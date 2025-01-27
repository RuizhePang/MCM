import random
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from joblib import Parallel, delayed

# ------------------------------
# 数据预处理优化（全局计算 + 类型优化）
# ------------------------------
def preprocess_data(combined_data_path, prediction_year):
    combined_data = pd.read_csv(combined_data_path)
    combined_data = combined_data.fillna(0)  # 用 0 填充 NaN
    few_medal_df = combined_data[combined_data['data_type'] == 'few_medal'].drop(columns=['data_type'])
    raw_athletes = combined_data[combined_data['data_type'] == 'raw_athletes'].drop(columns=['data_type'])
    
    # 计算参赛年限
    participation_span = raw_athletes.groupby('Name')['Year'].agg(
        lambda x: prediction_year - x.min() + 1
    ).rename('Participation_Span').astype('int')
    raw_athletes = raw_athletes.merge(participation_span, on='Name', how='left')
    
    # Step 1: 获取每个国家历史的参赛人数统计
    athletes_data = raw_athletes.groupby(['NOC', 'Year'])['Name'].count().rename('Athletes').reset_index()
    
    # Step 2: 预测每个国家2028年的参赛人数并存入字典
    country_participants_dict = {}
    for country in raw_athletes['NOC'].unique():
        predicted = predict_2028_athletes_num(athletes_data, country)
        country_participants_dict[country] = int(predicted)  # 转为整数
    
    # Step 3: 将预测值映射到原始数据
    raw_athletes['Country_Participants'] = raw_athletes['NOC'].map(country_participants_dict)
    
    # 生成 Medal_Score 并保留原始 Medal 列
    raw_athletes['Medal_Score'] = raw_athletes['Medal'].map(
        {'Gold':5, 'Silver':3, 'Bronze':1, 'No medal':0}
    ).fillna(0).astype('int')
    raw_athletes['Medal_Score'] = raw_athletes.groupby(['NOC', 'Name'])['Medal_Score'].transform('sum')
    
    # only keep the data of the prediction year
    raw_athletes = raw_athletes[raw_athletes['Year'] == (prediction_year - 4)].copy()
    raw_athletes['Year'] = prediction_year

    # 返回包含 Medal 列的数据
    return few_medal_df, raw_athletes[['Name', 'NOC', 'Year', 'Event', 'Medal_Score', 'Participation_Span', 'Country_Participants']]

def predict_2028_athletes_num(athletes_data, country):

    country_data = athletes_data[athletes_data['NOC'] == country]

    X = country_data['Year'].values.reshape(-1, 1)
    y = country_data['Athletes'].values
    
    if len(X) == 0:
        return 0

    model = LinearRegression()
    model.fit(X, y)

    year_to_predict = np.array([[2028]])
    predicted_athletes = model.predict(year_to_predict)

    return predicted_athletes[0]

# ------------------------------
# CRITIC权重计算（修复标准差为零的问题）
# ------------------------------
def critic_weight(matrix):
    # === 全局标准化 ===
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1e-6  # 零值保护
    norm_matrix = (matrix - means) / stds  # 所有列统一Z-score
    
    # === 重新计算有效列 ===
    valid_col_mask = (stds != 0)
    valid_norm_matrix = norm_matrix[:, valid_col_mask]
    
    # === 冲突度计算 ===
    if valid_norm_matrix.size == 0:
        return np.zeros(matrix.shape[1])
    
    try:
        corr = np.corrcoef(valid_norm_matrix.T)
    except:
        corr = np.ones((valid_norm_matrix.shape[1], valid_norm_matrix.shape[1]))
    
    conflict = np.sum(1 - np.abs(corr), axis=1)
    
    # === 权重计算 ===
    valid_std = valid_norm_matrix.std(axis=0)  # 标准化后的标准差
    combined = valid_std * conflict
    total = combined.sum()
    
    if total <= 1e-6:
        valid_weights = np.ones_like(combined) / len(combined)
    else:
        valid_weights = combined / total
    
    # === 构建完整权重 ===
    full_weights = np.zeros(matrix.shape[1])
    full_weights[valid_col_mask] = valid_weights
    return full_weights


# ------------------------------
# TOPSIS评分计算（原始向量化实现）
# ------------------------------
def topsis(matrix, weights):
    # 归一化处理
    norms = np.linalg.norm(matrix, axis=0)
    norms[norms == 0] = 1e-6  # 零值保护
    norm_matrix = matrix / norms
    
    # 加权矩阵
    weighted = norm_matrix * weights
    
    # 理想解
    ideal_best = np.max(weighted, axis=0)
    ideal_worst = np.min(weighted, axis=0)
    
    # 距离计算
    d_best = np.linalg.norm(weighted - ideal_best, axis=1)
    d_worst = np.linalg.norm(weighted - ideal_worst, axis=1)
    
    # 综合得分
    return d_worst / (d_best + d_worst + 1e-6)  # 防止除零

# ------------------------------
# 核心评价逻辑（并行化优化）
# ------------------------------
def process_single_group(group):
    # 准备评价矩阵
    criteria_matrix = group[['Medal_Score', 'Participation_Span', 'Country_Participants']].values
    
    # print(f"\n{group['Year'].iloc[0]}年 {group['Event'].iloc[0]}项目的评价矩阵:")
    # print(criteria_matrix)

    # 跳过无效分组
    if len(criteria_matrix) < 2:
        return None
    
    # 权重计算
    weights = critic_weight(criteria_matrix)
    if weights is None:
        return None
    
    # print("计算得到的权重:")
    # print(weights)

    # TOPSIS评分
    scores = topsis(criteria_matrix, weights)
    if np.all(np.isnan(scores)):
        return None
    
    # print("计算得到的TOPSIS得分:")
    # print(scores)
    
    # 结果排序与奖牌分配
    group['TOPSIS_Score'] = scores
    ranked = group.sort_values('TOPSIS_Score', ascending=False)
    top3 = ranked.head(3).reset_index(drop=True)
    top3['Awarded_Medal'] = ['Gold', 'Silver', 'Bronze'][:len(top3)]
    
    return top3[['Year', 'Event', 'Name', 'NOC', 'Awarded_Medal']]

# 并行处理所有分组
def evaluate_athletes_by_group(df):
    groups = list(df.groupby(['Event']))
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(process_single_group)(group) 
        for _, group in tqdm(groups, desc="Processing Groups", unit="group")
    )
    #-----------------------test-----------------------
    # # 随机选择一个分组元组 (key, subgroup_df)
    # selected_group_tuple = random.choice(groups)
    # group_key = selected_group_tuple[0]
    # group_data = selected_group_tuple[1]  # 提取子 DataFrame
    # # 处理选中的分组
    # result = process_single_group(group_data)
    
    # # 返回单个结果（或直接打印）
    # if result is not None:
    #     print(f"\n随机选择的分组键：{group_key}")
    #     print("对应的奖牌分配结果：")
    #     print(result)
    #     return result
    # else:
    #     print(f"分组 {group_key} 无有效结果")
    #     return None
    #-----------------------test-------------------------

    # 合并有效结果
    valid_results = [res for res in results if res is not None]
    return pd.concat(valid_results) if valid_results else None


# ------------------------------
# 预测函数（修复链式赋值警告）
# ------------------------------
def predict_nation_medals(few_medal_df, medal_winners_df):
    few_medal_df = few_medal_df.copy()
    
    if medal_winners_df is None or medal_winners_df.empty:
        few_medal_df.loc[:, 'Gold'] = 0
        few_medal_df.loc[:, 'Silver'] = 0
        few_medal_df.loc[:, 'Bronze'] = 0
        few_medal_df.loc[:, 'Total'] = 0
        return few_medal_df
    
    medal_counts = medal_winners_df.groupby(['NOC'])['Awarded_Medal'].agg(
        Gold=lambda x: x.eq('Gold').sum(),
        Silver=lambda x: x.eq('Silver').sum(),
        Bronze=lambda x: x.eq('Bronze').sum(),
        Total=lambda x: x.isin(['Gold', 'Silver', 'Bronze']).sum()
    ).reset_index()
    
    merged_df = few_medal_df.merge(medal_counts, on=['NOC'], how='left').fillna(0)
    return merged_df

# ------------------------------
# 主执行流程
# ------------------------------
if __name__ == "__main__":
    random.seed(233)
    year = 2028
    combined_data_path='./data/clustered_data.csv'
    match_df = pd.read_csv('./data/match.csv')
    # 预处理数据
    
    few_medal_df, raw_atheletes_df = preprocess_data(combined_data_path, year)
    
    # 执行评价
    medal_winners = evaluate_athletes_by_group(raw_atheletes_df)
    medal_winners = medal_winners.merge(match_df, left_on='NOC', right_on='abbr', how='left')
    medal_winners = medal_winners[['Year', 'Event', 'Name', 'name', 'Awarded_Medal']]
    medal_winners = medal_winners.rename(columns={'name': 'NOC'})

    if medal_winners is not None:
        print("\n按项目分组的奖牌获得者名单:")
        print(medal_winners)
        medal_winners.to_csv('event_medal_winners.csv', index=False)
    else:
        print("未找到有效的奖牌分配数据")
    
    few_medal_df = few_medal_df.merge(match_df, left_on='NOC', right_on='abbr', how='left')
    few_medal_df = few_medal_df[['name']]
    few_medal_df = few_medal_df.rename(columns={'name': 'NOC'})

    print(few_medal_df)

    # 加载待预测数据并执行预测
    medal_prediction = predict_nation_medals(few_medal_df[['NOC']], medal_winners)
    print("\n国家年度奖牌预测（基于模型分配结果）:")
    print(medal_prediction)
    medal_prediction.to_csv('predictions_few_data.csv', index=False)







