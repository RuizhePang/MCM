import random
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
from joblib import Parallel, delayed

# ------------------------------
# 数据预处理优化（全局计算 + 类型优化）
# ------------------------------
def preprocess_data(combined_data_path='output_data.csv'):
    combined_data = pd.read_csv(combined_data_path)
    few_medal_df = combined_data[combined_data['data_type'] == 'few_medal'].drop(columns=['data_type'])
    raw_athletes = combined_data[combined_data['data_type'] == 'raw_athletes'].drop(columns=['data_type'])
    
    # 计算参赛年限
    participation_span = raw_athletes.groupby('Name')['Year'].agg(
        lambda x: x.max() - x.min() + 1
    ).rename('Participation_Span').astype('int16')
    raw_athletes = raw_athletes.merge(participation_span, on='Name', how='left')
    
    # 计算国家参赛人数
    country_participants = raw_athletes.groupby(['Event', 'NOC']).size().rename('Country_Participants').astype('int16')
    raw_athletes = raw_athletes.merge(country_participants, on=['Event', 'NOC'], how='left')
    
    # 生成 Medal_Score 并保留原始 Medal 列
    raw_athletes['Medal_Score'] = raw_athletes['Medal'].map(
        {'Gold':150, 'Silver':100, 'Bronze':50, 'No medal':0}
    ).fillna(0).astype('int8')
    
    # 返回包含 Medal 列的数据
    return few_medal_df, raw_athletes[['Name', 'NOC', 'Year', 'Event', 'Medal', 'Medal_Score', 'Participation_Span', 'Country_Participants']]

# ------------------------------
# CRITIC权重计算（修复标准差为零的问题）
# ------------------------------
def critic_weight(matrix):
    # 计算原始标准差
    std = matrix.std(axis=0)
    
    # 创建有效指标掩码（标准差非零的列）
    valid_col_mask = (std != 0)
    valid_std = std[valid_col_mask]
    valid_matrix = matrix[:, valid_col_mask]
    
    # 处理全无效情况
    if valid_matrix.size == 0:
        return np.zeros(matrix.shape[1])
    
    # 处理单列特殊情况
    if valid_matrix.shape[1] == 1:
        full_weights = np.zeros(matrix.shape[1])
        full_weights[valid_col_mask] = 1.0  # 唯一有效列权重设为1
        return full_weights
    
    # 标准化有效列
    matrix_std = (valid_matrix - valid_matrix.mean(axis=0)) / valid_std
    
    # 计算相关系数矩阵
    try:
        corr = np.corrcoef(matrix_std.T)
    except:
        corr = np.ones((valid_matrix.shape[1], valid_matrix.shape[1]))
    
    # 冲突度计算（仅考虑有效列）
    conflict = np.sum(1 - np.abs(corr), axis=1)
    variability = valid_std
    
    # 权重计算（仅对有效列）
    denominator = np.sum(variability * conflict)
    if denominator <= 1e-6:
        valid_weights = np.ones(len(valid_std)) / len(valid_std)
    else:
        valid_weights = (variability * conflict) / denominator
    
    # 构建完整权重向量
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
    groups = list(df.groupby(['Year', 'Event']))
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
    
    medal_counts = medal_winners_df.groupby(['NOC', 'Year'])['Awarded_Medal'].agg(
        Gold=lambda x: x.eq('Gold').sum(),
        Silver=lambda x: x.eq('Silver').sum(),
        Bronze=lambda x: x.eq('Bronze').sum(),
        Total=lambda x: x.isin(['Gold', 'Silver', 'Bronze']).sum()
    ).reset_index()
    
    merged_df = few_medal_df.merge(medal_counts, on=['NOC', 'Year'], how='left').fillna(0)
    return merged_df

# ------------------------------
# 主执行流程
# ------------------------------
if __name__ == "__main__":
    random.seed(604)

    # 预处理数据
    few_medal_df, raw_atheletes_df = preprocess_data()
    
    # 执行评价
    medal_winners = evaluate_athletes_by_group(raw_atheletes_df)
    if medal_winners is not None:
        print("\n按项目分组的奖牌获得者名单:")
        print(medal_winners)
        medal_winners.to_csv('event_medal_winners.csv', index=False)
    else:
        print("未找到有效的奖牌分配数据")
    

    # 加载待预测数据并执行预测
    medal_prediction = predict_nation_medals(few_medal_df[['NOC', 'Year']], medal_winners)
    print("\n国家年度奖牌预测（基于模型分配结果）:")
    print(medal_prediction)
    medal_prediction.to_csv('nation_medal_predictions.csv', index=False)