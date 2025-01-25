import pandas as pd

# 加载数据
match_table = pd.read_csv('../data/match.csv')
medal_data = pd.read_csv('../data/summerOly_medal_counts.csv')
athletes_data = pd.read_csv('../data/summerOly_athletes.csv')
host_data = pd.read_csv('../data/summerOly_hosts.csv')
#events_data = pd.read_csv('../data/summerOly_programs.csv')

# 数据预处理
medal_data['NOC'] = medal_data['NOC'].str.strip()
athletes_data['NOC'] = athletes_data['NOC'].str.strip()
medal_data['Year'] = pd.to_numeric(medal_data['Year'], errors='coerce')
athletes_data['Year'] = pd.to_numeric(athletes_data['Year'], errors='coerce')

# 按国家和年份排序
medal_data_sorted = medal_data.sort_values(by=['NOC', 'Year'], ascending=[True, False])
# athletes_data_sorted = athletes_data.sort_values(by=['NOC', 'Year'], ascending=[True, False])

# 定义筛选条件函数
def filter_conditions(group):
    few_data_indices = []
    for idx, row in group.iterrows():
        current_year = row['Year']
        # 获取当前年份之前的记录（按年份降序）
        past_games = group[group['Year'] < current_year].sort_values(by='Year', ascending=False)
        # 取最近三次奥运会记录
        recent_3_games = past_games.head(3)
        # print(recent_3_games)
        # print("\n" )
        
        # 条件1：最近三次参赛次数 < 3
        condition1 = len(recent_3_games) < 3
        
        # 条件2：最近三次每次奖牌数 < 3
        if not recent_3_games.empty:
            condition2 = all(recent_3_games['Total'] < 3)
        else:
            condition2 = False
        
        # 满足任一条件则标记为few_data
        if condition1 or condition2:
            few_data_indices.append(idx)
    return few_data_indices

# 按国家分组应用筛选条件
grouped = medal_data_sorted.groupby('NOC', group_keys=False)
few_data_indices = grouped.apply(filter_conditions).explode().dropna().astype(int)
#print(few_data_indices)

# 分割数据
few_medal = medal_data_sorted.loc[few_data_indices]
#print(few_medal)
#print(few_medal)
abundant_medal = medal_data_sorted.drop(few_data_indices)

# ----------------------------合并国家代码（NOC）---------------------------------

# 合并运动员数据与国家全称映射表
athletes_data = pd.merge(
    athletes_data, 
    match_table, 
    left_on='NOC', 
    right_on='abbr', 
    how='left'
)

# 提取关键字段并重命名
athletes_data = athletes_data[['Name', 'Year', 'name', 'Event', 'Medal']]
athletes_data = athletes_data.rename(columns={'name': 'NOC'})

#------------------------------------------------------------------------------------

# 合并运动员数据（可选）
few_athletes = pd.merge(few_medal[['NOC', 'Year']], athletes_data, on=['NOC', 'Year'], how='left')
abundant_athletes = pd.merge(abundant_medal[['NOC', 'Year']], athletes_data, on=['NOC', 'Year'], how='left')

# 输出结果
#print("few_data 统计:")
#print(f"- 国家数量: {few_medal['NOC'].nunique()}")
#print(f"- 记录数量: {len(few_medal)}")
#print(f"- 关联运动员数量: {len(few_athletes)}")

#print(few_athletes)
#print(few_medal)

#print("\nabundant_data 统计:")
#print(f"- 国家数量: {abundant_medal['NOC'].nunique()}")
#print(f"- 记录数量: {len(abundant_medal)}")
#print(f"- 关联运动员数量: {len(abundant_athletes)}")
