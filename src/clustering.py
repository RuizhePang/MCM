import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd

# Load data
match_table = pd.read_csv('./data/match.csv')
medal_data = pd.read_csv('./data/summerOly_medal_counts.csv')
athletes_data = pd.read_csv('./data/summerOly_athletes.csv')

# Data preprocessing
medal_data['NOC'] = medal_data['NOC'].str.strip()
athletes_data['NOC'] = athletes_data['NOC'].str.strip()
medal_data['Year'] = pd.to_numeric(medal_data['Year'], errors='coerce')
athletes_data['Year'] = pd.to_numeric(athletes_data['Year'], errors='coerce')

# Change NOC to abbreviation to filter out outdated data
medal_data = pd.merge(medal_data, match_table, left_on='NOC', right_on='name', how='left')
medal_data = medal_data[['abbr', 'Year', 'Total', 'Gold', 'Silver', 'Bronze']]
medal_data = medal_data.rename(columns={'abbr': 'NOC'})

# Sort by country and year
medal_data_sorted = medal_data.sort_values(by=['NOC', 'Year'], ascending=[True, False])

# List of Olympic years
years = [1896, 1900, 1904, 1908, 1912, 1920, 1924, 1928, 1932, 1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024]

def filter_conditions(group):
    few_data_indices = []
    for idx, row in group.iterrows():
        current_year = row['Year']
        past_years = [year for year in years if year <= current_year]
        if len(past_years) >= 3:
            recent_3_years = past_years[-3:]
            recent_3_games = group[group['Year'].isin(recent_3_years)].sort_values(by='Year', ascending=False)
            condition1 = len(recent_3_games) < 3 
            condition2 = all(recent_3_games['Total'] < 3)
        else:
            condition1 = True
            condition2 = True

        if condition1 or condition2:
            few_data_indices.append(idx)
    return few_data_indices

def country_filter(medal_data, athletes_data, country, year):
    current_year = year
    past_years = [year for year in years if year <= current_year]
    group = medal_data[medal_data['NOC'] == country]
    if len(past_years) >= 3:
        recent_3_years = past_years[-3:]
        recent_3_games = group[group['Year'].isin(recent_3_years)].sort_values(by='Year', ascending=False)

        recent_3_attendances = athletes_data[athletes_data['NOC'] == country]
        recent_3_attendances = recent_3_attendances[recent_3_attendances['Year'].isin(recent_3_years)]
        
        #if country == 'Ceylon':
        #    print(recent_3_attendances)
        #    print(recent_3_games)
        #    raise
        
        # the country did not attend the recent 3 games
        if len(recent_3_attendances) == 0:
            return 2

        # the country attend the recent 3 games but did not gain any medal
        if (len(recent_3_games) > 0) and (all(recent_3_games['Total'] == 0)):
            return 3

        condition1 = len(recent_3_games) < 3
        condition2 = all(recent_3_games['Total'] < 3)
    else:
        condition1 = True
        condition2 = True

    return condition1 or condition2

# Apply filtering conditions by country group
grouped = medal_data_sorted.groupby('NOC', group_keys=False)
few_data_indices = grouped.apply(filter_conditions).explode().dropna().astype(int)
few_medal = medal_data_sorted.loc[few_data_indices]
abundant_medal = medal_data_sorted.drop(few_data_indices)

athletes_data = athletes_data[['Name', 'Year', 'NOC', 'Event', 'Medal']]

few_athletes = pd.merge(few_medal[['NOC', 'Year']], athletes_data, on=['NOC', 'Year'], how='left')
abundant_athletes = pd.merge(abundant_medal[['NOC', 'Year']], athletes_data, on=['NOC', 'Year'], how='left')

# Output results
# print("Few_data statistics:")
# print(f"- Number of countries: {few_medal['NOC'].nunique()}")
# print(f"- Number of records: {len(few_medal)}")
# print(f"- Number of associated athletes: {len(few_athletes)}")

# print(few_athletes)
# print(few_medal)

# print("\nAbundant_data statistics:")
# print(f"- Number of countries: {abundant_medal['NOC'].nunique()}")
# print(f"- Number of records: {len(abundant_medal)}")
# print(f"- Number of associated athletes: {len(abundant_athletes)}")

# Write to CSV file (single sheet)
few_medal['data_type'] = 'few_medal'
abundant_medal['data_type'] = 'abundant_medal'
few_athletes['data_type'] = 'few_athletes'
abundant_athletes['data_type'] = 'abundant_athletes'
medal_data['data_type'] = 'raw_medal'
athletes_data['data_type'] = 'raw_athletes'

combined_data = pd.concat([few_medal, abundant_medal, few_athletes, abundant_athletes, medal_data, athletes_data], ignore_index=True)
# combined_data.to_csv('./data/clustered_data.csv', index=False)




# 获取所有国家并排序
countries = medal_data_sorted['NOC'].unique()

# 创建国家到y轴位置的映射
country_to_y = {country: idx for idx, country in enumerate(countries)}

# 准备数据
x = medal_data_sorted['Year']
y = medal_data_sorted['NOC'].map(country_to_y)
# 确定颜色：few_data_indices对应红色，其他蓝色
colors = ['red' if idx in few_data_indices else 'blue' for idx in medal_data_sorted.index]

# 创建图形
plt.figure(figsize=(14, 10))
plt.scatter(x, y, c=colors, s=40, alpha=0.6, edgecolors='w', linewidth=0.5)

# 设置坐标轴标签
plt.yticks(ticks=range(len(countries)), labels=countries)
plt.xticks(ticks=years, labels=years, rotation=90)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.title('Data Availability by Country and Year', fontsize=14)

# 调整布局和网格
plt.grid(True, linestyle='--', alpha=0.4, which='both', axis='both')
plt.gca().set_axisbelow(True)  # 将网格线置于数据下方
plt.tight_layout()

plt.show()
# 创建国家到y轴位置的映射
countries = medal_data_sorted['NOC'].unique()
country_to_y = {country: idx for idx, country in enumerate(countries)}

# 生成颜色映射
color_map = {
    0: 'blue',   # 正常数据
    1: 'red',    # 满足条件1或2
}

# 生成颜色列表（核心修改部分）
colors = []
for _, row in medal_data_sorted.iterrows():
    result = country_filter(
        medal_data,          # 原始奖牌数据
        athletes_data,       # 原始运动员数据 
        row['NOC'],          # 当前国家
        row['Year']          # 当前年份
    )
    colors.append(color_map.get(result, 'blue'))  # 默认蓝色

# 创建图形
plt.figure(figsize=(16, 12))
scatter = plt.scatter(
    x=medal_data_sorted['Year'],
    y=medal_data_sorted['NOC'].map(country_to_y),
    c=colors,
    s=40,
    alpha=0.7,
    edgecolors='w',
    linewidth=0.5
)

# 坐标轴设置
y_tick_labels = [countries[i] if i % 3 == 2 else '' for i in range(len(countries))]
plt.yticks(ticks=range(len(countries)), labels=y_tick_labels)
plt.xticks(ticks=years, labels=years, rotation=90)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.title('Olympic Data Classification by Country and Year', fontsize=14)

# 添加自定义图例
legend_elements = [
    Patch(facecolor='blue', label='Abundant Data'),
    Patch(facecolor='red', label='Few Data'),
]
plt.legend(handles=legend_elements, loc='upper right')

# 优化显示
plt.grid(True, linestyle='--', alpha=0.4, which='both', axis='both')
plt.gca().set_axisbelow(True)
plt.tight_layout()
plt.savefig('classification_result.png', dpi=150)  # 添加此行
plt.show()