import pandas as pd

# Load data
match_table = pd.read_csv('../data/match.csv')
medal_data = pd.read_csv('../data/summerOly_medal_counts.csv')
athletes_data = pd.read_csv('../data/summerOly_athletes.csv')

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

# Apply filtering conditions by country group
grouped = medal_data_sorted.groupby('NOC', group_keys=False)
few_data_indices = grouped.apply(filter_conditions).explode().dropna().astype(int)
few_medal = medal_data_sorted.loc[few_data_indices]
abundant_medal = medal_data_sorted.drop(few_data_indices)

athletes_data = athletes_data[['Name', 'Year', 'NOC', 'Event', 'Medal']]

few_athletes = pd.merge(few_medal[['NOC', 'Year']], athletes_data, on=['NOC', 'Year'], how='left')
abundant_athletes = pd.merge(abundant_medal[['NOC', 'Year']], athletes_data, on=['NOC', 'Year'], how='left')

# Output results
print("Few_data statistics:")
print(f"- Number of countries: {few_medal['NOC'].nunique()}")
print(f"- Number of records: {len(few_medal)}")
print(f"- Number of associated athletes: {len(few_athletes)}")

print(few_athletes)
print(few_medal)

print("\nAbundant_data statistics:")
print(f"- Number of countries: {abundant_medal['NOC'].nunique()}")
print(f"- Number of records: {len(abundant_medal)}")
print(f"- Number of associated athletes: {len(abundant_athletes)}")

# Write to CSV file (single sheet)
few_medal['data_type'] = 'few_medal'
abundant_medal['data_type'] = 'abundant_medal'
few_athletes['data_type'] = 'few_athletes'
abundant_athletes['data_type'] = 'abundant_athletes'
medal_data['data_type'] = 'raw_medal'
athletes_data['data_type'] = 'raw_athletes'

combined_data = pd.concat([few_medal, abundant_medal, few_athletes, abundant_athletes, medal_data, athletes_data], ignore_index=True)
combined_data.to_csv('../data/clustered_data.csv', index=False)
