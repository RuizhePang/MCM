import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

match_table = pd.read_csv('../data/match.csv')

medal_data = pd.read_csv('../data/summerOly_medal_counts.csv')
medal_data['NOC'] = medal_data['NOC'].str.strip()

athletes_data = pd.read_csv('../data/summerOly_athletes.csv')
athletes_data = pd.merge(athletes_data, match_table, left_on='NOC', right_on='abbr', how='left')
athletes_data = athletes_data[['Name', 'Year', 'name']]
athletes_data = athletes_data.rename(columns={
    'name': 'NOC',
    'Name': 'Athletes'
})
athletes_data = athletes_data.groupby(['NOC', 'Year'])['Athletes'].count().reset_index()

host_data = pd.read_csv('../data/summerOly_hosts.csv')
def extract_country(host):
    if "Cancelled" in host:
        return None
    host = host.split('(')[0].strip()
    country = host.split(",")[-1].strip()

    # 将 United Kingdom 替换为 Great Britain
    if country == "United Kingdom":
        country = "Great Britain"
    return country
host_data["Country"] = host_data["Host"].apply(extract_country)

events_data = pd.read_csv('../data/summerOly_programs.csv', encoding='Windows-1252')
events_data = events_data.iloc[-3, 4:].reset_index()
events_data.columns = ['Year', 'Events']
events_data = events_data.drop(index=3).reset_index(drop=True)
events_data['Year'] = events_data['Year'].astype(int)

merged_data = pd.merge(medal_data, athletes_data, on=['NOC', 'Year'], how='left')
merged_data = pd.merge(merged_data, host_data, on='Year', how='left')
merged_data['Is_Host'] = (merged_data['Country'] == merged_data['NOC']).astype(int)
merged_data = pd.merge(merged_data, events_data, on='Year', how='left')

medal_pivot = merged_data.pivot(index='NOC', columns='Year', values='Total').fillna(0)
athletes_pivot = merged_data.pivot(index='NOC', columns='Year', values='Athletes').fillna(0)
host_pivot = merged_data.pivot(index='NOC', columns='Year', values='Is_Host').fillna(0)
events_pivot = merged_data.pivot(index='NOC', columns='Year', values='Events').fillna(0)

years = [1896, 1900, 1904, 1908, 1912, 1920, 1924, 1928, 1932, 1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020]
years_back = 3

for country in medal_pivot.index:
    #if country not in ['Japan']:
    if country not in ['Great Britain', 'United States','France', 'China', 'Japan', 'Australia']:
        continue
    #print(f"\nTraining Model - Country: {country}")
    
    X = []
    y = []

    for i in range(len(years) - years_back):
        medals = [medal_pivot.loc[country, years[i + j]] for j in range(years_back)]
        athletes = [athletes_pivot.loc[country, years[i + years_back]]]
        host = [host_pivot.loc[country, years[i + years_back]]]
        year = [np.float64(years[i + years_back])]
        events = [events_pivot.loc[country, years[i + years_back]]]

        features = medals + athletes + host + year + events

        medal_next = medal_pivot.loc[country, years[i + years_back]]
        
        X.append(features)
        y.append(medal_next)

X = pd.DataFrame(X, columns=[f'Medal_{i+1}' for i in range(years_back)] + ['Athletes'] + ['Host'] + ['Year'] + ['Events'])
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")

accuracies = []
for country in medal_pivot.index:
    #if country not in ['China']:
    if country not in ['Great Britain', 'United States','France', 'China', 'Japan', 'Australia']:
        continue
    print('\n')
    input_medals = [medal_pivot.loc[country, years[-years_back + i]] for i in range(years_back)]
    input_athletes = [athletes_pivot.loc[country, 2024]]
    input_host = [host_pivot.loc[country, 2024]]
    input_year = [np.float64(2024)]
    input_events = [events_pivot.loc[country, 2024]]
    input_data = input_medals + input_athletes + input_host + input_year + input_events
    input_df = pd.DataFrame([input_data], columns=[f'Medal_{i+1}' for i in range(years_back)] + ['Athletes'] + ['Host'] + ['Year'] + ['Events'])
    
    prediction = model.predict(input_df)[0]
    actual = medal_pivot.loc[country, 2024]

    print(f"{country} - Actual 2024 total medal number: ",medal_pivot.loc[country, 2024])
    print(f"{country} - Predicted 2024 total medal number: ",prediction)
    
    prediction = round(prediction)
    if actual == 0:
        if prediction == 0:
            accuracy = 1
        else:
            accuracy = 0 
    else:
        accuracy = 1 - abs(actual - prediction) / actual  # 计算准确率
    accuracies.append(accuracy)  # 将准确率添加到列表中
    print(f"{country} - Accuracy: {accuracy:.2%}")

average_accuracy = sum(accuracies) / len(accuracies)
print(f"\nAverage Accuracy: {average_accuracy:.2%}")
