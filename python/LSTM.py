import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Data Processing
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

# Training Data
for country in medal_pivot.index:
    if country not in ['United States', 'China', 'Great Britain', 'France','Japan', 'Australia']:
        continue
    
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

X = np.array(X)
y = np.array(y)

# Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2025)

# Model Defination
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=500)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=3000,
    batch_size=32,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Validation by Using the Data of 2024
accuracies = []
for country in medal_pivot.index:
    if country not in ['Great Britain', 'United States','France', 'China', 'Japan', 'Australia']:
        continue
    print('\n')
    input_medals = [medal_pivot.loc[country, years[-years_back + i]] for i in range(years_back)]
    input_athletes = [athletes_pivot.loc[country, 2024]]
    input_host = [host_pivot.loc[country, 2024]]
    input_year = [np.float64(2024)]
    input_events = [events_pivot.loc[country, 2024]]
    input_data = input_medals + input_athletes + input_host + input_year + input_events

    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)
    input_data = input_data.reshape((1, years_back + 4, 1))

    prediction = model.predict(input_data)[0][0]
    actual = medal_pivot.loc[country, 2024]

    print(f"{country} - Actual 2024 total medal number: ", actual)
    print(f"{country} - Predicted 2024 total medal number: ", prediction)

    if actual == 0:
        if round(prediction) == 0:
            accuracy = 1
        else:
            accuracy = 0
    else:
        accuracy = 1 - abs(actual - prediction) / actual
    print(f"{country} - Accuracy: {accuracy:.2%}")
    accuracies.append(accuracy)
average_accuracy = sum(accuracies) / len(accuracies)
print(average_accuracy)
