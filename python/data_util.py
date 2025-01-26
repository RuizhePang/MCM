import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

years = [1896, 1900, 1904, 1908, 1912, 1920, 1924, 1928, 1932, 1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020]
#years = [1896, 1900, 1904, 1908, 1912, 1920, 1924, 1928, 1932, 1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024]

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
        if len(recent_3_games) == 0:
            return 3

        condition1 = len(recent_3_games) < 3
        condition2 = all(recent_3_games['Total'] < 3)
    else:
        condition1 = True
        condition2 = True

    return condition1 or condition2

def predict_2028_events_num(events_data):
    df = pd.DataFrame(events_data)

    X = df['Year'].values.reshape(-1, 1)
    y = df['Events'].values

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    model_poly = LinearRegression()
    model_poly.fit(X_poly, y)

    year_to_predict = np.array([[2028]])
    year_to_predict_poly = poly.transform(year_to_predict)
    predicted_events = model_poly.predict(year_to_predict_poly)
    
    return predicted_events[0]

    #print(f"Predicted Events for 2028: {predicted_events[0]:.2f}")

    #df['Year'] = df['Year'].astype(float)
    #df['Events'] = df['Events'].astype(float)

    #x_range = np.linspace(1896, 2028, 100).reshape(-1, 1)
    #x_range_poly = poly.transform(x_range)

    #plt.scatter(df['Year'], df['Events'], color='blue', label='Actual Data')
    #plt.plot(x_range, model_poly.predict(x_range_poly), color='red', label='Fitted Line')
    #plt.scatter(2028, predicted_events, color='green', label='Prediction for 2028')

    #plt.xlabel('Year')
    #plt.ylabel('Events')
    #plt.legend()
    #plt.title('Event Prediction for 2028')

    #plt.show()

#predict_2028_events_num(events_data)

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

    #print(f"Predicted Athletes for {country} in 2028: {predicted_athletes[0]:.2f}")

    #plt.scatter(country_data['Year'], country_data['Athletes'], color='blue', label=f'Actual Data ({country})')
    #plt.plot(country_data['Year'], model.predict(X), color='red', label='Fitted Line')  # 绘制回归线
    #plt.scatter(2028, predicted_athletes, color='green', label='Prediction for 2028')

    #plt.xlabel('Year')
    #plt.ylabel('Number of Athletes')
    #plt.legend()
    #plt.title(f'Athlete Number Prediction for {country} in 2028')

    #plt.show()
