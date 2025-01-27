import random
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from joblib import Parallel, delayed

# ------------------------------
# Data Preprocessing
# ------------------------------
def preprocess_data(combined_data_path, prediction_year):
    combined_data = pd.read_csv(combined_data_path)
    combined_data = combined_data.fillna(0)  # Fill NaN with 0
    few_medal_df = combined_data[combined_data['data_type'] == 'few_medal'].drop(columns=['data_type'])
    raw_athletes = combined_data[combined_data['data_type'] == 'raw_athletes'].drop(columns=['data_type'])
    
    # Calculate participation span
    participation_span = raw_athletes.groupby('Name')['Year'].agg(
        lambda x: prediction_year - x.min() + 1
    ).rename('Participation_Span').astype('int')
    raw_athletes = raw_athletes.merge(participation_span, on='Name', how='left')
    
    # Step 1: Get historical athlete count per country
    athletes_data = raw_athletes.groupby(['NOC', 'Year'])['Name'].count().rename('Athletes').reset_index()
    
    # Step 2: Predict the number of athletes for each country in 2028
    country_participants_dict = {}
    for country in raw_athletes['NOC'].unique():
        predicted = predict_2028_athletes_num(athletes_data, country)
        country_participants_dict[country] = int(predicted)  # Convert to integer
    
    # Step 3: Map predicted values to the raw data
    raw_athletes['Country_Participants'] = raw_athletes['NOC'].map(country_participants_dict)
    
    # Generate Medal_Score and keep the original Medal column
    raw_athletes['Medal_Score'] = raw_athletes['Medal'].map(
        {'Gold':5, 'Silver':3, 'Bronze':1, 'No medal':0}
    ).fillna(0).astype('int')
    raw_athletes['Medal_Score'] = raw_athletes.groupby(['NOC', 'Name'])['Medal_Score'].transform('sum')
    
    # Only keep data for the prediction year
    raw_athletes = raw_athletes[raw_athletes['Year'] == (prediction_year - 4)].copy()
    raw_athletes['Year'] = prediction_year

    # Return data with Medal column
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
# CRITIC Weight Calculation
# ------------------------------
def critic_weight(matrix):
    # Normalize the matrix
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1e-6  # Zero value protection
    norm_matrix = (matrix - means) / stds  # Z-score normalization
    
    # Recalculate valid columns
    valid_col_mask = (stds != 0)
    valid_norm_matrix = norm_matrix[:, valid_col_mask]
    
    # Conflict calculation
    if valid_norm_matrix.size == 0:
        return np.zeros(matrix.shape[1])
    
    try:
        corr = np.corrcoef(valid_norm_matrix.T)
    except:
        corr = np.ones((valid_norm_matrix.shape[1], valid_norm_matrix.shape[1]))
    
    conflict = np.sum(1 - np.abs(corr), axis=1)
    
    # Weight calculation
    valid_std = valid_norm_matrix.std(axis=0)  # Standard deviation after normalization
    combined = valid_std * conflict
    total = combined.sum()
    
    if total <= 1e-6:
        valid_weights = np.ones_like(combined) / len(combined)
    else:
        valid_weights = combined / total
    
    # Build full weights
    full_weights = np.zeros(matrix.shape[1])
    full_weights[valid_col_mask] = valid_weights
    return full_weights

# ------------------------------
# TOPSIS Score Calculation
# ------------------------------
def topsis(matrix, weights):
    # Normalization
    norms = np.linalg.norm(matrix, axis=0)
    norms[norms == 0] = 1e-6  # Zero value protection
    norm_matrix = matrix / norms
    
    # Weighted matrix
    weighted = norm_matrix * weights
    
    # Ideal solutions
    ideal_best = np.max(weighted, axis=0)
    ideal_worst = np.min(weighted, axis=0)
    
    # Distance calculation
    d_best = np.linalg.norm(weighted - ideal_best, axis=1)
    d_worst = np.linalg.norm(weighted - ideal_worst, axis=1)
    
    # Composite score
    return d_worst / (d_best + d_worst + 1e-6)  # Prevent division by zero

# ------------------------------
# Core Evaluation Logic (Parallelized)
# ------------------------------
def process_single_group(group):
    # Prepare evaluation matrix
    criteria_matrix = group[['Medal_Score', 'Participation_Span', 'Country_Participants']].values
    
    # Skip invalid groups
    if len(criteria_matrix) < 2:
        return None
    
    # Weight calculation
    weights = critic_weight(criteria_matrix)
    if weights is None:
        return None
    
    # TOPSIS scoring
    scores = topsis(criteria_matrix, weights)
    if np.all(np.isnan(scores)):
        return None
    
    # Result sorting and medal allocation
    group['TOPSIS_Score'] = scores
    ranked = group.sort_values('TOPSIS_Score', ascending=False)
    top3 = ranked.head(3).reset_index(drop=True)
    top3['Awarded_Medal'] = ['Gold', 'Silver', 'Bronze'][:len(top3)]
    
    return top3[['Year', 'Event', 'Name', 'NOC', 'Awarded_Medal']]

# Parallel processing of all groups
def evaluate_athletes_by_group(df):
    groups = list(df.groupby(['Event']))
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(process_single_group)(group) 
        for _, group in tqdm(groups, desc="Processing Groups", unit="group")
    )

    # Merge valid results
    valid_results = [res for res in results if res is not None]
    return pd.concat(valid_results) if valid_results else None

# ------------------------------
# Prediction Function
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
# Main Execution Flow
# ------------------------------
#if __name__ == "__main__":
#    random.seed(233)
#    year = 2028
#    combined_data_path='../data/clustered_data.csv'
#    match_df = pd.read_csv('../data/match.csv')
    
    # Preprocess data
#    few_medal_df, raw_atheletes_df = preprocess_data(combined_data_path, year)
    
    # Execute evaluation
#    medal_winners = evaluate_athletes_by_group(raw_atheletes_df)
#    medal_winners = medal_winners.merge(match_df, left_on='NOC', right_on='abbr', how='left')
#    medal_winners = medal_winners[['Year', 'Event', 'Name', 'name', 'Awarded_Medal']]
#    medal_winners = medal_winners.rename(columns={'name': 'NOC'})

#    if medal_winners is not None:
#        print("\nMedal winners by event:")
#        print(medal_winners)
#        medal_winners.to_csv('event_medal_winners.csv', index=False)
#    else:
#        print("No valid medal allocation data found")
    
#    few_medal_df = few_medal_df.merge(match_df, left_on='NOC', right_on='abbr', how='left')
#    few_medal_df = few_medal_df[['name']]
#    few_medal_df = few_medal_df.rename(columns={'name': 'NOC'})

#    print(few_medal_df)

    # Load prediction data and execute prediction
#    medal_prediction = predict_nation_medals(few_medal_df[['NOC']], medal_winners)
#    print("\nNational medal prediction (based on model allocation):")
#    print(medal_prediction)
#    medal_prediction.to_csv('predictions_few_data.csv', index=False)
