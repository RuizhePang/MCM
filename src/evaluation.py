import random
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
# ------------------------------
# Visualization Functions
# ------------------------------
def plot_criteria_distribution(df):
    """Plot distribution of criteria (Medal_Score, Participation_Span, Country_Participants)"""
    plt.figure(figsize=(12, 6))
    plt.suptitle("Distribution of Criteria", fontsize=16)
    
    plt.subplot(1, 3, 1)
    sns.histplot(df['Medal_Score'], kde=True, color='blue')
    plt.title("Medal Score Distribution")
    
    plt.subplot(1, 3, 2)
    sns.histplot(df['Participation_Span'], kde=True, color='green')
    plt.title("Participation Span Distribution")
    
    plt.subplot(1, 3, 3)
    sns.histplot(df['Country_Participants'], kde=True, color='orange')
    plt.title("Country Participants Distribution")
    
    plt.tight_layout()
    plt.savefig('criteria_distribution.png')  # Save as image
    plt.close()

def plot_critic_weights(weights, criteria_names):
    """Plot CRITIC weights for each criterion"""
    plt.figure(figsize=(16, 10))
    sns.barplot(x=criteria_names, y=weights, palette='viridis')
    plt.title("CRITIC Weights for Criteria")
    plt.ylabel("Weight")
    plt.xlabel("Criteria")
    plt.xticks(rotation=45)
    plt.savefig('critic_weights.png')  # Save as image
    plt.close()

def plot_topsis_scores(scores):
    """Plot distribution of TOPSIS scores"""
    plt.figure(figsize=(8, 5))
    sns.histplot(scores, kde=True, color='purple')
    plt.title("Distribution of TOPSIS Scores")
    plt.xlabel("TOPSIS Score")
    plt.ylabel("Frequency")
    plt.savefig('topsis_scores.png')  # Save as image
    plt.close()

def plot_prediction_vs_actual(medal_prediction, raw_medal_df):
    """Plot predicted vs actual medal counts"""
    merged = medal_prediction.merge(
        raw_medal_df,
        on=["NOC", "Year"],
        suffixes=("_pred", "_true"),
        how="left"
    ).fillna(0)
    
    plt.figure(figsize=(12, 6))
    plt.suptitle("Predicted vs Actual Medal Counts", fontsize=16)
    
    medals = ['Gold', 'Silver', 'Bronze']
    for i, medal in enumerate(medals):
        plt.subplot(1, 3, i+1)
        sns.scatterplot(x=f"{medal}_true", y=f"{medal}_pred", data=merged, color='blue')
        plt.plot([0, merged[f"{medal}_true"].max()], [0, merged[f"{medal}_true"].max()], 'r--')
        plt.title(f"{medal} Medals")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
    
    plt.tight_layout()
    plt.savefig('prediction_vs_actual.png')  # Save as image
    plt.close()


# ------------------------------
# Data Preprocessing
# ------------------------------
def preprocess_data(combined_data_path, prediction_year):
    combined_data = pd.read_csv(combined_data_path)
    combined_data = combined_data.fillna(0)  # Fill NaN with 0
    few_medal = combined_data[combined_data['data_type'] == 'few_medal'].drop(columns=['data_type'])
    raw_athletes = combined_data[combined_data['data_type'] == 'raw_athletes'].drop(columns=['data_type'])
    raw_medal = combined_data[combined_data['data_type'] == 'raw_medal'].drop(columns=['data_type'])
    
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
    return few_medal, raw_athletes[['Name', 'NOC', 'Year', 'Event', 'Medal_Score', 'Participation_Span', 'Country_Participants']], raw_medal

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
def predict_nation_medals(few_medal_df, medal_winners_df, year_prediction):
    few_medal_df = few_medal_df.copy()
    
    if medal_winners_df is None or medal_winners_df.empty:
        few_medal_df['prediction_year'] = year_prediction
        few_medal_df = few_medal_df.drop_duplicates(subset='NOC', keep='first')
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
    
    medal_counts['prediction_year'] = year_prediction
    
    merged_df = few_medal_df.merge(medal_counts, on=['NOC'], how='left').fillna(0)
    merged_df = merged_df.drop_duplicates(subset='NOC', keep='first')  # 按国家去重
    
    merged_df['prediction_year'] = year_prediction
    return merged_df

def evaluate_prediction(model_prediction, raw_medal_df):
    merged = model_prediction.merge(
        raw_medal_df,
        on=["NOC", "Year"],
        suffixes=("_pred", "_true"),
        how = "left"
    ).fillna(0)
    
    if merged.empty:
        raise ValueError("No matching countries between prediction and true data.")
    
    y_true = merged[["Gold_true", "Silver_true", "Bronze_true", "Total_true"]].values
    y_pred = merged[["Gold_pred", "Silver_pred", "Bronze_pred", "Total_pred"]].values

    correlations = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1 and len(np.unique(y_pred[:, i])) > 1:
            corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
            correlations.append(corr)
        else:
            correlations.append(np.nan)  
    
    mse = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    
    avg_corr = np.nanmean(correlations)
    
    return avg_corr, mse, r2

# ------------------------------
# Main Execution Flow
# ------------------------------
if __name__ == "__main__":
    random.seed(233)
    year = 2024
    combined_data_path = './data/clustered_data.csv'
    match_df = pd.read_csv('./data/match.csv')
    
    # Preprocess data
    few_medal_df, raw_athletes_df, raw_medal_df = preprocess_data(combined_data_path, year)
    raw_athletes_df.fillna(0, inplace=True)
    raw_medal_df.fillna(0, inplace=True)
    raw_medal_temp = raw_medal_df.copy()

    # Plot criteria distribution
    plot_criteria_distribution(raw_athletes_df)

    # Execute evaluation
    medal_winners = evaluate_athletes_by_group(raw_athletes_df)

    # Plot CRITIC weights (example for one group)
    example_group = raw_athletes_df.groupby('Event').get_group(list(raw_athletes_df['Event'].unique())[0])
    weights = critic_weight(example_group[['Medal_Score', 'Participation_Span', 'Country_Participants']].values)
    plot_critic_weights(weights, ['Medal_Score', 'Participation_Span', 'Country_Participants'])

    # Plot TOPSIS scores (example for one group)
    example_scores = topsis(example_group[['Medal_Score', 'Participation_Span', 'Country_Participants']].values, weights)
    plot_topsis_scores(example_scores)

    # Load prediction data and execute prediction
    medal_prediction = predict_nation_medals(few_medal_df[['NOC']], medal_winners, year)
    medal_prediction.to_csv(f'predictions_few_data_{year}.csv', index=False)


    medal_prediction = medal_prediction.rename(columns={"prediction_year": "Year"})
    # Plot predicted vs actual medal counts
    plot_prediction_vs_actual(medal_prediction, raw_medal_temp)

    # Evaluate prediction
    avg_corr, mse, r2 = evaluate_prediction(medal_prediction, raw_medal_temp)
    print(f"\nEvaluation results for year {year}:")
    print(f"- Average Correlation: {avg_corr:.4f}")
    print(f"- Mean Squared Error: {mse}")
    print(f"- R² Score: {r2}")