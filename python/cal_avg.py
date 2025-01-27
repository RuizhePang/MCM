import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="Arguments")
parser.add_argument('--years_back', type=int, default=3, help="years to call back")
parser.add_argument('--medal_type', type=str, default='Total', help="which medal to predict")
args = parser.parse_args()

years_back = args.years_back
medal_type = args.medal_type

input_file = f'../result/result_{medal_type}_back{years_back}.xlsx'

df = pd.read_excel(input_file)
columns_to_average = ["SVM", "RandomForest", "LinearRegression", "Ridge", "Lasso", "WeightLinearRegression", "RidgeCV", "LassoCV"]

# Replace 'ABSENT' with NaN and convert the columns to numeric
df_no_absent = df[columns_to_average].replace("ABSENT", np.nan)

# Calculate the minimum and maximum for each row
df["Min"] = df_no_absent.min(axis=1, skipna=True)
df["Max"] = df_no_absent.max(axis=1, skipna=True)
df["Mid"] = (df_no_absent.min(axis=1, skipna=True) + df_no_absent.max(axis=1, skipna=True)) / 2

# Calculate the interval width
df["Interval"] = ((df["Max"] - df["Min"]) / 2).round(0)

# Format the result as 'X±Y'
df["Interval_Result"] = df.apply(lambda row: f"{row['Mid']:.0f}±{row['Interval']:.0f}" if pd.notna(row['Max']) and pd.notna(row['Min']) else "ABSENT", axis=1)

# Drop the "Min", "Max", and "Interval" columns if you no longer need them
df.drop(columns=["Min", "Max", "Interval"], inplace=True)

df.to_excel(input_file, index=False)
