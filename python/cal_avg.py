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
columns_to_average = ["SVM", "RandomForest", "DecisionTree", "LinearRegression"]
is_absent = df[columns_to_average].apply(lambda row: (row == 'ABSENT'), axis=1)
df_no_absent = df[columns_to_average].replace("ABSENT", np.nan)
df["Average"] = df_no_absent.apply(pd.to_numeric, errors='coerce').mean(axis=1, skipna=True)
df["Average"] = df["Average"].where(~is_absent.all(axis=1), "ABSENT")
df["Average"] = df["Average"].round(0)
df["Average"] = df["Average"].apply(lambda x: int(x) if isinstance(x, (float, int)) and pd.notna(x) else x)
df["Average"] = df["Average"].astype("object")
df.to_excel(input_file, index=False)
