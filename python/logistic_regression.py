import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

roc_abbr = pd.read_csv("../data/match.csv")
athletes = pd.read_csv("../data/summerOly_athletes.csv")
medal_counts = pd.read_csv("../data/summerOly_medal_counts.csv")
medal_counts["NOC"] = medal_counts["NOC"].str.strip()

country_set = set(athletes["NOC"])
country_total_medal_count = {country: 0 for country in country_set}
for index, row in medal_counts.iterrows():
    df = roc_abbr[roc_abbr["name"] == row["NOC"]]
    if len(df) != 1:
        continue
    country_total_medal_count[df["abbr"].iloc[0]] += row["Total"]

years = sorted(set(athletes["Year"]))
dt = 4

X = []
y = []

for country in country_set:
    names = roc_abbr[roc_abbr["abbr"] == country]["name"]
    if ((medal_counts["Year"].isin(years[:dt])) & (medal_counts["NOC"].isin(names))).any():
        continue
    country_df = athletes[athletes["NOC"] == country]
    for i in range(len(years) - dt - 1):
        nsports = 0
        nathletes = 0
        nevents = 0
        for j in range(dt):
            df = country_df[country_df["Year"] == years[i + j]]
            nsports += len(set(df["Sport"]))
            nathletes += len(set(df["Name"]))
            nevents += len(set(athletes[athletes["Year"] == years[i + j]]["Event"]))
        label = (
            (medal_counts["Year"] == years[i + dt]) & (medal_counts["NOC"].isin(names))
        ).any()
        if nsports == 0 and not label:
            continue
        nsports /= dt
        nathletes /= dt
        nevents /= dt
        X.append([nsports, nathletes, nevents, i + dt])
        y.append(label)
        if label:
            break

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", (y_test == y_pred).mean())
print("Recall:", recall_score(y_test, y_pred))

model.fit(X, y)
for year in (2024,):
    X_test = []
    y_test = []
    for country in country_set:
        names = roc_abbr[roc_abbr["abbr"] == country]["name"]
        country_df = athletes[athletes["NOC"] == country]
        if ((medal_counts["Year"] < year) & (medal_counts["NOC"].isin(names))).any():
            continue
        nsports = 0
        nathletes = 0
        nevents = 0
        for j in range(dt):
            df = country_df[country_df["Year"] == year - 4 * (j + 1)]
            nsports += len(set(df["Sport"]))
            nathletes += len(set(df["Name"]))
            nevents += len(
                set(athletes[athletes["Year"] == year - 4 * (j + 1)]["Event"])
            )
        nsports /= dt
        nathletes /= dt
        nevents /= dt
        label = (
            (medal_counts["Year"] == year) & (medal_counts["NOC"].isin(names))
        ).any()
        X_test.append([nsports, nathletes, nevents, years.index(year)])
        y_test.append(label)
    y_pred = model.predict(X_test)
    count = (y_pred == y_test).sum()
    print(f"2024 Accuracy: {count}/{len(y_test)}")
