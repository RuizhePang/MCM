import pandas as pd

athletes = pd.read_csv("../data/summerOly_athletes.csv")
medal_counts = pd.read_csv("../data/summerOly_medal_counts.csv")

noc_list = []
for index, row in medal_counts.iterrows():
    noc_list.append(row["NOC"].strip())

noc_list = sorted(set(noc_list))
noc_dict = {i: set() for i in noc_list}
noc_abbr_set = set()

for index, row in athletes.iterrows():
    team = row["Team"]
    noc_abbr = row["NOC"]
    for noc in noc_list:
        if str(team).startswith(noc):
            noc_dict[noc].add(noc_abbr)
            noc_abbr_set.add(noc_abbr)


sp_dict = {
    "Australia": ["AUS"],
    "Bohemia": ["BOH"],
    "British West Indies": ["WIF"],
    "Ceylon": ["SRI"],
    "Denmark": ["DEN"],
    "Dominica": ["DMA"],
    "FR Yugoslavia": ["YUG"],
    "France": ["FRA"],
    "Germany": ["GER"],
    "Great Britain": ["GBR"],
    "Ireland": ["IRL"],
    "Lebanon": ["LIB", "LBN"],
    "Netherlands": ["NED"],
    "Niger": ["NIG"],
    "ROC": ["ROC"],
    "Russia": ["RUS"],
    "Russian Empire": ["RUS"],
    "Serbia": ["SRB"],
    "Taiwan": ["TPE"],
    "United States": ["USA"],
    "United Team of Germany": ["GER"],
    "Virgin Islands": ["ISV"],
}

print("name,abbr")

for key, values in noc_dict.items():
    if len(values) != 1:
        if key not in sp_dict:
            continue
        values = sp_dict[key]
    for value in values:
        print(key + "," + value)
