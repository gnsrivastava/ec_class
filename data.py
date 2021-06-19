#!/usr/bin/python python3

from data_pandas.dataset_prep import find_combinations, separate_pure_mixed, find_min_pure
import pandas as pd

data = pd.read_csv("total.csv", sep=',', index_col=0)
header = data.columns.values

combinations = find_combinations()
pure, mixed = separate_pure_mixed(combinations, data)

pure_total = pd.DataFrame(columns=data.columns)
for key, _ in pure.items():
    pure_total = pure_total.append(pure.get(str(key)))

count = find_min_pure(pure)

downsample_pure = pd.DataFrame(columns=data.columns)
for key, _ in pure.items():
    downsample_pure = downsample_pure.append(pure.get(str(key)).sample(count))

#print(len(downsample_pure.index))
rest_pure = pd.DataFrame()
for i, row in pure_total.iterrows():
    if row.name not in downsample_pure.index:
        rest_pure[row.name] = row.values

rest_pure = rest_pure.T
rest_pure.columns = data.columns
# print(rest_pure)