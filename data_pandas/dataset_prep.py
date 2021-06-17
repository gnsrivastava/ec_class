#!/usr/bin/python python3

# The goal of the script is to separate pure vs mixed data points to do statification

import itertools
from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np

#write a function for pure class:
data = pd.read_csv("total.csv", sep=',', index_col=0)
header = data.columns.values

n = '100000'
ls = []
while n != '111111':
    for j in range(1, len(n)):
        a = sorted(np.unique(['_'.join(i) for i in itertools.permutations(n, 6)]), reverse=True)
        ls.extend(a)
        temp = "1"
        n = n[0:j]+temp+n[j+1:]

value = []
for factor in ls:
    df1 = data[data['EC1_EC2_EC3_EC4_EC5_EC6'].str.contains(factor)]
    df1.to_csv("ec_"+factor+".csv")