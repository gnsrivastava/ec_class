# -*- coding: utf-8 -*-
# Importing required Library
#!/usr/bin/python python3

# The goal of the script is to separate pure vs mixed data points to do statification

import itertools
import numpy as np

def find_combinations():
    n = '100000'
    combinations = []
    while n != '111111':
        for j in range(1, len(n)):
            a = sorted(np.unique(['_'.join(i) for i in itertools.permutations(n, 6)]), reverse=True)
            combinations.extend(a)
            temp = "1"
            n = n[0:j]+temp+n[j+1:]
    return combinations

def separate_pure_mixed(combinations, data):
    pure = dict()
    mixed = dict()
    for factor in combinations:
        df1 = data[data['EC1_EC2_EC3_EC4_EC5_EC6'].str.contains(factor)]
        #df1.to_csv("ec_"+factor+".csv")
        if (factor == '1_0_0_0_0_0' or factor == '0_1_0_0_0_0' or factor == '0_0_1_0_0_0' or factor == '0_0_0_1_0_0' or factor == '0_0_0_0_1_0' or factor == '0_0_0_0_0_1') and df1.empty == False:
            pure[factor] = df1
        elif df1.empty == False:
            mixed[factor] = df1
        else:
            continue
    return pure, mixed

def find_min_pure(pure):
    shape =[]
    for key, _ in pure.items():
        shape.append(pure[key].shape[0])
        count = min(shape)
    return count
