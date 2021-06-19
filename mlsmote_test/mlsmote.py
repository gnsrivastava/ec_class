# -*- coding: utf-8 -*-
# Importing required Library
import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors


# def create_dataset(n_sample=1000):
#     ''' 
#     Create a unevenly distributed sample data set multilabel  
#     classification using make_classification function
    
#     args
#     nsample: int, Number of sample to be created
#     weights: ratio of numbers of ones vs total number of samples
#     return
#     X: pandas.DataFrame, feature vector dataframe with 10 features 
#     y: pandas.DataFrame, target vector dataframe with 5 labels
#     '''
#     X, y = make_classification(n_classes=5, class_sep=2,
#                                weights=[0.1, 0.025, 0.205, 0.008, 0.9], n_informative=3, n_redundant=1, flip_y=0,
#                                n_features=10, n_clusters_per_class=1, n_samples=1000, random_state=10)
#     y = pd.get_dummies(y, prefix='class')
#     #print(X.shape, y.shape)
#     return pd.DataFrame(X), y
data = pd.read_csv("train.csv", sep=",", index_col=0)
X, y = data.iloc[:,0:123], data.iloc[:,123:]

def get_tail_label(df):
    """
    Give tail label colums of the given target dataframe
    
    args
    df: pandas.DataFrame, target label df whose tail label has to identified
    
    return
    tail_label: list, a list containing column name of all the tail label
    """
    columns = df.columns
    #print(columns)
    n = len(columns)
    #print(n)
    irpl = np.zeros(n)  # Imbalance ratio per label: irpl
    #print(irpl)
    for column in range(n):
        #print(df[columns[column]].value_counts()[1])   # number of 1 in a label    df[columns[column]].value_counts() => output: class 0: 900, class 1: 100
        irpl[column] = df[columns[column]].value_counts()[1]
    irpl = max(irpl)/irpl
    #print(irpl)
    mir = np.average(irpl)  # Mean Imbalance ratio: mir
    #print(mir)
    tail_label = []
    for i in range(n):
        if irpl[i] > mir:
            #print(columns[i])
            tail_label.append(columns[i])
    #print(tail_label)
    return tail_label

def get_index(df):
  """
  give the index of all tail_label rows
  args
  df: pandas.DataFrame, target label df from which index for tail label has to identified
    
  return
  index: list, a list containing index number of all the tail label
  """
  tail_labels = get_tail_label(df)
  #print(tail_labels)
  index = set()
  for tail_label in tail_labels:
    #print(df[df[tail_label] == 1].index)   # gives indices of each row with tail_label =1
    sub_index = set(df[df[tail_label] == 1].index)
    #print(sub_index)
    index = index.union(sub_index)
    #print(index)
  return list(index)


def get_minority_instace(X, y):
    """
    Give minority dataframe containing all the tail labels
    
    args
    X: pandas.DataFrame, the feature vector dataframe
    y: pandas.DataFrame, the target vector dataframe
    
    return
    X_sub: pandas.DataFrame, the feature vector minority dataframe
    y_sub: pandas.DataFrame, the target vector minority dataframe
    """
    index = get_index(y)
    X_sub = X[X.index.isin(index)].reset_index(drop=True)
    #print(X[X.index.isin(index)].reset_index(drop=True)) # reset_index() will convert index to column0, drop = True will remove the old index as column0
    y_sub = y[y.index.isin(index)].reset_index(drop=True)
    return X_sub, y_sub


def nearest_neighbour(X):
    """
    Give index of 5 nearest neighbor of all the instance
    
    args
    X: np.array, array whose nearest neighbor has to find
    
    return
    indices: list of list, index of 5 NN of each element in X
    """
    nbs = NearestNeighbors(
        n_neighbors=6, metric='euclidean', algorithm='kd_tree').fit(X)
    _, indices = nbs.kneighbors(X)
    #print(indices)
    return indices


def MLSMOTE(X, y, n_sample):
    """
    Give the augmented data using MLSMOTE algorithm
    
    args
    X: pandas.DataFrame, input vector DataFrame
    y: pandas.DataFrame, feature vector dataframe
    n_sample: int, number of newly generated sample
    
    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data
    """
    indices2 = nearest_neighbour(X) # Shape of indices2 = (33, 5) same as y_sub
    #print(indices2.shape)
    n = len(indices2) 
    #print(n)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0, n-1)  # give number between 0 and 32
        neighbour = random.choice(indices2[reference, 1:])
        #print(indices2[reference,0:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis=0, skipna=True)
        target[i] = np.array([1 if val > 2 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference, :] - X.loc[neighbour, :]
        new_X[i] = np.array(X.loc[reference, :] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    new_X = pd.concat([X, new_X], axis=0)
    target = pd.concat([y, target], axis=0)
    return new_X, target

if __name__ == 'main':
    """
    main function to use the MLSMOTE
    """
    #X, y = create_dataset()  # Creating a Dataframe
    #Getting minority instance of that datframe
    X_sub, y_sub = get_minority_instace(X, y)
    # Applying MLSMOTE to augment the dataframe
    X_res, y_res = MLSMOTE(X_sub, y_sub, 2000)
    X_mlsmote = pd.DataFrame(np.concatenate((X, X_res), axis=0), columns=X.columns)
    y_mlsmote = pd.DataFrame(np.concatenate((y, y_res), axis=0), columns=y.columns)

#Step1: create data
# X, y = create_dataset()


# EDA for labels
# classes_zero = y[y.iloc[:,1] == 0]
# classes_one = y[y.iloc[:,1] == 1]
# print(f'Class 0: {len(classes_zero)}')
# print(f'Class 1: {len(classes_one)}')

#Step2: get minority labels
# X_sub, y_sub = get_minority_instace(X, y)
# X_res, y_res = MLSMOTE(X_sub, y_sub, 2000)

# X_mlsmote = pd.DataFrame(np.concatenate((X, X_res),axis=0), columns=X.columns)
# y_mlsmote = pd.DataFrame(np.concatenate((y, y_res), axis=0), columns=y.columns)

# classes_zero = y_mlsmote[y_mlsmote.iloc[:,1] == 0.0]
# classes_one = y_mlsmote[y_mlsmote.iloc[:, 1] == 1.0]
# print(f'Class 0: {len(classes_zero)}')
# print(f'Class 1: {len(classes_one)}')
