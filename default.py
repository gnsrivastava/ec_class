#!/usr/bin/python python3

from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from skmultilearn.model_selection import iterative_train_test_split    #not working
#from sklearn.model_selection import train_test_split          #not working
#from sklearn.model_selection import StratifiedShuffleSplit    #not working
from skmultilearn.model_selection import IterativeStratification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, hamming_loss, classification_report

# Import Multilabel packages
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from skmultilearn.adapt import MLkNN

# Import ML packages
#pip install scikit-multilearn

from mlsmote import get_minority_instace, MLSMOTE

#from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import GaussianNB, MultinomialNB

def build_model(model, mlb_estimator, xtrain, ytrain, xtest, ytest):
    clf = mlb_estimator(model)
    clf.fit(xtrain, ytrain)
    #predict
    clf_prediction = clf.predict(xtest)

    # Check for accuracies and hamming loss
    acc = accuracy_score(ytest, clf_prediction)
    haml = hamming_loss(ytest, clf_prediction)
    results = {"accuracy": acc, "hamming loss": haml}
    return results

data_train = pd.read_csv("train.csv", sep=',', index_col=0)
#data_train.iloc[:, 123:] = data_train.iloc[:, 123:].astype('int')
#X_train = np.round(np.array(data_train.iloc[:, 0:123]), decimals=2)
#y = data.iloc[:,123:]
#y_train = np.round(np.array(data_train.iloc[:, 123:]))
X_train = data_train.iloc[:, 0:123]
y_train = data_train.iloc[:, 123:]

X_sub, y_sub = get_minority_instace(X_train, y_train)
X_res, y_res = MLSMOTE(X_sub, y_sub, 2000)
X_mlsmote_train = pd.DataFrame(np.concatenate((X_train, X_res), axis=0), columns=X_train.columns)
y_mlsmote_train = pd.DataFrame(np.concatenate((y_train, y_res), axis=0), columns=y_train.columns)

data_test = pd.read_csv("test.csv", sep=',', index_col=0)
#data_test.iloc[:, 123:] = data_test.iloc[:, 123:].astype('int')
#X_test = np.round(np.array(data_test.iloc[:, 0:123]), decimals=2)
#y_test = np.round(np.array(data_test.iloc[:, 123:]))

X_test = data_test.iloc[:, 0:123]
y_test = data_test.iloc[:, 123:]

X_sub_t, y_sub_t = get_minority_instace(X_test, y_test)
X_res_t, y_res_t = MLSMOTE(X_sub_t, y_sub_t, 600)
X_mlsmote_test = pd.DataFrame(np.concatenate((X_test, X_res_t), axis=0), columns=X_test.columns)
y_mlsmote_test = pd.DataFrame(np.concatenate((y_test, y_res_t), axis=0), columns=y_test.columns)

for n in [BinaryRelevance, ClassifierChain, LabelPowerset]:
    results = build_model(KNeighborsClassifier(n_neighbors=10), n, X_mlsmote_train, y_mlsmote_train, X_mlsmote_test, y_mlsmote_test)
    print("{} {}".format(str(n), results))

# # Choose max_samples:
# for n in [BinaryRelevance, ClassifierChain, LabelPowerset]:
#     for i in np.arange(0.1, 1.1, 0.1):
#         if i == 1.0:
#             i = None
#             results = build_model(RandomForestClassifier(
#                 max_samples=i), n, X_mlsmote_train, y_mlsmote_train, X_mlsmote_test, y_mlsmote_test)
#         else:
#             results = build_model(RandomForestClassifier(
#                 max_samples=i), n, X_mlsmote_train, y_mlsmote_train, X_mlsmote_test, y_mlsmote_test)

#         if i != None:
#             print("{} {} {}".format(str(n), round(i,2), results))
#         else:
#             print("{} {} {}".format(str(n), i, results))









# k_fold = IterativeStratification(n_splits=2, order=2)
# for train, test in k_fold.split(X, y):
#     X_train, y_train = X[train], y[train]
#     X_test, y_test = X[test], y[test]
#     for n in [BinaryRelevance, ClassifierChain, LabelPowerset]:
#         for i in [10, 50, 100, 500, 1000]:
#             results = build_model(RandomForestClassifier(max_samples = None, max_features = 81, n_estimators=i),LabelPowerset, X_train, y_train, X_test, y_test)
#             print("{} {} {}".format(str(n), i, results))




    

# stratified_split = StratifiedShuffleSplit(n_splits=2, test_size=0.33)
# for train_index, test_index in stratified_split.split(data_x, y):
#     x_train, x_test = data_x[train_index], data_x[test_index]
#     y_train, y_test = y[train_index], y[test_index]

#y = data[['EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6']]
#print(X, y)

# Exploratory data analysis to see how many positive counts are under each EC class
#sns.countplot(data['EC6'])
#plt.show()


# train test split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=data.iloc[:,123:], random_state=42)
#print(int(sqrt(X_train.shape[0])))

# Problem Transformation
# for n in [BinaryRelevance, ClassifierChain, LabelPowerset]:
#     for i in [10, 50, 100, 500, 1000, 1500]:
#        results = build_model(RandomForestClassifier(n_estimators=i),LabelPowerset, X_train, y_train, X_test, y_test)
#        print("{} {} {}".format(str(n), i, results)) 


# Choose max_samples:
# for n in [BinaryRelevance, ClassifierChain, LabelPowerset]:
#     for i in np.arange(0.1, 1.1, 0.1):
#         if i == 1.0:
#             i = None
#             results = build_model(RandomForestClassifier(max_samples=i),LabelPowerset, X_train, y_train, X_test, y_test)
#         else: 
#             results = build_model(RandomForestClassifier(max_samples=i),LabelPowerset, X_train, y_train, X_test, y_test)
    
#         if i != None:
#             print("{} {} {}".format(str(n), round(i,2), results))
#         else:
#             print("{} {} {}".format(str(n), i, results))

# mtry optimization

# for n in [BinaryRelevance, ClassifierChain, LabelPowerset]:
#     for i in range(int(sqrt(X_train.shape[1])), int(X_train.shape[1]), 10):
#         results = build_model(RandomForestClassifier(max_features=i),LabelPowerset, X_train, y_train, X_test, y_test)
#         print("{} {} {}".format(str(n), round(i,2), results))
