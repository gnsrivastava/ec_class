#!/usr/bin/python python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss, classification_report

# Import Multilabel packages
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from skmultilearn.adapt import MLkNN

# Import ML packages
#pip install scikit-multilearn

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB

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

data = pd.read_csv("training_set_2367_CIDs.csv", sep=',', index_col=0)
data.iloc[:,123:] = data.iloc[:,123:].astype('float')
X = data.iloc[:,0:123]
print(X.head())
# Exploratory data analysis to see how many positive counts are under each EC class
#sns.countplot(data['EC6'])
#plt.show()

# train test split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
