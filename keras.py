#!/usr/bin/python python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from mlsmote import get_minority_instace, MLSMOTE

data_train = pd.read_csv("train.csv", sep=',', index_col=0)
#data_train.iloc[:, 123:] = data_train.iloc[:, 123:].astype('int')
#X_train = np.round(np.array(data_train.iloc[:, 0:123]), decimals=2)
#y = data.iloc[:,123:]
#y_train = np.round(np.array(data_train.iloc[:, 123:]))
X_train = data_train.iloc[:, 0:123]
y_train = data_train.iloc[:, 123:]

X_sub, y_sub = get_minority_instace(X_train, y_train)
X_res, y_res = MLSMOTE(X_sub, y_sub, 2000)
X_mlsmote_train = pd.DataFrame(np.concatenate(
    (X_train, X_res), axis=0), columns=X_train.columns)
y_mlsmote_train = pd.DataFrame(np.concatenate(
    (y_train, y_res), axis=0), columns=y_train.columns)

data_test = pd.read_csv("test.csv", sep=',', index_col=0)
#data_test.iloc[:, 123:] = data_test.iloc[:, 123:].astype('int')
#X_test = np.round(np.array(data_test.iloc[:, 0:123]), decimals=2)
#y_test = np.round(np.array(data_test.iloc[:, 123:]))

X_test = data_test.iloc[:, 0:123]
y_test = data_test.iloc[:, 123:]

X_sub_t, y_sub_t = get_minority_instace(X_test, y_test)
X_res_t, y_res_t = MLSMOTE(X_sub_t, y_sub_t, 600)
X_mlsmote_test = pd.DataFrame(np.concatenate(
    (X_test, X_res_t), axis=0), columns=X_test.columns)
y_mlsmote_test = pd.DataFrame(np.concatenate(
    (y_test, y_res_t), axis=0), columns=y_test.columns)
#df = pd.read_csv("train.csv", sep=",", index_col=0)
plt.rcParams["figure.figsize"] = (10, 6)
plt.hist(y_mlsmote_train, bins=2)
plt.ylabel('Number of Samples')
plt.xlabel('No = 0 and Yes = 1')
plt.show()
# In the figure above the right side represents 1 = EC class being present and left represents 0 = EC class not there.
# Random Undersampling: drawing a subset from the original dataset, ensuring that you have equal numbers per class, effectively discarding many of the big-quantity class samples.
    # Usually undersampling is not a good idea for high imbalance set as you will be losing data

# Random Oversampling: drawing a subset from the original dataset, ensuring that you have equal numbers per class, effectively copying many of the low-quantity class samples.
# Applying class weights: by making classes with higher data quantities less important in the model optimization process, it is possible to achieve optimization-level class balance.
# Working with the F1 score instead of Precision and Recall: by using a metric that attempts to find a balance between relevance of all results and number of relevant results found, you could reduce the impact of class balance on your model without removing it.

# # Undersanpling
# # Count samples per class
# classes_zero = df[df.iloc[:,124] == 0]
# classes_one = df[df.iloc[:,124] == 1]

# # Print sizes
# print(f'Class 0: {len(classes_zero)}')
# print(f'Class 1: {len(classes_one)}')

# # Undersample zero to the size of one
# classes_zero = classes_zero.sample(len(classes_one))

# # Print sizes
# print(f'Class 0: {len(classes_zero)}')
# print(f'Class 1: {len(classes_one)}')


# Over sampling

# Count samples per class
# classes_zero = df[df.iloc[:,123] == 0]
# classes_one = df[df.iloc[:,123] == 1]

# # Print sizes
# print(f'Class 0: {len(classes_zero)}')
# print(f'Class 1: {len(classes_one)}')

# # Oversample one to the size of zero
# classes_one = classes_one.sample(len(classes_zero), replace=True)

# # Print sizes
# print(f'Class 0: {len(classes_zero)}')
# print(f'Class 1: {len(classes_one)}')


# Apply class weights

# Count samples per class
# classes_zero = df[df['EC6'] == 0]
# classes_one = df[df['EC6'] == 1]

# # Convert parts into NumPy arrays for weight computation
# zero_numpy = classes_zero['EC6'].to_numpy()
# one_numpy = classes_one['EC6'].to_numpy()
# all_together = np.concatenate((zero_numpy, one_numpy))
# unique_classes = np.unique(all_together)


#Compute weights
# weights = sklearn.utils.class_weight.compute_class_weight(
#     'balanced', unique_classes, all_together)

# print(weights)
# weights = sklearn.utils.class_weight.compute_sample_weight([{0:0.92633229, 1:1.08639706}, {0:0.75286624, 1:1.48866499}, {0:0.64274062, 1:2.25142857}, {0:0.56473961, 1:4.36162362}, {0:0.52070485, 1:12.57446809}, {0:0.52650334, 1:9.93277311}], y=df.iloc[:,123:])
# #print(weights.shape)
# for m in weights:
#     print(m)
