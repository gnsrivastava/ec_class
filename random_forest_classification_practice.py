# I will try to make a randome forest classifier in sklearn
# https://machinelearningmastery.com/random-forest-ensemble-in-python/
#!/usr/bin/python python3

# Problem: Classifiy substrate dataset into EC lasses using descriptors

# Pseudo code
#1 import all the modules, train_test_split, cross_val_score, RepreatedStratifiedKFold, numpy
# pandas, matplotlib, 

# 2 import training dataset
# 3   Run default model on training set to get the idea of performance
# Notes: RF is an ensemble ML algorithm
# In Random forest
#Regression: Prediction is the average prediction across the decision trees.
#Classification: Prediction is the majority vote class label predicted across the decision trees.
# Random forests’ tuning parameter is the number of randomly selected predictors, k, to choose from at each split, and is commonly referred to as mtry. 
# In the regression context, Breiman (2001) recommends setting mtry to be one-third of the number of predictors

# Three parameters in RF
# 1) Mtry
# 2) Depth of a tree
# 3) Number of trees in an ensemble


# from numpy.core.numeric import cross
# from sklearn.ensemble  import RandomForestClassifier # RandomForestClassifier is a class
# from sklearn import datasets

# iris = datasets.load_iris()

# import numpy as np
# import pandas as pd

# df = pd.DataFrame(iris.data, iris.target, columns = iris.feature_names)
# df.reset_index(level=0, inplace = True)
# df = df.rename(columns={'index':'target'}, inplace = False) # Here inplace = True returns None

# X = df.drop('target', axis =1).values
# y = df.target.values
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import RepeatedStratifiedKFold
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state = 42)

##############################
#### Step 0: default model

################################
# define model # 
# rf = RandomForestClassifier()

# # Evaluate model
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
# n_scores = cross_val_score(rf, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# # report performance
# from numpy import mean
# from numpy import std
# print('Accuracy: %.3f (%.3f)' %(mean(n_scores), std(n_scores)))

# Now that I see at these parameters accuracy is 0.956
# I can make predictions on test set

# Call fit method
# rf.fit(X_test, y_test)

# y_pred = rf.predict(X_test)
# print(y_pred)
######***********************************************

################################################
# Step 1: Find max_samples for bootstrapping
#################################################
# Next step would be to look at hyperparameter

# Explore the number of samples
#from matplotlib import pyplot as plt
# # get a list of models to evaluate
# def get_models():
# 	rf = dict()
# 	# explore ratios from 10% to 100% in 10% increments
# 	for i in np.arange(0.1, 1.1, 0.1):
# 		key = '%.1f' % i
# 		# set max_samples=None to use 100%
# 		if i == 1.0:
# 		    i = None
# 		rf[key] = RandomForestClassifier(max_samples=i)
# 	return rf

# # Evaluate models using cross-validation
# def evaluate_model(rf, X, y):
#     cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
#     scores = cross_val_score(rf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
#     return scores

# # define dataset
# X, y = X_train, y_train

# # get models to evaluate
# rf = get_models()
# # evaluate the models and store results
# results, names = list(), list()
# for name, model in rf.items():
# 	# evaluate the model
# 	scores = evaluate_model(rf[name], X, y)
# 	# store the results
# 	results.append(scores)
# 	names.append(name)
# 	# summarize the performance along the way
# 	print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
# # plot model performance for comparison
# plt.boxplot(results, labels=names, showmeans=True)
#plt.show()
##*****************************************************
# Accuracies at differnt sample sizes
# Figure: bootstrap_max_samples_figure
# >0.1 0.975 (0.024)
# >0.2 0.971 (0.029)
# >0.3 0.968 (0.028)
# >0.4 0.965 (0.027)
# >0.5 0.965 (0.027)
# >0.6 0.962 (0.026)
# >0.7 0.962 (0.026)
# >0.8 0.956 (0.032)
# >0.9 0.956 (0.032)
# >1.0 0.959 (0.029)
# The result implies that the bootstrap sample of 0.1of total is best
# So in my case 0.1 would be the max_sample fraction for bootstrapping

#################################
# Step 2: mtry
################################
# Explore Number of features
# The number of features that is randomly sampled 
# for each split point (mtry)

# get a list of models to evaluate
# def get_models():
# 	rf = dict()
# 	# explore number of features from 1 to 4 # Only using till4 because there are only 4 features in iris dataset
# 	for i in range(1,5):
# 		rf[str(i)] = RandomForestClassifier(max_samples = 0.1, max_features=i)
# 	return rf

# # evaluate a given model using cross-validation
# def evaluate_model(rf, X, y):
# 	# define the evaluation procedure
# 	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
# 	# evaluate the model and collect the results
# 	scores = cross_val_score(rf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# 	return scores

# # define dataset
# X, y = X_train, y_train
# # get the models to evaluate
# rf = get_models()
# # evaluate the models and store results
# results, names = list(), list()
# for name, model in rf.items():
# 	# evaluate the model
# 	scores = evaluate_model(rf[name], X, y)
# 	# store the results
# 	results.append(scores)
# 	names.append(name)
# 	# summarize the performance along the way
# 	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# # plot model performance for comparison
# plt.boxplot(results, labels=names, showmeans=True)
# plt.show()
#####*********************************************************************


# Accuracies at different mtry values
# Mtry 2 is performing the best at max_samples =0.1
# max_sampels was determined before when explring # of samples

#figure: mtry_max_features_figure.png
# >1 0.959 (0.042)
# >2 0.971 (0.029)
# >3 0.968 (0.028)
# >4 0.962 (0.043)

###################################
#Step 3: ntree
###################################

#Explore Number of Trees (ntree)

# number of trees can be set by n_estimators argument
# Default = 100 

# get a list of models to evaluate
# def get_models():
# 	rf = dict()
# 	# define number of trees to consider
# 	n_trees = [10, 50, 100, 500, 1000, 1500]
# 	for n in n_trees:
# 		rf[str(n)] = RandomForestClassifier(max_samples = 0.1, max_features= 2,n_estimators=n)
# 	return rf

# # evaluate a given model using cross-validation
# def evaluate_model(rf, X, y):
# 	# define the evaluation procedure
# 	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
# 	# evaluate the model and collect the results
# 	scores = cross_val_score(rf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# 	return scores

# # define dataset
# X, y = X_train, y_train
# # get the models to evaluate
# rf = get_models()
# # evaluate the models and store results
# results, names = list(), list()
# for name, model in rf.items():
# 	# evaluate the model
# 	scores = evaluate_model(rf[name], X, y)
# 	# store the results
# 	results.append(scores)
# 	names.append(name)
# 	# summarize the performance along the way
# 	print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
# # plot model performance for comparison
# plt.boxplot(results, labels=names, showmeans=True)
# plt.show()
#####**********************************************************************
# from the results below it is clear that there is no
#performance gain after n_estimators = 100 (ntree = 100)
# Figure: ntree_n_estimators_figure.png
# >10 0.940 (0.051)
# >50 0.965 (0.027)
# >100 0.968 (0.028)
# >500 0.968 (0.028)
# >1000 0.968 (0.033)
# >1500 0.968 (0.028)

################################
#Step 4: Tree depth
################################

# Explore tree depth

# get a list of models to evaluate
# def get_models():
# 	rf = dict()
# 	# consider tree depths from 1 to 7 and None=full
# 	depths = [i for i in range(1,8)] + [None]
# 	for n in depths:
# 		rf[str(n)] = RandomForestClassifier(max_samples = 0.1, max_features = 2, n_estimators = 100, max_depth=n)
# 	return rf

# # evaluate a given model using cross-validation
# def evaluate_model(rf, X, y):
# 	# define the evaluation procedure
# 	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
# 	# evaluate the model and collect the results
# 	scores = cross_val_score(rf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
#     #print(scores)
# 	return scores

# # define dataset
# X, y = X_train, y_train
# # get the models to evaluate
# rf = get_models()
# # evaluate the models and store results
# results, names = list(), list()
# for name, model in rf.items():
# 	# evaluate the model
# 	scores = evaluate_model(rf[name], X, y)
# 	# store the results
# 	results.append(scores)
# 	names.append(name)
# 	# summarize the performance along the way
# 	print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
# # plot model performance for comparison
# #print(results)
# plt.boxplot(results, labels=names, showmeans=True)
# plt.show()

# From the results it is clear that depth of 3 is better
#Figure: tree_depth_max_depth_figure.png
# >1 0.968 (0.028)
# >2 0.959 (0.038)
# >3 0.971 (0.029)
# >4 0.968 (0.038)
# >5 0.965 (0.032)
# >6 0.971 (0.034)
# >7 0.971 (0.034)
# >None 0.962 (0.031) # None =>  no depth restriction

# Evaluate model after hyper tuning
# rf = RandomForestClassifier(max_samples = 0.1, max_features = 2, n_estimators = 100, max_depth = 3)
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
# n_scores = cross_val_score(rf, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# # report performance
# from numpy import mean
# from numpy import std
# print('Training Accuracy: %.3f (%.3f)' %(mean(n_scores), std(n_scores)))

# #Call fit method
# rf.fit(X_train, y_train)
# prob = rf.predict_proba(X_test)
# #print(prob)
# y_pred = rf.predict(X_test)
# print(tuple(zip(list(np.amax(prob, 1)), list(y_pred))))
# # Print error
# from sklearn import metrics
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# # Evaluate algorithm
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# print(accuracy_score(y_test, y_pred))

# Output
# Training Accuracy: 0.962 (0.031)
# [2 1 2 2 2 2 1 1 0 2 0 0 2 2 0 2 1 0 0 0 1 0 1 2 2 1 1 1 1 0 2 2 2 0 2 0 0
#  0 0 2 1 0 2 2 1]
# Mean Absolute Error: 0.06666666666666667
# Mean Squared Error: 0.06666666666666667
# Root Mean Squared Error: 0.2581988897471611
# [[15  0  0]
#  [ 0 12  3]
#  [ 0  0 15]]
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00        15
#            1       1.00      0.80      0.89        15
#            2       0.83      1.00      0.91        15

#     accuracy                           0.93        45
#    macro avg       0.94      0.93      0.93        45
# weighted avg       0.94      0.93      0.93        45

# 0.9333333333333333