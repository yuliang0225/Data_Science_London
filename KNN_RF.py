#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 22:19:56 2017

@author: smuch
"""

#%%
#!python3
import numpy as np
#import sklearn as sk
#import matplotlib.pyplot as plt
import pandas as pd
import os
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import Perceptron
#from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import VotingClassifier
#from sklearn import svm
from sklearn.cross_validation import cross_val_score
#%%
#### READING OUR GIVEN DATA INTO PANDAS DATAFRAME ####
os.chdir(r"/home/smuch/文档/Data_Science_London")
# Data_read
train_data = pd.read_csv(os.getcwd()+'/Data/train.csv', header=None)
train_labels = pd.read_csv(os.getcwd()+'/Data/trainLabels.csv', header=None)
test_data = pd.read_csv(os.getcwd()+'/Data/test.csv', header=None)
x_train = np.asarray(train_data)
y_train = np.asarray(train_labels).ravel()
x_test = np.asarray(test_data)

print 'training_x Shape:',x_train.shape,',training_y Shape:',y_train.shape, ',testing_x Shape:',x_test.shape

#Checking the models
x_all = np.r_[x_train,x_test]
print 'x_all shape :',x_all.shape
#%% Feature Absorbtion
#### USING THE GAUSSIAN MIXTURE MODEL ####
from sklearn.mixture import GaussianMixture
lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
	for n_components in n_components_range:
        # Fit a mixture of Gaussians with EM
		gmm = GaussianMixture(n_components=n_components,covariance_type=cv_type)
		gmm.fit(x_all)
        bic.append(gmm.aic(x_all))
        if bic[-1] < lowest_bic:
        	lowest_bic = bic[-1]
        	best_gmm = gmm
		
best_gmm.fit(x_all)
x_train = best_gmm.predict_proba(x_train)
x_test = best_gmm.predict_proba(x_test)

#%% modeling
#### TAKING ONLY TWO MODELS FOR KEEPING IT SIMPLE ####
knn = KNeighborsClassifier()
rf = RandomForestClassifier()

param_grid = dict( )## ?????? Why this is empty??
#### GRID SEARCH for BEST TUNING PARAMETERS FOR KNN #####
grid_search_knn = GridSearchCV(knn,param_grid=param_grid,cv=10,scoring='accuracy').fit(x_train,y_train)
print 'best estimator KNN:',grid_search_knn.best_estimator_,'Best Score', grid_search_knn.best_estimator_.score(x_train,y_train)
knn_best = grid_search_knn.best_estimator_
	
#### GRID SEARCH for BEST TUNING PARAMETERS FOR RandomForest #####
grid_search_rf = GridSearchCV(rf, param_grid=dict( ), verbose=3,scoring='accuracy',cv=10).fit(x_train,y_train)
print 'best estimator RandomForest:',grid_search_rf.best_estimator_,'Best Score', grid_search_rf.best_estimator_.score(x_train,y_train)
rf_best = grid_search_rf.best_estimator_


knn_best.fit(x_train,y_train)
print knn_best.predict(x_test)[0:10]
rf_best.fit(x_train,y_train)
print rf_best.predict(x_test)[0:10]

#### SCORING THE MODELS ####
print 'Score for KNN :',cross_val_score(knn_best,x_train,y_train,cv=10,scoring='accuracy').mean()
print 'Score for Random Forest :',cross_val_score(rf_best,x_train,y_train,cv=10,scoring='accuracy').max()

### IN CASE WE WERE USING MORE THAN ONE CLASSIFIERS THEN VOTING CLASSIFIER CAN BE USEFUL ###
#clf = VotingClassifier(
#		estimators=[('knn_best',knn_best),('rf_best',rf_best)],
#		#weights=[871856020222,0.907895269918]
#	)
#clf.fit(x_train,y_train)
#print clf.predict(x_test)[0:10]

##### FRAMING OUR SOLUTION #####
knn_best_pred = pd.DataFrame(knn_best.predict(x_test))
rf_best_pred = pd.DataFrame(rf_best.predict(x_test))
#voting_clf_pred = pd.DataFrame(clf.predict(x_test))

knn_best_pred.index += 1
rf_best_pred.index += 1
#voting_clf_pred.index += 1

#knn_best_pred.to_csv('knn_best_pred.csv')
rf_best_pred.to_csv('Submission.csv')
#voting_clf_pred.to_csv('voting_clf_pred.csv')
