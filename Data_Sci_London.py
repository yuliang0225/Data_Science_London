# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 12:29:42 2017
Data Science London + Scikit-learn
https://www.kaggle.com/c/data-science-london-scikit-learn

#description

@author: smuch
"""
#%% Pakage import
import numpy as np
import pandas as pd
import glob, re
from sklearn import *
from sklearn.preprocessing import MinMaxScaler 
import os
from sklearn.neighbors import KNeighborsClassifier
#%% Set path
# os.chdir(r"/home/smuch/文档/Data_Science_London")
os.getcwd()
#%% Data import
data = {
    'traInput': pd.read_csv(os.getcwd()+'/Data/train.csv',header=None),
    'tesInput': pd.read_csv(os.getcwd()+'/Data/test.csv',header=None),
    'traOutput': pd.read_csv(os.getcwd()+'/Data/trainLabels.csv',header=None)
    }    
#%%
traIn = data['traInput']
tesIn = data['tesInput']
traOu = data['traOutput']    

#%% Normalized
scaler =MinMaxScaler()
scaler.fit(traIn)
traInScale = scaler.transform(traIn)
#%% Cross Validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(KNNclf,traIn,traOu,cv=10)
print(scores)
scores.mean()

#%% KNN Models
KNNclf =KNeighborsClassifier(n_neighbors = 3)
#%% KNN Models N selection
ScoreMean = []  # Accuracy Mean
ScoreSD =[]     # Accuracy SD
for n_neighbors in range(1,50):
    KNNclf =KNeighborsClassifier(n_neighbors = n_neighbors)
    scores = cross_val_score(KNNclf,traIn,traOu,cv=5)
    ScoreMean.append(scores.mean())
    ScoreSD.append(scores.var())
    
plt.errorbar(range(1,50),ScoreMean,yerr=ScoreSD)
#%%
ScoreMean.index(max(ScoreMean)) # N =3
#%%
KNNclf =KNeighborsClassifier(n_neighbors = 3)
KNNclf.fit(traIn,traOu)

KNNResult = KNNclf.predict(tesIn)
save = pd.DataFrame({'ID':range(1,9001),'Value':KNNResult}) 
save.to_csv('b.csv',index=False,sep=b',') 


#%% PCA 
from sklearn.decomposition import PCA
pca = PCA(n_components=None)
pca.fit(traInScale)
traIn_pca = pca.transform(traInScale)

#%%
temp=traIn_pca[:,1]




