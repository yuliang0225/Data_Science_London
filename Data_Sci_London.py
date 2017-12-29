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
import matplotlib.pyplot as plt 
#%% Set path
os.chdir(r"/home/smuch/文档/Data_Science_London")
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
max(ScoreMean) # N =3
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
#%% SVM
from sklearn.svm import SVC
svm = SVC(kernel = 'rbf', C=0.1, gamma = 1)
scores = cross_val_score(svm,traInScale,traOu,cv=10)
print(scores)
scores.mean()
#%% SVM RBF c g
def SvmRbfPara(tranIn,tranOut,step,Cmin,Cmax,gmin,gmax,cvnum):
    AccSum = list()
    m = (np.arange(Cmin,Cmax+1,step,dtype=float)).size
    n = (np.arange(gmin,gmax+1,step,dtype=float)).size
    for c in np.arange(Cmin,Cmax+1,step,dtype=float):
        for g in np.arange(gmin,gmax+1,step,dtype=float):
            svm = SVC(kernel = 'rbf', C=2**c, gamma = 2**g)
            scores = cross_val_score(svm,tranIn,tranOut,cv=cvnum)
            AccSum.append(scores.mean())
    rearr = np.asarray(AccSum).reshape(n,m)
    return rearr
#%% ALL parameters
temp = SvmRbfPara(traInScale,traOu,0.05,-5,5,-5,5,5)
np.savetxt("SVM_c_g_AP.csv", temp, delimiter=",")
#%% Plot
import seaborn as sns
plt.figure(figsize=(4,4),dpi=600)
SVMcg_df = pd.DataFrame(temp,columns=[x for x in np.arange(-5,5+1,0.05,dtype=float)],index=[x for x in np.arange(-5,5+1,0.05,dtype=float)])
sns.heatmap(SVMcg_df,annot=False,vmin=0.8)
plt.show()
#%% Get optimal C and g
index = np.where(temp==np.max(temp))
np.max(temp)
bestc = 2**np.arange(-5,5+1,0.05,dtype=float)[index[0][0]]
bestg = 2**np.arange(-5,5+1,0.05,dtype=float)[index[1][0]]
#%% 
Cmin = -2
Cmax = 2
gmin = -2
gmax = 2
cvnum = 5
tranOut = traOu
tranIn = traInScale
step=1
#%%
i = (np.arange(Cmin,Cmax+1,step)).size
j = (np.arange(gmin,gmax+1,step)).size
















