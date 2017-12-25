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
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
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
#%% Data View
train_dataframe = pd.DataFrame(data=traIn[0:50])

#%%
plt.plot(traIn[:][1],traIn[:][2],'o')

#%%
scaler =MinMaxScaler()
scaler.fit(traIn)
traInScale = scaler.transform(traIn)
#%%
