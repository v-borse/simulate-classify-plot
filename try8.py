#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 08:36:03 2021

@author: vborse
"""

from IPython.display import clear_output, display, HTML

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import integrate
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso,LassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn import linear_model
import statsmodels.api as sm
from matplotlib import cm
import matplotlib.pyplot as plt
plt.style.use('bmh')
import seaborn as sns
import numpy as np
from statistics import variance
import cmath
from mlxtend.evaluate import bias_variance_decomp
from sklearn.metrics import r2_score

import module
import module2
#save numpy array as npy file
from numpy import asarray
from numpy import save
# define data
#data = asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# save to npy file
#save('data.npy', data)

from numpy import load
# load array
rms1 = load('rmse_a1.npy')
rms2 = load('rmse_a2.npy')
#print the array
labels=["x1","x2","x3"]
#plt.boxplot(rms[:,:]) 
#plt.boxplot(rms2[:,:], vert=True, patch_artist=True, labels=labels) 
#plt.ylabel('observed value')
#plt.title('Multiple Box Plot : Vertical Version')
#plt.show()
"""
import matplotlib.pyplot as plt
import numpy as np
data = np.random.normal(0.1, size=(100,6))
data[76:79,:] = np.ones((3,6))+0.2

plt.figure(figsize=(4,3))
# option 1, specify props dictionaries
c = "red"
#plt.boxplot(rms[:,:3], positions=[1,2,3], notch=False, patch_artist=True,labels=labels,
#             boxprops=dict(facecolor=c, color=c))
#            capprops=dict(color=c),
#            whiskerprops=dict(color=c),
#            flierprops=dict(color=c, markeredgecolor=c),
#            medianprops=dict(color=c),
#            )


# option 2, set all colors individually
c2 = "yellow"
box1 = plt.boxplot(rms2[:,:3], positions=[1.25,2.25,3.25], notch=False, patch_artist=True,labels=labels)
for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box1[item], color='k')
plt.setp(box1["boxes"], facecolor=c2)
plt.setp(box1["fliers"], markeredgecolor=c)

c3 = "green"
box2 = plt.boxplot(rms[:,:3], positions=[1.5,2.5,3.5], notch=False, patch_artist=True,labels=labels)
for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box2[item], color='k')
plt.setp(box2["boxes"], facecolor=c3)
plt.setp(box2["fliers"], markeredgecolor=c)


#plt.xlim(0.5,4)
plt.xticks([1,2,3], [1,2,3])
plt.show()
"""
def ta_rmse():
    from numpy import load
    # load array
    rms1 = load('rmse_a1.npy')
    rms2 = load('rmse_a2.npy')
    #t_lag=1: NR1a11,NR2a11,NR1a21,NR2a21  :OLS
    #t_lag=15: NR1a12,NR2a12,NR1a22,NR2a22  : OLS
    #t_lag=1: NR1a11,NR2a11,NR1a21,NR2a21  :Ridge
    #t_lag=15: NR1a12,NR2a12,NR1a22,NR2a22  :Ridge
    t1a=np.array([np.mean(rms1[:100,1]),np.mean(rms1[:100,2]),np.mean(rms2[:100,1]),np.mean(rms2[:100,2])])
    t2a=np.array([np.mean(rms1[100:200,1]),np.mean(rms1[100:200,2]),np.mean(rms2[100:200,1]),np.mean(rms2[100:200,2])])
    t3a=np.array([np.mean(rms1[200:300,1]),np.mean(rms1[200:300,2]),np.mean(rms2[200:300,1]),np.mean(rms2[200:300,2])])
    t4a=np.array([np.mean(rms1[300:400,1]),np.mean(rms1[300:400,2]),np.mean(rms2[300:400,1]),np.mean(rms2[300:400,2])])
    
    t1ra=np.array([np.mean(rms1[:100,3]),np.mean(rms1[:100,4]),np.mean(rms2[:100,3]),np.mean(rms2[:100,4])])
    t2ra=np.array([np.mean(rms1[100:200,3]),np.mean(rms1[100:200,4]),np.mean(rms2[100:200,3]),np.mean(rms2[100:200,4])])
    t3ra=np.array([np.mean(rms1[200:300,3]),np.mean(rms1[200:300,4]),np.mean(rms2[200:300,3]),np.mean(rms2[200:300,4])])
    t4ra=np.array([np.mean(rms1[300:400,3]),np.mean(rms1[300:400,4]),np.mean(rms2[300:400,3]),np.mean(rms2[300:400,4])])
    
    mini=np.array([t1a[0],t2a[0],t3a[0],t4a[0]])
    index=np.array([0,0,0,0])
    mini_r=np.array([t1ra[0],t2ra[0],t3ra[0],t4ra[0]])
    index_r=np.array([0,0,0,0])
    
    
    for i in range(len(t1a)):
        
        #Non Recursive
        if (t1a[i]<mini[0]):
            mini[0]=t1a[i]
            index[0]=i
        if (t2a[i]<mini[1]):
            mini[1]=t2a[i]
            index[1]=i
        if (t3a[i]<mini[2]):
            mini[2]=t3a[i]
            index[2]=i
        if (t4a[i]<mini[3]):
            mini[3]=t4a[i]
            index[3]=i
            
        # Recursive   
        if (t1ra[i]<mini_r[0]):
            mini_r[0]=t1ra[i]
            index_r[0]=i
        if (t2ra[i]<mini_r[1]):
            mini_r[1]=t2ra[i]
            index_r[1]=i
        if (t3ra[i]<mini_r[2]):
            mini_r[2]=t3ra[i]
            index_r[2]=i
        if (t4ra[i]<mini_r[3]):
            mini_r[3]=t4ra[i]
            index_r[3]=i
    
    if (min(mini)<MEAN_train):
        best_rmse=int(np.where(mini==min(mini))[0][0])
    if (min(mini_r)<MEAN_train):
        best_rmse_r=int(np.where(mini_r==min(mini_r))[0][0])
    
    return index,index_r,mini,mini_r,best_rmse,best_rmse_r

def ta_cd():
    from numpy import load
    # load array
    cd1 = load('cd_a1.npy')
    cd2 = load('cd_a2.npy')
    #t_lag=1: NR1a11,NR2a11,NR1a21,NR2a21  :OLS
    #t_lag=15: NR1a12,NR2a12,NR1a22,NR2a22  : OLS
    #t_lag=1: NR1a11,NR2a11,NR1a21,NR2a21  :Ridge
    #t_lag=15: NR1a12,NR2a12,NR1a22,NR2a22  :Ridge
    t1a=np.array([np.mean(cd1[:100,0]),np.mean(cd1[:100,1]),np.mean(cd2[:100,2]),np.mean(rms2[:100,3])])
    t2a=np.array([np.mean(cd1[100:200,0]),np.mean(cd1[100:200,1]),np.mean(cd2[100:200,2]),np.mean(cd2[100:200,3])])
    t3a=np.array([np.mean(cd1[200:300,0]),np.mean(cd1[200:300,1]),np.mean(cd2[200:300,2]),np.mean(cd2[200:300,3])])
    t4a=np.array([np.mean(cd1[300:400,0]),np.mean(cd1[300:400,1]),np.mean(cd2[300:400,2]),np.mean(cd2[300:400,3])])
    
    t1ra=np.array([np.mean(cd1[:100,0]),np.mean(cd1[:100,1]),np.mean(cd2[:100,2]),np.mean(cd2[:100,3])])
    t2ra=np.array([np.mean(cd1[100:200,0]),np.mean(cd1[100:200,1]),np.mean(cd2[100:200,2]),np.mean(cd2[100:200,3])])
    t3ra=np.array([np.mean(cd1[200:300,0]),np.mean(cd1[200:300,1]),np.mean(cd2[200:300,2]),np.mean(cd2[200:300,3])])
    t4ra=np.array([np.mean(cd1[300:400,0]),np.mean(cd1[300:400,1]),np.mean(cd2[300:400,2]),np.mean(cd2[300:400,3])])
    
    mini=np.array([t1a[0],t2a[0],t3a[0],t4a[0]])
    index=np.array([0,0,0,0])
    mini_r=np.array([t1ra[0],t2ra[0],t3ra[0],t4ra[0]])
    index_r=np.array([0,0,0,0])
    
    
    for i in range(len(t1a)):
        
        #Non Recursive
        if (t1a[i]<mini[0]):
            mini[0]=t1a[i]
            index[0]=i
        if (t2a[i]<mini[1]):
            mini[1]=t2a[i]
            index[1]=i
        if (t3a[i]<mini[2]):
            mini[2]=t3a[i]
            index[2]=i
        if (t4a[i]<mini[3]):
            mini[3]=t4a[i]
            index[3]=i
            
        # Recursive   
        if (t1ra[i]<mini_r[0]):
            mini_r[0]=t1ra[i]
            index_r[0]=i
        if (t2ra[i]<mini_r[1]):
            mini_r[1]=t2ra[i]
            index_r[1]=i
        if (t3ra[i]<mini_r[2]):
            mini_r[2]=t3ra[i]
            index_r[2]=i
        if (t4ra[i]<mini_r[3]):
            mini_r[3]=t4ra[i]
            index_r[3]=i
    
    if (min(mini)<MEAN_train):
        best_cd=int(np.where(mini==min(mini))[0][0])
    if (min(mini_r)<MEAN_train):
        best_cd_r=int(np.where(mini_r==min(mini_r))[0][0])
    
    return index,index_r,mini,mini_r,best_cd,best_cd_r
rmse_ia,rmse_iar,rmse_mini_a,rmse_mini_ra,rmse_best_ind_a,rmse_best_ind_ra=ta_rmse()
cd_ia,cd_iar,cd_mini_a,cd_mini_ra,cd_best_ind_a,cd_best_ind_ra=ta_cd()

