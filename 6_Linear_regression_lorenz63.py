#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jan 29 04:25:27 2021

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
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

from module import lorenz
from module import single_traj



#------MAIN BODY of the code -----------------------------------------------------------------------------------------------------------
r=28
R=1
pts=single_traj(4,-14,21,r,0.001,100000)

with open('pts.npy', 'wb') as h:
    np.save(h, pts)
   
pts=np.load('pts.npy')
X=np.transpose(pts)

covMatrix = np.corrcoef(X,bias=True)
y=np.dot(covMatrix,X)
Y=np.dot(covMatrix,X)
Y2=np.dot(covMatrix,X2)
y_pred=np.dot(covMatrix,X)

for i,item in enumerate(X):
    for j,jitem in enumerate(X[i]):
        if (j!=(len(X[i])-1)):
            Y[i][j]=X[i][j+1]
        else:
            Y[i][-1]=0



pts2=single_traj(1,-1,2.05,r,0.001,100000)
X2=np.transpose(pts2)


for i,item in enumerate(X2):
    for j,jitem in enumerate(X2[i]):
        if (j!=(len(X[i])-1)):
            Y2[i][j]=X2[i][j+1]
        else:
            Y2[i][-1]=0


for i in range(3):
    model = LinearRegression().fit(X.T, Y[i])
    #model = LinearRegression().fit(X[2].reshape(-1,1), Y[0])
    r_sq = model.score(X.T, Y[i])
    #r_sq = model.score(X[2].reshape(-1,1), Y[2])
    y_pred2= model.predict(X2.T)
    #y_pred= model.predict(X[2].reshape(-1,1))
    intercept, coefficients = model.intercept_, model.coef_

    #---------------------------------------------------------------------------------------

    x_labels=[0.1,0.5,1,2,5,10]
    y_labels=[0,0.2,0.4,0.6,0.8,1.0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    #fig.suptitle(string1.format(c0,c1,c2,indir_01,indir_02,indir_21,r,R))
    fig.suptitle("Linear regression for X")
    
    sns.scatterplot(ax=axes[0], x=y_pred2[:-2], y=Y2[i][:-2])
    axes[0].set_title("x= y_pred; y=Y_ideal")
    
    
    sns.scatterplot(ax=axes[1], x=Y2[i][:-2], y=Y2[i][:-2])
    axes[1].set_title("x=Y_ideal; y=Y_ideal")
    
    
    #sns.plot(ax=axes[2], x=y_pred2[:-2]-Y2[2][:-2],y=np.arange(10000))
    #axes[2].set_title("x=delta t; error")
    
    plt.plot(y_pred2[:-2]-Y2[i][:-2],'k')
    plt.xlabel('time_steps')
    plt.ylabel('error')
    plt.title('Linear regression')
