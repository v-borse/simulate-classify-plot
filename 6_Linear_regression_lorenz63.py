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


def single_traj(x0,y0,z0,r,dt,num_steps):
    
    pt=[]
    #for single trajectory
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)
        
    # Set initial values
    xs[0], ys[0], zs[0]= (4, -14, 21)
    #xs[0], ys[0], zs[0] = (1., -1., 2.05)
        
        
    for i in range(num_steps):
      """
      EULER SCHEME
            
      dt*x_dot, dt*y_dot, dt*z_dot = lorenz(xs[i], ys[i], zs[i])
      xs[i + 1] = xs[i] + dt*x_dot
      ys[i + 1] = ys[i] + dt*y_dot
      zs[i + 1] = zs[i] + dt*z_dot
      """
      # RK 4 SCHEME
            
      
            
      k1_x, k1_y, k1_z = lorenz(xs[i], ys[i], zs[i], dt,r)
      k2_x, k2_y, k2_z = lorenz(xs[i]+0.5*k1_x, ys[i]+0.5*k1_y, zs[i]+0.5*k1_z, dt,r)
      k3_x, k3_y, k3_z = lorenz(xs[i]+0.5*k2_x, ys[i]+0.5*k2_y, zs[i]+0.5*k2_z, dt,r)
      k4_x, k4_y, k4_z = lorenz(xs[i]+k3_x, ys[i]+k3_y, zs[i]+k3_z, dt,r)
      xs[i + 1] = xs[i] + ((k1_x+2*k2_x+2*k3_x+k4_x) /6.0)
      ys[i + 1] = ys[i] + ((k1_y+2*k2_y+2*k3_y+k4_y) /6.0)
      zs[i + 1] = zs[i] + ((k1_z+2*k2_z+2*k3_z+k4_z) /6.0)
    
        
    
    pt=np.transpose(np.array([xs,ys,zs]))
    
    return pt


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
