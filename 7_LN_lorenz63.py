#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 15:38:43 2021

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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from module mport lorenz
from module import single_traj
from module import ideal
from module import non_recursive_LN
from module import recursive_LN
from module import m_non_recursive_LN
from module import m_recursive_LN
    
# ------------------MAIN BODY OF THE CODE-----------------------------    
r=28
R=1
pts=single_traj(4,-14,21,r,0.01,10000) 
#with open('pts.npy', 'wb') as h:
#    np.save(h, pts)
#pts=np.load('pts.npy')
X1=np.transpose(pts)


pts2=single_traj(1,-1,2.05,r,0.01,10000)
X2=np.transpose(pts2)

Y1=[]
Y2=[]
y_pred=[]
y_pred1=[]
y_pred2=[]

t_steps=50
index=0
# index can take any value (0,1,2) corresponding to X(x,y,z)

#------------UNIVARIATE LINEAR REGRESSION ---------------------------------


X1, Y1= ideal(pts,t_steps)
X2, Y2= ideal(pts2,t_steps)
i1,c1,y_pred1=non_recursive_LN (X2[index], t_steps,index,pts2)
i2,c2,y_pred2=recursive_LN (X2[index], t_steps,index,pts2)

fig, axs = plt.subplots(2, 2, sharex=False,sharey=False, figsize=(15,5))
fig.suptitle('Univariate Linear Regression; dt=0.01')
axs[0,0].scatter(y_pred1[:-(t_steps+1)],Y2[index][:-(t_steps+1)],s=2)
axs[0,0].plot(Y2[index][:-(t_steps+1)],Y2[index][:-(t_steps+1)],'r')
axs[0,0].set_xlabel("y_pred")
axs[0,0].set_ylabel("y_ideal")
axs[0,0].set_title("Non-recursive LN")
    
axs[0,1].scatter(y_pred2[:-(t_steps+1)],Y2[index][:-(t_steps+1)],s=2)
axs[0,1].plot(Y2[index][:-(t_steps+1)],Y2[index][:-(t_steps+1)],'r')
axs[0,1].set_xlabel("y_pred")
axs[0,1].set_ylabel("y_ideal")
axs[0,1].set_title("Recursive LN")
    
axs[1,0].plot(y_pred1[:-(t_steps+1)]-Y2[index][:-(t_steps+1)],'b')
axs[1,0].plot(y_pred2[:-(t_steps+1)]-Y2[index][:-(t_steps+1)],'r')
axs[1,0].set_ylabel("error")
axs[1,0].set_xlabel("time_steps")
axs[1,0].set_title("Errors for LN")
    
axs[1,1].scatter(X2[index][:-(t_steps+1)],Y2[index][:-(t_steps+1)],s=2)
axs[1,1].scatter(X2[index][:-(t_steps+1)],y_pred1[:-(t_steps+1)],s=2)
axs[1,1].scatter(X2[index][:-(t_steps+1)],y_pred2[:-(t_steps+1)],s=2)
axs[1,1].set_xlabel("X(x,y,z)")
axs[1,1].set_ylabel("Y")

#---------------MULTIVARIATE LINEAR REGRESSION----------------------------------------


X1, Y1= ideal(pts,t_steps)
X2, Y2= ideal(pts2,t_steps)
i1,c1,y_pred1=m_non_recursive_LN (X2, t_steps,index,pts2)
i2,c2,y_pred2=m_recursive_LN (X2, t_steps,index,pts2)

fig, axs = plt.subplots(2, 2, sharex=False,sharey=False, figsize=(15,5))
fig.suptitle('Multivariate Linear Regression; dt=0.01')
axs[0,0].scatter(y_pred1[:-(t_steps+1)],Y2[index][:-(t_steps+1)],s=2)
axs[0,0].plot(Y2[index][:-(t_steps+1)],Y2[index][:-(t_steps+1)],'r')
axs[0,0].set_xlabel("y_pred")
axs[0,0].set_ylabel("y_ideal")
axs[0,0].set_title("Non-recursive LN")
       
axs[0,1].scatter(y_pred2[:-(t_steps+1)],Y2[index][:-(t_steps+1)],s=2)
axs[0,1].plot(Y2[index][:-(t_steps+1)],Y2[index][:-(t_steps+1)],'r')
axs[0,1].set_xlabel("y_pred")
axs[0,1].set_ylabel("y_ideal")
axs[0,1].set_title("Recursive LN")
       
axs[1,0].plot(y_pred1[:-(t_steps+1)]-Y2[index][:-(t_steps+1)],'b')
axs[1,0].plot(y_pred2[:-(t_steps+1)]-Y2[index][:-(t_steps+1)],'r')
axs[1,0].set_ylabel("error")
axs[1,0].set_xlabel("time_steps")
axs[1,0].set_title("Errors for LN")
      
axs[1,1].scatter(X2[index][:-(t_steps+1)],Y2[index][:-(t_steps+1)],s=2)
axs[1,1].scatter(X2[index][:-(t_steps+1)],y_pred1[:-(t_steps+1)],s=2)
axs[1,1].scatter(X2[index][:-(t_steps+1)],y_pred2[:-(t_steps+1)],s=2)
axs[1,1].set_xlabel("X(x,y,z)")
axs[1,1].set_ylabel("Y")
