#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:54:38 2021

@author: vborse
"""

"""
Created on Thu Mar 18 18:22:02 2021

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

from module import lorenz
from module import single_traj
#from module import ideal
#from module import non_recursive_LN
#from module import recursive_LN
#from module import m_non_recursive_LN
#from module import m_recursive_LN

def ideal(pt, t_steps):
    
    print(pt)
    X = pt[:-t_steps-2]
    Y = pt[1:-t_steps-1]+pt[2:-t_steps]
    
    return X, Y



def ideal3 (pt,t_steps):
    XT = pt[:-t_steps-2]
    XT1= pt[1:-t_steps-1]
    XT2= pt[2:-t_steps]
    Y  = pt[1:-t_steps-1]+pt[2:-t_steps]
    
    return XT,XT1,XT2,Y

def plot_predicted_ts(X2,Y2, Yp, index):
    
    fig, axs = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(15, 5))
    fig.suptitle(' Linear Regression; dt=0.01')
    
    axs[0,0].scatter(Yp, Y2[:, index],s=2)
    axs[0,0].plot(Y2[:, index],Y2[:, index],'r')
    axs[0,0].set_xlim([-50,50])
    axs[0,0].set_ylim([-50,50])
    axs[0,0].set_xlabel("y_pred")
    axs[0,0].set_ylabel("y_ideal")
    axs[0,0].set_title("Non-recursive LN")
    
           
    axs[1,0].scatter(X2[:, index], Y2[:, index], s=2)
    axs[1,0].scatter(X2[:, index], Yp, s=2)
    
    axs[1,0].set_xlim([-50,50])
    axs[1,0].set_ylim([-50,50])
    axs[1,0].set_xlabel("X(x,y,z)")
    axs[1,0].set_ylabel("Y")
    axs[1,0].set_title("X and Y")
    
    axs[1,1].plot(X2[:, index])
    axs[1,1].plot(Y2[:, index])
    axs[1,1].plot(Yp)
    
    axs[1,1].set_ylim([-60,60])
    axs[1,1].set_xlabel("Time series")
    
    
    
    axs[2,0].plot(Yp-Y2[:, index], 'b')
    
    axs[2,0].set_ylim([-50,50])
    axs[2,0].set_ylabel("error")
    axs[2,0].set_xlabel("time_steps")
    axs[2,0].set_title("Errors for LN")
    
    
    
    axs[2,1].plot(Yp-Y2[:, index], 'b')
    
    axs[2,1].set_ylim([-50,150])
    axs[2,1].set_ylabel("squared error")
    axs[2,1].set_xlabel("time_steps")
    axs[2,1].set_title("Squared Errors for LN")
    
    plt.show()
    #plt.scatter(y_pred1, y_pred2)


def plot_traj(X2,Y2,YPx,YPy,YPz):
    
    fig1 = plt.figure(figsize=(15,10))
    ax1 = fig1.add_subplot(131, projection='3d')
    #ax1.scatter(y_pred10, y_pred11, y_pred12,color='g', alpha=1)
    ax1.scatter(X2.T[0], X2.T[1], X2.T[2],color='r', alpha=.01)
    ax1.scatter(Y2.T[0], Y2.T[1], Y2.T[2],color='b', alpha=.009)
    
    ax1.set_xlabel('X[0] ')
    ax1.set_ylabel('X[1] ')
    ax1.set_zlabel('X[2] ')
    
    
    
    fig1 = plt.figure(figsize=(15,10))
    ax1 = fig1.add_subplot(132, projection='3d')
    
    
    ax1.scatter(Y2.T[0], Y2.T[1], Y2.T[2],color='b',alpha=0.009)
    ax1.set_xlabel('Y2[0]')
    ax1.set_ylabel('Y2[1]')
    ax1.set_zlabel('Y2[2]')
    
    
    
    #fig2 = plt.figure()
    #ax2 = fig2.add_subplot(111, projection='3d')
    fig1 = plt.figure(figsize=(15,10))
    ax1 = fig1.add_subplot(133, projection='3d')
    ax1.scatter(YPx, YPy, YPz, color='g',alpha=0.05)
    ax1.scatter(Y2.T[0], Y2.T[1], Y2.T[2],color='b',alpha=0.009)
    ax1.set_xlabel('y_pred1[0]; non recursive')
    ax1.set_ylabel('y_pred1[1]')
    ax1.set_zlabel('y_pred1[2]')
    plt.show()
    
    
    
import pandas as pd 
from sklearn import linear_model
import statsmodels.api as sm

r=28
R=1
tlength = 10000
t_steps = 50
pts=single_traj(4,-14,21,r,0.01,tlength) 
pts2=single_traj(1,-1,2.05,r,0.01,tlength)

X1, Y1 = ideal(pts,1)
X2, Y2 = ideal(pts2,t_steps)
XT,XT1,XT2, Y = ideal3(pts,t_steps)
# intialise data of lists. 
data = {'XTx':XT[:,0], 
        'XT1x':XT1[:,0],
        'XT2x':XT2[:,0],
        'YTx':Y[:,0],
        'XTy':XT[:,1], 
        'XT1y':XT1[:,1],
        'XT2y':XT2[:,1],
        'YTy':Y[:,1],
        'XTz':XT[:,2], 
        'XT1z':XT1[:,2],
        'XT2z':XT2[:,2],
        'YTz':Y[:,2]} 
  
# Create DataFrame 
df = pd.DataFrame(data) 



# prediction with sklearn
xx=df[['XT1x','XT2x']] #Univariate
xy=df[['XT1y','XT2y']]
xz=df[['XT1z','XT2z']]
#x=df[['XT1x','XT2x','XT1y','XT2y','XT1z','XT2z']] # Multivariate

# prediction with sklearn

#x=df[['XT1x','XT2x','XT1y','XT2y','XT1z','XT2z']]

yx=df['YTx']
yy=df['YTy']
yz=df['YTz']
regr_x = linear_model.LinearRegression()
regr_x.fit(xx, yx)

print('Intercept: \n', regr_x.intercept_)
print('Coefficients: \n', regr_x.coef_)

# with statsmodels
xx = sm.add_constant(xx) # adding a constant
 
modelx = sm.OLS(yx, xx).fit()
predictions_x = modelx.predict(xx) 
 
print_modelx = modelx.summary()
print(print_modelx)
#--------------------------------------
regr_y = linear_model.LinearRegression()
regr_y.fit(xy, yy)

print('Intercept: \n', regr_y.intercept_)
print('Coefficients: \n', regr_y.coef_)

# with statsmodels
xy = sm.add_constant(xy) # adding a constant
 
modely = sm.OLS(yy, xy).fit()
predictions_y = modely.predict(xy) 
 
print_modely = modely.summary()
print(print_modely)

#--------------------------------------
regr_z = linear_model.LinearRegression()
regr_z.fit(xz, yz)

print('Intercept: \n', regr_z.intercept_)
print('Coefficients: \n', regr_z.coef_)

# with statsmodels
xz = sm.add_constant(xz) # adding a constant
 
modelz = sm.OLS(yz, xz).fit()
predictions_z = modelz.predict(xz) 
 
print_modelz = modelz.summary()
print(print_modelz)
  

#plt.plot(predictions)
#plt.plot(df.YTx,'r')
#plt.plot(df.YTx-predictions,'k')

#DATA_x = ([1, 2],
#          [2, 3],
#          [3, 4])
#
#DATA_y = DATA_x[::-1]
#
#DATA_z = DATA_x[::-1]
#
#XLIMS = [[-10, 10]] * 5
#YLIMS = [[-10, 10]] * 5
#ZLIMS = [[0, 10]] * 5
#
#for j, (x, y, z , xlim, ylim, zlim) in enumerate(zip(DATA_x, DATA_y, DATA_z, XLIMS, YLIMS, ZLIMS)):
#    fig1 = plt.figure()
#    #ax1 = fig1.add_subplot(111, projection='3d')
#    ax = plt.subplot(1, 3, j + 1,projection='3d')
#    ax.scatter(x, y, z)
#    ax.set_xlim(xlim)
#    ax.set_ylim(ylim)
#    ax.set_zlim(zlim)


#model.(sm.OLS(Y.T,df1[['XT1.T','XT2.T']]).fit()

#plt.plot(predictions)
#plt.plot(df.YTx,'r')
#plt.plot(df.YTx-predictions,'k')
plot_predicted_ts(X2,Y2,predictions_x,0)
plot_predicted_ts(X2,Y2,predictions_y,1)
plot_predicted_ts(X2,Y2,predictions_z,2)
#plot_traj(X2,Y2,predictions_x,predictions_y,predictions_z)
