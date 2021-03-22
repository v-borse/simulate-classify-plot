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

def ideal2(pt, t_steps):
    
    print(pt)
    X = pt[:-t_steps]
    Y = pt[t_steps:]
    
    return X, Y

def ideal3 (pt,t_steps):
    XT = pt[:-t_steps-2]
    XT1= pt[1:-t_steps-1]
    XT2= pt[2:-t_steps]
    Y  = pt[1:-t_steps-1]+pt[2:-t_steps]
    
    return XT,XT1,XT2,Y

import pandas as pd 
from sklearn import linear_model
import statsmodels.api as sm

r=28
R=1
tlength = 10000
t_steps = 5
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
x=df[['XT1x','XT2x']] #Univariate
x=df[['XT1x','XT2x','XT1y','XT2y','XT1z','XT2z']] # Multivariate
y=df['YTx']
regr = linear_model.LinearRegression()
regr.fit(x, y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
x = sm.add_constant(x) # adding a constant
 
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
 
print_model = model.summary()
print(print_model)
  
#plt.plot(predictions)
#plt.plot(df.YTx,'r')
plt.plot(df.YTx-predictions,'k')

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



