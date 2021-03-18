#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    X = pt[:-t_steps]
    Y = pt[t_steps:]
    
    return X, Y

def r_predict(Xt,t_steps):
    Xnew = Xt

    for i in range(t_steps):
        if i == 0:
            Xnew = Xt
            Xnew0= rx.predict(Xnew[:, 0, None].reshape(-1, 1))
            Xnew1= ry.predict(Xnew[:, 1, None].reshape(-1, 1))
            Xnew2= rz.predict(Xnew[:, 2, None].reshape(-1, 1))
        else:
            print(np.shape(Xnew[:, 0, None]))
            print(np.shape(Xnew0[None].T))
            print(Xnew0)
            Xnew[:, 0, None]=Xnew0[None].T
            Xnew[:, 1, None]=Xnew1[None].T
            Xnew[:, 2, None]=Xnew2[None].T
            Xnew0= rx.predict(Xnew[:, 0, None].reshape(-1, 1))
            Xnew1= ry.predict(Xnew[:, 1, None].reshape(-1, 1))
            Xnew2= rz.predict(Xnew[:, 2, None].reshape(-1, 1))
            
            return Xnew
        

def R_predict(Xt,t_steps):
    Xnew = Xt
    
    for i in range(t_steps):
        if i == 0:
            Xnew = Xt
            Xnew0= Rx.predict(Xnew)
            Xnew1= Ry.predict(Xnew)
            Xnew2= Rz.predict(Xnew)
            
        else:
            
            
            Xnew[:, 0, None]=Xnew0
            Xnew[:, 1, None]=Xnew1
            Xnew[:, 2, None]=Xnew2
            Xnew0= Rx.predict(Xnew)
            Xnew1= Ry.predict(Xnew)
            Xnew2= Rz.predict(Xnew)
            
    return Xnew
            

def plot1(X2,Y2, Yp, Yrp,index):
    
    fig, axs = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(15, 5))
    fig.suptitle(' Linear Regression; dt=0.01')
    
    axs[0,0].scatter(Yp, Y2[:, index],s=2)
    axs[0,0].plot(Y2[:, index],Y2[:, index],'r')
    axs[0,0].set_xlabel("y_pred")
    axs[0,0].set_ylabel("y_ideal")
    axs[0,0].set_title("Non-recursive LN")
    
    axs[0,1].scatter(Yrp[:, index], Y2[:, index], s=2)
    axs[0,1].plot(Y2[:, index], Y2[:, index], 'r')
    axs[0,1].set_xlabel("y_pred")
    axs[0,1].set_ylabel("y_ideal")
    axs[0,1].set_title("Recursive LN")
    
    axs[1,0].plot(Yp-Y2[:, index], 'b')
    axs[1,0].plot(Yrp[:, index]-Y2[:, index], 'r')
    axs[1,0].set_ylabel("error")
    axs[1,0].set_xlabel("time_steps")
    axs[1,0].set_title("Errors for LN")
    
    axs[1,1].scatter(X2[:, index], Y2[:, index], s=2)
    axs[1,1].scatter(X2[:, index], Yp, s=2)
    axs[1,1].scatter(X2[:, index], Yrp[:, index], s=2) 
    axs[1,1].set_xlabel("X(x,y,z)")
    axs[1,1].set_ylabel("Y")
    plt.show()
    
    plt.plot(X2[:, index])
    plt.plot(Y2[:, index])
    plt.plot(Yp)
    plt.plot(Yrp[:, index])
    plt.xlabel("Time series")
    plt.show()
    
    
    #plt.scatter(y_pred1, y_pred2)
    
def plot2(X2,Y2,YPx,YPy,YPz,YRP):
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    #ax1.scatter(y_pred10, y_pred11, y_pred12,color='g', alpha=1)
    ax1.scatter(X2.T[0], X2.T[1], X2.T[2],color='b', alpha=.01)
    ax1.set_xlabel('X[0] ')
    ax1.set_ylabel('X[1] ')
    ax1.set_zlabel('X[2] ')
    
    
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(Y2.T[0], Y2.T[1], Y2.T[2],color='r')
    ax1.set_xlabel('Y2[0]')
    ax1.set_ylabel('Y2[1]')
    ax1.set_zlabel('Y2[2]')
    
    
    
    #fig2 = plt.figure()
    #ax2 = fig2.add_subplot(111, projection='3d')
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(YRP[0], YRP[1], YRP[2],color='y')
    ax1.set_xlabel('y_pred2[0]; recursive')
    ax1.set_ylabel('y_pred2[1]')
    ax1.set_zlabel('y_pred2[2]')
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(YPx, YPy, YPz, color='g')
    ax1.set_xlabel('y_pred1[0]; non recursive')
    ax1.set_ylabel('y_pred1[1]')
    ax1.set_zlabel('y_pred1[2]')
    plt.show()
                
    
#---------MAIN BODY OF THE CODE------------------

r=28
R=1
tlength = 10000

pts=single_traj(4,-14,21,r,0.01,tlength) 
pts2=single_traj(1,-1,2.05,r,0.01,tlength)

t_steps = 5

index = 0


"""--------UNIVARIATE----------"""

X1, Y1 = ideal(pts,t_steps)
X2, Y2 = ideal(pts2,t_steps)

#--------NON-RECURSIVE (MODEL)-------------
nrx = LinearRegression().fit(X1[:, 0, None], Y1[:, 0])
nry = LinearRegression().fit(X1[:, 1, None], Y1[:, 1])
nrz = LinearRegression().fit(X1[:, 2, None], Y1[:, 2])

nrx_i = nrx.intercept_
nrx_c = nrx.coef_
nrx_rsq = nrx.score(X1[:, 0, None], Y1[:, 0])

nry_i = nry.intercept_
nry_c = nry.coef_
nry_rsq = nry.score(X1[:, 1, None], Y1[:, 1])

nrz_i = nrz.intercept_
nrz_c = nrz.coef_
nrz_rsq = nrz.score(X1[:, 2, None], Y1[:, 2])

#---------NON-RECURSIVE (PREDICT)----------------------
Ynrx =nrx.predict(X2[:, 0, None])
Ynry =nry.predict(X2[:, 1, None])
Ynrz =nrz.predict(X2[:, 2, None])

Ynrx_rsq = nrx.score(X1[:, 0, None], Ynrx)
Ynry_rsq = nrx.score(X1[:, 1, None], Ynry)
Ynrz_rsq = nrx.score(X1[:, 2, None], Ynrz)


#---------RECURSIVE  (MODEL)--------------------

rx = LinearRegression().fit(X1[:, 0, None], Y1[:, 0])
ry = LinearRegression().fit(X1[:, 1, None], Y1[:, 1])
rz = LinearRegression().fit(X1[:, 2, None], Y1[:, 2])

rx_i = rx.intercept_
rx_c = rx.coef_
rx_rsq = rx.score(X1[:, 0, None], Y1[:, 0])

ry_i = ry.intercept_
ry_c = ry.coef_
ry_rsq = ry.score(X1[:, 1, None], Y1[:, 1])

rz_i = rz.intercept_
rz_c = rz.coef_
rz_rsq = rz.score(X1[:, 2, None], Y1[:, 2])

Yr = r_predict(X2,t_steps)

#---------Plot1--------------------------

plot1(X2,Y2,Ynrx,Yr,0)
plot1(X2,Y2,Ynry,Yr,1)
plot1(X2,Y2,Ynrz,Yr,2)

#-------Plot2--------------------------------
plot2(X2,Y2,Ynrx,Ynry,Ynrz,Yr)



"""----------------MULTIVARIATE----------------"""

X1, Y1 = ideal(pts,t_steps)
X2, Y2 = ideal(pts2,t_steps)

#--------NON-RECURSIVE (MODEL)-------------

NRx = LinearRegression().fit(X1[:, :], Y1[:, 0, None])
NRy = LinearRegression().fit(X1[:, :], Y1[:, 1, None])
NRz = LinearRegression().fit(X1[:, :], Y1[:, 2, None])

NRx_i = NRx.intercept_
NRx_c = NRx.coef_
NRx_rsq = NRx.score(X1[:, :], Y1[:, 0, None])

NRy_i = NRy.intercept_
NRy_c = NRy.coef_
NRy_rsq = NRy.score(X1[:, :], Y1[:, 1, None])

NRz_i = NRz.intercept_
NRz_c = NRz.coef_
NRz_rsq = NRz.score(X1[:, :], Y1[:, 2, None])

#---------NON-RECURSIVE (PREDICT)----------------------

YNRx =NRx.predict(X2)
YNRy =NRy.predict(X2)
YNRz =NRz.predict(X2)

YNRx_rsq = NRx.score(X2[:, :], Y2[:, 0, None], YNRx)
YNRy_rsq = NRy.score(X2[:, :], Y2[:, 1, None], YNRy)
YNRz_rsq = NRz.score(X2[:, :], Y2[:, 2, None], YNRz)


#---------RECURSIVE  (MODEL)--------------------

Rx= LinearRegression().fit(X1[:, :], Y1[:, 0, None])
Ry= LinearRegression().fit(X1[:, :], Y1[:, 1, None])
Rz= LinearRegression().fit(X1[:, :], Y1[:, 2, None])

Rx_i = Rx.intercept_
Rx_c = Rx.coef_
Rx_rsq = Rx.score(X1[:, :], Y1[:, 0, None])

Ry_i = Ry.intercept_
Ry_c = Ry.coef_
Ry_rsq = Ry.score(X1[:, :], Y1[:, 1, None])

Rz_i = Rz.intercept_
Rz_c = Rz.coef_
Rz_rsq = Rz.score(X1[:, :], Y1[:, 2, None])

YR = R_predict(X2,t_steps)

#-----------------PLOT1----------------------

plot1(X2,Y2,YNRx,YR,0)
plot1(X2,Y2,YNRy,YR,1)
plot1(X2,Y2,YNRz,YR,2)

#---------------PLOT2---------------------------

plot2(X2,Y2,YNRx,YNRy,YNRz,YR)
