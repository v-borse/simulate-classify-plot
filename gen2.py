#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:35:38 2021

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
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn import linear_model
import statsmodels.api as sm
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from module import lorenz
from module import single_traj

def plot_predicted_ts(X,Y, Yp, Yrp,index):
    fig, axs = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(15, 15))
    fig.suptitle(' Linear Regression; dt=0.01')
    
    
    axs[0,0].scatter(Yp, Y,c='k',s=2)
    axs[0,0].plot(Y,Y,'g')
    axs[0,0].legend()
    #axs[0,0].set_xlim([-50,50])
    #axs[0,0].set_ylim([-50,50])
    axs[0,0].set_xlabel("y_pred")
    axs[0,0].set_ylabel("y_ideal")
    axs[0,0].set_title("Non-recursive LN")
    
    axs[0,1].scatter(Yrp[:, index], Y,c='k', s=2)
    axs[0,1].plot(Y, Y, 'g')
    #axs[0,1].set_xlim([-50,50])
    #axs[0,1].set_ylim([-50,50])
    axs[0,1].set_xlabel("y_pred")
    axs[0,1].set_ylabel("y_ideal")
    axs[0,1].set_title("Recursive LN")
      
    
    axs[1,0].plot((Yp-Y), 'b')
    axs[1,0].set_ylabel("error")
    axs[1,0].set_xlabel("time_steps")
    axs[1,0].set_title("Errors for non-recursive LN")
    
    axs[1,1].plot(Yrp[:, index]-Y, 'r')
    axs[1,1].set_ylabel("error")
    axs[1,1].set_xlabel("time_steps")
    axs[1,1].set_title("Errors for recursive LN")
     
    
    axs[2,0].plot(Yp[:100],'b^',Yrp[:100, index],'r.')
    axs[2,0].set_xlabel("predicted time series")
    
    axs[2,1].plot(Yp[500:600],'b^')
    axs[2,1].plot(Yrp[500:600,index],'r.')
    axs[2,1].set_xlabel("predicted time series")
       
    plt.show()
    

def gen_R_predict_multi(_X,modelx,modely,modelz,t_steps):
    _X=np.copy(_X) 
    for i in range(1,t_steps+1):
        
        Yrx=modelx.predict(_X[:,[0,1,2,3,4,5]])
        Yry=modely.predict(_X[:,[0,1,2,3,4,5]])
        Yrz=modelz.predict(_X[:,[0,1,2,3,4,5]])
        
        _X[:,0] = _X[:,3]
        _X[:,1] = _X[:,4]
        _X[:,2] = _X[:,5]
        
        _X[:,3] = Yrx
        _X[:,4] = Yry
        _X[:,5] = Yrz
        
    Xnew = np.array([Yrx,Yry,Yrz])
    return Xnew.T

def Ideal_lags(_X,tsteps,tlags):
    Xti=_X[:-(tsteps+tlags)]
    for i in range(1,tlags+1):
        
        Xti=np.concatenate((Xti, _X[i:-(tsteps+tlags-i)]), axis=1)
        #print(np.shape(Xti))
    
    Y  =_X[(tsteps+tlags):]
    
    return(Xti,Y)      

def grouped_col(ncol,t_lags,order=1):
    
    ff= np.arange(0,(ncol*(t_lags+1)),1)
    F=np.reshape(ff,((t_lags+1),ncol))
    
    return F


def GEN_R_predict_multi(_X,modelx,modely,modelz,t_steps,t_lags,ncol,order=1):
    
    _X = np.copy(_X)
    c=grouped_col(ncol,t_lags,order)
    print(_X)
    for i in range(1,t_steps+1):
        
        Ynrx=modelx.predict(_X)
        Ynry=modely.predict(_X)
        Ynrz=modelz.predict(_X)
        
        Xnew=np.array([Ynrx,Ynry,Ynrz])
        #print(np.shape(Xnew[:,:,None]))
        
        #Xnew=Xnew.reshape((len(Ynrx),1,ncol))
        #print(np.shape(Xnew[0]))
        #print(np.shape(Xnew.T))
        #print(np.shape(Xnew.T[:,:,None]))
        for k in range(t_lags+1):
            #print(k)
            
            if ((k+1)<t_lags):
                
                _X[:,[c[k]]] = _X[:,[c[k+1]]]
                #print(np.shape(_X[:,k]))
            #_X[:,-(k+1)] = Xnew[k]
            _X[:,c[-1]] = Xnew.T
        print(_X[:,c[-1]])
        print(Xnew.T)
    return Xnew.T




r=28
R=1
tlength = 10000

#N=len(pts)
#P=3

order=3
t_steps=1
t_lags=2
ncol=3

pts=single_traj(4,-14,21,r,0.01,tlength) 
pts2=single_traj(1,-1,2.05,r,0.01,tlength)
N=len(pts)

Xr_t1,Yr_t1 = Ideal_lags(pts,1,t_lags)
Xtrain, Ytrain = Ideal_lags(pts,t_steps,t_lags)
Xtest, Ytest = Ideal_lags(pts2,t_steps,t_lags)

#Xpr_t1,Ypr_t1 = Ideal_poly(pts,2,t_steps)
#Xp_train, Yp_train = Ideal_poly(pts,order,t_steps)
#Xp_test, Yp_test = Ideal_poly(pts2,order,t_steps)


#============MULTIVARIATE======================
c = grouped_col(3,t_lags,order)
#---------------Non-Recursive-----------------

model_x=linear_model.LinearRegression()
model_x.fit(Xtrain, Ytrain[:,0])

model_y=linear_model.LinearRegression()
model_y.fit(Xtrain, Ytrain[:,1])

model_z=linear_model.LinearRegression()
model_z.fit(Xtrain, Ytrain[:,2])

#====predict==================================

YNRx=model_x.predict(Xtest)
YNRy=model_y.predict(Xtest)
YNRz=model_z.predict(Xtest)

#----RECURSIVE-------------------------------------

Rx=linear_model.LinearRegression()
Rx.fit(Xr_t1, Yr_t1[:,0])

Ry=linear_model.LinearRegression()
Ry.fit(Xr_t1, Yr_t1[:,1])

Rz=linear_model.LinearRegression()
Rz.fit(Xr_t1, Yr_t1[:,2])

YR = GEN_R_predict_multi(Xtest,Rx,Ry,Rz,t_steps,t_lags,ncol,1)


#plot_predicted_ts(Xtest, Ytest[:,0],YNRx,YR,0)
#plot_predicted_ts(Xtest, Ytest[:,1],YNRy,YR,1)
#plot_predicted_ts(Xtest, Ytest[:,2],YNRz,YR,2)
#    
