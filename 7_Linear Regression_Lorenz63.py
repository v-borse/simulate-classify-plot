#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 16:28:15 2021

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

from module2 import grouped_col_multi,grouped_col_uni2
#from module import lorenz
#from module import single_traj
#
#from toy_model import Ideal_poly
#from toy_model import grouped_col_multi,swap, swap3,grouped_col_uni2,Ideal_poly3,Ideal_poly4


def plot_ts(Ytrue,ynrx,ynry,ynrz,Yr,YNRX,YNRY,YNRZ,YR):
    
    fig, axs = plt.subplots(6, 2, sharex=False, sharey=False, figsize=(15, 15))
    fig.suptitle(' Linear Regression; dt=0.01')
    
    axs[0,0].plot(Ytrue[:,0],'b')
    axs[0,0].plot(ynrx,'r')
    #axs[0,0].plot(Ytrue[:,0]-ynrx,'g')
    axs[0,0].set_xlabel("timesteps")
    axs[0,0].set_ylabel("blue:Ytrue ; red: pred" )
    axs[0,0].set_title("X component NR /Univariate")
    
    axs[1,0].plot(Ytrue[:,1],'b')
    axs[1,0].plot(ynry,'r')
    axs[1,0].set_xlabel("timesteps")
    axs[1,0].set_ylabel("blue:Ytrue ; red: pred" )
    axs[1,0].set_title("Y component NR /Univariate")
    
    axs[2,0].plot(Ytrue[:,2],'b')
    axs[2,0].plot(ynrz,'r')
    axs[2,0].set_xlabel("timesteps")
    axs[2,0].set_ylabel("blue:Ytrue ; red: pred" )
    axs[2,0].set_title("Z component NR /Univariate")
    
    axs[0,1].plot(Ytrue[:,0],'b')
    axs[0,1].plot(YNRX,'r')
    axs[0,1].set_xlabel("timesteps")
    axs[0,1].set_ylabel("blue:Ytrue ; red: pred" )
    axs[0,1].set_title("X component NR /Multivariate")
    
    axs[1,1].plot(Ytrue[:,1],'b')
    axs[1,1].plot(YNRY,'r')
    axs[1,1].set_xlabel("timesteps")
    axs[1,1].set_ylabel("blue:Ytrue ; red: pred" )
    axs[1,1].set_title("Y component NR /Multivariate")
    
    axs[2,1].plot(Ytrue[:,2],'b')
    axs[2,1].plot(YNRZ,'r')
    axs[2,1].set_xlabel("timesteps")
    axs[2,1].set_ylabel("blue:Ytrue ; red: pred" )
    axs[2,1].set_title("Z component NR /Multivariate")
    
    axs[3,0].plot(Ytrue[:,0])
    axs[3,0].plot(Yp[:,0])
    axs[3,0].set_xlabel("timesteps")
    axs[3,0].set_ylabel("blue:Ytrue ; orange: pred" )
    axs[3,0].set_title("X component R /Univariate")
    
    axs[4,0].plot(Ytrue[:,1])
    axs[4,0].plot(Yp[:,1])
    axs[4,0].set_xlabel("timesteps")
    axs[4,0].set_ylabel("blue:Ytrue ; orange: pred" )
    axs[4,0].set_title("Y component R /Univariate")
    
    axs[5,0].plot(Ytrue[:,2])
    axs[5,0].plot(Yp[:,2])
    axs[5,0].set_xlabel("timesteps")
    axs[5,0].set_ylabel("blue:Ytrue ; orange: pred" )
    axs[5,0].set_title("Z component R /Univariate")
    
    axs[3,1].plot(Ytrue[:,0])
    axs[3,1].plot(YP[:,0])
    axs[3,1].set_xlabel("timesteps")
    axs[3,1].set_ylabel("blue:Ytrue ; orange: pred" )
    axs[3,1].set_title("X component R /Multivariate")
    
    axs[4,1].plot(Ytrue[:,1])
    axs[4,1].plot(YP[:,1])
    axs[4,1].set_xlabel("timesteps")
    axs[4,1].set_ylabel("blue:Ytrue ; orange: pred" )
    axs[4,1].set_title("Y component R /Multivariate")
    
    axs[5,1].plot(Ytrue[:,2])
    axs[5,1].plot(YP[:,2])
    axs[5,1].set_xlabel("timesteps")
    axs[5,1].set_ylabel("blue:Ytrue ; orange: pred" )
    axs[5,1].set_title("Z component R /Multivariate")
    
    fig.tight_layout()
    
    
def Ideal_lags(_X,tsteps,tlags):
    Xti=_X[:-(tsteps+tlags)]
    for i in range(1,tlags+1):
        
        Xti=np.concatenate((Xti, _X[i:-(tsteps+tlags-i)]), axis=1)
            
    Y  =_X[(tsteps+tlags):]
    
    return(Xti,Y)   
    
def swap_multi(_X,_Y,modelx,modely,modelz,t_steps,t_lags,ncol,order):
    
      
    _X = np.copy(_X)
    
    c=grouped_col_multi(ncol,t_lags,order)

    for i in range(1,t_steps+1):
               
        Ynrx = modelx.predict(_X)
        Ynry = modely.predict(_X)
        Ynrz = modelz.predict(_X)
        
        Xnew=np.array([Ynrx,Ynry,Ynrz])
               
        for k in range(t_lags+1):
          
            
            if ((k+1)<=t_lags):
                
                _X.T[:,[c[k]]] = _X.T[:,[c[k+1]]]
               
            else:
                
                xx=Ideal_poly3(Xnew.T,order,t_steps)              
                _X[:,c[-1]] = xx
       
    return _X[:,c[-1]]

def swap_uni(_X,_Y,modelx,modely,modelz,t_steps,t_lags,ncol,order):
    
     
    _X = np.copy(_X)
    c=grouped_col_multi(ncol,t_lags,order)   
    cu=grouped_col_uni2(ncol,t_lags,order)

    for i in range(1,t_steps+1):
        

        YNrx = modelx.predict(_X[:,cu[0]])
        
        YNry = modely.predict(_X[:,cu[1]])
        YNrz = modelz.predict(_X[:,cu[2]])
        
       
        Xnew=np.array([YNrx,YNry,YNrz])
        
        for k in range(t_lags+1):
            
            
            if ((k+1)<=t_lags):
                
                _X.T[:,[c[k]]] = _X.T[:,[c[k+1]]]
                
            else:
                
#                A=Xnew.T
#                #print(np.shape(A))                
#                B=np.transpose(A,(1,0,2)).reshape(3,-1)
#                #print(np.shape(B))
#                xx=Ideal_poly3(B.T,order,t_steps)
                A=np.squeeze(Xnew, axis=2)
                xx=Ideal_poly3(A.T,order,t_steps)
                
                
                #print(np.shape(_X[:,c[-1]]))
                
                _X[:,c[-1]] = xx
        
    return _X[:,c[-1]]
    
#----FORMING LORENZ TRAJECTORIES-------

r=28
R=1
tlength = 10000

order=1
t_steps=5
t_lags=1
ncol=3

pts=single_traj(4,-14,21,r,0.01,tlength) 
pts2=single_traj(1,-1,2.05,r,0.01,tlength)
N=len(pts)

# --------CREATING DATASETS------------------

Xr_t1,Yr_t1 = Ideal_poly(pts,order,t_steps)
Xr_train, Yr_train = Ideal_lags(Xr_t1,1,t_lags)


Xnr_t1,Ynr_t1 = Ideal_poly(pts,order,t_steps)
Xtrain, Ytrain = Ideal_lags(Xnr_t1,t_steps,t_lags)

Xr_test,Yr_test=Ideal_poly(pts2,order,t_steps)
Xtest, Ytest = Ideal_lags(Xr_test,t_steps,t_lags)
Xtest = Xtrain
Ytest = Ytrain

#-----UNIVARIATE-----------------------

cu = grouped_col_uni2(3,t_lags,order)
#---------------Non-Recursive-----------------

regr_x=linear_model.LinearRegression()
regr_x.fit(Xtrain[:,cu[0]], Ytrain[:,0])

regr_y=linear_model.LinearRegression()
regr_y.fit(Xtrain[:,cu[1]], Ytrain[:,1])

regr_z=linear_model.LinearRegression()
regr_z.fit(Xtrain[:,cu[2]], Ytrain[:,2])

#====predict==================================

Ynrx=regr_x.predict(Xtest[:,cu[0]])
Ynry=regr_y.predict(Xtest[:,cu[1]])
Ynrz=regr_z.predict(Xtest[:,cu[2]])

#----RECURSIVE-------------------------------------

rx=linear_model.LinearRegression()
rx.fit(Xr_train[:,cu[0]], Yr_train[:,0,None])

ry=linear_model.LinearRegression()
ry.fit(Xr_train[:,cu[1]], Yr_train[:,1,None])

rz=linear_model.LinearRegression()
rz.fit(Xr_train[:,cu[2]], Yr_train[:,2,None])

Yp=swap_uni(Xtest,Ytest,rx,ry,rz,t_steps,t_lags,ncol,order)


#============MULTIVARIATE======================
c = grouped_col_multi(3,t_lags,order)
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
Rx.fit(Xr_train, Yr_train[:,0])

Ry=linear_model.LinearRegression()
Ry.fit(Xr_train, Yr_train[:,1])

Rz=linear_model.LinearRegression()
Rz.fit(Xr_train, Yr_train[:,2])

YP=swap_multi(Xtest,Ytest,Rx,Ry,Rz,t_steps,t_lags,ncol,order)

plot_ts(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP)
