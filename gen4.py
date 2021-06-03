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


from module import lorenz
from module import single_traj

from toy_model import Ideal_poly
from toy_model import grouped_col_multi,swap, swap3,grouped_col_uni,Ideal_poly3,Ideal_poly4


def Ideal_lags(_X,tsteps,tlags):
    Xti=_X[:-(tsteps+tlags)]
    for i in range(1,tlags+1):
        
        Xti=np.concatenate((Xti, _X[i:-(tsteps+tlags-i)]), axis=1)
        #print(np.shape(Xti))
    
    Y  =_X[(tsteps+tlags):]
    
    return(Xti,Y)   
    
def swap4(_X,_Y,modelx,modely,modelz,t_steps,t_lags,ncol,order):
    print("in swap4")
      
    _X = np.copy(_X)
    
    c=grouped_col_multi(ncol,t_lags,order)

    
    #print("after Ynrx")
    #print(np.shape(_Y))
    for i in range(1,t_steps+1):
               
        Ynrx = modelx.predict(_X)
        Ynry = modely.predict(_X)
        Ynrz = modelz.predict(_X)
        
        
        Xnew=np.array([Ynrx,Ynry,Ynrz])
        #print(np.shape(Xnew))
        
       
        for k in range(t_lags+1):
          
            
            if ((k+1)<=t_lags):
                
                _X.T[:,[c[k]]] = _X.T[:,[c[k+1]]]
               
            else:
                
                xx=Ideal_poly3(Xnew.T,order,t_steps)
                print(np.shape(_X[:,c[-1]]))
                print(np.shape(xx))
                _X[:,c[-1]] = xx
       
    return _X[:,c[-1]]

def swap6(_X,_Y,modelx,modely,modelz,t_steps,t_lags,ncol,order):
    print("in swap6")
     
    _X = np.copy(_X)
    c=grouped_col_multi(ncol,t_lags,order)   
    cu=grouped_col_uni(ncol,t_lags,order)

    for i in range(1,t_steps+1):
        

        YNrx = modelx.predict(_X[:,cu[0]])
        
        YNry = modely.predict(_X[:,cu[1]])
        YNrz = modelz.predict(_X[:,cu[2]])
        
       
        Xnew=np.array([YNrx,YNry,YNrz])
        print(np.shape(Xnew))
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
                
                print(np.shape(A))
                #print(np.shape(_X[:,c[-1]]))
                print(np.shape(xx))
                _X[:,c[-1]] = xx
        
    return _X[:,c[-1]]
    
#----FORMING LORENZ TRAJECTORIES-------

r=28
R=1
tlength = 10000

#N=len(pts)
#P=3

order=2
t_steps=5
t_lags=2
ncol=3

pts=single_traj(4,-14,21,r,0.01,tlength) 
pts2=single_traj(1,-1,2.05,r,0.01,tlength)
N=len(pts)

# --------CREATING DATASETS------------------

Xr_t1,Yr_t1 = Ideal_poly(pts,order,t_steps)
Xr_train, Yr_train = Ideal_lags(Xr_t1,1,t_lags)

#Xnr_t1,Ynr_t1=Ideal_poly(pts,order,t_steps)
Xnr_t1,Ynr_t1 = Ideal_poly(pts,order,t_steps)
Xtrain, Ytrain = Ideal_lags(Xnr_t1,t_steps,t_lags)

Xr_test,Yr_test=Ideal_poly(pts2,order,t_steps)
Xtest, Ytest = Ideal_lags(Xr_test,t_steps,t_lags)

#-----UNIVARIATE-----------------------
#-------Non-recursive------------------
cu = grouped_col_uni(3,t_lags,order)
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
#rx.fit(Xr_t1[:,cu[0]], Yr_t1[:,0])
rx.fit(Xr_train[:,cu[0]], Yr_train[:,0,None])

ry=linear_model.LinearRegression()
#ry.fit(Xr_t1[:,cu[1]], Yr_t1[:,1])
ry.fit(Xr_train[:,cu[1]], Yr_train[:,1,None])

rz=linear_model.LinearRegression()
#rz.fit(Xr_t1[:,cu[2]], Yr_t1[:,2])
rz.fit(Xr_train[:,cu[2]], Yr_train[:,2,None])
#Yr = GEN_R_predict_uni(Xtest,rx,ry,rz,t_steps,t_lags,ncol,1)

#Xp2,Yp2=Ideal_poly(Xtest,order,t_steps)
#Yp=swap6(Xp2,Yp2,rx,ry,rz,t_steps,t_lags,ncol,order)
Yp=swap6(Xtest,Ytest,rx,ry,rz,t_steps,t_lags,ncol,order)


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
#Rx.fit(Xnr_t1, Ynr_t1[:,0])
Rx.fit(Xr_train, Yr_train[:,0])

Ry=linear_model.LinearRegression()
#Ry.fit(Xnr_t1, Ynr_t1[:,1])
Ry.fit(Xr_train, Yr_train[:,1])

Rz=linear_model.LinearRegression()
#Rz.fit(Xnr_t1, Ynr_t1[:,2])
Rz.fit(Xr_train, Yr_train[:,2])
#YR = GEN_R_predict_multi(Xtest,Rx,Ry,Rz,t_steps,t_lags,ncol,1)

#Xp2,Yp2=Ideal_poly(Xtest,order,t_steps)
#Xt2,Yt2=Ideal_lags(Xp2,t_steps,t_lags)
#Yp=swap3(Xp2,Yp2,t_steps,t_lags,ncol,order)
#YP=swap4(Xtest,Ytest,Rx,Ry,Rz,t_steps,t_lags,ncol,order)

