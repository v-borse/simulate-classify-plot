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
    
def gen_r_predict_uni(_X,modelx,modely,modelz,t_steps):
    _X=np.copy(_X)
    for i in range(1,t_steps+1):
                 
        Yrx=modelx.predict(_X[:,[0,3]])
        Yry=modely.predict(_X[:,[1,4]])
        Yrz=modelz.predict(_X[:,[2,5]])
        
        _X[:,0] = _X[:,3]
        _X[:,1] = _X[:,4]
        _X[:,2] = _X[:,5]
        
        _X[:,3] = Yrx
        _X[:,4] = Yry
        _X[:,5] = Yrz
        
    Xnew = np.array([Yrx,Yry,Yrz])
    return Xnew.T



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



def Ideal_poly(_x,_order,t_steps):
    _X=_x[:-t_steps]
    Y =_x[t_steps:]
    XX=[]
    for i in range(1,_order+1):
        
        XX=np.concatenate((_X, np.power(_X,i)), axis=1)
        print(np.shape(XX))
    
    
    return XX,Y


    
def Ideal_lags(_X,tsteps,tlags):
    Xti=_X[:-(tsteps+tlags)]
    for i in range(1,tlags+1):
        
        Xti=np.concatenate((Xti, _X[i:-(tsteps+tlags-i)]), axis=1)
        print(np.shape(Xti))
    
    Y  =_X[(tsteps+tlags):]
    
    return(Xti,Y)      

r=28
R=1
tlength = 10000

#N=len(pts)
#P=3

order=3
t_steps=1
t_lags=1

pts=single_traj(4,-14,21,r,0.01,tlength) 
pts2=single_traj(1,-1,2.05,r,0.01,tlength)
N=len(pts)

Xr_t1,Yr_t1 = Ideal_lags(pts,1,t_lags)
Xtrain, Ytrain = Ideal_lags(pts,t_steps,t_lags)
Xtest, Ytest = Ideal_lags(pts2,t_steps,t_lags)

Xpr_t1,Ypr_t1 = Ideal_poly(pts,2,t_steps)
Xp_train, Yp_train = Ideal_poly(pts,order,t_steps)
Xp_test, Yp_test = Ideal_poly(pts2,order,t_steps)



#-------POLYNOMIAL COVARIATES----------------------------
#----------UNIVARIATE-------------------------------------

#---------Non- Recursive---------------------------------------


regr_x = linear_model.LinearRegression()
regr_x.fit(Xp_train[:, [0,3]], Yp_train[:, 0])

print('Intercept: \n', regr_x.intercept_)
print('Coefficients: \n', regr_x.coef_)

regr_y = linear_model.LinearRegression()
regr_y.fit(Xp_train[:, [1,4]], Yp_train[:, 1])


regr_z = linear_model.LinearRegression()
regr_z.fit(Xp_train[:, [2,5]], Yp_train[:, 2])




Ynrx=regr_x.predict(Xp_test[:, [0,3]])
Ynry=regr_y.predict(Xp_test[:, [1,4]])
Ynrz=regr_z.predict(Xp_test[:, [2,5]])

#plot_predicted_ts(X9[:,0],Y9[:,0],Ynrx,0)
#plot_predicted_ts(X9[:,1],Y9[:,1],Ynry,1)
#plot_predicted_ts(X9[:,2],Y9[:,2],Ynrz,2)


#------RECURSIVE----------------------


rx = linear_model.LinearRegression()
rx.fit(Xpr_t1[:, [0,3]], Ypr_t1[:, 0])

ry = linear_model.LinearRegression()
ry.fit(Xpr_t1[:, [1,4]], Ypr_t1[:, 1])

rz = linear_model.LinearRegression()
rz.fit(Xpr_t1[:, [2,5]], Ypr_t1[:, 2])

#-------prediction-------------------

Yr = gen_r_predict_uni(Xp_test,rx,ry,rz,t_steps)


plot_predicted_ts(Xp_test[:,0],Yp_test[:,0],Ynrx,Yr,0)
plot_predicted_ts(Xp_test[:,1],Yp_test[:,1],Ynry,Yr,1)
plot_predicted_ts(Xp_test[:,2],Yp_test[:,2],Ynrz,Yr,2)

#============MULTIVARIATE======================

#---------------Non-Recursive-----------------

model_x=linear_model.LinearRegression()
model_x.fit(Xp_train[:,[0,1,2,3,4,5]], Yp_train[:,0])

model_y=linear_model.LinearRegression()
model_y.fit(Xp_train[:,[0,1,2,3,4,5]], Yp_train[:,1])

model_z=linear_model.LinearRegression()
model_z.fit(Xp_train[:,[0,1,2,3,4,5]], Yp_train[:,2])

#====predict==================================



YNRx=model_x.predict(Xp_test[:,[0,1,2,3,4,5]])
YNRy=model_y.predict(Xp_test[:,[0,1,2,3,4,5]])
YNRz=model_z.predict(Xp_test[:,[0,1,2,3,4,5]])

#plot_predicted_ts(X9[:,3],Y9[:,0],Ynrx,Yr,0)
#plot_predicted_ts(X9[:,4],Y9[:,1],Ynry,Yr,1)
#plot_predicted_ts(X9[:,5],Y9[:,2],Ynrz,Yr,2)

#----RECURSIVE-------------------------------------

Rx=linear_model.LinearRegression()
Rx.fit(Xpr_t1[:,[0,1,2,3,4,5]], Ypr_t1[:,0])

Ry=linear_model.LinearRegression()
Ry.fit(Xpr_t1[:,[0,1,2,3,4,5]], Ypr_t1[:,1])

Rz=linear_model.LinearRegression()
Rz.fit(Xpr_t1[:,[0,1,2,3,4,5]], Ypr_t1[:,2])

YR = gen_R_predict_multi(Xp_test,Rx,Ry,Rz,t_steps)


plot_predicted_ts(Xp_test[:,0],Yp_test[:,0],YNRx,YR,0)
plot_predicted_ts(Xp_test[:,1],Yp_test[:,1],YNRy,YR,1)
plot_predicted_ts(Xp_test[:,2],Yp_test[:,2],YNRz,YR,2)
#sns.heatmap(np.corrcoef(Y7[:,0],YR[:,0]),vmin=0.9,vmax=1)
"""
#************************************************************


#-----UNIVARIATE-----------------------

#-------Non-recursive------------------

regr_x = linear_model.LinearRegression()
regr_x.fit(Xtrain[:, [0,3]], Ytrain[:, 0])

print('Intercept: \n', regr_x.intercept_)
print('Coefficients: \n', regr_x.coef_)

regr_y = linear_model.LinearRegression()
regr_y.fit(Xtrain[:, [1,4]], Ytrain[:, 1])


regr_z = linear_model.LinearRegression()
regr_z.fit(Xtrain[:, [2,5]], Ytrain[:, 2])

Ynrx=regr_x.predict(Xtest[:, [0,3]])
Ynry=regr_y.predict(Xtest[:, [1,4]])
Ynrz=regr_z.predict(Xtest[:, [2,5]])

#plot_predicted_ts(X5[:,0],Y5[:,0],Ynrx,0)
#plot_predicted_ts(X5[:,1],Y5[:,1],Ynry,1)
#plot_predicted_ts(X5[:,2],Y5[:,2],Ynrz,2)

#------RECURSIVE----------------------


rx = linear_model.LinearRegression()
rx.fit(Xr_t1[:,[0,3]], Yr_t1[:, 0])

ry = linear_model.LinearRegression()
ry.fit(Xr_t1[:,[1,4]], Yr_t1[:,1])

rz = linear_model.LinearRegression()
rz.fit(Xr_t1[:,[2,5]], Yr_t1[:,2])

#-------prediction-------------------

Yr = gen_r_predict_uni(Xtest,rx,ry,rz,t_steps)


plot_predicted_ts(Xtest[:,3],Ytest[:,0],Ynrx,Yr,0)
plot_predicted_ts(Xtest[:,4],Ytest[:,1],Ynry,Yr,1)
plot_predicted_ts(Xtest[:,5],Ytest[:,2],Ynrz,Yr,2)



#============MULTIVARIATE======================

#---------------Non-Recursive-----------------

model_x=linear_model.LinearRegression()
model_x.fit(Xtrain[:,[0,1,2,3,4,5]], Ytrain[:,0])

model_y=linear_model.LinearRegression()
model_y.fit(Xtrain[:,[0,1,2,3,4,5]], Ytrain[:,1])

model_z=linear_model.LinearRegression()
model_z.fit(Xtrain[:,[0,1,2,3,4,5]], Ytrain[:,2])

#====predict==================================

YNRx=model_x.predict(Xtest[:,[0,1,2,3,4,5]])
YNRy=model_y.predict(Xtest[:,[0,1,2,3,4,5]])
YNRz=model_z.predict(Xtest[:,[0,1,2,3,4,5]])

#----RECURSIVE-------------------------------------

Rx=linear_model.LinearRegression()
Rx.fit(Xr_t1[:,[0,1,2,3,4,5]], Yr_t1[:,0])

Ry=linear_model.LinearRegression()
Ry.fit(Xr_t1[:,[0,1,2,3,4,5]], Yr_t1[:,1])

Rz=linear_model.LinearRegression()
Rz.fit(Xr_t1[:,[0,1,2,3,4,5]], Yr_t1[:,2])

YR = gen_R_predict_multi(Xtest,Rx,Ry,Rz,t_steps)


plot_predicted_ts(Xtest[:,3], Ytest[:,0],YNRx,YR,0)
plot_predicted_ts(Xtest[:,4], Ytest[:,1],YNRy,YR,1)
plot_predicted_ts(Xtest[:,5], Ytest[:,2],YNRz,YR,2)
"""
