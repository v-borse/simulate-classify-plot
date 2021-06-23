#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 13:25:10 2021

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
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.linear_model import Lasso,LassoCV
from sklearn.model_selection import RepeatedKFold
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

import module
import module2

dt=0.01
r=28
R=1
tlength = 9999
t_length2=999
it=np.arange(0,tlength+1,1)
it2=np.arange(0,t_length2+1,0.1)



order=1
t_steps=2
t_lags=1
ncol=3

pts=single_traj(4,-14,21,r,0.01,tlength) 
pts2=single_traj(1,-1,2.05,r,0.01,tlength)
pts3=single_traj(2,-4,6.05,r,0.01,tlength)
N=len(pts)
ss=10
start=0
end=100
lead_time=[1,2,5,10,20]
cd_=[]

#YTest=np.empty((len(lead_time)+1,1000,3)

for i,item in enumerate(lead_time):
    
    t_steps=item
    # --------CREATING DATASETS------------------
    it=np.linspace(0,(tlength+1)*dt,tlength+1)
    it_d=np.linspace(0,(t_length2+1)*dt*ss,tlength+1)

    Xr_t1,Yr_t1 = Ideal_poly(pts[::ss],order,t_steps)
    Xr_train, Yr_train = Ideal_lags(Xr_t1,1,t_lags)
    
    
    Xnr_t1,Ynr_t1 = Ideal_poly(pts[::ss],order,t_steps)
    Xtrain, Ytrain = Ideal_lags(Xnr_t1,t_steps,t_lags)
    
    Xr_test,Yr_test=Ideal_poly(pts2[::ss],order,t_steps)
    Xtest, Ytest = Ideal_lags(Xr_test,t_steps,t_lags)
    #Xtest = Xtrain
    #Ytest = Ytrain

    X_cv,Y_cv=Ideal_poly(pts3[::ss],order,t_steps)
    Xcv, Ycv = Ideal_lags(X_cv,t_steps,t_lags)
    
    
    #cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    cv = None
    
    alphas=np.arange(1, 10, 1)
    alpha=1
    
    #-----UNIVARIATE-----------------------
    
    cu = grouped_col_uni2(3,t_lags,order)
    #---------------Non-Recursive-----------------
    regr_x= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error', store_cv_values=True)
    #regr_x=linear_model.LinearRegression(alpha)
    regr_x.fit(Xtrain[:,cu[0]], Ytrain[:,0])
    
    regr_y= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
    #regr_y=linear_model.Ridge(alpha)
    regr_y.fit(Xtrain[:,cu[1]], Ytrain[:,1])
    
    regr_z= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
    #regr_z=linear_model.Ridge(alpha)
    #regr_z=linear_model.LinearRegression()
    regr_z.fit(Xtrain[:,cu[2]], Ytrain[:,2])
    
    #====predict==================================
    
    Ynrx=regr_x.predict(Xtest[:,cu[0]])
    Ynry=regr_y.predict(Xtest[:,cu[1]])
    Ynrz=regr_z.predict(Xtest[:,cu[2]])
    
    #----RECURSIVE-------------------------------------
    rx= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
    #rx=linear_model.Ridge(alpha)
    #rx=linear_model.LinearRegression()
    rx.fit(Xr_train[:,cu[0]], Yr_train[:,0])
    
    ry= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
    #ry=linear_model.Ridge(alpha)
    #ry=linear_model.LinearRegression()
    ry.fit(Xr_train[:,cu[1]], Yr_train[:,1])
    
    rz= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
    #rz=linear_model.Ridge(alpha)
    #rz=linear_model.LinearRegression()
    rz.fit(Xr_train[:,cu[2]], Yr_train[:,2])
    
    Yp=swap_uni(Xtest,Ytest,rx,ry,rz,t_steps,t_lags,ncol,order)
    
    
    #============MULTIVARIATE======================
    c = grouped_col_multi(3,t_lags,order)
    #---------------Non-Recursive-----------------
    
    model_x= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
    #model_x=linear_model.Ridge(alpha)
    #model_x=linear_model.LinearRegression()
    model_x.fit(Xtrain, Ytrain[:,0])
    
    model_y= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
    #model_y=linear_model.Ridge(alpha)
    #model_y=linear_model.LinearRegression()
    model_y.fit(Xtrain, Ytrain[:,1])
    
    model_z= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
    #model_z=linear_model.Ridge(alpha)
    #model_z=linear_model.LinearRegression()
    model_z.fit(Xtrain, Ytrain[:,2])
    
    #====predict==================================
    
    YNRx=model_x.predict(Xtest)
    YNRy=model_y.predict(Xtest)
    YNRz=model_z.predict(Xtest)
    
    #----RECURSIVE-------------------------------------
    
    Rx= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
    #Rx=linear_model.Ridge(alpha)
    #Rx=linear_model.LinearRegression()
    Rx.fit(Xr_train, Yr_train[:,0])
    
    Ry= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
    #Ry=linear_model.Ridge(alpha)
    #Ry=linear_model.LinearRegression()
    Ry.fit(Xr_train, Yr_train[:,1])
    
    Rz= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
    #Rz=linear_model.Ridge(alpha)
    #Rz=linear_model.LinearRegression()
    Rz.fit(Xr_train, Yr_train[:,2])
    
    YP=swap_multi(Xtest,Ytest,Rx,Ry,Rz,t_steps,t_lags,ncol,order)
    
    cd_.append(coef_det(Xtest,Ytest,regr_x,regr_y,regr_z,rx,ry,rz,model_x,model_y,model_z,Rx,Ry,Rz,cu))
    CD=np.array(cd_)
    
    
#    plt.plot(it_d[start:end],Ynrx[start:end])
#    plt.plot(it_d[start:end],Ynry[start:end])
#    plt.plot(it_d[start:end],Ynrz[start:end])
#    
#    plt.plot(it_d[start:end],YNRx[start:end])
#    plt.plot(it_d[start:end],YNRy[start:end])
#    plt.plot(it_d[start:end],YNRz[start:end])
#    
#    plt.plot(it_d[start:end],Yp[start:end,0])
#    plt.plot(it_d[start:end],Yp[start:end,1])
#    plt.plot(it_d[start:end],Yp[start:end,2])
#
#    plt.plot(it_d[start:end],YP[start:end,0])
#    plt.plot(it_d[start:end],YP[start:end,1])
#    plt.plot(it_d[start:end],YP[start:end,2])
#    
    
    #print("lead_time",lead_time[i])
    #plot_ts_lt(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP,start,end,it_d)
    
    #YTest[i][:][:],YnrX[i],YnrY[i],YnrZ[i],Yp_[i],YNRX[i],YNRY[i],YNRZ[i],YP_[i],It_d[i],CD[i]=predict(pts,pts2,ss,tlength,t_length2,dt, t_steps,order,t_lags)
    #plot_ts(YTest[i],YnrX[i],YnrY[i],YnrZ[i],Yp_[i],YNRX[i],YNRY[i],YNRZ[i],YP_[i],start,end,It_d)
    #plot_error(Ytest[i],Ynrx[i],Ynry[i],Ynrz[i],Yp[i],YNRx[i],YNRy[i],YNRz[i],YP[i],start,end,it_d[i])
    #scatter_plots(Ytest[i],Ynrx[i],Ynry[i],Ynrz[i],Yp[i],YNRx[i],YNRy[i],YNRz[i],YP[i],start,end)
    
    plot_ts(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP,start,end,it_d)
    #plot_error(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP,start,end,it_d)
    #scatter_plots(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP,start,end=8000)

#plt.scatter(lead_time,CD[:,0])
cd_lt2(CD,lead_time)
#
#compare(pts2,it,it_d,ss,start=0,end=1000)
