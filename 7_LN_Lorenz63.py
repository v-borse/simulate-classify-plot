#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 09:40:39 2021

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
from sklearn.linear_model import Ridge
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
plt.style.use('bmh')
import seaborn as sns
import numpy as np
from statistics import variance
import cmath
from mlxtend.evaluate import bias_variance_decomp

import module
import module2

dt=0.01
r=28
R=1
tlength = 9999
t_length2=999
it=np.arange(0,tlength+1,1)
it2=np.arange(0,t_length2+1,0.1)

#it=np.linspace(0,(tlength+1)*dt,tlength+1)
#it_d=np.linspace(0,(t_length2+1)*dt*ss,tlength+1)

order=1
t_steps=10
t_lags=1
ncol=3

xx = [  2,  15, -26,   3, -17,   2,  -9,  27,  22, -14, -24,  12,   1,
         5,  22, -12, -28, -15,   5,  28, -26, -29,  21,  -8,  12, -29,
        16,  23,  19,  -9, -18,  25, -14,  25, -20,   4,  24,  -1,  15,
        12, -20, -18,  11, -10, -25,   6, -27,  13,  23,  -5,  20,   3,
        27,  28, -30,  12, -14,  -1,   4, -14,  24,  22, -20, -15,  19,
       -25, -17,  11, -11,  16,  17,  -4, -29,   7,  23, -30,   0,  11,
        10, -18,  26,   6,   8,   0,   0,  27, -25, -21,  25,  28,  24,
         2,  18,   7, -19, -26, -19, -12, -22,   4]
yy = [ 24, -14,  28,   9,  23, -16, -30,  -7, -24,  -4,  25,  21,   5,
         6,  -1, -24,  -5,  -2, -15,  -6, -20,  17, -27, -25, -24, -29,
        10,  -6,  21,  28,  13, -28, -25,  16,  -8,   6,   8, -13, -26,
        22,  17,  28, -30,  -4,   1,  -4, -10,  -9,  19, -17,  27,  28,
        26,  -2, -29,  10, -18,   5,   5,   1,   6,   6,  10, -15,   0,
        -5,  21,   1,   7,   4,  10,  -8, -29,  -5,  14,   0, -11, -23,
        29,   7,   2, -26,  19, -17,   8, -20, -29,   3,  -4,  11,  15,
        21,  11, -15, -27,  15, -19,   0, -27, -14]
zz = [ 7,  7,  9,  6, 13,  4, 27, 19,  1,  6, 17,  7, 25, 22, 26, 14,  4,
       23,  8,  7,  3, 17, 24,  6, 23, 23, 18,  4,  8,  3,  9, 19, 29, 22,
        6,  3, 12, 21, 20,  0, 28,  3, 10,  6,  0,  0, 15, 28,  3,  6,  3,
        9, 26, 25, 11,  1, 27,  5,  7, 25, 24, 17, 15, 26, 17, 19, 24,  3,
        7, 14, 14,  4, 25, 11,  0, 13,  9, 16,  3, 29, 27, 23,  2,  0,  2,
       14,  7, 15, 22,  5, 12, 21, 20, 21,  2, 11, 26,  1,  7, 21]



pts=single_traj(4,-14,21,r,0.01,tlength) 
#pts2=single_traj(1,-1,2.05,r,0.01,tlength)
#pts2=single_traj(14,-12,2.05,r,0.01,tlength)

pts3=single_traj(2,-4,6.05,r,0.01,tlength)
N=len(pts)
ss=50
start=0
end=2000
#end=len(Xtest[:,0])
#lead_time=[10,20,30,40,50]
lead_time=np.array([0.1,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0])
cd_=[]
rmse_=[]
Rmse_= np.empty((0, 12), float)
CD= np.empty((0, 12), float)

#YTest=np.empty((len(lead_time)+1,1000,3)
for a in range(len(xx[:2])):
    pts2=single_traj(xx[a],yy[a],zz[a],r,0.01,tlength)

    for i,item  in enumerate(lead_time) :
    
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
#        Xtest = Xtrain
#        Ytest = Ytrain
        
        X_cv,Y_cv=Ideal_poly(pts3[::ss],order,t_steps)
        Xcv, Ycv = Ideal_lags(X_cv,t_steps,t_lags)
        
        #cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        #alphas=np.arange(0.00000001, 0.0000001, 10)
        alphas=np.arange(1, 10, 10)
        alpha=1
        
        #-----UNIVARIATE-----------------------
        
        cu = grouped_col_uni2(3,t_lags,order)
        #---------------Non-Recursive-----------------
        
        regr_x=linear_model.LinearRegression()
        regr_x.fit(Xtrain[:,cu[0]], Ytrain[:,0])
        #print(cross_val_score(regr_x, Xtrain[:,cu[0]], Ytrain[:,0],scoring='neg_mean_squared_error', cv=10))
        
        regr_y=linear_model.LinearRegression()
        regr_y.fit(Xtrain[:,cu[1]], Ytrain[:,1])
        #print(cross_val_score(regr_y, Xtrain[:,cu[1]], Ytrain[:,1],scoring='neg_mean_squared_error', cv=10))
        
        regr_z=linear_model.LinearRegression()
        regr_z.fit(Xtrain[:,cu[2]], Ytrain[:,2])
        #print(cross_val_score(regr_x, Xtrain[:,cu[0]], Ytrain[:,0],scoring='neg_mean_squared_error', cv=10))
        #====predict==================================
        
        Ynrx=regr_x.predict(Xtest[:,cu[0]])
        Ynry=regr_y.predict(Xtest[:,cu[1]])
        Ynrz=regr_z.predict(Xtest[:,cu[2]])
        
        #----RECURSIVE-------------------------------------
        
        rx=linear_model.LinearRegression()
        rx.fit(Xr_train[:,cu[0]], Yr_train[:,0,])
        
        ry=linear_model.LinearRegression()
        ry.fit(Xr_train[:,cu[1]], Yr_train[:,1])
        
        rz=linear_model.LinearRegression()
        rz.fit(Xr_train[:,cu[2]], Yr_train[:,2])
        
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
        
        cd_=coef_det(Xtest,Ytest,regr_x,regr_y,regr_z,rx,ry,rz,model_x,model_y,model_z,Rx,Ry,Rz,cu)
        CD=np.append(CD,np.array([cd_]),axis=0)
        
#        rmse_.append(RMSE(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP))
#        Rmse_=np.array(rmse_)
        
        rmse_=RMSE(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP)
        Rmse_=np.append(Rmse_, np.array([rmse_]), axis=0)
        #plt.boxplot(Rmse_,positions=[a])
        
        
#        plt.plot(it_d[start:end],Ytest[start:end,0],'k+')
#        plt.plot(it_d[start:end],Ytest[start:end,1],'k+')
#        plt.plot(it_d[start:end],Ytest[start:end,2],'k+')
        
#        plt.plot(it_d[start:end],Ynrx[start:end])
#        plt.plot(it_d[start:end],Ynry[start:end])
#        plt.plot(it_d[start:end],Ynrz[start:end])
#        
#        plt.plot(it_d[start:end],YNRx[start:end])
#        plt.plot(it_d[start:end],YNRy[start:end])
#        plt.plot(it_d[start:end],YNRz[start:end])
#        
#        plt.plot(it_d[start:end],Yp[start:end,0])
#        plt.plot(it_d[start:end],Yp[start:end,1])
#        plt.plot(it_d[start:end],Yp[start:end,2])
#   
#        plt.plot(it_d[start:end],YP[start:end,0])
#        plt.plot(it_d[start:end],YP[start:end,1])
#        plt.plot(it_d[start:end],YP[start:end,2])
#        
        
        #print("lead_time",lead_time[i])
        #plot_ts_lt(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP,start,end,it_d)
        
        #YTest[i][:][:],YnrX[i],YnrY[i],YnrZ[i],Yp_[i],YNRX[i],YNRY[i],YNRZ[i],YP_[i],It_d[i],CD[i]=predict(pts,pts2,ss,tlength,t_length2,dt, t_steps,order,t_lags)
        #plot_ts(YTest[i],YnrX[i],YnrY[i],YnrZ[i],Yp_[i],YNRX[i],YNRY[i],YNRZ[i],YP_[i],start,end,It_d)
        #plot_error(Ytest[i],Ynrx[i],Ynry[i],Ynrz[i],Yp[i],YNRx[i],YNRy[i],YNRz[i],YP[i],start,end,it_d[i])
        #scatter_plots(Ytest[i],Ynrx[i],Ynry[i],Ynrz[i],Yp[i],YNRx[i],YNRy[i],YNRz[i],YP[i],start,end)
        
#        plot_ts(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP,start,end,it_d)
#        plot_error(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP,start,end,it_d)
#        scatter_plots(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP,start,end=8000)
        
    
        
#plt.scatter(lead_time,CD[:,0])
#cd_lt2(CD,lead_time)
        
rmse_lt(Rmse_,lead_time*dt*ss)

#compare(pts2,it,it_d,ss,start=0,end=1000)
    
def variance(data):
    # Number of observations
    n = len(data)
    # Mean of the data
    mean = sum(data) / n
     # Square deviations
    deviations = [(x - mean) ** 2 for x in data]
    # Variance
    variance = sum(deviations) / n
    return variance
 
def BIAS(actual,predicted):
    
    mse_= mean_squared_error(actual,predicted)
    mse_ = np.mean((actual - predicted)**2)

    vari= variance(actual-predicted)
    print (vari)
    print(mse_)
    print(mse_-vari)
    Bias=cmath.sqrt(mse_-vari)
    
    return Bias

def true_bias(actual,predicted):
    
    data= actual-predicted
    n = len(data)
    # Mean of the data
    mean = sum(data) / n
    
    return mean
    
#_bias=BIAS(Ytest[:,0],Ynrx)
#print(_bias)
#tr_bias=true_bias(Ytest[:,0],Ynrx)
#print(tr_bias)
#
#mse, bias, var = bias_variance_decomp(regr_x, Xtrain[:,cu[0]], Ytrain[:,0], Xtrain[:,cu[0]], Ytest[:,0], loss='mse', num_rounds=200, random_seed=1)

#print (np.corrcoef(Ytest[:,0],Ynrx))
#print(np.corrcoef(Ytest.T,YP.T))
#print(cd_)
#print(rmse_)

    
label=['Ynrx','Ynry','Ynrz','YNRx','YNRy','YNRz','Yrx','Yry','Yrz','YRx','YRy','YRz']
#for i in range(len(Rmse_[0])):
#    plt.boxplot(Rmse_.T[i],positions=[i])
#    plt.yscale('log')
#    #plt.plot(np.mean(Rmse_.T[i]))
#    plt.title('RMSE Boxplots for 100 trajectories')
#    plt.xlabel("Models")
#    plt.ylabel("RMSE")
#    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11],label)

#    

#for i in range(len(CD[0])):
#    plt.boxplot(CD.T[i],positions=[i])
#    plt.title('Coef_of dtrmntn Boxplots for 100 trajectories')
#    plt.xlabel("Models")
#    plt.ylabel("R^2")
#    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11],label)