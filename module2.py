#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 08:35:11 2021

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
from sklearn.metrics import mean_squared_error
import math
from statistics import variance


def grouped_col_uni2(ncol,t_lags,order):
    
    ff= np.arange(0,(ncol*(t_lags+1)*order),1)
    #F=np.reshape(ff,((t_lags+1),ncol*order))
    F=np.reshape(ff,((t_lags+1)*order,ncol))
    
    return F.T

def grouped_col_multi(ncol,t_lags,order):
    
    ff= np.arange(0,(ncol*(t_lags+1)*order),1)
    F=np.reshape(ff,((t_lags+1),ncol*order))
    
    
    return F

def Ideal_lags(_X,tsteps,tlags):
    Xti=_X[:-(tsteps+tlags)]
    for i in range(1,tlags+1):
        
        Xti=np.concatenate((Xti, _X[i:-(tsteps+tlags-i)]), axis=1)
            
    Y  =_X[(tsteps+tlags):]
    
    return(Xti,Y)  
    
def Ideal_poly(xX,_order,t_steps):
    _X=xX[:-t_steps]
    _Y=xX[t_steps:]
    xx=[]
    #yy=[]
    for i in range(1,_order+1):
        x=np.power(_X,i)
        #y=np.power(_Y,i)
        xx.append(x)
        #yy.append(y)
    l= np.size(xx,axis=1)
    #ly= np.size(yy,axis=1)
    #XX=np.transpose(xx,(1,0,2)).reshape(N,-1)
    XX=np.transpose(xx,(1,0,2)).reshape(l,-1)
    #YY=np.transpose(yy,(1,0,2)).reshape(ly,-1)
    
    return XX,_Y
    
def Ideal_poly3(_X,_order,t_steps):
#    _X=xX[:-t_steps]
#    _Y=xX[t_steps:]
    xx=[]
    #yy=[]
    for i in range(1,_order+1):
        x=np.power(_X,i)
        #y=np.power(_Y,i)
        xx.append(x)
        #yy.append(y)
    l= np.size(xx,axis=1)
#    print(l)
#    print(np.shape(xx))
    #ly= np.size(yy,axis=1)
    #XX=np.transpose(xx,(1,0,2)).reshape(N,-1)
    XX=np.transpose(xx,(1,0,2)).reshape(l,-1)
    #YY=np.transpose(yy,(1,0,2)).reshape(ly,-1)
    
    return XX

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
                #A=np.squeeze(Xnew, axis=2)
                xx=Ideal_poly3(Xnew.T,order,t_steps)
                
                
                _X[:,c[-1]] = xx
        
    return _X[:,c[-1]]

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



def plot_ts(Ytrue,ynrx,ynry,ynrz,Yr,YNRX,YNRY,YNRZ,YR,start,end,int_time):
    
    fig, axs = plt.subplots(6, 2, sharex=False, sharey=False, figsize=(15, 15))
    fig.suptitle(' Time series ; dt=0.01')
    
    axs[0,0].plot(int_time[start:end],Ytrue[start:end,0],'b')
    axs[0,0].plot(int_time[start:end],ynrx[start:end],'r')
    #axs[0,0].plot(Ytrue[:,0]-ynrx,'g')
    axs[0,0].set_xlabel("integration time")
    axs[0,0].set_ylabel("blue:Ytrue ; red: pred" )
    axs[0,0].set_title("X component NR /Univariate")
    axs[0,0].legend()
    
    axs[1,0].plot(int_time[start:end],Ytrue[start:end,1],'b')
    axs[1,0].plot(int_time[start:end],ynry[start:end],'r')
    axs[1,0].set_xlabel("integration time")
    axs[1,0].set_ylabel("blue:Ytrue ; red: pred" )
    axs[1,0].set_title("Y component NR /Univariate")
    
    axs[2,0].plot(int_time[start:end],Ytrue[start:end,2],'b')
    axs[2,0].plot(int_time[start:end],ynrz[start:end],'r')
    axs[2,0].set_xlabel("integration time")
    axs[2,0].set_ylabel("blue:Ytrue ; red: pred" )
    axs[2,0].set_title("Z component NR /Univariate")
    
    axs[0,1].plot(int_time[start:end],Ytrue[start:end,0],'b')
    axs[0,1].plot(int_time[start:end],YNRX[start:end],'r')
    axs[0,1].set_xlabel("integration time")
    axs[0,1].set_ylabel("blue:Ytrue ; red: pred" )
    axs[0,1].set_title("X component NR /Multivariate")
    
    axs[1,1].plot(int_time[start:end],Ytrue[start:end,1],'b')
    axs[1,1].plot(int_time[start:end],YNRY[start:end],'r')
    axs[1,1].set_xlabel("integration time")
    axs[1,1].set_ylabel("blue:Ytrue ; red: pred" )
    axs[1,1].set_title("Y component NR /Multivariate")
    
    axs[2,1].plot(int_time[start:end],Ytrue[start:end,2],'b')
    axs[2,1].plot(int_time[start:end],YNRZ[start:end],'r')
    axs[2,1].set_xlabel("integration time")
    axs[2,1].set_ylabel("blue:Ytrue ; red: pred" )
    axs[2,1].set_title("Z component NR /Multivariate")
    
    axs[3,0].plot(int_time[start:end],Ytrue[start:end,0])
    axs[3,0].plot(int_time[start:end],Yp[start:end,0])
    axs[3,0].set_xlabel("integration time")
    axs[3,0].set_ylabel("blue:Ytrue ; orange: pred" )
    axs[3,0].set_title("X component R /Univariate")
    
    axs[4,0].plot(int_time[start:end],Ytrue[start:end,1])
    axs[4,0].plot(int_time[start:end],Yp[start:end,1])
    axs[4,0].set_xlabel("integration time")
    axs[4,0].set_ylabel("blue:Ytrue ; orange: pred" )
    axs[4,0].set_title("Y component R /Univariate")
    
    axs[5,0].plot(int_time[start:end],Ytrue[start:end,2])
    axs[5,0].plot(int_time[start:end],Yp[start:end,2])
    axs[5,0].set_xlabel("integration time")
    axs[5,0].set_ylabel("blue:Ytrue ; orange: pred" )
    axs[5,0].set_title("Z component R /Univariate")
    
    axs[3,1].plot(int_time[start:end],Ytrue[start:end,0])
    axs[3,1].plot(int_time[start:end],YP[start:end,0])
    axs[3,1].set_xlabel("integration time")
    axs[3,1].set_ylabel("blue:Ytrue ; orange: pred" )
    axs[3,1].set_title("X component R /Multivariate")
    
    axs[4,1].plot(int_time[start:end],Ytrue[start:end,1])
    axs[4,1].plot(int_time[start:end],YP[start:end,1])
    axs[4,1].set_xlabel("integration time")
    axs[4,1].set_ylabel("blue:Ytrue ; orange: pred" )
    axs[4,1].set_title("Y component R /Multivariate")
    
    axs[5,1].plot(int_time[start:end],Ytrue[start:end,2])
    axs[5,1].plot(int_time[start:end],YP[start:end,2])
    axs[5,1].set_xlabel("integration time")
    axs[5,1].set_ylabel("blue:Ytrue ; orange: pred" )
    axs[5,1].set_title("Z component R /Multivariate")
    
    fig.tight_layout()
    


def plot_error(Ytrue,ynrx,ynry,ynrz,Yr,YNRX,YNRY,YNRZ,YR,start,end,int_time):
    
    fig, axs = plt.subplots(3, sharex=False, sharey=False, figsize=(15, 15))
    fig.suptitle('Error plots')
    
    axs[0].plot(int_time[start:end],Ytrue[start:end,0]-ynrx[start:end],'^',markersize=12)
    axs[0].plot(int_time[start:end],Ytrue[start:end,0]-Yp[start:end,0],'+',markersize=12)
    axs[0].plot(int_time[start:end],Ytrue[start:end,0]-YNRX[start:end],'*',markersize=12)
    axs[0].plot(int_time[start:end],Ytrue[start:end,0]-YP[start:end,0],'.',markersize=15)
    axs[0].set_xlabel("integration time")
    axs[0].set_ylabel("Error" )
    axs[0].set_title("X component")
    
    axs[1].plot(int_time[start:end],Ytrue[start:end,1]-ynry[start:end],'^',markersize=12)
    axs[1].plot(int_time[start:end],Ytrue[start:end,1]-Yp[start:end,1],'+',markersize=12)
    axs[1].plot(int_time[start:end],Ytrue[start:end,1]-YNRY[start:end],'*',markersize=12)
    axs[1].plot(int_time[start:end],Ytrue[start:end,1]-YP[start:end,1],'.',markersize=15)
    axs[1].set_xlabel("integration time")
    axs[1].set_ylabel("Error" )
    axs[1].set_title("Y component")
    
    
    axs[2].plot(int_time[start:end],Ytrue[start:end,2]-ynrz[start:end],'^',markersize=12)
    axs[2].plot(int_time[start:end],Ytrue[start:end,2]-Yp[start:end,2],'+',markersize=12)
    axs[2].plot(int_time[start:end],Ytrue[start:end,2]-YNRZ[start:end],'*',markersize=12)
    axs[2].plot(int_time[start:end],Ytrue[start:end,2]-YP[start:end,2],'.',markersize=15)
    axs[2].set_xlabel("integration time")
    axs[2].set_ylabel("Error" )
    axs[2].set_title("Z component")
    
    fig.tight_layout()

def scatter_plots(Ytrue,ynrx,ynry,ynrz,Yr,YNRX,YNRY,YNRZ,YR,start,end):
    
    fig, axs = plt.subplots(6, 2, sharex=False, sharey=False, figsize=(15, 15))
    fig.suptitle('Scatter plots: Predicted v/s true')
    
    axs[0,0].scatter(Ytrue[start:end,0],ynrx[start:end],s=10)
    #axs[0,0].plot(ynrx,'r')
    #axs[0,0].plot(Ytrue[:,0]-ynrx,'g')
    axs[0,0].set_xlabel("Ytrue")
    axs[0,0].set_ylabel("Ynrx" )
    axs[0,0].set_title("X component NR /Univariate")
   
    
    axs[1,0].scatter(Ytrue[start:end,1],ynry[start:end],s=10)
    #axs[1,0].plot(ynry,'r')
    axs[1,0].set_xlabel("Ytrue")
    axs[1,0].set_ylabel("Ynry")
    axs[1,0].set_title("Y component NR /Univariate")
    
    axs[2,0].scatter(Ytrue[start:end,2],ynrz[start:end],s=10)
    #axs[2,0].plot(ynrz,'r')
    axs[2,0].set_xlabel("Ytrue")
    axs[2,0].set_ylabel("Ynrz")
    axs[2,0].set_title("Z component NR /Univariate")
    
    axs[0,1].scatter(Ytrue[start:end,0],YNRX[start:end],s=10)
    #axs[0,1].plot(YNRX,'r')
    axs[0,1].set_xlabel("Ytrue")
    axs[0,1].set_ylabel("YNRx")
    axs[0,1].set_title("X component NR /Multivariate")
    
    axs[1,1].scatter(Ytrue[start:end,1],YNRY[start:end],s=10)
    #axs[1,1].plot(YNRY,'r')
    axs[1,1].set_xlabel("Ytrue")
    axs[1,1].set_ylabel("YNRy")
    axs[1,1].set_title("Y component NR /Multivariate")
    
    axs[2,1].scatter(Ytrue[start:end,2],YNRZ[start:end],s=10)
    #axs[2,1].plot(YNRZ,'r')
    axs[2,1].set_xlabel("Ytrue")
    axs[2,1].set_ylabel("YNRz")
    axs[2,1].set_title("Z component NR /Multivariate")
    
    axs[3,0].scatter(Ytrue[start:end,0],Yp[start:end,0],s=10)
    #axs[3,0].plot(Yp[:,0])
    axs[3,0].set_xlabel("Ytrue")
    axs[3,0].set_ylabel("Ypx")
    axs[3,0].set_title("X component R /Univariate")
    
    axs[4,0].scatter(Ytrue[start:end,1],Yp[start:end,1],s=10)
    #axs[4,0].plot(Yp[:,1])
    axs[4,0].set_xlabel("Ytrue")
    axs[4,0].set_ylabel("Ypy")
    axs[4,0].set_title("Y component R /Univariate")
    
    axs[5,0].scatter(Ytrue[start:end,2],Yp[start:end,2],s=10)
    #axs[5,0].plot(Yp[:,2])
    axs[5,0].set_xlabel("Ytrue")
    axs[5,0].set_ylabel("Ypz")
    axs[5,0].set_title("Z component R /Univariate")
    
    axs[3,1].scatter(Ytrue[start:end,0],YP[start:end,0],s=10)
    #axs[3,1].plot(YP[:,0])
    axs[3,1].set_xlabel("Ytrue")
    axs[3,1].set_ylabel("YPx")
    axs[3,1].set_title("X component R /Multivariate")
    
    axs[4,1].scatter(Ytrue[start:end,1],YP[start:end,1],s=10)
    #axs[4,1].plot(YP[:,1])
    axs[4,1].set_xlabel("Ytrue")
    axs[4,1].set_ylabel("YPy")
    axs[4,1].set_title("Y component R /Multivariate")
    
    axs[5,1].scatter(Ytrue[start:end,2],YP[start:end,2],s=10)
    #axs[5,1].plot(YP[:,2])
    axs[5,1].set_xlabel("Ytrue")
    axs[5,1].set_ylabel("YPz")
    axs[5,1].set_title("Z component R /Multivariate")
    
    fig.tight_layout()
    
def coef_det(Xtest,Ytest,regr_x,regr_y,regr_z,rx,ry,rz,model_x,model_y,model_z,Rx,Ry,Rz,cu):
    cd=[]
    cd.append(regr_x.score(Xtest[:,cu[0]],Ytest[:,0]))
    cd.append(regr_y.score(Xtest[:,cu[1]],Ytest[:,1]))
    cd.append(regr_z.score(Xtest[:,cu[2]],Ytest[:,2]))

    cd.append(rx.score(Xtest[:,cu[0]],Ytest[:,0]))
    cd.append(ry.score(Xtest[:,cu[1]],Ytest[:,1]))
    cd.append(rz.score(Xtest[:,cu[2]],Ytest[:,2]))

    cd.append(model_x.score(Xtest,Ytest[:,0]))
    cd.append(model_y.score(Xtest,Ytest[:,1]))
    cd.append(model_z.score(Xtest,Ytest[:,2]))
    
    cd.append(Rx.score(Xtest,Ytest[:,0]))
    cd.append(Ry.score(Xtest,Ytest[:,1]))
    cd.append(Rz.score(Xtest,Ytest[:,2]))
    
    Cd=np.array(cd)
    
    return Cd

def RMSE(Ytrue,ynrx,ynry,ynrz,Yr,YNRX,YNRY,YNRZ,YR):
    
    rmse=[]
    rmse.append(np.sqrt(mean_squared_error(Ytrue[:,0], ynrx)))
    rmse.append(np.sqrt(mean_squared_error(Ytrue[:,1], ynry)))
    rmse.append(np.sqrt(mean_squared_error(Ytrue[:,2], ynrz)))
    
    rmse.append(np.sqrt(mean_squared_error(Ytrue[:,0], YNRX)))
    rmse.append(np.sqrt(mean_squared_error(Ytrue[:,1], YNRY)))
    rmse.append(np.sqrt(mean_squared_error(Ytrue[:,2], YNRZ)))
    
    rmse.append(np.sqrt(mean_squared_error(Ytrue[:,0], Yr[:,0])))
    rmse.append(np.sqrt(mean_squared_error(Ytrue[:,1], Yr[:,1])))
    rmse.append(np.sqrt(mean_squared_error(Ytrue[:,2], Yr[:,2])))
    
    rmse.append(np.sqrt(mean_squared_error(Ytrue[:,0], YR[:,0])))
    rmse.append(np.sqrt(mean_squared_error(Ytrue[:,1], YR[:,1])))
    rmse.append(np.sqrt(mean_squared_error(Ytrue[:,2], YR[:,2])))
    
    Rmse=np.array(rmse)
    return Rmse


def predict(pts,pts2,ss,tlength,t_length2,dt, t_steps,order,t_lags):
    
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
    alphas=np.arange(1, 10000, 100)
    alpha=1

    #-----UNIVARIATE-----------------------
    
    cu = grouped_col_uni2(3,t_lags,order)
    #---------------Non-Recursive-----------------
    
    
    regr_x= LassoCV(alphas=alphas,cv=10, random_state=0)
    #regr_x= LassoCV(alphas=alphas,cv=10, max_iter=100000, normalize=True)
    #regr_x=linear_model.Lasso(alpha)
    #regr_x=linear_model.LinearRegression()
    regr_x.fit(Xtrain[:,cu[0]], Ytrain[:,0])
    
    regr_y= LassoCV(alphas=alphas,cv=10, random_state=0)
    #regr_y=linear_model.Lasso(alpha)
    #regr_y=linear_model.LinearRegression()
    regr_y.fit(Xtrain[:,cu[1]], Ytrain[:,1])
    
    regr_z= LassoCV(alphas=alphas,cv=10, random_state=0)
    #regr_z=linear_model.Lasso(alpha)
    #regr_z=linear_model.LinearRegression()
    regr_z.fit(Xtrain[:,cu[2]], Ytrain[:,2])
    
    #====predict==================================
    
    Ynrx=regr_x.predict(Xtest[:,cu[0]])
    Ynry=regr_y.predict(Xtest[:,cu[1]])
    Ynrz=regr_z.predict(Xtest[:,cu[2]])
    
    #----RECURSIVE-------------------------------------
    
    rx= LassoCV(alphas=alphas,cv=10, random_state=0)
    #rx=linear_model.Lasso(alpha)
    #rx=linear_model.LinearRegression()
    rx.fit(Xr_train[:,cu[0]], Yr_train[:,0])
    
    ry= LassoCV(alphas=alphas,cv=10, random_state=0)
    #ry=linear_model.Lasso(alpha)
    #ry=linear_model.LinearRegression()
    ry.fit(Xr_train[:,cu[1]], Yr_train[:,1])
    
    rz= LassoCV(alphas=alphas,cv=10, random_state=0)
    #rz=linear_model.Lasso(alpha)
    #rz=linear_model.LinearRegression()
    rz.fit(Xr_train[:,cu[2]], Yr_train[:,2])
    
    Yp=swap_uni(Xtest,Ytest,rx,ry,rz,t_steps,t_lags,ncol,order)
    
    
    #============MULTIVARIATE======================
    c = grouped_col_multi(3,t_lags,order)
    #---------------Non-Recursive-----------------
    model_x= LassoCV(alphas=alphas,cv=10, random_state=0)
    #model_x=linear_model.Lasso(alpha)
    #model_x=linear_model.LinearRegression()
    model_x.fit(Xtrain, Ytrain[:,0])
    
    model_y= LassoCV(alphas=alphas,cv=10, random_state=0)
    #model_y=linear_model.Lasso(alpha)
    #model_y=linear_model.LinearRegression()
    model_y.fit(Xtrain, Ytrain[:,1])
    
    model_z= LassoCV(alphas=alphas,cv=10, random_state=0)
    #model_z=linear_model.Lasso(alpha)
    #model_z=linear_model.LinearRegression()
    model_z.fit(Xtrain, Ytrain[:,2])
    
    #====predict==================================
    
    YNRx=model_x.predict(Xtest)
    YNRy=model_y.predict(Xtest)
    YNRz=model_z.predict(Xtest)
    
    #----RECURSIVE-------------------------------------
    
    Rx= LassoCV(alphas=alphas,cv=10, random_state=0)
    #Rx=linear_model.Lasso(alpha)
    #Rx=linear_model.LinearRegression()
    Rx.fit(Xr_train, Yr_train[:,0])
    
    Ry= LassoCV(alphas=alphas,cv=10,random_state=0)
    #Ry=linear_model.Lasso(alpha)
    #Ry=linear_model.LinearRegression()
    Ry.fit(Xr_train, Yr_train[:,1])
    
    Rz= LassoCV(alphas=alphas,cv=10,random_state=0)
    #Rz=linear_model.Lasso(alpha)
    #Rz=linear_model.LinearRegression()
    Rz.fit(Xr_train, Yr_train[:,2])
    
    YP=swap_multi(Xtest,Ytest,Rx,Ry,Rz,t_steps,t_lags,ncol,order)
    cd=coef_det(Xtest,Ytest,regr_x,regr_y,regr_z,rx,ry,rz,model_x,model_y,model_z,Rx,Ry,Rz,cu)
    #plot_ts(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP,start,end,it_d)
    #plot_error(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP,start,end,it_d)
    #scatter_plots(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP,start,end)
    
    return Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP,it_d,cd



    
def cd_lt2(CD,lead_time):
    
    fig, axs = plt.subplots(3, sharex=False, sharey=False, figsize=(15, 15))
    fig.suptitle(' Predited values v/s lead_time; dt=0.01')
    
    axs[0].plot(lead_time,CD[:,0],'^',markersize=12)
    axs[0].plot(lead_time,CD[:,3],'*',markersize=12)
    axs[0].plot(lead_time,CD[:,6],'+',markersize=12)
    axs[0].plot(lead_time,CD[:,9],'.',markersize=15)
    #axs[0,0].plot(ynrx,'r')
    #axs[0,0].plot(Ytrue[:,0]-ynrx,'g')
    axs[0].set_xlabel("lead time")
    axs[0].set_ylabel("coef_of_det")
    axs[0].set_title("X component")
    
    axs[1].plot(lead_time,CD[:,1],'^',markersize=12)
    axs[1].plot(lead_time,CD[:,4],'*',markersize=12)
    axs[1].plot(lead_time,CD[:,7],'+',markersize=12)
    axs[1].plot(lead_time,CD[:,10],'.',markersize=15)
    #axs[1,0].plot(ynry,'r')
    axs[1].set_xlabel("lead time")
    axs[1].set_ylabel("coef_of_det")
    axs[1].set_title("Y component")
    
    axs[2].plot(lead_time,CD[:,2],'^',markersize=12)
    axs[2].plot(lead_time,CD[:,5],'*',markersize=12)
    axs[2].plot(lead_time,CD[:,8],'+',markersize=12)
    axs[2].plot(lead_time,CD[:,11],'.',markersize=15)
    #axs[2,0].plot(ynrz,'r')
    axs[2].set_xlabel("lead time")
    axs[2].set_ylabel("coef_of_det")
    axs[2].set_title("Z component")
    
    
    
    fig.tight_layout()
    
def rmse_lt(Rmse,lead_time):
    
    fig, axs = plt.subplots(3, sharex=False, sharey=False, figsize=(15, 15))
    fig.suptitle(' Predited values v/s lead_time; dt=0.01')
    
    axs[0].plot(lead_time,Rmse[:,0],'^',markersize=12)
    axs[0].plot(lead_time,Rmse[:,3],'*',markersize=12)
    axs[0].plot(lead_time,Rmse[:,6],'+',markersize=12)
    axs[0].plot(lead_time,Rmse[:,9],'.',markersize=15)
    #axs[0,0].plot(ynrx,'r')
    #axs[0,0].plot(Ytrue[:,0]-ynrx,'g')
    axs[0].set_xlabel("lead time")
    axs[0].set_ylabel("RMSE")
    axs[0].set_title("X component")
    
    axs[1].plot(lead_time,Rmse[:,1],'^',markersize=12)
    axs[1].plot(lead_time,Rmse[:,4],'*',markersize=12)
    axs[1].plot(lead_time,Rmse[:,7],'+',markersize=12)
    axs[1].plot(lead_time,Rmse[:,10],'.',markersize=15)
    #axs[1,0].plot(ynry,'r')
    axs[1].set_xlabel("lead time")
    axs[1].set_ylabel("RMSE")
    axs[1].set_title("Y component")
    
    axs[2].plot(lead_time,Rmse[:,2],'^',markersize=12)
    axs[2].plot(lead_time,Rmse[:,5],'*',markersize=12)
    axs[2].plot(lead_time,Rmse[:,8],'+',markersize=12)
    axs[2].plot(lead_time,Rmse[:,11],'.',markersize=15)
    #axs[2,0].plot(ynrz,'r')
    axs[2].set_xlabel("lead time")
    axs[2].set_ylabel("RMSE")
    axs[2].set_title("Z component")
    
    
    
    fig.tight_layout()
    
    
def compare(pts2,it,it_d,ss,start=0,end=100):
    
    
    fig, axs = plt.subplots(3, sharex=False, sharey=False, figsize=(15, 15))
    fig.suptitle("Comparison of original and downscaled trajectories")
    
    axs[0].scatter(it[start:end],pts2[start:end,0],s=10)
    axs[0].scatter(it_d[start:end:ss],pts2[start:end:ss,0],s=10,marker='+')
    axs[0].set_xlabel("integration time")
    axs[0].set_ylabel("blue:orig, red: downscaled")
    axs[0].set_title("X component")
    
    axs[1].scatter(it[start:end],pts2[start:end,1],s=10)
    axs[1].scatter(it_d[start:end:ss],pts2[start:end:ss,1],s=10,marker='+')
    axs[1].set_xlabel("integration time")
    axs[1].set_ylabel("blue:orig, red: downscaled")
    axs[1].set_title("Y component")
    
    axs[2].scatter(it[start:end],pts2[start:end,2],s=10)
    axs[2].scatter(it_d[start:end:ss],pts2[start:end:ss,2],s=10,marker='+')
    axs[2].set_xlabel("integration time")
    axs[2].set_ylabel("blue:orig, red: downscaled")
    axs[2].set_title("Z component")
    
    fig.tight_layout()
    