#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 09:47:15 2021

@author: vborse
"""

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

from toy_model import Ideal_poly
from toy_model import grouped_col_multi,swap, swap3,grouped_col_uni,Ideal_poly3

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

#def grouped_col(ncol,t_lags,order=1):
#    
#    ff= np.arange(0,(ncol*(t_lags+1)),1)
#    F=np.reshape(ff,((t_lags+1),ncol))
#    
#    return F

def GEN_R_predict_uni(_X,modelx,modely,modelz,t_steps,t_lags,ncol,order=1):
    
    _X = np.copy(_X)
    c=grouped_col_uni(ncol,t_lags,order)
    print(_X)
    for i in range(1,t_steps+1):
        print(i)
        Ynrx=modelx.predict(_X[:,c[:,0]])
        Ynry=modely.predict(_X[:,c[:,1]])
        Ynrz=modelz.predict(_X[:,c[:,2]])
        
        Xnew=np.array([Ynrx,Ynry,Ynrz])
        #print(np.shape(Xnew.T))
        
        #>>Xnew2,Ynew2= (Ideal_poly(Xnew,order,t_steps))
        #print(np.shape(Xnew[:,:,None]))
        
        #Xnew=Xnew.reshape((len(Ynrx),1,ncol))
        #print(np.shape(Xnew[0]))
        #print(np.shape(Xnew.T))
        #print(np.shape(Xnew.T[:,:,None]))
        for k in range(t_lags+1):
            print(k)
            
            if ((k+1)<=t_lags):
                
                _X[:,[c[k]]] = _X[:,[c[k+1]]]
                #print(np.shape(_X[:,k]))
            #_X[:,-(k+1)] = Xnew[k]
            else:
                _X[:,c[-1]] = Xnew.T
                #_X[:,c[-1]] = Xnew2.T
#        print(_X[:,c[-1]])
            #print(Xnew)
        print(np.shape(Xnew.T))
        print(Xnew)
    #>>return Xnew2.T
    return Xnew.T


def GEN_R_predict_multi(_X,modelx,modely,modelz,t_steps,t_lags,ncol,order):
    
    _X = np.copy(_X)
    c=grouped_col_multi(ncol,t_lags,order)
    print(_X)
    for i in range(1,t_steps+1):
        print(i)
        Ynrx=modelx.predict(_X)
        Ynry=modely.predict(_X)
        Ynrz=modelz.predict(_X)
        
        Xnew=np.array([Ynrx,Ynry,Ynrz])
        #print(np.shape(Xnew.T))
        
        #>>Xnew2,Ynew2= (Ideal_poly(Xnew,order,t_steps))
        #print(np.shape(Xnew[:,:,None]))
        
        #Xnew=Xnew.reshape((len(Ynrx),1,ncol))
        #print(np.shape(Xnew[0]))
        #print(np.shape(Xnew.T))
        #print(np.shape(Xnew.T[:,:,None]))
        for k in range(t_lags+1):
            print(k)
            
            if ((k+1)<=t_lags):
                
                _X[:,[c[k]]] = _X[:,[c[k+1]]]
                #print(np.shape(_X[:,k]))
            #_X[:,-(k+1)] = Xnew[k]
            else:
                _X[:,c[-1]] = Xnew.T
                #_X[:,c[-1]] = Xnew2.T
#        print(_X[:,c[-1]])
            #print(Xnew)
        print(np.shape(Xnew.T))
        print(Xnew)
    #>>return Xnew2.T
    return Xnew.T

def swap4(_X,_Y,modelx,modely,modelz,t_steps,t_lags,ncol,order):
    print("in swap4")
    #print(_X)
    
    _X = np.copy(_X)
    print(np.shape(_X))
    c=grouped_col_multi(ncol,t_lags,order)
#    print(np.shape(_X))
#    print(np.shape(_Y))
    
#    print(c)
    #print(Xt4)
    #_Ynrx=modelx.predict(_X[:,])
    
    print("after Ynrx")
    print(np.shape(_Y))
    for i in range(1,t_steps+1):
        
#        Ynrx = modelx.predict(_X[:,c[0]])
#        Ynry = modely.predict(_X[:,c[1]])
#        Ynrz = modelz.predict(_X[:,c[2]])
        
        Ynrx = modelx.predict(_X)
        Ynry = modely.predict(_X)
        Ynrz = modelz.predict(_X)
        
        print(np.shape(Ynrx))
        Xnew=np.array([Ynrx,Ynry,Ynrz])
        print(np.shape(Xnew))
        for k in range(t_lags+1):
            #print(k)
            
            if ((k+1)<=t_lags):
                
                _X.T[:,[c[:,k]]] = _X.T[:,[c[:,k+1]]]
                #print(np.shape(_X[:,k]))
            #_X[:,-(k+1)] = Xnew[k]
            else:
                
                xx=Ideal_poly3(Xnew.T,order,t_steps)
                print (xx)
                #xx=Ideal_poly3(_Y[:,c[0]],order,t_steps)
                print(np.shape(_Y[:,c[0]] ))
                print(np.shape(_X[:,c[-1]] ))
                print(np.shape(xx))
                #_X[c[:,-1]] = xx
                #_X[:,c[-1]] = xx[:,c[-1]]
                _X[:,c[-1]] = xx
                
            #print(Xt4)
            #print(">>>",k)
            #print(Xt4[c[:,-1]])
        #print(_X[:,c[-1]])
#        print(_X)
#        print(_Y.T)
    return _X[:,c[-1]]

def swap6(_X,_Y,modelx,modely,modelz,t_steps,t_lags,ncol,order):
    print("in swap6")
    #print(_X)
    
    _X = np.copy(_X)
    print(np.shape(_X))
    cu=grouped_col_uni(ncol,t_lags,order)
#    print(np.shape(_X))
#    print(np.shape(_Y))
    
    print(cu)
    print(_X[:,cu[1]])
    #print(Xt4)
    #_Ynrx=modelx.predict(_X[:,])
    
    print("after Ynrx")
    print(np.shape(_Y))
    for i in range(1,t_steps+1):
        
#        Ynrx = modelx.predict(_X[:,c[0]])
#        Ynry = modely.predict(_X[:,c[1]])
#        Ynrz = modelz.predict(_X[:,c[2]])
        
        YNrx = modelx.predict(_X[:,cu[0]])
        print(np.shape(_X[:,cu[1]]))
        print(np.shape(YNrx))
        YNry = modely.predict(_X[:,cu[1]])
        YNrz = modelz.predict(_X[:,cu[2]])
        
        print(np.shape(YNrx))
        Xnew=np.array([YNrx,YNry,YNrz])
        print(np.shape(Xnew))
        for k in range(t_lags+1):
            #print(k)
            
            if ((k+1)<=t_lags):
                
                _X.T[:,[cu[k]]] = _X.T[:,[cu[k+1]]]
                #print(np.shape(_X[:,k]))
            #_X[:,-(k+1)] = Xnew[k]
            else:
                
                xx=Ideal_poly3(Xnew.T,order,t_steps)
                print (xx)
                #xx=Ideal_poly3(_Y[:,c[0]],order,t_steps)
                print(np.shape(_Y[:,cu[0]] ))
                print(np.shape(_X[:,cu[-1]] ))
                print(np.shape(xx))
                #_X[c[:,-1]] = xx
                #_X[:,c[-1]] = xx[:,c[-1]]
                _X[:,cu[-1]] = xx
                
            #print(Xt4)
            #print(">>>",k)
            #print(Xt4[c[:,-1]])
        #print(_X[:,c[-1]])
#        print(_X)
#        print(_Y.T)
    return _X[:,cu[-1]]

def Rsq(actual, predict):
     
    corr_matrix = np.corrcoef(actual, predict)
    corr = corr_matrix[0,1]
    R_sq = corr**2
 
    #print(R_sq)
    return R_sq

def plot_R_multi(Y,YR):
    rsq=[]
    print(Y)
    print(YR)
    rYRx=Rsq(Y[:,0],YR[:,0])
    rYRy=Rsq(Y[:,1],YR[:,1])
    rYRz=Rsq(Y[:,2],YR[:,2])
    
    rsq=np.array([rYRx,rYRy,rYRz])
    return rsq

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

"""
Xr_t1,Yr_t1=Ideal_poly(pts,order,t_steps)
#Xr_t1,Yr_t1 = Ideal_lags(Xr_p1,1,t_lags)
Xtrain, Ytrain = Ideal_lags(Xr_t1,t_steps,t_lags)

Xr_test,Yr_test=Ideal_poly(pts2,order,t_steps)
Xtest, Ytest = Ideal_lags(Xr_test,t_steps,t_lags)
"""
#Xr_t1,Yr_t1=Ideal_poly(pts,order,1)
Xr_t1,Yr_t1 = Ideal_lags(pts,1,t_lags)
Xr_train, Yr_train = Ideal_poly(Xr_t1,order,t_steps)

#Xnr_t1,Ynr_t1=Ideal_poly(pts,order,t_steps)
Xnr_t1,Ynr_t1 = Ideal_lags(pts,t_steps,t_lags)
Xtrain, Ytrain = Ideal_poly(Xnr_t1,order,t_steps)

Xr_test,Yr_test=Ideal_lags(pts2,t_steps,t_lags)
Xtest, Ytest = Ideal_poly(Xr_test,order,t_steps)

#Xpr_t1,Ypr_t1 = Ideal_poly(pts,2,t_steps)
#Xp_train, Yp_train = Ideal_poly(pts,order,t_steps)
#Xp_test, Yp_test = Ideal_poly(pts2,order,t_steps)

"""
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
"""
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
Yp=swap4(Xtest,Ytest,Rx,Ry,Rz,t_steps,t_lags,ncol,order)


#plot_predicted_ts(Xtest, Ytest[:,0],YNRx,YR,0)
#plot_predicted_ts(Xtest, Ytest[:,1],YNRy,YR,1)
#plot_predicted_ts(Xtest, Ytest[:,2],YNRz,YR,2)
    
#RSQ1=plot_R_multi(Ytest,YR)

#plt.plot(Ytest[:9984,0])
##plt.plot(Ytest[:,1])
##plt.plot(Ytest[:,2])
#
#plt.plot(YNRx[:9984])
##plt.plot(YNRy)
##plt.plot(YNRz)
#
#plt.plot(Yp[:9984,-3])
##plt.plot(Ytest[:,1])
##plt.plot(Ytest[:,2])
#-----------------------------
#plt.plot(Yp[:500,-6])
#plt.plot(pts2[:500,0])
#plt.plot(Ytest[:200,0])
#plt.plot(YNRx[:200])

#plt.plot(Yp[:200,-2])
#plt.plot(Ytest[:200,1])
#plt.plot(YNRy[:200])
#
#plt.plot(Yp[:200,-1])
#plt.plot(Ytest[:200,2])
#plt.plot(YNRz[:200])
#----------------------------
#plt.plot(Yp[:len(Yp),-3])
#plt.plot(Ytest[:len(Yp),0])
#plt.plot(YNRx[:len(Yp)])

#plt.plot(Yp[:len(Yp),-2])
#plt.plot(Ytest[:len(Yp),1])
#plt.plot(YNRy[:len(Yp)])
#
#plt.plot(Yp[:len(Yp),-1])
#plt.plot(Ytest[:len(Yp),2])
#plt.plot(YNRz[:len(Yp)])
#----------------------

#plt.plot(abs(Yp[:200,-6])-Ytest[:200,0])
#plt.plot(Ytest[:200,0]-YNRx[:200])

#plt.plot(Ytest[:len(Yp),0]-Yp[:len(Yp),-3])
#plt.plot(Ytest[:len(Yp),0]-YNRx[:len(Yp)])