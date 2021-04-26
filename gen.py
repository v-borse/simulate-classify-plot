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
    fig, axs = plt.subplots(4, 2, sharex=False, sharey=False, figsize=(15, 15))
    fig.suptitle(' Linear Regression; dt=0.01')
    
    
    axs[0,0].scatter(Yp, Y,c='k',s=2)
    axs[0,0].plot(Y,Y,'g')
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
    
    #axs[1,0].scatter(X2[:, index], Y2[:, index],c='k', s=2)
    
    axs[1,0].scatter(X, Yp,c='b', s=2)
    axs[1,0].scatter(X, Yrp[:, index],c='r', s=2) 
    #axs[1,0].set_xlim([-50,50])
    #axs[1,0].set_ylim([-50,50])
    axs[1,0].set_xlabel("X(x,y,z)")
    axs[1,0].set_ylabel("Y")
    axs[1,0].set_title("X and Y")
    
    axs[1,1].plot((Yp-Y), 'b')
    axs[1,1].plot(Yrp[:, index]-Y, 'r')
    #axs[2,0].set_ylim([-50,50])
    axs[1,1].set_ylabel("error")
    axs[1,1].set_xlabel("time_steps")
    axs[1,1].set_title("Errors for LN")
    
    
    axs[2,0].plot(X[:5000],'g^',Y[:5000],'y.')
    #axs[1,1].plot(Y2[:, index])
    #axs[1,1].plot(Yp)
    #axs[1,1].plot(Yrp[:, index])
    #axs[1,1].set_ylim([-60,60])
    axs[2,0].set_xlabel("X and Y_ideal Time series")
    
    axs[2,1].plot(X[5000:],'g^',Y[5000:],'y.')
    #axs[1,1].plot(Y2[:, index])
    #axs[1,1].plot(Yp)
    #axs[1,1].plot(Yrp[:, index])
    #axs[1,1].set_ylim([-60,60])
    axs[2,1].set_xlabel("X and Y_ideal Time series")
    
    
    axs[3,0].plot(Yp[:5000],'b^',Yrp[:5000, index],'r.')
    #axs[1,1].plot(Y2[:, index])
    #axs[1,1].plot(Yp)
    #axs[1,1].plot(Yrp[:, index])
    #axs[1,1].set_ylim([-60,60])
    axs[3,0].set_xlabel("predicted Time series")
    
    axs[3,1].plot(Yp[5000:],'b^')
    axs[3,1].plot(Yrp[5000:,index],'r.')
    #axs[3,1].plot(Yp[5000:],'b^',Yrp[5000:, index],'r.')
    #axs[1,1].plot(Y2[:, index])
    #axs[1,1].plot(Yp)
    #axs[1,1].plot(Yrp[:, index])
    #axs[1,1].set_ylim([-60,60])
    axs[3,1].set_xlabel("predicted Time series")
       
    plt.show()
    
def coef_of_detrm(actual,predicted):
     
    corr_matrix = np.corrcoef(actual, predicted)
    corr = corr_matrix[0,1]
    R_sq = corr**2
    sns.heatmap(corr_matrix)
    plt.show()
    return R_sq


def Ideal_poly(_X,_order):
    xx=[]
    for i in range(1,_order+1):
        x=np.power(_X,i)
        xx.append(x)
    XX=np.transpose(xx,(1,0,2)).reshape(N,-1)
    
    return XX

def Ideal_lags(_X,tsteps,tlags):
    Xti=[]
    for i in range(0,tlags+1):
        xti=_X[i:-(tsteps+tlags-i)]
        Xti.append(xti)
        print(np.shape(xti))
    
    l= np.size(Xti,axis=1)
    Xx=np.transpose(Xti,(1,0,2)).reshape(l,-1)   
    Y  =  _X[(tsteps+tlags):]
    
    return(Xx,Y)

def create_df(_X):
    
    col=[str(j).zfill(2) for j in range(1,(np.size(_X,axis=1))+1)]
    #print(col)
    dfg = pd.DataFrame(_X, columns = [col])
    
    return dfg

    

def gen_r_predict_uni(dfg,modelx,modely,modelz,t_steps):
#def gen_r_predict_uni(dfg,t_steps):
    
#    co=np.arange(0,np.size(dfg,axis=1),3)
#    print(co)
    cox=[str(j).zfill(2) for j in range(4,np.size(dfg,axis=1)+1,3)]
    coy=[str(j).zfill(2) for j in range(5,np.size(dfg,axis=1)+1,3)]
    coz=[str(j).zfill(2) for j in range(6,np.size(dfg,axis=1)+1,3)]
#    print(cox)
#    print(coy)
#    print(coz)
#    print(dfg)
    for i in range(1,t_steps+1):
        tx= dfg[[cox[0],cox[1]]]
        #print(tx)
        ty=dfg[[coy[0],coy[1]]]
        tz=dfg[[coz[0],coz[1]]]
         
        Yrx=modelx.predict(tx)
        Yry=modely.predict(ty)
        Yrz=modelz.predict(tz)
        
        dfg[cox[0]] = dfg[cox[1]]
        dfg[coy[0]] = dfg[coy[1]]
        dfg[coz[0]] = dfg[coz[1]]
        
        dfg[cox[0]] = Yrx
        dfg[coy[0]] = Yry
        dfg[coz[0]] = Yrz
        
    Xnew = np.array([Yrx,Yry,Yrz])
    print(dfg)
    return Xnew.T
    
def gen_R_predict_multi(dfg,modelx,modely,modelz,t_steps):
    
    cox=[str(j).zfill(2) for j in range(4,np.size(dfg,axis=1)+1,3)]
    coy=[str(j).zfill(2) for j in range(5,np.size(dfg,axis=1)+1,3)]
    coz=[str(j).zfill(2) for j in range(6,np.size(dfg,axis=1)+1,3)]
    
    for i in range(1,t_steps+1):
        
        tx=dfg[[cox[0],cox[1],coy[0],coy[1],coz[0],coz[1]]] # Multivariate
        
        Yrx=modelx.predict(tx)
        Yry=modely.predict(tx)
        Yrz=modelz.predict(tx)
        
        dfg[cox[0]] = dfg[cox[1]]
        dfg[coy[0]] = dfg[coy[1]]
        dfg[coz[0]] = dfg[coz[1]]
        
        dfg[cox[0]] = Yrx
        dfg[coy[0]] = Yry
        dfg[coz[0]] = Yrz
        
    Xnew = np.array([Yrx,Yry,Yrz])
    print(tx)
    return Xnew.T
    
r=28
R=1
tlength = 10000

N=len(pts)
#P=3

order=3
t_steps=5
t_lags=2

pts=single_traj(4,-14,21,r,0.01,tlength) 
pts2=single_traj(1,-1,2.05,r,0.01,tlength)

X6 = Ideal_poly(pts,order)
X5, Y5 = Ideal_lags(pts,t_steps,t_lags)
X7, Y7 = Ideal_lags(pts2,t_steps,t_lags)
df=create_df(X5)
#gen_r_predict_uni(df,t_steps)

#----------UNIVARIATE-------------------------------------

#---------Non- Recursive---------------------------------------
X_train=create_df(X5)
X_test=create_df(X7)

colx=[str(j).zfill(2) for j in range(4,np.size(X_train,axis=1)+1,3)]
coly=[str(j).zfill(2) for j in range(5,np.size(X_train,axis=1)+1,3)]
colz=[str(j).zfill(2) for j in range(6,np.size(X_train,axis=1)+1,3)]
# prediction with sklearn
xx=X_train[[colx[0],colx[1]]] #Univariate
xy=X_train[[coly[0],coly[1]]]
xz=X_train[[colz[0],colz[1]]]
yx=Y5[:,0]
yy=Y5[:,1]
yz=Y5[:,2]

regr_x = linear_model.LinearRegression()
regr_x.fit(xx, yx)

print('Intercept: \n', regr_x.intercept_)
print('Coefficients: \n', regr_x.coef_)

regr_y = linear_model.LinearRegression()
regr_y.fit(xy, yy)


regr_z = linear_model.LinearRegression()
regr_z.fit(xz, yz)


txx=X_test[[colx[0],colx[1]]] 
txy=X_test[[coly[0],coly[1]]] 
txz=X_test[[colz[0],colz[1]]] 


Ynrx=regr_x.predict(txx)
Ynry=regr_y.predict(txy)
Ynrz=regr_z.predict(txz)

#plot_predicted_ts(X2,Y2,predictions_x,0)
#plot_predicted_ts(X2,Y2,predictions_y,1)
#plot_predicted_ts(X2,Y2,predictions_z,2)


#------RECURSIVE----------------------


rx = linear_model.LinearRegression()
rx.fit(xx, yx)

ry = linear_model.LinearRegression()
ry.fit(xy, yy)

rz = linear_model.LinearRegression()
rz.fit(xz, yz)

#-------prediction-------------------

Yr = gen_r_predict_uni(X_test,rx,ry,rz,t_steps)

#plot_predicted_ts(X7[:,0],Y7[:,0],Ynrx,Yr[:len(Ynrx[:,None])],0)
#plot_predicted_ts(X7[:,1],Y7[:,1],Ynry,Yr[:len(Ynry[:,None])],1)
#plot_predicted_ts(X7[:,2],Y7[:,2],Ynrz,Yr[:len(Ynrz[:,None])],2)
#
#sns.heatmap(np.corrcoef(Y7[:,0],Yr[:,0]),vmin=0.9,vmax=1)


#============MULTIVARIATE======================

#---------------Non-Recursive-----------------

col_m=[str(j).zfill(2) for j in range(4,np.size(X_train,axis=1)+1)]
x=X_train[[col_m[0],col_m[1],col_m[2],col_m[3],col_m[4],col_m[5]]] # Multivariate

yx=Y5[:,0]
yy=Y5[:,1]
yz=Y5[:,2]


model_x=linear_model.LinearRegression()
model_x.fit(x,yx)

model_y=linear_model.LinearRegression()
model_y.fit(x,yy)

model_z=linear_model.LinearRegression()
model_z.fit(x,yz)

#====predict==================================

Tx=X_test[[col_m[0],col_m[1],col_m[2],col_m[3],col_m[4],col_m[5]]]

YNRx=model_x.predict(Tx)
YNRy=model_y.predict(Tx)
YNRz=model_z.predict(Tx)

#plot_predicted_ts(X2,Y2,Ynrx,0)
#plot_predicted_ts(X2,Y2,Ynry,1)
#plot_predicted_ts(X2,Y2,Ynrz,2)

#----RECURSIVE-------------------------------------

Rx=linear_model.LinearRegression()
Rx.fit(x,yx)

Ry=linear_model.LinearRegression()
Ry.fit(x,yy)

Rz=linear_model.LinearRegression()
Rz.fit(x,yz)

YR = gen_R_predict_multi(X_test,Rx,Ry,Rz,t_steps)

plot_predicted_ts(X7[:,0],Y7[:,0],YNRx,YR[:len(YNRx[:,None])],0)
plot_predicted_ts(X7[:,1],Y7[:,1],YNRy,YR[:len(YNRy[:,None])],1)
plot_predicted_ts(X7[:,2],Y7[:,2],YNRz,YR[:len(YNRz[:,None])],2)

sns.heatmap(np.corrcoef(Y7[:,0],YR[:,0]),vmin=0.9,vmax=1)
