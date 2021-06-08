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

def lorenz(x, y, z,dt, r, s=10, b=8/3):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
       t: step size
    Returns:
       dt*x_dot, dt*y_dot, dt*z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    
    return dt*x_dot, dt*y_dot, dt*z_dot

def single_traj(x0,y0,z0,r,dt,num_steps):
    
    pt=[]
    #for single trajectory
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)
        
    # Set initial values
    xs[0], ys[0], zs[0]= (x0, y0, z0) 
    
        
        
    for i in range(num_steps):
      """
      EULER SCHEME
            
      dt*x_dot, dt*y_dot, dt*z_dot = lorenz(xs[i], ys[i], zs[i])
      xs[i + 1] = xs[i] + dt*x_dot
      ys[i + 1] = ys[i] + dt*y_dot
      zs[i + 1] = zs[i] + dt*z_dot
      """
      # RK 4 SCHEME
            
      
            
      k1_x, k1_y, k1_z = lorenz(xs[i], ys[i], zs[i], dt,r)
      k2_x, k2_y, k2_z = lorenz(xs[i]+0.5*k1_x, ys[i]+0.5*k1_y, zs[i]+0.5*k1_z, dt,r)
      k3_x, k3_y, k3_z = lorenz(xs[i]+0.5*k2_x, ys[i]+0.5*k2_y, zs[i]+0.5*k2_z, dt,r)
      k4_x, k4_y, k4_z = lorenz(xs[i]+k3_x, ys[i]+k3_y, zs[i]+k3_z, dt,r)
      xs[i + 1] = xs[i] + ((k1_x+2*k2_x+2*k3_x+k4_x) /6.0)
      ys[i + 1] = ys[i] + ((k1_y+2*k2_y+2*k3_y+k4_y) /6.0)
      zs[i + 1] = zs[i] + ((k1_z+2*k2_z+2*k3_z+k4_z) /6.0)
    
        
    
    pt=np.transpose(np.array([xs,ys,zs]))
    
    return pt

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
                A=np.squeeze(Xnew, axis=2)
                xx=Ideal_poly3(A.T,order,t_steps)
                
                print(np.shape(A))
                #print(np.shape(_X[:,c[-1]]))
                print(np.shape(xx))
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
