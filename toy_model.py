#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:13:50 2021

@author: vborse
"""

import numpy as np



# STEP 2: CREATING POLYNOMIAL COVARIATES
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

def Ideal_lags2(_X,tsteps,tlags):
    Xti=_X[:-(tsteps+tlags)]
    for i in range(1,tlags+1):
        #xti=X[i:-(tsteps+tlags-i)]
        Xti=np.concatenate((Xti, _X[i:-(tsteps+tlags-i)]), axis=1)
        print(np.shape(Xti))
    
#    l= np.size(Xti,axis=1)
#    Xx=np.transpose(Xti,(1,0,2)).reshape(l,-1)   
    Y  =  _X[(tsteps+tlags):]
    
    return(Xti,Y)      
    
def grouped_col(ncol,t_lags,order=1):
    
    ff= np.arange(0,(ncol*(t_lags+1)),1)
    F=np.reshape(ff,((t_lags+1),ncol))
    
    return F

def swap(_X,_Y,t_steps,t_lags,ncol,order=1):
    
    _X = np.copy(_X)
    c=grouped_col(ncol,t_lags,order)
    print(_X)
    for i in range(1,t_steps+1):
        #Xnew=np.array([_Y,_Y,_Y])
        for k in range(t_lags+1):
            #print(k)
            
            if ((k+1)<=t_lags):
                
                _X[:,[c[k]]] = _X[:,[c[k+1]]]
                #print(np.shape(_X[:,k]))
            #_X[:,-(k+1)] = Xnew[k]
            else:
                _X[:,c[-1]] = _Y
            #print(_X)
        #print(_X[:,c[-1]])
        print(_X)
        print(_Y.T)
    return _Y




# STEP 1: CREATING X 

N=20
P=3

order=1
t_steps=1
t_lags=3
ncol=3

f = np.arange(1,N+1,1)
X=np.repeat(f,P).reshape(N,P)
  
Xt2,Yt2=Ideal_lags2(X,t_steps,t_lags)
Xt,Yt=Ideal_lags(X,t_steps,t_lags)
Xs=Ideal_poly(X,order)

c=grouped_col(3,3,1)
print(c)
Yr=swap(Xt2,Yt2,t_steps,t_lags,ncol,1)
    


