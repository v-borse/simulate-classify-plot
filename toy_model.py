#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:13:50 2021

@author: vborse
"""

import numpy as np

# STEP 1: CREATING X 

N=20
P=3

order=5
t_steps=1
t_lags=1


f = np.arange(1,N+1,1)
X=np.repeat(f,P).reshape(N,P)

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
        Xti=np.concatenate((_X[i:-(tsteps+tlags-i)], Xti), axis=1)
        print(np.shape(Xti))
    
#    l= np.size(Xti,axis=1)
#    Xx=np.transpose(Xti,(1,0,2)).reshape(l,-1)   
    Y  =  _X[(tsteps+tlags):]
    
    return(Xti,Y)      
    
Xt2,Yt2=Ideal_lags2(X,t_steps,t_lags)
Xt,Yt=Ideal_lags(X,t_steps,t_lags)
Xs=Ideal_poly(X,order)

    


#print(Xx)
#print(Yt)

#print(XX)
