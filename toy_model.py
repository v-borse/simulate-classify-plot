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

xx=[]
f = np.arange(1,N+1,1)
X=np.repeat(f,P).reshape(N,P)

# STEP 2: CREATING POLYNOMIAL COVARIATES
def Ideal_poly(X,order):
    for i in range(1,order+1):
        x=np.power(X,i)
        xx.append(x)
        
    return xx



def Ideal_lags(X,t_steps,t_lags):
    Xti=[]
    for i in range(0,t_lags+1):
        xti=X[i:-(t_steps+t_lags-i)]
        Xti.append(xti)
        print(np.shape(xti))
        
    Y  =  X[(t_steps+t_lags):]
    return(Xti,Y)
        
Xt,Yt=Ideal_lags(X,t_steps,t_lags)
l= np.size(Xt,axis=1)
Xx=np.transpose(Xt,(1,0,2)).reshape(l,-1)    

Xsss=Ideal_poly(X,order)
XX=np.transpose(Xsss,(1,0,2)).reshape(N,-1)

#print(Xx)
#print(Yt)
#
#print(XX)
