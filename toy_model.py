#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:13:50 2021

@author: vborse
"""

import numpy as np



# STEP 2: CREATING POLYNOMIAL COVARIATES
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
    #ly= np.size(yy,axis=1)
    #XX=np.transpose(xx,(1,0,2)).reshape(N,-1)
    XX=np.transpose(xx,(1,0,2)).reshape(l,-1)
    #YY=np.transpose(yy,(1,0,2)).reshape(ly,-1)
    
    return XX

def Ideal_poly2(_x,_order,t_steps):
    _X=_x[:-t_steps]
    Y =_x[t_steps:]
    XX=[]
    for i in range(1,_order+1):
        
        XX=np.concatenate((_X, np.power(_X,i)), axis=1)
        #print(np.shape(XX))
    
    
    return XX,Y



def Ideal_lags(_X,tsteps,tlags):
    Xti=[]
    for i in range(0,tlags+1):
        xti=_X[i:-(tsteps+tlags-i)]
        Xti.append(xti)
        #print(np.shape(xti))
    
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
    
def grouped_col(ncol,t_lags,order):
    
    ff= np.arange(0,(ncol*(t_lags+1)*order),1)
    F=np.reshape(ff,((t_lags+1),ncol*order))
    
    return F

def swap(_X,_Y,t_steps,t_lags,ncol,order):
    
    _X = np.copy(_X)
    c=grouped_col(ncol,t_lags,order)
    
#    print(c)
    print(_X)
    
    for i in range(1,t_steps+1):
        #Xnew=np.array([_Y,_Y,_Y])
        for k in range(t_lags+1):
            #print(k)
            
            if ((k+1)<=t_lags):
                
                _X.T[:,[c[k]]] = _X.T[:,[c[k+1]]]
                #print(np.shape(_X[:,k]))
            #_X[:,-(k+1)] = Xnew[k]
            else:
                _X.T[:,c[-1]] = _Y.T[:,c[0]]
                
            #print(_X)
            print(">>>",k)
            print(_X)
        #print(_X[:,c[-1]])
#        print(_X)
#        print(_Y.T)
    return _X.T[:,c[-1]]

def swap3(_X,_Y,t_steps,t_lags,ncol,order):
    
    _X = np.copy(_X)
    c=grouped_col(ncol,t_lags,order)
#    print(np.shape(_X))
#    print(np.shape(_Y))
    
#    print(c)
    #print(_X)
    
    for i in range(1,t_steps+1):
        #Xnew=np.array([_Y,_Y,_Y])
        for k in range(t_lags+1):
            #print(k)
            
            if ((k+1)<=t_lags):
                
                _X.T[:,[c[:,k]]] = _X.T[:,[c[:,k+1]]]
                #print(np.shape(_X[:,k]))
            #_X[:,-(k+1)] = Xnew[k]
            else:
                xx=Ideal_poly3(_Y[c[:,0]],order,t_steps)
#                print(np.shape(_X[c[:,-1]] ))
#                print(np.shape(xx))
                _X[c[:,-1]] = xx
                
            print(_X)
            print(">>>",k)
            print(_X[c[:,-1]])
        #print(_X[:,c[-1]])
#        print(_X)
#        print(_Y.T)
    return _X[:,c[-1]]


# STEP 1: CREATING X 

N=60
P=3

order=3
t_steps=1
t_lags=2
ncol=3

f = np.arange(1,N+1,1)
X=np.repeat(f,P).reshape(N,P)
cc=np.arange(1,order+1,1)
cp=grouped_col(ncol,t_lags,order)

"""
Xt2,Yt2=Ideal_lags2(X,t_steps,t_lags)
Xt,Yt=Ideal_lags(X,t_steps,t_lags)


Xs,Ys=Ideal_poly(Xt,order,t_steps)
Xp2,Yp2=Ideal_poly(X,5,t_steps)
#Yp=swap_poly(Xp2,Yp2,t_steps,t_lags,ncol,5)
Yp=swap(Xs,Ys,t_steps,t_lags,ncol,order)

"""
Xp3,Yp3=Ideal_poly(X,order,t_steps)
Xt3,Yt3=Ideal_lags(Xp3,t_steps,t_lags)
Yp33=swap3(Xt3,Yt3,t_steps,t_lags,ncol,order)   

#cxv=Ideal_poly3(X,order,t_steps)

#c=grouped_col(3,3,1)
##print(c)
#Yr=swap(Xt2,Yt2,t_steps,t_lags,ncol,1)
    



