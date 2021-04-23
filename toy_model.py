#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:13:50 2021

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

# STEP 1: CREATING X 
order=5
xx=[]
f = np.arange(1,21,1)
X=np.repeat(f,order).reshape(20,order)

# STEP 2: CREATING POLYNOMIAL COVARIATES
def Ideal_poly(X,order):
    for i in range(order):
        x=np.power(X[:,i],i+1)
        xx.append(x)
    Xx=np.transpose(xx)
    return Xx

def ideal3(pt,t_steps):
    XT2 = pt[:-t_steps-2]
    XT1= pt[1:-t_steps-1]
    XT= pt[2:-t_steps]
    Y  = pt[t_steps+2:]
    
    return XT,XT1,XT2,Y

def Ideal(X,t_steps,t_lags):
    Xti=[]
    for i in range(1,t_lags+1):
        xti=X[t_lags-i:-t_steps-t_lags-i]
        Xti.append(xti)
        print(Xti)
    #XTi=np.transpose(Xti)
    #Y  = X[t_steps+t_lags:]
    
    #l=np.size(Xti,1)
    
        
    Y  =  X[t_steps+(2*t_lags):]
    return(Xti,Y)

Xs=Ideal_poly(X,order)
Xss,Yss=Ideal(f,5,3)
print(Xss)
print(Yss)   
print(np.size(Xss,1))
print(np.size(Yss))