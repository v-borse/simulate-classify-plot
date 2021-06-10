#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 10:44:00 2021

@author: vborse
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:29:54 2021

@author: vborse
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 16:28:15 2021

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
from sklearn.linear_model import Lasso
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

import module
import module2
    
#----FORMING LORENZ TRAJECTORIES-------

r=28
R=1
tlength = 9999

order=1
t_steps=5
t_lags=2
ncol=3

pts=single_traj(4,-14,21.07,r,0.01,tlength) 
pts2=single_traj(1,-1,2.05,r,0.01,tlength)
N=len(pts)
ss=10
# --------CREATING DATASETS------------------

Xr_t1,Yr_t1 = Ideal_poly(pts[::ss],order,t_steps)
Xr_train, Yr_train = Ideal_lags(Xr_t1,1,t_lags)


Xnr_t1,Ynr_t1 = Ideal_poly(pts[::ss],order,t_steps)
Xtrain, Ytrain = Ideal_lags(Xnr_t1,t_steps,t_lags)

Xr_test,Yr_test=Ideal_poly(pts2[::ss],order,t_steps)
Xtest, Ytest = Ideal_lags(Xr_test,t_steps,t_lags)
#Xtest = Xtrain
#Ytest = Ytrain

alpha=1

#-----UNIVARIATE-----------------------

cu = grouped_col_uni2(3,t_lags,order)
#---------------Non-Recursive-----------------
regr_x=linear_model.Lasso(alpha)
#regr_x=linear_model.LinearRegression()
regr_x.fit(Xtrain[:,cu[0]], Ytrain[:,0])

regr_y=linear_model.Lasso(alpha)
#regr_y=linear_model.LinearRegression()
regr_y.fit(Xtrain[:,cu[1]], Ytrain[:,1])

regr_z=linear_model.Lasso(alpha)
#regr_z=linear_model.LinearRegression()
regr_z.fit(Xtrain[:,cu[2]], Ytrain[:,2])

#====predict==================================

Ynrx=regr_x.predict(Xtest[:,cu[0]])
Ynry=regr_y.predict(Xtest[:,cu[1]])
Ynrz=regr_z.predict(Xtest[:,cu[2]])

#----RECURSIVE-------------------------------------

rx=linear_model.Lasso(alpha)
#rx=linear_model.LinearRegression()
rx.fit(Xr_train[:,cu[0]], Yr_train[:,0])

ry=linear_model.Lasso(alpha)
#ry=linear_model.LinearRegression()
ry.fit(Xr_train[:,cu[1]], Yr_train[:,1])

rz=linear_model.Lasso(alpha)
#rz=linear_model.LinearRegression()
rz.fit(Xr_train[:,cu[2]], Yr_train[:,2])

Yp=swap_uni(Xtest,Ytest,rx,ry,rz,t_steps,t_lags,ncol,order)


#============MULTIVARIATE======================
c = grouped_col_multi(3,t_lags,order)
#---------------Non-Recursive-----------------

model_x=linear_model.Lasso(alpha)
#model_x=linear_model.LinearRegression()
model_x.fit(Xtrain, Ytrain[:,0])

model_y=linear_model.Lasso(alpha)
#model_y=linear_model.LinearRegression()
model_y.fit(Xtrain, Ytrain[:,1])

model_z=linear_model.Lasso(alpha)
#model_z=linear_model.LinearRegression()
model_z.fit(Xtrain, Ytrain[:,2])

#====predict==================================

YNRx=model_x.predict(Xtest)
YNRy=model_y.predict(Xtest)
YNRz=model_z.predict(Xtest)

#----RECURSIVE-------------------------------------

Rx=linear_model.Lasso(alpha)
#Rx=linear_model.LinearRegression()
Rx.fit(Xr_train, Yr_train[:,0])

Ry=linear_model.Lasso(alpha)
#Ry=linear_model.LinearRegression()
Ry.fit(Xr_train, Yr_train[:,1])

Rz=linear_model.Lasso(alpha)
#Rz=linear_model.LinearRegression()
Rz.fit(Xr_train, Yr_train[:,2])

YP=swap_multi(Xtest,Ytest,Rx,Ry,Rz,t_steps,t_lags,ncol,order)

plot_ts(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP)
#plot_error(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP)
