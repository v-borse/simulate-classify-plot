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
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import RepeatedKFold
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
t_steps=1
t_lags=1
ncol=3

pts=single_traj(4,-14,21,r,0.01,tlength) 
pts2=single_traj(1,-1,2.05,r,0.01,tlength)
pts3=single_traj(2,-4,6.05,r,0.01,tlength)
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

X_cv,Y_cv=Ideal_poly(pts3[::ss],order,t_steps)
Xcv, Ycv = Ideal_lags(X_cv,t_steps,t_lags)


cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
cv = None

alphas=np.arange(0.1, 10, 0.1)
alpha=1

#-----UNIVARIATE-----------------------

cu = grouped_col_uni2(3,t_lags,order)
#---------------Non-Recursive-----------------
regr_x= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error', store_cv_values=True)
#regr_x=linear_model.LinearRegression(alpha)
regr_x.fit(Xtrain[:,cu[0]], Ytrain[:,0])

regr_y= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
#regr_y=linear_model.Ridge(alpha)
regr_y.fit(Xtrain[:,cu[1]], Ytrain[:,1])

regr_z= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
#regr_z=linear_model.Ridge(alpha)
#regr_z=linear_model.LinearRegression()
regr_z.fit(Xtrain[:,cu[2]], Ytrain[:,2])

#====predict==================================

Ynrx=regr_x.predict(Xtest[:,cu[0]])
Ynry=regr_y.predict(Xtest[:,cu[1]])
Ynrz=regr_z.predict(Xtest[:,cu[2]])

#----RECURSIVE-------------------------------------
rx= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
#rx=linear_model.Ridge(alpha)
#rx=linear_model.LinearRegression()
rx.fit(Xr_train[:,cu[0]], Yr_train[:,0])

ry= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
#ry=linear_model.Ridge(alpha)
#ry=linear_model.LinearRegression()
ry.fit(Xr_train[:,cu[1]], Yr_train[:,1])

rz= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
#rz=linear_model.Ridge(alpha)
#rz=linear_model.LinearRegression()
rz.fit(Xr_train[:,cu[2]], Yr_train[:,2])

Yp=swap_uni(Xtest,Ytest,rx,ry,rz,t_steps,t_lags,ncol,order)


#============MULTIVARIATE======================
c = grouped_col_multi(3,t_lags,order)
#---------------Non-Recursive-----------------

model_x= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
#model_x=linear_model.Ridge(alpha)
#model_x=linear_model.LinearRegression()
model_x.fit(Xtrain, Ytrain[:,0])

model_y= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
#model_y=linear_model.Ridge(alpha)
#model_y=linear_model.LinearRegression()
model_y.fit(Xtrain, Ytrain[:,1])

model_z= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
#model_z=linear_model.Ridge(alpha)
#model_z=linear_model.LinearRegression()
model_z.fit(Xtrain, Ytrain[:,2])

#====predict==================================

YNRx=model_x.predict(Xtest)
YNRy=model_y.predict(Xtest)
YNRz=model_z.predict(Xtest)

#----RECURSIVE-------------------------------------

Rx= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
#Rx=linear_model.Ridge(alpha)
#Rx=linear_model.LinearRegression()
Rx.fit(Xr_train, Yr_train[:,0])

Ry= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
#Ry=linear_model.Ridge(alpha)
#Ry=linear_model.LinearRegression()
Ry.fit(Xr_train, Yr_train[:,1])

Rz= RidgeCV(alphas, cv=cv, scoring='neg_mean_absolute_error')
#Rz=linear_model.Ridge(alpha)
#Rz=linear_model.LinearRegression()
Rz.fit(Xr_train, Yr_train[:,2])

YP=swap_multi(Xtest,Ytest,Rx,Ry,Rz,t_steps,t_lags,ncol,order)

plot_ts(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP)
#plot_error(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP)

print("regr_x=",regr_x.score(Xtest[:,cu[0]],Ytest[:,0]))
print("regr_y=",regr_y.score(Xtest[:,cu[1]],Ytest[:,1]))
print("regr_z=",regr_z.score(Xtest[:,cu[2]],Ytest[:,2]))

print("rx=",rx.score(Xtest[:,cu[0]],Ytest[:,0]))
print("ry=",ry.score(Xtest[:,cu[1]],Ytest[:,1]))
print("rz=",rz.score(Xtest[:,cu[2]],Ytest[:,2]))

print("model_x=",model_x.score(Xtest,Ytest[:,0]))
print("model_y=",model_y.score(Xtest,Ytest[:,1]))
print("model_z=",model_z.score(Xtest,Ytest[:,2]))

print("Rx=",Rx.score(Xtest,Ytest[:,0]))
print("Ry=",Ry.score(Xtest,Ytest[:,1]))
print("Rz=",Rz.score(Xtest,Ytest[:,2]))


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

plt.plot([cd[0],cd[3],cd[6],cd[9]])
plt.plot([cd[1],cd[4],cd[7],cd[10]])
plt.plot([cd[2],cd[5],cd[8],cd[11]])

"""
plt.figure(figsize=(5, 3))

for Model in [Lasso,Ridge]:
    scores = [cross_val_score(Model(alpha), Xcv, Ycv, cv=10).mean()
            for alpha in alphas]
    #print(scores)
    plt.plot(alphas, scores, label=Model.__name__)

plt.legend(loc='lower left')
plt.xlabel('alpha')
plt.ylabel('cross validation score')
plt.tight_layout()
plt.show()
"""
