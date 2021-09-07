#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 16:26:01 2021

@author: vborse
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 14:47:48 2021

@author: vborse
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 09:28:38 2021

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
from sklearn.linear_model import Lasso,LassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn import linear_model
import statsmodels.api as sm
from matplotlib import cm
import matplotlib.pyplot as plt
plt.style.use('bmh')
import seaborn as sns
import numpy as np
from statistics import variance
import cmath
from mlxtend.evaluate import bias_variance_decomp
from sklearn.metrics import r2_score

import module
import module2


dt=0.01
r=28
R=1

xx = [  2,  15, -26,   3, -17,   2,  -9,  27,  22, -14, -24,  12,   1,
         5,  22, -12, -28, -15,   5,  28, -26, -29,  21,  -8,  12, -29,
        16,  23,  19,  -9, -18,  25, -14,  25, -20,   4,  24,  -1,  15,
        12, -20, -18,  11, -10, -25,   6, -27,  13,  23,  -5,  20,   3,
        27,  28, -30,  12, -14,  -1,   4, -14,  24,  22, -20, -15,  19,
       -25, -17,  11, -11,  16,  17,  -4, -29,   7,  23, -30,   0,  11,
        10, -18,  26,   6,   8,   0,   0,  27, -25, -21,  25,  28,  24,
         2,  18,   7, -19, -26, -19, -12, -22,   4]
yy = [ 24, -14,  28,   9,  23, -16, -30,  -7, -24,  -4,  25,  21,   5,
         6,  -1, -24,  -5,  -2, -15,  -6, -20,  17, -27, -25, -24, -29,
        10,  -6,  21,  28,  13, -28, -25,  16,  -8,   6,   8, -13, -26,
        22,  17,  28, -30,  -4,   1,  -4, -10,  -9,  19, -17,  27,  28,
        26,  -2, -29,  10, -18,   5,   5,   1,   6,   6,  10, -15,   0,
        -5,  21,   1,   7,   4,  10,  -8, -29,  -5,  14,   0, -11, -23,
        29,   7,   2, -26,  19, -17,   8, -20, -29,   3,  -4,  11,  15,
        21,  11, -15, -27,  15, -19,   0, -27, -14]
zz = [ 7,  7,  9,  6, 13,  4, 27, 19,  1,  6, 17,  7, 25, 22, 26, 14,  4,
       23,  8,  7,  3, 17, 24,  6, 23, 23, 18,  4,  8,  3,  9, 19, 29, 22,
        6,  3, 12, 21, 20,  0, 28,  3, 10,  6,  0,  0, 15, 28,  3,  6,  3,
        9, 26, 25, 11,  1, 27,  5,  7, 25, 24, 17, 15, 26, 17, 19, 24,  3,
        7, 14, 14,  4, 25, 11,  0, 13,  9, 16,  3, 29, 27, 23,  2,  0,  2,
       14,  7, 15, 22,  5, 12, 21, 20, 21,  2, 11, 26,  1,  7, 21]

tlength = 9999
t_length1=999
t_length2=199

it=np.arange(0,tlength+1,1)
it1=np.arange(0,t_length1+1,0.1)  # for ss=10
it2=np.arange(0,t_length2+1,0.02)  # for ss=50

pts_train=single_traj(4,-14,21,r,0.01,tlength) 
pts_test=single_traj(1,-1,2.05,r,0.01,tlength)

order=1
t_steps=20
t_lags=1
ncol=3

ss=40
lt_steps=[1]
filename1='rmse_d3.npy'
filename2=  'cd_d3.npy'
cd_=[]
rmse_=[]
Rmse_= np.empty((0, 5), float)  
CD= np.empty((0, 4), float)
MEAN_train= (np.mean(pts_train[:,0])+np.mean(pts_train[:,1])+np.mean(pts_train[:,2]))/3

for a in range(len(xx[:])):
    pts_test=single_traj(xx[a],yy[a],zz[a],r,0.01,tlength)
    for i,item  in enumerate(lt_steps) :
    
        t_steps=item
        # --------CREATING DATASETS------------------
        it=np.linspace(0,(tlength+1)*dt,tlength+1)
        it_d=np.linspace(0,(t_length1+1)*dt*ss,tlength+1)
        
        Xr_t1,Yr_t1 = Ideal_poly(pts_train[::ss],order,t_steps)
        Xr_train, Yr_train = Ideal_lags(Xr_t1,1,t_lags)
        

        Xnr_t1,Ynr_t1 = Ideal_poly(pts_train[::ss],order,t_steps)
        Xtrain, Ytrain = Ideal_lags(Xnr_t1,t_steps,t_lags)
        
        Xr_test,Yr_test=Ideal_poly(pts_test[::ss],order,t_steps)
        Xtest, Ytest = Ideal_lags(Xr_test,t_steps,t_lags)
        
        xxx=np.empty(len(Ytest[:,0]))
        xxx.fill(MEAN_train)
#        Xtest = Xtrain
#        Ytest = Ytrain
        
        cv = None
    
        alphas=np.arange(0.0000000001, 10, 10)
        #alphas=np.arange(0.000001, 10, 10)
        #alphas=np.arange(0, 10, 10)
        alpha=1
        
        #-----UNIVARIATE-----------------------
        
        cu = grouped_col_uni2(3,t_lags,order)
        #---------------Non-Recursive-----------------
        
        regr_x=LassoCV(alphas, cv=10, random_state=0)
        regr_x.fit(Xtrain[:,cu[0]], Ytrain[:,0])
        #print(cross_val_score(regr_x, Xtrain[:,cu[0]], Ytrain[:,0],scoring='neg_mean_squared_error', cv=10))
        
        regr_y=LassoCV(alphas, cv=10, random_state=0)
        regr_y.fit(Xtrain[:,cu[1]], Ytrain[:,1])
        #print(cross_val_score(regr_y, Xtrain[:,cu[1]], Ytrain[:,1],scoring='neg_mean_squared_error', cv=10))
        
        regr_z=LassoCV(alphas, cv=10, random_state=0)
        regr_z.fit(Xtrain[:,cu[2]], Ytrain[:,2])
        #print(cross_val_score(regr_x, Xtrain[:,cu[0]], Ytrain[:,2],scoring='neg_mean_squared_error', cv=10))
        #====predict==================================
        
        Ynrx=regr_x.predict(Xtest[:,cu[0]])
        Ynry=regr_y.predict(Xtest[:,cu[1]])
        Ynrz=regr_z.predict(Xtest[:,cu[2]])
        
        #----RECURSIVE-------------------------------------
        
        rx=LassoCV(alphas, cv=10, random_state=0)
        rx.fit(Xr_train[:,cu[0]], Yr_train[:,0,])
        
        ry=LassoCV(alphas, cv=10, random_state=0)
        ry.fit(Xr_train[:,cu[1]], Yr_train[:,1])
        
        rz=LassoCV(alphas, cv=10, random_state=0)
        rz.fit(Xr_train[:,cu[2]], Yr_train[:,2])
        
        Yp=swap_uni(Xtest,Ytest,rx,ry,rz,t_steps,t_lags,ncol,order)
        
        #============MULTIVARIATE======================
        c = grouped_col_multi(3,t_lags,order)
        #---------------Non-Recursive-----------------
        
        model_x=LassoCV(alphas, cv=10, random_state=0)
        model_x.fit(Xtrain, Ytrain[:,0])
        
        model_y=LassoCV(alphas, cv=10, random_state=0)
        model_y.fit(Xtrain, Ytrain[:,1])
        
        model_z=LassoCV(alphas, cv=10, random_state=0)
        model_z.fit(Xtrain, Ytrain[:,2])
        
        #====predict==================================
        
        YNRx=model_x.predict(Xtest)
        YNRy=model_y.predict(Xtest)
        YNRz=model_z.predict(Xtest)
        
        #----RECURSIVE-------------------------------------
        
        Rx=LassoCV(alphas, cv=10, random_state=0)
        Rx.fit(Xr_train, Yr_train[:,0])
        
        Ry=LassoCV(alphas, cv=10, random_state=0)
        Ry.fit(Xr_train, Yr_train[:,1])
        
        Rz=LassoCV(alphas, cv=10, random_state=0)
        Rz.fit(Xr_train, Yr_train[:,2])
        
        YP=swap_multi(Xtest,Ytest,Rx,Ry,Rz,t_steps,t_lags,ncol,order)
 
        #-----------------------------------------------------------
        print("x > ",regr_x.alpha_,model_x.alpha_,rx.alpha_,Rx.alpha_)
        print("y > ",regr_y.alpha_,model_y.alpha_,ry.alpha_,Ry.alpha_)
        print("z > ",regr_z.alpha_,model_z.alpha_,rz.alpha_,Rz.alpha_)
        cd_=coef_det_avg2(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP)
        #CD=np.append(CD,np.array([cd_]),axis=0)
        
        rmse_=RMSE_avg(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP,xxx)
        #Rmse_=np.append(Rmse_, np.array([rmse_]), axis=0)
        rmseee= RMSE(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP)
        Rmse_=np.append(Rmse_, np.array([rmse_]), axis=0)
        cddd=coef_det(Xtest,Ytest,regr_x,regr_y,regr_z,rx,ry,rz,model_x,model_y,model_z,Rx,Ry,Rz,cu)
        cd2=coef_det2(Ytest,Ynrx,Ynry,Ynrz,Yp,YNRx,YNRy,YNRz,YP)
        CD=np.append(CD,np.array([cd_]),axis=0)
        
        #plt.boxplot(Rmse_,positions=[a])
        #plt.boxplot(CD,positions=[a])
        
#

fig, axs = plt.subplots(2, sharex=False, sharey=False, figsize=(15, 15))
fig.suptitle(' Predited values v/s lead_time; dt=0.01')
    
axs[0].boxplot(Rmse_[:,1:])
axs[0].set_title("RMSE boxplots: lead time=0.3; t_lags=1; ss=5")
plt.xticks([1,2,3,4],['NR1c21','NR2c21','R1c21','R2c21'])
axs[0].set_xlabel("Models")
axs[0].set_ylabel("RMSE")
xxxx=np.empty(len(Rmse_[0]))
xxxx.fill(np.mean(Rmse_[:,0]))
axs[0].plot(xxxx,'g')

axs[1].boxplot(CD)
axs[1].set_title("R^2 boxplots: lead time=0.3; t_lags=1; ss=5")
plt.xticks([1,2,3,4],['NR1c21','NR2c21','R1c21','R2c21'])
axs[1].set_xlabel("Models")
axs[1].set_ylabel("Coef_of _deter")
xxxxx=np.zeros(len(CD[0])+1)
axs[1].plot(xxxxx,'g')


#plt.boxplot(CD)
#plt.title("R^2 boxplots: lead time=0.8; t_lags=15; ss=40")
#plt.xticks([1,2,3,4],['NR1c2','NR2c2','R1c2','R2c2'])
#plt.xlabel("Models")
#plt.ylabel("Coef_of _deter")
#xxxxx=np.zeros(len(CD[0])+1)
#plt.plot(xxxxx,'g')

#from numpy import asarray
#from numpy import save
## define data
#rmse_a = asarray(Rmse_)
#cd_a = asarray(CD)
## save to npy file
#save('rmse_e4.npy', rmse_a,'a')
#save('cd_e4.npy',cd_a,'a')

#from numpy import asarray
#from numpy import save
# define data
#rmse_a = asarray(Rmse_)
#cd_a = asarray(CD)
## save to npy file
#with open('rmse_a1.npy', 'wb') as f:
#    np.save(f, rmse_a)
#    
#with open('rmse_a1.npy', 'wb') as g:
#    np.save(g, rmse_a)
   
from npy_append_array import NpyAppendArray
import numpy as np

arr1 = np.array([[1,2],[3,4]])
arr2 = np.array([[1,2],[3,4],[5,6]])



# Appending to an array created by np.save is possible, but can fail in certain
# corner cases: e.g. a record as dtype, dim surpassing a critical threshold.
# Initialize the array with npaa.append directly (see below) so the
# header will be created with 64 byte of spare header space for growth.

# np.save(filename, arr1)
"""
npaa = NpyAppendArray(filename1)
npaa.append(Rmse_)
#npaa.append(arr2)
#npaa.append(arr2)

data = np.load(filename1, mmap_mode="r")


npaa2 = NpyAppendArray(filename2)
npaa2.append(CD)
#npaa.append(arr2)
#npaa.append(arr2)

data2 = np.load(filename2, mmap_mode="r")
"""



