#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 12:48:06 2021

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



def ideal(pt, t_steps):
    # for polynomial covariates; it replaces ideal()
    print(pt)
    X = pt[:-t_steps-2]
    Y = pt[t_steps+2:]
    
    return X, Y

def ideal5 (pt,t_steps):
    #for polynomial covariates; it replaces ideal3()
    XT = pt[:-t_steps]
    XT1= pt[:-t_steps]
    XT2= np.multiply(pt[:-t_steps],pt[:-t_steps])
    Y  = pt[t_steps:]
    
    return XT,XT1,XT2,Y

def ideal3(pt,t_steps):
    XT2 = pt[:-t_steps-2]
    XT1= pt[1:-t_steps-1]
    XT= pt[2:-t_steps]
    Y  = pt[t_steps+2:]
    
    return XT,XT1,XT2,Y

def gen_ideal(pt,t_steps,n):
    
    col=[str(j).zfill(2) for j in range(1,n+1)]
    print(col)
    dfg=pd.DataFrame()
    #dfg = pd.DataFrame(columns=[col])
    #print(dfg)
    X= pt[:-t_steps]
    Y = pt[t_steps:]
    for i in range(n+1):
        if(i+1<=n):
#            this_column = dfg.columns[i]
#            print(this_column)
#            dfg[this_column] = [i, i+1]
            XTi= np.power(X[:,0],i+1)
            print(XTi)
            #dfg[this_column]=pd.Series(XTi)
            #print(dfg)
            dfg = pd.concat([dfg, pd.Series(XTi)], axis=1)
            
            print(dfg)
            
    return X,Y,dfg
    
    

def plot_predicted_ts1(X2,Y2, Yp, index):
    
    fig, axs = plt.subplots(4, 2, sharex=False, sharey=False, figsize=(15, 5))
    fig.suptitle(' Linear Regression; dt=0.01')
    
    axs[0,0].scatter(Yp, Y2[:, index],s=2,c='k')
    axs[0,0].plot(Y2[:, index],Y2[:, index],'g')
    #axs[0,0].set_xlim([-50,50])
    #axs[0,0].set_ylim([-50,50])
    axs[0,0].set_xlabel("y_pred")
    axs[0,0].set_ylabel("y_ideal")
    axs[0,0].set_title("Non-recursive LN")
    
           
    axs[1,0].scatter(X2[:, index], Y2[:, index],c='k', s=2)
    axs[1,0].scatter(X2[:, index], Yp,c='b', s=2)
    
    #axs[1,0].set_xlim([-50,50])
    #axs[1,0].set_ylim([-50,50])
    axs[1,0].set_xlabel("X(x,y,z)")
    axs[1,0].set_ylabel("Y")
    axs[1,0].set_title("X and Y")
    
    axs[1,1].plot((Yp-Y2[:, index]), 'b')
    #axs[2,0].set_ylim([-50,50])
    axs[1,1].set_ylabel("error")
    axs[1,1].set_xlabel("time_steps")
    axs[1,1].set_title("Errors for LN")
    
    
    
    axs[2,0].plot(X2[:5000, index],'g^',Y2[:5000, index],'y.')
    axs[2,0].set_xlabel("X and Y_ideal Time series")
    
    axs[2,1].plot(X2[5000:, index],'g^',Y2[5000:, index],'y.')
    axs[2,1].set_xlabel("X and Y_ideal Time series")
    
    axs[3,0].plot(Yp[:5000],'b^')
    axs[3,0].set_xlabel("predicted Time series")
    
    axs[3,1].plot(Yp[5000:],'b^')
    axs[3,1].set_xlabel("predicted Time series")
    
    
    
    plt.show()

def plot_predicted_ts(X2,Y2, Yp, Yrp,index):
    fig, axs = plt.subplots(4, 2, sharex=False, sharey=False, figsize=(15, 15))
    fig.suptitle(' Linear Regression; dt=0.01')
    
    axs[0,0].scatter(Yp, Y2[:, index],c='k',s=2)
    axs[0,0].plot(Y2[:, index],Y2[:, index],'g')
    #axs[0,0].set_xlim([-50,50])
    #axs[0,0].set_ylim([-50,50])
    axs[0,0].set_xlabel("y_pred")
    axs[0,0].set_ylabel("y_ideal")
    axs[0,0].set_title("Non-recursive LN")
    
    axs[0,1].scatter(Yrp[:, index], Y2[:, index],c='k', s=2)
    axs[0,1].plot(Y2[:, index], Y2[:, index], 'g')
    #axs[0,1].set_xlim([-50,50])
    #axs[0,1].set_ylim([-50,50])
    axs[0,1].set_xlabel("y_pred")
    axs[0,1].set_ylabel("y_ideal")
    axs[0,1].set_title("Recursive LN")
    
       
    axs[1,0].scatter(X2[:, index], Y2[:, index],c='k', s=2)
    axs[1,0].scatter(X2[:, index], Yp,c='b', s=2)
    axs[1,0].scatter(X2[:, index], Yrp[:, index],c='r', s=2) 
    #axs[1,0].set_xlim([-50,50])
    #axs[1,0].set_ylim([-50,50])
    axs[1,0].set_xlabel("X(x,y,z)")
    axs[1,0].set_ylabel("Y")
    axs[1,0].set_title("X and Y")
    
    axs[1,1].plot((Yp-Y2[:, index]), 'b')
    axs[1,1].plot(Yrp[:, index]-Y2[:, index], 'r')
    #axs[2,0].set_ylim([-50,50])
    axs[1,1].set_ylabel("error")
    axs[1,1].set_xlabel("time_steps")
    axs[1,1].set_title("Errors for LN")
    
    
    axs[2,0].plot(X2[:5000, index],'g^',Y2[:5000, index],'y.')
    #axs[1,1].plot(Y2[:, index])
    #axs[1,1].plot(Yp)
    #axs[1,1].plot(Yrp[:, index])
    #axs[1,1].set_ylim([-60,60])
    axs[2,0].set_xlabel("X and Y_ideal Time series")
    
    axs[2,1].plot(X2[5000:, index],'g^',Y2[5000:, index],'y.')
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

def dataframe(XT,XT1,XT2, Y):
    
    Data = {'XTx':XT[:,0], 
        'XT1x':XT1[:,0],
        'XT2x':XT2[:,0],
        'YTx':Y[:,0],
        'XTy':XT[:,1], 
        'XT1y':XT1[:,1],
        'XT2y':XT2[:,1],
        'YTy':Y[:,1],
        'XTz':XT[:,2], 
        'XT1z':XT1[:,2],
        'XT2z':XT2[:,2],
        'YTz':Y[:,2]}
    return Data

def t_dataframe(tXT,tXT1,tXT2, tY):
    
    Data = {'tXTx':tXT[:,0], 
        'tXT1x':tXT1[:,0],
        'tXT2x':tXT2[:,0],
        'tYTx':tY[:,0],
        'tXTy':tXT[:,1], 
        'tXT1y':tXT1[:,1],
        'tXT2y':tXT2[:,1],
        'tYTy':tY[:,1],
        'tXTz':tXT[:,2], 
        'tXT1z':tXT1[:,2],
        'tXT2z':tXT2[:,2],
        'tYTz':tY[:,2]}
    return Data
    
def r_predict_uni(Xt,modelx,modely,modelz,t_steps):

    for i in range(t_steps):
        if i == 0:
            Xnew = np.copy(Xt)
        Xnew[:, 0] = modelx.predict(Xnew[:, 0, None])
        Xnew[:, 1] = modely.predict(Xnew[:, 1, None])
        Xnew[:, 2] = modelz.predict(Xnew[:, 2, None])
            
    return Xnew

def r_predict_uni2(pts2,modelx,modely,modelz,t_steps):
    
    tXT,tXT1,tXT2, tY= ideal3(pts2,1)
    data4= t_dataframe(tXT,tXT1,tXT2, tY)
    # Create DataFrame 
    df4 = pd.DataFrame(data4) 
    
    NUM_COLORS = 50
    #LINE_STYLES = ['solid','dotted']
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)

    sns.reset_orig()  # get default matplotlib styles back
    clrs = sns.color_palette('husl', n_colors=NUM_COLORS)  # a list of RGB tuples
    fig, ax = plt.subplots(1)
    for i in range(t_steps):
        tx=df4[['tXT1x','tXT2x']]
        ty=df4[['tXT1y','tXT2y']]
        tz=df4[['tXT1z','tXT2z']]
         
        Yrx=modelx.predict(tx)
        Yry=modely.predict(ty)
        Yrz=modelz.predict(tz)
        
        lines = ax.plot(Yrx[:5000],label=i)
        lines[0].set_color(clrs[i])
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        ax.legend(title='t_steps',bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlabel('timesteps')
        ax.set_ylabel('Y')
        ax.set_title("Recursive/Univariate LN")
        #ax.legend(lines[:i],[i],bbox_to_anchor=(1.05, 1), loc='upper left',frameon='False')
#        plt.plot(Yrx,label=i,colormap=cm.cubelix);
#        plt.legend(frameon='False')
#        lines=plt.plot(Yrx,label=i)
#        plt.legend(lines[:t_steps],['0','1','2','3','4']);
        
        
        

        
        df4['tXT2x'] = df4['tXT1x']
        df4['tXT2y'] = df4['tXT1y']
        df4['tXT2z'] = df4['tXT1z']
        df4.tXT1x = Yrx
        df4.tXT1y = Yry
        df4.tXT1z = Yrz
        #print(df4)
    #ax.legend(lines[:],[i],bbox_to_anchor=(1.05, 1), loc='upper left',frameon='False')
    Xnew = np.array([Yrx,Yry,Yrz])
    return Xnew.T
        
            
def R_predict_multi2(pts2,modelx,modely,modelz,t_steps):
    
    tXT,tXT1,tXT2, tY= ideal3(pts2,1)
    data5= t_dataframe(tXT,tXT1,tXT2, tY)
    # Create DataFrame 
    df5 = pd.DataFrame(data5) 
    NUM_COLORS = 50
    #LINE_STYLES = ['solid','dotted']
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)

    sns.reset_orig()  # get default matplotlib styles back
    clrs = sns.color_palette('husl', n_colors=NUM_COLORS)  # a list of RGB tuples
    fig, ax = plt.subplots(1)
    
    for i in range(t_steps):
        tx=df5[['tXT1x','tXT2x','tXT1y','tXT2y','tXT1z','tXT2z']] # Multivariate

        Yrx=modelx.predict(tx)
        Yry=modely.predict(tx)
        Yrz=modelz.predict(tx)
        
        lines = ax.plot(Yrx[:5000],label=i)
        lines[0].set_color(clrs[i])
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        ax.legend(title='t_steps',bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlabel('timesteps')
        ax.set_ylabel('Y')
        ax.set_title("Recursive/Multivariate LN")
        
        df5['tXT2x'] = df5['tXT1x']
        df5['tXT2y'] = df5['tXT1y']
        df5['tXT2z'] = df5['tXT1z']
        df5.tXT1x = Yrx
        df5.tXT1y = Yry
        df5.tXT1z = Yrz
    Xnew = np.array([Yrx,Yry,Yrz])
    return Xnew.T


r=28
R=1
tlength = 10000
t_steps = 5
pts=single_traj(4,-14,21,r,0.01,tlength) 
pts2=single_traj(1,-1,2.05,r,0.01,tlength)

X1, Y1 = ideal(pts,1)
X2, Y2 = ideal(pts2,t_steps)
X3, Y3 = ideal(pts2,1)
X4, Y4, dfgg = gen_ideal(pts,t_steps,t_steps)

"""
#----------UNIVARIATE-------------------------------------

#---------Non- Recursive---------------------------------------

XT,XT1,XT2, Y = ideal3(pts,t_steps)
data= dataframe(XT,XT1,XT2, Y)
# Create DataFrame 
df = pd.DataFrame(data) 

# prediction with sklearn
xx=df[['XT1x','XT2x']] #Univariate
xy=df[['XT1y','XT2y']]
xz=df[['XT1z','XT2z']]
yx=df['YTx']
yy=df['YTy']
yz=df['YTz']

regr_x = linear_model.LinearRegression()
regr_x.fit(xx, yx)

print('Intercept: \n', regr_x.intercept_)
print('Coefficients: \n', regr_x.coef_)

regr_y = linear_model.LinearRegression()
regr_y.fit(xy, yy)


regr_z = linear_model.LinearRegression()
regr_z.fit(xz, yz)

tXT,tXT1,tXT2, tY= ideal3(pts2,t_steps)
data2= t_dataframe(tXT,tXT1,tXT2, tY)

  
# Create DataFrame 
df2 = pd.DataFrame(data2) 

txx=df2[['tXT1x','tXT2x']] #Univariate
txy=df2[['tXT1y','tXT2y']]
txz=df2[['tXT1z','tXT2z']]
tyx=df2['tYTx']
tyy=df2['tYTy']
tyz=df2['tYTz']

Ynrx=regr_x.predict(txx)
Ynry=regr_y.predict(txy)
Ynrz=regr_z.predict(txz)

#plot_predicted_ts(X2,Y2,predictions_x,0)
#plot_predicted_ts(X2,Y2,predictions_y,1)
#plot_predicted_ts(X2,Y2,predictions_z,2)


#==========Recursive====================================

XT,XT1,XT2, Y = ideal3(pts,1)
data3= dataframe(XT,XT1,XT2, Y)
# Create DataFrame 
df3 = pd.DataFrame(data3) 

# prediction with sklearn
xx=df3[['XT1x','XT2x']] #Univariate
xy=df3[['XT1y','XT2y']]
xz=df3[['XT1z','XT2z']]
yx=df3['YTx']
yy=df3['YTy']
yz=df3['YTz']

rx = linear_model.LinearRegression()
rx.fit(xx, yx)

ry = linear_model.LinearRegression()
ry.fit(xy, yy)

rz = linear_model.LinearRegression()
rz.fit(xz, yz)

#-------preddiction-------------------

Yr = r_predict_uni2(pts2,rx,ry,rz,t_steps)
plot_predicted_ts(X2,Y2,Ynrx,Yr[:len(Ynrx[:,None])],0)
plot_predicted_ts(X2,Y2,Ynry,Yr[:len(Ynry[:,None])],1)
plot_predicted_ts(X2,Y2,Ynrz,Yr[:len(Ynrz[:,None])],2)




#----------MULTIVARIATE-------------------------------------

#---------Non- Recursive---------------------------------------

XT,XT1,XT2, Y = ideal3(pts,t_steps)
data1= dataframe(XT,XT1,XT2, Y)

# Create DataFrame 
df1 = pd.DataFrame(data1) 
x=df1[['XT1x','XT2x','XT1y','XT2y','XT1z','XT2z']] # Multivariate

yx=df1['YTx']
yy=df1['YTy']
yz=df1['YTz']

modelx=linear_model.LinearRegression()
modelx.fit(x,yx)

modely=linear_model.LinearRegression()
modely.fit(x,yy)

modelz=linear_model.LinearRegression()
modelz.fit(x,yz)

#====predict==================================

tXT,tXT1,tXT2, tY= ideal3(pts2,t_steps)
data2= t_dataframe(tXT,tXT1,tXT2, tY)

  
# Create DataFrame 
df2 = pd.DataFrame(data2) 

Tx=df2[['tXT1x','tXT2x','tXT1y','tXT2y','tXT1z','tXT2z']] # Multivariate


Tyx=df2['tYTx']
Tyy=df2['tYTy']
Tyz=df2['tYTz']

YNRx=modelx.predict(Tx)
YNRy=modely.predict(Tx)
YNRz=modelz.predict(Tx)

#plot_predicted_ts(X2,Y2,Ynrx,0)
#plot_predicted_ts(X2,Y2,Ynry,1)
#plot_predicted_ts(X2,Y2,Ynrz,2)

#-------------------------Recursive-----------------------------
XT,XT1,XT2, Y = ideal3(pts,1)
data3= dataframe(XT,XT1,XT2, Y)
# Create DataFrame 
df3 = pd.DataFrame(data3) 

x=df3[['XT1x','XT2x','XT1y','XT2y','XT1z','XT2z']] # Multivariate

yx=df3['YTx']
yy=df3['YTy']
yz=df3['YTz']

Rx=linear_model.LinearRegression()
Rx.fit(x,yx)

Ry=linear_model.LinearRegression()
Ry.fit(x,yy)

Rz=linear_model.LinearRegression()
Rz.fit(x,yz)

YR = R_predict_multi2(pts2,Rx,Ry,Rz,t_steps)
plot_predicted_ts(X2,Y2,YNRx,YR[:len(Ynrx[:,None])],0)
plot_predicted_ts(X2,Y2,YNRy,YR[:len(Ynry[:,None])],1)
plot_predicted_ts(X2,Y2,YNRz,YR[:len(Ynrz[:,None])],2)
"""