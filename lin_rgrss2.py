#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 14:29:18 2021

@author: vborse
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 04:25:27 2021

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
import random

"""Generating set of random points as initial conditions within a defined range"""
#xx = np.random.choice(range(-30,30),100)
#yy = np.random.choice(range(-30,30),100)
#zz = np.random.choice(range(0,30),100)

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

def single_traj(r,dt,num_steps):
    # Soulivanh: if you know the final length of the vector, you should preallocate it. here:
#    x = np.empty((num_steps + 1) * ntraj)
#    y = np.empty((num_steps + 1) * ntraj)
#    z = np.empty((num_steps + 1) * ntraj)
    #x=np.array([])
    #y=np.array([])
    #z=np.array([])
    pt=[]
    #for single trajectory
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)
        
    # Set initial values
    xs[0], ys[0], zs[0]= (4, -14, 21)
    #xs[0], ys[0], zs[0] = (1., -1., 2.05)
        
        
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
    
        # Saving values for each trajectory        
        #x=np.append(x,xs,axis=0)
        #print(len(x))
        #y=np.append(y,ys,axis=0)
        #z=np.append(z,zs,axis=0)
    #x=np.append(x,xs,axis=0)
    #y=np.append(y,ys,axis=0)
    #z=np.append(z,zs,axis=0)
    
    pt=np.transpose(np.array([xs,ys,zs]))
    
    return pt


def classify2(R,_pt,q):
    """
    Given:
    _pt: takes vector (set of coordinates lying on trajectories) 
    q: the center of spherical region
    Returns:
    ts: time series
    # Soulivanh: It is probably not necessary to return the _c and _r variables as it is easy to deduce them from ts with the command np.where
    _c0,_c1,_c2: number of points in region 0,1,2
    _r0,_r1,_r2: Indices of the points in region 0,1,2
    
    dst: distance between two points
    R: radius of the sphere
    """
    _c1=0
    _c2=0
    _c0=0
    _r0=[]
    _r1=[]
    _r2=[]
    # Soulivanh: if you know the final length of the vector, you should preallocate it. Here:
    # _ts = np.zeros(_pn.shape[0])
    _ts=[]
    #R=10
    for each_ind, each_line in enumerate(_pt):
        #calculates Eucledian distance between two points
        dst=np.linalg.norm(each_line-q)
        
        if (dst<=R and each_line[0]>0):
            #Soulivanh: _ts[each_ind] = 1
            _ts.append(1)
            _r1.append(each_ind)
            _c1+=1
            # region 1: set of points within the spherical bowl of radius R centered around q
            # dst <=R and x>0
        elif (dst>R and each_line[0]>0):
            _ts.append(2)
            _r2.append(each_ind)
            _c2+=1
            # region 2: set of points outside the spherical bowl and x>0
            #  dst>R and x>0
        elif(each_line[0]<=0):
            _ts.append(0)
            _r0.append(each_ind)
            _c0+=1
            # region 0: dst>R and x<=0
    
    return _ts,_c0,_c1,_c2,_r0,_r1,_r2

def fixed_points(r,b):
    qx=math.sqrt(b*(r-1))
    qy=math.sqrt(b*(b-1))
    qz=r-1
    q=np.array([qx,qy,qz])
    
    return q  

# forming trajectories

r=28
R=1
pts=single_traj(r,0.01,10000)

with open('pts.npy', 'wb') as h:
    np.save(h, pts)
   
pts=np.load('pts.npy')
X=np.transpose(pts)
##q=np.array([15.0,13.0,37.0])
#q=fixed_points(r,8/3)
#with open('q.npy', 'wb') as f:
#        np.save(f, q)   
#
## classifying regions
#ts,c0,c1,c2,R0,R1,R2=classify2(R,pts,q)
#print("c0,c1,c2=",c0,c1,c2)
#
#
#with open('ts.npy', 'wb') as g:
#    np.save(g, ts)


#covMatrix = np.cov(X,bias=True)
covMatrix = np.corrcoef(X,bias=True)
y=np.dot(covMatrix,X)
Y=np.dot(covMatrix,X)
y_pred=np.dot(covMatrix,X)
# Step 1: Import packages
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import math 
# Step 2a: Provide data

for i,item in enumerate(X):
    for j,jitem in enumerate(X[i]):
        if (j!=(len(X[i])-1)):
            Y[i][j]=X[i][j+1]
        else:
            Y[i][-1]=0

#model = LinearRegression().fit(X.T, Y[0])
model = LinearRegression().fit(X[2].reshape(-1,1), Y[2])
    # Step 4: Get results
#r_sq = model.score(X.T, Y[0])
r_sq = model.score(X[2].reshape(-1,1), Y[2])
intercept, coefficients = model.intercept_, model.coef_
print(intercept,coefficients)
print(X[i])
print(Y[i])
print(y[i])

    # Step 5: Predict
    #y_pred = model.predict(np.array([22]).reshape(-1,1))
#y_pred= model.predict(X.T)
y_pred= model.predict(X[2].reshape(-1,1))

print(y_pred)
print('\n')
    #plt.scatter(X[i],y_pred)
    
    
print('x')
print('y')
print('y_pred=',y_pred)
print("intercept=",intercept)
print("coefficients=",coefficients)
#plt.scatter(x,y)
#plt.plot(arr_z,y_pred,'r')
#
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Linear regression')
#plt.scatter(X[2],Y[2])
#plt.scatter(X[2],y[2])
#plt.scatter(X[2],y_pred)

#y_pred= model.predict(X.reshape(-1,1))
#y_pred=model.predict()
#plt.scatter(X[2],Y[2])
#plt.scatter(X[2],y[2])

#plt.scatter(y_pred,Y[0])
plt.scatter(y_pred[:-2],Y[2][:-2],s=2)
plt.xlabel('y_pred')
plt.ylabel('Y_ideal')
plt.title('Linear regression')

#plt.plot(Y[0],Y[0],'r')
plt.plot(Y[2][:-2],Y[2][:-2],'r')
#plt.hold(False)
#plt.plot(y_pred[:-2]-Y[1][:-2],'k')
#plt.xlabel('time_steps')
#plt.ylabel('error')
#plt.title('Linear regression')
"""
#sns.heatmap(covMatrix, annot=True, fmt='g')
plt.show()
"""