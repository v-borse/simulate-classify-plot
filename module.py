#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:33:30 2021

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
import matplotlib.pyplot as plt

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

def single_traj(x0,y0,z0,r,dt,num_steps):
    
    pt=[]
    #for single trajectory
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)
        
    # Set initial values
    #xs[0], ys[0], zs[0]= (4, -14, 21)
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
    
        
    
    pt=np.transpose(np.array([xs,ys,zs]))
    
    return pt

def trajectory(dt,num_steps,ntraj):

    x=np.array([])
    y=np.array([])
    z=np.array([])
    pt=[]
    
    #for n trajectories
    for j in range(ntraj):

        xs = np.empty(num_steps + 1)
        ys = np.empty(num_steps + 1)
        zs = np.empty(num_steps + 1)
        
        # Set initial values
        xs[0], ys[0], zs[0]= (xx[j], yy[j], zz[j])
        #xs[0], ys[0], zs[0] = (1., -1., 2.05)
        PT=single_traj(xs[0],ys[0],zs[0],r,dt,num_steps)        
        
    
        # Saving values for each trajectory        
        x=np.append(PT[0][:],xs,axis=0)
        y=np.append(PT[1][:],ys,axis=0)
        z=np.append(PT[2][:],zs,axis=0)
    
        pt=np.transpose(np.array([x,y,z]))
    
    return pt

def classify2(_pt,q):
    """
    Given:
    _pt: takes vector (set of coordinates lying on trajectories) 
    q: the center of spherical region
    Returns:
    _ts: time series
    
    
    dst: distance between two points
    R: radius of the sphere
    """
    _c1=0
    _c2=0
    _c0=0
    _r0=[]
    _r1=[]
    _r2=[]
   
    _ts=[]
    R=1
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
    
    return _ts

def fixed_points(r,b):
    qx=math.sqrt(b*(r-1))
    qy=math.sqrt(b*(b-1))
    qz=r-1
    q=np.array([qx,qy,qz])
    
    return q 

def WireframeSphere(centre, radius,
                    n_meridians=20, n_circles_latitude=None):
    """
    
    Create the arrays of values to plot the wireframe of a sphere.
    Parameters
    ----------
    centre: array like
        A point, defined as an iterable of three numerical values.
    radius: number
        The radius of the sphere.
    n_meridians: int
        The number of meridians to display (circles that pass on both poles).
    n_circles_latitude: int
        The number of horizontal circles (akin to the Equator) to display.
        Notice this includes one for each pole, and defaults to 4 or half
        of the *n_meridians* if the latter is larger.
    Returns
    -------
    sphere_x, sphere_y, sphere_z: arrays
        The arrays with the coordinates of the points to make the wireframe.
        Their shape is (n_meridians, n_circles_latitude).
    Examples
    --------
    >>> fig = plt.figure()
    >>> ax = fig.gca(projection='3d')
    >>> ax.set_aspect("equal")
    >>> sphere = ax.plot_wireframe(*WireframeSphere(), color="r", alpha=0.5)
    >>> fig.show()
    >>> fig = plt.figure()
    >>> ax = fig.gca(projection='3d')
    >>> ax.set_aspect("equal")
    >>> frame_xs, frame_ys, frame_zs = WireframeSphere()
    >>> sphere = ax.plot_wireframe(frame_xs, frame_ys, frame_zs, color="r", alpha=0.5)
    >>> fig.show()
    """
    if n_circles_latitude is None:
        n_circles_latitude = max(n_meridians/2, 4)
    u, v = np.mgrid[0:2*np.pi:n_meridians*1j, 0:np.pi:n_circles_latitude*1j]
    sphere_x = centre[0] + radius * np.cos(u) * np.sin(v)
    sphere_y = centre[1] + radius * np.sin(u) * np.sin(v)
    sphere_z = centre[2] + radius * np.cos(v)
    return sphere_x, sphere_y, sphere_z

def committor2(delta, ts, st1, st2):
    """
    Given:
        delta = # of time steps within which there is a switching
        ts = time series
        Example: if we have to switch from 0 to 1 then
        st1=0
        st2=1
    Returns:
        
        indirect = counts for indirect switching from st1 to st2
    """
        
    indirect = 0
    
    II = np.where(np.isin(ts,st1))[0]
    lenII = len(II)  
    for i,item in enumerate(II):
        if(item < (len(ts) - 1)):
            iend = min(item+delta+1, len(ts))
          
            trial=ts[item+1:iend]
            #print(trial)
         
            indirect +=  np.any(np.isin(trial,st2))
    p=indirect/len(II)
    #print(p)   
    return(indirect,lenII,p)


def ideal(pt, t_steps):
    
    print(pt)
    X = pt[:-t_steps]
    Y = pt[t_steps:]
    
    return X, Y

def non_recursive_LN(Xt,t_steps,ind,pt):
    
    X, Y = ideal(pt, t_steps)
    
    model = LinearRegression().fit(X[:, ind, None], Y[:, ind])
    intercept, coefficients = model.intercept_, model.coef_
    
    Ynew = model.predict(Xt)
    
    return intercept, coefficients, Ynew

def recursive_LN(Xt, t_steps, ind, pt):
    
    X, Y = ideal(pt, 1)
    
    model = LinearRegression().fit(X[:, ind, None], Y[:, ind])
    intercept, coefficients = model.intercept_, model.coef_
    
    Xnew = Xt

    for i in range(t_steps):
        if i == 0:
            Xnew = Xt
        else:
            Ynew = model.predict(Xnew.reshape(-1, 1))
            
    return intercept, coefficients, Ynew

def m_non_recursive_LN(Xt, t_steps, ind, pt):
    
    X, Y = ideal(pt, t_steps)
    
    model1 = LinearRegression().fit(X_tr[:, :],Y_tr[:, ind])
    r_sq = model1.score(X_tr[:, :], Y_tr[:, ind])
    intercept, coefficients = model.intercept_, model.coef_
    
    Ynew = model.predict(Xt)

    return intercept, coefficients, Ynew

def m_recursive_LN(Xt, t_steps, ind, pt):
    
    X, Y = ideal(pt, 1)
    
    model2 = LinearRegression().fit(X_tr[:, :], Y_tr[:, ind, None])
    r_sq = model2.score(X_tr[:, :], Y_tr[:, ind, None])
    intercept, coefficients = model.intercept_, model.coef_
    
    Xnew = Xt

    for i in range(t_steps):
        if i == 0:
            Xnew = Xt
        else:
            Ynew = model.predict(Xnew)
            
    return intercept, coefficients, Ynew
