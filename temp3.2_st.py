#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 04:29:17 2021

@author: vborse
"""

from IPython.display import clear_output, display, HTML

import numpy as np
from scipy import integrate

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
import random





def classify2(_pt,q):
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
    
    return _ts,_c0,_c1,_c2,_r0,_r1,_r2

#-----------------------------------------------------------------------------------------------------------------

#arr_x=np.load('x.npy')
#arr_y=np.load('y.npy')
#arr_z=np.load('z.npy')
#
## creating a vector
#pts=np.transpose(np.array([arr_x,arr_y,arr_z]))
pts=np.load('pts.npy')
q=np.array([15.0,13.0,37.0])
# Soulivanh: q was not saved
with open('q.npy', 'wb') as f:
        np.save(f, q)   


# classifying regions
ts,c0,c1,c2,R0,R1,R2=classify2(pts,q)

# Soulivanh: It is simpler to save ts inshea as it is easy to deduce them from ts with the command np.where
with open('ts.npy', 'wb') as g:
    np.save(g, ts)
#with open('R2.npy', 'wb') as h:
#        np.save(h, R2)   
#with open('R0.npy', 'wb') as m:
#        np.save(m, R0)    
#with open('R1.npy', 'wb') as n:
#        np.save(n, R1) 
