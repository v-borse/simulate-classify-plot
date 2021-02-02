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

#-----------------------------------------------------------------------------------------------------------------


pts=np.load('pts.npy')
q=np.array([15.0,13.0,37.0])

with open('q.npy', 'wb') as f:
        np.save(f, q)   


# classifying regions
ts=classify2(pts,q)


with open('ts.npy', 'wb') as g:
    np.save(g, ts)
