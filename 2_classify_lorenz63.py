#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 04:29:17 2021

@author: vborse
"""
import numpy as np
from module import classify2
from module import fixed_points




#-----------------------------------------------------------------------------------------------------------------


pts=np.load('pts.npy')
#q=np.array([15.0,13.0,37.0])
q=fixed_points(r=28,b=8/3)

with open('q.npy', 'wb') as f:
        np.save(f, q)   


# classifying regions
ts=classify2(pts,q)


with open('ts.npy', 'wb') as g:
    np.save(g, ts)
