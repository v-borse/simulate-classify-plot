#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 04:35:45 2021

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

#arr_x=np.load('x.npy')
#arr_y=np.load('y.npy')
#arr_z=np.load('z.npy')

pts=(np.load('pts.npy')).T

# Soulivanh: q was not loaded
q=np.load('q.npy')
ts=np.load('ts.npy')
#ix = np.isin(x, goodvalues)
#r2=np.isin(ts,2)
#R2=np.where(r2)
#r0=np.isin(ts,0)
#R0=np.where(r0)
#r1=np.isin(ts,1)
#R1=np.where(r1)
R2=np.where(np.isin(ts,2))
R0=np.where(np.isin(ts,0))
R1=np.where(np.isin(ts,1))
#R0=np.where(ts[:] == 0, ts)
#R1=np.where(ts[:] == 1, ts)
#R2=np.load('R2.npy')
#R0=np.load('R0.npy')
#R1=np.load('R1.npy')

# Soulivanh: is region used somewhere ?

region=[R2,R0,R1,q]    
fig3=plt.figure()


plt.scatter(pts[0][R2],pts[2][R2],alpha=0.1,marker='.',linewidths=0.01)
plt.scatter(pts[0][R0],pts[2][R0],alpha=0.1,marker='.',linewidths=0.01)
plt.scatter(pts[0][R1],pts[2][R1],alpha=1,linewidths=3,c='r')
plt.scatter(q[0],q[2],c='black',s=1,alpha=1,linewidths=3,marker='o')
plt.title('rho=28; dt=0.01; radius=1 unit; 2D plot')
plt.xlabel("X Axis")
plt.ylabel("Z Axis")
fig3.show()
#ax.plt.legend((line1, line2, line3,line4), region,loc="upper right",title="Regions", bbox_to_anchor=(1.2, 0.5))

fig=plt.figure()
#ax = fig.gca(projection='3d')
#ax = Axes3D(fig)
ax = fig.add_subplot(1, 2, 1, projection='3d')

# Soulivanh
ax.scatter(pts[0][R2],pts[1][R2],pts[2][R2],alpha=0.1,marker='.',linewidths=0.01)
ax.scatter(pts[0][R0],pts[1][R0],pts[2][R0],alpha=0.1,marker='.',linewidths=0.01) # Soulivanh: arr_z[R2] changed to arr_z[R0]
ax.scatter(pts[0][R1],pts[1][R1],pts[2][R1],alpha=1,linewidths=3,c='r') # Soulivanh: arr_z[R2] changed to arr_z[R1]

ax.scatter(q[0],q[1],q[2],c='black',s=1,alpha=1,linewidths=3,marker='o')
ax.set_title('rho=28; dt=0.01; radius=1 unit; 3D plot') # Soulivanh: ax.title changed to ax.set_title
ax.set_xlabel("X Axis")# Soulivanh: ax.xlabel changed to ax.set_xlabel
ax.set_ylabel("Y Axis")# Soulivanh: ax.ylabel changed to ax.set_ylabel
ax.set_zlabel("Z Axis")# Soulivanh: ax.zlabel changed to ax.set_zlabel
fig.show()
#ax.legend(region,loc="upper right",title="Regions", bbox_to_anchor=(1.2, 0.5))

"""
#ValueError: shape mismatch: objects cannot be broadcast to a single shape   

#This can be resolved if 600 trajectories are taken because shape of R1,R0, R2 are different

#Soulivanh: no, you just made a mistake in the indices when doing copy-paste

"""
