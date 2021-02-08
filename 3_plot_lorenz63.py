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

def fixed_points(r,b):
    qx=math.sqrt(b*(r-1))
    qy=math.sqrt(b*(b-1))
    qz=r-1
    q=np.array([qx,qy,qz])
    
    return q



pts=(np.load('pts.npy')).T


#q=np.load('q.npy')
q=fixed_points(r=28, b=8/3)
ts=np.load('ts.npy')

R2=np.where(np.isin(ts,2))
R0=np.where(np.isin(ts,0))
R1=np.where(np.isin(ts,1))


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


ax.scatter(pts[0][R2],pts[1][R2],pts[2][R2],alpha=0.1,marker='.',linewidths=0.01)
ax.scatter(pts[0][R0],pts[1][R0],pts[2][R0],alpha=0.1,marker='.',linewidths=0.01) 
ax.scatter(pts[0][R1],pts[1][R1],pts[2][R1],alpha=1,linewidths=3,c='r')

ax.scatter(q[0],q[1],q[2],c='black',s=1,alpha=1,linewidths=3,marker='o')
ax.set_title('rho=28; dt=0.01; radius=1 unit; 3D plot') 
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
fig.show()
#ax.legend(region,loc="upper right",title="Regions", bbox_to_anchor=(1.2, 0.5))

