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
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
sns.set() # Setting seaborn as default style even if use only matplotlib


def trajectory(r,dt,num_steps,ntraj):
   
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
        x=np.append(x,xs,axis=0)
        print(len(x))
        y=np.append(y,ys,axis=0)
        z=np.append(z,zs,axis=0)
    
    pt=np.transpose(np.array([x,y,z]))
    
    return pt
    

def committor3(delta,ts,st1,st2):
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
        
    indirect=0
    
    II=np.where(np.isin(ts,st1))[0]
    lenII=len(II)  
    
    for i,item in enumerate(II):
        if(item < (len(ts) - 1)):
            iend = min(item+delta+1, len(ts))
          
            trial=ts[item+1:iend]
            #print(trial)
         
            indirect +=  np.any(np.isin(trial,st2))
    #p=indirect/len(II)
    #print(p)   
    return(indirect,lenII)


#----MAIN BODY of the code --------------------------------------------------------------------------------------------------

# BOXPLOTS : Probabilities of transition within regions (for 100 trajectories)
r=64
R=1
pts=trajectory(r,0.01,10000,99)

with open('pts.npy', 'wb') as h:
    np.save(h, pts)
   
pts=np.load('pts.npy')

#q=np.array([15.0,13.0,37.0])
q=fixed_points(r,8/3)
with open('q.npy', 'wb') as f:
        np.save(f, q)   

# classifying regions
ts,c0,c1,c2,R0,R1,R2=classify2(R,pts,q)
#print("c0,c1,c2=",c0,c1,c2)


with open('ts.npy', 'wb') as g:
    np.save(g, ts)
    


TS=np.load('ts.npy') 
# time series for single trajectory can be directly loaded 
print(c0,c1,c2)

#st1=0
#st2=1
#print(c0,c1,c2)

delt=[10,50,100,200,500,1000]
x_labels=[0.1,0.5,1,2,5,10]
y_labels=[0,0.2,0.4,0.6,0.8,1.0]
#print("Switching from st1=",st1, "to st2=",st2)
P01=[]
P02=[]
P21=[]
Per_01=[]
INDIR_01=[]
string1="c0={0}; c1={1}; c2={2}; c01={3}; c02={4}; c21={5}; rho={6}; Rad={7}"
string2="c0={0}; c1={1}; c2={2}; rho={3}; Radius={4})"
string3="Nume={0}"
#print(string.format(c0,c1,c2,indir,lenII,r,R))

for k,kitem in enumerate(delt):
    deltak=kitem
    ii=1
    #P01=[]
    #P02=[]
    #P21=[]
    per_01=0
    Indir_01=0
    LenII_01=0
    for i in range(99):
      
      indir_01,lenII_01=committor3(deltak,TS[i*10001:ii*10001],0,1)
      Indir_01+=indir_01
      LenII_01+=lenII_01
      if (indir_01==0):
        per_01+=1
      p01=indir_01/lenII_01
      P01.append(p01)
      
      ii+=1
    Per_01.append(per_01)
    INDIR_01.append(Indir_01)
print(Per_01)


p=np.reshape(P01,(6,99))
print(len(p))
print (np.shape(p))

for k,kitem in enumerate(delt):
  plt.boxplot(p[k],positions=[k])
  plt.title(string2.format(c0,c1,c2,r,R))
  plt.suptitle(string3.format(INDIR_01))
  plt.xticks([0,1,2,3,4,5],x_labels)
  #plt.yticks([0,1,2,3,4,5],y_labels)
  plt.xlabel('delta t')
  plt.ylabel('P01')
  
  
