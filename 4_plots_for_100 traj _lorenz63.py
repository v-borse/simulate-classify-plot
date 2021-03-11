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

from module import trajectory
from module import committor3





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
  
  
