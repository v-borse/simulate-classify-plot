#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 09:21:29 2021

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

from module import committor2
    

    
#ts=[0,0,2,2,2,2,1,1,1,2,2,0,0,1,0,0,2,1,1,0,0,2,2,2,2,1,1,1,2,2,0,0,1,0,0,2,1,1]
#ts=[1,1,2,2,1,0,2,2,0,1,2,0]
TS=np.load('ts.npy') 
# time series for trajectories can be directly loaded 
diff=3

st1=0
st2=1


delt=[10,50,100,200,500,1000]
labels=[0.1,0.5,1,2,5,10]
print("Switching from st1=",st1, "to st2=",st2)

for k,kitem in enumerate(delt):
    deltak=kitem
    P=[]
    ii=1
    for i in range(99):
        #print(i)
    
        p=committor2(deltak,TS[i*10001:ii*10001],st1,st2)
        P.append(p)
    
        #print(" Probability= ",P )
        ii+=1
    plt.boxplot(P,positions=[k])
    plt.title('rho=64; dt=0.01; radius=5 units; Boxplot')
    plt.xlabel("delta t")
    plt.ylabel("P01")
    plt.xticks([0,1,2,3,4,5],labels)
