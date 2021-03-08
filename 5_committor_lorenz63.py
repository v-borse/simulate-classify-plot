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


    
def committor2(delta,ts,st1,st2):
    """
    Given:
        delta = # of time steps within which there is a switching
        ts = time series
        Example: if we have to switch from 0 to 1 then
        st1=0
        st2=1
    Returns:
        direct = counts for direct switching from st1 to st2
        indirect = counts for indirect switching from st1 to st2
    """
        
    indirect=0
    direct=0
    II=np.where(np.isin(ts,st1))[0]
        # II= list of indices in ts[i]==st1
    #print(II)
    #print(len(II))
    for i,item in enumerate(II):
        if(item < (len(ts) - 1)):
            iend = min(item+delta+1, len(ts))
            #print(item+1)
            #print(iend)
            trial=ts[item+1:iend]
            #print(trial)
            #print((np.isin(trial,st2)))
            indirect +=  np.any(np.isin(trial,st2))
    p=indirect/len(II)
    #print(p)   
    return(p)




    
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
