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


def single_traj(r,dt,num_steps):
    # Soulivanh: if you know the final length of the vector, you should preallocate it. here:
#    x = np.empty((num_steps + 1) * ntraj)
#    y = np.empty((num_steps + 1) * ntraj)
#    z = np.empty((num_steps + 1) * ntraj)
    #x=np.array([])
    #y=np.array([])
    #z=np.array([])
    pt=[]
    #for single trajectory
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)
        
    # Set initial values
    xs[0], ys[0], zs[0]= (4, -14, 21)
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


def fixed_points(r,b):
    qx=math.sqrt(b*(r-1))
    qy=math.sqrt(b*(b-1))
    qz=r-1
    q=np.array([qx,qy,qz])
    
    return q

def WireframeSphere(centre=[q[0],q[1],q[2]], radius=R,
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



#----MAIN BODY of the code-------------------------------------------------------------------------------------------------------------------------------------

# BOXPLOTS : Probabilities of transition within regions

# forming trajectories

r=64
R=10
pts=single_traj(r,0.01,10000)

with open('pts.npy', 'wb') as h:
    np.save(h, pts)
   
pts=np.load('pts.npy')

#q=np.array([15.0,13.0,37.0])
q=fixed_points(r,8/3)
with open('q.npy', 'wb') as f:
        np.save(f, q)   

# classifying regions
ts,c0,c1,c2,R0,R1,R2=classify2(R,pts,q)
print("c0,c1,c2=",c0,c1,c2)


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
string1="c0={0}; c1={1}; c2={2}; c01={3}; c02={4}; c21={5}; rho={6}; Rad={7}"
string2="c0={0}; c1={c1}; c2={c2}; rho={3}; Radius={4}"
string3="Nume={0}; Deno={1}"
#print(string.format(c0,c1,c2,indir,lenII,r,R))
for k,kitem in enumerate(delt):
    deltak=kitem
    
    indir_01,lenII_01,p01=committor2(deltak,TS,0,1)
    P01.append(p01)
    indir_02,lenII_01,p02=committor2(deltak,TS,0,2)
    P02.append(p02)
    indir_21,lenII_21,p21=committor2(deltak,TS,2,1)
    P21.append(p21)

#------2D LORENZ PLOT ---------------------------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle(string1.format(c0,c1,c2,indir_01,indir_02,indir_21,r,R))


sns.boxplot(ax=axes[0], x=x_labels, y=P01)
axes[0].set_title("x= delta t; y=P01")


sns.boxplot(ax=axes[1], x=x_labels, y=P02)
axes[1].set_title("x=delta t; y=P02")


sns.boxplot(ax=axes[2], x=x_labels, y=P21)
axes[2].set_title("x=delta t; y=P21")


pts=(np.load('pts.npy')).T

#-----3D LORENZ PLOT--------------------------------------------------------------------------------------------------------------------
#q=np.load('q.npy')
q=fixed_points(r, b=8/3)
ts=np.load('ts.npy')

R2=np.where(np.isin(ts,2))
R0=np.where(np.isin(ts,0))
R1=np.where(np.isin(ts,1))


region=[R2,R0,R1,q]    
fig3=plt.figure()


plt.scatter(pts[0][R2],pts[2][R2],alpha=0.1,marker='.',linewidths=0.01)
plt.scatter(pts[0][R0],pts[2][R0],alpha=0.1,marker='.',linewidths=0.01)
plt.scatter(pts[0][R1],pts[2][R1],alpha=1,linewidths=3,c='r')
#sphere= ax.plot_wireframe(*WireframeSphere()color='g',alpha=0.5)
plt.scatter(q[0],q[2],c='black',s=1,alpha=1,linewidths=3,marker='o')
plt.title('rho=28; dt=0.01; radius=1 units; 2D plot')
plt.xlabel("X Axis")
plt.ylabel("Z Axis")
fig3.show()
#ax.plt.legend((line1, line2, line3,line4), region,loc="upper right",title="Regions", bbox_to_anchor=(1.2, 0.5))

#----INTERSECTION OF 3D-LORENZ PLOT WITH SPHERICAL REGION CENTERED AROUND UNSTABLE FIXED POINT--------------------------------------------------------------------------------------------------------------------



fig=plt.figure()
ax = fig.gca(projection='3d')
#ax = Axes3D(fig)
#ax = fig.add_subplot(1, 2, 1, projection='3d')


ax.scatter(pts[0][R2],pts[1][R2],pts[2][R2],alpha=0.1,marker='.',linewidths=0.01)
ax.scatter(pts[0][R0],pts[1][R0],pts[2][R0],alpha=0.1,marker='.',linewidths=0.01) 
ax.scatter(pts[0][R1],pts[1][R1],pts[2][R1],alpha=1,linewidths=3,c='r')
sphere = ax.plot_wireframe(*WireframeSphere(), color="g", alpha=0.3)
ax.scatter(q[0],q[1],q[2],c='black',s=1,alpha=1,linewidths=3,marker='o')
ax.set_title('rho=28; dt=0.01; radius=1 units; 3D plot') 
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
#ax.view_init(30,360)
fig.show()
#ax.legend(region,loc="upper right",title="Regions", bbox_to_anchor=(1.2, 0.5))

#-----INTERACTIVE 3D PLOT----------------------------------------------------------------------------------------------------------------------

import plotly.express as px
import pandas as pd
q0=q[0]
q1=q[1]
q2=q[2]
qx=[q0,q0,q0]
qy=[q1,q1,q1]
qz=[q2,q2,q2]
#df = px.data.gapminder()
df3 = pd.DataFrame({'X' : qx, 'Y' : qy, 'Z' : qz, 'id' : f'id000{3}'})
df1 = pd.DataFrame({'X' : pts[0][R1], 'Y' : pts[1][R1], 'Z' : pts[2][R1], 'id':f'id000{1}'})
df2 = pd.DataFrame({'X': pts[0][R2], 'Y': pts[1][R2], 'Z': pts[2][R2], 'id': f'id000{2}' })
df0 = pd.DataFrame({'X': pts[0][R0], 'Y': pts[1][R0], 'Z': pts[2][R0], 'id': f'id000{0}'})
df = pd.concat([df0,df2,df1,df3])
fig = px.scatter_3d(df, x='X', y="Y", z="Z",color='id',size_max=1)
#fig = px.scatter_3d(df,x=q0, y=q1, z=q2)
#fig["layout"].pop("updatemenus") # optional, drop animation buttons
fig.update_traces(marker=dict(size=1,line=dict(width=0.1,color='DarkSlateGrey')),selector=dict(mode='markers'))

fig.show()

print (df)
