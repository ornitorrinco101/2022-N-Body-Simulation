# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:40:40 2022

@author: oscar
"""
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
#from matplotlib import animation as an
import time

start_time=time.time()

cAU=1.496*10**11
MO=1.989*10**30
G=6.6743*10**-11
cT=np.sqrt(cAU**3/(G*MO))
G=G*MO*cT**2/cAU**3
N=20
tf=0.5
dt=0.000001
n=tf/dt
t=np.arange(0,tf,dt)
n=int(len(t))

v=r=np.zeros((3,N))
r[0:3]=np.random.uniform(-1,1,(3,N))
r,v=r.T.reshape(N*3),v.T.reshape(N*3)

y0=np.concatenate((r,v))
M=np.abs(np.random.normal(0,size=N))*2
R0=np.zeros((3,N,N))
dim=[0,1,2]

def func(y,t):
    global G, M, R0, dim
    global N
    dy=np.zeros_like(y)
    N3=3*N
    dy[:N3]=y[N3:]
    A=y[:N3]
    A=A.reshape((N,3))
    A=np.array(np.split(A,3,axis=-1))
    R=R0
    R[0]=A[0,:]-A[0,:].T
    R[1]=A[1,:]-A[1,:].T
    R[2]=A[2,:]-A[2,:].T
    #R[R<10**-5]
    r=np.sqrt(R[0]**2+R[1]**2+R[2]**2)+np.identity(N)
    dd=-G*R/r**3
    ddR=np.sum(M*dd,axis=-1)
    dy[N3:]=ddR.T.reshape((N3))
    return dy

N3=3*N
y=sp.integrate.odeint(func,y0,t)
end_time = time.time()
R=np.array(np.split(y[:,:N3],N,axis=-1))
Mm=np.reshape(M,(N,1,1)) 
Rcm=np.sum(Mm*R,axis=-3)/sum(M)

fig=plt.figure(figsize=plt.figaspect(0.5))

ax=plt.axes(projection='3d')
#relative to cm
Rp=R-np.full((N,n,3),Rcm)
for i in range(N-1):
    ax.plot3D(Rp[i,:,0],Rp[i,:,1],Rp[i,:,2])
#ax.legend(np.round(M,2))
plt.show()

print('Elapsed time = ', repr(end_time - start_time))