# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:40:40 2022

@author: oscar
"""
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

cAU=1.496*10**11
cyrs=3600*24*365
c=cyrs/cAU
MO=1.989*10**30
G=6.6743*10**-11
G=G*MO*c**2/cAU
N=3     #number of particles

m1=1                #sun
m2=5.97*10**24/MO   #earth
m3=6.42*10**23/MO   #mars

t=np.arange(0,2,0.0001)
n=int(len(t))   #number of time steps

r1=np.array([0,0,0])
v1=np.array([0,0,1])*c*1000

r2=np.array([1,0,0])
v2=np.array([0,29.8,0])*c*1000

r3=np.array([228.0,0,0])*10**9/cAU
v3=np.array([0,24.1,0])*c*1000

#initial conditions and masses
y0=np.hstack((r1,r2,r3,v1,v2,v3))
M=np.array([m1,m2,m3])
R0=np.zeros((3,N,N))
dim=[0,1,2]

def func(y,t):  #version 2 of the integrated function
    global G, M, R0, dim
    global N, N3
    dy=np.zeros_like(y)
    N3=3*N
    dy[:N3]=y[N3:]
    A=y[:N3]
    A=A.reshape((N,3))
    A=np.array(np.split(A,3,axis=-1))
    R=R0
    for i in dim:   #vectorized d=ri-rj
        R[i]=A[i,:]-A[i,:].T
    r=np.sqrt(R[0]**2+R[1]**2+R[2]**2)+np.identity(N)
    #identity for i=j cases (diagonal line)
    #the distance is 0 and will diverge if not
    dd=-G*R/r**3
    ddR=np.sum(M*dd,axis=-1)
    dy[N3:]=ddR.T.reshape((N3))
    return dy

y=sp.integrate.odeint(func,y0,t)

R1=y[:,:3]
R2=y[:,3:6]
R3=y[:,6:9]
V1=y[:,9:12]
V2=y[:,12:15]
V3=y[:,15:]

Rcm=(m1*R1+m2*R2+m3*R3)/np.sum(M)

y=np.vstack([y])
R=np.array(np.split(y[:,:N3],N,axis=-1))
V=np.array(np.split(y[:,N3:],N,axis=-1))
Mm=np.reshape(M,(3,1,1)) 
Rcm=np.sum(Mm*R,axis=-3)/sum(M)

def getEp(npos):    #potential energy at time t[npos]
    global M, R, G, N, n
    pos=R[:,npos,:]
    x=pos[:,0:1]
    y=pos[:,1:2]
    z=pos[:,2:3]
    dx=x.T-x
    dy=y.T-y
    dz=z.T-z
    r=np.sqrt(dx**2+dy**2+dz**2)
    r[r>0] = 1.0/r[r>0]
    Ep=G*np.sum(np.sum(np.triu((-M*M.T)*r,1)))
    return Ep

Ep,Ec=np.zeros(n),np.zeros(n)

M=M.reshape((N,1))
for i in range(n):
    Ep[i]=getEp(i)
    Ec[i]=0.5*np.sum(np.sum(M*V[:,i,:]**2))
E=Ec+Ep
dEc=Ec[n-1]-Ec[0]
dEp=Ep[n-1]-Ep[0]
dE=(E[n-1]-E[0])/E[0]

plt.plot(t,E)   #figure 4
plt.xlabel('t (years)')
plt.ylabel('E (M☉⋅AU²/years²)')
plt.show()

fig=plt.figure(figsize=plt.figaspect(0.5)) #figure 3

ax=fig.add_subplot(1,3,1,projection='3d', title='Center of mass frame')
#relative to cm
R1p=R1-Rcm
R2p=R2-Rcm
R3p=R3-Rcm
ax.plot3D(R1p[:,0],R1p[:,1],R1p[:,2],'black')
ax.plot3D(R2p[:,0],R2p[:,1],R2p[:,2],'blue')
ax.plot3D(R3p[:,0],R3p[:,1],R3p[:,2],'red')
ax.set_xlabel('x (AU)')
ax.set_ylabel('y (AU)')
ax.set_zlabel('z (AU)')
ax.legend(['Sun','Earth','Mars'])


#inertial
ax=fig.add_subplot(1,3,2,projection='3d',title='Inertial frame')
ax.plot3D(R1[:,0],R1[:,1],R1[:,2],'black')
ax.plot3D(R2[:,0],R2[:,1],R2[:,2],'blue')
ax.plot3D(R3[:,0],R3[:,1],R3[:,2],'red')
ax.set_xlabel('x (AU)')
ax.set_ylabel('y (AU)')
ax.set_zlabel('z (AU)')
ax.legend(['Sun','Earth','Mars'])

ax=fig.add_subplot(1,3,3)
ax.plot(t,R1[:,0],'black')
ax.plot(t,R2[:,0],'blue')
ax.plot(t,R3[:,0],'red')
ax.set_xlabel('t (years)')
ax.set_ylabel('x (AU)')
ax.legend(['Sun','Earth','Mars'])

