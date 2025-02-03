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

m1=1
m2=1

t=np.arange(0,5,0.001)

r1=np.array([-0.5,0,0])
v1=np.array([0,-15,0])*c*1000

r2=np.array([0.5,0,0])
v2=np.array([0,15,0])*c*1000

y0=np.hstack((r1,r2,v1,v2)) #initial conditions

def func(y,t):  #integration function
    global G    #using 1st version of this function
    global m1   #later programs are the final "optimized"
    global m2   # version for N particles
    R1=y[:3]
    R2=y[3:6]
    R12=R2-R1
    r12=np.linalg.norm(R12)

    dy=np.zeros_like(y)
    dy[:6]=y[6:]
    dd12=G*R12/r12**3
    ddR1=dd12*m2
    ddR2=-dd12*m1
    dy[6:9]=ddR1    #force for m1
    dy[9:]=ddR2     #force for m2
    return dy

y=sp.integrate.odeint(func,y0,t)

R1=y[:,:3]  #unpacking values
R2=y[:,3:6]
V1=y[:,6:9]
V2=y[:,9:]

r1,r2,v1,v2=np.linalg.norm(R1,axis=-1),np.linalg.norm(R2,axis=-1),np.linalg.norm(V1,axis=-1),np.linalg.norm(V2,axis=-1)

R12=R2-R1
r12=np.linalg.norm(R12,axis=-1)

BC=(m1*R1+m2*R2)/(m1+m2) #center of mass
Ec=0.5*(m1*v1**2+m2*v2**2)
Ep=-G*m1*m2/r12
n=len(Ec)
E=Ec+Ep
dE=(E[n-1]-E[0])/E[0]
print('Change in E=',dE)
dEc=Ec[n-1]-Ec[0]
dEp=Ep[n-1]-Ep[0]

plt.plot(t,E)       #figure 2
plt.xlabel('t (years)')
plt.ylabel('E   (M☉⋅AU²/years²)')

fig=plt.figure()    #figure 1


ax=fig.add_subplot(1,2,1)
#relative to cm
R1p=R1-BC
R2p=R2-BC
ax.plot(R1p[:,0],R1p[:,1],'red')
ax.plot(R2p[:,0],R2p[:,1],'blue')
ax.set_xlabel('x (AU)')
ax.set_ylabel('y (AU)')
ax.legend(['m1','m2'])

ax=fig.add_subplot(1,2,2)
#relative to cm
R1p=R1-BC
R2p=R2-BC
ax.plot(t,R1[:,0],'red')
ax.plot(t,R2p[:,0],'blue')
ax.set_xlabel('time (years)')
ax.set_ylabel('x (AU)')
ax.legend(['m1','m2'])