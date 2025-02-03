# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:40:40 2022

@author: oscar
"""
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib import animation as an
import time
start_time=time.time()

cAU=1.496*10**11
cyrs=3600*24*365
c=cyrs/cAU
MO=1.989*10**30
G=6.6743*10**-11
G=G*MO*c**2/cAU
m1=1                #sun
m2=3.3*10**23/MO    #mercury
m3=4.87*10**24/MO   #venus
m4=5.97*10**24/MO   #earth
m5=6.42*10**23/MO   #mars
m6=1.898*10**27/MO  #jupiter
m7=5.68*10**26/MO   #saturn
m8=8.68*10**25/MO   #uranus
m9=1.02*10**26/MO   #neptune

N=9         #number of particles

tf=50       #end time 
            #setting t=170 takes about 70 seconds 
            #without Energy calculated
dt=0.001    #time step
t=np.arange(0,tf,dt)
n=int(len(t)) #number of time steps


r1=np.array([0,0,0])
v1=np.array([0,0,0])*c*1000

r2=np.array([46*10**9,0,0])/cAU
v2=np.array([0,47.4,0])*c*1000

r3=np.array([108.2*10**9,0,0])/cAU
v3=np.array([0,35.0,0])*c*1000

r4=np.array([1,0,0])
v4=np.array([0,29.8,0])*c*1000

r5=np.array([228.0,0,0])*10**9/cAU
v5=np.array([0,24.1,0])*c*1000

r6=np.array([-778.5,0,0])*10**9/cAU
v6=np.array([0,13.1,0])*c*1000

r7=np.array([1432.0,0,0])*10**9/cAU
v7=np.array([0,9.7,0])*c*1000

r8=np.array([2867.0,0,0])*10**9/cAU
v8=np.array([0,6.8,0])*c*1000

r9=np.array([4515.0,0,0])*10**9/cAU
v9=np.array([0,5.4,0])*c*1000

#initial conditions and masses
y0=np.hstack((r1,r2,r3,r4,r5,r6,r7,r8,r9,v1,v2,v3,v4,v5,v6,v7,v8,v9))
M=np.array([m1,m2,m3,m4,m5,m6,m7,m8,m9])
R0=np.zeros((3,N,N))
dim=[0,1,2]
N3=3*N

def func(y,t):      #version 2 of the integrated function
    global G, M, R0, dim
    global N, N3
    dy=np.zeros_like(y)
    dy[:N3]=y[N3:]
    A=y[:N3]
    A=A.reshape((N,3))
    A=np.array(np.split(A,3,axis=-1))
    R=R0
    for i in dim: #vectorized d=ri-rj
        R[i]=A[i,:]-A[i,:].T
    r=np.sqrt(R[0]**2+R[1]**2+R[2]**2)+np.identity(N) 
    #identity for i=j cases (diagonal line)
    #the distance is 0 and will diverge if not
    dd=-G*R/r**3
    ddR=np.sum(M*dd,axis=-1)
    dy[N3:]=ddR.T.reshape((N3))
    return dy

y=sp.integrate.odeint(func,y0,t)
R=np.array(np.split(y[:,:N3],N,axis=-1))
V=np.array(np.split(y[:,N3:],N,axis=-1))
Rcm=np.sum(M.reshape((N,1,1))*R,axis=-3)/sum(M)


def getEp(npos): #potential energy at time t[npos]
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
for i in range(n): #unoptimized loop
    Ep[i]=getEp(i) #doesn't slow process too much more
    Ec[i]=0.5*np.sum(np.sum(M*V[:,i,:]**2))
E=Ec+Ep
dEc=Ec[n-1]-Ec[0]
dEp=Ep[n-1]-Ep[0]
dE=(E[n-1]-E[0])/E[0]
print('Change in E=',dE)

plt.plot(t,E) #figure not shown on pdf
plt.show()


fig=plt.figure()


ax=fig.add_subplot(1,1,1, title='Solar System')
#relative to cm
R1p=R[0]-Rcm    #sun
R2p=R[1]-Rcm    #mercury
R3p=R[2]-Rcm    #venus
R4p=R[3]-Rcm    #earth
R5p=R[4]-Rcm    #mars
R6p=R[5]-Rcm    #jupiter
R7p=R[6]-Rcm    #saturn
R8p=R[7]-Rcm    #uranus
R9p=R[8]-Rcm    #neptune

#all of this is figure 5 xy graph
ax.plot(R1p[:,0],R1p[:,1],'black')
ax.plot(R2p[:,0],R2p[:,1],'grey')
ax.plot(R3p[:,0],R3p[:,1],'orange')
ax.plot(R4p[:,0],R4p[:,1],'blue')
ax.plot(R5p[:,0],R5p[:,1],'red')
ax.plot(R6p[:,0],R6p[:,1],'purple')
ax.plot(R7p[:,0],R7p[:,1],'violet')
ax.plot(R8p[:,0],R8p[:,1],'indigo')
ax.plot(R9p[:,0],R9p[:,1],'black')
ax.set_xlabel('distance (AU)')
ax.set_ylabel('distance (AU)')
ax.legend(['Sun','Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune'])

ax=plt.axes()
#figure 6 x(t) graph
for i in range(N):
    ax.plot(t,R[i,:,0])
ax.set_xlabel('t (years)')
ax.set_ylabel('x (AU)')
ax.legend(['Sun','Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune'])

end_time = time.time()
print('Elapsed time = ', repr(end_time - start_time))