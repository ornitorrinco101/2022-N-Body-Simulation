# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 15:49:47 2022

@author: oscar
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:40:40 2022

@author: oscar
"""
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib import animation as ang
import time
start_time=time.time()

cAU=1.496*10**11
MO=1.989*10**30
G=6.6743*10**-11
cT=np.sqrt(cAU**3/(G*MO))
G=G*MO*cT**2/cAU**3
N=3

m1=3
m2=4
m3=5
dt=0.00001
dt2=dt
tf=40
n1=int(10/dt)
n2=int(10/dt2)
t1=np.arange(0,tf,dt)
tf2=70
t2=np.arange(tf,tf2,dt)
t=np.concatenate([t1,t2])
n=int(len(t))


r1=np.array([1,3,0])
v=np.array([0,0,0])

r2=np.array([-2,-1,0])

r3=np.array([1,-1,0])

y0=np.hstack((r1,r2,r3,v,v,v))
M=np.array([m1,m2,m3])
R0=np.zeros((3,N,N))
dim=[0,1,2]
N3=3*N

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

y=sp.integrate.odeint(func,y0,t,atol=1*10**-19)
y=np.vstack([y])
R=np.array(np.split(y[:,:N3],N,axis=-1))

#can calculate Energy but it will significantly slow
#down the program

Mm=np.reshape(M,(3,1,1)) 
Rcm=np.sum(Mm*R,axis=-3)/sum(M)

'''
V=np.array(np.split(y[:,N3:],N,axis=-1))
def getEp(npos):
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
'''

R1=y[:,:3]
R2=y[:,3:6]
R3=y[:,6:9]
V1=y[:,9:12]
V2=y[:,12:15]
V3=y[:,15:]

fig=plt.figure(figsize=plt.figaspect(0.5)) #figure 7

ax=fig.add_subplot(2,4,1, title='t=0 to t=10')
#t 0-10
t10=n1
R1p=R1[0:t10]-Rcm[0:t10]
R2p=R2[0:t10]-Rcm[0:t10]
R3p=R3[0:t10]-Rcm[0:t10]
ax.plot(R1p[:,0],R1p[:,1],'r')
ax.plot(R2p[:,0],R2p[:,1],'b')
ax.plot(R3p[:,0],R3p[:,1],'g')
ax.set_xlabel('x (AU)')
ax.set_ylabel('y (AU)')
ax.legend(['m1','m2','m3'])


#t 10-20
t20=t10+n1
R1p=R1[t10:t20]-Rcm[t10:t20]
R2p=R2[t10:t20]-Rcm[t10:t20]
R3p=R3[t10:t20]-Rcm[t10:t20]
ax=fig.add_subplot(2,4,2,title='t=10 to t=20')
ax.plot(R1p[:,0],R1p[:,1],'red')
ax.plot(R2p[:,0],R2p[:,1],'blue')
ax.plot(R3p[:,0],R3p[:,1],'green')

#t 20-30
t30=t20+n1
R1p=R1[t20:t30]-Rcm[t20:t30]
R2p=R2[t20:t30]-Rcm[t20:t30]
R3p=R3[t20:t30]-Rcm[t20:t30]
ax=fig.add_subplot(2,4,3,title='t=20 to t=30')
ax.plot(R1p[:,0],R1p[:,1],'red')
ax.plot(R2p[:,0],R2p[:,1],'blue')
ax.plot(R3p[:,0],R3p[:,1],'green')

#t 30-40
t40=t30+n1
R1p=R1[t30:t40]-Rcm[t30:t40]
R2p=R2[t30:t40]-Rcm[t30:t40]
R3p=R3[t30:t40]-Rcm[t30:t40]
ax=fig.add_subplot(2,4,4,title='t=30 to t=40')
ax.plot(R1p[:,0],R1p[:,1],'red')
ax.plot(R2p[:,0],R2p[:,1],'blue')
ax.plot(R3p[:,0],R3p[:,1],'green')

#t 40-50
t50=t40+n2
R1p=R1[t40:t50]-Rcm[t40:t50]
R2p=R2[t40:t50]-Rcm[t40:t50]
R3p=R3[t40:t50]-Rcm[t40:t50]
ax=fig.add_subplot(2,4,5,title='t=40 to t=50')
ax.plot(R1p[:,0],R1p[:,1],'red')
ax.plot(R2p[:,0],R2p[:,1],'blue')
ax.plot(R3p[:,0],R3p[:,1],'green')


#t 50-60
t60=t50+n2
R1p=R1[t50:t60]-Rcm[t50:t60]
R2p=R2[t50:t60]-Rcm[t50:t60]
R3p=R3[t50:t60]-Rcm[t50:t60]
ax=fig.add_subplot(2,4,6,title='t=50 to t=60')
ax.plot(R1p[:,0],R1p[:,1],'red')
ax.plot(R2p[:,0],R2p[:,1],'blue')
ax.plot(R3p[:,0],R3p[:,1],'green')

#t 60-70
t70=t60+n2
R1p=R1[t60:t70]-Rcm[t60:t70]
R2p=R2[t60:t70]-Rcm[t60:t70]
R3p=R3[t60:t70]-Rcm[t60:t70]
ax=fig.add_subplot(2,4,7,title='t=60 to t=70')
ax.plot(R1p[:,0],R1p[:,1],'red')
ax.plot(R2p[:,0],R2p[:,1],'blue')
ax.plot(R3p[:,0],R3p[:,1],'green')

end_time = time.time()
print('Elapsed time = ', repr(end_time - start_time))