import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

times=1000
rng1=np.random.RandomState(0)
rng2=np.random.RandomState(1)
rng3=np.random.RandomState(1)
dt=0.1
rho=6
sigma=2*(rho**2*dt**2/4/np.pi-dt**2)
random_number1=0.025*rng1.normal(0,dt,times)
random_number2=0.025*rng2.normal(0,sigma,times)
random_jump=rng1.randn(times)

#'''
def judge(x):
    big=0.01
    small=0.0009
    if x>2:
        return big
    if x>1.5:
        return small
    if x<-2:
        return -big
    if x<-1.5:
        return -small
    else:
        return 0

#'''
Judge=[]
for i in range(times):
    Judge.append(judge(random_jump[i]))
V=[1]
alpha=0.3
beta=0.5
for i in range(1,times):
    V.append(V[i-1]+alpha*(beta-V[i-1])*dt+sigma**0.5*V[i-1]**0.5
             +random_number1[i])
S=[175]
lamb=6
r=0.015
A11=[]
A22=[]
for i in range(1,times):
    A1=(r-lamb*0.003)*dt
    A2=V[i]**0.5*random_number2[i]
    A11.append(A1)
    A22.append(A2)
    S.append(S[i-1]+(A1+A2+Judge[i])*S[i-1])
plt.plot(S)
#plt.ylim(-times0,times0)
