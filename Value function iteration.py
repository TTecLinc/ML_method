import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

'''
X=np.linspace(0,10,23)
y=X**0.5
plt.scatter(X,y)
f=sci.interpolate.interp1d(X,y,kind='quadratic')
X_new=np.linspace(0,10,100)
plt.plot(X_new,f(X_new))
'''
#'''
size=100
find_size=size*10
beta=0.9
a=0.6
start=1e-50
end=10
def f(x):
    return x**a
def u(x):
    return np.log(x)
def v_real(k):
    return (np.log(1-a*beta)+a*beta*np.log(a*beta)/(1-a*beta))/(1-beta)+a*np.log(k)/(1-a*beta)
k=np.linspace(start,end,size)
k_find=np.linspace(start,end,find_size)
v=np.zeros(size)
c=np.zeros(size)
max_v=[]
max_c=[]
signal=0

#'''
for time in range(61):
    v_f=interpolate.interp1d(k,v,kind='linear')
    c_f=interpolate.interp1d(k,c,kind='linear')
    if time%5==0:
        plt.subplot(211)
        plt.plot(k_find,v_f(k_find))
        plt.subplot(212)
        plt.plot(k_find,c_f(k_find))
    for i in range(size):
        for j in range(find_size):       
            max_c.append(f(k[i])-k_find[j])
            if max_c[-1]<0:
                break
            max_v.append(u(max_c[-1])+beta*v_f(k_find[j]))
        if abs(max(max_v)-v[i])<1e-3:
            signal=1
        v[i]=max(max_v)
        index=max_v.index(max(max_v))
        c[i]=max_c[index]
        max_v=[]
        max_c=[]
        if signal==1:
            break
    if signal==1:
        break
        #'''
#plt.plot(k_find,v_f(k_find))     
plt.subplot(211)
plt.plot(k_find,v_real(k_find))   
plt.ylim(-25,0)
#splt.xlim(1.5,5)  
#'''
