import random
import numpy as np
from lab3 import round2

def function(x, f):
    if f==1: #Rosenbrockova 'banana' funkcija
        return (100*(x[1]-x[0]**2)**2) + (1-x[0])**2
    elif f==2:
        return ((x[0]-4)**2) + 4*(x[1]-2)**2
    elif f==3:
        return (x[0]-2)**2+(x[1]+3)**2
    elif f==4:
        return (x[0]-3)**2+x[1]**2

def limits(x, zdt):
    if zdt==1:
        li1 = x[1]-x[0] >= 0
        li2 = 2-x[0] >= 0
        le = -100 <= x[0] <= 100 and -100 <= x[1] <= 100
        return li1 and li2 and le
    elif zdt==2:
        li1 = x[1]-x[0]
        li2 = 2-x[0]
        return li1 >= 0 and li2 >= 0, [li1, li2], 0
    elif zdt==3:
        li1 = 3-x[0]-x[1]
        li2 = 3+1.5*x[0]-x[1]
        li3 = x[1]-1
        return li1 >= 0 and li2 >= 0 and li3 == 0, [li1, li2], li3

def U(x, f, t, zdt):
    if not limits(x, zdt)[0]:
        return np.inf
    fu = function(x, f)
    g_sum = 0
    for limit in limits(x, zdt)[1]:
        g_sum -= np.log(limit) if limit > 0 else 1000000
    g_sum *= (1/t)
    h_sum = t * limits(x, zdt)[2]**2
    u = fu + g_sum + h_sum
    return u
    
def box(x0, f, zdt, xdi, xgi, alpha=1.3, e=0.00001):
    if not limits(x0, zdt): 
        print("x0 does not satisfy the limits!")
        return
    n = len(x0)
    xc = x0[:]
    xt = [xc]
    for t in range(2*n):
        xi = []
        for i in range(n):
            r = random.uniform(0,1)
            xi.append(xdi + r*(xgi-xdi))
        while True:
            if not limits(xi, zdt):
                for i in range(n):
                    xi[i] = (xi[i] + xc[i])/2
            else: break
        xt.append(xi)
        xc = [sum(x[i] for x in xt)/len(xt) for i in range(n)]
    while True:
        h = np.argmax([function(x, f) for x in xt])
        xh = xt.pop(h)
        h2 = np.argmax([function(x, f) for x in xt])
        xh2 = xt[h2]
        #centroid bez xh
        xc = [sum(x[i] for x in xt)/len(xt) for i in range(n)]
        #refleksija
        xr = [(1+alpha)*xc[i] - alpha*xh[i] for i in range(n)]
        #eksplicitna ogranicenja
        for i in range(n):
            if xr[i] < xdi:
                xr[i] = xdi
            elif xr[i] > xgi:
                xr[i] = xgi
        #implicitna ogranicenja
        while True:
            if not limits(xr, zdt):
                for i in range(n):
                    xr[i] = (xr[i] + xc[i])/2
            else: break
        if function(xr, f) > function(xh2, f):
            for i in range(n):
                xr[i] = (xr[i] + xc[i])/2
        xt.append(xr)
        res = [abs(xc[i]-xh[i]) <= e for i in range(n)]
        if all(res):
            return xc, function(xc, f)

def search(xP, dx, fun):
    x = xP[:]
    for i in range(len(x)):
        p = fun(x)
        x[i] += dx
        n = fun(x)
        if n > p:
            x[i] -= 2*dx
            n = fun(x)
            if n > p:
                x[i] += dx
    return x

def algHookeJeeves(x0, fun, dx=1, epsilon=0.00001):
    xP = xB = x0[:]
    while dx > epsilon:
        xN = search(xP, dx, fun)
        fxN = fun(xN)
        fxB = fun(xB)
        if fxN < fxB:
            for i in range(len(xP)):
                xP[i] = 2*xN[i] - xB[i]
            xB = xN
        else:
            dx /= 2
            xP = xB
    return xB

def G(x, zdt):
    g = 0
    for l in limits(x, zdt)[1]:
        if l < 0:
            t = 1
        else:
            t = 0
        g -= t * l
    return g

def innerPointSearch(x0, zdt, e=0.00001):
    x = x0[:] 
    while True:
        xs = x
        x = algHookeJeeves(xs, lambda a: G(a, zdt))
        res = [abs(xs[i]-x[i]) <= e for i in range(len(x0))]
        if all(res):
            return x
    
def transforme(x0, f, zdt, t=1, e=0.00001):
    x = x0[:]
    if not limits(x, zdt)[0]:
        x = innerPointSearch(x, zdt)
    while True:
        xprev = x
        x = algHookeJeeves(xprev, lambda a: U(a, f, t, zdt))
        res = [abs(xprev[i]-x[i]) <= e for i in range(len(x0))]
        if all(res):
            return x, function(x, f)
        t *= 10


print("\nZADATAK 1")
print("--------------")
x1, f1 = box(x0=[-1.9,2], f=1, zdt=1, xdi=-100, xgi=100)
print("xmin for f1=" + str(round2(x1)) + " f(xmin)=" + str(round(f1,2)))
x2, f2 = box(x0=[0.1,0.3], f=2, zdt=1, xdi=-100, xgi=100)
print("xmin for f2=" + str(round2(x2)) + " f(xmin)=" + str(round(f2,2)))

print("\nZADATAK 2")
print("--------------")
x1, f1 = transforme(x0=[0.5,2], f=1, zdt=2)
print("xmin for f1=" + str(round2(x1)) + " f(xmin)=" + str(round(f1,2)))
x2, f2 = transforme(x0=[0.1,0.3], f=2, zdt=2)
print("xmin for f2=" + str(round2(x2)) + " f(xmin)=" + str(round(f2,2)))

print("\nZADATAK 3")
print("--------------")
x1, f1 = transforme(x0=[5,5], f=4, zdt=3)
print("xmin for f4=" + str(x1) + " f(xmin)=" + str(round(f1,2)) + "\n")