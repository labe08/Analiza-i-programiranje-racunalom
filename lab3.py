import math as m
import numpy as np
import matplotlib.pyplot as plt
from lab1 import LUP, splitLU, forwardSubstitution, backwardSubstitution, Matrix

num_of_calls_f = 0 #funkcija
num_of_calls_H = 0 #Hesseova matrica
num_of_calls_g = 0 #gradijent
        
def round2(X):
    if type(X) == int:
        return X
    elif type(X) == float:
        return round(X,5)
    else:
        return [round(x,5) for x in X]

def function(x, f):
    global num_of_calls_f
    num_of_calls_f += 1
    if f==1: #Rosenbrockova 'banana' funkcija
        return (100*(x[1]-x[0]**2)**2) + (1-x[0])**2
    elif f==2:
        return ((x[0]-4)**2) + 4*(x[1]-2)**2
    elif f==3:
        return (x[0]-2)**2+(x[1]+3)**2
    elif f==4:
        return (1/4)*x[0]**4-x[0]**2+2*x[0]+(x[1]-1)**2
    elif f==5:
        return (x[0]**2+x[1]**2-1)**2 + (x[1]-x[0]**2)**2  
    elif f==6:
        measurements = [[1,3], [2,4], [3,4], [5,5], [6,6], [7,8]]
        fu = 0
        for t,y in measurements:
            fu += (x[0]*m.exp(x[1]*t)+x[2] - y)**2
        return fu

def gradient(x, f):
    global num_of_calls_g
    num_of_calls_g += 1
    if f==1: #Rosenbrockova 'banana' funkcija
        g1 = -400*(x[0]*x[1]-x[0]**3)-2+2*x[0]
        g2 = 200*x[1]-200*x[0]**2
    elif f==2:
        g1 = 2*x[0]-8
        g2 = 8*x[1]-16
    elif f==3:
        g1 = 2*x[0]-4
        g2 = 2*x[1]+6
    elif f==4:
        g1 = x[0]**3-2*x[0]+2
        g2 = 2*x[1]-2
    gradient = [g1, g2]
    return gradient

def HessianMatrix(x, f):
    global num_of_calls_H
    num_of_calls_H += 1
    if f==1: #Rosenbrockova 'banana' funkcija
        h1 = 1200*x[0]**2-400*x[1]+2
        h2 = -400*x[0]
        h3 = -400*x[0]
        h4 = 200
    elif f==2:
        h1 = 2
        h2 = 0
        h3 = 0
        h4 = 8
    elif f==3:
        h1 = 2
        h2 = 0
        h3 = 0
        h4 = 2
    elif f==4:
        h1 = 3*x[0]**2-2
        h2 = 0
        h3 = 0
        h4 = 2
    hessian = [[h1, h2], [h3, h4]]
    return hessian

def Jacobian(x, num):
    if num==4:
        j1 = -20*x[0]
        j2 = 10
        j3 = -1
        j4 = 0
    elif num==5:
        j1 = 2*x[0]
        j2 = 2*x[1]
        j3 = -2*x[0]
        j4 = 1
    elif num==6:
        j1 = m.exp(x[1]); j2 = x[0]*m.exp(x[1]); j3 = 1
        j4 = m.exp(2*x[1]); j5 = 2*x[0]*m.exp(2*x[1]); j6 = 1
        j7 = m.exp(3*x[1]); j8 = 3*x[0]*m.exp(3*x[1]); j9 = 1
        j10 = m.exp(5*x[1]); j11 = 5*x[0]*m.exp(5*x[1]); j12 = 1
        j13 = m.exp(6*x[1]); j14 = 6*x[0]*m.exp(6*x[1]); j15 = 1
        j16 = m.exp(7*x[1]); j17 = 7*x[0]*m.exp(7*x[1]); j18 = 1
        return [[j1,j2,j3], [j4,j5,j6], [j7,j8,j9], [j10,j11,j12], [j13,j14,j15], [j16,j17,j18]]
    jacobian = [[j1, j2], [j3, j4]]
    return jacobian

def Gx(x, num):
    global num_of_calls_g
    num_of_calls_g += 1
    if num == 4:
        g1 = 10*(x[1]-x[0]**2)
        g2 = 1-x[0]
    elif num == 5:
        g1 = x[0]**2+x[1]**2-1
        g2 = x[1]-x[0]**2
    elif num == 6:
        measurements = [[1,3], [2,4], [3,4], [5,5], [6,6], [7,8]]
        g = []
        for t,y in measurements:
            g.append(x[0]*m.exp(x[1]*t)+x[2] - y)
        return g
    return [g1,g2]

def solveEq(A, g):
    try:
        lu, P, s = LUP(A)
        L, U = splitLU(lu)
        y = forwardSubstitution(L, np.matmul(P.matrix, g))
        x = backwardSubstitution(U, y)
    except:
        return None
    return x

def unimodal(point, fun, h=1):
    l = point - h
    r = point + h
    m = point
    step = 1
    fm = fun(m)
    fl = fun(l)
    fr = fun(r)
    if fm < fr and fm < fl:
        return l,r
    elif fm > fr:
        while fm > fr:
            l = m
            m = r
            fm = fr
            step *= 2
            r = point + h * step
            fr = fun(r)
    else:
        while fm > fl:
            r = m
            m = l
            fm = fl
            step *= 2
            l = point - h * step
            fl = fun(l)   
    return l,r

def goldenSection(fun, x, v, b=None, e=0.00001):
    a = x[:]
    a1 = abs((a[0] - x[0]) / v[0])
    if not b:
        a1, b1 = unimodal(a1, fun)
    k = 0.5 * (m.sqrt(5) - 1)
    c = b1 - k * (b1 - a1)
    d = a1 + k * (b1 - a1)
    fc = fun(c)
    fd = fun(d)
    while ((b1 - a1) > e):
        if (fc < fd):
            b1 = d
            d = c
            c = b1 - k * (b1 - a1)
            fd = fc
            fc = fun(c)
        else:
            a1 = c
            c = d
            d = a1 + k * (b1 - a1)
            fc = fd
            fd = fun(d)
    return (b1+a1)/2

def fastestDescent(x0, f, e=0.00001, golden_section=False):
    global num_of_calls_f
    global num_of_calls_g
    num_of_calls_f = 0
    num_of_calls_g = 0

    x = x0[:]
    j = 0
    while True:
        f_prev = function(x, f)
        grad = gradient(x, f)
        v = [-g for g in grad]
        if golden_section:
            alpha = goldenSection(lambda a: function([x[j]-a*grad[j] for j in range(len(x))], f), x, v)
        else:
            alpha = 1
        for i in range(len(x)):
            x[i] = x[i] - alpha * grad[i]
        f_new = function(x, f)
        if f_new >= f_prev:
            j+=1
        else: j = 0
        res = [abs(g) <= e for g in grad]
        if all(res):
            return x, f_new, num_of_calls_f, num_of_calls_g
        if j >= 10:
            return None, None, None, None

def NewtonRaphson(x0, f, e=0.00001, golden_section=False):
    global num_of_calls_f
    global num_of_calls_g
    global num_of_calls_H
    num_of_calls_f = 0
    num_of_calls_g = 0
    num_of_calls_H = 0

    x = x0[:]
    j = 0
    while True:
        f_prev = function(x, f)
        grad = gradient(x, f)
        Hessian = Matrix(m=HessianMatrix(x, f))
        v = [g*(-1) for g in grad]
        delta_x = solveEq(Hessian, v)
        if golden_section:
            alpha = goldenSection(lambda a: function([x[j]-a*delta_x[j] for j in range(len(x))], f), x, v)
        else:
            alpha = 1
        for i in range(len(x)):
            x[i] = x[i] - alpha * delta_x[i]
        f_new = function(x, f)
        if f_new >= f_prev:
            j+=1
        else: j = 0
        res = [abs(g) <= e for g in grad]
        if all(res):
            return x, float(f_new), num_of_calls_f, num_of_calls_g, num_of_calls_H
        if j >= 10:
            return None, None, None, None, None

def GaussNewton(x0, f, num, e=0.00001, golden_section=False):
    global num_of_calls_f
    global num_of_calls_g
    num_of_calls_f = 0
    num_of_calls_g = 0
    
    x = x0[:]
    j = 0
    while True:
        f_prev = function(x, f)
        jacobian = Matrix(m=Jacobian(x, num))
        gx = Gx(x, num)
        Jt = Matrix(m=jacobian.transpose())
        a = Jt.multiply(jacobian)
        A = Matrix(m=a)
        grad = Jt.multiply(gx)  
        v = [-1*g for g in grad]  
        delta_x = solveEq(A, v)
        if delta_x is None:
            return -1, -1, -1, -1
        if golden_section:
            alpha = goldenSection(lambda a: function([x[j]+a*delta_x[j] for j in range(len(x))], f), x, v)
        else:
            alpha = 1
        x = [a+b*alpha for a,b in zip(x, delta_x)]
        f_new = function(x, f)
        if f_new >= f_prev:
            j+=1
        else: j = 0
        res = [abs(d_x * alpha) <= e for d_x in delta_x]
        if all(res):
            return x, float(f_new), num_of_calls_f, num_of_calls_g
        if j >= 10:
            return None, None, None, None


'''
zdt = int(input("Upisite broj zadatka (1-6):"))

if zdt==1:
    print("\nZADATAK 1")
    print("--------------")
    x1, f1, num_of_calls_f1, num_of_calls_g1 = fastestDescent(x0=[0,0], f=3)
    if x1:
        print("Without golden section for f3 and x0=[0,0]: xmin={} f(xmin)={} num_of_calls_f={} num_of_calls_g={}".format(round2(x1), round2(f1), round2(num_of_calls_f1), round2(num_of_calls_g1)))
    else: print("Without golden section for f3 and x0=[0,0]: divergence")
    x2, f2, num_of_calls_f2, num_of_calls_g2 = fastestDescent(x0=[0,0], f=3, golden_section=True)
    if x2:
        print("With golden section for f3 and x0=[0,0]: xmin={} f(xmin)={} num_of_calls_f={} num_of_calls_g={}".format(round2(x2), round2(f2), round2(num_of_calls_f2), round2(num_of_calls_g2)))
    else: print("With golden section for f3 and x0=[0,0]: divergence")

elif zdt==2:
    print("\nZADATAK 2")
    print("--------------")
    x1, f1, num_of_calls_f1, num_of_calls_g1 = fastestDescent(x0=[-1.9,2], f=1, golden_section=True)
    if x1:
        print("Fastest descent with golden section for f1 and x0=[-1.9,2]: xmin={} f(xmin)={} num_of_calls_f={} num_of_calls_g={}".format(round2(x1), round2(f1), round2(num_of_calls_f1), round2(num_of_calls_g1)))
    else: print("Fastest descent with golden section for f1 and x0=[-1.9,2]: divergence")
    x2, f2, num_of_calls_f2, num_of_calls_g2, num_of_calls_H2 = NewtonRaphson(x0=[-1.9,2], f=1, golden_section=True)
    if x2:
        print("Newton-Raphson with golden section for f1 and x0=[-1.9,2]: xmin={} f(xmin)={} num_of_calls_f={} num_of_calls_g={} num_of_calls_H={}".format(round2(x2), round2(f2), round2(num_of_calls_f2), round2(num_of_calls_g2), round2(num_of_calls_H2)))
    else: print("Newton-Raphson with golden section for f1 and x0=[-1.9,2]: divergence")
    x3, f3, num_of_calls_f3, num_of_calls_g3 = fastestDescent(x0=[0.1,0.3], f=2, golden_section=True)
    if x3:
        print("Fastest descent with golden section for f2 and x0=[0.1,0.3]: xmin={} f(xmin)={} num_of_calls_f={} num_of_calls_g={}".format(round2(x3), round2(f3), round2(num_of_calls_f3), round2(num_of_calls_g3)))
    else: print("Fastest descent with golden section for f2 and x0=[0.1,0.3]: divergence")
    x4, f4, num_of_calls_f4, num_of_calls_g4, num_of_calls_H4 = NewtonRaphson(x0=[0.1,0.3], f=2, golden_section=True)
    if x4:
        print("Newton-Raphson with golden section for f2 and x0=[0.1,0.3]: xmin={} f(xmin)={} num_of_calls_f={} num_of_calls_g={} num_of_calls_H={}".format(round2(x4), round2(f4), round2(num_of_calls_f4), round2(num_of_calls_g4), round2(num_of_calls_H4)))
    else: print("Newton-Raphson with golden section for f2 and x0=[0.1,0.3]: divergence")

elif zdt==3:
    print("\nZADATAK 3")
    print("--------------")
    x1, f1, num_of_calls_f1, num_of_calls_g1, num_of_calls_H1 = NewtonRaphson(x0=[3,3], f=4)
    if x1:
        print("Newton-Raphson without golden section for f4 and x0=[3,3]: xmin={} f(xmin)={} num_of_calls_f={} num_of_calls_g={} num_of_calls_H={}".format(round2(x1), round2(f1), round2(num_of_calls_f1), round2(num_of_calls_g1), round2(num_of_calls_H1)))
    else: print("Newton-Raphson without golden section for f4 and x0=[3,3]: divergence")
    x2, f2, num_of_calls_f2, num_of_calls_g2, num_of_calls_H2 = NewtonRaphson(x0=[1,2], f=4)
    if x2:
        print("Newton-Raphson without golden section for f4 and x0=[1,2]: xmin={} f(xmin)={} num_of_calls_f={} num_of_calls_g={} num_of_calls_H={}".format(round2(x2), round2(f2), round2(num_of_calls_f2), round2(num_of_calls_g2), round2(num_of_calls_H2)))
    else: print("Newton-Raphson without golden section for f4 and x0=[1,2]: divergence")
    x3, f3, num_of_calls_f3, num_of_calls_g3, num_of_calls_H3 = NewtonRaphson(x0=[3,3], f=4, golden_section=True)
    if x3:
        print("Newton-Raphson with golden section for f4 and x0=[3,3]: xmin={} f(xmin)={} num_of_calls_f={} num_of_calls_g={} num_of_calls_H={}".format(round2(x3), round2(f3), round2(num_of_calls_f3), round2(num_of_calls_g3), round2(num_of_calls_H3)))
    else: print("Newton-Raphson with golden section for f4 and x0=[3,3]: divergence")
    x4, f4, num_of_calls_f4, num_of_calls_g4, num_of_calls_H4 = NewtonRaphson(x0=[1,2], f=4, golden_section=True)
    if x4:
        print("Newton-Raphson with golden section for f4 and x0=[1,2]: xmin={} f(xmin)={} num_of_calls_f={} num_of_calls_g={} num_of_calls_H={}".format(round2(x4), round2(f4), round2(num_of_calls_f4), round2(num_of_calls_g4), round2(num_of_calls_H4)))
    else: print("Newton-Raphson with golden section for f4 and x0=[1,2]: divergence")

elif zdt==4:
    print("\nZADATAK 4")
    print("--------------")
    x1, f1, num_of_calls_f1, num_of_calls_g1 = GaussNewton(x0=[-1.9,2], f=1, num=4)
    if x1 == -1:
        print("Gauss-Newton not possible.")
    elif x1:
        print("Gauss-Newton without golden section for f1 and x0=[-1.9,2]: xmin={} f(xmin)={} num_of_calls_f={} num_of_calls_g={}".format(round2(x1), round2(f1), round2(num_of_calls_f1), round2(num_of_calls_g1)))
    else: print("Gauss-Newton without golden section for f1 and x0=[-1.9,2]: divergence")
    x2, f2, num_of_calls_f2, num_of_calls_g2 = GaussNewton(x0=[-1.9,2], f=1, num=4, golden_section=True)
    if x2 == -1:
        print("Gauss-Newton not possible.")
    elif x2:
        print("Gauss-Newton with golden section for f1 and x0=[-1.9,2]: xmin={} f(xmin)={} num_of_calls_f={} num_of_calls_g={}".format(round2(x2), round2(f2), round2(num_of_calls_f2), round2(num_of_calls_g2)))
    else: print("Gauss-Newton with golden section for f1 and x0=[-1.9,2]: divergence")

elif zdt == 5:
    print("\nZADATAK 5")
    print("--------------")
    x1, f1, num_of_calls_f1, num_of_calls_g1 = GaussNewton(x0=[-2,2], f=5, num=5, golden_section=True)
    if x1 == -1:
        print("Gauss-Newton not possible.")
    elif x1:
        print("Gauss-Newton with golden section for f5 and x0=[-2,2]: xmin={} f(xmin)={} num_of_calls_f={} num_of_calls_g={}".format(round2(x1), round2(f1), round2(num_of_calls_f1), round2(num_of_calls_g1)))
    else: print("Gauss-Newton with golden section for f1 and x0=[-2,2]: divergence")
    x2, f2, num_of_calls_f2, num_of_calls_g2 = GaussNewton(x0=[2,2], f=5, num=5, golden_section=True)
    if x2 == -1:
        print("Gauss-Newton not possible.")
    elif x2:
        print("Gauss-Newton with golden section for f5 and x0=[2,2]: xmin={} f(xmin)={} num_of_calls_f={} num_of_calls_g={}".format(round2(x2), round2(f2), round2(num_of_calls_f2), round2(num_of_calls_g2)))
    else: print("Gauss-Newton with golden section for f5 and x0=[2,2]: divergence")
    x3, f3, num_of_calls_f3, num_of_calls_g3 = GaussNewton(x0=[2,-2], f=5, num=5, golden_section=True)
    if x3 == -1:
        print("Gauss-Newton not possible.")
    elif x3:
        print("Gauss-Newton with golden section for f5 and x0=[2,-2]: xmin={} f(xmin)={} num_of_calls_f={} num_of_calls_g={}".format(round2(x3), round2(f3), round2(num_of_calls_f3), round2(num_of_calls_g3)))
    else: print("Gauss-Newton with golden section for f5 and x0=[2,-2]: divergence")

elif zdt == 6:
    print("\nZADATAK 6")
    print("--------------")
    x, f, num_of_calls_f, num_of_calls_g = GaussNewton(x0=[1,1,1], f=6, num=6, golden_section=True)
    if x == -1:
        print("Gauss-Newton not possible.")
    elif x:
        print("Gauss-Newton with golden section for f6 and x0=[1,1,1]: xmin={} f(xmin)={} num_of_calls_f={} num_of_calls_g={}".format(round2(x), round2(f), round2(num_of_calls_f), round2(num_of_calls_g)))

        data_points = np.array([(1, 3), (2, 4), (3, 4), (5, 5), (6, 6), (7, 8)])
        x_min = x
        def model(x, t):
            return x[0] * np.exp(x[1] * t) + x[2]
        model_values = [model(x_min, t) for t, _ in data_points]

        plt.scatter(data_points[:, 0], data_points[:, 1], label='Zadane točke')
        plt.plot(data_points[:, 0], model_values, label='Rezultat modela', color='red')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.legend()
        plt.title('Graf modela i zadanih točaka')
        plt.show()
    else: print("Gauss-Newton with golden section for f6 and x0=[1,1,1]: divergence")

'''  