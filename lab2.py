import math as m
import numpy as np

num_of_calls = 0

def readFile(filename):
    params = []
    with open(filename, 'r') as f:
        for line in f.read().splitlines():
            params.append([float(x) for x in line.split(' ')])
    return params
        
def round2(X):
    return [round(x,2) for x in X]

def function(x, f):
    global num_of_calls
    num_of_calls += 1
    if f==0:
        return (x[0]-3)**2
    elif f==1: #Rosenbrockova 'banana' funkcija
        return (100*(x[1]-x[0]**2)**2) + (1-x[0])**2
    elif f==2:
        return ((x[0]-4)**2) + 4*(x[1]-2)**2
    elif f==3:
        y = 0
        for i, xi in enumerate(x):
            y += (xi-i)**2
        return y
    elif f==4: #Jakobovićeva funkcija 
        return abs((x[0]-x[1])*(x[0]+x[1])) + m.sqrt((x[0]**2)+(x[1]**2))
    elif f==6: #Schaffer's function
        y = 0
        for xi in x:
            y += xi**2
        y2 = m.sin(m.sqrt(y))**2 - 0.5
        y3 = (1+0.001*y)**2
        return 0.5 + y2/y3

def unimodal(point, f, h=1):
    l = point - h
    r = point + h
    m = point
    step = 1
    fm = function([m], f)
    fl = function([l], f)
    fr = function([r], f)
    if fm < fr and fm < fl:
        return l,r
    elif fm > fr:
        while fm > fr:
            l = m
            m = r
            fm = fr
            step *= 2
            r = point + h * step
            fr = function([r], f)
    else:
        while fm > fl:
            r = m
            m = l
            fm = fl
            step *= 2
            l = point - h * step
            fl = function([l], f)   
    return l,r    

def goldenSection(f, x0, b=None, e=0.00001):
    a = x0[0]
    if not b:
        a, b = unimodal(a, f)
    k = 0.5 * (m.sqrt(5) - 1)
    c = b - k * (b - a)
    d = a + k * (b - a)
    fc = function([c], f)
    fd = function([d], f)
    while ((b - a) > e):
        if (fc < fd):
            b = d
            d = c
            c = b - k * (b - a)
            fd = fc
            fc = function([c], f)
        else:
            a = c
            c = d
            d = a + k * (b - a)
            fc = fd
            fd = function([d], f)
        #print("a={} fa={} b={} fb={} c={} fc={} d={} fd={}".format(round(a,2),round(function([a], f),2),round(b,2),round(function([b], f),2),round(c,2),round(fc,2),round(d,2),round(fd,2)))
    return (a + b) / 2

def unimodal2(point, f, x, i, h=1):
    l = point - h
    r = point + h
    m = point
    step = 1
    fm = function(x, f)
    xl = x[:]
    xl[i] = l
    fl = function(xl, f)
    xr = x[:]
    xr[i] = r
    fr = function(xr, f)
    if fm < fr and fm < fl:
        return l,r
    elif fm > fr:
        while fm > fr:
            l = m
            m = r
            fm = fr
            step *= 2
            r = point + h * step
            xr[i] = r
            fr = function(xr, f)
    else:
        while fm > fl:
            r = m
            m = l
            fm = fl
            step *= 2
            l = point - h * step
            xl[i] = l
            fl = function(xl, f)   
    return l,r

def goldenSection2(f, x, i, b=None, e=0.00001):
    a = x[i]
    xc = x[:]
    xd = x[:]
    if not b:
        a, b = unimodal2(a, f, x, i)
    k = 0.5 * (m.sqrt(5) - 1)
    c = b - k * (b - a)
    d = a + k * (b - a)
    xc[i] = c
    xd[i] = d
    fc = function(xc, f)
    fd = function(xd, f)
    while ((b - a) > e):
        if (fc < fd):
            b = d
            d = c
            c = b - k * (b - a)
            fd = fc
            xc[i] = c
            fc = function(xc, f)
        else:
            a = c
            c = d
            d = a + k * (b - a)
            fc = fd
            xd[i] = d
            fd = function(xd, f)
    return (b-a)/2

def coordinateSearch(x0, n, f, e=0.00001):
    global num_of_calls
    num_of_calls = 0
    x = x0
    while True:
        xs = x[:]
        for i in range(n):
            lambda1 = goldenSection2(f, x, i)
            x[i] = lambda1
        y = [x[i] - xs[i] for i in range(len(x))][0]
        if abs(y) <= e:
            return x, num_of_calls

def simplexNelderAndMead(x0, f, alpha=1, beta=0.5, gamma=2, sigma=0.00001, epsilon=0.05, displacement=1):
    global num_of_calls
    num_of_calls = 0
    X = [x0]
    for i in range(len(x0)):
        x0[i] += displacement
        X.append(x0.copy())
        x0[i] -= displacement
    while True:
        h = np.argmax([function(x, f) for x in X])
        Xh = X.pop(h)
        l = np.argmin([function(x, f) for x in X])
        Xc = []
        for i in range(len(X[0])):
            Xc.append(1/(len(X))*sum(x[i] for x in X))
        print("Centroid={} Funkcija cilja={}".format(round2(Xc),round(function(Xc,f))))
        #refleksija
        Xr = []
        for i in range(len(Xc)):
            Xr.append((1+alpha)*Xc[i] - alpha*Xh[i])
        fXr = function(Xr, f)
        fXl = function(X[l], f)
        fXh = function(Xh, f)
        if fXr < fXl:
            #ekspanzija
            Xe = []
            for i in range(len(Xc)):
                Xe.append((1-gamma)*Xc[i] + gamma*Xr[i])
            fXe = function(Xe, f)
            if fXe < fXl:
                X.insert(h,Xe)
            else:
                X.insert(h,Xr)
        else:
            if fXr > all([function(x, f) for x in X]):
                if fXr < fXh:
                    X.insert(h,Xr)
                    Xh = X[h]
                    fXh = function(Xh, f)
                #kontrakcija
                Xk = []
                for i in range(len(Xc)):
                    Xk.append((1-beta)*Xc[i] + beta*Xh[i])
                fXk = function(Xk, f)
                if fXk < fXh:
                    if len(X) == 3:
                        X.pop(h)
                    X.insert(h,Xk)
                else: #sažimanje
                    for i in range(len(X)):
                        for j in range(len(X[i])):
                            if X[l][j] > X[i][j]:
                                X[i][j] += sigma
                            else:
                                X[i][j] -= sigma
            else:
                X.insert(h,Xr)
        sum1 = 0
        fXc = function(Xc,f)
        for x in X:
            sum1 += len(X)*(function(x,f)-fXc)**2
        condition = m.sqrt(1/len(X)*sum1)
        if condition <= epsilon:
            h = np.argmax([function(x, f) for x in X])
            X.pop(h)
            x = []
            for i in range(len(X[0])):
                x.append((sum(x[i] for x in X)) / len(X))
            return x, num_of_calls

def search(xP, dx, f):
    x = xP[:]
    for i in range(len(x)):
        p = function(x, f)
        x[i] += dx
        n = function(x, f)
        if n > p:
            x[i] -= 2*dx
            n = function(x, f)
            if n > p:
                x[i] += dx
    return x

def algHookeJeeves(x0, f, dx=0.5, epsilon=0.00001):
    global num_of_calls
    num_of_calls = 0
    xP = xB = x0[:]
    while dx > epsilon:
        xN = search(xP, dx, f)
        fxN = function(xN, f)
        fxB = function(xB, f)
        if fxN < fxB:
            for i in range(len(xP)):
                xP[i] = 2*xN[i] - xB[i]
            xB = xN
        else:
            dx /= 2
            xP = xB
        print("XB={} fXB={} XP={} fXP={} XN={} fXN={}".format(round2(xB),round(fxB,2),round2(xP),round(function(xP,f),2),round2(xN),round(fxN,2)))
    return xB, num_of_calls


'''
zdt = int(input("Upisite broj zadatka (1-5):"))

if zdt==1:
    #zdt1
    print("\nZADATAK 1")
    print("--------------")

    p = readFile("zdt1.txt")
    min1, num = goldenSection(f=0, x0=p[1][:])
    print("\nZlatni rez={} Broj poziva={}\n".format(round(min1,2), num))
    min2, num = coordinateSearch(f=0, x0=p[1][:], n=1)
    print("\nPretraživanje po koordinatnim osima={} Broj poziva={}\n".format(round2(min2), num))
    min3, num = simplexNelderAndMead(f=0, x0=p[1][:])
    print("\nSimpleks postupak po Nelderu i Meadu={} Broj poziva={}\n".format(round2(min3), num))
    min4, num = algHookeJeeves(f=0, x0=p[1][:])
    print("\nHooke-Jeeves={} Broj poziva={}".format(round2(min4), num))

elif zdt==2:
    #zdt2
    print("\nZADATAK 2")
    print("--------------")
    fu = int(input("Odaberite funkciju (1-4):"))

    if fu==1:

        print("\nFunkcija 1\n")

        p = readFile("zdt2_1.txt")
        min1, num = simplexNelderAndMead(f=1, x0=p[1][:], epsilon=0.03)
        print("\nSimpleks postupak po Nelderu i Meadu={} Broj poziva={}\n".format(round2(min1), num))
        min2, num = algHookeJeeves(f=1, x0=p[1][:])
        print("\nHooke-Jeeves={} Broj poziva={}\n".format(round2(min2), num))
        min3, num = coordinateSearch(f=1,x0=p[1][:],n=len(p[1]))
        print("\nPretraživanje po koordinatnim osima={} Broj poziva={}\n".format(round2(min3), num))

    elif fu==2:

        print("\nFunkcija 2\n")

        p = readFile("zdt2_2.txt")
        min1, num = simplexNelderAndMead(f=2, x0=p[1][:], epsilon=0.5)
        print("\nSimpleks postupak po Nelderu i Meadu={} Broj poziva={}\n".format(round2(min1), num))
        min2, num = algHookeJeeves(f=2, x0=p[1][:])
        print("\nHooke-Jeeves={} Broj poziva={}\n".format(round2(min2), num))
        min3, num = coordinateSearch(f=2,x0=p[1][:],n=len(p[1]))
        print("\nPretraživanje po koordinatnim osima={} Broj poziva={}\n".format(round2(min3), num))

    elif fu==3:

        print("\nFunkcija 3\n")

        p = readFile("zdt2_3.txt")
        min1, num = simplexNelderAndMead(f=3, x0=p[1][:], epsilon=5)
        print("\nSimpleks postupak po Nelderu i Meadu={} Broj poziva={}\n".format(round2(min1), num))
        min2, num = algHookeJeeves(f=3, x0=p[1][:])
        print("\nHooke-Jeeves={} Broj poziva={}\n".format(round2(min2), num))
        min3, num = coordinateSearch(f=3,x0=p[1][:],n=len(p[1]))
        print("\nPretraživanje po koordinatnim osima={} Broj poziva={}\n".format(round2(min3), num))

    elif fu==4:
    
        print("\nFunkcija 4\n")

        p = readFile("zdt2_4.txt")
        min1, num = simplexNelderAndMead(f=4, x0=p[1][:])
        print("\nSimpleks postupak po Nelderu i Meadu={} Broj poziva={}\n".format(round2(min1), num))
        min2, num = algHookeJeeves(f=4, x0=p[1][:], epsilon=0.05)
        print("\nHooke-Jeeves={} Broj poziva={}\n".format(round2(min2), num))
        min3, num = coordinateSearch(f=4,x0=p[1][:],n=len(p[1]))
        print("\nPretraživanje po koordinatnim osima={} Broj poziva={}\n".format(round2(min3), num))


elif zdt==3:
    #zdt3
    print("\nZADATAK 3")
    print("--------------")

    p = readFile("zdt3.txt")
    min1, num = algHookeJeeves(f=4, x0=p[1][:], epsilon=0.05)
    print("\nHooke-Jeeves={} Broj poziva={}\n".format(round2(min1), num))
    min2, num = simplexNelderAndMead(f=4, x0=p[1][:])
    print("\nSimpleks postupak po Nelderu i Meadu={} Broj poziva={}\n".format(round2(min2), num))
        

elif zdt==4:
    #zdt4
    print("\nZADATAK 4")
    print("--------------")

    p = readFile("zdt4.txt")
    min1, num = simplexNelderAndMead(f=1, x0=p[1][:], displacement=p[3][0], epsilon=0.3)
    print("\nSimpleks postupak po Nelderu i Meadu={} Broj poziva={}\n".format(round2(min1), num))


elif zdt==5:
    #zdt5
    print("\nZADATAK 5")
    print("--------------")

    rand1 = round(random.uniform(-50, 50),2)
    rand2 = round(random.uniform(-50, 50),2)
    min1, num = coordinateSearch(f=6,x0=[rand1,rand2],n=2)
    fmin1 = function(min1,f=6)
    print("\nPretraživanje po koordinatnim osima={} Broj poziva={} Funkcija cilja={}\n".format(round2(min1), num, fmin1))     
'''