import numpy as np

epsilon = 0.00000001 

class Matrix:

    def readMatrix(self, file):
        rows = []
        with open(file, 'r') as f:
            for line in f:
                rows.append([float(x) for x in line.strip().split()])
        return np.array(rows)

    def __init__(self, file=None, m=None):
        if file is not None:
            self.matrix = self.readMatrix(file)
        elif m is not None:
            self.matrix = m
        else: self.matrix = []

    def getDimOfMatrix(self):
        return len(self.matrix), len(self.matrix[0])

    def print(self):
        print(self.matrix)

    def printTxt(self):
        with open("matrix.txt", 'a') as f:
            for row in self.matrix:
                for element in row:
                    f.write(str(element) + " ")
                f.write("\n")
            f.write("\n")
    
    def copyMatrix(self, arr):
        if type(arr) is Matrix:
            self.matrix = np.copy(arr.matrix)
        else:
            self.matrix = np.copy(arr)

    def getElement(self, i, j):
        return self.matrix[i][j]

    def setElement(self, i, j, value):
        self.matrix[i][j] = value
    
    def add(self, m):
        try:
            if type(m) is Matrix:
                return np.add(self.matrix, m.matrix)
            return np.add(self.matrix, m)
        except:
            print('Matrix dimensions are mismatched.')
            raise SystemExit()

    def substract(self, m):
        try:
            if type(m) is Matrix:
                return np.subtract(self.matrix, m.matrix)
            return np.subtract(self.matrix, m)
        except:
            print('Matrix dimensions are mismatched.')
            raise SystemExit()

    def multiply(self, m):
        try:
            if type(m) is Matrix:
                return np.matmul(self.matrix, m.matrix)
            return np.matmul(self.matrix, m)
        except:
            print('Matrix dimensions are mismatched.')
            raise SystemExit()

    def transpose(self):
        return np.transpose(self.matrix)
    
    def mulScalar(self, scalar):
        return self.matrix * scalar
    

def forwardSubstitution(matrix, vector):
    y = vector[:].flatten().tolist()
    for i in range(len(y)-1):
        for j in range(i+1, len(y)):
            y[j] -= (matrix[j][i] * y[i])
    return y
    
def backwardSubstitution(matrix, vector):
    try:
        n = len(vector)
        x = np.zeros(n)
        for i in range(n-1,-1,-1):
            tmp = vector[i]
            for j in range(i+1,n):
                tmp -= (matrix[i][j] * x[j])
            x[i] = tmp / matrix[i][i]
            if abs(x[i]) < epsilon:
                x[i] = 0
    except:
        print("No inverse. Singular matrix.")
        return
    return x

def choosePivot(matrix, column, row):
    values = dict()
    for i in range(row, len(matrix)):
        if abs(matrix[i][column]) not in values.keys():
            values[abs(matrix[i][column])] = i
    return (max(zip(values.keys(), values.values())))

def swapRows(matrix, i, j):
    matrix.matrix[[i,j]] = matrix.matrix[[j,i]]

def LU(matrix):
    if len(matrix.matrix) != len(matrix.matrix[0]):
        print('LU decompozition not possible. Matrix is not square.')
        return
    try:
        lu = Matrix()
        lu.copyMatrix(matrix.matrix)
        for i in range(len(lu.matrix)-1):
            for j in range(i+1, len(lu.matrix)):
                lu.matrix[j][i] /= lu.matrix[i][i]
                for k in range(i+1,len(lu.matrix)):
                    lu.matrix[j][k] -= (lu.matrix[j][i] * lu.matrix[i][k])
    except:
        print('LU decompozition not possible. Pivot element is 0.')
        return
    return lu

def LUP(matrix):
    if len(matrix.matrix) != len(matrix.matrix[0]):
        print('LUP decompozition not possible. Matrix is not square.')
        return
    try:
        lu = Matrix()
        lu.copyMatrix(matrix.matrix)
        p = np.identity(len(matrix.matrix))
        P = Matrix(None, p)
        swap_count = 0
        for i in range(len(lu.matrix)):
            pivot = choosePivot(lu.matrix, i, i)
            if abs(pivot[0]) < epsilon:
                print('LUP decompozition not possible. Matrix is singular.')
                return
            if pivot[1] != i:
                swapRows(lu, pivot[1], i)
                swapRows(P, pivot[1], i)
                swap_count += 1
            for j in range(i+1, len(lu.matrix)):
                lu.matrix[j][i] /= lu.matrix[i][i]
                for k in range(i+1, len(lu.matrix)):
                    lu.matrix[j][k] -= (lu.matrix[j][i] * lu.matrix[i][k])
    except:
        print('LUP decompozition not possible.')
        return
    return lu, P, swap_count

def splitLU(lu):
    L = np.tril(lu.matrix)
    np.fill_diagonal(L, 1)
    U = np.triu(lu.matrix)
    return L, U

def inverse(lu, p):
    try:
        P = p.matrix
        L, U = splitLU(lu)
        Inv = np.identity(len(lu.matrix))
        for i in range(len(lu.matrix)):
            y = forwardSubstitution(L, np.matmul(P,Inv[i]))
            x = backwardSubstitution(U, y)
            Inv[i] = np.array(x)   
        return Matrix(None, Inv.transpose())
    except:
        print("No inverse. Singular matrix.")
        return

def determinant(U, swap_count):
    product = 1
    for i in range(len(U)):
        product *= U[i][i]
    det = (-1)**swap_count * product
    return(det)

'''
#zdt2 
print("\nZADATAK 2")
A = Matrix("A.txt")
b = Matrix("B.txt")
lu1, P1, swap_count1 = LUP(A)
L, U = splitLU(lu1)
y = forwardSubstitution(L, P1.multiply(b))
x = backwardSubstitution(U, y)
print(x)

#zdt3 
print("\nZADATAK 3")
C = Matrix("C.txt")
vector = Matrix("vector.txt")
lu2 = LU(C)
lu2.print()
L2, U2 = splitLU(lu2)
y2 = forwardSubstitution(L2, vector.matrix)
x2 = backwardSubstitution(U2, y2)

#zdt4 
print("\nZADATAK 4")
D = Matrix("D.txt")
E = Matrix("E.txt")
lu3 = LU(D)
L3, U3 = splitLU(lu3)
lu4, P2, swap_count = LUP(D)
L4, U4 = splitLU(lu4)
y3 = forwardSubstitution(L3, E.matrix)
x3 = backwardSubstitution(U3, y3)
print(x3)
y4 = forwardSubstitution(L4, np.matmul(P2.matrix, E.matrix))
x4 = backwardSubstitution(U4, y4)
print(x4)

#zdt5 
print("\nZADATAK 5")
F = Matrix("F.txt")
G = Matrix("G.txt")
lu6, P3, swap_count = LUP(F)
L6, U6 = splitLU(lu6)
y6 = forwardSubstitution(L6, np.matmul(P3.matrix, G.matrix))
x6 = backwardSubstitution(U6, y6)
print(x6)

#zdt6 
print("\nZADATAK 6")
H = Matrix("H.txt")
I = Matrix("I.txt")
lu5 = LU(H)
L5, U5 = splitLU(lu5)
y5 = forwardSubstitution(L5, I.matrix)
x5 = backwardSubstitution(U5, y5)
print(x5)

#zdt7 
print("\nZADATAK 7")
inv = inverse(lu2, None)

#zdt8 
print("\nZADATAK 8")
J = Matrix("J.txt")
lu7, P4, swap_count = LUP(J)
inv = inverse(lu7, P4)
inv.print()

#zdt9 
print("\nZADATAK 9")
L7, U7 = splitLU(lu7)
det = determinant(U7, swap_count)
print(det)

#zdt10 
print("\nZADATAK 10")
det = determinant(U, swap_count1)
print(det)
'''