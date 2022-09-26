import numpy as np
from math import comb

def S(n,k):
    if n < k:
        return np.zeros((2**k,2**n))
    elif n == k:
        return np.identity(2**n)
    elif k == 0:
        return np.ones((1,2**n))
    else:
        b = S(n-1,k-1)
        U,L = np.split(S(n-1,k), 2, axis = 0)
        return np.block([[b,U],[L,b]])

def E(n,k):
    return np.sum(S(n,k)[0])

def shattered(k):
    if k == 1:
        return np.array([[0],[1]])
    else:
        b = shattered(k - 1)
        zeros = np.zeros((2**(k-1),1))
        ones = np.ones((2**(k-1),1))
        return np.block([[zeros, b],[ones,b]])

def IND(n):
    return shattered(n).transpose()

def DISJ(n):
    if n == 0:
        return np.identity(1)
    else:
        b = DISJ(n-1)
        return np.block([[b,b],[b,np.zeros((2**(n-1),2**(n-1)))]])

if __name__ == '__main__':
    print(DISJ(2))
    print(DISJ(3))
    print(DISJ(4))
