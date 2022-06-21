import numpy as np

def S(n,k):
    if n == k:
        return np.identity(2**n)
    elif k == 0:
        return np.ones((1,2**n))
    else:
        b = S(n-1,k-1)
        U,L = np.split(S(n-1,k), 2, axis = 0)
        return np.block([[b,U],[L,b]])
