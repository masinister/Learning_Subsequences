import numpy as np

def is_subseq(x, y):
    it = iter(y)
    return all(c in it for c in x)

def subseq_freq(S, k):
    res = np.zeros((2**k,))
    for x in S:
        
