import numpy as np

def bin_to_int(a):
    res = 0
    for b in a:
        res = (res << 1) | b
    return res

def sample_product_dist(P):
    return [np.random.choice([1, 0], p = [q, 1-q]) for q in P]

if __name__ == '__main__':
    x = sample_product_dist([0.5, 0.1, 1, 0 ,1])
    print(bin_to_int(x))
