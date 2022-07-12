from block_matrix import S
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from distributions import bin_to_int

def index_to_point(i, m):
    str = format(i, "b")
    res = deque([0]*m, maxlen = m)
    for b in str:
        res.append(int(b))
    return np.array(res)


if __name__ == '__main__':
    n = 4
    k = 3
    M = S(n,k)
    Z = S(n-1,k-1)
    Y = S(n-1, k)
    rowsum = np.sum(M[0])
    print("Row Sum: {}".format(rowsum))
    print(np.sum(Z[0]), np.sum(Y[0]))

    X = list(range(2**n))
    L = []
    for y in range(2**k):
        L.append(np.dot(X, M[y,:]))

    print(np.array(L) / 2**(2*n-1))
    print(list(L[x+1]-L[x] for x in range(2**k-1)))

    C = []
    for y in range(2**k):
            points = []
            for x in range(2**n):
                if M[y,x] == 1:
                    points.append(index_to_point(x, n))
            sum = np.sum(points, axis = 0)
            plt.plot(sum, label = format(y, "b"), linewidth = 3)
            print(list(np.around(sum,2)))
            C += [list(np.around(np.array(sum > 1/2, dtype = int), 2))]
    print(*C, sep = '\n')
    plt.legend()
    plt.show()
