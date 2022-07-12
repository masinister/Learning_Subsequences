import networkx as nx
from scipy import sparse
from block_matrix import S, shattered
import matplotlib.pyplot as plt
import string

if __name__ == '__main__':

    n = 4
    k = 3
    draw_bipartite = True

    M = sparse.csr_matrix(S(n,k))
    # M = sparse.csr_matrix(shattered(k))
    G = nx.bipartite.from_biadjacency_matrix(M)

    if draw_bipartite:
        top = nx.bipartite.sets(G)[0]
        pos = nx.bipartite_layout(G, top)
        nx.draw(G, pos, with_labels = True)
    else:
        color_map =  ['green']*2**k + ['gray']*2**n
        nx.draw(G, node_color = color_map)
        # nx.draw(G)

    plt.show()
