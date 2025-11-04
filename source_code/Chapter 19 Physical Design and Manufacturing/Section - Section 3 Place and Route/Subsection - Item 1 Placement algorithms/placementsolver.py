import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# nodes: N, nets: list of (i,j,weight)
def analytic_place(N, nets, anchors):
    # build Laplacian L and RHS b
    rows, cols, vals = [], [], []
    b = np.zeros(N)
    for i, j, w in nets:
        rows += [i, j, i, j]; cols += [i, j, j, i]
        vals += [ w, w, -w, -w ]             # Laplacian contributions
    L = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    # incorporate anchors as large-diagonal penalties
    for idx, xpos, strength in anchors:      # idx node pinned to xpos
        L[idx, idx] += strength
        b[idx] += strength * xpos
    x = spsolve(L, b)                        # solve sparse linear system
    return x                                 # x positions (1D)
# Example usage: feed nets and anchors for each placement region.