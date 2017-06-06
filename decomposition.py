import numpy as np
from numpy.linalg import norm
import heapq
"""
decompose X into two matrices S and VA
such that S has columns which are subset
of cols of X and V are any vectors.
A is a coefficient matrix.
rank of S is k1 and of V is k2
Also S is padded with zero columns to
match the dim of X
"""

def X_minus_S(X, selection):
    S = np.zeros(X.shape)
    S[:, selection] = X[:, selection]
    return (X - S)

def projection(x, U):
    return U.dot(U.T.dot(x))

def largest_projection(X, U, k = 1):
    r, c = X.shape
    Sel = []
    for i in range(c):
        p = projection(X[:,i], U)
        heapq.heappush(Sel, (-norm(p), i))
        Sel = Sel[:k]
    # Sel has (-norm of col, col index) pairs,
    # get the indices and return
    S = [t[1] for t in Sel]
    return S 

def largest_norm(X, k = 1):
    r, c = X.shape
    norm_list = []
    for i in range(c):
        n = norm(X[:, i])
        norm_list.append(-n)
    sorted_idx = np.argsort(np.array(norm_list))
    return sorted_idx[:k]


def decompose_S_VA(X, k1, k2, iters = 10):
    assert (k1 + k2) <= np.linalg.matrix_rank(X)

    def get_VA(R):
        U, s, V = np.linalg.svd(R)
        Uk2 = U[:, 0:k2]
        Vk2 = V[:, 0:k2]
        sk2 = s[0:k2]
        VA = Uk2.dot(np.diag(sk2).dot(Vk2.T))
        Uk1 = U[:, k2:(k2+k1)] 
        return VA, Uk1
    #def get_S(X, U):
    #    s_indices = largest_projection(X, U, k = k1)
    #    S = np.zeros(X.shape)
    #    S[:, s_indices] = X[:, s_indices]
    #    return S
    def get_S(X):
        s_indices = largest_norm(X, k1)
        S = np.zeros(X.shape)
        S[:, s_indices] = X[:, s_indices]
        return S

    R = X
    for i in range(iters):
        VA, U_rest_k1 = get_VA(R)
        S = get_S(X)
        R = X - S

    VA, _ = get_VA(R)
    return S, VA

if __name__ == "__main__":
    X = np.random.normal(size = (6,6))
    print(X.shape)
    print(np.linalg.matrix_rank(X))
    S, VA = decompose_S_VA(X, 2, 2)
    print(X)
    print(S)
    print(VA)
    print(norm(X), norm(X - S), norm(X - VA), norm(X - S - VA))
