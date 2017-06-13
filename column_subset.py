import numpy as np

def subtract(X, S):
    Xs = X.copy()
    Xs[:,S] = 0
    return Xs

def sort_cols_by_norm(X):
    r, c = X.shape
    norm_list = []
    for i in range(c):
        n = np.linalg.norm(X[:, i])
        norm_list.append(-n)
    sorted_idx = np.argsort(np.array(norm_list))
    return sorted_idx 

def maxnorm_subset(X, k = 1):
    sorted_idx = sort_cols_by_norm(X)
    return sorted_idx[:k]
