import numpy as np
from numpy.linalg import svd, pinv, norm
import itertools

def error(X, S, p):
    Xs = X[:,S]
    proj = Xs.dot(np.linalg.pinv(Xs))
    R = X - proj.dot(X)
    return norm_schatten_p(R, p)

def error_vec(X, S, p):
    proj = S.dot(np.linalg.pinv(S))
    R = X - proj.dot(X)
    return norm_schatten_p(R, p)

def residual(X, S, p):
    Xs = X[:,S]
    proj = Xs.dot(np.linalg.pinv(Xs))
    R = X - proj.dot(X)
    return R, norm_schatten_p(R, p)

def reduce_ksvd(X, k):
    U, s, V = np.linalg.svd(X)
    Uk = U[:,0:k]
    Vk = V[0:k,:]
    sk = s[0:k]
    Xk = Uk.dot(np.diag(sk).dot(Vk))
    return (X - Xk)

def reduce_ksvd2(X, k):
    U, s, V = np.linalg.svd(X)
    Uk = U[:,0:k]
    proj = Uk.dot(np.linalg.pinv(Uk))
    return (X - proj.dot(X))

#def VS(X, k1, k2, p):
#    Xk, Uk = reduce_ksvd(X, k2)
#    selection, err = select(Xk, k1, p)
#    Xs = X[:,selection]
#    SS = np.hstack((Xs, Uk))
#    err = error_vec(X, SS, p)
#    return selection, err

#def select(X, k, p):
#    m, n = X.shape
#    min_selection = None
#    min_err = float('inf')
#    for sub in itertools.combinations(range(n), k):
#        S = list(sub)
#        err = error(X, S, p)
#        if err < min_err:
#            min_err = err
#            min_selection = S
#    return min_selection, min_err

def select(X, k1, k2):
    m, n = X.shape
    min_selection = None
    min_err = float('inf')
    for sub in itertools.combinations(range(n), k1):
        S = list(sub)
        #print(S)
        #print(fg(X, S, k1, k2))
        X_Xs = X.copy()
        X_Xs[:, S] = 0
        #print(S, X_Xs)
        R = reduce_ksvd(X_Xs, k2)
        err = norm(R)**2
        #print(S, err)
        if err < min_err:
            min_err = err
            min_selection = S
    return min_selection, min_err

def fg(X, S, k1, k2):
    X_Xs = X.copy()
    X_Xs[:, S] = 0
    ki = len(S)
    Rf = reduce_ksvd(X_Xs, k2 + k1 - ki)
    Rg = reduce_ksvd(X_Xs, k2)
    return norm(Rf)**2, norm(Rg)**2

if __name__ == "__main__":
    X = np.genfromtxt('data/spectf.data')
    S, err = select(X, 3, 10)
    print("Selection :",S)
    print("Error:",err)

