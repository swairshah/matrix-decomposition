import numpy as np
import numpy.linalg as lin
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
    Vk = V[:,0:k]
    sk = s[0:k]
    Xk = Uk.dot(np.diag(sk).dot(Vk.T))
    return (X - Xk), Uk

def reduce_ksvd2(X, k):
    U, s, V = np.linalg.svd(X)
    Uk = U[:,0:k]
    proj = Uk.dot(np.linalg.pinv(Uk))
    R = X - proj.dot(X)
    return (X - R)

def VS(X, k1, k2, p):
    Xk, Uk = reduce_ksvd(X, k2)
    selection, err = select(Xk, k1, p)
    Xs = X[:,selection]
    SS = np.hstack((Xs, Uk))
    err = error_vec(X, SS, p)
    return selection, err

def select(X, k, p):
    m, n = X.shape
    min_selection = None
    min_err = float('inf')
    for sub in itertools.combinations(range(n), k):
        S = list(sub)
        err = error(X, S, p)
        if err < min_err:
            min_err = err
            min_selection = S
    return min_selection, min_err

if __name__ == "__main__":
    #X = np.genfromtxt('~/ml/data/libras.data', delimiter=' ')
    pass
