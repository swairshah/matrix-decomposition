import numpy as np
from numpy.linalg import matrix_rank as rank
from numpy.linalg import norm, svd
from column_subset import subtract
from brute import reduce_ksvd
import operator

def VA(X, k):
    U, s, V = svd(X, full_matrices = False)
    Uk = U[:,0:k]
    Vk = V[0:k,:]
    sk = s[0:k]

    V = Uk
    A = np.diag(sk).dot(Vk)
    return V, A

def outlier_pca(X, k1, k2, iters = 5):
    """
    C_s contains col indices included in S
    C_v contains rest 
    """
    m, n = X.shape
    C = np.arange(n) # indices of cols of X
    C_s = np.random.choice(n, size=k1, replace=False)
    #print(C_s)
    C_v = np.setdiff1d(C, C_s)

    def run_iter(C_s, C_v):
        err_dict = {}
        V, A = VA(subtract(X, C_s), k2)
        for idx, col in enumerate(C):
            err = norm(X[:, col] - V.dot(A[:,idx]))
            err_dict[col] = (-1*err)
	
	# the following returns a list of pairs sorted 
        # by values of dict which is neg of err
	# keys are the column indicies
        #print(err_dict.items())
        sorted_cols = sorted(err_dict.items(), key=operator.itemgetter(1))
        #print(sorted_cols)
    
        # now we get first k1 col indices outof it
        C_s = [i[0] for i in sorted_cols[0:k1]]

        C_v = np.setdiff1d(C, C_s)
        return C_s, C_v

    for i in range(iters):
        #print(C_s)
        C_s, C_v = run_iter(C_s, C_v)
        X_ = subtract(X, C_s)
        R = reduce_ksvd(X_, k2)
        error = norm(R)
        #print(error)

    print(error)
    return C_s

def avg_outlier_pca(X, k1, k2, iters = 5):
    np.set_printoptions(suppress = True, precision = True)
    """
    C_s contains col indices included in S
    C_v contains rest 
    we maintain a running avg of error x_i - V*a_i 
    for each column (in C_v).
    """
    m, n = X.shape
    C = np.arange(n) # indices of cols of X

    score_dict = {i: [0] for i in C}
    avg_score_dict = {i: np.mean(score_dict[i]) for i in score_dict}
    C_s = np.random.choice(n, size=k1, replace=False)
    #print(C_s)
    C_v = np.setdiff1d(C, C_s)

    iter = 1.0
    def run_iter(C_s, C_v):
        err_dict = {}
        V, A = VA(subtract(X, C_s), k2)
        for idx, col in enumerate(C):
            err = norm(X[:, col] - V.dot(A[:,idx]))
            score_dict[col].append(err)
	
        avg_score_dict = {i: np.mean(score_dict[i]) for i in score_dict}
	# the following returns a list of pairs sorted 
        # by values of dict which is running avg or err
	# keys are the column indicies
        #print(avg_score_dict)
        sorted_cols = sorted(avg_score_dict.items(), key=operator.itemgetter(1))
        # now we get last k1 col indices outof it
        C_s = [i[0] for i in sorted_cols[-k1:]]
        C_v = np.setdiff1d(C, C_s)
        return C_s, C_v

    for i in range(iters):
        #print(C_s)
        C_s, C_v = run_iter(C_s, C_v)
        X_ = subtract(X, C_s)
        R = reduce_ksvd(X_, k2)
        error = norm(R)
        iter += 1
        #print(error)

    print(error)
    return C_s

if __name__ == "__main__":
    np.set_printoptions(suppress = True, precision = True)
    X = np.random.uniform(size = (10,20), low = 0, high = 100)
    print(outlier_pca(X, 2, 4, iters = 30))
    print()
    print(avg_outlier_pca(X, 2, 4, iters = 500))
