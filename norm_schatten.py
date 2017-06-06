# -*- coding: utf-8 -*-
import numpy as np
def norm_schatten_p(X, p):
    """
    return ∑ (σ_i)^(p)
    """
    _, sigma, _ = np.linalg.svd(X)
    if p == float('inf'):
        return np.max(sigma)
    else:
        return np.sum(sigma**p)

def norm_schatten(X, p):
    """
    return (∑ (σ_i)^(p))^(1/p)
    """
    return norm_schatten_p(X,p)**(1.0/p)
