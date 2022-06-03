# Implementation of ADMM for RPCA
# Reference: https://jeremykarnowski.wordpress.com/2015/08/31/robust-principal-component-analysis-via-admm-in-python/

import numpy as np

def compute_S(x, b, l):
    return (1 / (l + 1)) * (x - b)

def compute_L(x, b, l, g, pl, pm):
    return pm(x - b, l * g, pl)

def prox_l1(v,lambdat):
    return np.maximum(0, v - lambdat) - np.maximum(0, -v - lambdat)

def prox_matrix(v, lambdat,prox_f):
    U,S,V = np.linalg.svd(v, full_matrices=False)
    S = S.reshape((len(S),1))
    pf = np.diagflat(prox_f(S,lambdat))
    return U @ pf @ V.conj()

def ADMM(M, max_iter):
    m, n = M.shape

    n_2 = np.linalg.norm(M, 2) * 0.15

    lambdap = 1.0

    S = L = np.zeros((m, n))
    T = np.zeros((m, 3 * n))
    U = np.zeros((m, n))

    for i in range(max_iter):
        if i % 20 == 0:
            print(f"迭代到了第{i}次")
        B = (S + L) /  2 - M / 2 + U

        S = compute_S(S, B, lambdap)
        L = compute_L(L, B, lambdap, n_2, prox_l1, prox_matrix)
        U = B
    
    return L, S