# Implementation of ADMM for RPCA
# Reference: https://jeremykarnowski.wordpress.com/2015/08/31/robust-principal-component-analysis-via-admm-in-python/

import numpy as np


def compute_T(x, b, l):
    return (1 / (l + 1)) * (x - b)


def compute_L(x, b, l, g, pl, pm):
    return pm(x - b, l * g, pl)


def compute_S(x, b, l, g, pl):
    return pl(x - b, l * g)


def prox_l1(v, lambdat):
    return np.maximum(0, v - lambdat) - np.maximum(0, -v - lambdat)


def prox_matrix(v, lambdat, prox_f):
    U, S, V = np.linalg.svd(v, full_matrices=False)
    S = S.reshape((len(S), 1))
    pf = np.diagflat(prox_f(S, lambdat))
    return U @ pf @ (V.conj())


def ADMM(M, max_iter):
    m, n = M.shape

    n_inf = np.linalg.norm(np.hstack(M).T, np.inf) * 0.15
    n_2 = np.linalg.norm(M, 2) * 0.15

    lambdap = 1.0

    S = L = T = np.zeros((m, n))
    U = np.zeros((m, n))
    z = np.zeros((m, 2 * n))

    for i in range(max_iter):
        if i % 20 == 0:
            print(f"迭代到了第{i}次")
        B = (S + L + T) / 3 - M / 3 + U

        T = compute_T(S, B, lambdap)
        S = compute_S(T, B, lambdap, n_inf, prox_l1)
        L = compute_L(L, B, lambdap, n_2, prox_l1, prox_matrix)
        U = B

        x = np.hstack([S, L, T])
        z_old = z
        z = x + np.tile(-(S + L + T) / 3 + M / 3, (1, 3))
        if np.linalg.norm(x - z, "fro") < (
            np.sqrt(m * n * 3) * 1e-4
            + 1e-2 * np.maximum(np.linalg.norm(x, "fro"), np.linalg.norm(-z, "fro"))
        ) and np.linalg.norm(-(z - z_old), "fro") < (
            np.sqrt(m * n * 3) * 1e-4 + 1e-2 * np.sqrt(3) * np.linalg.norm(U, "fro")
        ):
            break

    return L, S, i
