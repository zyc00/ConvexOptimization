# Reference:  Cand`es, E. J., Li, X., Ma, Y., and Wright, J. (2011).
# Robust principal componentanalysis?.  Journal of the ACM, 58(3), 1-37.  (Mathematical theory

import numpy as np


def gen_matrix():
    # L是一个低秩矩阵，S是一个稀疏矩阵，这里设置低秩和稀疏的要求
    p = 0.8
    S = np.random.uniform(0, 1, (15, 20))

    S = S * (np.random.uniform(0, 1, (15, 20)) >= p)
    L = np.random.uniform(0, 1, (15, 20))
    return S, L, S + L


def IALM(M: np.array, iter: int):
    mu = 0.25 / np.abs(M).mean()
    lamb = 1 / np.sqrt(max(M.shape[0], M.shape[1]))

    S = np.zeros(M.shape)
    Y = np.zeros(M.shape)
    L = np.zeros(M.shape)

    def computeS(X, t):
        y = np.maximum(X - t, 0)
        y = y + np.minimum(X + t, 0)
        return y

    def computeD(X, t):
        U, Sigma, VH = np.linalg.svd(X, full_matrices=False)
        rank = (Sigma > t).sum()
        Sigma = np.diag(Sigma[0:rank] - t)
        return U[:, 0:rank] @ Sigma @ VH[0:rank, :]

    for i in range(iter):
        if i % 20 == 0:
            print(f"迭代到了第{i}次")
        L = computeD(M - S + mu ** (-1) * Y, 1 / mu)
        S = computeS(M - L + mu ** (-1) * Y, lamb / mu)
        Y = Y + mu * (M - L - S)
        # print(S[-1])

    return L, S
