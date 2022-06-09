# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 11:48:05 2022

@author: 胡亦可
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 20 18:56:59 2022

@author: 胡亦可


EALM
"""

import numpy as np
import random


def Sfunc(tau, matrix):  #######shrink operation
    shape = np.shape(matrix)
    zeros = np.zeros(shape)
    difference = abs(matrix) - np.ones(shape) * tau
    y = np.where(matrix > 0, 1, -1)
    return y * np.maximum(difference, zeros)


def Dfunc(tau, matrix):  ####singular value thresholding operator
    U, S, V = np.linalg.svd(matrix, full_matrices=False)
    S_star = Sfunc(tau, np.diag(S))
    # print(U,S,V,S_star)
    return np.dot(U, np.dot(S_star, V))


def Fnorm(matrix):
    return np.sqrt(sum(sum(matrix**2)))


def init_Y(Y, lambbd):
    Y0 = np.where(Y > 0, 1, -1)
    l2 = np.sqrt(sum(sum(Y**2)))
    J = max(l2, np.max(Y) / lambbd)
    return Y0 / J


def update_mu(M, mu0, rau, epsino, Fn):
    if mu0 * Fn / Fnorm(M) < epsino:
        return rau * mu0
    else:
        return mu0


def EALM(M, max_iter):
    n1, n2 = M.shape
    delta = 10 ** (-7)
    lambd = 1 / (pow(max(n1, n2), 0.5))
    S1 = np.zeros((n1, n2))
    Y = init_Y(M, lambd)
    L = np.zeros((n1, n2))
    L_ = np.ones((n1, n2))
    F_M = Fnorm(M)
    mu = 0.5 / np.linalg.norm(np.sign(M), 2)
    rau = 6
    epsino = 10 ** (-6)
    for i in range(max_iter):
        if i % 20 == 0:
            print(f"迭代到了第{i}次")
        S = S1
        while Fnorm(L_ - L) > delta:
            # print(Fnorm(L_-L))
            L_ = L
            L = Dfunc(1 / mu, M - S1 + Y / mu)
            S1 = Sfunc(lambd / mu, M - L + Y / mu)
        L_ = np.ones((n1, n2))
        Y = Y + (M - L - S1) * mu
        Fn = Fnorm(S - S1)
        mu = update_mu(M, mu, rau, epsino, Fn)

        F = Fnorm(M - L - S1)

        if np.linalg.norm(M - L - S1, "fro") <= 1e-5 * np.linalg.norm(M, "fro"):
            break
    return L, S1, i
