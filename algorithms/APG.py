# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 16:03:02 2022

@author: 胡亦可
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 21:21:43 2022

@author: 胡亦可
"""

import numpy as np


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
    return sum(sum(matrix**2))


def APG(M, max_iter):
    n1, n2 = M.shape
    delta = 10 ** (-7)
    mu_0 = 100
    lambd = 1 / (pow(max(n1, n2), 0.5))
    eta = 0.5
    n = 0
    S_0 = np.zeros((n1, n2))
    S_1 = np.zeros((n1, n2))

    L_0 = np.zeros((n1, n2))
    L_1 = np.zeros((n1, n2))
    t_0 = 1
    t_1 = 1
    mu_bar = delta * mu_0

    for i in range(max_iter):
        if i % 20 == 0:
            print(f"迭代到了第{i}次")
        YL = L_1 + (t_0 - 1) / t_1 * (L_1 - L_0)
        YS = S_1 + (t_0 - 1) / t_1 * (S_1 - S_0)
        GL = YL - 0.5 * (YL + YS - M)
        L_0 = L_1
        L_1 = Dfunc(mu_0 / 2, GL)
        GS = YS - 0.5 * (YL + YS - M)
        S_0 = S_1
        S_1 = Sfunc(lambd * mu_0 * 0.5, GS)
        t_0 = t_1
        t_1 = (1 + pow(4 * t_0**2 + 1, 0.5)) / 2
        mu_0 = max(eta * mu_0, mu_bar)
        if np.linalg.norm(M - L_1 - S_1, "fro") <= 1e-5 * np.linalg.norm(M, "fro"):
            break

    return L_1, S_1, i
