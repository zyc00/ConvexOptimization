import numpy as np

def gen_fake_data(m, n, r, p):
    L = np.random.randn(m, r) @ np.random.randn(r, n)   # Low rank
    S = np.random.randn(m, n) * (np.random.uniform(0, 1, (m, n)) < p)   # sparse
    return L, S, L + S

def main(alg):
    m = 20
    n = 100
    r = 5
    p = 0.3

    print("生成矩阵...")
    L, S, M = gen_fake_data(m, n, r, p)
    pred_L, pred_S = alg(M, 1000)   
    print("fake data迭代结束")
