import numpy as np
import time


def gen_fake_data(m, n, r, p):
    L = np.random.randn(m, r) @ np.random.randn(r, n)  # Low rank
    S = np.random.randn(m, n) * (np.random.uniform(0, 1, (m, n)) < p)  # sparse
    return L, S, L + S


def main(alg):
    m = n = 100
    r = 10
    p = 0.3

    print("生成矩阵...")
    L, S, M = gen_fake_data(m, n, r, p)
    start_time = time.time()
    pred_L, pred_S, iters = alg(M, 1000)
    end_time = time.time()
    print(
        f"算法运行时长：%d分%.2f秒, 迭代次数：{iters}次"
        % ((end_time - start_time) // 60, ((end_time - start_time) % 60))
    )
    print("每次迭代用时：%.2f秒" % ((end_time - start_time) / (iters + 1)))
    print("fake data迭代结束")
