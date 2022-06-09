from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import os
import datetime
import time

data_dir = "./data/RawImages"


def main(alg):
    # 每次实验新建文件夹
    if not os.path.exists("output"):
        os.mkdir("output")
    time_tag = datetime.datetime.now().__format__("%Y%m%d_%H%M%S")
    os.mkdir(f"output/{time_tag}")
    os.mkdir(f"output/{time_tag}/L")
    os.mkdir(f"output/{time_tag}/S")

    # 加载数据
    data = []
    ids = []
    names = Path(data_dir)
    for name in names.iterdir():
        if str(name).endswith("bmp"):
            data.append(np.array(Image.open(name).resize((180, 120))))
            ids.append(str(name)[-8:-4])
    data = np.array(data)
    print("图片的维度是：", data.shape[1:3])

    # 灰度化和归一化
    data = np.mean(data, axis=3)
    data = data / 255

    # 按行拼接图片
    cat_data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))

    # 进入算法进行计算
    start_time = time.time()
    L, S, iters = alg(cat_data, 1000)
    end_time = time.time()
    print(
        f"算法运行时长：%d分%.2f秒, 迭代次数：{iters}次"
        % ((end_time - start_time) // 60, ((end_time - start_time) % 60))
    )
    print("每次迭代用时：%.2f秒" % ((end_time - start_time) / (iters + 1)))

    # 可视化生成的结果
    L_data = L.reshape((data.shape[0], data.shape[1], data.shape[2]))
    S_data = S.reshape((data.shape[0], data.shape[1], data.shape[2]))
    for _, (L, S, I) in enumerate(zip(L_data, S_data, ids)):
        cv2.imwrite(f"output/{time_tag}/L/{I}.jpg", L * 255)
        cv2.imwrite(f"output/{time_tag}/S/{I}.jpg", (S > 0.05) * 255)


if __name__ == "__main__":
    main()
