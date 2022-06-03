from PIL import Image
import cv2
import numpy as np
from pathlib import Path

from algorithms.APGM import APGM
from algorithms.ADMM import ADMM

data_dir = "./data"

def main(alg):
    # 加载数据
    data = []
    names = Path(data_dir)
    for name in names.iterdir():
        if str(name).endswith('jpeg'):
            data.append(np.array(Image.open(name).resize((560,420))))
    data = np.array(data)
    print("图片的维度是：", data.shape[1:3])

    # 灰度化和归一化
    data = np.mean(data, axis=3)
    data = data / 255

    # 按行拼接图片
    cat_data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))

    # 进入算法进行计算
    L, S = alg(cat_data, 1000)
    
    # 可视化生成的结果
    L_data = L.reshape((data.shape[0], data.shape[1], data.shape[2]))
    S_data = S.reshape((data.shape[0], data.shape[1], data.shape[2]))
    for i, (L, S) in enumerate(zip(L_data, S_data)):
        cv2.imwrite(f'output/{i}_l.jpg', L * 255)
        cv2.imwrite(f'output/{i}_s.jpg', S * 255)
    

if __name__ == '__main__':
    main()