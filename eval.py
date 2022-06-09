from statistics import mean
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import os
import datetime

data_dir = "./output/20220609_030149/S"
mask_dir = "./data/Masks"


def main():

    # 加载数据
    datas = {}
    names = Path(data_dir)
    for name in names.iterdir():
        if str(name).endswith("jpg"):
            img = np.array(Image.open(name).resize((180, 120))) / 255
            datas.update({str(name)[-8:-4]: img})

    masks = {}
    names = Path(mask_dir)
    for name in names.iterdir():
        if str(name).endswith("bmp"):
            img = np.array(Image.open(name).resize((180, 120)))
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
            img = img / 255
            masks.update({str(name)[-8:-4]: img})

    mses = []
    for key in datas.keys():
        data = datas[key]
        mask = masks[key]
        mse = np.mean((data - mask) ** 2)
        mses.append(mse)
    print(mean(mses))


if __name__ == "__main__":
    main()
