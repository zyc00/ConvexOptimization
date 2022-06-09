from statistics import mean
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import os
import datetime

data_dir = "./images"
mask_dir = "./data/Masks"
L_dir = "./output/20220609_022015/L"
S_dir = "./output/20220609_022015/S"


def main():

    # 加载数据
    datas = {}
    names = Path(data_dir)
    for name in names.iterdir():
        if str(name).endswith("png"):
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

    Ls = {}
    names = Path(L_dir)
    for name in names.iterdir():
        if str(name).endswith("jpg"):
            img = np.array(Image.open(name).resize((180, 120))) / 255
            Ls.update({str(name)[-8:-4]: img})

    Ss = {}
    names = Path(S_dir)
    for name in names.iterdir():
        if str(name).endswith("jpg"):
            img = np.array(Image.open(name).resize((180, 120))) / 255
            Ss.update({str(name)[-8:-4]: img})

    image = np.pad(datas["0448"], ((5, 5), (5, 5), (0, 0)), constant_values=1)
    mask = np.pad(
        masks["0448"][..., None].repeat(3, -1),
        ((5, 5), (5, 5), (0, 0)),
        constant_values=1,
    )
    image = np.concatenate([image, mask], axis=1)
    L = np.pad(
        Ls["0448"][..., None].repeat(3, -1), ((5, 5), (5, 5), (0, 0)), constant_values=1
    )
    S = np.pad(
        Ss["0448"][..., None].repeat(3, -1), ((5, 5), (5, 5), (0, 0)), constant_values=1
    )
    LS = np.concatenate([L, S], axis=1)
    image = np.concatenate([image, LS], axis=0)

    image1 = np.pad(datas["0375"], ((5, 5), (5, 5), (0, 0)), constant_values=1)
    mask = np.pad(
        masks["0375"][..., None].repeat(3, -1),
        ((5, 5), (5, 5), (0, 0)),
        constant_values=1,
    )
    image1 = np.concatenate([image1, mask], axis=1)
    L = np.pad(
        Ls["0375"][..., None].repeat(3, -1), ((5, 5), (5, 5), (0, 0)), constant_values=1
    )
    S = np.pad(
        Ss["0375"][..., None].repeat(3, -1), ((5, 5), (5, 5), (0, 0)), constant_values=1
    )
    LS = np.concatenate([L, S], axis=1)
    image1 = np.concatenate([image1, LS], axis=0)

    image = np.concatenate([image, image1], axis=1)

    cv2.imwrite("concat.png", image * 255)

    # for key in datas.keys():
    #     cv2.imwrite(f"images/{key}.png", datas[key][:, :, ::-1] * 255)


if __name__ == "__main__":
    main()
