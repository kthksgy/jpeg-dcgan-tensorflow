from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.transforms.jpeg import BlockwiseDCT

bwdct = BlockwiseDCT((8, 8))

zz = [
    0, 1, 5, 6, 14, 15, 27, 28,
    2, 4, 7, 13, 16, 26, 29, 42,
    3, 8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
]


def min_max_norm(x):
    x_min = x.min()
    x_max = x.max()
    if x_min == x_max:
        ret = x / x_min
    else:
        ret = (x - x_min) / (x_max - x_min)
    return ret


def value(p, q):
    cos = 0
    for m in range(8):
        for n in range(8):
            tmp = \
                np.cos(((2 * m + 1) * np.pi * p) / 8) \
                * np.cos(((2 * n + 1) * np.pi * q) / 8)
            if tmp > 0:
                cos += tmp
    ret = cos * 255
    ret *= np.sqrt((1 if p == 0 else 2) * (1 if q == 0 else 2) / 64)
    return ret


res = [[] for _ in range(8)]
for i in range(8):
    for j in range(8):
        a = np.zeros((1, 1, 8, 8), dtype=np.float64)
        v = value(i, j)
        a[:, :, i, j] = v
        image = bwdct.inverse(a.reshape(1, 1, 64))
        image = min_max_norm(image) * 255
        res[i].append(image.clip(0, 255).astype(np.uint8))

image = np.block(res)
plt.imshow(image, cmap='gray')
plt.show()

p_img = Path('./outputs/images/dct_basis.png')
p_img.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(p_img), image)
