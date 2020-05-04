import numpy as np


def zigzag(a: np.ndarray, block_size=8, inverse=False) -> np.ndarray:
    x = 0
    order = np.zeros(a.shape, dtype=np.int32)
    k = 1
    while x < block_size:
        x += 1
        order[k] = x
        k += 1
        for _ in range(x):
            order[k] = order[k-1] + 7
            k += 1
        if x == block_size - 1:
            break
        order[k] = order[k-1] + 8
        k += 1
        for _ in range(x):
            order[k] = order[k-1] - 7
            k += 1
        x += 1
        order[k] = x
        k += 1
    order[block_size**2//2+block_size//2:] = \
        63 - order[block_size**2//2-block_size//2-1::-1]
    ret = np.zeros(a.shape, dtype=a.dtype)
    if not inverse:
        for i in range(ret.shape[0]):
            ret[i] = a[order[i]]
    else:
        for i in range(ret.shape[0]):
            ret[i] = a[np.argwhere(order == i)[0]]
    return ret


if __name__ == '__main__':
    # DCTの出力
    zzo_2d = np.array([
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63]
    ])
    # JPEGファイルのDQT等の形式
    zzo_1d = np.arange(64)

    # 配列をジグザグスキャンする
    assert (zigzag(zzo_2d.ravel()) == zzo_1d).all()
    # ジグザグスキャン済みの配列を元に戻す
    assert (zzo_2d.ravel() == zigzag(zzo_1d, inverse=True)).all()
