from math import ceil

import numpy as np
import scipy
from . import scanning


def bwdct(
        img: np.ndarray, block_size=8,
        qt=None, is_zzsqt=False, trunc_only=False) -> np.ndarray:
    """JPEGで利用されるブロックごとのDCTを行います。

    Args:
        a: 変換対象の二次元配列
        block_size: DCTを行うブロックのサイズ

    Returns:
        ブロックごとのDCT係数

    Todo:
         * skimage.restoration.cycle_spinの調査
    """
    if qt is not None and is_zzsqt:
        qt = scanning.zigzag(qt, block_size=block_size, inverse=True)
    output_size = (
        ceil(img.shape[0] / block_size),
        ceil(img.shape[1] / block_size),
        block_size ** 2
    )
    img = np.pad(
        img,
        [
            [0, output_size[0] * block_size - img.shape[0]],
            [0, output_size[0] * block_size - img.shape[1]]
        ])
    coef = np.zeros(output_size, dtype=np.float64)
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            # float64
            block = scipy.fft.dctn(
                    img[
                        i*block_size:(i+1)*block_size,
                        j*block_size:(j+1)*block_size
                    ],
                    axes=[0, 1], norm='ortho'
                ).ravel()
            if qt is not None:
                block = np.round(block / qt)
                if trunc_only:
                    block = block * qt
            coef[i:i+1, j:j+1] = block
    return coef


def bwidct(
        coef: np.ndarray, block_size=8,
        qt=None, is_zzsqt=False, src_size=None) -> np.ndarray:
    """JPEGで利用されるブロックごとのDCT係数から元配列を復元します。

    Args:
        coef: 変換対象の二次元配列
        block_size: DCTが行われたブロックのサイズ

    Returns:
        復元された二次元配列

    Todo:
         * skimage.restoration.cycle_spinの調査
    """
    if qt is not None and is_zzsqt:
        qt = scanning.zigzag(qt, block_size=block_size, inverse=True)
    output_size = (
        coef.shape[0] * block_size,
        coef.shape[1] * block_size
    )
    img = np.zeros(output_size, dtype=np.uint8)
    for i in range(output_size[0] // block_size):
        for j in range(output_size[1] // block_size):
            block = coef[i, j]
            if qt is not None:
                block = block * qt
            # float32
            block = scipy.fft.idctn(
                    block.reshape((block_size, block_size)),
                    axes=[0, 1], norm='ortho'
                )
            img[
                i * block_size:(i + 1) * block_size,
                j * block_size:(j + 1) * block_size
            ] = block.clip(0, 255).astype(np.uint8)
    return img if src_size is None else img[:src_size[0], :src_size[1]]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    src = np.random.randint(0, 256, size=(28, 28), dtype=np.uint8)

    coefs = bwdct(src, with_pad=True)
    rstr = bwidct(coefs, src_size=src.shape)

    src = src.astype(np.uint8)
    rstr = rstr.astype(np.uint8)

    fig, axs = plt.subplots(1, 3)
    ax_src, ax_rstr, ax_diff = axs

    fig.suptitle('Blockwise DCT Sample')
    ax_src.imshow(src, cmap='gray')
    ax_rstr.imshow(rstr, cmap='gray')
    ax_diff.imshow((rstr - src).astype(np.float32) / 255., cmap='gray')
    # assert (rstr == src).all()

    plt.show()
