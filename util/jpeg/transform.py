import numpy as np
import scipy
from . import scanning


def bwdct(a: np.ndarray, block_size=8, qt=None, is_zzsqt=False, with_pad=True) -> np.ndarray:
    """JPEGで利用されるブロックごとのDCTを行います。

    Args:
        a: 変換対象の二次元配列
        block_size: DCTを行うブロックのサイズ

    Returns:
        ブロックごとのDCT係数

    Todo:
         * 並列計算に対応する。(skimage.restoration.cycle_spinとか?)
    """
    if qt is not None and is_zzsqt:
        qt = scanning.zigzag(qt, block_size=block_size, inverse=True)
    if with_pad:
        h_pad = block_size - a.shape[0] % block_size
        w_pad = block_size - a.shape[1] % block_size
        a = np.pad(a, [[0, h_pad], [0, w_pad]])
    h = a.shape[0] // block_size
    w = a.shape[1] // block_size
    ret = np.zeros((h, w, block_size ** 2), dtype=np.int32)
    for i in range(h):
        for j in range(w):
            block = scipy.fft.dctn(
                    a[
                        i*block_size:(i+1)*block_size,
                        j*block_size:(j+1)*block_size
                    ],
                    axes=[0, 1], norm='ortho'
                ).ravel()
            if qt is not None:
                block = np.round(block / qt)
            ret[i:i+1, j:j+1] = block
    return ret


def bwidct(
        a: np.ndarray, block_size=8,
        qt=None, is_zzsqt=False, src_size=None) -> np.ndarray:
    """JPEGで利用されるブロックごとのDCT係数から元配列を復元します。

    Args:
        a: 変換対象の二次元配列
        block_size: DCTが行われたブロックのサイズ

    Returns:
        復元された二次元配列

    Todo:
         * 並列計算に対応する。(skimage.restoration.cycle_spinとか?)
    """
    if qt is not None and is_zzsqt:
        qt = scanning.zigzag(qt, block_size=block_size, inverse=True)
    h = a.shape[0] * block_size
    w = a.shape[1] * block_size
    ret = np.zeros((h, w))
    for i in range(h // block_size):
        for j in range(w // block_size):
            block = a[i, j]
            if qt is not None:
                block = block * qt
            block = scipy.fft.idctn(
                    block.reshape((block_size, block_size)),
                    axes=[0, 1], norm='ortho'
                )
            ret[
                i*block_size:(i+1)*block_size,
                j*block_size:(j+1)*block_size] = block
    return ret if src_size is None else ret[:src_size[0], :src_size[1]]


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
