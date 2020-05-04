import cv2
import numpy as np
from . import transform


def compress(
        img: np.ndarray,
        block_size=8, y_qt=None, crcb_qt=None, is_zzsqt=False,
        ycrcb_ratio=2, with_pad=False):
    crcb = None
    if img.ndim == 2:
        y = img
    elif img.ndim == 3:
        if img.shape[-1] == 1:
            y = img[:, :, 0]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y = img[:, :, 0]
            crcb = cv2.resize(
                img, None,
                fx=1/ycrcb_ratio, fy=1/ycrcb_ratio,
                interpolation=cv2.INTER_AREA)[:, :, 1:3]
    y_coefs = transform.bwdct(
        y, block_size=block_size, qt=y_qt,
        is_zzsqt=is_zzsqt, with_pad=with_pad)

    if crcb is None:
        return y_coefs

    crcb_coefs = np.array([transform.bwdct(
        crcb[:, :, i], block_size=block_size, qt=crcb_qt,
        is_zzsqt=is_zzsqt, with_pad=with_pad) for i in range(2)])
    return y_coefs, crcb_coefs


def decompress(
        y_coefs: np.ndarray, crcb_coefs=None,
        block_size=8, y_qt=None, crcb_qt=None, is_zzsqt=False,
        ycrcb_ratio=2, src_size=None):
    y = transform.bwidct(
        y_coefs, block_size=block_size, qt=y_qt,
        is_zzsqt=is_zzsqt, src_size=src_size)
    y = np.clip(y, 0, 255).astype(np.uint8)

    if crcb_coefs is None:
        return y

    crcb = np.array([transform.bwidct(
        crcb_coefs[i], block_size=block_size, qt=crcb_qt,
        is_zzsqt=is_zzsqt, src_size=(src_size[0]//2, src_size[1]//2)
    ) for i in range(2)])
    crcb = np.array([cv2.resize(
        crcb[i], (src_size[1], src_size[0]),
        interpolation=cv2.INTER_LINEAR) for i in range(2)])
    crcb = np.clip(crcb, 0, 255).astype(np.uint8)

    img = np.concatenate((np.expand_dims(y, -1), crcb.transpose(1, 2, 0)), -1)
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    return img


if __name__ == '__main__':
    from argparse import ArgumentParser
    import matplotlib.pyplot as plt
    from pathlib import Path
    from time import perf_counter
    from jpegfile import inspect
    from scanning import zigzag
    ARGUMENT_PARSER = ArgumentParser(
        description='JPEG圧縮過程の離散コサイン変換と量子化のテストプログラムです。'
    )
    ARGUMENT_PARSER.add_argument(
        '--input', help='入力画像のパスを指定します。'
    )
    ARGUMENT_PARSER.add_argument(
        '--quantization-table', help='量子化テーブルの参照元のJPEG画像のパスを指定します。'
    )
    ARGUMENT_PARSER.add_argument(
        '--show', help='出力を表示します。',
        action='store_true'
    )
    ARGUMENT_PARSER.add_argument(
        '--transparent', help='図を透過します。',
        action='store_true'
    )
    ARGS = ARGUMENT_PARSER.parse_args()

    ASSETS_DIR = Path(__file__).joinpath('../../../assets')
    if ARGS.input is None:
        INPUT_PATH = ASSETS_DIR.joinpath('q90_444.jpg')
    if ARGS.quantization_table is None:
        QUANTIZATION_TABLE_PATH = ASSETS_DIR.joinpath('q70_420.jpg')

    src = cv2.imread(str(INPUT_PATH), cv2.IMREAD_UNCHANGED)

    info = inspect(QUANTIZATION_TABLE_PATH)
    y_qt = info['quantization_table0']
    y_qt = zigzag(y_qt, inverse=True)
    crcb_qt = info['quantization_table1']
    crcb_qt = zigzag(crcb_qt, inverse=True)
    print('>> 圧縮します。')
    begin_compress_time = perf_counter()
    y_coefs, crcb_coefs = compress(
        src, y_qt=y_qt, crcb_qt=crcb_qt, with_pad=True)
    print(f' - 経過時間: {perf_counter() - begin_compress_time:.7f}[s]')
    src_zero_ratio = 1 - np.count_nonzero(src) / np.prod(src.shape)
    coefs_zero_ratio = 1 -\
        (np.count_nonzero(y_coefs) + np.count_nonzero(crcb_coefs)) /\
        (np.prod(y_coefs.shape) + np.prod(crcb_coefs.shape))

    print(f' - 入力画像ゼロ率: {src_zero_ratio:.1%}')
    print(f' - DCT係数ゼロ率: {coefs_zero_ratio:.1%}')
    print('>> 展開します。')
    begin_decompress_time = perf_counter()
    rstr = decompress(
        y_coefs, crcb_coefs=crcb_coefs,
        y_qt=y_qt, crcb_qt=crcb_qt, src_size=src.shape[0:2])
    print(f' - 経過時間: {perf_counter() - begin_decompress_time:.7f}[s]')

    fig, axs = plt.subplots(1, 3)
    ax_src, ax_rstr, ax_diff = axs

    fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95,
        wspace=0.05, hspace=0.05)
    fig.suptitle('JPEG Quantization Loss')
    ax_src.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    ax_src.set_yticks([])
    ax_src.set_xticks([])
    ax_src.set_title('Source')
    ax_rstr.imshow(cv2.cvtColor(rstr, cv2.COLOR_BGR2RGB))
    ax_rstr.set_yticks([])
    ax_rstr.set_xticks([])
    ax_rstr.set_title('Decompressed')
    ax_diff.imshow(
        np.abs(rstr.astype(np.int32) - src.astype(np.int32)).astype(np.uint8))
    ax_diff.set_yticks([])
    ax_diff.set_xticks([])
    ax_diff.set_title('Difference')
    if ARGS.show:
        plt.show()
    else:
        plt.savefig(
            'result.png', dpi=350, transparent=ARGS.transparent,
            bbox_inches='tight')
