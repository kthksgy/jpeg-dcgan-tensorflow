from typing import Optional, Tuple

import cv2
import numpy as np
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dctn.html#scipy.fft.dctn
from scipy.fft import dctn, idctn


class BlockwiseDCT:
    def __init__(
        self, input_size: Tuple[int], block_size: Tuple[int] = (8, 8),
        floatx: np.dtype = np.float32, uintx: np.dtype = np.uint8
    ):
        self.input_size = input_size
        self.block_size = block_size
        assert self.input_size[0] % self.block_size[0] == 0, \
            '入力の縦幅がブロックの縦幅で割り切れません。'
        self.num_vblocks = self.input_size[0] // self.block_size[0]
        assert self.input_size[1] % self.block_size[1] == 0, \
            '入力の横幅がブロックの横幅で割り切れません。'
        self.num_hblocks = self.input_size[1] // self.block_size[1]
        self.num_coefficients = np.prod(block_size)
        self.floatx = floatx
        self.uintx = uintx
        ii = np.iinfo(self.uintx)
        self.uintx_min = ii.min
        self.uintx_max = ii.max

    def __call__(self, image, inplace=False):
        return dctn(
            self.__split(np.asarray(image)), type=2,
            axes=[1, 2],
            norm='ortho',
            overwrite_x=inplace,
            workers=-1
        ) \
            .reshape(
                self.num_vblocks,
                self.num_hblocks,
                self.num_coefficients) \
            .astype(self.floatx)

    def inverse(self, image, inplace=True):
        output = self.__concatenate(idctn(
            image.reshape(-1, self.block_size[0], self.block_size[1]), type=2,
            axes=[1, 2],
            norm='ortho',
            overwrite_x=inplace,
            workers=-1
        ))
        return output

    # https://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-multiple-smaller-2d-arrays/16873755#16873755
    def __split(self, image):
        return image \
            .reshape(
                self.num_vblocks,
                self.block_size[0],
                -1,
                self.block_size[1]) \
            .swapaxes(1, 2) \
            .reshape(-1, self.block_size[0], self.block_size[1])

    def __concatenate(self, blocks):
        return blocks \
            .reshape(
                self.num_vblocks,
                -1,
                self.block_size[0],
                self.block_size[1]) \
            .swapaxes(1, 2) \
            .reshape(self.input_size[0], self.input_size[1])


class JPEGQuantize:
    def __init__(
        self, *,
        quality: int, source: str,
        luma_region: Tuple[Tuple[int, int], Tuple[int, int]],
        chroma_region: Optional[
            Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        unit: str = 'pixel'
    ):
        self.quality = quality
        self.source = source
        self.luma, self.chroma = \
            self.get_table(quality=self.quality, source=self.source)
        self.luma = self.luma.ravel()
        self.chroma = self.chroma.ravel()
        self.luma_region = (
            (luma_region[0][0] // 8, luma_region[0][1] // 8),
            (luma_region[1][0] // 8, luma_region[1][1] // 8)
        ) if unit == 'pixel' else luma_region
        self.chroma_region = (
            (chroma_region[0][0] // 8, chroma_region[0][1] // 8),
            (chroma_region[1][0] // 8, chroma_region[1][1] // 8)
        ) if chroma_region is not None and unit == 'pixel' else chroma_region

    def __call__(self, blocks) -> np.ndarray:
        blocks[
            self.luma_region[0][0]:self.luma_region[0][1],
            self.luma_region[1][0]:self.luma_region[1][1]
        ] = np.round(
            blocks[
                self.luma_region[0][0]:self.luma_region[0][1],
                self.luma_region[1][0]:self.luma_region[1][1]
            ] / self.luma)
        if self.chroma_region is not None:
            blocks[
                self.chroma_region[0][0]:self.chroma_region[0][1],
                self.chroma_region[1][0]:self.chroma_region[1][1]
            ] = np.round(
                blocks[
                    self.chroma_region[0][0]:self.chroma_region[0][1],
                    self.chroma_region[1][0]:self.chroma_region[1][1]
                ] / self.chroma)
        return blocks

    def inverse(self, blocks) -> np.ndarray:
        blocks[
            self.luma_region[0][0]:self.luma_region[0][1],
            self.luma_region[1][0]:self.luma_region[1][1]
        ] = np.round(
            blocks[
                self.luma_region[0][0]:self.luma_region[0][1],
                self.luma_region[1][0]:self.luma_region[1][1]
            ] * self.luma)
        if self.chroma_region is not None:
            blocks[
                self.chroma_region[0][0]:self.chroma_region[0][1],
                self.chroma_region[1][0]:self.chroma_region[1][1]
            ] = np.round(
                blocks[
                    self.chroma_region[0][0]:self.chroma_region[0][1],
                    self.chroma_region[1][0]:self.chroma_region[1][1]
                ] * self.chroma)
        return blocks

    @classmethod
    def get_table(cls, *, quality: int = 50, source: str = 'jpeg_standard'):
        assert quality > 0, 'Qualityパラメータは1以上の整数で指定してください。'
        # https://www.impulseadventure.com/photo/jpeg-quantization.html
        if source == 'jpeg_standard':
            luma = np.asarray([
                [16, 11, 10, 16,  24,  40,  51,  61],
                [12, 12, 14, 19,  26,  58,  60,  55],
                [14, 13, 16, 24,  40,  57,  69,  56],
                [14, 17, 22, 29,  51,  87,  80,  62],
                [18, 22, 37, 56,  68, 109, 103,  77],
                [24, 35, 55, 64,  81, 104, 113,  92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103,  99]
            ])
            chroma = np.asarray([
                [17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]
            ])
        else:
            raise KeyError(f'指定した量子化テーブルのキー〈{source}〉は未実装です。')
        luma, chroma = \
            [
                np.floor((100 - quality) / 50 * table + 0.5).clip(min=1)
                if quality >= 50 else
                np.floor(50 / quality * table + 0.5)
                for table in [luma, chroma]
            ]
        return luma, chroma


class ChromaSubsampling:
    # https://en.wikipedia.org/wiki/Chroma_subsampling
    def __init__(self, image_size: Tuple[int, int], ratio: str):
        '''
        '''
        self.image_size = image_size
        self.ratio = ratio
        self.sampling_factor = self.get_sampling_factor(ratio=ratio)
        self.fy = 1 / self.sampling_factor[0]
        self.fx = 1 / self.sampling_factor[1]
        self.luma_size = self.image_size
        self.chroma_size = (
            self.image_size[0] // self.sampling_factor[0],
            self.image_size[1] // self.sampling_factor[1]
        )
        self.y_region = ((0, self.image_size[0]), (0, self.image_size[1]))
        self.cr_region = (
            (
                self.image_size[0],
                self.image_size[0] + self.chroma_size[0]
            ),
            (
                0,
                self.chroma_size[1]
            )
        )
        if self.sampling_factor[1] > 1:
            self.cb_region = (
                self.cr_region[0],
                (
                    self.cr_region[1][1],
                    self.cr_region[1][1] + self.chroma_size[1]
                )
            )
        else:
            self.cb_region = (
                (
                    self.cr_region[0][1],
                    self.cr_region[0][1] + self.chroma_size[0]
                ),
                (
                    0,
                    self.chroma_size[1]
                )
            )
        pass

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = image.transpose(2, 0, 1).reshape(-1, image.shape[1])
        image[
            self.cr_region[0][0]:self.cr_region[0][1],
            :self.cr_region[1][1]
        ] = cv2.resize(
                image[self.image_size[0]:self.image_size[0] * 2],
                None,
                fx=self.fx, fy=self.fy)
        image[
            self.cb_region[0][0]:self.cb_region[0][1],
            self.cb_region[1][0]:self.cb_region[1][1]
        ] = cv2.resize(
                image[self.image_size[0] * 2:self.image_size[0] * 3],
                None,
                fx=self.fx, fy=self.fy)
        return image[:self.cb_region[0][1], :self.cb_region[1][1]]

    def inverse(self, image: np.ndarray) -> np.ndarray:
        y = image[
            self.y_region[0][0]:self.y_region[0][1],
            self.y_region[1][0]:self.y_region[1][1]
        ]
        cr = cv2.resize(
                image[
                    self.cr_region[0][0]:self.cr_region[0][1],
                    self.cr_region[1][0]:self.cr_region[1][1]],
                None,
                fx=self.sampling_factor[1], fy=self.sampling_factor[0])
        cb = cv2.resize(
                image[
                    self.cb_region[0][0]:self.cb_region[0][1],
                    self.cb_region[1][0]:self.cb_region[1][1]],
                None,
                fx=self.sampling_factor[1], fy=self.sampling_factor[0])
        return np.stack([y, cr, cb], axis=-1)

    def get_regions(
        self
    ) -> Tuple[
        Tuple[Tuple[int, int], Tuple[int, int]],
        Tuple[Tuple[int, int], Tuple[int, int]],
        Tuple[Tuple[int, int], Tuple[int, int]],
    ]:
        return self.y_region, self.cr_region, self.cb_region

    def get_luma_region(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return self.y_region

    def get_chroma_region(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (
            (self.cr_region[0][0], self.cb_region[1][1]),
            (self.cr_region[1][0], self.cb_region[1][1])
        )

    @classmethod
    def get_sampling_factor(cls, *, ratio: str = '4:4:4'):
        if ratio == '4:4:4':
            sampling_factor = (1, 1)
        elif ratio == '4:2:2':
            sampling_factor = (1, 2)
        elif ratio == '4:2:0':
            sampling_factor = (2, 2)
        elif ratio == '4:4:0':
            sampling_factor = (2, 1)
        elif ratio == '4:1:1':
            sampling_factor = (1, 4)
        else:
            raise KeyError(f'指定した比率〈{ratio}〉は未実装です。')
        return sampling_factor


class LowPassFilter:
    def __init__(
        self, ratio: float, block_size: Tuple[int] = (8, 8),
    ):
        self.block_size = block_size
        self.new_block_size = (
            max(1, int(self.block_size[0] * ratio)),
            max(1, int(self.block_size[1] * ratio))
        )
        self.new_num_coefficients = np.prod(self.new_block_size)
        self.padding = (
            (0, 0), (0, 0),
            (0, self.block_size[0] - self.new_block_size[0]),
            (0, self.block_size[1] - self.new_block_size[1]),
        )

    def __call__(self, blocks):
        return blocks \
            .reshape(blocks.shape[:-1] + self.block_size)[
                :,
                :,
                :self.new_block_size[0],
                :self.new_block_size[1]] \
            .reshape(blocks.shape[:-1] + (-1,))

    def inverse(self, blocks):
        return np.pad(
            blocks.reshape(blocks.shape[:-1] + self.new_block_size),
            self.padding) \
            .reshape(blocks.shape[:-1] + (-1,))

    def get_num_features(self):
        return self.new_num_coefficients


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


def marker_sof(f):
    length = int.from_bytes(f.read(2), 'big')
    frame_header = {}
    frame_header['sample_precision'] = int.from_bytes(f.read(1), 'big')
    frame_header['height'] = int.from_bytes(f.read(2), 'big')
    frame_header['width'] = int.from_bytes(f.read(2), 'big')
    frame_header['num_channels'] = int.from_bytes(f.read(1), 'big')
    k = length - 8
    while k > 0:
        n = int.from_bytes(f.read(1), 'big')
        frame_header[f'channel{n}'] = {}
        tmp = int.from_bytes(f.read(1), 'big')
        hn = tmp >> 4
        vn = tmp - (hn << 4)
        frame_header[f'channel{n}']['horizontal_ratio'] = hn
        frame_header[f'channel{n}']['vertical_ratio'] = vn
        frame_header[f'channel{n}']['target_quantization_table'] = \
            int.from_bytes(f.read(1), 'big')
        k -= 3
    return frame_header


def marker_sof0(f):
    frame_header = marker_sof(f)
    frame_header['method'] = 'baseline'
    return frame_header


def marker_sof2(f):
    frame_header = marker_sof(f)
    frame_header['method'] = 'progressive'
    return frame_header


def marker_dqt(f):
    length = int.from_bytes(f.read(2), 'big')
    k = length - 2
    ret = {}
    while k > 0:
        tmp = int.from_bytes(f.read(1), 'big')
        pqn = tmp >> 4
        tqn = tmp - (pqn << 4)
        k -= 65 if pqn == 0 else 129
        ret[f'quantization_table{tqn}'] = np.array([
                int.from_bytes(f.read(1 if pqn == 0 else 2), 'big')
                for _ in range(64)],
                dtype=np.uint8 if pqn == 0 else np.uint16)
    return ret


MARKERS = {
    b'\xff\xc0': marker_sof0,
    b'\xff\xc2': marker_sof2,
    b'\xff\xdb': marker_dqt,
}


def inspect(path: str):
    f = open(str(path), 'rb')
    # SOI(Start of Image)
    assert f.read(2) == b'\xff\xd8'
    info = {}
    telled = 0
    while telled < f.tell():
        telled = f.tell()
        if f.read(1) != b'\xff':
            continue
        marker = b'\xff' + f.read(1)
        info.update(MARKERS.get(marker, lambda _: {})(f))
    return info


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
    print(zzo_2d)
    # JPEGファイルのDQT等の形式
    zzo_1d = np.arange(64)
    print(zzo_1d.reshape(8, 8))

    # 配列をジグザグスキャンする
    assert (zigzag(zzo_2d.ravel()) == zzo_1d).all()
    # ジグザグスキャン済みの配列を元に戻す
    assert (zzo_2d.ravel() == zigzag(zzo_1d, inverse=True)).all()
