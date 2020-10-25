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
        )).clip(self.uintx_min, self.uintx_max).astype(self.uintx)
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
