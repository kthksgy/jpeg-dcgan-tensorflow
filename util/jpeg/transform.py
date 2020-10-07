from typing import Tuple, Union

import numpy as np
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dctn.html#scipy.fft.dctn
from scipy.fft import dctn, idctn


class BlockwiseDCT:
    def __init__(
        self, input_size: Tuple[int], block_size: Tuple[int] = (8, 8),
        dtype=np.uint8
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

    def __call__(self, image, inplace=True):
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
            .astype(np.float32)

    def inverse(self, image, inplace=True, rescale=False):
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


class Quantize:
    def __init__(
        self, table: Union[np.ndarray, Tuple[np.ndarray]],
        round_only: bool = True
    ):
        # INDICES: y = 0, cb = 1, cr = 2
        self.table = \
            table if isinstance(table, tuple) \
            else (table,)
        assert all([
            isinstance(e, np.ndarray)
            and e.ndim == 1
            for e in self.table
        ]), '量子化テーブルは1次元のNumPy配列で指定してください。'
        self.round_only = round_only

    def __call__(self, blocks, idx: int = 0):
        blocks /= self.table[idx]
        blocks = np.round(blocks, out=blocks)
        if self.round_only:
            blocks *= self.table[idx]
        return blocks


class LowPassFilter:
    def __init__(
        self, ratio: float, block_size: Tuple[int] = (8, 8),
    ):
        self.block_size = block_size
        self.new_block_size = (
            int(self.block_size[0] * ratio),
            int(self.block_size[1] * ratio)
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
