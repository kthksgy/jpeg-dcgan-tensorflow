# 追加モジュール
import cv2
import numpy as np

# 自作モジュール
from utils.transforms.jpeg import (
    BlockwiseDCT,
    JPEGQuantize
)

np.set_printoptions(precision=1, floatmode='fixed', suppress=True)

img = cv2.imread('./assets/Y_BLOCK.png', 0)
print('画素値:')
print(img)

bwdct = BlockwiseDCT(img.shape)

blks = bwdct(img)
print('DCT係数:')
print(blks.reshape(8, 8))

jpeg_quantize = JPEGQuantize(quality=50, source='jpeg_standard', luma_region=((0, 8), (0, 8)))

q_blks = jpeg_quantize(blks)
print('量子化済みDCT係数:')
print(q_blks.reshape(8, 8).astype(np.int32))

print('量子化テーブル:')
print(jpeg_quantize.get_table(quality=50)[0].astype(np.int32))
