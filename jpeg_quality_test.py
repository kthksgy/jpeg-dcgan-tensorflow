# 標準モジュール
import argparse
from copy import deepcopy
from pathlib import Path

# 追加モジュール
import cv2
import numpy as np

# 自作モジュール
from utils.transforms.color import (
    YCrCb
)
from utils.transforms.jpeg import (
    ChromaSubsampling,
    BlockwiseDCT,
    JPEGQuantize,
)

# コマンドライン引数を取得するパーサー
parser = argparse.ArgumentParser(
    prog='JPEG Quality Test',
    description='画像を様々なJPEGのQualityで圧縮して保存します。',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    'path', help='画像のパスを指定します。',
    type=str
)
parser.add_argument(
    '--quality', help='出力するQualityを%(type)sで指定します。',
    type=int, nargs='+', metavar='Q', choices=range(1, 100),
    default=[1, 5, 10, 50, 75],
)
parser.add_argument(
    '--rgb', help='RGBを保存します。',
    action='store_true'
)
parser.add_argument(
    '--ycrcb', help='YCrCbを保存します。',
    action='store_true'
)
# コマンドライン引数をパースする
args = parser.parse_args()

path = Path(args.path)
if not path.exists():
    raise FileNotFoundError('ファイルが見つかりません。')

img = cv2.imread(str(path), 1)
fstem = path.stem

OUTPUT_DIR = Path(f'./outputs/images/{fstem}_quality')
OUTPUT_DIR.mkdir(parents=True)

ycrcb = YCrCb()
img = ycrcb(img)

chroma_subsampling = ChromaSubsampling(img.shape[:-1], '4:2:0')
luma_region = chroma_subsampling.get_luma_region()
chroma_region = chroma_subsampling.get_chroma_region()
img = chroma_subsampling(img)

bwdct = BlockwiseDCT(img.shape)
blks_orig = bwdct(img)

for q in args.quality:
    blks = deepcopy(blks_orig)
    jpeg_quantize = JPEGQuantize(
        quality=q, source='jpeg_standard',
        luma_region=luma_region, chroma_region=chroma_region)
    q_blks = jpeg_quantize(blks)
    d_blks = jpeg_quantize.inverse(q_blks)
    d_img = bwdct.inverse(d_blks)
    d_img = d_img.clip(0, 255).astype(np.uint8)
    d_img = chroma_subsampling.inverse(d_img)
    d_img = ycrcb.inverse(d_img)
    cv2.imwrite(str(OUTPUT_DIR.joinpath(f'{fstem}_q{q}.png')), d_img)
