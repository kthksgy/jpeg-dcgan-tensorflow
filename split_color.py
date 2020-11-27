# 標準モジュール
import argparse
from logging import (
    getLogger, basicConfig,
    DEBUG, INFO, WARNING
)
from pathlib import Path

# 追加モジュール
import cv2
import numpy as np

# コマンドライン引数を取得するパーサー
parser = argparse.ArgumentParser(
    prog='Split Color',
    description='入力された画像の色を分離してそれぞれ保存します。'
)
parser.add_argument(
    'path', help='画像のパスを指定します。',
    type=str
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
# ロギングの設定
basicConfig(
    format='%(asctime)s %(name)s %(funcName)s %(levelname)s: %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=DEBUG if args.debug else INFO if args.info else WARNING,
)
# ロガーの取得
logger = getLogger('main')

# 画像のパスをコマンドライン引数から取得
path = Path(args.path)
print(f' >> {path}を対象にします。')
if not path.exists():
    raise FileNotFoundError('指定された画像ファイルが見つかりません。')

img = cv2.imread(str(path), 1)
fstem = path.stem

OUTPUT_DIR = Path(f'./outputs/images/{fstem}_split')
OUTPUT_DIR.mkdir(parents=True)


if args.rgb:
    colors = ['b', 'g', 'r']
    for i in range(len(colors)):
        c = img[:, :, i]
        c = np.expand_dims(c, axis=-1)
        c = np.pad(c, [[0, 0], [0, 0], [i, 2 - i]])
        cv2.imwrite(str(OUTPUT_DIR.joinpath(f'{fstem}_{colors[i]}.png')), c)

if args.ycrcb:
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = tmp[:, :, 0]
    y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(OUTPUT_DIR.joinpath(f'{fstem}_y.png')), y)
    cr = tmp[:, :, 1]
    cr = np.expand_dims(cr, axis=-1)
    cr = np.pad(cr, [[0, 0], [0, 0], [2, 0]])
    cv2.imwrite(str(OUTPUT_DIR.joinpath(f'{fstem}_cr.png')), cr)
    cb = tmp[:, :, 2]
    cb = np.expand_dims(cb, axis=-1)
    cb = np.pad(cb, [[0, 0], [0, 0], [0, 2]])
    cv2.imwrite(str(OUTPUT_DIR.joinpath(f'{fstem}_cb.png')), cb)
