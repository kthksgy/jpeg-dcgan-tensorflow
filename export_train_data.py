# 標準モジュール
import argparse
from pathlib import Path
from PIL import Image
import re

# 追加モジュール
import cv2
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm

# 自作モジュール
from utils.common.datasets import load_dataset
from utils.common.image import make_grid

# コマンドライン引数を取得するパーサー
parser = argparse.ArgumentParser(
    prog='Export Train Data',
    description='PyTorchを用いて読み込んだ訓練データをそのままファイル出力します。'
)
parser.add_argument(
    '--dataset', help='データセットを指定します。',
    type=str, default='mnist', choices=['mnist', 'fashion_mnist', 'cifar10']
)
parser.add_argument(
    '--data-path', help='データセットのパスを指定します。',
    type=str, default='./data'
)
parser.add_argument(
    '--num-outputs', help='出力画像数を指定します。',
    type=int, default=100
)
parser.add_argument(
    '--scale', help='画像の拡大倍率を指定します。',
    type=float, default=1.0
)
parser.add_argument(
    '--pad', help='画像の上下左右のパディングを指定します。',
    type=int, default=0
)
parser.add_argument(
    '--grid', help='訓練画像をグリッド状の1枚の画像で保存します。',
    action='store_true'
)
parser.add_argument(
    '--num-samples', help='1クラス当たりのサンプル数を指定します。',
    type=int, default=10
)
# コマンドライン引数をパースする
args = parser.parse_args()

OUTPUT_DIR = Path(f'./outputs/images/{args.dataset}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dataset = load_dataset(
    args.dataset, root=args.data_path, transform=transforms.Pad(args.pad))
NUM_CLASSES = len(dataset.classes)
# PILのサイズは(Width, Height)
IMAGE_SIZE = dataset[0][0].size
OUTPUT_SIZE = tuple(map(lambda x: int(x * args.scale), IMAGE_SIZE))

total = min(len(dataset), args.num_outputs if not args.grid else len(dataset))
pbar = tqdm(
    zip(range(total), dataset),
    desc='訓練画像を抽出中... ',
    total=total,
    leave=False)
if args.grid:
    imgs = [[] for _ in range(NUM_CLASSES)]
    for _, (img, lbl) in pbar:
        img = np.asarray(img)
        img = cv2.resize(
            img, OUTPUT_SIZE,
            interpolation=cv2.INTER_NEAREST)
        if img.ndim == 3:
            img = img.transpose(2, 0, 1)
        imgs[lbl].append(img)
        filled = 0
        for arr in imgs:
            if len(arr) >= args.num_samples:
                filled += 1
        if filled >= len(imgs):
            break
    image = make_grid([arr[:args.num_samples] for arr in imgs])
    if image.ndim == 3:
        image = image.transpose(1, 2, 0)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(
        str(OUTPUT_DIR.joinpath(
            f'grid_{NUM_CLASSES}_{args.num_samples}_{OUTPUT_SIZE[1]}.png')),
        image)
else:
    label_names = []
    output_paths = []
    for s in dataset.classes:
        label_name = re.sub(
            r'[\\/:*?"<>|]+', ' ',
            s.rsplit(' - ', maxsplit=1)[-1])
        label_names.append(label_name)
        p = OUTPUT_DIR.joinpath(label_name)
        p.mkdir(parents=True, exist_ok=True)
        output_paths.append(p)

    for i, (img, lbl) in pbar:
        img = img.resize(OUTPUT_SIZE, Image.NEAREST)
        img.save(
            output_paths[lbl].joinpath(
                f'{label_names[lbl]}_{i + 1}_{OUTPUT_SIZE[1]}.png'))
pbar.close()
