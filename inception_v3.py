# 標準モジュール
import argparse
from logging import (
    getLogger, basicConfig,
    DEBUG, INFO, WARNING
)
from pathlib import Path

# 追加モジュール
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from utils.device import AutoDevice
from utils.common.datasets import load_dataset
from evaluation_index.frechet_inception_distance \
    import FrechetInceptionDistance

# コマンドライン引数を取得するパーサー
parser = argparse.ArgumentParser(
    prog='Inception V3 特徴抽出',
    description='FID計算に必要になるデータセットのInception V3特徴を計算します。'
)

parser.add_argument(
    '-b', '--batch-size', help='バッチサイズを指定します。',
    type=int, default=250, metavar='B'
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
    '--disable-cuda', '--cpu', help='CUDAを無効化しGPU上では計算を行わず全てCPU上で計算します。',
    action='store_true'
)
parser.add_argument(
    '--info', help='ログ表示レベルをINFOに設定し、詳細なログを表示します。',
    action='store_true'
)
parser.add_argument(
    '--debug', help='ログ表示レベルをDEBUGに設定し、より詳細なログを表示します。',
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
# 名前を指定してロガーを取得する
logger = getLogger('main')

# デバイスについての補助クラスをインスタンス化
auto_device = AutoDevice(disable_cuda=args.disable_cuda)
logger.info('デバイスの優先順位を計算しました。')
device = auto_device()
logger.info(f'メインデバイスとして〈{device}〉が選択されました。')

logger.info('画像に適用する変換のリストを定義します。')
dataset_transforms = []

if args.dataset in ['mnist', 'fashion_mnist']:
    dataset_transforms.append(
        transforms.Pad(2, fill=0, padding_mode='constant'))
    logger.info('変換リストにゼロパディングを追加しました。')
else:
    dataset_transforms.append(transforms.Grayscale())
    logger.info('変換リストにグレイスケール化を追加しました。')

IMAGE_SIZE = (32, 32)

dataset_transforms.append(
    transforms.ToTensor()
)

INCEPTION_V3_PATH = Path(args.data_path).joinpath('inception_v3')
fid = FrechetInceptionDistance(device=device)
NPZ_PATH = INCEPTION_V3_PATH\
    .joinpath(args.dataset).joinpath(
        f'{args.dataset}_{IMAGE_SIZE[0]}_{IMAGE_SIZE[1]}.npz')
NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
if NPZ_PATH.exists():
    print('既に特徴を保存したファイルが存在します。')
else:
    dataset = load_dataset(
        args.dataset, root=args.data_path,
        transform=dataset_transforms)
    NUM_CLASSES = len(dataset.classes)
    NUM_FEATURES = dataset[0][0].shape[0]
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, drop_last=True)
    pbar = tqdm(
        dataloader,
        desc='[準備] Inception-V3を適用中... ',
        total=len(dataloader),
        leave=False)
    tmp = [[] for _ in range(NUM_CLASSES)]
    with torch.no_grad():
        for imgs, lbls in pbar:
            feats = fid.get_features(imgs)
            for feat, lbl in zip(feats, lbls):
                tmp[lbl].append(feat)
            filled = 0
            for arr in tmp:
                if len(arr) >= 1000:
                    filled += 1
            if filled >= len(tmp):
                break
    fid_real_features = np.concatenate([
        np.stack(arr[:1000]) for arr in tmp])
    fid_labels = np.repeat(np.arange(0, 10), 1000)
    np.savez_compressed(
        NPZ_PATH, features=fid_real_features, labels=fid_labels)
    pbar.close()
