# 標準モジュール
import argparse
import csv
from datetime import datetime
from importlib import import_module
from logging import (
    getLogger, basicConfig,
    DEBUG, INFO, WARNING
)
from pathlib import Path
import random
import sys
from time import perf_counter

# 追加モジュール
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm

# 自作モジュール
from util.image.shape import tile_images
from utils.common.datasets import load_dataset
from utils.device import AutoDevice
from utils.transforms.common import Normalize
from utils.transforms.jpeg import (
    BlockwiseDCT,
    JPEGQuantize,
    LowPassFilter,
)
from models.mymodel32 import Generator, Discriminator
from evaluation_index.frechet_inception_distance \
    import FrechetInceptionDistance

# コマンドライン引数を取得するパーサー
parser = argparse.ArgumentParser(
    prog='PyTorch Generative Adversarial Network',
    description='PyTorchを用いてGANの画像生成を行います。'
)

# 訓練に関する引数
parser.add_argument(
    '-b', '--batch-size', help='バッチサイズを指定します。',
    type=int, default=250, metavar='B'
)
parser.add_argument(
    '-e', '--epochs', help='学習エポック数を指定します。',
    type=int, default=100, metavar='E'
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
    '--form', help='データをどのような形状で読み込むかを指定します。',
    type=str, default='grayscale', choices=['grayscale', 'jpeg']
)
parser.add_argument(
    '--criterion', help='訓練時に使用する損失関数を指定します。',
    type=str, default='binary_cross_entropy',
    choices=['binary_cross_entropy', 'hinge']
)
parser.add_argument(
    '-z', '--z-dim', help='潜在空間のサイズを指定します。',
    type=int, default=128, metavar='Z'
)
parser.add_argument(
    '--seed', help='乱数生成器のシード値を指定します。',
    type=int, default=42
)
# JPEG圧縮に関するコマンドライン引数
parser.add_argument(
    '--low-pass-ratio', help='Low Passフィルターの割合を設定します。1.0でフィルターしません。',
    type=float, default=1.0, metavar='[0.0-1.0]'
)
parser.add_argument(
    '--jpeg-quality', help='JPEG圧縮のQualityパラメータを指定します。量子化テーブルが変化します。',
    type=int, default=90, metavar='[1-100]'
)
parser.add_argument(
    '--quantization-table-source', help='JPEG圧縮の量子化テーブルのソースを指定します。',
    type=str, default='jpeg_standard', choices=['jpeg_standard']
)
# 出力に関するコマンドライン引数
parser.add_argument(
    '--dir-name', help='出力ディレクトリの名前を指定します。',
    type=str, default=None,
)
#   画像生成
parser.add_argument(
    '--plot', help='Matplotlibでサンプルを表示します。',
    action='store_true'
)
parser.add_argument(
    '--no-sample', help='結果を評価するためのサンプルを出力しません。',
    action='store_true'
)
parser.add_argument(
    '--num-samples', help='結果を見るための1クラス当たりのサンプル数を指定します。',
    type=int, default=10
)
parser.add_argument(
    '--sample-interval', help='生成画像の保存間隔をエポック数で指定します。',
    type=int, default=5,
)
#   モデルの保存
parser.add_argument(
    '--save', help='訓練したモデルを保存します。',
    action='store_true'
)
parser.add_argument(
    '--save-interval', help='モデルの保存間隔をエポック数で指定します。',
    type=int, default=10,
)
parser.add_argument(
    '--load', help='訓練したモデルのパスをGenerator、Discriminatorの順に指定します。',
    type=str, nargs=2, default=None
)
parser.add_argument(
    '--test', help='生成時間のテストを行います。',
    action='store_true'
)
# FIDに関する設定
parser.add_argument(
    '--fid', help='GeneratorをFIDで評価します。',
    action='store_true'
)
parser.add_argument(
    '--fid-batch-size', help='Inception-V3に与えるデータのバッチサイズを指定します。',
    type=int, default=100
)
parser.add_argument(
    '--fid-num-samples', help='FIDで検証するサンプル数を2048以上で指定します。',
    type=int, default=10000
)
parser.add_argument(
    '--fid-interval', help='FIDを計算する間隔を指定します。',
    type=int, default=5,
)
# PyTorchに関するコマンドライン引数
parser.add_argument(
    '--preload', help='事前に訓練データをメモリに読み込みます。',
    action='store_true'
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

# 結果を出力するために起動日時を保持する
LAUNCH_DATETIME = datetime.now()

# ロギングの設定
basicConfig(
    format='%(asctime)s %(name)s %(funcName)s %(levelname)s: %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=DEBUG if args.debug else INFO if args.info else WARNING,
)
# 名前を指定してロガーを取得する
logger = getLogger('main')

# Matplotlibの設定
pfp = \
    matplotlib \
    .font_manager \
    .FontProperties(fname='./fonts/KintoSans-Bold.ttf')
PFP = {'fontproperties': pfp}
matplotlib.rcParams['savefig.dpi'] = 350
logger.info('Matplotlibの設定を行いました。')

# 出力に関する定数
if args.dir_name is None:
    OUTPUT_DIR = Path(
        LAUNCH_DATETIME.strftime(
            f'./outputs/{args.dataset}/{args.form}/%Y%m%d%H%M%S'))
else:
    OUTPUT_DIR = Path(f'./outputs/{args.dataset}/{args.form}/{args.dir_name}')
OUTPUT_DIR.mkdir(parents=True)
logger.info(f'結果出力用のディレクトリ({OUTPUT_DIR})を作成しました。')
f_outputs = open(
    OUTPUT_DIR.joinpath('outputs.txt'), mode='w', encoding='utf-8')
f_outputs.write(' '.join(sys.argv) + '\n')
if not args.no_sample:
    OUTPUT_SAMPLE_DIR = OUTPUT_DIR.joinpath('samples')
    OUTPUT_SAMPLE_DIR.mkdir(parents=True)
    logger.info(f'画像用のディレクトリ({OUTPUT_SAMPLE_DIR})を作成しました。')
if args.save:
    OUTPUT_MODEL_DIR = OUTPUT_DIR.joinpath('models')
    OUTPUT_MODEL_DIR.mkdir(parents=True)
    logger.info(f'モデル用のディレクトリ({OUTPUT_MODEL_DIR})を作成しました。')

# 乱数生成器のシード値の設定
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
logger.info('乱数生成器のシード値を設定しました。')

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

# Grayscale Transform Instances
MEAN_GRAYSCALE = (0.5,)
STD_GRAYSCALE = (0.5,)
normalize = Normalize(MEAN_GRAYSCALE, STD_GRAYSCALE, inplace=True)
logger.info('グレイスケール用の正規化を定義しました。')

# JPEG Transform Instances
bwdct = BlockwiseDCT((32, 32))
logger.info('ブロックワイズ離散コサイン変換を定義しました。')
jpeg_quantize = JPEGQuantize(
    quality=args.jpeg_quality, source=args.quantization_table_source,
    luma_region=[(0, i) for i in IMAGE_SIZE])
logger.info('JPEG量子化を定義しました。')
low_pass_filter = LowPassFilter(args.low_pass_ratio)
logger.info('ローパスフィルタを定義しました。')

if args.form == 'jpeg':
    dataset_transforms.extend([
        bwdct,
        jpeg_quantize,
        low_pass_filter
    ])
    logger.info('変換リストにJPEG圧縮処理を追加しました。')

to_tensor = transforms.ToTensor()
dataset_transforms.append(to_tensor)
logger.info('変換リストにテンソル化を追加しました。')

if args.form == 'grayscale':
    dataset_transforms.append(normalize)
    logger.info('変換リストに正規化を追加しました。')

dataset = load_dataset(
    args.dataset, root=args.data_path, transform=dataset_transforms)
NUM_CLASSES = len(dataset.classes)
NUM_FEATURES = dataset[0][0].shape[0]
logger.info(f'データセット〈{args.dataset}〉を読み込みました。')

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True)
logger.info('データローダを生成しました。')

if args.preload:
    images_preloaded = []
    labels_preloaded = []
    for images, labels in tqdm(
        dataloader,
        desc='学習データの事前読み込み中...',
        total=len(dataloader),
        leave=False
    ):
        images_preloaded.append(images)
        labels_preloaded.append(labels)
    images_preloaded = torch.cat(images_preloaded)
    labels_preloaded = torch.cat(labels_preloaded)
    dataset = torch.utils.data.TensorDataset(
        images_preloaded, labels_preloaded)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)

INCEPTION_V3_PATH = Path(args.data_path).joinpath('inception_v3')
fid = FrechetInceptionDistance(device=device)
if args.fid:
    NPZ_PATH = Path(args.data_path) \
        .joinpath('inception_v3') \
        .joinpath(args.dataset) \
        .joinpath(f'{args.dataset}_{IMAGE_SIZE[0]}_{IMAGE_SIZE[1]}.npz')
    NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    if NPZ_PATH.exists():
        npz = np.load(NPZ_PATH)
        fid_real_features = npz['features']
        fid_labels = npz['labels']
    else:
        raise Exception('指定したデータセットのInception V3の特徴ファイルが見つかりません。')
    fid_num_batches = int(np.ceil(fid_labels.shape[0] / args.fid_batch_size))
    fid_labels_batch = np.pad(
        fid_labels,
        [0, fid_num_batches * args.fid_batch_size - fid_labels.shape[0]])\
        .reshape(fid_num_batches, args.fid_batch_size).astype(np.int64)

if args.fid_num_samples < 2048:
    args.fid_num_samples = 2048

model_g = Generator(
    args.z_dim, NUM_FEATURES,
    bias=True, num_classes=NUM_CLASSES, form=args.form).to(device)
model_d = Discriminator(
    NUM_FEATURES, num_classes=NUM_CLASSES, form=args.form).to(device)

##################
# Optimizerの設定 #
##################
optim_g = optim.Adam(
    model_g.parameters(),
    lr=0.001 * args.batch_size / 1000,
    betas=[0.5, 0.999])
optim_d = optim.Adam(
    model_d.parameters(),
    lr=0.00025 * args.batch_size / 1000,
    betas=[0.5, 0.999])

example_labels = \
    torch.arange(start=0, end=NUM_CLASSES, step=1, device=device)\
    .view(-1, 1).repeat(1, args.num_samples).flatten()
example_z = torch.randn(
    example_labels.size()[0], args.z_dim, device=device)

# 損失関数の定義を動的インポートする
criterion_module = import_module(f'criterions.{args.criterion}')
criterion = criterion_module.Criterion(args.batch_size, device)


def to_images(tensors, form: str = 'grayscale'):
    if form == 'grayscale':
        map(normalize.inverse, tensors)
        images = tensors.permute(0, 2, 3, 1).numpy()
    elif form == 'jpeg':
        coefs = tensors.permute(0, 2, 3, 1).numpy()
        images = np.zeros((coefs.shape[0], 32, 32, 1))
        for i, coef in enumerate(coefs):
            coef = low_pass_filter.inverse(coef)
            # coef = np.round(coef)
            coef = jpeg_quantize.inverse(coef)
            coef = bwdct.inverse(coef)
            images[i] = np.expand_dims(coef, -1)
            images[i] /= images[i].max()
    images *= 255
    return images.clip(0, 255).astype(np.uint8)


f_results = open(
    OUTPUT_DIR.joinpath('results.csv'), mode='w', encoding='utf-8')
csv_writer = csv.writer(f_results, lineterminator='\n')
result_items = [
    'Epoch', 'Generator Loss Mean', 'Discriminator Loss Mean',
    'Train Elapsed Time', 'FID', 'Sample Image File',
    'Generator Model File', 'Discriminator Model File'
]
csv_writer.writerow(result_items)
csv_idx = {item: i for i, item in enumerate(result_items)}

fig, ax = plt.subplots(1, 1)
for epoch in range(args.epochs):
    results = ['' for _ in range(len(csv_idx))]
    results[csv_idx['Epoch']] = f'{epoch + 1}'

    log_loss_g, log_loss_d = [], []

    pbar = tqdm(
        enumerate(dataloader),
        desc=f'[{epoch+1}/{args.epochs}] 訓練開始',
        total=len(dataset)//args.batch_size,
        leave=False)
    model_g.train()
    model_d.train()
    begin_time = perf_counter()  # 時間計測開始
    for i, (real_images, labels) in pbar:
        real_images = real_images.to(device)
        labels = labels.to(device)

        #######################################################################
        # Discriminatorの訓練
        #######################################################################
        model_d.zero_grad()
        output_d_real = model_d(real_images, labels)
        loss_d_real = criterion(output_d_real, real=True, generator=False)
        loss_d_real.backward()

        z = torch.randn(args.batch_size, args.z_dim, device=device)
        fake_images = model_g(z, labels)
        output_d_fake = model_d(fake_images.detach(), labels)
        loss_d_fake = criterion(output_d_fake, real=False, generator=False)
        loss_d_fake.backward()  # 同じ勾配をGeneratorでも利用する
        loss_d = loss_d_real + loss_d_fake
        log_loss_d.append(loss_d.item())
        optim_d.step()

        #######################################################################
        # Generatorの訓練
        #######################################################################
        model_g.zero_grad()
        output_g = model_d(fake_images, labels)
        loss_g = criterion(output_g, real=True, generator=True)
        loss_g.backward()
        log_loss_g.append(loss_g.item())
        optim_g.step()
        pbar.set_description_str(
            f'[{epoch+1}/{args.epochs}] 訓練中... '
            f'<損失: (G={loss_g.item():.016f}, D={loss_d.item():.016f})>')
    end_time = perf_counter()  # 時間計測終了
    pbar.close()

    loss_g_mean = np.mean(log_loss_g)
    loss_d_mean = np.mean(log_loss_d)
    results[csv_idx['Generator Loss Mean']] = f'{loss_g_mean:.016f}'
    results[csv_idx['Discriminator Loss Mean']] = f'{loss_d_mean:.016f}'

    train_elapsed_time = end_time - begin_time
    results[csv_idx['Train Elapsed Time']] = f'{train_elapsed_time:.07f}'

    print(
        f'[{epoch+1}/{args.epochs}] 訓練完了. '
        f'<エポック処理時間: {train_elapsed_time:.07f}[s/epoch]'
        f', 平均損失: (G={loss_g_mean:.016f}, D={loss_d_mean:.016f})>')

    model_g.eval()
    model_d.eval()

    # FID
    if args.fid and (
            epoch == 0
            or (epoch + 1) % args.fid_interval == 0
            or epoch == args.epochs - 1):
        pbar = tqdm(
            fid_labels_batch,
            desc=f'[{epoch+1}/{args.epochs}] 画像を生成中... ',
            total=fid_labels_batch.shape[0],
            leave=False)
        fid_fake_features = []
        with torch.no_grad():
            for labels in pbar:
                z = torch.randn(
                    args.fid_batch_size, args.z_dim, device=device)
                fake_images = to_images(
                    model_g(z, torch.from_numpy(labels).to(device))
                    .detach().cpu(),
                    form=args.form)
                fake_images = torch.stack([
                    to_tensor(image) for image in fake_images])
                fid_fake_features.append(fid.get_features(fake_images))
        fid_fake_features = np.concatenate(fid_fake_features)
        pbar.set_description_str(f'[{epoch+1}/{args.epochs}] FID計算中... ')
        fid_score = fid(fid_real_features, fid_fake_features)
        pbar.close()
        results[csv_idx['FID']] = f'{fid_score:.05f}'
        print(f'[{epoch+1}/{args.epochs}] FID計算完了. <FID: {fid_score:.05f}>')

    if not args.no_sample and (
            epoch == 0
            or (epoch + 1) % args.sample_interval == 0
            or epoch == args.epochs - 1):
        with torch.no_grad():
            example_images = model_g(example_z, example_labels).detach().cpu()
        example_images = to_images(
            example_images, form=args.form)
        image_shape = example_images.shape[1:]

        example_images_tiled = tile_images(example_images, 10, 10)
        if args.plot:
            title_text = f'生成画像 ({epoch+1}エポック完了)'
            fig.canvas.set_window_title(title_text)
            fig.suptitle(title_text, **PFP)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(image_shape[0]))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(image_shape[1]))
            ax.imshow(
                example_images_tiled,
                cmap='gray' if image_shape[-1] == 1 else None)
            plt.pause(0.01)
        sample_image_fname = f'{epoch+1:06d}.png'
        cv2.imwrite(
            str(OUTPUT_SAMPLE_DIR.joinpath(sample_image_fname)),
            example_images_tiled)
        results[csv_idx['Sample Image File']] = sample_image_fname

    if args.save and (
            (epoch + 1) % args.save_interval == 0
            or epoch == args.epochs - 1):
        model_g_fname = f'generator_{epoch+1:06d}.pt'
        # TODO: Generatorのセーブを行う。
        results[csv_idx['Generator Model File']] = model_g_fname

        model_d_fname = f'discriminator_{epoch+1:06d}.pt'
        # TODO: Generatorのセーブを行う。
        results[csv_idx['Discriminator Model File']] = model_d_fname
    csv_writer.writerow(results)
    f_results.flush()
f_results.close()

if args.test:
    print('生成時間の計測を開始します。')
    with torch.no_grad():
        z = torch.zeros(
            args.batch_size, args.z_dim,
            device=device, dtype=torch.float)
        labels = torch.zeros(
            args.batch_size, device=device, dtype=torch.long)
        begin_time = perf_counter()
        for _ in range(10):
            for _ in range(1000):
                torch.randn(args.batch_size, args.z_dim, out=z)
                torch.randint(0, NUM_CLASSES, (args.batch_size,), out=labels)
                model_g(z, labels)
        end_time = perf_counter()
    tmpstr = f'生成時間(1バッチ{args.batch_size}枚×{1000}回): '\
        f'{(end_time - begin_time) / 10:.07f}[s]'
    print(tmpstr)
    f_outputs.write(tmpstr)
f_outputs.close()
