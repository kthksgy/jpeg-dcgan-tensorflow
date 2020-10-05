from argparse import ArgumentParser
import csv
from datetime import datetime
from importlib import import_module
from pathlib import Path
import random
from time import perf_counter
from tqdm import tqdm

import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from util.image.shape import tile_images
from util.jpeg.transform import (bwdct, bwidct)
from util.jpeg.jpegfile import inspect
from util.jpeg.scanning import zigzag

from models.mymodel32 import Generator, Discriminator
from evaluation_index.frechet_inception_distance\
    import FrechetInceptionDistance

LAUNCH_DATETIME = datetime.now()

ARGUMENT_PARSER = ArgumentParser()

# 訓練に関する引数
ARGUMENT_PARSER.add_argument(
    '-b', '--batch-size', help='バッチサイズを指定します。',
    type=int, default=250, metavar='B'
)
ARGUMENT_PARSER.add_argument(
    '-e', '--epochs', help='学習エポック数を指定します。',
    type=int, default=100, metavar='E'
)
ARGUMENT_PARSER.add_argument(
    '--dataset', help='データセットを指定します。',
    type=str, default='mnist', choices=['mnist', 'fashion_mnist', 'cifar10']
)
ARGUMENT_PARSER.add_argument(
    '--data_path', help='データセットのパスを指定します。',
    type=str, default='./data'
)
ARGUMENT_PARSER.add_argument(
    '--form', help='データをどのような形状で読み込むかを指定します。',
    type=str, default='rgb_pixels', choices=['rgb_pixels', 'jpeg_blocks']
)
ARGUMENT_PARSER.add_argument(
    '--criterion', help='訓練時に使用する損失関数を指定します。',
    type=str, default='binary_cross_entropy',
    choices=['binary_cross_entropy', 'hinge']
)
# 画像生成に関する引数
ARGUMENT_PARSER.add_argument(
    '-z', '--z-dim', help='潜在空間のサイズを指定します。',
    type=int, default=128, metavar='Z'
)
# 結果に関する引数
ARGUMENT_PARSER.add_argument(
    '--plot', help='Matplotlibでサンプルを表示します。',
    action='store_true'
)
ARGUMENT_PARSER.add_argument(
    '--no-sample', help='結果を評価するためのサンプルを出力しません。',
    action='store_true'
)
ARGUMENT_PARSER.add_argument(
    '--num-samples', help='結果を見るための1クラス当たりのサンプル数を指定します。',
    type=int, default=10
)
ARGUMENT_PARSER.add_argument(
    '--sample-frequency', help='サンプルを出力する頻度をエポック数で指定します。',
    type=int, default=10
)
ARGUMENT_PARSER.add_argument(
    '--no-save', help='生成器と識別器の学習状況を保存しません。',
    action='store_true'
)
ARGUMENT_PARSER.add_argument(
    '--save-frequency', help='モデルの保存頻度をエポック数で指定します。',
    type=int, default=100,
)
ARGUMENT_PARSER.add_argument(
    '--test', help='生成時間のテストを行います。',
    action='store_true'
)
# FIDに関する設定
ARGUMENT_PARSER.add_argument(
    '--fid', help='FIDでGeneratorを評価します。',
    action='store_true'
)
ARGUMENT_PARSER.add_argument(
    '--fid-batch-size', help='Inception-V3に与えるデータのバッチサイズを指定します。',
    type=int, default=100
)
ARGUMENT_PARSER.add_argument(
    '--fid-num-samples', help='FIDで検証するサンプル数を2048以上で指定します。',
    type=int, default=10000
)
ARGUMENT_PARSER.add_argument(
    '--fid-frequency', help='FIDを計算する頻度を指定します。',
    type=int, default=10,
)
ARGS = ARGUMENT_PARSER.parse_args()

# Matplotlibの設定
pfp = \
    matplotlib \
    .font_manager \
    .FontProperties(fname='./fonts/KintoSans-Bold.ttf')
PFP = {'fontproperties': pfp}
matplotlib.rcParams['savefig.dpi'] = 350

# 出力に関する定数
OUTPUT_DIR = Path(
    LAUNCH_DATETIME.strftime(
        f'./outputs/{ARGS.dataset}/{ARGS.form}/%Y%m%d%H%M%S'))
if not ARGS.no_sample:
    OUTPUT_SAMPLE_DIR = OUTPUT_DIR.joinpath('samples')
    OUTPUT_SAMPLE_DIR.mkdir(parents=True)
if not ARGS.no_save:
    OUTPUT_MODEL_DIR = OUTPUT_DIR.joinpath('models')
    OUTPUT_MODEL_DIR.mkdir(parents=True)

# 乱数のシード値の設定
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class AutoDevice:
    def __init__(self):
        if torch.cuda.is_available():
            self.cuda_devices = []
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(f'cuda:{i}')
                self.cuda_devices.append({
                    'id': f'cuda:{i}',
                    'name': prop.name,
                    'capability': (prop.major, prop.minor),
                    'memory': prop.total_memory,
                    'num_processors': prop.multi_processor_count,
                })
            self.cuda_devices.sort(key=lambda d: d['memory'], reverse=True)
        else:
            self.cuda_devices = None

    def __call__(self, id):
        if self.cuda_devices is not None and id >= 0:
            return self.cuda_devices[min(id, len(self.cuda_devices) - 1)]['id']
        else:
            return 'cpu'


auto_device = AutoDevice()


class BlockwiseDCT(object):
    def __init__(self, jpeg_path: str = './assets/q90_420.jpg'):
        self.qt = inspect(jpeg_path)['quantization_table0']
        self.qt = zigzag(self.qt, inverse=True)

    def __call__(self, sample):
        sample = np.asarray(sample)
        compressed = bwdct(sample, qt=self.qt, trunc_only=True)
        # compressed = bwdct(sample)
        return compressed


class BlockwiseIDCT(object):
    def __init__(self, jpeg_path: str = './assets/q90_420.jpg'):
        self.qt = inspect(jpeg_path)['quantization_table0']
        self.qt = zigzag(self.qt, inverse=True)

    def __call__(self, sample):
        sample = np.asarray(sample)
        image = bwidct(sample)
        return image


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            # 正規化: t.sub_(m).div_(s)
            t.mul_(s).add_(m)
        return tensor


MEAN_GRAYSCALE = (0.5,)
STD_GRAYSCALE = (0.5,)

dataset_transforms = []

if ARGS.dataset == 'mnist' or ARGS.dataset == 'fashion_mnist':
    dataset_transforms.append(
        transforms.Pad(2, fill=0, padding_mode='constant'))
else:
    dataset_transforms.append(transforms.Grayscale())

if ARGS.form == 'jpeg_blocks':
    dataset_transforms.append(
        BlockwiseDCT())
    blockwise_idct = BlockwiseIDCT()

to_tensor = transforms.ToTensor()
dataset_transforms.append(to_tensor)

norm_funcs = {}

if ARGS.form == 'rgb_pixels':
    norm_funcs['rgb_pixels'] = (
        transforms.Normalize(MEAN_GRAYSCALE, STD_GRAYSCALE),
        Denormalize(MEAN_GRAYSCALE, STD_GRAYSCALE))
    dataset_transforms.append(norm_funcs['rgb_pixels'][0])


def load_dataset(dataset_name: str, transform=None):
    if isinstance(transform, (list, tuple)):
        transform = transforms.Compose(transform)
    if dataset_name == 'mnist':
        num_classes = 10
        dataset = dset.MNIST(
            root=ARGS.data_path, download=True, train=True,
            transform=transform)
    elif dataset_name == 'fashion_mnist':
        num_classes = 10
        dataset = dset.FashionMNIST(
            root=ARGS.data_path, download=True, train=True,
            transform=transform)
    return dataset, num_classes


dataset, NUM_CLASSES = load_dataset(ARGS.dataset, dataset_transforms)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=ARGS.batch_size, shuffle=True)

if ARGS.form == 'jpeg_blocks':
    images_preloaded = []
    labels_preloaded = []
    for images, labels in tqdm(
        dataloader,
        desc='DCT係数の事前読み込み中... ',
        total=len(dataloader),
        leave=False
    ):
        images_preloaded.append(images)
        labels_preloaded.append(labels)
    images_preloaded = torch.cat(images_preloaded)
    # STD_JPEG_BLOCKS, MEAN_JPEG_BLOCKS = \
    #     torch.std_mean(
    #         images_preloaded.permute(0, 2, 3, 1).reshape(-1, 64), dim=0)
    # norm_funcs['jpeg_blocks'] = (
    #     transforms.Normalize(MEAN_JPEG_BLOCKS, STD_JPEG_BLOCKS),
    #     Denormalize(MEAN_JPEG_BLOCKS, STD_JPEG_BLOCKS))
    # for image in images_preloaded:
    #     norm_funcs['jpeg_blocks'][0](image)
    labels_preloaded = torch.cat(labels_preloaded)
    dataset = torch.utils.data.TensorDataset(
        images_preloaded, labels_preloaded)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=ARGS.batch_size, shuffle=True)

INCEPTION_V3_PATH = Path(ARGS.data_path).joinpath('inception_v3')
fid = FrechetInceptionDistance(device=auto_device(1))
if ARGS.fid:
    NPZ_PATH = INCEPTION_V3_PATH\
        .joinpath(ARGS.dataset).joinpath(f'{ARGS.dataset}.npz')
    NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    if NPZ_PATH.exists():
        npz = np.load(NPZ_PATH)
        fid_real_features = npz['features']
        fid_labels = npz['labels']
    else:
        fid_dataset_transforms = [
            e for e in dataset_transforms
            if not isinstance(e, (BlockwiseDCT, transforms.Normalize))]
        fid_dataset, _ = load_dataset(ARGS.dataset, fid_dataset_transforms)
        fid_dataloader = torch.utils.data.DataLoader(
            fid_dataset, batch_size=ARGS.fid_batch_size, drop_last=True)
        pbar = tqdm(
            fid_dataloader,
            desc='[準備] Inception-V3を適用中... ',
            total=len(fid_dataloader),
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
    fid_num_batches = int(np.ceil(fid_labels.shape[0] / ARGS.fid_batch_size))
    fid_labels_batch = np.pad(
        fid_labels,
        [0, fid_num_batches * ARGS.fid_batch_size - fid_labels.shape[0]])\
        .reshape(fid_num_batches, ARGS.fid_batch_size).astype(np.int64)

if ARGS.fid_num_samples < 2048:
    ARGS.fid_num_samples = 2048

if ARGS.form == 'rgb_pixels':
    IMAGE_CHANNELS = 1
elif ARGS.form == 'jpeg_blocks':
    IMAGE_CHANNELS = 64

model_g = Generator(
    ARGS.z_dim, IMAGE_CHANNELS,
    bias=True, num_classes=NUM_CLASSES, form=ARGS.form).to(auto_device(0))
model_d = Discriminator(
    IMAGE_CHANNELS, num_classes=NUM_CLASSES, form=ARGS.form).to(auto_device(0))
# summary(model_g, (ARGS.z_dim,))
# summary(model_d, (1, 32, 32))

##################
# Optimizerの設定 #
##################
if ARGS.form == 'rgb_pixels':
    optim_g = optim.Adam(
        model_g.parameters(),
        lr=0.0016 * ARGS.batch_size / 1000,
        betas=[0.5, 0.999])
    optim_d = optim.Adam(
        model_d.parameters(),
        lr=0.0004 * ARGS.batch_size / 1000,
        betas=[0.5, 0.999])
elif ARGS.form == 'jpeg_blocks':
    optim_g = optim.Adam(
        model_g.parameters(),
        lr=0.002 * ARGS.batch_size / 1000,
        betas=[0.5, 0.999])
    optim_d = optim.Adam(
        model_d.parameters(),
        lr=0.00025 * ARGS.batch_size / 1000,
        betas=[0.5, 0.999])

example_labels = \
    torch.arange(start=0, end=NUM_CLASSES, step=1, device=auto_device(0))\
    .view(-1, 1).repeat(1, ARGS.num_samples).flatten()
example_z = torch.randn(
    example_labels.size()[0], ARGS.z_dim, device=auto_device(0))

criterion_module = import_module(f'criterions.{ARGS.criterion}')

criterion = criterion_module.Criterion(ARGS.batch_size, auto_device(0))


def to_images(tensors, form: str = 'rgb_pixels', norm_funcs=None):
    if form in norm_funcs:
        denormalize = norm_funcs[form][1]
        for tensor in tensors:
            denormalize(tensor)
    if form == 'rgb_pixels':
        images = tensors.permute(0, 2, 3, 1).numpy()
    elif form == 'jpeg_blocks':
        coefs = tensors.permute(0, 2, 3, 1).numpy()
        images = np.zeros((coefs.shape[0], 32, 32, 1))
        for i, coef in enumerate(coefs):
            images[i] = np.expand_dims(blockwise_idct(coef), -1)
            images[i] /= images[i].max()
    images *= 255
    return images.clip(0, 255).astype(np.uint8)


f_results = open(
    OUTPUT_DIR.joinpath('results.csv'), mode='w', encoding='utf-8')
csv_writer = csv.writer(f_results, lineterminator='\n')
csv_writer.writerow([
    'Epoch', 'Generator Loss Mean', 'Discriminator Loss Mean',
    'Epoch Duration', 'FID', 'Sample Image File',
    'Generator Model File', 'Discriminator Model File',
])
f_results.flush()

csv_idx = {
    'Epoch': 0, 'Generator Loss Mean': 1, 'Discriminator Loss Mean': 2,
    'Epoch Duration': 3, 'FID': 4, 'Sample Image File': 5,
    'Generator Model File': 6, 'Discriminator Model File': 7,
}

fig, ax = plt.subplots(1, 1)
for epoch in range(ARGS.epochs):
    results = ['' for _ in range(len(csv_idx))]
    results[csv_idx['Epoch']] = f'{epoch + 1}'

    log_loss_g, log_loss_d = [], []

    pbar = tqdm(
        enumerate(dataloader),
        desc=f'[{epoch+1}/{ARGS.epochs}] 訓練開始',
        total=len(dataset)//ARGS.batch_size,
        leave=False)
    begin_time = perf_counter()  # 時間計測開始
    for i, (real_images, labels) in pbar:
        real_images = real_images.to(auto_device(0))
        labels = labels.to(auto_device(0))

        #######################################################################
        # Discriminatorの訓練
        #######################################################################
        model_d.zero_grad()
        output_d_real = model_d(real_images, labels)
        loss_d_real = criterion(output_d_real, real=True, generator=False)
        loss_d_real.backward()

        z = torch.randn(ARGS.batch_size, ARGS.z_dim).to(auto_device(0))
        fake_images = model_g(z, labels)
        output_d_fake = model_d(fake_images, labels)
        loss_d_fake = criterion(output_d_fake, real=False, generator=False)
        loss_d_fake.backward(retain_graph=True)  # 同じ勾配をGeneratorでも利用する
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
            f'[{epoch+1}/{ARGS.epochs}] 訓練中... '
            f'<損失: (G={loss_g.item():.016f}, D={loss_d.item():.016f})>')
    end_time = perf_counter()  # 時間計測終了
    pbar.close()

    loss_g_mean = np.mean(log_loss_g)
    loss_d_mean = np.mean(log_loss_d)
    results[csv_idx['Generator Loss Mean']] = f'{loss_g_mean:.016f}'
    results[csv_idx['Discriminator Loss Mean']] = f'{loss_d_mean:.016f}'

    epoch_duration = end_time - begin_time
    results[csv_idx['Epoch Duration']] = f'{epoch_duration:.07f}'

    print(
        f'[{epoch+1}/{ARGS.epochs}] 訓練完了. '
        f'<エポック処理時間: {epoch_duration:.07f}[s/epoch]'
        f', 平均損失: (G={loss_g_mean:.016f}, D={loss_d_mean:.016f})>')

    # FID
    if ARGS.fid and (
            epoch == 0
            or (epoch + 1) % ARGS.fid_frequency == 0
            or epoch == ARGS.epochs - 1):
        pbar = tqdm(
            fid_labels_batch,
            desc=f'[{epoch+1}/{ARGS.epochs}] 画像を生成中... ',
            total=fid_labels_batch.shape[0],
            leave=False)
        fid_fake_features = []
        with torch.no_grad():
            for labels in pbar:
                z = torch.randn(
                    ARGS.fid_batch_size, ARGS.z_dim, device=auto_device(0))
                fake_images = to_images(
                    model_g(z, torch.from_numpy(labels).to(auto_device(0)))
                    .detach().cpu(),
                    form=ARGS.form, norm_funcs=norm_funcs)
                fake_images = torch.stack([
                    to_tensor(image) for image in fake_images])
                fid_fake_features.append(fid.get_features(fake_images))
        fid_fake_features = np.concatenate(fid_fake_features)
        pbar.set_description_str(f'[{epoch+1}/{ARGS.epochs}] FID計算中... ')
        fid_score = fid(fid_real_features, fid_fake_features)
        pbar.close()
        results[csv_idx['FID']] = f'{fid_score:.05f}'
        print(f'[{epoch+1}/{ARGS.epochs}] FID計算完了. <FID: {fid_score:.05f}>')

    if not ARGS.no_sample and (
            epoch == 0
            or (epoch + 1) % ARGS.sample_frequency == 0
            or epoch == ARGS.epochs - 1):
        with torch.no_grad():
            example_images = model_g(example_z, example_labels).detach().cpu()
        example_images = to_images(
            example_images, form=ARGS.form, norm_funcs=norm_funcs)
        image_shape = example_images.shape[1:]

        example_images_tiled = tile_images(example_images, 10, 10)
        if ARGS.plot:
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

    if not ARGS.no_save and (
            (epoch + 1) % ARGS.save_frequency == 0
            or epoch == ARGS.epochs - 1):
        model_g_fname = f'generator_{epoch+1:06d}.pt'
        # TODO: Generatorのセーブを行う。
        results[csv_idx['Generator Model File']] = model_g_fname

        model_d_fname = f'discriminator_{epoch+1:06d}.pt'
        # TODO: Generatorのセーブを行う。
        results[csv_idx['Discriminator Model File']] = model_d_fname
    csv_writer.writerow(results)
    f_results.flush()
f_results.close()

f_measurements = open(
    OUTPUT_DIR.joinpath('measurements.txt'), mode='w', encoding='utf-8')

if ARGS.test:
    print('生成時間の計測を開始します。')
    with torch.no_grad():
        z = torch.zeros(
            ARGS.batch_size, ARGS.z_dim,
            device=auto_device(0), dtype=torch.float)
        labels = torch.zeros(
            ARGS.batch_size, device=auto_device(0), dtype=torch.long)
        begin_time = perf_counter()
        for _ in range(1000):
            torch.randn(ARGS.batch_size, ARGS.z_dim, out=z)
            torch.randint(0, NUM_CLASSES, (ARGS.batch_size,), out=labels)
            model_g(z, labels)
        end_time = perf_counter()
    tmpstr = f'生成時間(1バッチ{ARGS.batch_size}枚×{1000}回): '\
        f'{end_time - begin_time:.07f}[s]'
    print(tmpstr)
    f_measurements.write(tmpstr)
f_measurements.close()
