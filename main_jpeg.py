from argparse import ArgumentParser
from datetime import datetime
from itertools import chain
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from util.data.mod import get_specific_labeled
from util.image.shape import tile_images
from util.jpeg.transform import (bwdct, bwidct)
from util.jpeg.jpegfile import inspect
from util.jpeg.scanning import zigzag

from train import train_step
from models.mnist_dct import (make_discriminator, make_generator)

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
# 画像生成に関する引数
ARGUMENT_PARSER.add_argument(
    '--target-labels', help='学習する画像のラベルを1つ以上指定します。',
    type=int, nargs='+', choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
)
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
    '--num-samples', help='結果を見るためのサンプル数を指定します。',
    type=int, default=16
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
ARGS = ARGUMENT_PARSER.parse_args()

# Matplotlibの設定
pfp = \
    matplotlib \
    .font_manager \
    .FontProperties(fname='./fonts/KintoSans-Bold.ttf')
PFP = {'fontproperties': pfp}
matplotlib.rcParams['savefig.dpi'] = 350

OUTPUT_DIR = Path(
    LAUNCH_DATETIME.strftime(f'./outputs/{ARGS.dataset}_jpeg_%Y%m%d%H%M%S'))
OUTPUT_SAMPLE_DIR = OUTPUT_DIR.joinpath('samples')
OUTPUT_SAMPLE_DIR.mkdir(parents=True)
OUTPUT_MODEL_DIR = OUTPUT_DIR.joinpath('models')
OUTPUT_MODEL_DIR.mkdir(parents=True)

# データのロード
if ARGS.dataset == 'mnist':
    train_data, test_data = keras.datasets.mnist.load_data()
elif ARGS.dataset == 'fashion_mnist':
    train_data, test_data = keras.datasets.fashion_mnist.load_data()
train_images, train_labels = train_data
test_images, test_labels = test_data

all_images = np.concatenate([train_images, test_images])
all_images = np.pad(all_images, [[0, 0], [2, 2], [2, 2]])
all_labels = np.concatenate([train_labels, test_labels])

if ARGS.target_labels is not None:
    images = get_specific_labeled(all_images, all_labels, ARGS.target_labels)
else:
    images = all_images

qt = inspect('./assets/q90_420.jpg')['quantization_table0']
qt = zigzag(qt, inverse=True)

compressed = np.zeros(
    (
        images.shape[0],
        int(np.ceil(images.shape[1] / 8.0)),
        int(np.ceil(images.shape[1] / 8.0)),
        8 * 8
    ), dtype=np.float32)

for i, image in enumerate(images):
    compressed[i] = bwdct(image, qt=qt, trunc_only=True)

# データセットの作成
dataset = \
    tf.data.Dataset.from_tensor_slices(compressed) \
    .shuffle(compressed.shape[0]) \
    .batch(ARGS.batch_size)

# 生成器と識別器の作成
generator = make_generator(ARGS.z_dim)
discriminator = make_discriminator(ARGS.z_dim)

# テスト画像を生成するためのシード値
seed = tf.random.normal([ARGS.num_samples, ARGS.z_dim])

# matplotlibで表示するための諸々
fig, axs = plt.subplots(1, ARGS.num_samples)
# fig.subplots_adjust(left=0, right=1, hspace=0.1, wspace=-0.6)
# axs = list(chain.from_iterable(axs))

output_images = np.zeros(
    (
        ARGS.num_samples,
        images.shape[1],
        images.shape[2],
    ), dtype=np.uint8
)

epoch_durations = []

# 訓練フェーズ
for epoch in range(ARGS.epochs):
    begin_time = perf_counter()
    for image_batch in dataset:
        gen_loss, dis_loss = \
            train_step(image_batch, ARGS.z_dim, generator, discriminator)
        print(f' - Generator Loss: {gen_loss:.6f}, ', end='')
        print(f'Discriminator Loss: {dis_loss:.6f}', end='\r')
    end_time = perf_counter()
    epoch_duration = end_time - begin_time
    epoch_durations.append(epoch_duration)
    print(f'\n>> {epoch+1}エポックが完了しました。 経過時間: {epoch_duration:.7f}[s]')
    if not ARGS.no_sample and (
            epoch == 0
            or (epoch + 1) % ARGS.sample_frequency == 0
            or epoch == ARGS.epochs - 1):
        coefs = generator(seed, training=False).numpy()
        print('出力: ', np.min(coefs), np.max(coefs), np.mean(coefs), np.std(coefs))
        for i, coef in enumerate(coefs):
            output_images[i] = bwidct(coef)
        print('出力: ', np.min(output_images), np.max(output_images), np.mean(output_images), np.std(output_images))
        fig.suptitle(f'生成画像 ( {epoch+1:6d} エポック学習済み )', **PFP)
        if ARGS.plot:
            for ax, img in zip(axs, output_images):
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(np.squeeze(img), cmap='gray')
            plt.pause(0.01)
        cv2.imwrite(
            str(OUTPUT_SAMPLE_DIR.joinpath(f'{epoch+1:06d}.png')),
            tile_images(output_images, 1, ARGS.num_samples))
    if not ARGS.no_save and (
            (epoch + 1) % ARGS.save_frequency == 0
            or epoch == ARGS.epochs - 1):
        generator.save(
            OUTPUT_MODEL_DIR.joinpath(f'generator_{epoch+1:06d}.h5'))
        discriminator.save(
            OUTPUT_MODEL_DIR.joinpath(f'discriminator_{epoch+1:06d}.h5'))

print(f'>> 平均エポック時間: {np.mean(epoch_durations[1:]):.7f}[s]')

generation_durations = []
for i in range(1000):
    print(f'>> 生成時間検証のために画像生成中... [{i+1}/1000]', end='\r')
    begin_time = perf_counter()
    test_seed = tf.random.normal([1000, ARGS.z_dim])
    coefs = generator(test_seed, training=False)
    end_time = perf_counter()
    generation_durations.append(end_time - begin_time)
print(f'\n>> 生成時間(1000枚): {np.mean(generation_durations):.7f}[s]')
