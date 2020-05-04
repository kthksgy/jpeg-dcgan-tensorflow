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

from train import train_step
from models.mnist_dct import (make_discriminator, make_generator)
from util.image.shape import tile_images
from util.jpeg.compression import (compress, decompress)
from util.jpeg.jpegfile import inspect
from util.jpeg.scanning import zigzag

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
# 画像生成に関する引数
ARGUMENT_PARSER.add_argument(
    '--target-labels', help='学習する画像のラベルを1つ以上指定します。',
    type=int, nargs='+', choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
)
ARGUMENT_PARSER.add_argument(
    '-z', '--z-dim', help='潜在空間のサイズを指定します。',
    type=int, default=100, metavar='Z'
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
    type=int, default=16, choices=[4, 16, 36, 64, 100]
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

OUTPUT_DIR = Path(LAUNCH_DATETIME.strftime('./outputs/mnist_%Y%m%d%H%M%S'))
OUTPUT_SAMPLE_DIR = OUTPUT_DIR.joinpath('samples')
OUTPUT_SAMPLE_DIR.mkdir(parents=True)
OUTPUT_MODEL_DIR = OUTPUT_DIR.joinpath('models')
OUTPUT_MODEL_DIR.mkdir(parents=True)

# 生成画像や訓練に関する定数の定義
NUM_SAMPLES_SQRT = int(np.sqrt(ARGS.num_samples))  # テストとして生成する画像数の平方根

# データのロード
train_data, test_data = keras.datasets.mnist.load_data()
train_images, train_labels = train_data
test_images, test_labels = test_data

qt = inspect('./assets/q70_420.jpg')['quantization_table0']
qt = zigzag(qt, inverse=True)

if ARGS.target_labels is None:
    num_target_images = train_images.shape[0] + test_images.shape[0]
else:
    num_target_images = 0
    for l in ARGS.target_labels:
        num_target_images += \
            np.count_nonzero(train_labels == l) + \
            np.count_nonzero(test_labels == l)

images = np.zeros((num_target_images, 4, 4, 64), dtype=np.float16)
k = 0
for image, label in \
    chain.from_iterable(
        [zip(train_images, train_labels), zip(test_images, test_labels)]):
    if ARGS.target_labels is not None and label not in ARGS.target_labels:
        continue
    images[k] = \
        compress(np.pad(image, [[2, 2], [2, 2]]), y_qt=qt)
    k += 1

del train_images
del train_labels
del test_images
del test_labels

# データセットの作成
dataset = \
    tf.data.Dataset.from_tensor_slices(images) \
    .shuffle(images.shape[0]) \
    .batch(ARGS.batch_size)

# 生成器と識別器の作成
generator = make_generator(ARGS.z_dim)
discriminator = make_discriminator(ARGS.z_dim)

# テスト画像を生成するためのシード値
seed = tf.random.normal([NUM_SAMPLES_SQRT ** 2, ARGS.z_dim])

# matplotlibで表示するための諸々
fig, axs = plt.subplots(NUM_SAMPLES_SQRT, NUM_SAMPLES_SQRT)
fig.subplots_adjust(left=0, right=1, hspace=0.1, wspace=-0.6)
axs = list(chain.from_iterable(axs))


# 訓練フェーズ
for epoch in range(ARGS.epochs):
    begin_time = perf_counter()
    for image_batch in dataset:
        gen_loss, dis_loss = \
            train_step(image_batch, ARGS.z_dim, generator, discriminator)
        print(f' - Generator Loss: {gen_loss:.6f}, ', end='')
        print(f'Discriminator Loss: {dis_loss:.6f}', end='\r')
    print(f'>> エポックが完了しました。 経過時間: {perf_counter() - begin_time:.7f}[s]')
    if not ARGS.no_sample or ARGS.plot:
        coefs = generator(seed, training=False).numpy()
        imgs = np.array(
            [decompress(c, y_qt=qt) for c in coefs], dtype=np.uint8)
        fig.suptitle(f'生成画像 ( {epoch+1:6d} エポック学習済み )', **PFP)
        if ARGS.plot:
            for ax, img in zip(axs, imgs):
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(np.squeeze(img), cmap='gray')
            plt.pause(0.01)
        if not ARGS.no_sample and epoch % ARGS.sample_frequency == 0:
            imgs = np.expand_dims(imgs, -1)
            cv2.imwrite(
                str(OUTPUT_SAMPLE_DIR.joinpath(f'{epoch+1:06d}.png')),
                tile_images(imgs, NUM_SAMPLES_SQRT, NUM_SAMPLES_SQRT))
        if epoch % ARGS.save_frequency == 0:
            generator.save(
                OUTPUT_MODEL_DIR.joinpath(f'generator_{epoch+1:06d}.h5'))
            discriminator.save(
                OUTPUT_MODEL_DIR.joinpath(f'discriminator_{epoch+1:06d}.h5'))

cv2.imwrite(
    str(OUTPUT_SAMPLE_DIR.joinpath(f'{epoch+1:06d}.png')),
    tile_images(
        np.array(
            [decompress(coefs, y_qt=qt)
                for coefs in generator(seed, training=False).numpy()],
            dtype=np.uint8),
        NUM_SAMPLES_SQRT, NUM_SAMPLES_SQRT))
generator.save(
    OUTPUT_MODEL_DIR.joinpath(f'generator_{epoch+1:06d}.h5'))
discriminator.save(
    OUTPUT_MODEL_DIR.joinpath(f'discriminator_{epoch+1:06d}.h5'))
