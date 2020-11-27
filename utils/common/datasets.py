from typing import Callable, List, Optional, Union
import torchvision.datasets as dset
import torchvision.transforms as transforms


def __dataset_mnist(**kwargs):
    return dset.MNIST(
            root=kwargs['root'], train=kwargs['split'] == 'train',
            transform=kwargs['transform'],
            target_transform=kwargs['target_transform'],
            download=kwargs['download'])


def __dataset_cifar10(**kwargs):
    return dset.CIFAR10(
            root=kwargs['root'], train=kwargs['split'] == 'train',
            transform=kwargs['transform'],
            target_transform=kwargs['target_transform'],
            download=kwargs['download'])


def __dataset_cifar100(**kwargs):
    return dset.CIFAR100(
            root=kwargs['root'], train=kwargs['split'] == 'train',
            transform=kwargs['transform'],
            target_transform=kwargs['target_transform'],
            download=kwargs['download'])


__DATASET_FUNC_PREFIX = '__dataset_'
print({
    k[len(__DATASET_FUNC_PREFIX):]: v
    for k, v in globals().items()
    if k.startswith(__DATASET_FUNC_PREFIX)})


def load_dataset(
    name: str, root: str, train: bool = True,
    transform: Optional[Union[List[Callable], Callable]] = None,
    download: bool = True, 
):
    if isinstance(transform, (list, tuple)):
        transform = transforms.Compose(transform)
    if name == 'mnist':
        dataset = dset.MNIST(
            root=root, download=download, train=train,
            transform=transform)
    elif name == 'fashion_mnist':
        dataset = dset.FashionMNIST(
            root=root, download=download, train=train,
            transform=transform)
    elif name == 'cifar10':
        dataset = dset.CIFAR10(
            root=root, download=download, train=train,
            transform=transform)
    else:
        raise Exception(f'指定されたデータセット〈{name}〉は未実装です。')
    return dataset


if __name__ == '__main__':
    # 標準モジュール
    import argparse
    # コマンドライン引数を取得するパーサー
    parser = argparse.ArgumentParser(
        prog='Dataset Downloader',
        description='Torch Visionを使ってデータセットをダウンロードします。'
    )
    parser.add_argument(
        '--dataset', help='データセットを指定します。',
        type=str, default='mnist', choices=['mnist', 'fashion_mnist', 'cifar10']
    )
    parser.add_argument(
        '--data-path', help='データセットのパスを指定します。',
        type=str, default='./data'
    )
