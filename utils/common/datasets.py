from typing import Callable, List, Optional, Union
import torchvision.datasets as dset
import torchvision.transforms as transforms


def load_dataset(
    name: str, root: str, download: bool = True, train: bool = True,
    transform: Optional[Union[List[Callable], Callable]] = None
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
