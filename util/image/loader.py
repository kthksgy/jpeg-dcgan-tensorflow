import cv2
import numpy as np
from pathlib import Path


def load_images(paths, size=None, max_images=0, exts=['.jpg', '.png'],
                grayscale=False, alpha=True, recursive=False, explicit=False,
                verbose=0, with_filename=False):
    if not hasattr(paths, '__iter__'):
        paths = [paths]
    n = 0
    for _p in paths:
        if max_images > 0 and n >= max_images:
            break
        if not isinstance(_p, Path):
            _p = Path(_p)
        if _p.is_dir() and recursive:
            print('>> サブディレクトリを走査します。')
            print(f' - {_p}')
            try:
                for img, name in load_images(
                    _p.glob('*'), size=size, max_images=max_images,
                    exts=exts, grayscale=grayscale,
                    alpha=alpha, recursive=recursive, verbose=verbose,
                    with_filename=True
                ):
                    n += 1
                    yield (img, name) if with_filename else img
            except StopIteration:
                print('>> サブディレクトリの走査が完了しました。')
            continue
        if _p.suffix not in exts:
            continue
        if verbose > 0:
            print(
                '>> {}を読み込み中です。'.format(_p),
                end='\r' if verbose == 1 else '\n')
        img = cv2.imread(str(_p), cv2.IMREAD_UNCHANGED)
        if isinstance(size, tuple) or isinstance(size, list):
            img = cv2.resize(img, size)
        mask = None
        if img.ndim == 3 and img.shape[-1] == 4:
            mask = img[:, :, -1]
            img = img[:, :, :-1]
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if mask is not None and alpha:
            if img.ndim == 2:
                img = np.stack([img, mask], -1)
            else:
                mask = np.expand_dims(mask, -1)
                img = np.concatenate([img, mask], -1)
        if explicit and img.ndim == 2:
            img = np.expand_dims(img, -1)
        n += 1
        yield (img, _p.name) if with_filename else img
    print('')
