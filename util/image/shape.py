import numpy as np


def tile_images(imgs: np.ndarray, num_h: int, num_w: int) -> np.ndarray:
    tile = np.zeros(
        (num_h*imgs.shape[1], num_w*imgs.shape[2], imgs.shape[3]),
        dtype=np.uint8)
    k = 0
    for i in range(0, tile.shape[0], imgs.shape[1]):
        for j in range(0, tile.shape[1], imgs.shape[2]):
            tile[i:i+imgs.shape[1], j:j+imgs.shape[2]] = \
                imgs[k]
            k += 1
    return np.squeeze(tile)
