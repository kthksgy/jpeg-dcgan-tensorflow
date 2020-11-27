import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.transforms.jpeg import (
    ChromaSubsampling,
    BlockwiseDCT,
    JPEGQuantize
)

image = cv2.imread('./assets/cat.png')
image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
# image[:, :, 0] = 255
# image[:, :, 2] = 0

image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
chroma_subsampling = ChromaSubsampling(image.shape[:-1], '4:2:0')
image = chroma_subsampling(image)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.show()

y_region, cr_region, cb_region = chroma_subsampling.get_regions()
y = image[y_region[0][0]:y_region[0][1], y_region[1][0]:y_region[1][1]]
cr = image[cr_region[0][0]:cr_region[0][1], cr_region[1][0]:cr_region[1][1]]
cb = image[cb_region[0][0]:cb_region[0][1], cb_region[1][0]:cb_region[1][1]]

plt.imshow(y, cmap='gray', vmin=0, vmax=255)
plt.show()
plt.imshow(cr, cmap='gray', vmin=0, vmax=255)
plt.show()
plt.imshow(cb, cmap='gray', vmin=0, vmax=255)
plt.show()

bwdct = BlockwiseDCT(image.shape)

blocks = bwdct(image)

luma_region = y_region
chroma_region = (
    (cr_region[0][0], cb_region[1][1]),
    (cr_region[1][0], cb_region[1][1])
)

jpeg_quantize = JPEGQuantize(quality=1, source='jpeg_standard', luma_region=luma_region, chroma_region=chroma_region)

blocks = jpeg_quantize(blocks)

blocks = jpeg_quantize.inverse(blocks)

image = bwdct.inverse(blocks)

image = image.clip(0, 255).astype(np.uint8)

image = chroma_subsampling.inverse(image)

image = cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)

plt.imshow(image, vmin=0, vmax=255)
plt.show()
