import numpy as np


def get_specific_labeled(images, labels, target_labels):
    extracted = []
    for image, label in zip(images, labels):
        if label not in target_labels:
            continue
        extracted.append(image)
    extracted_ndarray = np.array(extracted)
    return extracted_ndarray
