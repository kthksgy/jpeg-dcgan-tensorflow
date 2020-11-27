import numpy as np
from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms


class FrechetInceptionDistance:
    def __init__(self, inception_v3=None, device='cpu'):
        self.device = device
        if inception_v3 is not None:
            self.inception_v3 = inception_v3
        else:
            self.inception_v3 = nn.Sequential(
                *list(models.inception_v3(pretrained=True, aux_logits=False)
                            .children())[:-2]).to(self.device)
        self.inception_v3.eval()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            inplace=True)

    def __call__(self, real_vectors, fake_vectors):
        assert real_vectors.shape == fake_vectors.shape\
            and real_vectors.shape[0] >= 2048,\
            'FIDを計算する際は最低2048個の特徴ベクトルを用意する必要が有ります。'
        real_mean = np.mean(real_vectors, axis=0)
        real_cov = np.cov(real_vectors, rowvar=False)
        fake_mean = np.mean(fake_vectors, axis=0)
        fake_cov = np.cov(fake_vectors, rowvar=False)
        cov_mean, _ = sqrtm(np.dot(real_cov, fake_cov), disp=False)
        return np.sum((real_mean - fake_mean) ** 2)\
            + np.trace(real_cov) + np.trace(fake_cov)\
            - 2 * np.trace(cov_mean)

    def get_features(self, bat_images):
        with torch.no_grad():
            bat_images = F.interpolate(bat_images, (299, 299))
            if bat_images.size()[1] == 1:
                bat_images = bat_images.repeat(1, 3, 1, 1)
            map(self.normalize, bat_images)
            # tensors = self.inception_v3(bat_images.to(self.device)).detach()
            return np.squeeze(
                self.inception_v3(bat_images.to(self.device))
                .detach().cpu().numpy())
