from typing import Tuple, Union

import torch
from torch import nn
from torch.nn.utils import spectral_norm

from .basic_modules import ConditionalBatchNorm2d, SelfAttention


def init_xavier_uniform(layer):
    if hasattr(layer, "weight"):
        torch.nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if hasattr(layer.bias, "data"):
            layer.bias.data.fill_(0)


class GBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
        num_classes: int = 0
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride, padding, bias=bias)
        self.conv.apply(init_xavier_uniform)

        if num_classes == 0:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = ConditionalBatchNorm2d(out_channels, num_classes)
        self.activation = nn.LeakyReLU(0.3, True)

    def forward(self, inputs, labels=None):
        x = self.conv(inputs)
        if labels is not None:
            x = self.bn(x, labels)
        else:
            x = self.bn(x)
        return self.activation(x)


class Generator(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        bias: bool = True, num_classes: int = 0,
        form: str = 'rgb_pixels'
    ):
        super().__init__()
        self.form = form
        # (in_channels) -> (64, 4, 4)
        self.block1 = GBlock(
                in_channels, 64, 4, 1, padding=0,
                bias=bias, num_classes=num_classes)
        if self.form == 'rgb_pixels':
            # (64, 4, 4) -> (32, 8, 8)
            self.block2 = GBlock(
                    64, 32, 4, 2, padding=1,
                    bias=bias, num_classes=num_classes)
            # (32, 8, 8) -> (32, 16, 16)
            self.block3 = GBlock(
                    32, 32, 4, 2, padding=1,
                    bias=bias, num_classes=num_classes)
            # (32, 16, 16) -> (32, 32, 32)
            self.block4 = GBlock(
                    32, 16, 4, 2, padding=1,
                    bias=bias, num_classes=num_classes)
            self.out = nn.Sequential(
                # (32, 32, 32) -> (out_channels, 32, 32)
                nn.ConvTranspose2d(16, out_channels, 1, padding=0, bias=bias),
                nn.Tanh()
            )
        elif self.form == 'jpeg_blocks':
            # (64, 4, 4) -> (64, 4, 4)
            self.block2 = GBlock(
                    64, 64, 3, 1, padding=1,
                    bias=bias, num_classes=num_classes)
            self.block3 = None
            self.block4 = None
            self.out = nn.Sequential(
                SelfAttention(64),
                nn.LeakyReLU(0.3),
                # (64, 4, 4) -> (64, 4, 4)
                nn.ConvTranspose2d(
                    64, out_channels, 1, padding=0, bias=bias),
            )

    def forward(
        self, inputs, labels=None,
        form: str = 'rgb_pixels'
    ):
        x = inputs.view(inputs.size(0), inputs.size(1), 1, 1)
        x = self.block1(x, labels)
        x = self.block2(x, labels)
        # block3とblock4はJPEGブロックモードでは存在しない
        if self.block3 is not None:
            x = self.block3(x, labels)
        if self.block4 is not None:
            x = self.block4(x, labels)
        return self.out(x)


class DBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        num_classes: int = 0, prob_dropout: float = 0.5
    ):
        super().__init__()
        self.conv = spectral_norm(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size, stride, padding))
        self.dropout = nn.Dropout2d(prob_dropout)
        self.activation = nn.LeakyReLU(0.3, True)

        self.conv.apply(init_xavier_uniform)

    def forward(self, inputs):
        return self.activation(self.dropout(self.conv(inputs)))


class Discriminator(nn.Module):
    def __init__(
        self, in_channels: int,
        num_classes: int = 0, form: str = 'rgb_pixels'
    ):
        super().__init__()
        self.form = form
        self.__sequential = []
        if self.form == 'rgb_pixels':
            self.__sequential.extend([
                # (in_channels, 32, 32) -> (32, 16, 16)
                DBlock(in_channels, 32, 5, stride=2, padding=2),
                # (32, 16, 16) -> (48, 8, 8)
                DBlock(32, 48, 5, stride=2, padding=2),
                # (48, 8, 8) -> (64, 4, 4)
                DBlock(48, 64, 3, stride=2, padding=1),
                # (64, 4, 4) -> (96, 2, 2)
                DBlock(64, 96, 3, stride=2, padding=1),
            ])
        elif self.form == 'jpeg_blocks':
            self.__sequential.extend([
                # (64, 4, 4) -> (96, 2, 2)
                DBlock(in_channels, 96, 3, stride=2, padding=1),
            ])
        self.__sequential.extend([
            # (96, 2, 2) -> (128, 1, 1)
            spectral_norm(nn.Conv2d(96, 128, 2))
        ])

        self.main = nn.Sequential(*self.__sequential)
        self.last = nn.Linear(128, 1)
        if num_classes > 0:
            self.sn_embedding = spectral_norm(nn.Embedding(num_classes, 128))
            nn.init.xavier_normal_(self.sn_embedding.weight, gain=1)
        else:
            self.sn_embedding = None

    def forward(
        self, inputs, labels=None, form: str = 'rgb_pixels'
    ):
        h = self.main(inputs)
        # cGANs with Projection Discriminator
        h = h.view(inputs.size(0), -1)
        if labels is not None:
            h *= self.sn_embedding(labels)
        # Real(1) or Fake(0)を出力する
        return torch.sigmoid(self.last(h))
