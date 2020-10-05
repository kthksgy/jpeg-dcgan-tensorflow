import torch
import torch.nn as nn


class Criterion():
    def __init__(self, batch_size: int, device: torch.device):
        self.__ones = torch.ones(batch_size, 1).to(device)
        self.__zeros = torch.zeros(batch_size, 1).to(device)
        self.__loss_function = nn.BCELoss(reduction='mean')

    def __call__(self, outputs, real: bool, generator: bool):
        if real:
            return self.__loss_function(outputs, self.__ones)
        else:
            return self.__loss_function(outputs, self.__zeros)
