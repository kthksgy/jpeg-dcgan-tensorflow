import torch


class Criterion():
    def __init__(self, batch_size: int, device: torch.device):
        self.__ones = torch.ones(batch_size, 1, device=device)
        self.__zeros = torch.zeros(batch_size, 1, device=device)

    def __call__(self, outputs, real: bool, generator: bool):
        if generator:
            return -torch.mean(outputs)
        else:
            if real:
                return -torch.mean(torch.min(outputs - 1, self.__zeros))
            else:
                return - torch.mean(torch.min(-outputs - 1, self.__zeros))
