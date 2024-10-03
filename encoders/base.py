import torch.nn as nn


class BaseEncoder(nn.Module):
    def __init__(self, in_channels=3, wavelets_mode=False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.wavelets_mode = wavelets_mode

    def forward(self, _):
        pass
