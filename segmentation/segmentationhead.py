import torch.nn as nn
from .common import Activation


class SegmentationHead(nn.Module):
    def __init__(self, kernels_in=16, has_activation=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(
            kernels_in, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.has_activation = has_activation
        if self.has_activation:
            self.act = Activation()

    def forward(self, x):
        x = self.conv(x)
        if self.has_activation:
            x = self.act(x)
        return x
