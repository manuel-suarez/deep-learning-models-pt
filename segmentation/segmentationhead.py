import torch.nn as nn
from .common import Activation


class SegmentationHead(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(
            16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.act = Activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x
