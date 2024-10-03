from .base import ResNetBaseEncoder
import torch.nn as nn
from .common import BasicBlock


class ResNetEncoder(ResNetBaseEncoder):
    def __init__(self, in_channels=3, *args, **kwargs) -> None:
        super().__init__(in_channels, *args, **kwargs)
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            ),
            nn.BatchNorm2d(
                64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.encoder_block2 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
            ),
            nn.Sequential(
                BasicBlock(64, 64, has_downsample=False, stride=1),
                BasicBlock(64, 64, has_downsample=False, stride=1),
            ),
        )
        self.encoder_block3 = nn.Sequential(
            BasicBlock(64, 128, has_downsample=True, stride=2),
            BasicBlock(128, 128, has_downsample=False, stride=1),
        )
        self.encoder_block4 = nn.Sequential(
            BasicBlock(128, 256, has_downsample=True, stride=2),
            BasicBlock(256, 256, has_downsample=False, stride=1),
        )
        self.encoder_block5 = nn.Sequential(
            BasicBlock(256, 512, has_downsample=True, stride=2),
            BasicBlock(512, 512, has_downsample=False, stride=1),
        )
