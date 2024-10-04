from .base import ResNetBaseEncoder
import torch.nn as nn
from .common import Bottleneck


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
                Bottleneck(64, 64, 256, has_downsample=True, stride=1),
                Bottleneck(256, 64, 256, has_downsample=False, stride=1),
                Bottleneck(256, 64, 256, has_downsample=False, stride=1),
            ),
        )
        self.encoder_block3 = nn.Sequential(
            Bottleneck(256, 128, 512, has_downsample=True, stride=2),
            Bottleneck(512, 128, 512, has_downsample=False, stride=1),
            Bottleneck(512, 128, 512, has_downsample=False, stride=1),
            Bottleneck(512, 128, 512, has_downsample=False, stride=1),
        )
        self.encoder_block4 = nn.Sequential(
            Bottleneck(512, 256, 1024, has_downsample=True, stride=2),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
        )
        self.encoder_block5 = nn.Sequential(
            Bottleneck(1024, 512, 2048, has_downsample=True, stride=2),
            Bottleneck(2048, 512, 2048, has_downsample=False, stride=1),
            Bottleneck(2048, 512, 2048, has_downsample=False, stride=1),
        )
