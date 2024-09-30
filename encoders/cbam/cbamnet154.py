import torch.nn as nn
from .common import CBAMBottleneck


class CBAMNetEncoder(nn.Module):
    def __init__(self, in_channels=3) -> None:
        super().__init__()
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(
                64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(
                64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(
                128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.encoder_block2 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
            ),
            nn.Sequential(
                CBAMBottleneck(128, 128, 256, has_downsample=True, stride=1),
                CBAMBottleneck(256, 128, 256, has_downsample=False, stride=1),
                CBAMBottleneck(256, 128, 256, has_downsample=False, stride=1),
            ),
        )
        self.encoder_block3 = nn.Sequential(
            CBAMBottleneck(256, 256, 512, has_downsample=True, stride=2, se_size=32),
            CBAMBottleneck(512, 256, 512, has_downsample=False, stride=1, se_size=32),
            CBAMBottleneck(512, 256, 512, has_downsample=False, stride=1, se_size=32),
            CBAMBottleneck(512, 256, 512, has_downsample=False, stride=1, se_size=32),
            CBAMBottleneck(512, 256, 512, has_downsample=False, stride=1, se_size=32),
            CBAMBottleneck(512, 256, 512, has_downsample=False, stride=1, se_size=32),
            CBAMBottleneck(512, 256, 512, has_downsample=False, stride=1, se_size=32),
            CBAMBottleneck(512, 256, 512, has_downsample=False, stride=1, se_size=32),
        )
        self.encoder_block4 = nn.Sequential(
            CBAMBottleneck(512, 512, 1024, has_downsample=True, stride=2, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
            CBAMBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        )
        self.encoder_block5 = nn.Sequential(
            CBAMBottleneck(
                1024, 1024, 2048, has_downsample=True, stride=2, se_size=128
            ),
            CBAMBottleneck(
                2048, 1024, 2048, has_downsample=False, stride=1, se_size=128
            ),
            CBAMBottleneck(
                2048, 1024, 2048, has_downsample=False, stride=1, se_size=128
            ),
        )

    def forward(self, x):
        c1 = self.encoder_block1(x)
        c2 = self.encoder_block2(c1)
        c3 = self.encoder_block3(c2)
        c4 = self.encoder_block4(c3)
        c5 = self.encoder_block5(c4)
        return c1, c2, c3, c4, c5