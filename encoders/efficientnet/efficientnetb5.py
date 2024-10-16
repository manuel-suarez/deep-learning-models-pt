import torch.nn as nn
from .base import EfficientNetBaseEncoder
from .common import MBConvBlock, EfficientNetBaseEncoderBlock


class EfficientNetEncoder(EfficientNetBaseEncoder):
    def __init__(self, in_channels=3, wavelets_mode=False, *args, **kwargs) -> None:
        super().__init__(in_channels, wavelets_mode, *args, **kwargs)
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                48,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(
                48, eps=1e-3, momentum=0.01, affine=True, track_running_stats=True
            ),
            nn.SiLU(inplace=True),
        )
        self.encoder_block2 = EfficientNetBaseEncoderBlock(
            [
                MBConvBlock(
                    kernels=[
                        48
                        + (1 if wavelets_mode == 2 else 0)
                        + (1 if wavelets_mode == 3 else 0),
                        0,
                        48,
                        12,
                        48,
                        24,
                    ],
                    stride=1,
                    residual=False,
                ),
                MBConvBlock(
                    kernels=[24, 0, 24, 6, 24, 24],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[24, 0, 24, 6, 24, 24],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[24, 144, 144, 6, 144, 40],
                    stride=2,
                    residual=False,
                ),
                MBConvBlock(
                    kernels=[40, 240, 240, 10, 240, 40],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[40, 240, 240, 10, 240, 40],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[40, 240, 240, 10, 240, 40],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[40, 240, 240, 10, 240, 40],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[40, 240, 240, 10, 240, 40],
                    stride=1,
                    residual=True,
                ),
            ],
            wavelets_mode=wavelets_mode,
        )
        self.encoder_block3 = EfficientNetBaseEncoderBlock(
            [
                MBConvBlock(
                    kernels=[
                        40
                        + (1 if wavelets_mode == 2 else 0)
                        + (1 if wavelets_mode == 3 else 0),
                        240,
                        240,
                        10,
                        240,
                        64,
                    ],
                    stride=2,
                    residual=False,
                ),
                MBConvBlock(
                    kernels=[64, 384, 384, 16, 384, 64],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[64, 384, 384, 16, 384, 64],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[64, 384, 384, 16, 384, 64],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[64, 384, 384, 16, 384, 64],
                    stride=1,
                    residual=True,
                ),
            ],
            wavelets_mode=wavelets_mode,
        )
        self.encoder_block4 = EfficientNetBaseEncoderBlock(
            [
                MBConvBlock(
                    kernels=[
                        64
                        + (1 if wavelets_mode == 2 else 0)
                        + (1 if wavelets_mode == 3 else 0),
                        384,
                        384,
                        16,
                        384,
                        128,
                    ],
                    stride=2,
                    residual=False,
                ),
                MBConvBlock(
                    kernels=[128, 768, 768, 32, 768, 128],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[128, 768, 768, 32, 768, 128],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[128, 768, 768, 32, 768, 128],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[128, 768, 768, 32, 768, 128],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[128, 768, 768, 32, 768, 128],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[128, 768, 768, 32, 768, 128],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[128, 768, 768, 32, 768, 176],
                    stride=1,
                    residual=False,
                ),
                MBConvBlock(
                    kernels=[176, 1056, 1056, 44, 1056, 176],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[176, 1056, 1056, 44, 1056, 176],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[176, 1056, 1056, 44, 1056, 176],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[176, 1056, 1056, 44, 1056, 176],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[176, 1056, 1056, 44, 1056, 176],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[176, 1056, 1056, 44, 1056, 176],
                    stride=1,
                    residual=True,
                ),
            ],
            wavelets_mode=wavelets_mode,
        )
        self.encoder_block5 = EfficientNetBaseEncoderBlock(
            [
                MBConvBlock(
                    kernels=[
                        176
                        + (1 if wavelets_mode == 2 else 0)
                        + (1 if wavelets_mode == 3 else 0),
                        1056,
                        1056,
                        44,
                        1056,
                        304,
                    ],
                    stride=2,
                    residual=False,
                ),
                MBConvBlock(
                    kernels=[304, 1824, 1824, 76, 1824, 304],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[304, 1824, 1824, 76, 1824, 304],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[304, 1824, 1824, 76, 1824, 304],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[304, 1824, 1824, 76, 1824, 304],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[304, 1824, 1824, 76, 1824, 304],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[304, 1824, 1824, 76, 1824, 304],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[304, 1824, 1824, 76, 1824, 304],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[304, 1824, 1824, 76, 1824, 304],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[304, 1824, 1824, 76, 1824, 512],
                    stride=1,
                    residual=False,
                ),
                MBConvBlock(
                    kernels=[512, 3072, 3072, 128, 3072, 512],
                    stride=1,
                    residual=False,
                ),
                MBConvBlock(
                    kernels=[512, 3072, 3072, 128, 3072, 512],
                    stride=1,
                    residual=False,
                ),
            ],
            wavelets_mode=wavelets_mode,
        )
