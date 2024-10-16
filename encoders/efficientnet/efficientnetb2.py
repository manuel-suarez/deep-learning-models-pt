import torch.nn as nn
from .base import EfficientNetBaseEncoder
from .common import MBConvBlock, EfficientNetBaseEncoderBlock


class EfficientNetEncoder(EfficientNetBaseEncoder):
    def __init__(self, in_channels=3, wavelets_mode=False, *args, **kwargs) -> None:
        super().__init__(in_channels, wavelets_mode, *args, **kwargs)
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(
                32, eps=1e-3, momentum=0.01, affine=True, track_running_stats=True
            ),
            nn.SiLU(inplace=True),
        )
        self.encoder_block2 = EfficientNetBaseEncoderBlock(
            [
                MBConvBlock(
                    kernels=[
                        32
                        + (1 if wavelets_mode == 2 else 0)
                        + (1 if wavelets_mode == 3 else 0),
                        0,
                        32,
                        8,
                        32,
                        16,
                    ],
                    stride=1,
                    residual=False,
                ),
                MBConvBlock(
                    kernels=[16, 0, 16, 4, 16, 16],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[16, 96, 96, 4, 96, 24],
                    stride=2,
                    residual=False,
                ),
                MBConvBlock(
                    kernels=[24, 144, 144, 6, 144, 24],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[24, 144, 144, 6, 144, 24],
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
                        24
                        + (1 if wavelets_mode == 2 else 0)
                        + (1 if wavelets_mode == 3 else 0),
                        144,
                        144,
                        6,
                        144,
                        48,
                    ],
                    stride=2,
                    residual=False,
                ),
                MBConvBlock(
                    kernels=[48, 288, 288, 12, 288, 48],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[48, 288, 288, 12, 288, 48],
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
                        48
                        + (1 if wavelets_mode == 2 else 0)
                        + (1 if wavelets_mode == 3 else 0),
                        288,
                        288,
                        12,
                        288,
                        88,
                    ],
                    stride=2,
                    residual=False,
                ),
                MBConvBlock(
                    kernels=[88, 528, 528, 22, 528, 88],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[88, 528, 528, 22, 528, 88],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[88, 528, 528, 22, 528, 88],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[88, 528, 528, 22, 528, 120],
                    stride=1,
                    residual=False,
                ),
                MBConvBlock(
                    kernels=[120, 720, 720, 30, 720, 120],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[120, 720, 720, 30, 720, 120],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[120, 720, 720, 30, 720, 120],
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
                        120
                        + (1 if wavelets_mode == 2 else 0)
                        + (1 if wavelets_mode == 3 else 0),
                        720,
                        720,
                        30,
                        720,
                        208,
                    ],
                    stride=2,
                    residual=False,
                ),
                MBConvBlock(
                    kernels=[208, 1248, 1248, 52, 1248, 208],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[208, 1248, 1248, 52, 1248, 208],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[208, 1248, 1248, 52, 1248, 208],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[208, 1248, 1248, 52, 1248, 208],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[208, 1248, 1248, 52, 1248, 352],
                    stride=1,
                    residual=False,
                ),
                MBConvBlock(
                    kernels=[352, 2112, 2112, 88, 2112, 352],
                    stride=1,
                    residual=True,
                ),
            ],
            wavelets_mode=wavelets_mode,
        )
