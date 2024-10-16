import torch.nn as nn
from .base import EfficientNetBaseEncoder
from .common import (
    InitBlock,
    MBConvBlock,
    repeat_mbconvblock,
    EfficientNetBaseEncoderBlock,
)


class EfficientNetEncoder(EfficientNetBaseEncoder):
    def __init__(self, in_channels=3, wavelets_mode=False, *args, **kwargs) -> None:
        super().__init__(in_channels, wavelets_mode, *args, **kwargs)
        self.encoder_block1 = InitBlock(in_channels=in_channels, out_channels=32)
        self.encoder_block2 = EfficientNetBaseEncoderBlock(
            [
                MBConvBlock(
                    kernels=[
                        32
                        + (1 if wavelets_mode == 2 else 0)
                        + (1 if wavelets_mode == 3 else 0),
                        8,
                        16,
                    ],
                    stride=1,
                    residual=False,
                ),
                MBConvBlock(
                    kernels=[16, 4, 16],
                    stride=1,
                    residual=True,
                ),
                MBConvBlock(
                    kernels=[16, 96, 4, 24],
                    stride=2,
                    residual=False,
                ),
                *repeat_mbconvblock(
                    kernels=[24, 144, 6, 24], stride=1, residual=True, blocks=2
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
                        6,
                        40,
                    ],
                    stride=2,
                    residual=False,
                ),
                *repeat_mbconvblock(
                    kernels=[40, 240, 10, 40], stride=1, residual=True, blocks=2
                ),
            ],
            wavelets_mode=wavelets_mode,
        )
        self.encoder_block4 = EfficientNetBaseEncoderBlock(
            [
                MBConvBlock(
                    kernels=[
                        40
                        + (1 if wavelets_mode == 2 else 0)
                        + (1 if wavelets_mode == 3 else 0),
                        240,
                        10,
                        80,
                    ],
                    stride=2,
                    residual=False,
                ),
                *repeat_mbconvblock(
                    kernels=[80, 480, 20, 80], stride=1, residual=True, blocks=3
                ),
                MBConvBlock(
                    kernels=[80, 480, 20, 112],
                    stride=1,
                    residual=False,
                ),
                *repeat_mbconvblock(
                    kernels=[112, 672, 28, 112], stride=1, residual=True, blocks=3
                ),
            ],
            wavelets_mode=wavelets_mode,
        )
        self.encoder_block5 = EfficientNetBaseEncoderBlock(
            [
                MBConvBlock(
                    kernels=[
                        112
                        + (1 if wavelets_mode == 2 else 0)
                        + (1 if wavelets_mode == 3 else 0),
                        672,
                        28,
                        192,
                    ],
                    stride=2,
                    residual=False,
                ),
                *repeat_mbconvblock(
                    kernels=[192, 1152, 48, 192], stride=1, residual=True, blocks=4
                ),
                MBConvBlock(
                    kernels=[192, 1152, 48, 320],
                    stride=1,
                    residual=False,
                ),
                MBConvBlock(
                    kernels=[320, 1920, 80, 320],
                    stride=1,
                    residual=True,
                ),
            ],
            wavelets_mode=wavelets_mode,
        )
