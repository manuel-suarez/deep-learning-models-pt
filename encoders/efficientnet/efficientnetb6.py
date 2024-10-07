import torch.nn as nn
from .base import EfficientNetBaseEncoder
from .common import InitBlock, MBConvBlock, repeat_mbconvblock


class EfficientNetEncoder(EfficientNetBaseEncoder):
    def __init__(self, in_channels=3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder_block1 = InitBlock(in_channels=in_channels, out_channels=56)
        self.encoder_block2 = nn.Sequential(
            MBConvBlock(
                kernels=[56, 14, 32],
                stride=1,
                residual=False,
            ),
            *repeat_mbconvblock(
                kernels=[32, 8, 32],
                stride=1,
                residual=True,
                blocks=2,
            ),
            MBConvBlock(
                kernels=[32, 192, 8, 40],
                stride=2,
                residual=False,
            ),
            *repeat_mbconvblock(
                kernels=[40, 240, 10, 40],
                stride=1,
                residual=True,
                blocks=5,
            ),
        )
        self.encoder_block3 = nn.Sequential(
            MBConvBlock(
                kernels=[40, 240, 10, 72],
                stride=2,
                residual=False,
            ),
            *repeat_mbconvblock(
                kernels=[72, 432, 18, 72],
                stride=1,
                residual=True,
                blocks=5,
            ),
        )
        self.encoder_block4 = nn.Sequential(
            MBConvBlock(
                kernels=[72, 432, 18, 144],
                stride=2,
                residual=False,
            ),
            *repeat_mbconvblock(
                kernels=[144, 864, 36, 144],
                stride=1,
                residual=True,
                blocks=7,
            ),
            MBConvBlock(
                kernels=[144, 864, 36, 200],
                stride=1,
                residual=False,
            ),
            *repeat_mbconvblock(
                kernels=[200, 1200, 50, 200],
                stride=1,
                residual=True,
                blocks=7,
            ),
        )
        self.encoder_block5 = nn.Sequential(
            MBConvBlock(
                kernels=[200, 1200, 50, 344],
                stride=2,
                residual=False,
            ),
            *repeat_mbconvblock(
                kernels=[344, 2064, 86, 344],
                stride=1,
                residual=True,
                blocks=9,
            ),
            MBConvBlock(
                kernels=[344, 2064, 86, 576],
                stride=1,
                residual=False,
            ),
            *repeat_mbconvblock(
                kernels=[576, 3456, 144, 576],
                stride=1,
                residual=True,
                blocks=2,
            ),
        )
