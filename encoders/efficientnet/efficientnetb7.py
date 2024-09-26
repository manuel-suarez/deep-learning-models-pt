import torch.nn as nn
from .common import InitBlock, MBConvBlock, repeat_mbconvblock


class EfficientNetEncoder(nn.Module):
    def __init__(self, in_channels=3) -> None:
        super().__init__()
        self.encoder_block1 = InitBlock(
            in_channels=in_channels,
            out_channels=64,
        )
        self.encoder_block2 = nn.Sequential(
            MBConvBlock(
                kernels=[64, 16, 32],
                stride=1,
                residual=False,
            ),
            *repeat_mbconvblock(
                kernels=[32, 8, 32],
                stride=1,
                residual=True,
                blocks=3,
            ),
            MBConvBlock(
                kernels=[32, 192, 8, 48],
                stride=2,
                residual=False,
            ),
            *repeat_mbconvblock(
                kernels=[48, 288, 12, 48],
                stride=1,
                residual=True,
                blocks=6,
            ),
        )
        self.encoder_block3 = nn.Sequential(
            MBConvBlock(
                kernels=[48, 288, 12, 80],
                stride=2,
                residual=False,
            ),
            *repeat_mbconvblock(
                kernels=[80, 480, 20, 80],
                stride=1,
                residual=True,
                blocks=6,
            ),
        )
        self.encoder_block4 = nn.Sequential(
            MBConvBlock(
                kernels=[80, 480, 20, 160],
                stride=2,
                residual=False,
            ),
            *repeat_mbconvblock(
                kernels=[160, 960, 40, 160],
                stride=1,
                residual=True,
                blocks=9,
            ),
            MBConvBlock(
                kernels=[160, 960, 40, 224],
                stride=1,
                residual=False,
            ),
            *repeat_mbconvblock(
                kernels=[224, 1344, 56, 224],
                stride=1,
                residual=True,
                blocks=9,
            ),
        )
        self.encoder_block5 = nn.Sequential(
            MBConvBlock(
                kernels=[224, 1244, 56, 384],
                stride=2,
                residual=False,
            ),
            *repeat_mbconvblock(
                kernels=[384, 2304, 96, 384],
                stride=1,
                residual=True,
                blocks=11,
            ),
            MBConvBlock(
                kernels=[384, 2304, 96, 640],
                stride=1,
                residual=False,
            ),
            *repeat_mbconvblock(
                kernels=[640, 3840, 160, 640],
                stride=1,
                residual=True,
                blocks=3,
            ),
        )

    def forward(self, x):
        c1 = self.encoder_block1(x)
        c2 = self.encoder_block2(c1)
        c3 = self.encoder_block3(c2)
        c4 = self.encoder_block4(c3)
        c5 = self.encoder_block5(c4)

        return c1, c2, c3, c4, c5