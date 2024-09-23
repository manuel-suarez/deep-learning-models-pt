import torch.nn as nn
from .common import InitBlock, MBConvBlock, repeat_mbconvblock


class EfficientNetEncoder(nn.Module):
    def __init__(self, in_channels=3) -> None:
        super().__init__()
        self.encoder_block1 = InitBlock(in_channels=in_channels, out_channels=32)
        self.encoder_block2 = nn.Sequential(
            MBConvBlock(
                kernels=[32, 8, 16],
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
        )
        self.encoder_block3 = nn.Sequential(
            MBConvBlock(
                kernels=[24, 144, 6, 40],
                stride=2,
                residual=False,
            ),
            *repeat_mbconvblock(
                kernels=[40, 240, 10, 40], stride=1, residual=True, blocks=2
            ),
        )
        self.encoder_block4 = nn.Sequential(
            MBConvBlock(
                kernels=[40, 240, 10, 80],
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
        )
        self.encoder_block5 = nn.Sequential(
            MBConvBlock(
                kernels=[112, 672, 28, 192],
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
        )

    def forward(self, x):
        c1 = self.encoder_block1(x)
        c2 = self.encoder_block2(c1)
        c3 = self.encoder_block3(c2)
        c4 = self.encoder_block4(c3)
        c5 = self.encoder_block5(c4)

        return c1, c2, c3, c4, c5
