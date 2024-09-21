import torch.nn as nn
from .common import MBConvBlock


class EfficientNetEncoder(nn.Module):
    def __init__(self, in_channels=3) -> None:
        super().__init__()
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                40,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(
                40, eps=1e-3, momentum=0.01, affine=True, track_running_stats=True
            ),
            nn.SiLU(inplace=True),
        )
        self.encoder_block2 = nn.Sequential(
            MBConvBlock(
                kernels=[40, 0, 40, 10, 40, 24],
                stride=1,
                residual=False,
            ),
            MBConvBlock(
                kernels=[24, 0, 24, 6, 24, 24],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[24, 144, 144, 6, 144, 32],
                stride=2,
                residual=False,
            ),
            MBConvBlock(
                kernels=[32, 192, 192, 8, 192, 32],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[32, 192, 192, 8, 192, 32],
                stride=1,
                residual=True,
            ),
        )
        self.encoder_block3 = nn.Sequential(
            MBConvBlock(
                kernels=[32, 192, 192, 8, 192, 48],
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
        )
        self.encoder_block4 = nn.Sequential(
            MBConvBlock(
                kernels=[48, 288, 288, 12, 288, 96],
                stride=2,
                residual=False,
            ),
            MBConvBlock(
                kernels=[96, 576, 576, 24, 576, 96],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[96, 576, 576, 24, 576, 96],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[96, 576, 576, 24, 576, 96],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[96, 576, 576, 24, 576, 96],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[96, 576, 576, 24, 576, 136],
                stride=1,
                residual=False,
            ),
            MBConvBlock(
                kernels=[136, 816, 816, 34, 816, 136],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[136, 816, 816, 34, 816, 136],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[136, 816, 816, 34, 816, 136],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[136, 816, 816, 34, 816, 136],
                stride=1,
                residual=True,
            ),
        )
        self.encoder_block5 = nn.Sequential(
            MBConvBlock(
                kernels=[136, 816, 816, 34, 816, 232],
                stride=2,
                residual=False,
            ),
            MBConvBlock(
                kernels=[232, 1392, 1392, 58, 1392, 232],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[232, 1392, 1392, 58, 1392, 232],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[232, 1392, 1392, 58, 1392, 232],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[232, 1392, 1392, 58, 1392, 232],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[232, 1392, 1392, 58, 1392, 232],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[232, 1392, 1392, 58, 1392, 384],
                stride=1,
                residual=False,
            ),
            MBConvBlock(
                kernels=[384, 2304, 2304, 96, 2304, 384],
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
