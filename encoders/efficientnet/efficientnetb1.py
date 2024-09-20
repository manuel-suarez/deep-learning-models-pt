import torch.nn as nn
from .common import MBConvBlock


class EfficientNetEncoder(nn.Module):
    def __init__(self, in_channels=3) -> None:
        super().__init__()
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
        self.encoder_block2 = nn.Sequential(
            MBConvBlock(
                kernels=[32, 0, 32, 8, 32, 16],
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
        )
        self.encoder_block3 = nn.Sequential(
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
        )
        self.encoder_block4 = nn.Sequential(
            MBConvBlock(
                kernels=[40, 240, 240, 10, 240, 80],
                stride=2,
                residual=False,
            ),
            MBConvBlock(
                kernels=[80, 480, 480, 20, 480, 80],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[80, 480, 480, 20, 480, 80],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[80, 480, 480, 20, 480, 80],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[80, 480, 480, 20, 480, 112],
                stride=1,
                residual=False,
            ),
            MBConvBlock(
                kernels=[112, 672, 672, 28, 672, 112],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[112, 672, 672, 28, 672, 112],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[112, 672, 672, 28, 672, 112],
                stride=1,
                residual=True,
            ),
        )
        self.encoder_block5 = nn.Sequential(
            MBConvBlock(
                kernels=[112, 672, 672, 28, 672, 192],
                stride=2,
                residual=False,
            ),
            MBConvBlock(
                kernels=[192, 1152, 1152, 48, 1152, 192],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[192, 1152, 1152, 48, 1152, 192],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[192, 1152, 1152, 48, 1152, 192],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[192, 1152, 1152, 48, 1152, 192],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[192, 1152, 1152, 48, 1152, 320],
                stride=1,
                residual=False,
            ),
            MBConvBlock(
                kernels=[320, 1920, 1920, 80, 1920, 320],
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
