import torch.nn as nn
from .common import MBConvBlock


class EfficientNetEncoder(nn.Module):
    def __init__(self, in_channels=3) -> None:
        super().__init__()
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
        self.encoder_block2 = nn.Sequential(
            MBConvBlock(
                kernels=[48, 0, 48, 12, 48, 24],
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
            MBConvBlock(
                kernels=[32, 192, 192, 8, 192, 32],
                stride=1,
                residual=True,
            ),
        )
        self.encoder_block3 = nn.Sequential(
            MBConvBlock(
                kernels=[32, 192, 192, 8, 192, 56],
                stride=2,
                residual=False,
            ),
            MBConvBlock(
                kernels=[56, 336, 336, 14, 336, 56],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[56, 336, 336, 14, 336, 56],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[56, 336, 336, 14, 336, 56],
                stride=1,
                residual=True,
            ),
        )
        self.encoder_block4 = nn.Sequential(
            MBConvBlock(
                kernels=[56, 336, 336, 14, 336, 112],
                stride=2,
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
                kernels=[112, 672, 672, 28, 672, 160],
                stride=1,
                residual=False,
            ),
            MBConvBlock(
                kernels=[160, 960, 960, 40, 960, 160],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[160, 960, 960, 40, 960, 160],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[160, 960, 960, 40, 960, 160],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[160, 960, 960, 40, 960, 160],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[160, 960, 960, 40, 960, 160],
                stride=1,
                residual=True,
            ),
        )
        self.encoder_block5 = nn.Sequential(
            MBConvBlock(
                kernels=[160, 960, 960, 40, 960, 272],
                stride=2,
                residual=False,
            ),
            MBConvBlock(
                kernels=[272, 1632, 1632, 68, 1632, 272],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[272, 1632, 1632, 68, 1632, 272],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[272, 1632, 1632, 68, 1632, 272],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[272, 1632, 1632, 68, 1632, 272],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[272, 1632, 1632, 68, 1632, 272],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[272, 1632, 1632, 68, 1632, 272],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[272, 1632, 1632, 68, 1632, 272],
                stride=1,
                residual=True,
            ),
            MBConvBlock(
                kernels=[272, 1632, 1632, 68, 1632, 448],
                stride=1,
                residual=False,
            ),
            MBConvBlock(
                kernels=[448, 2688, 2688, 112, 2688, 448],
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