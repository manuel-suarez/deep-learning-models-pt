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
                kernels=[24, 144, 144, 6, 144, 48],
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
                kernels=[48, 288, 288, 12, 288, 88],
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
        )
        self.encoder_block5 = nn.Sequential(
            MBConvBlock(
                kernels=[120, 720, 720, 30, 720, 208],
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
        )

    def forward(self, x):
        c1 = self.encoder_block1(x)
        c2 = self.encoder_block2(c1)
        c3 = self.encoder_block3(c2)
        c4 = self.encoder_block4(c3)
        c5 = self.encoder_block5(c4)

        return c1, c2, c3, c4, c5