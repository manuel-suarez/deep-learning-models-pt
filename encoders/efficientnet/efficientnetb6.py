import torch.nn as nn
from .common import MBConvBlock


class EfficientNetEncoder(nn.Module):
    def __init__(self, in_channels=3) -> None:
        super().__init__()
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                56,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(
                56, eps=1e-3, momentum=0.01, affine=True, track_running_stats=True
            ),
            nn.SiLU(inplace=True),
        )
        # Estamos empezando el bloque 2
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
        )
        self.encoder_block3 = nn.Sequential(
            MBConvBlock(
                kernels=[40, 240, 240, 10, 240, 64],
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
        )
        self.encoder_block4 = nn.Sequential(
            MBConvBlock(
                kernels=[64, 384, 384, 16, 384, 128],
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
        )
        self.encoder_block5 = nn.Sequential(
            MBConvBlock(
                kernels=[176, 1056, 1056, 44, 1056, 304],
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
        )

    def forward(self, x):
        c1 = self.encoder_block1(x)
        c2 = self.encoder_block2(c1)
        c3 = self.encoder_block3(c2)
        c4 = self.encoder_block4(c3)
        c5 = self.encoder_block5(c4)

        return c1, c2, c3, c4, c5
