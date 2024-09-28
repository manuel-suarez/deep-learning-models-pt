import torch, torch.nn as nn, torch.nn.functional as F
from .common import Conv2dReLU


class PSPBlock(nn.Module):
    def __init__(
        self, in_channels=512, out_channels=128, in_size=28, out_size=1, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=out_size),
            Conv2dReLU(in_channels, out_channels),
        )
        self.in_size = in_size
        self.out_Size = out_size

    def forward(self, x):
        x1 = self.block(x)
        x1 = F.interpolate(x1, size=self.in_size)
        return x1


class PSPModule(nn.Module):
    def __init__(
        self, in_channels=512, out_channels=128, in_size=28, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.block1 = PSPBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            in_size=in_size,
            out_size=1,
        )
        self.block2 = PSPBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            in_size=in_size,
            out_size=2,
        )
        self.block3 = PSPBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            in_size=in_size,
            out_size=3,
        )
        self.block4 = PSPBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            in_size=in_size,
            out_size=6,
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        x4 = self.block4(x)
        x = torch.cat([x1, x2, x3, x4, x])
        return x


class PSPDecoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module = PSPModule()
        self.conv2drelu = Conv2dReLU(1024, 512)
        self.dropout = nn.Dropout2d(p=0.2, inplace=False)

    def forward(self, x):
        x1 = self.module(x)
        x1 = self.conv2drelu(x)
        x1 = self.dropout(x)
        return x
