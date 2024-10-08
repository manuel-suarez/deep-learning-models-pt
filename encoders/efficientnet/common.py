import torch, torch.nn as nn
from models.encoders.base import BaseEncoderBlock


class InitBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(
            out_channels,
            eps=1e-3,
            momentum=0.01,
            affine=True,
            track_running_stats=True,
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MBConvBlock(nn.Module):
    def __init__(self, kernels, stride=1, residual=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Conversion between versions
        if len(kernels) == 6:
            if kernels[1] == 0:
                kernels = [kernels[0], kernels[3], kernels[5]]
            else:
                kernels = [kernels[0], kernels[1], kernels[3], kernels[5]]
        self.has_block1a = len(kernels) == 4
        self.conv2d1 = nn.Conv2d(
            kernels[0],
            kernels[1] if self.has_block1a else kernels[0],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="same",
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(
            kernels[1] if self.has_block1a else kernels[0],
            eps=1e-3,
            momentum=0.01,
            affine=True,
            track_running_stats=True,
        )
        self.act1 = nn.SiLU(inplace=True)
        if self.has_block1a:
            padding = "same" if stride == 1 else (1, 1)
            self.conv2d1a = nn.Conv2d(
                kernels[1],
                kernels[1],
                kernel_size=(3, 3),
                stride=(stride, stride),
                padding=padding,
                bias=False,
            )
            self.bn1a = nn.BatchNorm2d(
                kernels[1],
                eps=1e-3,
                momentum=0.01,
                affine=True,
                track_running_stats=True,
            )
            self.act1a = nn.SiLU(inplace=True)
        self.conv2d2 = nn.Conv2d(
            kernels[1] if self.has_block1a else kernels[0],
            kernels[2] if self.has_block1a else kernels[1],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="same",
            bias=False,
        )
        self.act2 = nn.SiLU(inplace=True)
        self.conv2d3 = nn.Conv2d(
            kernels[2] if self.has_block1a else kernels[1],
            kernels[1] if self.has_block1a else kernels[0],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="same",
            bias=False,
        )
        self.act3 = nn.Sigmoid()
        self.conv2d4 = nn.Conv2d(
            kernels[1] if self.has_block1a else kernels[0],
            kernels[3] if self.has_block1a else kernels[2],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="same",
            bias=False,
        )
        self.bn4 = nn.BatchNorm2d(
            kernels[3] if self.has_block1a else kernels[2],
            eps=1e-3,
            momentum=0.01,
            affine=True,
            track_running_stats=True,
        )
        self.residual = residual

    def forward(self, x):
        x1 = self.conv2d1(x)
        x1 = self.bn1(x1)
        x1 = self.act1(x1)
        if self.has_block1a:
            x1 = self.conv2d1a(x1)
            x1 = self.bn1a(x1)
            x1 = self.act1a(x1)
        x2 = nn.functional.adaptive_avg_pool2d(x1, 1)
        x2 = self.conv2d2(x2)
        x2 = self.act2(x2)
        x2 = self.conv2d3(x2)
        x2 = self.act3(x2)
        x3 = torch.mul(x1, x2)
        x3 = self.conv2d4(x3)
        x3 = self.bn4(x3)
        if self.residual:
            x = torch.add(x3, x)
        else:
            x = x3
        return x


def repeat_mbconvblock(kernels, stride=1, residual=True, blocks=1, *args, **kwargs):
    return [
        MBConvBlock(kernels, stride=stride, residual=residual, *args, **kwargs)
        for _ in range(blocks)
    ]


class EfficientNetBaseEncoderBlock(BaseEncoderBlock):
    def __init__(
        self, blocks, wavelets_mode=False, pool_mode=False, *args, **kwargs
    ) -> None:
        super().__init__(
            wavelets_mode=wavelets_mode, pool_mode=pool_mode, *args, **kwargs
        )
        self.block = nn.Sequential(*blocks)

    def forward(self, x, w=None):
        print("Efficientnet base encoder block forward")
        if w is not None:
            print(f"Wavelets mode level: {self.wavelets_mode}")
            x = torch.add(w, x)
        x = self.block(x)
        return x
