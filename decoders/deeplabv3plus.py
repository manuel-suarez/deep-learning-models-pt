import torch, torch.nn as nn
from .common import SeparableConv2d


class ASPPSeparableConv(nn.Module):
    def __init__(self, kernels_in, kernels_out, padding, dilation):
        super().__init__()
        self.conv = SeparableConv2d(
            kernels_in, kernels_out, padding=padding, dilation=dilation
        )
        self.bn = nn.BatchNorm2d(
            kernels_out, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPPooling(nn.Module):
    def __init__(self, kernels_in, kernels_out) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Conv2d(
            kernels_in, kernels_out, kernel_size=(1, 1), stride=(1, 1), bias=False
        )
        self.bn = nn.BatchNorm2d(
            kernels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = nn.functional.interpolate(x, scale_factor=14)
        return x


class ASPP(nn.Module):
    def __init__(self, kernels_in, kernels_out) -> None:
        super().__init__()
        self.conv1 = ASPPSeparableConv(kernels_in, kernels_out, padding=12, dilation=12)
        self.conv2 = ASPPSeparableConv(kernels_in, kernels_out, padding=24, dilation=24)
        self.conv3 = ASPPSeparableConv(kernels_in, kernels_out, padding=36, dilation=36)
        self.pool = ASPPooling(kernels_in, kernels_out)
        self.block = nn.Sequential(
            nn.Conv2d(kernels_in, kernels_out), nn.BatchNorm2d(kernels_out), nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(kernels_out * 5, kernels_out),
            nn.BatchNorm2d(kernels_out),
            nn.ReLU(),
            nn.Dropout(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.pool(x)
        x5 = self.block(x)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.block2(x)
        return x


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, kernels_in, kernels_out) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            ASPP(kernels_in, kernels_out),
            SeparableConv2d(kernels_out, kernels_out, padding=1, groups=256),
            nn.BatchNorm2d(
                kernels_out,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=(3, 3)),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            SeparableConv2d(304, 256, padding=1, groups=304),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=4.0, mode="bilinear")

    def forward(self, x, skip):
        x = self.block1(x)
        x = self.upsampling(x)
        s = self.block2(skip)
        x = torch.cat([x, s], dim=1)
        x = self.block3(x)
        return x
