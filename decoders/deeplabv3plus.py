import torch, torch.nn as nn
from .common import SeparableConv2d


class ASPPSeparableConv(nn.Module):
    def __init__(self, kernels_in, kernels_out, padding, dilation, groups):
        super().__init__()
        self.conv = SeparableConv2d(
            kernels_in, kernels_out, padding=padding, dilation=dilation, groups=groups
        )
        self.bn = nn.BatchNorm2d(
            kernels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu = nn.ReLU()

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
        self.relu = nn.ReLU()

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
        self.block1 = nn.Sequential(
            nn.Conv2d(kernels_in, kernels_out, kernel_size=(1, 1)),
            nn.BatchNorm2d(
                kernels_out,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(),
        )
        self.conv1 = ASPPSeparableConv(
            kernels_in, kernels_out, padding=12, dilation=12, groups=512
        )
        self.conv2 = ASPPSeparableConv(
            kernels_in, kernels_out, padding=24, dilation=24, groups=512
        )
        self.conv3 = ASPPSeparableConv(
            kernels_in, kernels_out, padding=36, dilation=36, groups=512
        )
        self.pool = ASPPooling(kernels_in, kernels_out)
        self.project = nn.Sequential(
            nn.Conv2d(
                kernels_out * 5,
                kernels_out,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(
                kernels_out,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
        )

    def forward(self, x):
        print("\tASPP forward")
        print("\tx shape: ", x.shape)
        x1 = self.conv1(x)
        print("\tx1 shape: ", x1.shape)
        x2 = self.conv2(x)
        print("\tx2 shape: ", x2.shape)
        x3 = self.conv3(x)
        print("\tx3 shape: ", x3.shape)
        x4 = self.pool(x)
        print("\tx4 shape: ", x4.shape)
        x5 = self.block1(x)
        print("\tx5 shape: ", x5.shape)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        print("\tcat shape: ", x.shape)
        x = self.project(x)
        print("\tprojection shape: ", x.shape)
        return x


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, kernels_in, kernels_out, kernels=64) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            ASPP(kernels_in, kernels_out),
            SeparableConv2d(kernels_out, kernels_out, padding=1, groups=256),
            nn.BatchNorm2d(
                kernels_out,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(kernels, 48, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(
                48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            SeparableConv2d(304, 256, padding=1, groups=304),
            nn.BatchNorm2d(
                256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
        )
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=4.0)

    def forward(self, x, skip):
        print("DeepLabV3PlusDecoder forward: ")
        print("x shape: ", x.shape)
        print("skip shape: ", skip.shape)
        x = self.block1(x)
        print("block1 shape: ", x.shape)
        x = self.upsampling(x)
        print("upsampling shape: ", x.shape)
        s = self.block2(skip)
        print("skip block: ", s.shape)
        x = torch.cat([x, s], dim=1)
        print("cat shape: ", x.shape)
        x = self.block3(x)
        print("block2 shape: ", x.shape)
        return x
