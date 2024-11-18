import torch.nn as nn


class Conv2dReLU(nn.Module):
    def __init__(self, kernels_in, kernels_out, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            kernels_in,
            kernels_out,
            kernel_size=(kernel_size, kernel_size),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
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


class SeparableConv2d(nn.Module):
    def __init__(
        self, kernels_in, kernels_out, padding, dilation=1, groups=512
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            kernels_in,
            kernels_in,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            kernels_in, kernels_out, kernel_size=(1, 1), stride=(1, 1), bias=False
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
