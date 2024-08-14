import torch.nn as nn


class Conv2dReLU(nn.Module):
    def __init__(self, kernels_in, kernels_out):
        super().__init__()
        self.conv = nn.Conv2d(
            kernels_in,
            kernels_out,
            kernel_size=(3, 3),
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
