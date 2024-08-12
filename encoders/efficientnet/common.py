import torch, torch.nn as nn


class MBConvBlock(nn.Module):
    def __init__(
        self, kernels_inputs, kernels_outputs, stride=1, residual=False, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.conv2d1 = nn.Conv2d(
            kernels_inputs[0],
            kernels_outputs[0],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="same",
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(
            kernels_outputs[0],
            eps=1e-3,
            momentum=0.01,
            affine=True,
            track_running_stats=True,
        )
        self.act1 = nn.SiLU(inplace=True)
        self.has_block1a = kernels_inputs[1] != 0
        if kernels_inputs[1] != 0:
            padding = "same" if stride == 1 else (1, 1)
            self.conv2d1a = nn.Conv2d(
                kernels_inputs[1],
                kernels_outputs[1],
                kernel_size=(3, 3),
                stride=(stride, stride),
                padding=padding,
                bias=False,
            )
            self.bn1a = nn.BatchNorm2d(
                kernels_outputs[1],
                eps=1e-3,
                momentum=0.01,
                affine=True,
                track_running_stats=True,
            )
            self.act1a = nn.SiLU(inplace=True)
        self.conv2d2 = nn.Conv2d(
            kernels_inputs[2],
            kernels_outputs[2],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="same",
            bias=False,
        )
        self.act2 = nn.SiLU(inplace=True)
        self.conv2d3 = nn.Conv2d(
            kernels_inputs[3],
            kernels_outputs[3],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="same",
            bias=False,
        )
        self.act3 = nn.Sigmoid()
        self.conv2d4 = nn.Conv2d(
            kernels_inputs[4],
            kernels_outputs[4],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="same",
            bias=False,
        )
        self.bn4 = nn.BatchNorm2d(
            kernels_outputs[4],
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
