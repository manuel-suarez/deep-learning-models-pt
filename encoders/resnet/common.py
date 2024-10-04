import torch, torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(
        self, kernels_in, kernels_out, stride=1, has_downsample=False, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # Conv+BatchNorm+ReLU
        self.conv1 = nn.Conv2d(
            kernels_in,
            kernels_out,
            kernel_size=(
                3,
                3,
            ),
            stride=(stride, stride),
            padding=(1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(
            kernels_out, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            kernels_out,
            kernels_out,
            kernel_size=(
                3,
                3,
            ),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(
            kernels_out, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.has_downsample = has_downsample
        if has_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    kernels_in,
                    kernels_out,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(
                    kernels_out,
                    eps=1e-5,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
            )

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        if self.has_downsample:
            x2 = self.downsample(x)
            x = torch.add(x1, x2)
        else:
            x = torch.add(x1, x)
        x = self.relu2(x)
        return x


class Bottleneck(nn.Module):
    def __init__(
        self,
        kernels_in,
        kernels_bt,
        kernels_out,
        stride=1,
        has_downsample=False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # Conv+BatchNorm+ReLU
        self.conv1 = nn.Conv2d(
            kernels_in,
            kernels_bt,
            kernel_size=(
                3,
                3,
            ),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(
            kernels_bt, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            kernels_bt,
            kernels_bt,
            kernel_size=(
                3,
                3,
            ),
            stride=(stride, stride),
            padding=(1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(
            kernels_bt, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            kernels_bt,
            kernels_out,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(
            kernels_out, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False
        )
        self.relu3 = nn.ReLU(inplace=True)

        self.has_downsample = has_downsample
        if has_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    kernels_in,
                    kernels_out,
                    kernel_size=(3, 3),
                    stride=(stride, stride),
                    padding=(1, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(
                    kernels_out,
                    eps=1e-5,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
            )

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = self.conv3(x1)
        x1 = self.bn3(x1)
        if self.has_downsample:
            x2 = self.downsample(x)
            x = torch.add(x1, x2)
        else:
            x = torch.add(x1, x)
        x = self.relu3(x)
        return x


def repeat_bsconvblock(
    in_channels, out_channels, has_downsample, stride, num_blocks=1, *args, **kwargs
):
    return [
        BasicBlock(
            in_channels, out_channels, has_downsample=has_downsample, stride=stride
        )
        for _ in range(num_blocks)
    ]


def repeat_btconvblock(
    in_channels,
    bt_channels,
    out_channels,
    has_downsample,
    stride,
    num_blocks=1,
    *args,
    **kwargs
):
    return [
        Bottleneck(
            in_channels,
            bt_channels,
            out_channels,
            has_downsample=has_downsample,
            stride=stride,
            *args,
            **kwargs,
        )
        for _ in range(num_blocks)
    ]


class ResNetEncoderBasicBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_blocks=1, pool_block=False, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pool_block = pool_block
        if self.pool_block:
            self.pool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
            )
        self.block = nn.Sequential(
            BasicBlock(
                in_channels,
                out_channels,
                has_downsample=False if self.pool_block else True,
                stride=1 if self.pool_block else 2,
            ),
            *repeat_bsconvblock(
                out_channels,
                out_channels,
                has_downsample=False,
                stride=1,
                num_blocks=num_blocks,
            ),
        )

    def forward(self, x, w=None):
        if w is not None:
            print("x: ", x.shape)
            print("w: ", w.shape)
            x = torch.add(x, w)
        if self.pool_block:
            x = self.pool(x)
        x = self.block(x)
        return x


class ResNetEncoderBottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        bt_channels,
        out_channels,
        num_blocks=1,
        pool_block=False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pool_block = pool_block
        if self.pool_block:
            self.pool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
            )
        self.block = nn.Sequential(
            Bottleneck(
                in_channels,
                bt_channels,
                out_channels,
                has_downsample=True,
                stride=1 if self.pool_block else 2,
            ),
            *repeat_btconvblock(
                out_channels,
                bt_channels,
                out_channels,
                has_downsample=False,
                stride=1,
                num_blocks=num_blocks,
            ),
        )

    def forward(self, x, w=None):
        if w is not None:
            print("x: ", x.shape)
            print("w: ", w.shape)
            x = torch.add(x, w)
        if self.pool_block:
            x = self.pool(x)
        x = self.block(x)
        return x
