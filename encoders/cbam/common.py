import torch, torch.nn as nn
import torch.nn.functional as F
from models.encoders.base import BaseEncoderBlock


class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        kernel_size = 7
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Spatial channel compression (max and avg over channels)
        x1 = torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)),
            dim=1,
        )
        x1 = self.conv(x1)
        x1 = self.bn(x1)
        scale = F.sigmoid(x1)
        return x * scale


class ChannelAttentionModule(nn.Module):
    def __init__(self, num_channels, se_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=1)
        self.shared_block = nn.Sequential(
            nn.Conv2d(
                num_channels,
                se_size,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                se_size,
                num_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = self.avgpool(x)
        max_x = self.maxpool(x)
        avg_x = self.shared_block(avg_x)
        max_x = self.shared_block(max_x)
        x1 = torch.add(avg_x, max_x)
        scale = self.sigmoid(x1)
        return x * scale


class CBAMModule(nn.Module):
    def __init__(self, kernels_out, se_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.spatial_attmodule = SpatialAttentionModule(2, 1)
        self.channel_attmodule = ChannelAttentionModule(kernels_out, se_size)

    def forward(self, x):
        x1 = self.spatial_attmodule(x)
        x1 = self.channel_attmodule(x1)
        return x1


class CBAMBottleneck(nn.Module):
    def __init__(
        self,
        kernels_in,
        kernels_bt,
        kernels_out,
        stride=1,
        se_size=16,
        has_downsample=False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.conv2d1 = nn.Conv2d(
            kernels_in,
            kernels_bt,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(
            kernels_bt, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2d2 = nn.Conv2d(
            kernels_bt,
            kernels_out,
            kernel_size=(3, 3),
            stride=(stride, stride),
            padding=(1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(
            kernels_out, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2d3 = nn.Conv2d(
            kernels_out,
            kernels_out,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(
            kernels_out, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu3 = nn.ReLU(inplace=True)
        self.cbammodule = CBAMModule(kernels_out, se_size)
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
        x1 = self.conv2d1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2d2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = self.conv2d3(x1)
        x1 = self.bn3(x1)
        x1 = self.cbammodule(x1)
        if self.has_downsample:
            x2 = self.downsample(x)
            x = torch.add(x1, x2)
        else:
            x = torch.add(x1, x)
        x = self.relu3(x)
        return x


def repeat_cbamconvblock(
    in_channels,
    bt_channels,
    out_channels,
    has_downsample,
    stride,
    se_size,
    num_blocks=1,
    *args,
    **kwargs
):
    return [
        CBAMBottleneck(
            in_channels,
            bt_channels,
            out_channels,
            has_downsample=has_downsample,
            stride=stride,
            se_size=se_size,
            *args,
            **kwargs
        )
        for _ in range(num_blocks)
    ]


class CBAMEncoderBottleneckBlock(BaseEncoderBlock):
    def __init__(
        self,
        in_channels,
        bt_channels,
        out_channels,
        se_size,
        num_blocks=1,
        pool_block=False,
        wavelets_mode=False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(wavelets_mode=wavelets_mode, pool_mode=1, *args, **kwargs)
        self.pool_block = pool_block
        if self.pool_block:
            self.pool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
            )
        self.block = nn.Sequential(
            CBAMBottleneck(
                in_channels,
                bt_channels,
                out_channels,
                has_downsample=True,
                se_size=se_size,
                stride=1 if self.pool_block else 2,
            ),
            *repeat_cbamconvblock(
                out_channels,
                bt_channels,
                out_channels,
                has_downsample=False,
                stride=1,
                se_size=se_size,
                num_blocks=num_blocks,
            )
        )
        self.wblock = nn.Sequential(
            nn.Conv2d(
                1,
                1,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(
                1,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(inplace=True),
            # BasicBlock(
            #    1,
            #    out_channels // 2,
            #    has_downsample=False if self.pool_block else True,
            #    stride=1 if self.pool_block else 2,
            # ),
            # *repeat_bsconvblock(
            #    out_channels // 2,
            #    out_channels // 2,
            #    has_downsample=False,
            #    stride=1,
            #    num_blocks=num_blocks,
            # ),
        )
