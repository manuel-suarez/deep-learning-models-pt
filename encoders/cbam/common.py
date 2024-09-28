import torch, torch.nn as nn
import torch.nn.functional as F


class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=7, stride=1, padding=0, dilation=1
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Spatial channel compression (max and avg over channels)
        x = torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)),
            dim=1,
        )
        x = self.conv(x)
        x = self.bn(x)
        scale = F.sigmoid(x)
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


class SEBottleneck(nn.Module):
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
        self.spatial_attmodule = SpatialAttentionModule(kernels_out, se_size)
        self.channel_attmodule = ChannelAttentionModule(kernels_out, se_size)
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
        x1 = self.spatial_attmodule(x1)
        x1 = self.channel_attmodule(x1)
        if self.has_downsample:
            x2 = self.downsample(x)
            x = torch.add(x1, x2)
        else:
            x = torch.add(x1, x)
        x = self.relu3(x)
        return x
