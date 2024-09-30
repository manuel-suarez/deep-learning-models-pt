import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_blocks=2, pool_block=True
    ) -> None:
        super().__init__()
        self.pool_block = pool_block
        if self.pool_block:
            self.pool = nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            )
        if num_blocks == 1:
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
                nn.BatchNorm2d(
                    out_channels,
                    eps=1e-5,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.ReLU(inplace=True),
            )
        if num_blocks == 2:
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
                nn.BatchNorm2d(
                    out_channels,
                    eps=1e-5,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
                nn.BatchNorm2d(
                    out_channels,
                    eps=1e-5,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.ReLU(inplace=True),
            )
        if num_blocks == 3:
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
                nn.BatchNorm2d(
                    out_channels,
                    eps=1e-5,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
                nn.BatchNorm2d(
                    out_channels,
                    eps=1e-5,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
                nn.BatchNorm2d(
                    out_channels,
                    eps=1e-5,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.ReLU(inplace=True),
            )
        if num_blocks == 4:
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
                nn.BatchNorm2d(
                    out_channels,
                    eps=1e-5,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
                nn.BatchNorm2d(
                    out_channels,
                    eps=1e-5,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
                nn.BatchNorm2d(
                    out_channels,
                    eps=1e-5,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
                nn.BatchNorm2d(
                    out_channels,
                    eps=1e-5,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, w=None):
        if self.pool_block:
            x = self.pool(x)
        if w is not None:
            x = torch.add(x, w)
        x = self.block(x)
        return x
