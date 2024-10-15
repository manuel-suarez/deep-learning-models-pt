import torch
import torch.nn as nn
from models.encoders.base import BaseEncoderBlock


class EncoderBlock(BaseEncoderBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=2,
        pool_block=True,
        wavelets_mode=False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(wavelets_mode=wavelets_mode, pool_mode=0, *args, **kwargs)
        self.pool_block = pool_block
        if self.pool_block:
            self.pool = nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            )
        # We are simplifying the wavelets block
        self.wblock = nn.Sequential(
            nn.Conv2d(
                1,
                in_channels // 2,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(
                in_channels // 2,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(inplace=True),
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
