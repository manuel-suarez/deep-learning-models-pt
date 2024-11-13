from .base import ResNetBaseEncoder
import torch.nn as nn
from .common import ResNetEncoderBasicBlock


class ResNetEncoder(ResNetBaseEncoder):
    def __init__(
        self, in_channels=3, wavelets_mode=False, deeplab_arch=False, *args, **kwargs
    ) -> None:
        super().__init__(in_channels, wavelets_mode, *args, **kwargs)
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            ),
            nn.BatchNorm2d(
                64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.encoder_block2 = ResNetEncoderBasicBlock(
            in_channels=64
            + (1 if wavelets_mode == 2 else 0)
            + (1 if wavelets_mode == 3 else 0),
            out_channels=64
            + (1 if wavelets_mode == 2 else 0)
            + (1 if wavelets_mode == 3 else 0),
            num_blocks=1,
            pool_block=True,
            wavelets_mode=wavelets_mode,
        )
        # self.encoder_block2 = nn.Sequential(
        #    nn.MaxPool2d(
        #        kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        #    ),
        #    nn.Sequential(
        #        BasicBlock(64, 64, has_downsample=False, stride=1),
        #        BasicBlock(64, 64, has_downsample=False, stride=1),
        #    ),
        # )
        self.encoder_block3 = ResNetEncoderBasicBlock(
            in_channels=64
            + (2 if wavelets_mode == 2 else 0)
            + (2 if wavelets_mode == 3 else 0),
            out_channels=128
            + (2 if wavelets_mode == 2 else 0)
            + (2 if wavelets_mode == 3 else 0),
            num_blocks=1,
            pool_block=False,
            wavelets_mode=wavelets_mode,
        )
        # self.encoder_block3 = nn.Sequential(
        #    BasicBlock(64, 128, has_downsample=True, stride=2),
        #    BasicBlock(128, 128, has_downsample=False, stride=1),
        # )
        self.encoder_block4 = ResNetEncoderBasicBlock(
            in_channels=128
            + (3 if wavelets_mode == 2 else 0)
            + (3 if wavelets_mode == 3 else 0),
            out_channels=256
            + (3 if wavelets_mode == 2 else 0)
            + (3 if wavelets_mode == 3 else 0),
            num_blocks=1,
            pool_block=False,
            wavelets_mode=wavelets_mode,
        )
        # self.encoder_block4 = nn.Sequential(
        #    BasicBlock(128, 256, has_downsample=True, stride=2),
        #    BasicBlock(256, 256, has_downsample=False, stride=1),
        # )
        self.encoder_block5 = ResNetEncoderBasicBlock(
            in_channels=256
            + (4 if wavelets_mode == 2 else 0)
            + (4 if wavelets_mode == 3 else 0),
            out_channels=512
            + (4 if wavelets_mode == 2 else 0)
            + (4 if wavelets_mode == 3 else 0),
            num_blocks=1,
            pool_block=False,
            stride=deeplab_arch,
            wavelets_mode=wavelets_mode,
        )
        # self.encoder_block5 = nn.Sequential(
        #    BasicBlock(256, 512, has_downsample=True, stride=2),
        #    BasicBlock(512, 512, has_downsample=False, stride=1),
        # )
