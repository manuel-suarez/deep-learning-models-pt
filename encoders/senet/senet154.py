from .base import SeNetBaseEncoder
import torch.nn as nn
from .common import SEBottleneck, SENetEncoderBottleneckBlock


class SENetEncoder(SeNetBaseEncoder):
    def __init__(
        self, in_channels=3, wavelets_mode=False, deeplab_arch=False, *args, **kwargs
    ) -> None:
        super().__init__(in_channels, wavelets_mode, *args, *kwargs)
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(
                64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(
                64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(
                128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.encoder_block2 = SENetEncoderBottleneckBlock(
            in_channels=128
            + (1 if wavelets_mode == 2 else 0)
            + (1 if wavelets_mode == 3 else 0),
            bt_channels=128,
            out_channels=256
            + (1 if wavelets_mode == 2 else 0)
            + (1 if wavelets_mode == 3 else 0),
            se_size=16,
            num_blocks=2,
            pool_block=True,
            wavelets_mode=wavelets_mode,
        )
        # self.encoder_block2 = nn.Sequential(
        #    nn.MaxPool2d(
        #        kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        #    ),
        #    nn.Sequential(
        #        SEBottleneck(128, 128, 256, has_downsample=True, stride=1),
        #        SEBottleneck(256, 128, 256, has_downsample=False, stride=1),
        #        SEBottleneck(256, 128, 256, has_downsample=False, stride=1),
        #    ),
        # )
        self.encoder_block3 = SENetEncoderBottleneckBlock(
            in_channels=256
            + (2 if wavelets_mode == 2 else 0)
            + (2 if wavelets_mode == 3 else 0),
            bt_channels=256,
            out_channels=512
            + (2 if wavelets_mode == 2 else 0)
            + (2 if wavelets_mode == 3 else 0),
            se_size=32,
            num_blocks=7,
            pool_block=False,
            wavelets_mode=wavelets_mode,
        )
        # self.encoder_block3 = nn.Sequential(
        #    SEBottleneck(256, 256, 512, has_downsample=True, stride=2, se_size=32),
        #    SEBottleneck(512, 256, 512, has_downsample=False, stride=1, se_size=32),
        #    SEBottleneck(512, 256, 512, has_downsample=False, stride=1, se_size=32),
        #    SEBottleneck(512, 256, 512, has_downsample=False, stride=1, se_size=32),
        #    SEBottleneck(512, 256, 512, has_downsample=False, stride=1, se_size=32),
        #    SEBottleneck(512, 256, 512, has_downsample=False, stride=1, se_size=32),
        #    SEBottleneck(512, 256, 512, has_downsample=False, stride=1, se_size=32),
        #    SEBottleneck(512, 256, 512, has_downsample=False, stride=1, se_size=32),
        # )
        self.encoder_block4 = SENetEncoderBottleneckBlock(
            in_channels=512
            + (3 if wavelets_mode == 2 else 0)
            + (3 if wavelets_mode == 3 else 0),
            bt_channels=512,
            out_channels=1024
            + (3 if wavelets_mode == 2 else 0)
            + (3 if wavelets_mode == 3 else 0),
            se_size=64,
            num_blocks=35,
            pool_block=False,
            wavelets_mode=wavelets_mode,
        )
        # self.encoder_block4 = nn.Sequential(
        #    SEBottleneck(512, 512, 1024, has_downsample=True, stride=2, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        #    SEBottleneck(1024, 512, 1024, has_downsample=False, stride=1, se_size=64),
        # )
        self.encoder_block5 = SENetEncoderBottleneckBlock(
            in_channels=1024
            + (4 if wavelets_mode == 2 else 0)
            + (4 if wavelets_mode == 3 else 0),
            bt_channels=1024,
            out_channels=2048
            + (4 if wavelets_mode == 2 else 0)
            + (4 if wavelets_mode == 3 else 0),
            se_size=128,
            num_blocks=2,
            pool_block=False,
            stride=deeplab_arch,
            wavelets_mode=wavelets_mode,
        )
        # self.encoder_block5 = nn.Sequential(
        #    SEBottleneck(1024, 1024, 2048, has_downsample=True, stride=2, se_size=128),
        #    SEBottleneck(2048, 1024, 2048, has_downsample=False, stride=1, se_size=128),
        #    SEBottleneck(2048, 1024, 2048, has_downsample=False, stride=1, se_size=128),
        # )
