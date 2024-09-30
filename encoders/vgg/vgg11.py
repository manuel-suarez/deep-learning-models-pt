import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2) -> None:
        super().__init__()
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

    def forward(self, x, w=None):
        x = self.pool(x)
        if w is not None:
            x = torch.add(x, w)
        x = self.block(x)
        return x


class Vgg11Encoder(nn.Module):
    def __init__(self, in_channels=3) -> None:
        super().__init__()
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ),
            nn.BatchNorm2d(
                64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.encoder_block2 = EncoderBlock(64, 128, num_blocks=1)
        # self.encoder_block2 = nn.Sequential(
        #    nn.MaxPool2d(
        #        kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
        #    ),
        #    nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #    nn.BatchNorm2d(
        #        128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        #    ),
        #    nn.ReLU(inplace=True),
        # )
        self.encoder_block3 = EncoderBlock(128, 256, num_blocks=2)
        # self.encoder_block3 = nn.Sequential(
        #    nn.MaxPool2d(
        #        kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
        #    ),
        #    nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #    nn.BatchNorm2d(
        #        256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        #    ),
        #    nn.ReLU(inplace=True),
        #    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #    nn.BatchNorm2d(
        #        256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        #    ),
        #    nn.ReLU(inplace=True),
        # )
        self.encoder_block4 = EncoderBlock(256, 512, num_blocks=2)
        # self.encoder_block4 = nn.Sequential(
        #    nn.MaxPool2d(
        #        kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
        #    ),
        #    nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #    nn.BatchNorm2d(
        #        512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        #    ),
        #    nn.ReLU(inplace=True),
        #    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #    nn.BatchNorm2d(
        #        512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        #    ),
        #    nn.ReLU(inplace=True),
        # )
        self.encoder_block5 = EncoderBlock(512, 512, num_blocks=2)
        # self.encoder_block5 = nn.Sequential(
        #    nn.MaxPool2d(
        #        kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
        #    ),
        #    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #    nn.BatchNorm2d(
        #        512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        #    ),
        #    nn.ReLU(inplace=True),
        #    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #    nn.BatchNorm2d(
        #        512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        #    ),
        #    nn.ReLU(inplace=True),
        # )
        self.encoder_block6 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            )
        )

    def forward(self, inputs):
        # We need to obtain the wavelet decomposition factors (4 decomposition levels)
        # x, x1, x2, x3, x4 = inputs
        # Process and add decomposition level
        x = self.encoder_block1(inputs)
        c1 = self.encoder_block2(x)
        c2 = self.encoder_block3(c1)
        c3 = self.encoder_block4(c2)
        c4 = self.encoder_block5(c3)
        c5 = self.encoder_block6(c4)
        return c1, c2, c3, c4, c5
