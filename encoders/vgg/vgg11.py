from models.encoders.base import BaseEncoder
from .common import EncoderBlock
import torch
import torch.nn as nn


class Vgg11Encoder(BaseEncoder):
    def __init__(self, in_channels=3, wavelets_mode=False) -> None:
        super().__init__()
        self.wavelets_mode = wavelets_mode
        self.encoder_block1 = EncoderBlock(
            in_channels, 64, num_blocks=1, pool_block=False
        )
        self.encoder_block2 = EncoderBlock(64, 128, num_blocks=1)
        self.encoder_block3 = EncoderBlock(128, 256, num_blocks=2)
        self.encoder_block4 = EncoderBlock(256, 512, num_blocks=2)
        self.encoder_block5 = EncoderBlock(512, 512, num_blocks=2)
        self.encoder_block6 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            )
        )

    def forward(self, inputs):
        # We need to obtain the wavelet decomposition factors (4 decomposition levels)
        if self.wavelets_mode:
            x, x1, x2, x3, x4 = inputs
            # Process and add decomposition level
            x = self.encoder_block1(x)
            c1 = self.encoder_block2(x, x1)
            c2 = self.encoder_block3(c1, x2)
            c3 = self.encoder_block4(c2, x3)
            c4 = self.encoder_block5(c3, x4)
            c5 = self.encoder_block6(c4)
        else:
            x = self.encoder_block1(inputs)
            c1 = self.encoder_block2(x)
            c2 = self.encoder_block3(c1)
            c3 = self.encoder_block4(c2)
            c4 = self.encoder_block5(c3)
            c5 = self.encoder_block6(c4)
        return c1, c2, c3, c4, c5
