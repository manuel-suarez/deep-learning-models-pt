from .base import VggBaseEncoder
from .common import EncoderBlock
import torch
import torch.nn as nn


class Vgg11Encoder(VggBaseEncoder):
    def __init__(self, in_channels=3, wavelets_mode=False, *args, **kwargs) -> None:
        super().__init__(in_channels, wavelets_mode, *args, **kwargs)
        self.wavelets_mode = wavelets_mode
        self.encoder_block1 = EncoderBlock(
            in_channels, 64, num_blocks=1, pool_block=False
        )
        self.encoder_block2 = EncoderBlock(
            64 + (1 if wavelets_mode == 2 else 0),
            128,
            num_blocks=1,
            wavelets_mode=wavelets_mode,
        )
        self.encoder_block3 = EncoderBlock(
            128 + (1 if wavelets_mode == 2 else 0),
            256,
            num_blocks=2,
            wavelets_mode=wavelets_mode,
        )
        self.encoder_block4 = EncoderBlock(
            256 + (1 if wavelets_mode == 2 else 0),
            512,
            num_blocks=2,
            wavelets_mode=wavelets_mode,
        )
        self.encoder_block5 = EncoderBlock(
            512 + (1 if wavelets_mode == 2 else 0),
            512,
            num_blocks=2,
            wavelets_mode=wavelets_mode,
        )
        self.encoder_block6 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            )
        )
