from .common import EncoderBlock
import torch.nn as nn


class Vgg19Encoder(nn.Module):
    def __init__(self, in_channels=3) -> None:
        super().__init__()
        self.encoder_block1 = EncoderBlock(in_channels, 64, pool_block=False)
        self.encoder_block2 = EncoderBlock(64, 128, num_blocks=2)
        self.encoder_block3 = EncoderBlock(128, 256, num_blocks=4)
        self.encoder_block4 = EncoderBlock(256, 512, num_blocks=4)
        self.encoder_block5 = EncoderBlock(512, 512, num_blocks=4)
        self.encoder_block6 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            )
        )

    def forward(self, x):
        x = self.encoder_block1(x)
        c1 = self.encoder_block2(x)
        c2 = self.encoder_block3(c1)
        c3 = self.encoder_block4(c2)
        c4 = self.encoder_block5(c3)
        c5 = self.encoder_block6(c4)
        return c1, c2, c3, c4, c5
