import torch, torch.nn as nn
from .common import Conv2dReLU


class MaxPoolBlock(nn.Module):
    def __init__(
        self, pooling_times, in_channels, out_channels, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.block_pool = nn.MaxPool2d(pooling_times, pooling_times, ceil_mode=True)
        self.block_conv1 = Conv2dReLU(in_channels, out_channels, kernel_size=3)
        self.block_conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3)

    def forward(self, x):
        x = self.block_pool(x)
        x = self.block_conv1(x)
        x = self.block_conv2(x)

        return x


class UpSampleBlock(nn.Module):
    def __init__(
        self, scale_factor, in_channels, out_channels, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.block_upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear")
        self.block_conv1 = Conv2dReLU(in_channels, out_channels, kernel_size=3)
        self.block_conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3)

    def forward(self, x):
        x = self.block_upsample(x)
        x = self.block_conv1(x)
        x = self.block_conv2(x)

        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block_conv1 = Conv2dReLU(in_channels, out_channels, kernel_size=3)
        self.block_conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3)

    def forward(self, x):
        x = self.block_conv1(x)
        x = self.block_conv2(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, blocks, decoder, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.b1, self.b2, self.b3, self.b4, self.b5 = blocks
        self.decoder = decoder

    def forward(self, inputs):
        h1, h2, h3, h4, h5 = inputs
        hd1 = self.b1(h1)
        hd2 = self.b2(h2)
        hd3 = self.b3(h3)
        hd4 = self.b4(h4)
        hd5 = self.b5(h5)

        hdx = torch.cat((hd1, hd2, hd3, hd4, hd5), dim=1)
        hdx = self.decoder(hdx)
        return hdx


class Unet3PDecoder(nn.Module):
    def __init__(
        self, inputs, out_channels, cat_channels=64, cat_blocks=5, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cat_channels = cat_channels
        self.cat_blocks = cat_blocks
        self.up_channels = self.cat_channels * self.cat_blocks
        # Decoder Block 4
        self.decoder_block4 = DecoderBlock(
            blocks=[
                MaxPoolBlock(8, inputs[0], self.cat_channels),
                MaxPoolBlock(4, inputs[1], self.cat_channels),
                MaxPoolBlock(2, inputs[2], self.cat_channels),
                BasicBlock(inputs[3], self.cat_channels),
                UpSampleBlock(2, inputs[4], self.cat_channels),
            ],
            decoder=BasicBlock(self.up_channels, self.up_channels),
        )

        self.decoder_block3 = DecoderBlock(
            blocks=[
                MaxPoolBlock(4, inputs[0], self.cat_channels),
                MaxPoolBlock(2, inputs[1], self.cat_channels),
                BasicBlock(inputs[2], self.cat_channels),
                UpSampleBlock(2, self.up_channels, self.cat_channels),
                UpSampleBlock(4, inputs[4], self.cat_channels),
            ],
            decoder=BasicBlock(self.up_channels, self.up_channels),
        )

        self.decoder_block2 = DecoderBlock(
            blocks=[
                MaxPoolBlock(2, inputs[0], self.cat_channels),
                BasicBlock(inputs[1], self.cat_channels),
                UpSampleBlock(2, self.up_channels, self.cat_channels),
                UpSampleBlock(4, self.up_channels, self.cat_channels),
                UpSampleBlock(8, inputs[4], self.cat_channels),
            ],
            decoder=BasicBlock(self.up_channels, self.up_channels),
        )

        self.decoder_block1 = DecoderBlock(
            blocks=[
                BasicBlock(inputs[0], self.cat_channels),
                UpSampleBlock(2, self.up_channels, self.cat_channels),
                UpSampleBlock(4, self.up_channels, self.cat_channels),
                UpSampleBlock(8, self.up_channels, self.cat_channels),
                UpSampleBlock(16, inputs[4], self.cat_channels),
            ],
            decoder=BasicBlock(self.up_channels, self.up_channels),
        )

        self.output_block = Conv2dReLU(self.up_channels, out_channels, kernel_size=3)

    def forward(self, c5, skips):
        # print("Unet3PDecoder forward:")
        c1, c2, c3, c4 = skips
        # print("c1: ", c1.shape)
        # print("c2: ", c2.shape)
        # print("c3: ", c3.shape)
        # print("c4: ", c4.shape)
        # print("c5: ", c5.shape)
        hd4 = self.decoder_block4([c1, c2, c3, c4, c5])
        hd3 = self.decoder_block3([c1, c2, c3, hd4, c5])
        hd2 = self.decoder_block2([c1, c2, hd3, hd4, c5])
        hd1 = self.decoder_block1([c1, hd2, hd3, hd4, c5])
        hd1 = self.output_block(hd1)

        return hd1
