import torch, torch.nn as nn
from .common import Conv2dReLU


class MaxPoolBlock(nn.Module):
    def __init__(
        self, pooling_times, in_channels, out_channels, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.block_pool = nn.MaxPool2d(pooling_times, pooling_times, ceil_mode=True)
        self.block_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.block_bn = nn.BatchNorm2d(out_channels)
        self.block_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.block_pool(x)
        x = self.block_conv(x)
        x = self.block_bn(x)
        x = self.block_relu(x)

        return x


class UpSampleBlock(nn.Module):
    def __init__(
        self, scale_factor, in_channels, out_channels, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.block_upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear")
        self.block_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.block_bn = nn.BatchNorm2d(out_channels)
        self.block_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print("\tUpSampleBlock forward:")
        # print("\tx: ", x.shape)
        x = self.block_upsample(x)
        # print("\tup: ", x.shape)
        x = self.block_conv(x)
        x = self.block_bn(x)
        x = self.block_relu(x)

        return x


class CatBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.block_bn = nn.BatchNorm2d(out_channels)
        self.block_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.block_conv(x)
        x = self.block_bn(x)
        x = self.block_relu(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.block_bn = nn.BatchNorm2d(out_channels)
        self.block_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.block_conv(x)
        x = self.block_bn(x)
        x = self.block_relu(x)

        return x


class Unet3PDecoderBlock(nn.Module):
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
        self.decoder_block4 = Unet3PDecoderBlock(
            blocks=[
                MaxPoolBlock(8, inputs[0], self.cat_channels),
                MaxPoolBlock(4, inputs[1], self.cat_channels),
                MaxPoolBlock(2, inputs[2], self.cat_channels),
                CatBlock(inputs[3], self.cat_channels),
                UpSampleBlock(2, inputs[4], self.cat_channels),
            ],
            decoder=DecoderBlock(self.up_channels, self.up_channels),
        )

        self.decoder_block3 = Unet3PDecoderBlock(
            blocks=[
                MaxPoolBlock(4, inputs[0], self.cat_channels),
                MaxPoolBlock(2, inputs[1], self.cat_channels),
                CatBlock(inputs[2], self.cat_channels),
                UpSampleBlock(2, self.up_channels, self.cat_channels),
                UpSampleBlock(4, inputs[4], self.cat_channels),
            ],
            decoder=DecoderBlock(self.up_channels, self.up_channels),
        )

        self.decoder_block2 = Unet3PDecoderBlock(
            blocks=[
                MaxPoolBlock(2, inputs[0], self.cat_channels),
                CatBlock(inputs[1], self.cat_channels),
                UpSampleBlock(2, self.up_channels, self.cat_channels),
                UpSampleBlock(4, self.up_channels, self.cat_channels),
                UpSampleBlock(8, inputs[4], self.cat_channels),
            ],
            decoder=DecoderBlock(self.up_channels, self.up_channels),
        )

        self.decoder_block1 = Unet3PDecoderBlock(
            blocks=[
                CatBlock(inputs[0], self.cat_channels),
                UpSampleBlock(2, self.up_channels, self.cat_channels),
                UpSampleBlock(4, self.up_channels, self.cat_channels),
                UpSampleBlock(8, self.up_channels, self.cat_channels),
                UpSampleBlock(16, inputs[4], self.cat_channels),
            ],
            decoder=DecoderBlock(self.up_channels, self.up_channels),
        )

        self.output_block = nn.Sequential(
            nn.Conv2d(self.up_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, c5, skips):
        print("Unet3PDecoder forward:")
        c1, c2, c3, c4 = skips
        print("c1: ", c1.shape)
        print("c2: ", c2.shape)
        print("c3: ", c3.shape)
        print("c4: ", c4.shape)
        print("c5: ", c5.shape)
        # h1_hd4 = self.h1_hd4(c1)
        # print("h1_hd4: ", h1_hd4.shape)
        # h2_hd4 = self.h2_hd4(c2)
        # print("h2_hd4: ", h2_hd4.shape)
        # h3_hd4 = self.h3_hd4(c3)
        # print("h3_hd4: ", h3_hd4.shape)
        # h4_hd4 = self.h4_hd4(c4)
        # print("h4_hd4: ", h4_hd4.shape)
        # h5_hd4 = self.h5_hd4(c5)
        # print("h5_hd4: ", h5_hd4.shape)
        # hd4 = torch.cat([h1_hd4, h2_hd4, h3_hd4, h4_hd4, h5_hd4], dim=1)
        # print("hd4: ", hd4.shape)
        # hd4 = self.h4_dec(hd4)
        hd4 = self.decoder_block4([c1, c2, c3, c4, c5])

        # h1_hd3 = self.h1_hd3(c1)
        # print("h1_hd3: ", h1_hd3.shape)
        # h2_hd3 = self.h2_hd3(c2)
        # print("h2_hd3: ", h2_hd3.shape)
        # h3_hd3 = self.h3_hd3(c3)
        # print("h3_hd3: ", h3_hd3.shape)
        # h4_hd3 = self.h4_hd3(hd4)
        # print("h4_hd3: ", h4_hd3.shape)
        # h5_hd3 = self.h5_hd3(c5)
        # print("h5_hd3: ", h5_hd3.shape)
        # hd3 = torch.cat([h1_hd3, h2_hd3, h3_hd3, h4_hd3, h5_hd3], dim=1)
        # print("hd3: ", hd3.shape)
        # hd3 = self.h3_dec(hd3)
        hd3 = self.decoder_block3([c1, c2, c3, hd4, c5])

        # h1_hd2 = self.h1_hd2(c1)
        # print("h1_hd2: ", h1_hd2.shape)
        # h2_hd2 = self.h2_hd2(c2)
        # print("h2_hd2: ", h2_hd2.shape)
        # h3_hd2 = self.h3_hd2(hd3)
        # print("h3_hd2: ", h3_hd2.shape)
        # h4_hd2 = self.h4_hd2(hd4)
        # print("h4_hd2: ", h4_hd2.shape)
        # h5_hd2 = self.h5_hd2(c5)
        # print("h5_hd2: ", h5_hd2.shape)
        # hd2 = torch.cat([h1_hd2, h2_hd2, h3_hd2, h4_hd2, h5_hd2], dim=1)
        # print("hd2: ", hd2.shape)
        # hd2 = self.h2_dec(hd2)
        hd2 = self.decoder_block2([c1, c2, hd3, hd4, c5])

        # h1_hd1 = self.h1_hd1(c1)
        # print("h1_hd1: ", h1_hd1.shape)
        # h2_hd1 = self.h2_hd1(hd2)
        # print("h2_hd1: ", h2_hd1.shape)
        # h3_hd1 = self.h3_hd1(hd3)
        # print("h3_hd1: ", h3_hd1.shape)
        # h4_hd1 = self.h4_hd1(hd4)
        # print("h4_hd1: ", h4_hd1.shape)
        # h5_hd1 = self.h5_hd1(c5)
        # print("h5_hd1: ", h5_hd1.shape)
        # hd1 = torch.cat([h1_hd1, h2_hd1, h3_hd1, h4_hd1, h5_hd1], dim=1)
        # print("hd1: ", hd1.shape)
        # hd1 = self.h2_dec(hd1)
        hd1 = self.decoder_block1([c1, hd2, hd3, hd4, c5])
        hd1 = self.output_block(hd1)

        return hd1
