import torch, torch.nn as nn
from .common import Conv2dReLU


class DecoderBlock(nn.Module):
    def __init__(self, kernels_in, kernels_out):
        super().__init__()
        self.kernels_in = kernels_in
        self.kernels_out = kernels_out
        self.conv2drelu1 = Conv2dReLU(kernels_in, kernels_out)
        self.conv2drelu2 = Conv2dReLU(kernels_out, kernels_out)

    def forward(self, x, skip=None):
        x = nn.functional.interpolate(x, scale_factor=2)
        if skip != None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv2drelu1(x)
        x = self.conv2drelu2(x)
        return x


class UnetPlusPlusDecoder(nn.Module):
    def __init__(
        self,
        inputs_1=[512, 256, 128, 64],
        outputs_1=[256, 128, 64, 64],
        inputs_2=[256, 128, 64],
        outputs_2=[128, 64, 64],
        inputs_3=[128, 64],
        outputs_3=[64, 64],
        inputs_4=[64],
        outputs_4=[32],
        inputs_5=32,
        outputs_5=16,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.decoder_block1_1 = DecoderBlock(inputs_1[0], outputs_1[0])
        self.decoder_block1_2 = DecoderBlock(inputs_1[1], outputs_1[1])
        self.decoder_block1_3 = DecoderBlock(inputs_1[2], outputs_1[2])
        self.decoder_block1_4 = DecoderBlock(inputs_1[3], outputs_1[3])
        self.decoder_block2_1 = DecoderBlock(inputs_2[0], outputs_2[0])
        self.decoder_block2_2 = DecoderBlock(inputs_2[1], outputs_2[1])
        self.decoder_block2_3 = DecoderBlock(inputs_2[2], outputs_2[2])
        self.decoder_block3_1 = DecoderBlock(inputs_3[0], outputs_3[0])
        self.decoder_block3_2 = DecoderBlock(inputs_3[1], outputs_3[1])
        self.decoder_block4_1 = DecoderBlock(inputs_4[0], outputs_4[0])
        self.decoder_block5 = DecoderBlock(inputs_5, outputs_5)

    def forward(self, c5, skips):
        c1, c2, c3, c4 = skips
        # Block 1
        x1_1 = self.decoder_block1_1(c5, c4)
        x1_2 = self.decoder_block1_2(c4, c3)
        x1_3 = self.decoder_block1_3(c3, c2)
        x1_4 = self.decoder_block1_4(c2, c1)
        # Concatenations
        cat2_1 = torch.cat([x1_2, c3], dim=1)
        cat2_2 = torch.cat([x1_3, c2], dim=1)
        cat2_3 = torch.cat([x1_4, c1], dim=1)
        # Block 2
        x2_1 = self.decoder_block2_1(x1_1, cat2_1)
        x2_2 = self.decoder_block2_2(x1_2, cat2_2)
        x2_3 = self.decoder_block2_3(x1_3, cat2_3)
        # Concatenations
        cat3_1 = torch.cat([x2_2, x1_3, c2], dim=1)
        cat3_2 = torch.cat([x2_3, x1_4, c1], dim=1)
        # Block 3
        x3_1 = self.decoder_block3_1(x2_1, cat3_1)
        x3_2 = self.decoder_block3_2(x2_2, cat3_2)
        # Concatenations
        cat4_1 = torch.cat([x3_2, x1_4, x2_3, x3_2, c1], dim=1)
        # Block 4
        x4_1 = self.decoder_block4_1(x3_1, cat4_1)
        # Block 5
        x5 = self.decoder_block5(x4_1)
        return x5
