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
    def __init__(self, inputs, outputs, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decoder_block1_1 = DecoderBlock(inputs[0][0], outputs[0][0])
        self.decoder_block1_2 = DecoderBlock(inputs[0][1], outputs[0][1])
        self.decoder_block1_3 = DecoderBlock(inputs[0][2], outputs[0][2])
        self.decoder_block1_4 = DecoderBlock(inputs[0][3], outputs[0][3])
        self.decoder_block2_1 = DecoderBlock(inputs[1][0], outputs[1][0])
        self.decoder_block2_2 = DecoderBlock(inputs[1][1], outputs[1][1])
        self.decoder_block2_3 = DecoderBlock(inputs[1][2], outputs[1][2])
        self.decoder_block3_1 = DecoderBlock(inputs[2][0], outputs[2][0])
        self.decoder_block3_2 = DecoderBlock(inputs[2][1], outputs[2][1])
        self.decoder_block4_1 = DecoderBlock(inputs[3][0], outputs[3][0])
        self.decoder_block5 = DecoderBlock(inputs[4], outputs[4])

    def forward(self, c5, skips):
        c1, c2, c3, c4 = skips
        # print("UnetPlusPlus Decoder forward")
        # print("c1 shape: ", c1.shape)
        # print("c2 shape: ", c2.shape)
        # print("c3 shape: ", c3.shape)
        # print("c4 shape: ", c4.shape)
        # print("c5 shape: ", c5.shape)
        # Block 1
        # print("Block 1")
        x1_1 = self.decoder_block1_1(c5, c4)
        # print("x1_1 shape: ", x1_1.shape)
        x1_2 = self.decoder_block1_2(c4, c3)
        # print("x1_2 shape: ", x1_2.shape)
        x1_3 = self.decoder_block1_3(c3, c2)
        # print("x1_3 shape: ", x1_3.shape)
        x1_4 = self.decoder_block1_4(c2, c1)
        # print("x1_4 shape: ", x1_4.shape)
        # Concatenations
        cat2_1 = torch.cat([x1_2, c3], dim=1)
        cat2_2 = torch.cat([x1_3, c2], dim=1)
        cat2_3 = torch.cat([x1_4, c1], dim=1)
        # Block 2
        # print("Block 2")
        x2_1 = self.decoder_block2_1(x1_1, cat2_1)
        # print("x2_1 shape: ", x2_1.shape)
        x2_2 = self.decoder_block2_2(x1_2, cat2_2)
        # print("x2_2 shape: ", x2_2.shape)
        x2_3 = self.decoder_block2_3(x1_3, cat2_3)
        # print("x2_3 shape: ", x2_3.shape)
        # Concatenations
        cat3_1 = torch.cat([x2_2, x1_3, c2], dim=1)
        cat3_2 = torch.cat([x2_3, x1_4, c1], dim=1)
        # Block 3
        # print("Block 3")
        x3_1 = self.decoder_block3_1(x2_1, cat3_1)
        # print("x3_1 shape: ", x3_1.shape)
        x3_2 = self.decoder_block3_2(x2_2, cat3_2)
        # print("x3_2 shape: ", x3_2.shape)
        # Concatenations
        # print("Cat 3")
        cat4_1 = torch.cat([x3_2, x2_3, x1_4, c1], dim=1)
        # print("cat4_1 shape: ", cat4_1.shape)
        # Block 4
        # print("Block 4")
        x4_1 = self.decoder_block4_1(x3_1, cat4_1)
        # print("x4_1 shape: ", x4_1.shape)
        # Block 5
        x5 = self.decoder_block5(x4_1)
        return x5
