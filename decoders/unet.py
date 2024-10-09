import torch, torch.nn as nn
from .common import Conv2dReLU


class CenterBlock(nn.Module):
    def __init__(self, kernels_in, kernels_out, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv2drelu1 = Conv2dReLU(kernels_in, kernels_out)
        self.conv2drelu2 = Conv2dReLU(kernels_out, kernels_out)

    def forward(self, x):
        x = self.conv2drelu1(x)
        x = self.conv2drelu2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, kernels_in, kernels_out):
        super().__init__()
        self.kernels_in = kernels_in
        self.kernels_out = kernels_out
        self.conv2drelu1 = Conv2dReLU(kernels_in, kernels_out)
        self.conv2drelu2 = Conv2dReLU(kernels_out, kernels_out)

    def forward(self, x, skip=None):
        print("decoder block forward")
        print("x: ", x.shape)
        x = nn.functional.interpolate(x, scale_factor=2)
        print("x interpolate: ", x.shape)
        if skip != None:
            x = torch.cat([x, skip], dim=1)
            print("x cat: ", x.shape)
        print("kernels_in: ", self.kernels_in)
        x = self.conv2drelu1(x)
        print("x conv2drelu1: ", x.shape)
        print("kernels_out: ", self.kernels_out)
        x = self.conv2drelu2(x)
        print("x conv2drelu2: ", x.shape)
        return x


class UnetDecoder(nn.Module):
    def __init__(
        self,
        inputs=[3072, 768, 384, 128, 32],
        outputs=[256, 128, 64, 32, 16],
        has_center=False,
        center_size=512 + 4,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.has_center = has_center
        if has_center:
            self.center_block = CenterBlock(center_size, center_size)
        self.decoder_block1 = DecoderBlock(inputs[0], outputs[0])
        self.decoder_block2 = DecoderBlock(inputs[1], outputs[1])
        self.decoder_block3 = DecoderBlock(inputs[2], outputs[2])
        self.decoder_block4 = DecoderBlock(inputs[3], outputs[3])
        self.decoder_block5 = DecoderBlock(inputs[4], outputs[4])

    def forward(self, x, skips):
        print("Unet decoder forward: ")
        print("x: ", x.shape)
        c1, c2, c3, c4 = skips
        print("c1: ", c1.shape)
        print("c2: ", c2.shape)
        print("c3: ", c3.shape)
        print("c4: ", c4.shape)
        if self.has_center:
            x = self.center_block(x)
            print("center: ", x.shape)
        x = self.decoder_block1(x, c4)
        print("decoder block1: ", x.shape)
        x = self.decoder_block2(x, c3)
        print("decoder block2: ", x.shape)
        x = self.decoder_block3(x, c2)
        print("decoder block3: ", x.shape)
        x = self.decoder_block4(x, c1)
        print("decoder block4: ", x.shape)
        x = self.decoder_block5(x)
        print("decoder block5: ", x.shape)
        return x
