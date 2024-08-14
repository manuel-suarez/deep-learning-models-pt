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
        self.conv2drelu1 = Conv2dReLU(kernels_in, kernels_out)
        self.conv2drelu2 = Conv2dReLU(kernels_out, kernels_out)

    def forward(self, x, skip=None):
        x = nn.functional.interpolate(x, scale_factor=2)
        if skip != None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv2drelu1(x)
        x = self.conv2drelu2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
        self,
        inputs=[3072, 768, 384, 128, 32],
        outputs=[256, 128, 64, 32, 16],
        has_center=False,
        center_size=512,
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
        c1, c2, c3, c4 = skips
        if self.has_center:
            x = self.center_block(x)
        x = self.decoder_block1(x, c4)
        x = self.decoder_block2(x, c3)
        x = self.decoder_block3(x, c2)
        x = self.decoder_block4(x, c1)
        x = self.decoder_block5(x)
        return x
