import torch, torch.nn as nn
from .common import Conv2dReLU


class TransposeX2(nn.Module):
    def __init__(self, kernels_in, kernels_out, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.convtranspose = nn.ConvTranspose2d(
            kernels_in,
            kernels_out,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(
            kernels_out, eps=1e-5, momentum=0.01, affine=True, track_running_stats=True
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.convtranspose(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, kernels_in, kernels_mid, kernels_out, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decoder_block = nn.Sequential(
            Conv2dReLU(kernels_in, kernels_mid),
            TransposeX2(kernels_mid, kernels_mid),
            Conv2dReLU(kernels_mid, kernels_out),
        )

    def forward(self, x, skip=None):
        x = self.decoder_block(x)
        if skip != None:
            x = torch.add(x, skip)
        return x


class LinknetDecoder(nn.Module):
    def __init__(self, inputs, mids, outputs, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decoder_block1 = DecoderBlock(inputs[0], mids[0], outputs[0])
        self.decoder_block2 = DecoderBlock(inputs[1], mids[1], outputs[1])
        self.decoder_block3 = DecoderBlock(inputs[2], mids[2], outputs[2])
        self.decoder_block4 = DecoderBlock(inputs[3], mids[3], outputs[3])
        self.decoder_block5 = DecoderBlock(inputs[4], mids[4], outputs[4])

    def forward(self, x, skips):
        c1, c2, c3, c4 = skips
        x = self.decoder_block1(x, c4)
        x = self.decoder_block2(x, c3)
        x = self.decoder_block3(x, c2)
        x = self.decoder_block4(x, c1)
        x = self.decoder_block5(x)
        return x
