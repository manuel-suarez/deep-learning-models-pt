import torch.nn as nn
from .common import Activation


class SegmentationHead(nn.Module):
    def __init__(
        self,
        kernels_in=16,
        out_channels=1,
        has_activation=True,
        has_upsampling=False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(
            kernels_in,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.has_activation = has_activation
        self.has_upsampling = has_upsampling
        if self.has_activation:
            self.act = Activation()
        if self.has_upsampling:
            self.upsampling = nn.UpsamplingBilinear2d(scale_factor=4.0, mode="bilinear")

    def forward(self, x):
        x = self.conv(x)
        if self.has_activation:
            x = self.act(x)
        return x
