import torch
from models.encoders.base import BaseEncoderB6


class EfficientNetBaseEncoder(BaseEncoderB6):
    def __init__(self, in_channels=3, wavelets_mode=False, *args, **kwargs) -> None:
        super().__init__(in_channels, wavelets_mode, *args, **kwargs)

    def forward(self, inputs):
        if not self.wavelets_mode:
            c1 = self.encoder_block1(inputs)
            c2 = self.encoder_block2(c1)
            c3 = self.encoder_block3(c2)
            c4 = self.encoder_block4(c3)
            c5 = self.encoder_block5(c4)
            return c1, c2, c3, c4, c5
        # We need to obtain the wavelet decomposition factors (4 decomposition levels)
        if self.wavelets_mode == 1:
            x, x1, x2, x3, x4 = inputs
            # Process and add decomposition level
            c1 = self.encoder_block1(x)
            x1 = torch.add(c1, x1)
            c2 = self.encoder_block2(x1)
            x2 = torch.add(c2, x2)
            c3 = self.encoder_block3(x2)
            x3 = torch.add(c3, x3)
            c4 = self.encoder_block4(x3)
            x4 = torch.add(c4, x4)
            c5 = self.encoder_block5(x4)
            return c1, c2, c3, c4, c5
