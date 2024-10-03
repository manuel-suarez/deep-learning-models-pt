import torch.nn as nn


class BaseEncoder(nn.Module):
    def __init__(self, in_channels=3, wavelets_mode=False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.wavelets_mode = wavelets_mode

    def forward(self, inputs):
        # We need to obtain the wavelet decomposition factors (4 decomposition levels)
        if self.wavelets_mode:
            x, x1, x2, x3, x4 = inputs
            # Process and add decomposition level
            x = self.encoder_block1(x)
            c1 = self.encoder_block2(x, x1)
            c2 = self.encoder_block3(c1, x2)
            c3 = self.encoder_block4(c2, x3)
            c4 = self.encoder_block5(c3, x4)
            c5 = self.encoder_block6(c4)
        else:
            x = self.encoder_block1(inputs)
            c1 = self.encoder_block2(x)
            c2 = self.encoder_block3(c1)
            c3 = self.encoder_block4(c2)
            c4 = self.encoder_block5(c3)
            c5 = self.encoder_block6(c4)
        return c1, c2, c3, c4, c5
