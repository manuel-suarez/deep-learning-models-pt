import torch
import torch.nn as nn


class BaseEncoderB5(nn.Module):
    def __init__(self, in_channels=3, wavelets_mode=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.wavelets_mode = wavelets_mode

    def forward(self, inputs):
        # We will implement "multiresolution_mode" where we need to specify different levels of operations and decompositions
        # A value of "Falte" for wavelets_mode indicates that there will be no decomposition on the input
        if not self.wavelets_mode:
            x = self.encoder_block1(inputs)
            c1 = self.encoder_block2(x)
            c2 = self.encoder_block3(c1)
            c3 = self.encoder_block4(c2)
            c4 = self.encoder_block5(c3)
            c5 = self.encoder_block6(c4)
            return c1, c2, c3, c4, c5
        # Level 0 of wavelets_mode indicates that in fact there is no wavelet decomposition but image shrink (using the same scale values for pixels only subsampling)

        # Level 1 of wavelets_mode indicates that there are a wavelets decomposition of the input and we are using "add" operation between decompositions and filters of encoder blocks
        if self.wavelets_mode >= 0:
            x, x1, x2, x3, x4 = inputs
            # Process and add decomposition level
            x = self.encoder_block1(x)
            c1 = self.encoder_block2(x, x1)
            c2 = self.encoder_block3(c1, x2)
            c3 = self.encoder_block4(c2, x3)
            c4 = self.encoder_block5(c3, x4)
            c5 = self.encoder_block6(c4)
            return c1, c2, c3, c4, c5


class BaseEncoderB6(nn.Module):
    def __init__(self, in_channels=3, wavelets_mode=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.wavelets_mode = wavelets_mode

    def forward(self, inputs):
        if not self.wavelets_mode:
            c1 = self.encoder_block1(inputs)
            c2 = self.encoder_block2(c1)
            c3 = self.encoder_block3(c2)
            c4 = self.encoder_block4(c3)
            c5 = self.encoder_block5(c4)
            return c1, c2, c3, c4, c5
        # We need to obtain the wavelet decomposition factors (4 decomposition levels)
        if self.wavelets_mode >= 0:
            x, x1, x2, x3, x4 = inputs
            # Process and add decomposition level
            c1 = self.encoder_block1(x)
            c2 = self.encoder_block2(c1, x1)
            c3 = self.encoder_block3(c2, x2)
            c4 = self.encoder_block4(c3, x3)
            c5 = self.encoder_block5(c4, x4)
            return c1, c2, c3, c4, c5


class BaseEncoderBlock(nn.Module):
    def __init__(self, wavelets_mode=False, pool_mode=0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.wavelets_mode = wavelets_mode
        # Pool mode indicates when the pool block must be evaluated, after or before the multiresolution operation block
        self.pool_mode = pool_mode

    def forward(self, x, w=None):
        if self.pool_block and self.pool_mode == 0:
            x = self.pool(x)
        if w is not None:
            if self.wavelets_mode == 1:
                x = torch.add(x, w)
            if self.wavelets_mode == 2:
                x = torch.cat([x, w], dim=1)
        if self.pool_block and self.pool_mode == 1:
            x = self.pool(x)
        x = self.block(x)
        return x
