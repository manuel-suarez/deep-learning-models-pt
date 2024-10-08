from models.encoders.base import BaseEncoder


class SeNetBaseEncoder(BaseEncoder):
    def __init__(self, in_channels=3, wavelets_mode=False, *args, **kwargs) -> None:
        super().__init__(in_channels, wavelets_mode, *args, **kwargs)

    def forward(self, inputs):
        print("ResNetBaseEncoder forward: ")
        print("inputs: ", len(inputs))
        # We need to obtain the wavelet decomposition factors (4 decomposition levels)
        if self.wavelets_mode:
            x, x1, x2, x3, x4 = inputs
            print("x: ", x.shape)
            print("x1: ", x1.shape)
            print("x2: ", x2.shape)
            print("x3: ", x3.shape)
            print("x4: ", x4.shape)
            # Process and add decomposition level
            c1 = self.encoder_block1(x)
            print("c1: ", c1.shape)
            c2 = self.encoder_block2(c1, x1)
            print("c2: ", c2.shape)
            c3 = self.encoder_block3(c2, x2)
            print("c3: ", c3.shape)
            c4 = self.encoder_block4(c3, x3)
            print("c4: ", c4.shape)
            c5 = self.encoder_block5(c4, x4)
            print("c5: ", c5.shape)
        else:
            c1 = self.encoder_block1(inputs)
            c2 = self.encoder_block2(c1)
            c3 = self.encoder_block3(c2)
            c4 = self.encoder_block4(c3)
            c5 = self.encoder_block5(c4)
        return c1, c2, c3, c4, c5
