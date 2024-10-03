from models.encoders.base import BaseEncoder


class CBAMNetBaseEncoder(BaseEncoder):
    def __init__(self, in_channels=3, *args, **kwargs) -> None:
        super().__init__(in_channels, *args, **kwargs)

    def forward(self, inputs):
        c1 = self.encoder_block1(inputs)
        c2 = self.encoder_block2(c1)
        c3 = self.encoder_block3(c2)
        c4 = self.encoder_block4(c3)
        c5 = self.encoder_block5(c4)
        return c1, c2, c3, c4, c5
