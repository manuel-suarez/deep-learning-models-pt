from models.encoders.base import BaseEncoderB5


class VggBaseEncoder(BaseEncoderB5):
    def __init__(self, in_channels=3, *args, **kwargs) -> None:
        super().__init__(in_channels, *args, **kwargs)
