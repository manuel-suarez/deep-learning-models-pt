from models.encoders.base import BaseEncoderB6


class ResNetBaseEncoder(BaseEncoderB6):
    def __init__(self, in_channels=3, *args, **kwargs) -> None:
        super().__init__(in_channels, *args, **kwargs)
