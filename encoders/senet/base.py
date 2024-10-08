from models.encoders.base import BaseEncoderB6


class SeNetBaseEncoder(BaseEncoderB6):
    def __init__(self, in_channels=3, wavelets_mode=False, *args, **kwargs) -> None:
        super().__init__(in_channels, wavelets_mode, *args, **kwargs)
