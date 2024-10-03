from models.encoder.base import BaseEncoder


class ResNetBaseEncoder(BaseEncoder):
    def __init__(self, in_channels=3, *args, **kwargs) -> None:
        super().__init__(in_channels, args, kwargs)
