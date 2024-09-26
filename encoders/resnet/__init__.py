from .resnet18 import ResNetEncoder as resnet18_encoder
from .resnet34 import ResNetEncoder as resnet34_encoder
from .resnet50 import ResNetEncoder as resnet50_encoder
from .resnet101 import ResNetEncoder as resnet101_encoder
from .resnet152 import ResNetEncoder as resnet152_encoder

__all__ = [
    resnet18_encoder,
    resnet34_encoder,
    resnet50_encoder,
    resnet101_encoder,
    resnet152_encoder,
]
