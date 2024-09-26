from .vgg import vgg11_encoder, vgg13_encoder, vgg16_encoder, vgg19_encoder
from .resnet import (
    resnet18_encoder,
    resnet34_encoder,
    resnet50_encoder,
    resnet101_encoder,
    resnet152_encoder,
)
from .senet import senet154_encoder
from .efficientnet import (
    efficientnetb0_encoder,
    efficientnetb1_encoder,
    efficientnetb2_encoder,
    efficientnetb3_encoder,
    efficientnetb4_encoder,
    efficientnetb5_encoder,
    efficientnetb6_encoder,
    efficientnetb7_encoder,
)


class EncoderException(Exception):
    def __init__(self, encoder_name, *args: object) -> None:
        super().__init__(*args)
        self.encoder_name = encoder_name
        self.message = f"Encoder {encoder_name} not implemented"


encoders = {
    "vgg11": vgg11_encoder,
    "vgg13": vgg13_encoder,
    "vgg16": vgg16_encoder,
    "vgg19": vgg19_encoder,
    "resnet18": resnet18_encoder,
    "resnet34": resnet34_encoder,
    "resnet50": resnet50_encoder,
    "resnet101": resnet101_encoder,
    "resnet152": resnet152_encoder,
    "senet154": senet154_encoder,
    "efficientnetb0": efficientnetb0_encoder,
    "efficientnetb1": efficientnetb1_encoder,
    "efficientnetb2": efficientnetb2_encoder,
    "efficientnetb3": efficientnetb3_encoder,
    "efficientnetb4": efficientnetb4_encoder,
    "efficientnetb5": efficientnetb5_encoder,
    "efficientnetb6": efficientnetb6_encoder,
    "efficientnetb7": efficientnetb7_encoder,
}

unet_decoder_params = {
    "vgg11": {"inputs": [1024, 768, 384, 192, 32], "has_center": True},
    "vgg13": {"inputs": [1024, 768, 384, 192, 32], "has_center": True},
    "vgg16": {"inputs": [1024, 768, 384, 192, 32], "has_center": True},
    "vgg19": {"inputs": [1024, 768, 384, 192, 32], "has_center": True},
    "resnet18": {"inputs": [768, 384, 192, 128, 32], "has_center": True},
    "resnet34": {"inputs": [768, 384, 192, 128, 32], "has_center": True},
    "resnet50": {},
    "resnet101": {},
    "resnet152": {},
    "senet154": {"inputs": [3072, 768, 384, 192, 32]},
    "efficientnetb0": {"inputs": [432, 296, 152, 96, 32]},
    "efficientnetb1": {"inputs": [432, 296, 152, 96, 32]},
    "efficientnetb2": {"inputs": [472, 304, 152, 96, 32]},
    "efficientnetb3": {"inputs": [520, 304, 160, 104, 32]},
    "efficientnetb4": {"inputs": [608, 312, 160, 112, 32]},
    "efficientnetb5": {"inputs": [688, 320, 168, 112, 32]},
    "efficientnetb6": {"inputs": [776, 328, 168, 120, 32]},
    "efficientnetb7": {"inputs": [864, 336, 176, 128, 32]},
}

linknet_decoder_params = {
    "vgg11": {
        "inputs": [512, 512, 512, 256, 128],
        "mids": [128, 128, 128, 64, 32],
        "outputs": [512, 512, 256, 128, 32],
    },
    "vgg13": {
        "inputs": [512, 512, 512, 256, 128],
        "mids": [128, 128, 128, 64, 32],
        "outputs": [512, 512, 256, 128, 32],
    },
    "vgg16": {
        "inputs": [512, 512, 512, 256, 128],
        "mids": [128, 128, 128, 64, 32],
        "outputs": [512, 512, 256, 128, 32],
    },
    "vgg19": {
        "inputs": [512, 512, 512, 256, 128],
        "mids": [128, 128, 128, 64, 32],
        "outputs": [512, 512, 256, 128, 32],
    },
    "resnet18": {
        "inputs": [512, 256, 128, 64, 64],
        "mids": [128, 64, 32, 16, 16],
        "outputs": [256, 128, 64, 64, 32],
    },
    "resnet34": {
        "inputs": [512, 256, 128, 64, 64],
        "mids": [128, 64, 32, 16, 16],
        "outputs": [256, 128, 64, 64, 32],
    },
    "resnet50": {
        "inputs": [2048, 1024, 512, 256, 64],
        "mids": [512, 256, 128, 64, 16],
        "outputs": [1024, 512, 256, 64, 32],
    },
    "resnet101": {
        "inputs": [2048, 1024, 512, 256, 64],
        "mids": [512, 256, 128, 64, 16],
        "outputs": [1024, 512, 256, 64, 32],
    },
    "resnet152": {
        "inputs": [2048, 1024, 512, 256, 64],
        "mids": [512, 256, 128, 64, 16],
        "outputs": [1024, 512, 256, 64, 32],
    },
    "senet154": {
        "inputs": [2048, 1024, 512, 256, 128],
        "mids": [512, 256, 128, 64, 32],
        "outputs": [1024, 512, 256, 128, 32],
    },
    "efficientnetb0": {
        "inputs": [320, 112, 40, 24, 32],
        "mids": [80, 28, 10, 6, 8],
        "outputs": [112, 40, 24, 32, 32],
    },
    "efficientnetb1": {
        "inputs": [320, 112, 40, 24, 32],
        "mids": [80, 28, 10, 6, 8],
        "outputs": [112, 40, 24, 32, 32],
    },
    "efficientnetb2": {
        "inputs": [352, 120, 48, 24, 32],
        "mids": [88, 30, 12, 6, 8],
        "outputs": [120, 48, 24, 32, 32],
    },
    "efficientnetb3": {
        "inputs": [384, 136, 48, 32, 40],
        "mids": [96, 34, 12, 8, 10],
        "outputs": [136, 48, 32, 40, 32],
    },
    "efficientnetb4": {
        "inputs": [448, 160, 56, 32, 48],
        "mids": [112, 40, 14, 8, 12],
        "outputs": [160, 56, 32, 48, 32],
    },
    "efficientnetb5": {
        "inputs": [512, 176, 64, 40, 48],
        "mids": [128, 44, 16, 10, 12],
        "outputs": [176, 64, 40, 48, 32],
    },
    "efficientnetb6": {
        "inputs": [576, 200, 72, 40, 56],
        "mids": [144, 50, 18, 10, 14],
        "outputs": [200, 72, 40, 56, 32],
    },
    "efficientnetb7": {
        "inputs": [640, 224, 80, 48, 64],
        "mids": [160, 56, 20, 12, 16],
        "outputs": [224, 80, 48, 64, 32],
    },
}

fpn_decoder_params = {"vgg11": {}}


def get_encoder(name, in_channels=3):
    try:
        return encoders[name](in_channels=in_channels)
    except KeyError:
        raise EncoderException(encoder_name=name)


def get_unet_decoder_params(name):
    try:
        return unet_decoder_params[name]
    except KeyError:
        raise EncoderException(encoder_name=name)


def get_linknet_decoder_params(name):
    try:
        return linknet_decoder_params[name]
    except KeyError:
        raise EncoderException(encoder_name=name)


def get_fpn_decoder_params(name):
    try:
        return fpn_decoder_params[name]
    except KeyError:
        raise EncoderException(encoder_name=name)


__all__ = [
    get_encoder,
    get_unet_decoder_params,
    get_linknet_decoder_params,
    get_fpn_decoder_params,
]
