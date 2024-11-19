from .vgg import vgg11_encoder, vgg13_encoder, vgg16_encoder, vgg19_encoder
from .resnet import (
    resnet18_encoder,
    resnet34_encoder,
    resnet50_encoder,
    resnet101_encoder,
    resnet152_encoder,
)
from .senet import senet154_encoder
from .cbam import cbamnet154_encoder
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
        self.encoder_name = encoder_name
        self.message = f"Encoder {encoder_name} not implemented"
        super().__init__(self.message, *args)


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
    "cbamnet154": cbamnet154_encoder,
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
    "resnet18": {
        "inputs": [768, 384, 192, 128, 32],
        "has_center": True,
        "center_add": 0,
    },
    "resnet34": {
        "inputs": [768, 384, 192, 128, 32],
        "has_center": True,
        "center_add": 0,
    },
    "resnet50": {"inputs": [3072, 768, 384, 128, 32]},
    "resnet101": {"inputs": [3072, 768, 384, 128, 32]},
    "resnet152": {"inputs": [3072, 768, 384, 128, 32]},
    "senet154": {"inputs": [3072, 768, 384, 192, 32]},
    "cbamnet154": {"inputs": [3072, 768, 384, 192, 32]},
    "efficientnetb0": {"inputs": [432, 296, 152, 96, 32]},
    "efficientnetb1": {"inputs": [432, 296, 152, 96, 32]},
    "efficientnetb2": {"inputs": [472, 304, 152, 96, 32]},
    "efficientnetb3": {"inputs": [520, 304, 160, 104, 32]},
    "efficientnetb4": {"inputs": [608, 312, 160, 112, 32]},
    "efficientnetb5": {"inputs": [688, 320, 168, 112, 32]},
    "efficientnetb6": {"inputs": [776, 328, 168, 120, 32]},
    "efficientnetb7": {"inputs": [864, 336, 176, 128, 32]},
}

unetplusplus_decoder_params = {
    "vgg11": {
        "inputs": [
            [512 + 512, 512 + 512, 512 + 256, 256 + 128],
            [256 + 1024, 512 + 512, 256 + 256],
            [128 + 768, 256 + 384],
            [64 + 512],
            32,
        ],
        "outputs": [[256, 512, 256, 128], [128, 256, 128], [64, 128], [32], 16],
    },
    "vgg13": {
        "inputs": [
            [512 + 512, 512 + 512, 512 + 256, 256 + 128],
            [256 + 1024, 512 + 512, 256 + 256],
            [128 + 768, 256 + 384],
            [64 + 512],
            32,
        ],
        "outputs": [[256, 512, 256, 128], [128, 256, 128], [64, 128], [32], 16],
    },
    "vgg16": {
        "inputs": [
            [512 + 512, 512 + 512, 512 + 256, 256 + 128],
            [256 + 1024, 512 + 512, 256 + 256],
            [128 + 768, 256 + 384],
            [64 + 512],
            32,
        ],
        "outputs": [[256, 512, 256, 128], [128, 256, 128], [64, 128], [32], 16],
    },
    "vgg19": {
        "inputs": [
            [512 + 512, 512 + 512, 512 + 256, 256 + 128],
            [256 + 1024, 512 + 512, 256 + 256],
            [128 + 768, 256 + 384],
            [64 + 512],
            32,
        ],
        "outputs": [[256, 512, 256, 128], [128, 256, 128], [64, 128], [32], 16],
    },
    "resnet18": {
        "inputs": [
            [512 + 256, 256 + 128, 128 + 64, 64 + 64],
            [256 + 256, 128 + 128, 64 + 128],
            [128 + 192, 64 + 192],
            [64 + 256],
            32,
        ],
        "outputs": [[256, 128, 64, 64], [128, 64, 64], [64, 64], [32], 16],
    },
    "resnet34": {
        "inputs": [
            [512 + 256, 256 + 128, 128 + 64, 64 + 64],
            [256 + 256, 128 + 128, 64 + 128],
            [128 + 192, 64 + 192],
            [64 + 256],
            32,
        ],
        "outputs": [[256, 128, 64, 64], [128, 64, 64], [64, 64], [32], 16],
    },
    "resnet50": {
        "inputs": [
            [2048 + 1024, 1024 + 512, 512 + 256, 256 + 64],
            [256 + 1024, 512 + 512, 256 + 128],
            [128 + 768, 256 + 192],
            [64 + 256],
            32,
        ],
        "outputs": [[256, 512, 256, 64], [128, 256, 64], [64, 64], [32], 16],
    },
    "resnet101": {
        "inputs": [
            [2048 + 1024, 1024 + 512, 512 + 256, 256 + 64],
            [256 + 1024, 512 + 512, 256 + 128],
            [128 + 768, 256 + 192],
            [64 + 256],
            32,
        ],
        "outputs": [[256, 512, 256, 64], [128, 256, 64], [64, 64], [32], 16],
    },
    "resnet152": {
        "inputs": [
            [2048 + 1024, 1024 + 512, 512 + 256, 256 + 64],
            [256 + 1024, 512 + 512, 256 + 128],
            [128 + 768, 256 + 192],
            [64 + 256],
            32,
        ],
        "outputs": [[256, 512, 256, 64], [128, 256, 64], [64, 64], [32], 16],
    },
    "senet154": {
        "inputs": [
            [2048 + 1024, 1024 + 512, 512 + 256, 256 + 128],
            [256 + 1024, 512 + 512, 256 + 256],
            [128 + 768, 256 + 384],
            [64 + 512],
            32,
        ],
        "outputs": [[256, 512, 256, 128], [128, 256, 128], [64, 128], [32], 16],
    },
    "cbamnet154": {
        "inputs": [
            [2048 + 1024, 1024 + 512, 512 + 256, 256 + 128],
            [256 + 1024, 512 + 512, 256 + 256],
            [128 + 768, 256 + 384],
            [64 + 512],
            32,
        ],
        "outputs": [[256, 512, 256, 128], [128, 256, 128], [64, 128], [32], 16],
    },
    "efficientnetb0": {
        "inputs": [
            [320 + 112, 112 + 40, 40 + 24, 24 + 32],
            [256 + 80, 40 + 48, 24 + 64],
            [128 + 72, 24 + 96],
            [64 + 128],
            32,
        ],
        "outputs": [[256, 40, 24, 32], [128, 24, 32], [64, 32], [32], 16],
    },
    "efficientnetb1": {
        "inputs": [
            [320 + 112, 112 + 40, 40 + 24, 24 + 32],
            [256 + 80, 40 + 48, 24 + 64],
            [128 + 72, 24 + 96],
            [64 + 128],
            32,
        ],
        "outputs": [[256, 40, 24, 32], [128, 24, 32], [64, 32], [32], 16],
    },
    "efficientnetb2": {
        "inputs": [
            [352 + 120, 120 + 48, 48 + 24, 24 + 32],
            [256 + 96, 48 + 48, 24 + 64],
            [128 + 72, 24 + 96],
            [64 + 128],
            32,
        ],
        "outputs": [[256, 48, 24, 32], [128, 24, 32], [64, 32], [32], 16],
    },
    "efficientnetb3": {
        "inputs": [
            [384 + 136, 136 + 48, 48 + 32, 32 + 40],
            [256 + 96, 48 + 64, 32 + 80],
            [128 + 96, 32 + 120],
            [64 + 160],
            32,
        ],
        "outputs": [[256, 48, 32, 40], [128, 32, 40], [64, 40], [32], 16],
    },
    "efficientnetb4": {
        "inputs": [
            [448 + 160, 160 + 56, 56 + 32, 32 + 48],
            [256 + 112, 56 + 64, 32 + 96],
            [128 + 96, 32 + 144],
            [64 + 192],
            32,
        ],
        "outputs": [[256, 56, 32, 48], [128, 32, 48], [64, 48], [32], 16],
    },
    "efficientnetb5": {
        "inputs": [
            [512 + 176, 176 + 64, 64 + 40, 40 + 48],
            [256 + 128, 64 + 80, 40 + 96],
            [128 + 120, 40 + 144],
            [64 + 192],
            32,
        ],
        "outputs": [[256, 64, 40, 48], [128, 40, 48], [64, 48], [32], 16],
    },
    "efficientnetb6": {
        "inputs": [
            [576 + 200, 200 + 72, 72 + 40, 40 + 56],
            [256 + 144, 72 + 80, 40 + 112],
            [128 + 120, 40 + 168],
            [64 + 224],
            32,
        ],
        "outputs": [[256, 72, 40, 56], [128, 40, 56], [64, 56], [32], 16],
    },
    "efficientnetb7": {
        "inputs": [
            [640 + 224, 224 + 80, 80 + 48, 48 + 64],
            [256 + 160, 80 + 96, 48 + 128],
            [128 + 144, 48 + 192],
            [64 + 256],
            32,
        ],
        "outputs": [[256, 80, 48, 64], [128, 48, 64], [64, 64], [32], 16],
    },
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
    "cbamnet154": {
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

fpn_decoder_params = {
    "vgg11": {
        "fpn_inputs": [512, 512, 256],
        "fpn_outputs": [256, 256, 256],
    },
    "vgg13": {
        "fpn_inputs": [512, 512, 256],
        "fpn_outputs": [256, 256, 256],
    },
    "vgg16": {
        "fpn_inputs": [512, 512, 256],
        "fpn_outputs": [256, 256, 256],
    },
    "vgg19": {
        "fpn_inputs": [512, 512, 256],
        "fpn_outputs": [256, 256, 256],
    },
    "resnet18": {
        "fpn_inputs": [256, 128, 64],
        "fpn_outputs": [256, 256, 256],
    },
    "resnet18": {
        "fpn_inputs": [256, 128, 64],
        "fpn_outputs": [256, 256, 256],
    },
    "resnet34": {
        "fpn_inputs": [256, 128, 64],
        "fpn_outputs": [256, 256, 256],
    },
    "resnet50": {
        "in_channels": 2048,
        "fpn_inputs": [1024, 512, 256],
        "fpn_outputs": [256, 256, 256],
    },
    "resnet101": {
        "in_channels": 2048,
        "fpn_inputs": [1024, 512, 256],
        "fpn_outputs": [256, 256, 256],
    },
    "resnet152": {
        "in_channels": 2048,
        "fpn_inputs": [1024, 512, 256],
        "fpn_outputs": [256, 256, 256],
    },
    "senet154": {
        "in_channels": 2048,
        "fpn_inputs": [1024, 512, 256],
        "fpn_outputs": [256, 256, 256],
    },
    "cbamnet154": {
        "in_channels": 2048,
        "fpn_inputs": [1024, 512, 256],
        "fpn_outputs": [256, 256, 256],
    },
    "efficientnetb0": {
        "in_channels": 320,
        "fpn_inputs": [112, 40, 24],
        "fpn_outputs": [256, 256, 256],
    },
    "efficientnetb1": {
        "in_channels": 320,
        "fpn_inputs": [112, 40, 24],
        "fpn_outputs": [256, 256, 256],
    },
    "efficientnetb2": {
        "in_channels": 352,
        "fpn_inputs": [120, 48, 24],
        "fpn_outputs": [256, 256, 256],
    },
    "efficientnetb3": {
        "in_channels": 384,
        "fpn_inputs": [136, 48, 32],
        "fpn_outputs": [256, 256, 256],
    },
    "efficientnetb4": {
        "in_channels": 448,
        "fpn_inputs": [160, 56, 32],
        "fpn_outputs": [256, 256, 256],
    },
    "efficientnetb5": {
        "in_channels": 512,
        "fpn_inputs": [176, 64, 40],
        "fpn_outputs": [256, 256, 256],
    },
    "efficientnetb6": {
        "in_channels": 576,
        "fpn_inputs": [200, 72, 40],
        "fpn_outputs": [256, 256, 256],
    },
    "efficientnetb7": {
        "in_channels": 640,
        "fpn_inputs": [224, 80, 48],
        "fpn_outputs": [256, 256, 256],
    },
}

deeplabv3plus_decoder_params = {
    "resnet18": {"kernels_in": 512, "kernels_out": 256},
    "resnet34": {"kernels_in": 512, "kernels_out": 256},
    "resnet50": {"kernels_in": 2048, "kernels_out": 256, "kernels": 256},
    "resnet101": {"kernels_in": 2048, "kernels_out": 256, "kernels": 256},
    "resnet152": {"kernels_in": 2048, "kernels_out": 256, "kernels": 256},
    "senet154": {"kernels_in": 2048, "kernels_out": 256, "kernels": 256},
    "cbamnet154": {"kernels_in": 2048, "kernels_out": 256, "kernels": 256},
    "efficientnetb0": {
        "kernels_in": 320,
        "kernels_out": 256,
        "kernels": 24,
        "groups": 256,
        "aspp_groups": 320,
    },
    "efficientnetb1": {
        "kernels_in": 320,
        "kernels_out": 256,
        "kernels": 24,
        "groups": 256,
        "aspp_groups": 320,
    },
    "efficientnetb2": {
        "kernels_in": 352,
        "kernels_out": 256,
        "kernels": 24,
        "groups": 256,
        "aspp_groups": 352,
    },
    "efficientnetb3": {
        "kernels_in": 384,
        "kernels_out": 256,
        "kernels": 32,
        "groups": 256,
        "aspp_groups": 384,
    },
    "efficientnetb4": {
        "kernels_in": 448,
        "kernels_out": 256,
        "kernels": 32,
        "groups": 256,
        "aspp_groups": 448,
    },
    "efficientnetb5": {
        "kernels_in": 512,
        "kernels_out": 256,
        "kernels": 40,
        "groups": 256,
        "aspp_groups": 512,
    },
    "efficientnetb6": {
        "kernels_in": 576,
        "kernels_out": 256,
        "kernels": 40,
        "groups": 256,
        "aspp_groups": 576,
    },
    "efficientnetb7": {
        "kernels_in": 640,
        "kernels_out": 256,
        "kernels": 48,
        "groups": 256,
        "aspp_groups": 640,
    },
}

pspnet_decoder_params = {
    "vgg11": {},
    "vgg13": {},
    "vgg16": {},
    "vgg19": {},
    "resnet18": {},
    "resnet34": {},
    "resnet50": {},
    "resnet101": {},
    "resnet152": {},
    "senet154": {},
    "cbamnet154": {},
    "efficientnetb0": {},
    "efficientnetb1": {},
    "efficientnetb2": {},
    "efficientnetb3": {},
    "efficientnetb4": {},
    "efficientnetb5": {},
    "efficientnetb6": {},
    "efficientnetb7": {},
}

manet_decoder_params = {
    "vgg11": {
        "inputs": [
            512,
            512,
            256,
            128,
            64,
            32,
        ],
        "mid_channels": [
            512,
            512,
            256,
            128,
        ],
        "outputs": [512, 256, 128, 64, 32, 16],
    },
    "vgg13": {
        "encoder_channels": [
            512,
            512,
            256,
            128,
            64,
        ],
        "decoder_channels": [64, 256, 128, 64, 32],
    },
    "vgg16": {
        "encoder_channels": [
            512,
            512,
            256,
            128,
            64,
        ],
        "decoder_channels": [64, 256, 128, 64, 32],
    },
}


def get_encoder(name, in_channels=3, wavelets_mode=False, deeplab_arch=False):
    if name not in encoders:
        raise EncoderException(encoder_name=name)
    return encoders[name](
        in_channels=in_channels, wavelets_mode=wavelets_mode, deeplab_arch=deeplab_arch
    )


def get_unet_decoder_params(name, wavelets_mode):
    if name not in unet_decoder_params:
        raise EncoderException(encoder_name=name)
    result = dict(unet_decoder_params)
    if (wavelets_mode == 2 or wavelets_mode == 3) and (
        name.startswith("resnet") or name.startswith("senet") or name.startswith("cbam")
    ):
        result[name]["inputs"][0] += 7
        result[name]["inputs"][1] += 2
        result[name]["inputs"][2] += 1
        result[name]["center_add"] = 4
    return result[name]


def get_unetplusplus_decoder_params(name, wavelets_mode):
    if name not in unetplusplus_decoder_params:
        raise EncoderException(encoder_name=name)
    if (wavelets_mode == 2 or wavelets_mode == 3) and (
        name.startswith("resnet") or name.startswith("senet") or name.startswith("cbam")
    ):
        unetplusplus_decoder_params[name]["inputs"][0] += 7
        unetplusplus_decoder_params[name]["inputs"][1] += 2
        unetplusplus_decoder_params[name]["inputs"][2] += 1
        unetplusplus_decoder_params[name]["center_add"] = 4
    return unetplusplus_decoder_params[name]


def get_linknet_decoder_params(name):
    try:
        return linknet_decoder_params[name]
    except KeyError:
        raise EncoderException(encoder_name=name)


def get_fpn_decoder_params(name):
    if name not in fpn_decoder_params:
        raise EncoderException(encoder_name=name)
    return fpn_decoder_params[name]


def get_deeplabv3plus_decoder_params(name):
    if name not in deeplabv3plus_decoder_params:
        raise EncoderException(encoder_name=name)
    return deeplabv3plus_decoder_params[name]


def get_pspnet_decoder_params(name):
    if name not in pspnet_decoder_params:
        raise EncoderException(encoder_name=name)
    return pspnet_decoder_params[name]


def get_manet_decoder_params(name, wavelets_mode):
    if name not in manet_decoder_params:
        raise EncoderException(encoder_name=name)
    return manet_decoder_params[name]


__all__ = [
    get_encoder,
    get_unet_decoder_params,
    get_unetplusplus_decoder_params,
    get_linknet_decoder_params,
    get_fpn_decoder_params,
    get_deeplabv3plus_decoder_params,
    get_pspnet_decoder_params,
    get_manet_decoder_params,
]
