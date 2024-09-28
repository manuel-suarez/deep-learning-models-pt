from .fpn import FPN
from .unet import Unet
from .linknet import Linknet
from .pspnet import PSPNet


def get_model(arch, args, encoder=None):
    if arch == "unet":
        return Unet(encoder_name=encoder, **args)
    if arch == "linknet":
        return Linknet(encoder_name=encoder, **args)
    if arch == "fpn":
        return FPN(encoder_name=encoder, **args)
    if arch == "pspnet":
        return PSPNet(encoder_name=encoder, **args)
