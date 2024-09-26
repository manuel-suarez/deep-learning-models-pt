from .unet import Unet


def get_model(arch, args, encoder=None):
    if arch == "unet":
        return Unet(encoder_name=encoder, **args)
