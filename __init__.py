from models.unet_resnet34 import UnetResNet34


def get_model(arch, args, encoder=None):
    if arch == "unet":
        if encoder == "resnet34":
            return UnetResNet34(**args)
