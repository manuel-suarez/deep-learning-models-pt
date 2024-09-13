from models.unet_resnet34 import UnetResNet34


def get_model(arch, encoder=None):
    if arch == "unet":
        if encoder == "resnet34":
            return UnetResNet34()
