from models.unet_vgg11 import UnetVgg11
from models.unet_vgg13 import UnetVgg13
from models.unet_vgg16 import UnetVgg16
from models.unet_vgg19 import UnetVgg19
from models.unet_resnet18 import UnetResNet18
from models.unet_resnet34 import UnetResNet34
from models.unet_resnet50 import UnetResNet50
from models.unet_resnet101 import UnetResNet101
from models.unet_resnet152 import UnetResNet152
from models.unet_senet154 import UnetSENet154
from models.unet_efficientnetb0 import UnetEfficientNetB0


def get_model(arch, args, encoder=None):
    if arch == "unet":
        if encoder == "vgg11":
            return UnetVgg11(**args)
        if encoder == "vgg13":
            return UnetVgg13(**args)
        if encoder == "vgg16":
            return UnetVgg16(**args)
        if encoder == "vgg19":
            return UnetVgg19(**args)
        if encoder == "resnet18":
            return UnetResNet18(**args)
        if encoder == "resnet34":
            return UnetResNet34(**args)
        if encoder == "resnet50":
            return UnetResNet50(**args)
        if encoder == "resnet101":
            return UnetResNet101(**args)
        if encoder == "resnet152":
            return UnetResNet152(**args)
        if encoder == "senet154":
            return UnetSENet154(**args)
        if encoder == "efficientnetb0":
            return UnetEfficientNetB0(**args)
