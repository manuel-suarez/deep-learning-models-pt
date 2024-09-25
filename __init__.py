from .unet_vgg11 import UnetVgg11
from .unet_vgg13 import UnetVgg13
from .unet_vgg16 import UnetVgg16
from .unet_vgg19 import UnetVgg19
from .unet_resnet18 import UnetResNet18
from .unet_resnet34 import UnetResNet34
from .unet_resnet50 import UnetResNet50
from .unet_resnet101 import UnetResNet101
from .unet_resnet152 import UnetResNet152
from .unet_senet154 import UnetSENet154
from .unet_efficientnetb0 import UnetEfficientNetB0
from .unet_efficientnetb1 import UnetEfficientNetB1
from .unet_efficientnetb2 import UnetEfficientNetB2
from .unet_efficientnetb3 import UnetEfficientNetB3
from .unet_efficientnetb4 import UnetEfficientNetB4
from .unet_efficientnetb5 import UnetEfficientNetB5
from .unet_efficientnetb6 import UnetEfficientNetB6


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
        if encoder == "efficientnetb1":
            return UnetEfficientNetB1(**args)
        if encoder == "efficientnetb2":
            return UnetEfficientNetB2(**args)
        if encoder == "efficientnetb3":
            return UnetEfficientNetB3(**args)
        if encoder == "efficientnetb4":
            return UnetEfficientNetB4(**args)
        if encoder == "efficientnetb5":
            return UnetEfficientNetB5(**args)
        if encoder == "efficientnetb6":
            return UnetEfficientNetB6(**args)
