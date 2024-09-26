from .unet import Unet
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
from .unet_efficientnetb7 import UnetEfficientNetB7


def get_model(arch, args, encoder=None):
    if arch == "unet":
        return Unet(encoder_name=encoder, **args)
