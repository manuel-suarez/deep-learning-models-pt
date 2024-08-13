from .vgg import vgg11_encoder, vgg13_encoder, vgg16_encoder, vgg19_encoder


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
}

unet_decoder_params = {
    "vgg11": {"inputs": [1024, 768, 384, 192, 32], "has_center": True},
    "vgg13": {"inputs": [1024, 768, 384, 192, 32], "has_center": True},
    "vgg16": {"inputs": [1024, 768, 384, 192, 32], "has_center": True},
    "vgg19": {"inputs": [1024, 768, 384, 192, 32], "has_center": True},
}


def get_encoder(name):
    try:
        return encoders[name]()
    except KeyError:
        raise EncoderException(encoder_name=name)


def get_unet_decoder_params(name):
    try:
        return unet_decoder_params[name]
    except KeyError:
        raise EncoderException(encoder_name=name)


__all__ = [
    get_encoder,
    get_unet_decoder_params,
    vgg11_encoder,
    vgg13_encoder,
    vgg16_encoder,
    vgg19_encoder,
]
