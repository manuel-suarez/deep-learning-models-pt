import torch.nn as nn
from encoders import get_encoder, get_unet_decoder_params
from decoders.unet import UnetDecoder
from segmentation.segmentationhead import SegmentationHead


class Unet(nn.Module):
    def __init__(self, encoder_name, activation=False) -> None:
        super().__init__()
        ## Encoder
        self.encoder = get_encoder(encoder_name)
        ## Decoder
        self.decoder = UnetDecoder(**get_unet_decoder_params(encoder_name))
        ## Segmentation
        self.segmentation = SegmentationHead(activation)

    def forward(self, inputs):
        c1, c2, c3, c4, c5 = self.encoder(inputs)
        d1 = self.decoder(c5, [c1, c2, c3, c4])
        outputs = self.segmentation(d1)

        return outputs
