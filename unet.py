import torch.nn as nn
from models.encoders import get_encoder, get_unet_decoder_params
from models.decoders.unet import UnetDecoder
from models.segmentation.segmentationhead import SegmentationHead


class Unet(nn.Module):
    def __init__(
        self,
        encoder_name,
        in_channels=3,
        out_channels=1,
        activation=False,
        wavelets_mode=False,
    ) -> None:
        super().__init__()
        print(f"Initializing Unet model, wavelets mode: {wavelets_mode}")
        ## Encoder
        self.encoder = get_encoder(
            encoder_name, in_channels=in_channels, wavelets_mode=wavelets_mode
        )
        ## Decoder
        self.decoder = UnetDecoder(
            **get_unet_decoder_params(encoder_name, wavelets_mode)
        )
        ## Segmentation
        self.segmentation = SegmentationHead(
            out_channels=out_channels, has_activation=activation
        )

    def forward(self, inputs):
        c1, c2, c3, c4, c5 = self.encoder(inputs)
        d1 = self.decoder(c5, [c1, c2, c3, c4])
        outputs = self.segmentation(d1)

        return outputs
