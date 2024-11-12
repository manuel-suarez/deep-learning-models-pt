import torch.nn as nn
from models.encoders import get_encoder, get_unetplusplus_decoder_params
from models.decoders.unetplusplus import UnetPlusPlusDecoder
from models.segmentation.segmentationhead import SegmentationHead


class UnetPlusPlus(nn.Module):
    def __init__(
        self,
        encoder_name,
        in_channels=3,
        out_channels=1,
        activation=False,
        wavelets_mode=False,
    ) -> None:
        super().__init__()
        print(f"Initializing UnetPlusPlus model, wavelets mode: {wavelets_mode}")
        ## Encoder
        self.encoder = get_encoder(
            encoder_name, in_channels=in_channels, wavelets_mode=wavelets_mode
        )
        ## Decoder
        self.decoder = UnetPlusPlusDecoder(
            **get_unetplusplus_decoder_params(encoder_name, wavelets_mode)
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
