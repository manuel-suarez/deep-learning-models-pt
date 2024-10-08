import torch.nn as nn
from .encoders import get_encoder, get_linknet_decoder_params
from .decoders.linknet import LinknetDecoder
from .segmentation.segmentationhead import SegmentationHead


class Linknet(nn.Module):
    def __init__(
        self,
        encoder_name,
        in_channels=3,
        out_channels=1,
        activation=False,
        wavelets_mode=False,
    ) -> None:
        super().__init__()
        ## Encoder
        self.encoder = get_encoder(
            encoder_name, in_channels=in_channels, wavelets_mode=wavelets_mode
        )
        ## Decoder
        self.decoder = LinknetDecoder(**get_linknet_decoder_params(encoder_name))
        ## Segmentation
        self.segmentation = SegmentationHead(
            kernels_in=32, out_channels=out_channels, has_activation=activation
        )

    def forward(self, inputs):
        c1, c2, c3, c4, c5 = self.encoder(inputs)
        d1 = self.decoder(c5, [c1, c2, c3, c4])
        outputs = self.segmentation(d1)

        return outputs
