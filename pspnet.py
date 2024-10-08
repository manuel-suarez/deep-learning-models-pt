import torch.nn as nn
from .encoders import get_encoder, get_pspnet_decoder_params
from .decoders.pspnet import PSPDecoder
from .segmentation.segmentationhead import SegmentationHead


class PSPNet(nn.Module):
    def __init__(
        self,
        encoder_name,
        in_channels=3,
        out_channels=1,
        activation=False,
        wavelets_mode=False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        ## Encoder
        self.encoder = get_encoder(
            encoder_name, in_channels=in_channels, wavelets_mode=wavelets_mode
        )
        ## Decoder
        self.decoder = PSPDecoder(**get_pspnet_decoder_params(encoder_name))
        ## Segmentation
        self.segmentation = SegmentationHead(
            kernels_in=512, out_channels=out_channels, has_activation=activation
        )

    def forward(self, inputs):
        _, _, _, _, c5 = self.encoder(inputs)
        d1 = self.decoder(c5)
        outputs = self.segmentation(d1)

        return outputs
