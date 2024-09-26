import torch.nn as nn
from encoders import get_encoder, get_fpn_decoder_params
from decoders.fpn import FPNDecoder
from segmentation.segmentationhead import SegmentationHead


class FPN(nn.Module):
    def __init__(
        self, encoder_name, in_channels=3, out_channels=1, activation=False
    ) -> None:
        super().__init__()
        ## Encoder
        self.encoder = get_encoder(encoder_name, in_channels=in_channels)
        ## Decoder
        self.decoder = FPNDecoder(**get_fpn_decoder_params(encoder_name))
        ## Segmentation
        self.segmentation = SegmentationHead(
            kernels_in=128, out_channels=out_channels, has_activation=activation
        )

    def forward(self, inputs):
        _, c2, c3, c4, c5 = self.encoder(inputs)
        d1 = self.decoder(c5, [c2, c3, c4])
        outputs = self.segmentation(d1)

        return outputs
