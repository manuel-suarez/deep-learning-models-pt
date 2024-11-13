import torch.nn as nn
from models.encoders import get_encoder, get_deeplabv3plus_decoder_params
from models.decoders.deeplabv3plus import DeepLabV3PlusDecoder
from models.segmentation.segmentationhead import SegmentationHead


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        encoder_name,
        in_channels=3,
        out_channels=1,
        activation=False,
        wavelets_mode=False,
    ) -> None:
        super().__init__()
        print(f"Initializing DeepLabV3Plus model, wavelets mode: {wavelets_mode}")
        ## Encoder
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            wavelets_mode=wavelets_mode,
            deeplab_arch=True,
        )
        ## Decoder
        self.decoder = DeepLabV3PlusDecoder(
            **get_deeplabv3plus_decoder_params(encoder_name)
        )
        ## Segmentation
        self.segmentation = SegmentationHead(
            kernels_in=256, out_channels=out_channels, has_activation=activation
        )

    def forward(self, inputs):
        _, c2, _, _, c5 = self.encoder(inputs)
        d1 = self.decoder(c5, c2)
        outputs = self.segmentation(d1)

        return outputs
