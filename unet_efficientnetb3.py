import torch.nn as nn
from .encoders.efficientnet.efficientnetb3 import EfficientNetEncoder
from .decoders.unet import UnetDecoder
from .segmentation.segmentationhead import SegmentationHead


class UnetEfficientNetB3(nn.Module):
    def __init__(self, in_channels=3, out_channels=1) -> None:
        super().__init__()
        ## Encoder
        self.encoder = EfficientNetEncoder(in_channels=in_channels)
        ## Decoder
        self.decoder = UnetDecoder(inputs=[520, 304, 160, 104, 32])
        ## Segmentation
        self.segmentation = SegmentationHead(
            out_channels=out_channels, has_activation=False
        )

    def forward(self, inputs):
        c1, c2, c3, c4, c5 = self.encoder(inputs)
        d1 = self.decoder(c5, [c1, c2, c3, c4])
        outputs = self.segmentation(d1)

        return outputs


if __name__ == "__main__":
    from torchview import draw_graph

    model = UnetEfficientNetB3()
    print(model)
    draw_graph(
        model,
        input_size=(1, 3, 224, 224),
        depth=5,
        show_shapes=True,
        expand_nested=True,
        save_graph=True,
        filename="unet+efficientnetb3",
        directory="figures",
    )
