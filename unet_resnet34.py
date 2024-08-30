import torch.nn as nn
from .encoders.resnet.resnet34 import ResNetEncoder
from .decoders.unet import UnetDecoder
from .segmentation.segmentationhead import SegmentationHead


class UnetResNet34(nn.Module):
    def __init__(self, in_channels=3) -> None:
        super().__init__()
        ## Encoder
        self.encoder = ResNetEncoder(in_channels=in_channels)
        ## Decoder
        self.decoder = UnetDecoder(inputs=[768, 384, 192, 128, 32])
        ## Segmentation Head
        self.segmentation_head = SegmentationHead(has_activation=False)

    def forward(self, inputs):
        c1, c2, c3, c4, c5 = self.encoder(inputs)
        d1 = self.decoder(c5, [c1, c2, c3, c4])
        outputs = self.segmentation_head(d1)

        return outputs


if __name__ == "__main__":
    from torchview import draw_graph

    model = UnetResNet34()
    print(model)
    draw_graph(
        model,
        input_size=(1, 3, 224, 224),
        depth=5,
        show_shapes=True,
        expand_nested=True,
        save_graph=True,
        filename="unet+resnet34",
        directory="figures",
    )
