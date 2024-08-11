import torch.nn as nn
from encoders.resnet.resnet101 import ResNetEncoder
from decoders.unet import UnetDecoder
from segmentation.segmentationhead import SegmentationHead


class UnetResNet101(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ## Encoder
        self.encoder = ResNetEncoder()
        ## Decoder
        self.decoder = UnetDecoder()
        ## Segmentation Head
        self.segmentation_head = SegmentationHead()

    def forward(self, inputs):
        c1, c2, c3, c4, c5 = self.encoder(inputs)
        d1 = self.decoder(c5, [c1, c2, c3, c4])
        outputs = self.segmentation_head(d1)
        return outputs


if __name__ == "__main__":
    from torchview import draw_graph

    model = UnetResNet101()
    print(model)
    draw_graph(
        model,
        input_size=(1, 3, 224, 224),
        depth=5,
        show_shapes=True,
        expand_nested=True,
        save_graph=True,
        filename="unet+resnet101",
        directory="figures",
    )