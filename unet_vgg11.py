from unet import Unet


class UnetVgg11(Unet):
    def __init__(self, in_channels=3, out_channels=1) -> None:
        super().__init__("vgg11", in_channels=in_channels, out_channels=out_channels)


if __name__ == "__main__":
    from torchview import draw_graph

    model = UnetVgg11()
    print(model)
    draw_graph(
        model,
        input_size=(1, 3, 224, 224),
        depth=5,
        show_shapes=True,
        expand_nested=True,
        save_graph=True,
        filename="unet+vgg11",
        directory="figures",
    )
