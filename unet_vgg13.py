from unet import Unet


class UnetVgg13(Unet):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("vgg13", *args, **kwargs)


if __name__ == "__main__":
    from torchview import draw_graph

    model = UnetVgg13()
    print(model)
    draw_graph(
        model,
        input_size=(1, 3, 224, 224),
        depth=5,
        show_shapes=True,
        expand_nested=True,
        save_graph=True,
        filename="unet+vgg13",
        directory="figures",
    )
