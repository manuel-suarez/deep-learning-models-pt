from fpn import FPN


class FPNVgg11(FPN):
    def __init__(self) -> None:
        super().__init__("vgg11")


if __name__ == "__main__":
    from torchview import draw_graph

    model = FPNVgg11()
    print(model)
    draw_graph(
        model,
        input_size=(1, 3, 224, 224),
        depth=5,
        show_shapes=True,
        expand_nested=True,
        save_graph=True,
        filename="fpn+vgg11",
        directory="figures",
    )
