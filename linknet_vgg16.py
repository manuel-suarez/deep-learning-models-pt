from linknet import Linknet


class LinknetVgg16(Linknet):
    def __init__(self) -> None:
        super().__init__("vgg16")


if __name__ == "__main__":
    from torchview import draw_graph

    model = LinknetVgg16()
    print(model)
    draw_graph(
        model,
        input_size=(1, 3, 224, 224),
        depth=5,
        show_shapes=True,
        expand_nested=True,
        save_graph=True,
        filename="linknet+vgg16",
        directory="figures",
    )
