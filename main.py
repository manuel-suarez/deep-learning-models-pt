from unet_vgg16 import UnetVgg16
from torchview import draw_graph

model = UnetVgg16()
print(model)
draw_graph(
    model,
    input_size=(1, 3, 224, 224),
    depth=5,
    expand_nested=True,
    save_graph=True,
    filename="unet+vgg16",
)
