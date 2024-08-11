import torch.nn as nn


class Activation(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(x)
