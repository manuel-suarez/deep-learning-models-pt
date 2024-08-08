import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import MaxPool2d, padding


class ResidualBlock(nn.Module):
    def __init__(
        self, kernels_in, kernels_out, stride=1, has_downsample=False, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # Conv+BatchNorm+ReLU
        self.conv1 = nn.Conv2d(
            kernels_in,
            kernels_out,
            kernel_size=(
                3,
                3,
            ),
            stride=(stride, stride),
            padding=(1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(
            kernels_out, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            kernels_out,
            kernels_out,
            kernel_size=(
                3,
                3,
            ),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(
            kernels_out, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.has_downsample = has_downsample
        if has_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    kernels_in,
                    kernels_out,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(
                    kernels_out,
                    eps=1e-5,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
            )

    def forward(self, x, res):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.has_downsample:
            res = self.downsample(res)
        x = torch.add(x, res)
        x = self.relu2(x)
        return x


class UnetResNet34(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ## Encoder
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(
                3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            ),
            nn.BatchNorm2d(
                64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.encoder_block2 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
            ),
            nn.Sequential(
                ResidualBlock(64, 64, has_downsample=False, stride=1),
                ResidualBlock(64, 64, has_downsample=False, stride=1),
                ResidualBlock(64, 64, has_downsample=False, stride=1),
            ),
        )
        self.encoder_block3 = nn.Sequential(
            ResidualBlock(64, 128, has_downsample=True, stride=2),
            ResidualBlock(128, 128, has_downsample=False, stride=1),
            ResidualBlock(128, 128, has_downsample=False, stride=1),
            ResidualBlock(128, 128, has_downsample=False, stride=1),
        )
        self.encoder_block4 = nn.Sequential(
            ResidualBlock(128, 256, has_downsample=True, stride=2),
            ResidualBlock(256, 256, has_downsample=False, stride=1),
            ResidualBlock(256, 256, has_downsample=False, stride=1),
            ResidualBlock(256, 256, has_downsample=False, stride=1),
            ResidualBlock(256, 256, has_downsample=False, stride=1),
            ResidualBlock(256, 256, has_downsample=False, stride=1),
        )
        self.encoder_block5 = nn.Sequential(
            ResidualBlock(256, 512, has_downsample=True, stride=2),
            ResidualBlock(512, 512, has_downsample=False, stride=1),
            ResidualBlock(512, 512, has_downsample=False, stride=1),
        )

        ### Decoder
        self.decoder_block5 = nn.Sequential(
            nn.Conv2d(
                768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(
                256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(
                256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.decoder_block4 = nn.Sequential(
            nn.Conv2d(
                384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(
                128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(
                128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.decoder_block3 = nn.Sequential(
            nn.Conv2d(
                192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(
                64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(
                64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(
                128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(
                32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(
                32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(
                32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(
                16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(
                16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.segmentation_block = nn.Sequential(
            nn.Conv2d(
                16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        c1 = self.encoder_block1(inputs)
        c2 = self.encoder_block2[0](c1)
        c2 = self.encoder_block2[1][0](c2, c2)
        c2 = self.encoder_block2[1][1](c2, c2)
        c2 = self.encoder_block2[1][2](c2, c2)
        c3 = self.encoder_block3[0](c2, c2)
        c3 = self.encoder_block3[1](c3, c3)
        c3 = self.encoder_block3[2](c3, c3)
        c3 = self.encoder_block3[3](c3, c3)
        c4 = self.encoder_block4[0](c3, c3)
        c4 = self.encoder_block4[1](c4, c4)
        c4 = self.encoder_block4[2](c4, c4)
        c4 = self.encoder_block4[3](c4, c4)
        c4 = self.encoder_block4[4](c4, c4)
        c4 = self.encoder_block4[5](c4, c4)
        c5 = self.encoder_block5[0](c4, c4)
        c5 = self.encoder_block5[1](c5, c5)
        c5 = self.encoder_block5[2](c5, c5)
        # c3 = self.encoder_block3(c2)
        # c4 = self.encoder_block4(c3)
        # c5 = self.encoder_block5(c4)
        # ap = self.avgpool_block(c5)
        # d5 = self.bottleneck(ap)
        # d5 = F.interpolate(d5, scale_factor=2, mode="nearest")
        # c5 = torch.cat([c5, d5], dim=1)
        # d5 = self.decoder_block5(c5)
        d4 = F.interpolate(c5, scale_factor=2, mode="nearest")
        c4 = torch.cat([c4, d4], dim=1)
        d4 = self.decoder_block5(c4)
        d3 = F.interpolate(d4, scale_factor=2, mode="nearest")
        c3 = torch.cat([c3, d3], dim=1)
        d3 = self.decoder_block4(c3)
        d2 = F.interpolate(d3, scale_factor=2, mode="nearest")
        c2 = torch.cat([c2, d2], dim=1)
        d2 = self.decoder_block3(c2)
        d1 = F.interpolate(d2, scale_factor=2, mode="nearest")
        c1 = torch.cat([c1, d1], dim=1)
        d1 = self.decoder_block2(c1)
        d0 = F.interpolate(d1, scale_factor=2, mode="nearest")
        d0 = self.decoder_block1(d0)
        outputs = self.segmentation_block(d0)

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
