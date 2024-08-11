import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(
        self,
        kernels_in,
        kernels_bt,
        kernels_out,
        stride=1,
        has_downsample=False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # Conv+BatchNorm+ReLU
        self.conv1 = nn.Conv2d(
            kernels_in,
            kernels_bt,
            kernel_size=(
                3,
                3,
            ),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(
            kernels_bt, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            kernels_bt,
            kernels_bt,
            kernel_size=(
                3,
                3,
            ),
            stride=(stride, stride),
            padding=(1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(
            kernels_bt, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            kernels_bt,
            kernels_out,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(
            kernels_out, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False
        )
        self.relu3 = nn.ReLU(inplace=True)

        self.has_downsample = has_downsample
        if has_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    kernels_in,
                    kernels_out,
                    kernel_size=(3, 3),
                    stride=(stride, stride),
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

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = self.conv3(x1)
        x1 = self.bn3(x1)
        if self.has_downsample:
            x2 = self.downsample(x)
            x = torch.add(x1, x2)
        else:
            x = torch.add(x1, x)
        x = self.relu3(x)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
                Bottleneck(64, 64, 256, has_downsample=True, stride=1),
                Bottleneck(256, 64, 256, has_downsample=False, stride=1),
                Bottleneck(256, 64, 256, has_downsample=False, stride=1),
            ),
        )
        self.encoder_block3 = nn.Sequential(
            Bottleneck(256, 128, 512, has_downsample=True, stride=2),
            Bottleneck(512, 128, 512, has_downsample=False, stride=1),
            Bottleneck(512, 128, 512, has_downsample=False, stride=1),
            Bottleneck(512, 128, 512, has_downsample=False, stride=1),
            Bottleneck(512, 128, 512, has_downsample=False, stride=1),
            Bottleneck(512, 128, 512, has_downsample=False, stride=1),
            Bottleneck(512, 128, 512, has_downsample=False, stride=1),
            Bottleneck(512, 128, 512, has_downsample=False, stride=1),
        )
        self.encoder_block4 = nn.Sequential(
            Bottleneck(512, 256, 1024, has_downsample=True, stride=2),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
            Bottleneck(1024, 256, 1024, has_downsample=False, stride=1),
        )
        self.encoder_block5 = nn.Sequential(
            Bottleneck(1024, 512, 2048, has_downsample=True, stride=2),
            Bottleneck(2048, 512, 2048, has_downsample=False, stride=1),
            Bottleneck(2048, 512, 2048, has_downsample=False, stride=1),
        )

    def forward(self, x):
        c1 = self.encoder_block1(x)
        c2 = self.encoder_block2(c1)
        c3 = self.encoder_block3(c2)
        c4 = self.encoder_block4(c3)
        c5 = self.encoder_block5(c4)
        return c1, c2, c3, c4, c5


class Conv2dReLU(nn.Module):
    def __init__(self, kernels_in, kernels_out):
        super().__init__()
        self.conv = nn.Conv2d(
            kernels_in,
            kernels_out,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(
            kernels_out, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, kernels_in, kernels_out):
        super().__init__()
        self.conv2drelu1 = Conv2dReLU(kernels_in, kernels_out)
        self.conv2drelu2 = Conv2dReLU(kernels_out, kernels_out)

    def forward(self, x, skip=None):
        x = nn.functional.interpolate(x, scale_factor=2)
        if skip != None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv2drelu1(x)
        x = self.conv2drelu2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decoder_block1 = DecoderBlock(3072, 256)
        self.decoder_block2 = DecoderBlock(768, 128)
        self.decoder_block3 = DecoderBlock(384, 64)
        self.decoder_block4 = DecoderBlock(128, 32)
        self.decoder_block5 = DecoderBlock(32, 16)

    def forward(self, x, skips):
        c1, c2, c3, c4 = skips
        x = self.decoder_block1(x, c4)
        x = self.decoder_block2(x, c3)
        x = self.decoder_block3(x, c2)
        x = self.decoder_block4(x, c1)
        x = self.decoder_block5(x)
        return x


class Activation(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(x)


class SegmentationHead(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(
            16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.act = Activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class UnetResNet152(nn.Module):
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

    model = UnetResNet152()
    print(model)
    draw_graph(
        model,
        input_size=(1, 3, 224, 224),
        depth=5,
        show_shapes=True,
        expand_nested=True,
        save_graph=True,
        filename="unet+resnet152",
        directory="figures",
    )
