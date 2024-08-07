import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding


class UnetVgg16(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ## Encoder
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.encoder_block2 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.encoder_block3 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.encoder_block4 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.encoder_block5 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        # torchvision Avg Pool
        self.avgpool_block = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
            nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        )

        ## Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    512,
                    512,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(
                    512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
                ),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(
                    512,
                    512,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(
                    512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
                ),
                nn.ReLU(inplace=True),
            ),
        )

        ## Decoder
        self.decoder_block5 = nn.Sequential(
            nn.Conv2d(
                1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
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
                768, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
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
                384, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
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
                192, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
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
        c2 = self.encoder_block2(c1)
        c3 = self.encoder_block3(c2)
        c4 = self.encoder_block4(c3)
        c5 = self.encoder_block5(c4)
        ap = self.avgpool_block(c5)
        d5 = self.bottleneck(ap)
        d5 = F.interpolate(d5, scale_factor=2, mode="nearest")
        c5 = torch.cat([c5, d5], dim=1)
        d5 = self.decoder_block5(c5)
        d4 = F.interpolate(d5, scale_factor=2, mode="nearest")
        c4 = torch.cat([c4, d4], dim=1)
        d4 = self.decoder_block4(c4)
        d3 = F.interpolate(d4, scale_factor=2, mode="nearest")
        c3 = torch.cat([c3, d3], dim=1)
        d3 = self.decoder_block3(c3)
        d2 = F.interpolate(d3, scale_factor=2, mode="nearest")
        c2 = torch.cat([c2, d2], dim=1)
        d2 = self.decoder_block2(c2)
        d1 = F.interpolate(d2, scale_factor=2, mode="nearest")
        d1 = self.decoder_block1(d1)
        outputs = self.segmentation_block(d1)

        return outputs


if __name__ == "__main__":
    from torchview import draw_graph

    model = UnetVgg16()
    print(model)
    draw_graph(
        model,
        input_size=(1, 3, 224, 224),
        depth=5,
        show_shapes=True,
        expand_nested=True,
        save_graph=True,
        filename="unet+vgg16",
        directory="figures",
    )
