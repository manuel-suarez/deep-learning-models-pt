import torch.nn as nn


class Vgg13Encoder(nn.Module):
    def __init__(self, in_channels=3) -> None:
        super().__init__()
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ),
            nn.BatchNorm2d(
                64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(
                64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.encoder_block2 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(
                128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(
                128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.encoder_block3 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(
                256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(
                256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.encoder_block4 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(
                512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(
                512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.encoder_block5 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(
                512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(
                512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.encoder_block6 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            )
        )

    def forward(self, x):
        x = self.encoder_block1(x)
        c1 = self.encoder_block2(x)
        c2 = self.encoder_block3(c1)
        c3 = self.encoder_block4(c2)
        c4 = self.encoder_block5(c3)
        c5 = self.encoder_block6(c4)
        return c1, c2, c3, c4, c5
