import torch.nn as nn
from .common import BasicBlock


class ResNetEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
                BasicBlock(64, 64, has_downsample=False, stride=1),
                BasicBlock(64, 64, has_downsample=False, stride=1),
                BasicBlock(64, 64, has_downsample=False, stride=1),
            ),
        )
        self.encoder_block3 = nn.Sequential(
            BasicBlock(64, 128, has_downsample=True, stride=2),
            BasicBlock(128, 128, has_downsample=False, stride=1),
            BasicBlock(128, 128, has_downsample=False, stride=1),
            BasicBlock(128, 128, has_downsample=False, stride=1),
        )
        self.encoder_block4 = nn.Sequential(
            BasicBlock(128, 256, has_downsample=True, stride=2),
            BasicBlock(256, 256, has_downsample=False, stride=1),
            BasicBlock(256, 256, has_downsample=False, stride=1),
            BasicBlock(256, 256, has_downsample=False, stride=1),
            BasicBlock(256, 256, has_downsample=False, stride=1),
            BasicBlock(256, 256, has_downsample=False, stride=1),
        )
        self.encoder_block5 = nn.Sequential(
            BasicBlock(256, 512, has_downsample=True, stride=2),
            BasicBlock(512, 512, has_downsample=False, stride=1),
            BasicBlock(512, 512, has_downsample=False, stride=1),
        )

    def forward(self, x):
        c1 = self.encoder_block1(x)
        c2 = self.encoder_block2(c1)
        c3 = self.encoder_block3(c2)
        c4 = self.encoder_block4(c3)
        c5 = self.encoder_block5(c4)
        return c1, c2, c3, c4, c5
