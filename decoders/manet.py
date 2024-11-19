import torch, torch.nn as nn
import torch.nn.functional as F

from .common import Conv2dReLU


class PAB(nn.Module):
    def __init__(
        self, in_channels, out_channels, pab_channels=64, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.conv_top = nn.Conv2d(in_channels, pab_channels, kernel_size=1)
        self.conv_center = nn.Conv2d(in_channels, pab_channels, kernel_size=1)
        self.conv_bottom = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.map_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        print("PAB forward")
        print("x: ", x.shape, x.size())
        bsize = x.size()[0]
        h = x.size()[2]
        w = x.size()[3]

        # Apply convolutions
        x_top = self.conv_top(x)
        print("x top: ", x_top.shape)
        x_center = self.conv_center(x)
        print("x center: ", x_center.shape)
        x_bottom = self.conv_bottom(x)
        print("x bottom: ", x_bottom.shape)

        # Flatten
        x_top = torch.flatten(x_top, 2)
        print("x top flatten: ", x_top.shape)
        x_center = torch.flatten(x_center, 2)
        print("x center flatten: ", x_center.shape)
        x_bottom = torch.flatten(x_bottom, 2)
        print("x bottom flatten: ", x_bottom.shape)

        # Transpose
        x_center = torch.transpose(x_center, 1, 2)
        print("x center transpose: ", x_center.shape)
        x_bottom = torch.transpose(x_bottom, 1, 2)
        print("x bottom transpose: ", x_bottom.shape)

        # Matmul & view & softmax
        sp_map = torch.matmul(x_center, x_top)
        print("sp_map matmul: ", sp_map.shape)
        sp_map = sp_map.view(bsize, -1)
        print("sp_map view: ", sp_map.shape)
        sp_map = self.map_softmax(sp_map)
        print("sp_map softmax: ", sp_map.shape)
        sp_map = sp_map.view(bsize, h * w, h * w)
        print("sp_map view: ", sp_map.shape)
        sp_map = torch.matmul(sp_map, x_bottom)
        print("sp_map matmul: ", sp_map.shape)
        sp_map = torch.reshape(sp_map, (bsize, self.in_channels, h, w))
        print("sp_map reshape: ", sp_map.shape)
        x = torch.add(x, sp_map)
        print("x add: ", x.shape)
        x = self.conv_out(x)
        print("x out: ", x.shape)
        return x


class MFAB(nn.Module):
    def __init__(
        self, in_channels, mid_channels, out_channels, reduced_channels, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.block_1 = nn.Sequential(
            Conv2dReLU(in_channels, in_channels), Conv2dReLU(in_channels, mid_channels)
        )
        self.block_2l = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(mid_channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, mid_channels, 1),
            nn.Sigmoid(),
        )
        self.block_2r = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(mid_channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, mid_channels, 1),
            nn.Sigmoid(),
        )
        self.conv1 = Conv2dReLU(2 * mid_channels, out_channels)
        self.conv2 = Conv2dReLU(out_channels, out_channels)

    def forward(self, x, skip=None):
        print("MFAB forward")
        print("x: ", x.shape)
        x_1 = self.block_1(x)
        print("x_1 block_1: ", x_1.shape)
        x_1 = F.interpolate(x_1, scale_factor=2, mode="nearest")
        print("x_1 interpolate: ", x_1.shape)
        x_2l = self.block_2l(x_1)
        print("x_2l: ", x_2l.shape)
        if skip is not None:
            print("skip: ", skip.shape)
            x_2r = self.block_2r(skip)
            print("x_2r: ", x_2r.shape)
            x_2l = torch.add(x_2l, x_2r)
            print("x_2l+x_2r: ", x_2l.shape)
        print("before matmul: ", x_1.shape, x_2l.shape)
        x = x_1 * x_2l
        print("x matmul: ", x.shape)
        x = torch.cat([x, skip], dim=1)
        print("x cat: ", x.shape)
        x = self.conv1(x)
        print("x conv1: ", x.shape)
        x = self.conv2(x)
        print("x conv2: ", x.shape)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = Conv2dReLU(in_channels, out_channels, kernel_size=3)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MAnetDecoder(nn.Module):
    def __init__(self, inputs, mid_channels, outputs, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Position Attention Block
        self.pab_block = PAB(inputs[0], outputs[0])

        self.mfab_block1 = MFAB(inputs[1], mid_channels[0], outputs[1], 32)
        self.mfab_block2 = MFAB(inputs[2], mid_channels[1], outputs[2], 32)
        self.mfab_block3 = MFAB(inputs[3], mid_channels[2], outputs[3], 16)
        self.mfab_block4 = MFAB(inputs[4], mid_channels[3], outputs[4], 8)

        self.decoder_block = DecoderBlock(inputs[5], outputs[5])

    def forward(self, x, skips):
        c1, c2, c3, c4 = skips

        x = self.pab_block(x)
        x = self.mfab_block1(x, c4)
        x = self.mfab_block2(x, c3)
        x = self.mfab_block3(x, c2)
        x = self.mfab_block4(x, c1)
        x = self.decoder_block(x)

        return x
