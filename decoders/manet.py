import torch, torch.nn as nn
import torch.nn.functional as F

from .common import Conv2dReLU


class PAB(nn.Module):
    def __init__(
        self, in_channels, out_channels, pab_channels=64, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.top_conv = nn.Conv2d(in_channels, pab_channels, kernel_size=1)
        self.center_conv = nn.Conv2d(in_channels, pab_channels, kernel_size=1)
        self.bottom_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.map_softmax = nn.Softmax(dim=1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        bsize = x.size()[0]
        h = x.size()[2]
        w = x.size()[3]
        x_top = self.top_conv(x)
        x_center = self.center_conv(x)
        x_bottom = self.bottom_conv(x)

        x_top = x_top.flatten(2)
        x_center = x_center.flatten(2).transpose(1, 2)
        x_bottom = x_bottom.flatten(2).transpose(1, 2)

        sp_map = torch.matmul(x_center, x_top)
        sp_map = self.map_softmax(sp_map.view(bsize, -1)).view(bsize, h * w, h * w)
        sp_map = torch.matmul(sp_map, x_bottom)
        sp_map = sp_map.reshape(bsize, self.in_channels, h, w)
        x = x + sp_map
        x = self.out_conv(x)
        return x


class MFAB(nn.Module):
    def __init__(
        self, in_channels, skip_channels, out_channels, reduction=16, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hl_conv = nn.Sequential(
            Conv2dReLU(in_channels, in_channels), Conv2dReLU(in_channels, skip_channels)
        )
        reduced_channels = max(1, skip_channels // reduction)
        self.SE_ll = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(skip_channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, skip_channels, 1),
            nn.Sigmoid(),
        )
        self.SE_hl = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(skip_channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, skip_channels, 1),
            nn.Sigmoid(),
        )
        self.conv1 = Conv2dReLU(skip_channels + skip_channels, out_channels)
        self.conv2 = Conv2dReLU(out_channels, out_channels)

    def forward(self, x, skip=None):
        x = self.hl_conv(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        attention_hl = self.SE_hl(x)
        if skip is not None:
            attention_ll = self.SE_ll(skip)
            attention_hl = attention_hl + attention_ll
            x = x * attention_hl
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels, skip_channels, out_channels, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels, out_channels, kernel_size=3
        )
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MAnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        reduction=16,
        pab_channels=64,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]

        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        self.center = PAB(head_channels, head_channels, pab_channels=pab_channels)
        blocks = [
            (
                MFAB(in_ch, skip_ch, out_ch, reduction=reduction)
                if skip_ch > 0
                else DecoderBlock(in_ch, skip_ch, out_ch)
            )
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[1:]
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
