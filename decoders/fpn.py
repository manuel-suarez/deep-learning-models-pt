import torch, torch.nn as nn


class Conv3x3GNReLU(nn.Module):
    def __init__(self, kernels_in, kernels_out, has_interpolate=True) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                kernels_in,
                kernels_out,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.GroupNorm(32, kernels_out, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
        )
        self.has_interpolate = has_interpolate

    def forward(self, x):
        x = self.block(x)
        if self.has_interpolate:
            x = nn.functional.interpolate(x, scale_factor=2)
        return x


class SegmentationBlock(nn.Module):
    def __init__(
        self, num_blocks, kernels_in, kernels_out, has_interpolate=True, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        blocks = [
            Conv3x3GNReLU(
                kernels_in if i == 0 else kernels_out, kernels_out, has_interpolate
            )
            for i in range(num_blocks)
        ]
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.block(x)
        return x


class FPNBlock(nn.Module):
    def __init__(self, kernels_in, kernels_out, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv2d = nn.Conv2d(
            kernels_in,
            kernels_out,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

    def forward(self, x, skips):
        x = nn.functional.interpolate(x, scale_factor=2)
        x1 = self.conv2d(skips)
        x = torch.add(x, x1)
        return x


class MergeBlock(nn.Module):
    def __init__(self, policy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, x):
        if self.policy == "add":
            return sum(x)
        elif self.policy == "cat":
            return torch.cat(x, dim=1)
        else:
            raise ValueError(self.policy)


class FPNDecoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv2d = nn.Conv2d(
            512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.decoder_block1 = SegmentationBlock(3, 256, 128)
        self.decoder_block2 = SegmentationBlock(2, 256, 128)
        self.decoder_block3 = SegmentationBlock(1, 256, 128)
        self.decoder_block4 = SegmentationBlock(1, 256, 128, has_interpolate=False)
        self.fpn_block1 = FPNBlock(512, 256)
        self.fpn_block2 = FPNBlock(512, 256)
        self.fpn_block3 = FPNBlock(256, 256)
        self.merge = MergeBlock(policy="add")
        self.dropout = nn.Dropout2d(p=0.2, inplace=True)

    def forward(self, x, skips):
        c1, c2, c3 = skips
        x = self.conv2d(x)
        d1 = self.decoder_block1(x)
        f1 = self.fpn_block1(x, c3)
        d2 = self.decoder_block2(f1)
        f2 = self.fpn_block2(f1, c2)
        d3 = self.decoder_block3(f2)
        f3 = self.fpn_block3(f2, c1)
        d4 = self.decoder_block4(f3)
        x = self.merge([d1, d2, d3, d4])
        x = self.dropout(x)
        return x
