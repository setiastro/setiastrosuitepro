# src/setiastro/saspro/syqon_prism_model/model.py
from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            channels * 2,
            channels * 2,
            kernel_size=3,
            padding=1,
            groups=channels * 2,
            bias=True,
        )
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True)
        self.ffn2 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv2(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.ffn1(y)
        y = self.sg(y)
        y = self.ffn2(y)
        x = x + y * self.gamma
        return x


class NAFNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        width: int = 48,
        enc_blk_nums: list[int] | tuple[int, ...] = (2, 4, 6, 8),
        dec_blk_nums: list[int] | tuple[int, ...] = (2, 2, 2, 2),
        middle_blk_num: int = 4,
        use_sigmoid: bool = False,
    ):
        super().__init__()
        self.intro = nn.Conv2d(in_ch, width, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, out_ch, kernel_size=3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        ch = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(num)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, kernel_size=2, stride=2, bias=True))
            ch *= 2

        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch * 2, kernel_size=1, bias=True),
                    nn.PixelShuffle(2),
                )
            )
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(num)]))

        self.out_act = nn.Sigmoid() if use_sigmoid else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.intro(x)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for up, decoder in zip(self.ups, self.decoders):
            x = up(x)
            x = x + skips.pop()
            x = decoder(x)
        x = self.ending(x)
        return self.out_act(x)


def create_model(base_ch: int = 48, depth: int = 4, use_sigmoid: bool = False) -> nn.Module:
    depth = max(1, min(8, int(depth)))
    if depth >= 4:
        base = (2, 4, 6, 8)
        enc = base + tuple([8] * (depth - 4))
        dec = tuple([2] * depth)
        middle = 4
    elif depth == 3:
        enc = (2, 4, 6)
        dec = (2, 2, 2)
        middle = 3
    elif depth == 2:
        enc = (2, 4)
        dec = (2, 2)
        middle = 2
    else:
        enc = (2,)
        dec = (2,)
        middle = 2

    return NAFNet(
        width=int(base_ch),
        enc_blk_nums=enc,
        dec_blk_nums=dec,
        middle_blk_num=middle,
        use_sigmoid=use_sigmoid,
    )