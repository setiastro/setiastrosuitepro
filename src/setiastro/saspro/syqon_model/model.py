#src/setiastro/saspro/syqon_model/model.py
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
        width: int = 32,
        enc_blk_nums: list[int] | tuple[int, ...] = (2, 2, 4, 8),
        dec_blk_nums: list[int] | tuple[int, ...] = (2, 2, 2, 2),
        middle_blk_num: int = 2,
        use_sigmoid: bool = True,
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

        self.use_sigmoid = use_sigmoid
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


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        g = min(groups, out_ch)
        while out_ch % g != 0:
            g -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=g, num_channels=out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=g, num_channels=out_ch),
            nn.SiLU(inplace=True),
        )
        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        skip = x if self.skip is None else self.skip(x)
        return out + skip


class SelfAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        self.to_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b, self.heads, c // self.heads, h * w)
        k = k.reshape(b, self.heads, c // self.heads, h * w)
        v = v.reshape(b, self.heads, c // self.heads, h * w)
        attn = torch.softmax(torch.einsum("bhcn,bhcm->bhnm", q, k) * self.scale, dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        out = out.reshape(b, c, h, w)
        return self.proj(out) + x


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, groups=groups)
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res(x)
        down = self.down(x)
        return x, down


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )
        self.res = ResBlock(out_ch + skip_ch, out_ch, groups=groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x)


class UNetStar(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        base_ch: int = 64,
        depth: int = 4,
        groups: int = 8,
        use_sigmoid: bool = True,
        use_attention: bool = False,
    ):
        super().__init__()
        self.stem = ResBlock(in_ch, base_ch, groups=groups)
        downs = []
        ch = base_ch
        skip_chs = []
        for _ in range(depth):
            downs.append(DownBlock(ch, ch * 2, groups=groups))
            ch *= 2
            skip_chs.append(ch)
        self.downs = nn.ModuleList(downs)
        attention = SelfAttention(ch, heads=4) if use_attention else nn.Identity()
        self.mid = nn.Sequential(
            ResBlock(ch, ch, groups=groups),
            attention,
            ResBlock(ch, ch, groups=groups),
        )
        ups = []
        for skip_ch in reversed(skip_chs):
            up_ch = max(base_ch, skip_ch // 2)
            ups.append(UpBlock(ch, skip_ch, up_ch, groups=groups))
            ch = up_ch
        self.ups = nn.ModuleList(ups)
        out_layers = [
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, out_ch, kernel_size=1),
        ]
        if use_sigmoid:
            out_layers.append(nn.Sigmoid())
        self.out = nn.Sequential(*out_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        skips = []
        for down in self.downs:
            skip, x = down(x)
            skips.append(skip)
        x = self.mid(x)
        for up in self.ups:
            x = up(x, skips.pop())
        return self.out(x)


def create_model(
    base_ch: int = 48,
    depth: int = 4,
    groups: int = 8,
    use_sigmoid: bool = True,
    use_attention: bool = False,
) -> nn.Module:
    _ = groups
    _ = use_attention
    depth = max(1, min(100, depth))
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
        width=base_ch,
        enc_blk_nums=enc,
        dec_blk_nums=dec,
        middle_blk_num=middle,
        use_sigmoid=False,
    )
