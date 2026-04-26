# src/setiastro/saspro/syqon_model/model.py
from __future__ import annotations

from typing import Tuple, Union, Optional


# =============================================================================
# Core building blocks (shared)  — lazy builders
# =============================================================================

def _build_LayerNorm2d():
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

    return LayerNorm2d


def _build_SimpleGate():
    import torch
    import torch.nn as nn

    class SimpleGate(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x.chunk(2, dim=1)
            return x1 * x2

    return SimpleGate


def _build_NAFBlock():
    import torch
    import torch.nn as nn
    LayerNorm2d = _build_LayerNorm2d()
    SimpleGate = _build_SimpleGate()

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

    return NAFBlock


def _build_NAFNet():
    import torch
    import torch.nn as nn
    NAFBlock = _build_NAFBlock()

    class NAFNet(nn.Module):
        def __init__(
            self,
            in_ch: int = 3,
            out_ch: int = 3,
            width: int = 32,
            enc_blk_nums: Union[Tuple[int, ...], list] = (2, 2, 4, 8),
            dec_blk_nums: Union[Tuple[int, ...], list] = (2, 2, 2, 2),
            middle_blk_num: int = 2,
            use_sigmoid: bool = True,
            ups_mode: str = "bilinear",  # "bilinear" or "pixelshuffle"
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
                if ups_mode == "pixelshuffle":
                    # ups.N.0.weight — Nadir older builds
                    self.ups.append(
                        nn.Sequential(
                            nn.Conv2d(ch, ch * 2, kernel_size=1, bias=True),
                            nn.PixelShuffle(2),
                        )
                    )
                else:
                    # ups.N.1.weight — SyQon Nadir/AxiomV2 current
                    self.ups.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                            nn.Conv2d(ch, ch // 2, kernel_size=3, padding=1, bias=True),
                        )
                    )
                ch //= 2
                self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(num)]))

            self.use_sigmoid = bool(use_sigmoid)
            self.out_act = nn.Sigmoid() if self.use_sigmoid else nn.Identity()

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

    return NAFNet


# =============================================================================
# Legacy / extra models — all kept for compatibility, all lazified
# =============================================================================

def _build_ResBlock():
    import torch.nn as nn

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

        def forward(self, x):
            out = self.block(x)
            skip = x if self.skip is None else self.skip(x)
            return out + skip

    return ResBlock


def _build_SelfAttention():
    import torch
    import torch.nn as nn

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

    return SelfAttention


def _build_DownBlock():
    import torch.nn as nn
    ResBlock = _build_ResBlock()

    class DownBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
            super().__init__()
            self.res = ResBlock(in_ch, out_ch, groups=groups)
            self.down = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)

        def forward(self, x):
            x = self.res(x)
            down = self.down(x)
            return x, down

    return DownBlock


def _build_UpBlock():
    import torch
    import torch.nn as nn
    ResBlock = _build_ResBlock()

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

    return UpBlock


def _build_UNetStar():
    import torch.nn as nn
    ResBlock = _build_ResBlock()
    SelfAttention = _build_SelfAttention()
    DownBlock = _build_DownBlock()
    UpBlock = _build_UpBlock()

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

        def forward(self, x):
            x = self.stem(x)
            skips = []
            for down in self.downs:
                skip, x = down(x)
                skips.append(skip)
            x = self.mid(x)
            for up in self.ups:
                x = up(x, skips.pop())
            return self.out(x)

    return UNetStar


# =============================================================================
# NAFNetLite (Axiom 2.1) — lazy build
# =============================================================================

def _build_NAFNetLite():
    import torch
    import torch.nn as nn

    class LayerNorm2d_Flat(nn.Module):
        """1D weight/bias LayerNorm — used by NAFNetLite (Axiom 2.1)."""
        def __init__(self, num_channels: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias   = nn.Parameter(torch.zeros(num_channels))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            mu    = x.mean(1, keepdim=True)
            sigma = x.var(1, keepdim=True, unbiased=False)
            x = (x - mu) / torch.sqrt(sigma + self.eps)
            return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

    class SimplifiedChannelAttention(nn.Module):
        def __init__(self, num_channels: int):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv2d(num_channels, num_channels, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.conv(self.pool(x))

    class SimpleGate(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x.chunk(2, dim=1)
            return x1 * x2

    class NAFBlockLite(nn.Module):
        """NAFBlock as used in Axiom 2.1 NAFNetLite."""
        def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2):
            super().__init__()
            dw_ch  = channels * dw_expand
            ffn_ch = channels * ffn_expand

            # Spatial mixing
            self.norm1 = LayerNorm2d_Flat(channels)
            self.conv1 = nn.Conv2d(channels, dw_ch, 1)
            self.conv2 = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch)  # depthwise
            self.gate1 = SimpleGate()                                           # dw_ch -> dw_ch//2
            self.sca   = SimplifiedChannelAttention(dw_ch // 2)
            self.conv3 = nn.Conv2d(dw_ch // 2, channels, 1)

            # Channel mixing (FFN)
            self.norm2 = LayerNorm2d_Flat(channels)
            self.conv4 = nn.Conv2d(channels, ffn_ch, 1)
            self.gate2 = SimpleGate()                                           # ffn_ch -> ffn_ch//2
            self.conv5 = nn.Conv2d(ffn_ch // 2, channels, 1)

            self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.norm1(x)
            y = self.conv1(y)
            y = self.conv2(y)
            y = self.gate1(y)
            y = self.sca(y)
            y = self.conv3(y)
            x = x + y * self.beta

            y = self.norm2(x)
            y = self.conv4(y)
            y = self.gate2(y)
            y = self.conv5(y)
            x = x + y * self.gamma
            return x

    class NAFNetLite(nn.Module):
        def __init__(
            self,
            in_channels:    int   = 3,
            out_channels:   int   = 3,
            width:          int   = 32,
            enc_blk_nums:   tuple = (2, 2, 4),
            middle_blk_num: int   = 4,
            dec_blk_nums:   tuple = (2, 2, 2),
        ):
            super().__init__()
            assert len(enc_blk_nums) == len(dec_blk_nums)

            self.intro   = nn.Conv2d(in_channels, width, 3, padding=1)
            self.ending  = nn.Conv2d(width, out_channels, 3, padding=1)

            self.encoders = nn.ModuleList()
            self.downs    = nn.ModuleList()
            self.decoders = nn.ModuleList()
            self.ups      = nn.ModuleList()
            self.fusions  = nn.ModuleList()

            ch = width
            for num in enc_blk_nums:
                self.encoders.append(nn.Sequential(*[NAFBlockLite(ch) for _ in range(num)]))
                self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2))
                ch *= 2

            self.middle = nn.Sequential(*[NAFBlockLite(ch) for _ in range(middle_blk_num)])

            for num in reversed(dec_blk_nums):
                # 3x3 conv before PixelShuffle — matches Axiom 2.1 checkpoint
                # ups.N.0.weight shape: (ch*2, ch, 3, 3)
                self.ups.append(nn.Sequential(
                    nn.Conv2d(ch, ch * 2, 3, padding=1),
                    nn.PixelShuffle(2),   # ch*2/4 = ch//2
                ))
                ch //= 2
                self.fusions.append(nn.Conv2d(ch * 2, ch, 1))
                self.decoders.append(nn.Sequential(*[NAFBlockLite(ch) for _ in range(num)]))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            inp = x
            x = self.intro(x)

            skips = []
            for encoder, down in zip(self.encoders, self.downs):
                x = encoder(x)
                skips.append(x)
                x = down(x)

            x = self.middle(x)

            for decoder, up, fusion, skip in zip(
                self.decoders, self.ups, self.fusions, reversed(skips)
            ):
                x = up(x)
                x = fusion(torch.cat([x, skip], dim=1))
                x = decoder(x)

            x = self.ending(x)
            # Global residual: model predicts correction, output is clamped starless
            return torch.clamp(inp + x, 0.0, 1.0)

    return NAFNetLite


# =============================================================================
# Model config helpers (pure Python, no torch)
# =============================================================================

def _cfg_nadir(depth: int) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
    depth = max(1, min(100, int(depth)))
    if depth >= 4:
        base = (2, 4, 6, 8)
        enc = base + tuple([8] * (depth - 4))
        dec = tuple([2] * depth)
        middle = 4
    elif depth == 3:
        enc = (2, 4, 6); dec = (2, 2, 2); middle = 3
    elif depth == 2:
        enc = (2, 4); dec = (2, 2); middle = 2
    else:
        enc = (2,); dec = (2,); middle = 2
    return enc, dec, middle


def _cfg_axiomv2(depth: int) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
    """
    Matches the SyQon axiomv2 snippet — kept separate from Nadir in case SyQon
    changes defaults later.
    """
    return _cfg_nadir(depth)


def infer_variant_from_state_dict(sd: dict) -> str:
    """Best-effort variant inference from a state dict. Returns 'nadir' by default."""
    _ = sd
    return "nadir"


# =============================================================================
# Public factory
# =============================================================================

def create_model(
    base_ch: int = 48,
    depth: int = 4,
    groups: int = 8,
    use_sigmoid: bool = True,
    use_attention: bool = False,
    *,
    variant: str = "nadir",
):
    """
    Unified factory for SyQon Nadir / AxiomV2 models.

    Parameters used for NAFNet inference:
      - base_ch  (width)
      - depth    (number of encoder/decoder stages; maps to enc/dec block counts)
      - variant  'nadir' or 'axiomv2'

    Other args (groups/use_attention/use_sigmoid) are retained for compatibility
    with legacy callers. For NAFNet inference we force use_sigmoid=False because
    SyQon starless models are trained in linear space (no clamping).

    Torch is imported lazily — safe to call at module load time.
    """
    _ = groups
    _ = use_attention
    _ = use_sigmoid  # ignored for NAFNet outputs; engine clips/sanitizes later

    v = (variant or "nadir").lower().strip()
    if v in ("axiomv2", "axiom", "ax2"):
        enc, dec, middle = _cfg_axiomv2(depth)
    else:
        enc, dec, middle = _cfg_nadir(depth)

    NAFNet = _build_NAFNet()
    return NAFNet(
        width=int(base_ch),
        enc_blk_nums=enc,
        dec_blk_nums=dec,
        middle_blk_num=int(middle),
        use_sigmoid=False,  # IMPORTANT: keep linear output
    )


# =============================================================================
# Public lazy constructors for callers that instantiate classes directly
# =============================================================================

def build_NAFNet(**kwargs):
    return _build_NAFNet()(**kwargs)

def build_NAFNetLite(**kwargs):
    return _build_NAFNetLite()(**kwargs)

def build_UNetStar(**kwargs):
    return _build_UNetStar()(**kwargs)


class NAFNet:
    """Lazy proxy — instantiating this builds and returns the real NAFNet."""
    def __new__(cls, **kwargs):
        return _build_NAFNet()(**kwargs)

class NAFNetLite:
    """Lazy proxy — instantiating this builds and returns the real NAFNetLite."""
    def __new__(cls, **kwargs):
        return _build_NAFNetLite()(**kwargs)

class NAFBlock:
    """Lazy proxy."""
    def __new__(cls, **kwargs):
        return _build_NAFBlock()(**kwargs)

class LayerNorm2d:
    """Lazy proxy."""
    def __new__(cls, **kwargs):
        return _build_LayerNorm2d()(**kwargs)