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
            ups_mode: str = "bilinear",
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
                    self.ups.append(
                        nn.Sequential(
                            nn.Conv2d(ch, ch * 2, kernel_size=1, bias=True),
                            nn.PixelShuffle(2),
                        )
                    )
                else:
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
# StarNetGenerator (AxiomV2.2) — lazy builder
# =============================================================================

def _build_StarNetGenerator():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _SafeBatchNorm2d(nn.BatchNorm2d):
        def forward(self, x):
            if self.training and x.numel() // x.shape[1] == 1:
                return F.batch_norm(
                    x, self.running_mean, self.running_var,
                    self.weight, self.bias, False, self.momentum, self.eps,
                )
            return super().forward(x)

    class _ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch, use_bn=True):
            super().__init__()
            layers = [
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            ]
            if use_bn:
                layers.append(_SafeBatchNorm2d(out_ch, eps=1e-5, momentum=0.1))
            self.block = nn.Sequential(*layers)

        def forward(self, x):
            return self.block(x)

    class _DeconvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.block = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                _SafeBatchNorm2d(out_ch, eps=1e-5, momentum=0.1),
            )

        def forward(self, x):
            return self.block(x)

    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
            nn.init.zeros_(module.bias)

    class StarNetGenerator(nn.Module):
        """
        AxiomV2.2 StarNet-style generator.

        Three-branch output head:
          - removal branch  → relu(raw_removal)
          - inpaint branch  → tanh(raw_inpainted)
          - alpha gate      → sigmoid(raw_alpha) → avg_pool smoothed

        Final output:
          output = (1 - alpha) * (input - removal) + alpha * inpainted
        """
        def __init__(self, in_channels=3, repair_scale=0.35, alpha_smooth_kernel=5):
            super().__init__()
            self.in_channels         = in_channels
            self.repair_scale        = repair_scale        # kept for checkpoint compat
            self.alpha_smooth_kernel = int(alpha_smooth_kernel)

            # Encoder
            self.g_conv0 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
            self.g_conv1 = _ConvBlock(64,  128)
            self.g_conv2 = _ConvBlock(128, 256)
            self.g_conv3 = _ConvBlock(256, 512)
            self.g_conv4 = _ConvBlock(512, 512)
            self.g_conv5 = _ConvBlock(512, 512)
            self.g_conv6 = _ConvBlock(512, 512)
            self.g_conv7 = _ConvBlock(512, 512)

            # Decoder
            self.g_deconv0 = _DeconvBlock(512,  512)
            self.g_deconv1 = _DeconvBlock(1024, 512)
            self.g_deconv2 = _DeconvBlock(1024, 512)
            self.g_deconv3 = _DeconvBlock(1024, 512)
            self.g_deconv4 = _DeconvBlock(1024, 256)
            self.g_deconv5 = _DeconvBlock(512,  128)
            self.g_deconv6 = _DeconvBlock(256,  64)

            # V2.2 output head
            self.g_deconv7_feat = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
            # outputs: removal(in_ch) + inpainted(in_ch) + alpha(1)
            self.g_out = nn.Conv2d(64, in_channels * 2 + 1, kernel_size=3, stride=1, padding=1)

            # Weight init
            self.apply(_init_weights)
            # Bias alpha gate toward "no inpainting" at init
            nn.init.constant_(self.g_out.bias[in_channels * 2:], -3.0)

        def forward(self, x, return_aux=False):
            c0 = self.g_conv0(x)
            c1 = self.g_conv1(c0)
            c2 = self.g_conv2(c1)
            c3 = self.g_conv3(c2)
            c4 = self.g_conv4(c3)
            c5 = self.g_conv5(c4)
            c6 = self.g_conv6(c5)
            c7 = self.g_conv7(c6)

            d0 = self.g_deconv0(c7)
            d1 = self.g_deconv1(torch.cat([d0, c6], dim=1))
            d2 = self.g_deconv2(torch.cat([d1, c5], dim=1))
            d3 = self.g_deconv3(torch.cat([d2, c4], dim=1))
            d4 = self.g_deconv4(torch.cat([d3, c3], dim=1))
            d5 = self.g_deconv5(torch.cat([d4, c2], dim=1))
            d6 = self.g_deconv6(torch.cat([d5, c1], dim=1))

            features  = self.g_deconv7_feat(torch.cat([d6, c0], dim=1))
            out_preds = self.g_out(features)

            raw_removal   = out_preds[:, 0:self.in_channels, :, :]
            raw_inpainted = out_preds[:, self.in_channels:self.in_channels * 2, :, :]
            raw_alpha     = out_preds[:, self.in_channels * 2:, :, :]

            removal          = F.relu(raw_removal)
            residual_output  = x - removal
            inpainted_output = torch.tanh(raw_inpainted)
            alpha            = torch.sigmoid(raw_alpha)

            # ── Smooth alpha to feather residual/inpaint transition edges ──
            k = self.alpha_smooth_kernel
            if k % 2 == 0:
                k += 1
            alpha = F.avg_pool2d(alpha, kernel_size=k, stride=1, padding=k // 2)
            alpha = torch.clamp(alpha, 0.0, 1.0)
            # ──────────────────────────────────────────────────────────────

            output = (1.0 - alpha) * residual_output + alpha * inpainted_output

            if return_aux:
                return output, {
                    "alpha":     alpha,
                    "residual":  residual_output,
                    "inpainted": inpainted_output,
                    "removal":   removal,
                }
            return output

    return StarNetGenerator

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
        def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2):
            super().__init__()
            dw_ch  = channels * dw_expand
            ffn_ch = channels * ffn_expand

            self.norm1 = LayerNorm2d_Flat(channels)
            self.conv1 = nn.Conv2d(channels, dw_ch, 1)
            self.conv2 = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch)
            self.gate1 = SimpleGate()
            self.sca   = SimplifiedChannelAttention(dw_ch // 2)
            self.conv3 = nn.Conv2d(dw_ch // 2, channels, 1)

            self.norm2 = LayerNorm2d_Flat(channels)
            self.conv4 = nn.Conv2d(channels, ffn_ch, 1)
            self.gate2 = SimpleGate()
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
                self.ups.append(nn.Sequential(
                    nn.Conv2d(ch, ch * 2, 3, padding=1),
                    nn.PixelShuffle(2),
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
    return _cfg_nadir(depth)


def infer_variant_from_state_dict(sd: dict) -> str:
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
    _ = groups
    _ = use_attention
    _ = use_sigmoid

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
        use_sigmoid=False,
    )


# =============================================================================
# Public lazy constructors
# =============================================================================

def build_NAFNet(**kwargs):
    return _build_NAFNet()(**kwargs)

def build_NAFNetLite(**kwargs):
    return _build_NAFNetLite()(**kwargs)

def build_UNetStar(**kwargs):
    return _build_UNetStar()(**kwargs)

def build_StarNetGenerator(**kwargs):
    return _build_StarNetGenerator()(**kwargs)


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

class StarNetGenerator:
    """Lazy proxy — instantiating this builds and returns the real StarNetGenerator."""
    def __new__(cls, **kwargs):
        return _build_StarNetGenerator()(**kwargs)