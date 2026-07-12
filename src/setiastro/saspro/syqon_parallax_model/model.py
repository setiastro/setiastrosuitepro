# ============================================================================
# SyQon Parallax — model architectures
#
# Three models, all part of the Parallax suite:
#   correction   — StellarDirectNet  (aberration correction, no level cond)
#   star_reduce  — StellarDirectNet  (star reduction, level conditioning 1-6)
#   sharpen      — AstroNAFLiteDeblur (NAFNet-lite deblur/sharpen)
#
# Architectures are exact reproductions of SyQon's released source:
#   model.py      → StellarDirectNet
#   sharp_lite.py → AstroNAFLiteDeblur / build_nafnet_lite
# ============================================================================

from __future__ import annotations
from typing import Literal

ParallaxVariant = Literal["correction", "star_reduce", "sharpen"]


# ---------------------------------------------------------------------------
# StellarDirectNet  (correction + star_reduce)
# Exact architecture from SyQon model.py
# ---------------------------------------------------------------------------

def _build_StellarDirectNet():
    import torch
    import torch.nn as nn

    class SEBlock(nn.Module):
        def __init__(self, channels: int, reduction: int = 8):
            super().__init__()
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, max(channels // reduction, 4), bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(max(channels // reduction, 4), channels, bias=False),
                nn.Sigmoid(),
            )

        def forward(self, x):
            b, c, _, _ = x.shape
            return x * self.fc(x).view(b, c, 1, 1)

    class ResBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
            self.relu  = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
            self.se    = SEBlock(channels)

        def forward(self, x):
            out = self.relu(self.conv1(x))
            out = self.conv2(out)
            out = self.se(out)
            return out + x

    class StellarDirectNet(nn.Module):
        """
        U-Net with SE-attention ResBlocks and optional CoordConv + level conditioning.

        correction  : cond_level=False  (boolean pass, no level input)
        star_reduce : cond_level=True   (level 1-6, normalised to level/10)
        """
        def __init__(
            self,
            in_channels: int = 3,
            base_channels: int = 32,
            num_res_blocks: int = 2,
            coord_conv: bool = True,
            cond_level: bool = False,
            aggressive_correction: bool = False,
        ):
            super().__init__()
            self.coord_conv   = coord_conv
            self.cond_level   = cond_level
            self.max_residual = 0.5 if aggressive_correction else 0.35

            extra    = (2 if coord_conv else 0) + (1 if cond_level else 0)
            input_ch = in_channels + extra

            self.enc1_conv = nn.Conv2d(input_ch, base_channels, 3, padding=1)
            self.enc1_res  = nn.Sequential(*[ResBlock(base_channels) for _ in range(num_res_blocks)])
            self.down1     = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)

            self.enc2_res  = nn.Sequential(*[ResBlock(base_channels * 2) for _ in range(num_res_blocks)])
            self.down2     = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)

            self.bottleneck = nn.Sequential(*[ResBlock(base_channels * 4) for _ in range(num_res_blocks)])

            self.up2       = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.dec2_conv = nn.Conv2d(base_channels * 4 + base_channels * 2, base_channels * 2, 3, padding=1)
            self.dec2_res  = nn.Sequential(*[ResBlock(base_channels * 2) for _ in range(num_res_blocks)])

            self.up1       = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.dec1_conv = nn.Conv2d(base_channels * 2 + base_channels, base_channels, 3, padding=1)
            self.dec1_res  = nn.Sequential(*[ResBlock(base_channels) for _ in range(num_res_blocks)])

            self.final_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)
            nn.init.zeros_(self.final_conv.weight)
            nn.init.zeros_(self.final_conv.bias)

        def forward(self, x, level=None):
            identity = x
            b, _, h, w = x.shape
            feats = [x]

            if self.coord_conv:
                yg = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
                xg = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
                feats += [yg, xg]

            if self.cond_level and level is not None:
                lv = level.view(b, 1, 1, 1).expand(b, 1, h, w)
                feats.append(lv)

            inp = torch.cat(feats, dim=1)

            e1 = self.enc1_res(self.enc1_conv(inp))
            e2 = self.enc2_res(self.down1(e1))
            bn = self.bottleneck(self.down2(e2))

            d2 = self.dec2_res(self.dec2_conv(torch.cat([self.up2(bn), e2], dim=1)))
            d1 = self.dec1_res(self.dec1_conv(torch.cat([self.up1(d2), e1], dim=1)))

            delta = torch.tanh(self.final_conv(d1)) * self.max_residual
            return torch.clamp(identity + delta, 0.0, 1.0)

    return StellarDirectNet


# ---------------------------------------------------------------------------
# AstroNAFLiteDeblur  (sharpen)
# Exact architecture from SyQon sharp_lite.py
# ---------------------------------------------------------------------------

def _build_AstroNAFLiteDeblur():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LayerNorm2d(nn.Module):
        def __init__(self, n: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, n, 1, 1))
            self.bias   = nn.Parameter(torch.zeros(1, n, 1, 1))
            self.eps    = eps

        def forward(self, x):
            mu  = x.mean(1, keepdim=True)
            var = (x - mu).pow(2).mean(1, keepdim=True)
            return (x - mu) / torch.sqrt(var + self.eps) * self.weight + self.bias

    class SimpleGate(nn.Module):
        def forward(self, x):
            a, b = x.chunk(2, dim=1)
            return a * b

    class SimplifiedChannelAttention(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv2d(channels, channels, 1, bias=True)

        def forward(self, x):
            return x * self.conv(self.pool(x))

    class NAFBlock(nn.Module):
        def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2, dropout_rate: float = 0.0):
            super().__init__()
            dw_c  = channels * dw_expand
            ffn_c = channels * ffn_expand
            self.norm1 = LayerNorm2d(channels)
            self.conv1 = nn.Conv2d(channels, dw_c, 1, bias=True)
            self.conv2 = nn.Conv2d(dw_c, dw_c, 3, padding=1, groups=dw_c, bias=True)
            self.sg    = SimpleGate()
            self.sca   = SimplifiedChannelAttention(dw_c // 2)
            self.conv3 = nn.Conv2d(dw_c // 2, channels, 1, bias=True)
            self.drop1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.norm2 = LayerNorm2d(channels)
            self.conv4 = nn.Conv2d(channels, ffn_c, 1, bias=True)
            self.conv5 = nn.Conv2d(ffn_c // 2, channels, 1, bias=True)
            self.drop2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        def forward(self, x):
            r = x
            x = self.norm1(x); x = self.conv1(x); x = self.conv2(x)
            x = self.sg(x); x = x * self.sca(x); x = self.conv3(x)
            x = self.drop1(x); y = r + x * self.beta
            x = self.norm2(y); x = self.conv4(x); x = self.sg(x)
            x = self.conv5(x); x = self.drop2(x)
            return y + x * self.gamma

    class PixelShuffleUp(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, 1, bias=True), nn.PixelShuffle(2))

        def forward(self, x):
            return self.body(x)

    class BilinearUp(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.proj = nn.Conv2d(channels, channels // 2, 3, padding=1, bias=True)

        def forward(self, x):
            return self.proj(F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False))

    class AstroNAFLiteDeblur(nn.Module):
        def __init__(
            self,
            img_channels: int = 3,
            width: int = 32,
            enc_blocks=None,
            middle_blocks: int = 3,
            dec_blocks=None,
            dropout_rate: float = 0.0,
            upsample_mode: str = "bilinear",
        ):
            super().__init__()
            enc_blocks = list(enc_blocks or [1, 2, 2])
            dec_blocks = list(dec_blocks or [1, 1])

            if len(dec_blocks) < len(enc_blocks):
                dec_blocks = dec_blocks + [1] * (len(enc_blocks) - len(dec_blocks))
            else:
                dec_blocks = dec_blocks[:len(enc_blocks)]

            self.intro  = nn.Conv2d(img_channels, width, 3, padding=1, bias=True)
            self.ending = nn.Conv2d(width, img_channels, 3, padding=1, bias=True)
            self.encoders = nn.ModuleList()
            self.downs    = nn.ModuleList()
            self.ups      = nn.ModuleList()
            self.decoders = nn.ModuleList()

            chan = width
            for n in enc_blocks:
                self.encoders.append(nn.Sequential(*[NAFBlock(chan, dropout_rate=dropout_rate) for _ in range(n)]))
                self.downs.append(nn.Conv2d(chan, chan * 2, 2, stride=2, bias=True))
                chan *= 2

            self.middle = nn.Sequential(*[NAFBlock(chan, dropout_rate=dropout_rate) for _ in range(middle_blocks)])

            for n in dec_blocks:
                up = PixelShuffleUp(chan) if upsample_mode == "pixelshuffle" else BilinearUp(chan)
                self.ups.append(up)
                chan //= 2
                self.decoders.append(nn.Sequential(*[NAFBlock(chan, dropout_rate=dropout_rate) for _ in range(n)]))

            self.padder_size = 2 ** len(enc_blocks)

        def forward(self, x):
            h, w = x.shape[-2:]
            x = self._pad(x)
            res = x
            x = self.intro(x)
            skips = []
            for enc, dn in zip(self.encoders, self.downs):
                x = enc(x); skips.append(x); x = dn(x)
            x = self.middle(x)
            for dec, up, sk in zip(self.decoders, self.ups, reversed(skips)):
                x = up(x); x = x + sk; x = dec(x)
            return (self.ending(x) + res)[..., :h, :w]

        def _pad(self, x):
            _, _, h, w = x.shape
            ph = (self.padder_size - h % self.padder_size) % self.padder_size
            pw = (self.padder_size - w % self.padder_size) % self.padder_size
            return F.pad(x, (0, pw, 0, ph), mode="reflect") if ph or pw else x

    _PRESETS = {
        "lite":          {"width": 24, "enc_blocks": [1, 1, 2], "middle_blocks": 2, "dec_blocks": [1, 1]},
        "balanced":      {"width": 32, "enc_blocks": [1, 2, 2], "middle_blocks": 3, "dec_blocks": [1, 1]},
        "quality_light": {"width": 40, "enc_blocks": [2, 2, 3], "middle_blocks": 4, "dec_blocks": [2, 1]},
    }

    def build_nafnet_lite(preset_name: str = "balanced", upsample_mode: str = "bilinear", img_channels: int = 3):
        cfg = _PRESETS.get(preset_name, _PRESETS["balanced"])
        return AstroNAFLiteDeblur(
            img_channels=img_channels,
            width=cfg["width"],
            enc_blocks=cfg["enc_blocks"],
            middle_blocks=cfg["middle_blocks"],
            dec_blocks=cfg["dec_blocks"],
            upsample_mode=upsample_mode,
        )

    return AstroNAFLiteDeblur, build_nafnet_lite

# ---------------------------------------------------------------------------
# NAFNet  (aesthetics correction + star_reduce + sharpen)
# Standard NAFNet with LayerNorm2d (permute), PixelUnshuffle down,
# PixelShuffle up, and global residual add. Dims inferred from state dict.
# ---------------------------------------------------------------------------

def _build_NAFNet():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LayerNorm2d(nn.Module):
        def __init__(self, n: int, eps: float = 1e-6):
            super().__init__()
            self.norm = nn.LayerNorm(n, eps=eps)

        def forward(self, x):
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            return x.permute(0, 3, 1, 2)

    class SimpleGate(nn.Module):
        def forward(self, x):
            a, b = x.chunk(2, dim=1)
            return a * b

    class SimplifiedChannelAttention(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv2d(channels, channels, 1, bias=True)

        def forward(self, x):
            return x * self.conv(self.pool(x))

    class NAFBlock(nn.Module):
        def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2, dropout_rate: float = 0.0):
            super().__init__()
            dw_c  = channels * dw_expand
            ffn_c = channels * ffn_expand
            self.norm1 = LayerNorm2d(channels)
            self.conv1 = nn.Conv2d(channels, dw_c, 1, bias=True)
            self.conv2 = nn.Conv2d(dw_c, dw_c, 3, padding=1, groups=dw_c, bias=True)
            self.sg    = SimpleGate()
            self.sca   = SimplifiedChannelAttention(dw_c // 2)
            self.conv3 = nn.Conv2d(dw_c // 2, channels, 1, bias=True)
            self.drop1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.norm2 = LayerNorm2d(channels)
            self.conv4 = nn.Conv2d(channels, ffn_c, 1, bias=True)
            self.conv5 = nn.Conv2d(ffn_c // 2, channels, 1, bias=True)
            self.drop2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        def forward(self, x):
            identity = x
            x = self.norm1(x); x = self.conv1(x); x = self.conv2(x)
            x = self.sg(x); x = self.sca(x); x = self.conv3(x)
            x = self.drop1(x)
            y = identity + x * self.beta

            identity = y
            y = self.norm2(y); y = self.conv4(y); y = self.sg(y)
            y = self.conv5(y); y = self.drop2(y)
            return identity + y * self.gamma

    class NAFNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=3, width=32,
                     middle_blk_num=4, enc_blk_nums=(2, 2, 2), dec_blk_nums=(2, 2, 2)):
            super().__init__()
            self.intro  = nn.Conv2d(in_channels,  width, 3, padding=1, bias=True)
            self.ending = nn.Conv2d(width, out_channels, 3, padding=1, bias=True)

            self.encoders    = nn.ModuleList()
            self.decoders    = nn.ModuleList()
            self.upsamples   = nn.ModuleList()
            self.downsamples = nn.ModuleList()

            chan = width
            for num in enc_blk_nums:
                self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
                self.downsamples.append(nn.Sequential(
                    nn.PixelUnshuffle(2),
                    nn.Conv2d(chan * 4, chan * 2, 1, bias=False),
                ))
                chan *= 2

            self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

            for num in dec_blk_nums:
                self.upsamples.append(nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                ))
                chan //= 2
                self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

            self.padder_size = 2 ** len(enc_blk_nums)

        def forward(self, inp):
            _, _, H, W = inp.shape
            ph = (self.padder_size - H % self.padder_size) % self.padder_size
            pw = (self.padder_size - W % self.padder_size) % self.padder_size
            x = F.pad(inp, (0, pw, 0, ph), mode="reflect") if (ph or pw) else inp

            x_intro = self.intro(x)
            skips = []
            e = x_intro
            for enc, dn in zip(self.encoders, self.downsamples):
                e = enc(e); skips.append(e); e = dn(e)
            m = self.middle_blks(e)
            d = m
            for up, dec, sk in zip(self.upsamples, self.decoders, reversed(skips)):
                d = up(d); d = d + sk; d = dec(d)

            out = self.ending(d)
            # global residual — add input's first out_channels back (SyQon pattern)
            out = out + x[:, :out.shape[1], :, :]

            if ph or pw:
                out = out[:, :, :H, :W]
            return out

    return NAFNet


def _infer_nafnet_dims(sd: dict, mc: dict) -> dict:
    """Infer in/out/width from a NAFNet state dict, falling back to config."""
    if "intro.weight" in sd:
        width       = int(sd["intro.weight"].shape[0])
        in_channels = int(sd["intro.weight"].shape[1])
        dims_source = "state_dict"
    else:
        width       = int(mc.get("width", 32))
        in_channels = int(mc.get("img_channels", mc.get("in_channels", 3)))
        dims_source = "config_default"
    if "ending.weight" in sd:
        out_channels = int(sd["ending.weight"].shape[0])
    else:
        out_channels = int(mc.get("out_channels", 3))
    return {
        "in_channels":  in_channels,
        "out_channels": out_channels,
        "width":        width,
        "source":       dims_source,
    }

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _clean_state_dict(sd: dict) -> dict:
    """Strip DataParallel 'module.' and torch.compile '_orig_mod.' prefixes."""
    if not sd:
        return sd
    keys = list(sd.keys())
    if keys and all(k.startswith("module.") for k in keys):
        sd = {k[7:]: v for k, v in sd.items()}
    return {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}


def _extract_state_dict(ckpt) -> dict:
    """Find the actual state dict inside a checkpoint. SyQon uses various keys."""
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ("model", "model_state_dict", "state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


def _load_pth(path: str) -> dict:
    import torch
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"Unrecognised checkpoint format at {path}: {type(obj)}")


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_parallax_model(variant: ParallaxVariant = "correction"):
    """
    Factory for SyQon Parallax models.

    Returns an unloaded (random weights) model instance of the correct
    architecture for the given variant. Call load_parallax_model() to
    load real weights from a checkpoint file.

    Variants:
        correction  — StellarDirectNet, no level conditioning
        star_reduce — StellarDirectNet, level conditioning (1-6 / 10)
        sharpen     — AstroNAFLiteDeblur (preset auto-detected from checkpoint)
    """
    variant = str(variant or "correction").strip().lower()

    if variant in ("correction", "star_reduce"):
        StellarDirectNet = _build_StellarDirectNet()
        return StellarDirectNet(
            in_channels=3,
            base_channels=32,
            num_res_blocks=2,
            coord_conv=True,
            cond_level=(variant == "star_reduce"),
            aggressive_correction=False,
        )

    if variant == "sharpen":
        _, build_nafnet_lite = _build_AstroNAFLiteDeblur()
        return build_nafnet_lite(preset_name="balanced", upsample_mode="bilinear", img_channels=3)

    raise ValueError(f"Unknown Parallax variant: {variant!r}")

def load_parallax_model(ckpt_path: str,
                        variant: ParallaxVariant = "correction",
                        mode: str = "classic"):
    """
    Load a SyQon Parallax model.

    variant: "correction" | "star_reduce" | "sharpen"
    mode:    "classic"    — StellarDirectNet (correction/star_reduce) or AstroNAFLiteDeblur (sharpen)
             "aesthetics" — NAFNet for all three variants; star_reduce takes 4-channel input
                            (RGB + level plane); correction/sharpen take 3-channel.

    Returns (model, config_dict)
    """
    ckpt   = _load_pth(ckpt_path)
    config = ckpt.get("config", {})
    mode   = str(mode or "classic").strip().lower()

    if mode == "aesthetics":
        NAFNet = _build_NAFNet()
        sd = _clean_state_dict(_extract_state_dict(ckpt))
        mc = config.get("model", {})
        dims = _infer_nafnet_dims(sd, mc)

        # Variant/channel sanity check — catches the "installed correction
        # model in the star reduction slot" mistake before it surfaces as
        # an opaque tensor shape error inside the tiling loop.
        expected_in = 4 if variant == "star_reduce" else 3
        if dims["in_channels"] != expected_in and dims["source"] == "state_dict":
            raise RuntimeError(
                f"SyQon Parallax (Aesthetics/{variant}): the installed model at\n"
                f"  {ckpt_path}\n"
                f"has intro.weight with {dims['in_channels']} input channels, but the "
                f"'{variant}' stage requires a model with {expected_in}. "
                f"This usually means the wrong file was chosen when installing "
                f"the aesthetics models — the Star Reduction aesthetics model is "
                f"the 4-channel file (RGB + level plane), while Correction and "
                f"Sharpen are 3-channel. Re-install the correct file from the "
                f"SyQon Tools hub."
            )

        model = NAFNet(
            in_channels    = dims["in_channels"],
            out_channels   = dims["out_channels"],
            width          = dims["width"],
            middle_blk_num = mc.get("middle_blk_num", 4),
            enc_blk_nums   = tuple(mc.get("enc_blk_nums", [2, 2, 2])),
            dec_blk_nums   = tuple(mc.get("dec_blk_nums", [2, 2, 2])),
        )
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[Parallax/aesthetics] state_dict mismatch — missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print(f"  first missing: {list(missing)[:5]}")
            if unexpected:
                print(f"  first unexpected: {list(unexpected)[:5]}")
        model.eval()
        return model, config

    # classic (unchanged from before)
    if variant in ("correction", "star_reduce"):
        StellarDirectNet = _build_StellarDirectNet()
        mc = config.get("model", {})
        tc = config.get("training", {})
        model = StellarDirectNet(
            in_channels=mc.get("in_channels", 3),
            base_channels=mc.get("base_channels", 32),
            num_res_blocks=mc.get("num_res_blocks", 2),
            coord_conv=mc.get("coord_conv", True),
            cond_level=(variant == "star_reduce"),
            aggressive_correction=tc.get("aggressive_correction", False),
        )
        sd = _clean_state_dict(_extract_state_dict(ckpt))
        model.load_state_dict(sd, strict=True)

    elif variant == "sharpen":
        _, build_nafnet_lite = _build_AstroNAFLiteDeblur()
        args          = ckpt.get("args", {})
        preset        = args.get("preset", "balanced")
        upsample_mode = args.get("upsample_mode", "bilinear")
        model = build_nafnet_lite(preset_name=preset, upsample_mode=upsample_mode, img_channels=3)
        sd = _clean_state_dict(_extract_state_dict(ckpt))
        model.load_state_dict(sd, strict=True)

    else:
        raise ValueError(f"Unknown Parallax variant: {variant!r}")

    model.eval()
    return model, config