from __future__ import annotations

# ============================================================================
# SyQon Parallax — Sharpening model architecture
# Placeholder — will be replaced with SyQon's actual architecture when released.
#
# Three models, all part of the Parallax suite:
#   deblur      — non-stellar deblur
#   star_reduce — star reduction
#   star_abcorr — star aberration correction
# ============================================================================

from typing import Literal
import torch
import torch.nn as nn

ParallaxVariant = Literal["deblur", "star_reduce", "star_abcorr"]


class _PlaceholderParallaxNet(nn.Module):
    """
    Passthrough placeholder — returns input unchanged.
    Replace with SyQon's actual Parallax architecture when released.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def create_parallax_model(variant: ParallaxVariant = "deblur") -> nn.Module:
    """
    Factory for SyQon Parallax sharpening models.

    Variants:
        deblur      — non-stellar deblur model
        star_reduce — star reduction model
        star_abcorr — star aberration correction model

    Returns a placeholder passthrough model until SyQon releases the architecture.
    All three variants share the same placeholder until real architectures are known.
    """
    variant = str(variant or "deblur").strip().lower()
    if variant not in ("deblur", "star_reduce", "star_abcorr"):
        raise ValueError(f"Unknown Parallax variant: {variant!r}")
    return _PlaceholderParallaxNet()