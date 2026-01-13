# src/setiastro/saspro/ser_stack_config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Union, Sequence

from setiastro.saspro.imageops.serloader import PlanetaryFrameSource

TrackMode = Literal["off", "planetary", "surface"]
PlanetarySource = Union[str, Sequence[str], PlanetaryFrameSource]

@dataclass
class SERStackConfig:
    source: PlanetarySource
    roi: Optional[Tuple[int, int, int, int]] = None
    track_mode: TrackMode = "planetary"
    surface_anchor: Optional[Tuple[int, int, int, int]] = None
    keep_percent: float = 20.0

    # AP / alignment
    ap_size: int = 64
    ap_spacing: int = 48
    ap_min_mean: float = 0.03
    ap_multiscale: bool = False
    ssd_refine_bruteforce: bool = False

    # ✅ Drizzle
    drizzle_scale: float = 1.0          # 1.0 = off, 1.5, 2.0
    drizzle_pixfrac: float = 0.80       # "drop shrink" in output pixels (roughly)
    drizzle_kernel: str = "gaussian"    # "square" | "circle" | "gaussian"
    drizzle_sigma: float = 0.0          # only used for gaussian; 0 => auto from pixfrac

    def __init__(self, source: PlanetarySource, **kwargs):
        # Allow deprecated/ignored kwargs without crashing
        kwargs.pop("multipoint", None)  # accept but ignore

        self.source = source
        self.roi = kwargs.pop("roi", None)
        self.track_mode = kwargs.pop("track_mode", "planetary")
        self.surface_anchor = kwargs.pop("surface_anchor", None)
        self.keep_percent = float(kwargs.pop("keep_percent", 20.0))

        self.ap_size = int(kwargs.pop("ap_size", 64))
        self.ap_spacing = int(kwargs.pop("ap_spacing", 48))
        self.ap_min_mean = float(kwargs.pop("ap_min_mean", 0.03))
        self.ap_multiscale = bool(kwargs.pop("ap_multiscale", False))
        self.ssd_refine_bruteforce = bool(kwargs.pop("ssd_refine_bruteforce", False))

        # ✅ NEW: Drizzle params
        self.drizzle_scale = float(kwargs.pop("drizzle_scale", 1.0))
        if self.drizzle_scale not in (1.0, 1.5, 2.0):
            self.drizzle_scale = 1.0

        self.drizzle_pixfrac = float(kwargs.pop("drizzle_pixfrac", 0.80))
        self.drizzle_kernel = str(kwargs.pop("drizzle_kernel", "gaussian")).strip().lower()
        self.drizzle_sigma = float(kwargs.pop("drizzle_sigma", 0.0))

        # sanitize a bit
        if self.drizzle_scale < 1.0:
            self.drizzle_scale = 1.0
        if self.drizzle_pixfrac <= 0.0:
            self.drizzle_pixfrac = 0.01
        if self.drizzle_kernel not in ("square", "circle", "gaussian"):
            self.drizzle_kernel = "gaussian"
        if self.drizzle_sigma < 0.0:
            self.drizzle_sigma = 0.0

        if kwargs:
            raise TypeError(f"Unexpected config keys: {sorted(kwargs.keys())}")
