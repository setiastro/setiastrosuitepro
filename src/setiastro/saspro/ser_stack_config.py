# src/setiastro/saspro/ser_stack_config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Union, Sequence, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import numpy as np
    KeepMask = "np.ndarray"
else:
    KeepMask = object

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
    bayer_pattern: Optional[str] = None

    # AP / alignment
    ap_size: int = 64
    ap_spacing: int = 48
    ap_min_mean: float = 0.03
    ap_multiscale: bool = False
    ssd_refine_bruteforce: bool = False
    keep_mask: Optional[KeepMask] = None
    planet_smooth_sigma: float = 1.5
    planet_thresh_pct: float = 92.0
    planet_use_norm: bool = True
    planet_norm_lo_pct: float = 1.0
    planet_norm_hi_pct: float = 99.5
    planet_min_val: float = 0.02
    # ✅ Drizzle
    drizzle_scale: float = 1.0          # 1.0 = off, 1.5, 2.0
    drizzle_pixfrac: float = 0.80       # "drop shrink" in output pixels (roughly)
    drizzle_kernel: str = "gaussian"    # "square" | "circle" | "gaussian"
    drizzle_sigma: float = 0.0          # only used for gaussian; 0 => auto from pixfrac
    derotate_enabled: bool = False
    derotate_deg_per_min: float = 0.0   # user controlled
    derotate_center: tuple[float, float] | None = None  # ROI-space center
    derotate_radius: float | None = None
    derotate_overlay_mode: str = "none"  # "none"|"saturn"|"galaxy"
    derotate_rings: tuple[float, float, float, float] | None = None  # (pa, tilt, kout, kin)

    def __init__(self, source: PlanetarySource, **kwargs):
        # Allow deprecated/ignored kwargs without crashing
        kwargs.pop("multipoint", None)  # accept but ignore

        self.source = source
        self.roi = kwargs.pop("roi", None)
        self.track_mode = kwargs.pop("track_mode", "planetary")
        self.surface_anchor = kwargs.pop("surface_anchor", None)
        self.keep_percent = float(kwargs.pop("keep_percent", 20.0))
        self.bayer_pattern = kwargs.pop("bayer_pattern", None)
        if isinstance(self.bayer_pattern, str):
            s = self.bayer_pattern.strip().upper()
            self.bayer_pattern = s if s in ("RGGB", "BGGR", "GRBG", "GBRG") else None
        else:
            self.bayer_pattern = None
        self.ap_size = int(kwargs.pop("ap_size", 64))
        self.ap_spacing = int(kwargs.pop("ap_spacing", 48))
        self.ap_min_mean = float(kwargs.pop("ap_min_mean", 0.03))
        self.ap_multiscale = bool(kwargs.pop("ap_multiscale", False))
        self.ssd_refine_bruteforce = bool(kwargs.pop("ssd_refine_bruteforce", False))
        self.keep_mask = kwargs.pop("keep_mask", None)
        # Planetary centroid knobs (pure data, no UI references)
        self.planet_smooth_sigma = float(kwargs.pop("planet_smooth_sigma", 1.5))
        self.planet_thresh_pct   = float(kwargs.pop("planet_thresh_pct", 92.0))
        self.planet_min_val      = float(kwargs.pop("planet_min_val", 0.02))
        self.planet_use_norm     = bool(kwargs.pop("planet_use_norm", True))
        self.planet_norm_lo_pct  = float(kwargs.pop("planet_norm_lo_pct", 1.0))
        self.planet_norm_hi_pct  = float(kwargs.pop("planet_norm_hi_pct", 99.5))

        # sanitize
        self.planet_smooth_sigma = max(0.0, self.planet_smooth_sigma)
        self.planet_thresh_pct   = float(np.clip(self.planet_thresh_pct, 0.0, 100.0)) if "np" in globals() else self.planet_thresh_pct
        self.planet_min_val      = float(max(0.0, min(1.0, self.planet_min_val)))
        self.planet_norm_lo_pct  = float(np.clip(self.planet_norm_lo_pct, 0.0, 100.0)) if "np" in globals() else self.planet_norm_lo_pct
        self.planet_norm_hi_pct  = float(np.clip(self.planet_norm_hi_pct, 0.0, 100.0)) if "np" in globals() else self.planet_norm_hi_pct
        if self.planet_norm_hi_pct <= self.planet_norm_lo_pct:
            self.planet_norm_hi_pct = min(100.0, self.planet_norm_lo_pct + 1.0)
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
        # ✅ NEW: Derotation UI params (viewer/stacker)
        self.derotate_enabled = bool(kwargs.pop("derotate_enabled", False))
        self.derotate_deg_per_min = float(kwargs.pop("derotate_deg_per_min", 0.0))

        dc = kwargs.pop("derotate_center", None)
        if dc is None:
            self.derotate_center = None
        else:
            try:
                self.derotate_center = (float(dc[0]), float(dc[1]))
            except Exception:
                self.derotate_center = None

        dr = kwargs.pop("derotate_radius", None)
        self.derotate_radius = None if dr is None else float(dr)

        self.derotate_overlay_mode = str(kwargs.pop("derotate_overlay_mode", "none")).strip().lower()

        rings = kwargs.pop("derotate_rings", None)
        if rings is None:
            self.derotate_rings = None
        else:
            try:
                # (pa, tilt, kout, kin)
                self.derotate_rings = (float(rings[0]), float(rings[1]), float(rings[2]), float(rings[3]))
            except Exception:
                self.derotate_rings = None

        if kwargs:
            raise TypeError(f"Unexpected config keys: {sorted(kwargs.keys())}")
