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
    track_mode: str = "planetary"
    surface_anchor: Optional[Tuple[int,int,int,int]] = None
    keep_percent: float = 20.0
    ap_size: int = 64
    ap_spacing: int = 48
    ap_min_mean: float = 0.03
    ap_multiscale: bool = False

    def __init__(self, source: PlanetarySource, **kwargs):
        # Allow deprecated/ignored kwargs without crashing
        kwargs.pop("multipoint", None)  # accept but ignore
        # Now assign known fields
        self.source = source
        self.roi = kwargs.pop("roi", None)
        self.track_mode = kwargs.pop("track_mode", "planetary")
        self.surface_anchor = kwargs.pop("surface_anchor", None)
        self.keep_percent = float(kwargs.pop("keep_percent", 20.0))
        self.ap_size = int(kwargs.pop("ap_size", 64))
        self.ap_spacing = int(kwargs.pop("ap_spacing", 48))
        self.ap_min_mean = float(kwargs.pop("ap_min_mean", 0.03))
        self.ap_multiscale = bool(kwargs.pop("ap_multiscale", False))
        if kwargs:
            raise TypeError(f"Unexpected config keys: {sorted(kwargs.keys())}")
