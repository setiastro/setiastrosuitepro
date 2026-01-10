# src/setiastro/saspro/ser_stack_config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

TrackMode = Literal["off", "planetary", "surface"]

@dataclass
class SERStackConfig:
    ser_path: str
    roi: Optional[Tuple[int, int, int, int]] = None
    track_mode: str = "planetary"          # planetary/surface/off
    surface_anchor: Optional[Tuple[int,int,int,int]] = None
    keep_percent: float = 20.0

    # --- multipoint alignment ---
    multipoint: bool = False
    ap_size: int = 64
    ap_spacing: int = 48
    ap_min_mean: float = 0.03     
