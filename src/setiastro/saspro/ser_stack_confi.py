# src/setiastro/saspro/ser_stack_config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

TrackMode = Literal["off", "planetary", "surface"]

@dataclass
class SERStackConfig:
    ser_path: str
    roi: Optional[Tuple[int,int,int,int]] = None     # x,y,w,h
    track_mode: TrackMode = "planetary"
    surface_anchor: Optional[Tuple[int,int,int,int]] = None  # x,y,w,h (in ROI coords or full-frame coords - pick one!)
    keep_percent: float = 20.0
