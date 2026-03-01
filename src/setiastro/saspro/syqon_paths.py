# src/setiastro/saspro/syqon_paths.py
from __future__ import annotations

from pathlib import Path
from setiastro.saspro.resources import get_resources


def _models_root_fallback() -> Path:
    return Path.home() / ".saspro" / "models"


def syqon_family_dir(family: str) -> Path:
    family = (family or "").strip().lower()
    if family not in ("starless", "denoise", "sharpen"):
        family = "starless"

    folder_map = {
        "starless": "syqon_starless",
        "denoise": "syqon_denoise",
        "sharpen": "syqon_sharpen",
    }

    try:
        r = get_resources()
        base = Path(r.MODELS_DIR)
    except Exception:
        base = _models_root_fallback()

    d = base / folder_map[family]
    d.mkdir(parents=True, exist_ok=True)
    return d


def syqon_prism_model_path(model_kind: str) -> Path:
    mk = (model_kind or "prism_mini").strip().lower()
    d = syqon_family_dir("denoise")

    if mk == "prism_deep":
        return d / "prism_deep.pt"
    return d / "prism_mini.pt"


def syqon_parallax_model_path(model_kind: str = "parallax") -> Path:
    d = syqon_family_dir("sharpen")
    return d / "parallax.pt"