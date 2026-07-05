# pro/fx_module.py
from __future__ import annotations
import colorsys
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal, QTimer, QPointF, QEvent, QSettings
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QLabel, QSlider, QHBoxLayout, QComboBox,
    QPushButton, QMessageBox, QScrollArea, QWidget, QGroupBox
)
from PyQt6.QtGui import QPixmap, QImage

try:
    import cv2
except Exception:
    cv2 = None

from setiastro.saspro.widgets.image_utils import (
    to_float01 as _to_float01,
    extract_mask_from_document as _active_mask_array_from_doc
)


# ─── image helpers ────────────────────────────────────────────────────────────

def _as_qimage_rgb8(float01: np.ndarray) -> QImage:
    f = np.asarray(float01, dtype=np.float32)
    if f.ndim == 2:
        f = np.stack([f] * 3, axis=-1)
    elif f.ndim == 3 and f.shape[2] == 1:
        f = np.repeat(f, 3, axis=2)
    buf8 = np.ascontiguousarray(np.clip(f, 0.0, 1.0) * 255.0, dtype=np.uint8)
    h, w, _ = buf8.shape
    return QImage(buf8.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888).copy()


def _luminance(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] >= 3:
        R, G, B = image[..., 0], image[..., 1], image[..., 2]
        return 0.2126 * R + 0.7152 * G + 0.0722 * B
    return image.squeeze() if image.ndim == 3 else image


def _hue_to_rgb(hue_deg: float) -> np.ndarray:
    r, g, b = colorsys.hsv_to_rgb((hue_deg % 360.0) / 360.0, 1.0, 1.0)
    return np.array([r, g, b], dtype=np.float32)


# ─── blend-mode primitives (shared by several effects) ───────────────────────

_BLEND_MODES = ["Screen", "Soft Light", "Lighten"]
MAX_PREVIEW = 1200    # max pixel dimension for the downsampled preview buffer


def _blend_screen(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - base) * (1.0 - blend)


def _blend_soft_light(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    # W3C compositing-spec soft-light formula (matches Photoshop's Soft Light)
    d_lo = base - (1.0 - 2.0 * blend) * base * (1.0 - base)
    d_hi_inner = np.where(base <= 0.25,
                         ((16.0 * base - 12.0) * base + 4.0) * base,
                         np.sqrt(np.clip(base, 0.0, 1.0)))
    d_hi = base + (2.0 * blend - 1.0) * (d_hi_inner - base)
    return np.where(blend <= 0.5, d_lo, d_hi)


def _blend_lighten(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return np.maximum(base, blend)


_BLEND_FUNCS = {
    "Screen":     _blend_screen,
    "Soft Light": _blend_soft_light,
    "Lighten":    _blend_lighten,
}


def _gaussian(img: np.ndarray, sigma: float) -> Optional[np.ndarray]:
    if cv2 is None:
        return None
    sigma = max(0.1, sigma)
    k = int(2 * round(3 * sigma) + 1) | 1
    try:
        return cv2.GaussianBlur(np.ascontiguousarray(img, dtype=np.float32), (k, k), sigma)
    except Exception:
        return None


def _recombine_luma(layer_a: np.ndarray, layer_b: np.ndarray,
                    blended: np.ndarray, amount: float) -> np.ndarray:
    """
    Screen (and similar) blends of two near-identical images run away in
    brightness — screen(x,x) = 2x - x^2, always brighter than x. This pulls
    the result back down: target luma is the average of the two source
    layers' luma, and `blended` gets rescaled per-pixel toward that target
    while keeping its colour/detail shape. `amount` (0..1) is a dry/wet mix
    between the raw blend and the fully luma-corrected version.
    """
    if amount <= 0.001:
        return blended
    lum_a       = _luminance(layer_a)
    lum_b       = _luminance(layer_b)
    lum_target  = 0.5 * (lum_a + lum_b)
    lum_blended = _luminance(blended)
    ratio = lum_target / (lum_blended + 1e-6)
    if blended.ndim == 3 and ratio.ndim == 2:
        ratio = ratio[:, :, None]
    corrected = np.clip(blended * ratio, 0.0, 1.0)
    return blended * (1.0 - amount) + corrected * amount


# ─── effect implementations ───────────────────────────────────────────────────

def _fx_orton_glow(img: np.ndarray, p: dict) -> np.ndarray:
    """Classic Orton effect: blur + brighten a duplicate, blend back at partial opacity."""
    opacity = float(p["opacity"])
    if opacity < 0.001:
        return img
    blurred = _gaussian(img, float(p["blur_radius"]))
    if blurred is None:
        return img
    boosted = np.clip(blurred * max(0.01, float(p["glow_brightness"])), 0.0, 1.0)
    blend_fn = _BLEND_FUNCS.get(p["blend_mode"], _blend_screen)
    blended = np.clip(blend_fn(img, boosted), 0.0, 1.0)
    blended = _recombine_luma(img, boosted, blended, float(p.get("luma_recovery", 0.0)))

    highlight_protect = float(p["highlight_protect"])
    if highlight_protect > 0.0:
        lum = _luminance(img)
        rolloff = np.clip((lum - 0.7) / 0.3, 0.0, 1.0)
        protect = 1.0 - highlight_protect * rolloff
        if img.ndim == 3 and protect.ndim == 2:
            protect = protect[:, :, None]
    else:
        protect = 1.0

    eff = opacity * protect
    return np.clip(img * (1.0 - eff) + blended * eff, 0.0, 1.0).astype(np.float32)


def _fx_soft_focus(img: np.ndarray, p: dict) -> np.ndarray:
    """Gentle diffusion — straight blur/opacity mix, no brightening. Classic portrait diffusion."""
    opacity = float(p["opacity"])
    if opacity < 0.001:
        return img
    blurred = _gaussian(img, float(p["blur_radius"]))
    if blurred is None:
        return img
    out = img * (1.0 - opacity) + blurred * opacity
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _fx_bloom(img: np.ndarray, p: dict) -> np.ndarray:
    """Isolates bright highlights, blurs just those, screens them back — halo around stars/discs."""
    opacity = float(p["opacity"])
    if opacity < 0.001:
        return img
    threshold = float(p["threshold"])
    lum = _luminance(img)
    mask = np.clip((lum - threshold) / max(1e-3, 1.0 - threshold), 0.0, 1.0)
    mask3 = mask[..., None] if img.ndim == 3 else mask
    highlights = img * mask3

    blurred = _gaussian(highlights, float(p["blur_radius"]))
    if blurred is None:
        return img
    boosted = np.clip(blurred * max(0.01, float(p["brightness"])), 0.0, 1.0)
    composite = np.clip(_blend_screen(img, boosted), 0.0, 1.0)
    composite = _recombine_luma(img, boosted, composite, float(p.get("luma_recovery", 0.0)))
    out = img * (1.0 - opacity) + composite * opacity
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _fx_vignette(img: np.ndarray, p: dict) -> np.ndarray:
    """Radial darkening toward the frame edges."""
    amount = float(p["amount"])
    if amount < 0.001:
        return img
    radius   = float(p["radius"])
    softness = float(p["softness"])

    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ny = (yy - cy) / max(cy, 1e-6)
    nx = (xx - cx) / max(cx, 1e-6)
    dist = np.sqrt(nx ** 2 + ny ** 2)

    lo = max(1e-3, radius - softness)
    hi = max(lo + 1e-3, radius + softness)
    t = np.clip((dist - lo) / (hi - lo), 0.0, 1.0)
    vig = 1.0 - amount * t
    vig3 = vig[..., None] if img.ndim == 3 else vig
    return np.clip(img * vig3, 0.0, 1.0).astype(np.float32)


def _fx_film_grain(img: np.ndarray, p: dict) -> np.ndarray:
    """Adds organic monochrome or colour grain. Fixed seed keeps the pattern stable while
    other sliders are adjusted, rather than re-randomizing on every preview tick."""
    intensity = float(p["intensity"])
    if intensity < 0.001:
        return img
    size = float(p["size"])
    mono = p.get("mono", "Yes") == "Yes"

    rng = np.random.RandomState(42)
    h, w = img.shape[:2]

    if mono or img.ndim == 2:
        noise = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
        if size > 0.01 and cv2 is not None:
            sigma = max(0.3, size)
            k = int(2 * round(3 * sigma) + 1) | 1
            noise = cv2.GaussianBlur(noise, (k, k), sigma)
            noise = noise / (noise.std() + 1e-6)
        noise3 = noise[..., None] if img.ndim == 3 else noise
    else:
        noise = rng.normal(0.0, 1.0, (h, w, 3)).astype(np.float32)
        if size > 0.01 and cv2 is not None:
            sigma = max(0.3, size)
            k = int(2 * round(3 * sigma) + 1) | 1
            for c in range(3):
                noise[..., c] = cv2.GaussianBlur(noise[..., c], (k, k), sigma)
            noise = noise / (noise.std() + 1e-6)
        noise3 = noise

    out = img + noise3 * intensity * 0.15
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _fx_split_tone(img: np.ndarray, p: dict) -> np.ndarray:
    """Tints shadows and highlights with independent hues."""
    strength = float(p["strength"])
    if strength < 0.001 or img.ndim != 3 or img.shape[2] < 3:
        return img
    balance = float(p["balance"])
    lum  = _luminance(img)
    w_hi = np.clip(lum + balance * 0.5, 0.0, 1.0)
    w_lo = 1.0 - w_hi

    shadow_rgb    = _hue_to_rgb(float(p["shadow_hue"]))
    highlight_rgb = _hue_to_rgb(float(p["highlight_hue"]))
    tint = w_lo[..., None] * shadow_rgb + w_hi[..., None] * highlight_rgb

    out = img * (1.0 - strength) + np.clip(img * 2.0 * tint, 0.0, 1.0) * strength
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ─── param spec + effect registry ─────────────────────────────────────────────

@dataclass
class ParamSpec:
    key: str
    label: str
    kind: str = "slider"          # "slider" | "combo"
    slider_lo: int = 0
    slider_hi: int = 100
    slider_default: int = 50
    scale: float = 1.0            # real_value = raw_slider_value / scale
    fmt: Callable[[float], str] = field(default=lambda v: f"{v:.2f}")
    tooltip: str = ""
    choices: Optional[List[str]] = None
    default_choice: Optional[str] = None


EFFECTS: List[dict] = [
    {
        "key": "orton_glow",
        "name": "Orton Glow",
        "description": "Classic film glow — blur + brighten a duplicate, blend back at "
                       "partial opacity. Dreamy, hazy look.",
        "process": _fx_orton_glow,
        "params": [
            ParamSpec("blur_radius", "Blur radius", "slider", 10, 1000, 150, 10.0,
                      lambda v: f"{v:.1f} px", "Size of the soft glow halo."),
            ParamSpec("glow_brightness", "Glow brightness", "slider", 50, 300, 140, 100.0,
                      lambda v: f"{v:.2f}×",
                      "Brightness boost of the blurred duplicate before blending — mirrors "
                      "the film technique of overexposing the soft exposure."),
            ParamSpec("opacity", "Opacity", "slider", 0, 100, 50, 100.0,
                      lambda v: f"{v:.2f}", "Overall blend strength."),
            ParamSpec("blend_mode", "Blend mode", "combo", choices=_BLEND_MODES,
                      default_choice="Screen",
                      tooltip="Screen: brightest/hazy. Soft Light: gentler, protects shadows. "
                             "Lighten: least colour shift."),
            ParamSpec("highlight_protect", "Highlight protection", "slider", 0, 100, 50, 100.0,
                      lambda v: f"{v:.2f}",
                      "Fades the glow out near already-clipped highlights (e.g. a bright "
                      "solar disc) so it doesn't blow out further."),
            ParamSpec("luma_recovery", "Luma recovery", "slider", 0, 100, 70, 100.0,
                      lambda v: f"{v:.2f}",
                      "Screening two similar images always overshoots in brightness. This "
                      "rescales the blend back toward the average luma of the two source "
                      "layers — higher = closer to the original exposure level."),
        ],
    },
    {
        "key": "soft_focus",
        "name": "Soft Focus",
        "description": "Gentle diffusion blend — softens micro-contrast without brightening. "
                       "Classic portrait/glamour look.",
        "process": _fx_soft_focus,
        "params": [
            ParamSpec("blur_radius", "Blur radius", "slider", 10, 1000, 100, 10.0,
                      lambda v: f"{v:.1f} px", "Softness radius."),
            ParamSpec("opacity", "Opacity", "slider", 0, 100, 40, 100.0,
                      lambda v: f"{v:.2f}", "Blend strength."),
        ],
    },
    {
        "key": "bloom",
        "name": "Bloom / Star Glow",
        "description": "Isolates bright highlights, blurs just those, and screens them back — "
                       "halo around stars or bright disc edges.",
        "process": _fx_bloom,
        "params": [
            ParamSpec("threshold", "Highlight threshold", "slider", 0, 100, 70, 100.0,
                      lambda v: f"{v:.2f}",
                      "Luminance above which pixels are treated as highlights."),
            ParamSpec("blur_radius", "Blur radius", "slider", 10, 1000, 200, 10.0,
                      lambda v: f"{v:.1f} px", "Halo size."),
            ParamSpec("brightness", "Highlight brightness", "slider", 50, 400, 150, 100.0,
                      lambda v: f"{v:.2f}×",
                      "Boost applied to isolated highlights before blending."),
            ParamSpec("opacity", "Opacity", "slider", 0, 100, 60, 100.0,
                      lambda v: f"{v:.2f}", "Overall blend strength."),
            ParamSpec("luma_recovery", "Luma recovery", "slider", 0, 100, 70, 100.0,
                      lambda v: f"{v:.2f}",
                      "Screening two similar highlight layers overshoots in brightness. "
                      "This rescales back toward the average luma of the two layers."),
        ],
    },
    {
        "key": "vignette",
        "name": "Vignette",
        "description": "Radial darkening toward the frame edges.",
        "process": _fx_vignette,
        "params": [
            ParamSpec("amount", "Amount", "slider", 0, 100, 50, 100.0,
                      lambda v: f"{v:.2f}", "Darkening strength at the edges."),
            ParamSpec("radius", "Radius", "slider", 10, 200, 100, 100.0,
                      lambda v: f"{v:.2f}",
                      "Distance from centre where falloff begins (1.0 ≈ edge)."),
            ParamSpec("softness", "Softness", "slider", 1, 150, 40, 100.0,
                      lambda v: f"{v:.2f}", "Falloff softness."),
        ],
    },
    {
        "key": "film_grain",
        "name": "Film Grain",
        "description": "Adds organic monochrome or colour grain.",
        "process": _fx_film_grain,
        "params": [
            ParamSpec("intensity", "Intensity", "slider", 0, 100, 30, 100.0,
                      lambda v: f"{v:.2f}", "Grain strength."),
            ParamSpec("size", "Grain size", "slider", 0, 50, 10, 10.0,
                      lambda v: f"{v:.1f}",
                      "Grain clump size (0 = fine, higher = coarser)."),
            ParamSpec("mono", "Colour", "combo", choices=["Yes", "No"], default_choice="Yes",
                      tooltip="Yes = monochrome grain, No = independent colour grain."),
        ],
    },
    {
        "key": "split_tone",
        "name": "Split Tone",
        "description": "Tints shadows and highlights with independent hues.",
        "process": _fx_split_tone,
        "params": [
            ParamSpec("shadow_hue", "Shadow hue", "slider", 0, 360, 220, 1.0,
                      lambda v: f"{v:.0f}°", "Hue applied to shadows."),
            ParamSpec("highlight_hue", "Highlight hue", "slider", 0, 360, 40, 1.0,
                      lambda v: f"{v:.0f}°", "Hue applied to highlights."),
            ParamSpec("balance", "Balance", "slider", -100, 100, 0, 100.0,
                      lambda v: f"{v:.2f}", "Shifts the shadow/highlight split point."),
            ParamSpec("strength", "Strength", "slider", 0, 100, 30, 100.0,
                      lambda v: f"{v:.2f}", "Overall tint strength."),
        ],
    },
]

_EFFECTS_BY_KEY = {e["key"]: e for e in EFFECTS}


def _process_fx(effect_key: str, img: np.ndarray, params: dict) -> np.ndarray:
    eff = _EFFECTS_BY_KEY.get(effect_key)
    if eff is None:
        return img
    s = img
    if s.ndim == 3 and s.shape[2] == 1:
        s2 = s[..., 0]
        out = eff["process"](s2, params)
        out = out[..., None]
    else:
        out = eff["process"](s, params)
    return out.astype(np.float32, copy=False)


def _blend_mask(out: np.ndarray, src: np.ndarray,
                mask01: np.ndarray | None) -> np.ndarray:
    if mask01 is None:
        return out
    h, w = out.shape[:2]
    m = mask01
    if m.shape != (h, w):
        if cv2 is not None:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            yi = np.linspace(0, m.shape[0] - 1, h).astype(np.int32)
            xi = np.linspace(0, m.shape[1] - 1, w).astype(np.int32)
            m = m[yi][:, xi]
    if out.ndim == 3 and m.ndim == 2:
        m = m[:, :, None]
    src_f = src
    if out.ndim != src_f.ndim:
        if out.ndim == 2 and src_f.ndim == 3:
            src_f = src_f.squeeze()
        elif out.ndim == 3 and src_f.ndim == 2:
            src_f = np.repeat(src_f[:, :, None], out.shape[2], axis=2)
    return np.clip(src_f * (1.0 - m) + out * m, 0.0, 1.0).astype(np.float32)


# ─── headless entry point ─────────────────────────────────────────────────────

def fx_headless(doc, effect: str = "orton_glow", **params):
    """
    Apply an FX effect headlessly.

    `effect` selects the registry key (see EFFECTS). Any remaining keyword
    arguments are passed straight through as that effect's params dict —
    unlike the other tools' headless functions (which take fixed named
    kwargs), this one is intentionally generic since each effect has a
    different parameter set. If your replay dispatcher expects a fixed
    signature per command_id, splat the stored preset dict as
    fx_headless(doc, **preset) where preset = {"effect": key, **params}.
    """
    if doc is None or getattr(doc, "image", None) is None:
        return
    src = _to_float01(np.asarray(doc.image))
    if src is None:
        return
    eff = _EFFECTS_BY_KEY.get(effect)
    if eff is None:
        return
    # fill any missing params with their defaults so partial presets still work
    full_params = {
        spec.key: (spec.slider_default / spec.scale if spec.kind == "slider" else spec.default_choice)
        for spec in eff["params"]
    }
    full_params.update(params)

    out = _process_fx(effect, src, full_params)
    out = _blend_mask(out, src, _active_mask_array_from_doc(doc))
    doc.apply_edit(out, metadata={
        "step_name": f"FX — {eff['name']}",
        "command_id": "fx",
        "effect": effect,
        "preset": full_params,
        "fx": full_params,
    }, step_name=f"FX — {eff['name']}")


# ─── worker ───────────────────────────────────────────────────────────────────

class FXWorker(QThread):
    preview_ready = pyqtSignal(object)   # np.ndarray

    def __init__(self, image: np.ndarray, effect: str, params: dict,
                 mask01: np.ndarray | None = None):
        super().__init__()
        self.image   = image
        self.effect  = effect
        self.params  = params
        self.mask01  = mask01
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        src = _to_float01(self.image)
        if src is None or self._cancel:
            return
        out = _process_fx(self.effect, src, self.params)
        if self._cancel:
            return
        out = _blend_mask(out, src, self.mask01)
        if not self._cancel:
            self.preview_ready.emit(out)


class _FXApplyWorker(QObject):
    """
    Runs the full-resolution FX processing off the GUI thread. Pure numpy
    work only — never touches the document, Qt widgets, or anything else
    that isn't thread-safe. The main thread does doc.apply_edit() once this
    emits finished().
    """
    finished = pyqtSignal(bool, str, object)  # ok, error_message, result_array

    def __init__(self, source: np.ndarray, effect_key: str, params: dict,
                 mask01: Optional[np.ndarray] = None):
        super().__init__()
        self._source     = source
        self._effect_key = effect_key
        self._params     = params
        self._mask01     = mask01

    def run(self):
        try:
            out = _process_fx(self._effect_key, self._source, self._params)
            out = _blend_mask(out, self._source, self._mask01)
            self.finished.emit(True, "", out)
        except Exception as e:
            import traceback
            self.finished.emit(False, f"{e}\n\n{traceback.format_exc()}", None)


# ─── dialog ───────────────────────────────────────────────────────────────────

class FXDialog(QDialog):
    def __init__(self, main, doc, parent=None, initial_effect: str = "orton_glow"):
        super().__init__(parent)
        self.main = main
        self.doc  = doc
        self.setWindowTitle("FX")
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)

        # preview state
        self._pix_processed: QPixmap | None = None
        self._pix_original:  QPixmap | None = None
        self._preview_src:   Optional[np.ndarray] = None   # downsampled buffer for fast preview
        self._showing_original = False
        self._zoom = 0.25
        self._panning = False
        self._pan_start = QPointF()
        self._did_initial_fit = False
        self._worker: FXWorker | None = None
        self._apply_thread: QThread | None = None
        self._apply_worker: _FXApplyWorker | None = None

        # effect state
        self._param_widgets: dict = {}   # key -> (kind, widget, ParamSpec)
        self._current_effect = _EFFECTS_BY_KEY.get(initial_effect, EFFECTS[0])

        # active-document tracking
        self._connected_doc_change = False
        if hasattr(self.main, "currentDocumentChanged"):
            try:
                self.main.currentDocumentChanged.connect(self._on_active_doc_changed)
                self._connected_doc_change = True
            except Exception:
                pass

        # debounce timer
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self._trigger_preview)

        self._build_ui()
        self._cache_original()
        self._select_effect(self._current_effect["key"], initial=True)

    # ── cache original ────────────────────────────────────────────────────────

    def _build_preview_src(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min(1.0, MAX_PREVIEW / max(h, w, 1))
        if scale < 1.0:
            nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
            if cv2 is not None:
                prev = cv2.resize(np.ascontiguousarray(img, dtype=np.float32),
                                  (nw, nh), interpolation=cv2.INTER_AREA)
            else:
                yi = np.linspace(0, h - 1, nh).astype(np.int32)
                xi = np.linspace(0, w - 1, nw).astype(np.int32)
                prev = img[yi][:, xi]
        else:
            prev = img.copy()
        return prev.astype(np.float32)

    def _cache_original(self):
        if self.doc is None or getattr(self.doc, "image", None) is None:
            self._pix_original = None
            self._preview_src  = None
            return
        src = _to_float01(np.asarray(self.doc.image))
        if src is None:
            self._preview_src = None
            return
        self._preview_src = self._build_preview_src(src)
        self._pix_original = QPixmap.fromImage(_as_qimage_rgb8(self._preview_src))

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── left: controls ────────────────────────────────────────────────────
        left_w = QWidget()
        left_w.setMinimumWidth(300)
        left_w.setMaximumWidth(340)
        left = QVBoxLayout(left_w)
        left.setSpacing(6)

        effect_row = QHBoxLayout()
        effect_row.addWidget(QLabel("Effect:"))
        self.combo_effect = QComboBox()
        self.combo_effect.addItems([e["name"] for e in EFFECTS])
        self.combo_effect.currentIndexChanged.connect(self._on_effect_index_changed)
        effect_row.addWidget(self.combo_effect, 1)
        left.addLayout(effect_row)

        self._lbl_desc = QLabel("", self)
        self._lbl_desc.setWordWrap(True)
        self._lbl_desc.setStyleSheet("font-size: 10px; color: #999; margin-bottom: 2px;")
        left.addWidget(self._lbl_desc)

        self._params_grp = QGroupBox("Parameters")
        self._controls_lay = QVBoxLayout(self._params_grp)
        self._controls_lay.setSpacing(4)
        left.addWidget(self._params_grp)

        left.addSpacing(4)

        # ── before/after toggle ───────────────────────────────────────────────
        self.btn_toggle = QPushButton("⇄  Toggle Before / After")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setToolTip(
            "Instantly switch between original and processed preview.\n"
            "No recalculation — uses cached images."
        )
        self.btn_toggle.toggled.connect(self._on_toggle)
        left.addWidget(self.btn_toggle)

        self.lbl_status = QLabel("Ready.", self)
        self.lbl_status.setStyleSheet("color: #777; font-size: 10px;")
        left.addWidget(self.lbl_status)

        left.addStretch(1)

        # action buttons
        btn_row = QHBoxLayout()
        self.btn_apply  = QPushButton("Apply")
        self.btn_reset  = QPushButton("Reset")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_apply.clicked.connect(self._apply)
        self.btn_reset.clicked.connect(self._reset)
        self.btn_cancel.clicked.connect(self.close)
        for b in (self.btn_apply, self.btn_reset, self.btn_cancel):
            btn_row.addWidget(b)
        left.addLayout(btn_row)

        root.addWidget(left_w, 0)

        # ── right: preview ────────────────────────────────────────────────────
        right = QVBoxLayout()

        zoom_row = QHBoxLayout()
        for label, slot in (("–", self._zoom_out), ("+", self._zoom_in),
                             ("Fit", self._fit)):
            b = QPushButton(label)
            b.setFixedWidth(44)
            b.clicked.connect(slot)
            zoom_row.addWidget(b)
        zoom_row.addStretch(1)
        right.addLayout(zoom_row)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.viewport().installEventFilter(self)
        self.preview_lbl = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.preview_lbl)
        right.addWidget(self.scroll, 1)

        root.addLayout(right, 1)
        self.resize(1050, 620)

    # ── effect selection / dynamic controls ───────────────────────────────────

    def _on_effect_index_changed(self, idx: int):
        if 0 <= idx < len(EFFECTS):
            self._select_effect(EFFECTS[idx]["key"])

    def _select_effect(self, effect_key: str, initial: bool = False):
        eff = _EFFECTS_BY_KEY.get(effect_key)
        if eff is None:
            return
        self._current_effect = eff

        if not initial:
            idx = next((i for i, e in enumerate(EFFECTS) if e["key"] == effect_key), 0)
            if self.combo_effect.currentIndex() != idx:
                self.combo_effect.blockSignals(True)
                self.combo_effect.setCurrentIndex(idx)
                self.combo_effect.blockSignals(False)

        self._lbl_desc.setText(eff["description"])
        self._rebuild_controls(eff)
        self._did_initial_fit = False
        self._unsync_toggle()
        self._trigger_preview()

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
            elif item.layout() is not None:
                self._clear_layout(item.layout())

    def _rebuild_controls(self, eff: dict):
        self._clear_layout(self._controls_lay)
        self._param_widgets = {}

        for spec in eff["params"]:
            if spec.kind == "slider":
                real_default = spec.slider_default / spec.scale
                lbl = QLabel(f"{spec.label}: {spec.fmt(real_default)}")
                sld = QSlider(Qt.Orientation.Horizontal)
                sld.setRange(spec.slider_lo, spec.slider_hi)
                sld.setValue(spec.slider_default)
                sld.setToolTip(spec.tooltip)
                sld.valueChanged.connect(
                    lambda v, s=spec, l=lbl: self._on_slider_changed(s, l, v))
                self._controls_lay.addWidget(lbl)
                self._controls_lay.addWidget(sld)
                self._param_widgets[spec.key] = ("slider", sld, spec)
            else:
                row = QHBoxLayout()
                row.addWidget(QLabel(f"{spec.label}:"))
                combo = QComboBox()
                combo.addItems(spec.choices or [])
                if spec.default_choice:
                    combo.setCurrentText(spec.default_choice)
                combo.setToolTip(spec.tooltip)
                combo.currentTextChanged.connect(lambda _t: self._on_combo_changed())
                row.addWidget(combo, 1)
                self._controls_lay.addLayout(row)
                self._param_widgets[spec.key] = ("combo", combo, spec)

    def _on_slider_changed(self, spec: ParamSpec, lbl: QLabel, raw: int):
        real = raw / spec.scale
        lbl.setText(f"{spec.label}: {spec.fmt(real)}")
        self._unsync_toggle()
        self._preview_timer.start()

    def _on_combo_changed(self):
        self._unsync_toggle()
        self._preview_timer.start()

    def _unsync_toggle(self):
        if self.btn_toggle.isChecked():
            self.btn_toggle.blockSignals(True)
            self.btn_toggle.setChecked(False)
            self.btn_toggle.blockSignals(False)
            self._showing_original = False

    def _get_params(self) -> dict:
        params = {}
        for key, (kind, widget, spec) in self._param_widgets.items():
            if kind == "slider":
                params[key] = widget.value() / spec.scale
            else:
                params[key] = widget.currentText()
        return params

    # ── before/after toggle ───────────────────────────────────────────────────

    def _on_toggle(self, showing_before: bool):
        self._showing_original = showing_before
        if showing_before:
            if self._pix_original is not None:
                self._show_pixmap(self._pix_original)
        else:
            if self._pix_processed is not None:
                self._show_pixmap(self._pix_processed)
            else:
                self._trigger_preview()

    # ── preview ───────────────────────────────────────────────────────────────

    def _trigger_preview(self):
        if self._preview_src is None:
            return
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(300)

        params = self._get_params()
        mask01 = _active_mask_array_from_doc(self.doc)
        self._worker = FXWorker(self._preview_src, self._current_effect["key"], params, mask01)
        self._worker.preview_ready.connect(self._on_preview_ready)
        self._worker.start()

    def _on_preview_ready(self, out: np.ndarray):
        self._pix_processed = QPixmap.fromImage(_as_qimage_rgb8(out))
        if not self._showing_original:
            self._show_pixmap(self._pix_processed)
        if not self._did_initial_fit:
            self._did_initial_fit = True
            QTimer.singleShot(0, self._fit)

    def _show_pixmap(self, pix: QPixmap):
        scaled = pix.scaled(
            pix.size() * self._zoom,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_lbl.setPixmap(scaled)
        self.preview_lbl.resize(scaled.size())

    # ── reset ─────────────────────────────────────────────────────────────────

    def _reset(self):
        self._rebuild_controls(self._current_effect)
        self._preview_timer.start()

    # ── apply ─────────────────────────────────────────────────────────────────

    def _set_controls_enabled(self, enabled: bool):
        self.btn_apply.setEnabled(enabled)
        self.btn_reset.setEnabled(enabled)
        self.btn_cancel.setEnabled(enabled)
        self.combo_effect.setEnabled(enabled)
        for _kind, widget, _spec in self._param_widgets.values():
            widget.setEnabled(enabled)

    def _apply(self):
        if self.doc is None or getattr(self.doc, "image", None) is None:
            return
        if self._apply_thread is not None:
            return  # already running

        effect_key  = self._current_effect["key"]
        effect_name = self._current_effect["name"]
        params      = self._get_params()

        try:
            src = _to_float01(np.asarray(self.doc.image))
            if src is None:
                QMessageBox.information(self, "FX", "No image to process.")
                return
            mask01 = _active_mask_array_from_doc(self.doc)

            self.lbl_status.setText(
                f"Applying {effect_name} — processing full-resolution image…")
            self._set_controls_enabled(False)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

            self._apply_thread = QThread(self)
            self._apply_worker = _FXApplyWorker(src, effect_key, params, mask01)
            self._apply_worker.moveToThread(self._apply_thread)
            self._apply_thread.started.connect(
                self._apply_worker.run, Qt.ConnectionType.QueuedConnection)
            self._apply_worker.finished.connect(
                lambda ok, msg, out: self._on_apply_finished(
                    ok, msg, out, effect_key, effect_name, params),
                Qt.ConnectionType.QueuedConnection)
            self._apply_thread.start()
        except Exception:
            import traceback
            QMessageBox.critical(self, "FX", traceback.format_exc())
            self._teardown_apply_thread()
            self._set_controls_enabled(True)
            QApplication.restoreOverrideCursor()

    def _teardown_apply_thread(self):
        if self._apply_thread is not None:
            try:
                self._apply_thread.quit()
                self._apply_thread.wait()
            except Exception:
                pass
        self._apply_thread = None
        self._apply_worker = None

    def _on_apply_finished(self, ok: bool, message: str, out,
                           effect_key: str, effect_name: str, params: dict):
        self._teardown_apply_thread()
        self._set_controls_enabled(True)
        QApplication.restoreOverrideCursor()

        if not ok:
            self.lbl_status.setText("Apply failed.")
            QMessageBox.critical(self, "FX", message)
            return

        try:
            self.doc.apply_edit(out, metadata={
                "step_name": f"FX — {effect_name}",
                "command_id": "fx",
                "effect": effect_key,
                "preset": params,
                "fx": params,
            }, step_name=f"FX — {effect_name}")

            try:
                mw = self.main
                if hasattr(mw, "_remember_last_headless_command"):
                    preset = {"effect": effect_key, **params}
                    mw._remember_last_headless_command(
                        "fx", preset, description=f"FX — {effect_name}",
                    )
            except Exception:
                pass

            self.lbl_status.setText("Applied.")
            self._save_geometry()
            QTimer.singleShot(400, self.close)
        except Exception:
            import traceback
            self.lbl_status.setText("Apply failed.")
            QMessageBox.critical(self, "FX", traceback.format_exc())

    # ── active doc change ─────────────────────────────────────────────────────

    def _on_active_doc_changed(self, doc):
        if doc is None or getattr(doc, "image", None) is None:
            return
        if doc is self.doc:
            return
        self.doc = doc
        title = doc.display_name() if hasattr(doc, "display_name") else "Image"
        self.setWindowTitle(f"FX — {title}")
        self._pix_processed = None
        self._showing_original = False
        self.btn_toggle.blockSignals(True)
        self.btn_toggle.setChecked(False)
        self.btn_toggle.blockSignals(False)
        self._cache_original()
        self._did_initial_fit = False
        self._trigger_preview()

    # ── zoom / pan ────────────────────────────────────────────────────────────
    def _zoom_in(self):
        vp = self.scroll.viewport()
        self._set_zoom(self._zoom * 1.25,
                       anchor=QPointF(vp.width() / 2.0, vp.height() / 2.0))

    def _zoom_out(self):
        vp = self.scroll.viewport()
        self._set_zoom(self._zoom / 1.25,
                       anchor=QPointF(vp.width() / 2.0, vp.height() / 2.0))

    def _set_zoom(self, z: float, anchor: QPointF | None = None):
        old_zoom = self._zoom
        self._zoom = max(0.05, min(z, 5.0))

        pix = (self._pix_original if self._showing_original
               else self._pix_processed)
        if pix is None:
            return

        scaled = pix.scaled(
            pix.size() * self._zoom,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_lbl.setPixmap(scaled)
        self.preview_lbl.resize(scaled.size())

        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()

        if anchor is None:
            vp = self.scroll.viewport()
            anchor = QPointF(vp.width() / 2.0, vp.height() / 2.0)

        cx = hbar.value() + anchor.x()
        cy = vbar.value() + anchor.y()

        factor = self._zoom / max(old_zoom, 1e-9)

        hbar.setValue(int(cx * factor - anchor.x()))
        vbar.setValue(int(cy * factor - anchor.y()))

    def _fit(self):
        pix = (self._pix_original if self._showing_original
               else self._pix_processed)
        if pix is None:
            return
        vp = self.scroll.viewport().size()
        s  = min(vp.width() / pix.width(), vp.height() / pix.height())
        self._set_zoom(max(0.05, s))

    def eventFilter(self, obj, ev):
        if obj is self.scroll.viewport():
            t = ev.type()
            if (t == QEvent.Type.Wheel
                    and ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
                factor = 1.25 if ev.angleDelta().y() > 0 else 0.8
                self._set_zoom(self._zoom * factor,
                               anchor=ev.position())
                ev.accept(); return True
            if t == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = True
                self._pan_start = ev.position()
                self.scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                ev.accept(); return True
            if t == QEvent.Type.MouseMove and self._panning:
                d = ev.position() - self._pan_start
                self.scroll.horizontalScrollBar().setValue(
                    self.scroll.horizontalScrollBar().value() - int(d.x()))
                self.scroll.verticalScrollBar().setValue(
                    self.scroll.verticalScrollBar().value() - int(d.y()))
                self._pan_start = ev.position()
                ev.accept(); return True
            if (t == QEvent.Type.MouseButtonRelease
                    and ev.button() == Qt.MouseButton.LeftButton
                    and self._panning):
                self._panning = False
                self.scroll.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                ev.accept(); return True
        return super().eventFilter(obj, ev)

    # ── geometry persistence ──────────────────────────────────────────────────

    def _save_geometry(self):
        try:
            QSettings().setValue("fx/window_geometry", self.saveGeometry())
        except Exception:
            pass

    def _restore_geometry(self):
        try:
            g = QSettings().value("fx/window_geometry")
            if g is not None:
                self.restoreGeometry(g)
        except Exception:
            pass

    def showEvent(self, ev):
        super().showEvent(ev)
        if not getattr(self, "_geom_restored", False):
            self._geom_restored = True
            QTimer.singleShot(0, lambda: (self._restore_geometry(),
                                          self._fit() if self._pix_processed else None))

    def closeEvent(self, ev):
        if self._apply_thread is not None and self._apply_thread.isRunning():
            ev.ignore()
            return
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(500)
        if self._connected_doc_change and hasattr(self.main, "currentDocumentChanged"):
            try:
                self.main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
            except Exception:
                pass
        self._save_geometry()
        super().closeEvent(ev)


# ─── public entry point ───────────────────────────────────────────────────────

def open_fx_dialog(main, doc=None, initial_effect: str = "orton_glow", preset: dict | None = None):
    if doc is None:
        doc = getattr(main, "_active_doc", None)
        if callable(doc):
            doc = doc()
    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.information(main, "FX", "Open an image first.")
        return
    dlg = FXDialog(main, doc, parent=main, initial_effect=initial_effect)
    dlg.show()
    return dlg