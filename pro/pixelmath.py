# pro/pixelmath.py
from __future__ import annotations
import os
import re
import json
import numpy as np

from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QIcon, QCursor, QImage, QPixmap, QTransform, QActionGroup
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout, QLabel,
    QPushButton, QPlainTextEdit, QComboBox, QDialogButtonBox, QRadioButton, QApplication, QSplitter,
    QTabWidget, QWidget, QMessageBox, QMenu, QScrollArea, QButtonGroup, QListWidget, QListWidgetItem, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QToolButton
)

from pro.autostretch import autostretch

# Import shared utilities
from pro.widgets.image_utils import nearest_resize_2d as _nearest_resize_2d
from pro.widgets.image_utils import float_to_qimage_rgb8 as _float_to_qimage_rgb8
from pro.widgets.themed_buttons import themed_toolbtn

# ---- Optional accelerators from legacy.numba_utils -------------------------
try:
    from legacy.numba_utils import fast_mad as _fast_mad
except Exception:
    _fast_mad = None

# _float_to_qimage_rgb8 imported from pro.widgets.image_utils


# =============================================================================
# PixelImage wrapper (vector ops, indexing, ^ as exponent, ~ as invert)
# =============================================================================
class PixelImage:
    """
    Lightweight wrapper to enable intuitive pixel math:
      • Supports per-channel indexing: img[0], img[1], img[2] → (H,W) planes
      • Broadcasts (H,W) ⇄ (H,W,3) for +,-,*,/, power, and comparisons
      • ~img means (1 - img)
    """
    __array_priority__ = 10_000  # ensure numpy uses our dunder ops

    def __init__(self, array: np.ndarray):
        self.array = np.asarray(array, dtype=np.float32)

    # ---- channel indexing ----
    def __getitem__(self, ch):
        a = self.array
        if a.ndim < 3:
            raise ValueError("This image has no channel dimension to index.")
        if not (0 <= ch < a.shape[2]):
            raise IndexError(f"Channel index {ch} out of range for shape {a.shape}")
        return PixelImage(a[..., ch])

    # ---- shape coercion (H,W) ⇄ (H,W,3) ----
    @staticmethod
    def _coerce(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if a.ndim == 3 and b.ndim == 2:
            b = np.repeat(b[..., None], a.shape[2], axis=2)
        elif a.ndim == 2 and b.ndim == 3:
            a = np.repeat(a[..., None], b.shape[2], axis=2)
        return a, b

    # ---- binary arithmetic helpers ----
    def _bin(self, other, op):
        a = self.array
        b = other.array if isinstance(other, PixelImage) else other
        a, b = self._coerce(a, b)
        return PixelImage(op(a, b))

    # ---- comparisons with coercion (return ndarray masks) ----
    def _cmp(self, other, op):
        a = self.array
        b = other.array if isinstance(other, PixelImage) else other
        a, b = self._coerce(a, b)
        return op(a, b)

    # ---- arithmetic ----
    __add__      = lambda self, o: self._bin(o, np.add)
    __radd__     = __add__
    __sub__      = lambda self, o: self._bin(o, np.subtract)
    __mul__      = lambda self, o: self._bin(o, np.multiply)
    __rmul__     = __mul__
    __truediv__  = lambda self, o: self._bin(o, np.divide)

    def __rsub__(self, o):
        a, b = self._coerce(o.array if isinstance(o, PixelImage) else o, self.array)
        return PixelImage(np.subtract(a, b))

    def __rtruediv__(self, o):
        a, b = self._coerce(o.array if isinstance(o, PixelImage) else o, self.array)
        return PixelImage(np.divide(a, b))

    # power ** and ^
    def __pow__(self, o):
        a = self.array; b = o.array if isinstance(o, PixelImage) else o
        a, b = self._coerce(a, b)
        return PixelImage(np.power(a, b))

    def __rpow__(self, o):
        a = o.array if isinstance(o, PixelImage) else o; b = self.array
        a, b = self._coerce(a, b)
        return PixelImage(np.power(a, b))

    # keep ^ as alias for power for convenience
    def __xor__(self, o):
        return self.__pow__(o)

    def __rxor__(self, o):
        return self.__rpow__(o)

    # invert (~img) → 1 - img
    def __invert__(self):
        return PixelImage(1.0 - self.array)

    # ---- comparisons (return boolean ndarray) ----
    __lt__ = lambda self, o: self._cmp(o, np.less)
    __le__ = lambda self, o: self._cmp(o, np.less_equal)
    __eq__ = lambda self, o: self._cmp(o, np.equal)
    __ne__ = lambda self, o: self._cmp(o, np.not_equal)
    __gt__ = lambda self, o: self._cmp(o, np.greater)
    __ge__ = lambda self, o: self._cmp(o, np.greater_equal)

    def __repr__(self):
        return f"PixelImage(shape={self.array.shape}, dtype={self.array.dtype})"



# =============================================================================
# Helpers
# =============================================================================
_ID_RX = re.compile(r'[^0-9a-zA-Z_]+')
def _sanitize_ident(name: str) -> str:
    s = _ID_RX.sub('_', str(name)).strip('_')
    if not s: s = "view"
    if s[0].isdigit(): s = "_" + s
    return s

def _as_rgb(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    a = np.clip(a, 0.0, 1.0)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    return a

# _nearest_resize_2d imported from pro.widgets.image_utils

def _get_doc_active_mask_2d(doc, H: int, W: int) -> np.ndarray | None:
    """
    Returns the active mask as a 2-D float32 array in [0..1], resized to (H,W).
    """
    if doc is None:
        return None
    mid = getattr(doc, "active_mask_id", None)
    if not mid:
        return None
    masks = getattr(doc, "masks", {}) or {}
    layer = masks.get(mid)
    if layer is None:
        return None

    # Extract data robustly without using `or` on arrays
    data = None
    # object-style
    for attr in ("data", "mask", "image", "array"):
        if hasattr(layer, attr):
            val = getattr(layer, attr)
            if val is not None:
                data = val
                break
    # dict-style
    if data is None and isinstance(layer, dict):
        for key in ("data", "mask", "image", "array"):
            if key in layer and layer[key] is not None:
                data = layer[key]
                break
    # ndarray
    if data is None and isinstance(layer, np.ndarray):
        data = layer
    if data is None:
        return None

    m = np.asarray(data)
    if m.ndim == 3:           # collapse RGB(A) → gray
        m = m.mean(axis=2)
    m = m.astype(np.float32, copy=False)

    # normalize to [0..1]
    if m.max(initial=0.0) > 1.0:
        m /= float(m.max())

    m = np.clip(m, 0.0, 1.0)
    return _nearest_resize_2d(m, H, W)

def _mask_for_ref(doc, ref_like: np.ndarray) -> np.ndarray | None:
    """
    Returns a mask shaped for `ref_like`:
      - 2-D for mono ref
      - H×W×C (broadcast) for color ref
    """
    ref = np.asarray(ref_like)
    H, W = ref.shape[:2]
    m2d = _get_doc_active_mask_2d(doc, H, W)
    if m2d is None:
        return None
    if ref.ndim == 3:
        return np.repeat(m2d[:, :, None], ref.shape[2], axis=2)
    return m2d

def _blend_masked(base: np.ndarray, out: np.ndarray, m: np.ndarray) -> np.ndarray:
    base = _as_rgb(base)  # (H,W,3)
    out  = _as_rgb(out)   # (H,W,3)
    m    = np.asarray(m, dtype=np.float32)
    m    = np.clip(m, 0.0, 1.0)

    # Allow 2-D or 3-D masks
    if m.ndim == 2:
        m = m[..., None]              # (H,W,1)
    elif m.ndim == 3 and m.shape[2] not in (1, 3):
        raise ValueError("Mask must be 2-D or have 1 or 3 channels.")

    return np.clip(base * (1.0 - m) + out * m, 0.0, 1.0)


# =============================================================================
# Headless apply
# =============================================================================
def apply_pixel_math_to_doc(parent, doc, preset: dict | None):
    if doc is None or getattr(doc, "image", None) is None:
        raise RuntimeError("Document has no image.")
    expr = (preset or {}).get("expr", "").strip()
    ev = _Evaluator(parent, doc)
    if expr:
        out = ev.eval_single(expr)
    else:
        r = (preset or {}).get("expr_r", "").strip()
        g = (preset or {}).get("expr_g", "").strip()
        b = (preset or {}).get("expr_b", "").strip()
        if not (r or g or b):
            raise RuntimeError("Pixel Math preset empty.")
        out = ev.eval_rgb(r, g, b, default_channels=(0, 1, 2))

    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
    if hasattr(doc, "set_image"):
        doc.set_image(out, step_name="Pixel Math")
    elif hasattr(doc, "apply_numpy"):
        doc.apply_numpy(out, step_name="Pixel Math")
    else:
        doc.image = out

# =============================================================================
# Evaluator
# =============================================================================
class _Evaluator:
    def __init__(self, parent, doc):
        self.parent = parent
        self.doc = doc
        self._build_namespace()

    def _build_namespace(self):
        self.ns = {
            "np": np,
            # existing:
            "med": self._med, "mean": self._mean, "min": self._min, "max": self._max,
            "std": self._std, "mad": self._mad, "log": self._log, "iff": self._iff, "mtf": self._mtf,
            # new math helpers:
            "clamp": self._clamp,
            "rescale": self._rescale,
            "gamma": self._gamma,
            "pow_safe": self._pow_safe,
            "absf": self._absf,
            "expf": self._expf,
            "sqrtf": self._sqrtf,
            "arcsin": self._arcsin,
            "sigmoid": self._sigmoid,
            "smoothstep": self._smoothstep,
            "lerp": self._lerp, "mix": self._lerp,
            # stats / normalization:
            "percentile": self._percentile,
            "normalize01": self._normalize01,
            "zscore": self._zscore,
            # channels & color:
            "ch": self._ch,
            "luma": self._luma,
            "compose": self._compose,
            # mask helpers:
            "mask": self._mask_fn,
            "apply_mask": self._apply_mask_fn,
            # optional filters (cv2-backed):
            "boxblur": self._boxblur,
            "gauss": self._gauss,
            "median": self._median,
            "unsharp": self._unsharp,
            # constants:
            "pi": float(np.pi), "e": float(np.e), "EPS": 1e-8,
        }

        cur = np.asarray(self.doc.image, dtype=np.float32)
        self._img_shape = cur.shape
        self.ns["img"] = PixelImage(_as_rgb(cur))

        H, W = cur.shape[:2]
        C = 1 if cur.ndim == 2 else cur.shape[2]
        self.ns["H"], self.ns["W"], self.ns["C"] = int(H), int(W), int(C)
        self.ns["shape"] = (int(H), int(W), int(C))

        # Normalized coordinate grids (2-D, float32)
        xx = np.linspace(0.0, 1.0, W, dtype=np.float32)
        yy = np.linspace(0.0, 1.0, H, dtype=np.float32)
        X, Y = np.meshgrid(xx, yy)
        self.ns["X"] = X
        self.ns["Y"] = Y

        # map: raw title → ident (existing)
        self.title_map = []
        open_docs = []
        if hasattr(self.parent, "_subwindow_docs"):
            open_docs = list(self.parent._subwindow_docs())
        else:
            open_docs = [(getattr(self.doc, "display_name", lambda: "view")(), self.doc)]

        used = set(self.ns.keys())
        for raw_title, d in open_docs:
            ident = _sanitize_ident(raw_title or "view")
            base, i = ident, 2
            while ident in used:
                ident = f"{base}_{i}"; i += 1
            used.add(ident)
            arr = getattr(d, "image", None)
            if arr is None:
                continue
            self.ns[ident] = PixelImage(np.asarray(arr, dtype=np.float32))  # keep native 2D/3D
            self.title_map.append((str(raw_title), ident))

    # -------- expression rewriting: allow raw window titles in user code
    def _rewrite_names(self, expr: str) -> str:
        if not expr: return expr
        out = expr
        for raw, ident in self.title_map:
            # raw title
            pat = re.compile(rf'(?<![\w]){re.escape(raw)}(?![\w])')
            out = pat.sub(ident, out)
            # basename without extension
            base = os.path.splitext(raw)[0]
            if base and base != raw:
                pat2 = re.compile(rf'(?<![\w]){re.escape(base)}(?![\w])')
                out = pat2.sub(ident, out)
        return out

    # -------- functions
    def _med(self, x):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x)
        if a.ndim == 2:
            v = np.median(a); out = np.full_like(a, v)
        else:
            v = np.median(a, axis=(0, 1)); out = np.tile(v, (*a.shape[:2], 1))
        return PixelImage(out) if isinstance(x, PixelImage) else out

    def _mean(self, x):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x)
        if a.ndim == 2:
            v = np.mean(a); out = np.full_like(a, v)
        else:
            v = np.mean(a, axis=(0, 1)); out = np.tile(v, (*a.shape[:2], 1))
        return PixelImage(out) if isinstance(x, PixelImage) else out

    def _min(self, x):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x)
        if a.ndim == 2:
            v = np.min(a); out = np.full_like(a, v)
        else:
            v = np.min(a, axis=(0, 1)); out = np.tile(v, (*a.shape[:2], 1))
        return PixelImage(out) if isinstance(x, PixelImage) else out

    def _max(self, x):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x)
        if a.ndim == 2:
            v = np.max(a); out = np.full_like(a, v)
        else:
            v = np.max(a, axis=(0, 1)); out = np.tile(v, (*a.shape[:2], 1))
        return PixelImage(out) if isinstance(x, PixelImage) else out

    def _std(self, x):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x)
        if a.ndim == 2:
            v = np.std(a); out = np.full_like(a, v)
        else:
            v = np.std(a, axis=(0, 1)); out = np.tile(v, (*a.shape[:2], 1))
        return PixelImage(out) if isinstance(x, PixelImage) else out

    def _mad(self, x):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x)
        if a.ndim == 2:
            if _fast_mad is not None:
                v = float(_fast_mad(a))
            else:
                m = np.median(a); v = np.median(np.abs(a - m))
            out = np.full_like(a, v)
        else:
            out = np.empty_like(a)
            for c in range(a.shape[2]):
                ch = a[..., c]
                if _fast_mad is not None:
                    v = float(_fast_mad(ch))
                else:
                    m = np.median(ch); v = np.median(np.abs(ch - m))
                out[..., c] = v
        return PixelImage(out) if isinstance(x, PixelImage) else out

    def _log(self, x):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x)
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.log(np.clip(a, 1e-12, None))
        return PixelImage(y) if isinstance(x, PixelImage) else y

    def _iff(self, cond, a, b):
        c = cond.array if isinstance(cond, PixelImage) else cond
        av = a.array if isinstance(a, PixelImage) else a
        bv = b.array if isinstance(b, PixelImage) else b
        r = np.where(c, av, bv)
        return PixelImage(r) if any(isinstance(z, PixelImage) for z in (cond, a, b)) else r

    def _mtf(self, x, m):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x)
        with np.errstate(divide='ignore', invalid='ignore'):
            y = ((m - 1.0) * a) / (((2.0 * m - 1.0) * a) - m)
        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
        return PixelImage(y) if isinstance(x, PixelImage) else y

    # ---- math helpers ----
    def _clamp(self, x, lo=0.0, hi=1.0):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x, dtype=np.float32)
        y = np.clip(a, float(lo), float(hi))
        return PixelImage(y) if isinstance(x, PixelImage) else y

    def _rescale(self, x, a, b, lo=0.0, hi=1.0):
        a = np.asarray(x.array if isinstance(x, PixelImage) else x, dtype=np.float32)
        src_lo, src_hi = float(a.min()), float(a.max())
        if np.isfinite(a).any():
            src_lo, src_hi = float(a), float(b)
        # avoid div-by-zero
        denom = max(src_hi - src_lo, 1e-12)
        y = (a - src_lo) / denom
        y = y * (hi - lo) + lo
        return PixelImage(y) if isinstance(x, PixelImage) else y

    def _gamma(self, x, g):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x, dtype=np.float32)
        y = np.power(np.clip(a, 0.0, 1.0), float(g))
        return PixelImage(y) if isinstance(x, PixelImage) else y

    def _pow_safe(self, x, p):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x, dtype=np.float32)
        y = np.power(np.clip(a, 1e-8, None), float(p))
        return PixelImage(y) if isinstance(x, PixelImage) else y

    def _absf(self, x):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x, dtype=np.float32)
        y = np.abs(a)
        return PixelImage(y) if isinstance(x, PixelImage) else y

    def _expf(self, x):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x, dtype=np.float32)
        y = np.exp(a)
        return PixelImage(y) if isinstance(x, PixelImage) else y

    def _sqrtf(self, x):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x, dtype=np.float32)
        y = np.sqrt(np.clip(a, 0.0, None))
        return PixelImage(y) if isinstance(x, PixelImage) else y

    def _arcsin(self, x):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x, dtype=np.float32)
        y = np.arcsin(np.clip(a, -1.0, 1.0))
        return PixelImage(y) if isinstance(x, PixelImage) else y


    def _sigmoid(self, x, k=10.0, mid=0.5):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x, dtype=np.float32)
        y = 1.0 / (1.0 + np.exp(-float(k) * (a - float(mid))))
        return PixelImage(y) if isinstance(x, PixelImage) else y

    def _smoothstep(self, e0, e1, x):
        e0, e1 = float(e0), float(e1)
        a = x.array if isinstance(x, PixelImage) else np.asarray(x, dtype=np.float32)
        t = np.clip((a - e0) / max(e1 - e0, 1e-12), 0.0, 1.0)
        y = t * t * (3 - 2 * t)
        return PixelImage(y) if isinstance(x, PixelImage) else y

    def _lerp(self, a, b, t):
        av = a.array if isinstance(a, PixelImage) else np.asarray(a, dtype=np.float32)
        bv = b.array if isinstance(b, PixelImage) else np.asarray(b, dtype=np.float32)
        tv = t.array if isinstance(t, PixelImage) else np.asarray(t, dtype=np.float32)
        y = av * (1.0 - tv) + bv * tv
        return PixelImage(y) if any(isinstance(z, PixelImage) for z in (a, b, t)) else y

    # ---- stats/normalization ----
    def _percentile(self, x, p):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x, dtype=np.float32)
        if a.ndim == 2:
            v = np.percentile(a, float(p))
            out = np.full_like(a, v)
        else:
            out = np.empty_like(a)
            for c in range(a.shape[2]):
                v = np.percentile(a[..., c], float(p))
                out[..., c] = v
        return PixelImage(out) if isinstance(x, PixelImage) else out

    def _normalize01(self, x):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x, dtype=np.float32)
        if a.ndim == 2:
            lo, hi = float(a.min()), float(a.max())
            out = (a - lo) / max(hi - lo, 1e-12)
        else:
            out = np.empty_like(a)
            for c in range(a.shape[2]):
                ch = a[..., c]
                lo, hi = float(ch.min()), float(ch.max())
                out[..., c] = (ch - lo) / max(hi - lo, 1e-12)
        return PixelImage(np.clip(out, 0.0, 1.0)) if isinstance(x, PixelImage) else np.clip(out, 0.0, 1.0)

    def _zscore(self, x):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x, dtype=np.float32)
        if a.ndim == 2:
            m, s = float(a.mean()), float(a.std())
            out = (a - m) / max(s, 1e-12)
        else:
            out = np.empty_like(a)
            for c in range(a.shape[2]):
                ch = a[..., c]
                m, s = float(ch.mean()), float(ch.std())
                out[..., c] = (ch - m) / max(s, 1e-12)
        return PixelImage(out) if isinstance(x, PixelImage) else out

    # ---- channels & color ----
    def _ch(self, x, i):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x, dtype=np.float32)
        if a.ndim != 3: raise ValueError("ch(x,i) expects RGB image")
        return a[..., int(i)]

    def _luma(self, x):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x, dtype=np.float32)
        if a.ndim == 2:
            return a
        y = 0.2126*a[...,0] + 0.7152*a[...,1] + 0.0722*a[...,2]
        return y

    def _compose(self, r, g, b):
        R = r.array if isinstance(r, PixelImage) else np.asarray(r, dtype=np.float32)
        G = g.array if isinstance(g, PixelImage) else np.asarray(g, dtype=np.float32)
        B = b.array if isinstance(b, PixelImage) else np.asarray(b, dtype=np.float32)
        if R.ndim != 2 or G.ndim != 2 or B.ndim != 2:
            raise ValueError("compose(r,g,b) expects three 2-D planes")
        return np.stack([R, G, B], axis=2)

    # ---- mask helpers exposed to the user ----
    def _mask_fn(self):
        ref = _as_rgb(np.asarray(self.doc.image, dtype=np.float32))
        m = _mask_for_ref(self.doc, ref)
        if m is None:
            m = np.zeros(ref.shape[:2], dtype=np.float32)
        if m.ndim == 3:
            m = m[...,0]
        return m

    def _apply_mask_fn(self, base, out, m):
        basev = base.array if isinstance(base, PixelImage) else np.asarray(base, dtype=np.float32)
        outv  = out.array  if isinstance(out,  PixelImage)  else np.asarray(out,  dtype=np.float32)
        mv    = m.array    if isinstance(m,    PixelImage)  else np.asarray(m,    dtype=np.float32)
        return _blend_masked(basev, outv, mv)  # _blend_masked now handles 2-D or 3-D


    # ---- tiny filters (cv2 optional) ----
    def _apply_per_channel(self, a, fn):
        if a.ndim == 2:
            return fn(a)
        out = np.empty_like(a)
        for c in range(a.shape[2]):
            out[..., c] = fn(a[..., c])
        return out

    def _boxblur(self, x, k=3):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x, dtype=np.float32)
        try:
            import cv2
            k = int(max(1, k))
            y = self._apply_per_channel(a, lambda ch: cv2.blur(ch, (k, k)))
        except Exception:
            # naive fallback
            from math import floor
            k = int(max(1, k))
            r = k//2
            y = a.copy()
            # very simple and slow fallback; okay as last resort
            for i in range(a.shape[0]):
                i0, i1 = max(0, i-r), min(a.shape[0], i+r+1)
                for j in range(a.shape[1]):
                    j0, j1 = max(0, j-r), min(a.shape[1], j+r+1)
                    y[i, j] = a[i0:i1, j0:j1].mean(axis=(0,1))
        return PixelImage(y) if isinstance(x, PixelImage) else y

    def _gauss(self, x, sigma=1.0):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x, dtype=np.float32)
        try:
            import cv2
            s = float(sigma)
            k = int(max(1, 2*int(3*s)+1))
            y = self._apply_per_channel(a, lambda ch: cv2.GaussianBlur(ch, (k, k), s))
        except Exception:
            # approximate with box blur passes
            y = self._boxblur(a, k=max(1, int(2*sigma)+1))
            y = y.array if isinstance(y, PixelImage) else y
            y = self._boxblur(PixelImage(y), k=max(1, int(2*sigma)+1))
            y = y.array if isinstance(y, PixelImage) else y
        return PixelImage(y) if isinstance(x, PixelImage) else y

    def _median(self, x, k=3):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x, dtype=np.float32)
        try:
            import cv2
            k = int(max(1, k)) | 1  # must be odd
            y = self._apply_per_channel(a, lambda ch: cv2.medianBlur(ch, k))
        except Exception:
            # crude fallback: percentile in local box
            y = self._boxblur(a, k=k)  # not truly median, but better than nothing
            y = y.array if isinstance(y, PixelImage) else y
        return PixelImage(y) if isinstance(x, PixelImage) else y

    def _unsharp(self, x, sigma=1.5, amount=1.0):
        a = x.array if isinstance(x, PixelImage) else np.asarray(x, dtype=np.float32)
        blur = self._gauss(PixelImage(a), sigma)
        blur = blur.array if isinstance(blur, PixelImage) else blur
        y = np.clip(a + float(amount) * (a - blur), 0.0, 1.0)
        return PixelImage(y) if isinstance(x, PixelImage) else y


    # -------- core eval
    def _eval_multiline(self, expr: str):
        lines = [ln for ln in (expr or "").splitlines() if ln.strip()]
        if not lines:
            return 0
        scope = dict(self.ns)
        for ln in lines[:-1]:
            exec(ln, {"__builtins__": None}, scope)
        return eval(lines[-1], {"__builtins__": None}, scope)

    def eval_single(self, expr: str) -> np.ndarray:
        expr = self._rewrite_names(expr)
        r = self._eval_multiline(expr)
        if isinstance(r, PixelImage):
            r = r.array

        ref = _as_rgb(np.asarray(self.doc.image, dtype=np.float32))
        if np.isscalar(r):
            r = np.full(ref.shape, float(r), dtype=np.float32)
        r = _as_rgb(r.astype(np.float32, copy=False))

        m = _mask_for_ref(self.doc, ref)
        if m is not None:
            r = _blend_masked(ref, r, m)
        return r

    def eval_rgb(self, er: str, eg: str, eb: str, default_channels=(0, 1, 2)) -> np.ndarray:
        er, eg, eb = self._rewrite_names(er), self._rewrite_names(eg), self._rewrite_names(eb)
        ref = _as_rgb(np.asarray(self.doc.image, dtype=np.float32))
        H, W, _ = ref.shape

        def one(e, ci: int):
            if not e:
                return 0
            v = self._eval_multiline(e)
            if isinstance(v, PixelImage):
                v = v.array

            # Scalars become (H,W) plane later, so handle that below
            if np.isscalar(v):
                return np.full((H, W), float(v), dtype=np.float32)

            v = np.asarray(v, dtype=np.float32)

            # NEW: if user returned a color (HxWx3) in per-channel slot, assume the tab's channel
            if v.ndim == 3:
                if v.shape[2] == 1:
                    v = v[..., 0]
                else:
                    # auto-pick requested channel
                    v = v[..., int(ci)]

            # At this point expect 2-D plane
            if v.ndim != 2:
                raise ValueError("Per-channel mode expects a 2-D result (or an RGB where the tab's channel can be taken).")
            return v

        R = one(er, default_channels[0])
        G = one(eg, default_channels[1])
        B = one(eb, default_channels[2])
        out = np.stack([R, G, B], axis=2)

        m = _mask_for_ref(self.doc, ref)
        if m is not None:
            out = _blend_masked(ref, out, m)
        return out

class _PreviewView(QGraphicsView):
    """QGraphicsView with left-drag panning and Ctrl+wheel zoom."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)  # click & drag to pan
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._zoom = 1.0
        self._min_zoom = 0.05
        self._max_zoom = 20.0

    def wheelEvent(self, ev):
        # Ctrl + wheel → zoom; otherwise, default scroll behavior
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            angle = ev.angleDelta().y()
            step = 1.25 if angle > 0 else 1/1.25
            new_zoom = max(self._min_zoom, min(self._zoom * step, self._max_zoom))
            step = new_zoom / self._zoom  # clamp-aware step
            self._zoom = new_zoom
            self.scale(step, step)
            ev.accept()
        else:
            super().wheelEvent(ev)

    # helpers to keep external controls in sync
    def zoom_reset(self):
        self.resetTransform()
        self._zoom = 1.0

    def zoom_by(self, factor: float):
        new_zoom = max(self._min_zoom, min(self._zoom * float(factor), self._max_zoom))
        factor = new_zoom / self._zoom
        self._zoom = new_zoom
        self.scale(factor, factor)


# =============================================================================
# Dialog
# =============================================================================
class PixelMathDialogPro(QDialog):
    """
    Pixel Math with view-name variables.
      • img → active view
      • one variable per OPEN VIEW using the window title (sanitized).
      • Output: Overwrite active OR Create new view
    """
    def __init__(self, parent, doc, icon: QIcon | None = None):
        super().__init__(parent)
        self.setWindowTitle("Pixel Math")
        if icon:
            try:
                self.setWindowIcon(icon)
            except Exception:
                pass

        self.doc = doc
        self.ev = _Evaluator(parent, doc)

        self._load_autostretch_prefs()

        # ──────────────────────────────────────────────────────────────────────────
        # Root split layout: controls (left, scrollable) | preview (right, flexible)
        # ──────────────────────────────────────────────────────────────────────────
        root = QHBoxLayout(self)

        # Left column (controls)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_panel = QWidget()
        left_col = QVBoxLayout(left_panel)
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(8)
        left_scroll.setWidget(left_panel)

        # Right column (preview)
        right_panel = QWidget()
        right_col = QVBoxLayout(right_panel)
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(8)

        # Put them into a splitter so user can drag the boundary
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)   # left: fixed-ish
        splitter.setStretchFactor(1, 1)   # right: grows

        # Give the left side a reasonable minimum so it doesn't disappear
        left_scroll.setMinimumWidth(260)

        root.addWidget(splitter)
        self._splitter = splitter  # optional, if you ever want to tweak later


        # ──────────────────────────────────────────────────────────────────────────
        # Variables mapping (raw title → identifier)
        # ──────────────────────────────────────────────────────────────────────────
        vars_grp = QGroupBox("Variables")
        vars_layout = QVBoxLayout(vars_grp)

        self.vars_list = QListWidget()
        self.vars_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.vars_list.setAlternatingRowColors(True)
        self.vars_list.setTextElideMode(Qt.TextElideMode.ElideRight)

        def _shorten_title(raw_title: str, ident: str, max_chars: int = 48) -> tuple[str, str]:
            """
            Return (display_text, full_text).
            display_text is shortened so the left panel doesn't explode.
            full_text is put into the tooltip.
            """
            base = str(raw_title)
            if len(base) > max_chars:
                head = max_chars // 2 - 1
                tail = max_chars - head - 1
                base = base[:head] + "…" + base[-tail:]
            disp = f"{base} → {ident}"
            full = f"{raw_title} → {ident}"
            return disp, full


        # First item = active view
        active_item = QListWidgetItem("img (active)")
        active_item.setData(Qt.ItemDataRole.UserRole, "img")   # ← stash the real name
        active_item.setToolTip("img (active)")
        self.vars_list.addItem(active_item)

        # Other open views
        for raw, ident in self.ev.title_map:
            disp, full = _shorten_title(raw, ident)
            it = QListWidgetItem(disp)
            it.setData(Qt.ItemDataRole.UserRole, ident)   # ← stash the ident
            it.setToolTip(full)
            self.vars_list.addItem(it)

        # Comfortable height; scroll appears as needed
        self.vars_list.setMinimumHeight(120)
        self.vars_list.setMaximumHeight(180)

        hint = QLabel("Tip: double-click to insert the identifier at the cursor")
        hint.setStyleSheet("color: gray; font-size: 11px;")

        vars_layout.addWidget(self.vars_list)
        vars_layout.addWidget(hint)

        def _insert_ident_into_current_editor(item: QListWidgetItem):
            ident = item.data(Qt.ItemDataRole.UserRole) or item.text().split("→", 1)[-1].strip()
            ed = self.ed_single if self.rb_single.isChecked() else (
                self.ed_r if self.tabs.currentIndex()==0 else self.ed_g if self.tabs.currentIndex()==1 else self.ed_b
            )
            ed.setFocus()
            ed.insertPlainText(str(ident))

        self._on_var_dblclick = _insert_ident_into_current_editor
        try:
            self.vars_list.itemDoubleClicked.connect(self._on_var_dblclick, Qt.ConnectionType.UniqueConnection)
        except TypeError:
            # Already connected; ignore
            pass

        left_col.addWidget(vars_grp)

        # ──────────────────────────────────────────────────────────────────────────
        # Output group
        # ──────────────────────────────────────────────────────────────────────────
        out_grp = QGroupBox("Output")
        out_row = QHBoxLayout(out_grp)
        self.rb_out_overwrite = QRadioButton("Overwrite active"); self.rb_out_overwrite.setChecked(True)
        self.rb_out_new       = QRadioButton("Create new view")
        out_row.addWidget(self.rb_out_overwrite)
        out_row.addWidget(self.rb_out_new)
        out_row.addStretch(1)
        left_col.addWidget(out_grp)

        # ──────────────────────────────────────────────────────────────────────────
        # Mode (single expression vs per-channel)
        # ──────────────────────────────────────────────────────────────────────────
        mode_row = QHBoxLayout()
        self.rb_single = QRadioButton("Single Expression"); self.rb_single.setChecked(True)
        self.rb_sep    = QRadioButton("Separate (R / G / B)")
        mode_row.addWidget(self.rb_single)
        mode_row.addWidget(self.rb_sep)
        mode_row.addStretch(1)
        left_col.addLayout(mode_row)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.rb_single)
        self.mode_group.addButton(self.rb_sep)

        # Editors
        self.ed_single = QPlainTextEdit()
        self.ed_single.setPlaceholderText("e.g. (img + otherView) / 2")
        left_col.addWidget(self.ed_single)

        self.tabs = QTabWidget(); self.tabs.setVisible(False)
        self.ed_r, self.ed_g, self.ed_b = QPlainTextEdit(), QPlainTextEdit(), QPlainTextEdit()
        for ed, name in ((self.ed_r, "Red"), (self.ed_g, "Green"), (self.ed_b, "Blue")):
            w = QWidget(); lay = QVBoxLayout(w); lay.addWidget(ed); self.tabs.addTab(w, name)
        left_col.addWidget(self.tabs)

        self.rb_single.toggled.connect(lambda on: self._mode(on))

        glossary_btn = QPushButton("Glossary…")
        glossary_btn.clicked.connect(self._open_glossary)
        left_col.addWidget(glossary_btn)

        # ──────────────────────────────────────────────────────────────────────────
        # Preview (right side)
        # ──────────────────────────────────────────────────────────────────────────
        preview_grp = QGroupBox("Preview")
        pv_lay = QVBoxLayout(preview_grp)

        # Toolbar
        tb = QHBoxLayout()
        self.btn_preview  = QPushButton("Preview")
        self.btn_preview.setToolTip("Compute Pixel Math and show the result here without committing.")

        # NEW: Auto-stretch toggle with a dropdown menu
        self.btn_autostretch = QToolButton()
        self.btn_autostretch.setText("Auto-stretch")
        self.btn_autostretch.setCheckable(True)
        self.btn_autostretch.setChecked(self._as_enabled)
        self.btn_autostretch.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)

        as_menu = QMenu(self)
        act_toggle = as_menu.addAction("Enable Auto-stretch")
        act_toggle.setCheckable(True); act_toggle.setChecked(self._as_enabled)
        as_menu.addSeparator()

        # Target presets
        tgt_menu = as_menu.addMenu("Target median")
        self._tgt_group = QActionGroup(self); self._tgt_group.setExclusive(True)
        for label, val in (("0.18 (soft)", 0.18), ("0.25 (default)", 0.25), ("0.35 (brighter)", 0.35)):
            a = tgt_menu.addAction(label)
            a.setCheckable(True)
            a.setChecked(abs(self._as_target - val) < 1e-6)
            self._tgt_group.addAction(a)
            a.triggered.connect(lambda _=False, v=val: self._set_as_target(v))

        # Sigma presets
        sig_menu = as_menu.addMenu("Black-point sigma")
        self._sig_group = QActionGroup(self); self._sig_group.setExclusive(True)
        for label, val in (("σ=2.5", 2.5), ("σ=3 (default)", 3.0), ("σ=4 (deeper black)", 4.0)):
            a = sig_menu.addAction(label)
            a.setCheckable(True)
            a.setChecked(abs(self._as_sigma - val) < 1e-6)
            self._sig_group.addAction(a)
            a.triggered.connect(lambda _=False, v=val: self._set_as_sigma(v))

        # Linked channels
        act_linked = as_menu.addAction("Linked channels (use luminance)")
        act_linked.setCheckable(True); act_linked.setChecked(self._as_linked)
        as_menu.addSeparator()

        # Output precision
        act_16 = as_menu.addAction("Use 16-bit stats")
        act_16.setCheckable(True); act_16.setChecked(self._as_16bit)

        self.btn_autostretch.setMenu(as_menu)

        # Keep toggle and menu in sync
        def _apply_menu_state():
            self._as_enabled = self.btn_autostretch.isChecked()
            act_toggle.setChecked(self._as_enabled)
            self._save_autostretch_prefs()
            self._rerun_preview_if_any()

        self.btn_autostretch.toggled.connect(_apply_menu_state)

        act_toggle.toggled.connect(lambda on: (self.btn_autostretch.setChecked(on),
                                            self._save_autostretch_prefs(),
                                            self._rerun_preview_if_any()))

        act_linked.toggled.connect(lambda on: (setattr(self, "_as_linked", on),
                                            self._save_autostretch_prefs(),
                                            self._rerun_preview_if_any()))

        act_16.toggled.connect(   lambda on: (setattr(self, "_as_16bit",  on),
                                            self._save_autostretch_prefs(),
                                            self._rerun_preview_if_any()))

        # tiny helpers for radio menus
        def _refresh_target_checks():
            for a in self._tgt_group.actions():
                txt = a.text()  # "0.18 (soft)" / "0.25 (default)" / "0.35 (brighter)"
                v = 0.18 if txt.startswith("0.18") else 0.25 if txt.startswith("0.25") else 0.35
                a.setChecked(abs(self._as_target - v) < 1e-6)

        def _refresh_sigma_checks():
            for a in self._sig_group.actions():
                # texts are "σ=2.5", "σ=3 (default)", "σ=4 (deeper black)"
                tail = a.text().split("=", 1)[-1].strip().split()[0]  # "2.5" / "3" / "4"
                v = float(tail)
                a.setChecked(abs(self._as_sigma - v) < 1e-6)

        # API used by the actions above
        def _set_target(v: float):
            self._as_target = float(v)
            self._save_autostretch_prefs()
            _refresh_target_checks()
            self._rerun_preview_if_any()

        def _set_sigma(v: float):
            self._as_sigma = float(v)
            self._save_autostretch_prefs()
            _refresh_sigma_checks()
            self._rerun_preview_if_any()

        self._set_as_target = _set_target
        self._set_as_sigma  = _set_sigma

        # existing zoom buttons
        self.btn_zoom_in  = themed_toolbtn("zoom-in", "Zoom In")
        self.btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        self.btn_zoom_1_1 = themed_toolbtn("zoom-original", "1:1")
        self.btn_fit      = themed_toolbtn("zoom-fit-best", "Fit to Preview")

        tb.addWidget(self.btn_preview)
        tb.addWidget(self.btn_autostretch)          # ← NEW
        tb.addSpacing(12)
        tb.addWidget(self.btn_zoom_in); tb.addWidget(self.btn_zoom_out)
        tb.addWidget(self.btn_zoom_1_1); tb.addWidget(self.btn_fit)
        tb.addStretch(1)
        pv_lay.addLayout(tb)

        # Graphics view
        self.preview_view = _PreviewView()  # <-- was QGraphicsView()
        self.preview_view.setRenderHints(self.preview_view.renderHints())
        self.preview_scene = QGraphicsScene(self.preview_view)
        self.preview_view.setScene(self.preview_scene)
        self._preview_item: QGraphicsPixmapItem | None = None
        self._preview_zoom = 1.0  # keep if you like; the view tracks its own zoom too
        pv_lay.addWidget(self.preview_view, 1)

        right_col.addWidget(preview_grp, 1)

        # Wire up preview actions
        self.btn_preview.clicked.connect(self._do_preview)
        self.btn_zoom_in.clicked.connect(lambda: self._zoom_by(1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom_by(1/1.25))
        self.btn_zoom_1_1.clicked.connect(self._zoom_reset_1_1)
        self.btn_fit.clicked.connect(self._fit_to_view)

        # ──────────────────────────────────────────────────────────────────────────
        # Examples (insertable templates)
        # ──────────────────────────────────────────────────────────────────────────
        ex_row = QHBoxLayout()
        ex_row.addWidget(QLabel("Examples:"))
        self.cb_examples = QComboBox()
        self.cb_examples.addItem("Insert example…")
        for title, kind, payload in self._examples_list():
            self.cb_examples.addItem(title, (kind, payload))
        self.cb_examples.currentIndexChanged.connect(self._apply_example_from_combo)
        ex_row.addWidget(self.cb_examples, 1)
        left_col.addLayout(ex_row)

        # ──────────────────────────────────────────────────────────────────────────
        # Favorites
        # ──────────────────────────────────────────────────────────────────────────
        fav_row = QHBoxLayout()
        self.cb_fav = QComboBox(); self.cb_fav.addItem("Select a favorite expression")
        self._load_favorites()
        self.cb_fav.currentTextChanged.connect(self._pick_favorite)

        b_save = QPushButton("Save as Favorite")
        b_del  = QPushButton("Delete Favorite")

        b_save.clicked.connect(self._save_favorite)
        b_del.clicked.connect(self._delete_favorite)

        fav_row.addWidget(self.cb_fav, 1)
        fav_row.addWidget(b_save)
        fav_row.addWidget(b_del)
        left_col.addLayout(fav_row)

        def _fav_context_menu(point):
            if self.cb_fav.currentIndex() <= 0:
                return
            menu = QMenu(self)
            act_del = menu.addAction("Delete this favorite")
            act = menu.exec(self.cb_fav.mapToGlobal(point))
            if act == act_del:
                self._delete_favorite()

        self.cb_fav.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.cb_fav.customContextMenuRequested.connect(_fav_context_menu)

        # ──────────────────────────────────────────────────────────────────────────
        # Buttons + Help (left)
        # ──────────────────────────────────────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self._apply)
        btns.rejected.connect(self.reject)
        b_help = btns.addButton("Help", QDialogButtonBox.ButtonRole.HelpRole)
        b_help.clicked.connect(self._help)
        left_col.addWidget(btns)

        # Output group selection model
        self.out_group = QButtonGroup(self)
        self.out_group.setExclusive(True)
        self.out_group.addButton(self.rb_out_overwrite)
        self.out_group.addButton(self.rb_out_new)

        # Initialize editor visibility
        QTimer.singleShot(0, lambda: self._mode(self.rb_single.isChecked()))

        # A little wider to favor the preview
        self.resize(940, 700)
        QTimer.singleShot(0, lambda: self._splitter.setSizes([320, max(620, self.width() - 320)]))

    # ─────────────── Auto-stretch prefs ───────────────
    def _as_settings(self):
        p = self.parent()
        return getattr(p, "settings", None)

    def _load_autostretch_prefs(self):
        s = self._as_settings()
        self._as_enabled = False
        self._as_linked  = True
        self._as_target  = 0.25
        self._as_sigma   = 3.0
        self._as_16bit   = True
        if s:
            self._as_enabled = s.value("pixelmath/preview_autostretch", False, type=bool)
            self._as_linked  = s.value("pixelmath/preview_as_linked",  True,  type=bool)
            self._as_target  = s.value("pixelmath/preview_as_target",  0.25,  type=float)
            self._as_sigma   = s.value("pixelmath/preview_as_sigma",   3.0,   type=float)
            self._as_16bit   = s.value("pixelmath/preview_as_16bit",   True,  type=bool)

    def _save_autostretch_prefs(self):
        s = self._as_settings()
        if not s: return
        s.setValue("pixelmath/preview_autostretch", self._as_enabled)
        s.setValue("pixelmath/preview_as_linked",   self._as_linked)
        s.setValue("pixelmath/preview_as_target",   float(self._as_target))
        s.setValue("pixelmath/preview_as_sigma",    float(self._as_sigma))
        s.setValue("pixelmath/preview_as_16bit",    self._as_16bit)

    def _rerun_preview_if_any(self):
        # If we already showed something, recompute so user sees the change immediately
        if self._preview_item is not None:
            QTimer.singleShot(0, self._do_preview)


    # ---------- Preview helpers ------------------------------------------------
    def _do_preview(self):
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            if self.rb_single.isChecked():
                out = self.ev.eval_single(self.ed_single.toPlainText().strip())
            else:
                out = self.ev.eval_rgb(
                    self.ed_r.toPlainText().strip(),
                    self.ed_g.toPlainText().strip(),
                    self.ed_b.toPlainText().strip(),
                    default_channels=(0, 1, 2)
                )
            out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

            # NEW: optional auto-stretch (preview only)
            if getattr(self, "_as_enabled", False):
                out = autostretch(
                    out,
                    target_median=float(getattr(self, "_as_target", 0.25)),
                    linked=bool(getattr(self, "_as_linked", True)),
                    sigma=float(getattr(self, "_as_sigma", 3.0)),
                    use_16bit=bool(getattr(self, "_as_16bit", True)),
                )

            self._set_preview_image(out)
            if self._preview_item is not None and abs(self._preview_zoom - 1.0) < 1e-6:
                self._fit_to_view()
        except Exception as e:
            msg = str(e)
            if "name '" in msg and "' is not defined" in msg:
                msg += "\n\nTip: use the identifier listed in Variables (or the raw title; it’s auto-mapped)."
            QMessageBox.critical(self, "Pixel Math Preview", f"Failed:\n{msg}")
        finally:
            QApplication.restoreOverrideCursor()

    def _set_preview_image(self, img: np.ndarray):
        """Render numpy float image into the preview scene, preserving zoom/pan."""
        qim = _float_to_qimage_rgb8(img)
        pm  = QPixmap.fromImage(qim)

        # --- capture current view state (before we clear/replace) ---
        view = self.preview_view
        old_transform = QTransform(view.transform())
        old_zoom = getattr(view, "_zoom", 1.0)

        old_center_norm = None
        if self._preview_item is not None:
            # current center in scene coords → normalize to old image size
            center_scene = view.mapToScene(view.viewport().rect().center())
            old_pix = self._preview_item.pixmap()
            ow, oh = float(old_pix.width()), float(old_pix.height())
            if ow > 0 and oh > 0:
                old_center_norm = QPointF(center_scene.x() / ow, center_scene.y() / oh)

        # --- replace scene content ---
        self.preview_scene.clear()
        self._preview_item = self.preview_scene.addPixmap(pm)
        self.preview_scene.setSceneRect(self._preview_item.boundingRect())

        # --- restore transform and center ---
        # (don’t call zoom_reset / fit here—respect user’s current view)
        view.setTransform(old_transform)
        view._zoom = float(old_zoom)
        self._preview_zoom = float(old_zoom)

        if old_center_norm is not None:
            nw, nh = float(pm.width()), float(pm.height())
            new_center = QPointF(old_center_norm.x() * nw, old_center_norm.y() * nh)
            view.centerOn(new_center)

        self.preview_view.viewport().update()    

    def _zoom_by(self, factor: float):
        if self._preview_item is None:
            return
        self.preview_view.zoom_by(float(factor))
        # mirror into our logical tracker (optional)
        self._preview_zoom = self.preview_view._zoom

    def _zoom_reset_1_1(self):
        if self._preview_item is None:
            return
        self.preview_view.zoom_reset()
        self._preview_zoom = 1.0

    def _fit_to_view(self):
        if self._preview_item is None:
            return
        # Fit the item, then record logical zoom as 1.0 (we treat "fit" as baseline)
        self.preview_view.fitInView(self._preview_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.preview_view._zoom = 1.0
        self._preview_zoom = 1.0

    # ---------- examples -------------------------------------------------------
    def _examples_list(self):
        a = "img"
        others = [ident for (_, ident) in self.ev.title_map if ident != a]
        b = others[0] if others else a
        c = others[1] if len(others) > 1 else a

        return [
            # --- existing basics ---
            ("Average two views", "single", f"({a} + {b}) / 2"),
            ("Difference (A - B)", "single", f"{a} - {b}"),
            ("Invert active", "single", f"~{a}"),
            ("Subtract median (bias remove)", "single", f"{a} - med({a})"),
            ("Zero-center by mean", "single", f"{a} - mean({a})"),
            ("Min + Max combine", "single", f"min({a}) + max({a})"),
            ("Log transform", "single", f"log({a} + 1e-6)"),
            ("Midtones transform m=0.25", "single", f"mtf({a}, 0.25)"),
            ("If darker than median → 0 else 1", "single", f"iff({a} < med({a}), 0, 1)"),

            ("Per-channel: swap R↔B", "rgb", (f"{a}[2]", f"{a}[1]", f"{a}[0]")),
            ("Per-channel: avg A & B", "rgb", (f"({a}[0]+{b}[0])/2", f"({a}[1]+{b}[1])/2", f"({a}[2]+{b}[2])/2")),
            ("Per-channel: build RGB from A,B,C", "rgb", (f"{a}[0]", f"{b}[1]", f"{c}[2]")),

            # --- new, single-expression tone/normalization ---
            ("Normalize to 0–1 (per-channel)", "single", f"normalize01({a})"),
            ("Sigmoid contrast (k=12, mid=0.4)", "single", f"sigmoid({a}, k=12, mid=0.4)"),
            ("Gamma 0.6 (brighten midtones)", "single", f"gamma({a}, 0.6)"),
            ("Percentile stretch 0.5–99.5%", "single",
            f"lo = percentile({a}, 0.5)\nhi = percentile({a}, 99.5)\nclamp(({a} - lo) / (hi - lo), 0, 1)"),

            # --- blending & masking ---
            ("Blend A→B by horizontal gradient X", "single", f"t = X\nlerp({a}, {b}, t)"),
            ("Apply active mask to blend A→B", "single", f"m = mask()\napply_mask({a}, {b}, m)"),

            # --- sharpening with mask (multiline) ---
            ("Masked unsharp (luma-based)", "single",
            f"base = {a}\nsh = unsharp({a}, sigma=1.2, amount=0.8)\n"
            f"m = smoothstep(0.10, 0.60, luma({a}))\napply_mask(base, sh, m)"),

            # --- view matching / calibration ---
            ("Match medians of A to B", "single", f"{a} * (med({b}) / med({a}))"),

            # --- small filters ---
            ("Gaussian blur σ=2", "single", f"gauss({a}, sigma=2.0)"),
            ("Median filter k=3", "single", f"median({a}, k=3)"),

            # --- per-channel examples using new helpers ---
            ("Per-channel: luma to all channels", "rgb", (f"luma({a})", f"luma({a})", f"luma({a})")),
            ("Per-channel: A’s R, B’s G, C’s B (normed)", "rgb",
            (f"normalize01({a}[0])", f"normalize01({b}[1])", f"normalize01({c}[2])")),
        ]

    def _function_glossary(self):
        # name -> (signature / template, short description)
        return {
            "clamp": ("clamp(x, lo=0, hi=1)", "Limit values to [lo..hi]."),
            "rescale": ("rescale(x, a, b, lo=0, hi=1)", "Map range [a..b] to [lo..hi]."),
            "gamma": ("gamma(x, g)", "Apply gamma curve."),
            "pow_safe": ("pow_safe(x, p)", "Power with EPS floor."),
            "absf": ("absf(x)", "Absolute value."),
            "expf": ("expf(x)", "Exponential."),
            "sqrtf": ("sqrtf(x)", "Square root (clamped to ≥0)."),
            "arcsin": ("arcsin(x)", "Inverse sine (radians), input clipped to [-1,1]."),
            "sigmoid": ("sigmoid(x, k=10, mid=0.5)", "S-shaped tone curve."),
            "smoothstep": ("smoothstep(e0, e1, x)", "Cubic smooth ramp."),
            "lerp/mix": ("lerp(a, b, t)", "Linear blend."),
            "percentile": ("percentile(x, p)", "Per-channel percentile image."),
            "normalize01": ("normalize01(x)", "Per-channel [0..1] normalization."),
            "zscore": ("zscore(x)", "Per-channel (x-mean)/std."),
            "ch": ("ch(x, i)", "Extract channel i (0/1/2) as 2-D."),
            "luma": ("luma(x)", "Rec.709 luminance as 2-D."),
            "compose": ("compose(R, G, B)", "Stack three planes to RGB."),
            "mask": ("m = mask()", "Active mask (2-D, [0..1])."),
            "apply_mask": ("apply_mask(base, out, m)", "Blend by mask."),
            "boxblur": ("boxblur(x, k=3)", "Box blur (cv2 if available)."),
            "gauss": ("gauss(x, sigma=1.0)", "Gaussian blur."),
            "median": ("median(x, k=3)", "Median filter (cv2 if avail)."),
            "unsharp": ("unsharp(x, sigma=1.5, amount=1.0)", "Unsharp mask."),
            "mtf": ("mtf(x, m)", "Midtones transfer (existing)."),
            "iff": ("iff(cond, a, b)", "Conditional (existing)."),
            "X / Y": ("X, Y", "Normalized coordinates in [0..1]."),
            "H/W/C": ("H, W, C, shape", "Image dimensions."),
        }

    def _open_glossary(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Pixel Math Glossary")
        lay = QVBoxLayout(dlg)

        info = QLabel("Double-click to insert a template at the cursor.")
        info.setStyleSheet("color: gray;")
        lay.addWidget(info)

        from PyQt6.QtWidgets import QLineEdit, QListWidget, QListWidgetItem, QHBoxLayout, QPushButton
        search = QLineEdit()
        search.setPlaceholderText("Search…")
        lay.addWidget(search)

        lst = QListWidget()
        lst.setMinimumHeight(220)
        lay.addWidget(lst, 1)

        # fill
        gl = self._function_glossary()
        def _refill():
            q = search.text().strip().lower()
            lst.clear()
            for name, (sig, desc) in gl.items():
                if not q or q in name.lower() or q in sig.lower() or q in desc.lower():
                    item = QListWidgetItem(f"{sig} — {desc}")
                    item.setData(Qt.ItemDataRole.UserRole, sig)
                    lst.addItem(item)
        _refill()

        def _insert_current():
            item = lst.currentItem()
            if not item: return
            sig = item.data(Qt.ItemDataRole.UserRole) or ""
            ed = self.ed_single if self.rb_single.isChecked() else (self.ed_r if self.tabs.currentIndex()==0 else self.ed_g if self.tabs.currentIndex()==1 else self.ed_b)
            ed.insertPlainText(sig)

        lst.itemDoubleClicked.connect(lambda *_: (_insert_current(), None))
        search.textChanged.connect(lambda *_: _refill())

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        insert_btn = QPushButton("Insert")
        btns.addButton(insert_btn, QDialogButtonBox.ButtonRole.ApplyRole)
        insert_btn.clicked.connect(_insert_current)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        dlg.resize(620, 400)
        dlg.exec()


    def _delete_favorite(self):
        text = self.cb_fav.currentText()
        if text == "Select a favorite expression":
            return
        # Remove from in-memory list
        try:
            idx_in_list = self._favs.index(text)
        except ValueError:
            return

        self._favs.pop(idx_in_list)

        # Rebuild combo to keep indices clean
        self.cb_fav.blockSignals(True)
        self.cb_fav.clear()
        self.cb_fav.addItem("Select a favorite expression")
        for f in self._favs:
            self.cb_fav.addItem(f)
        self.cb_fav.setCurrentIndex(0)
        self.cb_fav.blockSignals(False)

        # Persist
        s = self._settings()
        if s:
            s.setValue("pixelmath_favorites", json.dumps(self._favs))


    def _apply_example_from_combo(self, idx: int):
        if idx <= 0:  # "Insert example…"
            return
        kind, payload = self.cb_examples.currentData()
        # Switch mode first, then inject text on the next event loop tick to avoid any race with toggled()
        if kind == "single":
            self.rb_single.setChecked(True)
            def set_text():
                self._mode(True)
                self.ed_single.setPlainText(str(payload))
            QTimer.singleShot(0, set_text)
        else:
            self.rb_sep.setChecked(True)
            def set_text_rgb():
                self._mode(False)
                r, g, b = payload
                self.ed_r.setPlainText(r)
                self.ed_g.setPlainText(g)
                self.ed_b.setPlainText(b)
            QTimer.singleShot(0, set_text_rgb)
        # reset the combo back to the prompt so it can be used repeatedly
        QTimer.singleShot(0, lambda: self.cb_examples.setCurrentIndex(0))

    # ---------- favorites ------------------------------------------------------
    def _settings(self):
        p = self.parent(); return getattr(p, "settings", None)

    def _load_favorites(self):
        self._favs = []
        s = self._settings()
        if s:
            raw = s.value("pixelmath_favorites", "", type=str) or ""
            try: self._favs = json.loads(raw) if raw else []
            except Exception: self._favs = []
        for f in self._favs: self.cb_fav.addItem(f)

    def _save_favorite(self):
        if self.rb_single.isChecked():
            expr = self.ed_single.toPlainText().strip()
        else:
            expr = f"[R]{self.ed_r.toPlainText().strip()} | [G]{self.ed_g.toPlainText().strip()} | [B]{self.ed_b.toPlainText().strip()}"
        if not expr or expr in self._favs: return
        self._favs.append(expr); self.cb_fav.addItem(expr)
        s = self._settings()
        if s: s.setValue("pixelmath_favorites", json.dumps(self._favs))

    def _pick_favorite(self, text):
        if text == "Select a favorite expression": return
        if "[R]" in text or "[G]" in text or "[B]" in text:
            self.rb_sep.setChecked(True); self._mode(False)
            parts = {}
            for p in [t.strip() for t in text.split("|") if t.strip()]:
                parts[p[:3]] = p[3:].strip()
            self.ed_r.setPlainText(parts.get("[R]", "")); self.ed_g.setPlainText(parts.get("[G]", "")); self.ed_b.setPlainText(parts.get("[B]", ""))
        else:
            self.rb_single.setChecked(True); self._mode(True)
            self.ed_single.setPlainText(text)

    # =============================================================================
    # New-view delivery helper (used by PixelMathDialogPro)
    # =============================================================================

    @staticmethod
    def _deliver_new_view(parent, src_doc, img: np.ndarray, step_name: str = "Pixel Math"):
        dm = getattr(parent, "doc_manager", None)
        if dm is None:
            if hasattr(src_doc, "set_image"):
                src_doc.set_image(img, step_name=step_name)
            else:
                src_doc.image = img
            return src_doc

        base = src_doc.display_name() if callable(getattr(src_doc, "display_name", None)) else getattr(src_doc, "display_name", "Untitled")
        base = base if isinstance(base, str) and base else "Untitled"
        new_title = f"{base} — {step_name}"

        meta = dict(getattr(src_doc, "metadata", {}) or {})
        meta["step_name"] = step_name

        new_doc = dm.open_array(np.asarray(img, dtype=np.float32), metadata=meta, title=new_title)
        if hasattr(parent, "_spawn_subwindow_for"):
            parent._spawn_subwindow_for(new_doc)
        return new_doc


    # ---------- UI helpers -----------------------------------------------------
    def _mode(self, single_on: bool):
        self.ed_single.setVisible(single_on)
        self.tabs.setVisible(not single_on)

    def _help(self):
        gl = self._function_glossary()
        lines = [
            "Operators: +  -  *  /   ^(power)   ~(invert)",
            "Comparisons: <, ==   (use inside iff)",
            "",
            "Variables:",
            "  • img (active) and one per open view (by window title, auto-mapped).",
            "  • Coordinates: X, Y in [0..1].",
            "  • Sizes: H, W, C, shape.",
            "",
            "Per-channel indexing: view[0], view[1], view[2].",
            "Multiline: last line is the result.",
            "Output: Overwrite active or Create new view.",
            "",
            "Functions:"
        ]
        # Pretty column-ish dump
        for name, (sig, desc) in gl.items():
            lines.append(f"  {sig}\n    {desc}")
        QMessageBox.information(self, "Pixel Math Help", "\n".join(lines))

    # ---------- Apply ----------------------------------------------------------
    # ---------- Apply ----------------------------------------------------------
    def _apply(self):
        try:
            # Capture expressions first so we can store them for replay
            if self.rb_single.isChecked():
                mode   = "single"
                expr   = self.ed_single.toPlainText().strip()
                expr_r = ""
                expr_g = ""
                expr_b = ""
                out = self.ev.eval_single(expr)
            else:
                mode   = "rgb"
                expr   = ""
                expr_r = self.ed_r.toPlainText().strip()
                expr_g = self.ed_g.toPlainText().strip()
                expr_b = self.ed_b.toPlainText().strip()
                out = self.ev.eval_rgb(
                    expr_r,
                    expr_g,
                    expr_b,
                    default_channels=(0, 1, 2)
                )

            out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

            # Output route
            if self.rb_out_new.isChecked():
                self._deliver_new_view(self.parent(), self.doc, out, "Pixel Math")
            else:
                if hasattr(self.doc, "set_image"):
                    self.doc.set_image(out, step_name="Pixel Math")
                elif hasattr(self.doc, "apply_numpy"):
                    self.doc.apply_numpy(out, step_name="Pixel Math")
                else:
                    self.doc.image = out

            # ── Register as last_headless_command for replay ──────────
            try:
                main = self.parent()
                if main is not None:
                    preset = {
                        "mode": mode,
                        "expr": expr,
                        "expr_r": expr_r,
                        "expr_g": expr_g,
                        "expr_b": expr_b,
                    }
                    payload = {
                        "command_id": "pixel_math",
                        "preset": dict(preset),
                    }
                    setattr(main, "_last_headless_command", payload)

                    # optional log
                    try:
                        if hasattr(main, "_log"):
                            if mode == "single" and expr:
                                desc = expr
                            else:
                                desc = f"R:{expr_r} G:{expr_g} B:{expr_b}"
                            main._log(f"[Replay] Registered Pixel Math as last action → {desc}")
                    except Exception:
                        pass
            except Exception:
                # don't break apply if replay wiring fails
                pass
            # ───────────────────────────────────────────────────────────

            self.accept()
        except Exception as e:
            msg = str(e)
            if "name '" in msg and "' is not defined" in msg:
                msg += "\n\nTip: use the identifier shown beside Variables (e.g. 'andromeda_png'), "
                msg += "or just type the raw title; it will be auto-mapped."
            QMessageBox.critical(self, "Pixel Math", f"Failed:\n{msg}")
