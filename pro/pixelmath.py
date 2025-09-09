# pro/pixelmath.py
from __future__ import annotations
import os, re, json
import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QCursor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout, QLabel,
    QPushButton, QPlainTextEdit, QComboBox, QDialogButtonBox, QRadioButton,
    QTabWidget, QWidget, QMessageBox, QMenu
)

# ---- Optional accelerators from legacy.numba_utils -------------------------
try:
    from legacy.numba_utils import fast_mad as _fast_mad
except Exception:
    _fast_mad = None

# =============================================================================
# PixelImage wrapper (vector ops, indexing, ^ as exponent, ~ as invert)
# =============================================================================
class PixelImage:
    def __init__(self, array: np.ndarray):
        self.array = array

    def __getitem__(self, ch):
        a = self.array
        if a.ndim < 3:
            raise ValueError("This image has no channel dimension to index.")
        if not (0 <= ch < a.shape[2]):
            raise IndexError(f"Channel index {ch} out of range for shape {a.shape}")
        return PixelImage(a[..., ch])

    def _bin(self, other, op):
        return PixelImage(op(self.array, other.array if isinstance(other, PixelImage) else other))
    __add__ = lambda self, o: self._bin(o, np.add)
    __radd__ = __add__
    __sub__ = lambda self, o: self._bin(o, np.subtract)
    __rsub__ = lambda self, o: PixelImage((o.array if isinstance(o, PixelImage) else o) - self.array)
    __mul__ = lambda self, o: self._bin(o, np.multiply)
    __rmul__ = __mul__
    __truediv__ = lambda self, o: self._bin(o, np.divide)

    def __invert__(self):
        return PixelImage(1.0 - self.array)

    def __xor__(self, other):
        return PixelImage(np.power(self.array, other.array if isinstance(other, PixelImage) else other))
    def __rxor__(self, other):
        return PixelImage(np.power(other.array if isinstance(other, PixelImage) else other, self.array))

    def __lt__(self, other):  return self.array < (other.array if isinstance(other, PixelImage) else other)
    def __eq__(self, other):  return self.array == (other.array if isinstance(other, PixelImage) else other)

    def __repr__(self): return f"PixelImage(shape={self.array.shape}, dtype={self.array.dtype})"


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

def _nearest_resize_2d(m: np.ndarray, H: int, W: int) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.shape == (H, W):
        return m
    try:
        import cv2
        return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32, copy=False)
    except Exception:
        yi = (np.linspace(0, m.shape[0] - 1, H)).astype(np.int32)
        xi = (np.linspace(0, m.shape[1] - 1, W)).astype(np.int32)
        return m[yi][:, xi].astype(np.float32, copy=False)

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
    base = np.asarray(base, dtype=np.float32)
    out  = np.asarray(out,  dtype=np.float32)
    m    = np.clip(np.asarray(m, dtype=np.float32), 0.0, 1.0)
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
        out = ev.eval_rgb(r, g, b)

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
            "med": self._med,
            "mean": self._mean,
            "min": self._min,
            "max": self._max,
            "std": self._std,
            "mad": self._mad,
            "log": self._log,
            "iff": self._iff,
            "mtf": self._mtf,
        }
        cur = np.asarray(self.doc.image, dtype=np.float32)
        self._img_shape = cur.shape
        self.ns["img"] = PixelImage(_as_rgb(cur))

        # map: raw title -> identifier (stored for rewriting + UI)
        self.title_map: list[tuple[str, str]] = []
        open_docs = []
        if hasattr(self.parent, "_subwindow_docs"):
            open_docs = list(self.parent._subwindow_docs())  # [(title, doc), ...]
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
            self.ns[ident] = PixelImage(_as_rgb(arr))
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

    def eval_rgb(self, er: str, eg: str, eb: str) -> np.ndarray:
        er, eg, eb = self._rewrite_names(er), self._rewrite_names(eg), self._rewrite_names(eb)
        ref = _as_rgb(np.asarray(self.doc.image, dtype=np.float32))
        H, W, _ = ref.shape

        def one(e):
            if not e:
                return 0
            v = self._eval_multiline(e)
            if isinstance(v, PixelImage):
                v = v.array
            if np.isscalar(v):
                return np.full((H, W), float(v), dtype=np.float32)
            if v.ndim == 3:
                raise ValueError("Per-channel mode expects 2D results (use viewName[0/1/2]).")
            return v.astype(np.float32, copy=False)

        R = one(er); G = one(eg); B = one(eb)
        out = np.stack([R, G, B], axis=2)

        m = _mask_for_ref(self.doc, ref)
        if m is not None:
            out = _blend_masked(ref, out, m)
        return out

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
            try: self.setWindowIcon(icon)
            except Exception: pass

        self.doc = doc
        self.ev = _Evaluator(parent, doc)

        v = QVBoxLayout(self)

        # Variables mapping (raw title → identifier) so the user can copy/paste
        map_lines = ["img (active)"] + [f"{raw} → {ident}" for raw, ident in self.ev.title_map]
        lbl = QLabel("Variables:\n  " + ",  ".join(map_lines))
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(lbl)

        # ----- Output group (very visible) ------------------------------------
        out_grp = QGroupBox("Output")
        out_row = QHBoxLayout(out_grp)
        self.rb_out_overwrite = QRadioButton("Overwrite active"); self.rb_out_overwrite.setChecked(True)
        self.rb_out_new       = QRadioButton("Create new view")
        out_row.addWidget(self.rb_out_overwrite)
        out_row.addWidget(self.rb_out_new)
        out_row.addStretch(1)
        v.addWidget(out_grp)

        # ----- Mode group ------------------------------------------------------
        mode_row = QHBoxLayout()
        self.rb_single = QRadioButton("Single Expression"); self.rb_single.setChecked(True)
        self.rb_sep    = QRadioButton("Separate (R / G / B)")
        mode_row.addWidget(self.rb_single); mode_row.addWidget(self.rb_sep); mode_row.addStretch(1)
        v.addLayout(mode_row)

        # Editors
        self.ed_single = QPlainTextEdit()
        self.ed_single.setPlaceholderText("e.g. (img + otherView) / 2")
        v.addWidget(self.ed_single)

        self.tabs = QTabWidget(); self.tabs.setVisible(False)
        self.ed_r, self.ed_g, self.ed_b = QPlainTextEdit(), QPlainTextEdit(), QPlainTextEdit()
        for ed, name in ((self.ed_r,"Red"), (self.ed_g,"Green"), (self.ed_b,"Blue")):
            w = QWidget(); lay = QVBoxLayout(w); lay.addWidget(ed); self.tabs.addTab(w, name)
        v.addWidget(self.tabs)

        self.rb_single.toggled.connect(lambda on: self._mode(on))

        # ----- Examples (SAS-style list you can drop down and insert) ----------
        ex_row = QHBoxLayout()
        ex_row.addWidget(QLabel("Examples:"))
        self.cb_examples = QComboBox()
        self.cb_examples.addItem("Insert example…")
        for title, kind, payload in self._examples_list():
            # store (kind, payload) as userData for easy retrieval
            self.cb_examples.addItem(title, (kind, payload))
        self.cb_examples.currentIndexChanged.connect(self._apply_example_from_combo)
        ex_row.addWidget(self.cb_examples, 1)
        v.addLayout(ex_row)

        # Favorites
        fav_row = QHBoxLayout()
        self.cb_fav = QComboBox(); self.cb_fav.addItem("Select a favorite expression")
        self._load_favorites()
        self.cb_fav.currentTextChanged.connect(self._pick_favorite)
        b_save = QPushButton("Save as Favorite"); b_save.clicked.connect(self._save_favorite)
        fav_row.addWidget(self.cb_fav); fav_row.addWidget(b_save)
        v.addLayout(fav_row)

        # Buttons + Help
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self._apply); btns.rejected.connect(self.reject)
        b_help = btns.addButton("Help", QDialogButtonBox.ButtonRole.HelpRole); b_help.clicked.connect(self._help)
        v.addWidget(btns)

        self.resize(860, 580)

    # ---------- examples -------------------------------------------------------
    def _examples_list(self):
        a = "img"
        others = [ident for (_, ident) in self.ev.title_map if ident != a]
        b = others[0] if others else a
        c = others[1] if len(others) > 1 else a
        return [
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
        ]

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
        QMessageBox.information(self, "Pixel Math Help",
            "Operators: + - * /   ^(power)   ~(invert)\n"
            "Comparisons: <, == (use inside iff)\n"
            "Functions: med, mean, min, max, std, mad, log, iff(cond,a,b), mtf(x,m)\n"
            "Variables: 'img' (active) and one per open view (window title).\n"
            "You can also type the raw window title (e.g. 'andromeda.png'); the dialog maps it automatically.\n"
            "Per-channel indexing: viewName[0], viewName[1], viewName[2].\n"
            "Multiline: last line is the result.\n"
            "Output: choose Overwrite active or Create new view."
        )

    # ---------- Apply ----------------------------------------------------------
    def _apply(self):
        try:
            if self.rb_single.isChecked():
                out = self.ev.eval_single(self.ed_single.toPlainText().strip())
            else:
                out = self.ev.eval_rgb(self.ed_r.toPlainText().strip(),
                                       self.ed_g.toPlainText().strip(),
                                       self.ed_b.toPlainText().strip())
            out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

            # Output route
            if self.rb_out_new.isChecked():
                self._deliver_new_view(self.parent(), self.doc, out, "Pixel Math")
            else:
                if hasattr(self.doc, "set_image"): self.doc.set_image(out, step_name="Pixel Math")
                elif hasattr(self.doc, "apply_numpy"): self.doc.apply_numpy(out, step_name="Pixel Math")
                else: self.doc.image = out

            self.accept()
        except Exception as e:
            msg = str(e)
            if "name '" in msg and "' is not defined" in msg:
                msg += "\n\nTip: use the identifier shown beside Variables (e.g. 'andromeda_png'), "
                msg += "or just type the raw title; it will be auto-mapped."
            QMessageBox.critical(self, "Pixel Math", f"Failed:\n{msg}")