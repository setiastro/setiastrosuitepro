#saspro/add_stars.py
from __future__ import annotations
import os
import numpy as np

# Qt
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QSettings, QByteArray, QThread, QCoreApplication
from PyQt6.QtGui import QImage, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QScrollArea, QSizePolicy,
    QComboBox, QSlider, QMessageBox, QFileDialog, QFormLayout
)

# I/O (use your legacy functions)
from setiastro.saspro.legacy.image_manager import load_image
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
from setiastro.saspro.help_support import make_help_button
from setiastro.saspro.thread_keeper import start_owned_thread, thread_running
from setiastro.saspro.widgets.factory_reset import (
    make_reset_button, snapshot_controls, restore_controls)

try:
    import cv2
except Exception:
    cv2 = None


def _tr(text: str) -> str:
    return QCoreApplication.translate("AddStars", text)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers to enumerate docs and masks from the Pro app
# ──────────────────────────────────────────────────────────────────────────────
# REPLACE OLD _iter_open_docs WITH THIS
def _iter_open_docs(main):
    from PyQt6 import sip

    def _label_for(obj, fallback):
        name = ""
        try:
            if hasattr(obj, "display_name") and callable(obj.display_name):
                name = obj.display_name()
            else:
                name = getattr(obj, "name", "") or ""
        except Exception:
            name = ""
        return name or fallback or f"View {len(items)}"

    def _image_from_any(x):
        if x is None:
            return None
        chain = [x, getattr(x, "doc", None), getattr(x, "_doc", None), getattr(x, "document", None)]
        for c in chain:
            if c is None:
                continue
            try:
                if isinstance(c, sip.wrappertype) and sip.isdeleted(c):
                    continue
            except Exception:
                pass
            img = getattr(c, "image", None)
            if img is not None:
                try:
                    a = np.asarray(img)
                    if a is not None and a.size:
                        return a
                except Exception:
                    pass
            for m in ("get_image", "current_image", "image_array"):
                f = getattr(c, m, None)
                if callable(f):
                    try:
                        a = f()
                        a = np.asarray(a) if a is not None else None
                        if a is not None and a.size:
                            return a
                    except Exception:
                        pass
        return None

    def _add_item(obj, label_hint=None):
        try:
            if isinstance(obj, sip.wrappertype) and sip.isdeleted(obj):
                return
        except Exception:
            pass
        img = _image_from_any(obj)
        if img is None:
            return
        key = id(getattr(obj, "image", obj))
        if key in seen:
            return
        seen.add(key)
        items.append((_label_for(obj, label_hint), obj))

    items, seen = [], set()

    # 1) docman sources
    dm = getattr(main, "docman", None)
    if dm is not None:
        for attr in ("documents", "docs", "open_docs", "views"):
            coll = getattr(dm, attr, None)
            if isinstance(coll, dict):
                for d in coll.values():
                    _add_item(d)
            elif isinstance(coll, (list, tuple, set)):
                for d in coll:
                    _add_item(d)
        for meth in ("iter_docs", "all_docs", "iter"):
            fn = getattr(dm, meth, None)
            if callable(fn):
                try:
                    for d in fn():
                        _add_item(d)
                except Exception:
                    pass

    # 2) any QMdiArea on main
    for attr in dir(main):
        try:
            val = getattr(main, attr)
        except Exception:
            continue

        # Guard: skip deleted Qt objects before touching them at all
        try:
            if isinstance(val, sip.wrappertype) and sip.isdeleted(val):
                continue
        except Exception:
            pass

        try:
            has_swl = hasattr(val, "subWindowList")
        except Exception:
            continue

        if not has_swl:
            continue

        try:
            sw_list = val.subWindowList()
        except Exception:
            continue

        for sw in sw_list:
            try:
                if sip.isdeleted(sw):
                    continue
            except Exception:
                continue

            title = ""
            try:
                title = sw.windowTitle()
            except Exception:
                pass

            w = None
            try:
                w = sw.widget()
            except Exception:
                continue

            if w is None:
                continue

            try:
                if sip.isdeleted(w):
                    continue
            except Exception:
                continue

            for candidate in (
                getattr(w, "doc", None),
                getattr(w, "_doc", None),
                getattr(w, "document", None),
                w,
            ):
                if candidate is None:
                    continue
                try:
                    if sip.isdeleted(candidate):
                        continue
                except Exception:
                    pass
                if _image_from_any(candidate) is not None:
                    _add_item(candidate, label_hint=title)
                    break
    return items

def _doc_image(doc_like) -> np.ndarray | None:
    """
    Accepts a doc or a view widget and returns a float32 image array
    (mono 2D or RGB 3D). No boolean ops on arrays to avoid ambiguity.
    """
    def _grab(x):
        if x is None:
            return None
        # direct attribute
        img = getattr(x, "image", None)
        if img is not None:
            return img
        # method fallbacks
        for m in ("get_image", "current_image", "image_array"):
            fn = getattr(x, m, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    pass
        return None

    img = _grab(doc_like)
    if img is None:
        img = _grab(getattr(doc_like, "doc", None))
    if img is None:
        img = _grab(getattr(doc_like, "_doc", None))
    if img is None:
        img = _grab(getattr(doc_like, "document", None))
    if img is None:
        return None

    a = np.asarray(img).astype(np.float32, copy=False)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    # Defensive normalization for big float ranges
    if a.dtype.kind == "f" and a.size:
        mx = float(a.max())
        if mx > 5.0:
            a = a / mx
    return a




def _active_mask_array_from_doc(doc) -> np.ndarray | None:
    """
    Return active mask (H,W) float32 in [0,1] from the document, if present.
    """
    try:
        mid = getattr(doc, "active_mask_id", None)
        if not mid:
            return None
        masks = getattr(doc, "masks", {}) or {}
        layer = masks.get(mid)
        data = getattr(layer, "data", None) if layer is not None else None
        if data is None:
            return None
        a = np.asarray(data)
        if a.ndim == 3:
            if cv2 is not None:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            else:
                a = a.mean(axis=2)
        a = a.astype(np.float32, copy=False)
        a = np.clip(a, 0.0, 1.0)
        return a
    except Exception:
        return None


def _resize_mask_nearest(m: np.ndarray, h: int, w: int) -> np.ndarray:
    if m.shape != (h, w):
        if cv2 is not None:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            yi = (np.linspace(0, m.shape[0]-1, h)).astype(np.int32)
            xi = (np.linspace(0, m.shape[1]-1, w)).astype(np.int32)
            m = m[yi][:, xi]
    return m


def _blend_add_stars(starless: np.ndarray, stars_only: np.ndarray, mode: str, r: float,
                     m: np.ndarray | None) -> np.ndarray:
    """Starless + stars-only (RGB, [0,1], same shape) by Screen or Add at
    ratio r; where the (H,W) mask m is given, only it gets the stars."""
    if mode == "Screen":
        base = starless + stars_only - (starless * stars_only)
    else:
        base = starless + stars_only

    blended = (1.0 - r) * starless + r * base
    blended = np.clip(blended, 0.0, 1.0)

    if m is not None:
        m3 = m[:, :, None]   # broadcast: the same values as repeating it 3x
        # only replace where mask==1; keep original starless elsewhere
        blended = np.clip(starless * (1.0 - m3) + blended * m3, 0.0, 1.0).astype(np.float32, copy=False)
    return blended


# Every control change restarts this timer; the preview is built once the
# controls have been still that long, on a worker, one at a time.
_PREVIEW_DEBOUNCE_MS = 120


class _AddStarsJob(QThread):
    """Runs fn(**kwargs) off the GUI thread; its dict comes back through
    `ready` ({"error": text} if it raised). No parent: started with
    thread_keeper.start_owned_thread, so the dialog may close mid-run."""
    ready = pyqtSignal(object)

    def __init__(self, fn, kwargs):
        super().__init__(None)
        self._fn = fn
        self._kwargs = kwargs

    def run(self):
        try:
            out = self._fn(**self._kwargs)
        except Exception as e:
            out = {"error": str(e)}
        self._kwargs = None
        self.ready.emit(out)


def _proxy_rgb(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """img (RGB [0,1]) at (w, h): INTER_AREA when shrinking. Preview only."""
    if img.shape[1] == w and img.shape[0] == h:
        return img
    if cv2 is not None:
        return cv2.resize(np.ascontiguousarray(img), (w, h), interpolation=cv2.INTER_AREA)
    yi = np.linspace(0, img.shape[0] - 1, h).astype(np.int32)
    xi = np.linspace(0, img.shape[1] - 1, w).astype(np.int32)
    return img[yi][:, xi]


def _add_stars_preview_job(starless, stars_only, ps, pt, w, h, scale, mode, r, mask):
    """The preview: the Apply blend on proxies at the displayed size (w, h),
    i.e. full size x scale (full size when scale >= 1, then scaled up as
    before). Returns the proxies (for reuse), the blend and the QImage."""
    if ps is None:
        ps = _proxy_rgb(starless, w, h)
    if pt is None:
        pt = _proxy_rgb(stars_only, w, h)
    m = None
    if mask is not None:
        if mask.shape != (h, w) and cv2 is not None:
            m = cv2.resize(np.ascontiguousarray(mask), (w, h), interpolation=cv2.INTER_AREA)
        else:
            m = _resize_mask_nearest(mask, h, w)
    out = _blend_add_stars(ps, pt, mode, r, m)
    u8 = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    q = QImage(u8.data, w, h, u8.strides[0], QImage.Format.Format_RGB888).copy()
    if scale > 1.0:
        q = q.scaled(max(1, round(w*scale)), max(1, round(h*scale)), Qt.AspectRatioMode.KeepAspectRatio,
                     Qt.TransformationMode.SmoothTransformation)
    return {"starless": starless, "stars_only": stars_only, "ps": ps, "pt": pt,
            "w": w, "h": h, "scale": scale, "out": out, "qimg": q}


def _add_stars_full_job(starless, stars_only, mode, r, mask):
    """Apply's full-resolution blend (the old preview math, off the GUI thread)."""
    m = None
    if mask is not None:
        h, w = starless.shape[:2]
        m = _resize_mask_nearest(mask, h, w)
    return {"result": _blend_add_stars(starless, stars_only, mode, r, m)}


# ──────────────────────────────────────────────────────────────────────────────
# Dialog
# ──────────────────────────────────────────────────────────────────────────────
class AddStarsDialog(QDialog):
    stars_added = pyqtSignal(object, object)
    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Add Stars to Image"))

        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        # Non-modal: allow user to switch between images while dialog is open
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        #self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass  # older PyQt6 versions
        self.main = main
        self.starless = None
        self.stars_only = None
        self.blended_image = None   # last preview blend, at display size (_pix_scale)
        self.scale_factor = 1.0
        self._pix = None
        self._pix_scale = None
        self._pv_timer = QTimer(self)    # debounce: restarted on every change
        self._pv_timer.setSingleShot(True)
        self._pv_timer.timeout.connect(self._launch_preview)
        self._pv_thread = None
        self._pv_pending = False         # a change arrived while a preview ran
        self._proxies = {}               # (id(img), w, h) -> (img, proxy)
        self._apply_thread = None
        self._apply_target = None
        self._fit_once = False
        self._settings = QSettings()
        self._persist_prefix = "add_stars"
        self._geom_restored = False
        self._restoring_ui = True   # guards saves while we build/restore/populate
        self._build_ui()
        self._populate_doc_combos()
        # Factory defaults: saved settings only land on first show
        # (_restore_ui_state). The image pickers are left alone.
        self._factory = snapshot_controls(self, exclude=(self.cmb_starless, self.cmb_stars))

    def _k(self, key: str) -> str:
        return f"{self._persist_prefix}/{key}"

    def _reset_to_factory(self):
        """Blend type and ratio back to their defaults (saved on change, as
        any edit is). Not while Apply computes."""
        if thread_running(self._apply_thread):
            return
        restore_controls(self._factory)

    # UI -----------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Preview
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.preview_label.setScaledContents(False)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.preview_label)
        layout.addWidget(self.scroll_area)

        # Zoom row (standardized themed toolbuttons)
        zrow = QHBoxLayout()
        self.btn_zoom_in  = themed_toolbtn("zoom-in",       self.tr("Zoom In"))
        self.btn_zoom_out = themed_toolbtn("zoom-out",      self.tr("Zoom Out"))
        self.btn_fit      = themed_toolbtn("zoom-fit-best", self.tr("Fit to Preview"))

        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_fit.clicked.connect(self.fit_to_preview)

        zrow.addWidget(self.btn_zoom_in)
        zrow.addWidget(self.btn_zoom_out)
        zrow.addWidget(self.btn_fit)
        zrow.addStretch(1)
        layout.addLayout(zrow)

        # Selection + blend
        grid = QGridLayout()

        # Blend type
        grid.addWidget(QLabel(self.tr("Blend Type:")), 0, 0)
        self.cmb_blend = QComboBox(); self.cmb_blend.addItems(["Screen", "Add"])
        self.cmb_blend.currentIndexChanged.connect(self.update_preview)
        grid.addWidget(self.cmb_blend, 0, 1)

        # Starless source
        grid.addWidget(QLabel(self.tr("Starless View:")), 1, 0)
        self.cmb_starless = QComboBox(); grid.addWidget(self.cmb_starless, 1, 1)
        btn_sless_file = QPushButton(self.tr("Load from File")); btn_sless_file.clicked.connect(lambda: self._load_from_file('starless'))
        grid.addWidget(btn_sless_file, 1, 2)

        # Stars-only source
        grid.addWidget(QLabel(self.tr("Stars-Only View:")), 2, 0)
        self.cmb_stars   = QComboBox(); grid.addWidget(self.cmb_stars, 2, 1)
        btn_stars_file = QPushButton(self.tr("Load from File")); btn_stars_file.clicked.connect(lambda: self._load_from_file('stars'))
        grid.addWidget(btn_stars_file, 2, 2)

        layout.addLayout(grid)

        refresh_row = QHBoxLayout()
        btn_refresh = QPushButton(self.tr("Refresh Views"))
        btn_refresh.clicked.connect(self._populate_doc_combos)
        refresh_row.addStretch(1)
        refresh_row.addWidget(btn_refresh)
        layout.addLayout(refresh_row)

        # Ratio slider
        row = QHBoxLayout()
        row.addWidget(QLabel(self.tr("Blend Ratio (Screen/Add Intensity):")))
        self.slider_ratio = QSlider(Qt.Orientation.Horizontal)
        self.slider_ratio.setRange(0, 100); self.slider_ratio.setValue(100)
        self.slider_ratio.setTickInterval(10); self.slider_ratio.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_ratio.valueChanged.connect(self.update_preview)
        row.addWidget(self.slider_ratio)
        layout.addLayout(row)

        # Buttons
        brow = QHBoxLayout()
        brow.addStretch(1)
        btn_apply = QPushButton(self.tr("Apply")); btn_apply.clicked.connect(self._apply)
        self.btn_apply = btn_apply
        self.btn_reset = make_reset_button(self); self.btn_reset.clicked.connect(self._reset_to_factory)
        btn_cancel= QPushButton(self.tr("Cancel")); btn_cancel.clicked.connect(self.reject)
        brow.addWidget(btn_apply); brow.addWidget(self.btn_reset); brow.addWidget(btn_cancel)
        layout.addLayout(brow)

        # Grip + help: their own row at the bottom
        from setiastro.saspro.shortcuts import ShortcutGrip
        self.btn_help = make_help_button("add_stars", self)
        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.addWidget(ShortcutGrip("add_stars", self))
        grip_row.addStretch(1)
        grip_row.addWidget(self.btn_help)
        layout.addLayout(grip_row)

        self.setMinimumSize(900, 650)

        # signals for combos
        self.cmb_starless.currentIndexChanged.connect(self._pick_starless_from_combo)
        self.cmb_stars.currentIndexChanged.connect(self._pick_stars_from_combo)

        def _save():
            self._save_ui_state()

        self.cmb_blend.currentIndexChanged.connect(lambda *_: _save())
        self.slider_ratio.valueChanged.connect(lambda *_: _save())
        self.cmb_starless.currentIndexChanged.connect(lambda *_: _save())
        self.cmb_stars.currentIndexChanged.connect(lambda *_: _save())

    # Populate combos with open docs (+ sentinel for file)
    def _populate_doc_combos(self):
        items = [("Select View", None)]
        for name, d in _iter_open_docs(self.main):
            items.append((name, d))
        items.append(("Load from File", "file"))

        # prevent currentIndexChanged from firing while we rebuild lists
        self.cmb_starless.blockSignals(True)
        self.cmb_stars.blockSignals(True)
        try:
            self.cmb_starless.clear()
            self.cmb_stars.clear()
            for label, data in items:
                self.cmb_starless.addItem(label, data)
                self.cmb_stars.addItem(label, data)
        finally:
            self.cmb_starless.blockSignals(False)
            self.cmb_stars.blockSignals(False)

    # File load ----------------------------------------------------------------
    def _load_from_file(self, which: str):
        fn, _ = QFileDialog.getOpenFileName(
            self, self.tr("Select Starless Image") if which=='starless' else self.tr("Select Stars-Only Image"), "",
            "Image Files (*.png *.tif *.tiff *.fits *.fit *.xisf *.jpg *.jpeg)"
        )
        if not fn:
            return
        img, _, _, _ = load_image(fn)
        if img is None:
            QMessageBox.critical(self, self.tr("Load Error"), self.tr("Failed to load: {0}").format(os.path.basename(fn)))
            return
        if which == 'starless':
            self.starless = self._to_rgb01(img)
            self.cmb_starless.setCurrentIndex(self.cmb_starless.count()-1)  # "Load from File"
        else:
            self.stars_only = self._to_rgb01(img)
            self.cmb_stars.setCurrentIndex(self.cmb_stars.count()-1)
        self.update_preview()

    @staticmethod
    def _resolve_doc_object(doc_like):
        if doc_like is None:
            return None
        for c in (doc_like,
                getattr(doc_like, "doc", None),
                getattr(doc_like, "_doc", None),
                getattr(doc_like, "document", None)):
            if c is None:
                continue
            if hasattr(c, "apply_edit") and any(
                hasattr(c, a) for a in ("image", "get_image", "current_image", "image_array")
            ):
                return c
        return None

    def _target_doc_for_mask(self):
        """Use the selected Starless View's doc (fallback to active doc)."""
        sel = self.cmb_starless.currentData()
        if sel is None or sel == "file":
            doc = getattr(self.main, "_active_doc", None)
            if callable(doc): doc = doc()
            return self._resolve_doc_object(doc)
        return self._resolve_doc_object(sel)

    # Combo selects ------------------------------------------------------------
    def _pick_starless_from_combo(self):
        data = self.cmb_starless.currentData()
        if data is None or data == "file":
            # None or "Load from File" (the button sets image)
            self.update_preview()
            return
        img = _doc_image(data)
        if img is None:
            QMessageBox.warning(self, self.tr("Empty View"), self.tr("Selected starless view has no image."))
            return
        self.starless = self._to_rgb01(img)
        self.update_preview()

    def _pick_stars_from_combo(self):
        data = self.cmb_stars.currentData()
        if data is None or data == "file":
            self.update_preview()
            return
        img = _doc_image(data)
        if img is None:
            QMessageBox.warning(self, self.tr("Empty View"), self.tr("Selected stars-only view has no image."))
            return
        self.stars_only = self._to_rgb01(img)
        self.update_preview()

    # Math ---------------------------------------------------------------------
    @staticmethod
    def _to_rgb01(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a).astype(np.float32, copy=False)
        if a.ndim == 2:
            a = np.stack([a]*3, axis=-1)
        elif a.ndim == 3 and a.shape[2] == 1:
            a = np.repeat(a, 3, axis=2)
        a = np.clip(a, 0.0, 1.0)
        return a

    def _blend_inputs(self):
        """(mode, ratio, raw mask of the destination doc or None) for a blend."""
        mode = self.cmb_blend.currentText()
        r = self.slider_ratio.value() / 100.0
        # mask from the *destination* doc (selected Starless View)
        tgt = self._target_doc_for_mask()
        m = _active_mask_array_from_doc(tgt) if tgt is not None else None
        return mode, r, m

    def _blend_images(self) -> np.ndarray | None:
        """Full-resolution blend on the calling thread (Apply runs the same on a worker)."""
        if self.starless is None or self.stars_only is None:
            return None

        # same size?
        if self.starless.shape != self.stars_only.shape:
            QMessageBox.critical(self, self.tr("Size Mismatch"), self.tr("Starless and Stars-Only views are different sizes."))
            return None

        mode, r, m = self._blend_inputs()
        return _add_stars_full_job(self.starless, self.stars_only, mode, r, m)["result"]

    # Preview ------------------------------------------------------------------
    def update_preview(self):
        """Debounced: the preview is rebuilt once the controls have been
        still for _PREVIEW_DEBOUNCE_MS (_launch_preview)."""
        self._pv_timer.start(_PREVIEW_DEBOUNCE_MS)

    def _proxy_for(self, img, w, h):
        hit = self._proxies.get((id(img), w, h))
        return hit[1] if hit is not None and hit[0] is img else None

    def _launch_preview(self):
        """Build the preview on a worker, one at a time, at the displayed
        size; a change made while one runs re-runs it when it lands."""
        self._pv_timer.stop()
        if thread_running(self._pv_thread):
            self._pv_pending = True
            return
        if self.starless is None or self.stars_only is None:
            self.blended_image = None
            self._pix = None
            self.preview_label.clear()
            return
        # same size?
        if self.starless.shape != self.stars_only.shape:
            self.blended_image = None
            self._pix = None
            self.preview_label.clear()
            QMessageBox.critical(self, self.tr("Size Mismatch"), self.tr("Starless and Stars-Only views are different sizes."))
            return
        H, W = self.starless.shape[:2]
        scale = float(self.scale_factor)
        s = min(1.0, scale)
        w, h = max(1, round(W*s)), max(1, round(H*s))
        mode, r, m = self._blend_inputs()
        thr = _AddStarsJob(_add_stars_preview_job, dict(
            starless=self.starless, stars_only=self.stars_only,
            ps=self._proxy_for(self.starless, w, h), pt=self._proxy_for(self.stars_only, w, h),
            w=w, h=h, scale=scale, mode=mode, r=r, mask=m))
        thr.ready.connect(self._on_preview_ready)
        thr.finished.connect(self._on_preview_finished)
        self._pv_thread = thr
        start_owned_thread(thr)

    def _on_preview_ready(self, res: dict):
        if "error" in res:
            self.preview_label.setText(self.tr("Preview failed: {0}").format(res['error']))
            return
        if res["starless"] is not self.starless or res["stars_only"] is not self.stars_only:
            return                       # a source changed meanwhile; its own preview follows
        w, h = res["w"], res["h"]
        self._proxies = {(id(res["starless"]), w, h): (res["starless"], res["ps"]),
                         (id(res["stars_only"]), w, h): (res["stars_only"], res["pt"])}
        self.blended_image = res["out"]
        self._pix = QPixmap.fromImage(res["qimg"])
        self._pix_scale = res["scale"]
        # keep scroll position
        hs = self.scroll_area.horizontalScrollBar().value()
        vs = self.scroll_area.verticalScrollBar().value()
        self._refresh_scaled()
        self.scroll_area.horizontalScrollBar().setValue(hs)
        self.scroll_area.verticalScrollBar().setValue(vs)

    def _on_preview_finished(self):
        if self._pv_pending:
            self._pv_pending = False
            self._launch_preview()

    def _to_pixmap(self, img: np.ndarray) -> QPixmap:
        im = np.clip(img, 0.0, 1.0)
        u8 = (im * 255.0 + 0.5).astype(np.uint8)
        if u8.ndim == 2:
            q = QImage(u8.data, u8.shape[1], u8.shape[0], u8.strides[0], QImage.Format.Format_Grayscale8)
        else:
            # RGB888
            q = QImage(u8.data, u8.shape[1], u8.shape[0], u8.strides[0], QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q)

    # Zoom/fit -----------------------------------------------------------------
    def wheelEvent(self, ev: QWheelEvent):
        if ev.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        ev.accept()

    def zoom_in(self):
        self.scale_factor *= 1.25
        self._refresh_scaled()

    def zoom_out(self):
        self.scale_factor /= 1.25
        self._refresh_scaled()

    def fit_to_preview(self):
        if self.blended_image is None:
            return
        QTimer.singleShot(0, self._do_fit)

    def _do_fit(self):
        if self.blended_image is None or self.starless is None:
            return
        H, W = self.starless.shape[:2]
        vsz = self.scroll_area.viewport().size()
        if W == 0 or H == 0:
            return
        sw = vsz.width() / W
        sh = vsz.height() / H
        self.scale_factor = min(sw, sh)
        self._refresh_scaled()

    def _refresh_scaled(self):
        if self._pix is None or self.starless is None:
            return
        if self._pix_scale == self.scale_factor:
            self.preview_label.setPixmap(self._pix)
        else:
            # Rescale what is on screen at once, and rebuild it at this zoom.
            H, W = self.starless.shape[:2]
            self.preview_label.setPixmap(self._pix.scaled(
                max(1, round(W*self.scale_factor)), max(1, round(H*self.scale_factor)),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation))
            self.update_preview()
        self.preview_label.adjustSize()

    # Apply --------------------------------------------------------------------
    def _apply(self):
        """
        Applies the blended image to the selected *Starless View* (or, if the starless
        source is "Load from File", falls back to the active doc).
        """
        if thread_running(self._apply_thread):
            return
        if (self.starless is None or self.stars_only is None
                or self.starless.shape != self.stars_only.shape):
            QMessageBox.warning(self, self.tr("No Blend"), self.tr("No blended image to apply."))
            return

        sel = self.cmb_starless.currentData()
        target_doc = None

        if sel is None:  # "Select View"
            # Fallback: active doc
            doc = getattr(self.main, "_active_doc", None)
            if callable(doc):
                doc = doc()
            target_doc = self._resolve_doc_object(doc)
        elif sel == "file":
            # Starless came from a file; no view to overwrite → fallback to active
            doc = getattr(self.main, "_active_doc", None)
            if callable(doc):
                doc = doc()
            target_doc = self._resolve_doc_object(doc)
        else:
            # A real view/doc was chosen
            target_doc = self._resolve_doc_object(sel)

        if target_doc is None:
            QMessageBox.warning(self, self.tr("No Target"),
                                self.tr("Pick a starless view to overwrite (or activate a destination window)."))
            return

        mode, r, m = self._blend_inputs()
        thr = _AddStarsJob(_add_stars_full_job, dict(
            starless=self.starless, stars_only=self.stars_only, mode=mode, r=r, mask=m))
        self._apply_target = target_doc
        thr.ready.connect(self._on_apply_ready)
        self._apply_thread = thr
        self.btn_apply.setEnabled(False)
        start_owned_thread(thr)

    def _on_apply_ready(self, res: dict):
        self._apply_thread = None
        target_doc, self._apply_target = self._apply_target, None
        self.btn_apply.setEnabled(True)
        if "error" in res:
            QMessageBox.critical(self, self.tr("Add Stars"), self.tr("Failed to blend:\n{0}").format(res['error']))
            return
        # Emit (target_doc, blended_image)
        self.stars_added.emit(target_doc, res["result"].astype(np.float32, copy=False))
        # Close UI after apply
        self.accept()   # or: self.close()

    def done(self, r):
        # Cancel/Esc while Apply computes: hide now; the dialog closes once
        # the result is emitted.
        if thread_running(self._apply_thread):
            self.hide()
            return
        super().done(r)

    # Ensure initial fit once shown
    def showEvent(self, ev):
        super().showEvent(ev)

        self._populate_doc_combos()

        if not self._geom_restored:
            self._geom_restored = True
            def _after():
                self._restore_ui_state()
                self._save_ui_state()  # optional: commit what we restored
            QTimer.singleShot(0, _after)

        if not self._fit_once:
            self._fit_once = True
            QTimer.singleShot(0, self.fit_to_preview)

    def _save_ui_state(self):
        # Don’t save while we’re initializing/restoring or before the first show
        if getattr(self, "_restoring_ui", False) or not getattr(self, "_geom_restored", False):
            return        
        try:
            s = self._settings
            s.setValue(self._k("window_geometry"), self.saveGeometry())

            s.setValue(self._k("blend_type"), str(self.cmb_blend.currentText()))
            s.setValue(self._k("ratio"), int(self.slider_ratio.value()))

            # Save selections as LABELS (stable), not doc pointers
            s.setValue(self._k("starless_label"), str(self.cmb_starless.currentText()))
            s.setValue(self._k("stars_label"),   str(self.cmb_stars.currentText()))

            try:
                s.sync()
            except Exception:
                pass
        except Exception:
            pass

    def _restore_ui_state(self):
        self._restoring_ui = True
        try:
            s = self._settings

            g = s.value(self._k("window_geometry"), None)
            if g is not None:
                self.restoreGeometry(g)

            # blend type
            bt = str(s.value(self._k("blend_type"), "Screen"))
            i = self.cmb_blend.findText(bt)
            if i >= 0:
                self.cmb_blend.blockSignals(True)
                self.cmb_blend.setCurrentIndex(i)
                self.cmb_blend.blockSignals(False)

            # ratio
            r = int(s.value(self._k("ratio"), 100))
            self.slider_ratio.blockSignals(True)
            self.slider_ratio.setValue(max(0, min(100, r)))
            self.slider_ratio.blockSignals(False)

            # selections by label
            sl_lab = str(s.value(self._k("starless_label"), "Select View"))
            st_lab = str(s.value(self._k("stars_label"), "Select View"))

            j = self.cmb_starless.findText(sl_lab)
            if j >= 0:
                self.cmb_starless.blockSignals(True)
                self.cmb_starless.setCurrentIndex(j)
                self.cmb_starless.blockSignals(False)

            k = self.cmb_stars.findText(st_lab)
            if k >= 0:
                self.cmb_stars.blockSignals(True)
                self.cmb_stars.setCurrentIndex(k)
                self.cmb_stars.blockSignals(False)

        finally:
            self._restoring_ui = False

        # Now load images from the restored selections and run preview
        # Do this after _restoring_ui is False so signals work normally
        self._pick_starless_from_combo()
        self._pick_stars_from_combo()
        QTimer.singleShot(0, self.update_preview)

    def closeEvent(self, ev):
        if thread_running(self._apply_thread):
            # Apply still computing: it closes the dialog once it has emitted.
            self.hide()
            ev.ignore()
            return
        self._pv_timer.stop()
        try:
            self._save_ui_state()
        except Exception:
            pass
        super().closeEvent(ev)

# ──────────────────────────────────────────────────────────────────────────────
# Public entry point: open dialog, then apply to active doc
# ──────────────────────────────────────────────────────────────────────────────
def add_stars(main):
    doc = getattr(main, "_active_doc", None)
    if callable(doc):
        doc = doc()
    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.warning(main, _tr("No Image"), _tr("Please activate a destination image window first."))
        return

    dlg = AddStarsDialog(main, parent=main)
    dlg.stars_added.connect(lambda target, arr: _apply_to_doc(main, target, arr))

    # IMPORTANT: keep a strong reference (non-modal show)
    if not hasattr(main, "_tool_dialogs"):
        main._tool_dialogs = []
    main._tool_dialogs.append(dlg)

    # When the dialog closes, drop the reference
    def _cleanup(_=None, d=dlg):
        try:
            if hasattr(main, "_tool_dialogs") and d in main._tool_dialogs:
                main._tool_dialogs.remove(d)
        except Exception:
            pass

    try:
        dlg.finished.connect(_cleanup)     # QDialog signal
    except Exception:
        pass
    try:
        dlg.destroyed.connect(_cleanup)    # QObject signal (extra safety)
    except Exception:
        pass

    dlg.show()
    dlg.raise_()
    dlg.activateWindow()


def _apply_to_doc(main, doc, arr: np.ndarray):
    """Overwrite the given document with the blended (stars added) result."""
    if doc is None:
        QMessageBox.warning(main, _tr("No Target Document"), _tr("No document to apply to."))
        return
    try:
        meta = {
            "step_name": "Stars Added",
            "bit_depth": "32-bit floating point",
            "is_mono": (arr.ndim == 2),
        }
        doc.apply_edit(arr.astype(np.float32, copy=False), metadata=meta, step_name="Stars Added")
        if hasattr(main, "_log"):
            main._log("Stars Added")
    except Exception as e:
        QMessageBox.critical(main, _tr("Add Stars"), _tr("Failed to apply result:\n{0}").format(e))
