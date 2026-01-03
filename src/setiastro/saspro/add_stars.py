# pro/add_stars.py
from __future__ import annotations
import os
import numpy as np

# Qt
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QScrollArea, QSizePolicy,
    QComboBox, QSlider, QMessageBox, QFileDialog, QFormLayout
)

# I/O (use your legacy functions)
from setiastro.saspro.legacy.image_manager import load_image
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn


try:
    import cv2
except Exception:
    cv2 = None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers to enumerate docs and masks from the Pro app
# ──────────────────────────────────────────────────────────────────────────────
# REPLACE OLD _iter_open_docs WITH THIS
def _iter_open_docs(main):
    """
    Find open views/docs by:
    1) docman.{documents|docs|open_docs|views|iter_docs|all_docs}
    2) Any attribute on main that has subWindowList() (QMdiArea), pulling docs
       from subwindow.widget().{doc,_doc,document} or the widget itself if it
       exposes an image.
    Returns a list of (label, provider) where provider can be a doc or widget.
    """
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
        """Robustly get a numpy-ish image from doc/widget."""
        if x is None: 
            return None
        chain = [x, getattr(x, "doc", None), getattr(x, "_doc", None), getattr(x, "document", None)]
        for c in chain:
            if c is None:
                continue
            img = getattr(c, "image", None)
            if img is not None:
                try:
                    a = np.asarray(img)
                    if a is not None and a.size:
                        return a
                except Exception:
                    pass
            # method fallbacks
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
        img = _image_from_any(obj)
        if img is None:
            return
        key = id(getattr(obj, "image", obj))  # stable-ish identity
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
        if hasattr(val, "subWindowList"):
            try:
                for sw in val.subWindowList():
                    title = ""
                    try:
                        title = sw.windowTitle()
                    except Exception:
                        pass
                    w = None
                    try:
                        w = sw.widget()
                    except Exception:
                        pass
                    # prefer an actual doc if present; fallback to widget
                    for candidate in (
                        getattr(w, "doc", None),
                        getattr(w, "_doc", None),
                        getattr(w, "document", None),
                        w,
                    ):
                        if candidate is None:
                            continue
                        if _image_from_any(candidate) is not None:
                            _add_item(candidate, label_hint=title)
                            break
            except Exception:
                continue

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


# ──────────────────────────────────────────────────────────────────────────────
# Dialog
# ──────────────────────────────────────────────────────────────────────────────
class AddStarsDialog(QDialog):
    stars_added = pyqtSignal(object, object)
    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Add Stars to Image"))

        self.setWindowFlag(Qt.WindowType.Window, True)
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
        self.blended_image = None
        self.scale_factor = 1.0
        self._fit_once = False

        self._build_ui()
        self._populate_doc_combos()

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
        self.btn_zoom_in  = themed_toolbtn("zoom-in",       "Zoom In")
        self.btn_zoom_out = themed_toolbtn("zoom-out",      "Zoom Out")
        self.btn_fit      = themed_toolbtn("zoom-fit-best", "Fit to Preview")

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
        brow = QHBoxLayout(); brow.addStretch(1)
        btn_apply = QPushButton(self.tr("Apply")); btn_apply.clicked.connect(self._apply)
        btn_cancel= QPushButton(self.tr("Cancel")); btn_cancel.clicked.connect(self.reject)
        brow.addWidget(btn_apply); brow.addWidget(btn_cancel)
        layout.addLayout(brow)

        self.setMinimumSize(900, 650)

        # signals for combos
        self.cmb_starless.currentIndexChanged.connect(self._pick_starless_from_combo)
        self.cmb_stars.currentIndexChanged.connect(self._pick_stars_from_combo)

    # Populate combos with open docs (+ sentinel for file)
    def _populate_doc_combos(self):
        items = [("Select View", None)]
        for name, d in _iter_open_docs(self.main):
            items.append((name, d))
        items.append(("Load from File", "file"))

        self.cmb_starless.clear()
        self.cmb_stars.clear()
        for label, data in items:
            self.cmb_starless.addItem(label, data)
            self.cmb_stars.addItem(label, data)

    # File load ----------------------------------------------------------------
    def _load_from_file(self, which: str):
        fn, _ = QFileDialog.getOpenFileName(
            self, f"Select {'Starless' if which=='starless' else 'Stars-Only'} Image", "",
            "Image Files (*.png *.tif *.tiff *.fits *.fit *.xisf *.jpg *.jpeg)"
        )
        if not fn:
            return
        img, _, _, _ = load_image(fn)
        if img is None:
            QMessageBox.critical(self, "Load Error", f"Failed to load: {os.path.basename(fn)}")
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
            QMessageBox.warning(self, "Empty View", "Selected starless view has no image.")
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
            QMessageBox.warning(self, "Empty View", "Selected stars-only view has no image.")
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

    def _blend_images(self) -> np.ndarray | None:
        if self.starless is None or self.stars_only is None:
            return None

        # same size?
        if self.starless.shape != self.stars_only.shape:
            QMessageBox.critical(self, "Size Mismatch", "Starless and Stars-Only views are different sizes.")
            return None

        mode = self.cmb_blend.currentText()
        r = self.slider_ratio.value() / 100.0

        if mode == "Screen":
            base = self.starless + self.stars_only - (self.starless * self.stars_only)
        else:
            base = self.starless + self.stars_only

        blended = (1.0 - r) * self.starless + r * base
        blended = np.clip(blended, 0.0, 1.0)

        # mask from the *destination* doc (selected Starless View)
        tgt = self._target_doc_for_mask()
        if tgt is not None:
            m = _active_mask_array_from_doc(tgt)
            if m is not None:
                h, w = blended.shape[:2]
                if m.shape != (h, w):
                    if cv2 is not None:
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    else:
                        yi = (np.linspace(0, m.shape[0]-1, h)).astype(np.int32)
                        xi = (np.linspace(0, m.shape[1]-1, w)).astype(np.int32)
                        m = m[yi][:, xi]
                m3 = np.repeat(m[:, :, None], 3, axis=2)
                # only replace where mask==1; keep original starless elsewhere
                blended = np.clip(self.starless * (1.0 - m3) + blended * m3, 0.0, 1.0).astype(np.float32, copy=False)

        return blended

    # Preview ------------------------------------------------------------------
    def update_preview(self):
        out = self._blend_images()
        self.blended_image = out
        if out is None:
            self.preview_label.clear()
            return

        pix = self._to_pixmap(out)
        # keep scroll position
        hs = self.scroll_area.horizontalScrollBar().value()
        vs = self.scroll_area.verticalScrollBar().value()

        scaled = pix.scaled(
            pix.size() * self.scale_factor,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled)
        self.preview_label.adjustSize()
        self.scroll_area.horizontalScrollBar().setValue(hs)
        self.scroll_area.verticalScrollBar().setValue(vs)

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
        if self.blended_image is None:
            return
        pix = self._to_pixmap(self.blended_image)
        vsz = self.scroll_area.viewport().size()
        if pix.isNull() or pix.width() == 0 or pix.height() == 0:
            return
        sw = vsz.width() / pix.width()
        sh = vsz.height() / pix.height()
        self.scale_factor = min(sw, sh)
        self.update_preview()

    def _refresh_scaled(self):
        if self.blended_image is None:
            return
        pix = self._to_pixmap(self.blended_image)
        scaled = pix.scaled(
            pix.size() * self.scale_factor,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled)
        self.preview_label.adjustSize()

    # Apply --------------------------------------------------------------------
    def _apply(self):
        """
        Applies the blended image to the selected *Starless View* (or, if the starless
        source is "Load from File", falls back to the active doc).
        """
        if self.blended_image is None:
            QMessageBox.warning(self, "No Blend", "No blended image to apply.")
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
            QMessageBox.warning(self, "No Target",
                                "Pick a starless view to overwrite (or activate a destination window).")
            return

        # Emit (target_doc, blended_image)
        self.stars_added.emit(target_doc, self.blended_image.astype(np.float32, copy=False))
        # Close UI after apply
        self.accept()   # or: self.close()
        return

    # Ensure initial fit once shown
    def showEvent(self, ev):
        super().showEvent(ev)
        # repopulate in case windows opened after dialog construction
        self._populate_doc_combos()
        if not self._fit_once:
            self._fit_once = True
            QTimer.singleShot(0, self.fit_to_preview)


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point: open dialog, then apply to active doc
# ──────────────────────────────────────────────────────────────────────────────
def add_stars(main):
    doc = getattr(main, "_active_doc", None)
    if callable(doc):
        doc = doc()
    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.warning(main, "No Image", "Please activate a destination image window first.")
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
        QMessageBox.warning(main, "No Target Document", "No document to apply to.")
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
        QMessageBox.critical(main, "Add Stars", f"Failed to apply result:\n{e}")
