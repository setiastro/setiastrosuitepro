# pro/ghs_dialog_pro.py
from PyQt6.QtCore import Qt, QEvent, QPointF, QTimer
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QScrollArea, QComboBox, QSlider, QToolButton, QWidget, QMessageBox)
from PyQt6.QtGui import QPixmap, QImage, QPen, QColor
import numpy as np



# Reuse the engine from curves_editor_pro
from .curve_editor_pro import (
    CurveEditor, _CurvesWorker, _apply_mode_any, build_curve_lut,
    _float_to_qimage_rgb8, _downsample_for_preview, ImageLabel
)

class GhsDialogPro(QDialog):
    """
    Hyperbolic Stretch dialog:
    - Left: α/β/γ + LP/HP + channel selector
    - Right: same preview/zoom/pan as CurvesDialogPro
    - Uses CurveEditor for the actual curve, but the points are generated from parameters.
    """
    def __init__(self, parent, document):
        super().__init__(parent)
        self.setWindowTitle("Hyperbolic Stretch")
        self.doc = document
        self._preview_img = None
        self._full_img = None
        self._pix = None
        self._zoom = 0.25
        self._panning = False
        self._pan_start = QPointF()
        self._sym_u = 0.5   # pivot in [0..1]

        # ---------- layout ----------
        main = QHBoxLayout(self)

        # Left controls
        left = QVBoxLayout()
        self.editor = CurveEditor(self)
        left.addWidget(self.editor)

        hint = QLabel("Tip: Ctrl+Click (or double-click) the image to set the symmetry pivot")
        hint.setStyleSheet("color: #888; font-size: 11px;")
        left.addWidget(hint)
        self.editor.setToolTip("Ctrl+Click (or double-click) the image to set the symmetry pivot")

        # channel selector
        ch_row = QHBoxLayout()
        ch_row.addWidget(QLabel("Channel:"))
        self.cmb_ch = QComboBox(self)
        self.cmb_ch.addItems(["K (Brightness)", "R", "G", "B"])
        ch_row.addWidget(self.cmb_ch)
        left.addLayout(ch_row)

        # α / β / γ
        def _mk_slider_row(name, rng, val):
            row = QHBoxLayout()
            lab = QLabel(name); row.addWidget(lab)
            s = QSlider(Qt.Orientation.Horizontal); s.setRange(*rng); s.setValue(val); row.addWidget(s)
            v = QLabel(f"{val/100:.2f}" if name=="γ" else f"{val/50:.2f}"); row.addWidget(v)
            return row, s, v

        rowA, self.sA, self.labA = _mk_slider_row("α", (1, 500), 50)   # 1.0
        rowB, self.sB, self.labB = _mk_slider_row("β", (1, 500), 50)   # 1.0
        rowG, self.sG, self.labG = _mk_slider_row("γ", (1, 500), 100)  # 1.0
        left.addLayout(rowA); left.addLayout(rowB); left.addLayout(rowG)

        # LP / HP (protect)
        rowLP = QHBoxLayout(); rowHP = QHBoxLayout()
        rowLP.addWidget(QLabel("LP")); self.sLP = QSlider(Qt.Orientation.Horizontal); self.sLP.setRange(0,360); rowLP.addWidget(self.sLP); self.labLP = QLabel("0.00"); rowLP.addWidget(self.labLP)
        rowHP.addWidget(QLabel("HP")); self.sHP = QSlider(Qt.Orientation.Horizontal); self.sHP.setRange(0,360); rowHP.addWidget(self.sHP); self.labHP = QLabel("0.00"); rowHP.addWidget(self.labHP)
        left.addLayout(rowLP); left.addLayout(rowHP)

        # Buttons
        rowb = QHBoxLayout()
        self.btn_apply = QPushButton("Apply")
        self.btn_reset = QToolButton(); self.btn_reset.setText("Reset")
        rowb.addWidget(self.btn_apply); rowb.addWidget(self.btn_reset)
        left.addLayout(rowb)
        left.addStretch(1)

        main.addLayout(left, 0)

        # --- Right preview panel ---
        right = QVBoxLayout()
        zoombar = QHBoxLayout()
        b_out = QPushButton("Zoom Out"); b_in = QPushButton("Zoom In"); b_fit = QPushButton("Fit to Preview")
        zoombar.addWidget(b_out); zoombar.addWidget(b_in); zoombar.addWidget(b_fit)
        right.addLayout(zoombar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # CREATE LABEL FIRST
        self.label = ImageLabel(self)                 # <- make sure ImageLabel is imported
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.mouseMoved.connect(self._on_preview_mouse_moved)
        self.label.installEventFilter(self)

        self.scroll.setWidget(self.label)
        # INSTALL FILTERS AFTER label exists
        self.scroll.viewport().installEventFilter(self)

        right.addWidget(self.scroll, 1)
        main.addLayout(right, 1)

        # ---------- wiring ----------
        self.editor.setPreviewCallback(lambda _lut8: self._quick_preview())
        self.editor.setSymmetryCallback(self._on_symmetry_pick)

        self.sA.valueChanged.connect(self._rebuild_from_params)
        self.sB.valueChanged.connect(self._rebuild_from_params)
        self.sG.valueChanged.connect(self._rebuild_from_params)
        self.sLP.valueChanged.connect(self._rebuild_from_params)
        self.sHP.valueChanged.connect(self._rebuild_from_params)
        self.cmb_ch.currentTextChanged.connect(self._recolor_curve)

        self.btn_apply.clicked.connect(self._apply)
        self.btn_reset.clicked.connect(self._reset)

        b_out.clicked.connect(lambda: self._set_zoom(self._zoom / 1.25))
        b_in .clicked.connect(lambda: self._set_zoom(self._zoom * 1.25))
        b_fit.clicked.connect(self._fit)

        # seed image data
        self._load_from_doc()

        # start with Fit to Preview (avoids offset issues)
        QTimer.singleShot(0, self._fit)
        

        # first curve
        self._rebuild_from_params()

    # ---------- params → handles/curve ----------
    def _on_symmetry_pick(self, u, v):
        self._sym_u = float(u)
        self._rebuild_from_params()

    def _rebuild_from_params(self):
        a = self.sA.value()/50.0
        b = self.sB.value()/50.0
        g = self.sG.value()/100.0
        self.labA.setText(f"{a:.2f}")
        self.labB.setText(f"{b:.2f}")
        self.labG.setText(f"{g:.2f}")

        # number of handles (keep existing count or default to 20)
        N = len(self.editor.control_points) or 20
        if len(self.editor.control_points) == 0:
            for _ in range(N):
                self.editor.addControlPoint(0, 0)

        SP  = float(self._sym_u)
        eps = 1e-6

        # --- sample around 0.5, then REMAP x to SP (this is the key) ---
        us = np.linspace(0.0, 1.0, N)              # even sampling
        left = us <= 0.5
        right = ~left

        # generalized hyperbolic (two shapes, mirrored at 0.5)
        rawL = us**a / (us**a + b*(1.0-us)**a)
        rawR = us**a / (us**a + (1.0/b)*(1.0-us)**a)

        midL = (0.5**a) / (0.5**a + b*(0.5)**a)
        midR = (0.5**a) / (0.5**a + (1.0/b)*(0.5)**a)

        # map domain to pivoted x ("up") and scaled y ("vp")
        up = np.empty_like(us)
        vp = np.empty_like(us)

        # left half → [0 .. SP]
        up[left] = 2.0 * SP * us[left]
        vp[left] = rawL[left] * (SP / max(midL, eps))

        # right half → [SP .. 1]
        up[right] = SP + 2.0*(1.0 - SP)*(us[right] - 0.5)
        vp[right] = SP + (rawR[right] - midR) * ((1.0 - SP) / max(1.0 - midR, eps))

        # LP/HP protection: blend toward identity (vp == up)
        LP = self.sLP.value()/360.0
        HP = self.sHP.value()/360.0
        if LP > 0:
            m = up <= SP
            vp[m] = (1.0 - LP)*vp[m] + LP*up[m]
        if HP > 0:
            m = up >= SP
            vp[m] = (1.0 - HP)*vp[m] + HP*up[m]

        # gamma lift
        if abs(g - 1.0) > 1e-6:
            vp = np.clip(vp, 0.0, 1.0) ** (1.0 / g)

        # keep in range & gently enforce monotonicity to avoid tiny dips
        vp = np.clip(vp, 0.0, 1.0)
        vp = np.maximum.accumulate(vp)

        # write handles back (x rightward, y inverted for the grid)
        xs = up * 360.0
        ys = (1.0 - vp) * 360.0
        pts = list(zip(xs.astype(float), ys.astype(float)))

        cps_sorted = sorted(self.editor.control_points, key=lambda p: p.scenePos().x())
        for p, (x, y) in zip(cps_sorted, pts):
            p.setPos(x, y)

        self._recolor_curve()
        self.editor.updateCurve()
        self._quick_preview()


    def _recolor_curve(self):
        color_map = {
            "K (Brightness)": Qt.GlobalColor.white,
            "R": Qt.GlobalColor.red, "G": Qt.GlobalColor.green, "B": Qt.GlobalColor.blue
        }
        ch = self.cmb_ch.currentText()
        if getattr(self.editor, "curve_item", None):
            pen = QPen(color_map[ch]); pen.setWidth(3)
            self.editor.curve_item.setPen(pen)
        self._quick_preview()

    # ---------- preview/apply (same as CurvesDialogPro) ----------
    def _build_lut01(self):
        fn = getattr(self.editor, "getCurveFunction", None)
        if not fn: return None
        f = fn()
        if f is None: return None
        try:
            return build_curve_lut(f, size=65536)
        except Exception:
            return None
    
    def _quick_preview(self):
        if self._preview_img is None:
            return
        lut01 = self._build_lut01()
        if lut01 is None:
            return
        mode = self.cmb_ch.currentText()
        out = _apply_mode_any(self._preview_img, mode, lut01)
        out = self._blend_with_mask(out)             # ✅ blend with mask
        self._update_preview_pix(out)

    def _apply(self):
        if self._full_img is None:
            return

        luts = self._build_all_active_luts()

        self.btn_apply.setEnabled(False)
        self._thr = _CurvesWorker(self._full_img, luts, self)
        # ⬇️ use the handler you ALREADY have, which commits + metadata + reset
        self._thr.done.connect(self._on_apply_ready)
        self._thr.finished.connect(lambda: self.btn_apply.setEnabled(True))
        self._thr.start()

    def _build_all_active_luts(self) -> dict[str, np.ndarray]:
        """
        For GHS we really only have ONE curve at a time – the one in self.editor –
        and we apply it to the currently selected channel.
        The worker wants a dict like {"K": lut} or {"R": lut}.
        """
        lut = self._build_lut01()
        if lut is None:
            return {}

        ch = self.cmb_ch.currentText()
        # map UI text → worker key
        ui2key = {
            "K (Brightness)": "K",
            "R": "R",
            "G": "G",
            "B": "B",
        }
        key = ui2key.get(ch, "K")
        return {key: lut}

    def _apply_all_curves_once(self, img: np.ndarray, luts: dict[str, np.ndarray]) -> np.ndarray:
        """
        This is what _CurvesWorker will call.
        We only ever expect 0 or 1 LUT here.
        """
        if not luts:
            return img

        # pull the single entry
        (key, lut), = luts.items()

        # map worker key → mode string used by _apply_mode_any
        key2mode = {
            "K": "K (Brightness)",
            "R": "R",
            "G": "G",
            "B": "B",
        }
        mode = key2mode.get(key, "K (Brightness)")

        out = _apply_mode_any(img, mode, lut)
        return out.astype(np.float32, copy=False)

    def _on_apply_commit_ready(self, out01: np.ndarray):
        # honor mask, same as preview
        out01 = self._blend_with_mask(out01)

        # 🔴 safety: if the document currently holds RGB but we got mono back,
        # make it 3-channel so apply_edit doesn’t silently ignore it
        doc_img = np.asarray(self.doc.image)
        if doc_img.ndim == 3 and out01.ndim == 2:
            out01 = np.repeat(out01[..., None], 3, axis=2)

        # now do the normal commit (history, reload, reset curves, etc.)
        self._commit(out01)


    def _on_apply_ready(self, out01: np.ndarray):
        try:
            out_masked = self._blend_with_mask(out01)

            _marr, mid, mname = self._active_mask_layer()
            meta = {
                "step_name": "Hyperbolic Stretch",
                "ghs": {
                    "alpha": self.sA.value()/50.0, "beta": self.sB.value()/50.0,
                    "gamma": self.sG.value()/100.0,
                    "lp": self.sLP.value()/360.0, "hp": self.sHP.value()/360.0,
                    "pivot": self._sym_u,
                    "channel": self.cmb_ch.currentText()
                },
                "masked": bool(mid),
                "mask_id": mid,
                "mask_name": mname,
                "mask_blend": "m*out + (1-m)*src",
            }

            # Commit result to the document
            self.doc.apply_edit(out_masked.copy(), metadata=meta, step_name="Hyperbolic Stretch")

            # 🔁 Refresh buffers from the updated doc
            self._load_from_doc()

            # 🔄 Reset pivot + curve drawing for the next pass
            self._sym_u = 0.5
            self.editor.clearSymmetryLine()
            self.editor.initCurve()        # clear handles & redraw baseline
            self.sA.setValue(50); self.sB.setValue(50); self.sG.setValue(100)
            self.sLP.setValue(0); self.sHP.setValue(0)            
            self._rebuild_from_params()    # repopulate curve from current sliders (now at default pivot)
            QTimer.singleShot(0, self._fit)

        except Exception as e:
            QMessageBox.critical(self, "Apply failed", str(e))


    # ---------- image plumbing / zoom/pan ----------
    def _load_from_doc(self):
        img = self.doc.image
        if img is None:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        arr = np.asarray(img).astype(np.float32)
        if arr.dtype.kind in "ui":
            arr = arr / np.iinfo(img.dtype).max
        self._full_img = arr
        self._preview_img = _downsample_for_preview(arr, 1200)
        self._update_preview_pix(self._preview_img)

    def _update_preview_pix(self, img01):
        if img01 is None:
            self.label.clear(); self._pix = None; return
        qimg = _float_to_qimage_rgb8(img01)
        pm = QPixmap.fromImage(qimg)
        self._pix = pm
        self._apply_zoom()

    def _apply_zoom(self):
        if self._pix is None: return
        scaled = self._pix.scaled(self._pix.size()*self._zoom,
                                  Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(scaled)
        self.label.resize(scaled.size())

    def _set_zoom(self, z):
        self._zoom = float(max(0.05, min(z, 8.0)))
        self._apply_zoom()

    def _fit(self):
        if self._pix is None: return
        vp = self.scroll.viewport().size()
        if self._pix.width()==0 or self._pix.height()==0: return
        s = min(vp.width()/self._pix.width(), vp.height()/self._pix.height())
        self._set_zoom(max(0.05, s))

    def _k_from_label_point(self, lbl_pt):
        """lbl_pt is in label (pixmap) coordinates."""
        if self._preview_img is None or self.label.pixmap() is None:
            return None
        pix = self.label.pixmap()
        pw, ph = pix.width(), pix.height()
        x, y = int(lbl_pt.x()), int(lbl_pt.y())
        if not (0 <= x < pw and 0 <= y < ph):
            return None
        ih, iw = self._preview_img.shape[:2]
        ix = int(x * iw / pw)
        iy = int(y * ih / ph)
        ix = max(0, min(iw - 1, ix))
        iy = max(0, min(ih - 1, iy))
        px = self._preview_img[iy, ix]
        k = float(np.mean(px)) if self._preview_img.ndim == 3 else float(px)
        return max(0.0, min(1.0, k))

    # ctrl+wheel zoom + panning + ctrl+click on preview to move pivot
    def eventFilter(self, obj, ev):
        lbl = getattr(self, "label", None)
        if lbl is None:
            return False
        # --- set pivot on DOUBLE-CLICK (or Ctrl+click) anywhere over the image ---
        if (obj is self.label or obj is self.scroll.viewport()):
            # Double-click → set pivot
            if ev.type() == QEvent.Type.MouseButtonDblClick and ev.button() == Qt.MouseButton.LeftButton:
                lbl_pt = (ev.position().toPoint() if obj is self.label
                        else self.label.mapFrom(self.scroll.viewport(), ev.position().toPoint()))
                k = self._k_from_label_point(lbl_pt)
                if k is not None:
                    self._sym_u = k
                    self.editor.setSymmetryPoint(k * 360.0, 0)
                    self._rebuild_from_params()
                    ev.accept(); return True

            # Keep Ctrl+single-click support too
            if (ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton
                    and (ev.modifiers() & Qt.KeyboardModifier.ControlModifier)):
                lbl_pt = (ev.position().toPoint() if obj is self.label
                        else self.label.mapFrom(self.scroll.viewport(), ev.position().toPoint()))
                k = self._k_from_label_point(lbl_pt)
                if k is not None:
                    self._sym_u = k
                    self.editor.setSymmetryPoint(k * 360.0, 0)
                    self._rebuild_from_params()
                    ev.accept(); return True

        # --- existing zoom/pan handling (unchanged) ---
        if obj is self.scroll.viewport():
            if ev.type() == QEvent.Type.Wheel and (ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self._set_zoom(self._zoom * (1.25 if ev.angleDelta().y() > 0 else 0.8))
                ev.accept(); return True
            if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = True; self._pan_start = ev.position()
                self.scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                ev.accept(); return True
            if ev.type() == QEvent.Type.MouseMove and self._panning:
                d = ev.position() - self._pan_start
                h = self.scroll.horizontalScrollBar(); v = self.scroll.verticalScrollBar()
                h.setValue(h.value() - int(d.x())); v.setValue(v.value() - int(d.y()))
                self._pan_start = ev.position()
                ev.accept(); return True
            if ev.type() == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = False
                self.scroll.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                ev.accept(); return True

        return super().eventFilter(obj, ev)

    def _on_preview_mouse_moved(self, x: float, y: float):
        if self._panning or self._preview_img is None or self._pix is None:
            return
        ix = int(x / max(self._zoom, 1e-6))
        iy = int(y / max(self._zoom, 1e-6))
        ix = max(0, min(self._pix.width()  - 1, ix))
        iy = max(0, min(self._pix.height() - 1, iy))

        img = self._preview_img
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            v = float(img[iy, ix] if img.ndim == 2 else img[iy, ix, 0])
            v = float(np.clip(v, 0.0, 1.0))
            self.editor.updateValueLines(v, 0.0, 0.0, grayscale=True)
        else:
            r, g, b = img[iy, ix, 0], img[iy, ix, 1], img[iy, ix, 2]
            r = float(np.clip(r, 0.0, 1.0)); g = float(np.clip(g, 0.0, 1.0)); b = float(np.clip(b, 0.0, 1.0))
            self.editor.updateValueLines(r, g, b, grayscale=False)

    # --- mask helpers ---------------------------------------------------
    def _active_mask_layer(self):
        """Return (mask_float01, mask_id, mask_name) or (None, None, None)."""
        mid = getattr(self.doc, "active_mask_id", None)
        if not mid: return None, None, None
        layer = getattr(self.doc, "masks", {}).get(mid)
        if layer is None: return None, None, None
        m = np.asarray(getattr(layer, "data", None))
        if m is None or m.size == 0: return None, None, None
        m = m.astype(np.float32, copy=False)
        if m.dtype.kind in "ui":
            m /= float(np.iinfo(m.dtype).max)
        else:
            mx = float(m.max()) if m.size else 1.0
            if mx > 1.0: m /= mx
        return np.clip(m, 0.0, 1.0), mid, getattr(layer, "name", "Mask")

    def _resample_mask_if_needed(self, mask: np.ndarray, out_hw: tuple[int,int]) -> np.ndarray:
        """Nearest-neighbor resize via integer indexing."""
        mh, mw = mask.shape[:2]
        th, tw = out_hw
        if (mh, mw) == (th, tw): return mask
        yi = np.linspace(0, mh - 1, th).astype(np.int32)
        xi = np.linspace(0, mw - 1, tw).astype(np.int32)
        return mask[yi][:, xi]

    def _blend_with_mask(self, processed: np.ndarray) -> np.ndarray:
        """
        Blend processed image with original using active mask (if any).
        Chooses original from preview/full buffers to match shape.
        """
        mask, _mid, _mname = self._active_mask_layer()
        if mask is None:
            return processed

        out = processed.astype(np.float32, copy=False)

        # choose the matching original buffer (same HxW as 'out')
        if (hasattr(self, "_full_img") and self._full_img is not None
                and out.shape[:2] == self._full_img.shape[:2]):
            src = self._full_img
        else:
            src = self._preview_img

        m = self._resample_mask_if_needed(mask, out.shape[:2])
        if out.ndim == 3 and out.shape[2] == 3:
            m = m[..., None]

        # reconcile mono vs RGB
        if src.ndim == 2 and out.ndim == 3:
            src = np.stack([src]*3, axis=-1)
        elif src.ndim == 3 and out.ndim == 2:
            src = src[..., 0]

        return (m * out + (1.0 - m) * src).astype(np.float32, copy=False)


    def _reset(self):
        self.sA.setValue(50); self.sB.setValue(50); self.sG.setValue(100)
        self.sLP.setValue(0);  self.sHP.setValue(0)
        self._sym_u = 0.5
        self.editor.clearSymmetryLine()
        self._rebuild_from_params()
