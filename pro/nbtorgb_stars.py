# pro/nbtorgb_stars.py
from __future__ import annotations
import os
import numpy as np

from PyQt6.QtCore import (
    Qt, QSize, QEvent, QTimer, QPoint, QThread, pyqtSignal
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QFileDialog, QInputDialog, QMessageBox, QGridLayout, QCheckBox, QSizePolicy,
    QSlider
)
from PyQt6.QtGui import (
    QPixmap, QImage, QIcon, QPainter, QPen, QColor, QFont, QFontMetrics,
    QCursor, QMovie
)

# Legacy I/O (same used elsewhere in SASpro)
from legacy.image_manager import load_image as legacy_load_image

from legacy.numba_utils import applySCNR_numba, adjust_saturation_numba
from pro.widgets.themed_buttons import themed_toolbtn


# Optional: your stretch helpers (only used if you’d like to pre-stretch inputs)
# from imageops.stretch import stretch_mono_image, stretch_color_image


class NBtoRGBStars(QWidget):
    """
    SASpro version of NB→RGB Stars:
    - Ha/OIII/SII mono (any subset) and/or OSC stars image
    - Ha↔OIII ratio
    - Optional "star stretch"
    - Live preview on the right (with PPP-style zoom/pan/fit)
    - Push final to a new view via DocManager
    """
    THUMB_ICON_SIZE = QSize(22, 22)  # just for button decoration if icon_path provided

    def __init__(self, doc_manager=None, parent=None, icon_path: str | None = None):
        super().__init__(parent)
        self.doc_manager = doc_manager
        self.setWindowTitle("NB→RGB Stars")

        if icon_path:
            try:
                self.setWindowIcon(QIcon(icon_path))
            except Exception:
                pass

        # raw inputs (float32 ~[0..1])
        self.ha   : np.ndarray | None = None
        self.oiii : np.ndarray | None = None
        self.sii  : np.ndarray | None = None
        self.osc  : np.ndarray | None = None    # 3-channel stars-only (optional)

        # filenames / metadata (best-effort)
        self._file_ha   = None
        self._file_oiii = None
        self._file_sii  = None
        self._file_osc  = None

        # output
        self.final: np.ndarray | None = None

        # preview pixmap/zoom state
        self._base_pm: QPixmap | None = None
        self._zoom    = 1.0
        self._min_zoom = 0.05
        self._max_zoom = 20.0
        self._panning = False
        self._pan_last: QPoint | None = None

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QHBoxLayout(self)

        # -------- left controls
        left = QVBoxLayout()
        left_host = QWidget(self); left_host.setLayout(left); left_host.setFixedWidth(320)

        left.addWidget(QLabel(
            "<b>NB→RGB Stars</b><br>"
            "Load Ha / OIII / (optional SII) and/or OSC stars.<br>"
            "Tune ratio and preview; push to a new view."
        ))

        # Load buttons + status labels
        self.btn_ha   = QPushButton("Load Ha…");   self.btn_ha.clicked.connect(lambda: self._load_channel("Ha"))
        self.btn_oiii = QPushButton("Load OIII…"); self.btn_oiii.clicked.connect(lambda: self._load_channel("OIII"))
        self.btn_sii  = QPushButton("Load SII (optional)…");  self.btn_sii.clicked.connect(lambda: self._load_channel("SII"))
        self.btn_osc  = QPushButton("Load OSC stars (optional)…"); self.btn_osc.clicked.connect(lambda: self._load_channel("OSC"))

        self.lbl_ha   = QLabel("No Ha loaded.")
        self.lbl_oiii = QLabel("No OIII loaded.")
        self.lbl_sii  = QLabel("No SII loaded.")
        self.lbl_osc  = QLabel("No OSC stars loaded.")

        for lab in (self.lbl_ha, self.lbl_oiii, self.lbl_sii, self.lbl_osc):
            lab.setWordWrap(True); lab.setStyleSheet("color:#888; margin-left:8px;")

        for btn, lab in ((self.btn_ha, self.lbl_ha),
                         (self.btn_oiii, self.lbl_oiii),
                         (self.btn_sii, self.lbl_sii),
                         (self.btn_osc, self.lbl_osc)):
            left.addWidget(btn); left.addWidget(lab)

        # Ratio (Ha to OIII)
        row = QHBoxLayout()
        self.lbl_ratio = QLabel("Ha:OIII ratio = 0.30")
        self.sld_ratio = QSlider(Qt.Orientation.Horizontal); self.sld_ratio.setRange(0, 100); self.sld_ratio.setValue(30)
        self.sld_ratio.valueChanged.connect(lambda v: self.lbl_ratio.setText(f"Ha:OIII ratio = {v/100:.2f}"))
        row.addWidget(self.lbl_ratio); left.addLayout(row)
        left.addWidget(self.sld_ratio)

        # Star Stretch
        self.chk_star_stretch = QCheckBox("Enable star stretch"); self.chk_star_stretch.setChecked(True)
        left.addWidget(self.chk_star_stretch)

        row2 = QHBoxLayout()
        self.lbl_stretch = QLabel("Stretch factor = 5.00")
        self.sld_stretch = QSlider(Qt.Orientation.Horizontal); self.sld_stretch.setRange(0, 800); self.sld_stretch.setValue(500)
        self.sld_stretch.valueChanged.connect(lambda v: self.lbl_stretch.setText(f"Stretch factor = {v/100:.2f}"))
        row2.addWidget(self.lbl_stretch); left.addLayout(row2)
        left.addWidget(self.sld_stretch)

        row3 = QHBoxLayout()
        self.lbl_sat = QLabel("Saturation = 1.00×")
        self.sld_sat = QSlider(Qt.Orientation.Horizontal)
        self.sld_sat.setRange(0, 300)         # 0.00× … 3.00×
        self.sld_sat.setValue(100)            # 1.00× by default
        self.sld_sat.valueChanged.connect(lambda v: self.lbl_sat.setText(f"Saturation = {v/100:.2f}×"))
        row3.addWidget(self.lbl_sat)
        left.addLayout(row3)
        left.addWidget(self.sld_sat)

        # Actions
        act = QHBoxLayout()
        self.btn_preview = QPushButton("Preview Combine"); self.btn_preview.clicked.connect(self._preview_combine)
        self.btn_push = QPushButton("Push Final to New View"); self.btn_push.clicked.connect(self._push_final)
        act.addWidget(self.btn_preview); act.addWidget(self.btn_push)
        left.addLayout(act)

        self.btn_clear = QPushButton("Clear Inputs"); self.btn_clear.clicked.connect(self._clear_inputs)
        left.addWidget(self.btn_clear)

        # Spinner (optional)
        self.spinner = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.spinner_movie = QMovie(os.path.join(os.path.dirname(__file__), "spinner.gif"))
        self.spinner.setMovie(self.spinner_movie); self.spinner.hide()
        left.addWidget(self.spinner)

        left.addStretch(1)
        root.addWidget(left_host, 0)

        # -------- right: preview (zoom/pan like PPP)
        right = QVBoxLayout()

        tools = QHBoxLayout()
        self.btn_zoom_in  = QPushButton("Zoom +"); self.btn_zoom_in.clicked.connect(lambda: self._zoom_at(1.25))
        self.btn_zoom_out = QPushButton("Zoom −"); self.btn_zoom_out.clicked.connect(lambda: self._zoom_at(0.8))
        self.btn_fit      = QPushButton("Fit to Preview"); self.btn_fit.clicked.connect(self._fit_to_preview)
        tools.addWidget(self.btn_zoom_in); tools.addWidget(self.btn_zoom_out); tools.addWidget(self.btn_fit)
        right.addLayout(tools)

        self.scroll = QScrollArea(self); self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.preview)
        self.preview.setMouseTracking(True)

        # Intercept wheel everywhere to prevent scroll while zooming; pan via drag
        for obj in (self.preview, self.scroll, self.scroll.viewport(),
                    self.scroll.horizontalScrollBar(), self.scroll.verticalScrollBar()):
            obj.installEventFilter(self)

        right.addWidget(self.scroll, 1)
        self.status = QLabel(""); right.addWidget(self.status, 0)

        right_host = QWidget(self); right_host.setLayout(right)
        root.addWidget(right_host, 1)

        self.setLayout(root)
        self.setMinimumSize(980, 640)

    def showEvent(self, e):
        super().showEvent(e)
        QTimer.singleShot(0, self._center_scrollbars)

    # ---------- file/view loading ----------
    def _set_status_label(self, which: str, text: str | None):
        lab = getattr(self, f"lbl_{which.lower()}")
        if text:
            lab.setText(text); lab.setStyleSheet("color:#2a7; font-weight:600; margin-left:8px;")
        else:
            lab.setText(f"No {which} loaded."); lab.setStyleSheet("color:#888; margin-left:8px;")

    def _load_channel(self, which: str):
        src, ok = QInputDialog.getItem(self, f"Load {which}", "Source:", ["From View", "From File"], 0, False)
        if not ok: return

        out = self._load_from_view(which) if src == "From View" else self._load_from_file(which)
        if out is None: return
        img, header, bit_depth, is_mono, path, label = out

        # Normalize to floats in [0,1]; collapse mono to 2D; ensure OSC is RGB
        if which in ("Ha","OIII","SII"):
            if img.ndim == 3: img = img[...,0]
            setattr(self, which.lower(), self._as_float01(img))
        else:  # OSC
            if img.ndim == 2: img = np.stack([img]*3, axis=-1)
            setattr(self, which.lower(), self._as_float01(img))

        setattr(self, f"_file_{which.lower()}", path)
        self._set_status_label(which, label)
        self.status.setText(f"{which} loaded ({'mono' if img.ndim==2 else 'RGB'}) shape={img.shape}")

    def _load_from_view(self, which):
        views = self._list_open_views()
        if not views:
            QMessageBox.warning(self, "No Views", "No open image views found."); return None
        labels = [lab for lab, _ in views]
        choice, ok = QInputDialog.getItem(self, f"Select View for {which}", "Choose a view:", labels, 0, False)
        if not ok or not choice: return None
        sw = dict(views)[choice]
        doc = getattr(sw, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.warning(self, "Empty View", "Selected view has no image."); return None
        img = doc.image
        meta = getattr(doc, "metadata", {}) or {}
        header = meta.get("original_header", None)
        bit_depth = meta.get("bit_depth", "Unknown")
        is_mono = (img.ndim == 2) or (img.ndim == 3 and img.shape[2] == 1)
        path = meta.get("file_path", None)
        return img, header, bit_depth, is_mono, path, f"From View: {choice}"

    def _load_from_file(self, which):
        filt = "Images (*.png *.tif *.tiff *.fits *.fit *.xisf)"
        path, _ = QFileDialog.getOpenFileName(self, f"Select {which} File", "", filt)
        if not path: return None
        img, header, bit_depth, is_mono = legacy_load_image(path)
        if img is None:
            QMessageBox.critical(self, "Load Error", f"Could not load {os.path.basename(path)}"); return None
        return img, header, bit_depth, is_mono, path, f"From File: {os.path.basename(path)}"

    # ---------- combine / preview ----------
    def _preview_combine(self):
        if (self.osc is None) and not (self.ha is not None and self.oiii is not None):
            QMessageBox.warning(self, "Missing Images", "Load OSC, or Ha+OIII (SII optional).")
            return

        self.spinner.show(); self.spinner_movie.start()

        ratio = self.sld_ratio.value() / 100.0
        stretch_enabled = self.chk_star_stretch.isChecked()
        stretch_factor = self.sld_stretch.value() / 100.0
        sat_factor = self.sld_sat.value() / 100.0

        try:
            # 1) combine (no SCNR here)
            rgb = self._combine_nb_rgb(ratio, stretch_enabled, stretch_factor)

            # 2) ensure float32 & contiguous for numba
            rgb = np.ascontiguousarray(rgb.astype(np.float32))

            # 3) SCNR (numba)
            rgb = applySCNR_numba(rgb)

            # 4) Saturation (numba)
            if abs(sat_factor - 1.0) > 1e-3:
                rgb = adjust_saturation_numba(rgb, sat_factor)

            self.final = np.clip(rgb, 0.0, 1.0)

        except Exception as e:
            self.spinner.hide(); self.spinner_movie.stop()
            QMessageBox.critical(self, "Combine Error", str(e))
            return

        self._set_preview_image(self._to_qimage(self.final))
        self.status.setText("Preview updated.")
        self.spinner.hide(); self.spinner_movie.stop()


    def _combine_nb_rgb(self, ratio: float, star_stretch: bool, stretch_k: float) -> np.ndarray:
        """
        Combine to RGB:
        - If OSC present: use channels from OSC, optionally blend Ha/SII/OO into them.
        - Else NB-only: R ~ 0.5*(Ha+SII), G ~ mix(Ha, OIII) via ratio, B ~ OIII.
        Shapes must match.
        """
        # Ensure shapes
        shapes = [x.shape[:2] for x in (self.ha, self.oiii, self.sii) if x is not None]
        if self.osc is not None:
            shapes.append(self.osc.shape[:2])
        if shapes and len(set(shapes)) != 1:
            raise ValueError(f"Channel sizes differ: {set(shapes)}")

        if self.osc is not None:
            r = self.osc[...,0]; g = self.osc[...,1]; b = self.osc[...,2]
            sii = self.sii if self.sii is not None else r
            ha  = self.ha  if self.ha  is not None else r
            oiii= self.oiii if self.oiii is not None else b

            r_out = 0.5*r + 0.5*sii
            g_out = ratio*ha + (1.0 - ratio)*g
            b_out = oiii
        else:
            if self.ha is None or self.oiii is None:
                raise ValueError("Need Ha and OIII if no OSC image is provided.")
            ha = self.ha
            sii = self.sii if self.sii is not None else ha
            oiii = self.oiii
            r_out = 0.5*ha + 0.5*sii
            g_out = ratio*ha + (1.0 - ratio)*oiii
            b_out = oiii

        rgb = np.stack([r_out, g_out, b_out], axis=2).astype(np.float32)
        rgb = np.clip(rgb, 0, 1)

        if star_stretch:
            # Simple non-linear boost; bounded and monotonic
            # ((3^k)*x) / ((3^k - 1)*x + 1)
            t = 3.0 ** float(stretch_k)
            rgb = (t*rgb) / ((t - 1.0)*rgb + 1.0)
            rgb = np.clip(rgb, 0, 1)

        return rgb


    # ---------- preview helpers + zoom/pan ----------
    def _set_preview_image(self, qimg: QImage):
        self._base_pm = QPixmap.fromImage(qimg)
        self._zoom = 1.0
        self._update_preview_pixmap()
        QTimer.singleShot(0, self._center_scrollbars)

    def _update_preview_pixmap(self):
        if self._base_pm is None: return
        scaled = self._base_pm.scaled(
            self._base_pm.size() * self._zoom,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview.setPixmap(scaled)
        self.preview.resize(scaled.size())

    def _set_zoom(self, new_zoom: float):
        self._zoom = max(self._min_zoom, min(self._max_zoom, new_zoom))
        self._update_preview_pixmap()

    def _zoom_at(self, factor: float = 1.25, anchor_vp: QPoint | None = None):
        if self._base_pm is None: return

        old_zoom = self._zoom
        new_zoom = max(self._min_zoom, min(self._max_zoom, old_zoom * factor))
        ratio = new_zoom / max(1e-6, old_zoom)

        vp = self.scroll.viewport()
        if anchor_vp is None:
            anchor_vp = QPoint(vp.width() // 2, vp.height() // 2)  # center of view

        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()

        content_x = hbar.value() + anchor_vp.x()
        content_y = vbar.value() + anchor_vp.y()

        self._set_zoom(new_zoom)

        if self.preview.width() <= vp.width():
            hbar.setValue((hbar.maximum() + hbar.minimum()) // 2)
        else:
            new_h = int(content_x * ratio - anchor_vp.x())
            hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), new_h)))

        if self.preview.height() <= vp.height():
            vbar.setValue((vbar.maximum() + vbar.minimum()) // 2)
        else:
            new_v = int(content_y * ratio - anchor_vp.y())
            vbar.setValue(max(vbar.minimum(), min(vbar.maximum(), new_v)))

    def _fit_to_preview(self):
        if self._base_pm is None: return
        vp = self.scroll.viewport().size()
        pm = self._base_pm.size()
        if pm.width() == 0 or pm.height() == 0: return
        k = min(vp.width() / pm.width(), vp.height() / pm.height())
        self._set_zoom(max(self._min_zoom, min(self._max_zoom, k)))
        self._center_scrollbars()

    def _center_scrollbars(self):
        h = self.scroll.horizontalScrollBar()
        v = self.scroll.verticalScrollBar()
        h.setValue((h.maximum() + h.minimum()) // 2)
        v.setValue((v.maximum() + v.minimum()) // 2)

    # ---------- utilities ----------
    def _clear_inputs(self):
        self.ha = self.oiii = self.sii = self.osc = None
        self._file_ha = self._file_oiii = self._file_sii = self._file_osc = None
        self.final = None
        self.preview.clear(); self._base_pm = None
        for which in ("Ha","OIII","SII","OSC"):
            self._set_status_label(which, None)
        self.status.setText("Cleared inputs.")

    @staticmethod
    def _as_float01(arr):
        a = np.asarray(arr)
        if a.dtype == np.uint8:  return a.astype(np.float32)/255.0
        if a.dtype == np.uint16: return a.astype(np.float32)/65535.0
        return np.clip(a.astype(np.float32), 0.0, 1.0)

    @staticmethod
    def _to_qimage(arr):
        a = np.clip(arr, 0, 1)
        if a.ndim == 2:
            u = (a * 255).astype(np.uint8); h, w = u.shape
            return QImage(u.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        if a.ndim == 3 and a.shape[2] == 3:
            u = (a * 255).astype(np.uint8); h, w, _ = u.shape
            return QImage(u.data, w, h, w*3, QImage.Format.Format_RGB888).copy()
        raise ValueError(f"Unexpected image shape: {a.shape}")

    def _find_main_window(self):
        w = self
        from PyQt6.QtWidgets import QMainWindow, QApplication
        while w is not None and not isinstance(w, QMainWindow):
            w = w.parentWidget()
        if w: return w
        for tlw in QApplication.topLevelWidgets():
            if isinstance(tlw, QMainWindow):
                return tlw
        return None

    def _list_open_views(self):
        mw = self._find_main_window()
        if not mw: return []
        try:
            from pro.subwindow import ImageSubWindow
            subs = mw.findChildren(ImageSubWindow)
        except Exception:
            subs = []
        out = []
        for sw in subs:
            title = getattr(sw, "view_title", None) or sw.windowTitle() or getattr(sw.document, "display_name", lambda: "Untitled")()
            out.append((str(title), sw))
        return out

    def _push_final(self):
        if self.final is None:
            QMessageBox.warning(self, "No Image", "Preview first, then push."); return
        mw = self._find_main_window()
        dm = getattr(mw, "docman", None)
        if not mw or not dm:
            QMessageBox.critical(self, "UI", "Main window or DocManager not available."); return
        title = "NB→RGB Stars"
        try:
            if hasattr(dm, "open_array"):
                doc = dm.open_array(self.final, metadata={"is_mono": False}, title=title)
            elif hasattr(dm, "create_document"):
                doc = dm.create_document(image=self.final, metadata={"is_mono": False}, name=title)
            else:
                raise RuntimeError("DocManager lacks open_array/create_document")
            if hasattr(mw, "_spawn_subwindow_for"):
                mw._spawn_subwindow_for(doc)
            else:
                from pro.subwindow import ImageSubWindow
                sw = ImageSubWindow(doc, parent=mw); sw.setWindowTitle(title); sw.show()
            self.status.setText("Opened final composite in a new view.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open new view:\n{e}")

    # ---------- event filter (zoom/pan like PPP) ----------
    def eventFilter(self, obj, ev):
        # Ctrl+wheel zoom at mouse (prevent scrolling); wheel without Ctrl: eat it (no scroll)
        if ev.type() == QEvent.Type.Wheel and (
            obj is self.preview
            or obj is self.scroll
            or obj is self.scroll.viewport()
            or obj is self.scroll.horizontalScrollBar()
            or obj is self.scroll.verticalScrollBar()
        ):
            if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                factor = 1.25 if ev.angleDelta().y() > 0 else 0.8
                # safer: compute anchor in viewport coords via global position
                try:
                    anchor_vp = self.scroll.viewport().mapFromGlobal(ev.globalPosition().toPoint())
                except Exception:
                    vp = self.scroll.viewport()
                    anchor_vp = QPoint(vp.width()//2, vp.height()//2)
                self._zoom_at(factor, anchor_vp)
            ev.accept()
            return True

        # click-drag pan on viewport
        if obj is self.scroll.viewport():
            if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = True
                self._pan_last = ev.position().toPoint()
                self.scroll.viewport().setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
                return True
            if ev.type() == QEvent.Type.MouseMove and self._panning:
                cur = ev.position().toPoint()
                delta = cur - (self._pan_last or cur)
                self._pan_last = cur
                h = self.scroll.horizontalScrollBar()
                v = self.scroll.verticalScrollBar()
                h.setValue(h.value() - delta.x())
                v.setValue(v.value() - delta.y())
                return True
            if ev.type() == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = False
                self._pan_last = None
                self.scroll.viewport().setCursor(QCursor(Qt.CursorShape.ArrowCursor))
                return True

        return super().eventFilter(obj, ev)
