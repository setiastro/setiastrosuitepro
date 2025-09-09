# pro/perfect_palette_picker.py
from __future__ import annotations
import os
import numpy as np

from PyQt6.QtCore import Qt, QSize, QEvent, QTimer, QPoint
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QFileDialog, QInputDialog, QMessageBox, QGridLayout, QCheckBox, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QImage, QIcon, QPainter, QPen, QColor, QFont, QFontMetrics, QCursor

# legacy loader (same one DocManager uses)
from legacy.image_manager import load_image as legacy_load_image

# your statistical stretch (mono + color) like SASv2
# (same signatures you use elsewhere)
from imageops.stretch import stretch_mono_image, stretch_color_image

class PerfectPalettePicker(QWidget):
    THUMB_CROP = 512  # side length for thumbnail center crops
    PALETTES = [
        "SHO","HOO","HSO","HOS",
        "OSS","OHH","OSH","OHS",
        "HSS","Realistic1","Realistic2","Foraxx"
    ]

    def __init__(self, doc_manager=None, parent=None):
        super().__init__(parent)
        self.doc_manager = doc_manager
        self.setWindowTitle("Perfect Palette Picker")

        # raw channels (float32 ~[0..1])
        self.ha   = None
        self.oiii = None
        self.sii  = None
        self.osc1 = None
        self.osc2 = None

        # stretched cache (per input name → stretched array)
        self._stretched: dict[str, np.ndarray] = {}

        self.final = None
        self.current_palette = None
        self._thumb_base_pm: dict[str, QPixmap] = {}   # palette name -> base pixmap (image only)
        self._selected_name: str | None = None

        # thumbs
        self._thumb_buttons: dict[str, QPushButton] = {}

        self._base_pm: QPixmap | None = None
        self._zoom = 1.0
        self._min_zoom = 0.05
        self._max_zoom = 6.0
        self._panning = False
        self._pan_last: QPoint | None = None

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QHBoxLayout(self)

        # -------- left controls
        left = QVBoxLayout()
        left_host = QWidget(self); left_host.setLayout(left); left_host.setFixedWidth(300)

        left.addWidget(QLabel("<b>Load channels</b>"))

        # Load buttons + status labels
        self.btn_ha   = QPushButton("Load Ha…");   self.btn_ha.clicked.connect(lambda: self._load_channel("Ha"))
        self.btn_oiii = QPushButton("Load OIII…"); self.btn_oiii.clicked.connect(lambda: self._load_channel("OIII"))
        self.btn_sii  = QPushButton("Load SII…");  self.btn_sii.clicked.connect(lambda: self._load_channel("SII"))
        self.btn_osc1 = QPushButton("Load OSC1 (Ha/OIII)…"); self.btn_osc1.clicked.connect(lambda: self._load_channel("OSC1"))
        self.btn_osc2 = QPushButton("Load OSC2 (SII/OIII)…"); self.btn_osc2.clicked.connect(lambda: self._load_channel("OSC2"))

        self.lbl_ha   = QLabel("No Ha loaded.")
        self.lbl_oiii = QLabel("No OIII loaded.")
        self.lbl_sii  = QLabel("No SII loaded.")
        self.lbl_osc1 = QLabel("No OSC1 loaded.")
        self.lbl_osc2 = QLabel("No OSC2 loaded.")
        for lab in (self.lbl_ha, self.lbl_oiii, self.lbl_sii, self.lbl_osc1, self.lbl_osc2):
            lab.setWordWrap(True); lab.setStyleSheet("color:#888; margin-left:8px;")

        for btn, lab in (
            (self.btn_ha, self.lbl_ha),
            (self.btn_oiii, self.lbl_oiii),
            (self.btn_sii, self.lbl_sii),
            (self.btn_osc1, self.lbl_osc1),
            (self.btn_osc2, self.lbl_osc2),
        ):
            left.addWidget(btn); left.addWidget(lab)

        # Linear toggle (stretch BEFORE palette build)
        self.chk_linear = QCheckBox("Linear input (apply statistical stretch before build)")
        self.chk_linear.setChecked(True)
        self.chk_linear.stateChanged.connect(self._rebuild_stretch_cache_for_all)
        left.addSpacing(6); left.addWidget(self.chk_linear)

        # Actions
        self.btn_clear = QPushButton("Clear Loaded Channels")
        self.btn_clear.clicked.connect(self._clear_channels)
        left.addWidget(self.btn_clear)

        self.btn_create = QPushButton("Create Palettes")
        self.btn_create.clicked.connect(self._create_palettes)
        left.addWidget(self.btn_create)

        self.btn_push = QPushButton("Push Final to New View")
        self.btn_push.clicked.connect(self._push_final)
        left.addWidget(self.btn_push)

        left.addStretch(1)
        root.addWidget(left_host, 0)

        # -------- right: preview + fixed-size 4×3 grid
        right = QVBoxLayout()

        # zoom toolbar
        tools = QHBoxLayout()
        self.btn_zoom_in  = QPushButton("Zoom +"); self.btn_zoom_in.clicked.connect(lambda: self._zoom_at(1.25))
        self.btn_zoom_out = QPushButton("Zoom −"); self.btn_zoom_out.clicked.connect(lambda: self._zoom_at(0.8))
        self.btn_fit      = QPushButton("Fit to Preview"); self.btn_fit.clicked.connect(self._fit_to_preview)
        tools.addWidget(self.btn_zoom_in); tools.addWidget(self.btn_zoom_out); tools.addWidget(self.btn_fit)
        right.addLayout(tools)

        # main preview (expands)
        self.scroll = QScrollArea(self); self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.preview)
        self.preview.setMouseTracking(True)
        self.preview.installEventFilter(self)
        self.scroll.viewport().installEventFilter(self)     
        self.scroll.installEventFilter(self)  
        self.scroll.horizontalScrollBar().installEventFilter(self)  # NEW
        self.scroll.verticalScrollBar().installEventFilter(self)    # NEW        
        right.addWidget(self.scroll, 1)

        # fixed-size grid
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(8); self.grid.setVerticalSpacing(8)
        self.grid.setContentsMargins(8, 8, 8, 8)

        self.thumb_size = QSize(220, 110)
        btn_w = self.thumb_size.width() + 2
        btn_h = self.thumb_size.height() + 2
        cols, rows = 4, 3

        for idx, name in enumerate(self.PALETTES):
            r, c = divmod(idx, cols)
            b = QPushButton("")  # we draw the text onto the icon itself
            b.setToolTip(name)
            b.setIconSize(self.thumb_size)
            b.setFixedSize(btn_w, btn_h)
            b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            b.clicked.connect(lambda _=None, n=name: self._on_palette_clicked(n))
            b.setStyleSheet("QPushButton{background:#222;border:1px solid #333;} QPushButton:hover{border-color:#555;}")
            self._thumb_buttons[name] = b
            self.grid.addWidget(b, r, c)

        grid_host = QWidget(self); grid_host.setLayout(self.grid)
        hspacing = self.grid.horizontalSpacing(); vspacing = self.grid.verticalSpacing()
        m = self.grid.contentsMargins()
        grid_w = cols*btn_w + (cols-1)*hspacing + m.left() + m.right()
        grid_h = rows*btn_h + (rows-1)*vspacing + m.top() + m.bottom()
        grid_host.setFixedSize(grid_w, grid_h)
        grid_host.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        right.addWidget(grid_host, 0, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.status = QLabel(""); right.addWidget(self.status, 0)

        right_host = QWidget(self); right_host.setLayout(right)
        root.addWidget(right_host, 1)

        self.setLayout(root)
        self.setMinimumSize(left_host.width() + grid_w + 48, max(560, grid_h + 200))

    # ---------- status helpers ----------
    def _set_status_label(self, which: str, text: str | None):
        lab = getattr(self, f"lbl_{which.lower()}")
        if text:
            lab.setText(text)
            lab.setStyleSheet("color:#2a7; font-weight:600; margin-left:8px;")
        else:
            lab.setText(f"No {which} loaded.")
            lab.setStyleSheet("color:#888; margin-left:8px;")

    # ------------- load by view/file -------------
    def _load_channel(self, which: str):
        src, ok = QInputDialog.getItem(
            self, f"Load {which}", "Source:", ["From View", "From File"], 0, False
        )
        if not ok:
            return

        if src == "From View":
            out = self._load_from_view(which)
        else:
            out = self._load_from_file(which)
        if out is None:
            return

        img, header, bit_depth, is_mono, path, label = out

        # NB channels → mono; OSC → RGB
        if which in ("Ha","OIII","SII"):
            if img.ndim == 3:
                img = img[:, :, 0]
        else:
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)

        # store raw, normalized
        setattr(self, which.lower(), self._as_float01(img))
        self._set_status_label(which, label)
        self.status.setText(f"{which} loaded ({'mono' if img.ndim==2 else 'RGB'}) shape={img.shape}")

        # build/clear stretched cache for this input
        self._cache_stretch(which)

        if self.current_palette is None:
            self.current_palette = "SHO"

    def _load_from_view(self, which):
        views = self._list_open_views()
        if not views:
            QMessageBox.warning(self, "No Views", "No open image views were found.")
            return None

        labels = [lab for lab, _ in views]
        choice, ok = QInputDialog.getItem(
            self, f"Select View for {which}", "Choose a view (by name):", labels, 0, False
        )
        if not ok or not choice:
            return None

        sw = dict(views)[choice]
        doc = getattr(sw, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.warning(self, "Empty View", "Selected view has no image.")
            return None

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
        if not path:
            return None
        img, header, bit_depth, is_mono = legacy_load_image(path)
        if img is None:
            QMessageBox.critical(self, "Load Error", f"Could not load {os.path.basename(path)}")
            return None
        label = f"From File: {os.path.basename(path)}"
        return img, header, bit_depth, is_mono, path, label

    def showEvent(self, e):
        super().showEvent(e)
        QTimer.singleShot(0, self._center_scrollbars)

    # ------------- build/caches -------------
    def _cache_stretch(self, which: str):
        """Compute and cache stretched version of a just-loaded input (if linear checked)."""
        arr = getattr(self, which.lower())
        if arr is None:
            self._stretched.pop(which, None); return
        if not self.chk_linear.isChecked():
            self._stretched.pop(which, None); return
        self._stretched[which] = self._stretch_input(arr)

    def _rebuild_stretch_cache_for_all(self, _state: int):
        """Rebuild (or clear) stretched cache for all loaded inputs when checkbox toggles."""
        for which in ("Ha","OIII","SII","OSC1","OSC2"):
            self._cache_stretch(which)

    def _render_thumb(self, name: str):
        base = self._thumb_base_pm.get(name)
        if base is None:
            return
        pm = base.copy()

        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = QFont("Helvetica", 10, QFont.Weight.DemiBold)
        p.setFont(font)
        fm = QFontMetrics(font)

        pad = 6
        strip_h = fm.height() + pad * 2
        strip = pm.rect().adjusted(0, pm.height() - strip_h, 0, 0)

        # translucent bottom strip
        p.fillRect(strip, QColor(0, 0, 0, 160))
        color = QColor(102, 255, 102) if self._selected_name == name else QColor(255, 255, 255)
        p.setPen(QPen(color))
        p.drawText(strip, Qt.AlignmentFlag.AlignCenter, name)
        p.end()

        btn = self._thumb_buttons[name]
        btn.setIcon(QIcon(pm))
        btn.setIconSize(self.thumb_size)  # <- ensures no clipping

    # ------------- thumbnails -------------
    def _create_palettes(self):
        """
        Build the 12 palette thumbnails from a **center crop of the stretched inputs**
        and draw the palette name directly on each thumbnail. Names turn green when selected.
        """
        ha, oo, si = self._prepared_channels(for_thumbs=True)
        if oo is None or (ha is None and si is None):
            QMessageBox.warning(self, "Need Channels", "Load at least OIII + (Ha or SII).")
            return

        built = 0
        for name in self.PALETTES:
            r, g, b = self._map_channels_or_special(name, ha, oo, si)
            if any(ch is None for ch in (r, g, b)):
                self._thumb_base_pm.pop(name, None)
                self._thumb_buttons[name].setIcon(QIcon())
                continue

            r = np.clip(np.nan_to_num(r), 0, 1)
            g = np.clip(np.nan_to_num(g), 0, 1)
            b = np.clip(np.nan_to_num(b), 0, 1)
            rgb = np.stack([r, g, b], axis=2).astype(np.float32)

            # scale the thumbnail to EXACTLY the button icon size first
            pm = QPixmap.fromImage(self._to_qimage(rgb)).scaled(
                self.thumb_size, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self._thumb_base_pm[name] = pm
            self._render_thumb(name) 
            built += 1

        self.status.setText(f"Created {built} palette previews.")


    def _on_palette_clicked(self, name: str):
        self._selected_name = name
        for n in self.PALETTES:
            self._render_thumb(n)
        self.current_palette = name
        self._generate_for_palette(name)

    # ------------- palette build helpers -------------
    def _center_crop(self, img: np.ndarray, side: int) -> np.ndarray:
        """Center-crop to a square of size 'side' (no upscaling)."""
        h, w = img.shape[:2]; s = min(side, h, w)
        y0 = (h - s) // 2; x0 = (w - s) // 2
        return img[y0:y0+s, x0:x0+s] if img.ndim == 2 else img[y0:y0+s, x0:x0+s, :]

    def _center_crop_all_to_side(self, side: int, *imgs):
        """Center-crop all provided images to the same square side (no upscaling)."""
        s = None
        for im in imgs:
            if im is None: continue
            h, w = im.shape[:2]
            s = min(side, h, w) if s is None else min(s, h, w, side)
        if s is None: s = side
        return [self._center_crop(im, s) if im is not None else None for im in imgs], s

    def _prepared_channels(self, for_thumbs: bool = False):
        """
        Build Ha/OIII/SII bases from inputs. If 'Linear input' is checked,
        **use stretched versions** (cached). Then optionally center-crop for thumbnails.
        """
        # choose raw vs stretched
        def pick(name):
            if self.chk_linear.isChecked() and (name in self._stretched):
                return self._stretched[name]
            return getattr(self, name.lower())

        ha = pick("Ha")
        oo = pick("OIII")
        si = pick("SII")
        o1 = pick("OSC1")
        o2 = pick("OSC2")

        # synthesize from stretched OSC first (stretch-before-crop)
        if o1 is not None:  # OSC1: R≈Ha, mean(G,B)≈OIII
            h1 = o1[..., 0]
            g1b1 = o1[..., 1:3].mean(axis=2)
            ha = h1 if ha is None else 0.5*ha + 0.5*h1
            oo = g1b1 if oo is None else 0.5*oo + 0.5*g1b1

        if o2 is not None:  # OSC2: R≈SII, mean(G,B)≈OIII
            s2 = o2[..., 0]
            g2b2 = o2[..., 1:3].mean(axis=2)
            si = s2 if si is None else 0.5*si + 0.5*s2
            oo = g2b2 if oo is None else 0.5*oo + 0.5*g2b2

        # shapes must match for full-size
        shapes = [x.shape for x in (ha, oo, si) if x is not None]
        if len(shapes) and len(set(shapes)) > 1 and not for_thumbs:
            QMessageBox.critical(self, "Size Mismatch", f"Channel sizes differ: {set(shapes)}")
            return None, None, None

        # thumbnails: crop AFTER stretch/synth
        if for_thumbs:
            (ha, oo, si), _ = self._center_crop_all_to_side(self.THUMB_CROP, ha, oo, si)

        return ha, oo, si

    def _generate_for_palette(self, pal: str):
        ha, oo, si = self._prepared_channels()
        if oo is None or (ha is None and si is None):
            return

        r,g,b = self._map_channels_or_special(pal, ha, oo, si)
        if any(ch is None for ch in (r,g,b)):
            QMessageBox.critical(self, "Palette Error", f"Could not build palette {pal}."); return

        r = np.clip(np.nan_to_num(r), 0, 1)
        g = np.clip(np.nan_to_num(g), 0, 1)
        b = np.clip(np.nan_to_num(b), 0, 1)
        rgb = np.stack([r,g,b], axis=2).astype(np.float32)

        mx = float(rgb.max()) or 1.0
        self.final = (rgb / mx).astype(np.float32)

        self._set_preview_image(self._to_qimage(self.final))
        self.status.setText(f"Preview generated: {pal}")

    def _set_preview_image(self, qimg: QImage):
        self._base_pm = QPixmap.fromImage(qimg)
        self._zoom = 1.0
        self._update_preview_pixmap()
        QTimer.singleShot(0, self._center_scrollbars)

    def _update_preview_pixmap(self):
        if self._base_pm is None:
            return
        # explicit int size (QSize * float can crash on some PyQt6 builds)
        base_sz = self._base_pm.size()
        w = max(1, int(base_sz.width() * self._zoom))
        h = max(1, int(base_sz.height() * self._zoom))
        scaled = self._base_pm.scaled(
            w, h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview.setPixmap(scaled)
        self.preview.resize(scaled.size())

    def _set_zoom(self, new_zoom: float):
        self._zoom = max(self._min_zoom, min(self._max_zoom, new_zoom))
        self._update_preview_pixmap()

    def _zoom_at(self, factor: float = 1.25, anchor_vp: QPoint | None = None):
        if self._base_pm is None:
            return

        vp = self.scroll.viewport()
        if anchor_vp is None:
            anchor_vp = QPoint(vp.width() // 2, vp.height() // 2)  # view center

        # label coords under the anchor *before* zoom
        lbl_before = self.preview.mapFrom(vp, anchor_vp)

        old_zoom = self._zoom
        new_zoom = max(self._min_zoom, min(self._max_zoom, old_zoom * factor))
        ratio = new_zoom / max(old_zoom, 1e-6)
        if abs(ratio - 1.0) < 1e-6:
            return

        # apply zoom (updates label size & scrollbar ranges)
        self._zoom = new_zoom
        self._update_preview_pixmap()

        # desired label coords *after* zoom
        lbl_after_x = int(lbl_before.x() * ratio)
        lbl_after_y = int(lbl_before.y() * ratio)

        # move scrollbars so anchor_vp keeps the same content point
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), lbl_after_x - anchor_vp.x())))
        vbar.setValue(max(vbar.minimum(), min(vbar.maximum(), lbl_after_y - anchor_vp.y())))


    def _fit_to_preview(self):
        if self._base_pm is None:
            return
        vp = self.scroll.viewport().size()
        pm = self._base_pm.size()
        if pm.width() == 0 or pm.height() == 0:
            return
        k = min(vp.width() / pm.width(), vp.height() / pm.height())
        self._set_zoom(max(self._min_zoom, min(self._max_zoom, k)))
        self._center_scrollbars()

    def _center_scrollbars(self):
        # center the view on the image
        h = self.scroll.horizontalScrollBar()
        v = self.scroll.verticalScrollBar()
        h.setValue((h.maximum() + h.minimum()) // 2)
        v.setValue((v.maximum() + v.minimum()) // 2)

    def _map_channels_or_special(self, name, ha, oo, si):
        # substitution
        if ha is None and si is not None: ha = si
        if si is None and ha is not None: si = ha

        basic = {
            "SHO": (si, ha, oo),
            "HOO": (ha, oo, oo),
            "HSO": (ha, si, oo),
            "HOS": (ha, oo, si),
            "OSS": (oo, si, si),
            "OHH": (oo, ha, ha),
            "OSH": (oo, si, ha),
            "OHS": (oo, ha, si),
            "HSS": (ha, si, si),
        }
        if name in basic:
            return basic[name]

        try:
            if name == "Realistic1":
                r = (ha + si)/2 if (ha is not None and si is not None) else (ha if ha is not None else 0)
                g = 0.3*(ha if ha is not None else 0) + 0.7*(oo if oo is not None else 0)
                b = 0.9*(oo if oo is not None else 0) + 0.1*(ha if ha is not None else 0)
                return r,g,b
            if name == "Realistic2":
                r = 0.7*(ha if ha is not None else 0) + 0.3*(si if si is not None else 0)
                g = 0.3*(si if si is not None else 0) + 0.7*(oo if oo is not None else 0)
                b = (oo if oo is not None else 0)
                return r,g,b
            if name == "Foraxx":
                if ha is not None and oo is not None and si is None:
                    r = ha; b = oo
                    t = ha * oo
                    g = (t**(1 - t))*ha + (1 - (t**(1 - t)))*oo
                    return r,g,b
                if ha is not None and oo is not None and si is not None:
                    t = np.clip(oo, 1e-6, 1.0)**(1 - np.clip(oo, 1e-6, 1.0))
                    r = t*si + (1 - t)*ha
                    t2 = ha * oo
                    g = (t2**(1 - t2))*ha + (1 - (t2**(1 - t2)))*oo
                    b = oo
                    return r,g,b
                return basic["SHO"]
        except Exception:
            return basic.get("SHO", (ha, oo, si))

        return basic.get("SHO", (ha, oo, si))

    # ------------- push to new subwindow -------------
    def _push_final(self):
        if self.final is None:
            QMessageBox.warning(self, "No Image", "Generate a palette first."); return
        mw = self._find_main_window()
        dm = getattr(mw, "docman", None)
        if not mw or not dm:
            QMessageBox.critical(self, "UI", "Main window or DocManager not available."); return
        title = self.current_palette or "Palette"
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
            self.status.setText("Opened final palette in a new view.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open new view:\n{e}")

    # ------------- utilities -------------
    def _clear_channels(self):
        self.ha = self.oiii = self.sii = self.osc1 = self.osc2 = None
        self._stretched.clear()
        self.final = None
        self.preview.clear()
        for which in ("Ha","OIII","SII","OSC1","OSC2"):
            self._set_status_label(which, None)
        for name, b in self._thumb_buttons.items():
            b.setIcon(QIcon())
        self._thumb_base_pm.clear()
        self._selected_name = None
        for b in self._thumb_buttons.values():
            b.setIcon(QIcon())
        self.status.setText("Cleared all loaded channels.")

    def _as_float01(self, arr):
        a = np.asarray(arr)
        if a.dtype == np.uint8:  return a.astype(np.float32)/255.0
        if a.dtype == np.uint16: return a.astype(np.float32)/65535.0
        return np.clip(a.astype(np.float32), 0.0, 1.0)

    def _stretch_input(self, img):
        """Run statistical stretch on mono or color inputs (target_median=0.25)."""
        if img.ndim == 2:
            return np.clip(stretch_mono_image(img, target_median=0.25), 0.0, 1.0)
        if img.ndim == 3 and img.shape[2] == 3:
            return np.clip(stretch_color_image(img, target_median=0.25, linked=False), 0.0, 1.0)
        if img.ndim == 3 and img.shape[2] == 1:
            mono = img[...,0]
            return np.clip(stretch_mono_image(mono, target_median=0.25), 0.0, 1.0)
        return img

    def _to_qimage(self, arr):
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
        if not mw:
            return []
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
    
    def eventFilter(self, obj, ev):
        # Ctrl+wheel = zoom at mouse (no scrolling). Wheel without Ctrl = eaten.
        if ev.type() == QEvent.Type.Wheel and (
            obj is self.preview
            or obj is self.scroll
            or obj is self.scroll.viewport()
            or obj is self.scroll.horizontalScrollBar()
            or obj is self.scroll.verticalScrollBar()
        ):
            # always stop the wheel from scrolling
            ev.accept()

            # Zoom only when Ctrl is held
            if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                factor = 1.25 if ev.angleDelta().y() > 0 else 0.8

                # Get mouse position in global screen coords and map into the viewport
                vp = self.scroll.viewport()
                anchor_vp = vp.mapFromGlobal(ev.globalPosition().toPoint())

                # Clamp to viewport rect (robust if the event originated on scrollbars)
                r = vp.rect()
                if not r.contains(anchor_vp):
                    anchor_vp.setX(max(r.left(),  min(r.right(),  anchor_vp.x())))
                    anchor_vp.setY(max(r.top(),   min(r.bottom(), anchor_vp.y())))

                self._zoom_at(factor, anchor_vp)
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