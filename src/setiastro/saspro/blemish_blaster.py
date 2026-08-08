# pro/blemish_blaster.py
from __future__ import annotations
import math
import numpy as np
from typing import Optional
from PyQt6.QtCore import Qt, QEvent, QPointF, QRunnable, QThreadPool, pyqtSlot, QObject, pyqtSignal, QSettings, QTimer, QByteArray
from PyQt6.QtGui import QImage, QPixmap, QPen, QBrush, QAction, QKeySequence, QColor, QWheelEvent, QIcon
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel, QPushButton, QSlider, QSizePolicy, QSpinBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem, QMessageBox, QScrollArea, QCheckBox, QDoubleSpinBox, QWidget, QFrame    
)
from setiastro.saspro.imageops.stretch import stretch_color_image, stretch_mono_image 

from dataclasses import dataclass
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn


@dataclass
class BlemishOp:
    x: int
    y: int
    radius: int
    feather: float
    opacity: float
    channels: list[int]


# ──────────────────────────────────────────────────────────────────────────────
# Worker
# ──────────────────────────────────────────────────────────────────────────────

class _BBWorkerSignals(QObject):
    finished = pyqtSignal(np.ndarray)

class _BlemishWorker(QRunnable):
    def __init__(self, image: np.ndarray, x: int, y: int, radius: int, feather: float, opacity: float,
                 channels_to_process: list[int]):
        super().__init__()
        self.image = image.copy()
        self.x, self.y = int(x), int(y)
        self.radius = int(radius)
        self.feather = float(feather)
        self.opacity = float(opacity)
        self.channels_to_process = channels_to_process
        self.signals = _BBWorkerSignals()
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass  # older PyQt6 versions
    @pyqtSlot()
    def run(self):
        out = self._remove_blemish(
            self.image, self.x, self.y, self.radius, self.feather, self.opacity, self.channels_to_process
        )
        self.signals.finished.emit(out)

    # ── the exact SASv2 logic (minor tidy) ────────────────────────────────────
    def _remove_blemish(self, image, x, y, radius, feather, opacity, channels_to_process):
        corrected_image = image.copy()
        h, w = image.shape[:2]

        # 6 neighbors
        angles = [0, 60, 120, 180, 240, 300]
        centers = []
        for ang in angles:
            r = math.radians(ang)
            dx = int(math.cos(r) * (radius * 1.5))
            dy = int(math.sin(r) * (radius * 1.5))
            centers.append((x + dx, y + dy))

        tgt_median = self._median_circle(image, x, y, radius, channels_to_process)
        neigh_medians = [self._median_circle(image, cx, cy, radius, channels_to_process) for (cx, cy) in centers]

        diffs = [abs(m - tgt_median) for m in neigh_medians]
        idxs = np.argsort(diffs)[:3]
        sel_centers = [centers[i] for i in idxs]

        for c in channels_to_process:
            for i in range(max(y - radius, 0), min(y + radius + 1, h)):
                yi = i - y
                for j in range(max(x - radius, 0), min(x + radius + 1, w)):
                    xj = j - x
                    dist = math.hypot(xj, yi)
                    if dist > radius:
                        continue

                    weight = 1.0 if feather <= 0 else max(0.0, min(1.0, (radius - dist) / (radius * feather)))

                    samples = []
                    for (cx, cy) in sel_centers:
                        sj = j + (cx - x)
                        si = i + (cy - y)
                        if 0 <= si < h and 0 <= sj < w:
                            if image.ndim == 2:
                                samples.append(image[si, sj])
                            elif image.ndim == 3 and image.shape[2] == 1:
                                samples.append(image[si, sj, 0])
                            elif image.ndim == 3 and c < image.shape[2]:
                                samples.append(image[si, sj, c])

                    if samples:
                        median_val = float(np.median(samples))
                    else:
                        if image.ndim == 2:
                            median_val = float(image[i, j])
                        elif image.ndim == 3 and image.shape[2] == 1:
                            median_val = float(image[i, j, 0])
                        else:
                            median_val = float(image[i, j, c])

                    if image.ndim == 2:
                        orig = float(image[i, j])
                        corrected_image[i, j] = (1 - opacity * weight) * orig + (opacity * weight) * median_val
                    elif image.ndim == 3 and image.shape[2] == 1:
                        orig = float(image[i, j, 0])
                        corrected_image[i, j, 0] = (1 - opacity * weight) * orig + (opacity * weight) * median_val
                    elif image.ndim == 3 and c < image.shape[2]:
                        orig = float(image[i, j, c])
                        corrected_image[i, j, c] = (1 - opacity * weight) * orig + (opacity * weight) * median_val

        return corrected_image

    def _median_circle(self, image, cx, cy, radius, channels):
        vals = []
        y0 = max(cy - radius, 0); y1 = min(cy + radius + 1, image.shape[0])
        x0 = max(cx - radius, 0); x1 = min(cx + radius + 1, image.shape[1])
        if y0 >= y1 or x0 >= x1:
            return 0.0

        for c in channels:
            if image.ndim == 2:
                roi = image[y0:y1, x0:x1]
            elif image.ndim == 3:
                if image.shape[2] == 1:
                    roi = image[y0:y1, x0:x1, 0]
                elif c < image.shape[2]:
                    roi = image[y0:y1, x0:x1, c]
                else:
                    continue
            else:
                continue

            yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
            mask = (xx - (cx - x0))**2 + (yy - (cy - y0))**2 <= radius**2
            vals.extend(roi[mask].ravel())

        return float(np.median(vals)) if len(vals) else 0.0


class _LineDefectWorker(QRunnable):
    """Linear Blaster: heal a straight defect (satellite trail, cosmic ray) by
    sampling parallel lines offset perpendicular to the defect."""
    def __init__(self, image, p0, p1, mid, thickness, feather, opacity,
                 channels_to_process, full_line):
        super().__init__()
        self.image = image.copy()
        self.p0 = (int(p0[0]), int(p0[1]))
        self.p1 = (int(p1[0]), int(p1[1]))
        self.mid = (float(mid[0]), float(mid[1])) if mid is not None else None
        self.thickness = int(thickness)
        self.feather = float(feather)
        self.opacity = float(opacity)
        self.channels_to_process = channels_to_process
        self.full_line = bool(full_line)
        self.signals = _BBWorkerSignals()
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass

    @pyqtSlot()
    def run(self):
        out = self._remove_linear_defect(
            self.image, self.p0, self.p1, self.mid, self.thickness,
            self.feather, self.opacity, self.channels_to_process, self.full_line
        )
        self.signals.finished.emit(out)

    def _remove_linear_defect(self, image, p0, p1, mid, thickness, feather, opacity,
                              channels, full_line):
        import math
        import numpy as np

        corrected = image.copy()
        H, W = image.shape[:2]
        ax, ay = float(p0[0]), float(p0[1])
        bx, by = float(p1[0]), float(p1[1])
        if math.hypot(bx - ax, by - ay) < 1e-6:
            return corrected

        # ---- centerline vertices: straight, or a bent quadratic Bézier ----
        bend = mid is not None and (abs(mid[0] - 0.5 * (ax + bx)) > 0.5
                                    or abs(mid[1] - 0.5 * (ay + by)) > 0.5)
        if bend:
            # control point so the curve passes through the handle at t=0.5
            Mx = 2.0 * float(mid[0]) - 0.5 * (ax + bx)
            My = 2.0 * float(mid[1]) - 0.5 * (ay + by)
            arclen = math.hypot(Mx - ax, My - ay) + math.hypot(bx - Mx, by - My)
            n = int(min(64, max(4, arclen / 24.0)))
            verts = []
            for k in range(n + 1):
                t = k / float(n)
                verts.append(((1 - t) ** 2 * ax + 2 * (1 - t) * t * Mx + t * t * bx,
                              (1 - t) ** 2 * ay + 2 * (1 - t) * t * My + t * t * by))
            tanA = (Mx - ax, My - ay)
            tanB = (bx - Mx, by - My)
        else:
            verts = [(ax, ay), (bx, by)]
            tanA = tanB = (bx - ax, by - ay)

        # ---- full-line: straight extensions along the end tangents to the border ----
        if full_line:
            eA = self._ray_rect_exit(ax, ay, -tanA[0], -tanA[1], W, H)
            eB = self._ray_rect_exit(bx, by, tanB[0], tanB[1], W, H)
            if eA is not None:
                verts = [eA] + verts
            if eB is not None:
                verts = verts + [eB]

        # ---- blast each sub-segment into a shared buffer (keep-strongest heal) ----
        alpha = np.zeros((H, W), np.float32)
        T = float(max(1, thickness))
        nseg = len(verts) - 1
        for i in range(nseg):
            cap_start = (i == 0) and (not full_line)
            cap_end   = (i == nseg - 1) and (not full_line)
            self._blast_one_segment(image, corrected, alpha,
                                    verts[i], verts[i + 1], T, feather, opacity,
                                    channels, cap_start, cap_end)
        return corrected

    def _blast_one_segment(self, image, corrected, alpha, P0, P1, T, feather, opacity,
                           channels, cap_start, cap_end):
        import math
        import numpy as np
        import cv2

        H, W = image.shape[:2]
        p0x, p0y = float(P0[0]), float(P0[1])
        p1x, p1y = float(P1[0]), float(P1[1])
        dx, dy = p1x - p0x, p1y - p0y
        Lseg = math.hypot(dx, dy)
        if Lseg < 1e-6:
            return
        ux, uy = dx / Lseg, dy / Lseg
        nx, ny = -uy, ux
        hw = max(0.5, 0.5 * T)
        overlap = max(2.0, hw)           # flat joints overlap so they don't seam
        OFFS = np.array([T, 2 * T, 3 * T, -T, -2 * T, -3 * T], dtype=np.float32)

        pad = hw + overlap + 2.0
        x0 = max(0, int(math.floor(min(p0x, p1x) - pad))); x1 = min(W, int(math.ceil(max(p0x, p1x) + pad)))
        y0 = max(0, int(math.floor(min(p0y, p1y) - pad))); y1 = min(H, int(math.ceil(max(p0y, p1y) + pad)))
        if x0 >= x1 or y0 >= y1:
            return

        jj, ii = np.meshgrid(np.arange(x0, x1, dtype=np.float32),
                             np.arange(y0, y1, dtype=np.float32))
        rx, ry = jj - p0x, ii - p0y
        t = rx * ux + ry * uy
        s = rx * nx + ry * ny
        abss = np.abs(s)

        # distance-to-band: rounded caps at true ends, flat (extended) at joints
        d = abss.copy()
        before = t < 0.0
        after  = t > Lseg
        if cap_start:
            d = np.where(before, np.hypot(jj - p0x, ii - p0y), d)
        else:
            d = np.where(before & (t < -overlap), np.inf, d)
        if cap_end:
            d = np.where(after, np.hypot(jj - p1x, ii - p1y), d)
        else:
            d = np.where(after & (t > Lseg + overlap), np.inf, d)

        band = d <= hw
        if not band.any():
            return

        if feather <= 0:
            wv = band.astype(np.float32)
        else:
            wv = np.clip((hw - d) / (hw * feather), 0.0, 1.0).astype(np.float32) * band.astype(np.float32)
        a_seg = (opacity * wv).astype(np.float32)

        cur = alpha[y0:y1, x0:x1]
        upd = a_seg > cur
        if not upd.any():
            return

        t_lo = 0.0 if cap_start else -overlap
        t_hi = Lseg if cap_end else (Lseg + overlap)
        nT = max(1, int(math.ceil(t_hi - t_lo)) + 1)
        col = np.clip(np.round(t - t_lo).astype(np.int32), 0, nT - 1)
        tcols = t_lo + np.arange(nT, dtype=np.float32)
        cxs = p0x + tcols * ux
        cys = p0y + tcols * uy
        nS = max(1, int(round(2 * hw)) + 1)
        srel = np.arange(nS, dtype=np.float32) - (nS - 1) * 0.5
        MX = cxs[None, :] + srel[:, None] * nx
        MY = cys[None, :] + srel[:, None] * ny

        is_mono = (image.ndim == 2)
        n_ch = 1 if is_mono else image.shape[2]

        for c in channels:
            if is_mono:
                plane = image
            elif n_ch == 1:
                plane = image[..., 0]
            elif c < n_ch:
                plane = image[..., c]
            else:
                continue
            plane = np.ascontiguousarray(plane, dtype=np.float32)

            tgt = cv2.remap(plane, MX, MY, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT101)
            tgt_med = np.median(tgt, axis=0)
            off_med = np.empty((6, nT), np.float32)
            for oi in range(6):
                osamp = cv2.remap(plane, MX + OFFS[oi] * nx, MY + OFFS[oi] * ny,
                                  interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT101)
                off_med[oi] = np.median(osamp, axis=0)
            order = np.argsort(np.abs(off_med - tgt_med[None, :]), axis=0)[:3, :]
            sel_px = OFFS[order][:, col]

            samp = np.empty((3,) + jj.shape, np.float32)
            for k in range(3):
                samp[k] = cv2.remap(plane,
                                    (jj + sel_px[k] * nx).astype(np.float32),
                                    (ii + sel_px[k] * ny).astype(np.float32),
                                    interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT101)
            healed = np.median(samp, axis=0)

            orig = plane[y0:y1, x0:x1]
            out = (1.0 - a_seg) * orig + a_seg * healed
            if is_mono:
                blk = corrected[y0:y1, x0:x1]
            elif n_ch == 1:
                blk = corrected[y0:y1, x0:x1, 0]
            else:
                blk = corrected[y0:y1, x0:x1, c]
            blk[upd] = out[upd]

        alpha[y0:y1, x0:x1] = np.where(upd, a_seg, cur)

    @staticmethod
    def _ray_rect_exit(px, py, dx, dy, W, H):
        """Point where the ray (px,py)+t·(dx,dy), t>0, exits the image rect."""
        import math
        L = math.hypot(dx, dy)
        if L < 1e-9:
            return None
        ux, uy = dx / L, dy / L
        ts = []
        if abs(ux) > 1e-9:
            for xb in (0.0, W - 1.0):
                t = (xb - px) / ux
                if t > 1e-6 and -1.0 <= (py + t * uy) <= H:
                    ts.append(t)
        if abs(uy) > 1e-9:
            for yb in (0.0, H - 1.0):
                t = (yb - py) / uy
                if t > 1e-6 and -1.0 <= (px + t * ux) <= W:
                    ts.append(t)
        if not ts:
            return None
        t = min(ts)
        return (px + t * ux, py + t * uy)

    @staticmethod
    def _chord_trange(ax, ay, ux, uy, W, H):
        """Arc-length range where the infinite line A + t*u crosses the image rect."""
        import math
        cand = []
        if abs(ux) > 1e-9:
            for xb in (0.0, W - 1.0):
                t = (xb - ax) / ux; y = ay + t * uy
                if -1.0 <= y <= H:
                    cand.append(t)
        if abs(uy) > 1e-9:
            for yb in (0.0, H - 1.0):
                t = (yb - ay) / uy; x = ax + t * ux
                if -1.0 <= x <= W:
                    cand.append(t)
        if not cand:
            return None, None
        return min(cand), max(cand)

# ──────────────────────────────────────────────────────────────────────────────
# Dialog
# ──────────────────────────────────────────────────────────────────────────────

class BlemishBlasterDialogPro(QDialog):
    """
    Interactive blemish remover (preview + click to heal) that writes back to the
    provided document when 'Apply' is pressed.
    """
    def __init__(self, parent, doc):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Blemish Blaster"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.setMinimumSize(900, 650)
        self._did_initial_fit = False

        self._doc = doc
        base = getattr(doc, "image", None)
        if base is None:
            raise RuntimeError("Document has no image.")

        # normalize to float32 [0..1]
        self._orig_shape = base.shape
        self._orig_mono = (base.ndim == 2) or (base.ndim == 3 and base.shape[2] == 1)

        img = np.asarray(base, dtype=np.float32)
        if img.dtype.kind in "ui":
            # Best-effort normalize
            maxv = float(np.nanmax(img)) or 1.0
            img = img / max(1.0, maxv)
        img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)

        # display buffer is 3-channels for visualization
        if img.ndim == 2:
            img3 = np.repeat(img[:, :, None], 3, axis=2)
        elif img.ndim == 3 and img.shape[2] == 1:
            img3 = np.repeat(img, 3, axis=2)
        elif img.ndim == 3 and img.shape[2] >= 3:
            img3 = img[:, :, :3]
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        self._image   = img3.copy()      # linear, edited by worker
        self._display = self._image.copy()

        # ── Scene/View (unchanged) ─────────────────────────────────────────
        self.scene = QGraphicsScene(self)
        self.view  = QGraphicsView(self.scene)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pix   = QGraphicsPixmapItem()
        self.scene.addItem(self.pix)

        self.circle = QGraphicsEllipseItem()
        self.circle.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine))
        self.circle.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.circle.setVisible(False)
        self.scene.addItem(self.circle)

        # ── Linear Blaster overlay items ──────────────────────────
        from PyQt6.QtWidgets import QGraphicsLineItem, QGraphicsPathItem
        self._mode = 0                  # 0 = Blemish, 1 = Linear
        self._lin_pts = []              # scene-coord endpoints (0, 1, or 2 QPointF)
        self._lin_mid = None            # bend control handle (QPointF), set once 2 pts exist
        self._lin_drag = None           # 0/1 = endpoint, 2 = mid handle, or None
        self._lin_line = QGraphicsPathItem()   # centerline (straight or bent)
        self._lin_line.setPen(QPen(QColor(0, 200, 255), 2, Qt.PenStyle.DashLine))
        self._lin_line.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._lin_line.setVisible(False)
        self.scene.addItem(self._lin_line)
        self._lin_mid_handle = QGraphicsEllipseItem()
        self._lin_mid_handle.setPen(QPen(QColor(120, 255, 120), 2))
        self._lin_mid_handle.setBrush(QBrush(QColor(120, 255, 120, 80)))
        self._lin_mid_handle.setVisible(False)
        self.scene.addItem(self._lin_mid_handle)
        self._lin_ext = QGraphicsLineItem()   # dim extension for full-line preview
        self._lin_ext.setPen(QPen(QColor(0, 200, 255, 110), 1, Qt.PenStyle.DotLine))
        self._lin_ext.setVisible(False)
        self.scene.addItem(self._lin_ext)
        from PyQt6.QtWidgets import QGraphicsPathItem
        self._lin_band = QGraphicsPathItem()  # thickness outline (what gets blasted)
        self._lin_band.setPen(QPen(QColor(255, 80, 80), 1, Qt.PenStyle.DashLine))
        self._lin_band.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._lin_band.setVisible(False)
        self.scene.addItem(self._lin_band)
        self._lin_handles = []
        for _ in range(2):
            hnd = QGraphicsEllipseItem()
            hnd.setPen(QPen(QColor(0, 200, 255), 2))
            hnd.setBrush(QBrush(QColor(0, 200, 255, 70)))
            hnd.setVisible(False)
            self.scene.addItem(hnd)
            self._lin_handles.append(hnd)

        # scroll container
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.view)
        self.scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # --- Zoom controls (buttons) ---------------------------------
        # --- Zoom controls (standard themed toolbuttons) ---------------
        self._zoom = 1.0  # initial zoom factor

        self.btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        self.btn_zoom_in  = themed_toolbtn("zoom-in", "Zoom In")
        self.btn_zoom_fit = themed_toolbtn("zoom-fit-best", "Fit to Preview")


        # ── Controls
        ctrls = QGroupBox(self.tr("Controls"))
        form  = QFormLayout(ctrls)

        # existing sliders (now each paired with a live spinbox)
        self.s_radius  = QSlider(Qt.Orientation.Horizontal); self.s_radius.setRange(1, 900); self.s_radius.setValue(12)
        self.s_feather = QSlider(Qt.Orientation.Horizontal); self.s_feather.setRange(0, 100); self.s_feather.setValue(50)
        self.s_opacity = QSlider(Qt.Orientation.Horizontal); self.s_opacity.setRange(0, 100); self.s_opacity.setValue(100)

        self.sb_radius  = self._mk_slider_spin(self.s_radius,  suffix=" px")
        self.sb_feather = self._mk_slider_spin(self.s_feather, suffix=" %")
        self.sb_opacity = self._mk_slider_spin(self.s_opacity, suffix=" %")

        form.addRow(self.tr("Radius:"),  self._slider_spin_row(self.s_radius,  self.sb_radius))
        form.addRow(self.tr("Feather:"), self._slider_spin_row(self.s_feather, self.sb_feather))
        form.addRow(self.tr("Opacity:"), self._slider_spin_row(self.s_opacity, self.sb_opacity))

        # ── Mode + Linear Blaster controls ────────────────────────
        from PyQt6.QtWidgets import QRadioButton, QButtonGroup
        self.rb_blemish = QRadioButton(self.tr("Blemish Blaster"))
        self.rb_linear  = QRadioButton(self.tr("Linear Blaster"))
        self.rb_blemish.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.rb_blemish, 0)
        self._mode_group.addButton(self.rb_linear, 1)
        _mode_row = QHBoxLayout()
        _mode_row.addWidget(self.rb_blemish); _mode_row.addWidget(self.rb_linear)
        _mode_row.addStretch(1)
        _mode_w = QWidget(self); _mode_w.setLayout(_mode_row)
        form.addRow(self.tr("Mode:"), _mode_w)

        self.lbl_thickness = QLabel(self.tr("Thickness:"))
        self.s_thickness = QSlider(Qt.Orientation.Horizontal)
        self.s_thickness.setRange(1, 100); self.s_thickness.setValue(10)
        self.sb_thickness = self._mk_slider_spin(self.s_thickness, suffix=" px")
        self._thickness_row = self._slider_spin_row(self.s_thickness, self.sb_thickness)
        form.addRow(self.lbl_thickness, self._thickness_row)
        # keep the outline in sync while dragging thickness
        self.s_thickness.valueChanged.connect(lambda _=None: self._lin_refresh())

        self.rb_extent_seg  = QRadioButton(self.tr("Segment"))
        self.rb_extent_full = QRadioButton(self.tr("Full line across image"))
        self.rb_extent_seg.setChecked(True)
        self._extent_group = QButtonGroup(self)
        self._extent_group.addButton(self.rb_extent_seg, 0)
        self._extent_group.addButton(self.rb_extent_full, 1)
        _ext_row = QHBoxLayout()
        _ext_row.addWidget(self.rb_extent_seg); _ext_row.addWidget(self.rb_extent_full)
        _ext_row.addStretch(1)
        self.ext_w = QWidget(self); self.ext_w.setLayout(_ext_row)
        self.lbl_extent = QLabel(self.tr("Blast extent:"))
        form.addRow(self.lbl_extent, self.ext_w)

        self.btn_blast_line = QPushButton(self.tr("⚡ Blast Line"))
        self.btn_blast_line.setEnabled(False)
        form.addRow("", self.btn_blast_line)

        # start hidden (Blemish mode is default)
        for _w in (self.lbl_thickness, self._thickness_row,
                   self.lbl_extent, self.ext_w, self.btn_blast_line):
            _w.setVisible(False)

        self.rb_blemish.toggled.connect(self._on_mode_changed)
        self.rb_extent_full.toggled.connect(lambda _=None: self._lin_refresh())
        self.btn_blast_line.clicked.connect(self._blast_line)
        self.lbl_bracket_help = QLabel(self.tr("Tip: Use [ and ] to decrease/increase brush size."))
        self.lbl_bracket_help.setStyleSheet("color:#888;")
        self.lbl_bracket_help.setWordWrap(True)
        form.addRow(self.lbl_bracket_help)
        # --- PREVIEW AUTOSTRETCH (display only) ---
        self.cb_autostretch = QCheckBox(self.tr("Auto-stretch preview"))
        self.cb_autostretch.setChecked(True)
        form.addRow(self.cb_autostretch)

        self.s_target_median = QDoubleSpinBox()
        self.s_target_median.setRange(0.01, 0.60)
        self.s_target_median.setSingleStep(0.01)
        self.s_target_median.setDecimals(3)
        self.s_target_median.setValue(0.25)
        form.addRow(self.tr("Target median:"), self.s_target_median)

        self.cb_linked = QCheckBox(self.tr("Linked color channels"))
        self.cb_linked.setChecked(True)
        form.addRow(self.cb_linked)

        # react to UI
        self.cb_autostretch.toggled.connect(self._update_display_autostretch)
        self.s_target_median.valueChanged.connect(self._update_display_autostretch)
        self.cb_linked.toggled.connect(self._update_display_autostretch)
        # (nice-to-have: disable fields when off)
        self.cb_autostretch.toggled.connect(lambda on: (self.s_target_median.setEnabled(on),
                                                        self.cb_linked.setEnabled(on)))

        # buttons / layout (unchanged)
        bb = QHBoxLayout()

        self.btn_undo  = QPushButton(self.tr("Undo"))
        self.btn_redo  = QPushButton(self.tr("Redo"))
        self.btn_apply = QPushButton(self.tr("Apply to Document"))
        self.btn_close = QPushButton(self.tr("Close"))

        self.btn_undo.setEnabled(False)
        self.btn_redo.setEnabled(False)

        bb.addStretch()
        bb.addWidget(self.btn_undo)
        bb.addWidget(self.btn_redo)
        bb.addSpacing(12)
        bb.addWidget(self.btn_apply)
        bb.addWidget(self.btn_close)

        # ─────────────────────────────────────────────────────────────
        # Layout: Left = Controls, Right = Preview (so preview gets height)
        # ─────────────────────────────────────────────────────────────
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        # ---- LEFT: Zoom + Controls + Buttons (scrollable on small screens) ----
        left = QVBoxLayout()
        left.setSpacing(10)

        zoom_box = QGroupBox(self.tr("Zoom"))
        zoom_lay = QHBoxLayout(zoom_box)
        zoom_lay.addStretch(1)
        zoom_lay.addWidget(self.btn_zoom_out)
        zoom_lay.addWidget(self.btn_zoom_in)
        zoom_lay.addWidget(self.btn_zoom_fit)
        zoom_lay.addStretch(1)

        left.addWidget(zoom_box)
        left.addWidget(ctrls, 0)

        btn_row = QWidget(self)
        btn_row.setLayout(bb)
        left.addWidget(btn_row, 0)

        left.addStretch(1)

        left_widget = QWidget(self)
        left_widget.setLayout(left)

        left_scroll = QScrollArea(self)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(320)   # tweak 300–360
        left_scroll.setWidget(left_widget)

        # ---- vertical separator ----
        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)

        # ---- RIGHT: Preview ----
        right = QVBoxLayout()
        right.setSpacing(8)
        right.addWidget(self.scroll, 1)

        # Add in order: controls left, preview right
        root.addWidget(left_scroll, 0)
        root.addWidget(sep)
        root.addLayout(right, 1)


        # behavior
        self._threadpool = QThreadPool.globalInstance()
        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)

        # zoom buttons
        self.btn_zoom_out.clicked.connect(lambda: self._set_zoom(self._zoom / 1.25))
        self.btn_zoom_in.clicked.connect(lambda: self._set_zoom(self._zoom * 1.25))
        self.btn_zoom_fit.clicked.connect(self._fit_view)

        self.btn_apply.clicked.connect(self._commit_to_doc)
        self.btn_close.clicked.connect(self.reject)
        self.btn_undo.clicked.connect(self._undo_step)
        self.btn_redo.clicked.connect(self._redo_step)
        # undo/redo inside dialog (simple)
        self._undo, self._redo = [], []
        self._update_undo_redo_buttons()

        self._update_display_autostretch()
        #self._fit_view()

        # shortcuts
        a_undo = QAction(self)
        a_undo.setShortcut(QKeySequence.StandardKey.Undo)
        a_undo.triggered.connect(self._undo_step)

        a_redo = QAction(self)
        a_redo.setShortcut(QKeySequence.StandardKey.Redo)
        a_redo.triggered.connect(self._redo_step)

        a_radius_down = QAction(self)
        a_radius_down.setShortcut(QKeySequence(Qt.Key.Key_BracketLeft))
        a_radius_down.triggered.connect(lambda: self._adjust_brush_radius(-2))

        a_radius_up = QAction(self)
        a_radius_up.setShortcut(QKeySequence(Qt.Key.Key_BracketRight))
        a_radius_up.triggered.connect(lambda: self._adjust_brush_radius(+2))

        self.addAction(a_undo)
        self.addAction(a_redo)
        self.addAction(a_radius_down)
        self.addAction(a_radius_up)

    def _update_undo_redo_buttons(self):
        try:
            self.btn_undo.setEnabled(len(self._undo) > 0)
            self.btn_redo.setEnabled(len(self._redo) > 0)
        except Exception:
            pass


    def _update_display_autostretch(self):
        """Rebuilds self._display from the current linear working image."""
        src = self._image  # linear data (HxWx3)
        if not self.cb_autostretch.isChecked():
            self._display = src.astype(np.float32, copy=False)
            self._refresh_pix()
            return

        tm = float(self.s_target_median.value())
        if not self._orig_mono:
            # true color source
            disp = stretch_color_image(src, target_median=tm, linked=self.cb_linked.isChecked(),
                                    normalize=False, apply_curves=False)
        else:
            # original was mono; channels in src are identical
            mono = src[..., 0]
            mono_st = stretch_mono_image(mono, target_median=tm, normalize=False, apply_curves=False)
            disp = np.stack([mono_st]*3, axis=-1)

        self._display = disp.astype(np.float32, copy=False)
        self._refresh_pix()


    # ── Event filter for hover/click + wheel zoom
    def eventFilter(self, src, ev):
        if src is self.view.viewport():
            et = ev.type()

            # ── Linear Blaster mode ──────────────────────────────
            if getattr(self, "_mode", 0) == 1:
                if et == QEvent.Type.MouseButtonPress:
                    pos = self.view.mapToScene(ev.position().toPoint())
                    if ev.button() == Qt.MouseButton.LeftButton:
                        self._lin_left_press(pos); return True
                    if ev.button() == Qt.MouseButton.RightButton:
                        self._lin_right_press(); return True
                elif et == QEvent.Type.MouseButtonDblClick and ev.button() == Qt.MouseButton.LeftButton:
                    pos = self.view.mapToScene(ev.position().toPoint())
                    self._lin_double_click(pos); return True
                elif et == QEvent.Type.MouseMove:
                    pos = self.view.mapToScene(ev.position().toPoint())
                    self._lin_mouse_move(pos); return True
                elif et == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
                    self._lin_drag = None; return True
                elif et == QEvent.Type.Wheel:
                    self._wheel_zoom(ev); self._lin_refresh(); return True
                return super().eventFilter(src, ev)

            # ── Blemish Blaster mode (original behavior) ─────────
            if et == QEvent.Type.MouseMove:
                pos = self.view.mapToScene(ev.position().toPoint())
                r = self.s_radius.value()
                self.circle.setRect(pos.x()-r, pos.y()-r, 2*r, 2*r)
                self.circle.setVisible(True)
            elif et == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                pos = self.view.mapToScene(ev.position().toPoint())
                self._heal_at(pos)
                return True
            elif et == QEvent.Type.Wheel:
                self._wheel_zoom(ev)
                return True
        return super().eventFilter(src, ev)

    # ── Heal logic
    def _heal_at(self, scene_pos: QPointF):
        x, y = int(round(scene_pos.x())), int(round(scene_pos.y()))
        if not (0 <= x < self._image.shape[1] and 0 <= y < self._image.shape[0]):
            return
        radius  = int(self.s_radius.value())
        feather = float(self.s_feather.value()) / 100.0
        opacity = float(self.s_opacity.value()) / 100.0

        # Snapshot only the affected region before healing
        h, w = self._image.shape[:2]
        pad = int(radius * 1.5) + radius + 2
        py0 = max(0, y - pad)
        px0 = max(0, x - pad)
        py1 = min(h, y + pad + 1)
        px1 = min(w, x + pad + 1)
        pre_patch = (py0, px0, self._image[py0:py1, px0:px1].copy())

        chans = [0, 1, 2]
        worker = _BlemishWorker(self._image, x, y, radius, feather, opacity, chans)
        worker.signals.finished.connect(lambda corrected: self._on_worker_done(corrected, pre_patch))
        self.setEnabled(False)
        self._threadpool.start(worker)

    # ── Linear Blaster ────────────────────────────────────────────
    def _mk_slider_spin(self, slider: QSlider, *, suffix: str = ""):
        """Create a QSpinBox two-way bound to an int slider."""
        sb = QSpinBox(self)
        sb.setRange(slider.minimum(), slider.maximum())
        sb.setValue(slider.value())
        if suffix:
            sb.setSuffix(suffix)
        sb.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        sb.setFixedWidth(78)

        def _sld_to_sb(v):
            if sb.value() != v:
                sb.blockSignals(True); sb.setValue(v); sb.blockSignals(False)

        def _sb_to_sld(v):
            if slider.value() != v:
                slider.blockSignals(True); slider.setValue(v); slider.blockSignals(False)
                slider.valueChanged.emit(v)   # keep existing slider hooks firing

        slider.valueChanged.connect(_sld_to_sb)
        sb.valueChanged.connect(_sb_to_sld)
        return sb

    def _slider_spin_row(self, slider: QSlider, spin: QSpinBox) -> QWidget:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(slider, 1)
        row.addWidget(spin, 0)
        w = QWidget(self)
        w.setLayout(row)
        return w

    def _on_mode_changed(self, *_):
        self._mode = 1 if self.rb_linear.isChecked() else 0
        lin = (self._mode == 1)
        for lbl, fld in ((self.lbl_thickness, self._thickness_row),
                         (self.lbl_extent, self.ext_w)):
            lbl.setVisible(lin); fld.setVisible(lin)
        self.btn_blast_line.setVisible(lin)
        self.circle.setVisible(False)     # brush circle only in blemish mode
        self._lin_reset()

    def _lin_handle_r(self) -> float:
        # scene-space grab radius; scaled by zoom so it stays clickable zoomed out
        return max(4.0, 8.0 / max(getattr(self, "_zoom", 1.0), 1e-3))

    def _lin_clamp_pt(self, pos: QPointF) -> QPointF:
        h, w = self._image.shape[:2]
        return QPointF(float(min(max(pos.x(), 0), w - 1)),
                       float(min(max(pos.y(), 0), h - 1)))

    def _lin_hit_handle(self, pos: QPointF):
        r = self._lin_handle_r() * 1.6
        # mid (bend) handle first so it's grabbable even near an endpoint
        if self._lin_mid is not None and len(self._lin_pts) == 2:
            if math.hypot(pos.x() - self._lin_mid.x(), pos.y() - self._lin_mid.y()) <= r:
                return 2
        for k, p in enumerate(self._lin_pts):
            if math.hypot(pos.x() - p.x(), pos.y() - p.y()) <= r:
                return k
        return None

    def _lin_left_press(self, pos: QPointF):
        hit = self._lin_hit_handle(pos)
        if hit is not None:
            self._lin_drag = hit
        elif len(self._lin_pts) < 2:
            self._lin_pts.append(self._lin_clamp_pt(pos))
            self._lin_drag = None
            if len(self._lin_pts) == 2:
                a, b = self._lin_pts
                self._lin_mid = QPointF(0.5 * (a.x() + b.x()), 0.5 * (a.y() + b.y()))
        self._lin_refresh()

    def _lin_right_press(self):
        if self._lin_pts:
            self._lin_pts.pop()
            self._lin_drag = None
            if len(self._lin_pts) < 2:
                self._lin_mid = None
        self._lin_refresh()

    def _lin_double_click(self, pos: QPointF):
        # double-click the mid handle to straighten the line
        if self._lin_mid is not None and len(self._lin_pts) == 2:
            r = self._lin_handle_r() * 1.6
            if math.hypot(pos.x() - self._lin_mid.x(), pos.y() - self._lin_mid.y()) <= r:
                a, b = self._lin_pts
                self._lin_mid = QPointF(0.5 * (a.x() + b.x()), 0.5 * (a.y() + b.y()))
                self._lin_refresh()

    def _lin_mouse_move(self, pos: QPointF):
        if self._lin_drag == 2 and self._lin_mid is not None:
            self._lin_mid = self._lin_clamp_pt(pos)
            self._lin_refresh()
        elif self._lin_drag is not None and self._lin_drag < len(self._lin_pts):
            self._lin_pts[self._lin_drag] = self._lin_clamp_pt(pos)
            self._lin_refresh()
        elif len(self._lin_pts) == 1:
            self._lin_refresh(cursor=pos)

    def _lin_bend(self) -> bool:
        if self._lin_mid is None or len(self._lin_pts) != 2:
            return False
        a, b = self._lin_pts
        mx, my = 0.5 * (a.x() + b.x()), 0.5 * (a.y() + b.y())
        return math.hypot(self._lin_mid.x() - mx, self._lin_mid.y() - my) > 0.5

    def _lin_ray_exit(self, px, py, dx, dy):
        h, w = self._image.shape[:2]
        L = math.hypot(dx, dy)
        if L < 1e-9:
            return None
        ux, uy = dx / L, dy / L
        ts = []
        if abs(ux) > 1e-9:
            for xb in (0.0, w - 1.0):
                t = (xb - px) / ux
                if t > 1e-6 and -1.0 <= (py + t * uy) <= h:
                    ts.append(t)
        if abs(uy) > 1e-9:
            for yb in (0.0, h - 1.0):
                t = (yb - py) / uy
                if t > 1e-6 and -1.0 <= (px + t * ux) <= w:
                    ts.append(t)
        if not ts:
            return None
        t = min(ts)
        return (px + t * ux, py + t * uy)

    def _lin_center_path(self, cursor: QPointF | None = None):
        from PyQt6.QtGui import QPainterPath
        path = QPainterPath()
        n = len(self._lin_pts)
        full = self.rb_extent_full.isChecked()
        if n == 2:
            a, b = self._lin_pts
            ax, ay, bx, by = a.x(), a.y(), b.x(), b.y()
            if self._lin_bend():
                mcx = 2.0 * self._lin_mid.x() - 0.5 * (ax + bx)
                mcy = 2.0 * self._lin_mid.y() - 0.5 * (ay + by)
                start = (ax, ay)
                if full:
                    eA = self._lin_ray_exit(ax, ay, -(mcx - ax), -(mcy - ay))
                    if eA is not None:
                        start = eA
                path.moveTo(start[0], start[1])
                if start != (ax, ay):
                    path.lineTo(ax, ay)
                path.quadTo(mcx, mcy, bx, by)
                if full:
                    eB = self._lin_ray_exit(bx, by, bx - mcx, by - mcy)
                    if eB is not None:
                        path.lineTo(eB[0], eB[1])
            else:
                if full:
                    eA = self._lin_ray_exit(ax, ay, ax - bx, ay - by)
                    eB = self._lin_ray_exit(bx, by, bx - ax, by - ay)
                    sx, sy = eA if eA is not None else (ax, ay)
                    ex, ey = eB if eB is not None else (bx, by)
                    path.moveTo(sx, sy); path.lineTo(ex, ey)
                else:
                    path.moveTo(ax, ay); path.lineTo(bx, by)
        elif n == 1 and cursor is not None:
            a = self._lin_pts[0]
            path.moveTo(a.x(), a.y()); path.lineTo(cursor.x(), cursor.y())
        return path

    def _lin_refresh(self, cursor: QPointF | None = None):
        hr = self._lin_handle_r()
        for k, hnd in enumerate(self._lin_handles):
            if k < len(self._lin_pts):
                p = self._lin_pts[k]
                hnd.setRect(p.x() - hr, p.y() - hr, 2 * hr, 2 * hr)
                hnd.setVisible(True)
            else:
                hnd.setVisible(False)

        if self._lin_mid is not None and len(self._lin_pts) == 2:
            self._lin_mid_handle.setRect(self._lin_mid.x() - hr, self._lin_mid.y() - hr,
                                         2 * hr, 2 * hr)
            self._lin_mid_handle.setVisible(True)
        else:
            self._lin_mid_handle.setVisible(False)

        path = self._lin_center_path(cursor=cursor)
        if path.elementCount() > 0:
            self._lin_line.setPath(path)
            self._lin_line.setVisible(True)
        else:
            self._lin_line.setVisible(False)
        self._lin_ext.setVisible(False)   # extent is shown by the centerline itself now

        self._lin_update_band(cursor=cursor)
        self.btn_blast_line.setEnabled(self._mode == 1 and len(self._lin_pts) == 2)

    def _lin_update_band(self, cursor: QPointF | None = None):
        """Outline the exact blast region by stroking the centerline at the
        current thickness with round caps — straight, bent, or full-line alike."""
        from PyQt6.QtGui import QPainterPathStroker
        path = self._lin_center_path(cursor=cursor)
        if path.elementCount() == 0:
            self._lin_band.setVisible(False)
            return
        stroker = QPainterPathStroker()
        stroker.setWidth(max(1.0, float(self.s_thickness.value())))
        stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
        stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        self._lin_band.setPath(stroker.createStroke(path))
        self._lin_band.setVisible(True)

    def _lin_reset(self):
        self._lin_pts = []
        self._lin_mid = None
        self._lin_drag = None
        self._lin_line.setVisible(False)
        self._lin_ext.setVisible(False)
        self._lin_band.setVisible(False)
        self._lin_mid_handle.setVisible(False)
        for hnd in self._lin_handles:
            hnd.setVisible(False)
        self.btn_blast_line.setEnabled(False)

    def _blast_line(self):
        if len(self._lin_pts) != 2:
            return
        h, w = self._image.shape[:2]
        p0, p1 = self._lin_pts[0], self._lin_pts[1]
        x0 = int(round(min(max(p0.x(), 0), w - 1))); y0 = int(round(min(max(p0.y(), 0), h - 1)))
        x1 = int(round(min(max(p1.x(), 0), w - 1))); y1 = int(round(min(max(p1.y(), 0), h - 1)))
        thickness = int(self.s_thickness.value())
        feather   = float(self.s_feather.value()) / 100.0
        opacity   = float(self.s_opacity.value()) / 100.0
        full_line = self.rb_extent_full.isChecked()
        mid = None
        if self._lin_bend() and self._lin_mid is not None:
            mid = (float(self._lin_mid.x()), float(self._lin_mid.y()))

        # a line can span the frame, so snapshot the whole image for undo
        pre_patch = (0, 0, self._image.copy())

        chans = [0, 1, 2]
        worker = _LineDefectWorker(self._image, (x0, y0), (x1, y1), mid,
                                   thickness, feather, opacity, chans, full_line)
        worker.signals.finished.connect(lambda corrected: self._on_line_done(corrected, pre_patch))
        self.setEnabled(False)
        self._threadpool.start(worker)

    def _on_line_done(self, corrected: np.ndarray, pre_patch: tuple):
        self._on_worker_done(corrected, pre_patch)
        self._lin_reset()

    def _on_worker_done(self, corrected: np.ndarray, pre_patch: tuple):
        self._undo.append(pre_patch)
        if len(self._undo) > 20:
            self._undo.pop(0)
        self._redo.clear()
        self._image = corrected.astype(np.float32, copy=False)
        self._display = self._image.copy()
        self._update_display_autostretch()
        self.setEnabled(True)
        self._update_undo_redo_buttons()

    # ── Zoom
    # ── Zoom
    def _wheel_zoom(self, ev: QWheelEvent):
        """
        Mouse wheel / trackpad zoom.
        Supports both plain wheel and Ctrl+wheel.
        Zooms around the cursor position in the view.
        """
        old_scene_pos = self.view.mapToScene(ev.position().toPoint())

        # Prefer smooth trackpad deltas when available
        dy = ev.pixelDelta().y()
        if dy != 0:
            abs_dy = abs(dy)
            ctrl_down = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)

            if abs_dy <= 3:
                base_factor = 1.012 if ctrl_down else 1.010
            elif abs_dy <= 10:
                base_factor = 1.025 if ctrl_down else 1.020
            else:
                base_factor = 1.040 if ctrl_down else 1.030

            factor = base_factor if dy > 0 else 1.0 / base_factor
        else:
            dy = ev.angleDelta().y()
            if dy == 0:
                ev.accept()
                return

            ctrl_down = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
            step = 1.25 if ctrl_down else 1.15
            factor = step if dy > 0 else 1.0 / step

        new_zoom = max(0.05, min(4.0, self._zoom * factor))
        if abs(new_zoom - self._zoom) < 1e-6:
            ev.accept()
            return

        self._set_zoom(new_zoom)

        # Keep the point under the cursor anchored after zoom
        new_scene_pos = self.view.mapToScene(ev.position().toPoint())
        delta = new_scene_pos - old_scene_pos

        hsb = self.view.horizontalScrollBar()
        vsb = self.view.verticalScrollBar()
        hsb.setValue(int(round(hsb.value() + delta.x() * self._zoom)))
        vsb.setValue(int(round(vsb.value() + delta.y() * self._zoom)))

        ev.accept()


    def _set_zoom(self, z: float):
        """Clamp and apply zoom to the graphics view."""
        z = float(max(0.05, min(4.0, z)))
        if abs(z - self._zoom) < 1e-4:
            return
        self._zoom = z
        self.view.resetTransform()
        self.view.scale(self._zoom, self._zoom)

    def _fit_view(self):
        """Fit the image nicely into the view without drifting."""
        if self.pix is None or self.pix.pixmap().isNull():
            return

        # Make sure the scene rect matches the pixmap bounds
        br = self.pix.boundingRect()
        if br.isNull():
            return
        self.scene.setSceneRect(br)

        # Reset any old transform and let QGraphicsView handle the fit
        self.view.resetTransform()
        self.view.fitInView(br, Qt.AspectRatioMode.KeepAspectRatio)

        # Track the resulting uniform scale as our current zoom
        t = self.view.transform()
        self._zoom = t.m11()

    # ── Undo/Redo
    def _undo_step(self):
        if not self._undo:
            return
        py0, px0, pre = self._undo.pop()
        ph, pw = pre.shape[:2]
        redo_patch = (py0, px0, self._image[py0:py0 + ph, px0:px0 + pw].copy())
        self._redo.append(redo_patch)
        self._image[py0:py0 + ph, px0:px0 + pw] = pre
        self._display = self._image.copy()
        self._update_display_autostretch()
        self._update_undo_redo_buttons()

    def _redo_step(self):
        if not self._redo:
            return
        py0, px0, post = self._redo.pop()
        ph, pw = post.shape[:2]
        undo_patch = (py0, px0, self._image[py0:py0 + ph, px0:px0 + pw].copy())
        self._undo.append(undo_patch)
        self._image[py0:py0 + ph, px0:px0 + pw] = post
        self._display = self._image.copy()
        self._update_display_autostretch()
        self._update_undo_redo_buttons()

    def _adjust_brush_radius(self, delta: int):
        """Increase/decrease blemish radius from keyboard shortcuts."""
        try:
            cur = int(self.s_radius.value())
            new_val = max(self.s_radius.minimum(), min(self.s_radius.maximum(), cur + int(delta)))
            if new_val != cur:
                self.s_radius.setValue(new_val)
                self._update_brush_circle_from_cursor()
        except Exception:
            pass

    def _update_brush_circle_from_cursor(self):
        """Refresh the preview circle at the current mouse position."""
        try:
            vp = self.view.viewport()
            global_pos = vp.mapFromGlobal(self.cursor().pos())
            if not vp.rect().contains(global_pos):
                return
            pos = self.view.mapToScene(global_pos)
            r = int(self.s_radius.value())
            self.circle.setRect(pos.x() - r, pos.y() - r, 2 * r, 2 * r)
            self.circle.setVisible(True)
        except Exception:
            pass

    # ── Commit back to the document
    def _commit_to_doc(self):
        # convert back to original channels if needed
        out = self._image
        if self._orig_mono:
            # collapse to single channel
            mono = np.mean(out, axis=2, dtype=np.float32)
            if len(self._orig_shape) == 3 and self._orig_shape[2] == 1:
                mono = mono[:, :, None]
            out = mono.astype(np.float32, copy=False)
        else:
            # ensure 3 channels
            if out.ndim == 2:
                out = np.repeat(out[:, :, None], 3, axis=2)
            elif out.ndim == 3 and out.shape[2] >= 3:
                out = out[:, :, :3]

        out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

        # Try common doc APIs
        applied = False
        try:
            if hasattr(self._doc, "set_image"):
                self._doc.set_image(out, step_name="Blemish Blaster"); applied = True
            elif hasattr(self._doc, "apply_numpy"):
                self._doc.apply_numpy(out, step_name="Blemish Blaster"); applied = True
            elif hasattr(self._doc, "image"):
                self._doc.image = out; applied = True
        except Exception as e:
            QMessageBox.critical(self, "Blemish Blaster", f"Failed to write to document:\n{e}")
            return

        if applied and hasattr(self.parent(), "_refresh_active_view"):
            try: self.parent()._refresh_active_view()
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        try:
            self._save_window_geometry()
        except Exception:
            pass
        self.accept()

    # ── display helpers
    def _np_to_qpix(self, img: np.ndarray) -> QPixmap:
        arr = np.ascontiguousarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
        if arr.ndim == 2:
            h, w = arr.shape
            arr = np.repeat(arr[:, :, None], 3, axis=2)
            qimg = QImage(arr.data, w, h, 3*w, QImage.Format.Format_RGB888)
        else:
            h, w, _ = arr.shape
            qimg = QImage(arr.data, w, h, 3*w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _refresh_pix(self):
        self.pix.setPixmap(self._np_to_qpix(self._display))
        self.circle.setVisible(False)

        if not getattr(self, "_did_initial_fit", False):
            self._did_initial_fit = True
            QTimer.singleShot(0, self._fit_view)

    def _restore_window_geometry(self):
        try:
            s = QSettings()
            g = s.value("blemish_blaster/window_geometry", None)
            if g is not None:
                self.restoreGeometry(g)
        except Exception:
            pass

    def _save_window_geometry(self):
        try:
            s = QSettings()
            s.setValue("blemish_blaster/window_geometry", self.saveGeometry())
        except Exception:
            pass

    def showEvent(self, ev):
        super().showEvent(ev)

        if not getattr(self, "_geom_restored", False):
            self._geom_restored = True

            def _after_restore_refit():
                self._restore_window_geometry()
                # after geometry restore, viewport changes → refit
                self._fit_view()

            QTimer.singleShot(0, _after_restore_refit)
            return

        # Normal shows: optional safety refit if you want
        QTimer.singleShot(0, self._fit_view)        

    def closeEvent(self, ev):
        try:
            self._save_window_geometry()
        except Exception:
            pass
        self._image = None
        self._display = None
        self._undo.clear()
        self._redo.clear()
        super().closeEvent(ev)       