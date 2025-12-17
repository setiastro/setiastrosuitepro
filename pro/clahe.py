# pro/clahe.py
from __future__ import annotations
import numpy as np
import cv2

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QLabel, QPushButton, QSlider, QGraphicsScene,
    QGraphicsPixmapItem, QMessageBox
)

# Import centralized widgets
from pro.widgets.graphics_views import ZoomableGraphicsView
from pro.widgets.image_utils import extract_mask_resized as _get_active_mask_resized
from pro.widgets.themed_buttons import themed_toolbtn


# ----------------------- Core -----------------------
def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    # ... (unchanged)
    if image is None:
        raise ValueError("image is None")
    arr = np.asarray(image, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    was_hw1 = (arr.ndim == 3 and arr.shape[2] == 1)
    if arr.ndim == 3 and arr.shape[2] == 3:
        lab = cv2.cvtColor((arr * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        return np.clip(enhanced, 0.0, 1.0)
    mono = arr.squeeze()
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    cl = clahe.apply((mono * 255.0).astype(np.uint8)).astype(np.float32) / 255.0
    cl = np.clip(cl, 0.0, 1.0)
    if was_hw1:
        cl = cl[..., None]
    return cl

# Note: _get_active_mask_resized imported from pro.widgets.image_utils

def apply_clahe_to_doc(doc, preset: dict | None):
    """
    Apply CLAHE to doc.image using a preset.

    Backward compatible:
      - old presets: {"clip_limit": 2.0, "tile": 8}  # tile count across min dimension
      - new presets: {"clip_limit": 2.0, "tile_px": 128}  # tile size in pixels
    """
    if doc is None or getattr(doc, "image", None) is None:
        raise RuntimeError("Document has no image.")

    img = np.asarray(doc.image)

    # --- preset decode (supports old + new) ---
    p = preset or {}
    clip = float(p.get("clip_limit", 2.0))

    # Resolve tile_grid_size for OpenCV
    if "tile_px" in p:
        tile_px = int(p.get("tile_px", 128))
        h, w = img.shape[:2]
        s = float(min(h, w))
        tile_px = max(8, tile_px)
        n = int(round(s / float(tile_px)))
        n = max(2, min(n, 128))
        tile_grid = (n, n)
    else:
        # legacy: treat "tile" as OpenCV tileGridSize count (tiles across)
        tile = int(p.get("tile", 8))
        tile = max(2, min(tile, 128))
        tile_grid = (tile, tile)

    out = apply_clahe(img, clip_limit=clip, tile_grid_size=tile_grid)
    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    # Blend with active mask if present
    H, W = out.shape[:2]
    m = _get_active_mask_resized(doc, H, W)
    if m is not None:
        base = np.asarray(doc.image, dtype=np.float32)
        if base.dtype.kind in "ui":
            maxv = float(np.iinfo(base.dtype).max)
            base = base / max(1.0, maxv)
        else:
            base = np.clip(base, 0.0, 1.0)

        if out.ndim == 3:
            if base.ndim == 2:
                base = base[:, :, None].repeat(out.shape[2], axis=2)
            elif base.ndim == 3 and base.shape[2] == 1:
                base = base.repeat(out.shape[2], axis=2)
            M = np.repeat(m[:, :, None], out.shape[2], axis=2).astype(np.float32)
            out = np.clip(base * (1.0 - M) + out * M, 0.0, 1.0)
        else:
            if base.ndim == 3 and base.shape[2] == 1:
                base = base.squeeze(axis=2)
            out = np.clip(base * (1.0 - m) + out * m, 0.0, 1.0)

    # Commit
    if hasattr(doc, "set_image"):
        doc.set_image(out, step_name="CLAHE")
    elif hasattr(doc, "apply_numpy"):
        doc.apply_numpy(out, step_name="CLAHE")
    else:
        doc.image = out

# ----------------------- Dialog -----------------------
class CLAHEDialogPro(QDialog):
    def __init__(self, parent, doc, icon: QIcon | None = None):
        super().__init__(parent)
        self.setWindowTitle("CLAHE")
        if icon:
            try: self.setWindowIcon(icon)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        self.doc = doc
        self.orig = np.clip(np.asarray(doc.image, dtype=np.float32), 0.0, 1.0)
        disp = self.orig
        if disp.ndim == 2: disp = disp[..., None].repeat(3, axis=2)
        elif disp.ndim == 3 and disp.shape[2] == 1: disp = disp.repeat(3, axis=2)
        self._disp_base = disp

        v = QVBoxLayout(self)

        # ---- Params (unchanged) ----
        grp = QGroupBox("CLAHE Parameters"); grid = QGridLayout(grp)
        self.s_clip = QSlider(Qt.Orientation.Horizontal); self.s_clip.setRange(1, 40); self.s_clip.setValue(20)
        self.lbl_clip = QLabel("2.0")
        self.s_clip.valueChanged.connect(lambda val: self.lbl_clip.setText(f"{val/10.0:.1f}"))
        self.s_clip.valueChanged.connect(self._debounce_preview)

        # tile size slider (pixels) — intuitive control
        self.s_tile = QSlider(Qt.Orientation.Horizontal)
        self.s_tile.setRange(8, 512)          # 4 is pointless; you clamp to >=8 anyway
        self.s_tile.setSingleStep(8)
        self.s_tile.setPageStep(64)
        self.s_tile.setValue(128)             # nice default
        self.s_tile.setToolTip("CLAHE tile size in pixels (larger = coarser, smaller = finer).")

        self.lbl_tile = QLabel("128 px")
        self.lbl_tile.setToolTip(self.s_tile.toolTip())

        self.s_tile.valueChanged.connect(lambda v: self.lbl_tile.setText(f"{v} px"))
        self.s_tile.valueChanged.connect(self._debounce_preview)

        grid.addWidget(QLabel("Tile Size (px):"), 1, 0)
        grid.addWidget(self.s_tile, 1, 1)
        grid.addWidget(self.lbl_tile, 1, 2)



        grid.addWidget(QLabel("Clip Limit:"), 0, 0); grid.addWidget(self.s_clip, 0, 1); grid.addWidget(self.lbl_clip, 0, 2)

        v.addWidget(grp)

        # ---- Preview with zoom/pan ----
        self.scene = QGraphicsScene(self)
        self.view  = ZoomableGraphicsView(self.scene)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pix   = QGraphicsPixmapItem()
        self.scene.addItem(self.pix)
        v.addWidget(self.view, 1)

        # ---- Zoom bar ----
        # ---- Zoom bar (themed) ----
        z = QHBoxLayout()
        z.addStretch(1)

        self.btn_zoom_in  = themed_toolbtn("zoom-in", "Zoom In")
        self.btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        self.btn_zoom_fit = themed_toolbtn("zoom-fit-best", "Fit to Preview")

        self.btn_zoom_in.clicked.connect(self.view.zoom_in)
        self.btn_zoom_out.clicked.connect(self.view.zoom_out)
        self.btn_zoom_fit.clicked.connect(lambda: self.view.fit_to_item(self.pix))

        z.addWidget(self.btn_zoom_in)
        z.addWidget(self.btn_zoom_out)
        z.addWidget(self.btn_zoom_fit)

        v.addLayout(z)


        # ---- Buttons (unchanged) ----
        row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply");  self.btn_apply.clicked.connect(self._apply)
        self.btn_reset = QPushButton("Reset");  self.btn_reset.clicked.connect(self._reset)
        self.btn_close = QPushButton("Cancel"); self.btn_close.clicked.connect(self.reject)
        row.addStretch(1); row.addWidget(self.btn_apply); row.addWidget(self.btn_reset); row.addWidget(self.btn_close)
        v.addLayout(row)

        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.timeout.connect(self._update_preview)

        self._set_pix(self._disp_base)
        self._update_preview()
        # initial fit
        self.view.fit_to_item(self.pix)

    def _debounce_preview(self): self._timer.start(250)

    def _set_pix(self, rgb):
        arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        h, w, _ = arr.shape
        q = QImage(arr.data, w, h, 3*w, QImage.Format.Format_RGB888)
        self.pix.setPixmap(QPixmap.fromImage(q))
        self.scene.setSceneRect(self.pix.boundingRect())

    def _update_preview(self):
        clip = self.s_clip.value() / 10.0
        tile_px = int(self.s_tile.value())

        try:
            tile_grid = self._tile_grid_from_px(tile_px, self._disp_base.shape[:2])

            out = apply_clahe(
                self._disp_base,
                clip_limit=float(clip),
                tile_grid_size=tile_grid
            )

            # Respect active mask (preview works on _disp_base size)
            H, W = out.shape[:2]
            m = _get_active_mask_resized(self.doc, H, W)
            if m is not None:
                if out.ndim == 3:
                    M = np.repeat(m[:, :, None], out.shape[2], axis=2).astype(np.float32)
                else:
                    M = m.astype(np.float32)

                base = self._disp_base.astype(np.float32, copy=False)
                out = np.clip(base * (1.0 - M) + out * M, 0.0, 1.0)

            self._set_pix(out)
            self._preview = out

        except Exception as e:
            QMessageBox.warning(self, "CLAHE", f"Preview failed:\n{e}")


    def _apply(self):
        try:
            clip = float(self.s_clip.value() / 10.0)
            tile_px = int(self.s_tile.value())

            tile_grid = self._tile_grid_from_px(tile_px, self.orig.shape[:2])

            out = apply_clahe(
                self.orig,
                clip_limit=clip,
                tile_grid_size=tile_grid
            )
            out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

            # Mask-respectful commit
            H, W = out.shape[:2]
            m = _get_active_mask_resized(self.doc, H, W)
            if m is not None:
                base = np.asarray(self.doc.image, dtype=np.float32)

                # Normalize base into [0..1] for blending
                if base.dtype.kind in "ui":
                    maxv = float(np.iinfo(base.dtype).max)
                    base = base / max(1.0, maxv)
                else:
                    base = np.clip(base, 0.0, 1.0)

                if out.ndim == 3:
                    if base.ndim == 2:
                        base = base[:, :, None].repeat(out.shape[2], axis=2)
                    elif base.ndim == 3 and base.shape[2] == 1:
                        base = base.repeat(out.shape[2], axis=2)

                    M = np.repeat(m[:, :, None], out.shape[2], axis=2).astype(np.float32)
                    out = np.clip(base * (1.0 - M) + out * M, 0.0, 1.0)
                else:
                    if base.ndim == 3 and base.shape[2] == 1:
                        base = base.squeeze(axis=2)
                    out = np.clip(base * (1.0 - m) + out * m, 0.0, 1.0)

                out = out.astype(np.float32, copy=False)

            # Commit to document
            if hasattr(self.doc, "set_image"):
                self.doc.set_image(out, step_name="CLAHE")
            elif hasattr(self.doc, "apply_numpy"):
                self.doc.apply_numpy(out, step_name="CLAHE")
            else:
                self.doc.image = out

            # ── Register as last_headless_command for replay ─────────────
            try:
                main = self.parent()
                if main is not None:
                    preset = {
                        "clip_limit": float(clip),
                        "tile_px": int(tile_px),   # NEW, intuitive
                        # (optional debug)
                        # "tile": int(tile_grid[0]),
                    }
                    payload = {"command_id": "clahe", "preset": dict(preset)}
                    setattr(main, "_last_headless_command", payload)

                    try:
                        if hasattr(main, "_log"):
                            main._log(
                                f"[Replay] Registered CLAHE as last action "
                                f"(clip_limit={preset['clip_limit']}, tile_px={preset['tile_px']})"
                            )
                    except Exception:
                        pass
            except Exception:
                pass
            # ─────────────────────────────────────────────────────────────

            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "CLAHE", f"Failed to apply:\n{e}")

    def _tile_grid_from_px(self, tile_px: int, hw: tuple[int, int]) -> tuple[int, int]:
        """
        Convert desired tile size (pixels) into OpenCV tileGridSize=(n,n)
        where n is number of tiles across the *min dimension*.
        """
        h, w = hw
        s = float(min(h, w))
        tile_px = max(8, int(tile_px))
        n = int(round(s / float(tile_px)))
        n = max(2, min(n, 128))
        return (n, n)

    def _reset(self):
        self.s_clip.setValue(20); self.s_tile.setValue(8)
        self._set_pix(self._disp_base)
        self.view.fit_to_item(self.pix)
