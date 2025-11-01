# pro/rgbalign.py
from __future__ import annotations

import os
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QMessageBox, QProgressBar, QPlainTextEdit
)

# soft deps
try:
    import astroalign
except Exception:
    astroalign = None

try:
    import cv2
except Exception:
    cv2 = None

# try to reuse poly from star_alignment if present
try:
    from pro.star_alignment import PolynomialTransform
except Exception:
    PolynomialTransform = None


# ─────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────
class RGBAlignWorker(QThread):
    progress = pyqtSignal(int, str)     # (percent, message)
    done = pyqtSignal(np.ndarray)       # aligned RGB image
    failed = pyqtSignal(str)

    def __init__(self, img: np.ndarray, model: str):
        super().__init__()
        self.img = img
        self.model = model

        self.r_xform = None   # (kind, matrix_or_callable)
        self.b_xform = None   # (kind, matrix_or_callable)
        self.r_pairs = None   # (src_pts, dst_pts) for R
        self.b_pairs = None   # (src_pts, dst_pts) for B

    def run(self):
        if self.img is None or self.img.ndim != 3 or self.img.shape[2] < 3:
            self.failed.emit("Image must be RGB (3 channels).")
            return
        if astroalign is None:
            self.failed.emit("astroalign is not available.")
            return

        try:
            self.progress.emit(5, "Preparing channels…")
            R = np.ascontiguousarray(self.img[..., 0].astype(np.float32, copy=False))
            G = np.ascontiguousarray(self.img[..., 1].astype(np.float32, copy=False))
            B = np.ascontiguousarray(self.img[..., 2].astype(np.float32, copy=False))

            # R → G
            self.progress.emit(15, "Aligning Red → Green…")
            kind_R, X_R, (r_src, r_dst) = self._estimate_transform(R, G, self.model)
            self.r_xform = (kind_R, X_R)
            self.r_pairs = (r_src, r_dst)
            self.progress.emit(35, f"Red transform = {kind_R}")
            R_aligned = self._warp_channel(R, kind_R, X_R, G.shape)

            # B → G
            self.progress.emit(55, "Aligning Blue → Green…")
            kind_B, X_B, (b_src, b_dst) = self._estimate_transform(B, G, self.model)
            self.b_xform = (kind_B, X_B)
            self.b_pairs = (b_src, b_dst)
            self.progress.emit(75, f"Blue transform = {kind_B}")
            B_aligned = self._warp_channel(B, kind_B, X_B, G.shape)

            out = np.stack([R_aligned, G, B_aligned], axis=2).astype(self.img.dtype, copy=False)
            self.progress.emit(100, "Done.")
            self.done.emit(out)
        except Exception as e:
            self.failed.emit(str(e))


    # ───── helpers (basically mini versions of your big star alignment logic) ─────
    def _estimate_transform(self, src: np.ndarray, ref: np.ndarray, model: str):
        H, W = ref.shape[:2]

        tform, (src_pts, dst_pts) = astroalign.find_transform(
            np.ascontiguousarray(src),
            np.ascontiguousarray(ref)
        )

        P = np.asarray(tform.params, dtype=np.float64)
        src_xy = np.asarray(src_pts, dtype=np.float32)
        dst_xy = np.asarray(dst_pts, dtype=np.float32)

        # affine
        if model == "affine":
            if cv2 is None:
                return ("affine", P[0:2, :], (src_xy, dst_xy))
            A, _ = cv2.estimateAffine2D(src_xy, dst_xy, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if A is None:
                return ("affine", P[0:2, :], (src_xy, dst_xy))
            return ("affine", A, (src_xy, dst_xy))

        # homography
        if model == "homography":
            if cv2 is None:
                if P.shape == (3, 3):
                    return ("homography", P, (src_xy, dst_xy))
                A3 = np.vstack([P[0:2, :], [0, 0, 1]])
                return ("homography", A3, (src_xy, dst_xy))
            Hh, _ = cv2.findHomography(src_xy, dst_xy, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if Hh is None:
                if P.shape == (3, 3):
                    return ("homography", P, (src_xy, dst_xy))
                A3 = np.vstack([P[0:2, :], [0, 0, 1]])
                return ("homography", A3, (src_xy, dst_xy))
            return ("homography", Hh, (src_xy, dst_xy))

        # poly3 / poly4
        if model in ("poly3", "poly4") and PolynomialTransform is not None and cv2 is not None:
            order = 3 if model == "poly3" else 4
            scale_vec = np.array([W, H], dtype=np.float32)
            src_n = src_xy / scale_vec
            dst_n = dst_xy / scale_vec

            t_poly = PolynomialTransform()
            ok = t_poly.estimate(dst_n, src_n, order=order)  # dst → src
            if not ok:
                Hh, _ = cv2.findHomography(src_xy, dst_xy, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                return ("homography", Hh, (src_xy, dst_xy))

            def _warp_poly(img: np.ndarray, out_shape: tuple[int, int]):
                Hh_, Ww_ = out_shape
                yy, xx = np.mgrid[0:Hh_, 0:Ww_].astype(np.float32)
                coords = np.stack([xx, yy], axis=-1).reshape(-1, 2)
                coords_n = coords / scale_vec
                mapped_n = t_poly(coords_n)
                mapped = mapped_n * scale_vec
                map_x = mapped[:, 0].reshape(Hh_, Ww_).astype(np.float32)
                map_y = mapped[:, 1].reshape(Hh_, Ww_).astype(np.float32)
                return cv2.remap(
                    img, map_x, map_y,
                    interpolation=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )

            return (model, _warp_poly, (src_xy, dst_xy))

        # fallback → homography
        if cv2 is None:
            if P.shape == (3, 3):
                return ("homography", P, (src_xy, dst_xy))
            A3 = np.vstack([P[0:2, :], [0, 0, 1]])
            return ("homography", A3, (src_xy, dst_xy))

        Hh, _ = cv2.findHomography(src_xy, dst_xy, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        return ("homography", Hh, (src_xy, dst_xy))


    def _warp_channel(self, ch: np.ndarray, kind: str, X, ref_shape):
        H, W = ref_shape[:2]
        if kind == "affine":
            if cv2 is None:
                return ch
            A = np.asarray(X, dtype=np.float32).reshape(2, 3)
            return cv2.warpAffine(ch, A, (W, H), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if kind == "homography":
            if cv2 is None:
                return ch
            Hm = np.asarray(X, dtype=np.float32).reshape(3, 3)
            return cv2.warpPerspective(ch, Hm, (W, H), flags=cv2.INTER_LANCZOS4,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if kind.startswith("poly"):
            return X(ch, (H, W))

        return ch


# ─────────────────────────────────────────────────────────────────────
# Dialog
# ─────────────────────────────────────────────────────────────────────
class RGBAlignDialog(QDialog):
    def __init__(self, parent=None, document=None):
        super().__init__(parent)
        self.setWindowTitle("RGB Align")
        self.parent = parent
        # document could be a view; try to unwrap
        self.doc_view = document
        self.doc = getattr(document, "document", document)
        self.image = getattr(self.doc, "image", None) if self.doc is not None else None

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Align R and B channels to G using astroalign.\n"
                             "Select model and run."))

        hl = QHBoxLayout()
        hl.addWidget(QLabel("Alignment model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Homography (default)",
            "Affine",
            "Poly 3",
            "Poly 4",
        ])
        self.model_combo.setCurrentIndex(0)
        hl.addWidget(self.model_combo)
        lay.addLayout(hl)

        self.chk_new_doc = QCheckBox("Create new document (keep original)")
        self.chk_new_doc.setChecked(True)
        lay.addWidget(self.chk_new_doc)

        # progress
        self.progress_label = QLabel("Idle.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        lay.addWidget(self.progress_label)
        lay.addWidget(self.progress_bar)

        self.summary_box = QPlainTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setPlaceholderText("Transform summary will appear here…")
        self.summary_box.setMinimumHeight(140)
        # optional: monospace
        self.summary_box.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 11px;")
        lay.addWidget(self.summary_box)

        btns = QHBoxLayout()
        self.btn_run = QPushButton("Align")
        self.btn_close = QPushButton("Close")
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_close)
        lay.addLayout(btns)

        self.btn_run.clicked.connect(self._start_align)
        self.btn_close.clicked.connect(self.close)

        self.worker: RGBAlignWorker | None = None

    def _start_align(self):
        if self.image is None:
            QMessageBox.warning(self, "RGB Align", "No image found in active view.")
            return
        if self.image.ndim != 3 or self.image.shape[2] < 3:
            QMessageBox.warning(self, "RGB Align", "Image must be RGB (3 channels).")
            return
        if astroalign is None:
            QMessageBox.warning(self, "RGB Align", "astroalign is not available.")
            return

        model = self._selected_model()
        self.progress_label.setText("Starting…")
        self.progress_bar.setValue(0)

        self.worker = RGBAlignWorker(self.image, model)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.done.connect(self._on_worker_done)
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.start()
        self.btn_run.setEnabled(False)

    def _selected_model(self) -> str:
        txt = self.model_combo.currentText().lower()
        if "affine" in txt:
            return "affine"
        if "poly 3" in txt:
            return "poly3"
        if "poly 4" in txt:
            return "poly4"
        return "homography"

    # slots
    def _on_worker_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.progress_label.setText(msg)

    def _on_worker_failed(self, err: str):
        self.btn_run.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Failed.")
        QMessageBox.critical(self, "RGB Align", err)

    def _on_worker_done(self, out: np.ndarray):
        self.btn_run.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_label.setText("Applying…")

        summary_lines = []
        w = self.worker  # type: ignore

        if w is not None:
            def _fmt_mat(M):
                return "\n".join([
                    "    " + "  ".join(f"{v: .6f}" for v in row)
                    for row in M
                ])

            # R
            if w.r_xform is not None:
                kind, X = w.r_xform
                summary_lines.append("Red → Green:")
                summary_lines.append(f"  model: {kind}")
                if kind == "affine":
                    A = np.asarray(X, dtype=float).reshape(2, 3)
                    M = np.vstack([A, [0, 0, 1]])
                    summary_lines.append(_fmt_mat(M))
                elif kind == "homography":
                    H = np.asarray(X, dtype=float).reshape(3, 3)
                    summary_lines.append(_fmt_mat(H))
                else:
                    summary_lines.append("  (non-matrix; warp callable)")

            # B
            if w.b_xform is not None:
                kind, X = w.b_xform
                summary_lines.append("")
                summary_lines.append("Blue → Green:")
                summary_lines.append(f"  model: {kind}")
                if kind == "affine":
                    A = np.asarray(X, dtype=float).reshape(2, 3)
                    M = np.vstack([A, [0, 0, 1]])
                    summary_lines.append(_fmt_mat(M))
                elif kind == "homography":
                    H = np.asarray(X, dtype=float).reshape(3, 3)
                    summary_lines.append(_fmt_mat(H))
                else:
                    summary_lines.append("  (non-matrix; warp callable)")

        summary_text = "\n".join(summary_lines) if summary_lines else "No transform info."

        # ─── show in dialog ───
        self.summary_box.setPlainText(summary_text)

        # ─── also log to parent if available ───
        if self.parent is not None and hasattr(self.parent, "_log") and callable(self.parent._log):
            self.parent._log("[RGB Align]\n" + summary_text)

        # ─── apply to doc(s) ───
        try:
            if self.chk_new_doc.isChecked():
                dm = getattr(self.parent, "docman", None)
                if dm is not None:
                    dm.open_array(out, {"display_name": "RGB Aligned"}, title="RGB Aligned")
                else:
                    if hasattr(self.doc, "apply_edit"):
                        self.doc.apply_edit(out, {"step_name": "RGB Align"}, step_name="RGB Align")
                    else:
                        self.doc.image = out
            else:
                if hasattr(self.doc, "apply_edit"):
                    self.doc.apply_edit(out, {"step_name": "RGB Align"}, step_name="RGB Align")
                else:
                    self.doc.image = out

            # ✅ finish the status so the user knows it’s done
            self.progress_label.setText("Done.")
        except Exception as e:
            self.progress_label.setText("Apply failed.")
            QMessageBox.warning(self, "RGB Align", f"Aligned image created, but applying failed:\n{e}")




def align_rgb_array(img: np.ndarray, model: str = "homography") -> np.ndarray:
    """
    Headless core: returns a new RGB image with R,B aligned to G.
    Raises RuntimeError on problems.
    """
    if img is None or img.ndim != 3 or img.shape[2] < 3:
        raise RuntimeError("Image must be RGB (3 channels).")
    if astroalign is None:
        raise RuntimeError("astroalign is not available.")

    worker = RGBAlignWorker(img, model)

    try:
        R = np.ascontiguousarray(img[..., 0].astype(np.float32, copy=False))
        G = np.ascontiguousarray(img[..., 1].astype(np.float32, copy=False))
        B = np.ascontiguousarray(img[..., 2].astype(np.float32, copy=False))

        def _estimate_and_warp(src, ref):
            # NOTE: _estimate_transform now returns 3 values
            kind, X, _pairs = worker._estimate_transform(src, ref, model)
            return worker._warp_channel(src, kind, X, ref.shape)

        R_aligned = _estimate_and_warp(R, G)
        B_aligned = _estimate_and_warp(B, G)

        out = np.stack([R_aligned, G, B_aligned], axis=2)
        if img.dtype != out.dtype:
            out = out.astype(img.dtype, copy=False)
        return out
    except Exception as e:
        raise RuntimeError(str(e))

def run_rgb_align_headless(main_window, document, preset: dict | None = None):
    """
    Headless entry used by shortcuts.
    main_window: your QMainWindow (has docman, statusBar, etc.)
    document: current ImageDocument
    preset: {"model": "homography"/"affine"/"poly3"/"poly4", "new_doc": bool}
    """
    if document is None:
        QMessageBox.warning(main_window, "RGB Align", "No active document.")
        return

    img = np.asarray(document.image)
    p = dict(preset or {})
    model = p.get("model", "homography").lower()
    create_new = bool(p.get("new_doc", False))

    sb = getattr(main_window, "statusBar", None)
    if callable(sb):
        sb().showMessage(f"RGB Align ({model})…", 3000)

    try:
        out = align_rgb_array(img, model=model)
    except Exception as e:
        QMessageBox.critical(main_window, "RGB Align (headless)", str(e))
        return

    if create_new:
        dm = getattr(main_window, "docman", None)
        if dm is not None:
            dm.open_array(out, {"display_name": "RGB Aligned"}, title="RGB Aligned")
        else:
            # fallback to replace if we can't create new
            try:
                document.apply_edit(out, {"step_name": "RGB Align"})
            except Exception:
                document.image = out
    else:
        # in-place
        if hasattr(document, "apply_edit"):
            document.apply_edit(out, {"step_name": "RGB Align"}, step_name="RGB Align")
        else:
            document.image = out

    if callable(sb):
        sb().showMessage("RGB Align done.", 3000)
