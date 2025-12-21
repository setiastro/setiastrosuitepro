# pro/linear_fit.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QDialogButtonBox,
    QPushButton, QGroupBox, QMessageBox, QGridLayout, QWidget, QProgressBar
)

# --------------------------------------------------------------------------------------
# Preset editor (used by Shortcuts “Edit Preset…”). Import into shortcuts.py like:
#   from setiastro.saspro.linear_fit import _LinearFitPresetDialog
# and then store/load via your existing _load_preset/_save_preset helpers.
# --------------------------------------------------------------------------------------

class _LinearFitPresetDialog(QDialog):
    """
    Stores defaults for Linear Fit when run via shortcuts/DnD.
    For mono images the preset does not store a specific reference;
    we will ask the user if needed.
    """
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Linear Fit — Preset")
        init = dict(initial or {})
        v = QVBoxLayout(self)

        gb = QGroupBox("RGB strategy", self)
        grid = QGridLayout(gb)
        self.combo_rgb_mode = QComboBox(self)
        self.combo_rgb_mode.addItems([
            "Match to Highest Median",
            "Match to Lowest Median",
            "Match to Red",
            "Match to Green",
            "Match to Blue",
        ])
        self.combo_rgb_mode.setCurrentIndex(int(init.get("rgb_mode_idx", 0)))
        grid.addWidget(QLabel("Target channel:"), 0, 0)
        grid.addWidget(self.combo_rgb_mode, 0, 1)
        v.addWidget(gb)

        gb2 = QGroupBox("Out-of-range handling", self)
        h2 = QHBoxLayout(gb2)
        self.combo_rescale = QComboBox(self)
        self.combo_rescale.addItems([
            "Clip to [0..1]",
            "Normalize to [0..1] if needed",
            "Leave values as-is",
        ])
        self.combo_rescale.setCurrentIndex(int(init.get("rescale_mode_idx", 1)))
        h2.addWidget(QLabel("Mode:"))
        h2.addWidget(self.combo_rescale, 1)
        v.addWidget(gb2)

        info = QLabel("Mono images will be matched to a reference view's median.\n"
                      "If reference isn’t provided in the headless path, you'll be asked.")
        info.setWordWrap(True)
        v.addWidget(info)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        v.addWidget(btns)

    def result_dict(self) -> dict:
        return {
            "rgb_mode_idx": int(self.combo_rgb_mode.currentIndex()),
            "rescale_mode_idx": int(self.combo_rescale.currentIndex()),
        }

# --------------------------------------------------------------------------------------
# Engine: pure NumPy Linear Fit helpers
# --------------------------------------------------------------------------------------

def _nanmedian(x: np.ndarray) -> float:
    try:
        m = float(np.nanmedian(x))
        if np.isfinite(m):
            return m
    except Exception:
        pass
    return 0.0

def _postprocess(arr: np.ndarray, rescale_mode_idx: int) -> np.ndarray:
    """
    rescale_mode_idx:
      0 = clip to [0..1]
      1 = normalize to [0..1] if min<0 or max>1
      2 = leave values as-is
    """
    if rescale_mode_idx == 2:
        return arr
    if rescale_mode_idx == 0:
        return np.clip(arr, 0.0, 1.0)
    # normalize if needed
    a_min = float(np.nanmin(arr))
    a_max = float(np.nanmax(arr))
    if a_min >= 0.0 and a_max <= 1.0:
        return arr
    rng = max(a_max - a_min, 1e-12)
    return (arr - a_min) / rng

def linear_fit_rgb(img: np.ndarray, rgb_mode_idx: int, rescale_mode_idx: int) -> Tuple[np.ndarray, int, List[float], List[float]]:
    """
    Fit each channel to a reference channel by median.
    Returns (out, ref_idx, medians_before, scales).
    """
    assert img.ndim == 3 and img.shape[2] >= 3, "RGB image expected"
    work = img.astype(np.float32, copy=False)
    meds = [_nanmedian(work[..., c]) for c in range(3)]
    eps = 1e-12

    if rgb_mode_idx == 0:      # Highest
        ref_idx = int(np.argmax(meds))
    elif rgb_mode_idx == 1:    # Lowest
        ref_idx = int(np.argmin(meds))
    elif rgb_mode_idx == 2:    # Red
        ref_idx = 0
    elif rgb_mode_idx == 3:    # Green
        ref_idx = 1
    else:                      # Blue
        ref_idx = 2

    m_ref = max(meds[ref_idx], eps)
    scales = []
    out = work.copy()
    for c in range(3):
        m_c = max(meds[c], eps)
        s = m_ref / m_c
        scales.append(float(s))
        out[..., c] *= float(s)

    out = _postprocess(out, rescale_mode_idx)
    return out, ref_idx, meds, scales

def linear_fit_mono_to_ref(mono: np.ndarray, ref: np.ndarray, rescale_mode_idx: int) -> Tuple[np.ndarray, float, float]:
    """
    Scale mono image median to the reference image median (RGB ref uses luminance proxy).
    Returns (out, m_src, m_ref).
    """
    mono = mono.astype(np.float32, copy=False)
    if ref.ndim == 3 and ref.shape[2] >= 3:
        ref_lum = 0.2126*ref[...,0] + 0.7152*ref[...,1] + 0.0722*ref[...,2]
        m_ref = _nanmedian(ref_lum)
    else:
        m_ref = _nanmedian(ref)

    m_src = _nanmedian(mono)
    eps = 1e-12
    s = (m_ref) / max(m_src, eps)
    out = mono * float(s)
    out = _postprocess(out, rescale_mode_idx)
    return out, m_src, m_ref

# --------------------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------------------

@dataclass
class _Job:
    mode: str                         # "rgb" or "mono"
    rgb_mode_idx: int = 0
    rescale_mode_idx: int = 1
    src: Optional[np.ndarray] = None
    ref: Optional[np.ndarray] = None  # only for mono mode

class _LinearFitWorker(QThread):
    progress = pyqtSignal(int, str)
    failed = pyqtSignal(str)
    done = pyqtSignal(object, str)    # (np.ndarray, step_name)

    def __init__(self, job: _Job):
        super().__init__()
        self.job = job

    def run(self):
        try:
            j = self.job
            if j.src is None:
                raise RuntimeError("No source image")
            self.progress.emit(5, "Analyzing…")

            if j.mode == "rgb":
                out, ref_idx, meds, scales = linear_fit_rgb(j.src, j.rgb_mode_idx, j.rescale_mode_idx)
                names = ["R", "G", "B"]
                target = {
                    0: "highest median", 1: "lowest median",
                    2: "Red", 3: "Green", 4: "Blue"
                }.get(j.rgb_mode_idx, "highest median")
                step = f"Linear Fit (RGB → {names[ref_idx]} / {target})"
                self.progress.emit(100, "Done")
                self.done.emit(out, step)
                return

            if j.mode == "mono":
                if j.ref is None:
                    raise RuntimeError("No reference image selected")
                out, m_src, m_ref = linear_fit_mono_to_ref(j.src, j.ref, j.rescale_mode_idx)
                step = "Linear Fit (mono → reference median)"
                self.progress.emit(100, "Done")
                self.done.emit(out, step)
                return

            raise RuntimeError("Unknown mode")

        except Exception as e:
            self.failed.emit(str(e))

# --------------------------------------------------------------------------------------
# Modal dialog to configure & run on the ACTIVE view
# --------------------------------------------------------------------------------------

class LinearFitDialog(QDialog):
    """
    One-shot UI: works on the active doc image.
    For RGB → choose target channel strategy.
    For mono → pick a reference view from doc_manager.
    Applies result back through doc_manager.apply_edit_to_active().
    """
    def __init__(self, parent, doc_manager, active_doc):
        super().__init__(parent)
        self.setWindowTitle("Linear Fit")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.dm = doc_manager
        self.doc = active_doc
        self.worker: Optional[_LinearFitWorker] = None

        if active_doc is None or getattr(active_doc, "image", None) is None:
            raise RuntimeError("No active image/view")

        img = np.asarray(active_doc.image)
        self._src = img.astype(np.float32, copy=False)

        v = QVBoxLayout(self)

        # Determine mode
        is_rgb = (self._src.ndim == 3 and self._src.shape[2] >= 3)
        self.mode = "rgb" if is_rgb else "mono"

        if self.mode == "rgb":
            gb = QGroupBox("RGB options", self)
            g = QGridLayout(gb)
            self.combo_rgb = QComboBox(self)
            self.combo_rgb.addItems([
                "Match to Highest Median",
                "Match to Lowest Median",
                "Match to Red",
                "Match to Green",
                "Match to Blue",
            ])
            self.combo_rgb.setCurrentIndex(0)
            g.addWidget(QLabel("Target channel:"), 0, 0)
            g.addWidget(self.combo_rgb, 0, 1)
            v.addWidget(gb)
        else:
            gb = QGroupBox("Mono reference", self)
            g = QGridLayout(gb)
            self.combo_ref = QComboBox(self)
            self._ref_docs: list = []
            for d in self.dm.all_documents():
                if d is active_doc:
                    continue
                if getattr(d, "image", None) is None:
                    continue
                self._ref_docs.append(d)
                self.combo_ref.addItem(d.display_name())
            if not self._ref_docs:
                self.combo_ref.addItem("(no other views open)")
            g.addWidget(QLabel("Reference view:"), 0, 0)
            g.addWidget(self.combo_ref, 0, 1)
            note = QLabel("If the reference is RGB, a luminance proxy is used to compute its median.")
            note.setWordWrap(True)
            g.addWidget(note, 1, 0, 1, 2)
            v.addWidget(gb)

        # Common: out-of-range handling
        gb2 = QGroupBox("Out-of-range handling", self)
        h2 = QHBoxLayout(gb2)
        self.combo_rescale = QComboBox(self)
        self.combo_rescale.addItems([
            "Clip to [0..1]",
            "Normalize to [0..1] if needed",
            "Leave values as-is",
        ])
        self.combo_rescale.setCurrentIndex(1)
        h2.addWidget(QLabel("Mode:"))
        h2.addWidget(self.combo_rescale, 1)
        v.addWidget(gb2)

        # Progress
        self.status = QLabel("")
        self.bar = QProgressBar(self); self.bar.setRange(0, 100)
        v.addWidget(self.status)
        v.addWidget(self.bar)

        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self._go)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

        # Small pre-read medians for info (non-blocking)
        try:
            if self.mode == "rgb":
                meds = [_nanmedian(self._src[...,i]) for i in range(3)]
                self.status.setText(f"Channel medians R/G/B: {meds[0]:.4g} / {meds[1]:.4g} / {meds[2]:.4g}")
            else:
                self.status.setText("Mono image selected. Choose a reference view.")
        except Exception:
            pass

    def _go(self):
        rescale_idx = int(self.combo_rescale.currentIndex())
        job = _Job(mode=self.mode, rescale_mode_idx=rescale_idx, src=self._src)

        if self.mode == "rgb":
            job.rgb_mode_idx = int(self.combo_rgb.currentIndex())
        else:
            if not self._ref_docs:
                QMessageBox.warning(self, "Linear Fit", "No reference view available.")
                return
            ref_doc = self._ref_docs[self.combo_ref.currentIndex()]
            job.ref = np.asarray(ref_doc.image).astype(np.float32, copy=False)

        self._run(job)

    def _run(self, job: _Job):
        self.bar.setValue(0)
        self.status.setText("Working…")
        self.setEnabled(False)

        self.worker = _LinearFitWorker(job)
        self.worker.progress.connect(self._on_prog)
        self.worker.failed.connect(self._on_fail)
        self.worker.done.connect(self._on_done)
        self.worker.start()

    def _on_prog(self, pct: int, msg: str):
        self.bar.setValue(pct); self.status.setText(msg)

    def _on_fail(self, err: str):
        self.setEnabled(True)
        self.status.setText("Failed.")
        QMessageBox.critical(self, "Linear Fit", err)

    def _on_done(self, out_img: np.ndarray, step_name: str):
        self.setEnabled(True)
        self.status.setText("Done.")

        # 1) Apply result via DocManager (ROI/full handled there)
        try:
            self.dm.apply_edit_to_active(out_img, step_name=step_name)
        except Exception as e:
            QMessageBox.warning(self, "Linear Fit", f"Applied, but could not update document:\n{e}")

        # 2) Remember this as the last headless-style command for Replay
        try:
            preset: dict = {
                "rescale_mode_idx": int(self.combo_rescale.currentIndex()),
                "mode": self.mode,
            }
            if self.mode == "rgb":
                preset["rgb_mode_idx"] = int(self.combo_rgb.currentIndex())
            else:
                # Mono: stash reference info for future enhancements
                if getattr(self, "_ref_docs", None):
                    idx = int(self.combo_ref.currentIndex())
                    if 0 <= idx < len(self._ref_docs):
                        ref_doc = self._ref_docs[idx]
                        ref_uid = getattr(ref_doc, "uid", None)
                        if ref_uid:
                            preset["ref_uid"] = ref_uid
                        preset["ref_name"] = ref_doc.display_name()

            # Walk up to a parent that knows how to remember headless commands
            mw = self.parent()
            while mw is not None and not hasattr(mw, "_remember_last_headless_command"):
                mw = mw.parent() if hasattr(mw, "parent") else None

            if mw is not None and hasattr(mw, "_remember_last_headless_command"):
                mw._remember_last_headless_command(
                    "linear_fit",
                    preset,
                    description=step_name or "Linear Fit",
                )
        except Exception:
            # Replay tracking should never break the dialog
            pass

        self.accept()


# --------------------------------------------------------------------------------------
# Public helpers for wiring into MainWindow
# --------------------------------------------------------------------------------------

def open_linear_fit_dialog(parent, doc_manager) -> None:
    """
    Bring up the Linear Fit dialog for the active view.
    Applies to active view via doc_manager on success.
    """
    doc = getattr(doc_manager, "get_active_document", lambda: None)()
    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.information(parent, "Linear Fit", "No active image.")
        return
    try:
        dlg = LinearFitDialog(parent, doc_manager, doc)
        dlg.exec()
    except Exception as e:
        QMessageBox.critical(parent, "Linear Fit", str(e))

def apply_linear_fit_via_preset(parent, doc_manager, active_doc, preset: dict | None) -> None:
    """
    Headless/DnD path: apply using a preset dict (from Shortcuts).
    If mono and no reference provided, asks the user to pick one.
    Expected preset keys:
        - rgb_mode_idx (int: 0..4)
        - rescale_mode_idx (int: 0..2)
    """
    preset = dict(preset or {})
    rescale_idx = int(preset.get("rescale_mode_idx", 1))

    img = np.asarray(active_doc.image)
    if img.ndim == 3 and img.shape[2] >= 3:
        rgb_idx = int(preset.get("rgb_mode_idx", 0))
        out, ref_idx, _, _ = linear_fit_rgb(img, rgb_idx, rescale_idx)
        names = ["R","G","B"]
        target = {0:"highest median", 1:"lowest median", 2:"Red", 3:"Green", 4:"Blue"}.get(rgb_idx, "highest median")
        step = f"Linear Fit (RGB → {names[ref_idx]} / {target})"
        doc_manager.apply_edit_to_active(out, step_name=step)
        return

    # MONO → prompt for reference
    # Enumerate other docs
    others = []
    for d in doc_manager.all_documents():
        if d is active_doc:
            continue
        if getattr(d, "image", None) is None:
            continue
        others.append(d)

    if not others:
        QMessageBox.information(parent, "Linear Fit", "Mono image requires a reference view.\nOpen another image and try again.")
        return

    # small inline pick
    pick = QDialog(parent)
    pick.setWindowTitle("Choose Reference View")
    vv = QVBoxLayout(pick)
    cb = QComboBox(pick)
    for d in others:
        cb.addItem(d.display_name())
    vv.addWidget(QLabel("Reference view (median target):"))
    vv.addWidget(cb)
    bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=pick)
    bb.accepted.connect(pick.accept); bb.rejected.connect(pick.reject)
    vv.addWidget(bb)
    if pick.exec() != QDialog.DialogCode.Accepted:
        return
    ref = np.asarray(others[cb.currentIndex()].image)

    out, _, _ = linear_fit_mono_to_ref(img, ref, rescale_idx)
    step = f"Linear Fit (mono → {others[cb.currentIndex()].display_name()})"
    doc_manager.apply_edit_to_active(out, step_name=step)

def apply_linear_fit_to_doc(parent, target_doc, preset: dict | None) -> None:
    """
    Replay helper: apply Linear Fit to a specific ImageDocument
    (usually the *base* doc when 'Replay Last on Base' is used).

    Currently supports RGB images; mono replay-on-base will just
    show a friendly message so you don't get a silent no-op.
    """
    if target_doc is None or getattr(target_doc, "image", None) is None:
        QMessageBox.information(parent, "Linear Fit", "No target image.")
        return

    preset = dict(preset or {})
    rescale_idx = int(preset.get("rescale_mode_idx", 1))

    img = np.asarray(target_doc.image)
    if img.ndim == 3 and img.shape[2] >= 3:
        rgb_idx = int(preset.get("rgb_mode_idx", 0))
        out, ref_idx, _, _ = linear_fit_rgb(img, rgb_idx, rescale_idx)

        names = ["R", "G", "B"]
        target = {
            0: "highest median",
            1: "lowest median",
            2: "Red",
            3: "Green",
            4: "Blue",
        }.get(rgb_idx, "highest median")

        step = f"Linear Fit (RGB → {names[ref_idx]} / {target})"
        meta = {"step_name": step, "bit_depth": "32-bit floating point"}
        try:
            target_doc.apply_edit(out.astype(np.float32, copy=False),
                                  metadata=meta,
                                  step_name=step)
        except Exception as e:
            QMessageBox.warning(parent, "Linear Fit", f"Replay apply failed:\n{e}")
        return

    # Mono replay-on-base: we don't have the reference baked into the preset yet.
    QMessageBox.information(
        parent,
        "Linear Fit",
        "Replay-on-base for mono Linear Fit is not implemented yet.\n"
        "Please re-run Linear Fit on this image via the dialog."
    )

# -------- headless command runner (Scripts / Presets / Replay) ---------------
from setiastro.saspro.headless_utils import normalize_headless_main, unwrap_docproxy

def run_linear_fit_via_preset(main, preset=None, target_doc=None):
    from PyQt6.QtWidgets import QMessageBox
    from setiastro.saspro.linear_fit import apply_linear_fit_via_preset

    p = dict(preset or {})
    main, doc, dm = normalize_headless_main(main, target_doc)

    if dm is None:
        QMessageBox.warning(main or None, "Linear Fit", "DocManager not available.")
        return
    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.warning(main or None, "Linear Fit", "Load an image first.")
        return

    apply_linear_fit_via_preset(main, dm, doc, p)
