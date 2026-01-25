# setiastro/saspro/cosmicclarity.py
from __future__ import annotations
import os
import sys
import glob
import time
import tempfile
import uuid
import numpy as np

from PyQt6.QtCore import Qt, QTimer, QSettings, QThread, pyqtSignal, QFileSystemWatcher, QEvent
from PyQt6.QtGui import QIcon, QAction, QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout, QLabel, QPushButton,
    QSlider, QCheckBox, QComboBox, QMessageBox, QWidget, QRadioButton, QProgressBar,
    QTextEdit, QFileDialog, QTreeWidget, QTreeWidgetItem, QMenu, QInputDialog
)
from PyQt6.QtCore import QProcess

# ---- bring in your image IO helpers ----
# Adjust these imports to your project structure if needed.
from setiastro.saspro.legacy.image_manager import load_image, save_image  

from setiastro.saspro.imageops.stretch import stretch_mono_image, stretch_color_image

from setiastro.saspro.cosmicclarity_engines.sharpen_engine import sharpen_rgb01
from setiastro.saspro.cosmicclarity_engines.denoise_engine import denoise_rgb01
from setiastro.saspro.cosmicclarity_engines.superres_engine import superres_rgb01
from setiastro.saspro.cosmicclarity_engines.satellite_engine import (
    get_satellite_models,
    satellite_remove_image,
)
# Import centralized preview dialog
from setiastro.saspro.widgets.preview_dialogs import ImagePreviewDialog

import shutil
import subprocess


def resolve_cosmic_root(parent=None) -> str:
    s = QSettings()
    root = s.value("paths/cosmic_clarity", "", type=str) or ""
    if root and os.path.isdir(root):
        return root

    # Try common relatives to the app executable
    appdir = os.path.dirname(os.path.abspath(sys.argv[0]))
    candidates = [
        appdir,
        os.path.join(appdir, "cosmic_clarity"),
        os.path.join(appdir, "CosmicClarity"),
        os.path.dirname(appdir),  # one up
    ]
    exe_names = {
        "win": ["SetiAstroCosmicClarity.exe", "SetiAstroCosmicClarity_denoise.exe"],
        "mac": ["SetiAstroCosmicClaritymac", "SetiAstroCosmicClarity_denoisemac"],
        "nix": ["SetiAstroCosmicClarity", "SetiAstroCosmicClarity_denoise"],
    }
    key = "win" if os.name == "nt" else ("mac" if sys.platform=="darwin" else "nix")

    for c in candidates:
        if all(os.path.exists(os.path.join(c, name)) for name in exe_names[key]):
            # ensure in/out exist
            os.makedirs(os.path.join(c, "input"), exist_ok=True)
            os.makedirs(os.path.join(c, "output"), exist_ok=True)
            s.setValue("paths/cosmic_clarity", c); s.sync()
            return c

    # Prompt user once
    QMessageBox.information(parent, "Cosmic Clarity",
        "Please select your Cosmic Clarity folder (the one that contains the CC executables and input/output).")
    folder = QFileDialog.getExistingDirectory(parent, "Select Cosmic Clarity Folder", "")
    if folder:
        s.setValue("paths/cosmic_clarity", folder); s.sync()
        os.makedirs(os.path.join(folder, "input"), exist_ok=True)
        os.makedirs(os.path.join(folder, "output"), exist_ok=True)
        return folder
    return ""  # caller should handle "not set"


# =============================================================================
# Small helpers
# =============================================================================
def _satellite_exe_name() -> str:
    base = "setiastrocosmicclarity_satellite"
    return f"{base}.exe" if os.name == "nt" else base




class WaitDialog(QDialog):
    cancelled = pyqtSignal()
    def __init__(self, title="Processing…", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        v = QVBoxLayout(self)
        self.lbl = QLabel("Processing, please wait…")
        self.txt = QTextEdit(); self.txt.setReadOnly(True)
        self.pb  = QProgressBar(); self.pb.setRange(0, 100)
        btn = QPushButton("Cancel"); btn.clicked.connect(self.cancelled.emit)
        v.addWidget(self.lbl); v.addWidget(self.txt); v.addWidget(self.pb); v.addWidget(btn)
    def append_output(self, line: str): self.txt.append(line)
    def set_progress(self, p: int):      self.pb.setValue(int(max(0, min(100, p))))

class CosmicClarityEngineWorker(QThread):
    progress = pyqtSignal(int)      # 0..100
    log      = pyqtSignal(str)
    result   = pyqtSignal(object, str)  # (np.ndarray float32 RGB01, final_step_title)
    error    = pyqtSignal(str)

    def __init__(self, img_rgb01: np.ndarray, preset: dict, parent=None):
        super().__init__(parent)
        self._img = np.asarray(img_rgb01, dtype=np.float32)
        self._preset = dict(preset)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    # ---- progress adapter ----
    def _mk_progress_cb(self, stage_label: str, stage_weight: float, base_pct: float, total_stages: int):
        """
        stage_weight: fraction of total progress reserved for this stage (0..1)
        base_pct: starting % for this stage
        """
        def _cb(done: int, total: int):
            if self._cancel:
                return
            if total <= 0:
                return
            frac = float(done) / float(total)
            pct = base_pct + stage_weight * 100.0 * frac
            self.progress.emit(int(max(0, min(100, round(pct)))))
            # optional: throttle logs; keep it quiet
        return _cb

    def run(self):
        try:
            p = self._preset
            mode = str(p.get("mode", "sharpen")).lower()

            use_gpu = bool(p.get("gpu", True))
            create_new_view = bool(p.get("create_new_view", False))  # not used here; dialog decides

            img = np.clip(self._img, 0.0, 1.0).astype(np.float32, copy=False)

            # Decide stage plan
            stages = []
            if mode == "sharpen":
                stages = ["sharpen"]
            elif mode == "denoise":
                stages = ["denoise"]
            elif mode == "both":
                stages = ["sharpen", "denoise"]
            elif mode == "superres":
                stages = ["superres"]
            else:
                stages = ["sharpen"]

            n = max(1, len(stages))
            stage_weight = 1.0 / float(n)

            out = img
            self.progress.emit(0)

            for si, st in enumerate(stages):
                if self._cancel:
                    self.error.emit("Cancelled.")
                    return

                base_pct = (100.0 * si) / float(n)
                self.log.emit(f"Running {st}…")

                if st == "sharpen":
                    # UI preset fields mirror your existing UI naming
                    sharpening_mode = p.get("sharpening_mode", "Both")
                    stellar_amount = float(p.get("stellar_amount", 0.5))
                    nonstellar_amount = float(p.get("nonstellar_amount", 0.5))
                    nonstellar_psf = float(p.get("nonstellar_psf", 3.0))
                    auto_psf = bool(p.get("auto_psf", True))
                    sharpen_sep = bool(p.get("sharpen_channels_separately", False))

                    prog = self._mk_progress_cb("sharpen", stage_weight, base_pct, n)

                    out = sharpen_rgb01(
                        out,
                        sharpening_mode=str(sharpening_mode),
                        stellar_amount=float(stellar_amount),
                        nonstellar_amount=float(nonstellar_amount),
                        nonstellar_strength=float(nonstellar_psf),
                        auto_detect_psf=bool(auto_psf),
                        separate_channels=bool(sharpen_sep),
                        use_gpu=bool(use_gpu),
                        progress_cb=prog,
                    )

                elif st == "denoise":
                    den_luma = float(p.get("denoise_luma", 0.5))
                    den_col = float(p.get("denoise_color", 0.5))
                    den_mode = str(p.get("denoise_mode", "full"))
                    sep = bool(p.get("separate_channels", False))

                    prog = self._mk_progress_cb("denoise", stage_weight, base_pct, n)

                    out = denoise_rgb01(
                        out,
                        denoise_strength=float(den_luma),
                        denoise_mode=str(den_mode),
                        separate_channels=bool(sep),
                        color_denoise_strength=float(den_col),
                        use_gpu=bool(use_gpu),
                        progress_cb=prog,
                    )

                elif st == "superres":
                    scale = int(p.get("scale", 2))
                    prog = self._mk_progress_cb("superres", stage_weight, base_pct, n)

                    out = superres_rgb01(
                        out,
                        scale=int(scale),
                        use_gpu=True,   # keep matching your old UI behavior (GPU hidden for SR)
                        progress_cb=prog,
                    )

                else:
                    raise RuntimeError(f"Unknown stage: {st}")

            self.progress.emit(100)

            # Title for history
            if mode == "both":
                step_title = "Cosmic Clarity – Sharpen + Denoise"
            elif mode == "superres":
                step_title = "Cosmic Clarity – Super Resolution"
            else:
                step_title = f"Cosmic Clarity – {mode.title()}"

            self.result.emit(np.clip(out, 0.0, 1.0).astype(np.float32, copy=False), step_title)

        except Exception as e:
            self.error.emit(str(e))


# =============================================================================
# Dialog
# =============================================================================
class CosmicClarityDialogPro(QDialog):
    """
    Pro port of SASv2 Cosmic Clarity panel:
      • Modes: Sharpen, Denoise, Both, Super Resolution
      • GPU toggle
      • PSF, stellar/nonstellar amounts
      • Denoise strengths/mode
      • Super-res scale
      • Apply target: overwrite / new view
    Uses QSettings key: paths/cosmic_clarity
    """
    def __init__(self, parent, doc, icon: QIcon | None = None, *, headless: bool=False, bypass_guard: bool=False):
        super().__init__(parent)
        # Hard guard unless explicitly bypassed (used by preset runner)
        if not bypass_guard and self._headless_guard_active():
            # avoid any flash; never show
            try: self.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            QTimer.singleShot(0, self.reject)
            return        
        self.setWindowTitle(self.tr("Cosmic Clarity"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass  # older PyQt6 versions        
        if icon: 
            try: self.setWindowIcon(icon)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        self.parent_ref = parent
        self.doc = doc
        self.orig = np.clip(np.asarray(doc.image, dtype=np.float32), 0.0, 1.0)
        self.cosmic_root = ""   # no longer used by in-process engines

        v = QVBoxLayout(self)

        # ---------------- Controls ----------------
        grp = QGroupBox(self.tr("Parameters"))
        grid = QGridLayout(grp)

        # Mode
        grid.addWidget(QLabel(self.tr("Mode:")), 0, 0)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Sharpen", "Denoise", "Both", "Super Resolution"])
        self.cmb_mode.currentIndexChanged.connect(self._mode_changed)
        grid.addWidget(self.cmb_mode, 0, 1, 1, 2)

        # GPU
        grid.addWidget(QLabel(self.tr("Use GPU:")), 1, 0)
        self.cmb_gpu = QComboBox(); self.cmb_gpu.addItems([self.tr("Yes"), self.tr("No")])
        grid.addWidget(self.cmb_gpu, 1, 1)

        # Sharpen block
        self.lbl_sh_mode = QLabel("Sharpening Mode:")
        self.cmb_sh_mode = QComboBox(); self.cmb_sh_mode.addItems(["Both", "Stellar Only", "Non-Stellar Only"])
        grid.addWidget(self.lbl_sh_mode, 2, 0); grid.addWidget(self.cmb_sh_mode, 2, 1)

        self.chk_sh_sep = QCheckBox("Sharpen RGB channels separately")
        self.chk_sh_sep.setToolTip(
            "Run the mono sharpening model independently on R, G, and B instead of a shared color model.\n"
            "Use for difficult color data where channels need slightly different sharpening."
        )
        grid.addWidget(self.chk_sh_sep, 3, 0)

        self.chk_auto_psf = QCheckBox("Auto Detect PSF"); self.chk_auto_psf.setChecked(True)
        grid.addWidget(self.chk_auto_psf, 3, 1)

        self.lbl_psf = QLabel("Non-Stellar PSF (1.0–8.0): 3.0")
        self.sld_psf = QSlider(Qt.Orientation.Horizontal); self.sld_psf.setRange(10, 80); self.sld_psf.setValue(30)
        self.sld_psf.valueChanged.connect(self._psf_label)
        grid.addWidget(self.lbl_psf, 4, 0, 1, 2); grid.addWidget(self.sld_psf, 5, 0, 1, 3)

        self.lbl_st_amt = QLabel("Stellar Amount (0–1): 0.50")
        self.sld_st_amt = QSlider(Qt.Orientation.Horizontal); self.sld_st_amt.setRange(0, 100); self.sld_st_amt.setValue(50)

        self.sld_st_amt.valueChanged.connect(self._on_st_amt)
        grid.addWidget(self.lbl_st_amt, 6, 0, 1, 2); grid.addWidget(self.sld_st_amt, 7, 0, 1, 3)

        self.lbl_nst_amt = QLabel("Non-Stellar Amount (0–1): 0.50")
        self.sld_nst_amt = QSlider(Qt.Orientation.Horizontal); self.sld_nst_amt.setRange(0, 100); self.sld_nst_amt.setValue(50)

        self.sld_nst_amt.valueChanged.connect(self._on_nst_amt)
        grid.addWidget(self.lbl_nst_amt, 8, 0, 1, 2); grid.addWidget(self.sld_nst_amt, 9, 0, 1, 3)

        # Denoise block
        self.lbl_dn_lum = QLabel("Luminance Denoise (0–1): 0.50")
        self.sld_dn_lum = QSlider(Qt.Orientation.Horizontal); self.sld_dn_lum.setRange(0, 100); self.sld_dn_lum.setValue(50)
        self.sld_dn_lum.valueChanged.connect(lambda v: self.lbl_dn_lum.setText(f"Luminance Denoise (0–1): {v/100:.2f}"))
        grid.addWidget(self.lbl_dn_lum, 10, 0, 1, 2); grid.addWidget(self.sld_dn_lum, 11, 0, 1, 3)

        self.lbl_dn_col = QLabel("Color Denoise (0–1): 0.50")
        self.sld_dn_col = QSlider(Qt.Orientation.Horizontal); self.sld_dn_col.setRange(0, 100); self.sld_dn_col.setValue(50)
        self.sld_dn_col.valueChanged.connect(lambda v: self.lbl_dn_col.setText(f"Color Denoise (0–1): {v/100:.2f}"))
        grid.addWidget(self.lbl_dn_col, 12, 0, 1, 2); grid.addWidget(self.sld_dn_col, 13, 0, 1, 3)

        self.lbl_dn_mode = QLabel("Denoise Mode:")
        self.cmb_dn_mode = QComboBox(); self.cmb_dn_mode.addItems(["full", "luminance"])
        grid.addWidget(self.lbl_dn_mode, 14, 0); grid.addWidget(self.cmb_dn_mode, 14, 1)

        self.chk_dn_sep = QCheckBox("Process RGB channels separately")
        grid.addWidget(self.chk_dn_sep, 15, 1)

        # Super-res
        self.lbl_scale = QLabel("Scale Factor:")
        self.cmb_scale = QComboBox(); self.cmb_scale.addItems(["2x", "3x", "4x"])
        grid.addWidget(self.lbl_scale, 16, 0); grid.addWidget(self.cmb_scale, 16, 1)

        # Apply target
        grid.addWidget(QLabel("Apply to:"), 17, 0)
        self.cmb_target = QComboBox(); self.cmb_target.addItems(["Overwrite active view", "Create new view"])
        grid.addWidget(self.cmb_target, 17, 1, 1, 2)

        v.addWidget(grp)

        # Buttons
        row = QHBoxLayout()
        b_run   = QPushButton(self.tr("Execute")); b_run.clicked.connect(self._run_main)
        b_close = QPushButton(self.tr("Close"));   b_close.clicked.connect(self.reject)
        row.addStretch(1); row.addWidget(b_run); row.addWidget(b_close)
        v.addLayout(row)

        self._mode_changed()  # set initial visibility

        self._wait = None


        self._headless = bool(headless)
        if self._headless:
            # Don’t show the control panel; we’ll still exec() to run the event loop.
            try: self.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        self.resize(560, 540)

    # ----- UI helpers -----
    def _headless_guard_active(self) -> bool:
        # 1) fast path: flags on the main window
        try:
            p = self.parent()
            if p and (getattr(p, "_cosmicclarity_guard", False) or getattr(p, "_cosmicclarity_headless_running", False)):
                return True
        except Exception:
            pass
        # 2) cross-module path: QSettings flag set by the preset runner
        try:
            s = QSettings()
            v = s.value("cc/headless_in_progress", False, type=bool)
            return bool(v)
        except Exception:
            # fallback if type kwarg unsupported in some Qt builds
            try:
                return bool(QSettings().value("cc/headless_in_progress", False))
            except Exception:
                return False

    # Never show if guard is active
    def showEvent(self, e):
        if self._headless_guard_active():
            e.ignore()
            QTimer.singleShot(0, self.reject)
            return
        return super().showEvent(e)

    # Never exec if guard is active
    def exec(self) -> int:
        if self._headless_guard_active():
            return 0
        return super().exec()


    def _on_st_amt(self, v: int): self.lbl_st_amt.setText(f"Stellar Amount (0–1): {v/100:.2f}")
    def _on_nst_amt(self, v: int): self.lbl_nst_amt.setText(f"Non-Stellar Amount (0–1): {v/100:.2f}")

    def _psf_label(self):
        self.lbl_psf.setText(f"Non-Stellar PSF (1.0–8.0): {self.sld_psf.value()/10:.1f}")

    def _mode_changed(self):
        idx = self.cmb_mode.currentIndex()  # 0 Sharpen, 1 Denoise, 2 Both, 3 Super-Res
        # Sharpen controls visible if Sharpen or Both
        show_sh = idx in (0, 2)
        for w in (self.lbl_sh_mode, self.cmb_sh_mode, self.chk_sh_sep, self.chk_auto_psf, self.lbl_psf, self.sld_psf, self.lbl_st_amt, self.sld_st_amt, self.lbl_nst_amt, self.sld_nst_amt):
            w.setVisible(show_sh)

        # Denoise controls visible if Denoise or Both
        show_dn = idx in (1, 2)
        for w in (self.lbl_dn_lum, self.sld_dn_lum, self.lbl_dn_col, self.sld_dn_col, self.lbl_dn_mode, self.cmb_dn_mode, self.chk_dn_sep):
            w.setVisible(show_dn)

        # Super-res controls visible if Super-Res
        show_sr = idx == 3
        for w in (self.lbl_scale, self.cmb_scale):
            w.setVisible(show_sr)

        # GPU hidden for superres (matches your SASv2)
        self.cmb_gpu.setVisible(not show_sr)
        self.parentWidget()

    # ----- Validation -----
    def _validate_root(self) -> bool:
        return True

    # ----- Execution -----
    def _run_main(self):
        # --- Basic safety: make sure we have an image ---
        try:
            img = np.asarray(getattr(self.doc, "image", None))
            if img is None or img.size == 0:
                QMessageBox.warning(self, "Cosmic Clarity", "No image loaded in the active view.")
                return
        except Exception:
            QMessageBox.warning(self, "Cosmic Clarity", "No image loaded in the active view.")
            return

        # --- Register this run as "last action" for replay (same as you had) ---
        try:
            main = self.parent_ref or self.parent()
            if main is not None:
                preset = self.build_preset_from_ui()
                payload = {
                    "cid": "cosmic_clarity",
                    "preset": preset,
                    "label": f"Cosmic Clarity ({preset.get('mode', 'sharpen')})",
                }
                if hasattr(main, "_set_last_headless_command"):
                    main._set_last_headless_command(payload)
                else:
                    setattr(main, "_last_headless_command", payload)
                    if hasattr(main, "_update_replay_button"):
                        main._update_replay_button()
        except Exception:
            pass  # never block processing

        # --- Snapshot UI preset ---
        preset = self.build_preset_from_ui()
        mode = str(preset.get("mode", "sharpen")).lower()

        # --- Normalize to float32 RGB01 for the engines ---
        # Your engines expect RGB01; keep mono as 3-ch internally.
        arr = np.asarray(self.doc.image, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[..., :3]
        else:
            QMessageBox.critical(self, "Cosmic Clarity", f"Unsupported image shape: {arr.shape}")
            return

        arr = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)

        # --- Show wait/progress UI ---
        title = "Cosmic Clarity – " + (
            "Sharpen + Denoise" if mode == "both" else
            "Super Resolution"  if mode == "superres" else
            mode.title()
        )
        self._wait = WaitDialog(title, self)
        self._wait.set_progress(0)
        self._wait.append_output("Starting…")
        self._wait.cancelled.connect(self._cancel_all_engine)
        self._wait.show()

        # --- Run the in-process engine worker ---
        self._engine_thread = CosmicClarityEngineWorker(arr, preset, parent=self)

        self._engine_thread.progress.connect(lambda p: self._wait.set_progress(int(p)) if self._wait else None)
        self._engine_thread.log.connect(lambda s: self._wait.append_output(str(s)) if self._wait else None)
        self._engine_thread.error.connect(self._on_engine_error)
        self._engine_thread.result.connect(self._on_engine_result)

        self._engine_thread.start()


    def _cancel_all_engine(self):
        try:
            if getattr(self, "_engine_thread", None) is not None:
                self._engine_thread.cancel()
        except Exception:
            pass
        if self._wait:
            self._wait.close()
            self._wait = None

    def _on_engine_error(self, msg: str):
        if self._wait:
            self._wait.close()
            self._wait = None
        self._engine_thread = None
        QMessageBox.critical(self, "Cosmic Clarity", msg)

    def _on_engine_result(self, out_arr: np.ndarray, step_title: str):
        if self._wait:
            self._wait.close()
            self._wait = None
        self._engine_thread = None

        create_new = (self.cmb_target.currentIndex() == 1)
        if create_new:
            ok = self._spawn_new_doc_from_numpy(out_arr, step_title)
            if not ok:
                self._apply_to_active(out_arr, step_title)
        else:
            self._apply_to_active(out_arr, step_title)

        self.accept()


    def _base_name(self) -> str:
        fp = getattr(self.doc, "file_path", None)
        if isinstance(fp, str) and fp:
            return os.path.splitext(os.path.basename(fp))[0]
        name = getattr(self.doc, "display_name", None)
        if callable(name):
            try:
                n = name() or ""
                if n:
                    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in n).strip("_") or "image"
            except Exception:
                pass
        return "image"


    def _apply_to_active(self, arr: np.ndarray, step_title: str):
        """Overwrite the active document image."""
        if hasattr(self.doc, "set_image"):
            self.doc.set_image(arr, step_name=step_title)
        elif hasattr(self.doc, "apply_numpy"):
            self.doc.apply_numpy(arr, step_name=step_title)
        else:
            self.doc.image = arr

    def _spawn_new_doc_from_numpy(self, arr: np.ndarray, step_title: str) -> bool:
        """Create a brand-new document + view from a numpy array. Returns True on success."""
        mw = self.parent()
        dm = getattr(mw, "docman", None)
        if dm is None:
            return False

        # build a reasonable title and metadata
        base_name = getattr(self.doc, "display_name", None)
        base = base_name() if callable(base_name) else (base_name or "Image")
        title = f"{base} [{step_title}]"

        meta = {
            "bit_depth": "32-bit floating point",
            "is_mono": (arr.ndim == 2) or (arr.ndim == 3 and arr.shape[2] == 1),
            "source": "Cosmic Clarity",
            "original_header": getattr(self.doc, "original_header", None),
        }

        try:
            new_doc = dm.open_array(arr.astype(np.float32, copy=False), metadata=meta, title=title)
            if hasattr(mw, "_spawn_subwindow_for"):   # same hook used in ABE
                mw._spawn_subwindow_for(new_doc)
            return True
        except Exception as e:
            QMessageBox.critical(self, "Cosmic Clarity", f"Failed to create new view:\n{e}")
            return False


    # ----- Super-resolution -----

    def apply_preset(self, p: dict):
        # Mode
        mode = str(p.get("mode","sharpen")).lower()
        self.cmb_mode.setCurrentIndex({"sharpen":0,"denoise":1,"both":2,"superres":3}.get(mode,0))
        # GPU
        self.cmb_gpu.setCurrentIndex(0 if p.get("gpu", True) else 1)
        # Target
        self.cmb_target.setCurrentIndex(1 if p.get("create_new_view", False) else 0)
        # Sharpen
        self.cmb_sh_mode.setCurrentText(p.get("sharpening_mode","Both"))
        self.chk_auto_psf.setChecked(bool(p.get("auto_psf", True)))
        self.sld_psf.setValue(int(max(10, min(80, round(float(p.get("nonstellar_psf",3.0))*10)))))
        self.sld_st_amt.setValue(int(max(0, min(100, round(float(p.get("stellar_amount",0.5))*100)))))
        self.sld_nst_amt.setValue(int(max(0, min(100, round(float(p.get("nonstellar_amount",0.5))*100)))))
        # NEW: allow presets to opt into per-channel sharpen (still defaults off without a preset)
        self.chk_sh_sep.setChecked(bool(p.get("sharpen_channels_separately", False)))

        # Denoise
        self.sld_dn_lum.setValue(int(max(0, min(100, round(float(p.get("denoise_luma",0.5))*100)))))
        self.sld_dn_col.setValue(int(max(0, min(100, round(float(p.get("denoise_color",0.5))*100)))))
        self.cmb_dn_mode.setCurrentText(str(p.get("denoise_mode","full")))
        self.chk_dn_sep.setChecked(bool(p.get("separate_channels", False)))
        # Super-Res
        self.cmb_scale.setCurrentText(str(int(p.get("scale",2))))

    def build_preset_from_ui(self) -> dict:
        """Snapshot current UI state into a preset dict usable by headless runner / replay."""
        idx = self.cmb_mode.currentIndex()  # 0 Sharpen, 1 Denoise, 2 Both, 3 Super-Res
        mode = {0: "sharpen", 1: "denoise", 2: "both", 3: "superres"}.get(idx, "sharpen")

        preset: dict = {
            "mode": mode,
            "gpu": (self.cmb_gpu.currentIndex() == 0),
            "create_new_view": (self.cmb_target.currentIndex() == 1),
        }

        # Sharpen / Both block
        if mode in ("sharpen", "both"):
            preset.update({
                "sharpening_mode": self.cmb_sh_mode.currentText(),
                "auto_psf": self.chk_auto_psf.isChecked(),
                "nonstellar_psf": self.sld_psf.value() / 10.0,         # slider 10–80 → 1.0–8.0
                "stellar_amount": self.sld_st_amt.value() / 100.0,     # 0–100 → 0–1
                "nonstellar_amount": self.sld_nst_amt.value() / 100.0, # 0–100 → 0–1
                "sharpen_channels_separately": self.chk_sh_sep.isChecked(),
            })

        # Denoise / Both block
        if mode in ("denoise", "both"):
            preset.update({
                "denoise_luma": self.sld_dn_lum.value() / 100.0,
                "denoise_color": self.sld_dn_col.value() / 100.0,
                "denoise_mode": self.cmb_dn_mode.currentText(),
                "separate_channels": self.chk_dn_sep.isChecked(),
            })

        # Super-res
        if mode == "superres":
            try:
                scale_txt = self.cmb_scale.currentText()
                # can be "2x" in the main dialog or just "2" in the preset dialog
                scale_txt = scale_txt.replace("x", "")
                preset["scale"] = int(scale_txt)
            except Exception:
                preset["scale"] = 2

        return preset


# =============================================================================
# Satellite removal 
# =============================================================================


class CosmicClaritySatelliteDialogPro(QDialog):
    """
    Pro dialog that mirrors SASv2 Cosmic Clarity Satellite tab:
      • Select input/output folders, live monitor, or batch process
      • GPU toggle, mode (full/luminance), clip trail, sensitivity, skip-save
      • Tree views for input/output with preview (autostretch + zoom)
    Uses QSettings key: paths/cosmic_clarity
    """
    def __init__(self, parent, doc=None, icon: QIcon | None = None):
        super().__init__(parent)
        self.setWindowTitle("Cosmic Clarity – Satellite Removal")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        if icon:
            try: self.setWindowIcon(icon)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        self.settings = QSettings()

        self.input_folder  = ""
        self.output_folder = ""
        self.sensitivity = 0.10  # 0.01–0.50
        self.doc = doc

        self.file_watcher = QFileSystemWatcher()
        self.file_watcher.directoryChanged.connect(self._on_folder_changed)

        self._sat_thread = None
        self._wait = None

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        main = QHBoxLayout(self)

        # Left controls
        left = QVBoxLayout()

        # Input/Output folder chooser row
        row_io = QHBoxLayout()
        self.btn_in  = QPushButton("Select Input Folder");  self.btn_in.clicked.connect(self._choose_input)
        self.btn_out = QPushButton("Select Output Folder"); self.btn_out.clicked.connect(self._choose_output)
        row_io.addWidget(self.btn_in); row_io.addWidget(self.btn_out)
        left.addLayout(row_io)

        # GPU
        left.addWidget(QLabel("Use GPU Acceleration:"))
        self.cmb_gpu = QComboBox(); self.cmb_gpu.addItems(["Yes", "No"])
        left.addWidget(self.cmb_gpu)

        # Mode
        left.addWidget(QLabel("Satellite Removal Mode:"))
        self.cmb_mode = QComboBox(); self.cmb_mode.addItems(["Full", "Luminance"])
        left.addWidget(self.cmb_mode)

        # Clip trail
        self.chk_clip = QCheckBox("Clip Satellite Trail to 0.000"); self.chk_clip.setChecked(True)
        left.addWidget(self.chk_clip)

        # Sensitivity slider
        row_sens = QHBoxLayout()
        row_sens.addWidget(QLabel("Clipping Sensitivity (Lower = more aggressive):"))
        self.sld_sens = QSlider(Qt.Orientation.Horizontal)
        self.sld_sens.setRange(1, 50)           # 0.01–0.50
        self.sld_sens.setValue(int(self.sensitivity * 100))
        self.sld_sens.setTickInterval(1)
        self.sld_sens.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sld_sens.valueChanged.connect(self._on_sens_change)
        row_sens.addWidget(self.sld_sens)
        self.lbl_sens_val = QLabel(f"{self.sensitivity:.2f}")
        row_sens.addWidget(self.lbl_sens_val)
        left.addLayout(row_sens)

        # Skip save if no trail
        self.chk_skip = QCheckBox("Skip Save if No Satellite Trail Detected")
        self.chk_skip.setChecked(False)
        left.addWidget(self.chk_skip)

        # Process row: single image / batch
        row_proc = QHBoxLayout()
        self.btn_single = QPushButton("Process Single Image"); self.btn_single.clicked.connect(self._process_single_image)
        self.btn_batch  = QPushButton("Batch Process Input Folder"); self.btn_batch.clicked.connect(self._batch_process)
        row_proc.addWidget(self.btn_single); row_proc.addWidget(self.btn_batch)
        left.addLayout(row_proc)

        # Live monitor
        self.btn_monitor = QPushButton("Live Monitor Input Folder"); self.btn_monitor.clicked.connect(self._live_monitor)
        left.addWidget(self.btn_monitor)


        left.addStretch(1)

        # Right: trees
        right = QVBoxLayout()
        right.addWidget(QLabel("Input Folder Files:"))
        self.tree_in = QTreeWidget(); self.tree_in.setHeaderLabels(["Filename"])
        self.tree_in.itemDoubleClicked.connect(lambda *_: self._preview_from_tree(self.tree_in, is_input=True))
        self.tree_in.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_in.customContextMenuRequested.connect(lambda pos: self._context_menu(self.tree_in, pos, is_input=True))
        right.addWidget(self.tree_in)

        right.addWidget(QLabel("Output Folder Files:"))
        self.tree_out = QTreeWidget(); self.tree_out.setHeaderLabels(["Filename"])
        self.tree_out.itemDoubleClicked.connect(lambda *_: self._preview_from_tree(self.tree_out, is_input=False))
        self.tree_out.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_out.customContextMenuRequested.connect(lambda pos: self._context_menu(self.tree_out, pos, is_input=False))
        right.addWidget(self.tree_out)

        main.addLayout(left, 2)
        main.addLayout(right, 1)

        self.resize(900, 600)



    # ---------- IO folders ----------
    def _choose_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder", self.input_folder or "")
        if not folder: return
        self.input_folder = folder
        self.btn_in.setText(f"Input: {os.path.basename(folder)}")
        self._watch(folder)
        self._refresh_tree(self.tree_in, folder)

    def _choose_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.output_folder or "")
        if not folder: return
        self.output_folder = folder
        self.btn_out.setText(f"Output: {os.path.basename(folder)}")
        self._watch(folder)
        self._refresh_tree(self.tree_out, folder)

    def _watch(self, folder):
        try:
            if folder and folder not in self.file_watcher.directories():
                self.file_watcher.addPath(folder)
        except Exception:
            pass

    def _on_folder_changed(self, path):
        if path == self.input_folder:
            self._refresh_tree(self.tree_in, self.input_folder)
        elif path == self.output_folder:
            self._refresh_tree(self.tree_out, self.output_folder)

    def _refresh_tree(self, tree: QTreeWidget, folder: str):
        tree.clear()
        if not folder or not os.path.isdir(folder): return
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(('.png', '.tif', '.tiff', '.fit', '.fits', '.xisf',
                                    '.cr2', '.nef', '.arw', '.dng', '.raf', '.orf', '.rw2', '.pef', '.jpg', '.jpeg')):
                QTreeWidgetItem(tree, [fn])

    # ---------- Sensitivity ----------
    def _on_sens_change(self, v: int):
        self.sensitivity = v / 100.0
        self.lbl_sens_val.setText(f"{self.sensitivity:.2f}")

    # ---------- Context menu ----------
    def _context_menu(self, tree: QTreeWidget, pos, is_input: bool):
        item = tree.itemAt(pos)
        if not item: return
        menu = QMenu(self)
        act_del = QAction("Delete File", self)
        act_ren = QAction("Rename File", self)
        act_del.triggered.connect(lambda: self._delete_file(tree, is_input))
        act_ren.triggered.connect(lambda: self._rename_file(tree, is_input))
        menu.addAction(act_del); menu.addAction(act_ren)
        menu.exec(tree.viewport().mapToGlobal(pos))

    def _folder_of(self, is_input: bool) -> str:
        return self.input_folder if is_input else self.output_folder

    def _delete_file(self, tree: QTreeWidget, is_input: bool):
        item = tree.currentItem()
        if not item: return
        folder = self._folder_of(is_input)
        fp = os.path.join(folder, item.text(0))
        if not os.path.exists(fp): return
        if QMessageBox.question(self, "Confirm Delete", f"Delete {item.text(0)}?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            os.remove(fp)
            self._refresh_tree(tree, folder)

    def _rename_file(self, tree: QTreeWidget, is_input: bool):
        item = tree.currentItem()
        if not item: return
        folder = self._folder_of(is_input)
        fp = os.path.join(folder, item.text(0))
        new, ok = QInputDialog.getText(self, "Rename File", "Enter new name:", text=item.text(0))
        if ok and new:
            np = os.path.join(folder, new)
            os.rename(fp, np)
            self._refresh_tree(tree, folder)

    # ---------- Preview ----------
    def _preview_from_tree(self, tree: QTreeWidget, is_input: bool):
        item = tree.currentItem()
        if not item: return
        folder = self._folder_of(is_input)
        fp = os.path.join(folder, item.text(0))
        if not os.path.isfile(fp): return
        try:
            img, _, _, is_mono = load_image(fp)
            if img is None:
                QMessageBox.critical(self, "Error", "Failed to load image for preview.")
                return
            dlg = ImagePreviewDialog(img, is_mono=is_mono, parent=self)
            dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            dlg.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to preview image:\n{e}")

    # ---------- Single image processing ----------
    def _process_single_image(self):
        from PyQt6.QtCore import QCoreApplication, Qt
        from PyQt6.QtWidgets import QMessageBox, QFileDialog, QInputDialog, QProgressDialog

        from setiastro.saspro.cosmicclarity_engines.satellite_engine import (
            get_satellite_models,
            satellite_remove_image,
        )

        # ----------------------------
        # Decide source: view or file
        # ----------------------------
        views = self._collect_open_views()
        use_view = False

        if views:
            mb = QMessageBox(self)
            mb.setWindowTitle("Process Single Image")
            mb.setText("Choose the source to process:")
            btn_view = mb.addButton("Open View", QMessageBox.ButtonRole.AcceptRole)
            btn_file = mb.addButton("File on Disk", QMessageBox.ButtonRole.AcceptRole)
            mb.addButton(QMessageBox.StandardButton.Cancel)
            mb.exec()

            if mb.clickedButton() is btn_view:
                use_view = True
            elif mb.clickedButton() is btn_file:
                use_view = False
            else:
                return

        # ----------------------------
        # Gather engine params from UI
        # ----------------------------
        use_gpu = (self.cmb_gpu.currentText() == "Yes")
        mode = self.cmb_mode.currentText().lower()
        clip_trail = bool(self.chk_clip.isChecked())
        sensitivity = float(self.sensitivity)
        skip_if_none = bool(self.chk_skip.isChecked())

        # ----------------------------------------------------
        # Acquire input image FIRST (no progress dialog yet)
        # ----------------------------------------------------
        hdr = None
        mono = False
        chosen_doc = None
        file_path = None

        if use_view:
            # Choose which doc
            if len(views) == 1:
                chosen_doc = views[0][1]
            else:
                titles = [t for (t, _) in views]
                sel, ok = QInputDialog.getItem(self, "Select View", "Choose an open view:", titles, 0, False)
                if not ok:
                    return
                chosen_doc = views[titles.index(sel)][1]

            if chosen_doc is None or getattr(chosen_doc, "image", None) is None:
                QMessageBox.warning(self, "Warning", "Selected view has no image.")
                return

            try:
                img = np.asarray(chosen_doc.image, dtype=np.float32)
                img = np.clip(img, 0.0, 1.0)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read image from view:\n{e}")
                return

        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Image", "",
                "Image Files (*.png *.tif *.tiff *.fit *.fits *.xisf *.cr2 *.nef *.arw *.dng *.raf *.orf *.rw2 *.pef *.jpg *.jpeg)"
            )
            if not file_path:
                return  # user cancelled file dialog

            try:
                img, hdr, bd, mono = load_image(file_path)
                if img is None:
                    raise RuntimeError("load_image returned None.")
                img = np.asarray(img, dtype=np.float32)
                img = np.clip(img, 0.0, 1.0)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")
                return

        # ----------------------------
        # Load/cached models ONCE
        # ----------------------------
        try:
            models = get_satellite_models(use_gpu=use_gpu, status_cb=lambda s: None)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load Satellite models:\n{e}")
            return

        # ----------------------------
        # Progress dialog helper
        #   IMPORTANT: create/show AFTER input is chosen
        # ----------------------------
        pd = QProgressDialog("Satellite removal…", "Cancel", 0, 100, self)
        pd.setWindowModality(Qt.WindowModality.WindowModal)
        pd.setMinimumDuration(250)   # helps prevent weird cancel behavior
        pd.setAutoClose(True)
        pd.setAutoReset(True)
        pd.setValue(0)

        cancelled = {"flag": False}

        def _on_cancel():
            cancelled["flag"] = True

        pd.canceled.connect(_on_cancel)

        def _progress(done: int, total: int):
            if total > 0:
                pd.setValue(int((done * 100) / total))
            else:
                pd.setValue(0)

            QCoreApplication.processEvents()

            # IMPORTANT: return True to continue, False to cancel
            return (not cancelled["flag"]) and (not pd.wasCanceled())

        # ----------------------------
        # Run engine
        # ----------------------------
        try:
            cancelled["flag"] = False  # reset right before run
            pd.show()

            out, detected = satellite_remove_image(
                img,
                models=models,
                mode=mode,
                clip_trail=clip_trail,
                sensitivity=sensitivity,
                progress_cb=_progress,
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing image:\n{e}")
            return

        finally:
            try:
                pd.close()
            except Exception:
                pass

        # ----------------------------
        # Handle cancel / skip-save
        # ----------------------------
        if cancelled["flag"]:
            QMessageBox.information(self, "Cancelled", "Operation cancelled.")
            return

        if skip_if_none and (not detected):
            QMessageBox.information(self, "Satellite Removal", "No satellite trail detected (skip-save enabled).")
            return

        out = np.asarray(out, dtype=np.float32, order="C")

        # ----------------------------
        # Apply or save
        # ----------------------------
        if use_view:
            if hasattr(chosen_doc, "set_image"):
                chosen_doc.set_image(out, step_name="Cosmic Clarity – Satellite Removal")
            elif hasattr(chosen_doc, "apply_numpy"):
                chosen_doc.apply_numpy(out, step_name="Cosmic Clarity – Satellite Removal")
            else:
                chosen_doc.image = out
            return

        # file path save
        try:
            base, ext = os.path.splitext(file_path)
            dst = base + "_satellited" + ext

            el = ext.lower()
            if el in (".tif", ".tiff"):
                fmt, bitdepth = "tiff", "32-bit floating point"
            elif el in (".fit", ".fits"):
                fmt, bitdepth = "fits", "32-bit floating point"
            elif el == ".xisf":
                fmt, bitdepth = "xisf", "32-bit floating point"
            else:
                fmt, bitdepth = "auto", None

            save_image(out, dst, fmt, bitdepth, hdr, mono)
            QMessageBox.information(self, "Success", f"Processed image saved to:\n{dst}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save result:\n{e}")

    def _collect_open_views(self):
        """
        Return a list of (title, doc) for all open MDI views with an image.
        Includes self.doc if supplied and valid.
        """
        views = []
        # include self.doc first if valid
        if getattr(self, "doc", None) is not None and getattr(self.doc, "image", None) is not None:
            title = getattr(self.doc, "display_name", lambda: "Active View")()
            views.append((title, self.doc))

        # try to enumerate MDI subwindows on the parent main window
        try:
            main = self.parent()
            mdi = getattr(main, "mdi", None)
            if mdi is not None:
                for sw in mdi.subWindowList():
                    w = sw.widget()
                    d = getattr(w, "document", None)
                    if d is not None and getattr(d, "image", None) is not None:
                        t = w.windowTitle() if hasattr(w, "windowTitle") else getattr(d, "display_name", lambda:"View")()
                        # don’t duplicate self.doc if it’s the same object
                        if not any(d is existing for _, existing in views):
                            views.append((t, d))
        except Exception:
            pass

        return views

    def _base_name_for_doc(self, d):
        """Derive a simple basename for staging temp files from a document."""
        fp = getattr(d, "file_path", None)
        if isinstance(fp, str) and fp:
            return os.path.splitext(os.path.basename(fp))[0]
        name = getattr(d, "display_name", None)
        if callable(name):
            try:
                n = name() or ""
                if n:
                    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in n).strip("_") or "image"
            except Exception:
                pass
        return "image"


    def _batch_process(self):
        if not self.input_folder or not self.output_folder:
            QMessageBox.warning(self, "Warning", "Please select both input and output folders.")
            return

        self._run_engine_thread(monitor=False, title="Satellite – Batch processing")


    def _live_monitor(self):
        if not self.input_folder or not self.output_folder:
            QMessageBox.warning(self, "Warning", "Please select both input and output folders.")
            return

        self.sld_sens.setEnabled(False)
        self._run_engine_thread(monitor=True, title="Satellite – Live monitoring",
                                on_finish=lambda: self.sld_sens.setEnabled(True))

    def _run_engine_thread(self, *, monitor: bool, title: str, on_finish=None):
        use_gpu = (self.cmb_gpu.currentText() == "Yes")
        mode = self.cmb_mode.currentText().lower()  # "full" / "luminance"
        clip_trail = bool(self.chk_clip.isChecked())
        sensitivity = float(self.sensitivity)
        skip_save = bool(self.chk_skip.isChecked())

        self._wait = WaitDialog(title, self)
        self._wait.show()

        self._sat_thread = SatelliteEngineThread(
            input_dir=self.input_folder,
            output_dir=self.output_folder,
            use_gpu=use_gpu,
            mode=mode,
            clip_trail=clip_trail,
            sensitivity=sensitivity,
            skip_save=skip_save,
            monitor=monitor,
        )
        self._sat_thread.log_signal.connect(self._wait.append_output)
        self._sat_thread.finished_signal.connect(lambda: self._on_thread_finished(on_finish))
        self._wait.cancelled.connect(self._cancel_sat_thread)
        self._sat_thread.start()


    # ---------- Command / run ----------

    def _cancel_sat_thread(self):
        if self._sat_thread:
            self._sat_thread.cancel()
        if self._wait:
            self._wait.close()
            self._wait = None

    def _on_thread_finished(self, on_finish):
        if self._wait: self._wait.close(); self._wait = None
        if callable(on_finish):
            try: on_finish()
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        QMessageBox.information(self, "Done", "Processing finished.")


    # ---------- Utils ----------
    @staticmethod
    def _create_temp_folder(base="~"):
        user_dir = os.path.expanduser(base)
        temp_folder = os.path.join(user_dir, "CosmicClarityTemp")
        os.makedirs(temp_folder, exist_ok=True)
        return temp_folder

class SatelliteEngineThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    progress_signal = pyqtSignal(int, int)  # done, total

    def __init__(self, *, input_dir: str, output_dir: str,
                 use_gpu: bool, mode: str, clip_trail: bool,
                 sensitivity: float, skip_save: bool, monitor: bool,
                 poll_seconds: float = 1.0):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.use_gpu = use_gpu
        self.mode = mode
        self.clip_trail = clip_trail
        self.sensitivity = sensitivity
        self.skip_save = skip_save
        self.monitor = monitor
        self.poll_seconds = poll_seconds

        self._cancel = False
        self._seen = set()

    def cancel(self):
        self._cancel = True

    def _iter_files(self):
        exts = ('.png', '.tif', '.tiff', '.fit', '.fits', '.xisf',
                '.cr2', '.nef', '.arw', '.dng', '.raf', '.orf', '.rw2', '.pef', '.jpg', '.jpeg')
        try:
            for fn in sorted(os.listdir(self.input_dir)):
                if fn.lower().endswith(exts):
                    yield fn
        except Exception:
            return

    def run(self):
        try:
            from setiastro.saspro.cosmicclarity_engines.satellite_engine import (
                get_satellite_models, satellite_remove_image
            )

            self.log_signal.emit("Loading Satellite models...")
            models = get_satellite_models(use_gpu=self.use_gpu, status_cb=lambda s: None)
            self.log_signal.emit("Models loaded.")

            os.makedirs(self.output_dir, exist_ok=True)

            def process_one(fp_in: str, fp_out: str):
                # NOTE: assumes load_image/save_image exist in your module scope
                img, hdr, bd, mono = load_image(fp_in)
                if img is None:
                    self.log_signal.emit(f"Failed to load: {os.path.basename(fp_in)}")
                    return

                img = np.asarray(img, dtype=np.float32)
                img = np.clip(img, 0.0, 1.0)

                out, detected = satellite_remove_image(
                    img,
                    models=models,
                    mode=self.mode,
                    clip_trail=self.clip_trail,
                    sensitivity=float(self.sensitivity),
                    progress_cb=None,  # keep this simple; we emit per-file progress instead
                )

                if self.skip_save and (not detected):
                    self.log_signal.emit(f"Skip (no trail): {os.path.basename(fp_in)}")
                    return

                out = np.asarray(out, dtype=np.float32, order="C")
                out = np.clip(out, 0.0, 1.0)

                # Choose save behavior: keep original extension/name
                # (You can change naming here if you want.)
                save_image(out, fp_out, "auto", None, hdr, mono)
                self.log_signal.emit(f"Saved: {os.path.basename(fp_out)}")

            # -------- batch (single pass) OR monitor (loop) --------
            while not self._cancel:
                files = list(self._iter_files())

                # monitor: only process new files
                todo = [fn for fn in files if fn not in self._seen]
                total = len(todo)
                done = 0

                for fn in todo:
                    if self._cancel:
                        break

                    self._seen.add(fn)
                    fp_in = os.path.join(self.input_dir, fn)
                    fp_out = os.path.join(self.output_dir, fn)

                    # if output already exists, treat as seen
                    if os.path.exists(fp_out):
                        self.log_signal.emit(f"Exists, skipping: {fn}")
                        done += 1
                        self.progress_signal.emit(done, total)
                        continue

                    self.log_signal.emit(f"Processing: {fn}")
                    try:
                        process_one(fp_in, fp_out)
                    except Exception as e:
                        self.log_signal.emit(f"Error {fn}: {e}")

                    done += 1
                    self.progress_signal.emit(done, total)

                if not self.monitor:
                    break

                # monitor mode: sleep/poll
                self.msleep(int(max(50, self.poll_seconds * 1000)))

            if self._cancel:
                self.log_signal.emit("Cancelled.")
            else:
                self.log_signal.emit("Done.")

        except Exception as e:
            self.log_signal.emit(f"Unexpected error: {e}")
        finally:
            self.finished_signal.emit()
