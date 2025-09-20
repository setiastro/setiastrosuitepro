# pro/cosmicclarity.py
from __future__ import annotations
import os, sys, glob, time
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
from legacy.image_manager import load_image, save_image  

from imageops.stretch import stretch_mono_image, stretch_color_image

import shutil, subprocess




# =============================================================================
# Small helpers
# =============================================================================
def _satellite_exe_name() -> str:
    base = "setiastrocosmicclarity_satellite"
    return f"{base}.exe" if os.name == "nt" else base


def _get_cosmic_root_from_settings() -> str:
    # Uses the same key your SettingsDialog writes:
    # self.settings.setValue("paths/cosmic_clarity", <path>)
    s = QSettings()
    return s.value("paths/cosmic_clarity", "", type=str) or ""

def _ensure_dirs(root: str):
    os.makedirs(os.path.join(root, "input"), exist_ok=True)
    os.makedirs(os.path.join(root, "output"), exist_ok=True)

_IMG_EXTS = ('.png', '.tif', '.tiff', '.fit', '.fits', '.xisf',
             '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef',
             '.jpg', '.jpeg')

def _purge_dir(path: str, *, prefix: str | None = None):
    """Delete lingering image-like files in a folder. Safe: files only."""
    try:
        if not os.path.isdir(path):
            return
        for fn in os.listdir(path):
            fp = os.path.join(path, fn)
            if not os.path.isfile(fp):
                continue
            if prefix and not fn.startswith(prefix):
                continue
            if os.path.splitext(fn)[1].lower() in _IMG_EXTS:
                try: os.remove(fp)
                except Exception: pass
    except Exception:
        pass

def _purge_cc_io(root: str, *, clear_input: bool, clear_output: bool, prefix: str | None = None):
    """Convenience to purge CC input/output dirs."""
    try:
        if clear_input:
            _purge_dir(os.path.join(root, "input"),  prefix=prefix)
        if clear_output:
            _purge_dir(os.path.join(root, "output"), prefix=prefix)
    except Exception:
        pass

def _platform_exe_names(mode: str) -> str:
    """
    Return executable filename for sharpen/denoise based on OS.
    Matches SASv2 you pasted:
      - Windows: SetiAstroCosmicClarity.exe / SetiAstroCosmicClarity_denoise.exe
      - macOS  : SetiAstroCosmicClaritymac / SetiAstroCosmicClarity_denoisemac
      - Linux  : SetiAstroCosmicClarity / SetiAstroCosmicClarity_denoise
    """
    is_win = os.name == "nt"
    is_mac = sys.platform == "darwin"
    if mode == "sharpen":
        return "SetiAstroCosmicClarity.exe" if is_win else ("SetiAstroCosmicClaritymac" if is_mac else "SetiAstroCosmicClarity")
    elif mode == "denoise":
        return "SetiAstroCosmicClarity_denoise.exe" if is_win else ("SetiAstroCosmicClarity_denoisemac" if is_mac else "SetiAstroCosmicClarity_denoise")
    elif mode == "superres":
        # SASv2 used lowercase for superres on Windows
        return "setiastrocosmicclarity_superres.exe" if is_win else "setiastrocosmicclarity_superres"
    else:
        return ""


# =============================================================================
# Wait UI
# =============================================================================
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


class WaitForFileWorker(QThread):
    fileFound = pyqtSignal(str)
    cancelled = pyqtSignal()
    error     = pyqtSignal(str)
    def __init__(self, glob_pat: str, timeout_sec=1800, parent=None):
        super().__init__(parent)
        self._glob = glob_pat
        self._timeout = timeout_sec
        self._running = True
    def run(self):
        start = time.time()
        while self._running and (time.time() - start < self._timeout):
            m = glob.glob(self._glob)
            if m:
                self.fileFound.emit(m[0]); return
            time.sleep(1)
        if self._running: self.error.emit("Output file not found within timeout.")
        else:             self.cancelled.emit()
    def stop(self): self._running = False


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
            except Exception: pass
            QTimer.singleShot(0, self.reject)
            return        
        self.setWindowTitle("Cosmic Clarity")
        if icon: 
            try: self.setWindowIcon(icon)
            except Exception: pass

        self.parent_ref = parent
        self.doc = doc
        self.orig = np.clip(np.asarray(doc.image, dtype=np.float32), 0.0, 1.0)
        self.cosmic_root = _get_cosmic_root_from_settings()

        v = QVBoxLayout(self)

        # ---------------- Controls ----------------
        grp = QGroupBox("Parameters")
        grid = QGridLayout(grp)

        # Mode
        grid.addWidget(QLabel("Mode:"), 0, 0)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Sharpen", "Denoise", "Both", "Super Resolution"])
        self.cmb_mode.currentIndexChanged.connect(self._mode_changed)
        grid.addWidget(self.cmb_mode, 0, 1, 1, 2)

        # GPU
        grid.addWidget(QLabel("Use GPU:"), 1, 0)
        self.cmb_gpu = QComboBox(); self.cmb_gpu.addItems(["Yes", "No"])
        grid.addWidget(self.cmb_gpu, 1, 1)

        # Sharpen block
        self.lbl_sh_mode = QLabel("Sharpening Mode:")
        self.cmb_sh_mode = QComboBox(); self.cmb_sh_mode.addItems(["Both", "Stellar Only", "Non-Stellar Only"])
        grid.addWidget(self.lbl_sh_mode, 2, 0); grid.addWidget(self.cmb_sh_mode, 2, 1)

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
        b_run   = QPushButton("Execute"); b_run.clicked.connect(self._run_main)
        b_close = QPushButton("Close");   b_close.clicked.connect(self.reject)
        row.addStretch(1); row.addWidget(b_run); row.addWidget(b_close)
        v.addLayout(row)

        self._mode_changed()  # set initial visibility

        self._wait = None
        self._wait_thread = None
        self._proc = None

        self._headless = bool(headless)
        if self._headless:
            # Don’t show the control panel; we’ll still exec() to run the event loop.
            try: self.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            except Exception: pass
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
        for w in (self.lbl_sh_mode, self.cmb_sh_mode, self.chk_auto_psf, self.lbl_psf, self.sld_psf, self.lbl_st_amt, self.sld_st_amt, self.lbl_nst_amt, self.sld_nst_amt):
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
        if not self.cosmic_root:
            QMessageBox.warning(self, "Cosmic Clarity", "No Cosmic Clarity folder is set. Set it in Preferences (Settings).")
            return False
        # basic presence check (don’t force a specific exe here, we do that later)
        if not os.path.isdir(self.cosmic_root):
            QMessageBox.warning(self, "Cosmic Clarity", "The Cosmic Clarity folder in Settings doesn’t exist anymore.")
            return False
        return True

    # ----- Execution -----
    def _run_main(self):
        if not self._validate_root(): return
        _ensure_dirs(self.cosmic_root)
        _purge_cc_io(self.cosmic_root, clear_input=True, clear_output=False)

        # Determine queue of operations
        mode_idx = self.cmb_mode.currentIndex()
        if mode_idx == 3:
            # Super-res path
            self._run_superres(); return
        elif mode_idx == 0:
            ops = [("sharpen", "_sharpened")]
        elif mode_idx == 1:
            ops = [("denoise", "_denoised")]
        else:
            ops = [("sharpen", "_sharpened"), ("denoise", "_denoised")]

        # Save current doc image to input
        base = self._base_name()
        in_path = os.path.join(self.cosmic_root, "input", f"{base}.tif")
        try:
            # Use your save_image helper. 32-bit float is what SASv2 used for input.
            save_image(self.orig, in_path, "tiff", "32-bit floating point", getattr(self.doc, "original_header", None), getattr(self.doc, "is_mono", False))
        except Exception as e:
            QMessageBox.critical(self, "Cosmic Clarity", f"Failed to save input TIFF:\n{e}")
            return

        # Run queue
        self._op_queue = ops
        self._current_input = in_path
        self._run_next()

    def _run_next(self):
        if not self._op_queue:
            # If we ever get here without more steps, we’re done.
            self.accept()
            return
        mode, suffix = self._op_queue.pop(0)
        exe_name = _platform_exe_names(mode)
        exe_path = os.path.join(self.cosmic_root, exe_name)
        if not os.path.exists(exe_path):
            QMessageBox.critical(self, "Cosmic Clarity", f"Executable not found:\n{exe_path}")
            return

        # Build args (SASv2 flags mirrored)
        args = []
        if mode == "sharpen":
            psf = self.sld_psf.value()/10.0
            args += ["--sharpening_mode", self.cmb_sh_mode.currentText(),
                     "--stellar_amount", f"{self.sld_st_amt.value()/100:.2f}",
                     "--nonstellar_strength", f"{psf:.1f}",
                     "--nonstellar_amount", f"{self.sld_nst_amt.value()/100:.2f}"]
            if self.chk_auto_psf.isChecked():
                args.append("--auto_detect_psf")
        elif mode == "denoise":
            args += ["--denoise_strength", f"{self.sld_dn_lum.value()/100:.2f}",
                     "--color_denoise_strength", f"{self.sld_dn_col.value()/100:.2f}",
                     "--denoise_mode", self.cmb_dn_mode.currentText()]
            if self.chk_dn_sep.isChecked():
                args.append("--separate_channels")

        if self.cmb_gpu.currentText() == "No" and mode in ("sharpen","denoise"):
            args.append("--disable_gpu")

        # Run process
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

        self._proc.readyReadStandardOutput.connect(self._read_proc_output_main)
        from functools import partial
        self._proc.finished.connect(partial(self._on_proc_finished, mode, suffix))
        self._proc.setProgram(exe_path)
        self._proc.setArguments(args)
        self._proc.start()
        if not self._proc.waitForStarted(3000):
            QMessageBox.critical(self, "Cosmic Clarity", "Failed to start process.")
            return

        # Wait for output file
        base = self._base_name()
        out_glob = os.path.join(self.cosmic_root, "output", f"{base}{suffix}.*")
        self._wait = WaitDialog(f"Cosmic Clarity – {mode.title()}", self)
        self._wait.cancelled.connect(self._cancel_all)
        self._wait.show()

        self._wait_thread = WaitForFileWorker(out_glob, timeout_sec=1800, parent=self)
        self._wait_thread.fileFound.connect(lambda path, mode=mode: self._on_output_file(path, mode))
        self._wait_thread.error.connect(self._on_wait_error)
        self._wait_thread.cancelled.connect(self._on_wait_cancel)
        self._wait_thread.start()

    def _read_proc_output_main(self):
        self._read_proc_output(self._proc, which="main")

    def _read_proc_output(self, proc: QProcess, which="main"):
        out = proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        if not self._wait: return
        for line in out.splitlines():
            line = line.strip()
            if not line: continue
            if line.startswith("Progress:"):
                try:
                    pct = float(line.split()[1].replace("%",""))
                    self._wait.set_progress(int(pct))
                except Exception:
                    pass
            else:
                self._wait.append_output(line)

    def _on_proc_finished(self, mode, suffix, code, status):
        if code != 0:
            if self._wait: self._wait.append_output(f"Process exited with code {code}.")
            # still let the file-watcher decide success/failure (some exes write before exit)

    def _on_output_file(self, out_path: str, mode: str):
        # stop waiting UI
        if self._wait: self._wait.close(); self._wait = None
        if self._wait_thread: self._wait_thread.stop(); self._wait_thread = None

        # Load processed image we just got
        try:
            img, hdr, bd, mono = load_image(out_path)
            if img is None:
                raise RuntimeError("Unable to load output image.")
        except Exception as e:
            QMessageBox.critical(self, "Cosmic Clarity", f"Failed to load output:\n{e}")
            return

        dest = img.astype(np.float32, copy=False)

        # Apply to document (so the user sees the step result immediately)
        step_title = f"Cosmic Clarity – {mode.title()}"
        create_new = (self.cmb_target.currentIndex() == 1)

        if create_new:
            ok = self._spawn_new_doc_from_numpy(dest, step_title)
            if not ok:
                # fall back to overwriting if we couldn’t spawn a new doc
                self._apply_to_active(dest, step_title)
        else:
            self._apply_to_active(dest, step_title)

        # Will we run another step (i.e., we're in "Both")?
        has_more = bool(self._op_queue)
        base     = self._base_name()
        next_in  = os.path.join(self.cosmic_root, "input", f"{base}.tif")
        prev_in  = getattr(self, "_current_input", None)

        try:
            if has_more:
                # IMPORTANT: Stage the current result as the next step's input
                save_image(
                    dest, next_in, "tiff", "32-bit floating point",
                    getattr(self.doc, "original_header", None),
                    getattr(self.doc, "is_mono", False)
                )
                self._current_input = next_in

            # Now it’s safe to clean up the produced output
            if out_path and os.path.exists(out_path):
                os.remove(out_path)

            # Remove the previous input file if it’s different from the new one
            if prev_in and prev_in != next_in and os.path.exists(prev_in):
                os.remove(prev_in)

        except Exception as e:
            QMessageBox.critical(self, "Cosmic Clarity", f"Failed while staging next step:\n{e}")
            self._op_queue.clear()
            return

        # Continue or finish
        if has_more:
            QTimer.singleShot(100, self._run_next)
        else:
            # Nothing else queued — we're done
            try:
                # 🔸 Final cleanup: clear both input & output
                _purge_cc_io(self.cosmic_root, clear_input=True, clear_output=True)
            except Exception:
                pass
            self.accept()


    def _on_wait_error(self, msg: str):
        if self._wait: self._wait.close(); self._wait = None
        if self._wait_thread: self._wait_thread.stop(); self._wait_thread = None
        QMessageBox.critical(self, "Cosmic Clarity", msg)

    def _on_wait_cancel(self):
        if self._wait: self._wait.close(); self._wait = None
        if self._wait_thread: self._wait_thread.stop(); self._wait_thread = None

    def _cancel_all(self):
        try:
            if self._proc: self._proc.kill()
        except Exception: pass
        self._on_wait_cancel()

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
    def _run_superres(self):
        exe_name = _platform_exe_names("superres")
        exe_path = os.path.join(self.cosmic_root, exe_name)
        if not os.path.exists(exe_path):
            QMessageBox.critical(self, "Cosmic Clarity", f"Super Resolution executable not found:\n{exe_path}")
            return

        _ensure_dirs(self.cosmic_root)
        # 🔸 purge output too so any file that appears is from THIS run
        _purge_cc_io(self.cosmic_root, clear_input=True, clear_output=True)

        base = self._base_name()
        in_path = os.path.join(self.cosmic_root, "input", f"{base}.tif")
        try:
            save_image(self.orig, in_path, "tiff", "32-bit floating point",
                    getattr(self.doc, "original_header", None),
                    getattr(self.doc, "is_mono", False))
        except Exception as e:
            QMessageBox.critical(self, "Cosmic Clarity", f"Failed to save input TIFF:\n{e}")
            return
        self._current_input = in_path

        scale = int(self.cmb_scale.currentText().replace("x", ""))
        # keep args as-is if your superres build expects explicit paths
        args = [
            "--input", in_path,
            "--output_dir", os.path.join(self.cosmic_root, "output"),
            "--scale", str(scale),
            "--model_dir", self.cosmic_root
        ]

        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._read_superres_output_main)
        # finished handler not required; the file watcher drives success
        self._proc.setProgram(exe_path)
        self._proc.setArguments(args)
        self._proc.start()
        if not self._proc.waitForStarted(3000):
            QMessageBox.critical(self, "Cosmic Clarity", "Failed to start Super Resolution process.")
            return

        self._wait = WaitDialog("Cosmic Clarity – Super Resolution", self)
        self._wait.cancelled.connect(self._cancel_all)
        self._wait.show()

        # 🔸 Watch broadly; we purged output so the first file is from this run.
        # We'll still re-pick the exact file in the slot for safety.
        self._sr_base = base
        self._sr_scale = scale
        out_glob = os.path.join(self.cosmic_root, "output", "*.*")

        self._wait_thread = WaitForFileWorker(out_glob, timeout_sec=1800, parent=self)
        self._wait_thread.fileFound.connect(self._on_superres_file)   # path arg is ignored; we reselect
        self._wait_thread.error.connect(self._on_wait_error)
        self._wait_thread.cancelled.connect(self._on_wait_cancel)
        self._wait_thread.start()


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
        # Denoise
        self.sld_dn_lum.setValue(int(max(0, min(100, round(float(p.get("denoise_luma",0.5))*100)))))
        self.sld_dn_col.setValue(int(max(0, min(100, round(float(p.get("denoise_color",0.5))*100)))))
        self.cmb_dn_mode.setCurrentText(str(p.get("denoise_mode","full")))
        self.chk_dn_sep.setChecked(bool(p.get("separate_channels", False)))
        # Super-Res
        self.cmb_scale.setCurrentText(str(int(p.get("scale",2))))


    def _read_superres_output_main(self):
        self._read_superres_output(self._proc)

    def _read_superres_output(self, proc: QProcess):
        out = proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        if not self._wait: return
        for line in out.splitlines():
            if line.startswith("PROGRESS:") or line.startswith("Progress:"):
                try:
                    tail = line.split(":",1)[1] if ":" in line else line.split()[1]
                    pct = int(float(tail.strip().replace("%","")))
                    self._wait.set_progress(pct)
                except Exception:
                    pass
            else:
                self._wait.append_output(line)

    def _pick_superres_output(self, base: str, scale: int) -> str | None:
        """
        Find the most plausible super-res output file. We try several common
        name patterns, then fall back to the newest/largest file in the output dir.
        """
        out_dir = os.path.join(self.cosmic_root, "output")

        def _best(paths: list[str]) -> str | None:
            if not paths:
                return None
            # prefer bigger file; tie-break by newest mtime
            paths.sort(key=lambda p: (os.path.getsize(p), os.path.getmtime(p)), reverse=True)
            return paths[0]

        # common patterns used by different builds
        patterns = [
            f"{base}_upscaled{scale}.*",
            f"{base}_upscaled*.*",
            f"{base}*upscal*.*",
            f"{base}*superres*.*",
        ]
        for pat in patterns:
            hit = _best(glob.glob(os.path.join(out_dir, pat)))
            if hit:
                return hit

        # fallback: anything in output (we purge it first, so whatever appears is ours)
        return _best(glob.glob(os.path.join(out_dir, "*.*")))


    def _on_superres_file(self, _first_path_from_watcher: str):
        # stop waiting UI
        if self._wait: self._wait.close(); self._wait = None
        if self._wait_thread: self._wait_thread.stop(); self._wait_thread = None

        # pick the actual output (robust to naming)
        base  = getattr(self, "_sr_base", self._base_name())
        scale = int(getattr(self, "_sr_scale", int(self.cmb_scale.currentText().replace("x",""))))
        out_path = self._pick_superres_output(base, scale)
        if not out_path or not os.path.exists(out_path):
            QMessageBox.critical(self, "Cosmic Clarity", "Super Resolution output file not found.")
            return

        try:
            img, hdr, bd, mono = load_image(out_path)
            if img is None:
                raise RuntimeError("Unable to load output image.")
        except Exception as e:
            QMessageBox.critical(self, "Cosmic Clarity", f"Failed to load Super Resolution output:\n{e}")
            return

        dest = img.astype(np.float32, copy=False)
        step_title = "Cosmic Clarity – Super Resolution"
        create_new = (self.cmb_target.currentIndex() == 1)

        if create_new:
            ok = self._spawn_new_doc_from_numpy(dest, step_title)
            if not ok:
                self._apply_to_active(dest, step_title)
        else:
            self._apply_to_active(dest, step_title)

        # cleanup mirrors sharpen/denoise
        try:
            if getattr(self, "_current_input", None) and os.path.exists(self._current_input):
                os.remove(self._current_input)
            if os.path.exists(out_path):
                os.remove(out_path)
            _purge_cc_io(self.cosmic_root, clear_input=True, clear_output=True)
        except Exception:
            pass

        self.accept()



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
        if icon:
            try: self.setWindowIcon(icon)
            except Exception: pass

        self.settings = QSettings()
        self.cosmic_clarity_folder = self.settings.value("paths/cosmic_clarity", "", type=str) or ""
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

        # Folder display + chooser for Cosmic Clarity root
        self.lbl_root = QLabel(f"Folder: {self.cosmic_clarity_folder or 'Not set'}")
        left.addWidget(self.lbl_root)
        self.btn_pick_root = QPushButton("Choose Cosmic Clarity Folder…"); self.btn_pick_root.clicked.connect(self._choose_root)
        left.addWidget(self.btn_pick_root)

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

    # ---------- Settings / root ----------
    def _choose_root(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Cosmic Clarity Folder", self.cosmic_clarity_folder or "")
        if not folder: return
        self.cosmic_clarity_folder = folder
        self.settings.setValue("paths/cosmic_clarity", folder)
        self.lbl_root.setText(f"Folder: {folder}")

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
                                    '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef', '.jpg', '.jpeg')):
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
        # Gather possible open views
        views = self._collect_open_views()

        # Decide source: view or file
        use_view = False
        if views:
            mb = QMessageBox(self)
            mb.setWindowTitle("Process Single Image")
            mb.setText("Choose the source to process:")
            btn_view  = mb.addButton("Open View", QMessageBox.ButtonRole.AcceptRole)
            btn_file  = mb.addButton("File on Disk", QMessageBox.ButtonRole.AcceptRole)
            mb.addButton(QMessageBox.StandardButton.Cancel)
            mb.exec()
            if mb.clickedButton() is btn_view:
                use_view = True
            elif mb.clickedButton() is None or mb.clickedButton() == mb.buttons()[-1]:  # Cancel
                return

        # --- Branch 1: Process an OPEN VIEW ---
        if use_view:
            # If multiple views, ask which one
            chosen_doc = None
            if len(views) == 1:
                chosen_doc = views[0][1]
                base_name  = self._base_name_for_doc(chosen_doc)
            else:
                titles = [t for (t, _) in views]
                sel, ok = QInputDialog.getItem(self, "Select View", "Choose an open view:", titles, 0, False)
                if not ok:
                    return
                idx = titles.index(sel)
                chosen_doc = views[idx][1]
                base_name  = self._base_name_for_doc(chosen_doc)

            # Stage image from the chosen view
            temp_in  = self._create_temp_folder()
            temp_out = self._create_temp_folder()
            staged_in = os.path.join(temp_in, f"{base_name}.tif")

            try:
                # 32-bit float TIFF like SASv2
                img = np.clip(np.asarray(chosen_doc.image, dtype=np.float32), 0.0, 1.0)
                save_image(
                    img, staged_in,
                    "tiff", "32-bit floating point",
                    getattr(chosen_doc, "original_header", None),
                    getattr(chosen_doc, "is_mono", False)
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to stage view for processing:\n{e}")
                return

            # Run satellite
            try:
                self._run_satellite(input_dir=temp_in, output_dir=temp_out, live=False)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error processing image:\n{e}")
                return

            # Pick up result and apply back to the view
            out = glob.glob(os.path.join(temp_out, "*_satellited.*"))
            if not out:
                # Likely --skip-save and no trail, or failure
                QMessageBox.information(self, "Satellite Removal", "No output produced (possibly no satellite trail detected).")
            else:
                out_path = out[0]
                try:
                    result, hdr, bd, mono = load_image(out_path)
                    if result is None:
                        raise RuntimeError("Unable to load output image.")
                    result = result.astype(np.float32, copy=False)

                    # Apply back to the chosen doc
                    if hasattr(chosen_doc, "set_image"):
                        chosen_doc.set_image(result, step_name="Cosmic Clarity – Satellite Removal")
                    elif hasattr(chosen_doc, "apply_numpy"):
                        chosen_doc.apply_numpy(result, step_name="Cosmic Clarity – Satellite Removal")
                    else:
                        chosen_doc.image = result
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to apply result to view:\n{e}")
                    # fall through to cleanup
                finally:
                    # Clean up temp files
                    try:
                        if os.path.exists(out_path): os.remove(out_path)
                    except Exception:
                        pass

            # Clean up temp dirs
            try:
                shutil.rmtree(temp_in, ignore_errors=True)
                shutil.rmtree(temp_out, ignore_errors=True)
            except Exception:
                pass

            return  # done

        # --- Branch 2: Process a FILE on disk ---
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Image Files (*.png *.tif *.tiff *.fit *.fits *.xisf *.cr2 *.nef *.arw *.dng *.orf *.rw2 *.pef *.jpg *.jpeg)"
        )
        if not file_path:
            QMessageBox.warning(self, "Warning", "No file selected.")
            return

        temp_in  = self._create_temp_folder()
        temp_out = self._create_temp_folder()
        try:
            shutil.copy(file_path, temp_in)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to stage input:\n{e}")
            return

        try:
            self._run_satellite(input_dir=temp_in, output_dir=temp_out, live=False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing image:\n{e}")
            return

        # Move output back next to original
        out = glob.glob(os.path.join(temp_out, "*_satellited.*"))
        if out:
            dst = os.path.join(os.path.dirname(file_path), os.path.basename(out[0]))
            try:
                shutil.move(out[0], dst)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save result:\n{e}")
                return
            QMessageBox.information(self, "Success", f"Processed image saved to:\n{dst}")
        else:
            QMessageBox.warning(self, "Warning", "No output file found.")

        # Cleanup
        try:
            shutil.rmtree(temp_in, ignore_errors=True)
            shutil.rmtree(temp_out, ignore_errors=True)
        except Exception:
            pass

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


    # ---------- Batch ----------
    def _batch_process(self):
        if not self.input_folder or not self.output_folder:
            QMessageBox.warning(self, "Warning", "Please select both input and output folders.")
            return
        exe = os.path.join(self.cosmic_clarity_folder, _satellite_exe_name())
        if not os.path.exists(exe):
            QMessageBox.critical(self, "Error", f"Executable not found:\n{exe}")
            return

        cmd = self._build_cmd(exe, self.input_folder, self.output_folder, batch=True, monitor=False)
        self._run_threaded(cmd, title="Satellite – Batch processing")

    # ---------- Live monitor ----------
    def _live_monitor(self):
        if not self.input_folder or not self.output_folder:
            QMessageBox.warning(self, "Warning", "Please select both input and output folders.")
            return
        exe = os.path.join(self.cosmic_clarity_folder, _satellite_exe_name())
        if not os.path.exists(exe):
            QMessageBox.critical(self, "Error", f"Executable not found:\n{exe}")
            return

        cmd = self._build_cmd(exe, self.input_folder, self.output_folder, batch=False, monitor=True)
        self.sld_sens.setEnabled(False)
        self._run_threaded(cmd, title="Satellite – Live monitoring", on_finish=lambda: self.sld_sens.setEnabled(True))

    # ---------- Command / run ----------
    def _build_cmd(self, exe_path: str, in_dir: str, out_dir: str, *, batch: bool, monitor: bool):
        cmd = [
            exe_path,
            "--input", in_dir,
            "--output", out_dir,
            "--mode", self.cmb_mode.currentText().lower(),
        ]
        if self.cmb_gpu.currentText() == "Yes":
            cmd.append("--use-gpu")
        if self.chk_clip.isChecked():
            cmd.append("--clip-trail")
        else:
            cmd.append("--no-clip-trail")
        if self.chk_skip.isChecked():
            cmd.append("--skip-save")
        if batch:
            cmd.append("--batch")
        if monitor:
            cmd.append("--monitor")
        cmd += ["--sensitivity", f"{self.sensitivity}"]
        return cmd

    def _run_threaded(self, cmd, title="Processing…", on_finish=None):
        # Wait dialog + threaded subprocess (mirrors SASv2 SatelliteProcessingThread)
        self._wait = WaitDialog(title, self)
        self._wait.show()

        self._sat_thread = SatelliteProcessingThread(cmd)
        self._sat_thread.log_signal.connect(self._wait.append_output)
        self._sat_thread.finished_signal.connect(lambda: self._on_thread_finished(on_finish))
        self._sat_thread.start()

    def _on_thread_finished(self, on_finish):
        if self._wait: self._wait.close(); self._wait = None
        if callable(on_finish):
            try: on_finish()
            except Exception: pass
        QMessageBox.information(self, "Done", "Processing finished.")

    def _run_satellite(self, *, input_dir: str, output_dir: str, live: bool):
        if not self.cosmic_clarity_folder:
            raise RuntimeError("Cosmic Clarity folder not set. Choose it in Preferences or with the button below.")
        exe = os.path.join(self.cosmic_clarity_folder, _satellite_exe_name())
        if not os.path.exists(exe):
            raise FileNotFoundError(f"Executable not found: {exe}")

        cmd = self._build_cmd(exe, input_dir, output_dir, batch=not live, monitor=live)
        print("Running command:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    # ---------- Utils ----------
    @staticmethod
    def _create_temp_folder(base="~"):
        user_dir = os.path.expanduser(base)
        temp_folder = os.path.join(user_dir, "CosmicClarityTemp")
        os.makedirs(temp_folder, exist_ok=True)
        return temp_folder

class ImagePreviewDialog(QDialog):
    def __init__(self, np_image, is_mono=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Preview")
        self.resize(640, 480)
        self.autostretch_enabled = False
        self.is_mono = is_mono
        self.zoom_factor = 1.0
        self.np_image = np.clip(np_image, 0, 1).astype(np.float32)

        v = QVBoxLayout(self)

        row = QHBoxLayout()
        self.btn_auto = QPushButton("AutoStretch (Off)")
        self.btn_auto.setCheckable(True)
        self.btn_auto.toggled.connect(self._toggle_autostretch)
        row.addWidget(self.btn_auto)

        self.btn_zi = QPushButton("Zoom In");  self.btn_zi.clicked.connect(self._zoom_in)
        self.btn_zo = QPushButton("Zoom Out"); self.btn_zo.clicked.connect(self._zoom_out)
        row.addWidget(self.btn_zi); row.addWidget(self.btn_zo)
        v.addLayout(row)

        self.scroll = QTextEdit()  # placeholder, replaced by QLabel in a scroll area
        # Real scroll area:
        self.area = QFileDialog()  # placeholder to keep names unique
        self.area = QWidget()      # not used

        self.scroll_area = QFileDialog()  # placeholder
        from PyQt6.QtWidgets import QScrollArea
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        v.addWidget(self.scroll_area)

        self.lbl = QLabel()
        self.scroll_area.setWidget(self.lbl)
        self._display_qimage(self.np_image)

        self.lbl.installEventFilter(self)

        QTimer.singleShot(0, self._center_scrollbars)

    def _display_qimage(self, np_img):
        arr = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 3:
            h, w, _ = arr.shape; bpl = 3 * w
            qimg = QImage(arr.tobytes(), w, h, bpl, QImage.Format.Format_RGB888)
        elif arr.ndim == 2:
            h, w = arr.shape; bpl = w
            qimg = QImage(arr.tobytes(), w, h, bpl, QImage.Format.Format_Grayscale8)
        else:
            raise ValueError(f"Unexpected image shape: {arr.shape}")
        pm = QPixmap.fromImage(qimg)
        pm = pm.scaled(int(pm.width()*self.zoom_factor), int(pm.height()*self.zoom_factor),
                       Qt.AspectRatioMode.KeepAspectRatio)
        self.lbl.setPixmap(pm); self.lbl.adjustSize()

    def _toggle_autostretch(self, checked):
        self.autostretch_enabled = checked
        self.btn_auto.setText("AutoStretch (On)" if checked else "AutoStretch (Off)")
        self._apply_autostretch()

    def _apply_autostretch(self):
        tgt = 0.25
        if self.autostretch_enabled:
            if self.np_image.ndim == 2:
                disp = np.stack([stretch_mono_image(self.np_image, tgt)]*3, axis=-1)
            elif self.np_image.ndim == 3 and self.np_image.shape[2] == 3:
                disp = stretch_color_image(self.np_image, tgt, linked=False)
            else:
                disp = self.np_image
        else:
            if self.np_image.ndim == 2:
                disp = np.stack([self.np_image]*3, axis=-1)
            else:
                disp = self.np_image
        self._display_qimage(disp)

    def _zoom_in(self):
        self.zoom_factor *= 1.2
        self._apply_autostretch() if self.autostretch_enabled else self._display_qimage(self.np_image)

    def _zoom_out(self):
        self.zoom_factor /= 1.2
        self._apply_autostretch() if self.autostretch_enabled else self._display_qimage(self.np_image)

    def eventFilter(self, source, event):
        if source == self.lbl and event.type() == QEvent.Type.Wheel:
            self._zoom_in() if event.angleDelta().y() > 0 else self._zoom_out()
            return True
        return super().eventFilter(source, event)

    def _center_scrollbars(self):
        sa = self.scroll_area
        h = sa.horizontalScrollBar(); v = sa.verticalScrollBar()
        h.setValue((h.maximum()+h.minimum())//2)
        v.setValue((v.maximum()+v.minimum())//2)


class SatelliteProcessingThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    def __init__(self, command):
        super().__init__()
        self.command = command
    def run(self):
        try:
            self.log_signal.emit("Running command: " + " ".join(self.command))
            subprocess.run(self.command, check=True)
            self.log_signal.emit("Processing complete.")
        except subprocess.CalledProcessError as e:
            self.log_signal.emit(f"Processing failed: {e}")
        except Exception as e:
            self.log_signal.emit(f"Unexpected error: {e}")
        finally:
            self.finished_signal.emit()
