# pro/cosmicclarity_preset.py
from __future__ import annotations
from setiastro.saspro.main_helpers import non_blocking_sleep
import os
import sys
import time
import glob
import shutil
import subprocess
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QSettings, QLockFile
from PyQt6.QtWidgets import QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox, QComboBox, QCheckBox, QMessageBox, QApplication, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QWidget
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QSettings, QLockFile, QEventLoop


# reuse your legacy IO + helpers
from setiastro.saspro.legacy.image_manager import load_image, save_image

from .remove_stars import _ProcThread, _ProcDialog
from .cosmicclarity import CosmicClarityDialogPro

# pull tiny helpers from the main CC module (or re-declare)
def _platform_exe_names(mode: str) -> str:
    is_win = os.name == "nt"
    is_mac = sys.platform == "darwin"
    if mode == "sharpen":
        return "SetiAstroCosmicClarity.exe" if is_win else ("SetiAstroCosmicClaritymac" if is_mac else "SetiAstroCosmicClarity")
    elif mode == "denoise":
        return "SetiAstroCosmicClarity_denoise.exe" if is_win else ("SetiAstroCosmicClarity_denoisemac" if is_mac else "SetiAstroCosmicClarity_denoise")
    elif mode == "superres":
        return "setiastrocosmicclarity_superres.exe" if is_win else "setiastrocosmicclarity_superres"
    return ""

def _cosmic_root(main) -> str:
    s = getattr(main, "settings", None)
    if not s:
        return ""
    try:
        return s.value("paths/cosmic_clarity", "", type=str) or ""
    except Exception:
        return s.value("paths/cosmic_clarity", "") or ""

def _base_from_doc(doc) -> str:
    fp = getattr(doc, "file_path", None)
    if isinstance(fp, str) and fp:
        return os.path.splitext(os.path.basename(fp))[0]
    name = getattr(doc, "display_name", None)
    if callable(name):
        try:
            n = name() or ""
            if n:
                return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in n).strip("_") or "image"
        except Exception:
            pass
    return "image"


# ---------------- Worker (no UI) ----------------
class _CCHeadlessWorker(QThread):
    finished_ok = pyqtSignal(np.ndarray)
    failed      = pyqtSignal(str)
    log         = pyqtSignal(str)
    progress    = pyqtSignal(int)   # 0..100
    step_changed= pyqtSignal(str)   # "sharpen" | "denoise" | "superres"

    def __init__(self, cosmic_root: str, doc, ops: list[tuple[str, str]], params: dict, create_new: bool):
        super().__init__()
        self.root = cosmic_root
        self.doc = doc
        self.ops = ops
        self.p   = params
        self.create_new = create_new
        self._stop  = False
        self._proc  = None

    def cancel(self):
        self._stop = True
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                try: self._proc.wait(timeout=5)
                except Exception: self._proc.kill()
        except Exception:
            pass

    def _emit_progress_line(self, line: str):
        s = line.strip()
        if not s:
            return
        self.log.emit(s)
        # Parse both formats used by your tools
        if s.startswith("Progress:"):
            try: self.progress.emit(int(float(s.split()[1].replace("%",""))))
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        elif s.startswith("PROGRESS:"):
            try: self.progress.emit(int(s.split(":",1)[1].strip().replace("%","")))
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

    def run(self):
        try:
            print("starting cc headless worker")
            in_dir  = os.path.join(self.root, "input")
            out_dir = os.path.join(self.root, "output")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)

            base    = _base_from_doc(self.doc)
            in_path = os.path.join(in_dir, f"{base}.tif")

            # Stage current image
            img = np.clip(np.asarray(self.doc.image), 0.0, 1.0).astype(np.float32, copy=False)
            save_image(img, in_path, "tiff", "32-bit floating point",
                       getattr(self.doc, "original_header", None),
                       getattr(self.doc, "is_mono", False))

            staged_input = in_path
            result = None
            print("entering cc headless run")
            for (mode, suffix) in self.ops:
                if self._stop: raise RuntimeError("Cancelled")
                self.step_changed.emit(mode)

                exe = os.path.join(self.root, _platform_exe_names(mode))
                if not os.path.exists(exe):
                    raise RuntimeError(f"Cosmic Clarity executable not found:\n{exe}")

                # Build args
                args = []
                if mode == "sharpen":
                    if not self.p.get("gpu", True):
                        args.append("--disable_gpu")
                    if self.p.get("auto_psf", True):
                        args.append("--auto_detect_psf")
                    args += [
                        "--sharpening_mode", self.p.get("sharpening_mode", "Both"),
                        "--stellar_amount", f"{float(self.p.get('stellar_amount', 0.50)):.2f}",
                        "--nonstellar_strength", f"{float(self.p.get('nonstellar_psf', 3.0)):.1f}",
                        "--nonstellar_amount", f"{float(self.p.get('nonstellar_amount', 0.50)):.2f}",
                    ]
                    # NEW: per-channel sharpen flag from preset
                    if self.p.get("sharpen_channels_separately", False):
                        args.append("--sharpen_channels_separately")
                elif mode == "denoise":
                    if not self.p.get("gpu", True): args.append("--disable_gpu")
                    if self.p.get("separate_channels", False): args.append("--separate_channels")
                    args += [
                        "--denoise_strength", f"{float(self.p.get('denoise_luma', 0.50)):.2f}",
                        "--color_denoise_strength", f"{float(self.p.get('denoise_color', 0.50)):.2f}",
                        "--denoise_mode", self.p.get("denoise_mode", "full"),
                    ]
                elif mode == "superres":
                    scale = int(self.p.get("scale", 2))
                    args = ["--input", staged_input, "--output_dir", out_dir,
                            "--scale", str(scale), "--model_dir", self.root]
                else:
                    raise RuntimeError(f"Unknown mode: {mode}")

                self.log.emit(f"Launching: {os.path.basename(exe)} {' '.join(args)}")
                self.progress.emit(0)

                # Run process and stream stdout
                self._proc = subprocess.Popen([exe] + args, cwd=self.root,
                                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                              text=True, universal_newlines=True)
                try:
                    for line in self._proc.stdout:
                        if self._stop: break
                        self._emit_progress_line(line)
                except Exception:
                    pass
                rc = self._proc.wait()
                self._proc = None
                if self._stop: raise RuntimeError("Cancelled")
                if rc != 0:
                    raise RuntimeError(f"{os.path.basename(exe)} exited with code {rc}")

                # Wait for produced file
                if mode == "superres":
                    scale = int(self.p.get("scale", 2))
                    pat = os.path.join(out_dir, f"{base}_upscaled{scale}.*")
                else:
                    pat = os.path.join(out_dir, f"{base}{suffix}.*")
                self.log.emit("Waiting for output file…")
                out_path = self._wait_for_file(pat, timeout=1800)
                if not out_path:
                    raise RuntimeError("Output file not found.")

                arr, _, _, _ = load_image(out_path)
                if arr is None:
                    raise RuntimeError("Failed to load output image.")

                result = np.asarray(arr).astype(np.float32, copy=False)
                self.progress.emit(100)

                # Stage as next input if more to do
                if (mode, suffix) != self.ops[-1]:
                    save_image(result, in_path, "tiff", "32-bit floating point",
                               getattr(self.doc, "original_header", None),
                               getattr(self.doc, "is_mono", False))
                    staged_input = in_path

                # Cleanup produced out file
                try:
                    if os.path.exists(out_path): os.remove(out_path)
                except Exception:
                    pass

            # Cleanup input
            try:
                if os.path.exists(in_path): os.remove(in_path)
            except Exception:
                pass

            if result is None:
                raise RuntimeError("No result produced.")
            self.finished_ok.emit(result)

        except Exception as e:
            self.failed.emit(str(e))

    def _wait_for_file(self, pattern: str, timeout: float = 1800.0, poll: float = 0.25):
        """
        Wait for a file matching glob `pattern`. Returns most recent match or "".
        """
        t0 = time.time()
        last = ""
        while time.time() - t0 < timeout:
            matches = glob.glob(pattern)
            if matches:
                try:
                    # pick newest file (mtime)
                    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                except Exception:
                    matches.sort()
                last = matches[0]
                # make sure it’s non-zero and stable-ish
                try:
                    if os.path.getsize(last) > 0:
                        return last
                except Exception:
                    return last
            non_blocking_sleep(poll)
        return ""


# ---------------- Public entry ----------------
def run_cosmicclarity_via_preset(main, preset: dict | None = None, *, doc=None):
    """Run CC headlessly by driving the same pipeline as the Execute button."""
    p = dict(preset or {})

    # ---- Record for Replay Last Action ----
    try:
        remember = getattr(main, "remember_last_headless_command", None)
        if remember is None:
            remember = getattr(main, "_remember_last_headless_command", None)
        if callable(remember):
            remember("cosmic_clarity", p, description="Cosmic Clarity")
        else:
            setattr(main, "_last_headless_command", {
                "command_id": "cosmic_clarity",
                "preset": dict(p),
            })
    except Exception:
        pass
    # --------------------------------------

    # Guard so users can’t open another CC panel while this runs
    setattr(main, "_cosmicclarity_headless_running", True)
    setattr(main, "_cosmicclarity_guard", True)
    s = QSettings()
    try:
        s.setValue("cc/headless_in_progress", True); s.sync()
    except Exception:
        pass

    try:
        # Prefer the explicit doc (from target_sw); otherwise fall back to active
        if doc is None:
            doc = getattr(main, "_active_doc", None)
            if callable(doc):
                doc = doc()

        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.warning(main, "Cosmic Clarity", "Load an image first.")
            return

        dlg = CosmicClarityDialogPro(main, doc, headless=True, bypass_guard=True)
        if getattr(dlg, "_headless", False) is not True:
            return

        try:
            dlg.apply_preset(p)
        except Exception:
            mode = str(p.get("mode","sharpen")).lower()
            dlg.cmb_mode.setCurrentIndex({"sharpen":0,"denoise":1,"both":2,"superres":3}.get(mode,0))
            dlg.cmb_gpu.setCurrentIndex(0 if p.get("gpu", True) else 1)
            dlg.cmb_target.setCurrentIndex(1 if p.get("create_new_view", False) else 0)

        dlg._run_main()

        loop = QEventLoop()
        dlg.finished.connect(loop.quit)
        loop.exec_() if hasattr(loop, "exec_") else loop.exec()

    finally:
        try:
            s.setValue("cc/headless_in_progress", False); s.sync()
        except Exception:
            pass
        for name in ("_cosmicclarity_headless_running", "_cosmicclarity_guard"):
            try:
                delattr(main, name)
            except Exception:
                setattr(main, name, False)



# ---------------- Optional: tiny preset editor for the shortcut button ----------------
class _CosmicClarityPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Cosmic Clarity — Preset")
        p = dict(initial or {})

        # Check if correction model is available
        try:
            from setiastro.saspro.resources import get_resources
            _r = get_resources()
            _cp = getattr(_r, "CC_C_PTH", None)
            self._correct_available = bool(_cp and os.path.exists(_cp))
        except Exception:
            self._correct_available = False

        f = QFormLayout(self)

        self.mode = QComboBox()
        self.mode.addItems(["sharpen", "denoise", "both", "superres"])
        self.mode.setCurrentText(str(p.get("mode", "sharpen")))
        self.mode.currentTextChanged.connect(self._mode_changed)
        f.addRow("Mode:", self.mode)

        self.gpu = QCheckBox("Use GPU")
        self.gpu.setChecked(bool(p.get("gpu", True)))
        f.addRow(self.gpu)

        self.ab_first = QCheckBox("Run Aberration Remover first")
        self.ab_first.setChecked(False)   # permanently disabled — superseded by native correction
        self.ab_first.setVisible(False)   # hidden
        f.addRow(self.ab_first)

        self.newview = QCheckBox("Create new view")
        self.newview.setChecked(bool(p.get("create_new_view", False)))
        f.addRow(self.newview)

        # --- Global controls (apply to sharpen AND denoise) ---
        self.compat = QCheckBox("GPU Compatibility Mode (slower, safer)")
        self.compat.setChecked(bool(p.get("compat_mode", p.get("denoise_compat_mode", False))))
        self.compat.setToolTip(
            "Safer inference settings. Recommended if sharpen/denoise fails,\n"
            "hangs, or produces GPU errors on older hardware."
        )
        f.addRow(self.compat)

        self.temp_stretch = QCheckBox("Temporary Stretch for AI (linear assist)")
        self.temp_stretch.setChecked(bool(p.get("temp_stretch", False)))
        f.addRow(self.temp_stretch)

        self.target_median = QDoubleSpinBox()
        self.target_median.setRange(0.01, 0.50)
        self.target_median.setSingleStep(0.01)
        try:
            self.target_median.setValue(max(0.01, min(0.50, float(p.get("target_median", 0.25)))))
        except Exception:
            self.target_median.setValue(0.25)
        self._target_median_label = QLabel("Target Median (0.01–0.50):")
        f.addRow(self._target_median_label, self.target_median)
        self.temp_stretch.toggled.connect(
            lambda on: (self.target_median.setEnabled(on), self._target_median_label.setEnabled(on)))
        self.target_median.setEnabled(self.temp_stretch.isChecked())
        self._target_median_label.setEnabled(self.temp_stretch.isChecked())

        # --- Stellar Correction Mode (only shown if model installed) ---
        self._corr_label = QLabel("Stellar Correction:")
        corr_row = QWidget()
        corr_lay = QHBoxLayout(corr_row)
        corr_lay.setContentsMargins(0, 0, 0, 0)
        from PyQt6.QtWidgets import QRadioButton
        self.rb_correct_only    = QRadioButton("Correct Only")
        self.rb_correct_sharpen = QRadioButton("Correct + Sharpen")
        self.rb_sharpen_only    = QRadioButton("Sharpen Only")
        self.rb_sharpen_only.setChecked(True)
        corr_lay.addWidget(self.rb_correct_only)
        corr_lay.addWidget(self.rb_correct_sharpen)
        corr_lay.addWidget(self.rb_sharpen_only)
        corr_lay.addStretch(1)
        f.addRow(self._corr_label, corr_row)
        self._corr_row_widget = corr_row

        # Restore correction mode from preset
        scm = p.get("stellar_correct_mode", "sharpen_only")
        self.rb_correct_only.setChecked(scm == "correct_only")
        self.rb_correct_sharpen.setChecked(scm == "correct_sharpen")
        self.rb_sharpen_only.setChecked(scm not in ("correct_only", "correct_sharpen"))

        # Connect to update sharpen sub-control visibility
        self.rb_correct_only.toggled.connect(self._mode_changed)
        self.rb_correct_sharpen.toggled.connect(self._mode_changed)
        self.rb_sharpen_only.toggled.connect(self._mode_changed)

        # --- Sharpen controls ---
        self.sh_mode = QComboBox()
        self.sh_mode.addItems(["Both", "Stellar Only", "Non-Stellar Only"])
        self.sh_mode.setCurrentText(p.get("sharpening_mode", "Both"))

        self.auto_psf = QCheckBox("Auto PSF")
        self.auto_psf.setChecked(bool(p.get("auto_psf", True)))

        self.psf = QDoubleSpinBox()
        self.psf.setRange(1.0, 8.0)
        self.psf.setSingleStep(0.1)
        self.psf.setValue(float(p.get("nonstellar_psf", 3.0)))

        self.st_amt = QDoubleSpinBox()
        self.st_amt.setRange(0.0, 1.0)
        self.st_amt.setSingleStep(0.05)
        self.st_amt.setValue(float(p.get("stellar_amount", 0.5)))

        self.nst_amt = QDoubleSpinBox()
        self.nst_amt.setRange(0.0, 1.0)
        self.nst_amt.setSingleStep(0.05)
        self.nst_amt.setValue(float(p.get("nonstellar_amount", 0.5)))

        self.sh_sep = QCheckBox("Sharpen RGB channels separately")
        self.sh_sep.setChecked(bool(p.get("sharpen_channels_separately", False)))

        self._sh_mode_label = QLabel("Sharpening Mode:")
        f.addRow(self._sh_mode_label, self.sh_mode)
        f.addRow(self.auto_psf)
        self._psf_label = QLabel("Non-stellar PSF:")
        f.addRow(self._psf_label, self.psf)
        self._st_label = QLabel("Stellar Amount (0-1):")
        f.addRow(self._st_label, self.st_amt)
        self._nst_label = QLabel("Non-stellar Amount (0-1):")
        f.addRow(self._nst_label, self.nst_amt)
        f.addRow(self.sh_sep)
        self.correct_conservative = QCheckBox("Conservative White Compression (reduces tiling on high-contrast images)")
        self.correct_conservative.setChecked(bool(p.get("correct_conservative", False)))
        self.correct_conservative.setToolTip(
            "Uses stronger white point compression (0.75 vs 0.95) during aberration correction.\n"
            "Reduces tiling artifacts on images with extreme bright/dark contrast within a single tile."
        )
        f.addRow(self.correct_conservative)

        # Correction model version (sharpen-only; mirrors main dialog combo)
        self.correct_ver = QComboBox()
        self.correct_ver.addItems(["V2 (latest)", "V1"])
        cv = str(p.get("correct_model_version", "V2 (latest)"))
        if self.correct_ver.findText(cv) >= 0:
            self.correct_ver.setCurrentText(cv)
        self._correct_ver_label = QLabel("Correction Model:")
        f.addRow(self._correct_ver_label, self.correct_ver)

        # Tiling (sharpen-only; mirrors main dialog chunk/overlap dropdowns)
        self.chunk = QComboBox()
        self.chunk.addItems(["128", "192", "256", "320", "384", "512", "640", "768", "1024"])
        self.chunk.setCurrentText(str(int(p.get("chunk_size", 256))))
        self._chunk_label = QLabel("Chunk Size:")
        f.addRow(self._chunk_label, self.chunk)

        self.overlap = QComboBox()
        self.overlap.addItems(["16", "32", "48", "64", "80", "96", "128", "192", "256", "320", "384", "512"])
        self.overlap.setCurrentText(str(int(p.get("overlap", 64))))
        self._overlap_label = QLabel("Overlap:")
        f.addRow(self._overlap_label, self.overlap)
        # Track sharpen sub-widgets for visibility toggling
        self._sharpen_sub_widgets = [
            self._sh_mode_label, self.sh_mode,
            self.auto_psf,
            self._psf_label, self.psf,
            self._st_label, self.st_amt,
            self._nst_label, self.nst_amt,
            self.sh_sep,
            self.correct_conservative,
            self._correct_ver_label, self.correct_ver,
            self._chunk_label, self.chunk,
            self._overlap_label, self.overlap,
        ]

        # --- Denoise controls ---
        self.dn_lum = QDoubleSpinBox()
        self.dn_lum.setRange(0.0, 1.0)
        self.dn_lum.setSingleStep(0.05)
        self.dn_lum.setValue(float(p.get("denoise_luma", 0.5)))

        self.dn_col = QDoubleSpinBox()
        self.dn_col.setRange(0.0, 1.0)
        self.dn_col.setSingleStep(0.05)
        self.dn_col.setValue(float(p.get("denoise_color", 0.5)))

        self.dn_mode = QComboBox()
        self.dn_mode.addItems(["full", "luminance"])
        self.dn_mode.setCurrentText(p.get("denoise_mode", "full"))

        self.dn_sep = QCheckBox("Separate RGB channels")
        self.dn_sep.setChecked(bool(p.get("separate_channels", False)))

        self.dn_model = QComboBox()
        self.dn_model.addItems(["Standard", "Walking Noise", "Lite (faster)"])
        if p.get("denoise_walking", False):
            self.dn_model.setCurrentText("Walking Noise")
        elif p.get("denoise_lite", False):
            self.dn_model.setCurrentText("Lite (faster)")
        else:
            self.dn_model.setCurrentText("Standard")

        self._dn_lum_label = QLabel("Denoise Luma:")
        self._dn_col_label = QLabel("Denoise Color:")
        self._dn_mode_label = QLabel("Denoise Mode:")
        self._dn_model_label = QLabel("Denoise Model:")
        f.addRow(self._dn_lum_label, self.dn_lum)
        f.addRow(self._dn_col_label, self.dn_col)
        f.addRow(self._dn_mode_label, self.dn_mode)
        f.addRow(self.dn_sep)
        f.addRow(self._dn_model_label, self.dn_model)

        self._denoise_sub_widgets = [
            self._dn_lum_label, self.dn_lum,
            self._dn_col_label, self.dn_col,
            self._dn_mode_label, self.dn_mode,
            self.dn_sep,
            self._dn_model_label, self.dn_model,
        ]

        # --- Super-res ---
        self.scale = QComboBox()
        self.scale.addItems(["2", "3", "4"])
        self.scale.setCurrentText(str(int(p.get("scale", 2))))
        self._scale_label = QLabel("Super-Res Scale:")
        f.addRow(self._scale_label, self.scale)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        f.addRow(btns)

        self._mode_changed()

    def _mode_changed(self, *_):
        m = self.mode.currentText()
        show_sh = m in ("sharpen", "both")
        show_dn = m in ("denoise", "both")
        show_sr = m == "superres"

        corr_visible = show_sh and self._correct_available
        self._corr_label.setVisible(corr_visible)
        self._corr_row_widget.setVisible(corr_visible)
        self.correct_conservative.setVisible(corr_visible)  # ADD THIS

        sharpen_active = show_sh and not (
            self._correct_available and self.rb_correct_only.isChecked()
        )
        for w in self._sharpen_sub_widgets:
            w.setVisible(sharpen_active)

        for w in self._denoise_sub_widgets:
            w.setVisible(show_dn)

        # temp-stretch / target-median / compat apply to sharpen+denoise, not superres
        globals_active = not show_sr
        for w in (self.compat, self.temp_stretch,
                  self._target_median_label, self.target_median):
            w.setVisible(globals_active)

        self._scale_label.setVisible(show_sr)
        self.scale.setVisible(show_sr)

    def result_dict(self) -> dict:
        m = self.mode.currentText()
        compat = bool(self.compat.isChecked())

        out = {
            "mode": m,
            "gpu": bool(self.gpu.isChecked()),
            "create_new_view": bool(self.newview.isChecked()),
            "aberration_first": False,

            # global compatibility controls (mirror build_preset_from_ui's derivation)
            "compat_mode": compat,
            "execution_mode": "compatibility" if compat else "auto",
            "batch_size_override": 1 if compat else 0,
        }
        if m in ("sharpen", "both"):
            scm = (
                "correct_only"    if self.rb_correct_only.isChecked()    else
                "correct_sharpen" if self.rb_correct_sharpen.isChecked() else
                "sharpen_only"
            ) if self._correct_available else "sharpen_only"

            out.update({
                "stellar_correct_mode": scm,
                "sharpening_mode": self.sh_mode.currentText(),
                "auto_psf": bool(self.auto_psf.isChecked()),
                "nonstellar_psf": float(self.psf.value()),
                "stellar_amount": float(self.st_amt.value()),
                "nonstellar_amount": float(self.nst_amt.value()),
                "sharpen_channels_separately": bool(self.sh_sep.isChecked()),
                "correct_conservative": bool(self.correct_conservative.isChecked()),
                "correct_model_version": self.correct_ver.currentText(),
                "chunk_size": int(self.chunk.currentText()),
                "overlap": int(self.overlap.currentText()),
                "temp_stretch": bool(self.temp_stretch.isChecked()),
                "target_median": float(self.target_median.value()),

                # sharpen compatibility controls
                "sharpen_execution_mode": "compatibility" if compat else "auto",
                "sharpen_batch_size_override": 1 if compat else 0,
            })
        if m in ("denoise", "both"):
            dn_model = self.dn_model.currentText()
            out.update({
                "denoise_luma": float(self.dn_lum.value()),
                "denoise_color": float(self.dn_col.value()),
                "denoise_mode": self.dn_mode.currentText(),
                "separate_channels": bool(self.dn_sep.isChecked()),
                "denoise_lite":    (dn_model == "Lite (faster)"),
                "denoise_walking": (dn_model == "Walking Noise"),

                # denoise compatibility controls
                "denoise_compat_mode": compat,
                "denoise_execution_mode": "compatibility" if compat else "auto",
                "denoise_batch_size_override": 1 if compat else 0,
                "temp_stretch": bool(self.temp_stretch.isChecked()),
                "target_median": float(self.target_median.value()),
            })
        if m == "superres":
            out["scale"] = int(self.scale.currentText())
        return out