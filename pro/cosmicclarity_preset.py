# pro/cosmicclarity_preset.py
from __future__ import annotations
import os, sys, time, glob, shutil, subprocess
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QSettings, QLockFile
from PyQt6.QtWidgets import QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox, QComboBox, QCheckBox, QMessageBox, QApplication, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QSettings, QLockFile, QEventLoop


# reuse your legacy IO + helpers
from legacy.image_manager import load_image, save_image

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
            except Exception: pass
        elif s.startswith("PROGRESS:"):
            try: self.progress.emit(int(s.split(":",1)[1].strip().replace("%","")))
            except Exception: pass

    def run(self):
        try:
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

            for (mode, suffix) in self.ops:
                if self._stop: raise RuntimeError("Cancelled")
                self.step_changed.emit(mode)

                exe = os.path.join(self.root, _platform_exe_names(mode))
                if not os.path.exists(exe):
                    raise RuntimeError(f"Cosmic Clarity executable not found:\n{exe}")

                # Build args
                args = []
                if mode == "sharpen":
                    if not self.p.get("gpu", True): args.append("--disable_gpu")
                    if self.p.get("auto_psf", True): args.append("--auto_detect_psf")
                    args += [
                        "--sharpening_mode", self.p.get("sharpening_mode", "Both"),
                        "--stellar_amount", f"{float(self.p.get('stellar_amount', 0.50)):.2f}",
                        "--nonstellar_strength", f"{float(self.p.get('nonstellar_psf', 3.0)):.1f}",
                        "--nonstellar_amount", f"{float(self.p.get('nonstellar_amount', 0.50)):.2f}",
                    ]
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


# ---------------- Public entry ----------------
def run_cosmicclarity_via_preset(main, preset: dict | None = None, *, doc=None):
    """Run CC headlessly by driving the same pipeline as the Execute button."""
    p = dict(preset or {})

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

        # Build the dialog in headless mode, bypassing the guard (we're the owner)
        dlg = CosmicClarityDialogPro(main, doc, headless=True, bypass_guard=True)
        if getattr(dlg, "_headless", False) is not True:
            return

        # Apply preset to the dialog widgets so behavior == user pressed Execute
        try:
            dlg.apply_preset(p)
        except Exception:
            mode = str(p.get("mode","sharpen")).lower()
            dlg.cmb_mode.setCurrentIndex({"sharpen":0,"denoise":1,"both":2,"superres":3}.get(mode,0))
            dlg.cmb_gpu.setCurrentIndex(0 if p.get("gpu", True) else 1)
            dlg.cmb_target.setCurrentIndex(1 if p.get("create_new_view", False) else 0)

        # Kick the exact same execution path
        dlg._run_main()

        # Block here until the dialog finishes.
        loop = QEventLoop()
        dlg.finished.connect(loop.quit)
        loop.exec_() if hasattr(loop, "exec_") else loop.exec()

    finally:
        # Clear guards even on exceptions
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
    """
    Minimal preset editor for the shortcut:
      Mode, GPU, create_new_view plus a few key params per mode.
    """
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Cosmic Clarity — Preset")
        p = dict(initial or {})
        f = QFormLayout(self)

        self.mode = QComboBox(); self.mode.addItems(["sharpen", "denoise", "both", "superres"])
        self.mode.setCurrentText(str(p.get("mode", "sharpen")))
        f.addRow("Mode:", self.mode)

        self.gpu = QCheckBox("Use GPU"); self.gpu.setChecked(bool(p.get("gpu", True)))
        f.addRow(self.gpu)

        self.newview = QCheckBox("Create new view"); self.newview.setChecked(bool(p.get("create_new_view", False)))
        f.addRow(self.newview)

        # Sharpen
        self.sh_mode = QComboBox(); self.sh_mode.addItems(["Both", "Stellar Only", "Non-Stellar Only"])
        self.sh_mode.setCurrentText(p.get("sharpening_mode", "Both"))
        self.auto_psf = QCheckBox("Auto PSF"); self.auto_psf.setChecked(bool(p.get("auto_psf", True)))
        self.psf = QDoubleSpinBox(); self.psf.setRange(1.0, 8.0); self.psf.setSingleStep(0.1); self.psf.setValue(float(p.get("nonstellar_psf", 3.0)))
        self.st_amt = QDoubleSpinBox(); self.st_amt.setRange(0.0, 1.0); self.st_amt.setSingleStep(0.05); self.st_amt.setValue(float(p.get("stellar_amount", 0.50)))
        self.nst_amt= QDoubleSpinBox(); self.nst_amt.setRange(0.0, 1.0); self.nst_amt.setSingleStep(0.05); self.nst_amt.setValue(float(p.get("nonstellar_amount", 0.50)))
        f.addRow("Sharpening Mode:", self.sh_mode)
        f.addRow(self.auto_psf)
        f.addRow("Non-stellar PSF:", self.psf)
        f.addRow("Stellar Amount:", self.st_amt)
        f.addRow("Non-stellar Amount:", self.nst_amt)

        # Denoise
        self.dn_lum = QDoubleSpinBox(); self.dn_lum.setRange(0.0, 1.0); self.dn_lum.setSingleStep(0.05); self.dn_lum.setValue(float(p.get("denoise_luma", 0.50)))
        self.dn_col = QDoubleSpinBox(); self.dn_col.setRange(0.0, 1.0); self.dn_col.setSingleStep(0.05); self.dn_col.setValue(float(p.get("denoise_color", 0.50)))
        self.dn_mode= QComboBox(); self.dn_mode.addItems(["full","luminance"]); self.dn_mode.setCurrentText(p.get("denoise_mode", "full"))
        self.dn_sep = QCheckBox("Separate RGB channels"); self.dn_sep.setChecked(bool(p.get("separate_channels", False)))
        f.addRow("Denoise Luma:", self.dn_lum)
        f.addRow("Denoise Color:", self.dn_col)
        f.addRow("Denoise Mode:", self.dn_mode)
        f.addRow(self.dn_sep)

        # Super-res
        self.scale = QComboBox(); self.scale.addItems(["2","3","4"]); self.scale.setCurrentText(str(int(p.get("scale", 2))))
        f.addRow("Super-Res Scale:", self.scale)

        # OK/Cancel
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        f.addRow(btns)


    def result_dict(self) -> dict:
        m = self.mode.currentText()
        out = {
            "mode": m,
            "gpu": bool(self.gpu.isChecked()),
            "create_new_view": bool(self.newview.isChecked()),
        }
        if m in ("sharpen","both"):
            out.update({
                "sharpening_mode": self.sh_mode.currentText(),
                "auto_psf": bool(self.auto_psf.isChecked()),
                "nonstellar_psf": float(self.psf.value()),
                "stellar_amount": float(self.st_amt.value()),
                "nonstellar_amount": float(self.nst_amt.value()),
            })
        if m in ("denoise","both"):
            out.update({
                "denoise_luma": float(self.dn_lum.value()),
                "denoise_color": float(self.dn_col.value()),
                "denoise_mode": self.dn_mode.currentText(),
                "separate_channels": bool(self.dn_sep.isChecked()),
            })
        if m == "superres":
            out["scale"] = int(self.scale.currentText())
        return out
