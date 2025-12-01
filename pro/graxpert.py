# pro/graxpert.py
from __future__ import annotations
import os, platform, shutil, tempfile, stat, glob, subprocess

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTextEdit, QPushButton, QFileDialog,
    QMessageBox, QInputDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox,
    QRadioButton, QLabel, QComboBox, QCheckBox, QWidget
)
from pro.config import Config

# Prefer the exact loader you used in SASv2
try:
    # adjust this import path if your loader lives elsewhere
    from legacy.image_manager import load_image as _legacy_load_image
except Exception:
    _legacy_load_image = None


class GraXpertOperationDialog(QDialog):
    """Choose operation + parameter (smoothing or strength) + (optional) denoise model."""
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("GraXpert")
        root = QVBoxLayout(self)

        # radios
        self.rb_bg  = QRadioButton("Remove gradient")
        self.rb_dn  = QRadioButton("Denoise")
        self.rb_bg.setChecked(True)

        # param widgets
        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.0, 1.0)
        self.spin.setDecimals(2)
        self.spin.setSingleStep(0.01)
        self.spin.setValue(0.10)   # default for smoothing

        # dynamic label
        self.param_label = QLabel("Smoothing (0–1):")

        # denoise model (optional)
        self.model_label = QLabel("Denoise model:")
        self.model_combo = QComboBox()
        # Index 0 = auto/latest (empty payload → omit flag)
        self.model_combo.addItem("Latest (auto)", "")     # omit -ai_version
        for v in ["3.0.2", "3.0.1", "3.0.0", "2.0.0", "1.1.0", "1.0.0"]:
            self.model_combo.addItem(v, v)

        # GPU toggle (persists via QSettings if available)
        self.cb_gpu = QCheckBox("Use GPU acceleration")
        use_gpu_default = True
        try:
            settings = getattr(parent, "settings", None)
            if settings is not None:
                use_gpu_default = settings.value("graxpert/use_gpu", True, type=bool)
        except Exception:
            pass
        self.cb_gpu.setChecked(bool(use_gpu_default))


        # layout
        form = QFormLayout()
        form.addRow(self.rb_bg)
        form.addRow(self.rb_dn)
        form.addRow(self.param_label, self.spin)
        form.addRow(self.model_label, self.model_combo)
        form.addRow(self.cb_gpu)
        root.addLayout(form)

        # switch label/defaults and enable/disable model picker
        def _to_bg():
            self.param_label.setText("Smoothing (0–1):")
            # If param was the denoise default, flip back to smoothing default
            self.spin.setValue(0.10 if abs(self.spin.value() - 0.50) < 1e-6 else self.spin.value())
            self.model_label.setEnabled(False)
            self.model_combo.setEnabled(False)

        def _to_dn():
            self.param_label.setText("Strength (0–1):")
            # If param was the smoothing default, flip to denoise default
            self.spin.setValue(0.50 if abs(self.spin.value() - 0.10) < 1e-6 else self.spin.value())
            self.model_label.setEnabled(True)
            self.model_combo.setEnabled(True)

        self.rb_bg.toggled.connect(lambda checked: _to_bg() if checked else None)
        self.rb_dn.toggled.connect(lambda checked: _to_dn() if checked else None)

        # initialize state
        _to_bg()

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def result(self):
        op = "background" if self.rb_bg.isChecked() else "denoise"
        val = float(self.spin.value())
        ai_version = self.model_combo.currentData() if not self.rb_bg.isChecked() else ""
        use_gpu = self.cb_gpu.isChecked()
        return op, val, (ai_version or None), use_gpu

def _build_graxpert_cmd(
    exe: str,
    operation: str,
    input_path: str,
    *,
    smoothing: float | None = None,
    strength: float | None = None,
    ai_version: str | None = None,
    gpu: bool = True,
    batch_size: int | None = None
) -> list[str]:
    op = "denoising" if operation == "denoise" else "background-extraction"
    cmd = [exe, "-cmd", op, input_path, "-cli", "-gpu", "true" if gpu else "false"]
    if op == "denoising":
        if strength is not None:
            cmd += ["-strength", f"{strength:.2f}"]
        if batch_size is not None:
            cmd += ["-batch_size", str(int(batch_size))]
        # Only include if user chose a specific model
        if ai_version:
            cmd += ["-ai_version", ai_version]
    else:
        if smoothing is not None:
            cmd += ["-smoothing", f"{smoothing:.2f}"]
    return cmd

# ---------- Public entry point (call this from your main window) ----------
def remove_gradient_with_graxpert(main_window, target_doc=None):
    """
    Exactly mirror SASv2 flow:
      - write input_image.tif
      - run GraXpert
      - read input_image_GraXpert.{fits|tif|tiff|png} using legacy loader
      - apply to target document
    """
    if getattr(main_window, "_graxpert_headless_running", False):
        return
    if getattr(main_window, "_graxpert_guard", False):   # cool-down guard
        return

    # 1) pick the document: explicit > fallback
    doc = target_doc

    if doc is None:
        # Backwards compatibility: fall back to _active_doc
        doc = getattr(main_window, "_active_doc", None)
        if callable(doc):
            doc = doc()

    if doc is None and hasattr(main_window, "mdi"):
        # Extra fallback: resolve from active subwindow if possible
        try:
            sw = main_window.mdi.activeSubWindow()
            if sw is not None:
                view = sw.widget()
                doc = getattr(view, "document", None)
        except Exception:
            pass

    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.warning(
            main_window,
            "No Image",
            "Please load an image before removing the gradient."
        )
        return

    # 2) smoothing/denoise prompt
    op_dlg = GraXpertOperationDialog(main_window)
    if op_dlg.exec() != QDialog.DialogCode.Accepted:
        return
    operation, param, ai_version, use_gpu = op_dlg.result()

    # 3) resolve GraXpert executable
    exe = _resolve_graxpert_exec(main_window)
    if not exe:
        return

    # Persist the checkbox choice for next time
    try:
        if hasattr(main_window, "settings"):
            main_window.settings.setValue("graxpert/use_gpu", bool(use_gpu))
    except Exception:
        pass

    # 🔁 NEW: record this as a replayable headless-style command
    try:
        remember = getattr(main_window, "remember_last_headless_command", None)
        if remember is None:
            remember = getattr(main_window, "_remember_last_headless_command", None)

        if callable(remember):
            preset = {
                "op": operation,           # "background" or "denoise"
                "gpu": bool(use_gpu),
            }
            if operation == "background":
                preset["smoothing"] = float(param)
                desc = "GraXpert Gradient Removal"
            else:
                preset["strength"] = float(param)
                if ai_version:
                    preset["ai_version"] = ai_version
                desc = "GraXpert Denoise"

            remember("graxpert", preset, description=desc)

            # Optional log entry, if you want:
            if hasattr(main_window, "_log"):
                try:
                    main_window._log(
                        f"[Replay] GraXpert preset stored from dialog: "
                        f"op={operation}, keys={list(preset.keys())}"
                    )
                except Exception:
                    pass
    except Exception:
        # Don't let replay bookkeeping break GraXpert itself
        pass

    # 4) write input to a temp working dir but KEEP THE SAME BASENAMES as v2
    workdir = tempfile.mkdtemp(prefix="saspro_graxpert_")
    input_basename = "input_image"
    input_path = os.path.join(workdir, f"{input_basename}.tif")
    try:
        _write_tiff_float32(doc.image, input_path)
    except Exception as e:
        QMessageBox.critical(main_window, "GraXpert", f"Failed to write temporary input:\n{e}")
        shutil.rmtree(workdir, ignore_errors=True)
        return

    # 5) build the exact v2 command (now with optional ai_version for denoise)
    command = _build_graxpert_cmd(
        exe,
        operation,
        input_path,
        smoothing=param if operation == "background" else None,
        strength=param if operation == "denoise" else None,
        ai_version=ai_version if operation == "denoise" else None,
        gpu=bool(use_gpu),
        batch_size=(4 if use_gpu else 1)
    )

    # Label + metadata for history/undo
    op_label = "GraXpert Denoise" if operation == "denoise" else "GraXpert Gradient Removal"
    meta_extras = {
        "graxpert_operation": operation,  # "denoise" | "background"
        "graxpert_param": float(param),
        "graxpert_ai_version": (ai_version or "latest") if operation == "denoise" else None,
        "graxpert_gpu": bool(use_gpu),
    }

    # 6) run and wait with a small log dialog
    output_basename = f"{input_basename}_GraXpert"
    _run_graxpert_command(
        main_window,
        command,
        output_basename,
        workdir,
        target_doc=doc,
        op_label=op_label,
        meta_extras=meta_extras,
    )


# ---------- helpers ----------
def _resolve_graxpert_exec(main_window) -> str | None:
    # prefer QSettings if available (all OS)
    path = None
    if hasattr(main_window, "settings"):
        try:
            path = main_window.settings.value("paths/graxpert", type=str)
        except Exception:
            path = None
    if path and os.path.exists(path):
        _ensure_exec_bit(path)
        return path

    sysname = platform.system()
    default = Config.get_graxpert_default_path()
    
    if sysname == "Windows":
        # rely on PATH (like v2) or default
        return default if default else "GraXpert.exe"
        
    if sysname == "Darwin":
        if default and os.path.exists(default):
            _ensure_exec_bit(default)
            if hasattr(main_window, "settings"):
                main_window.settings.setValue("paths/graxpert", default)
            return default
        return _pick_graxpert_path_and_store(main_window)
        
    if sysname == "Linux":
        # in v2 you asked user and saved; do the same
        return _pick_graxpert_path_and_store(main_window)

    QMessageBox.critical(main_window, "GraXpert", f"Unsupported operating system: {sysname}")
    return None

def _pick_graxpert_path_and_store(main_window) -> str | None:
    path, _ = QFileDialog.getOpenFileName(main_window, "Select GraXpert Executable")
    if not path:
        QMessageBox.warning(main_window, "Cancelled", "GraXpert path selection was cancelled.")
        return None
    try:
        _ensure_exec_bit(path)
    except Exception as e:
        QMessageBox.critical(main_window, "GraXpert", f"Failed to set execute permissions:\n{e}")
        return None
    if hasattr(main_window, "settings"):
        main_window.settings.setValue("paths/graxpert", path)
    return path


def _ensure_exec_bit(path: str) -> None:
    if platform.system() == "Windows":
        return
    try:
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass


def _write_tiff_float32(image, path: str, *, clip01: bool = True):
    """
    Always write a 32-bit floating-point TIFF for GraXpert.
    - Mono stays 2D; RGB stays HxWx3.
    - Values are clipped to [0,1] by default to avoid weird HDR ranges.
    """
    import numpy as np

    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]

    # Convert to float32 in [0,1]
    if np.issubdtype(arr.dtype, np.floating):
        a32 = arr.astype(np.float32, copy=False)
        if clip01:
            a32 = np.clip(a32, 0.0, 1.0)
    elif np.issubdtype(arr.dtype, np.integer):
        # Scale integers to [0,1] float32
        maxv = np.float32(np.iinfo(arr.dtype).max)
        a32 = (arr.astype(np.float32) / maxv)
    else:
        a32 = arr.astype(np.float32)

    if clip01:
        a32 = np.clip(a32, 0.0, 1.0)

    # Prefer tifffile to guarantee float32 TIFFs
    try:
        import tifffile as tiff
        # Write a plain, contiguous, uncompressed float32 TIFF
        # (GraXpert doesn't need ImageJ tags; photometric=minisblack is fine)
        tiff.imwrite(
            path,
            a32,
            dtype=np.float32,
            photometric='minisblack' if a32.ndim == 2 else None,
            planarconfig='contig',
            compression=None,
            imagej=False,
        )
        return
    except Exception as e1:
        pass

    # Fallback: imageio (uses tifffile under the hood in many installs)
    try:
        import imageio.v3 as iio
        iio.imwrite(path, a32.astype(np.float32))
        return
    except Exception as e2:
        raise RuntimeError(
            "Could not write 32-bit TIFF for GraXpert. "
            "Please install 'tifffile' or 'imageio'.\n"
            f"tifffile error: {e1}\nimageio error: {e2}"
        )



# ---------- runner + dialog ----------
class _GraXpertThread(QThread):
    stdout_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, command: list[str], cwd: str | None = None, parent=None):
        super().__init__(parent)
        self.command = command
        self.cwd = cwd

    def run(self):
        env = os.environ.copy()
        for k in ("PYTHONHOME", "PYTHONPATH", "DYLD_LIBRARY_PATH",
                  "DYLD_FALLBACK_LIBRARY_PATH", "PYTHONEXECUTABLE"):
            env.pop(k, None)
        try:
            p = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # merge; avoids ResourceWarning + deadlocks
                text=True,
                universal_newlines=True,
                env=env,
                start_new_session=True
            )
            for line in iter(p.stdout.readline, ""):
                if not line:
                    break
                self.stdout_signal.emit(line.rstrip())
            try:
                p.stdout.close()
            except Exception:
                pass
            rc = p.wait()
        except Exception as e:
            self.stdout_signal.emit(str(e))
            rc = -1
        self.finished_signal.emit(rc)


def _run_graxpert_command(parent, command: list[str], output_basename: str,
                          working_dir: str, target_doc,
                          op_label: str | None = None,
                          meta_extras: dict | None = None):
    dlg = QDialog(parent)
    dlg.setWindowTitle("GraXpert Progress")
    dlg.setMinimumSize(600, 420)
    lay = QVBoxLayout(dlg)
    log = QTextEdit(readOnly=True)
    lay.addWidget(log)
    btn_cancel = QPushButton("Cancel")
    lay.addWidget(btn_cancel)

    thr = _GraXpertThread(command, cwd=working_dir)
    thr.stdout_signal.connect(lambda s: log.append(s))
    thr.finished_signal.connect(
        lambda code: _on_graxpert_finished(
            parent,
            code,
            output_basename,
            working_dir,
            target_doc,
            dlg,
            op_label,
            meta_extras,
        )
    )
    btn_cancel.clicked.connect(thr.terminate)

    thr.start()
    dlg.exec()



# ---------- finish: import EXACT base like v2, via legacy loader ----------
def _persist_output_file(src_path: str) -> str | None:
    """Optional: move/copy GraXpert output to an app cache we control."""
    try:
        from PyQt6.QtCore import QStandardPaths
        cache_root = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)
    except Exception:
        cache_root = None
    try:
        base = os.path.join(cache_root or os.path.expanduser("~/.saspro_cache"), "graxpert")
        os.makedirs(base, exist_ok=True)
        dst = os.path.join(base, os.path.basename(src_path))
        # prefer move (cheaper); fall back to copy if cross-device issues
        try:
            shutil.move(src_path, dst)
        except Exception:
            shutil.copy2(src_path, dst)
        return dst
    except Exception:
        return None


def _on_graxpert_finished(parent,
                          return_code: int,
                          output_basename: str,
                          working_dir: str,
                          target_doc,
                          dlg,
                          op_label: str | None = None,
                          meta_extras: dict | None = None):
    try:
        dlg.close()
    except Exception:
        pass

    if return_code != 0:
        QMessageBox.critical(parent, "GraXpert", "GraXpert process failed.")
        shutil.rmtree(working_dir, ignore_errors=True)
        return

    # 1) find output file in the temp working dir
    output_file = _pick_exact_output(working_dir, output_basename)
    if not output_file:
        QMessageBox.critical(parent, "GraXpert", "GraXpert output file not found.")
        shutil.rmtree(working_dir, ignore_errors=True)
        return

    # 2) read pixels (header is optional)
    arr, header = None, None
    if _legacy_load_image is not None:
        try:
            a, h, _, _ = _legacy_load_image(output_file)
            arr, header = a, h
        except Exception:
            arr = None

    if arr is None:
        arr = _fallback_read_float01(output_file)

    if arr is None or arr.size == 0:
        QMessageBox.critical(parent, "GraXpert", "Could not read GraXpert output.")
        shutil.rmtree(working_dir, ignore_errors=True)
        return

    # Decide how it appears in history/undo
    step_label = op_label or "GraXpert Gradient Removal"

    # 3) base metadata
    meta = {
        "step_name": step_label,
        "bit_depth": "32-bit floating point",
        "is_mono": (arr.ndim == 2) or (arr.ndim == 3 and arr.shape[2] == 1),
        "description": step_label,
    }
    if header is not None:
        meta["original_header"] = header
    if meta_extras:
        meta.update(meta_extras)

    # 4) apply to the target doc
    try:
        target_doc.apply_edit(
            arr.astype(np.float32, copy=False),
            metadata=meta,
            step_name=step_label,
        )
    except Exception as e:
        QMessageBox.critical(parent, "GraXpert", f"Failed to apply result:\n{e}")
    finally:
        shutil.rmtree(working_dir, ignore_errors=True)



def _pick_exact_output(folder: str, base: str) -> str | None:
    # exact filenames only, like v2 did
    exts = ("fits", "tif", "tiff", "png")
    for ext in exts:
        p = os.path.join(folder, f"{base}.{ext}")
        if os.path.exists(p):
            return p
        # also try case-variants just in case
        for q in glob.glob(os.path.join(folder, f"{base}.*")):
            if q.lower().endswith("." + ext):
                return q
    return None


def _fallback_read_float01(path: str) -> np.ndarray | None:
    """Basic loader: return float32 in [0,1], mono or RGB, without being too clever."""
    try:
        import imageio.v3 as iio
        arr = iio.imread(path)
    except Exception:
        try:
            import tifffile as tiff
            arr = tiff.imread(path)
        except Exception:
            try:
                from astropy.io import fits
                with fits.open(path, memmap=False) as hdul:
                    arr = hdul[0].data
            except Exception:
                try:
                    import cv2
                    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if arr is not None and arr.ndim == 3:
                        arr = arr[..., ::-1]  # BGR->RGB
                except Exception:
                    arr = None
    if arr is None:
        return None

    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.dtype.kind in "ui":
        scale = 65535.0 if arr.dtype.itemsize >= 2 else 255.0
        arr = arr.astype(np.float32) / scale
    else:
        arr = arr.astype(np.float32, copy=False)
        mx = float(arr.max()) if arr.size else 1.0
        if mx > 5.0:
            arr = arr / mx
    return arr