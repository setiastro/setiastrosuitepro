# pro/graxpert.py
from __future__ import annotations
import os, platform, shutil, tempfile, stat, glob, subprocess
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTextEdit, QPushButton, QFileDialog,
    QMessageBox, QInputDialog
)

# Prefer the exact loader you used in SASv2
try:
    # adjust this import path if your loader lives elsewhere
    from legacy.image_manager import load_image as _legacy_load_image
except Exception:
    _legacy_load_image = None


# ---------- Public entry point (call this from your main window) ----------
def remove_gradient_with_graxpert(main_window):
    """
    Exactly mirror SASv2 flow:
      - write input_image.tif
      - run GraXpert
      - read input_image_GraXpert.{fits|tif|tiff|png} using legacy loader
      - apply to active document
    """
    if getattr(main_window, "_graxpert_headless_running", False):
        return
    if getattr(main_window, "_graxpert_guard", False):   # <-- new: cool-down guard
        return
    
    # ⛑️ don’t open the smoothing dialog if a headless preset is running
    if getattr(main_window, "_graxpert_headless_running", False):
        return    
    # 1) active doc & image
    doc = getattr(main_window, "_active_doc", None)
    if callable(doc):
        doc = doc()
    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.warning(main_window, "No Image", "Please load an image before removing the gradient.")
        return

    # 2) smoothing prompt (same defaults as v2)
    smoothing, ok = QInputDialog.getDouble(
        main_window,
        "GraXpert Smoothing Amount",
        "Enter smoothing amount (0.0 to 1.0):",
        decimals=2, min=0.0, max=1.0, value=0.10
    )
    if not ok:
        return

    # 3) resolve GraXpert executable
    exe = _resolve_graxpert_exec(main_window)
    if not exe:
        return

    # 4) write input to a temp working dir but KEEP THE SAME BASENAMES as v2
    workdir = tempfile.mkdtemp(prefix="saspro_graxpert_")
    input_basename = "input_image"
    input_path = os.path.join(workdir, f"{input_basename}.tif")
    try:
        _write_tiff_like_v2(doc.image, input_path)
    except Exception as e:
        QMessageBox.critical(main_window, "GraXpert", f"Failed to write temporary input:\n{e}")
        shutil.rmtree(workdir, ignore_errors=True)
        return

    # 5) build the exact v2 command
    command = [
        exe,
        "-cmd", "background-extraction",
        input_path,
        "-cli",
        "-smoothing", str(smoothing),
        "-gpu", "true"
    ]

    # 6) run and wait with a small log dialog
    output_basename = f"{input_basename}_GraXpert"
    _run_graxpert_command(main_window, command, output_basename, workdir, target_doc=doc)


# ---------- helpers ----------
def _resolve_graxpert_exec(main_window) -> str | None:
    # prefer QSettings if available (all OS)
    path = None
    if hasattr(main_window, "settings"):
        try:
            path = main_window.settings.value("graxpert/path", type=str)
        except Exception:
            path = None
    if path and os.path.exists(path):
        _ensure_exec_bit(path)
        return path

    sysname = platform.system()
    if sysname == "Windows":
        # rely on PATH (like v2)
        return "GraXpert.exe"
    if sysname == "Darwin":
        default = "/Applications/GraXpert.app/Contents/MacOS/GraXpert"
        if os.path.exists(default):
            _ensure_exec_bit(default)
            if hasattr(main_window, "settings"):
                main_window.settings.setValue("graxpert/path", default)
            return default
        return _pick_graxpert_path_and_store(main_window)
    if sysname == "Linux":
        # in v2 you asked user and saved; do the same
        return _pick_graxpert_path_and_store(main_window)

    QMessageBox.critical(main_window, "GraXpert", f"Unsupported operating system: {sysname}")
    return None


def _pick_graxpert_path_and_store(parent) -> str | None:
    path, _ = QFileDialog.getOpenFileName(parent, "Select GraXpert Executable", "", "Executable Files (*)")
    if not path:
        QMessageBox.warning(parent, "Cancelled", "GraXpert path selection was cancelled.")
        return None
    try:
        _ensure_exec_bit(path)
    except Exception as e:
        QMessageBox.critical(parent, "GraXpert", f"Failed to set execute permissions:\n{e}")
        return None
    if hasattr(parent, "settings"):
        parent.settings.setValue("graxpert/path", path)
    return path


def _ensure_exec_bit(path: str) -> None:
    if platform.system() == "Windows":
        return
    try:
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass


def _write_tiff_like_v2(image, path: str):
    """Preserve v2’s behavior: save to 16-bit if <=1.0 range, else float32."""
    arr = np.asarray(image)
    # channel-last & non-empty
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if np.issubdtype(arr.dtype, np.floating):
        mx = float(arr.max()) if arr.size else 1.0
        if mx <= 1.0:
            arr = (np.clip(arr, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
        else:
            arr = arr.astype(np.float32, copy=False)
    elif arr.dtype == np.uint8:
        arr = (arr.astype(np.float32) / 255.0 * 65535.0 + 0.5).astype(np.uint16)
    elif arr.dtype != np.uint16:
        arr = arr.astype(np.uint16, copy=False)

    # try tifffile then imageio, like before
    try:
        import tifffile as tiff
        tiff.imwrite(path, arr)
        return
    except Exception:
        pass
    try:
        import imageio.v3 as iio
        iio.imwrite(path, arr)
        return
    except Exception:
        pass
    # last resort OpenCV (8-bit fallback)
    try:
        import cv2
        w = arr
        if arr.dtype != np.uint8:
            # OpenCV encodes 8-bit TIFF well enough for GraXpert intake
            denom = 65535.0 if arr.dtype == np.uint16 else float(arr.max() or 1.0)
            w = (arr.astype(np.float32) / denom * 255.0).astype(np.uint8)
        if w.ndim == 3 and w.shape[2] == 3:
            w = w[..., ::-1]  # RGB->BGR
        cv2.imwrite(path, w)
    except Exception as e:
        raise RuntimeError(f"Could not save TIFF: {e}")


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


def _run_graxpert_command(parent, command: list[str], output_basename: str, working_dir: str, target_doc):
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
    thr.finished_signal.connect(lambda code: _on_graxpert_finished(parent, code, output_basename, working_dir, target_doc, dlg))
    btn_cancel.clicked.connect(thr.terminate)

    thr.start()
    dlg.exec()


# ---------- finish: import EXACT base like v2, via legacy loader ----------
def _on_graxpert_finished(parent, return_code: int, output_basename: str, working_dir: str, target_doc, dlg):
    try:
        dlg.close()
    except Exception:
        pass

    if return_code != 0:
        QMessageBox.critical(parent, "GraXpert", "GraXpert process failed.")
        shutil.rmtree(working_dir, ignore_errors=True)
        return

    # v2 logic: look ONLY for {base}.{fits|tif|tiff|png} in working_dir
    output_file = _pick_exact_output(working_dir, output_basename)
    if not output_file:
        QMessageBox.critical(parent, "GraXpert", "GraXpert output file not found.")
        shutil.rmtree(working_dir, ignore_errors=True)
        return

    # Use the SAME loader as v2
    arr = None
    if _legacy_load_image is not None:
        try:
            arr, _, _, _ = _legacy_load_image(output_file)
        except Exception:
            arr = None

    # Fallback loader if legacy not available
    if arr is None:
        arr = _fallback_read_float01(output_file)

    if arr is None or arr.size == 0:
        QMessageBox.critical(parent, "GraXpert", "Could not read GraXpert output.")
        shutil.rmtree(working_dir, ignore_errors=True)
        return

    # v2 coerced mono -> RGB for its image_manager; your doc stack supports mono,
    # so we keep channels as-is. (If you want RGB, uncomment below.)
    # if arr.ndim == 2:
    #     arr = np.stack([arr] * 3, axis=-1)

    doc = target_doc
    try:
        meta = {
            "step_name": "GraXpert Gradient Removal",
            "bit_depth": "32-bit floating point",
            "is_mono": (arr.ndim == 2) or (arr.ndim == 3 and arr.shape[2] == 1),
            "file_path": output_file,
            "description": "GraXpert Gradient Removed",
        }
        doc.apply_edit(arr.astype(np.float32, copy=False), metadata=meta, step_name="GraXpert Gradient Removal")

        # Optional: if your active view had display-stretch on and this looks “flat”,
        # you can toggle it here. Leaving unchanged for consistency with ABE flow.

        #QMessageBox.information(parent, "Success", "Gradient removed successfully.")
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
