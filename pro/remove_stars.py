# pro/remove_stars.py
from __future__ import annotations
import os, platform, shutil, stat, tempfile
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QInputDialog, QMessageBox, QFileDialog,
    QDialog, QVBoxLayout, QTextEdit, QPushButton,
    QLabel, QComboBox, QCheckBox, QSpinBox, QFormLayout, QDialogButtonBox, QWidget, QHBoxLayout
)

# use your legacy I/O functions (as requested)
from legacy.image_manager import save_image, load_image

try:
    import cv2
except Exception:
    cv2 = None


# ------------------------------------------------------------
# Settings helper
# ------------------------------------------------------------
def _get_setting_any(settings, keys: tuple[str, ...], default: str = "") -> str:
    if not settings:
        return default
    for k in keys:
        try:
            v = settings.value(k, "", type=str)
        except Exception:
            v = settings.value(k, "")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return default


# ------------------------------------------------------------
# Public entry
# ------------------------------------------------------------
def remove_stars(main):
    """Choose StarNet or CosmicClarityDarkStar, process active doc, update starless in-place, open stars-only as new doc."""
    tool, ok = QInputDialog.getItem(
        main, "Select Star Removal Tool", "Choose a tool:",
        ["StarNet", "CosmicClarityDarkStar"], 0, False
    )
    if not ok:
        return

    # active doc
    doc = getattr(main, "_active_doc", None)
    if callable(doc):
        doc = doc()
    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.warning(main, "No Image", "Please load an image before removing stars.")
        return

    if tool == "CosmicClarityDarkStar":
        _run_darkstar(main, doc)
    else:
        _run_starnet(main, doc)


# ------------------------------------------------------------
# StarNet (SASv2-like: 16-bit TIFF in StarNet folder)
# ------------------------------------------------------------
def _run_starnet(main, doc):
    exe = _get_setting_any(getattr(main, "settings", None),
                           ("starnet/exe_path", "paths/starnet"), "")
    if not exe or not os.path.exists(exe):
        exe_path, _ = QFileDialog.getOpenFileName(main, "Select StarNet Executable", "", "Executable Files (*)")
        if not exe_path:
            return
        exe = exe_path
        s = getattr(main, "settings", None)
        if s:
            s.setValue("starnet/exe_path", exe)
            s.setValue("paths/starnet", exe)

    if platform.system() in ("Darwin", "Linux"):
        _ensure_exec_bit(exe)

    sysname = platform.system()
    if sysname not in ("Windows", "Darwin", "Linux"):
        QMessageBox.critical(main, "Unsupported OS",
                             f"The current operating system '{sysname}' is not supported.")
        return

    # SASv2: ask linearity
    reply = QMessageBox.question(
        main, "Image Linearity", "Is the current image linear?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No
    )
    is_linear = (reply == QMessageBox.StandardButton.Yes)

    # ensure RGB for StarNet
    src = np.asarray(doc.image)
    if src.ndim == 2:
        processing_image = np.stack([src] * 3, axis=-1)
    elif src.ndim == 3 and src.shape[2] == 1:
        processing_image = np.repeat(src, 3, axis=2)
    else:
        processing_image = src
    processing_image = processing_image.astype(np.float32, copy=False)

    # optional stretch (SASv2)
    did_stretch = False
    if is_linear and hasattr(main, "stretch_image") and callable(getattr(main, "stretch_image")):
        try:
            processing_image = main.stretch_image(processing_image)
            did_stretch = True
        except Exception:
            pass

    # write input/output paths in StarNet folder
    starnet_dir = os.path.dirname(exe) or os.getcwd()
    input_image_path = os.path.join(starnet_dir, "imagetoremovestars.tif")
    output_image_path = os.path.join(starnet_dir, "starless.tif")

    try:
        # StarNet requires 16-bit TIFF
        save_image(processing_image, input_image_path,
                   original_format="tif", bit_depth="16-bit",
                   original_header=None, is_mono=False,
                   image_meta=None, file_meta=None)
    except Exception as e:
        QMessageBox.critical(main, "StarNet", f"Failed to write input TIFF:\n{e}")
        return

    exe_name = os.path.basename(exe).lower()
    if sysname in ("Windows", "Linux"):
        command = [exe, input_image_path, output_image_path, "256"]
    else:  # macOS
        if "starnet2" in exe_name:
            command = [exe, "--input", input_image_path, "--output", output_image_path]
        else:
            command = [exe, input_image_path, output_image_path]

    dlg = _ProcDialog(main, title="StarNet Progress")
    thr = _ProcThread(command, cwd=starnet_dir)
    thr.output_signal.connect(dlg.append_text)
    thr.finished_signal.connect(
        lambda rc: _on_starnet_finished(main, doc, rc, dlg, input_image_path, output_image_path, did_stretch)
    )
    dlg.cancel_button.clicked.connect(thr.terminate)

    dlg.show()
    thr.start()
    dlg.exec()


def _on_starnet_finished(main, doc, return_code, dialog, input_path, output_path, did_stretch):
    dialog.append_text(f"\nProcess finished with return code {return_code}.\n")
    if return_code != 0:
        QMessageBox.critical(main, "StarNet Error", f"StarNet failed with return code {return_code}.")
        _safe_rm(input_path); _safe_rm(output_path)
        dialog.close()
        return

    if not os.path.exists(output_path):
        QMessageBox.critical(main, "StarNet Error", "Starless image was not created.")
        _safe_rm(input_path)
        dialog.close()
        return

    dialog.append_text(f"Starless image found at {output_path}. Loading image...\n")
    starless_rgb, _, _, _ = load_image(output_path)
    if starless_rgb is None:
        QMessageBox.critical(main, "StarNet Error", "Failed to load starless image.")
        _safe_rm(input_path); _safe_rm(output_path)
        dialog.close()
        return

    # ensure 3ch
    if starless_rgb.ndim == 2 or (starless_rgb.ndim == 3 and starless_rgb.shape[2] == 1):
        starless_rgb = np.stack([starless_rgb] * 3, axis=-1)
    starless_rgb = starless_rgb.astype(np.float32, copy=False)

    # unstretch (if we stretched)
    if did_stretch and hasattr(main, "unstretch_image") and callable(getattr(main, "unstretch_image")):
        dialog.append_text("Unstretching the starless image...\n")
        try:
            starless_rgb = main.unstretch_image(starless_rgb)
            dialog.append_text("Starless image unstretched successfully.\n")
        except Exception:
            dialog.append_text("Unstretch failed; continuing with returned starless.\n")

    # original image (from the doc)
    orig = np.asarray(doc.image)
    if orig.ndim == 2:
        original_rgb = np.stack([orig] * 3, axis=-1)
    elif orig.ndim == 3 and orig.shape[2] == 1:
        original_rgb = np.repeat(orig, 3, axis=2)
    else:
        original_rgb = orig
    original_rgb = original_rgb.astype(np.float32, copy=False)

    # Stars-Only (SASv2 formula)
    dialog.append_text("Generating stars-only image...\n")
    with np.errstate(divide='ignore', invalid='ignore'):
        stars_only = (original_rgb - starless_rgb) / np.clip(1.0 - starless_rgb, 1e-6, None)
        stars_only = np.nan_to_num(stars_only, nan=0.0, posinf=0.0, neginf=0.0)
    stars_only = np.clip(stars_only, 0.0, 1.0)

    # apply active mask (doc-based)
    m3 = _active_mask3_from_doc(doc, stars_only.shape[1], stars_only.shape[0])
    if m3 is not None:
        stars_only *= m3
        dialog.append_text("✅ Applied active mask to the stars-only image.\n")
    else:
        dialog.append_text("ℹ️ No active mask for stars-only; skipping.\n")

    # push Stars-Only as new document with suffix _stars
    _push_as_new_doc(main, doc, stars_only, title_suffix="_stars", source="Stars-Only (StarNet)")
    dialog.append_text("Stars-only image pushed.\n")

    # mask-blend starless with original using active mask
    dialog.append_text("Preparing to update current view with starless (mask-blend)...\n")
    final_starless = _mask_blend_with_doc_mask(doc, starless_rgb, original_rgb)

    # overwrite the current doc view
    try:
        meta = {
            "step_name": "Stars Removed",
            "bit_depth": "32-bit floating point",
            "is_mono": False,
        }
        doc.apply_edit(final_starless.astype(np.float32, copy=False), metadata=meta, step_name="Stars Removed")
        if hasattr(main, "_log"):
            main._log("Stars Removed (StarNet)")
    except Exception as e:
        QMessageBox.critical(main, "StarNet Error", f"Failed to apply starless result:\n{e}")

    _safe_rm(input_path); _safe_rm(output_path)
    dialog.append_text("Temporary files cleaned up.\n")
    dialog.close()


# ------------------------------------------------------------
# CosmicClarityDarkStar
# ------------------------------------------------------------
def _run_darkstar(main, doc):
    exe, base = _resolve_darkstar_exe(main)
    if not exe or not base:
        QMessageBox.critical(main, "Cosmic Clarity Folder Error",
                             "Cosmic Clarity Dark Star executable not set.")
        return

    # Input/output folders per SASv2
    input_dir  = os.path.join(base, "input")
    output_dir = os.path.join(base, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Save the current image as 32-bit float TIFF (no stretch)
    in_path = os.path.join(input_dir, "imagetoremovestars.tif")
    try:
        save_image(doc.image, in_path, original_format="tif",
                   bit_depth="32-bit floating point",
                   original_header=None, is_mono=False, image_meta=None, file_meta=None)
    except Exception as e:
        QMessageBox.critical(main, "Cosmic Clarity", f"Failed to write input TIFF:\n{e}")
        return

    # Show SASv2-style config dialog
    cfg = DarkStarConfigDialog(main)
    if not cfg.exec():
        _safe_rm(in_path)
        return
    params = cfg.get_values()
    disable_gpu = params["disable_gpu"]
    mode = params["mode"]                         # "unscreen" or "additive"
    show_extracted_stars = params["show_extracted_stars"]
    stride = params["stride"]                     # 64..1024, default 512

    # Build CLI exactly like SASv2 (using --chunk_size, not chunk_size)
    args = []
    if disable_gpu:
        args.append("--disable_gpu")
    args += ["--star_removal_mode", mode]
    if show_extracted_stars:
        args.append("--show_extracted_stars")
    args += ["--chunk_size", str(stride)]

    command = [exe] + args

    dlg = _ProcDialog(main, title="CosmicClarityDarkStar Progress")
    thr = _ProcThread(command, cwd=output_dir)
    thr.output_signal.connect(dlg.append_text)
    thr.finished_signal.connect(lambda rc: _on_darkstar_finished(main, doc, rc, dlg, in_path, output_dir))
    dlg.cancel_button.clicked.connect(thr.terminate)

    dlg.show()
    thr.start()
    dlg.exec()



def _resolve_darkstar_exe(main):
    """
    Return (exe_path, base_folder) or (None, None) on cancel/error.
    Accepts either a folder (stored) or a direct executable path.
    Saves the folder back to QSettings under 'paths/cosmic_clarity'.
    """
    settings = getattr(main, "settings", None)
    raw = _get_setting_any(settings, ("paths/cosmic_clarity", "cosmic_clarity_folder"), "")

    def _platform_exe_name():
        return "setiastrocosmicclarity_darkstar.exe" if platform.system() == "Windows" \
               else "setiastrocosmicclarity_darkstar"

    exe_name = _platform_exe_name()

    exe_path = None
    base_folder = None

    if raw:
        if os.path.isfile(raw):
            # user stored the executable path directly
            exe_path = raw
            base_folder = os.path.dirname(raw)
        elif os.path.isdir(raw):
            # user stored the parent folder
            base_folder = raw
            exe_path = os.path.join(base_folder, exe_name)

    # if missing or invalid, let user pick the executable directly
    if not exe_path or not os.path.exists(exe_path):
        picked, _ = QFileDialog.getOpenFileName(main, "Select CosmicClarityDarkStar Executable", "", "Executable Files (*)")
        if not picked:
            return None, None
        exe_path = picked
        base_folder = os.path.dirname(picked)

    # ensure exec bit on POSIX
    if platform.system() in ("Darwin", "Linux"):
        _ensure_exec_bit(exe_path)

    # persist folder (not the exe) to the canonical key
    if settings:
        settings.setValue("paths/cosmic_clarity", base_folder)
        settings.sync()

    return exe_path, base_folder


def _on_darkstar_finished(main, doc, return_code, dialog, in_path, output_dir):
    dialog.append_text(f"\nProcess finished with return code {return_code}.\n")
    if return_code != 0:
        QMessageBox.critical(main, "CosmicClarityDarkStar Error",
                             f"CosmicClarityDarkStar failed with return code {return_code}.")
        _safe_rm(in_path); dialog.close(); return

    starless_path = os.path.join(output_dir, "imagetoremovestars_starless.tif")
    if not os.path.exists(starless_path):
        QMessageBox.critical(main, "CosmicClarityDarkStar Error", "Starless image was not created.")
        _safe_rm(in_path); dialog.close(); return

    dialog.append_text(f"Loading starless image from {starless_path}...\n")
    starless, _, _, _ = load_image(starless_path)
    if starless is None:
        QMessageBox.critical(main, "CosmicClarityDarkStar Error", "Failed to load starless image.")
        _safe_rm(in_path); dialog.close(); return

    if starless.ndim == 2 or (starless.ndim == 3 and starless.shape[2] == 1):
        starless_rgb = np.stack([starless] * 3, axis=-1)
    else:
        starless_rgb = starless
    starless_rgb = starless_rgb.astype(np.float32, copy=False)

    src = np.asarray(doc.image)
    if src.ndim == 2:
        original_rgb = np.stack([src] * 3, axis=-1)
    elif src.ndim == 3 and src.shape[2] == 1:
        original_rgb = np.repeat(src, 3, axis=2)
    else:
        original_rgb = src
    original_rgb = original_rgb.astype(np.float32, copy=False)

    # stars-only optional push
    stars_path = os.path.join(output_dir, "imagetoremovestars_stars_only.tif")
    if os.path.exists(stars_path):
        dialog.append_text(f"Loading stars-only image from {stars_path}...\n")
        stars_only, _, _, _ = load_image(stars_path)
        if stars_only is not None:
            if stars_only.ndim == 2 or (stars_only.ndim == 3 and stars_only.shape[2] == 1):
                stars_only = np.stack([stars_only] * 3, axis=-1)
            stars_only = stars_only.astype(np.float32, copy=False)
            m3 = _active_mask3_from_doc(doc, stars_only.shape[1], stars_only.shape[0])
            if m3 is not None:
                stars_only *= m3
                dialog.append_text("✅ Applied active mask to stars-only image.\n")
            else:
                dialog.append_text("ℹ️ Mask not active for stars-only; skipping.\n")
            _push_as_new_doc(main, doc, stars_only, title_suffix="_stars", source="Stars-Only (DarkStar)")
        else:
            dialog.append_text("Failed to load stars-only image.\n")
    else:
        dialog.append_text("No stars-only image generated.\n")

    # mask-blend starless → overwrite current doc
    dialog.append_text("Mask-blending starless image before update...\n")
    final_starless = _mask_blend_with_doc_mask(doc, starless_rgb, original_rgb)
    try:
        meta = {
            "step_name": "Stars Removed",
            "bit_depth": "32-bit floating point",
            "is_mono": False,
        }
        doc.apply_edit(final_starless.astype(np.float32, copy=False), metadata=meta, step_name="Stars Removed")
        if hasattr(main, "_log"):
            main._log("Stars Removed (DarkStar)")
    except Exception as e:
        QMessageBox.critical(main, "CosmicClarityDarkStar", f"Failed to apply result:\n{e}")

    # cleanup
    try:
        _safe_rm(in_path)
        sp = starless_path
        if os.path.exists(sp): _safe_rm(sp)
        sp2 = os.path.join(output_dir, "imagetoremovestars_stars_only.tif")
        if os.path.exists(sp2): _safe_rm(sp2)
        dialog.append_text("Temporary files cleaned up.\n")
    except Exception as e:
        dialog.append_text(f"Cleanup error: {e}\n")

    dialog.close()


# ------------------------------------------------------------
# Mask helpers (doc-centric)
# ------------------------------------------------------------
def _active_mask_array_from_doc(doc) -> np.ndarray | None:
    """Return active mask (H,W) float32 in [0,1] from the document, if present."""
    try:
        mid = getattr(doc, "active_mask_id", None)
        if not mid:
            return None
        masks = getattr(doc, "masks", {}) or {}
        layer = masks.get(mid)
        data = getattr(layer, "data", None) if layer is not None else None
        if data is None:
            return None
        a = np.asarray(data)
        if a.ndim == 3:
            if cv2 is not None:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            else:
                a = a.mean(axis=2)
        a = a.astype(np.float32, copy=False)
        a = np.clip(a, 0.0, 1.0)
        return a
    except Exception:
        return None


def _active_mask3_from_doc(doc, w, h) -> np.ndarray | None:
    """Return 3-channel mask resized to (h,w) if a doc-level mask exists; else None."""
    m = _active_mask_array_from_doc(doc)
    if m is None:
        return None
    if m.shape != (h, w):
        if cv2 is not None:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            yi = (np.linspace(0, m.shape[0] - 1, h)).astype(np.int32)
            xi = (np.linspace(0, m.shape[1] - 1, w)).astype(np.int32)
            m = m[yi][:, xi]
    return np.repeat(m[:, :, None], 3, axis=2).astype(np.float32, copy=False)


def _mask_blend_with_doc_mask(doc, starless_rgb: np.ndarray, original_rgb: np.ndarray) -> np.ndarray:
    """Blend using mask from doc if present: result = starless*m + original*(1-m)."""
    m = _active_mask_array_from_doc(doc)
    if m is None:
        return starless_rgb
    h, w = starless_rgb.shape[:2]
    if m.shape != (h, w):
        if cv2 is not None:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            yi = (np.linspace(0, m.shape[0] - 1, h)).astype(np.int32)
            xi = (np.linspace(0, m.shape[1] - 1, w)).astype(np.int32)
            m = m[yi][:, xi]
    m3 = np.repeat(m[:, :, None], 3, axis=2)
    return np.clip(starless_rgb * m3 + original_rgb * (1.0 - m3), 0.0, 1.0).astype(np.float32, copy=False)


# ------------------------------------------------------------
# New document helper
# ------------------------------------------------------------
def _push_as_new_doc(main, doc, arr: np.ndarray, title_suffix="_stars", source="Stars-Only"):
    dm = getattr(main, "docman", None)
    if not dm or not hasattr(dm, "open_array"):
        return
    try:
        base = ""
        if hasattr(doc, "display_name") and callable(doc.display_name):
            base = doc.display_name()
        else:
            base = getattr(doc, "name", "") or "Image"
        meta = {
            "bit_depth": "32-bit floating point",
            "is_mono": (arr.ndim == 2),
            "source": source
        }
        newdoc = dm.open_array(arr.astype(np.float32, copy=False), metadata=meta, title=f"{base}{title_suffix}")
        if hasattr(main, "_spawn_subwindow_for"):
            main._spawn_subwindow_for(newdoc)
    except Exception:
        pass


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def _ensure_exec_bit(path: str):
    if platform.system() == "Windows":
        return
    try:
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass


def _safe_rm(p):
    try:
        if p and os.path.exists(p):
            os.remove(p)
    except Exception:
        pass


# ------------------------------------------------------------
# Proc runner & dialog (merged stdout/stderr)
# ------------------------------------------------------------
class _ProcThread(QThread):
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, command: list[str], cwd: str | None = None, parent=None):
        super().__init__(parent)
        self.command = command
        self.cwd = cwd

    def run(self):
        import subprocess, os
        env = os.environ.copy()
        for k in ("PYTHONHOME","PYTHONPATH","DYLD_LIBRARY_PATH","DYLD_FALLBACK_LIBRARY_PATH","PYTHONEXECUTABLE"):
            env.pop(k, None)
        rc = -1
        try:
            p = subprocess.Popen(
                self.command, cwd=self.cwd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, text=True, start_new_session=True, env=env
            )
            for line in iter(p.stdout.readline, ""):
                if not line: break
                self.output_signal.emit(line.rstrip())
            try:
                p.stdout.close()
            except Exception:
                pass
            rc = p.wait()
        except Exception as e:
            self.output_signal.emit(str(e))
            rc = -1
        self.finished_signal.emit(rc)


class _ProcDialog(QDialog):
    def __init__(self, parent, title="Process"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(600, 420)
        lay = QVBoxLayout(self)
        self.text = QTextEdit(self); self.text.setReadOnly(True)
        lay.addWidget(self.text)
        self.cancel_button = QPushButton("Cancel", self)
        lay.addWidget(self.cancel_button)

    def append_text(self, s: str):
        try:
            self.text.append(s)
        except Exception:
            pass


class DarkStarConfigDialog(QDialog):
    """
    SASv2-style config UI:
      - Disable GPU: Yes/No (default No)
      - Star Removal Mode: unscreen | additive (default unscreen)
      - Show Extracted Stars: Yes/No (default No)
      - Stride (powers of 2): 64,128,256,512,1024 (default 512)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CosmicClarity Dark Star Settings")

        self.chk_disable_gpu = QCheckBox("Disable GPU")
        self.chk_disable_gpu.setChecked(False)  # default No (unchecked)

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["unscreen", "additive"])
        self.cmb_mode.setCurrentText("unscreen")

        self.chk_show_stars = QCheckBox("Show Extracted Stars")
        self.chk_show_stars.setChecked(True)

        self.cmb_stride = QComboBox()
        for v in (64, 128, 256, 512, 1024):
            self.cmb_stride.addItem(str(v), v)
        self.cmb_stride.setCurrentText("512")  # default 512

        form = QFormLayout()
        form.addRow("Star Removal Mode:", self.cmb_mode)
        form.addRow("Stride (power of two):", self.cmb_stride)
        form.addRow("", self.chk_disable_gpu)
        form.addRow("", self.chk_show_stars)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(btns)

    def get_values(self):
        return {
            "disable_gpu": self.chk_disable_gpu.isChecked(),
            "mode": self.cmb_mode.currentText(),
            "show_extracted_stars": self.chk_show_stars.isChecked(),
            "stride": int(self.cmb_stride.currentData()),
        }
