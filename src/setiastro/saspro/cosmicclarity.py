# setiastro/saspro/cosmicclarity.py
from __future__ import annotations
import os
import sys
import glob
import time
import tempfile
import uuid
import numpy as np

from PyQt6.QtCore import Qt, QTimer, QSettings, QThread, pyqtSignal, QFileSystemWatcher, QEvent, QTimer
from PyQt6.QtGui import QIcon, QAction, QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout, QLabel, QPushButton,
    QSlider, QCheckBox, QComboBox, QMessageBox, QWidget, QRadioButton, QProgressBar,
    QTextEdit, QFileDialog, QTreeWidget, QTreeWidgetItem, QMenu, QInputDialog, QDoubleSpinBox
)
from PyQt6.QtCore import QProcess, QCoreApplication, QT_TRANSLATE_NOOP

# ---- bring in your image IO helpers ----
# Adjust these imports to your project structure if needed.
from setiastro.saspro.legacy.image_manager import load_image, save_image  

from setiastro.saspro.imageops.stretch import stretch_mono_image, stretch_color_image
# (deleted — aberration_ai is no longer used; kept only as dead code in worker)
from setiastro.saspro.resources import get_icons

from setiastro.saspro.cosmicclarity_engines.sharpen_engine import sharpen_rgb01
from setiastro.saspro.help_support import make_help_button
from setiastro.saspro.widgets.factory_reset import (
    make_reset_button, snapshot_controls, restore_controls)
from setiastro.saspro.cosmicclarity_engines.denoise_engine import denoise_rgb01
from setiastro.saspro.cosmicclarity_engines.superres_engine import superres_rgb01
from setiastro.saspro.cosmicclarity_engines._backend_guard import EngineCancelled
from setiastro.saspro.cosmicclarity_engines.satellite_engine import (
    get_satellite_models,
    satellite_remove_image,
)
from setiastro.saspro.cosmicclarity_headless import (
    resolve_overlap_px,
    parse_overlap_pct,
    nearest_overlap_pct,
    OVERLAP_PCT_CHOICES,
    DEFAULT_OVERLAP_PCT,
)
# Import centralized preview dialog
from setiastro.saspro.widgets.preview_dialogs import ImagePreviewDialog

import shutil
import subprocess


def _tr(text: str) -> str:
    return QCoreApplication.translate("CosmicClarity", text)


# ---------------------------------------------------------------------------
# Input formats the satellite tool accepts
#
# This list used to be three separate inlined tuples -- _refresh_tree,
# SatelliteEngineThread._iter_files and the single-image file dialog -- and all
# three were missing .fz / .fts / gzipped FITS. load_image() has always read
# those and Blink's folder loader has always offered them, so a frame you could
# flag in Blink was invisible here: the folder trees came up empty and batch
# found nothing to do. One list now, shared by all three.
# ---------------------------------------------------------------------------

# Extensions os.path.splitext() cannot see whole. Checked longest-first.
_COMPOUND_EXTS = (".fits.gz", ".fit.gz", ".fts.gz")


def split_ext_compound(path: str):
    """os.path.splitext, except ".fits.gz" comes back as ONE extension.

    Plain splitext hands back ".gz" for a gzipped FITS. That is why the
    ".fits.gz" branch in _save_fmt_and_bitdepth_from_path_for_legacy below
    could never fire, and why "x.fits.gz" produced the output name
    "x.fits_satellited.gz".
    """
    low = path.lower()
    for ce in _COMPOUND_EXTS:
        if low.endswith(ce):
            return path[:-len(ce)], path[-len(ce):]
    return os.path.splitext(path)


SAT_INPUT_EXTS = (
    ".png", ".tif", ".tiff",
    ".fit", ".fits", ".fts",
    ".fits.gz", ".fit.gz", ".fts.gz", ".fz",
    ".xisf",
    ".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef",
    ".jpg", ".jpeg",
)

SAT_INPUT_FILTER = "Image Files (" + " ".join("*" + e for e in SAT_INPUT_EXTS) + ")"

# Input extension -> extension the RESULT is actually written with.
# save_image has no Rice/gzip FITS encoder and no RAW writer, so every one of
# those comes back as plain .fits. Resolving that HERE, rather than letting
# save_image quietly rewrite the name afterwards, is what keeps the batch
# loop's "Exists, skipping" test pointing at the file that really gets
# created -- otherwise every re-run reprocessed every .fz in the folder.
_SAT_OUT_EXT = {
    ".fit": ".fit", ".fits": ".fits", ".fts": ".fits",
    ".fits.gz": ".fits", ".fit.gz": ".fits", ".fts.gz": ".fits", ".fz": ".fits",
    ".tif": ".tif", ".tiff": ".tiff",
    ".xisf": ".xisf",
    ".png": ".png", ".jpg": ".jpg", ".jpeg": ".jpeg",
}


def satellite_output_name(filename: str, suffix: str = "_satellited") -> str:
    """The name (or path) the processed result for `filename` is saved under."""
    stem, ext = split_ext_compound(filename)
    return stem + suffix + _SAT_OUT_EXT.get(ext.lower(), ".fits")


def _save_fmt_and_bitdepth_from_path_for_legacy(path: str):
    ext = split_ext_compound(path)[1].lower()

    if ext in (".fit", ".fits", ".fts", ".fits.gz", ".fit.gz", ".fts.gz", ".fz"):
        return "fits", "32-bit floating point"

    if ext in (".tif", ".tiff"):
        return "tif", "32-bit floating point"

    if ext == ".xisf":
        return "xisf", "32-bit floating point"

    if ext == ".png":
        return "png", None  # legacy always saves 8-bit PNG

    if ext in (".jpg", ".jpeg"):
        return "jpg", None  # legacy always saves 8-bit JPG

    return None, None


def _save_image_no_auto_legacy(img, dst_path: str, hdr, mono, *, image_meta=None, file_meta=None, wcs_header=None):
    fmt, bd = _save_fmt_and_bitdepth_from_path_for_legacy(dst_path)

    # Unknown/unsupported extension -> force FITS output
    if fmt is None:
        base, _ = os.path.splitext(dst_path)
        dst_path = base + ".fits"
        fmt, bd = "fits", "32-bit floating point"

    save_image(
        img,
        dst_path,
        fmt,
        bd,
        hdr,
        mono,
        image_meta=image_meta,
        file_meta=file_meta,
        wcs_header=wcs_header,
    )
    return dst_path


def _cc_models_installed(kind: str = "core") -> tuple[bool, str]:
    """
    Returns (ok, detail)

    kind:
        "core"      -> normal Cosmic Clarity (sharpen / denoise / superres)
        "satellite" -> satellite removal only
        "all"       -> require both core + satellite
    """
    models_dir = ""
    try:
        from setiastro.saspro.resources import get_models_dir
        models_dir = get_models_dir()
    except Exception:
        pass

    if not models_dir or not os.path.isdir(models_dir):
        return False, _tr("Models directory is missing.")

    # Pick one stable sentinel per feature set.
    sentinels = {
        "core": [
            "deep_sharp_stellar_AI4.pth",  

        ],
        "satellite": [
            "satelliteRemovalAI4.pth",
        ],
        "all": [
            "deep_sharp_stellar_AI4.pth",
            "satelliteRemovalAI4.pth",
        ],
    }

    required = sentinels.get(kind, sentinels["core"])

    missing = [name for name in required if not os.path.exists(os.path.join(models_dir, name))]
    if missing:
        return False, _tr("Missing model(s): {0}").format(", ".join(missing))

    return True, ""

def _warn_models_missing_and_close(parent: QWidget, detail: str = ""):
    from setiastro.saspro.setup_center import offer_setup
    offer_setup(parent, "models", "Cosmic Clarity",
                _tr("Cosmic Clarity needs its AI models, and they are not installed yet."),
                detail)


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

    #sasprompt user once
    QMessageBox.information(parent, "Cosmic Clarity",
        _tr("Please select your Cosmic Clarity folder (the one that contains the CC executables and input/output)."))
    folder = QFileDialog.getExistingDirectory(parent, _tr("Select Cosmic Clarity Folder"), "")
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
    """Progress window for a Cosmic Clarity run.

    The bar is a busy indicator until the engine reports real progress, and an
    elapsed clock ticks on the GUI thread, so a long model load or tile pass reads
    as working rather than hung. Closing it (title-bar X or Esc) means Cancel: it is
    never just hidden over a worker that keeps going. The owner calls finish() once
    the worker has actually stopped.
    """
    cancelled = pyqtSignal()

    def __init__(self, title=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title if title is not None else self.tr("Processing…"))
        self._finished = False
        self._cancel_requested = False
        self._has_progress = False
        self._t0 = time.monotonic()

        v = QVBoxLayout(self)
        self.lbl = QLabel(self.tr("Processing, please wait…"))
        self.txt = QTextEdit()
        self.txt.setReadOnly(True)

        self.pb = QProgressBar()
        self.pb.setRange(0, 0)          # busy until the engine reports progress
        self.pb.setFormat("%p%")

        self.lbl_elapsed = QLabel("")
        self.btn_cancel = QPushButton(self.tr("Cancel"))
        self.btn_cancel.clicked.connect(self.request_cancel)

        v.addWidget(self.lbl)
        v.addWidget(self.txt)
        v.addWidget(self.pb)
        v.addWidget(self.lbl_elapsed)
        v.addWidget(self.btn_cancel)

        self._clock = QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._update_elapsed)
        self._clock.start()
        self._update_elapsed()

    def _update_elapsed(self):
        secs = int(time.monotonic() - self._t0)
        h, rem = divmod(secs, 3600)
        m, sec = divmod(rem, 60)
        self.lbl_elapsed.setText(self.tr("Elapsed {0}:{1:02d}:{2:02d}").format(h, m, sec) if h else self.tr("Elapsed {0}:{1:02d}").format(m, sec))

    def append_output(self, line: str):
        self.txt.append(line)

    def set_stage(self, text: str):
        if not self._cancel_requested:
            self.lbl.setText(str(text))

    def mark_cancelling(self):
        """Show that a cancel is under way. Does not emit cancelled."""
        if self._cancel_requested or self._finished:
            return
        self._cancel_requested = True
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setText(self.tr("Cancelling…"))
        self.lbl.setText(self.tr("Cancelling: stopping after the current tile batch…"))
        self.txt.append("Cancel requested… waiting for the worker to stop.")

    def request_cancel(self):
        if self._finished or self._cancel_requested:
            return
        self.mark_cancelling()
        self.cancelled.emit()

    def finish(self):
        """Close for real. Call once the worker has stopped."""
        self._finished = True
        self._clock.stop()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.close()

    def closeEvent(self, e):
        # The title-bar X while the worker runs is a cancel, not a hide. A
        # programmatic close() (not spontaneous) still closes as it always did.
        if not self._finished and e.spontaneous():
            e.ignore()
            self.request_cancel()
            return
        self._clock.stop()
        super().closeEvent(e)

    def keyPressEvent(self, e):
        # QDialog maps Esc to reject(), which would only hide the window.
        if e.key() == Qt.Key.Key_Escape and not self._finished:
            e.accept()
            self.request_cancel()
            return
        super().keyPressEvent(e)

    def set_progress(self, done: int, total: int | None = None):
        """
        Supports both:
            set_progress(percent)
        and:
            set_progress(done, total)
        """

        # Percent mode (CC dialog)
        if total is None:
            pct = max(0, min(100, int(done)))
            if pct <= 0 and not self._has_progress:
                return              # stay busy until real work has been reported
            self._has_progress = True
            if self.pb.maximum() != 100:
                self.pb.setRange(0, 100)
                self.pb.setFormat("%p%")
            self.pb.setValue(pct)
            return

        # Indeterminate mode
        if total <= 0:
            if not self._cancel_requested:
                self.lbl.setText(self.tr("Waiting for files…"))
            self.pb.setRange(0, 0)
            return

        # done / total mode
        done = max(0, min(int(done), int(total)))
        if not self._cancel_requested:
            self.lbl.setText(self.tr("Batch processing files…"))
        self.pb.setRange(0, int(total))
        self.pb.setValue(done)
        self.pb.setFormat(self.tr("{0} / {1} files").format(done, total))


# Worker threads are owned by thread_keeper, not by any window: closing or
# deleting a dialog mid-run must never destroy a running QThread (a qFatal).
from setiastro.saspro.thread_keeper import (  # noqa: E402
    start_owned_thread as _start_cc_thread,
    thread_running as _cc_thread_running,
)


def _blend_cc_mask(processed: np.ndarray, src: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """out = m*processed + (1-m)*src, with the dialog's historical mask handling.

    src is the image the run started from (3-channel); a blank mask is no effect,
    and a non-finite result falls back to the unblended output."""
    m = np.asarray(mask).astype(np.float32)
    if m.size == 0:
        return processed
    mx = float(m.max())
    if mx > 1.0:
        m = m / mx
    elif mx == 0.0:
        return processed
    m = np.clip(m, 0.0, 1.0)

    mh, mw = m.shape[:2]
    oh, ow = processed.shape[:2]
    if (mh, mw) != (oh, ow):
        yi = np.linspace(0, mh - 1, oh).astype(np.int32)
        xi = np.linspace(0, mw - 1, ow).astype(np.int32)
        m = m[yi][:, xi]

    src = np.asarray(src, dtype=np.float32)
    out = processed.astype(np.float32)
    if out.ndim == 3 and out.shape[2] >= 3:
        m = m[..., None]
        if src.ndim == 2:
            src = np.stack([src] * 3, axis=-1)
    elif out.ndim == 2:
        if src.ndim == 3:
            src = src[..., 0]

    result = m * out + (1.0 - m) * src
    if not np.isfinite(result).all():
        return processed
    return result.astype(np.float32)


class CosmicClarityEngineWorker(QThread):
    progress = pyqtSignal(int)      # 0..100
    log      = pyqtSignal(str)
    result   = pyqtSignal(object, str)  # (np.ndarray float32 RGB01, final_step_title)
    error    = pyqtSignal(str)
    stage    = pyqtSignal(str)      # label for the running stage
    cancelled = pyqtSignal()        # stopped on request; nothing to apply

    _STAGE_LABELS = {
        "sharpen": QT_TRANSLATE_NOOP("CosmicClarity", "Sharpening"),
        "denoise": QT_TRANSLATE_NOOP("CosmicClarity", "Denoising"),
        "superres": QT_TRANSLATE_NOOP("CosmicClarity", "Super resolution"),
        "aberration": QT_TRANSLATE_NOOP("CosmicClarity", "Aberration correction"),
    }

    def __init__(self, img_rgb01: np.ndarray, preset: dict, parent=None, *,
                 mask: np.ndarray | None = None, collapse_to_mono: bool = False):
        super().__init__(parent)
        self._img = np.asarray(img_rgb01, dtype=np.float32)
        self._preset = dict(preset)
        self._cancel = False
        # Finishing (mono collapse, mask blend) runs here, not in the GUI slot:
        # a full-frame blend on the GUI thread freezes the window on big images.
        self._mask = mask
        self._collapse_to_mono = bool(collapse_to_mono)
        self._last_pct = -1

    def cancel(self):
        self._cancel = True

    def _stop_requested(self) -> bool:
        return bool(self._cancel or self.isInterruptionRequested())

    def _finish_output(self, out: np.ndarray, src: np.ndarray, mode: str) -> np.ndarray:
        if self._collapse_to_mono and out.ndim == 3:
            out = out.mean(axis=2).astype(np.float32)
        if self._mask is None:
            return out
        if mode == "superres":
            # The mask is in the source geometry and super-res changes it (the
            # old GUI-side blend raised on the shape mismatch).
            self.log.emit("Mask not used: Super Resolution changes the image size.")
            return out
        return _blend_cc_mask(out, src, self._mask)

    # ---- progress adapter ----
    def _mk_progress_cb(self, stage_label: str, stage_weight: float, base_pct: float, total_stages: int):
        """
        stage_weight: fraction of total progress reserved for this stage (0..1)
        base_pct: starting % for this stage
        """
        def _cb(done: int, total: int):
            if self._stop_requested():
                return False
            if total <= 0:
                return True
            frac = float(done) / float(total)
            pct = int(max(0, min(100, round(base_pct + stage_weight * 100.0 * frac))))
            if pct != self._last_pct:       # one signal per visible step, not per tile
                self._last_pct = pct
                self.progress.emit(pct)
            return True
        return _cb

    def run(self):
        try:
            p = self._preset
            mode = str(p.get("mode", "sharpen")).lower()

            use_gpu = bool(p.get("gpu", True))
            create_new_view = bool(p.get("create_new_view", False))  # not used here; dialog decides
            temp_stretch = bool(p.get("temp_stretch", False))
            target_median = float(p.get("target_median", 0.25))
            target_median = max(0.01, min(0.50, target_median))
            img = np.clip(self._img, 0.0, 1.0).astype(np.float32, copy=False)
            apply_aberration = bool(p.get("aberration_first", False))
            ab_patch = int(p.get("aberration_patch", 512))
            ab_overlap = int(p.get("aberration_overlap", 64))
            ab_border = int(p.get("aberration_border_px", 10))
            ab_auto_gpu = bool(p.get("aberration_auto_gpu", True))
            ab_provider = p.get("aberration_provider", None)

            # Decide stage plan
            stages = []
            if apply_aberration:
                stages.append("aberration")
            if mode == "sharpen":
                stages.append("sharpen")
            elif mode == "denoise":
                stages.append("denoise")
            elif mode == "both":
                stages.extend(["sharpen", "denoise"])
            elif mode == "superres":
                stages.append("superres")
            else:
                stages = ["sharpen"]

            n = max(1, len(stages))
            stage_weight = 1.0 / float(n)

            out = img
            self.progress.emit(0)

            for si, st in enumerate(stages):
                if self._stop_requested():
                    raise EngineCancelled("Cancelled.")

                base_pct = (100.0 * si) / float(n)
                self.log.emit(f"Running {st}…")
                _lbl = self._STAGE_LABELS.get(st)
                self.stage.emit((QCoreApplication.translate("CosmicClarity", _lbl) if _lbl else st.title()) + "…")

                if st == "sharpen":
                    sharpening_mode = p.get("sharpening_mode", "Both")
                    stellar_amount = float(p.get("stellar_amount", 0.5))
                    nonstellar_amount = float(p.get("nonstellar_amount", 0.5))
                    nonstellar_psf = float(p.get("nonstellar_psf", 3.0))
                    auto_psf = bool(p.get("auto_psf", True))
                    sharpen_sep = bool(p.get("sharpen_channels_separately", False))

                    prog = self._mk_progress_cb("sharpen", stage_weight, base_pct, n)

                    sharpen_execution_mode = str(
                        p.get("sharpen_execution_mode", p.get("execution_mode", "auto"))
                    )
                    sharpen_batch_override = int(
                        p.get("sharpen_batch_size_override", p.get("batch_size_override", 0))
                    )

                    self.log.emit(f"Sharpen execution mode: {sharpen_execution_mode}")

                    out = sharpen_rgb01(
                        out,
                        sharpening_mode=str(sharpening_mode),
                        stellar_amount=float(stellar_amount),
                        nonstellar_amount=float(nonstellar_amount),
                        nonstellar_strength=float(nonstellar_psf),
                        auto_detect_psf=bool(auto_psf),
                        separate_channels=bool(sharpen_sep),
                        use_gpu=bool(use_gpu),
                        temp_stretch=temp_stretch,
                        target_median=target_median,
                        chunk_size=int(p.get("chunk_size", 256)),
                        overlap=resolve_overlap_px(p),
                        execution_mode=sharpen_execution_mode,
                        batch_size_override=sharpen_batch_override,
                        stellar_correct_mode=str(p.get("stellar_correct_mode", "sharpen_only")),
                        correct_model_version=str(p.get("correct_model_version", "V2 (latest)")),
                        correct_conservative=bool(p.get("correct_conservative", False)),
                        progress_cb=prog,
                    )

                elif st == "denoise":
                    prog = self._mk_progress_cb("denoise", stage_weight, base_pct, n)

                    self.log.emit(
                        f"Denoise execution mode: {self._preset.get('denoise_execution_mode', 'auto')}"
                    )

                    out = denoise_rgb01(
                        out,
                        denoise_strength=float(self._preset.get("denoise_luma", 0.5)),
                        denoise_mode=str(self._preset.get("denoise_mode", "luminance")),
                        separate_channels=bool(self._preset.get("separate_channels", False)),
                        color_denoise_strength=float(self._preset.get("denoise_color", 0.5)),
                        chunk_size=int(self._preset.get("chunk_size", 256)),
                        overlap=resolve_overlap_px(self._preset),
                        use_gpu=bool(self._preset.get("gpu", True)),
                        lite=bool(self._preset.get("denoise_lite", False)),
                        walking=bool(self._preset.get("denoise_walking", False)),  # ← new
                        temp_stretch=bool(self._preset.get("temp_stretch", False)),
                        target_median=float(self._preset.get("target_median", 0.25)),
                        progress_cb=prog,
                        execution_mode=str(self._preset.get("denoise_execution_mode", "auto")),
                        batch_size_override=int(self._preset.get("denoise_batch_size_override", 0)),
                    )
                elif st == "aberration":
                    pass  # aberration stage removed

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

            if self._stop_requested():
                raise EngineCancelled("Cancelled.")

            # Title for history
            if mode == "both":
                step_title = "Cosmic Clarity – Sharpen + Denoise"
            elif mode == "superres":
                step_title = "Cosmic Clarity – Super Resolution"
            else:
                step_title = f"Cosmic Clarity – {mode.title()}"

            out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
            out = self._finish_output(out, img, mode)
            if self._stop_requested():
                raise EngineCancelled("Cancelled.")
            self.progress.emit(100)
            self.result.emit(out, step_title)

        except EngineCancelled:
            self.cancelled.emit()
        except Exception as e:
            if self._stop_requested():
                # An engine path that surfaced the cancel as a plain error.
                self.cancelled.emit()
            else:
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
        self._engine_thread = None
        self._closing_after_cancel = False
        self._wait = None
        self.icons = get_icons()
        self._orig_was_mono = False
 
        # Hard guard unless explicitly bypassed (used by preset runner)
        if not bypass_guard and self._headless_guard_active():
            try:
                self.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            QTimer.singleShot(0, self.reject)
            return
 
        ok, detail = _cc_models_installed("core")
        if not ok:
            _warn_models_missing_and_close(self, detail)
            try:
                self.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            except Exception:
                pass
            QTimer.singleShot(0, self.reject)
            return
 
        self.setWindowTitle(self.tr("Cosmic Clarity"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass
 
        if icon:
            try:
                self.setWindowIcon(icon)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
 
        self.parent_ref = parent
        self.doc = doc
        self._engine_thread = None
        self._closing_after_cancel = False
        # The mask blend uses the image as it is at Execute (in the worker), so the
        # dialog no longer holds a full-frame copy for as long as it is open.
        self.orig = None
        self.cosmic_root = ""
 
        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------
        def _add_inline_slider(layout: QGridLayout, row: int, label: QLabel, slider: QSlider, col_span: int = 1):
            layout.addWidget(label, row, 0)
            layout.addWidget(slider, row, 1, 1, col_span)
 
        # ------------------------------------------------------------------
        # Main layout
        # ------------------------------------------------------------------
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)
 
        # ---------------- Header ----------------
        header = QHBoxLayout()
        header.setSpacing(10)
 
        self.lbl_logo = QLabel(self)
        self.lbl_logo.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        try:
            pm = QPixmap(self.icons.COSMICCLARITY)
            if not pm.isNull():
                self.lbl_logo.setPixmap(
                    pm.scaledToWidth(220, Qt.TransformationMode.SmoothTransformation)
                )
                header.addWidget(self.lbl_logo, 0)
        except Exception:
            pass
 
        hdr_text = QVBoxLayout()
        hdr_text.setContentsMargins(0, 0, 0, 0)
        text_wrap = QWidget()
        text_lay = QVBoxLayout(text_wrap)
        text_lay.setContentsMargins(0, 0, 0, 0)
        text_lay.setSpacing(2)
        lbl_title = QLabel(self.tr("Cosmic Clarity"))
        try:
            f = lbl_title.font()
            f.setBold(True)
            f.setPointSize(max(f.pointSize(), 12))
            lbl_title.setFont(f)
        except Exception:
            pass
        lbl_sub = QLabel(self.tr("Sharpen, denoise, and super-resolution for the active image."))
        lbl_sub.setWordWrap(True)
        text_lay.addWidget(lbl_title)
        text_lay.addWidget(lbl_sub)
        hdr_text.addStretch(1)
        hdr_text.addWidget(text_wrap, 0, Qt.AlignmentFlag.AlignVCenter)
        hdr_text.addStretch(1)
        header.addLayout(hdr_text, 1)
        outer.addLayout(header)
 
        # ---------------- General group ----------------
        grp_general = QGroupBox(self.tr("General"))
        gen = QGridLayout(grp_general)
        gen.setContentsMargins(8, 8, 8, 8)
        gen.setHorizontalSpacing(8)
        gen.setVerticalSpacing(6)
 
        gen.addWidget(QLabel(self.tr("Mode:")), 0, 0)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems([self.tr("Sharpen"), self.tr("Denoise"), self.tr("Both"), self.tr("Super Resolution")])
        gen.addWidget(self.cmb_mode, 0, 1)
 
        self.lbl_gpu = QLabel(self.tr("Use GPU:"))
        gen.addWidget(self.lbl_gpu, 0, 2)
        self.cmb_gpu = QComboBox()
        self.cmb_gpu.addItems([self.tr("Yes"), self.tr("No")])
        gen.addWidget(self.cmb_gpu, 0, 3)
 
        gen.addWidget(QLabel(self.tr("Apply to:")), 1, 0)
        self.cmb_target = QComboBox()
        self.cmb_target.addItems([self.tr("Overwrite active view"), self.tr("Create new view")])
        gen.addWidget(self.cmb_target, 1, 1, 1, 3)
 
        gen.setColumnStretch(1, 1)
        gen.setColumnStretch(3, 1)
        outer.addWidget(grp_general)
 
        # ---------------- Two-column body ----------------
        body = QHBoxLayout()
        body.setSpacing(10)
        left_col = QVBoxLayout()
        left_col.setSpacing(8)
        right_col = QVBoxLayout()
        right_col.setSpacing(8)
 
        # ---------------- Sharpen group ----------------
        grp_sh = QGroupBox(self.tr("Sharpen"))
        sh = QGridLayout(grp_sh)
        sh.setContentsMargins(8, 8, 8, 8)
        sh.setHorizontalSpacing(8)
        sh.setVerticalSpacing(6)
 
        self.lbl_stellar_correct = QLabel(self.tr("Stellar Correction:"))
        sh.addWidget(self.lbl_stellar_correct, 0, 0)

        corr_row = QHBoxLayout()
        self.rb_correct_only    = QRadioButton(self.tr("Correct Only"))
        self.rb_correct_sharpen = QRadioButton(self.tr("Correct + Sharpen"))
        self.rb_sharpen_only    = QRadioButton(self.tr("Sharpen Only"))
        self.rb_correct_sharpen.setChecked(True)
        self.rb_correct_only.setToolTip(self.tr(
            "Run the Aberration Correction model only.\n"
            "Fixes pinched optics, coma, spherical aberration, chromatic aberration etc.\n"
            "No sharpening is applied."
        ))
        self.rb_correct_sharpen.setToolTip(self.tr(
            "Run Aberration Correction first, then Stellar Sharpening.\n"
            "Best for images with both shape aberrations and soft stars."
        ))
        self.rb_sharpen_only.setToolTip(self.tr(
            "Standard Cosmic Clarity stellar sharpening — no correction pre-pass."
        ))
        corr_row.addWidget(self.rb_correct_only)
        corr_row.addWidget(self.rb_correct_sharpen)
        corr_row.addWidget(self.rb_sharpen_only)
        corr_row.addStretch(1)
        sh.addLayout(corr_row, 0, 1)
         # V1 / V2 model selector — shown only when correction is available
        self.lbl_correct_ver = QLabel(self.tr("Correction Model:"))
        self.cmb_correct_ver = QComboBox()
        self.cmb_correct_ver.addItems(["V2 (latest)", "V1"])
        self.cmb_correct_ver.setToolTip(self.tr(
            "V2: Updated weights, improved correction quality (recommended).\n"
            "V1: Original correction model."
        ))
        sh.addWidget(self.lbl_correct_ver, 1, 0)
        sh.addWidget(self.cmb_correct_ver, 1, 1)

        self.lbl_sh_mode = QLabel(self.tr("Sharpening Mode:"))
        self.cmb_sh_mode = QComboBox()
        self.cmb_sh_mode.addItems(["Both", "Stellar Only", "Non-Stellar Only"])
        sh.addWidget(self.lbl_sh_mode, 2, 0)
        sh.addWidget(self.cmb_sh_mode, 2, 1)
 
        self.chk_sh_sep = QCheckBox(self.tr("Sharpen RGB channels separately"))
        self.chk_sh_sep.setToolTip(self.tr(
            "Run the mono sharpening model independently on R, G, and B instead of a shared color model.\n"
            "Use for difficult color data where channels need slightly different sharpening."
        ))
        sh.addWidget(self.chk_sh_sep, 3, 0, 1, 2)

        self.chk_auto_psf = QCheckBox(self.tr("Auto Detect PSF"))
        self.chk_auto_psf.setChecked(True)
        sh.addWidget(self.chk_auto_psf, 4, 0, 1, 2)

        self.lbl_psf = QLabel(self.tr("Non-Stellar PSF (1.0–8.0): {0:.1f}").format(3.0))
        self.sld_psf = QSlider(Qt.Orientation.Horizontal)
        self.sld_psf.setRange(10, 80)
        self.sld_psf.setValue(30)
        _add_inline_slider(sh, 5, self.lbl_psf, self.sld_psf)
 
        self.lbl_st_amt = QLabel(self.tr("Stellar Amount (0–1): {0:.2f}").format(0.5))
        self.sld_st_amt = QSlider(Qt.Orientation.Horizontal)
        self.sld_st_amt.setRange(0, 100)
        self.sld_st_amt.setValue(50)
        _add_inline_slider(sh, 6, self.lbl_st_amt, self.sld_st_amt)
 
        self.lbl_nst_amt = QLabel(self.tr("Non-Stellar Amount (0–1): {0:.2f}").format(0.5))
        self.sld_nst_amt = QSlider(Qt.Orientation.Horizontal)
        self.sld_nst_amt.setRange(0, 100)
        self.sld_nst_amt.setValue(50)
        _add_inline_slider(sh, 7, self.lbl_nst_amt, self.sld_nst_amt)

        self.chk_correct_conservative = QCheckBox(self.tr("Conservative White Compression (reduces tiling on high-contrast images)"))
        self.chk_correct_conservative.setToolTip(self.tr(
            "When enabled, uses stronger white point compression during aberration correction.\n"
            "Reduces tiling artifacts on images with extreme bright/dark contrast within a single tile.\n"
        ))
        sh.addWidget(self.chk_correct_conservative, 8, 0, 1, 2)

        sh.setColumnStretch(1, 1)
        left_col.addWidget(grp_sh)
 
        # ---------------- Denoise group ----------------
        grp_dn = QGroupBox(self.tr("Denoise"))
        dn = QGridLayout(grp_dn)
        dn.setContentsMargins(8, 8, 8, 8)
        dn.setHorizontalSpacing(8)
        dn.setVerticalSpacing(6)
 
        self.lbl_dn_lum = QLabel(self.tr("Luminance Denoise (0–1): {0:.2f}").format(0.5))
        self.sld_dn_lum = QSlider(Qt.Orientation.Horizontal)
        self.sld_dn_lum.setRange(0, 100)
        self.sld_dn_lum.setValue(50)
        _add_inline_slider(dn, 0, self.lbl_dn_lum, self.sld_dn_lum)
 
        self.lbl_dn_col = QLabel(self.tr("Color Denoise (0–1): {0:.2f}").format(0.5))
        self.sld_dn_col = QSlider(Qt.Orientation.Horizontal)
        self.sld_dn_col.setRange(0, 100)
        self.sld_dn_col.setValue(50)
        _add_inline_slider(dn, 1, self.lbl_dn_col, self.sld_dn_col)
 
        self.lbl_dn_mode = QLabel(self.tr("Denoise Mode:"))
        self.cmb_dn_mode = QComboBox()
        self.cmb_dn_mode.addItems(["full", "luminance"])
        dn.addWidget(self.lbl_dn_mode, 2, 0)
        dn.addWidget(self.cmb_dn_mode, 2, 1)

        self.chk_dn_sep = QCheckBox(self.tr("Denoise RGB channels separately"))
        dn.addWidget(self.chk_dn_sep, 3, 0, 1, 2)

        self.lbl_dn_model = QLabel(self.tr("Denoise Model:"))
        self.cmb_dn_model = QComboBox()
        self.cmb_dn_model.addItems(["Standard", "Walking Noise", "Lite (faster)"])
        self.cmb_dn_model.setToolTip(self.tr(
            "Standard: general-purpose AI4 denoise model.\n"
            "Walking Noise: specialist model for fixed-pattern drift streaks\n"
            "  (polar misalignment, no guiding). Also handles normal noise.\n"
            "Lite: smaller faster model, slightly less detail."
        ))
        dn.addWidget(self.lbl_dn_model, 4, 0)
        dn.addWidget(self.cmb_dn_model, 4, 1)

        self.chk_dn_compat = QCheckBox(self.tr("GPU Compatibility Mode (slower, safer)"))
        self.chk_dn_compat.setToolTip(self.tr(
            "Uses safer inference settings for Cosmic Clarity processing.\n"
            "Recommended if sharpen or denoise fails, hangs, or produces GPU errors on older or fragile hardware."
        ))
        dn.addWidget(self.chk_dn_compat, 5, 0, 1, 2)
 
        dn.setColumnStretch(1, 1)
        left_col.addWidget(grp_dn)
        left_col.addStretch(1)
 
        # ---------------- Super Resolution group ----------------
        grp_sr = QGroupBox(self.tr("Super Resolution"))
        sr = QGridLayout(grp_sr)
        sr.setContentsMargins(8, 8, 8, 8)
        sr.setHorizontalSpacing(8)
        sr.setVerticalSpacing(6)
        self.lbl_scale = QLabel(self.tr("Scale Factor:"))
        self.cmb_scale = QComboBox()
        self.cmb_scale.addItems(["2x", "3x", "4x"])
        sr.addWidget(self.lbl_scale, 0, 0)
        sr.addWidget(self.cmb_scale, 0, 1)
        sr.setColumnStretch(1, 1)
        right_col.addWidget(grp_sr)
 
        # ---------------- Advanced group ----------------
        grp_adv = QGroupBox(self.tr("Advanced"))
        adv = QGridLayout(grp_adv)
        adv.setContentsMargins(8, 8, 8, 8)
        adv.setHorizontalSpacing(8)
        adv.setVerticalSpacing(6)
 
        self.chk_temp_stretch = QCheckBox(self.tr("Temporary Stretch for AI (linear assist)"))
        self.chk_temp_stretch.setChecked(True)
        self.chk_temp_stretch.setToolTip(self.tr(
            "Optionally applies a temporary stretch before AI inference (then reverses it).\n"
            "Useful for very linear / low-signal data. If off, engines use automatic behavior."
        ))
        adv.addWidget(self.chk_temp_stretch, 0, 0, 1, 2)

        self.lbl_target_median = QLabel(self.tr("Target Median (0.01–0.50): {0:.2f}").format(0.25))
        self.sld_target_median = QSlider(Qt.Orientation.Horizontal)
        self.sld_target_median.setRange(1, 50)
        self.sld_target_median.setValue(25)
        _add_inline_slider(adv, 1, self.lbl_target_median, self.sld_target_median)
 
        self.chk_aberration_first = QCheckBox(self.tr("Run Aberration Remover first"))
        self.chk_aberration_first.setChecked(False)   # permanently disabled
        self.chk_aberration_first.setVisible(False)   # hidden — superseded by native correction
        adv.addWidget(self.chk_aberration_first, 2, 0, 1, 2)

        self.lbl_chunk = QLabel(self.tr("Chunk Size:"))
        self.cmb_chunk = QComboBox()
        self.cmb_chunk.addItems(["128", "192", "256", "320", "384", "512", "640", "768", "1024"])
        self.cmb_chunk.setCurrentText("256")
        adv.addWidget(self.lbl_chunk, 3, 0)
        adv.addWidget(self.cmb_chunk, 3, 1)
 
        self.lbl_ov = QLabel(self.tr("Overlap (% of tile):"))
        self.cmb_ov = QComboBox()
        self.cmb_ov.addItems([f"{c}%" for c in OVERLAP_PCT_CHOICES])
        self.cmb_ov.setCurrentText(f"{int(DEFAULT_OVERLAP_PCT)}%")
        self.cmb_ov.setToolTip(self.tr(
            "Tile overlap as a percentage of the chunk size.\n"
            "Higher overlap reduces edge seams but is slower — runtime climbs\n"
            "steeply as overlap grows, so it is capped at 50% of the tile.\n"
            "25% is a good default."
        ))
        adv.addWidget(self.lbl_ov, 4, 0)
        adv.addWidget(self.cmb_ov, 4, 1)
 
        adv.setColumnStretch(1, 1)
        right_col.addWidget(grp_adv)
        right_col.addStretch(1)
 
        body.addLayout(left_col, 1)
        body.addLayout(right_col, 1)
        outer.addLayout(body, 1)
 
        # ---------------- Buttons ----------------
        row = QHBoxLayout()
        self.btn_clear_cache = QPushButton(self.tr("Clear AI Cache"))
        self.btn_clear_cache.setToolTip(self.tr(
            "Clears cached AI models from RAM/VRAM.\n"
            "This makes the next run slower (models must reload).\n\n"
            "Use if you are low on memory or troubleshooting."
        ))
        self.btn_clear_cache.clicked.connect(self._clear_ai_cache_clicked)
        b_run = QPushButton(self.tr("Execute"))
        b_run.clicked.connect(self._run_main)
        self.btn_run = b_run
        b_close = QPushButton(self.tr("Close"))
        b_close.clicked.connect(self.reject)
        b_reset = make_reset_button(self)
        b_reset.clicked.connect(self._reset_to_factory)
        row.addWidget(self.btn_clear_cache)
        row.addStretch(1)
        row.addWidget(make_help_button("cosmic_clarity", self))
        row.addWidget(b_run)
        row.addWidget(b_reset)
        row.addWidget(b_close)
        outer.addLayout(row)

        # --- preset drag handle (grip), pinned lower-left ---
        try:
            from setiastro.saspro.shortcuts import PresetDragHandle
            try:
                from setiastro.saspro.resources import cosmic_path
                _grip_icon = QIcon(cosmic_path)
            except Exception:
                _grip_icon = QIcon()
            drag_row = QHBoxLayout()
            drag_row.setContentsMargins(0, 0, 0, 0)
            self.preset_drag_handle = PresetDragHandle(
                "cosmic_clarity", self.get_preset, icon=_grip_icon,
                tooltip=self.tr(
                    "Drag to the canvas to create a Cosmic Clarity shortcut with these settings.\n"
                    "Drop directly on an image to apply them headlessly."),
                parent=self,
            )
            drag_row.addWidget(self.preset_drag_handle)
            drag_row.addStretch(1)   # push grip hard to the LEFT edge
            outer.addLayout(drag_row)
        except Exception:
            pass
 
        # Factory state, before any saved setting loads. The correction model
        # defaults to what is installed (Step 3), so it is recorded there.
        self._factory = snapshot_controls(self, exclude=(self.cmb_correct_ver,))

        # ------------------------------------------------------------------
        # Step 1: load chunk/overlap (no save signals yet)
        # ------------------------------------------------------------------
        self._load_chunk_overlap_settings()
 
        # chunk/overlap have their own save keys — wire those now.
        # (Overlap is a % of the tile capped at 50%, so it is always valid
        #  relative to chunk size — no cross-enforcement needed.)
        self.cmb_chunk.currentTextChanged.connect(self._save_chunk_overlap_settings)
        self.cmb_ov.currentTextChanged.connect(self._save_chunk_overlap_settings)
 
        # ------------------------------------------------------------------
        # Step 2: load all other settings with ALL signals blocked
        # ------------------------------------------------------------------
        self._load_cc_ui_settings()
 
        # ------------------------------------------------------------------
        # Step 3: wire correct model availability FIRST so _mode_changed sees it
        # ------------------------------------------------------------------
        from setiastro.saspro.resources import get_resources
        _r = get_resources()
        _cp  = getattr(_r, "CC_C_PTH",  None)
        _cp2 = getattr(_r, "CC_C2_PTH", None)
        self._correct_available    = bool((_cp and os.path.exists(_cp)) or (_cp2 and os.path.exists(_cp2)))
        self._correct_v2_available = bool(_cp2 and os.path.exists(_cp2))

        # Disable V2 option if not installed
        v2_idx = self.cmb_correct_ver.findText("V2 (latest)")
        if v2_idx >= 0:
            item_flags = self.cmb_correct_ver.model().item(v2_idx).flags()
            from PyQt6.QtCore import Qt as _Qt
            from PyQt6.QtGui import QStandardItem
            item = self.cmb_correct_ver.model().item(v2_idx)
            if self._correct_v2_available:
                item.setFlags(item.flags() | _Qt.ItemFlag.ItemIsEnabled | _Qt.ItemFlag.ItemIsSelectable)
                item.setToolTip(self.tr("V2 correction model — installed"))
            else:
                item.setFlags(item.flags() & ~_Qt.ItemFlag.ItemIsEnabled & ~_Qt.ItemFlag.ItemIsSelectable)
                item.setToolTip(self.tr("V2 correction model — not installed (click Download/Update Models in Preferences)"))

        # Default selection
        if not self._correct_v2_available:
            self.cmb_correct_ver.setCurrentText("V1")
        else:
            self.cmb_correct_ver.setCurrentText("V2 (latest)")
        self._factory_correct_ver = self.cmb_correct_ver.currentText()

        # ------------------------------------------------------------------
        # Step 4: now that _correct_available is set, apply dependent UI state
        # ------------------------------------------------------------------
        self._mode_changed()
        self._update_denoise_dependent_ui()

        # Sync temp-stretch enable state without saving
        self.lbl_target_median.setEnabled(self.chk_temp_stretch.isChecked())
        self.sld_target_median.setEnabled(self.chk_temp_stretch.isChecked())

        # Sync PSF label
        self._psf_label()
        self._on_st_amt(self.sld_st_amt.value())
        self._on_nst_amt(self.sld_nst_amt.value())
        self.lbl_dn_lum.setText(self.tr("Luminance Denoise (0–1): {0:.2f}").format(self.sld_dn_lum.value()/100))
        self.lbl_dn_col.setText(self.tr("Color Denoise (0–1): {0:.2f}").format(self.sld_dn_col.value()/100))
        tm = self.sld_target_median.value()
        self.lbl_target_median.setText(self.tr("Target Median (0.01–0.50): {0:.2f}").format(tm/100))
 
        # ------------------------------------------------------------------
        # Step 5: wire headless flag
        # ------------------------------------------------------------------
        self._headless = bool(headless)
        if self._headless:
            try:
                self.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
 
        self._wait = None
 
        # ------------------------------------------------------------------
        # Step 6: connect ALL signals that trigger saves — LAST
        # ------------------------------------------------------------------
        # Label-update only (no save)
        self.sld_psf.valueChanged.connect(self._psf_label)
        self.sld_st_amt.valueChanged.connect(self._on_st_amt)
        self.sld_nst_amt.valueChanged.connect(self._on_nst_amt)
        self.sld_dn_lum.valueChanged.connect(lambda v: self.lbl_dn_lum.setText(self.tr("Luminance Denoise (0–1): {0:.2f}").format(v/100)))
        self.sld_dn_col.valueChanged.connect(lambda v: self.lbl_dn_col.setText(self.tr("Color Denoise (0–1): {0:.2f}").format(v/100)))
        self.sld_target_median.valueChanged.connect(self._on_target_median)
        self.chk_temp_stretch.toggled.connect(self._temp_stretch_changed)
        self.chk_dn_sep.toggled.connect(self._dn_sep_changed)
 
        # Mode visibility
        self.cmb_mode.currentIndexChanged.connect(self._mode_changed)
        self.rb_correct_only.toggled.connect(self._mode_changed)
        self.rb_correct_sharpen.toggled.connect(self._mode_changed)
        self.rb_sharpen_only.toggled.connect(self._mode_changed)
        self.cmb_dn_model.currentIndexChanged.connect(self._dn_model_changed)
 
        # Save on any user change
        self.sld_st_amt.valueChanged.connect(self._save_cc_ui_settings)
        self.sld_nst_amt.valueChanged.connect(self._save_cc_ui_settings)
        self.sld_psf.valueChanged.connect(self._save_cc_ui_settings)
        self.sld_dn_lum.valueChanged.connect(self._save_cc_ui_settings)
        self.sld_dn_col.valueChanged.connect(self._save_cc_ui_settings)
        self.sld_target_median.valueChanged.connect(self._save_cc_ui_settings)
        self.chk_auto_psf.toggled.connect(self._save_cc_ui_settings)
        self.chk_sh_sep.toggled.connect(self._save_cc_ui_settings)
        self.chk_dn_sep.toggled.connect(self._save_cc_ui_settings)
        self.chk_dn_compat.toggled.connect(self._save_cc_ui_settings)
        self.chk_aberration_first.toggled.connect(self._save_cc_ui_settings)
        self.cmb_correct_ver.currentTextChanged.connect(self._save_cc_ui_settings)
        self.chk_temp_stretch.toggled.connect(self._save_cc_ui_settings)
        self.cmb_sh_mode.currentTextChanged.connect(self._save_cc_ui_settings)
        self.cmb_dn_mode.currentTextChanged.connect(self._save_cc_ui_settings)
        self.cmb_mode.currentIndexChanged.connect(self._save_cc_ui_settings)
        self.cmb_gpu.currentIndexChanged.connect(self._save_cc_ui_settings)
        self.cmb_target.currentIndexChanged.connect(self._save_cc_ui_settings)
        self.cmb_scale.currentTextChanged.connect(self._save_cc_ui_settings)
        self.rb_correct_only.toggled.connect(self._save_cc_ui_settings)
        self.rb_correct_sharpen.toggled.connect(self._save_cc_ui_settings)
        self.rb_sharpen_only.toggled.connect(self._save_cc_ui_settings)
        self.chk_correct_conservative.toggled.connect(self._save_cc_ui_settings)
 
        self.setMinimumSize(760, 520)
        self._restore_cc_geometry()
        QTimer.singleShot(0, self._deferred_resize)

        # Warm the engine while settings are chosen (cosmicclarity_engines.warmup):
        # once the window is up, and again when a choice changes which models or
        # shapes the run will use. Debounced, so flicking a combo queues one.
        self._warm_timer = QTimer(self)
        self._warm_timer.setSingleShot(True)
        self._warm_timer.setInterval(400)
        self._warm_timer.timeout.connect(self._start_warmup)
        for _sig in (self.cmb_mode.currentIndexChanged, self.cmb_gpu.currentIndexChanged,
                     self.cmb_sh_mode.currentTextChanged, self.cmb_dn_mode.currentTextChanged):
            _sig.connect(lambda *_: self._warm_timer.start())
        self._warm_timer.start()

    def _reset_to_factory(self):
        # Every control saves itself on change, so the restore is also saved.
        if _cc_thread_running(getattr(self, "_engine_thread", None)):
            return
        restore_controls(self._factory)
        self.cmb_correct_ver.setCurrentText(self._factory_correct_ver)

    def _start_warmup(self):
        try:
            img = getattr(getattr(self, "doc", None), "image", None)
            if img is None or not self.isVisible():
                return
            from setiastro.saspro.cosmicclarity_engines.warmup import warm_cosmic_clarity
            warm_cosmic_clarity(self.build_preset_from_ui(), img)
        except Exception as e:
            print(f"[CC warm-up] not started ({type(e).__name__}: {e})")

    def _dn_model_changed(self, idx: int):
            self._save_cc_ui_settings()

    def _clear_ai_cache_clicked(self):
        # Don’t allow cache clearing mid-run (can invalidate active model refs)
        if self._engine_thread is not None and self._engine_thread.isRunning():
            QMessageBox.information(self, "Cosmic Clarity", self.tr("Please wait for the current run to finish before clearing cache."))
            return

        # Optional: “aggressive” flush
        aggressive = True  # set False if you want gentler behavior

        try:
            # Denoise cache
            try:
                from setiastro.saspro.cosmicclarity_engines.denoise_engine import clear_denoise_models_cache
                clear_denoise_models_cache(aggressive=aggressive, status_cb=print)
            except Exception as e:
                print(f"[CC] clear_denoise_models_cache failed: {type(e).__name__}: {e}")

            # Sharpen cache (if exists)
            try:
                from setiastro.saspro.cosmicclarity_engines.sharpen_engine import clear_sharpen_models_cache
                clear_sharpen_models_cache(aggressive=aggressive, status_cb=print)
            except Exception:
                pass

            # Superres cache (if you have one)
            # try:
            #     from setiastro.saspro.cosmicclarity_engines.superres_engine import clear_superres_models_cache
            #     clear_superres_models_cache(aggressive=aggressive, status_cb=print)
            # except Exception:
            #     pass

            try:        # models are gone: let the next dialog warm them up again
                from setiastro.saspro.cosmicclarity_engines.warmup import forget
                forget()
            except Exception:
                pass

            QMessageBox.information(
                self,
                "Cosmic Clarity",
                self.tr("AI cache cleared.\n\nNext AI run will take longer because models must reload.")
            )

        except Exception as e:
            QMessageBox.warning(self, "Cosmic Clarity", self.tr("Failed to clear AI cache:\n{0}").format(e))

    def _on_target_median(self, v: int):
        tm = max(0.01, min(0.50, v / 100.0))
        self.lbl_target_median.setText(self.tr("Target Median (0.01–0.50): {0:.2f}").format(tm))
        self._save_cc_ui_settings()

    def _temp_stretch_changed(self, checked: bool):
        self.lbl_target_median.setEnabled(bool(checked))
        self.sld_target_median.setEnabled(bool(checked))
        self._save_cc_ui_settings()

 
    def _load_cc_ui_settings(self):
        """Load all persisted settings, blocking all widget signals to prevent save-during-load."""
        s = QSettings()
 
        # Collect every widget that might fire a signal during setValue/setChecked
        all_widgets = [
            self.cmb_dn_model, self.chk_dn_compat, self.chk_temp_stretch,
            self.sld_target_median, self.chk_aberration_first,
            self.sld_st_amt, self.sld_nst_amt, self.sld_psf,
            self.chk_auto_psf, self.chk_sh_sep, self.cmb_sh_mode,
            self.sld_dn_lum, self.sld_dn_col, self.cmb_dn_mode, self.chk_dn_sep,
            self.cmb_mode, self.cmb_gpu, self.cmb_target, self.cmb_scale,
            self.rb_correct_only, self.rb_correct_sharpen, self.rb_sharpen_only,self.cmb_correct_ver,
            self.chk_correct_conservative,
        ]
 
        # Block all signals before touching any widget
        for w in all_widgets:
            w.blockSignals(True)
 
        try:
            # Denoise model + compat
            dn_model = s.value("cc/denoise_model", "Standard")
            idx = self.cmb_dn_model.findText(str(dn_model))
            self.cmb_dn_model.setCurrentIndex(max(0, idx))
 
            compat = s.value("cc/gpu_compat_mode", None)
            if compat is None:
                compat = s.value("cc/denoise_compat_mode", False, type=bool)
            else:
                compat = s.value("cc/gpu_compat_mode", False, type=bool)
            self.chk_dn_compat.setChecked(bool(compat))
 
            # Temp stretch + target median
            self.chk_temp_stretch.setChecked(s.value("cc/temp_stretch", True, type=bool))
            try:
                tm = max(1, min(50, int(float(s.value("cc/target_median", 0.25)) * 100)))
            except Exception:
                tm = 25
            self.sld_target_median.setValue(tm)
 
            self.chk_aberration_first.setChecked(False) 
 
            # Sharpen sliders
            try:
                self.sld_st_amt.setValue(int(s.value("cc/stellar_amount", 50)))
            except Exception:
                self.sld_st_amt.setValue(50)
            try:
                self.sld_nst_amt.setValue(int(s.value("cc/nonstellar_amount", 50)))
            except Exception:
                self.sld_nst_amt.setValue(50)
            try:
                self.sld_psf.setValue(int(s.value("cc/nonstellar_psf", 30)))
            except Exception:
                self.sld_psf.setValue(30)
 
            self.chk_auto_psf.setChecked(s.value("cc/auto_psf", True, type=bool))
            self.chk_sh_sep.setChecked(s.value("cc/sharpen_sep", False, type=bool))
            corr_ver = s.value("cc/correct_model_version", "V2 (latest)", type=str)
            idx = self.cmb_correct_ver.findText(corr_ver)
            self.cmb_correct_ver.setCurrentIndex(max(0, idx))
            self.chk_correct_conservative.setChecked(s.value("cc/correct_conservative", False, type=bool))
            sh_mode = str(s.value("cc/sharpening_mode", "Both"))
            if self.cmb_sh_mode.findText(sh_mode) >= 0:
                self.cmb_sh_mode.setCurrentText(sh_mode)
 
            # Denoise sliders
            try:
                self.sld_dn_lum.setValue(int(s.value("cc/denoise_luma", 50)))
            except Exception:
                self.sld_dn_lum.setValue(50)
            try:
                self.sld_dn_col.setValue(int(s.value("cc/denoise_color", 50)))
            except Exception:
                self.sld_dn_col.setValue(50)
 
            dn_mode = str(s.value("cc/denoise_mode", "full"))
            if self.cmb_dn_mode.findText(dn_mode) >= 0:
                self.cmb_dn_mode.setCurrentText(dn_mode)
 
            self.chk_dn_sep.setChecked(s.value("cc/denoise_sep", False, type=bool))
 
            # General combos
            try:
                self.cmb_mode.setCurrentIndex(int(s.value("cc/mode", 0)))
            except Exception:
                self.cmb_mode.setCurrentIndex(0)
            try:
                self.cmb_gpu.setCurrentIndex(int(s.value("cc/gpu", 0)))
            except Exception:
                self.cmb_gpu.setCurrentIndex(0)
            try:
                self.cmb_target.setCurrentIndex(int(s.value("cc/target", 0)))
            except Exception:
                self.cmb_target.setCurrentIndex(0)
 
            scale = str(s.value("cc/scale", "2x"))
            if self.cmb_scale.findText(scale) >= 0:
                self.cmb_scale.setCurrentText(scale)
 
            # Stellar correct mode radio
            scm = str(s.value("cc/stellar_correct_mode", "correct_sharpen"))
            if scm == "correct_only":
                self.rb_correct_only.setChecked(True)
            elif scm == "sharpen_only":
                self.rb_sharpen_only.setChecked(True)
            else:
                self.rb_correct_sharpen.setChecked(True)
 
        finally:
            # Always unblock — even if something raised
            for w in all_widgets:
                w.blockSignals(False)


    def _save_cc_ui_settings(self):
        s = QSettings()
        s.setValue("cc/denoise_model", self.cmb_dn_model.currentText())
        s.setValue("cc/gpu_compat_mode", self.chk_dn_compat.isChecked())
        s.setValue("cc/denoise_compat_mode", self.chk_dn_compat.isChecked())
        s.setValue("cc/temp_stretch", self.chk_temp_stretch.isChecked())
        s.setValue("cc/target_median", self.sld_target_median.value() / 100.0)
        s.setValue("cc/aberration_first", self.chk_aberration_first.isChecked())
        s.setValue("cc/stellar_amount", self.sld_st_amt.value())
        s.setValue("cc/correct_model_version", self.cmb_correct_ver.currentText())
        s.setValue("cc/correct_conservative", self.chk_correct_conservative.isChecked())
        s.setValue("cc/nonstellar_amount", self.sld_nst_amt.value())
        s.setValue("cc/nonstellar_psf", self.sld_psf.value())
        s.setValue("cc/auto_psf", self.chk_auto_psf.isChecked())
        s.setValue("cc/sharpen_sep", self.chk_sh_sep.isChecked())
        s.setValue("cc/sharpening_mode", self.cmb_sh_mode.currentText())
        s.setValue("cc/denoise_luma", self.sld_dn_lum.value())
        s.setValue("cc/denoise_color", self.sld_dn_col.value())
        s.setValue("cc/denoise_mode", self.cmb_dn_mode.currentText())
        s.setValue("cc/denoise_sep", self.chk_dn_sep.isChecked())
        s.setValue("cc/mode", self.cmb_mode.currentIndex())
        s.setValue("cc/gpu", self.cmb_gpu.currentIndex())
        s.setValue("cc/target", self.cmb_target.currentIndex())
        s.setValue("cc/scale", self.cmb_scale.currentText())
        s.setValue("cc/stellar_correct_mode",
            "correct_only"    if self.rb_correct_only.isChecked()    else
            "correct_sharpen" if self.rb_correct_sharpen.isChecked() else
            "sharpen_only"
        )



    def _dn_sep_changed(self, checked: bool):
        """Separate-channels means mono model per RGB channel, so disable chroma controls."""
        self._update_denoise_dependent_ui()

    def _update_denoise_dependent_ui(self):
        # Only relevant when Denoise controls are currently in play (Denoise or Both mode)
        idx = self.cmb_mode.currentIndex()
        denoise_active = idx in (1, 2)  # 1 Denoise, 2 Both
        if not denoise_active:
            return

        sep = bool(self.chk_dn_sep.isChecked())

        # Option A: DISABLE (recommended: user sees it exists but is inactive)
        self.lbl_dn_col.setEnabled(not sep)
        self.sld_dn_col.setEnabled(not sep)

        self.lbl_dn_mode.setEnabled(not sep)
        self.cmb_dn_mode.setEnabled(not sep)

        # Optional: tooltip explanation when disabled
        tip = self.tr("Disabled because 'Process RGB channels separately' uses the mono model on R/G/B "
            "and does not use the color model.")
        if sep:
            self.lbl_dn_col.setToolTip(tip)
            self.sld_dn_col.setToolTip(tip)
            self.lbl_dn_mode.setToolTip(tip)
            self.cmb_dn_mode.setToolTip(tip)
        else:
            self.lbl_dn_col.setToolTip("")
            self.sld_dn_col.setToolTip("")
            self.lbl_dn_mode.setToolTip("")
            self.cmb_dn_mode.setToolTip("")


    def _set_overlap_pct(self, pct):
        """Select the overlap combo item nearest to the given percentage."""
        self.cmb_ov.setCurrentText(f"{nearest_overlap_pct(pct)}%")

    def _load_chunk_overlap_settings(self):
        s = QSettings()
        chunk = s.value("cc/chunk_size", "256")
        if self.cmb_chunk.findText(str(chunk)) >= 0:
            self.cmb_chunk.setCurrentText(str(chunk))

        # Overlap is now stored as a percentage of the tile (0-50).
        pct = s.value("cc/overlap_pct", None)
        if pct is None:
            # Migrate a legacy pixel overlap -> percentage using the saved chunk.
            try:
                legacy_px = float(s.value("cc/overlap", 64))
                ck = int(self.cmb_chunk.currentText())
                pct = round(100.0 * legacy_px / max(1, ck))
            except Exception:
                pct = DEFAULT_OVERLAP_PCT
        self._set_overlap_pct(pct)

    def _save_chunk_overlap_settings(self):
        s = QSettings()
        s.setValue("cc/chunk_size", self.cmb_chunk.currentText())
        s.setValue("cc/overlap_pct", int(parse_overlap_pct(self.cmb_ov.currentText())))


    def _ensure_models_installed_or_bail(self) -> bool:
        ok, detail = _cc_models_installed("core")
        if ok:
            return True

        msg = self.tr(
            "Cosmic Clarity AI models are not installed (or are incomplete).\n\n"
            "To install them:\n"
            "  Preferences → AI Models → Download/Update Models\n"
        )
        if detail:
            msg += self.tr("\nDetails: {0}\n").format(detail)

        QMessageBox.warning(self, "Cosmic Clarity", msg)
        QTimer.singleShot(0, self.reject)
        return False



    # ----- UI helpers -----
    def _headless_guard_active(self) -> bool:
        # A headless run in this process: the preset runner flags the main
        # window, this panel's parent, and clears it in a finally. Not the
        # QSettings copy (cc/headless_in_progress): that is on disk, so after a
        # SASpro that closed during a run, this panel closed itself unseen
        # whenever it was opened from a shortcut.
        try:
            p = self.parent()
            return bool(p and (getattr(p, "_cosmicclarity_guard", False)
                               or getattr(p, "_cosmicclarity_headless_running", False)))
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


    def _on_st_amt(self, v: int): self.lbl_st_amt.setText(self.tr("Stellar Amount (0–1): {0:.2f}").format(v/100))
    def _on_nst_amt(self, v: int): self.lbl_nst_amt.setText(self.tr("Non-Stellar Amount (0–1): {0:.2f}").format(v/100))

    def _psf_label(self):
        self.lbl_psf.setText(self.tr("Non-Stellar PSF (1.0–8.0): {0:.1f}").format(self.sld_psf.value()/10))

    def _mode_changed(self):
        idx = self.cmb_mode.currentIndex()  # 0 Sharpen, 1 Denoise, 2 Both, 3 Super-Res
        show_sh = idx in (0, 2)
        show_dn = idx in (1, 2)
        show_sr = idx == 3
        show_gpu = not show_sr
 
        # Correction radio buttons only visible when Sharpen group is shown AND model is installed
        _ca = getattr(self, "_correct_available", False)
        _cv2 = getattr(self, "_correct_v2_available", False)
        corr_active = show_sh and _ca
        for w in (self.lbl_stellar_correct,
                  self.rb_correct_only, self.rb_correct_sharpen, self.rb_sharpen_only):
            w.setVisible(corr_active)

        # Show version selector only when correction is active and at least one version installed
        self.lbl_correct_ver.setVisible(corr_active)
        self.cmb_correct_ver.setVisible(corr_active)
        self.chk_correct_conservative.setVisible(corr_active)
 
        # Sharpening sub-controls hidden when Correct Only is selected
        _ca = getattr(self, "_correct_available", False)
        sharpen_active = show_sh and not (_ca and self.rb_correct_only.isChecked())
        for w in (self.lbl_sh_mode, self.cmb_sh_mode, self.chk_sh_sep,
                  self.chk_auto_psf, self.lbl_psf, self.sld_psf,
                  self.lbl_st_amt, self.sld_st_amt,
                  self.lbl_nst_amt, self.sld_nst_amt):
            w.setVisible(sharpen_active)
 
        for w in (self.lbl_dn_lum, self.sld_dn_lum,
                  self.lbl_dn_col, self.sld_dn_col,
                  self.lbl_dn_mode, self.cmb_dn_mode,
                  self.chk_dn_sep, self.cmb_dn_model, self.lbl_dn_model):
            w.setVisible(show_dn)
 
        self.chk_dn_compat.setVisible(idx in (0, 1, 2))
 
        for w in (self.lbl_scale, self.cmb_scale):
            w.setVisible(show_sr)
 
        self.lbl_gpu.setVisible(show_gpu)
        self.cmb_gpu.setVisible(show_gpu)
 
        self._update_denoise_dependent_ui()
        self._relayout_after_visibility_change()

    def _relayout_after_visibility_change(self):
        from PyQt6.QtCore import QTimer
        def _do():
            try:
                self.layout().activate()
                for child in self.findChildren(QGroupBox):
                    child.layout().activate()
                    child.updateGeometry()
                self.adjustSize()
                # if we have a saved geometry, don't let adjustSize shrink below it
                s = QSettings()
                g = s.value("cc/dialog_geometry")
                if g:
                    self.restoreGeometry(g)
            except Exception:
                pass
        QTimer.singleShot(0, _do)

    # ----- Validation -----
    def _validate_root(self) -> bool:
        return True

    # ----- Execution -----
    def _run_main(self):
        # One run at a time: a second Execute used to replace (and orphan) the first worker.
        if _cc_thread_running(getattr(self, "_engine_thread", None)):
            if self._wait is not None:
                self._wait.raise_()
                self._wait.activateWindow()
            return
        # --- Basic safety: make sure we have an image ---
        try:
            img = np.asarray(getattr(self.doc, "image", None))
            if img is None or img.size == 0:
                QMessageBox.warning(self, "Cosmic Clarity", self.tr("No image loaded in the active view."))
                return
        except Exception:
            QMessageBox.warning(self, "Cosmic Clarity", self.tr("No image loaded in the active view."))
            return

        # --- Register this run as "last action" for replay (same as you had) ---
        try:
            main = self.parent_ref or self.parent()
            if main is not None:
                preset = self.build_preset_from_ui()
                if hasattr(main, "_remember_last_headless_command"):
                    main._remember_last_headless_command(
                        "cosmic_clarity", preset,
                        description=f"Cosmic Clarity ({preset.get('mode', 'sharpen')})"
                    )
                else:
                    main._last_headless_command = {
                        "command_id": "cosmic_clarity",
                        "preset": preset,
                    }
                    if hasattr(main, "_update_replay_button"):
                        main._update_replay_button()
        except Exception:
            pass

        # --- Snapshot UI preset ---
        # --- Snapshot UI preset ---
        preset = self.build_preset_from_ui()
        mode = str(preset.get("mode", "sharpen")).lower()
        self._run_preset = dict(preset)     # recorded with the result, whatever the UI shows by then

        # --- Walking noise model availability check ---
        if preset.get("denoise_walking", False) and mode in ("denoise", "both"):
            try:
                # Not through get_resources(): its lookup raises when the file is
                # missing, which this try would swallow.
                from setiastro.saspro.model_manager import optional_model_path
                missing = [f for f in ("deep_denoise_mono_AI4_1w.pth",
                                       "deep_denoise_color_AI4_1w.pth")
                           if optional_model_path(f) is None]
                if missing:
                    from setiastro.saspro.setup_center import offer_setup
                    offer_setup(self, "models", self.tr("Walking Noise Model Not Installed"),
                                self.tr("The Walking Noise denoise model is not installed yet."),
                                "\n".join(f"  • {m}" for m in missing))
                    return
            except Exception:
                pass

        arr = np.asarray(self.doc.image, dtype=np.float32)

        # Track mono BEFORE expanding to 3-channel for the engine
        orig_was_mono = (arr.ndim == 2) or (arr.ndim == 3 and arr.shape[2] == 1)

        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[..., :3]
        else:
            QMessageBox.critical(self, "Cosmic Clarity", self.tr("Unsupported image shape: {0}").format(arr.shape))
            return

        arr = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)

        # Stash for use in result handler
        self._orig_was_mono = orig_was_mono
        self._run_cancel_requested = False

        # Snapshot the active mask now. The worker blends with the mask as it was
        # at Execute, so editing the mask mid-run cannot change this run's result.
        mask, mask_id = None, None
        try:
            mask_id = self._active_mask_id()
            layer = getattr(self.doc, "masks", {}).get(mask_id) if mask_id else None
            # doc.add_mask() stores a bare float32 array; mask_mixin stores a layer with .data
            data = layer if isinstance(layer, np.ndarray) else getattr(layer, "data", None)
            if data is not None and np.asarray(data).size:
                mask = np.array(data, dtype=np.float32, copy=True)
        except Exception:
            mask = None
        self._run_mask_id = mask_id if mask is not None else None
        if mode == "superres":
            self._run_mask_id = None    # the worker does not use a mask for super-res

        # --- Show wait/progress UI ---
        title = {
            "both": self.tr("Cosmic Clarity – Sharpen + Denoise"),
            "superres": self.tr("Cosmic Clarity – Super Resolution"),
            "sharpen": self.tr("Cosmic Clarity – Sharpen"),
            "denoise": self.tr("Cosmic Clarity – Denoise"),
        }.get(mode) or ("Cosmic Clarity – " + mode.title())
        self._wait = WaitDialog(title, self)
        self._wait.append_output("Starting…")
        self._wait.set_stage(self.tr("Starting…"))
        self._wait.cancelled.connect(self._cancel_all_engine)
        self._wait.show()

        # --- Run the in-process engine worker ---
        # No Qt parent: _start_cc_thread owns the thread until it finishes, so this
        # dialog can close, or be deleted, mid-run without destroying a running
        # QThread. The slots are bound methods, which Qt disconnects if the dialog
        # goes first (the old lambdas would have called into a deleted dialog).
        thr = CosmicClarityEngineWorker(
            arr, preset, parent=None,
            mask=mask, collapse_to_mono=orig_was_mono,
        )
        thr.progress.connect(self._on_engine_progress)
        thr.log.connect(self._on_engine_log)
        thr.stage.connect(self._on_engine_stage)
        thr.error.connect(self._on_engine_error)
        thr.cancelled.connect(self._on_engine_cancelled)
        thr.result.connect(self._on_engine_result)
        thr.finished.connect(self._engine_cleanup_and_maybe_close)
        self._engine_thread = thr

        # No cache clearing and no second Execute while this run is active.
        self._set_run_controls_enabled(False)
        _start_cc_thread(thr)

    def _set_run_controls_enabled(self, enabled: bool):
        for name in ("btn_clear_cache", "btn_run"):
            w = getattr(self, name, None)
            if w is not None:
                try:
                    w.setEnabled(bool(enabled))
                except RuntimeError:
                    pass

    def _on_engine_progress(self, pct: int):
        if self._wait is not None:
            self._wait.set_progress(int(pct))

    def _on_engine_log(self, line: str):
        if self._wait is not None:
            self._wait.append_output(str(line))

    def _on_engine_stage(self, text: str):
        if self._wait is not None:
            self._wait.set_stage(str(text))

    def _cancel_all_engine(self):
        self._run_cancel_requested = True
        thr = getattr(self, "_engine_thread", None)
        if thr is not None:
            try:
                thr.cancel()                # checked at every tile batch
                thr.requestInterruption()   # Qt-native interruption request
            except RuntimeError:
                pass
        # The wait window stays up, showing "Cancelling…", until the worker stops.
        if self._wait is not None:
            self._wait.mark_cancelling()

    def _close_wait(self):
        if self._wait is not None:
            try:
                self._wait.finish()
            except RuntimeError:
                pass
            self._wait = None

    def _on_engine_cancelled(self):
        self._close_wait()
        if self._headless:
            self._headless_error = "Cancelled."
            # reject() defers until the worker has stopped, then closes.
            self.reject()

    def _on_engine_error(self, msg: str):
        self._close_wait()

        text = str(msg)
        self._headless_error = text
        # A headless dialog is invisible (WA_DontShowOnScreen): parent the
        # message to the main window so it is actually seen.
        host = self
        if self._headless:
            host = getattr(self, "parent_ref", None) or self.parent() or None

        # Runtime-diagnosed acceleration problems: show the message runtime_torch
        # composed, not a fixed generic one. It already carries the missing-package
        # list AND, when the in-process fallback tripped over a real error (typical
        # cases: a torch and torchvision built against different releases surface
        # as ``operator torchvision::nms does not exist``; a HIP/CUDA library the
        # installed torch expects but the driver does not really support surfaces
        # as an ``undefined symbol`` or a load failure), the underlying error too.
        # That detail is the searchable string a user actually needs. The old
        # handler swallowed it and told everyone to click Install/Repair, which
        # does not help when the runtime is installed-but-broken.
        if "GPU acceleration runtime" in text or "Hardware acceleration runtime" in text:
            from setiastro.saspro.setup_center import offer_setup
            offer_setup(host, "accel", "Cosmic Clarity",
                        self.tr("Hardware acceleration is not usable for this run."), text)
        else:
            QMessageBox.critical(host, "Cosmic Clarity", text)

        if self._headless:
            # The headless runner waits on finished(); nothing else closes an
            # invisible dialog. reject() defers until the worker has stopped.
            self.reject()

    def _on_engine_result(self, out_arr: np.ndarray, step_title: str):
        self._close_wait()
        if getattr(self, "_run_cancel_requested", False):
            # Cancel was pressed after the engine's last batch: cancel still wins.
            self._on_engine_cancelled()
            return
        if getattr(self, "doc", None) is None:
            return

        # The preset this run used, snapshotted at Execute.
        preset = dict(getattr(self, "_run_preset", None) or {})
        if not preset:
            try:
                preset = self.build_preset_from_ui()
            except Exception:
                preset = {}

        # The mono collapse and the mask blend already ran in the worker.

        create_new = (self.cmb_target.currentIndex() == 1)
        if create_new:
            ok = self._spawn_new_doc_from_numpy(out_arr, step_title)
            if not ok:
                self._apply_to_active(out_arr, step_title, preset=preset)
        else:
            self._apply_to_active(out_arr, step_title, preset=preset)

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

    def _active_mask_id(self) -> str | None:
        return getattr(self.doc, "active_mask_id", None) or None

    def _apply_to_active(self, arr: np.ndarray, step_title: str, preset: dict | None = None):
        mid = getattr(self, "_run_mask_id", None)     # the mask this run was blended with

        if preset is None:
            preset = {}

        meta = {
            "step_name": step_title,
            "command_id": "cosmic_clarity",
            "preset": preset,
            "masked": bool(mid),
            "mask_id": mid,
            "mask_blend": "m*out+(1-m)*src",
        }
        if hasattr(self.doc, "apply_edit"):
            self.doc.apply_edit(arr, meta, step_name=step_title)
        elif hasattr(self.doc, "set_image"):
            self.doc.set_image(arr, step_name=step_title)
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
            QMessageBox.critical(self, "Cosmic Clarity", self.tr("Failed to create new view:\n{0}").format(e))
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
        self.sld_st_amt.setValue(int(max(0, min(100, round(float(p.get("stellar_amount", 0.5)) * 100)))))
        self.sld_nst_amt.setValue(int(max(0, min(100, round(float(p.get("nonstellar_amount",0.5))*100)))))
        self.chk_correct_conservative.setChecked(bool(p.get("correct_conservative", False)))
        # NEW: allow presets to opt into per-channel sharpen (still defaults off without a preset)
        self.chk_sh_sep.setChecked(bool(p.get("sharpen_channels_separately", False)))
        scm = p.get("stellar_correct_mode", "sharpen_only")
        self.rb_correct_only.setChecked(scm == "correct_only")
        self.rb_correct_sharpen.setChecked(scm == "correct_sharpen")
        self.rb_sharpen_only.setChecked(scm not in ("correct_only", "correct_sharpen"))
        self._mode_changed()   # refresh sub-control visibility
        corr_ver = p.get("correct_model_version", "V2 (latest)")
        idx = self.cmb_correct_ver.findText(corr_ver)
        if idx >= 0:
            self.cmb_correct_ver.setCurrentIndex(idx)
        if "chunk_size" in p:
            self.cmb_chunk.setCurrentText(str(int(p["chunk_size"])))
        if p.get("overlap_pct") is not None:
            self._set_overlap_pct(p["overlap_pct"])
        elif "overlap" in p:
            # Legacy pixel overlap -> percentage of this preset's chunk size.
            try:
                ck = int(p.get("chunk_size", self.cmb_chunk.currentText()))
                self._set_overlap_pct(round(100.0 * float(p["overlap"]) / max(1, ck)))
            except Exception:
                self._set_overlap_pct(DEFAULT_OVERLAP_PCT)
        # Denoise
        self.sld_dn_lum.setValue(int(max(0, min(100, round(float(p.get("denoise_luma",0.5))*100)))))
        self.sld_dn_col.setValue(int(max(0, min(100, round(float(p.get("denoise_color",0.5))*100)))))
        # Denoise model variant
        if p.get("denoise_walking", False):
            self.cmb_dn_model.setCurrentText("Walking Noise")
        elif p.get("denoise_lite", False):
            self.cmb_dn_model.setCurrentText("Lite (faster)")
        else:
            self.cmb_dn_model.setCurrentText("Standard")
        self.chk_dn_sep.setChecked(bool(p.get("separate_channels", False)))

        self.chk_dn_compat.setChecked(bool(
            p.get("compat_mode", p.get("denoise_compat_mode", False))
        ))
        # NEW: temp stretch
        self.chk_temp_stretch.setChecked(bool(p.get("temp_stretch", False)))
        self.chk_aberration_first.setChecked(bool(p.get("aberration_first", False)))
        try:
            tm = float(p.get("target_median", 0.25))
        except Exception:
            tm = 0.25
        tm = max(0.01, min(0.50, tm))
        self.sld_target_median.setValue(int(round(tm * 100)))
        self._temp_stretch_changed(self.chk_temp_stretch.isChecked())
        self._update_denoise_dependent_ui()
        # Super-Res
        self.cmb_scale.setCurrentText(str(int(p.get("scale",2))))

    def build_preset_from_ui(self) -> dict:
        """Snapshot current UI state into a preset dict usable by headless runner / replay."""
        idx = self.cmb_mode.currentIndex()  # 0 Sharpen, 1 Denoise, 2 Both, 3 Super-Res
        mode = {0: "sharpen", 1: "denoise", 2: "both", 3: "superres"}.get(idx, "sharpen")

        compat = self.chk_dn_compat.isChecked()

        preset: dict = {
            "mode": mode,
            "gpu": (self.cmb_gpu.currentIndex() == 0),
            "create_new_view": (self.cmb_target.currentIndex() == 1),
            "aberration_first": False,

            # global compatibility controls
            "compat_mode": compat,
            "execution_mode": "compatibility" if compat else "auto",
            "batch_size_override": 1 if compat else 0,
        }


        # Sharpen / Both block
        # Sharpen / Both block
        if mode in ("sharpen", "both"):
            preset.update({
                "sharpening_mode": self.cmb_sh_mode.currentText(),
                "auto_psf": self.chk_auto_psf.isChecked(),
                "nonstellar_psf": self.sld_psf.value() / 10.0,
                "stellar_amount": (self.sld_st_amt.value() / 100.0),
                "nonstellar_amount": self.sld_nst_amt.value() / 100.0,
                "sharpen_channels_separately": self.chk_sh_sep.isChecked(),
                "correct_model_version": self.cmb_correct_ver.currentText(),
                "correct_conservative": self.chk_correct_conservative.isChecked(),
                "chunk_size": int(self.cmb_chunk.currentText()),
                "overlap_pct": parse_overlap_pct(self.cmb_ov.currentText()),
                "temp_stretch": self.chk_temp_stretch.isChecked(),
                "target_median": self.sld_target_median.value() / 100.0,
                "stellar_correct_mode": (
                    (
                        "correct_only"    if self.rb_correct_only.isChecked()    else
                        "correct_sharpen" if self.rb_correct_sharpen.isChecked() else
                        "sharpen_only"
                    ) if getattr(self, "_correct_available", False) else "sharpen_only"
                ),

                # sharpen compatibility controls
                "sharpen_execution_mode": "compatibility" if compat else "auto",
                "sharpen_batch_size_override": 1 if compat else 0,
            })

        # Denoise / Both block
        if mode in ("denoise", "both"):
            dn_model = self.cmb_dn_model.currentText()
            preset.update({
                "denoise_luma": self.sld_dn_lum.value() / 100.0,
                "denoise_color": self.sld_dn_col.value() / 100.0,
                "denoise_mode": self.cmb_dn_mode.currentText(),
                "separate_channels": self.chk_dn_sep.isChecked(),
                "denoise_lite":    (dn_model == "Lite (faster)"),
                "denoise_walking": (dn_model == "Walking Noise"),
                "denoise_compat_mode": compat,
                "denoise_execution_mode": "compatibility" if compat else "auto",
                "denoise_batch_size_override": 1 if compat else 0,
                "temp_stretch": self.chk_temp_stretch.isChecked(),
                "target_median": self.sld_target_median.value() / 100.0,
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

    # ── preset emit / seed (grip + double-click) ───────────────────────
    def get_preset(self) -> dict:
        try:
            p = self.build_preset_from_ui()
            return p
        except Exception as e:
            import traceback
            print(f"[CC-GRIP] get_preset RAISED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            raise

    def seed_from_preset(self, p: dict | None):
        """Seed side for double-click — exact inverse of get_preset. apply_preset
        already inverts every divisor, sets mode first, and drives _mode_changed
        itself, so it is the seed. (dict(p or {}) guards the None case, which
        apply_preset does not.)

        A preset that names only the mode (a canned workflow step) sets only the
        mode. apply_preset gives every key it lacks a fixed default, and the
        dialog saves what changes, so it would replace the user's own GPU, temp
        stretch and strength settings."""
        p = dict(p or {})
        if set(p) == {"mode"}:
            mode = str(p["mode"]).lower()
            self.cmb_mode.setCurrentIndex({"sharpen": 0, "denoise": 1, "both": 2, "superres": 3}.get(mode, 0))
            self._mode_changed()
            return
        self.apply_preset(p)

    def _deferred_resize(self):
        self.adjustSize()
        s = QSettings()
        g = s.value("cc/dialog_geometry")
        if g:
            self.restoreGeometry(g)

    def _restore_cc_geometry(self):
        s = QSettings()
        g = s.value("cc/dialog_geometry")
        if g:
            self.restoreGeometry(g)
        else:
            self.resize(860, 560)

    def closeEvent(self, e):
        thr = getattr(self, "_engine_thread", None)
        if _cc_thread_running(thr):
            self._closing_after_cancel = True
            self._cancel_all_engine()
            e.ignore()
            return
        try:
            QSettings().setValue("cc/dialog_geometry", self.saveGeometry())
            QSettings().sync()
        except Exception:
            pass
        try:
            self._release_memory()
        except Exception:
            pass
        super().closeEvent(e)

    def _release_memory(self):
        """Drop full-frame orig buffer on close."""
        if getattr(self, "_memory_released", False):
            return
        self._memory_released = True

        self.orig = None
        self.doc = None
        self._engine_thread = None

        try:
            from setiastro.saspro.free_torch_memory import trim_process_memory
            trim_process_memory()
        except Exception:
            pass

    def reject(self):
        thr = getattr(self, "_engine_thread", None)
        if _cc_thread_running(thr):
            self._closing_after_cancel = True
            self._cancel_all_engine()
            return
        QSettings().setValue("cc/dialog_geometry", self.saveGeometry())
        try:
            self._release_memory()
        except Exception:
            pass
        super().reject()

    def _engine_cleanup_and_maybe_close(self):
        # Called when the thread has truly stopped. _start_cc_thread owns and
        # deletes the thread object; this only drops the dialog's handle.
        self._engine_thread = None
        self._close_wait()      # never leave a progress window behind
        self._set_run_controls_enabled(True)

        if self._closing_after_cancel:
            self._closing_after_cancel = False
            try:
                self._release_memory()
            except Exception:
                pass
            super().reject()

def open_cosmic_clarity_with_preset(main_window, preset: dict | None = None):
    doc = None
    try:
        sw = main_window.mdi.activeSubWindow()
        if sw is not None:
            doc = getattr(sw.widget(), "document", None)
    except Exception:
        doc = None
    if doc is None:
        dm = getattr(main_window, "doc_manager", getattr(main_window, "docman", None))
        if dm is not None:
            try:
                doc = (dm.get_active_document() if hasattr(dm, "get_active_document")
                       else getattr(dm, "active_document", None))
            except Exception:
                doc = None
    if doc is None:
        try:
            ad = getattr(main_window, "_active_doc", None)
            if callable(ad):
                doc = ad()
        except Exception:
            doc = None
    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.information(main_window, "Cosmic Clarity", _tr("No active image view."))
        return None

    _icon = QIcon()
    try:
        from setiastro.saspro.resources import cosmic_path
        _icon = QIcon(cosmic_path)
    except Exception:
        _icon = QIcon()

    # Interactive open. bypass_guard stays False so a genuine in-progress headless
    # run still blocks a second panel; the guard is inactive during normal use.
    dlg = CosmicClarityDialogPro(main_window, doc, icon=_icon)

    # If __init__ early-rejected (models missing / guard active), its widgets were
    # never built — don't seed a zombie; let its own scheduled reject fire.
    if not hasattr(dlg, "cmb_mode"):
        return dlg

    try:
        dlg.seed_from_preset(preset or {})
    except Exception:
        pass
    try:
        main_window._cosmic_clarity_dialog = dlg   # retain against GC (WA_DeleteOnClose)
        dlg.destroyed.connect(
            lambda *_: setattr(main_window, "_cosmic_clarity_dialog", None))
    except Exception:
        pass
    dlg.show(); dlg.raise_(); dlg.activateWindow()
    return dlg

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
        self._engine_thread = None
        self._closing_after_cancel = False
        self._wait = None
        ok, detail = _cc_models_installed("satellite")
        if not ok:
            _warn_models_missing_and_close(self, detail)
            try:
                self.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            except Exception:
                pass
            QTimer.singleShot(0, self.reject)
            return

        self.setWindowTitle(self.tr("Cosmic Clarity – Satellite Removal"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        if icon:
            try: self.setWindowIcon(icon)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        self.settings = QSettings()
        self.icons = get_icons()

        self.input_folder  = ""
        self.output_folder = ""
        self.sensitivity = 0.10  # 0.01–0.50
        self.doc = doc

        self.file_watcher = QFileSystemWatcher()
        self.file_watcher.directoryChanged.connect(self._on_folder_changed)

        self._sat_thread = None
        self._wait = None

        self._build_ui()
        # Factory state, before saved settings load (folders are not widgets).
        self._factory = snapshot_controls(self)
        self._load_satellite_settings()

    def _reset_to_factory(self):
        # Settings are saved on close, as for any change.
        if self._sat_busy():
            return
        restore_controls(self._factory)

    # ---------- UI ----------
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        self.lbl_logo = QLabel(self)
        self.lbl_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        try:
            pm = QPixmap(self.icons.COSMICCLARITYSAT)
            if not pm.isNull():
                self.lbl_logo.setPixmap(
                    pm.scaledToWidth(280, Qt.TransformationMode.SmoothTransformation)
                )
                outer.addWidget(self.lbl_logo, 0, Qt.AlignmentFlag.AlignHCenter)
        except Exception:
            pass

        content = QHBoxLayout()

        # Left controls
        left = QVBoxLayout()

        row_io = QHBoxLayout()
        self.btn_in  = QPushButton(self.tr("Select Input Folder"));  self.btn_in.clicked.connect(self._choose_input)
        self.btn_out = QPushButton(self.tr("Select Output Folder")); self.btn_out.clicked.connect(self._choose_output)
        row_io.addWidget(self.btn_in)
        row_io.addWidget(self.btn_out)
        row_io.addWidget(make_help_button("cosmic_clarity_satellite", self))
        left.addLayout(row_io)

        left.addWidget(QLabel(self.tr("Use GPU Acceleration:")))
        self.cmb_gpu = QComboBox()
        self.cmb_gpu.addItems(["Yes", "No"])
        left.addWidget(self.cmb_gpu)

        self.chk_gpu_compat = QCheckBox(self.tr("GPU Compatibility Mode (slower, safer for low-VRAM GPUs)"))
        self.chk_gpu_compat.setChecked(False)
        left.addWidget(self.chk_gpu_compat)

        left.addWidget(QLabel(self.tr("Satellite Removal Mode:")))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Full", "Luminance"])
        left.addWidget(self.cmb_mode)

        self.chk_clip = QCheckBox(self.tr("Clip Satellite Trail to 0.000"))
        self.chk_clip.setChecked(True)
        left.addWidget(self.chk_clip)

        row_sens = QHBoxLayout()
        row_sens.addWidget(QLabel(self.tr("Clipping Sensitivity (Lower = more aggressive):")))
        self.sld_sens = QSlider(Qt.Orientation.Horizontal)
        self.sld_sens.setRange(1, 50)
        self.sld_sens.setValue(int(self.sensitivity * 100))
        self.sld_sens.setTickInterval(1)
        self.sld_sens.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sld_sens.valueChanged.connect(self._on_sens_change)
        row_sens.addWidget(self.sld_sens)
        self.lbl_sens_val = QLabel(f"{self.sensitivity:.2f}")
        row_sens.addWidget(self.lbl_sens_val)
        left.addLayout(row_sens)

        self.chk_skip = QCheckBox(self.tr("Skip Save if No Satellite Trail Detected"))
        self.chk_skip.setChecked(False)
        left.addWidget(self.chk_skip)

        self.chk_delete_original = QCheckBox(self.tr("Delete Original Image After Saving"))
        self.chk_delete_original.setChecked(False)
        left.addWidget(self.chk_delete_original)

        row_proc = QHBoxLayout()
        self.btn_single = QPushButton(self.tr("Process Single Image"))
        self.btn_single.clicked.connect(self._process_single_image)
        self.btn_batch = QPushButton(self.tr("Batch Process Input Folder"))
        self.btn_batch.clicked.connect(self._batch_process)
        btn_reset = make_reset_button(self)
        btn_reset.clicked.connect(self._reset_to_factory)
        row_proc.addWidget(self.btn_single)
        row_proc.addWidget(self.btn_batch)
        row_proc.addWidget(btn_reset)
        left.addLayout(row_proc)

        self.btn_monitor = QPushButton(self.tr("Live Monitor Input Folder"))
        self.btn_monitor.clicked.connect(self._live_monitor)
        left.addWidget(self.btn_monitor)

        left.addStretch(1)

        # Right side
        right = QVBoxLayout()
        right.addWidget(QLabel(self.tr("Input Folder Files:")))
        self.tree_in = QTreeWidget()
        self.tree_in.setHeaderLabels([self.tr("Filename")])
        self.tree_in.itemDoubleClicked.connect(lambda *_: self._preview_from_tree(self.tree_in, is_input=True))
        self.tree_in.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_in.customContextMenuRequested.connect(lambda pos: self._context_menu(self.tree_in, pos, is_input=True))
        right.addWidget(self.tree_in)

        right.addWidget(QLabel(self.tr("Output Folder Files:")))
        self.tree_out = QTreeWidget()
        self.tree_out.setHeaderLabels([self.tr("Filename")])
        self.tree_out.itemDoubleClicked.connect(lambda *_: self._preview_from_tree(self.tree_out, is_input=False))
        self.tree_out.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_out.customContextMenuRequested.connect(lambda pos: self._context_menu(self.tree_out, pos, is_input=False))
        right.addWidget(self.tree_out)

        content.addLayout(left, 2)
        content.addLayout(right, 1)

        outer.addLayout(content, 1)

        self.resize(900, 600)

    def _load_satellite_settings(self):
        s = self.settings
        
        # Folders
        self.input_folder = s.value("satellite/input_folder", "", type=str) or ""
        self.output_folder = s.value("satellite/output_folder", "", type=str) or ""
        if self.input_folder:
            self.btn_in.setText(self.tr("Input: {0}").format(os.path.basename(self.input_folder)))
            self._watch(self.input_folder)
            self._refresh_tree(self.tree_in, self.input_folder)
        if self.output_folder:
            self.btn_out.setText(self.tr("Output: {0}").format(os.path.basename(self.output_folder)))
            self._watch(self.output_folder)
            self._refresh_tree(self.tree_out, self.output_folder)

        gpu = s.value("satellite/use_gpu", "Yes", type=str)
        self.cmb_gpu.setCurrentText(gpu if gpu in ("Yes", "No") else "Yes")
        self.chk_gpu_compat.setChecked(s.value("satellite/gpu_compat", False, type=bool))

        mode = s.value("satellite/mode", "Full", type=str)
        idx = self.cmb_mode.findText(mode)
        self.cmb_mode.setCurrentIndex(idx if idx >= 0 else 0)

        self.chk_clip.setChecked(s.value("satellite/clip_trail", True, type=bool))

        sens = s.value("satellite/sensitivity", 10, type=int)
        self.sld_sens.setValue(max(1, min(50, sens)))

        self.chk_skip.setChecked(s.value("satellite/skip_save", False, type=bool))
        self.chk_delete_original.setChecked(s.value("satellite/delete_original", False, type=bool))

    def _save_satellite_settings(self):
        s = self.settings
        s.setValue("satellite/input_folder", self.input_folder)
        s.setValue("satellite/output_folder", self.output_folder)
        s.setValue("satellite/use_gpu", self.cmb_gpu.currentText())
        s.setValue("satellite/gpu_compat", self.chk_gpu_compat.isChecked())
        s.setValue("satellite/mode", self.cmb_mode.currentText())
        s.setValue("satellite/clip_trail", self.chk_clip.isChecked())
        s.setValue("satellite/sensitivity", self.sld_sens.value())
        s.setValue("satellite/skip_save", self.chk_skip.isChecked())
        s.setValue("satellite/delete_original", self.chk_delete_original.isChecked())
        s.sync()

    def _confirm_delete_originals(self) -> bool:
        """If delete-original is enabled, show ONE warning before a run.
        Returns True to proceed, False to abort."""
        if not self.chk_delete_original.isChecked():
            return True
        resp = QMessageBox.warning(
            self,
            self.tr("Delete Originals Enabled"),
            self.tr("The 'Delete Original Image After Saving' option is checked.\n\n"
            "After each image is successfully processed and saved, the ORIGINAL "
            "file will be permanently deleted. This cannot be undone.\n\n"
            "If satellite removal produces an unexpected result, the original "
            "sub will be gone.\n\n"
            "Do you want to proceed?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return resp == QMessageBox.StandardButton.Yes

    # ---------- IO folders ----------
    def _choose_input(self):
        folder = QFileDialog.getExistingDirectory(self, self.tr("Select Input Folder"), self.input_folder or "")
        if not folder:
            return
        self.input_folder = folder
        self.btn_in.setText(self.tr("Input: {0}").format(os.path.basename(folder)))
        self._watch(folder)
        self._refresh_tree(self.tree_in, folder)
        self._save_satellite_settings()

    def _choose_output(self):
        folder = QFileDialog.getExistingDirectory(self, self.tr("Select Output Folder"), self.output_folder or "")
        if not folder:
            return
        self.output_folder = folder
        self.btn_out.setText(self.tr("Output: {0}").format(os.path.basename(folder)))
        self._watch(folder)
        self._refresh_tree(self.tree_out, folder)
        self._save_satellite_settings()

    def _watch(self, folder):
        try:
            if folder and folder not in self.file_watcher.directories():
                self.file_watcher.addPath(folder)
        except Exception:
            pass

    def _on_folder_changed(self, path):
        if path == self.input_folder:
            self._refresh_tree(self.tree_in, self.input_folder)
        if path == self.output_folder:
            self._refresh_tree(self.tree_out, self.output_folder)

    def _refresh_tree(self, tree: QTreeWidget, folder: str):
        tree.clear()
        if not folder or not os.path.isdir(folder): return
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(SAT_INPUT_EXTS):
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
        act_del = QAction(self.tr("Delete File"), self)
        act_ren = QAction(self.tr("Rename File"), self)
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
        if QMessageBox.question(self, self.tr("Confirm Delete"), self.tr("Delete {0}?").format(item.text(0)),
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            os.remove(fp)
            self._refresh_tree(tree, folder)

    def _rename_file(self, tree: QTreeWidget, is_input: bool):
        item = tree.currentItem()
        if not item: return
        folder = self._folder_of(is_input)
        fp = os.path.join(folder, item.text(0))
        new, ok = QInputDialog.getText(self, self.tr("Rename File"), self.tr("Enter new name:"), text=item.text(0))
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
                QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to load image for preview."))
                return
            dlg = ImagePreviewDialog(img, is_mono=is_mono, parent=self)
            dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            dlg.show()
        except Exception as e:
            QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to preview image:\n{0}").format(e))

    # ---------- Single image processing ----------
    def _process_single_image(self):
        if self._sat_busy():
            return
        from PyQt6.QtWidgets import QMessageBox, QFileDialog, QInputDialog

        # ----------------------------
        # Decide source: view or file
        # ----------------------------
        views = self._collect_open_views()
        use_view = False
        compatibility_mode = bool(self.chk_gpu_compat.isChecked())

        if views:
            mb = QMessageBox(self)
            mb.setWindowTitle(self.tr("Process Single Image"))
            mb.setText(self.tr("Choose the source to process:"))
            btn_view = mb.addButton(self.tr("Open View"), QMessageBox.ButtonRole.AcceptRole)
            btn_file = mb.addButton(self.tr("File on Disk"), QMessageBox.ButtonRole.AcceptRole)
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
                sel, ok = QInputDialog.getItem(self, self.tr("Select View"), self.tr("Choose an open view:"), titles, 0, False)
                if not ok:
                    return
                chosen_doc = views[titles.index(sel)][1]

            if chosen_doc is None or getattr(chosen_doc, "image", None) is None:
                QMessageBox.warning(self, self.tr("Warning"), self.tr("Selected view has no image."))
                return

            try:
                img = np.asarray(chosen_doc.image, dtype=np.float32)
                img = np.clip(img, 0.0, 1.0)
            except Exception as e:
                QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to read image from view:\n{0}").format(e))
                return

        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self, self.tr("Select Image"), "",
                SAT_INPUT_FILTER
            )
            if not file_path:
                return  # user cancelled file dialog
            if not file_path:
                return  # user cancelled file dialog

            if not self._confirm_delete_originals():
                return
            try:
                result = load_image(file_path, return_metadata=True)
                if result is None or result[0] is None:
                    raise RuntimeError("load_image returned None.")
                img, hdr, bd, mono, meta = result
                img = np.asarray(img, dtype=np.float32)
                img = np.clip(img, 0.0, 1.0)
            except Exception as e:
                QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to load image:\n{0}").format(e))
                return

        # ----------------------------------------------------
        # Run off the GUI thread: model load, removal and, for a
        # file, the save. The old path ran the engine right here and
        # kept the window alive with processEvents().
        # ----------------------------------------------------
        dst = ""
        image_meta = None
        if not use_view:
            # Not splitext + same ext: there is no Rice/gzip FITS writer and no
            # RAW writer, so a .fz / .fits.gz / .cr2 source is saved as plain
            # .fits and the name has to say so up front.
            dst = satellite_output_name(file_path)
            image_meta = meta.get("xisf_meta") if meta else None
            if image_meta is not None and not isinstance(image_meta, list):
                image_meta = [image_meta]

        thr = SatelliteSingleImageThread(
            img,
            use_gpu=use_gpu,
            mode=mode,
            clip_trail=clip_trail,
            sensitivity=sensitivity,
            compatibility_mode=compatibility_mode,
            skip_if_none=skip_if_none,
            save_to=dst,
            save_hdr=hdr,
            save_mono=mono,
            save_bd=(bd if not use_view else None),
            image_meta=image_meta,
        )
        self._single_ctx = {
            "use_view": use_view, "doc": chosen_doc, "file_path": file_path,
            "skip_if_none": skip_if_none,
        }
        self._sat_cancel_requested = False
        self._wait = WaitDialog(self.tr("Satellite removal"), self)
        self._wait.set_stage(self.tr("Removing satellite trails…"))
        self._wait.cancelled.connect(self._cancel_sat_thread)
        thr.progress.connect(self._on_single_progress)
        thr.log.connect(self._wait.append_output)
        thr.result.connect(self._on_single_result)
        thr.error.connect(self._on_single_error)
        thr.cancelled.connect(self._on_single_cancelled)
        thr.finished.connect(self._on_single_finished)
        self._single_thread = thr
        self._set_sat_buttons_enabled(False)
        self._wait.show()
        _start_cc_thread(thr)

    def _on_single_progress(self, pct: int):
        if self._wait is not None:
            self._wait.set_progress(int(pct))

    def _on_single_result(self, out, detected: bool, saved: str):
        self._close_sat_wait()
        ctx = getattr(self, "_single_ctx", None) or {}
        stopped = bool(getattr(self, "_sat_cancel_requested", False) or self._closing_after_cancel)

        if ctx.get("skip_if_none") and not detected:
            if not stopped:
                QMessageBox.information(self, self.tr("Satellite Removal"),
                                        self.tr("No satellite trail detected (skip-save enabled)."))
            return

        if ctx.get("use_view"):
            if stopped:
                return      # cancel raced the last tile: leave the view untouched
            doc = ctx.get("doc")
            if doc is None:
                return
            if hasattr(doc, "set_image"):
                doc.set_image(out, step_name="Cosmic Clarity – Satellite Removal")
            else:
                doc.image = out
            return

        if not saved:
            return
        QMessageBox.information(self, self.tr("Success"), self.tr("Processed image saved to:\n{0}").format(saved))
        if self.chk_delete_original.isChecked():
            file_path = ctx.get("file_path") or ""
            try:
                if file_path and os.path.abspath(file_path) != os.path.abspath(saved):
                    os.remove(file_path)
                    self._refresh_tree(self.tree_in, self.input_folder)
            except Exception as ex:
                QMessageBox.warning(
                    self, self.tr("Delete Original"),
                    self.tr("Could not delete original:\n{0}").format(ex)
                )

    def _on_single_error(self, msg: str):
        self._close_sat_wait()
        QMessageBox.critical(self, self.tr("Error"), str(msg))

    def _on_single_cancelled(self):
        self._close_sat_wait()
        if not self._closing_after_cancel:
            QMessageBox.information(self, self.tr("Cancelled"), self.tr("Operation cancelled."))

    def _on_single_finished(self):
        self._single_thread = None
        self._close_sat_wait()
        self._set_sat_buttons_enabled(True)
        if self._closing_after_cancel:
            self._closing_after_cancel = False
            QTimer.singleShot(0, self.close)

    def _collect_open_views(self):
        """
        Return a list of (title, doc) for all open MDI views with an image.
        Includes self.doc if supplied and valid.
        """
        views = []
        # include self.doc first if valid
        if getattr(self, "doc", None) is not None and getattr(self.doc, "image", None) is not None:
            title = getattr(self.doc, "display_name", lambda: self.tr("Active View"))()
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
                        t = w.windowTitle() if hasattr(w, "windowTitle") else getattr(d, "display_name", lambda: self.tr("View"))()
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
            QMessageBox.warning(self, self.tr("Warning"), self.tr("Please select both input and output folders."))
            return
        if not self._confirm_delete_originals():
            return
        self._run_engine_thread(monitor=False, title=self.tr("Satellite – Batch processing"))


    def _live_monitor(self):
        if not self.input_folder or not self.output_folder:
            QMessageBox.warning(self, self.tr("Warning"), self.tr("Please select both input and output folders."))
            return
        if not self._confirm_delete_originals():
            return
        self.sld_sens.setEnabled(False)
        self._run_engine_thread(monitor=True, title=self.tr("Satellite – Live monitoring"),
                                on_finish=lambda: self.sld_sens.setEnabled(True))

    def _sat_busy(self) -> bool:
        return (_cc_thread_running(getattr(self, "_sat_thread", None))
                or _cc_thread_running(getattr(self, "_single_thread", None)))

    def _set_sat_buttons_enabled(self, enabled: bool):
        for name in ("btn_single", "btn_batch", "btn_monitor"):
            w = getattr(self, name, None)
            if w is not None:
                try:
                    w.setEnabled(bool(enabled))
                except RuntimeError:
                    pass

    def _close_sat_wait(self):
        if self._wait is not None:
            try:
                self._wait.finish()
            except RuntimeError:
                pass
            self._wait = None

    def _run_engine_thread(self, *, monitor: bool, title: str, on_finish=None):
        if self._sat_busy():
            return
        use_gpu = (self.cmb_gpu.currentText() == "Yes")
        mode = self.cmb_mode.currentText().lower()  # "full" / "luminance"
        clip_trail = bool(self.chk_clip.isChecked())
        sensitivity = float(self.sensitivity)
        skip_save = bool(self.chk_skip.isChecked())
        compatibility_mode = bool(self.chk_gpu_compat.isChecked())
        delete_original = bool(self.chk_delete_original.isChecked())
        self._sat_cancel_requested = False
        self._wait = WaitDialog(title, self)
        self._wait.show()

        # No Qt parent, owned by _start_cc_thread: closing this window can no
        # longer garbage-collect a running thread (a qFatal, i.e. a silent exit).
        self._sat_thread = SatelliteEngineThread(
            input_dir=self.input_folder,
            output_dir=self.output_folder,
            use_gpu=use_gpu,
            mode=mode,
            clip_trail=clip_trail,
            sensitivity=sensitivity,
            skip_save=skip_save,
            monitor=monitor,
            compatibility_mode=compatibility_mode,
            delete_original=delete_original,
        )
        self._sat_on_finish = on_finish
        self._sat_thread.log_signal.connect(self._wait.append_output)
        self._sat_thread.progress_signal.connect(self._on_sat_progress)
        self._sat_thread.finished_signal.connect(self._on_sat_thread_finished)
        self._wait.cancelled.connect(self._cancel_sat_thread)
        self._set_sat_buttons_enabled(False)
        _start_cc_thread(self._sat_thread)

    def _on_sat_progress(self, done: int, total: int):
        if not self._wait:
            return
        self._wait.set_progress(done, total)

    # ---------- Command / run ----------

    def _cancel_sat_thread(self):
        self._sat_cancel_requested = True
        for thr in (getattr(self, "_sat_thread", None), getattr(self, "_single_thread", None)):
            if _cc_thread_running(thr):
                try:
                    thr.cancel()
                    thr.requestInterruption()
                except RuntimeError:
                    pass
        # Keep the progress window up, showing "Cancelling…", until the worker
        # has actually stopped; the finished handlers close it.
        if self._wait is not None:
            self._wait.mark_cancelling()

    def _on_sat_thread_finished(self):
        self._on_thread_finished(getattr(self, "_sat_on_finish", None))

    def _on_thread_finished(self, on_finish):
        thr = getattr(self, "_sat_thread", None)
        monitor = bool(getattr(thr, "monitor", False))
        self._close_sat_wait()
        self._sat_thread = None
        self._set_sat_buttons_enabled(True)
        if callable(on_finish):
            try: on_finish()
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        if self._closing_after_cancel:
            self._closing_after_cancel = False
            QTimer.singleShot(0, self.close)
            return
        if monitor:
            QMessageBox.information(self, self.tr("Live Monitor"), self.tr("Live monitor stopped."))
        elif getattr(self, "_sat_cancel_requested", False):
            QMessageBox.information(self, self.tr("Cancelled"), self.tr("Processing cancelled."))
        else:
            QMessageBox.information(self, self.tr("Done"), self.tr("Processing finished."))

    def closeEvent(self, e):
        if self._sat_busy():
            # Stop the worker first; the window closes itself once it has stopped.
            # A live monitor used to keep running with no window at all.
            self._closing_after_cancel = True
            self._cancel_sat_thread()
            e.ignore()
            return
        self._save_satellite_settings()
        super().closeEvent(e)

    # ---------- Utils ----------
    @staticmethod
    def _create_temp_folder(base="~"):
        user_dir = os.path.expanduser(base)
        temp_folder = os.path.join(user_dir, "CosmicClarityTemp")
        os.makedirs(temp_folder, exist_ok=True)
        return temp_folder

class SatelliteSingleImageThread(QThread):
    """One Satellite Removal run off the GUI thread: model load, removal, file save."""
    progress  = pyqtSignal(int)                  # 0..100
    log       = pyqtSignal(str)
    result    = pyqtSignal(object, bool, str)    # image, trail detected, saved path ("" = none)
    error     = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, img, *, use_gpu: bool, mode: str, clip_trail: bool,
                 sensitivity: float, compatibility_mode: bool, skip_if_none: bool,
                 save_to: str = "", save_hdr=None, save_mono: bool = False,
                 save_bd=None, image_meta=None):
        super().__init__()
        self.img = img
        self.use_gpu = bool(use_gpu)
        self.mode = mode
        self.clip_trail = bool(clip_trail)
        self.sensitivity = float(sensitivity)
        self.compatibility_mode = bool(compatibility_mode)
        self.skip_if_none = bool(skip_if_none)
        self.save_to = save_to or ""
        self.save_hdr = save_hdr
        self.save_mono = bool(save_mono)
        self.save_bd = save_bd
        self.image_meta = image_meta
        self._cancel = False
        self._last_pct = -1

    def cancel(self):
        self._cancel = True

    def _stop_requested(self) -> bool:
        return bool(self._cancel or self.isInterruptionRequested())

    def _progress(self, done: int, total: int):
        if self._stop_requested():
            return False                # the engine raises EngineCancelled
        pct = int((done * 100) / total) if total > 0 else 0
        if pct != self._last_pct:
            self._last_pct = pct
            self.progress.emit(pct)
        return True

    def run(self):
        phase = "load"
        try:
            from setiastro.saspro.cosmicclarity_engines.satellite_engine import (
                get_satellite_models,
                satellite_remove_image,
            )
            self.log.emit("Loading Satellite models…")
            models = get_satellite_models(use_gpu=self.use_gpu, status_cb=lambda s: None)
            if self._stop_requested():
                raise EngineCancelled("Cancelled.")

            phase = "process"
            self.log.emit("Removing satellite trails…")
            out, detected = satellite_remove_image(
                self.img,
                models=models,
                mode=self.mode,
                clip_trail=self.clip_trail,
                sensitivity=self.sensitivity,
                compatibility_mode=self.compatibility_mode,
                progress_cb=self._progress,
            )
            if self._stop_requested():
                raise EngineCancelled("Cancelled.")
            out = np.asarray(out, dtype=np.float32, order="C")

            saved = ""
            if self.save_to and not (self.skip_if_none and not detected):
                phase = "save"
                fmt, _ = _save_fmt_and_bitdepth_from_path_for_legacy(self.save_to)
                save_image(
                    out, self.save_to, fmt or "fits",
                    self.save_bd or "32-bit floating point",
                    self.save_hdr, self.save_mono,
                    image_meta=self.image_meta,
                    file_meta=None,
                )
                saved = self.save_to
            self.result.emit(out, bool(detected), saved)

        except EngineCancelled:
            self.cancelled.emit()
        except Exception as e:
            if self._stop_requested():
                self.cancelled.emit()
            elif phase == "load":
                import traceback as _tb
                self.error.emit(self.tr("Failed to load Satellite models:\n{0}\n\n{1}").format(e, _tb.format_exc()))
            elif phase == "save":
                self.error.emit(self.tr("Failed to save result:\n{0}").format(e))
            else:
                self.error.emit(self.tr("Error processing image:\n{0}").format(e))


class SatelliteEngineThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    progress_signal = pyqtSignal(int, int)  # done, total

    def __init__(self, *, input_dir: str, output_dir: str,
                use_gpu: bool, mode: str, clip_trail: bool,
                sensitivity: float, skip_save: bool, monitor: bool,
                compatibility_mode: bool,
                delete_original: bool = False,
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
        self.compatibility_mode = compatibility_mode
        self.delete_original = delete_original

        self._cancel = False
        self._seen = set()

    def cancel(self):
        self._cancel = True

    def _iter_files(self):
        try:
            for fn in sorted(os.listdir(self.input_dir)):
                # split_ext_compound, not splitext: "x_satellited.fits.gz"
                # splits to stem "x_satellited.fits", which does not end with
                # the suffix, so an already-processed file was re-queued.
                stem = split_ext_compound(fn)[0]
                # Skip files we already processed
                if stem.endswith('_satellited'):
                    continue
                if fn.lower().endswith(SAT_INPUT_EXTS):
                    yield fn
        except Exception:
            return

    def run(self):
        try:
            from setiastro.saspro.cosmicclarity_engines.satellite_engine import (
                get_satellite_models, satellite_remove_image,
                satellite_detect_tiles,
                ArtifactMapBuilder,
            )

            self.log_signal.emit("Loading Satellite models...")
            try:
                models = get_satellite_models(use_gpu=self.use_gpu, status_cb=self.log_signal.emit)
            except Exception as e:
                import traceback as _tb
                self.log_signal.emit(f"Failed to load Satellite models: {e}\n{_tb.format_exc()}")
                return
            self.log_signal.emit("Models loaded.")

            os.makedirs(self.output_dir, exist_ok=True)

            # ── PASS 1 (batch only): fixed-pattern artifact map ─────────────
            #
            # Build it per IMAGE SHAPE, since a pixel map cannot be shared
            # across sensors and shape is the only grouping key available from
            # a bare folder listing. Shapes are read during the detection scan
            # itself, so nothing extra is loaded.
            #
            # LIVE MONITOR DELIBERATELY OPTS OUT: frames arrive one at a time
            # and the output is written before persistence could ever be
            # established. There is no safe veto to apply there.
            clean_files: set = set()     # scanned and known to have no real trail
            real_tiles_by_file: dict = {}   # fn -> REAL trail tiles from pass 1

            def _build_artifact_maps(file_names):
                """Scan the batch: builds the fixed-pattern map AND records
                which files have no real trail at all.

                The scan runs even when the batch is too small for the map,
                because knowing a file is clean is worth having on its own —
                pass 2 can then skip it outright instead of re-deriving the
                same answer through a full load + pad + tile + detect."""
                if self.monitor or not file_names:
                    return

                self.log_signal.emit(
                    f"Pass 1 of 2: scanning {len(file_names)} file(s) for "
                    f"diffraction spikes / amp glow (nothing is written)…"
                )
                builder = ArtifactMapBuilder()
                scanned_any = False
                for i, fn in enumerate(file_names):
                    if self._cancel:
                        return
                    fp = os.path.join(self.input_dir, fn)
                    try:
                        res = load_image(fp, return_metadata=True)
                        if res is None or res[0] is None:
                            continue
                        im = np.clip(np.asarray(res[0], dtype=np.float32), 0.0, 1.0)
                        shape_hw = tuple(int(v) for v in im.shape[:2])
                        tiles = satellite_detect_tiles(
                            im, models,
                            # MUST be 0: the persistence map is tested back
                            # against whole tiles, so the scan grid has to be
                            # fixed and non-overlapping or a fixed artifact
                            # lands in a different tile subset each frame and
                            # the mask shrinks to an unusable sliver.
                            overlap=0,
                            compatibility_mode=self.compatibility_mode,
                            stop_after=None,
                            status_cb=lambda _m: None,
                        )
                    except Exception as e:
                        self.log_signal.emit(f"Pass 1 skipped {fn}: {e}")
                        continue

                    # Group by Object/Filter/Exposure/Instrument, NOT by shape:
                    # Ha and Green off one camera share a shape, and pooling
                    # them hides the Ha amp glow below the persistence
                    # threshold so every Ha frame flags.
                    builder.add(fn, res[1], shape_hw, tiles)
                    scanned_any = True
                    self.progress_signal.emit(i + 1, len(file_names))

                if self._cancel or not scanned_any:
                    return

                # Same engine helper Blink and Calibrate_Lights use, so all
                # three agree on what counts as fixed-pattern structure.
                _clean, _real, _n_art = builder.finalize(
                    status_cb=self.log_signal.emit)
                clean_files.update(_clean)
                real_tiles_by_file.update(_real)
                self.log_signal.emit(
                    f"Fixed-pattern filter: {_n_art} tile(s) ignored across "
                    f"{builder.n_groups} group(s); {len(_real)} file(s) have "
                    f"real trails, {len(_clean)} are clean."
                )

            def process_one(fp_in: str, fp_out: str, known_tiles=None):
                result = load_image(fp_in, return_metadata=True)
                if result is None or result[0] is None:
                    self.log_signal.emit(f"Failed to load: {os.path.basename(fp_in)}")
                    return
                img, hdr, bd, mono, meta = result

                img = np.asarray(img, dtype=np.float32)
                img = np.clip(img, 0.0, 1.0)

                out, detected = satellite_remove_image(
                    img,
                    models=models,
                    mode=self.mode,
                    clip_trail=self.clip_trail,
                    sensitivity=float(self.sensitivity),
                    compatibility_mode=self.compatibility_mode,
                    # Cancel lands mid-file (EngineCancelled), not only between files.
                    progress_cb=lambda _d, _t: not self._cancel,
                    detect_tiles=known_tiles,
                    status_cb=self.log_signal.emit,
                )

                if self.skip_save and (not detected):
                    self.log_signal.emit(f"Skip (no trail): {os.path.basename(fp_in)}")
                    return

                out = np.asarray(out, dtype=np.float32, order="C")
                out = np.clip(out, 0.0, 1.0)

                # Extract image_meta and file_meta for XISF round-trip
                image_meta = meta.get("xisf_meta") if meta else None
                if image_meta is not None and not isinstance(image_meta, list):
                    image_meta = [image_meta]

                fmt, _ = _save_fmt_and_bitdepth_from_path_for_legacy(fp_out)
                save_image(
                    out, fp_out, fmt or "fits",
                    bd or "32-bit floating point",
                    hdr, mono,
                    image_meta=image_meta,
                    file_meta=None,
                )
                self.log_signal.emit(f"Saved: {os.path.basename(fp_out)}")

                if self.delete_original:
                    try:
                        # never delete the file we just wrote
                        if os.path.abspath(fp_in) != os.path.abspath(fp_out):
                            os.remove(fp_in)
                            self.log_signal.emit(f"Deleted original: {os.path.basename(fp_in)}")
                    except Exception as e:
                        self.log_signal.emit(f"Could not delete original {os.path.basename(fp_in)}: {e}")

            # -------- batch (two-pass) OR monitor (loop, single-pass) --------
            _maps_built = False
            while not self._cancel:
                files = list(self._iter_files())

                # monitor: only process new files
                todo = [fn for fn in files if fn not in self._seen]

                # Pass 1 runs once, over the batch we are about to process.
                if not _maps_built:
                    _maps_built = True
                    _build_artifact_maps(todo)
                    if self._cancel:
                        break
                    if not self.monitor:
                        self.log_signal.emit("Pass 2 of 2: removing trails…")

                total = len(todo)
                done = 0
                self.progress_signal.emit(done, total)

                for fn in todo:
                    if self._cancel:
                        break

                    self._seen.add(fn)
                    fp_in = os.path.join(self.input_dir, fn)
                    fp_out = os.path.join(self.output_dir,
                                          satellite_output_name(fn))

                    # if output already exists, treat as done
                    if os.path.exists(fp_out):
                        self.log_signal.emit(f"Exists, skipping: {fn}")
                        done += 1
                        self.progress_signal.emit(done, total)
                        continue

                    # Pass 1 already established this file has no real trail.
                    # Running the pipeline would reproduce that answer at the
                    # cost of a load, a 512px reflect pad, a full tiling and a
                    # detection sweep — all to hand back the input unchanged.
                    if fn in clean_files:
                        if self.skip_save:
                            # No output wanted, and NOT deleting the original:
                            # that matches process_one, which returns before
                            # its delete block on the skip-save path. Deleting
                            # here would destroy a frame we never wrote out.
                            self.log_signal.emit(f"Skip (no trail): {fn}")
                        else:
                            # The pipeline's output for a no-trail frame is the
                            # input round-tripped through load+save, so copy the
                            # bytes instead — identical data, no decode/encode,
                            # and nothing for the round-trip to drop.
                            try:
                                if (split_ext_compound(fn)[1].lower()
                                        == split_ext_compound(fp_out)[1].lower()):
                                    import shutil as _sh
                                    _sh.copy2(fp_in, fp_out)
                                    self.log_signal.emit(f"No trail, copied: {fn}")
                                else:
                                    # Container change (.fz / gzipped FITS /
                                    # RAW -> .fits). The source bytes are not
                                    # valid inside the output container, so
                                    # copying them would write a file whose
                                    # extension lies. Round-trip through
                                    # load+save instead -- still no inference,
                                    # which is the whole point of this path.
                                    _r = load_image(fp_in, return_metadata=True)
                                    if _r is None or _r[0] is None:
                                        raise RuntimeError("load returned nothing")
                                    _im, _hdr, _bd, _mono, _meta = _r
                                    _imeta = _meta.get("xisf_meta") if _meta else None
                                    if _imeta is not None and not isinstance(_imeta, list):
                                        _imeta = [_imeta]
                                    _fmt, _ = _save_fmt_and_bitdepth_from_path_for_legacy(fp_out)
                                    save_image(
                                        np.clip(np.asarray(_im, dtype=np.float32), 0.0, 1.0),
                                        fp_out, _fmt or "fits",
                                        _bd or "32-bit floating point",
                                        _hdr, _mono,
                                        image_meta=_imeta, file_meta=None,
                                    )
                                    self.log_signal.emit(f"No trail, converted: {fn}")
                                if self.delete_original and \
                                        os.path.abspath(fp_in) != os.path.abspath(fp_out):
                                    try:
                                        os.remove(fp_in)
                                        self.log_signal.emit(f"Deleted original: {fn}")
                                    except Exception as e:
                                        self.log_signal.emit(
                                            f"Could not delete original {fn}: {e}")
                            except Exception as e:
                                self.log_signal.emit(f"Copy failed for {fn}: {e}")
                        done += 1
                        self.progress_signal.emit(done, total)
                        continue

                    self.log_signal.emit(f"Processing: {fn}")
                    try:
                        process_one(fp_in, fp_out, real_tiles_by_file.get(fn))
                    except EngineCancelled:
                        break
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