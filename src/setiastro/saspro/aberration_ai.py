# pro/aberration_ai.py
from __future__ import annotations
import os
import webbrowser
import requests
import numpy as np
import sys
import platform  # add
import time
import subprocess

IS_APPLE_ARM = (sys.platform == "darwin" and platform.machine() == "arm64")

def _has_nvidia_gpu() -> bool:
    """Check if system has an NVIDIA GPU (Linux/Windows)."""
    try:
        if platform.system() == "Linux":
            r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=2)
            return "GPU" in (r.stdout.decode("utf-8", errors="ignore") or "")
        elif platform.system() == "Windows":
            try:
                ps = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ';'"],
                    capture_output=True, timeout=2
                )
                out = (ps.stdout.decode("utf-8", errors="ignore") or "").lower()
                return "nvidia" in out
            except Exception:
                w = subprocess.run(["wmic", "path", "win32_VideoController", "get", "name"],
                                  capture_output=True, timeout=2)
                return "nvidia" in (w.stdout.decode("utf-8", errors="ignore") or "").lower()
    except Exception:
        pass
    return False

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QStandardPaths, QSettings
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QComboBox, QSpinBox, QProgressBar, QMessageBox, QCheckBox, QLineEdit, QApplication
)
from PyQt6.QtGui import QIcon
from setiastro.saspro.config import Config

# Optional import (soft dep)
try:
    import onnxruntime as ort
except Exception:
    ort = None


# ---------- GitHub model fetching ----------
GITHUB_REPO = Config.GITHUB_ABERRATION_REPO
LATEST_API  = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

def _model_required_patch(model_path: str) -> int | None:
    """
    Returns the fixed spatial size the model expects (e.g. 512), or None if dynamic.
    """
    if ort is None or not os.path.isfile(model_path):
        return None
    try:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        shp = sess.get_inputs()[0].shape  # e.g. [1, 1, 512, 512] or ['N','C',512,512]
        h = shp[-2]; w = shp[-1]
        if isinstance(h, int) and isinstance(w, int) and h == w:
            return int(h)
    except Exception:
        pass
    return None


def _app_model_dir() -> str:
    d = Config.get_aberration_models_dir()
    os.makedirs(d, exist_ok=True)
    return d

def _download_latest_model_sync(
    dst_dir: str | None = None,
    *,
    progress_cb=None,   # callable(frac_0_to_1)
    cancel_cb=None,     # callable() -> bool
    log_cb=None,        # callable(str)
) -> str:
    """
    Download the latest aberration ONNX model synchronously.
    Returns the downloaded model path.
    Raises RuntimeError on failure/cancel.
    """
    if cancel_cb and cancel_cb():
        raise RuntimeError("Canceled")

    dst_dir = dst_dir or _app_model_dir()
    os.makedirs(dst_dir, exist_ok=True)

    if log_cb:
        log_cb("⬇️ Aberration AI: checking latest model release...")

    try:
        r = requests.get(LATEST_API, timeout=10)
        if r.status_code != 200:
            raise RuntimeError(f"GitHub API error: {r.status_code}")

        js = r.json()
        assets = js.get("assets", [])
        onnx_assets = [a for a in assets if a.get("name", "").lower().endswith(".onnx")]
        if not onnx_assets:
            raise RuntimeError("No .onnx asset found in latest release.")

        asset = onnx_assets[0]
        url = asset["browser_download_url"]
        name = asset["name"]
        out_path = os.path.join(dst_dir, name)
        tmp_path = out_path + ".part"

        if log_cb:
            log_cb(f"⬇️ Aberration AI: downloading model {name}...")

        got = 0
        with requests.get(url, stream=True, timeout=60) as rr:
            rr.raise_for_status()
            total = int(rr.headers.get("Content-Length", "0") or 0)
            chunk = 1 << 20

            with open(tmp_path, "wb") as f:
                for blk in rr.iter_content(chunk):
                    if cancel_cb and cancel_cb():
                        raise RuntimeError("Canceled")
                    if blk:
                        f.write(blk)
                        got += len(blk)
                        if total > 0 and progress_cb is not None:
                            try:
                                progress_cb(min(1.0, got / total))
                            except Exception:
                                pass

        os.replace(tmp_path, out_path)
        QSettings().setValue("AberrationAI/model_path", out_path)

        if log_cb:
            log_cb(f"✅ Aberration AI: downloaded model to {out_path}")

        return out_path

    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise RuntimeError(str(e))

class _DownloadWorker(QThread):
    progressed = pyqtSignal(int)      # 0..100 (downloaded)
    failed     = pyqtSignal(str)
    finished_ok= pyqtSignal(str)      # path

    def __init__(self, dst_dir: str):
        super().__init__()
        self.dst_dir = dst_dir

    def run(self):
        try:
            r = requests.get(LATEST_API, timeout=10)
            if r.status_code != 200:
                raise RuntimeError(f"GitHub API error: {r.status_code}")
            js = r.json()
            assets = js.get("assets", [])
            onnx_assets = [a for a in assets if a.get("name","").lower().endswith(".onnx")]
            if not onnx_assets:
                raise RuntimeError("No .onnx asset found in latest release.")
            asset = onnx_assets[0]
            url = asset["browser_download_url"]
            name = asset["name"]
            out_path = os.path.join(self.dst_dir, name)

            with requests.get(url, stream=True, timeout=60) as rr:
                rr.raise_for_status()
                total = int(rr.headers.get("Content-Length", "0") or 0)
                got = 0
                chunk = 1 << 20
                with open(out_path, "wb") as f:
                    for blk in rr.iter_content(chunk):
                        if blk:
                            f.write(blk)
                            got += len(blk)
                            if total > 0:
                                self.progressed.emit(int(got * 100 / total))
            self.finished_ok.emit(out_path)
        except Exception as e:
            self.failed.emit(str(e))


# ---------- core: tiling + hann blend ----------
def _hann2d(n: int) -> np.ndarray:
    w = np.hanning(n).astype(np.float32)
    return (w[:, None] * w[None, :])

def _tile_indices(n: int, patch: int, overlap: int) -> list[int]:
    stride = patch - overlap
    if patch >= n:
        return [0]
    idx, pos = [], 0
    while True:
        if pos + patch >= n:
            idx.append(n - patch)
            break
        idx.append(pos); pos += stride
    return sorted(set(idx))

def _pad_C_HW(arr: np.ndarray, patch: int) -> tuple[np.ndarray, int, int]:
    C, H, W = arr.shape
    pad_h = max(0, patch - H)
    pad_w = max(0, patch - W)
    if pad_h or pad_w:
        arr = np.pad(arr, ((0,0),(0,pad_h),(0,pad_w)), mode="edge")
    return arr, H, W

def _prepare_input(img: np.ndarray) -> tuple[np.ndarray, bool, bool]:
    """
    Returns (C,H,W) float32 in [0..1]; also returns (channels_last, was_uint16)
    """
    channels_last = (img.ndim == 3)
    if channels_last:
        arr = img.transpose(2,0,1)  # (C,H,W)
    else:
        arr = img[np.newaxis, ...]   # (1,H,W)
    was_uint16 = (arr.dtype == np.uint16)
    if was_uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32)
    return arr, channels_last, was_uint16

def _restore_output(arr: np.ndarray, channels_last: bool, was_uint16: bool, H: int, W: int) -> np.ndarray:
    arr = arr[:, :H, :W]
    arr = np.clip(np.nan_to_num(arr), 0.0, 1.0)
    if was_uint16:
        arr = (arr * 65535.0).astype(np.uint16)
    if channels_last:
        arr = arr.transpose(1,2,0)   # (H,W,C)
    else:
        arr = arr[0]                 # (H,W)
    return arr

def run_onnx_tiled(session, img: np.ndarray, patch_size=512, overlap=64,
                  progress_cb=None, cancel_cb=None) -> np.ndarray:
    """
    session: onnxruntime.InferenceSession
    img: mono (H,W) or RGB (H,W,3) numpy array

    cancel_cb: callable -> bool, return True to cancel
    """
    arr, channels_last, was_uint16 = _prepare_input(img)      # (C,H,W)
    arr, H0, W0 = _pad_C_HW(arr, patch_size)
    C, H, W = arr.shape

    win = _hann2d(patch_size)
    out = np.zeros_like(arr, dtype=np.float32)
    wgt = np.zeros_like(arr, dtype=np.float32)

    hs = _tile_indices(H, patch_size, overlap)
    ws = _tile_indices(W, patch_size, overlap)

    inp_name = session.get_inputs()[0].name
    total = len(hs) * len(ws) * C
    done = 0

    for c in range(C):
        for i in hs:
            for j in ws:
                if cancel_cb and cancel_cb():
                    raise RuntimeError("Canceled")

                patch = arr[c:c+1, i:i+patch_size, j:j+patch_size]  # (1,P,P)
                inp = np.ascontiguousarray(patch[np.newaxis, ...], dtype=np.float32)  # (1,1,P,P)

                out_patch = session.run(None, {inp_name: inp})[0]   # (1,1,P,P)
                out_patch = np.squeeze(out_patch, axis=0)           # (1,P,P)

                out[c:c+1, i:i+patch_size, j:j+patch_size] += out_patch * win
                wgt[c:c+1, i:i+patch_size, j:j+patch_size] += win

                done += 1
                if progress_cb:
                    progress_cb(done / max(1, total))

    wgt[wgt == 0] = 1.0
    arr = out / wgt
    return _restore_output(arr, channels_last, was_uint16, H0, W0)

# ---------- providers ----------
def pick_providers(auto_gpu=True) -> list[str]:
    """
    Windows: DirectML → CUDA → CPU
    mac(Intel): CPU → CoreML (optional)
    mac(Apple Silicon): **CPU only** (avoid CoreML artifact path)
    """
    if ort is None:
        return []

    avail = set(ort.get_available_providers())

    # Apple Silicon: always CPU ( CoreML has 16,384-dim constraint and can artifact )
    if IS_APPLE_ARM:
        return ["CPUExecutionProvider"] if "CPUExecutionProvider" in avail else []

    # Non-Apple ARM
    if not auto_gpu:
        return ["CPUExecutionProvider"] if "CPUExecutionProvider" in avail else []

    order = []
    if "DmlExecutionProvider" in avail:
        order.append("DmlExecutionProvider")
    if "CUDAExecutionProvider" in avail:
        order.append("CUDAExecutionProvider")

    # mac(Intel) can still use CoreML if someone insists, but we won't put it first.
    if "CPUExecutionProvider" in avail:
        order.append("CPUExecutionProvider")
    if "CoreMLExecutionProvider" in avail:
        order.append("CoreMLExecutionProvider")

    return order

def resolve_aberration_model_path() -> str:
    """
    Resolve the active Aberration AI model path from QSettings.
    Prefers custom model when enabled; otherwise uses downloaded model.
    Returns "" if no valid model exists.
    """
    s = QSettings()
    use_custom = s.value("AberrationAI/use_custom_model", False, type=bool)
    downloaded = s.value("AberrationAI/model_path", type=str) or ""
    custom = s.value("AberrationAI/custom_model_path", type=str) or ""

    if use_custom and custom and os.path.isfile(custom):
        return custom
    if downloaded and os.path.isfile(downloaded):
        return downloaded
    return ""

def resolve_or_download_aberration_model_path(
    requested_model_path: str | None = None,
    *,
    auto_download: bool = True,
    progress_cb=None,   # callable(frac_0_to_1)
    cancel_cb=None,     # callable() -> bool
    log_cb=None,        # callable(str)
) -> tuple[str, bool]:
    """
    Resolve the best usable model path.

    Order:
      1) explicit requested_model_path if valid
      2) custom model if enabled and valid
      3) downloaded model if valid
      4) auto-download latest model if allowed

    Returns:
        (model_path, did_download)

    Raises:
        RuntimeError if no model can be resolved/downloaded.
    """
    s = QSettings()
    use_custom = s.value("AberrationAI/use_custom_model", False, type=bool)
    downloaded = s.value("AberrationAI/model_path", type=str) or ""
    custom = s.value("AberrationAI/custom_model_path", type=str) or ""

    # Explicit path wins if valid
    if requested_model_path and os.path.isfile(requested_model_path):
        return requested_model_path, False

    # If caller passed a bad explicit path, log it but continue to fallback
    if requested_model_path and not os.path.isfile(requested_model_path):
        if log_cb:
            log_cb(f"⚠️ Aberration AI: requested model not found: {requested_model_path}")

    # Valid custom model
    if use_custom and custom and os.path.isfile(custom):
        return custom, False

    # Custom was enabled but missing: warn and continue
    if use_custom and custom and not os.path.isfile(custom):
        if log_cb:
            log_cb("⚠️ Aberration AI: custom model is enabled but missing; falling back to downloaded/latest model.")

    # Valid downloaded model
    if downloaded and os.path.isfile(downloaded):
        return downloaded, False

    # Nothing valid: auto-download latest
    if auto_download:
        if log_cb:
            log_cb("⬇️ Aberration AI: no local model found; downloading latest model automatically...")
        path = _download_latest_model_sync(
            _app_model_dir(),
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
            log_cb=log_cb,
        )
        return path, True

    raise RuntimeError("No valid Aberration AI model is configured.")

def run_aberration_ai_on_array(
    image: np.ndarray,
    *,
    model_path: str | None = None,
    patch: int = 512,
    overlap: int = 64,
    border_px: int = 10,
    auto_gpu: bool = True,
    provider: str | None = None,
    providers: list[str] | None = None,
    progress_cb=None,   # callable(frac_0_to_1)
    cancel_cb=None,     # callable() -> bool
    log_cb=None,        # callable(str)
) -> tuple[np.ndarray, str]:
    """
    Synchronous reusable Aberration AI runner.

    Returns:
        (output_image, used_provider)

    Raises:
        RuntimeError on failure/cancel
    """
    if ort is None:
        raise RuntimeError("onnxruntime is not installed.")

    if image is None:
        raise RuntimeError("No image supplied.")

    img = np.asarray(image)
    orig = img.copy()

    if cancel_cb and cancel_cb():
        raise RuntimeError("Canceled")

    # Resolve or auto-download model if needed
    did_download = False
    model_path, did_download = resolve_or_download_aberration_model_path(
        model_path,
        auto_download=True,
        progress_cb=(lambda f: progress_cb(0.15 * float(f))) if progress_cb is not None else None,
        cancel_cb=cancel_cb,
        log_cb=log_cb,
    )

    if not model_path or not os.path.isfile(model_path):
        raise RuntimeError("No valid Aberration AI model is configured.")

    # Provider selection
    if IS_APPLE_ARM:
        resolved_providers = ["CPUExecutionProvider"]
    elif providers:
        resolved_providers = list(providers)
    else:
        if auto_gpu:
            resolved_providers = pick_providers(auto_gpu=True)
        else:
            resolved_providers = [provider or "CPUExecutionProvider"]

    if not resolved_providers:
        resolved_providers = ["CPUExecutionProvider"]

    avail_providers = ort.get_available_providers()
    has_nvidia = _has_nvidia_gpu()

    if log_cb:
        log_cb(f"🔍 Available ONNX providers: {', '.join(avail_providers)}")
        log_cb(f"🔍 Attempting providers: {', '.join(resolved_providers)}")

    if has_nvidia and "CUDAExecutionProvider" not in avail_providers:
        msg = ("⚠️ GPU NVIDIA détecté mais CUDAExecutionProvider n'est pas disponible.\n"
               "   Vous devez installer 'onnxruntime-gpu' au lieu de 'onnxruntime'.\n"
               "   Commande: pip uninstall onnxruntime && pip install onnxruntime-gpu")
        if log_cb:
            log_cb(msg)

    # Match model-required patch if fixed
    req = _model_required_patch(model_path)
    if req and req > 0:
        patch = req

    # CoreML guard
    if "CoreMLExecutionProvider" in resolved_providers and req and req > 128:
        if log_cb:
            log_cb(f"CoreML limited to small tiles; model requires {req}px → using CPU.")
        resolved_providers = ["CPUExecutionProvider"]

    gpu_providers = [p for p in resolved_providers if p != "CPUExecutionProvider"]

    # Init session, fallback to CPU if needed
    try:
        sess = ort.InferenceSession(model_path, providers=resolved_providers)
        used_provider = (sess.get_providers()[0] if sess.get_providers() else "CPUExecutionProvider")

        if log_cb:
            if used_provider != "CPUExecutionProvider" and gpu_providers:
                log_cb(f"✅ Aberration AI: Using GPU provider {used_provider}")
            elif has_nvidia and used_provider == "CPUExecutionProvider":
                log_cb("⚠️ GPU NVIDIA détecté mais utilisation du CPU.\n"
                       "   Installez 'onnxruntime-gpu' pour utiliser le GPU.")
            else:
                log_cb(f"✅ Aberration AI using provider: {used_provider}")

    except Exception as e:
        error_msg = str(e)

        if log_cb:
            log_cb(f"⚠️ Aberration AI: GPU provider failed: {error_msg}")
            log_cb(f"Available providers: {', '.join(avail_providers)}")
            log_cb(f"Attempted providers: {', '.join(resolved_providers)}")

            if "CUDAExecutionProvider" in resolved_providers and "CUDAExecutionProvider" not in avail_providers:
                if has_nvidia:
                    log_cb("❌ CUDAExecutionProvider non disponible alors qu'un GPU NVIDIA est présent.\n"
                           "   Installez 'onnxruntime-gpu': pip uninstall onnxruntime && pip install onnxruntime-gpu")
                else:
                    log_cb("⚠️ CUDAExecutionProvider not available. You may need to install onnxruntime-gpu instead of onnxruntime.")

        try:
            sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            used_provider = "CPUExecutionProvider"
            if log_cb:
                log_cb(f"⚠️ Aberration AI: Falling back to CPU (GPU initialization failed: {error_msg})")
        except Exception as e2:
            raise RuntimeError(f"Failed to init ONNX session:\nGPU error: {error_msg}\nCPU error: {e2}")

    def _cb(frac):
        if cancel_cb and cancel_cb():
            raise RuntimeError("Canceled")
        if progress_cb is not None:
            try:
                frac = float(frac)
                if did_download:
                    progress_cb(0.15 + 0.85 * frac)
                else:
                    progress_cb(frac)
            except Exception:
                pass

    out = run_onnx_tiled(
        sess,
        img,
        patch_size=int(patch),
        overlap=int(overlap),
        progress_cb=_cb,
        cancel_cb=cancel_cb,
    )

    if cancel_cb and cancel_cb():
        raise RuntimeError("Canceled")

    out = _preserve_border(out, orig, int(border_px))
    return out, used_provider

def _preserve_border(dst: np.ndarray, src: np.ndarray, px: int = 10) -> np.ndarray:
    """
    Copy a px-wide ring from src → dst, in-place. Handles mono/RGB.
    Expects same shape for src and dst. Clamps px to image size.
    """
    if px <= 0 or dst is None or src is None:
        return dst
    if dst.shape != src.shape:
        return dst  # shapes differ; skip quietly

    h, w = dst.shape[:2]
    px = int(max(0, min(px, h // 2, w // 2)))
    if px == 0:
        return dst

    s = src.astype(dst.dtype, copy=False)

    # top & bottom
    dst[:px, ...]  = s[:px, ...]
    dst[-px:, ...] = s[-px:, ...]
    # left & right
    dst[:, :px, ...]  = s[:, :px, ...]
    dst[:, -px:, ...] = s[:, -px:, ...]

    return dst

# ---------- worker ----------
class _ONNXWorker(QThread):
    progressed   = pyqtSignal(int)          # 0..100
    failed       = pyqtSignal(str)
    finished_ok  = pyqtSignal(np.ndarray)
    canceled     = pyqtSignal()
    log_message  = pyqtSignal(str)          # for console logging

    def __init__(self, model_path: str, image: np.ndarray, patch: int, overlap: int, providers: list[str]):
        super().__init__()
        self.model_path = model_path
        self.image = image
        self.patch = patch
        self.overlap = overlap
        self.providers = providers
        self.used_provider = None
        self._cancel = False  # cooperative flag

    def cancel(self):
        # Safe to call from UI thread
        self._cancel = True
        self.requestInterruption()

    def _is_canceled(self) -> bool:
        return self._cancel or self.isInterruptionRequested()

    def run(self):
        if ort is None:
            self.failed.emit("onnxruntime is not installed.")
            return

        if self._is_canceled():
            self.canceled.emit()
            return

        def _prog(frac):
            self.progressed.emit(int(frac * 100))

        try:
            out, used_provider = run_aberration_ai_on_array(
                self.image,
                model_path=self.model_path,
                patch=self.patch,
                overlap=self.overlap,
                border_px=0,  # dialog preserves border afterward
                providers=self.providers,   # <-- THIS is the key fix
                progress_cb=_prog,
                cancel_cb=self._is_canceled,
                log_cb=lambda s: self.log_message.emit(str(s)),
            )
            self.used_provider = used_provider
        except Exception as e:
            msg = str(e) or "Error"
            if "Canceled" in msg:
                self.canceled.emit()
            else:
                self.failed.emit(msg)
            return

        if self._is_canceled():
            self.canceled.emit()
            return

        self.finished_ok.emit(out)


# ---------- dialog ----------
class AberrationAIDialog(QDialog):
    def __init__(self, parent, docman, get_active_doc_callable, icon: QIcon | None = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("R.A.'s Aberration Correction (AI)"))
        if icon is not None:
            self.setWindowIcon(icon)

        # Normalize window behavior across platforms
        self.setWindowFlag(Qt.WindowType.Window, True)
        # Non-modal: allow user to switch between images while dialog is open
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        #self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self.docman = docman
        self.get_active_doc = get_active_doc_callable
        self._t_start = None
        self._last_used_provider = None

        v = QVBoxLayout(self)

        # Model row
        row = QHBoxLayout()
        row.addWidget(QLabel(self.tr("Model:")))
        self.model_label = QLabel("—")
        self.model_label.setToolTip("")
        btn_browse = QPushButton(self.tr("Browse…")); btn_browse.clicked.connect(self._browse_active_model)
        row.addWidget(self.model_label, 1)
        row.addWidget(btn_browse)
        v.addLayout(row)
        # Custom model row (NEW)
        row_custom = QHBoxLayout()
        self.chk_use_custom = QCheckBox(self.tr("Use custom model file"))
        self.chk_use_custom.setChecked(False)
        self.chk_use_custom.toggled.connect(self._on_use_custom_toggled)

        self.le_custom_model = QLineEdit()
        self.le_custom_model.setReadOnly(True)
        self.le_custom_model.setPlaceholderText(self.tr("No custom model selected"))
        self.le_custom_model.setToolTip("")

        btn_custom_clear = QPushButton(self.tr("Clear"))
        btn_custom_clear.clicked.connect(self._clear_custom_model)

        row_custom.addWidget(self.chk_use_custom)
        row_custom.addWidget(self.le_custom_model, 1)

        row_custom.addWidget(btn_custom_clear)
        v.addLayout(row_custom)
        # Providers row
        row2 = QHBoxLayout()
        self.chk_auto = QCheckBox(self.tr("Auto GPU (if available)"))
        self.chk_auto.setChecked(True)
        row2.addWidget(self.chk_auto)
        self.cmb_provider = QComboBox()
        row2.addWidget(QLabel(self.tr("Provider:")))
        row2.addWidget(self.cmb_provider, 1)
        v.addLayout(row2)

        # Params row
        row3 = QHBoxLayout()
        row3.addWidget(QLabel(self.tr("Patch")))
        self.spin_patch = QSpinBox(minimum=128, maximum=2048); self.spin_patch.setValue(512)
        row3.addWidget(self.spin_patch)
        row3.addWidget(QLabel(self.tr("Overlap")))
        self.spin_overlap = QSpinBox(minimum=16, maximum=512); self.spin_overlap.setValue(64)
        row3.addWidget(self.spin_overlap)
        v.addLayout(row3)

        # Download / Open folder
        row4 = QHBoxLayout()
        btn_latest = QPushButton(self.tr("Download latest model…"))
        btn_latest.clicked.connect(self._download_latest_model)
        row4.addWidget(btn_latest)
        btn_openfolder = QPushButton(self.tr("Open model folder"))
        btn_openfolder.clicked.connect(self._open_model_folder)
        row4.addWidget(btn_openfolder)
        row4.addStretch(1)
        v.addLayout(row4)

        # Progress + actions
        self.progress = QProgressBar(); self.progress.setRange(0, 100); v.addWidget(self.progress)
        row5 = QHBoxLayout()
        self.btn_run = QPushButton(self.tr("Run")); self.btn_run.clicked.connect(self._run)
        btn_close = QPushButton(self.tr("Close")); btn_close.clicked.connect(self.reject)
        row5.addStretch(1); row5.addWidget(self.btn_run); row5.addWidget(btn_close)
        v.addLayout(row5)

        info = QLabel(
            "Model and weights © Riccardo Alberghi — "
            "<a href='https://github.com/riccardoalberghi'>more information</a>."
        )
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        info.setOpenExternalLinks(True)
        info.setWordWrap(True)
        info.setStyleSheet("color:#888; font-size:11px; margin-top:4px;")
        v.addWidget(info)

        self._model_path = None
        self._refresh_providers()
        self._load_last_model_from_settings()
        self._load_last_custom_model_from_settings()
        use_custom = QSettings().value("AberrationAI/use_custom_model", False, type=bool)
        self.chk_use_custom.setChecked(bool(use_custom))
        if IS_APPLE_ARM:
            self.chk_auto.setChecked(False)
            self.chk_auto.setEnabled(False)

    # ----- model helpers -----
    def _set_model_path(self, p: str | None):
        self._model_path = p
        if p:
            self.model_label.setText(os.path.basename(p))
            self.model_label.setToolTip(p)
            QSettings().setValue("AberrationAI/model_path", p)
        else:
            self.model_label.setText("—")
            self.model_label.setToolTip("")
            QSettings().remove("AberrationAI/model_path")

    def _load_last_model_from_settings(self):
        p = QSettings().value("AberrationAI/model_path", type=str)
        if p and os.path.isfile(p):
            self._set_model_path(p)

    def _browse_active_model(self):
        """
        Single Browse button.
        - If user picks a file inside the app model folder -> treat as "downloaded" selection (use_custom_model=False)
        - If user picks a file outside -> treat as "custom" (use_custom_model=True)
        """
        app_dir = os.path.abspath(_app_model_dir())

        # Start in last-used folder if possible
        last_custom = QSettings().value("AberrationAI/custom_model_path", type=str) or ""
        last_downloaded = QSettings().value("AberrationAI/model_path", type=str) or ""
        start_dir = None
        for candidate in (last_custom, last_downloaded):
            if candidate and os.path.isfile(candidate):
                d = os.path.dirname(candidate)
                if os.path.isdir(d):
                    start_dir = d
                    break
        if start_dir is None:
            start_dir = app_dir

        p, _ = QFileDialog.getOpenFileName(self, "Select ONNX model", start_dir, "ONNX (*.onnx)")
        if not p:
            return

        p_abs = os.path.abspath(p)
        # Determine if picked file is inside app model folder
        in_app_dir = False
        try:
            in_app_dir = os.path.commonpath([app_dir, p_abs]) == app_dir
        except Exception:
            in_app_dir = p_abs.startswith(app_dir)

        if in_app_dir:
            # "Downloaded" selection
            self._set_model_path(p_abs)
            self._set_custom_model_path(None)
            QSettings().setValue("AberrationAI/use_custom_model", False)
            if hasattr(self, "chk_use_custom"):
                self.chk_use_custom.setChecked(False)
        else:
            # "Custom" selection
            self._set_custom_model_path(p_abs)
            QSettings().setValue("AberrationAI/use_custom_model", True)
            if hasattr(self, "chk_use_custom"):
                self.chk_use_custom.setChecked(True)

        # Keep visuals in sync
        self._refresh_model_label()
        self._refresh_custom_row_visibility()


    def _refresh_model_label(self):
        downloaded = QSettings().value("AberrationAI/model_path", type=str) or ""
        custom     = QSettings().value("AberrationAI/custom_model_path", type=str) or ""
        use_custom = QSettings().value("AberrationAI/use_custom_model", False, type=bool)

        if use_custom and custom:
            self.model_label.setText(f"Custom: {os.path.basename(custom)}")
            self.model_label.setToolTip(custom)
        elif downloaded:
            self.model_label.setText(f"Downloaded: {os.path.basename(downloaded)}")
            self.model_label.setToolTip(downloaded)
        else:
            self.model_label.setText("—")
            self.model_label.setToolTip("")


    def _open_model_folder(self):
        d = _app_model_dir()
        try:
            if os.name == "nt":
                os.startfile(d)  # type: ignore
            elif sys.platform == "darwin":
                import subprocess; subprocess.Popen(["open", d])
            else:
                import subprocess; subprocess.Popen(["xdg-open", d])
        except Exception:
            webbrowser.open(f"file://{d}")
    # ----- custom model helpers (NEW) -----
    def _set_custom_model_path(self, p: str | None):
        if p:
            self.le_custom_model.setText(os.path.basename(p))
            self.le_custom_model.setToolTip(p)
            QSettings().setValue("AberrationAI/custom_model_path", p)
        else:
            self.le_custom_model.clear()
            self.le_custom_model.setToolTip("")
            QSettings().remove("AberrationAI/custom_model_path")

    def _load_last_custom_model_from_settings(self):
        p = QSettings().value("AberrationAI/custom_model_path", type=str)
        if p:
            if os.path.isfile(p):
                self._set_custom_model_path(p)
            else:
                # Keep the broken path visible in tooltip for debugging
                if hasattr(self, "le_custom_model"):
                    self.le_custom_model.setText(os.path.basename(p) + "  (missing)")
                    self.le_custom_model.setToolTip(p)

        # After both loads, sync labels/visibility
        self._refresh_model_label()
        self._refresh_custom_row_visibility()

    def _refresh_custom_row_visibility(self):
        """
        If you keep the custom row in the UI, hide the path field unless custom is enabled.
        """
        if not hasattr(self, "le_custom_model"):
            return
        use_custom = QSettings().value("AberrationAI/use_custom_model", False, type=bool)
        self.le_custom_model.setVisible(bool(use_custom))


    def _refresh_model_label(self):
        downloaded = QSettings().value("AberrationAI/model_path", type=str) or ""
        custom     = QSettings().value("AberrationAI/custom_model_path", type=str) or ""
        use_custom = QSettings().value("AberrationAI/use_custom_model", False, type=bool)

        # Prefer custom only if enabled AND the file exists
        if use_custom and custom:
            if os.path.isfile(custom):
                self.model_label.setText(f"Custom: {os.path.basename(custom)}")
                self.model_label.setToolTip(custom)
                return
            else:
                self.model_label.setText(f"Custom: {os.path.basename(custom)}  (missing)")
                self.model_label.setToolTip(custom)
                return

        # Otherwise show downloaded if valid
        if downloaded and os.path.isfile(downloaded):
            self.model_label.setText(f"Downloaded: {os.path.basename(downloaded)}")
            self.model_label.setToolTip(downloaded)
        else:
            self.model_label.setText("—")
            self.model_label.setToolTip("")


    def _browse_custom_model(self):
        # Start at last dir if possible, else app model dir
        last = QSettings().value("AberrationAI/custom_model_path", type=str) or ""
        start_dir = os.path.dirname(last) if last and os.path.isdir(os.path.dirname(last)) else _app_model_dir()
        p, _ = QFileDialog.getOpenFileName(self, "Select custom ONNX model", start_dir, "ONNX (*.onnx)")
        if p:
            self._set_custom_model_path(p)
            QSettings().setValue("AberrationAI/use_custom_model", True)
            if not self.chk_use_custom.isChecked():
                self.chk_use_custom.setChecked(True)

    def _clear_custom_model(self):
        self._set_custom_model_path(None)
        QSettings().setValue("AberrationAI/use_custom_model", False)
        if hasattr(self, "chk_use_custom"):
            self.chk_use_custom.setChecked(False)

        self._refresh_model_label()
        self._refresh_custom_row_visibility()


    def _on_use_custom_toggled(self, on: bool):
        QSettings().setValue("AberrationAI/use_custom_model", bool(on))

        if on:
            p = QSettings().value("AberrationAI/custom_model_path", type=str) or ""
            if not (p and os.path.isfile(p)):
                # Don’t spawn another browse button path; use the ONE browse if they want
                QMessageBox.information(
                    self,
                    self.tr("Custom model"),
                    self.tr("Custom model is enabled, but no custom file is selected.\n"
                            "Click Browse… to choose a model file.")
                )
                # Optional: auto-open the single browse:
                # self._browse_active_model()
                # return

        self._refresh_model_label()
        self._refresh_custom_row_visibility()


    # ----- provider UI -----
    def _log(self, msg: str):  # NEW
        mw = self.parent()
        try:
            if hasattr(mw, "_log"):
                mw._log(msg)
            elif hasattr(mw, "update_status"):
                mw.update_status(msg)
        except Exception:
            pass

    def _refresh_providers(self):
        if ort is None:
            self.cmb_provider.clear()
            self.cmb_provider.addItem("onnxruntime not installed")
            self.cmb_provider.setEnabled(False)
            return

        avail = ort.get_available_providers()
        self.cmb_provider.clear()

        if IS_APPLE_ARM:
            # Hard lock to CPU on M-series
            self.cmb_provider.addItem("CPUExecutionProvider")
            self.cmb_provider.setCurrentText("CPUExecutionProvider")
            self.cmb_provider.setEnabled(False)
            # also turn off Auto GPU and disable that checkbox
            self.chk_auto.setChecked(False)
            self.chk_auto.setEnabled(False)
            return

        # Other platforms: show all, sane default
        for name in avail:
            self.cmb_provider.addItem(name)

        if "DmlExecutionProvider" in avail:
            self.cmb_provider.setCurrentText("DmlExecutionProvider")
        elif "CUDAExecutionProvider" in avail:
            self.cmb_provider.setCurrentText("CUDAExecutionProvider")
        elif "CPUExecutionProvider" in avail:
            self.cmb_provider.setCurrentText("CPUExecutionProvider")
        elif "CoreMLExecutionProvider" in avail:
            self.cmb_provider.setCurrentText("CoreMLExecutionProvider")

    # ----- download -----
    def _download_latest_model(self):
        if requests is None:
            QMessageBox.warning(self, "Network", "The 'requests' package is required."); return
        dst = _app_model_dir()
        self.progress.setRange(0, 0)  # busy
        self.btn_run.setEnabled(False)
        self._dl = _DownloadWorker(dst)
        self._dl.progressed.connect(self.progress.setValue)
        self._dl.failed.connect(self._on_download_failed)
        self._dl.finished_ok.connect(self._on_download_ok)
        self._dl.finished.connect(lambda: (self.progress.setRange(0, 100), self.btn_run.setEnabled(True)))
        self._dl.start()

    def _on_download_failed(self, msg: str):
        QMessageBox.critical(self, "Download", msg)

    def _on_download_ok(self, path: str):
        self.progress.setValue(100)
        self._set_model_path(path)

        # Download becomes the active model unless custom is explicitly enabled
        if not QSettings().value("AberrationAI/use_custom_model", False, type=bool):
            self._set_custom_model_path(None)

        QMessageBox.information(self, "Model", f"Downloaded: {os.path.basename(path)}")

        self._refresh_model_label()
        self._refresh_custom_row_visibility()

    # ----- run -----
    def _run(self):
        if ort is None:
            QMessageBox.critical(
                self,
                "Unsupported ONNX Runtime",
                "The currently installed onnxruntime is not supported on this machine.\n"
                "Please try installing an earlier version (for example 1.19.x) and try again."
            )
            return
        
        # Choose model path (normal vs custom)
        use_custom = QSettings().value("AberrationAI/use_custom_model", False, type=bool)
        downloaded = QSettings().value("AberrationAI/model_path", type=str) or ""
        custom     = QSettings().value("AberrationAI/custom_model_path", type=str) or ""

        model_path = custom if use_custom else downloaded
        # Choose model path (normal vs custom), but allow auto-download fallback later
        use_custom = QSettings().value("AberrationAI/use_custom_model", False, type=bool)
        downloaded = QSettings().value("AberrationAI/model_path", type=str) or ""
        custom     = QSettings().value("AberrationAI/custom_model_path", type=str) or ""

        model_path = custom if use_custom else downloaded

        if self.chk_use_custom.isChecked():
            cp = QSettings().value("AberrationAI/custom_model_path", type=str) or ""
            if cp and os.path.isfile(cp):
                model_path = cp
            else:
                self._log("Aberration AI: custom model is enabled but missing; will download/use latest release model.")
                model_path = ""
        elif model_path and not os.path.isfile(model_path):
            self._log("Aberration AI: configured model is missing; will download latest release model automatically.")
            model_path = ""

        doc = self.get_active_doc()
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.warning(self, "Image", "No active image.")
            return

        img = np.asarray(doc.image)
        self._orig_for_border = img.copy()

        patch   = int(self.spin_patch.value())
        overlap = int(self.spin_overlap.value())

        # -------- providers (always choose, then always run) --------
        if IS_APPLE_ARM:
            providers = ["CPUExecutionProvider"]
            self.chk_auto.setChecked(False)
        else:
            if self.chk_auto.isChecked():
                providers = pick_providers(auto_gpu=True)
            else:
                sel = self.cmb_provider.currentText()
                providers = [sel] if sel else ["CPUExecutionProvider"]

        # --- make patch match the model's requirement (if fixed) ---
        req = _model_required_patch(model_path)
        if req and req > 0:
            patch = req
            try:
                self.spin_patch.blockSignals(True)
                self.spin_patch.setValue(req)
            finally:
                self.spin_patch.blockSignals(False)

        # --- CoreML guard on Intel: if model needs >128, run on CPU instead ---
        if ("CoreMLExecutionProvider" in providers) and (req and req > 128):
            self._log(f"CoreML limited to small tiles; model requires {req}px → using CPU.")
            providers = ["CPUExecutionProvider"]
            try:
                self.cmb_provider.setCurrentText("CPUExecutionProvider")
                self.chk_auto.setChecked(False)
            except Exception:
                pass

        self._t_start = time.perf_counter()
        prov_txt = ("auto" if self.chk_auto.isChecked() else self.cmb_provider.currentText() or "CPU")
        model_txt = os.path.basename(model_path) if (model_path and os.path.isfile(model_path)) else "auto-download/latest"
        self._log(f"🚀 Aberration AI: model={model_txt}, provider={prov_txt}, patch={patch}, overlap={overlap}")
        
        self._effective_model_path = model_path

        # -------- run worker --------
        self.progress.setValue(0)
        self.btn_run.setEnabled(False)

        self._worker = _ONNXWorker(model_path, img, patch, overlap, providers)
        self._worker.progressed.connect(self.progress.setValue)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished_ok.connect(self._on_ok)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.log_message.connect(self._log)  # Connect log messages to console
        self._worker.start()


    def _on_failed(self, msg: str):
        model_path = getattr(self, "_effective_model_path", self._model_path)
        self._log(f"❌ Aberration AI failed: {msg}")
        QMessageBox.critical(self, "ONNX Error", msg)
        self.reject()   # closes the dialog

    def _on_ok(self, out: np.ndarray):
        used = getattr(self._worker, "used_provider", None) or \
            (self.cmb_provider.currentText() if not self.chk_auto.isChecked() else "auto")        
        model_path = getattr(self, "_effective_model_path", self._model_path)
        if not model_path or not os.path.isfile(model_path):
            model_path = resolve_aberration_model_path()
        doc = self.get_active_doc()
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.warning(self, "Image", "No active image.")
            return

        # 1) Preserve a thin border from the original image (prevents “eaten” edges)
        BORDER_PX = 10
        src = getattr(self, "_orig_for_border", None)
        if src is None or src.shape != out.shape:
            try:
                src = np.asarray(doc.image)
            except Exception:
                src = None
        out = _preserve_border(out, src, BORDER_PX)

        # 2) Metadata for this step (stored on the document)
        meta = {
            "is_mono": (out.ndim == 2),
            "processing_parameters": {
                **(getattr(doc, "metadata", {}) or {}).get("processing_parameters", {}),
                "AberrationAI": {
                    "model_path": model_path,
                    "patch_size": int(self.spin_patch.value()),
                    "overlap": int(self.spin_overlap.value()),
                    "provider": used,
                    "border_px": BORDER_PX,
                }
            }
        }

        # 3) Apply through history-aware API (either path is fine)
        try:
            # Preferred: directly on the document
            if hasattr(doc, "apply_edit"):
                doc.apply_edit(out, meta, step_name="Aberration AI")
            # Or via DocManager (same effect)
            elif hasattr(self.docman, "update_active_document"):
                self.docman.update_active_document(out, metadata=meta, step_name="Aberration AI")
            else:
                # Last-resort fallback (no undo): avoid if possible
                doc.image = out
                try:
                    doc.metadata.update(meta)
                    doc.changed.emit()
                except Exception:
                    pass
        except Exception as e:
            self._log(f"❌ Aberration AI apply failed: {e}")
            QMessageBox.critical(self, "Apply Error", f"Failed to apply result:\n{e}")
            return

        # 3.5) Register this as last_headless_command for Replay Last Action  ← NEW
        try:
            main = self.parent()
            if main is not None:
                auto_gpu = bool(self.chk_auto.isChecked())
                preset = {
                    "model": model_path,
                    "patch": int(self.spin_patch.value()),
                    "overlap": int(self.spin_overlap.value()),
                    "border_px": int(BORDER_PX),
                    "auto_gpu": auto_gpu,
                }
                if not auto_gpu:
                    preset["provider"] = self.cmb_provider.currentText() or "CPUExecutionProvider"

                payload = {
                    "command_id": "aberrationai",
                    "preset": preset,
                }
                setattr(main, "_last_headless_command", payload)

                # optional log
                try:
                    if hasattr(main, "_log"):
                        prov = preset.get("provider", "auto" if auto_gpu else "CPUExecutionProvider")
                        main._log(
                            f"[Replay] Registered Aberration AI as last action "
                            f"(patch={preset['patch']}, overlap={preset['overlap']}, "
                            f"border={preset['border_px']}px, provider={prov})"
                        )
                except Exception:
                    pass
        except Exception:
            # never break the tool if replay wiring fails
            pass

        # 4) Refresh the active view
        mw = self.parent()
        sw = getattr(getattr(mw, "mdi", None), "activeSubWindow", lambda: None)()
        if sw and hasattr(sw, "widget"):
            w = sw.widget()
            if hasattr(w, "reload_from_doc"):
                try: w.reload_from_doc()
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            elif hasattr(w, "update_view"):
                try: w.update_view()
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            elif hasattr(w, "update"):
                w.update()

        dt = 0.0
        try:
            if self._t_start is not None:
                dt = time.perf_counter() - self._t_start
        except Exception:
            pass
        used = getattr(self._worker, "used_provider", None) or \
               (self.cmb_provider.currentText() if not self.chk_auto.isChecked() else "auto")
        BORDER_PX = 10  # same value used above
        self._log(
            f"✅ Aberration AI applied "
            f"(model={os.path.basename(model_path)}, provider={used}, "
            f"patch={int(self.spin_patch.value())}, overlap={int(self.spin_overlap.value())}, "
            f"border={BORDER_PX}px, time={dt:.2f}s)"
        )

        self.progress.setValue(100)
        # NEW: close this UI after a successful run
        self.accept()   # or self.close()
        return

    def _on_worker_finished(self):
        # Dialog might have been closed by _on_ok()
        if not self.isVisible():
            return

        if hasattr(self, "btn_run"):
            try:
                self.btn_run.setEnabled(True)
            except RuntimeError:
                pass
        self._worker = None