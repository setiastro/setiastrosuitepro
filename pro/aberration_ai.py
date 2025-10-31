# pro/aberration_ai.py
from __future__ import annotations
import os, webbrowser, requests
import numpy as np
import sys, platform  # add
import time

IS_APPLE_ARM = (sys.platform == "darwin" and platform.machine() == "arm64")

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QStandardPaths, QSettings
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QComboBox, QSpinBox, QProgressBar, QMessageBox, QCheckBox
)
from PyQt6.QtGui import QIcon

# Optional import (soft dep)
try:
    import onnxruntime as ort
except Exception:
    ort = None

import importlib

try:
    from importlib import metadata as _importlib_metadata
except Exception:
    _importlib_metadata = None

# will be filled by _probe_onnxruntime()
_ORT_FLAVOR: str | None = None         # e.g. "onnxruntime", "onnxruntime-directml"
_ORT_IMPORT_ERR: str | None = None     # last import error string
_ORT_FOUND_DISTS: list[str] = []       # names we found via importlib.metadata

# ---------- GitHub model fetching ----------
GITHUB_REPO = "riccardoalberghi/abberation_models"
LATEST_API  = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

def _model_required_patch(model_path: str) -> int | None:
    rt = _probe_onnxruntime()
    if rt is None or not os.path.isfile(model_path):
        return None
    try:
        sess = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        shp = sess.get_inputs()[0].shape
        h = shp[-2]; w = shp[-1]
        if isinstance(h, int) and isinstance(w, int) and h == w:
            return int(h)
    except Exception:
        pass
    return None


def _app_model_dir() -> str:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    if not base:
        base = os.path.expanduser("~/.local/share/SetiAstro")
    d = os.path.join(base, "Models", "aberration_ai")
    os.makedirs(d, exist_ok=True)
    return d


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

def run_onnx_tiled(session, img: np.ndarray, patch_size=512, overlap=64, progress_cb=None) -> np.ndarray:
    """
    session: onnxruntime.InferenceSession
    img: mono (H,W) or RGB (H,W,3) numpy array
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
                patch = arr[c:c+1, i:i+patch_size, j:j+patch_size]  # (1, P, P)
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
    rt = _probe_onnxruntime()
    if rt is None:
        return []

    avail = set(rt.get_available_providers())

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

# ---------- hardened ONNXRuntime loader ----------
def _probe_onnxruntime():
    """
    Try very hard to get an onnxruntime module.
    Returns the module or None.
    Also populates _ORT_FLAVOR, _ORT_IMPORT_ERR, _ORT_FOUND_DISTS for UI.
    """
    global ort, _ORT_FLAVOR, _ORT_IMPORT_ERR, _ORT_FOUND_DISTS

    # already loaded?
    if "ort" in globals() and ort is not None:
        return ort

    # 1) the normal one
    try:
        import onnxruntime as _ort
        ort = _ort
        _ORT_FLAVOR = "onnxruntime"
        _ORT_IMPORT_ERR = None
        return ort
    except Exception as e1:
        ort = None
        _ORT_IMPORT_ERR = f"import onnxruntime failed: {e1!r}"

    # 2) check if any *package* is installed even if import failed (wrong env / broken DLL)
    _ORT_FOUND_DISTS = []
    if _importlib_metadata:
        for dist_name in ("onnxruntime", "onnxruntime-directml", "onnxruntime-gpu", "onnx"):
            try:
                ver = _importlib_metadata.version(dist_name)
                _ORT_FOUND_DISTS.append(f"{dist_name}=={ver}")
            except Exception:
                pass

    # 3) some people try to import the package name directly; try once
    # (note: most of these *still* expose `onnxruntime` as the module name,
    # but this at least lets us catch weird installs)
    for alt_mod in ("onnxruntime_directml", "onnxruntime_gpu"):
        try:
            _alt = importlib.import_module(alt_mod)
            # some alt packages actually re-export the real module
            ort = getattr(_alt, "onnxruntime", _alt)
            _ORT_FLAVOR = alt_mod
            _ORT_IMPORT_ERR = None
            return ort
        except Exception:
            pass

    return None


# ---------- worker ----------
class _ONNXWorker(QThread):
    progressed = pyqtSignal(int)         # 0..100
    failed     = pyqtSignal(str)
    finished_ok= pyqtSignal(np.ndarray)

    def __init__(self, model_path: str, image: np.ndarray, patch: int, overlap: int, providers: list[str]):
        super().__init__()
        self.model_path = model_path
        self.image = image
        self.patch = patch
        self.overlap = overlap
        self.providers = providers
        self.used_provider = None

    def run(self):
        rt = _probe_onnxruntime()
        if rt is None:
            # include detail so we can tell if it's a "wrong env" case
            extra = ""
            if _ORT_FOUND_DISTS:
                extra = (
                    "\n\nI can see these ONNX/ONNXRuntime packages installed, but not in *this* Python:\n  - "
                    + "\n  - ".join(_ORT_FOUND_DISTS)
                    + f"\n\nThis SASpro is running under: {sys.executable}"
                    + "\nInstall into *this* Python* with:\n"
                    f'  "{sys.executable}" -m pip install onnxruntime'
                )
            elif _ORT_IMPORT_ERR:
                extra = f"\n\nImport error was:\n{_ORT_IMPORT_ERR}"

            self.failed.emit("onnxruntime is not installed or could not be imported." + extra)
            return
        try:
            sess = rt.InferenceSession(self.model_path, providers=self.providers)
            self.used_provider = (sess.get_providers()[0] if sess.get_providers() else None)
        except Exception:
            # fallback CPU if GPU fails
            try:
                sess = rt.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
                self.used_provider = "CPUExecutionProvider"
            except Exception as e2:
                self.failed.emit(f"Failed to init ONNX session:\n{e2}")
                return

        def cb(frac):
            self.progressed.emit(int(frac * 100))

        try:
            out = run_onnx_tiled(sess, self.image, self.patch, self.overlap, cb)
        except Exception as e:
            self.failed.emit(str(e)); return

        self.finished_ok.emit(out)


# ---------- dialog ----------
class AberrationAIDialog(QDialog):
    """
    UI:
      - Download latest model (from GitHub) OR browse existing .onnx
      - Auto GPU & provider
      - Patch/Overlap
      - Run → opens corrected result in new view
    """
    def __init__(self, parent, docman, get_active_doc_callable, icon: QIcon | None = None):
        super().__init__(parent)
        self.setWindowTitle("R.A.'s Aberration Correction (AI)")
        if icon is not None:
            self.setWindowIcon(icon)  # <-- here
        self.docman = docman
        self.get_active_doc = get_active_doc_callable
        self._t_start = None              # NEW: timing
        self._last_used_provider = None   # NEW: what ORT actually used

        v = QVBoxLayout(self)

        # Model row
        row = QHBoxLayout()
        row.addWidget(QLabel("Model:"))
        self.model_label = QLabel("—")
        self.model_label.setToolTip("")
        btn_browse = QPushButton("Browse…"); btn_browse.clicked.connect(self._browse_model)
        row.addWidget(self.model_label, 1)
        row.addWidget(btn_browse)
        v.addLayout(row)

        # Providers row
        row2 = QHBoxLayout()
        self.chk_auto = QCheckBox("Auto GPU (if available)")
        self.chk_auto.setChecked(True)
        row2.addWidget(self.chk_auto)
        self.cmb_provider = QComboBox()
        row2.addWidget(QLabel("Provider:"))
        row2.addWidget(self.cmb_provider, 1)
        v.addLayout(row2)

        # Params row
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Patch"))
        self.spin_patch = QSpinBox(minimum=128, maximum=2048); self.spin_patch.setValue(512)
        row3.addWidget(self.spin_patch)
        row3.addWidget(QLabel("Overlap"))
        self.spin_overlap = QSpinBox(minimum=16, maximum=512); self.spin_overlap.setValue(64)
        row3.addWidget(self.spin_overlap)
        v.addLayout(row3)

        # Download / Open folder
        row4 = QHBoxLayout()
        btn_latest = QPushButton("Download latest model…")
        btn_latest.clicked.connect(self._download_latest_model)
        row4.addWidget(btn_latest)
        btn_openfolder = QPushButton("Open model folder")
        btn_openfolder.clicked.connect(self._open_model_folder)
        row4.addWidget(btn_openfolder)
        row4.addStretch(1)
        v.addLayout(row4)

        # Progress + actions
        self.progress = QProgressBar(); self.progress.setRange(0, 100); v.addWidget(self.progress)
        row5 = QHBoxLayout()
        self.btn_run = QPushButton("Run"); self.btn_run.clicked.connect(self._run)
        btn_close = QPushButton("Close"); btn_close.clicked.connect(self.reject)
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

    def _browse_model(self):
        start_dir = _app_model_dir()
        p, _ = QFileDialog.getOpenFileName(self, "Select ONNX model", start_dir, "ONNX (*.onnx)")
        if p:
            self._set_model_path(p)

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
        rt = _probe_onnxruntime()
        if rt is None:
            self.cmb_provider.clear()
            # friendlier message
            if _ORT_FOUND_DISTS:
                self.cmb_provider.addItem("onnxruntime in another env")
                self.cmb_provider.setToolTip(
                    "I detected: " + ", ".join(_ORT_FOUND_DISTS)
                    + f"\nBut this Python ({sys.executable}) cannot import it.\n"
                    f'Run: "{sys.executable}" -m pip install onnxruntime'
                )
            else:
                msg = "onnxruntime not installed"
                if _ORT_IMPORT_ERR:
                    msg += f" ({_ORT_IMPORT_ERR.splitlines()[-1]})"
                self.cmb_provider.addItem(msg)
            self.cmb_provider.setEnabled(False)
            return

        avail = rt.get_available_providers()
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
        QMessageBox.information(self, "Model", f"Downloaded: {os.path.basename(path)}")

    # ----- run -----
    def _run(self):
        rt = _probe_onnxruntime()
        if rt is None:
            extra = ""
            if _ORT_FOUND_DISTS:
                extra = (
                    "\n\nI found these packages installed:\n  - "
                    + "\n  - ".join(_ORT_FOUND_DISTS)
                    + f"\n…but they are not importable from this Python: {sys.executable}\n"
                    f'Try: "{sys.executable}" -m pip install onnxruntime'
                )
            elif _ORT_IMPORT_ERR:
                extra = f"\n\nImport error was:\n{_ORT_IMPORT_ERR}"

            QMessageBox.critical(
                self, "Missing dependency",
                "onnxruntime is not installed or could not be imported."
                "\n\nInstall one of:\n"
                "  pip install onnxruntime  (CPU)\n"
                "  pip install onnxruntime-directml  (Windows)\n"
                "  pip install onnxruntime-gpu  (CUDA)"
                + extra
            )
            return

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
        req = _model_required_patch(self._model_path)
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
        self._log(f"🚀 Aberration AI: model={os.path.basename(self._model_path)}, "
                  f"provider={prov_txt}, patch={patch}, overlap={overlap}")

        # -------- run worker --------
        self.progress.setValue(0)
        self.btn_run.setEnabled(False)

        self._worker = _ONNXWorker(self._model_path, img, patch, overlap, providers)
        self._worker.progressed.connect(self.progress.setValue)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished_ok.connect(self._on_ok)
        self._worker.finished.connect(lambda: self.btn_run.setEnabled(True))
        self._worker.start()


    def _on_failed(self, msg: str):
        self._log(f"❌ Aberration AI failed: {msg}")   # NEW
        QMessageBox.critical(self, "ONNX Error", msg)

    def _on_ok(self, out: np.ndarray):
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
                    "model_path": self._model_path,
                    "patch_size": int(self.spin_patch.value()),
                    "overlap": int(self.spin_overlap.value()),
                    "provider": (self.cmb_provider.currentText()
                                if not self.chk_auto.isChecked() else "auto"),
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

        # 4) Refresh the active view
        mw = self.parent()
        sw = getattr(getattr(mw, "mdi", None), "activeSubWindow", lambda: None)()
        if sw and hasattr(sw, "widget"):
            w = sw.widget()
            if hasattr(w, "reload_from_doc"):
                try: w.reload_from_doc()
                except Exception: pass
            elif hasattr(w, "update_view"):
                try: w.update_view()
                except Exception: pass
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
            f"(model={os.path.basename(self._model_path)}, provider={used}, "
            f"patch={int(self.spin_patch.value())}, overlap={int(self.spin_overlap.value())}, "
            f"border={BORDER_PX}px, time={dt:.2f}s)"
        )


        self.progress.setValue(100)
        self.accept()