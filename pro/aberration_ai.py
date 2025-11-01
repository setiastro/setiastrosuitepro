# pro/aberration_ai.py
from __future__ import annotations
import os, webbrowser, requests
import numpy as np
import sys, platform
import time
import importlib
import subprocess
import traceback
import contextlib

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QStandardPaths, QSettings
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QComboBox, QSpinBox, QProgressBar, QMessageBox, QCheckBox
)
from PyQt6.QtGui import QIcon

__all__ = [
    "AberrationAIDialog",
    "_probe_onnxruntime",
    "pick_providers",
]

# IMPORTANT: do NOT import onnxruntime at module load time – it can crash
# the process on some Windows setups with “bad” PATHs.
ort = None  # filled lazily

IS_APPLE_ARM = (sys.platform == "darwin" and platform.machine() == "arm64")

try:
    from importlib import metadata as _importlib_metadata
except Exception:
    _importlib_metadata = None

# will be filled by _probe_onnxruntime()
_ORT_FLAVOR: str | None = None         # e.g. "onnxruntime", "onnxruntime-directml"
_ORT_IMPORT_ERR: str | None = None     # last import error string (full traceback)
_ORT_FOUND_DISTS: list[str] = []       # names we found via importlib.metadata

# ---------- GitHub model fetching ----------
GITHUB_REPO = "riccardoalberghi/abberation_models"
LATEST_API  = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"


# ------------------------------------------------------------
# PATH sanitizer – so we only import from *this* venv on Windows
# ------------------------------------------------------------
@contextlib.contextmanager
def _minimal_windows_path_for_ort():
    if os.name != "nt":
        # non-Windows: do nothing
        yield
        return

    old_path = os.environ.get("PATH", "")
    py_dir   = os.path.dirname(sys.executable)
    winroot  = os.environ.get("SystemRoot", r"C:\Windows")
    new_parts = [
        py_dir,
        os.path.join(winroot, "System32"),
        winroot,
    ]
    os.environ["PATH"] = os.pathsep.join(p for p in new_parts if p)
    try:
        yield
    finally:
        os.environ["PATH"] = old_path


def _app_model_dir() -> str:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    if not base:
        base = os.path.expanduser("~/.local/share/SetiAstro")
    d = os.path.join(base, "Models", "aberration_ai")
    os.makedirs(d, exist_ok=True)
    return d


def _win_try_add_ort_dirs():
    """
    Windows-only: add likely onnxruntime DLL dirs to the search path.
    This does nothing on non-Windows.
    """
    if os.name != "nt":
        return

    candidates: list[str] = []
    # common venv location
    candidates.append(os.path.join(sys.prefix, "Lib", "site-packages", "onnxruntime", "capi"))
    candidates.append(os.path.join(sys.prefix, "Lib", "site-packages", "onnxruntime_directml", "capi"))

    # try to find via metadata
    if _importlib_metadata:
        for dist_name in ("onnxruntime", "onnxruntime-directml"):
            try:
                files = _importlib_metadata.files(dist_name) or []
            except Exception:
                files = []
            for f in files:
                p = os.path.join(sys.prefix, "Lib", "site-packages", str(f))
                if p.lower().endswith(os.path.join("onnxruntime", "capi").lower()):
                    candidates.append(p)

    seen = set()
    for p in candidates:
        p = os.path.abspath(p)
        if p in seen:
            continue
        seen.add(p)
        if os.path.isdir(p):
            try:
                os.add_dll_directory(p)
            except Exception:
                pass


def _subproc_can_import_ort() -> tuple[bool, str]:
    """
    Run a tiny Python in a separate process to see if `import onnxruntime` works
    in THIS venv. Returns (ok, output).
    On Windows we also give the subprocess the *sanitized* PATH so the check
    matches the in-process import conditions.
    """
    code = r"""
import sys
try:
    import onnxruntime as ort
except Exception as e:
    print("ERR", repr(e))
    sys.exit(1)
else:
    print("OK", ort.__version__, ort.get_available_providers())
    sys.exit(0)
"""
    env = None
    if os.name == "nt":
        # give subprocess the same short PATH we will use in-process
        py_dir   = os.path.dirname(sys.executable)
        winroot  = os.environ.get("SystemRoot", r"C:\Windows")
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join([
            py_dir,
            os.path.join(winroot, "System32"),
            winroot,
        ])

    try:
        cp = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            env=env,
        )
    except Exception as e:
        # couldn't even run python
        return False, f"subprocess failed to run: {e!r}"

    out = (cp.stdout or "") + (cp.stderr or "")
    return (cp.returncode == 0, out.strip())


def _probe_onnxruntime():
    """
    Try VERY HARD to get an onnxruntime module without crashing the process.
    1. Collect which dists are installed (for UI)
    2. On Windows: try in a subprocess FIRST (safe)
    3. On Windows: if subprocess OK → do an in-process import but with a SHRUNK PATH
    4. Add DLL dirs on Windows and try again
    5. Try alt module names
    """
    global ort, _ORT_FLAVOR, _ORT_IMPORT_ERR, _ORT_FOUND_DISTS

    if ort is not None:
        return ort

    # 1) collect installed dists (for the dialog)
    installed = {}
    _ORT_FOUND_DISTS = []
    if _importlib_metadata:
        for dist_name in ("onnxruntime", "onnxruntime-directml", "onnxruntime-gpu"):
            try:
                ver = _importlib_metadata.version(dist_name)
                installed[dist_name] = ver
            except Exception:
                pass
    _ORT_FOUND_DISTS = [f"{k}=={v}" for k, v in installed.items()]

    # if BOTH cpu + dml are in the same venv → tell the user explicitly
    if "onnxruntime" in installed and "onnxruntime-directml" in installed:
        _ORT_IMPORT_ERR = (
            "Both 'onnxruntime' (CPU) and 'onnxruntime-directml' (Windows GPU) are "
            "installed in this venv. They provide the same Python package and the "
            "native DLL can refuse to initialize on Windows.\n\n"
            f'Uninstall ONE of them, for example:\n  "{sys.executable}" -m pip uninstall -y onnxruntime-directml\n'
            "Then re-open this tool."
        )
        return None

    # 2) Windows: run the SAFE subprocess check first
    if os.name == "nt":
        ok, out = _subproc_can_import_ort()
        if not ok:
            # subprocess couldn’t import → don’t try in-process, just report it
            _ORT_IMPORT_ERR = f"Subprocess import check failed:\n{out}"
            return None

        # 3) subprocess worked → try in-process, but with a CLEAN PATH
        try:
            with _minimal_windows_path_for_ort():
                import onnxruntime as _ort
            ort = _ort
            _ORT_FLAVOR = "onnxruntime"
            _ORT_IMPORT_ERR = None
            return ort
        except Exception as e1:
            _ORT_IMPORT_ERR = (
                "in-process import onnxruntime (with sanitized PATH) failed: "
                f"{e1!r}\n{traceback.format_exc()}"
            )
            ort = None
    else:
        # non-Windows: normal import
        try:
            import onnxruntime as _ort
            ort = _ort
            _ORT_FLAVOR = "onnxruntime"
            _ORT_IMPORT_ERR = None
            return ort
        except Exception as e1:
            _ORT_IMPORT_ERR = (
                f"in-process import onnxruntime failed: {e1!r}\n"
                f"{traceback.format_exc()}"
            )
            ort = None

    # 4) Windows: try adding DLL dirs and import again
    if os.name == "nt":
        _win_try_add_ort_dirs()
        try:
            with _minimal_windows_path_for_ort():
                import onnxruntime as _ort  # retry after DLL add
            ort = _ort
            _ORT_FLAVOR = "onnxruntime (after DLL add)"
            _ORT_IMPORT_ERR = None
            return ort
        except Exception as e2:
            _ORT_IMPORT_ERR = (
                "in-process import onnxruntime (after DLL add) failed: "
                f"{e2!r}\n{traceback.format_exc()}"
            )
            ort = None

    # 5) try alternate module names people install
    for alt_mod in ("onnxruntime_directml", "onnxruntime_gpu"):
        try:
            _alt = importlib.import_module(alt_mod)
            sys.modules.setdefault("onnxruntime", _alt)
            ort = _alt
            _ORT_FLAVOR = alt_mod
            _ORT_IMPORT_ERR = None
            return ort
        except Exception:
            pass

    return None


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


class _DownloadWorker(QThread):
    progressed = pyqtSignal(int)
    failed     = pyqtSignal(str)
    finished_ok= pyqtSignal(str)

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
    channels_last = (img.ndim == 3)
    if channels_last:
        arr = img.transpose(2,0,1)
    else:
        arr = img[np.newaxis, ...]
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
        arr = arr.transpose(1,2,0)
    else:
        arr = arr[0]
    return arr


def run_onnx_tiled(session, img: np.ndarray, patch_size=512, overlap=64, progress_cb=None) -> np.ndarray:
    arr, channels_last, was_uint16 = _prepare_input(img)
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
                patch = arr[c:c+1, i:i+patch_size, j:j+patch_size]
                inp = np.ascontiguousarray(patch[np.newaxis, ...], dtype=np.float32)
                out_patch = session.run(None, {inp_name: inp})[0]
                out_patch = np.squeeze(out_patch, axis=0)
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
    rt = _probe_onnxruntime()
    if rt is None:
        return []

    avail = set(rt.get_available_providers())

    if IS_APPLE_ARM:
        return ["CPUExecutionProvider"] if "CPUExecutionProvider" in avail else []

    if not auto_gpu:
        return ["CPUExecutionProvider"] if "CPUExecutionProvider" in avail else []

    order = []
    if "DmlExecutionProvider" in avail:
        order.append("DmlExecutionProvider")
    if "CUDAExecutionProvider" in avail:
        order.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in avail:
        order.append("CPUExecutionProvider")
    if "CoreMLExecutionProvider" in avail:
        order.append("CoreMLExecutionProvider")
    return order


def _preserve_border(dst: np.ndarray, src: np.ndarray, px: int = 10) -> np.ndarray:
    if px <= 0 or dst is None or src is None:
        return dst
    if dst.shape != src.shape:
        return dst

    h, w = dst.shape[:2]
    px = int(max(0, min(px, h // 2, w // 2)))
    if px == 0:
        return dst

    s = src.astype(dst.dtype, copy=False)
    dst[:px, ...]  = s[:px, ...]
    dst[-px:, ...] = s[-px:, ...]
    dst[:, :px, ...]  = s[:, :px, ...]
    dst[:, -px:, ...] = s[:, -px:, ...]
    return dst


# ---------- worker ----------
class _ONNXWorker(QThread):
    progressed = pyqtSignal(int)
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
            extra = ""
            if _ORT_FOUND_DISTS:
                extra = (
                    "\n\nI can see these ONNX/ONNXRuntime packages installed, but not in *this* Python:\n  - "
                    + "\n  - ".join(_ORT_FOUND_DISTS)
                    + f"\n\nThis SASpro is running under: {sys.executable}"
                    + "\nIf you only installed the DirectML wheel, also install the CPU one with:\n"
                    f'  "{sys.executable}" -m pip install onnxruntime'
                )
            if _ORT_IMPORT_ERR:
                extra += "\n\nActual import error was:\n" + _ORT_IMPORT_ERR
            self.failed.emit("onnxruntime is not installed or could not be imported." + extra)
            return

        try:
            sess = rt.InferenceSession(self.model_path, providers=self.providers)
            self.used_provider = (sess.get_providers()[0] if sess.get_providers() else None)
        except Exception:
            # try fallback CPU
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
            self.failed.emit(str(e))
            return

        self.finished_ok.emit(out)


# ---------- dialog ----------
class AberrationAIDialog(QDialog):
    def __init__(self, parent, docman, get_active_doc_callable, icon: QIcon | None = None):
        super().__init__(parent)
        self.setWindowTitle("R.A.'s Aberration Correction (AI)")
        if icon is not None:
            self.setWindowIcon(icon)
        self.docman = docman
        self.get_active_doc = get_active_doc_callable
        self._t_start = None
        self._last_used_provider = None

        v = QVBoxLayout(self)

        # model row
        row = QHBoxLayout()
        row.addWidget(QLabel("Model:"))
        self.model_label = QLabel("—")
        self.model_label.setToolTip("")
        btn_browse = QPushButton("Browse…"); btn_browse.clicked.connect(self._browse_model)
        row.addWidget(self.model_label, 1)
        row.addWidget(btn_browse)
        v.addLayout(row)

        # provider row
        row2 = QHBoxLayout()
        self.chk_auto = QCheckBox("Auto GPU (if available)")
        self.chk_auto.setChecked(True)
        row2.addWidget(self.chk_auto)
        self.cmb_provider = QComboBox()
        row2.addWidget(QLabel("Provider:"))
        row2.addWidget(self.cmb_provider, 1)
        v.addLayout(row2)

        # params
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Patch"))
        self.spin_patch = QSpinBox(minimum=128, maximum=2048); self.spin_patch.setValue(512)
        row3.addWidget(self.spin_patch)
        row3.addWidget(QLabel("Overlap"))
        self.spin_overlap = QSpinBox(minimum=16, maximum=512); self.spin_overlap.setValue(64)
        row3.addWidget(self.spin_overlap)
        v.addLayout(row3)

        # download / open
        row4 = QHBoxLayout()
        btn_latest = QPushButton("Download latest model…")
        btn_latest.clicked.connect(self._download_latest_model)
        row4.addWidget(btn_latest)
        btn_openfolder = QPushButton("Open model folder")
        btn_openfolder.clicked.connect(self._open_model_folder)
        row4.addWidget(btn_openfolder)
        row4.addStretch(1)
        v.addLayout(row4)

        # progress + run
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

    def _log(self, msg: str):
        mw = self.parent()
        try:
            if hasattr(mw, "_log"):
                mw._log(msg)
            elif hasattr(mw, "update_status"):
                mw.update_status(msg)
        except Exception:
            pass

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
    def _refresh_providers(self):
        rt = _probe_onnxruntime()
        if rt is None:
            self.cmb_provider.clear()
            if _ORT_FOUND_DISTS:
                self.cmb_provider.addItem("onnxruntime is in another env / broken")
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
            self.cmb_provider.addItem("CPUExecutionProvider")
            self.cmb_provider.setCurrentText("CPUExecutionProvider")
            self.cmb_provider.setEnabled(False)
            self.chk_auto.setChecked(False)
            self.chk_auto.setEnabled(False)
            return

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
            QMessageBox.warning(self, "Network", "The 'requests' package is required.")
            return
        dst = _app_model_dir()
        self.progress.setRange(0, 0)
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
            parts = [
                "onnxruntime is not installed or could not be imported.",
                "",
                "Install one of:",
                "  pip install onnxruntime           (CPU)",
                "  pip install onnxruntime-directml   (Windows GPU - DirectML)",
                "  pip install onnxruntime-gpu        (CUDA)",
                "",
            ]
            if _ORT_FOUND_DISTS:
                parts.append("I found these packages installed:")
                for d in _ORT_FOUND_DISTS:
                    parts.append(f"  - {d}")
                parts.append(f"...but they are not importable from this Python:\n  {sys.executable}")
            if _ORT_IMPORT_ERR:
                parts.append("")
                parts.append("Actual import error was:")
                parts.append(_ORT_IMPORT_ERR)
            parts.append("")
            parts.append("Try this in a terminal:")
            parts.append(f'  "{sys.executable}" -c "import onnxruntime; print(onnxruntime.__version__)"')

            QMessageBox.critical(self, "Missing dependency", "\n".join(parts))
            return

        doc = self.get_active_doc()
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.warning(self, "Image", "No active image.")
            return

        img = np.asarray(doc.image)
        self._orig_for_border = img.copy()

        patch   = int(self.spin_patch.value())
        overlap = int(self.spin_overlap.value())

        if IS_APPLE_ARM:
            providers = ["CPUExecutionProvider"]
            self.chk_auto.setChecked(False)
        else:
            if self.chk_auto.isChecked():
                providers = pick_providers(auto_gpu=True)
            else:
                sel = self.cmb_provider.currentText()
                providers = [sel] if sel else ["CPUExecutionProvider"]

        req = _model_required_patch(self._model_path)
        if req and req > 0:
            patch = req
            try:
                self.spin_patch.blockSignals(True)
                self.spin_patch.setValue(req)
            finally:
                self.spin_patch.blockSignals(False)

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

        self.progress.setValue(0)
        self.btn_run.setEnabled(False)

        self._worker = _ONNXWorker(self._model_path, img, patch, overlap, providers)
        self._worker.progressed.connect(self.progress.setValue)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished_ok.connect(self._on_ok)
        self._worker.finished.connect(lambda: self.btn_run.setEnabled(True))
        self._worker.start()

    def _on_failed(self, msg: str):
        self._log(f"❌ Aberration AI failed: {msg}")
        QMessageBox.critical(self, "ONNX Error", msg)

    def _on_ok(self, out: np.ndarray):
        doc = self.get_active_doc()
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.warning(self, "Image", "No active image.")
            return

        BORDER_PX = 10
        src = getattr(self, "_orig_for_border", None)
        if src is None or src.shape != out.shape:
            try:
                src = np.asarray(doc.image)
            except Exception:
                src = None
        out = _preserve_border(out, src, BORDER_PX)

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

        try:
            if hasattr(doc, "apply_edit"):
                doc.apply_edit(out, meta, step_name="Aberration AI")
            elif hasattr(self.docman, "update_active_document"):
                self.docman.update_active_document(out, metadata=meta, step_name="Aberration AI")
            else:
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

        self._log(
            f"✅ Aberration AI applied (model={os.path.basename(self._model_path)}, "
            f"provider={used}, patch={int(self.spin_patch.value())}, "
            f"overlap={int(self.spin_overlap.value())}, border={BORDER_PX}px, "
            f"time={dt:.2f}s)"
        )

        self.progress.setValue(100)
        self.accept()
