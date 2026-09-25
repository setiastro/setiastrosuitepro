# src/setiastro/saspro/syqon_studio.py
"""
SyQon Studio integration for SASpro.

This is the consolidated "Studio" path for the SyQon Tools hub. Instead of the
legacy per-tool integrated PyTorch engines (Prism / Parallax / Axiom / Starless),
this drives SyQon's single all-inclusive local **neural CLI** that ships inside
SyQon Studio (see https://syqon.eu/develop).

The CLI runs every SyQon neural process behind one stable, versioned contract:

    syqon-cli --model MODEL [OPTIONS] INPUT OUTPUT

Design notes / invariants (per the published developer contract):

  * The executable is discovered, never assumed to be on PATH. Order:
      1. previously validated path (QSettings)
      2. $SYQON_CLI_PATH
      3. standard install location for the platform
      4. Windows App Paths registry key
      5. user-selected path (Locate…)
    It is validated with `--version` (exit 0) before use.

  * Entitlement is device-bound to the signed-in Studio account. Flags never
    unlock a model; a licensed model the account lacks fails with exit 4. We
    surface that as an "authenticate in Studio" message rather than an error.

  * We never touch model weights. We select a public identifier (e.g.
    `prism-essential`) and the licensed CLI resolves the installed asset.

  * Process contract:
      - shell disabled, argv array
      - stdout on exit 0 == absolute saved-output path + newline
      - stderr == progress ("model 42%") + diagnostics
      - exit code carries the failure class (0..7, 130)
      - SIGINT/terminate cancels; no partial output is published

  * SASpro documents are in-memory float arrays, so we round-trip through a
    32-bit-float temp file (the same idiom the legacy standalone Starless path
    uses) and let the CLI own all preprocessing / tiling / reconstruction. No
    SASpro-side MTF stretch is applied on this path — that is the whole point
    of the consolidation.

This module is intentionally standalone. `syqon_tools.py` imports the page
lazily so there is no import cycle (this module imports a couple of shared
helpers from `syqon_tools` at load time; `syqon_tools` must not import this at
module top level).
"""
from __future__ import annotations

import os
import re
import sys
import time
import shutil
import tempfile
import subprocess
import threading
from pathlib import Path

import numpy as np

from PyQt6.QtCore import Qt, QSettings, QUrl, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap, QDesktopServices
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QGroupBox,
    QLabel, QPushButton, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QSlider, QLineEdit, QProgressBar, QStackedWidget, QMessageBox, QFileDialog,
)

from setiastro.saspro.legacy.image_manager import save_image, load_image

# Shared helpers from the hub. syqon_tools imports THIS module lazily, so this
# top-level import does not create a cycle.
from setiastro.saspro.syqon_tools import _WorkerCloseGuardMixin, _blend_result_with_mask

# Stars-only / gradient side-products are pushed into a new view.
from setiastro.saspro.remove_stars import _push_as_new_doc


# ─────────────────────────────────────────────────────────────────────────────
# Model registry — stable CLI identifiers (from --list-models / the dev docs).
# Order here is the dropdown order. `family` drives which option page shows.
# ─────────────────────────────────────────────────────────────────────────────
# (identifier, friendly label, access, family, input contract note)
STUDIO_MODELS: list[tuple[str, str, str, str, str]] = [
    ("prism-essential", "Prism Essential — Denoise",       "Included", "prism",         "Linear RGB or mono"),
    ("prism-advanced",  "Prism Deep Advanced — Denoise",   "Licensed", "prism",         "Linear RGB or mono"),
    ("prism-ultra",     "Prism Deep Ultra — Denoise",      "Licensed", "prism",         "Linear or non-linear"),
    ("prism-max",       "Prism Deep Max — Denoise",        "Licensed", "prism",         "Linear or non-linear"),
    ("prism-legacy-v1", "Prism Legacy V1 — Denoise",       "Licensed", "prism",         "Legacy display-domain input"),
    ("prism-legacy-v2", "Prism Legacy V2 — Denoise",       "Licensed", "prism",         "Robust linear input"),
    ("parallax",        "Parallax — Correct / Reduce / Deblur", "Licensed", "parallax",  "Linear or non-linear"),
    ("axiom",           "Axiom V3 — Starless",             "Licensed", "axiom",         "Linear (Axiom stretch) or non-linear"),
    ("deep-gradient",   "Deep Gradient (Untested Beta) — Gradient removal", "Included", "deep-gradient", "Linear RGB or mono"),
]

_MODEL_INDEX = {m[0]: m for m in STUDIO_MODELS}


def studio_family_for(model_id: str) -> str:
    m = _MODEL_INDEX.get(str(model_id))
    return m[3] if m else "prism"


_STUDIO_BUY_URL = "https://syqon.eu/syqonstudio"


# ─────────────────────────────────────────────────────────────────────────────
# CLI discovery / validation
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_bundle(path: str) -> str:
    """If given a macOS .app bundle, resolve to the syqon-cli binary inside it."""
    if not path:
        return path
    if sys.platform == "darwin" and path.endswith(".app") and os.path.isdir(path):
        inner = os.path.join(path, "Contents", "MacOS", "syqon-cli")
        if os.path.isfile(inner):
            return inner
    return path


def _windows_app_path(exe_name: str = "syqon-cli.exe") -> str | None:
    if os.name != "nt":
        return None
    try:
        import winreg  # type: ignore
        key = r"Software\Microsoft\Windows\CurrentVersion\App Paths\\" + exe_name
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as k:
            val, _ = winreg.QueryValueEx(k, "")
            val = str(val or "").strip().strip('"')
            return val or None
    except Exception:
        return None


def studio_cli_candidates() -> list[str]:
    """Ordered, de-duplicated list of paths to probe (env → standard → registry)."""
    out: list[str] = []

    env = os.environ.get("SYQON_CLI_PATH", "").strip()
    if env:
        out.append(env)

    if sys.platform == "darwin":
        out.append("/Applications/SyQon Studio.app/Contents/MacOS/syqon-cli")
        out.append(str(Path.home() / "Applications" / "SyQon Studio.app" / "Contents" / "MacOS" / "syqon-cli"))
    elif os.name == "nt":
        reg = _windows_app_path()
        if reg:
            out.append(reg)
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            out.append(str(Path(local) / "Programs" / "SyQon Studio" / "syqon-cli.exe"))
    else:
        # No official Linux install location is documented yet; rely on
        # $SYQON_CLI_PATH and the user-selected fallback.
        pass

    # de-dup, keep order
    seen: set[str] = set()
    uniq: list[str] = []
    for p in out:
        p = _resolve_bundle(p.strip())
        if p and p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def validate_studio_cli(path: str, timeout: float = 20.0) -> tuple[bool, str]:
    """Run `<cli> --version`. Returns (ok, version_or_error_text)."""
    path = _resolve_bundle((path or "").strip())
    if not path or not os.path.isfile(path):
        return False, "No such file"
    try:
        r = subprocess.run(
            [path, "--version"],
            capture_output=True, text=True, timeout=timeout,
            **({"creationflags": 0x08000000} if os.name == "nt" else {}),  # CREATE_NO_WINDOW
        )
        text = (r.stdout or r.stderr or "").strip().splitlines()
        ver = text[0].strip() if text else ""
        return (r.returncode == 0), (ver or f"exit {r.returncode}")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def list_studio_models(path: str, timeout: float = 20.0) -> set[str]:
    """
    Best-effort `--list-models`. Returns the set of known identifiers that
    appear in the CLI's output (so the UI can annotate what's actually
    installed/entitled). Empty set means "couldn't determine".
    """
    path = _resolve_bundle((path or "").strip())
    if not path or not os.path.isfile(path):
        return set()
    try:
        r = subprocess.run(
            [path, "--list-models"],
            capture_output=True, text=True, timeout=timeout,
            **({"creationflags": 0x08000000} if os.name == "nt" else {}),
        )
        blob = (r.stdout or "") + "\n" + (r.stderr or "")
        found = {mid for mid in _MODEL_INDEX if re.search(rf"(?<![\w-]){re.escape(mid)}(?![\w-])", blob)}
        return found
    except Exception:
        return set()


def find_studio_cli(settings: QSettings | None = None) -> str | None:
    """Return the first candidate (incl. remembered path) that validates."""
    cands: list[str] = []
    if settings is not None:
        remembered = str(settings.value("syqon/studio/cli_path", "", type=str) or "").strip()
        if remembered:
            cands.append(remembered)
    cands.extend(studio_cli_candidates())

    seen: set[str] = set()
    for c in cands:
        c = _resolve_bundle(c)
        if not c or c in seen:
            continue
        seen.add(c)
        ok, _ = validate_studio_cli(c)
        if ok:
            return c
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Argument construction (pure function — unit-testable without Qt)
# ─────────────────────────────────────────────────────────────────────────────
def build_studio_argv(
    exe: str,
    model_id: str,
    in_path: str,
    out_path: str,
    *,
    domain: str = "auto",
    tile: int | None = None,
    overlap: int | None = None,
    application: float | None = None,
    # parallax
    parallax_family: str = "aesthetics",
    parallax_correction: bool = True,
    parallax_reduction: bool = True,
    parallax_reduction_level: int = 5,
    parallax_deblur: bool = True,
    parallax_deblur_strength: float = 0.5,
    # axiom
    axiom_stretch: str = "auto",
    axiom_black: float | None = None,
    axiom_mid: float | None = None,
    axiom_white: float | None = None,
    stars_output: str | None = None,
    # deep gradient
    gradient_output: str | None = None,
    precision: str = "f32",
    overwrite: bool = True,
) -> list[str]:
    """
    Build the full argv for one Studio CLI invocation. `model_id` chooses which
    model-specific flags are emitted. Positional INPUT/OUTPUT come last, per the
    canonical `syqon-cli --model MODEL [OPTIONS] INPUT OUTPUT`.
    """
    family = studio_family_for(model_id)
    argv: list[str] = [exe, "--model", model_id]

    # Domain: pass explicitly (auto lets the CLI read container metadata).
    if domain:
        argv += ["--domain", str(domain)]

    # Output sample representation. We keep the in-app round-trip at f32.
    argv += ["--precision", str(precision)]

    # Tile / overlap are advisory for fixed-contract models; Deep Gradient
    # manages its own fixed image contract, so we don't send them there.
    if family in ("prism", "parallax"):
        if tile is not None:
            argv += ["--tile-size", str(int(tile))]
        if overlap is not None:
            argv += ["--overlap", str(int(overlap))]

    # Blend inferred result with original — meaningful for Prism.
    if family == "prism" and application is not None:
        argv += ["--application", f"{float(application):.4f}"]

    if family == "parallax":
        argv += ["--family", str(parallax_family)]
        argv += ["--correction", "true" if parallax_correction else "false"]
        argv += ["--reduction", "true" if parallax_reduction else "false"]
        if parallax_reduction:
            argv += ["--reduction-level", str(int(parallax_reduction_level))]
        argv += ["--deblur", "true" if parallax_deblur else "false"]
        if parallax_deblur:
            argv += ["--deblur-strength", f"{float(parallax_deblur_strength):.4f}"]

    elif family == "axiom":
        argv += ["--axiom-stretch", str(axiom_stretch)]
        if axiom_stretch == "custom":
            if axiom_black is not None:
                argv += ["--axiom-black", f"{float(axiom_black):.6f}"]
            if axiom_mid is not None:
                argv += ["--axiom-mid", f"{float(axiom_mid):.6f}"]
            if axiom_white is not None:
                argv += ["--axiom-white", f"{float(axiom_white):.6f}"]
        if stars_output:
            argv += ["--stars-output", str(stars_output)]

    elif family == "deep-gradient":
        if gradient_output:
            argv += ["--gradient-output", str(gradient_output)]

    if overwrite:
        argv += ["--overwrite"]

    argv += [str(in_path), str(out_path)]
    return argv


# Failure-contract exit codes → human text (developer docs §"Failure contract").
_EXIT_MESSAGES = {
    1: "SyQon CLI reported a general error. The original image was kept.",
    2: ("Invalid option or incompatible domain. Try declaring the input domain "
        "explicitly (Linear / Non-linear) — some models require linear input."),
    3: "Input/output path error (existence, permissions, or overwrite policy).",
    4: ("Authentication or entitlement denied. Sign in through SyQon Studio and "
        "confirm this model is included with your account."),
    5: "Inference failed — no output was produced.",
    6: "Output encoding failed. Try a different precision or destination.",
    7: "Output was saved, but the application handoff step failed.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Worker thread
# ─────────────────────────────────────────────────────────────────────────────
class _SyQonStudioCLIThread(QThread):
    # (primary_rgb01 | None, secondary_rgb01 | None, info: dict, err: str)
    # err == "__cancelled__" for a user cancel.
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, object, dict, str)

    def __init__(self, *, input_rgb01, scale_factor, exe, model_id, argv_opts: dict,
                 want_secondary: bool, secondary_flag: str | None, parent=None):
        super().__init__(parent)
        self.input_rgb01 = np.asarray(input_rgb01, dtype=np.float32)
        self.scale_factor = float(scale_factor)
        self.exe = str(exe)
        self.model_id = str(model_id)
        self.argv_opts = dict(argv_opts or {})
        self.want_secondary = bool(want_secondary)
        self.secondary_flag = secondary_flag  # "--stars-output" | "--gradient-output" | None
        self._cancel = False
        self._p = None
        self._tmp_in = None
        self._tmp_out = None
        self._tmp_sec = None

    def cancel(self):
        self._cancel = True
        p = self._p
        try:
            if p is not None and p.poll() is None:
                # SIGTERM on POSIX; terminate() maps to TerminateProcess on Windows.
                p.terminate()
        except Exception:
            pass

    def _emit(self, pct: int, stage: str):
        self.progress.emit(int(np.clip(pct, 0, 100)), str(stage or ""))

    def _load_rgb01(self, path: str):
        arr, _, _, _ = load_image(path)
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        else:
            arr = arr[..., :3]
        if self.scale_factor > 1.01:
            arr = arr * self.scale_factor
        return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)

    def run(self):
        info: dict = {"engine": "syqon_studio_cli", "model": self.model_id, "exe": self.exe}
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        re_pct = re.compile(r"(\d+)\s*%")

        def _reader(pipe, sink, is_stderr):
            try:
                for raw in pipe:
                    line = raw.rstrip("\r\n")
                    sink.append(line)
                    print(f"[SyQon Studio {'err' if is_stderr else 'out'}] {line}", flush=True)
                    if is_stderr:
                        m = re_pct.search(line)
                        if m:
                            self._emit(min(int(m.group(1)), 99), "SyQon Studio processing…")
            except Exception as e:
                print(f"[SyQon Studio reader] {type(e).__name__}: {e}", flush=True)

        try:
            if self._cancel:
                raise RuntimeError("__cancelled__")

            exe = _resolve_bundle(self.exe)
            if not exe or not os.path.isfile(exe):
                raise RuntimeError(f"SyQon CLI not found:\n{self.exe}")

            tag = f"studio_{os.getpid()}_{int(time.time() * 1000)}"
            tmpdir = Path(tempfile.gettempdir()) / "SASpro_SyQon_Studio"
            tmpdir.mkdir(parents=True, exist_ok=True)
            self._tmp_in = str(tmpdir / f"{tag}_in.tif")
            self._tmp_out = str(tmpdir / f"{tag}_out.tif")
            if self.want_secondary and self.secondary_flag:
                self._tmp_sec = str(tmpdir / f"{tag}_sec.tif")

            self._emit(2, "Writing temp image…")
            save_image(
                self.input_rgb01, self._tmp_in,
                original_format="tif", bit_depth="32-bit floating point",
                original_header=None, is_mono=False, image_meta=None, file_meta=None,
            )
            if self._cancel:
                raise RuntimeError("__cancelled__")

            opts = dict(self.argv_opts)
            if self.want_secondary and self.secondary_flag == "--stars-output":
                opts["stars_output"] = self._tmp_sec
            elif self.want_secondary and self.secondary_flag == "--gradient-output":
                opts["gradient_output"] = self._tmp_sec

            argv = build_studio_argv(exe, self.model_id, self._tmp_in, self._tmp_out, **opts)
            info["command"] = argv
            print("\n[SyQon Studio launch]\nCMD:", " ".join(argv), flush=True)
            self._emit(5, "Launching SyQon CLI…")

            self._p = subprocess.Popen(
                argv,
                cwd=str(Path(exe).parent),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                text=True, bufsize=1, universal_newlines=True,
                **({"creationflags": 0x08000000} if os.name == "nt" else {}),  # CREATE_NO_WINDOW
            )

            t_out = threading.Thread(target=_reader, args=(self._p.stdout, stdout_lines, False), daemon=True)
            t_err = threading.Thread(target=_reader, args=(self._p.stderr, stderr_lines, True), daemon=True)
            t_out.start(); t_err.start()

            self._p.wait()
            t_out.join(timeout=2.0)
            t_err.join(timeout=2.0)

            rc = self._p.returncode
            info["returncode"] = rc
            info["stdout_tail"] = "\n".join(stdout_lines[-100:])
            info["stderr_tail"] = "\n".join(stderr_lines[-200:])

            if self._cancel or rc == 130:
                raise RuntimeError("__cancelled__")

            if rc != 0:
                msg = _EXIT_MESSAGES.get(rc, f"SyQon CLI exited with code {rc}.")
                tail = "\n".join(stderr_lines[-6:]).strip()
                if tail:
                    msg = f"{msg}\n\n{tail}"
                raise RuntimeError(msg)

            # stdout is the absolute saved-output path; prefer it, fall back to ours.
            declared = (stdout_lines[-1].strip() if stdout_lines else "")
            primary_path = declared if (declared and os.path.exists(declared)) else self._tmp_out
            if not os.path.exists(primary_path):
                raise RuntimeError("SyQon CLI reported success but no output file was found.")

            self._emit(96, "Loading result…")
            primary = self._load_rgb01(primary_path)
            if primary is None:
                raise RuntimeError("Could not load the SyQon CLI output.")

            secondary = None
            if self.want_secondary and self._tmp_sec and os.path.exists(self._tmp_sec):
                try:
                    secondary = self._load_rgb01(self._tmp_sec)
                except Exception:
                    secondary = None

            self._emit(100, "Complete")
            self.finished.emit(primary, secondary, info, "")

        except Exception as e:
            import traceback
            info["traceback"] = traceback.format_exc()
            msg = str(e)
            self.finished.emit(None, None, info, msg)

        finally:
            try:
                if self._p is not None and self._p.poll() is None:
                    self._p.terminate()
            except Exception:
                pass
            for p in (self._tmp_in, self._tmp_out, self._tmp_sec):
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# Hub page
# ─────────────────────────────────────────────────────────────────────────────
class _SyQonStudioHubPage(_WorkerCloseGuardMixin, QWidget):
    _WORKER_LABEL = "SyQon Studio"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc_thr = None
        self.doc = None
        self.main = None
        self._scale_factor = 1.0
        self._orig_was_mono = False
        self.settings = QSettings()

        lay = QVBoxLayout(self)
        lay.setSpacing(6)

        # Compact header. The full-size SyQon logo (a ~240px black wordmark
        # square) ate too much vertical space, so we show a small icon beside a
        # text wordmark instead. If the icon asset doesn't load, the text alone
        # still reads fine.
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(8)
        self.lbl_logo = QLabel(self)
        try:
            from setiastro.saspro.resources import syqon_path
            pm = QPixmap(syqon_path)
            if not pm.isNull():
                self.lbl_logo.setPixmap(pm.scaledToHeight(24, Qt.TransformationMode.SmoothTransformation))
                hdr.addWidget(self.lbl_logo)
        except Exception:
            pass
        _title = QLabel("SyQon Studio", self)
        _title.setStyleSheet("font-size:15px; font-weight:600;")
        hdr.addWidget(_title)
        hdr.addStretch(1)
        _hdr_wrap = QWidget(self)
        _hdr_wrap.setLayout(hdr)
        lay.addWidget(_hdr_wrap)

        # ── CLI discovery box ──────────────────────────────────────────────
        cli_box = QGroupBox("SyQon Studio CLI", self)
        cli_form = QFormLayout(cli_box)

        self.edt_cli = QLineEdit(self)
        self.edt_cli.setPlaceholderText("Path to syqon-cli (auto-detected from SyQon Studio)")
        self.btn_locate = QPushButton("Locate…", self)
        self.btn_validate = QPushButton("Validate", self)
        self.btn_locate.clicked.connect(self._locate_cli)
        self.btn_validate.clicked.connect(self._validate_cli)

        cli_row = QHBoxLayout()
        cli_row.addWidget(self.edt_cli, 1)
        cli_row.addWidget(self.btn_locate)
        cli_row.addWidget(self.btn_validate)
        cli_wrap = QWidget(self); cli_wrap.setLayout(cli_row)
        cli_form.addRow("Executable:", cli_wrap)

        self.lbl_cli_status = QLabel("", self)
        self.lbl_cli_status.setWordWrap(True)
        self.lbl_cli_status.setStyleSheet("color:#8e8e93;")
        cli_form.addRow("", self.lbl_cli_status)

        self.btn_get_studio = QPushButton("Get SyQon Studio…", self)
        self.btn_get_studio.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(_STUDIO_BUY_URL))
        )
        get_row = QHBoxLayout()
        get_row.addWidget(self.btn_get_studio)
        get_row.addStretch(1)
        get_wrap = QWidget(self); get_wrap.setLayout(get_row)
        cli_form.addRow("", get_wrap)

        lay.addWidget(cli_box)

        # ── Model + options box ────────────────────────────────────────────
        box = QGroupBox("Neural process", self)
        form = QFormLayout(box)

        self.cmb_model = QComboBox(self)
        for mid, label, access, _fam, _note in STUDIO_MODELS:
            tag = "" if access == "Included" else "  ·  Licensed"
            self.cmb_model.addItem(f"{label}{tag}", userData=mid)
        self.cmb_model.currentIndexChanged.connect(self._on_model_changed)
        form.addRow("Model:", self.cmb_model)

        self.lbl_contract = QLabel("", self)
        self.lbl_contract.setWordWrap(True)
        self.lbl_contract.setStyleSheet("color:#8e8e93; font-size:11px;")
        form.addRow("", self.lbl_contract)

        # Common options
        self.cmb_domain = QComboBox(self)
        self.cmb_domain.addItem("Auto (read metadata)", userData="auto")
        self.cmb_domain.addItem("Linear", userData="linear")
        self.cmb_domain.addItem("Non-linear", userData="nonlinear")
        form.addRow("Input domain:", self.cmb_domain)

        self.spin_tile = QSpinBox(self)
        self.spin_tile.setRange(256, 2048)
        self.spin_tile.setSingleStep(64)
        self.spin_tile.setValue(int(self.settings.value("syqon/studio/tile", 512)))
        form.addRow("Tile size:", self.spin_tile)

        self.spin_overlap = QSpinBox(self)
        self.spin_overlap.setRange(0, 512)
        self.spin_overlap.setValue(int(self.settings.value("syqon/studio/overlap", 64)))
        form.addRow("Overlap:", self.spin_overlap)

        # Application (blend) — Prism only
        saved_app = float(self.settings.value("syqon/studio/application", 1.0))
        self.sld_app = QSlider(Qt.Orientation.Horizontal, self)
        self.sld_app.setRange(0, 100)
        self.sld_app.setValue(int(round(saved_app * 100)))
        self.spin_app = QDoubleSpinBox(self)
        self.spin_app.setRange(0.0, 1.0)
        self.spin_app.setSingleStep(0.05)
        self.spin_app.setDecimals(2)
        self.spin_app.setValue(saved_app)
        self._sync_app = False

        def _a_s2b(v):
            if self._sync_app:
                return
            self._sync_app = True; self.spin_app.setValue(v / 100.0); self._sync_app = False

        def _a_b2s(v):
            if self._sync_app:
                return
            self._sync_app = True; self.sld_app.setValue(int(round(v * 100))); self._sync_app = False

        self.sld_app.valueChanged.connect(_a_s2b)
        self.spin_app.valueChanged.connect(_a_b2s)
        app_row = QWidget(self)
        app_lay = QHBoxLayout(app_row); app_lay.setContentsMargins(0, 0, 0, 0)
        app_lay.addWidget(self.sld_app, 1); app_lay.addWidget(self.spin_app)
        self.lbl_app = QLabel("Application (blend):", self)
        form.addRow(self.lbl_app, app_row)
        self._app_row_widget = app_row

        lay.addWidget(box)

        # ── Per-model option stack ─────────────────────────────────────────
        self.opt_stack = QStackedWidget(self)
        self.opt_stack.addWidget(self._build_prism_page())
        self.opt_stack.addWidget(self._build_parallax_page())
        self.opt_stack.addWidget(self._build_axiom_page())
        self.opt_stack.addWidget(self._build_gradient_page())
        self._family_page = {"prism": 0, "parallax": 1, "axiom": 2, "deep-gradient": 3}
        lay.addWidget(self.opt_stack)

        # Progress + cancel
        self.pbar = QProgressBar(self); self.pbar.setRange(0, 100); self.pbar.setValue(0)
        self.pbar.setVisible(False)
        lay.addWidget(self.pbar)
        self.lbl_status = QLabel("", self); self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)

        lay.addStretch(1)
        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self._cancel_processing)
        lay.addWidget(self.btn_cancel)

        # restore model
        saved_model = str(self.settings.value("syqon/studio/model", "prism-essential", type=str) or "prism-essential")
        i = self.cmb_model.findData(saved_model)
        if i >= 0:
            self.cmb_model.setCurrentIndex(i)
        j = self.cmb_domain.findData(str(self.settings.value("syqon/studio/domain", "auto", type=str) or "auto"))
        if j >= 0:
            self.cmb_domain.setCurrentIndex(j)

        # auto-detect CLI on open
        remembered = str(self.settings.value("syqon/studio/cli_path", "", type=str) or "").strip()
        if remembered:
            self.edt_cli.setText(remembered)
        else:
            found = find_studio_cli(self.settings)
            if found:
                self.edt_cli.setText(found)
                self.settings.setValue("syqon/studio/cli_path", found)
        self._update_cli_status(initial=True)

        self._on_model_changed()

        # preset drag handle
        try:
            from setiastro.saspro.shortcuts import PresetDragHandle
            try:
                from setiastro.saspro.resources import syqon_path
                _grip = QIcon(syqon_path)
            except Exception:
                _grip = QIcon()
            drag_row = QHBoxLayout(); drag_row.setContentsMargins(0, 0, 0, 0)
            self.preset_drag_handle = PresetDragHandle(
                "syqontools", self.get_preset, icon=_grip,
                tooltip="Drag to the canvas to create a SyQon Studio shortcut.\n"
                        "Drop directly on an image to apply it headlessly.",
                parent=self,
            )
            drag_row.addWidget(self.preset_drag_handle); drag_row.addStretch(1)
            self.layout().addLayout(drag_row)
        except Exception:
            pass

    # ---- per-model option pages -------------------------------------------
    def _build_prism_page(self) -> QWidget:
        w = QWidget(self)
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(
            "Denoise · Application blends the result with the original.", self
        )
        lbl.setWordWrap(True); lbl.setStyleSheet("color:#8e8e93;")
        v.addWidget(lbl)
        return w

    def _build_parallax_page(self) -> QWidget:
        w = QWidget(self)
        form = QFormLayout(w)

        self.cmb_px_family = QComboBox(self)
        self.cmb_px_family.addItem("Aesthetics", userData="aesthetics")
        self.cmb_px_family.addItem("Classic", userData="classic")
        f = self.cmb_px_family.findData(str(self.settings.value("syqon/studio/px_family", "aesthetics", type=str) or "aesthetics"))
        if f >= 0:
            self.cmb_px_family.setCurrentIndex(f)
        form.addRow("Family:", self.cmb_px_family)

        self.chk_px_correct = QCheckBox("Stellar correction / defect repair", self)
        self.chk_px_correct.setChecked(bool(self.settings.value("syqon/studio/px_correct", True, type=bool)))
        form.addRow("", self.chk_px_correct)

        red_row = QWidget(self); red_lay = QHBoxLayout(red_row); red_lay.setContentsMargins(0, 0, 0, 0)
        self.chk_px_reduce = QCheckBox("Star reduction", self)
        self.chk_px_reduce.setChecked(bool(self.settings.value("syqon/studio/px_reduce", True, type=bool)))
        self.spin_px_reduce = QSpinBox(self); self.spin_px_reduce.setRange(0, 10)
        self.spin_px_reduce.setValue(int(self.settings.value("syqon/studio/px_reduce_level", 5)))
        self.spin_px_reduce.setFixedWidth(60)
        red_lay.addWidget(self.chk_px_reduce); red_lay.addWidget(QLabel("level:", self))
        red_lay.addWidget(self.spin_px_reduce); red_lay.addStretch(1)
        form.addRow("", red_row)

        deb_row = QWidget(self); deb_lay = QHBoxLayout(deb_row); deb_lay.setContentsMargins(0, 0, 0, 0)
        self.chk_px_deblur = QCheckBox("Deblur (deconvolution)", self)
        self.chk_px_deblur.setChecked(bool(self.settings.value("syqon/studio/px_deblur", True, type=bool)))
        self.spin_px_deblur = QDoubleSpinBox(self); self.spin_px_deblur.setRange(0.0, 1.0)
        self.spin_px_deblur.setSingleStep(0.05); self.spin_px_deblur.setDecimals(2)
        self.spin_px_deblur.setValue(float(self.settings.value("syqon/studio/px_deblur_strength", 0.5)))
        self.spin_px_deblur.setFixedWidth(70)
        deb_lay.addWidget(self.chk_px_deblur); deb_lay.addWidget(QLabel("strength:", self))
        deb_lay.addWidget(self.spin_px_deblur); deb_lay.addStretch(1)
        form.addRow("", deb_row)

        self.chk_px_reduce.toggled.connect(lambda on: self.spin_px_reduce.setEnabled(on))
        self.chk_px_deblur.toggled.connect(lambda on: self.spin_px_deblur.setEnabled(on))
        self.spin_px_reduce.setEnabled(self.chk_px_reduce.isChecked())
        self.spin_px_deblur.setEnabled(self.chk_px_deblur.isChecked())

        note = QLabel("Order: correction → reduction → deblur · enable at least one.", self)
        note.setWordWrap(True); note.setStyleSheet("color:#8e8e93; font-size:11px;")
        form.addRow("", note)
        return w

    def _build_axiom_page(self) -> QWidget:
        w = QWidget(self)
        form = QFormLayout(w)

        self.cmb_ax_stretch = QComboBox(self)
        self.cmb_ax_stretch.addItem("Auto", userData="auto")
        self.cmb_ax_stretch.addItem("Identity (already stretched)", userData="identity")
        self.cmb_ax_stretch.addItem("Custom", userData="custom")
        s = self.cmb_ax_stretch.findData(str(self.settings.value("syqon/studio/ax_stretch", "auto", type=str) or "auto"))
        if s >= 0:
            self.cmb_ax_stretch.setCurrentIndex(s)
        form.addRow("Axiom stretch:", self.cmb_ax_stretch)

        cust_row = QWidget(self); cust_lay = QHBoxLayout(cust_row); cust_lay.setContentsMargins(0, 0, 0, 0)
        self.spin_ax_black = QDoubleSpinBox(self); self.spin_ax_black.setRange(0.0, 1.0)
        self.spin_ax_black.setDecimals(4); self.spin_ax_black.setSingleStep(0.001)
        self.spin_ax_black.setValue(float(self.settings.value("syqon/studio/ax_black", 0.0)))
        self.spin_ax_mid = QDoubleSpinBox(self); self.spin_ax_mid.setRange(0.0, 1.0)
        self.spin_ax_mid.setDecimals(4); self.spin_ax_mid.setSingleStep(0.001)
        self.spin_ax_mid.setValue(float(self.settings.value("syqon/studio/ax_mid", 0.25)))
        self.spin_ax_white = QDoubleSpinBox(self); self.spin_ax_white.setRange(0.0, 1.0)
        self.spin_ax_white.setDecimals(4); self.spin_ax_white.setSingleStep(0.001)
        self.spin_ax_white.setValue(float(self.settings.value("syqon/studio/ax_white", 1.0)))
        for cap, sp in (("black", self.spin_ax_black), ("mid", self.spin_ax_mid), ("white", self.spin_ax_white)):
            cust_lay.addWidget(QLabel(cap, self)); cust_lay.addWidget(sp)
        cust_lay.addStretch(1)
        self._ax_custom_row = cust_row
        form.addRow("Custom points:", cust_row)

        self.chk_ax_stars = QCheckBox("Also create a Stars-Only document", self)
        self.chk_ax_stars.setChecked(bool(self.settings.value("syqon/studio/ax_make_stars", True, type=bool)))
        form.addRow("", self.chk_ax_stars)

        def _toggle_custom(*_):
            on = (self.cmb_ax_stretch.currentData() == "custom")
            self._ax_custom_row.setEnabled(on)
        self.cmb_ax_stretch.currentIndexChanged.connect(_toggle_custom)
        _toggle_custom()

        note = QLabel("Starless applied to this view; stars-only opens as a new view.", self)
        note.setWordWrap(True); note.setStyleSheet("color:#8e8e93; font-size:11px;")
        form.addRow("", note)
        return w

    def _build_gradient_page(self) -> QWidget:
        w = QWidget(self)
        form = QFormLayout(w)
        self.chk_dg_gradient = QCheckBox("Also create the extracted-gradient document", self)
        self.chk_dg_gradient.setChecked(bool(self.settings.value("syqon/studio/dg_make_gradient", False, type=bool)))
        form.addRow("", self.chk_dg_gradient)
        note = QLabel("Fixed internal contract (tile/overlap managed by the CLI) · linear input.", self)
        note.setWordWrap(True); note.setStyleSheet("color:#8e8e93; font-size:11px;")
        form.addRow("", note)
        return w

    # ---- CLI helpers -------------------------------------------------------
    def _cli_path(self) -> str:
        return _resolve_bundle(self.edt_cli.text().strip())

    def _locate_cli(self):
        start = self._cli_path() or (studio_cli_candidates()[0] if studio_cli_candidates() else "")
        filt = "SyQon CLI (syqon-cli syqon-cli.exe);;Applications (*.app);;All Files (*)"
        path, _ = QFileDialog.getOpenFileName(self, "Locate syqon-cli", start, filt)
        if not path:
            return
        path = _resolve_bundle(path)
        self.edt_cli.setText(path)
        self._validate_cli()

    def _validate_cli(self):
        path = self._cli_path()
        if not path:
            self.lbl_cli_status.setText("No executable selected.")
            return
        ok, text = validate_studio_cli(path)
        if ok:
            self.settings.setValue("syqon/studio/cli_path", path)
            avail = list_studio_models(path)
            extra = ""
            if avail:
                extra = f"\nAvailable models: {', '.join(sorted(avail))}"
            self.lbl_cli_status.setStyleSheet("color:#2ed573;")
            self.lbl_cli_status.setText(f"✔  {text}{extra}")
        else:
            self.lbl_cli_status.setStyleSheet("color:#ff6b6b;")
            self.lbl_cli_status.setText(f"✘  Could not validate: {text}")

    def _update_cli_status(self, initial=False):
        path = self._cli_path()
        if not path:
            self.lbl_cli_status.setStyleSheet("color:#ff6b6b;")
            self.lbl_cli_status.setText(
                "SyQon Studio CLI not found. Install SyQon Studio and sign in, then "
                "click Locate… if it isn't detected automatically."
            )
            return
        if initial:
            ok, text = validate_studio_cli(path)
            if ok:
                self.lbl_cli_status.setStyleSheet("color:#2ed573;")
                self.lbl_cli_status.setText(f"✔  {text}")
            else:
                self.lbl_cli_status.setStyleSheet("color:#8e8e93;")
                self.lbl_cli_status.setText("Executable set — click Validate to confirm.")

    # ---- model change ------------------------------------------------------
    def current_model(self) -> str:
        return str(self.cmb_model.currentData() or "prism-essential")

    def _on_model_changed(self, *_):
        mid = self.current_model()
        m = _MODEL_INDEX.get(mid)
        fam = studio_family_for(mid)
        self.settings.setValue("syqon/studio/model", mid)

        if m:
            access = m[2]; note = m[4]
            tag = "included with your account" if access == "Included" else "requires a licensed entitlement"
            self.lbl_contract.setText(f"Input contract: {note}  ·  {tag}.")

        self.opt_stack.setCurrentIndex(self._family_page.get(fam, 0))

        # Application only applies to Prism; tile/overlap not used by deep-gradient.
        is_prism = (fam == "prism")
        self.lbl_app.setVisible(is_prism)
        self._app_row_widget.setVisible(is_prism)

        uses_tiles = fam in ("prism", "parallax")
        self.spin_tile.setEnabled(uses_tiles)
        self.spin_overlap.setEnabled(uses_tiles)

    # ---- preset ------------------------------------------------------------
    def get_preset(self) -> dict:
        return {
            "family": "studio",
            "studio_model": self.current_model(),
            "studio_domain": str(self.cmb_domain.currentData() or "auto"),
            "studio_tile": int(self.spin_tile.value()),
            "studio_overlap": int(self.spin_overlap.value()),
            "studio_application": float(self.spin_app.value()),
            "studio_px_family": str(self.cmb_px_family.currentData() or "aesthetics"),
            "studio_px_correct": bool(self.chk_px_correct.isChecked()),
            "studio_px_reduce": bool(self.chk_px_reduce.isChecked()),
            "studio_px_reduce_level": int(self.spin_px_reduce.value()),
            "studio_px_deblur": bool(self.chk_px_deblur.isChecked()),
            "studio_px_deblur_strength": float(self.spin_px_deblur.value()),
            "studio_ax_stretch": str(self.cmb_ax_stretch.currentData() or "auto"),
            "studio_ax_black": float(self.spin_ax_black.value()),
            "studio_ax_mid": float(self.spin_ax_mid.value()),
            "studio_ax_white": float(self.spin_ax_white.value()),
            "studio_ax_make_stars": bool(self.chk_ax_stars.isChecked()),
            "studio_dg_make_gradient": bool(self.chk_dg_gradient.isChecked()),
        }

    def seed_from_preset(self, preset: dict | None):
        p = dict(preset or {})
        if "studio_model" in p:
            i = self.cmb_model.findData(str(p["studio_model"]))
            if i >= 0:
                self.cmb_model.setCurrentIndex(i)
        if "studio_domain" in p:
            i = self.cmb_domain.findData(str(p["studio_domain"]))
            if i >= 0:
                self.cmb_domain.setCurrentIndex(i)

        def _seti(sp, key, lo, hi):
            if key in p:
                try:
                    sp.setValue(max(lo, min(hi, int(p[key]))))
                except (TypeError, ValueError):
                    pass

        def _setf(sp, key, lo, hi):
            if key in p:
                try:
                    sp.setValue(max(lo, min(hi, float(p[key]))))
                except (TypeError, ValueError):
                    pass

        _seti(self.spin_tile, "studio_tile", 256, 2048)
        _seti(self.spin_overlap, "studio_overlap", 0, 512)
        _setf(self.spin_app, "studio_application", 0.0, 1.0)
        if "studio_px_family" in p:
            i = self.cmb_px_family.findData(str(p["studio_px_family"]))
            if i >= 0:
                self.cmb_px_family.setCurrentIndex(i)
        if "studio_px_correct" in p:
            self.chk_px_correct.setChecked(bool(p["studio_px_correct"]))
        if "studio_px_reduce" in p:
            self.chk_px_reduce.setChecked(bool(p["studio_px_reduce"]))
        _seti(self.spin_px_reduce, "studio_px_reduce_level", 0, 10)
        if "studio_px_deblur" in p:
            self.chk_px_deblur.setChecked(bool(p["studio_px_deblur"]))
        _setf(self.spin_px_deblur, "studio_px_deblur_strength", 0.0, 1.0)
        if "studio_ax_stretch" in p:
            i = self.cmb_ax_stretch.findData(str(p["studio_ax_stretch"]))
            if i >= 0:
                self.cmb_ax_stretch.setCurrentIndex(i)
        _setf(self.spin_ax_black, "studio_ax_black", 0.0, 1.0)
        _setf(self.spin_ax_mid, "studio_ax_mid", 0.0, 1.0)
        _setf(self.spin_ax_white, "studio_ax_white", 0.0, 1.0)
        if "studio_ax_make_stars" in p:
            self.chk_ax_stars.setChecked(bool(p["studio_ax_make_stars"]))
        if "studio_dg_make_gradient" in p:
            self.chk_dg_gradient.setChecked(bool(p["studio_dg_make_gradient"]))
        self._on_model_changed()

    # ---- run ---------------------------------------------------------------
    def _persist(self):
        s = self.settings
        s.setValue("syqon/studio/model", self.current_model())
        s.setValue("syqon/studio/domain", str(self.cmb_domain.currentData() or "auto"))
        s.setValue("syqon/studio/tile", int(self.spin_tile.value()))
        s.setValue("syqon/studio/overlap", int(self.spin_overlap.value()))
        s.setValue("syqon/studio/application", float(self.spin_app.value()))
        s.setValue("syqon/studio/px_family", str(self.cmb_px_family.currentData() or "aesthetics"))
        s.setValue("syqon/studio/px_correct", bool(self.chk_px_correct.isChecked()))
        s.setValue("syqon/studio/px_reduce", bool(self.chk_px_reduce.isChecked()))
        s.setValue("syqon/studio/px_reduce_level", int(self.spin_px_reduce.value()))
        s.setValue("syqon/studio/px_deblur", bool(self.chk_px_deblur.isChecked()))
        s.setValue("syqon/studio/px_deblur_strength", float(self.spin_px_deblur.value()))
        s.setValue("syqon/studio/ax_stretch", str(self.cmb_ax_stretch.currentData() or "auto"))
        s.setValue("syqon/studio/ax_black", float(self.spin_ax_black.value()))
        s.setValue("syqon/studio/ax_mid", float(self.spin_ax_mid.value()))
        s.setValue("syqon/studio/ax_white", float(self.spin_ax_white.value()))
        s.setValue("syqon/studio/ax_make_stars", bool(self.chk_ax_stars.isChecked()))
        s.setValue("syqon/studio/dg_make_gradient", bool(self.chk_dg_gradient.isChecked()))

    def _collect_opts(self) -> dict:
        return dict(
            domain=str(self.cmb_domain.currentData() or "auto"),
            tile=int(self.spin_tile.value()),
            overlap=int(self.spin_overlap.value()),
            application=float(self.spin_app.value()),
            parallax_family=str(self.cmb_px_family.currentData() or "aesthetics"),
            parallax_correction=bool(self.chk_px_correct.isChecked()),
            parallax_reduction=bool(self.chk_px_reduce.isChecked()),
            parallax_reduction_level=int(self.spin_px_reduce.value()),
            parallax_deblur=bool(self.chk_px_deblur.isChecked()),
            parallax_deblur_strength=float(self.spin_px_deblur.value()),
            axiom_stretch=str(self.cmb_ax_stretch.currentData() or "auto"),
            axiom_black=float(self.spin_ax_black.value()),
            axiom_mid=float(self.spin_ax_mid.value()),
            axiom_white=float(self.spin_ax_white.value()),
            precision="f32",
        )

    def process_document(self, doc, main):
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.warning(self, "SyQon Studio", "No active image.")
            return

        exe = self._cli_path()
        if not exe or not os.path.isfile(exe):
            QMessageBox.warning(self, "SyQon Studio",
                                "SyQon Studio CLI not found. Click Locate… to select syqon-cli.")
            return

        mid = self.current_model()
        fam = studio_family_for(mid)

        # Parallax needs at least one stage.
        if fam == "parallax" and not any((
            self.chk_px_correct.isChecked(),
            self.chk_px_reduce.isChecked(),
            self.chk_px_deblur.isChecked(),
        )):
            QMessageBox.warning(self, "SyQon Studio",
                                "Parallax needs at least one stage enabled "
                                "(correction, reduction, or deblur).")
            return

        self.doc = doc
        self.main = main
        self._persist()

        src = np.asarray(doc.image).astype(np.float32, copy=False)
        orig_was_mono = (src.ndim == 2) or (src.ndim == 3 and src.shape[2] == 1)
        x = np.nan_to_num(src, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        scale_factor = float(np.max(x)) if x.size else 1.0
        x01 = np.clip(x / scale_factor, 0.0, 1.0) if scale_factor > 1.01 else np.clip(x, 0.0, 1.0)

        if x01.ndim == 2:
            xrgb = np.stack([x01] * 3, axis=-1)
        elif x01.ndim == 3 and x01.shape[2] == 1:
            xrgb = np.repeat(x01, 3, axis=2)
        else:
            xrgb = x01[..., :3]

        self._scale_factor = scale_factor
        self._orig_was_mono = orig_was_mono

        # Secondary output selection
        want_secondary = False
        secondary_flag = None
        if fam == "axiom" and self.chk_ax_stars.isChecked():
            want_secondary = True; secondary_flag = "--stars-output"
        elif fam == "deep-gradient" and self.chk_dg_gradient.isChecked():
            want_secondary = True; secondary_flag = "--gradient-output"

        try:
            if self.proc_thr is not None and self.proc_thr.isRunning():
                self.proc_thr.cancel(); self.proc_thr.wait(250)
        except Exception:
            pass

        self._set_busy(True)
        self.lbl_status.setText("Processing…")

        self.proc_thr = _SyQonStudioCLIThread(
            input_rgb01=xrgb,
            scale_factor=scale_factor,
            exe=exe,
            model_id=mid,
            argv_opts=self._collect_opts(),
            want_secondary=want_secondary,
            secondary_flag=secondary_flag,
            parent=None,  # keep the worker out of the widget tree
        )
        self.proc_thr.progress.connect(self._on_worker_progress)
        self.proc_thr.finished.connect(self._on_worker_finished)
        self.proc_thr.finished.connect(self.proc_thr.deleteLater)
        self.proc_thr.start()

    def _cancel_processing(self):
        thr = getattr(self, "proc_thr", None)
        if thr is not None and thr.isRunning():
            thr.cancel()
            self.btn_cancel.setEnabled(False)
            self.lbl_status.setText("Cancelling…")

    def _set_busy(self, busy: bool):
        for w in (self.cmb_model, self.cmb_domain, self.spin_tile, self.spin_overlap,
                  self.sld_app, self.spin_app, self.opt_stack, self.btn_locate,
                  self.btn_validate, self.edt_cli):
            try:
                w.setEnabled(not busy)
            except Exception:
                pass
        self.pbar.setVisible(busy)
        self.btn_cancel.setVisible(busy)
        self.btn_cancel.setEnabled(busy)
        if busy:
            self.pbar.setValue(0)

    def _on_worker_progress(self, pct: int, stage: str):
        self.pbar.setValue(int(pct))
        if stage:
            self.lbl_status.setText(stage)

    def _on_worker_finished(self, primary, secondary, info: dict, err: str):
        if getattr(self, "proc_thr", None) is None and err == "__cancelled__":
            return

        if err == "__cancelled__":
            self._set_busy(False)
            self.lbl_status.setText("Cancelled.")
            self.proc_thr = None
            return

        if err:
            self._set_busy(False)
            self.proc_thr = None
            QMessageBox.critical(self, "SyQon Studio", err)
            self.lbl_status.setText("Failed.")
            return

        mid = self.current_model()
        fam = studio_family_for(mid)
        orig_was_mono = self._orig_was_mono

        primary = np.asarray(primary, dtype=np.float32)
        if primary.ndim == 2:
            primary = np.stack([primary] * 3, axis=-1)

        orig = np.asarray(self.doc.image).astype(np.float32, copy=False)
        orig = np.nan_to_num(orig, nan=0.0, posinf=0.0, neginf=0.0)
        if orig.ndim == 2:
            orig_rgb = np.stack([orig] * 3, axis=-1)
        elif orig.ndim == 3 and orig.shape[2] == 1:
            orig_rgb = np.repeat(orig, 3, axis=2)
        else:
            orig_rgb = orig[..., :3]

        # Mask-blend the primary against the original using the doc's active
        # mask (all families). No mask ⇒ blend returns the primary unchanged.
        final_rgb = _blend_result_with_mask(primary, orig_rgb, self.doc)
        final_to_apply = final_rgb.mean(axis=2).astype(np.float32, copy=False) if orig_was_mono else final_rgb
        final_to_apply = np.clip(final_to_apply, 0.0, 1.0).astype(np.float32, copy=False)

        step_name = {
            "prism": "Denoised",
            "parallax": "Sharpened",
            "axiom": "Stars Removed",
            "deep-gradient": "Gradient Removed",
        }.get(fam, "SyQon Studio")

        meta = {
            "step_name": step_name,
            "command_id": "syqontools",
            "preset": self.get_preset(),
            "bit_depth": "32-bit floating point",
            "is_mono": bool(orig_was_mono),
            "masked": bool(getattr(self.doc, "active_mask_id", None)),
            "mask_id": getattr(self.doc, "active_mask_id", None) or None,
            "mask_blend": "m*out+(1-m)*src",
            "replay_last": {
                "op": "syqon_studio",
                "params": {**self.get_preset(), "label": f"SyQon Studio ({mid})"},
            },
        }

        # Apply the primary to the original doc FIRST, before pushing the
        # secondary as a new doc. If _push_as_new_doc shifts active-doc focus
        # or otherwise mutates state that apply_edit relies on, doing the
        # apply_edit first eliminates that as a variable.
        import sys as _sys
        print(f"[SyQon apply] fam={fam} secondary_present={secondary is not None} "
              f"final_shape={final_to_apply.shape} "
              f"stats=(min={float(final_to_apply.min()):.4f}, max={float(final_to_apply.max()):.4f}, mean={float(final_to_apply.mean()):.4f})",
              file=_sys.stderr, flush=True)
        try:
            self.doc.apply_edit(final_to_apply, metadata=meta, step_name=step_name)
            print("[SyQon apply] apply_edit returned normally", file=_sys.stderr, flush=True)
        except Exception as _ae:
            import traceback
            print(f"[SyQon apply] apply_edit RAISED: {_ae!r}", file=_sys.stderr, flush=True)
            traceback.print_exc(file=_sys.stderr); _sys.stderr.flush()

        # Now push the secondary as a new view (stars-only for Axiom, gradient
        # for Deep Gradient). Done AFTER apply_edit so nothing it does can
        # interfere with the primary being written to the original doc.
        # The mask scopes the secondary too: the CLI operates on the whole
        # image, but the user only asked us to touch pixels the mask allows,
        # so we blend the secondary against a ZERO image via the same mask.
        # Result: masked region shows the extracted stars/gradient, the rest
        # is black. With no mask set, _blend_result_with_mask returns the
        # secondary unchanged (full-image behavior, as before).
        if secondary is not None:
            try:
                sec = np.asarray(secondary, dtype=np.float32)
                if sec.ndim == 2:
                    sec = np.stack([sec] * 3, axis=-1)
                if fam == "axiom":
                    suffix, source = "_stars", "Stars-Only (SyQon Axiom)"
                else:
                    suffix, source = "_gradient", "Gradient (SyQon Deep Gradient)"
                sec_zero = np.zeros_like(sec, dtype=np.float32)
                sec_masked = _blend_result_with_mask(sec, sec_zero, self.doc)
                sec_push = sec_masked.mean(axis=2).astype(np.float32, copy=False) if orig_was_mono else sec_masked
                sec_push = np.clip(sec_push, 0.0, 1.0).astype(np.float32, copy=False)
                _push_as_new_doc(self.main, self.doc, sec_push, title_suffix=suffix, source=source)
            except Exception as _se:
                print(f"[SyQon apply] secondary push failed: {_se!r}", file=_sys.stderr, flush=True)
        try:
            self.main._last_headless_command = {"command_id": "syqontools", "preset": self.get_preset()}
        except Exception:
            pass
        try:
            if hasattr(self.main, "_log"):
                self.main._log(f"SyQon Studio: {mid} (device={info.get('returncode')})")
        except Exception:
            pass

        self.proc_thr = None
        self._set_busy(False)
        self.lbl_status.setText("Complete!")

    def closeEvent(self, ev):
        if not self._safe_to_close():
            ev.ignore()
            return
        super().closeEvent(ev)