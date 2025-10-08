# pro/remove_stars_preset.py
from __future__ import annotations
import os, platform, shutil
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMessageBox, QDialog, QFormLayout, QDialogButtonBox, QComboBox, QCheckBox, QSpinBox, QLabel

from legacy.image_manager import save_image, load_image

# Reuse helpers & plumbing from the interactive module
from .remove_stars import (
    _ProcThread, _ProcDialog,
    _stat_stretch_rgb, _stat_unstretch_rgb,
    _active_mask3_from_doc, _mask_blend_with_doc_mask, _push_as_new_doc,
    _ensure_exec_bit,
)

# ---------- Headless public entry ----------
def run_remove_stars_via_preset(main, preset: dict | None = None):
    """
    Headless star removal from a shortcut preset.
      preset = {
        "tool": "starnet" | "darkstar",
        # StarNet:
        "linear": True/False,                # default True
        "exe": "/path/to/starnet(.exe)",     # optional override; else QSettings
        # DarkStar:
        "disable_gpu": False,                # default False
        "mode": "unscreen" | "additive",     # default "unscreen"
        "show_extracted_stars": True/False,  # default True
        "stride": 512,                       # 64..1024 power of two
        "exe": "/path/to/setiastrocosmicclarity_darkstar(.exe)"  # optional override
      }
    """
    p = dict(preset or {})
    tool = str(p.get("tool", "starnet")).lower()

    # active doc
    doc = getattr(main, "_active_doc", None)
    if callable(doc): doc = doc()
    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.warning(main, "Remove Stars", "Load an image first.")
        return

    # mark headless + short cool-down to block the interactive path
    setattr(main, "_remove_stars_headless_running", True)
    setattr(main, "_remove_stars_guard", True)

    def _clear_flags():
        for name in ("_remove_stars_headless_running", "_remove_stars_guard"):
            try: delattr(main, name)
            except Exception: setattr(main, name, False)

    try:
        if tool in ("starnet", "star_net", "sn"):
            _run_starnet_headless(main, doc, p)
        elif tool in ("darkstar", "cosmicclarity", "cosmic_clarity"):
            _run_darkstar_headless(main, doc, p)
        else:
            QMessageBox.critical(main, "Remove Stars", f"Unknown tool '{tool}'.")
    finally:
        # let event loop settle (image apply/logging) before allowing UI again
        QTimer.singleShot(1200, _clear_flags)


# ---------- StarNet headless ----------
def _resolve_starnet_exe_headless(main, override: str | None) -> str | None:
    if override and os.path.exists(override):
        if platform.system() in ("Darwin", "Linux"): _ensure_exec_bit(override)
        return override
    s = getattr(main, "settings", None)
    exe = None
    if s:
        for key in ("starnet/exe_path", "paths/starnet"):
            try:
                val = s.value(key, "", type=str)
            except Exception:
                val = s.value(key, "")
            if isinstance(val, str) and val.strip() and os.path.exists(val):
                exe = val.strip(); break
    if exe and platform.system() in ("Darwin", "Linux"): _ensure_exec_bit(exe)
    return exe

def _run_starnet_headless(main, doc, p):
    exe = _resolve_starnet_exe_headless(main, p.get("exe"))
    if not exe:
        QMessageBox.warning(main, "StarNet", "StarNet path not set. Open the interactive tool once to set it.")
        return

    # RGB float32 [0..1]
    src = np.asarray(doc.image)
    if src.ndim == 2: processing_image = np.stack([src]*3, axis=-1)
    elif src.ndim == 3 and src.shape[2] == 1: processing_image = np.repeat(src, 3, axis=2)
    else: processing_image = src
    processing_image = processing_image.astype(np.float32, copy=False)

    is_linear = bool(p.get("linear", True))
    did_stretch = False
    stretch_params = None
    if is_linear:
        processing_image, stretch_params = _stat_stretch_rgb(processing_image)
        did_stretch = True
        setattr(main, "_starnet_last_stretch_params", stretch_params)
    else:
        if hasattr(main, "_starnet_last_stretch_params"):
            delattr(main, "_starnet_last_stretch_params")

    starnet_dir = os.path.dirname(exe) or os.getcwd()
    in_path  = os.path.join(starnet_dir, "imagetoremovestars.tif")
    out_path = os.path.join(starnet_dir, "starless.tif")

    try:
        save_image(processing_image, in_path, original_format="tif",
                   bit_depth="16-bit", original_header=None, is_mono=False,
                   image_meta=None, file_meta=None)
    except Exception as e:
        QMessageBox.critical(main, "StarNet", f"Failed to write input TIFF:\n{e}")
        return

    exe_name = os.path.basename(exe).lower()
    sysname = platform.system()
    if sysname in ("Windows", "Linux"):
        command = [exe, in_path, out_path, "256"]
    else:
        if "starnet2" in exe_name:
            command = [exe, "--input", in_path, "--output", out_path]
        else:
            command = [exe, in_path, out_path]

    dlg = _ProcDialog(main, title="StarNet Progress")
    thr = _ProcThread(command, cwd=starnet_dir)
    thr.output_signal.connect(dlg.append_text)
    thr.finished_signal.connect(lambda rc: _finish_starnet(main, doc, rc, dlg, in_path, out_path, did_stretch))
    dlg.cancel_button.clicked.connect(thr.terminate)
    dlg.show(); thr.start(); dlg.exec()

def _finish_starnet(main, doc, rc, dlg, in_path, out_path, did_stretch):
    if rc != 0 or not os.path.exists(out_path):
        QMessageBox.critical(main, "StarNet", "StarNet failed or no output image produced.")
        _safe_rm(in_path); _safe_rm(out_path); dlg.close(); return

    starless_rgb, _, _, _ = load_image(out_path)
    if starless_rgb is None:
        QMessageBox.critical(main, "StarNet", "Failed to load starless image.")
        _safe_rm(in_path); _safe_rm(out_path); dlg.close(); return

    if starless_rgb.ndim == 2 or (starless_rgb.ndim == 3 and starless_rgb.shape[2] == 1):
        starless_rgb = np.stack([starless_rgb]*3, axis=-1)
    starless_rgb = starless_rgb.astype(np.float32, copy=False)

    if did_stretch:
        try:
            params = getattr(main, "_starnet_last_stretch_params", None)
            if params: starless_rgb = _stat_unstretch_rgb(starless_rgb, params)
        except Exception:
            pass

    # original as RGB
    orig = np.asarray(doc.image)
    if orig.ndim == 2: original_rgb = np.stack([orig]*3, axis=-1)
    elif orig.ndim == 3 and orig.shape[2] == 1: original_rgb = np.repeat(orig, 3, axis=2)
    else: original_rgb = orig
    original_rgb = original_rgb.astype(np.float32, copy=False)

    # Stars-Only (same as interactive)
    with np.errstate(divide='ignore', invalid='ignore'):
        stars_only = (original_rgb - starless_rgb) / np.clip(1.0 - starless_rgb, 1e-6, None)
        stars_only = np.nan_to_num(stars_only, nan=0.0, posinf=0.0, neginf=0.0)
    stars_only = np.clip(stars_only, 0.0, 1.0)
    m3 = _active_mask3_from_doc(doc, stars_only.shape[1], stars_only.shape[0])
    if m3 is not None: stars_only *= m3
    _push_as_new_doc(main, doc, stars_only, title_suffix="_stars", source="Stars-Only (StarNet)")

    # mask-blend starless, then commit
    final_starless = _mask_blend_with_doc_mask(doc, starless_rgb, original_rgb)
    try:
        meta = {"step_name": "Stars Removed", "bit_depth": "32-bit floating point", "is_mono": False}
        doc.apply_edit(final_starless.astype(np.float32, copy=False), metadata=meta, step_name="Stars Removed")
        if hasattr(main, "_log"): main._log("Stars Removed (StarNet, headless)")
    except Exception as e:
        QMessageBox.critical(main, "StarNet", f"Failed to apply starless result:\n{e}")

    _safe_rm(in_path); _safe_rm(out_path); dlg.close()


# ---------- DarkStar headless ----------
def _resolve_darkstar_exe_headless(main, override: str | None) -> tuple[str | None, str | None]:
    if override and os.path.exists(override):
        base = os.path.dirname(override)
        if platform.system() in ("Darwin","Linux"): _ensure_exec_bit(override)
        return override, base
    s = getattr(main, "settings", None)
    base = None
    if s:
        for key in ("paths/cosmic_clarity", "cosmic_clarity_folder"):
            try:
                v = s.value(key, "", type=str)
            except Exception:
                v = s.value(key, "")
            if isinstance(v, str) and v.strip() and os.path.isdir(v):
                base = v.strip(); break
    if not base: return None, None
    exe = os.path.join(base, "setiastrocosmicclarity_darkstar.exe" if platform.system()=="Windows"
                                else "setiastrocosmicclarity_darkstar")
    if not os.path.exists(exe): return None, None
    if platform.system() in ("Darwin","Linux"): _ensure_exec_bit(exe)
    return exe, base

def _run_darkstar_headless(main, doc, p):
    exe, base = _resolve_darkstar_exe_headless(main, p.get("exe"))
    if not exe or not base:
        QMessageBox.warning(main, "CosmicClarity DarkStar", "DarkStar path not set. Open the interactive tool once to set it.")
        return

    input_dir  = os.path.join(base, "input")
    output_dir = os.path.join(base, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    in_path = os.path.join(input_dir, "imagetoremovestars.tif")
    try:
        save_image(doc.image, in_path, original_format="tif",
                   bit_depth="32-bit floating point",
                   original_header=None, is_mono=False, image_meta=None, file_meta=None)
    except Exception as e:
        QMessageBox.critical(main, "CosmicClarity DarkStar", f"Failed to write input TIFF:\n{e}")
        return

    disable_gpu = bool(p.get("disable_gpu", False))
    mode = str(p.get("mode", "unscreen"))
    show = bool(p.get("show_extracted_stars", True))
    stride = int(p.get("stride", 512))

    args = []
    if disable_gpu: args.append("--disable_gpu")
    args += ["--star_removal_mode", mode]
    if show: args.append("--show_extracted_stars")
    args += ["--chunk_size", str(stride)]

    command = [exe] + args

    dlg = _ProcDialog(main, title="CosmicClarityDarkStar Progress")
    thr = _ProcThread(command, cwd=output_dir)
    thr.output_signal.connect(dlg.append_text)
    thr.finished_signal.connect(lambda rc: _finish_darkstar(main, doc, rc, dlg, in_path, output_dir))
    dlg.cancel_button.clicked.connect(thr.terminate)
    dlg.show(); thr.start(); dlg.exec()

def _finish_darkstar(main, doc, rc, dlg, in_path, output_dir):
    if rc != 0:
        QMessageBox.critical(main, "CosmicClarity DarkStar", "Process failed.")
        _safe_rm(in_path); dlg.close(); return

    starless_path = os.path.join(output_dir, "imagetoremovestars_starless.tif")
    if not os.path.exists(starless_path):
        QMessageBox.critical(main, "CosmicClarity DarkStar", "Starless image not found.")
        _safe_rm(in_path); dlg.close(); return

    starless, _, _, _ = load_image(starless_path)
    if starless is None:
        QMessageBox.critical(main, "CosmicClarity DarkStar", "Failed to load starless image.")
        _safe_rm(in_path); dlg.close(); return

    if starless.ndim == 2 or (starless.ndim == 3 and starless.shape[2] == 1):
        starless_rgb = np.stack([starless]*3, axis=-1)
    else:
        starless_rgb = starless
    starless_rgb = starless_rgb.astype(np.float32, copy=False)

    src = np.asarray(doc.image)
    if src.ndim == 2: original_rgb = np.stack([src]*3, axis=-1)
    elif src.ndim == 3 and src.shape[2] == 1: original_rgb = np.repeat(src, 3, axis=2)
    else: original_rgb = src
    original_rgb = original_rgb.astype(np.float32, copy=False)

    # stars-only (if DarkStar produced it)
    stars_path = os.path.join(output_dir, "imagetoremovestars_stars_only.tif")
    if os.path.exists(stars_path):
        stars_only, _, _, _ = load_image(stars_path)
        if stars_only is not None:
            if stars_only.ndim == 2 or (stars_only.ndim == 3 and stars_only.shape[2] == 1):
                stars_only = np.stack([stars_only]*3, axis=-1)
            stars_only = stars_only.astype(np.float32, copy=False)
            m3 = _active_mask3_from_doc(doc, stars_only.shape[1], stars_only.shape[0])
            if m3 is not None: stars_only *= m3
            _push_as_new_doc(main, doc, stars_only, title_suffix="_stars", source="Stars-Only (DarkStar)")

    final_starless = _mask_blend_with_doc_mask(doc, starless_rgb, original_rgb)
    try:
        meta = {"step_name": "Stars Removed", "bit_depth": "32-bit floating point", "is_mono": False}
        doc.apply_edit(final_starless.astype(np.float32, copy=False), metadata=meta, step_name="Stars Removed")
        if hasattr(main, "_log"): main._log("Stars Removed (DarkStar, headless)")
    except Exception as e:
        QMessageBox.critical(main, "CosmicClarity DarkStar", f"Failed to apply starless result:\n{e}")

    # cleanup
    try:
        _safe_rm(in_path)
        for fn in ("imagetoremovestars_starless.tif","imagetoremovestars_stars_only.tif"):
            p = os.path.join(output_dir, fn)
            if os.path.exists(p): _safe_rm(p)
    except Exception:
        pass
    dlg.close()


# ---------- Simple preset editor (for the shortcut button) ----------
class RemoveStarsPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Remove Stars — Preset")
        p = dict(initial or {})

        self.cmb_tool = QComboBox()
        self.cmb_tool.addItems(["StarNet", "CosmicClarity DarkStar"])
        self.cmb_tool.setCurrentIndex(0 if str(p.get("tool","starnet")).lower().startswith("star") else 1)

        # StarNet options
        self.chk_linear = QCheckBox("Image is linear (apply temporary stretch)")
        self.chk_linear.setChecked(bool(p.get("linear", True)))

        # DarkStar options
        self.chk_disable_gpu = QCheckBox("Disable GPU")
        self.chk_disable_gpu.setChecked(bool(p.get("disable_gpu", False)))
        self.cmb_mode = QComboBox(); self.cmb_mode.addItems(["unscreen","additive"])
        self.cmb_mode.setCurrentText(str(p.get("mode","unscreen")))
        self.chk_show = QCheckBox("Show extracted stars")
        self.chk_show.setChecked(bool(p.get("show_extracted_stars", True)))
        self.cmb_stride = QComboBox()
        for v in (64,128,256,512,1024): self.cmb_stride.addItem(str(v), v)
        self.cmb_stride.setCurrentText(str(int(p.get("stride", 512))))

        form = QFormLayout(self)
        form.addRow("Tool:", self.cmb_tool)
        form.addRow(QLabel("— StarNet —"))
        form.addRow("", self.chk_linear)
        form.addRow(QLabel("— DarkStar —"))
        form.addRow("Mode:", self.cmb_mode)
        form.addRow("Stride:", self.cmb_stride)
        form.addRow("", self.chk_disable_gpu)
        form.addRow("", self.chk_show)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        form.addRow(btns)

        # show/hide sections depending on tool
        def _toggle():
            star = self.cmb_tool.currentIndex() == 0
            self.chk_linear.setEnabled(star)
            for w in (self.cmb_mode, self.cmb_stride, self.chk_disable_gpu, self.chk_show):
                w.setEnabled(not star)
        self.cmb_tool.currentIndexChanged.connect(lambda _: _toggle())
        _toggle()

    def result_dict(self) -> dict:
        if self.cmb_tool.currentIndex() == 0:
            return {"tool": "starnet", "linear": bool(self.chk_linear.isChecked())}
        return {
            "tool": "darkstar",
            "disable_gpu": bool(self.chk_disable_gpu.isChecked()),
            "mode": self.cmb_mode.currentText(),
            "show_extracted_stars": bool(self.chk_show.isChecked()),
            "stride": int(self.cmb_stride.currentData()),
        }


# ---------- local util ----------
def _safe_rm(p):
    try:
        if p and os.path.exists(p): os.remove(p)
    except Exception: pass
