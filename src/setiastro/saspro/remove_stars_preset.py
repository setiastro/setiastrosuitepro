# pro/remove_stars_preset.py
from __future__ import annotations
import os
import platform
import shutil
import numpy as np

from PyQt6.QtWidgets import QMessageBox, QDialog, QFormLayout, QDialogButtonBox, QComboBox, QCheckBox, QSpinBox, QLabel
from PyQt6.QtCore import QThread, pyqtSignal
from setiastro.saspro.legacy.image_manager import save_image, load_image
from setiastro.saspro.cosmicclarity_engines.darkstar_engine import (
    darkstar_starremoval_rgb01,
    DarkStarParams,
)
# Reuse helpers & plumbing from the interactive module
from .remove_stars import (
    _ProcThread, _ProcDialog,
    _mtf_params_unlinked, _apply_mtf_unlinked_rgb, _invert_mtf_unlinked_rgb,
    _active_mask3_from_doc, _mask_blend_with_doc_mask, _push_as_new_doc,
    _ensure_exec_bit,_pad_reflect, _crop_unpad, _extract_stars_only,
)
from setiastro.saspro.starless_engines.syqon_nafnet_engine import nafnet_starless_rgb01

# ---------- Headless public entry ----------
def run_remove_stars_via_preset(main, doc_or_preset=None, preset: dict | None = None, target_doc=None):
    """
    Headless star removal from a shortcut preset.

    Supports BOTH call shapes:
      1) New CommandRunner shape:
            run_remove_stars_via_preset(main, target_doc, preset)
      2) Legacy shape:
            run_remove_stars_via_preset(main, preset_dict, target_doc=doc)
            run_remove_stars_via_preset(main, preset_dict)
    """
    from PyQt6.QtWidgets import QMessageBox
    from PyQt6.QtCore import QTimer
    import os
    import platform

    # ---- Interpret arguments for backward compat / new executor ----
    if preset is None and isinstance(doc_or_preset, dict):
        # Legacy: (main, preset_dict, target_doc=?)
        p = dict(doc_or_preset or {})
        doc = target_doc
    else:
        # New executor: (main, doc, preset_dict)
        p = dict(preset or {})
        doc = target_doc if target_doc is not None else doc_or_preset

    # Resolve active doc if still None
    if doc is None:
        d = getattr(main, "_active_doc", None)
        doc = d() if callable(d) else d

    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.warning(main, "Remove Stars", "Load an image first.")
        return

    tool = str(p.get("tool", "starnet")).lower()

    # mark headless + short cool-down to block the interactive path
    setattr(main, "_remove_stars_headless_running", True)
    setattr(main, "_remove_stars_guard", True)

    def _clear_flags():
        for name in ("_remove_stars_headless_running", "_remove_stars_guard"):
            try:
                delattr(main, name)
            except Exception:
                setattr(main, name, False)

    try:
        if tool in ("starnet", "star_net", "sn"):
            _run_starnet_headless(main, doc, p)
        elif tool in ("darkstar", "cosmicclarity", "cosmic_clarity"):
            _run_darkstar_headless(main, doc, p)
        elif tool in ("syqon", "syqon_starless"):
            _run_syqon_headless(main, doc, p)   # NEW
        else:
            QMessageBox.critical(main, "Remove Stars", f"Unknown tool '{tool}'.")
    finally:
        QTimer.singleShot(1200, _clear_flags)


def apply_remove_stars_to_doc(parent, target_doc, preset: dict | None):
    """
    Replay helper: apply Remove Stars to a specific doc (base/ROI).
    """
    if parent is None:
        return
    main = parent
    # walk up to main window if needed
    while main is not None and not hasattr(main, "doc_manager"):
        main = main.parent() if hasattr(main, "parent") else None
    if main is None:
        main = parent

    run_remove_stars_via_preset(main, preset, target_doc=target_doc)


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
    exe = _resolve_starnet_exe_headless(main, p.get("starnet_exe") or p.get("exe"))
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
    did_stretch = is_linear

    # sanitize + normalize if needed (keep exactly like interactive)
    processing_image = np.nan_to_num(processing_image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    scale_factor = float(np.max(processing_image)) if processing_image.size else 1.0
    processing_norm = (processing_image / scale_factor) if scale_factor > 1.0 else processing_image
    processing_norm = np.clip(processing_norm, 0.0, 1.0)

    img_for_starnet = processing_norm

    if is_linear:
        mtf_params = _mtf_params_unlinked(processing_norm, shadows_clipping=-2.8, targetbg=0.25)
        img_for_starnet = _apply_mtf_unlinked_rgb(processing_norm, mtf_params)

        # stash for inverse step (same keys as interactive)
        try:
            setattr(main, "_starnet_stat_meta", {
                "scheme": "siril_mtf",
                "s": np.asarray(mtf_params["s"], dtype=np.float32),
                "m": np.asarray(mtf_params["m"], dtype=np.float32),
                "h": np.asarray(mtf_params["h"], dtype=np.float32),
                "scale": float(scale_factor),
            })
        except Exception:
            pass
    else:
        try:
            if hasattr(main, "_starnet_stat_meta"):
                delattr(main, "_starnet_stat_meta")
        except Exception:
            pass


    starnet_dir = os.path.dirname(exe) or os.getcwd()
    in_path  = os.path.join(starnet_dir, "imagetoremovestars.tif")
    out_path = os.path.join(starnet_dir, "starless.tif")

    try:
        save_image(img_for_starnet, in_path, original_format="tif",
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
        meta = getattr(main, "_starnet_stat_meta", None)
        if isinstance(meta, dict) and meta.get("scheme") == "siril_mtf":
            try:
                p = {
                    "s": np.asarray(meta.get("s"), dtype=np.float32),
                    "m": np.asarray(meta.get("m"), dtype=np.float32),
                    "h": np.asarray(meta.get("h"), dtype=np.float32),
                }
                inv = _invert_mtf_unlinked_rgb(starless_rgb, p)
                sc = float(meta.get("scale", 1.0))
                if sc > 1.0:
                    inv *= sc
                starless_rgb = np.clip(inv, 0.0, 1.0).astype(np.float32, copy=False)
            except Exception:
                pass

        # cleanup so it can't leak
        try:
            if hasattr(main, "_starnet_stat_meta"):
                delattr(main, "_starnet_stat_meta")
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
    # --- Build input for engine: float32 [0..1] ---
    src = np.asarray(doc.image)
    orig_was_mono = (src.ndim == 2) or (src.ndim == 3 and src.shape[2] == 1)

    x = np.nan_to_num(src.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize to [0..1] for engine if needed (matches interactive)
    scale_factor = float(np.max(x)) if x.size else 1.0
    if scale_factor > 1.0:
        x = x / scale_factor
    x = np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)

    disable_gpu = bool(p.get("disable_gpu", False))
    mode = str(p.get("mode", "unscreen"))
    show = bool(p.get("show_extracted_stars", True))
    stride = int(p.get("stride", 512))

    params = DarkStarParams(
        use_gpu=(not disable_gpu),
        chunk_size=int(stride),
        overlap_frac=0.125,
        mode=mode,
        output_stars_only=show,
    )

    dlg = _ProcDialog(main, title="Dark Star Progress")
    dlg.append_text("Starting Dark Star (integrated engine)…\n")

    thr = _DarkStarEngineThread(x, params, parent=dlg)

    def _on_prog(done, total, stage):
        dlg.set_progress(done, total, stage)

    def _on_done(starless, stars_only, was_mono, err):
        if err:
            QMessageBox.critical(main, "CosmicClarity DarkStar", f"DarkStar failed:\n{err}")
            dlg.close()
            return

        # restore scale if we normalized (>1 domain docs)
        if scale_factor > 1.0:
            try:
                starless = starless * scale_factor
                if stars_only is not None:
                    stars_only = stars_only * scale_factor
            except Exception:
                pass

        # convert to RGB for blend math
        starless_arr = np.asarray(starless, dtype=np.float32)
        if starless_arr.ndim == 2 or (starless_arr.ndim == 3 and starless_arr.shape[2] == 1):
            starless_rgb = np.stack([starless_arr.squeeze()] * 3, axis=-1)
        else:
            starless_rgb = starless_arr[..., :3]

        orig = np.asarray(doc.image, dtype=np.float32)
        if orig.ndim == 2:
            original_rgb = np.stack([orig] * 3, axis=-1)
        elif orig.ndim == 3 and orig.shape[2] == 1:
            original_rgb = np.repeat(orig, 3, axis=2)
        else:
            original_rgb = orig[..., :3]

        # push stars-only doc if requested + available
        if show and stars_only is not None:
            so = np.asarray(stars_only, dtype=np.float32)
            if so.ndim == 2 or (so.ndim == 3 and so.shape[2] == 1):
                so_rgb = np.stack([so.squeeze()] * 3, axis=-1)
            else:
                so_rgb = so[..., :3]

            m3 = _active_mask3_from_doc(doc, so_rgb.shape[1], so_rgb.shape[0])
            if m3 is not None:
                so_rgb *= m3
            _push_as_new_doc(main, doc,
                             so_rgb.mean(axis=2).astype(np.float32) if orig_was_mono else so_rgb,
                             title_suffix="_stars", source="Stars-Only (DarkStar)")

        final_starless = _mask_blend_with_doc_mask(doc, starless_rgb, original_rgb)
        final_to_apply = final_starless.mean(axis=2).astype(np.float32, copy=False) if orig_was_mono else final_starless
        final_to_apply = np.clip(final_to_apply, 0.0, 1.0).astype(np.float32, copy=False)

        try:
            meta = {"step_name": "Stars Removed", "bit_depth": "32-bit floating point", "is_mono": bool(orig_was_mono)}
            doc.apply_edit(final_to_apply, metadata=meta, step_name="Stars Removed")
            if hasattr(main, "_log"):
                main._log("Stars Removed (DarkStar, headless/integrated)")
        except Exception as e:
            QMessageBox.critical(main, "CosmicClarity DarkStar", f"Failed to apply starless result:\n{e}")

        dlg.close()

    thr.progress_signal.connect(_on_prog)
    thr.finished_signal.connect(_on_done)

    dlg.cancel_button.clicked.connect(lambda: dlg.append_text("Cancel not supported for in-process engine.\n"))
    dlg.show()
    thr.start()
    dlg.exec()


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
        self.cmb_tool.addItems(["StarNet", "CosmicClarity DarkStar", "SyQon"])
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

        # SyQon options
        self.syq_tile = QSpinBox(); self.syq_tile.setRange(128, 2048); self.syq_tile.setSingleStep(64)
        self.syq_overlap = QSpinBox(); self.syq_overlap.setRange(16, 512); self.syq_overlap.setSingleStep(16)
        self.syq_make_stars = QCheckBox("Also create stars-only document (_stars)")
        self.syq_make_stars.setChecked(True)

        self.syq_pad = QCheckBox("Pad edges (reflect)")
        self.syq_pad.setChecked(True)
        self.syq_pad_px = QSpinBox(); self.syq_pad_px.setRange(0, 1024); self.syq_pad_px.setSingleStep(16); self.syq_pad_px.setValue(128)

        self.syq_extract = QComboBox(); self.syq_extract.addItems(["subtract", "unscreen"])


        form = QFormLayout(self)
        form.addRow("Tool:", self.cmb_tool)
        form.addRow(QLabel("— StarNet —"))
        form.addRow("", self.chk_linear)
        form.addRow(QLabel("— DarkStar —"))
        form.addRow("Mode:", self.cmb_mode)
        form.addRow("Stride:", self.cmb_stride)
        form.addRow("", self.chk_disable_gpu)
        form.addRow("", self.chk_show)
        form.addRow(QLabel("— SyQon —"))
        form.addRow("Tile size:", self.syq_tile)
        form.addRow("Overlap:", self.syq_overlap)
        form.addRow("", self.syq_make_stars)
        form.addRow("", self.syq_pad)
        form.addRow("Pad pixels:", self.syq_pad_px)
        form.addRow("Stars-only extraction:", self.syq_extract)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        form.addRow(btns)

        # show/hide sections depending on tool
        def _toggle():
            idx = self.cmb_tool.currentIndex()
            is_starnet = (idx == 0)
            is_darkstar = (idx == 1)
            is_syqon = (idx == 2)

            self.chk_linear.setEnabled(is_starnet)

            for w in (self.cmb_mode, self.cmb_stride, self.chk_disable_gpu, self.chk_show):
                w.setEnabled(is_darkstar)

            for w in (self.syq_tile, self.syq_overlap, self.syq_make_stars, self.syq_pad, self.syq_pad_px, self.syq_extract):
                w.setEnabled(is_syqon)

        self.cmb_tool.currentIndexChanged.connect(lambda _: _toggle())
        _toggle()

    def result_dict(self) -> dict:
        idx = self.cmb_tool.currentIndex()
        if idx == 0:
            return {"tool": "starnet", "linear": bool(self.chk_linear.isChecked())}
        if idx == 1:
            return {
                "tool": "darkstar",
                "disable_gpu": bool(self.chk_disable_gpu.isChecked()),
                "mode": self.cmb_mode.currentText(),
                "show_extracted_stars": bool(self.chk_show.isChecked()),
                "stride": int(self.cmb_stride.currentData()),
            }
        # SyQon
        return {
            "tool": "syqon",
            "tile_size": int(self.syq_tile.value()),
            "overlap": int(self.syq_overlap.value()),
            "make_stars": bool(self.syq_make_stars.isChecked()),
            "pad_edges": bool(self.syq_pad.isChecked()),
            "pad_pixels": int(self.syq_pad_px.value()),
            "stars_extract": str(self.syq_extract.currentText()),
        }


class _DarkStarEngineThread(QThread):
    progress_signal = pyqtSignal(int, int, str)           # done, total, stage
    finished_signal = pyqtSignal(object, object, bool, str)  # starless, stars_only, was_mono, errstr

    def __init__(self, img_rgb01: np.ndarray, params: DarkStarParams, parent=None):
        super().__init__(parent)
        self._img = img_rgb01
        self._params = params

    def run(self):
        try:
            def prog(done, total, stage):
                self.progress_signal.emit(int(done), int(total), str(stage))

            starless, stars_only, was_mono = darkstar_starremoval_rgb01(
                self._img,
                params=self._params,
                progress_cb=prog,
                status_cb=lambda s: None,
            )
            self.finished_signal.emit(starless, stars_only, bool(was_mono), "")
        except Exception as e:
            self.finished_signal.emit(None, None, False, str(e))

class _SyQonEngineThread(QThread):
    progress_signal = pyqtSignal(int, int, str)           # done, total, stage
    finished_signal = pyqtSignal(object, object, dict, str)  # starless_s, info, aux, err

    def __init__(self, x_for_net_rgb01: np.ndarray, ckpt_path: str,
                 tile: int, overlap: int, parent=None):
        super().__init__(parent)
        self._x = x_for_net_rgb01
        self._ckpt = ckpt_path
        self._tile = int(tile)
        self._overlap = int(overlap)

    def run(self):
        try:
            def prog(done, total, stage):
                self.progress_signal.emit(int(done), int(total), str(stage))

            starless_s, stars_s, info = nafnet_starless_rgb01(
                self._x,
                ckpt_path=self._ckpt,
                tile=self._tile,
                overlap=self._overlap,
                prefer_cuda=True,
                residual_mode=True,
                progress_cb=prog,
            )
            self.finished_signal.emit(starless_s, info, {"stars_s": stars_s}, "")
        except Exception as e:
            self.finished_signal.emit(None, None, {}, str(e))

def _run_syqon_headless(main, doc, p):
    # ---- model path (same location as your dialog installs) ----
    # IMPORTANT: point this at exactly the same syqon_starless/nadir that the dialog uses.
    # If you already have a helper like _syqon_data_dir/_syqon_model_path in remove_stars.py,
    # import and call it instead of reimplementing.
    try:
        from .remove_stars import _syqon_data_dir, _syqon_model_path
        ckpt_path = str(_syqon_model_path(_syqon_data_dir()))
    except Exception:
        ckpt_path = str(p.get("model_path", ""))

    if not ckpt_path or not os.path.exists(ckpt_path):
        QMessageBox.warning(main, "SyQon", "SyQon model not installed. Run the interactive SyQon tool once to install it.")
        return

    # ---- params ----
    tile = int(p.get("tile_size", 512))
    overlap = int(p.get("overlap", 64))
    shadow_clip = float(p.get("shadow_clip", 2.8))  # not used by nafnet wrapper here unless your engine uses it
    make_stars = bool(p.get("make_stars", True))

    pad_edges = bool(p.get("pad_edges", True))
    pad_pixels = int(p.get("pad_pixels", 128))
    stars_extract = str(p.get("stars_extract", "subtract")).lower()  # subtract | unscreen

    # ---- prep image (float32) ----
    src = np.asarray(doc.image)
    orig_was_mono = (src.ndim == 2) or (src.ndim == 3 and src.shape[2] == 1)

    x = np.nan_to_num(src.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

    # normalize to [0..1]
    scale_factor = float(np.max(x)) if x.size else 1.0
    x01 = (x / scale_factor) if scale_factor > 1.01 else x
    x01 = np.clip(x01, 0.0, 1.0).astype(np.float32, copy=False)

    # to RGB
    if x01.ndim == 2:
        xrgb = np.stack([x01] * 3, axis=-1)
    elif x01.ndim == 3 and x01.shape[2] == 1:
        xrgb = np.repeat(x01, 3, axis=2)
    else:
        xrgb = x01[..., :3]

    H0, W0 = xrgb.shape[:2]

    if pad_edges and pad_pixels > 0:
        xrgb = _pad_reflect(xrgb, pad_pixels)

    # MTF stretch for net
    mtf_params = _mtf_params_unlinked(xrgb, shadows_clipping=-2.8, targetbg=0.25)
    x_for_net = _apply_mtf_unlinked_rgb(xrgb, mtf_params)

    # ---- UI progress dialog + thread ----
    dlg = _ProcDialog(main, title="SyQon Progress")
    dlg.append_text("Starting SyQon…\n")

    thr = _SyQonEngineThread(x_for_net, ckpt_path=ckpt_path, tile=tile, overlap=overlap, parent=dlg)

    def _on_prog(done, total, stage):
        dlg.set_progress(done, total, stage)

    def _on_done(starless_s, info, aux, err):
        if err or starless_s is None:
            QMessageBox.critical(main, "SyQon", f"SyQon failed:\n{err or 'Unknown error'}")
            dlg.close()
            return

        starless_s = np.asarray(starless_s, dtype=np.float32)
        if starless_s.ndim == 2:
            starless_s = np.stack([starless_s] * 3, axis=-1)

        # inverse MTF
        starless_lin = _invert_mtf_unlinked_rgb(starless_s, mtf_params)

        # restore original scale domain (if we normalized)
        if scale_factor > 1.01:
            starless_lin = np.clip(starless_lin * scale_factor, 0.0, 1.0).astype(np.float32, copy=False)

        # unpad back to original size
        if pad_edges and pad_pixels > 0:
            starless_lin = _crop_unpad(starless_lin, pad_pixels, H0, W0)

        # original RGB (for stars-only/blend)
        orig = np.asarray(doc.image, dtype=np.float32)
        if orig.ndim == 2:
            original_rgb = np.stack([orig] * 3, axis=-1)
        elif orig.ndim == 3 and orig.shape[2] == 1:
            original_rgb = np.repeat(orig, 3, axis=2)
        else:
            original_rgb = orig[..., :3]

        starless_rgb = starless_lin.astype(np.float32, copy=False)

        # optional stars-only doc
        if make_stars:
            stars_only = _extract_stars_only(original_rgb, starless_rgb, mode=stars_extract)

            m3 = _active_mask3_from_doc(doc, stars_only.shape[1], stars_only.shape[0])
            if m3 is not None:
                stars_only *= m3

            stars_to_push = stars_only.mean(axis=2).astype(np.float32, copy=False) if orig_was_mono else stars_only
            _push_as_new_doc(main, doc, stars_to_push, title_suffix="_stars", source="Stars-Only (SyQon)")

        # blend w/ active mask and apply
        final_starless = _mask_blend_with_doc_mask(doc, starless_rgb, original_rgb)
        final_to_apply = final_starless.mean(axis=2).astype(np.float32, copy=False) if orig_was_mono else final_starless
        final_to_apply = np.clip(final_to_apply, 0.0, 1.0).astype(np.float32, copy=False)

        meta = {
            "step_name": "Stars Removed",
            "bit_depth": "32-bit floating point",
            "is_mono": bool(orig_was_mono),
            "replay_last": {
                "op": "remove_stars",
                "params": {
                    "tool": "syqon",
                    "tile_size": tile,
                    "overlap": overlap,
                    "shadow_clip": shadow_clip,
                    "make_stars": make_stars,
                    "pad_edges": pad_edges,
                    "pad_pixels": pad_pixels,
                    "stars_extract": stars_extract,
                    "model_path": ckpt_path,
                }
            }
        }

        try:
            doc.apply_edit(final_to_apply, metadata=meta, step_name="Stars Removed")
            if hasattr(main, "_log"):
                main._log("Stars Removed (SyQon, headless)")
        except Exception as e:
            QMessageBox.critical(main, "SyQon", f"Failed to apply result:\n{e}")

        dlg.close()

    thr.progress_signal.connect(_on_prog)
    thr.finished_signal.connect(_on_done)

    dlg.cancel_button.clicked.connect(lambda: dlg.append_text("Cancel not supported for SyQon thread yet.\n"))
    dlg.show()
    thr.start()
    dlg.exec()


# ---------- local util ----------
def _safe_rm(p):
    try:
        if p and os.path.exists(p): os.remove(p)
    except Exception as e:
        import logging
        logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
