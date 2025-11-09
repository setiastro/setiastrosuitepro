# pro/debayer.py
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
import cv2
import os

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QDialogButtonBox,
    QGroupBox, QMessageBox, QProgressBar
)

# fast kernels you already have
try:
    from legacy.numba_utils import debayer_fits_fast
except Exception as e:  # very unlikely in your env
    debayer_fits_fast = None


_RAW_EXTS = (".raf", ".raw", ".rw2", ".arw", ".nef", ".cr2", ".cr3", ".dng", ".orf", ".pef")

def _find_raw_sibling(path: Optional[str]) -> Optional[str]:
    """
    If we only have a derived FITS/XISF path, try to locate a plausible RAW
    with the same stem in the same folder.
    """
    if not path:
        return None
    base, _ = os.path.splitext(os.path.basename(path))
    folder = os.path.dirname(path)
    if not folder or not base:
        return None
    try:
        for ext in _RAW_EXTS:
            cand = os.path.join(folder, base + ext)
            if os.path.exists(cand):
                return cand
    except Exception:
        pass
    return None

# -------- helpers ------------------------------------------------------------
_BAYER_METHODS = [
    ("Edge-aware (Numba)", "edge"),
    ("Bilinear (Numba)", "bilinear"),
]
_XTRANS_METHODS = [
    ("AHD (rawpy)", "AHD"),
    ("DHT (rawpy)", "DHT"),
]
_VALID = {"RGGB", "BGGR", "GRBG", "GBRG"}

def _normalize_bayer_token(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.upper().replace(",", "").replace(" ", "").replace("/", "").replace("|", "")
    if len(s) == 4 and set(s) <= {"R", "G", "B"}:
        if s.count("R") == 1 and s.count("G") == 2 and s.count("B") == 1:
            return s if s in _VALID else None
    return None

def _detect_bayer_from_header(doc) -> Optional[str]:
    """
    Best-effort read of a Bayer pattern from the document header/metadata.
    Returns 'RGGB'/'BGGR'/'GRBG'/'GBRG' or None if not found.
    """
    hdr, meta, _ = _extract_doc_info(doc)

    probe = {}
    # FITS-like header (astropy Header behaves like a dict, case-insensitive)
    if hdr is not None:
        try:
            keys = list(hdr.keys()) if hasattr(hdr, "keys") else []
            for k in keys:
                try:
                    v = hdr.get(k) if hasattr(hdr, "get") else hdr[k]
                except Exception:
                    v = None
                probe[str(k).upper()] = "" if v is None else str(v)
        except Exception:
            pass

    # Merge doc metadata (strings only)
    if isinstance(meta, dict):
        for k, v in list(meta.items()):
            try:
                probe[str(k).upper()] = "" if v is None else str(v)
            except Exception:
                continue

    keys = [
        "BAYERPAT", "BAYERPATN", "BAYER_PATTERN", "BAYERPATTERN",
        "CFAPATTERN", "CFA_PATTERN", "PATTERN", "COLORTYPE", "COLORFILTERARRAY"
    ]
    for k in keys:
        raw = probe.get(k)
        if not raw:
            continue
        s = str(raw).upper()
        for pat in _VALID:
            if pat in s:
                return pat
        norm = _normalize_bayer_token(s)
        if norm:
            return norm
    return None


def _detect_cfa_family(doc) -> Optional[str]:
    """
    Returns 'BAYER', 'XTRANS', or None.
    Uses header/meta hints; if a RAW path exists, asks rawpy for ground truth.
    """
    hdr, meta, src_path = _extract_doc_info(doc)

    # Build a searchable blob of header + meta
    def _safe_blob(x) -> str:
        try:
            return str(x)
        except Exception:
            return ""
    blob = (_safe_blob(hdr) + " " + _safe_blob(meta)).upper()

    # Direct tokens first
    if any(k in blob for k in ("X-TRANS", "XTRANS", "FUJIFILM X-TRANS", "X TRANS")):
        return "XTRANS"
    if any(k in blob for k in ("BAYER", "RGGB", "BGGR", "GRBG", "GBRG")):
        return "BAYER"

    # Camera model hint
    model = (str(meta.get("MODEL")
              or meta.get("CameraModel")
              or (hdr.get("MODEL") if hasattr(hdr, "get") else None)
              or "")).upper()
    make  = (str(meta.get("MAKE")
              or (hdr.get("MAKE") if hasattr(hdr, "get") else None)
              or "")).upper()
    if "FUJIFILM" in make and any(tag in model for tag in (
        "X-T1","X-T2","X-T3","X-T4","X-T5",
        "X-E2","X-E2S","X-E3","X-PRO1","X-PRO2","X-PRO3",
        "X-H1","X-H2S","X-S10","X-S20","X100","X70","X30"
    )):
        return "XTRANS"

    # If we have a source path, ask rawpy
    if src_path:
        try:
            import rawpy  # type: ignore
            with rawpy.imread(src_path) as rp:
                if getattr(rp, "xtrans_pattern", None) is not None:
                    return "XTRANS"
                pat = getattr(rp, "raw_pattern", None)
                if pat is not None:
                    pat = np.array(pat)
                    if getattr(pat, "shape", None) == (2, 2):
                        return "BAYER"
        except Exception:
            pass

    return None


def _score_rgb(rgb: np.ndarray) -> float:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    def grad_energy(x):
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.mean(gx*gx + gy*gy))
    e_rg = grad_energy(r - g)
    e_bg = grad_energy(b - g)
    return e_rg + e_bg

def _autodetect_bayer_by_scoring(mono: np.ndarray) -> str:
    candidates = ["RGGB", "BGGR", "GRBG", "GBRG"]
    best_pat, best_score = None, float("inf")
    for pat in candidates:
        try:
            rgb = debayer_fits_fast(mono, pat, cfa_drizzle=False)
            rgb32 = rgb.astype(np.float32, copy=False)
            s = _score_rgb(rgb32)
            if s < best_score:
                best_score, best_pat = s, pat
        except Exception:
            continue
    return best_pat or "RGGB"

def _debayer_xtrans_via_rawpy(src_path: str,
                              use_cam_wb: bool = True,
                              output_bps: int = 16,
                              alg: str = "AHD") -> np.ndarray:
    """
    X-Trans demosaic via rawpy with selectable algorithm: 'AHD' or 'DHT'.
    Returns float32 RGB in [0,1].
    """
    import rawpy  # type: ignore
    alg_map = {
        "AHD": rawpy.DemosaicAlgorithm.AHD,
        "DHT": rawpy.DemosaicAlgorithm.DHT,
    }
    dem = alg_map.get((alg or "AHD").upper(), rawpy.DemosaicAlgorithm.AHD)
    with rawpy.imread(src_path) as rp:
        rgb16 = rp.postprocess(
            demosaic_algorithm=dem,
            no_auto_bright=True,
            gamma=(1.0, 1.0),
            output_bps=output_bps,
            use_camera_wb=use_cam_wb,
            fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off,
            four_color_rgb=False,
            half_size=False,
            bright=1.0,
            highlight_mode=rawpy.HighlightMode.Clip
        )
    return (rgb16.astype(np.float32) / (65535.0 if output_bps == 16 else 255.0))

def _doc_is_managed(dm, doc) -> bool:
    """Best-effort: is this document tracked by DocManager?"""
    if dm is None or doc is None:
        return False
    # Explicit API if you have it
    fn = getattr(dm, "is_managed_document", None)
    if callable(fn):
        try:
            return bool(fn(doc))
        except Exception:
            pass
    # Common internal collections
    for attr in ("_docs", "documents", "open_documents", "all_docs"):
        coll = getattr(dm, attr, None)
        if isinstance(coll, (list, tuple)):
            try:
                return any(d is doc for d in coll)
            except Exception:
                continue
    return False

def _apply_result_to_doc(dm, doc, rgb: np.ndarray, step_name: str = "Debayer"):
    """
    Always apply to the specific 'doc' (never assume 'active' view).
    """

    # --- A) Best: the document can commit its own undoable edit ---
    if hasattr(doc, "apply_edit") and callable(doc.apply_edit):
        meta = dict(getattr(doc, "metadata", {}) or {})
        meta["is_mono"] = False
        meta.setdefault("bit_depth", "32-bit floating point")
        try:
            doc.apply_edit(rgb.copy(), metadata=meta, step_name=step_name)
        except TypeError:
            # older signature: (image, metadata) or (image,)
            try: doc.apply_edit(rgb.copy(), metadata=meta)
            except Exception: doc.apply_edit(rgb.copy())
        _refresh_view_for_doc(dm, doc)
        return

    # --- B) Next best: mutate the doc directly, then refresh its view ---
    try:
        doc.image = rgb
        meta = getattr(doc, "metadata", None)
        if isinstance(meta, dict):
            meta["is_mono"] = False
            meta.setdefault("bit_depth", "32-bit floating point")
    except Exception:
        pass
    _refresh_view_for_doc(dm, doc)
    return

def _refresh_view_for_doc(dm, doc):
    """
    Find the subwindow that displays 'doc' and ask it to repaint,
    without switching the active view.
    """
    try:
        # Try a dedicated API if your app exposes it
        fn = getattr(dm, "refresh_subwindow_for_document", None)
        if callable(fn):
            fn(doc); return
    except Exception:
        pass

    # Generic fallback: walk MDI subwindows and update the one bound to 'doc'
    try:
        mw = getattr(dm, "main_window", None) or getattr(dm, "mw", None)
        if mw is None: return
        mdi = getattr(mw, "mdi", None)
        if mdi is None: return
        for sw in mdi.subWindowList():
            w = getattr(sw, "widget", lambda: None)()
            if getattr(w, "document", None) is doc:
                # Common update hooks
                upd = getattr(w, "refresh_pixmap_from_document", None) \
                      or getattr(w, "refresh_view", None) \
                      or getattr(w, "update_from_doc", None)
                if callable(upd):
                    upd(); return
                # Absolute fallback: force a repaint
                try:
                    w.update(); sw.update()
                except Exception:
                    pass
                return
    except Exception:
        pass


# -------- worker -------------------------------------------------------------

class _DebayerWorker(QThread):
    progress = pyqtSignal(int, str)
    failed = pyqtSignal(str)
    finished = pyqtSignal(np.ndarray, str)  # (rgb, used_pattern)

    def __init__(self, mono: np.ndarray, pattern: str, method: str = "edge"):
        super().__init__()
        self.mono = mono
        self.pattern = pattern
        self.method = (method or "edge")

    def run(self):
        try:
            if debayer_fits_fast is None:
                raise RuntimeError("Numba debayer kernels not available.")

            # enforce kernel-friendly layout/dtype
            img = _mono_as_float32_contig(self.mono)

            if img.ndim != 2:
                raise ValueError("Debayer expects a single-channel (mosaic) image.")
            if self.pattern not in _VALID:
                raise ValueError(f"Unsupported pattern: {self.pattern}")

            self.progress.emit(5, f"Debayering ({self.pattern}, {self.method}) …")
            rgb = debayer_fits_fast(img, self.pattern, cfa_drizzle=False, method=self.method)
            self.progress.emit(96, "Finalizing …")
            self.finished.emit(rgb, self.pattern)
        except Exception as e:
            self.failed.emit(str(e))

def _extract_doc_info(doc) -> tuple[dict | None, dict, Optional[str]]:
    meta = getattr(doc, "metadata", {}) or {}
    hdr = (meta.get("original_header")
           or meta.get("fits_header")
           or meta.get("header")
           or getattr(doc, "header", None))

    # try multiple fields for a source path
    def _first_nonempty(*vals):
        for v in vals:
            if v:
                return v
        return None

    # header cards that might store original RAW path
    hdr_raw = None
    try:
        if hdr is not None:
            for k in ("RAW_PATH","RAWFILE","ORIGFILE","ORIGINAL","ORIGPATH","RAWORIG","SOURCE","SRCFILE"):
                v = hdr.get(k) if hasattr(hdr, "get") else hdr[k]  # may raise → caught
                if v:
                    hdr_raw = str(v)
                    break
    except Exception:
        pass

    path = _first_nonempty(
        meta.get("raw_source_path"),
        hdr_raw,
        meta.get("file_path"),
        getattr(doc, "path", None),
        getattr(doc, "file_path", None),
    )
    return hdr, meta, path

def _mono_as_float32_contig(arr: np.ndarray) -> np.ndarray:
    """
    Ensure mono mosaic is 2D, C-contiguous, float32 in [0,1] for numba kernels.
    Scales integer inputs by their max (8/16/32 bits).
    """
    a = np.asarray(arr)
    if a.ndim != 2:
        raise RuntimeError("Debayer expects a single-channel (mosaic) image.")
    if np.issubdtype(a.dtype, np.integer):
        # pick a sensible scale based on dtype
        info = np.iinfo(a.dtype)
        a = a.astype(np.float32, copy=False) / float(info.max if info.max > 0 else 1.0)
    else:
        a = a.astype(np.float32, copy=False)
        # if it looks like 0..65535 in float, normalize too
        if a.max() > 2.0:
            a = a / 65535.0
    return np.ascontiguousarray(a)


# -------- dialog -------------------------------------------------------------

class DebayerDialog(QDialog):
    """
    One-shot debayer UI for the active view. Uses your numba kernels.
    If the image is already RGB, will warn and exit.
    """
    def __init__(self, parent, doc_manager, active_doc):
        super().__init__(parent)
        self.setWindowTitle("Debayer")
        self.dm = doc_manager
        self.doc = active_doc
        self.worker: Optional[_DebayerWorker] = None

        img = getattr(active_doc, "image", None)
        if img is None:
            raise RuntimeError("No image in active document.")
        arr = np.asarray(img)

        # Reject non-mosaic early
        if arr.ndim == 3 and arr.shape[2] >= 3:
            QMessageBox.information(self, "Debayer", "Image already has 3 channels.")
            self.setEnabled(False)
            self.close()
            return
        if arr.ndim != 2:
            QMessageBox.warning(self, "Debayer", "Only single-channel mosaics can be debayered.")
            self.setEnabled(False)
            self.close()
            return

        # ✅ normalize for numba kernels
        self._src = _mono_as_float32_contig(arr)

        # detect CFA family (BAYER/XTRANS/None)
        self._cfa_family = _detect_cfa_family(active_doc)

        v = QVBoxLayout(self)

        # pattern selection
        detected = _detect_bayer_from_header(active_doc)
        self._detected_pattern = detected  # store for later

        gb = QGroupBox("Bayer pattern", self)
        h = QHBoxLayout(gb)
        self.combo_pattern = QComboBox(self)
        self.combo_pattern.addItems([
            "Auto (from header)",
            "RGGB", "BGGR", "GRBG", "GBRG",
        ])
        self.combo_pattern.setCurrentIndex(0)
        self.lbl_detect = QLabel(f"Detected: {detected or '(unknown)'}")
        h.addWidget(self.combo_pattern, 1)
        h.addWidget(self.lbl_detect)
        v.addWidget(gb)

        if self._cfa_family == 'XTRANS':
            self.combo_pattern.setEnabled(False)
            self.lbl_detect.setText("Detected: X-Trans (rawpy)")
        else:
            norm = _normalize_bayer_token(self._detected_pattern or "")
            self.lbl_detect.setText(f"Detected: {norm or '(unknown)'}")

        self.method_group = QGroupBox("Method", self)
        hm = QHBoxLayout(self.method_group)
        self.combo_method = QComboBox(self)

        if self._cfa_family == 'XTRANS':
            for label, _tok in _XTRANS_METHODS:
                self.combo_method.addItem(label)
        else:
            for label, _tok in _BAYER_METHODS:
                self.combo_method.addItem(label)

        print(f"[Debayer] CFA family auto-detect → {self._cfa_family}  "
            f"path={_extract_doc_info(active_doc)[2]}  "
            f"model={getattr(active_doc, 'metadata', {}).get('MODEL')}")

        self.combo_method.setCurrentIndex(0)
        hm.addWidget(self.combo_method)
        hm.addStretch(1)
        v.addWidget(self.method_group)

        # progress + buttons
        self.status = QLabel("")
        self.bar = QProgressBar(self); self.bar.setRange(0, 100)
        v.addWidget(self.status)
        v.addWidget(self.bar)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self._go)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)


    def _chosen_pattern(self) -> str:
        if self._cfa_family == 'XTRANS':
            return "XTRANS"
        txt = self.combo_pattern.currentText()
        if txt.startswith("Auto"):
            norm = _normalize_bayer_token(self._detected_pattern or "")
            return norm or _autodetect_bayer_by_scoring(self._src)
        return txt

    def _go(self):
        pat = self._chosen_pattern()
        method_label = self.combo_method.currentText()

        # X-Trans path
        if pat == "XTRANS":
            src_path = (getattr(self.doc, "file_path", None) or getattr(self.doc, "path", None)
                        or (getattr(self.doc, "metadata", {}) or {}).get("file_path"))
            if not src_path or not str(src_path).lower().endswith(_RAW_EXTS):
                # try sibling RAW next to this file
                sib = _find_raw_sibling(getattr(self.doc, "file_path", None) or (getattr(self.doc, "metadata", {}) or {}).get("file_path"))
                if sib:
                    src_path = sib
                else:
                    QMessageBox.warning(self, "Debayer",
                        "X-Trans detected, but original RAW path was not found.\n"
                        "Open the RAF directly, or embed RAW_PATH in the header, or place the RAW next to the file.")
                    return
            # map label → rawpy alg token
            alg = next((tok for (label, tok) in _XTRANS_METHODS if label == method_label), "AHD")
            try:
                self.status.setText(f"Demosaicing X-Trans via rawpy ({alg}) …")
                self.bar.setValue(10)
                rgb = _debayer_xtrans_via_rawpy(src_path, use_cam_wb=True, output_bps=16, alg=alg)
                self.bar.setValue(96)
                _apply_result_to_doc(self.dm, self.doc, rgb.astype(np.float32, copy=False),
                                    step_name=f"Debayer (X-Trans/{alg})")
                self.status.setText("Done.")
                self.accept()
            except Exception as e:
                QMessageBox.critical(self, "Debayer", f"X-Trans demosaic failed:\n{e}")
                self.status.setText("Failed.")
            return

        # Bayer path (Numba) with method
        if pat not in _VALID:
            QMessageBox.warning(self, "Debayer", "Unknown pattern (auto-detect failed). Choose a pattern explicitly.")
            return

        bayer_method = next((tok for (label, tok) in _BAYER_METHODS if label == method_label), "edge")
        self.status.setText(f"Debayering as {pat} ({bayer_method}) …")
        self.bar.setValue(0)
        self.worker = _DebayerWorker(self._src, pat, method=bayer_method)
        self.worker.progress.connect(self._on_prog)
        self.worker.failed.connect(self._on_fail)
        self.worker.finished.connect(self._on_done)
        self.worker.start()


    def _on_prog(self, p: int, msg: str):
        self.bar.setValue(p); self.status.setText(msg)

    def _on_fail(self, err: str):
        QMessageBox.critical(self, "Debayer", err)
        self.status.setText("Failed.")

    def _on_done(self, rgb: np.ndarray, used_pattern: str):
        # Hand back to doc manager with an undo step name
        _apply_result_to_doc(self.dm, self.doc, rgb, step_name=f"Debayer ({used_pattern})")
        self.status.setText("Done.")
        self.accept()


# -------- headless (shortcut / DnD) -----------------------------------------

def apply_debayer_preset_to_doc(dm, doc, preset: dict) -> Tuple[str, np.ndarray]:
    """
    preset = {
        "pattern": "auto|RGGB|BGGR|GRBG|GBRG",
        "method":  "auto|edge|bilinear|AHD|DHT"
    }
    Returns (used_pattern, rgb_array).
    """
    if getattr(doc, "image", None) is None:
        raise RuntimeError("No image in document.")

    # ✅ normalize for numba kernels & ensure 2D
    mono_in = np.asarray(doc.image)
    if mono_in.ndim != 2:
        raise RuntimeError("Debayer expects a single-channel (mosaic) image.")
    mono = _mono_as_float32_contig(mono_in)

    family = _detect_cfa_family(doc)
    want_method = str(preset.get("method", "auto"))

    # X-Trans → rawpy path
    if family == "XTRANS":
        src_path = (getattr(doc, "file_path", None) or getattr(doc, "path", None)
                    or (getattr(doc, "metadata", {}) or {}).get("file_path"))

        if not src_path or not str(src_path).lower().endswith(_RAW_EXTS):
            hdr, meta, _ = _extract_doc_info(doc)
            src_path = (meta.get("raw_source_path") or
                        (hdr.get("RAW_PATH") if hasattr(hdr, "get") else None) or
                        _find_raw_sibling(meta.get("file_path") or getattr(doc, "file_path", None)))
        if not src_path or not str(src_path).lower().endswith(_RAW_EXTS):
            raise RuntimeError("X-Trans detected, but no RAW found. "
                               "Embed RAW_PATH/raw_source_path or place the RAW next to the file.")
        alg = want_method if want_method in ("AHD", "DHT") else "AHD"
        rgb = _debayer_xtrans_via_rawpy(src_path, use_cam_wb=True, output_bps=16, alg=alg)
        _apply_result_to_doc(dm, doc, rgb, step_name=f"Debayer (X-Trans/{alg})")
        return "XTRANS", rgb

    # Bayer → Numba path
    want = str(preset.get("pattern", "auto")).upper()
    if want == "AUTO":
        pat = _normalize_bayer_token(_detect_bayer_from_header(doc) or "")
        if pat not in _VALID:
            pat = _autodetect_bayer_by_scoring(mono)
    else:
        pat = want
    if pat not in _VALID:
        raise ValueError(f"Unsupported Bayer pattern: {pat}")

    method_tok = (want_method.lower() if want_method.lower() in ("edge", "bilinear") else "edge")

    if debayer_fits_fast is None:
        raise RuntimeError("Numba debayer kernels not available.")

    rgb = debayer_fits_fast(mono, pat, cfa_drizzle=False, method=method_tok)
    _apply_result_to_doc(dm, doc, rgb, step_name=f"Debayer ({pat}/{method_tok})")
    return pat, rgb


