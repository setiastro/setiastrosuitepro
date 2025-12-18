# pro/star_alignment_preset.py
from __future__ import annotations
import os
import json
import re
import numpy as np
import cv2
import astroalign
from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QDialogButtonBox, QComboBox, QLineEdit, QCheckBox,
    QSpinBox, QPushButton, QFileDialog, QWidget, QMessageBox
)
from PyQt6.QtCore import Qt

def _norm_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    # strip common UI decorations
    s = re.sub(r"^[■●◆▲▪▫•◼◻◾◽]\s*", "", s)
    s = re.sub(r"^Active View:\s*", "", s, flags=re.I)
    # ignore extensions when users type a short name
    base = os.path.splitext(s)[0]
    return base.casefold()

def _iter_views_with_titles(app):
    """Yield (visible_title, doc, subwin). Visible title honors per-view overrides."""
    mdi = getattr(app, "mdi", None)
    if mdi is None:
        return
    for sw in mdi.subWindowList() or []:
        try:
            w = sw.widget()
            doc = getattr(w, "document", None)
            if doc is None:
                continue
            if hasattr(w, "_effective_title"):
                title = str(w._effective_title())
            else:
                title = str(sw.windowTitle() or doc.display_name())
            yield (title, doc, sw)
        except Exception:
            continue

def _resolve_ref_doc_by_view_name(app, name: str):
    """Return (doc or None, debug_list)."""
    wanted = _norm_name(name)
    if not wanted:
        return None, []

    views = list(_iter_views_with_titles(app))
    seen = [t for (t, _, __) in views]

    # 1) exact title or exact doc name
    for t, d, _ in views:
        if _norm_name(t) == wanted or _norm_name(getattr(d, "display_name")()) == wanted:
            return d, seen

    # 2) prefix match
    for t, d, _ in views:
        if _norm_name(t).startswith(wanted) or _norm_name(getattr(d, "display_name")()).startswith(wanted):
            return d, seen

    # 3) substring
    for t, d, _ in views:
        if wanted in _norm_name(t) or wanted in _norm_name(getattr(d, "display_name")()):
            return d, seen

    return None, seen

# ---------------------------
# Preset dialog (like others)
# ---------------------------
class StarAlignmentPresetDialog(QDialog):
    """
    Preset UI for headless Star Alignment:
      ref_mode:  active | view_name | file
      ref_name:  (string, used when ref_mode=view_name)
      ref_file:  (path,   used when ref_mode=file)
      overwrite: bool
      downsample:int (>=1)
    """
    def __init__(self, parent: QWidget | None = None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Star Alignment — Preset")
        init = dict(initial or {})

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["active", "view_name", "file"])
        self.cmb_mode.setCurrentText(str(init.get("ref_mode", "active")).lower())

        self.edt_name = QLineEdit()
        self.edt_name.setPlaceholderText("Exact view title (case-insensitive)")
        self.edt_name.setText(str(init.get("ref_name", "")))

        self.edt_file = QLineEdit()
        self.edt_file.setPlaceholderText("/absolute/path/to/reference.fits|fit|xisf|tif")
        self.edt_file.setText(str(init.get("ref_file", "")))
        self.btn_browse = QPushButton("Browse…")
        def _pick_file():
            p, _ = QFileDialog.getOpenFileName(self, "Choose reference file", "",
                "Images (*.fits *.fit *.xisf *.tif *.tiff *.png *.jpg);;All Files (*)")
            if p:
                self.edt_file.setText(p)
        self.btn_browse.clicked.connect(_pick_file)

        self.chk_overwrite = QCheckBox("Overwrite target view")
        self.chk_overwrite.setChecked(bool(init.get("overwrite", False)))

        self.spin_down = QSpinBox()
        self.spin_down.setRange(1, 8)
        self.spin_down.setValue(int(init.get("downsample", 2)))
        self.spin_down.setToolTip("Downsample factor during transform solve (speed vs precision)")

        lay = QFormLayout(self)
        lay.addRow("Reference mode:", self.cmb_mode)
        lay.addRow("View name (if view_name):", self.edt_name)
        rowf = QWidget(); rf = QFormLayout(rowf); rf.setContentsMargins(0,0,0,0)
        rf.addRow(self.edt_file, self.btn_browse)
        lay.addRow("File (if file):", rowf)
        lay.addRow(self.chk_overwrite)
        lay.addRow("Downsample factor:", self.spin_down)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        lay.addRow(btns)

        # enable/disable inputs per mode
        def _upd():
            m = self.cmb_mode.currentText()
            self.edt_name.setEnabled(m == "view_name")
            self.edt_file.setEnabled(m == "file")
            self.btn_browse.setEnabled(m == "file")
        self.cmb_mode.currentTextChanged.connect(_upd); _upd()

    def result_dict(self) -> dict:
        return {
            "ref_mode":   self.cmb_mode.currentText(),
            "ref_name":   self.edt_name.text().strip(),
            "ref_file":   self.edt_file.text().strip(),
            "overwrite":  bool(self.chk_overwrite.isChecked()),
            "downsample": int(self.spin_down.value()),
        }


# ---------------------------
# Headless runner
# ---------------------------
def run_star_alignment_via_preset(app, preset: dict, target_doc) -> None:
    """
    app: main window (has mdi, docman/doc_manager, _log)
    target_doc: document object for the drop target
    Behavior mirrors other *_via_preset helpers: overwrite or create new view.
    """
    from setiastro.saspro.legacy.image_manager import load_image  # for file ref

    # ---- resolve target image ----
    if target_doc is None or getattr(target_doc, "image", None) is None:
        raise RuntimeError("Target view has no image.")
    tgt = np.asarray(target_doc.image, dtype=np.float32)
    if tgt.ndim not in (2, 3):
        raise RuntimeError("Unsupported target image shape.")

    # ---- resolve reference image per preset ----
    pp   = dict(preset or {})
    mode = str(pp.get("ref_mode", "active")).lower()

    def _doc_display_name(d):
        try:
            if callable(getattr(d, "display_name", None)):
                return d.display_name() or ""
            nm = getattr(d, "title", None)
            return nm or ""
        except Exception:
            return ""

    ref = None
    ref_name = "Reference"

    if mode == "file":
        path = pp.get("ref_file") or ""
        if not path or not os.path.exists(path):
            raise RuntimeError("Reference file does not exist.")
        ref, _, _, _ = load_image(path)
        if ref is None:
            raise RuntimeError("Failed to load reference file.")
        ref = np.asarray(ref, dtype=np.float32)
        ref_name = os.path.basename(path)

    elif mode == "view_name":
        # NEW: resolve against visible titles (per-view overrides) w/ normalization
        raw_name = (pp.get("ref_name") or "").strip()
        if not raw_name:
            raise RuntimeError("Reference view name is empty.")

        ref_doc, seen_titles = _resolve_ref_doc_by_view_name(app, raw_name)
        if ref_doc is None or getattr(ref_doc, "image", None) is None:
            # helpful error includes visible open view titles (no glyphs/exts)
            opts = ", ".join(os.path.splitext(t)[0] for t in seen_titles) if seen_titles else "(no open views)"
            raise RuntimeError(f"Reference view '{raw_name}' not found.\nOpen views: {opts}")

        ref = np.asarray(ref_doc.image, dtype=np.float32)

        # Prefer the visible title of the matched subwindow for the label
        ref_name = None
        for t, d, _ in _iter_views_with_titles(app):
            if d is ref_doc:
                ref_name = os.path.splitext(t)[0]
                break
        ref_name = ref_name or (_doc_display_name(ref_doc) or "Reference")

    else:  # "active" (default) → use the app's active subwindow doc
        sw = getattr(app, "mdi", None).activeSubWindow() if getattr(app, "mdi", None) else None
        if sw is None:
            raise RuntimeError("No active view to use as reference.")
        ref_doc = getattr(sw.widget(), "document", None)
        if ref_doc is None or getattr(ref_doc, "image", None) is None:
            raise RuntimeError("Active view has no image to use as reference.")
        ref = np.asarray(ref_doc.image, dtype=np.float32)
        # Prefer visible title if available
        try:
            for t, d, _ in _iter_views_with_titles(app):
                if d is ref_doc:
                    ref_name = os.path.splitext(t)[0]
                    break
            if ref_name == "Reference":
                ref_name = _doc_display_name(ref_doc) or "Reference"
        except Exception:
            ref_name = _doc_display_name(ref_doc) or "Reference"

    # ---- grayscale + optional downsample for solve ----
    def _to_gray(a):
        return np.mean(a, axis=2) if a.ndim == 3 else a

    ds = int(max(1, pp.get("downsample", 2)))
    ref_g = _to_gray(ref); tgt_g = _to_gray(tgt)
    if ds > 1:
        wR, hR = ref_g.shape[1] // ds, ref_g.shape[0] // ds
        wT, hT = tgt_g.shape[1] // ds, tgt_g.shape[0] // ds
        ref_s = cv2.resize(ref_g, (max(1, wR), max(1, hR)), interpolation=cv2.INTER_AREA)
        tgt_s = cv2.resize(tgt_g, (max(1, wT), max(1, hT)), interpolation=cv2.INTER_AREA)
    else:
        ref_s, tgt_s = ref_g, tgt_g

    # ---- find transform (map target → reference) with gentle backoff ----
    tries = [
        dict(detection_sigma=5,  min_area=7,  max_control_points=75),
        dict(detection_sigma=12, min_area=9,  max_control_points=75),
        dict(detection_sigma=20, min_area=9,  max_control_points=75),
        dict(detection_sigma=30, min_area=11, max_control_points=75),
    ]
    last = None
    M = None
    for kw in tries:
        try:
            T, _ = astroalign.find_transform(tgt_s.astype(np.float32), ref_s.astype(np.float32), **kw)
            M = T.params[0:2, :].astype(np.float32)
            break
        except Exception as e:
            last = e
            # bump SEP pixstack if that was the issue
            try:
                import sep
                if "pixel buffer full" in str(e).lower():
                    sep.set_extract_pixstack(int(sep.get_extract_pixstack() * 2))
            except Exception:
                pass
    if M is None:
        raise RuntimeError(f"Astroalign failed: {last}")

    if ds > 1:  # scale the translation back up
        M = M.copy()
        M[0, 2] *= ds
        M[1, 2] *= ds

    # ---- warp target to reference geometry ----
    H, W = ref_g.shape[:2]
    if tgt.ndim == 2:
        aligned = cv2.warpAffine(
            tgt, M, (W, H),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
    else:
        planes = [
            cv2.warpAffine(
                tgt[..., i], M, (W, H),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            ) for i in range(tgt.shape[2])
        ]
        aligned = np.stack(planes, axis=2)
    aligned = aligned.astype(np.float32, copy=False)

    # ---- write back (overwrite or new view) ----
    overwrite = bool(pp.get("overwrite", False))
    if overwrite:
        if hasattr(target_doc, "set_image"):
            target_doc.set_image(aligned, step_name=f"Star Alignment → {ref_name}")
        elif hasattr(target_doc, "apply_numpy"):
            target_doc.apply_numpy(aligned, step_name=f"Star Alignment → {ref_name}")
        else:
            target_doc.image = aligned
        try:
            if hasattr(target_doc, "changed"):
                target_doc.changed.emit()
        except Exception:
            pass
    else:
        dm = getattr(app, "docman", None) or getattr(app, "doc_manager", None)
        if dm is None:
            raise RuntimeError("Document manager not available.")
        # title
        base = ""
        try:
            base = target_doc.display_name() if callable(getattr(target_doc, "display_name", None)) \
                   else (getattr(target_doc, "title", None) or "")
        except Exception:
            pass
        base = base or "Image"
        title = f"{base} [Aligned → {ref_name}]"
        meta = {
            "step_name":   "Star Alignment",
            "description": f"Aligned to {ref_name}",
            "is_mono": bool(aligned.ndim == 2 or (aligned.ndim == 3 and aligned.shape[2] == 1)),
        }
        newdoc = dm.open_array(aligned, metadata=meta, title=title)
        if hasattr(app, "_spawn_subwindow_for"):
            app._spawn_subwindow_for(newdoc)
