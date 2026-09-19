from __future__ import annotations

import json
import os
import platform
import re
from dataclasses import dataclass

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QProgressDialog, QPushButton, QRadioButton,
    QVBoxLayout, QWidget,
)

from setiastro.saspro.file_utils import sanitize_filename

_HEADER_KEYS = ("fits_header", "original_header", "header")
_WCS_STRUCTURAL = {
    "SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "NAXIS4",
    "BSCALE", "BZERO", "EXTEND", "END", "PCOUNT", "GCOUNT",
}
_SAVE_MUTATED_KEYS = (
    "file_path", "original_format", "bit_depth", "original_header", "fits_header",
)

COMMAND_ID = "export_fits"
_SETTINGS_KEY = "paths/last_export_fits_dir"
_SPEC_SETTINGS_KEY = "export_fits/spec_v1"

# Formats offered by the merged tool. "source" == keep the source file's
# extension (the historical Checkpoint Save behaviour).
FORMAT_CHOICES = (
    ("Same as source", "source"),
    ("FITS (.fits)",    "fits"),
    ("XISF (.xisf)",    "xisf"),
    ("TIFF (.tiff)",    "tiff"),
    ("PNG (.png)",      "png"),
    ("JPEG (.jpg)",     "jpeg"),
    ("WebP (.webp)",    "webp"),
    ("PSB (.psb)",      "psb"),
)

_FORMAT_EXT = {
    "source": None,  # resolved from the source path at build time
    "fits": ".fits", "fit": ".fits", "fts": ".fits",
    "xisf": ".xisf",
    "tiff": ".tiff", "tif": ".tiff",
    "png": ".png",
    "jpeg": ".jpg", "jpg": ".jpg",
    "webp": ".webp",
    "psb": ".psb",
}

# Numbered-checkpoint naming, ported from the old Checkpoint Save.
_PROC_SCAN_RE = re.compile(r"^(?P<prefix>.+)_proc(?P<n>\d+)(?P<tag>_[^.]*)?$", re.IGNORECASE)

try:
    from setiastro.saspro.resources import disk_path
except Exception:
    disk_path = None


@dataclass
class ExportFitsResult:
    ok: bool
    path: str | None = None
    skipped: bool = False
    reason: str = ""


# --------------------------------------------------------------------------- #
# Naming helpers (ported from Checkpoint Save)
# --------------------------------------------------------------------------- #
def _best_name(doc) -> str:
    try:
        from setiastro.saspro.main_helpers import best_doc_name
        return best_doc_name(doc)
    except Exception:
        meta = getattr(doc, "metadata", {}) or {}
        return str(meta.get("display_name") or meta.get("name") or "untitled")


def _split_stars_suffix(name_no_ext: str):
    if name_no_ext.lower().endswith("_stars"):
        return name_no_ext[:-6], True
    return name_no_ext, False


def _strip_proc_tail(base_no_ext: str) -> str:
    """If base already ends in _procN or _procN_tag, strip that whole tail."""
    m = _PROC_SCAN_RE.match(base_no_ext)
    return m.group("prefix") if m else base_no_ext


def _next_proc_index_for(prefix: str, workdir: str, stars: bool, tag: str, ext: str) -> int:
    """Next free N for files matching prefix_procN{tag}{_stars}{ext} in workdir."""
    tag_part = tag or ""
    stars_part = "_stars" if stars else ""
    patt = re.compile(
        r"^" + re.escape(prefix)
        + r"_proc(\d+)"
        + re.escape(tag_part)
        + re.escape(stars_part)
        + re.escape(ext)
        + r"$",
        re.IGNORECASE,
    )
    max_n = 0
    try:
        for fn in os.listdir(workdir):
            m = patt.match(fn)
            if m:
                max_n = max(max_n, int(m.group(1)))
    except Exception:
        pass
    return max_n + 1


def _source_fs_path(doc) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    orig = str(meta.get("file_path", "") or "")
    if "::" in orig:
        orig = orig.split("::", 1)[0]
    return orig


def export_fits_stem(doc) -> str:
    orig = _source_fs_path(doc)
    if orig:
        stem = os.path.splitext(os.path.basename(orig))[0]
        if stem:
            return sanitize_filename(stem) or "untitled"
    name = _best_name(doc)
    stem = os.path.splitext(str(name or "untitled"))[0]
    return sanitize_filename(stem) or "untitled"


def _ext_for_format(fmt: str, source_path: str) -> str:
    fmt = (fmt or "source").strip().lower()
    if fmt.startswith("."):
        fmt = fmt[1:]
    mapped = _FORMAT_EXT.get(fmt, None)
    if fmt == "source" or mapped is None:
        e = os.path.splitext(source_path or "")[1].lower()
        if e:
            return e
        if mapped:  # unknown fmt but had a mapping? (won't happen) - fall through
            return mapped
        # source requested but no source ext known -> FITS is the safe default
        return ".fits" if fmt in ("source", "") else ("." + fmt)
    return mapped


# --------------------------------------------------------------------------- #
# Preset schema (superset; backward compatible with old {out_dir, overwrite})
# --------------------------------------------------------------------------- #
def normalize_export_fits_preset(preset: dict | None) -> dict:
    """
    Normalise any preset (old or new) to the full merged schema.

    Old export_fits presets only carried {out_dir, overwrite}; their absent keys
    default to the *old* behaviour (folder destination, FITS, no numbering, no
    pairing), so existing canvas shortcuts and Function Bundles keep working.
    """
    p = dict(preset or {})
    out_dir = str(p.get("out_dir") or p.get("folder") or p.get("output_dir") or "").strip()

    dest_mode = str(p.get("dest_mode") or "").strip().lower()
    if not dest_mode:
        dest_mode = "beside_source" if p.get("beside_source") else "folder"
    if dest_mode not in ("folder", "beside_source"):
        dest_mode = "folder"

    fmt = str(p.get("format") or p.get("fmt") or "fits").strip().lower()

    tag = str(p.get("tag") or "").strip()
    if tag and not tag.startswith("_"):
        tag = "_" + tag

    return {
        "dest_mode": dest_mode,
        "out_dir": out_dir,
        "overwrite": bool(p.get("overwrite", True)),
        "format": fmt,
        "auto_number": bool(p.get("auto_number", False)),
        "tag": tag,
        "include_paired_stars": bool(p.get("include_paired_stars", False)),
        "mode_all_open": bool(p.get("mode_all_open", False)),
        # internal: force a specific _procN (used to pair a _stars companion)
        "_force_number": p.get("_force_number"),
    }


def checkpoint_preset() -> dict:
    """The one-click Checkpoint Save identity: beside source, auto-numbered,
    same format as source, paired stars, active view only."""
    return normalize_export_fits_preset({
        "dest_mode": "beside_source",
        "format": "source",
        "auto_number": True,
        "include_paired_stars": True,
        "mode_all_open": False,
        "overwrite": True,
    })


# --------------------------------------------------------------------------- #
# Header resolution (unchanged from the user's export process)
# --------------------------------------------------------------------------- #
def _as_fits_header(value):
    if value is None:
        return None
    try:
        from astropy.io.fits import Header
    except Exception:
        return None
    if isinstance(value, Header):
        return value
    if isinstance(value, dict):
        hdr = Header()
        for key, val in value.items():
            if isinstance(key, str) and key.startswith("XISF:"):
                continue
            if isinstance(val, dict) and "value" in val:
                val = val["value"]
            try:
                hdr[str(key)] = val
            except Exception:
                continue
        return hdr if len(hdr) else None
    return None


def _header_from_snapshot(snap):
    if not isinstance(snap, dict):
        return None
    try:
        from setiastro.saspro.project_io import _deserialize_header_any
    except Exception:
        return None
    try:
        return _as_fits_header(_deserialize_header_any(snap))
    except Exception:
        return None


def _wcs_header_from_doc(doc, meta: dict):
    sources = (
        meta.get("wcs_header"),
        meta.get("wcs"),
        getattr(doc, "wcs", None),
    )
    try:
        from astropy.wcs import WCS
    except Exception:
        WCS = None
    for src in sources:
        if src is None:
            continue
        if WCS is not None:
            try:
                if isinstance(src, WCS):
                    return src.to_header(relax=True)
            except Exception:
                pass
        hdr = _as_fits_header(src)
        if hdr is not None:
            return hdr
    return None


def _merge_wcs_cards(hdr, wcs_hdr):
    if hdr is None:
        return wcs_hdr
    if wcs_hdr is None:
        return hdr
    out = hdr.copy() if hasattr(hdr, "copy") else hdr
    for key, val in wcs_hdr.items():
        if key in _WCS_STRUCTURAL:
            continue
        try:
            out[key] = val
        except Exception:
            continue
    return out


def resolve_header_for_export(doc):
    """Best FITS header for export: live cards, then snapshot, then WCS."""
    meta = getattr(doc, "metadata", None)
    if not isinstance(meta, dict):
        meta = {}
    hdr = None
    for key in _HEADER_KEYS:
        hdr = _as_fits_header(meta.get(key))
        if hdr is not None:
            break
    if hdr is None:
        hdr = _as_fits_header(getattr(doc, "original_header", None))
    if hdr is None:
        hdr = _header_from_snapshot(meta.get("__header_snapshot__"))
    wcs_hdr = _wcs_header_from_doc(doc, meta)
    merged = _merge_wcs_cards(hdr, wcs_hdr)
    if merged is None:
        return None
    return merged.copy() if hasattr(merged, "copy") else merged


# --------------------------------------------------------------------------- #
# Path building
# --------------------------------------------------------------------------- #
def _naming_parts(doc, spec: dict):
    """Return (target_dir, prefix, ext, is_stars, tag) or None if unresolvable."""
    source = _source_fs_path(doc)

    if spec["dest_mode"] == "beside_source":
        if not source:
            return None
        target_dir = os.path.dirname(source)
        base_stem = os.path.splitext(os.path.basename(source))[0]
    else:
        target_dir = spec["out_dir"]
        if not target_dir:
            return None
        base_stem = export_fits_stem(doc)

    if not target_dir:
        return None

    ext = _ext_for_format(spec["format"], source)

    # Stars companion detection: from the on-screen doc title OR the base stem.
    try:
        name = _best_name(doc)
    except Exception:
        name = base_stem
    title_stem = os.path.splitext(str(name or base_stem))[0]
    base_stem, base_stars = _split_stars_suffix(base_stem)
    _, title_stars = _split_stars_suffix(title_stem)
    is_stars = base_stars or title_stars

    tag = spec.get("tag") or ""

    # Only strip a proc tail when we're (re)building a numbered series; leaving it
    # alone otherwise preserves the exact filename old export_fits presets produce.
    prefix = _strip_proc_tail(base_stem) if spec["auto_number"] else base_stem
    prefix = sanitize_filename(prefix) or "untitled"
    return target_dir, prefix, ext, is_stars, tag


def _compute_target(doc, spec: dict):
    """Return (path, ext, reason). path is None (with a reason) when skipped."""
    parts = _naming_parts(doc, spec)
    if parts is None:
        if spec["dest_mode"] == "beside_source":
            return None, "", ("Unsaved document — no source file to sit beside. "
                              "Choose 'Into folder' instead.")
        return None, "", "No output folder chosen."

    target_dir, prefix, ext, is_stars, tag = parts
    stars_part = "_stars" if is_stars else ""

    if spec["auto_number"]:
        n = spec.get("_force_number")
        if not n:
            n = _next_proc_index_for(prefix, target_dir, is_stars, tag, ext)
        out_base = f"{prefix}_proc{int(n)}{tag}{stars_part}"
    else:
        out_base = f"{prefix}{tag}{stars_part}"

    return os.path.join(os.path.abspath(str(target_dir)), out_base + ext), ext, ""


def _peek_number(doc, spec: dict):
    parts = _naming_parts(doc, spec)
    if parts is None:
        return None
    target_dir, prefix, ext, is_stars, tag = parts
    return _next_proc_index_for(prefix, target_dir, is_stars, tag, ext)


# Kept for callers that want the plain FITS-into-folder path (old signature).
def export_fits_path(doc, out_dir: str) -> str:
    return os.path.join(os.path.abspath(str(out_dir)), f"{export_fits_stem(doc)}.fits")


# --------------------------------------------------------------------------- #
# Core single-document save (format-aware; keeps the doc untouched on disk)
# --------------------------------------------------------------------------- #
def export_document_as_fits(docman, doc, preset: dict | None = None) -> ExportFitsResult:
    """
    Write `doc` to disk per `preset` (destination / format / numbering / tag).

    Does not retarget the in-memory document (file_path / dirty stay unchanged).
    Name kept for backward compatibility with existing shortcut / bundle callers.
    """
    spec = normalize_export_fits_preset(preset)
    if doc is None or getattr(doc, "image", None) is None:
        return ExportFitsResult(ok=False, skipped=True, reason="No image to export")

    path, ext, reason = _compute_target(doc, spec)
    if path is None:
        return ExportFitsResult(ok=False, skipped=True, reason=reason)

    target_dir = os.path.dirname(path)
    try:
        os.makedirs(target_dir, exist_ok=True)
    except Exception as e:
        return ExportFitsResult(ok=False, skipped=True, path=path, reason=f"Cannot create folder: {e}")

    # Overwrite guard only bites when we're NOT auto-numbering (numbering already
    # guarantees a fresh name).
    if (not spec["auto_number"]) and (not spec["overwrite"]) and os.path.exists(path):
        return ExportFitsResult(ok=False, skipped=True, path=path, reason="File exists")

    meta = getattr(doc, "metadata", None)
    if not isinstance(meta, dict):
        meta = {}
        if hasattr(doc, "metadata"):
            doc.metadata = meta
    dirty = bool(getattr(doc, "dirty", False))
    restore = [(key, key in meta, meta.get(key)) for key in _SAVE_MUTATED_KEYS]
    bit_depth = meta.get("bit_depth")

    ext_low = ext.lower()
    is_fits_like = ext_low in (".fits", ".fit", ".fts", ".xisf")
    hdr = resolve_header_for_export(doc) if is_fits_like else None
    if hdr is not None:
        meta["original_header"] = hdr
        meta["fits_header"] = hdr

    jpeg_quality = meta.get("jpeg_quality") if ext_low in (".jpg", ".jpeg") else None

    try:
        docman.save_document(
            doc, path,
            bit_depth_override=bit_depth,
            jpeg_quality=jpeg_quality,
            export_opts=None,
        )
    except Exception as e:
        return ExportFitsResult(ok=False, skipped=False, path=path, reason=str(e))
    finally:
        for key, existed, val in restore:
            if existed:
                meta[key] = val
            else:
                meta.pop(key, None)
        if hasattr(doc, "metadata"):
            doc.metadata = meta
        if hasattr(doc, "dirty"):
            doc.dirty = dirty
    return ExportFitsResult(ok=True, path=path)


# --------------------------------------------------------------------------- #
# Settings helpers
# --------------------------------------------------------------------------- #
def _last_export_dir(mw, preset: dict | None = None) -> str:
    spec = normalize_export_fits_preset(preset)
    if spec["out_dir"] and os.path.isdir(spec["out_dir"]):
        return spec["out_dir"]
    settings = getattr(mw, "settings", None)
    if settings is not None:
        for key in (_SETTINGS_KEY, "paths/last_save_dir"):
            try:
                v = settings.value(key, "", type=str) or ""
            except Exception:
                v = ""
            if v and os.path.isdir(v):
                return v
    return spec["out_dir"]


def _remember_export_dir(mw, out_dir: str) -> None:
    settings = getattr(mw, "settings", None)
    if settings is None or not out_dir:
        return
    try:
        settings.setValue(_SETTINGS_KEY, out_dir)
    except Exception:
        pass


def _load_saved_spec(mw) -> dict:
    settings = getattr(mw, "settings", None)
    if settings is None:
        return {}
    try:
        raw = settings.value(_SPEC_SETTINGS_KEY, "", type=str) or ""
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def _remember_spec(mw, spec: dict) -> None:
    settings = getattr(mw, "settings", None)
    if settings is None:
        return
    keep = {k: spec.get(k) for k in (
        "dest_mode", "out_dir", "overwrite", "format",
        "auto_number", "tag", "include_paired_stars", "mode_all_open",
    )}
    try:
        settings.setValue(_SPEC_SETTINGS_KEY, json.dumps(keep))
    except Exception:
        pass


def _active_doc_from_main(mw):
    getter = getattr(mw, "_active_doc", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            pass
    dm = getattr(mw, "docman", None) or getattr(mw, "doc_manager", None)
    if dm is not None and hasattr(dm, "get_active_document"):
        try:
            return dm.get_active_document()
        except Exception:
            pass
    return None


def _docman(mw):
    return getattr(mw, "docman", None) or getattr(mw, "doc_manager", None)


def _open_documents(mw):
    fn = getattr(mw, "_collect_open_documents", None)
    if callable(fn):
        try:
            return list(fn() or [])
        except Exception:
            pass
    return []


# --------------------------------------------------------------------------- #
# Orchestration: build an export plan (with paired-stars) and run it
# --------------------------------------------------------------------------- #
def _build_plan(mw, docs, spec: dict):
    """
    Turn a list of primary docs into [(doc, spec_for_doc), ...].

    When include_paired_stars is on, each non-stars primary pulls in its open
    `<name>_stars` companion and both share the same _procN.
    """
    include = bool(spec.get("include_paired_stars"))
    name_map = {}
    if include:
        for d in _open_documents(mw):
            try:
                name_map[os.path.splitext(_best_name(d))[0]] = d
            except Exception:
                pass

    plan = []
    handled = set()
    for d in docs:
        if d is None or id(d) in handled:
            continue
        handled.add(id(d))

        force = _peek_number(d, spec) if spec.get("auto_number") else None
        s = dict(spec)
        if force is not None:
            s["_force_number"] = force
        plan.append((d, s))

        if include:
            title = os.path.splitext(_best_name(d))[0]
            _, is_stars = _split_stars_suffix(title)
            if not is_stars:
                comp = name_map.get(title + "_stars")
                if comp is not None and comp is not d and id(comp) not in handled:
                    handled.add(id(comp))
                    s2 = dict(spec)
                    if force is not None:
                        s2["_force_number"] = force
                    plan.append((comp, s2))
    return plan


def _run_plan(mw, plan, *, parent=None, spec_for_dir=None, title="Saving"):
    dm = _docman(mw)
    if dm is None:
        return 0, ["Document manager not available."]

    log = getattr(mw, "_log", None)
    applied = 0
    errors = []
    total = max(1, len(plan))

    pd = None
    if len(plan) > 1 and parent is not None:
        pd = QProgressDialog(f"{title}…", None, 0, total, parent)
        pd.setWindowTitle(title)
        pd.setWindowFlag(Qt.WindowType.Window, True)
        pd.setMinimumWidth(420)
        pd.setMinimumHeight(110)
        pd.setMinimumDuration(0)
        pd.setWindowModality(Qt.WindowModality.ApplicationModal)
        pd.setAutoClose(False)
        pd.setAutoReset(False)
        pd.setCancelButton(None)
        pd.resize(420, 120)
        pd.show()
        QApplication.processEvents()

    try:
        for i, (doc, s) in enumerate(plan, start=1):
            if pd is not None:
                pd.setValue(i - 1)
                pd.setLabelText(f"{title} ({i}/{total})…")
                QApplication.processEvents()
            result = export_document_as_fits(dm, doc, s)
            if result.ok:
                applied += 1
                if callable(log):
                    log(f"Saved: {result.path}")
            else:
                reason = result.reason or "Save failed."
                errors.append(reason)
                if callable(log):
                    log(f"Save skipped: {reason}")
        if pd is not None:
            pd.setValue(total)
    finally:
        if pd is not None:
            pd.close()
            pd.deleteLater()
            QApplication.processEvents()

    if spec_for_dir and spec_for_dir.get("dest_mode") == "folder" and spec_for_dir.get("out_dir"):
        _remember_export_dir(mw, spec_for_dir["out_dir"])
    return applied, errors


def run_checkpoint_now(mw):
    """One-click Checkpoint Save: numbered copy beside the source, active view
    (+ its paired _stars view), same format. No dialog."""
    doc = _active_doc_from_main(mw)
    if doc is None or getattr(doc, "image", None) is None:
        return
    spec = checkpoint_preset()
    plan = _build_plan(mw, [doc], spec)
    applied, errors = _run_plan(mw, plan, parent=mw, spec_for_dir=spec, title="Checkpoint Save")
    if applied == 0 and errors:
        QMessageBox.warning(mw, "Checkpoint Save", errors[0])


# --------------------------------------------------------------------------- #
# Preset editor (used by the shortcut / drag-handle system)
# --------------------------------------------------------------------------- #
class ExportFitsPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Save / Export — Preset")
        init = normalize_export_fits_preset(initial)

        self.rb_folder = QRadioButton("Into folder")
        self.rb_beside = QRadioButton("Beside source file")
        grp = QButtonGroup(self)
        grp.addButton(self.rb_folder)
        grp.addButton(self.rb_beside)
        (self.rb_beside if init["dest_mode"] == "beside_source" else self.rb_folder).setChecked(True)
        dest_row = QHBoxLayout()
        dest_row.addWidget(self.rb_folder)
        dest_row.addWidget(self.rb_beside)
        dest_row.addStretch(1)
        dest_w = QWidget()
        dest_w.setLayout(dest_row)

        self.edit_dir = QLineEdit(init["out_dir"])
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.edit_dir, 1)
        row.addWidget(btn_browse)
        self.folder_w = QWidget()
        self.folder_w.setLayout(row)

        self.cmb_format = QComboBox()
        for label, val in FORMAT_CHOICES:
            self.cmb_format.addItem(label, val)
        self._select_format(init["format"])

        self.chk_autonumber = QCheckBox("Auto-number (_proc1, _proc2, …)")
        self.chk_autonumber.setChecked(init["auto_number"])
        self.edit_tag = QLineEdit(init["tag"])
        self.edit_tag.setPlaceholderText("_tag (optional, e.g. _dbe)")
        self.chk_paired = QCheckBox("Also save paired _stars view")
        self.chk_paired.setChecked(init["include_paired_stars"])
        self.chk_overwrite = QCheckBox("Overwrite existing files")
        self.chk_overwrite.setChecked(init["overwrite"])

        form = QFormLayout(self)
        form.addRow("Destination:", dest_w)
        form.addRow("Output folder:", self.folder_w)
        form.addRow("Format:", self.cmb_format)
        form.addRow("", self.chk_autonumber)
        form.addRow("Tag:", self.edit_tag)
        form.addRow("", self.chk_paired)
        form.addRow("", self.chk_overwrite)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

        self.rb_folder.toggled.connect(self._sync_enabled)
        self.chk_autonumber.toggled.connect(self._sync_enabled)
        self._sync_enabled()

    def _select_format(self, val):
        idx = self.cmb_format.findData(val)
        self.cmb_format.setCurrentIndex(idx if idx >= 0 else 0)

    def _sync_enabled(self):
        folder = self.rb_folder.isChecked()
        self.folder_w.setEnabled(folder)
        # Overwrite only meaningful in folder mode without auto-numbering.
        self.chk_overwrite.setEnabled(folder and not self.chk_autonumber.isChecked())

    def _browse(self):
        start = self.edit_dir.text().strip() or os.path.expanduser("~")
        chosen = QFileDialog.getExistingDirectory(self, "Choose output folder", start)
        if chosen:
            self.edit_dir.setText(chosen)

    def result_dict(self) -> dict:
        return normalize_export_fits_preset({
            "dest_mode": "folder" if self.rb_folder.isChecked() else "beside_source",
            "out_dir": self.edit_dir.text().strip(),
            "format": self.cmb_format.currentData(),
            "auto_number": self.chk_autonumber.isChecked(),
            "tag": self.edit_tag.text().strip(),
            "include_paired_stars": self.chk_paired.isChecked(),
            "overwrite": self.chk_overwrite.isChecked(),
        })


# --------------------------------------------------------------------------- #
# Main dialog (rich shell from the user's process + checkpoint options)
# --------------------------------------------------------------------------- #
class ExportFitsDialog(QDialog):
    def __init__(self, main_window, preset: dict | None = None):
        super().__init__(main_window)
        self._mw = main_window
        self.setWindowTitle("Save Checkpoint / Export")
        self.setWindowFlag(Qt.WindowType.Window, True)
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.setMinimumSize(560, 360)

        # Seed: explicit preset first, else last-used spec, else old dir default.
        init = normalize_export_fits_preset(preset)
        if not preset:
            saved = _load_saved_spec(main_window)
            if saved:
                init = normalize_export_fits_preset(saved)
        init["out_dir"] = _last_export_dir(main_window, init)

        # Destination
        self.rb_folder = QRadioButton(self.tr("Into folder"))
        self.rb_beside = QRadioButton(self.tr("Beside source file"))
        grp = QButtonGroup(self)
        grp.addButton(self.rb_folder)
        grp.addButton(self.rb_beside)
        (self.rb_beside if init["dest_mode"] == "beside_source" else self.rb_folder).setChecked(True)
        dest_row = QHBoxLayout()
        dest_row.addWidget(self.rb_folder)
        dest_row.addWidget(self.rb_beside)
        dest_row.addStretch(1)
        dest_w = QWidget()
        dest_w.setLayout(dest_row)

        # Folder picker
        self.edit_dir = QLineEdit(init["out_dir"])
        btn_browse = QPushButton(self.tr("Browse…"))
        btn_browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.edit_dir, 1)
        row.addWidget(btn_browse)
        self.folder_w = QWidget()
        self.folder_w.setLayout(row)

        # Format + numbering + tag + companions + overwrite + scope
        self.cmb_format = QComboBox()
        for label, val in FORMAT_CHOICES:
            self.cmb_format.addItem(label, val)
        self._select_format(init["format"])

        self.chk_autonumber = QCheckBox(self.tr("Auto-number (_proc1, _proc2, …)"))
        self.chk_autonumber.setChecked(init["auto_number"])
        self.edit_tag = QLineEdit(init["tag"])
        self.edit_tag.setPlaceholderText(self.tr("_tag (optional, e.g. _dbe)"))
        self.chk_paired = QCheckBox(self.tr("Also save paired _stars view"))
        self.chk_paired.setChecked(init["include_paired_stars"])
        self.chk_overwrite = QCheckBox(self.tr("Overwrite existing files"))
        self.chk_overwrite.setChecked(init["overwrite"])
        self.chk_all_open = QCheckBox(self.tr("Apply to all open views"))
        self.chk_all_open.setChecked(init["mode_all_open"])

        hint = QLabel(self.tr(
            "Saves the current view to disk. 'Beside source file' writes a "
            "numbered checkpoint (…_proc1, …_proc2) next to the original; "
            "'Into folder' exports into a folder you choose. Use Apply to View "
            "Bundle, or drop a shortcut on a bundle, to save many views at once."
        ))
        hint.setWordWrap(True)

        v = QVBoxLayout(self)
        v.addWidget(hint)
        form = QFormLayout()
        form.addRow(self.tr("Destination:"), dest_w)
        form.addRow(self.tr("Output folder:"), self.folder_w)
        form.addRow(self.tr("Format:"), self.cmb_format)
        form.addRow("", self.chk_autonumber)
        form.addRow(self.tr("Tag:"), self.edit_tag)
        form.addRow("", self.chk_paired)
        form.addRow("", self.chk_overwrite)
        form.addRow("", self.chk_all_open)
        v.addLayout(form)

        btns = QHBoxLayout()
        btn_apply = QPushButton(self.tr("Apply"))
        btn_apply.clicked.connect(self._apply)
        self.btn_apply_vbundle = QPushButton(self.tr("Apply to View Bundle…"))
        self.btn_apply_vbundle.clicked.connect(self._apply_to_view_bundle)
        btn_close = QPushButton(self.tr("Close"))
        btn_close.clicked.connect(self.close)
        btns.addWidget(btn_apply)
        btns.addWidget(self.btn_apply_vbundle)
        btns.addStretch(1)
        btns.addWidget(btn_close)
        v.addLayout(btns)

        from setiastro.saspro.shortcuts import PresetDragHandle
        grip = QHBoxLayout()
        icon = QIcon(disk_path) if disk_path else QIcon()
        self.preset_drag_handle = PresetDragHandle(
            COMMAND_ID,
            self.current_preset,
            icon=icon,
            tooltip=(
                "Drag to the canvas to create a Save shortcut with these settings.\n"
                "Drop on an image, or add to a Function Bundle, to save headlessly."
            ),
            parent=self,
        )
        grip.addWidget(self.preset_drag_handle)
        grip.addStretch(1)
        v.addLayout(grip)
        self.resize(600, 400)

        self.rb_folder.toggled.connect(self._sync_enabled)
        self.chk_autonumber.toggled.connect(self._sync_enabled)
        self._sync_enabled()

    # -- widget helpers -----------------------------------------------------
    def _select_format(self, val):
        idx = self.cmb_format.findData(val)
        self.cmb_format.setCurrentIndex(idx if idx >= 0 else 0)

    def _sync_enabled(self):
        folder = self.rb_folder.isChecked()
        self.folder_w.setEnabled(folder)
        self.chk_overwrite.setEnabled(folder and not self.chk_autonumber.isChecked())

    def current_preset(self) -> dict:
        return normalize_export_fits_preset({
            "dest_mode": "folder" if self.rb_folder.isChecked() else "beside_source",
            "out_dir": self.edit_dir.text().strip(),
            "format": self.cmb_format.currentData(),
            "auto_number": self.chk_autonumber.isChecked(),
            "tag": self.edit_tag.text().strip(),
            "include_paired_stars": self.chk_paired.isChecked(),
            "overwrite": self.chk_overwrite.isChecked(),
            "mode_all_open": self.chk_all_open.isChecked(),
        })

    def seed_from_preset(self, preset: dict | None):
        spec = normalize_export_fits_preset(preset)
        (self.rb_beside if spec["dest_mode"] == "beside_source" else self.rb_folder).setChecked(True)
        if spec["out_dir"]:
            self.edit_dir.setText(spec["out_dir"])
        self._select_format(spec["format"])
        self.chk_autonumber.setChecked(spec["auto_number"])
        self.edit_tag.setText(spec["tag"])
        self.chk_paired.setChecked(spec["include_paired_stars"])
        self.chk_overwrite.setChecked(spec["overwrite"])
        self.chk_all_open.setChecked(spec["mode_all_open"])
        self._sync_enabled()

    def _browse(self):
        start = self.edit_dir.text().strip() or os.path.expanduser("~")
        chosen = QFileDialog.getExistingDirectory(self, "Choose output folder", start)
        if chosen:
            self.edit_dir.setText(chosen)

    def _require_destination(self) -> dict | None:
        spec = self.current_preset()
        if spec["dest_mode"] == "folder" and not spec["out_dir"]:
            self._browse()
            spec = self.current_preset()
        if spec["dest_mode"] == "folder" and not spec["out_dir"]:
            QMessageBox.warning(self, "Save Checkpoint / Export", "Choose an output folder first.")
            return None
        return spec

    # -- actions ------------------------------------------------------------
    def _target_docs(self, spec):
        if spec["mode_all_open"]:
            docs = _open_documents(self._mw)
        else:
            d = _active_doc_from_main(self._mw)
            docs = [d] if d is not None else []
        return [d for d in docs if d is not None and getattr(d, "image", None) is not None]

    def _apply(self):
        spec = self._require_destination()
        if spec is None:
            return
        if _docman(self._mw) is None:
            QMessageBox.warning(self, "Save Checkpoint / Export", "Document manager not available.")
            return
        docs = self._target_docs(spec)
        if not docs:
            QMessageBox.information(self, "Save Checkpoint / Export", "No image in the active view.")
            return
        plan = _build_plan(self._mw, docs, spec)
        applied, errors = _run_plan(self._mw, plan, parent=self, spec_for_dir=spec,
                                    title="Save Checkpoint / Export")
        _remember_spec(self._mw, spec)
        if applied == 0 and errors:
            QMessageBox.warning(self, "Save Checkpoint / Export", errors[0])
            return
        if errors:
            QMessageBox.warning(
                self, "Save Checkpoint / Export",
                f"Saved {applied} file(s).\n\nSkipped:\n" + "\n".join(errors[:12]),
            )
        if applied:
            self.close()

    def _load_view_bundle_choices(self):
        settings = QSettings()
        settings.sync()
        raw = ""
        for key in ("viewbundles/v3", "viewbundles/v2", "viewbundles/v1"):
            raw = settings.value(key, "", type=str) or ""
            if raw:
                break
        try:
            data = json.loads(raw or "[]")
        except Exception:
            data = []
        choices = []
        for bundle in data:
            if not isinstance(bundle, dict):
                continue
            name = str(bundle.get("name") or "Bundle").strip() or "Bundle"
            ptrs = []
            for x in (bundle.get("doc_ptrs") or []):
                try:
                    ptrs.append(int(x))
                except Exception:
                    pass
            files = [str(p) for p in (bundle.get("file_paths") or []) if p]
            choices.append((name, ptrs, files))
        return choices

    def _pick_view_bundle(self):
        choices = self._load_view_bundle_choices()
        if not choices:
            QMessageBox.information(self, "Save Checkpoint / Export", "No View Bundles found.")
            return None
        dlg = QDialog(self)
        dlg.setWindowTitle("Apply to View Bundle…")
        dlg.setMinimumSize(420, 280)
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("Select a View Bundle:"))
        lb = QListWidget(dlg)
        for name, ptrs, files in choices:
            it = QListWidgetItem(f"{name}  ({len(ptrs)} views, {len(files)} files)")
            it.setData(Qt.ItemDataRole.UserRole, (ptrs, files))
            lb.addItem(it)
        if lb.count():
            lb.setCurrentRow(0)
        v.addWidget(lb, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dlg,
        )
        v.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        cur = lb.currentItem()
        if not cur:
            return None
        return cur.data(Qt.ItemDataRole.UserRole)

    def _doc_from_file(self, path: str):
        from setiastro.saspro.doc_manager import ImageDocument
        from setiastro.saspro.legacy.image_manager import load_image
        img, header, bit_depth, is_mono = load_image(path)
        if img is None:
            raise RuntimeError(f"Could not load: {path}")
        ext = os.path.splitext(path)[1].lower().lstrip(".") or "fits"
        return ImageDocument(img, metadata={
            "file_path": path,
            "original_header": header,
            "bit_depth": bit_depth,
            "is_mono": is_mono,
            "original_format": ext,
        })

    def _apply_to_view_bundle(self):
        spec = self._require_destination()
        if spec is None:
            return
        picked = self._pick_view_bundle()
        if not picked:
            return
        ptrs, files = picked
        from setiastro.saspro.view_bundle import _find_main_window, _resolve_doc_and_subwindow
        mw = _find_main_window(self) or self._mw
        docs = []
        for ptr in ptrs or []:
            doc = None
            if mw is not None:
                doc, sw = _resolve_doc_and_subwindow(mw, ptr)
                if sw is not None and hasattr(mw, "mdi"):
                    try:
                        mw.mdi.setActiveSubWindow(sw)
                        QApplication.processEvents()
                    except Exception:
                        pass
            if doc is not None and getattr(doc, "image", None) is not None:
                docs.append(doc)
        for path in files or []:
            try:
                docs.append(self._doc_from_file(path))
            except Exception as e:
                QMessageBox.warning(self, "Save Checkpoint / Export", str(e))
                return
        if not docs:
            QMessageBox.information(self, "Save Checkpoint / Export", "No valid targets in the selected bundle.")
            return
        plan = _build_plan(mw, docs, spec)
        applied, errors = _run_plan(mw, plan, parent=self, spec_for_dir=spec,
                                    title="Save Checkpoint / Export")
        _remember_spec(self._mw, spec)
        if applied == 0 and errors:
            QMessageBox.warning(self, "Save Checkpoint / Export", errors[0])
        elif errors:
            QMessageBox.warning(
                self, "Save Checkpoint / Export",
                f"Saved {applied} file(s).\n\nSkipped:\n" + "\n".join(errors[:12]),
            )
        else:
            self.close()


def open_export_fits_with_preset(main_window, preset: dict | None = None):
    dlg = getattr(main_window, "_export_fits_dialog", None)
    if dlg is None:
        dlg = ExportFitsDialog(main_window, preset or {})
        try:
            main_window._export_fits_dialog = dlg
            dlg.destroyed.connect(
                lambda *_: setattr(main_window, "_export_fits_dialog", None)
                if getattr(main_window, "_export_fits_dialog", None) is dlg
                else None
            )
        except Exception:
            pass
    else:
        try:
            dlg.seed_from_preset(preset or {})
        except Exception:
            pass
    if preset:
        try:
            dlg.seed_from_preset(preset)
        except Exception:
            pass
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg