# ops/scripts.py
from __future__ import annotations

import os
import sys
import traceback
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Any
import numpy as np
from PyQt6.QtCore import QStandardPaths, QObject
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import QUrl

from ops.command_runner import run_command as _run_command

# -----------------------------------------------------------------------------
# Scripts folder  (FIXED ROOT: SASpro/scripts)
# -----------------------------------------------------------------------------
def get_scripts_dir() -> Path:
    """
    Per-user scripts folder, pinned to a stable 'SASpro/scripts' root.

      Windows: %LOCALAPPDATA%/SASpro/scripts
      macOS:   ~/Library/Application Support/SASpro/scripts
      Linux:   ~/.local/share/SASpro/scripts   (or $XDG_DATA_HOME)

    This intentionally does NOT use Qt's AppLocalDataLocation so it won't
    land under SetiAstro/Seti Astro Suite Pro.
    """
    # Windows
    if sys.platform.startswith("win"):
        base = os.getenv("LOCALAPPDATA")
        if base:
            root = Path(base)
        else:
            root = Path.home() / "AppData" / "Local"
        scripts = root / "SASpro" / "scripts"

    # macOS
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
        scripts = root / "SASpro" / "scripts"

    # Linux / other
    else:
        root = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        scripts = root / "SASpro" / "scripts"

    scripts.mkdir(parents=True, exist_ok=True)
    return scripts

def migrate_old_scripts_if_needed():
    """
    One-time best-effort migration from the old Qt-derived folder into
    the new SASpro/scripts folder.

    Safe: only copies *.py that don't already exist in new location.
    """
    try:
        new_dir = get_scripts_dir()

        old_dirs: list[Path] = []

        if sys.platform.startswith("win"):
            old_dirs.append(
                Path.home() / "AppData" / "Local" / "SetiAstro" / "Seti Astro Suite Pro" / "scripts"
            )
        elif sys.platform == "darwin":
            old_dirs.append(
                Path.home() / "Library" / "Application Support" / "SetiAstro" / "Seti Astro Suite Pro" / "scripts"
            )
        else:
            old_dirs.append(
                Path.home() / ".local" / "share" / "SetiAstro" / "Seti Astro Suite Pro" / "scripts"
            )

        for old in old_dirs:
            if not old.exists() or not old.is_dir():
                continue
            for p in old.glob("*.py"):
                dest = new_dir / p.name
                if not dest.exists():
                    try:
                        dest.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
                    except Exception:
                        # fallback binary copy if encoding chokes
                        import shutil
                        shutil.copy2(p, dest)

    except Exception:
        pass


# -----------------------------------------------------------------------------
# Script context exposed to user scripts
# -----------------------------------------------------------------------------
class ScriptContext:
    """
    Minimal, stable API for user scripts.
    Add helpers over time; try not to break existing ones.
    """
    def __init__(self, app_window, *, on_base: bool = False):
        self.app = app_window
        self._on_base = bool(on_base)

    def main_window(self):
        """Return the main SASpro window (stable helper for scripts)."""
        return self.app

    # ---- logging ----
    def log(self, msg: str):
        try:
            self.app._log(f"[Script] {msg}")
        except Exception:
            print("[Script]", msg)

    # ------------------------------------------------------------------
    # File-based image I/O (canonical SASpro routes)
    # ------------------------------------------------------------------
    def load_image(self, filename: str, *, return_metadata: bool = False,
                   max_retries: int = 3, wait_seconds: int = 3):
        """
        Load an image from disk using SASpro's canonical loader.

        This does NOT open or register a document or subwindow.
        It is purely file I/O.

        Returns:
            If return_metadata=False (default):
                (img, original_header, bit_depth, is_mono)
            If return_metadata=True:
                whatever legacy.image_manager.load_image returns in metadata mode
                (typically includes image_meta/file_meta)
        """
        from legacy import image_manager  # canonical route
        return image_manager.load_image(
            filename,
            max_retries=max_retries,
            wait_seconds=wait_seconds,
            return_metadata=bool(return_metadata),
        )

    def save_image(self, img_array, filename: str, *,
                   original_format: str | None = None,
                   bit_depth=None,
                   original_header=None,
                   is_mono: bool = False,
                   image_meta=None,
                   file_meta=None):
        """
        Save an image to disk using SASpro's canonical saver.

        This does NOT require an open document.
        It writes exactly through legacy.image_manager.save_image.

        Args:
            img_array: numpy array (mono or RGB). Any dtype accepted; saver handles it.
            filename: output path
            original_format: e.g. "fits", "tiff", "png". If None, inferred from suffix.
            bit_depth/original_header/is_mono/image_meta/file_meta:
                passed through to legacy saver.

        Returns:
            Whatever legacy.image_manager.save_image returns (often None or success flag).
        """
        from legacy import image_manager  # canonical route
        from pathlib import Path

        p = Path(filename)
        fmt = original_format
        if fmt is None or not str(fmt).strip():
            # infer from extension (".fits", ".fit", ".fz", ".tif", ".tiff", ".png", etc.)
            ext = p.suffix.lower().lstrip(".")
            if ext in ("fit", "fits", "fz", "fits.gz", "fit.gz"):
                fmt = "fits"
            elif ext in ("tif", "tiff"):
                fmt = "tiff"
            else:
                fmt = ext  # png/jpg/x

        return image_manager.save_image(
            img_array,
            str(p),
            fmt,
            bit_depth=bit_depth,
            original_header=original_header,
            is_mono=bool(is_mono),
            image_meta=image_meta,
            file_meta=file_meta,
        )

    # Friendly aliases (optional, but nice UX)
    open_image = load_image
    write_image = save_image

    # ---- active view/doc access ----
    def active_subwindow(self):
        try:
            return self.app.mdi.activeSubWindow()
        except Exception:
            return None

    def active_view(self):
        sw = self.active_subwindow()
        return sw.widget() if sw else None

    def base_document(self):
        sw = self.active_subwindow()
        if sw and hasattr(self.app, "_target_doc_from_subwindow"):
            try:
                return self.app._target_doc_from_subwindow(sw)
            except Exception:
                pass
        return self.active_document(fallback_to_base=False)

    def _docman(self):
        return getattr(self.app, "doc_manager", None)

    def active_document(self):
        """
        Normal run:
          - return DocManager.get_active_document() so Preview tabs yield _RoiViewDocument.
        Run-on-base:
          - force base doc even if Preview is active.
        """
        dm = self._docman()

        if dm and hasattr(dm, "get_active_document"):
            if self._on_base:
                # focused base is sticky and ignores ROI wrappers
                base = None
                try:
                    base = dm.get_focused_base_document()
                except Exception:
                    base = None
                return base or self.base_document()

            # normal run: ROI-aware
            try:
                return dm.get_active_document()
            except Exception:
                pass

        # fallback (should rarely happen)
        view = self.active_view()
        return getattr(view, "document", None) if view else None

    def get_image(self):
        doc = self.active_document()
        return getattr(doc, "image", None) if doc else None

    def set_image(self, img, step_name: str = "Script"):
        dm = self._docman()
        if dm is None:
            raise RuntimeError("DocManager not available.")

        img = np.asarray(img)
        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)

        if self._on_base:
            # ✅ Bypass ROI branch: write to base doc directly
            base_doc = None
            try:
                base_doc = dm.get_focused_base_document()
            except Exception:
                base_doc = None
            base_doc = base_doc or self.base_document()

            if base_doc is None:
                raise RuntimeError("No base document to update.")

            base_doc.apply_edit(img, metadata={}, step_name=step_name)

            # force full repaint, including any preview
            try:
                dm.imageRegionUpdated.emit(base_doc, None)
            except Exception:
                pass

            # if a preview is active, ask it to repaint too
            try:
                roi = dm._active_preview_roi()  # returns (x,y,w,h) or None
                if roi:
                    dm.previewRepaintRequested.emit(base_doc, roi)
            except Exception:
                pass
            return

        # ✅ Normal run: let DocManager decide (ROI preview vs full)
        dm.update_active_document(img, metadata={}, step_name=step_name)

    # ---- convenience wrappers into main window ----
    def run_command(self, command_id: str, preset=None, **kwargs):
        return _run_command(self, command_id, preset, **kwargs)

    def is_frozen(self) -> bool:
        return bool(getattr(sys, "frozen", False))

    # ------------------------------------------------------------------
    # View / document lookup helpers for scripts
    # ------------------------------------------------------------------
    def _iter_open_subwindows(self):
        """Yield (subwindow, widget) for all open MDI subwindows."""
        try:
            mdi = getattr(self.app, "mdi", None)
            if mdi is None:
                return
            for sw in mdi.subWindowList():
                try:
                    w = sw.widget()
                except Exception:
                    w = None
                if w is not None:
                    yield sw, w
        except Exception:
            return

    def _base_doc_for_widget(self, w):
        """
        Best-effort unwrap:
        - ImageSubWindow.base_document / _base_document / document
        - LiveViewDocument -> underlying base (_base)
        - ROI wrapper -> parent
        """
        doc = (
            getattr(w, "base_document", None)
            or getattr(w, "_base_document", None)
            or getattr(w, "document", None)
        )
        if doc is None:
            return None

        # LiveViewDocument exposes _base
        base = getattr(doc, "_base", None)
        if base is not None:
            doc = base

        # ROI wrapper -> parent base
        parent = getattr(doc, "_parent_doc", None)
        if parent is not None:
            doc = parent

        return doc

    def list_views(self):
        """
        Return list of open views with stable info.
        Each item:
          {
            "title": <window title>,
            "name": <doc display name>,
            "uid": <doc uid or None>,
            "file_path": <metadata file_path or ''>,
            "is_active": bool
          }
        """
        out = []
        active_sw = None
        try:
            active_sw = self.active_subwindow()
        except Exception:
            active_sw = None

        for sw, w in self._iter_open_subwindows():
            base_doc = self._base_doc_for_widget(w)
            if base_doc is None:
                continue

            # titles / names
            try:
                title = str(sw.windowTitle() or "")
            except Exception:
                title = ""
            try:
                name = str(base_doc.display_name())
            except Exception:
                name = str(getattr(base_doc, "metadata", {}).get("display_name", title) or title)

            uid = getattr(base_doc, "uid", None)
            file_path = ""
            try:
                file_path = str(getattr(base_doc, "metadata", {}).get("file_path", "") or "")
            except Exception:
                pass

            out.append({
                "title": title,
                "name": name,
                "uid": uid,
                "file_path": file_path,
                "is_active": (sw is active_sw),
            })
        return out

    def list_view_names(self):
        """Convenience: return a list of human-visible names for open views."""
        return [v["name"] or v["title"] for v in self.list_views()]

    def get_document(self, view_name_or_uid: str, *, prefer_title: bool = False):
        """
        Look up an open document by:
          - display name (doc.display_name())
          - or window title (subwindow.windowTitle())
          - or uid (exact)
        Matching is case-insensitive for names/titles.
        Returns base ImageDocument (never a ROI wrapper).
        """
        if not view_name_or_uid:
            return None

        key = str(view_name_or_uid).strip()
        key_low = key.lower()

        for sw, w in self._iter_open_subwindows():
            base_doc = self._base_doc_for_widget(w)
            if base_doc is None:
                continue

            uid = getattr(base_doc, "uid", None)
            if uid is not None and str(uid) == key:
                return base_doc

            # compare names/titles
            try:
                doc_name = str(base_doc.display_name() or "").strip()
            except Exception:
                doc_name = str(getattr(base_doc, "metadata", {}).get("display_name", "") or "").strip()

            try:
                title = str(sw.windowTitle() or "").strip()
            except Exception:
                title = ""

            if prefer_title:
                if title and title.lower() == key_low:
                    return base_doc
                if doc_name and doc_name.lower() == key_low:
                    return base_doc
            else:
                if doc_name and doc_name.lower() == key_low:
                    return base_doc
                if title and title.lower() == key_low:
                    return base_doc

        return None

    def get_image_for(self, view_name_or_uid: str):
        """Get image ndarray for a named/uid view (base doc)."""
        doc = self.get_document(view_name_or_uid)
        return getattr(doc, "image", None) if doc else None

    def set_image_for(self, view_name_or_uid: str, img, step_name: str = "Script"):
        """
        Set image on a named/uid view (base doc), with undo + repaint.
        This updates the full doc, not an ROI preview.
        """
        dm = self._docman()
        if dm is None:
            raise RuntimeError("DocManager not available.")

        doc = self.get_document(view_name_or_uid)
        if doc is None:
            raise RuntimeError(f"No open view matches '{view_name_or_uid}'")

        arr = np.asarray(img)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)

        # Apply edit to that doc directly (full-image semantics)
        doc.apply_edit(arr, metadata={}, step_name=step_name)

        # Clear/invalidate any ROI caches for this base doc so previews don't stale
        try:
            dm._invalidate_roi_cache(doc, None)
        except Exception:
            pass

        # Repaint any views showing this doc
        try:
            dm.imageRegionUpdated.emit(doc, None)
        except Exception:
            pass

    def activate_view(self, view_name_or_uid: str) -> bool:
        """
        Bring a view to front by name/title/uid.
        Returns True if activated.
        """
        key = str(view_name_or_uid).strip().lower()
        mdi = getattr(self.app, "mdi", None)
        if mdi is None:
            return False

        for sw, w in self._iter_open_subwindows():
            base_doc = self._base_doc_for_widget(w)
            if base_doc is None:
                continue

            uid = getattr(base_doc, "uid", None)
            try:
                doc_name = str(base_doc.display_name() or "").strip().lower()
            except Exception:
                doc_name = ""
            try:
                title = str(sw.windowTitle() or "").strip().lower()
            except Exception:
                title = ""

            if (uid is not None and str(uid) == view_name_or_uid) or doc_name == key or title == key:
                try:
                    mdi.setActiveSubWindow(sw)
                except Exception:
                    pass
                try:
                    sw.show()
                    sw.raise_()
                except Exception:
                    pass
                return True
        return False

    # ---- view enumeration / lookup by user-visible view name ----
    def list_image_views(self):
        """
        Return a list of (view_title, doc) for all open image subwindows.
        The title is the current MDI window title (what the user renamed it to).
        """
        out = []
        mdi = getattr(self.app, "mdi", None)
        if mdi is None:
            return out

        try:
            subwins = mdi.subWindowList()
        except Exception:
            subwins = []

        for sw in subwins:
            try:
                w = sw.widget()
            except Exception:
                continue

            doc = (
                getattr(w, "document", None)
                or getattr(w, "base_document", None)
                or getattr(w, "_base_document", None)
            )
            if doc is None or getattr(doc, "image", None) is None:
                continue

            try:
                title = sw.windowTitle() or ""
            except Exception:
                title = ""

            if not title:
                # fallback to doc display name if window title missing
                try:
                    title = doc.display_name()
                except Exception:
                    title = "Untitled"

            out.append((title, doc))

        return out

    def get_document_by_view_name(self, name: str):
        """
        Find the first open image doc whose *view title* matches name.
        Matching is case-insensitive; exact match preferred, else unique prefix.
        """
        name_l = (name or "").strip().lower()
        if not name_l:
            return None

        views = self.list_image_views()

        # exact match
        for title, doc in views:
            if title.strip().lower() == name_l:
                return doc

        # unique prefix match
        pref = [(t, d) for (t, d) in views if t.strip().lower().startswith(name_l)]
        if len(pref) == 1:
            return pref[0][1]

        return None

    def get_image_by_view_name(self, name: str):
        doc = self.get_document_by_view_name(name)
        return getattr(doc, "image", None) if doc else None

    def open_new_document(self, img, metadata=None, name: str | None = None):
        """
        Convenience for scripts: create/register a new ImageDocument from an array.
        """
        dm = self._docman()
        if dm is None:
            raise RuntimeError("DocManager not available.")
        return dm.open_array(np.asarray(img, dtype=np.float32), metadata=metadata, title=name)


# -----------------------------------------------------------------------------
# Script registry entries
# -----------------------------------------------------------------------------
@dataclass
class ScriptEntry:
    path: Path
    name: str
    group: str = ""
    shortcut: Optional[str] = None
    module: Any = None
    run: Optional[Callable[[ScriptContext], None]] = None


# -----------------------------------------------------------------------------
# Script manager
# -----------------------------------------------------------------------------
class ScriptManager(QObject):
    """
    Owns script discovery/loading and menu binding.
    Main window delegates to this.
    """
    def __init__(self, app_window):
        super().__init__(app_window)
        self.app = app_window
        self.registry: list[ScriptEntry] = []

    # ---- internal log ----
    def _log(self, msg: str):
        try:
            self.app._log(msg)
        except Exception:
            print(msg)

    # ---- loading ----
    def load_registry(self):
        migrate_old_scripts_if_needed() 
        scripts_dir = get_scripts_dir()
        self.registry = []

        for path in sorted(scripts_dir.glob("*.py")):
            try:
                entry = self._load_one_script(path)
                if entry:
                    self.registry.append(entry)
            except Exception:
                self._log(f"[Scripts] Failed to load {path.name}:\n{traceback.format_exc()}")

        self._log(f"[Scripts] Loaded {len(self.registry)} script(s) from {scripts_dir}")

    def _load_one_script(self, path: Path) -> ScriptEntry | None:
        # Make a unique module name so reload actually reloads
        try:
            mtime_ns = path.stat().st_mtime_ns
        except Exception:
            mtime_ns = 0
        module_name = f"saspro_user_script_{path.stem}_{mtime_ns}"

        spec = importlib.util.spec_from_file_location(module_name, path)
        if not spec or not spec.loader:
            return None

        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)  # type: ignore
        except Exception:
            self._log(f"[Scripts] Error importing {path.name}:\n{traceback.format_exc()}")
            return None

        # ---- entrypoint: allow run(ctx) OR main(ctx) ----
        run_func = getattr(mod, "run", None)
        if not callable(run_func):
            run_func = getattr(mod, "main", None)

        if not callable(run_func):
            self._log(f"[Scripts] {path.name} has no run(ctx) or main(ctx); skipping.")
            return None

        # ---- metadata: allow CAPS or lowercase ----
        def _pick(*names, default=None):
            for n in names:
                if hasattr(mod, n):
                    return getattr(mod, n)
            return default

        name = _pick("SCRIPT_NAME", "script_name", default=path.stem)
        group = _pick("SCRIPT_GROUP", "script_group", default="")
        shortcut = _pick("SCRIPT_SHORTCUT", "script_shortcut", default=None)

        entry = ScriptEntry(
            path=path,
            name=str(name),
            group=str(group or ""),
            shortcut=str(shortcut) if shortcut else None,
            module=mod,
            run=run_func,
        )
        return entry


    # ---- menu wiring ----
    def rebuild_menu(self, menu_scripts):
        """
        Clears and rebuilds the Scripts menu from registry.
        Expects base actions already created on app window:
        act_script_editor, act_open_scripts_folder, act_reload_scripts, act_create_sample_script
        """
        menu_scripts.clear()

        # --- fixed top actions ---
        if hasattr(self.app, "act_script_editor"):
            menu_scripts.addAction(self.app.act_script_editor)
            menu_scripts.addSeparator()

        menu_scripts.addAction(self.app.act_open_scripts_folder)
        menu_scripts.addAction(self.app.act_reload_scripts)
        menu_scripts.addAction(self.app.act_create_sample_script)
        menu_scripts.addSeparator()

        # group -> submenu
        group_menus: dict[str, Any] = {}

        for entry in self.registry:
            group = entry.group.strip()
            if group:
                sub = group_menus.get(group)
                if sub is None:
                    sub = menu_scripts.addMenu(group)
                    group_menus[group] = sub
                target_menu = sub
            else:
                target_menu = menu_scripts

            act = QAction(entry.name, self.app)
            if entry.shortcut:
                try:
                    act.setShortcut(entry.shortcut)
                except Exception:
                    pass

            act.triggered.connect(lambda _=False, e=entry: self.run_entry(e))
            target_menu.addAction(act)


    # ---- running ----
    def run_entry(self, entry: ScriptEntry, *, on_base: bool = False):
        ctx = ScriptContext(self.app, on_base=on_base)
        try:
            self._log(f"[Scripts] Running '{entry.name}' ({entry.path.name}) on_base={on_base}")
            entry.run(ctx)  # type: ignore
            self._log(f"[Scripts] Finished '{entry.name}'")
        except Exception as e:
            tb = traceback.format_exc()
            self._log(f"[Scripts] ERROR in '{entry.name}':\n{tb}")
            try:
                QMessageBox.critical(self.app, "Script Error",
                                     f"{entry.name} failed:\n\n{e}")
            except Exception:
                pass


    # ---- convenience actions ----
    def open_scripts_folder(self):
        folder = get_scripts_dir()
        try:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        except Exception:
            # OS fallback
            try:
                if sys.platform.startswith("win"):
                    os.startfile(folder)  # type: ignore
                elif sys.platform == "darwin":
                    os.system(f'open "{folder}"')
                else:
                    os.system(f'xdg-open "{folder}"')
            except Exception:
                self._log(f"[Scripts] Couldn't open scripts folder: {folder}")

    def create_sample_script(self):
        folder = get_scripts_dir()

        samples: dict[str, str] = {}

        # ------------------------------------------------------------------
        # 1) sample_invert.py  (existing)
        # ------------------------------------------------------------------
        samples["sample_invert.py"] = """\
# Sample SASpro script
# Put scripts in this folder; they appear in Scripts menu.
# Required entrypoint:
#   def run(ctx):
#       ...

SCRIPT_NAME = "Invert Image (Sample)"
SCRIPT_GROUP = "Samples"

import numpy as np

def run(ctx):
    img = ctx.get_image()
    if img is None:
        ctx.log("No active image.")
        return

    ctx.log(f"Inverting image... shape={img.shape}, dtype={img.dtype}")

    f = img.astype(np.float32)
    mx = float(np.nanmax(f)) if f.size else 1.0
    if mx > 1.0:
        f = f / mx
    f = np.clip(f, 0.0, 1.0)

    out = 1.0 - f
    ctx.set_image(out, step_name="Invert via Script")
    ctx.log("Done.")
"""

        # ------------------------------------------------------------------
        # 2) sample_star_preview_ui.py  (SEP demo)
        # ------------------------------------------------------------------
        samples["sample_star_preview_ui.py"] = """\
from __future__ import annotations

# =========================
# SASpro Script Metadata
# =========================
SCRIPT_NAME     = "Star Preview UI (SEP Demo)"
SCRIPT_GROUP    = "Samples"
SCRIPT_SHORTCUT = ""   # optional

# -------------------------
# Star Preview UI sample
# -------------------------

import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QCheckBox, QMessageBox, QApplication, QWidget
)

# your libs already bundled in SASpro
from imageops.stretch import stretch_color_image, stretch_mono_image
from imageops.starbasedwhitebalance import apply_star_based_white_balance

# (optional) for applying result back to active doc
from pro.whitebalance import apply_white_balance_to_doc


def _to_float01(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img).astype(np.float32, copy=False)
    if a.size == 0:
        return a
    m = float(np.nanmax(a))
    if np.isfinite(m) and m > 1.0:
        a = a / m
    return np.clip(a, 0.0, 1.0)


class StarPreviewDialog(QDialog):
    \"""
    Sample script UI:
    - Shows active image (auto-updates when subwindow changes)
    - Runs SEP detection + ellipse overlay
    - Zoom controls + Fit/1:1
    - Demo Apply WB to active image
    \"""
    def __init__(self, ctx, parent: QWidget | None = None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("Sample Script: Star Preview UI")
        self.resize(980, 640)

        self._zoom = 1.0
        self._img01: np.ndarray | None = None
        self._overlay01: np.ndarray | None = None

        self._build_ui()
        self._wire()

        # debounce for slider/checkbox
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(500)
        self._debounce.timeout.connect(self._rebuild_overlay)

        # watch active base doc so preview isn't blank
        try:
            dm = getattr(self.ctx.app, "doc_manager", None)
            if dm is not None and hasattr(dm, "activeBaseChanged"):
                dm.activeBaseChanged.connect(lambda _=None: self._load_active_image())
        except Exception:
            pass

        # initial load
        QTimer.singleShot(0, self._load_active_image)

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QVBoxLayout(self)

        self.preview = QLabel("No active image.")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("border: 1px solid #333; background:#1f1f1f;")
        self.preview.setMinimumSize(720, 420)
        root.addWidget(self.preview, stretch=1)

        # Zoom bar
        zrow = QHBoxLayout()
        self.btn_zoom_in  = QPushButton("Zoom +")
        self.btn_zoom_out = QPushButton("Zoom −")
        self.btn_fit      = QPushButton("Fit")
        self.btn_1to1     = QPushButton("1:1")
        zrow.addWidget(self.btn_zoom_in)
        zrow.addWidget(self.btn_zoom_out)
        zrow.addWidget(self.btn_fit)
        zrow.addWidget(self.btn_1to1)
        zrow.addStretch(1)
        root.addLayout(zrow)

        # SEP controls
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("SEP threshold (σ):"))
        self.thr_slider = QSlider(Qt.Orientation.Horizontal)
        self.thr_slider.setRange(1, 100)
        self.thr_slider.setValue(50)
        self.thr_slider.setTickInterval(10)
        self.thr_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        ctrl.addWidget(self.thr_slider, stretch=1)

        self.thr_label = QLabel("50")
        self.thr_label.setFixedWidth(30)
        ctrl.addWidget(self.thr_label)

        self.chk_autostretch = QCheckBox("Autostretch preview")
        self.chk_autostretch.setChecked(True)
        ctrl.addWidget(self.chk_autostretch)

        root.addLayout(ctrl)

        # bottom buttons
        brow = QHBoxLayout()
        brow.addStretch(1)
        self.btn_apply_demo = QPushButton("Apply WB to Active Image (demo)")
        self.btn_close = QPushButton("Close")
        brow.addWidget(self.btn_apply_demo)
        brow.addWidget(self.btn_close)
        root.addLayout(brow)

    def _wire(self):
        self.btn_close.clicked.connect(self.reject)

        self.btn_zoom_in.clicked.connect(lambda: self._set_zoom(self._zoom * 1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._set_zoom(self._zoom / 1.25))
        self.btn_fit.clicked.connect(self._zoom_fit)
        self.btn_1to1.clicked.connect(lambda: self._set_zoom(1.0))

        self.thr_slider.valueChanged.connect(self._on_thr_changed)
        self.chk_autostretch.toggled.connect(lambda _=None: self._debounce.start())

        self.btn_apply_demo.clicked.connect(self._apply_demo_wb)

    # ------------- Active image -------------
    def _load_active_image(self):
        try:
            doc = self.ctx.active_document()
        except Exception:
            doc = None

        if doc is None or getattr(doc, "image", None) is None:
            self._img01 = None
            self._overlay01 = None
            self.preview.setText("No active image.")
            self.preview.setPixmap(QPixmap())
            return

        img = _to_float01(np.asarray(doc.image))
        self._img01 = img
        self._zoom_fit()
        self._rebuild_overlay()

    # ------------- SEP overlay -------------
    def _on_thr_changed(self, v: int):
        self.thr_label.setText(str(v))
        self._debounce.start()

    def _rebuild_overlay(self):
        if self._img01 is None:
            return
        try:
            thr = float(self.thr_slider.value())
            auto = bool(self.chk_autostretch.isChecked())

            img = self._img01
            # if mono, make a fake RGB for visualization / SEP expects gray anyway
            if img.ndim == 2:
                rgb = np.repeat(img[..., None], 3, axis=2)
            elif img.ndim == 3 and img.shape[2] == 1:
                rgb = np.repeat(img, 3, axis=2)
            else:
                rgb = img

            # Use your WB star detector just for overlay
            # (balanced output ignored; we only want overlay + count)
            _balanced, count, overlay = apply_star_based_white_balance(
                rgb, threshold=thr, autostretch=auto,
                reuse_cached_sources=False, return_star_colors=False
            )

            self._overlay01 = overlay
            self._render_pixmap()
            self.setWindowTitle(f"Sample Script: Star Preview UI  —  {count} stars")

        except Exception as e:
            self._overlay01 = None
            self.preview.setText(f"Star detection failed:\\n{e}")

    # ------------- Rendering / zoom -------------
    def _render_pixmap(self):
        if self._overlay01 is None:
            return
        ov = np.clip(self._overlay01, 0, 1)
        h, w, c = ov.shape
        qimg = QImage((ov * 255).astype(np.uint8).data, w, h, 3*w, QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(qimg)

        # apply zoom
        zw = int(pm.width() * self._zoom)
        zh = int(pm.height() * self._zoom)
        pmz = pm.scaled(zw, zh, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.preview.setPixmap(pmz)

    def _set_zoom(self, z: float):
        self._zoom = float(np.clip(z, 0.05, 20.0))
        self._render_pixmap()

    def _zoom_fit(self):
        if self._overlay01 is None and self._img01 is None:
            return
        # fit based on raw image size
        base = self._overlay01 if self._overlay01 is not None else self._img01
        h, w = base.shape[:2]
        vw = max(1, self.preview.width())
        vh = max(1, self.preview.height())
        self._zoom = min(vw / w, vh / h)
        self._render_pixmap()

    # ------------- Demo apply -------------
    def _apply_demo_wb(self):
        try:
            doc = self.ctx.active_document()
            if doc is None:
                raise RuntimeError("No active document.")
            # Reuse your headless preset WB as an example of applying edits
            preset = {"mode": "star", "threshold": float(self.thr_slider.value())}
            apply_white_balance_to_doc(doc, preset)
            QMessageBox.information(self, "Demo", "White Balance applied to active image.")
            # refresh preview after edit
            self._load_active_image()
        except Exception as e:
            QMessageBox.critical(self, "Demo", f"Failed to apply WB:\\n{e}")


def run(ctx):
    \"""
    SASpro entry point.
    \"""
    w = StarPreviewDialog(ctx, parent=ctx.app)
    w.exec()
"""

        # ------------------------------------------------------------------
        # 3) sample_average_two_docs_ui.py  (NEW)
        # ------------------------------------------------------------------
        samples["sample_average_two_docs_ui.py"] = """\
# Sample SASpro script
# UI with two dropdowns listing open views by their CURRENT window titles.
# Averages the two selected documents and opens a new document.

from __future__ import annotations

SCRIPT_NAME  = "Average Two Documents (UI Sample)"
SCRIPT_GROUP = "Samples"

import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QMessageBox
)


class AverageTwoDocsDialog(QDialog):
    def __init__(self, ctx):
        super().__init__(parent=ctx.app)
        self.ctx = ctx
        self.setWindowTitle("Average Two Documents")
        self.resize(520, 180)

        self._title_to_doc = {}

        root = QVBoxLayout(self)

        # Row A
        row_a = QHBoxLayout()
        row_a.addWidget(QLabel("Document A:"))
        self.combo_a = QComboBox()
        row_a.addWidget(self.combo_a, 1)
        root.addLayout(row_a)

        # Row B
        row_b = QHBoxLayout()
        row_b.addWidget(QLabel("Document B:"))
        self.combo_b = QComboBox()
        row_b.addWidget(self.combo_b, 1)
        root.addLayout(row_b)

        # Buttons
        brow = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh List")
        self.btn_avg = QPushButton("Average → New Doc")
        self.btn_close = QPushButton("Close")
        brow.addStretch(1)
        brow.addWidget(self.btn_refresh)
        brow.addWidget(self.btn_avg)
        brow.addWidget(self.btn_close)
        root.addLayout(brow)

        self.btn_refresh.clicked.connect(self._populate)
        self.btn_avg.clicked.connect(self._do_average)
        self.btn_close.clicked.connect(self.reject)

        self._populate()

    def _populate(self):
        self.combo_a.clear()
        self.combo_b.clear()
        self._title_to_doc.clear()

        try:
            views = self.ctx.list_image_views()
        except Exception:
            views = []

        for title, doc in views:
            # if duplicate names exist, disambiguate slightly
            key = title
            if key in self._title_to_doc:
                # add uid or a counter suffix
                try:
                    uid = getattr(doc, "uid", "")[:6]
                    key = f"{title} [{uid}]"
                except Exception:
                    n = 2
                    while f"{title} ({n})" in self._title_to_doc:
                        n += 1
                    key = f"{title} ({n})"

            self._title_to_doc[key] = doc
            self.combo_a.addItem(key)
            self.combo_b.addItem(key)

        if self.combo_a.count() == 0:
            self.combo_a.addItem("<no image views>")
            self.combo_b.addItem("<no image views>")
            self.btn_avg.setEnabled(False)
        else:
            self.btn_avg.setEnabled(True)

    def _do_average(self):
        key_a = self.combo_a.currentText()
        key_b = self.combo_b.currentText()

        doc_a = self._title_to_doc.get(key_a)
        doc_b = self._title_to_doc.get(key_b)

        if doc_a is None or doc_b is None:
            QMessageBox.warning(self, "Average", "Please select two valid documents.")
            return

        img_a = getattr(doc_a, "image", None)
        img_b = getattr(doc_b, "image", None)

        if img_a is None or img_b is None:
            QMessageBox.warning(self, "Average", "One of the selected documents has no image.")
            return

        a = np.asarray(img_a, dtype=np.float32)
        b = np.asarray(img_b, dtype=np.float32)

        # reconcile mono/color
        if a.ndim == 2:
            a = a[..., None]
        if b.ndim == 2:
            b = b[..., None]
        if a.shape[2] == 1 and b.shape[2] == 3:
            a = np.repeat(a, 3, axis=2)
        if b.shape[2] == 1 and a.shape[2] == 3:
            b = np.repeat(b, 3, axis=2)

        if a.shape != b.shape:
            QMessageBox.warning(
                self, "Average",
                f"Shape mismatch:\\nA: {a.shape}\\nB: {b.shape}\\n\\n"
                "For this sample, images must match exactly."
            )
            return

        out = 0.5 * (a + b)

        # name the new doc based on view titles
        new_name = f"Average({key_a}, {key_b})"

        try:
            self.ctx.open_new_document(out, metadata={}, name=new_name)
            QMessageBox.information(self, "Average", f"Created new document:\\n{new_name}")
        except Exception as e:
            QMessageBox.critical(self, "Average", f"Failed to create new doc:\\n{e}")


def run(ctx):
    dlg = AverageTwoDocsDialog(ctx)
    dlg.exec()
"""

        created = []
        skipped = []

        for fname, text in samples.items():
            path = folder / fname
            if path.exists():
                skipped.append(fname)
                continue
            try:
                path.write_text(text, encoding="utf-8")
                created.append(fname)
                self._log(f"[Scripts] Wrote sample script: {path}")
            except Exception:
                self._log(f"[Scripts] Failed to write {fname}:\n{traceback.format_exc()}")

        # user message
        try:
            if created and not skipped:
                QMessageBox.information(
                    self.app, "Sample Scripts Created",
                    "Created sample scripts:\n\n" + "\n".join(created) +
                    "\n\nReload Scripts to see them."
                )
            elif created and skipped:
                QMessageBox.information(
                    self.app, "Sample Scripts Created",
                    "Created:\n" + "\n".join(created) +
                    "\n\nAlready existed:\n" + "\n".join(skipped) +
                    "\n\nReload Scripts to see new ones."
                )
            else:
                QMessageBox.information(
                    self.app, "Sample Scripts",
                    "All sample scripts already exist:\n\n" + "\n".join(skipped)
                )
        except Exception:
            pass

            self._log(f"[Scripts] Failed to write sample script:\n{traceback.format_exc()}")
