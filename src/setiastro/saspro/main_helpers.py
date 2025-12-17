# pro/main_helpers.py
"""
Helper functions extracted from the main module.

Contains utility functions used throughout the main window:
- File path utilities
- Document name/type detection
- Widget safety checks
- WCS/FITS header utilities
"""

import os
from typing import Optional, Tuple

from PyQt6 import sip

from setiastro.saspro.file_utils import (
    _normalize_ext,
    _sanitize_filename,
    _exts_from_filter,
    REPLACE_SPACES_WITH_UNDERSCORES,
    WIN_RESERVED_NAMES,
)


def safe_join_dir_and_name(directory: str, basename: str) -> str:
    """
    Join directory + sanitized basename. 
    Ensures the directory exists or raises a clear error.
    """
    safe_name = _sanitize_filename(basename)
    final_dir = directory or ""
    if final_dir and not os.path.isdir(final_dir):
        try:
            os.makedirs(final_dir, exist_ok=True)
        except Exception:
            pass
    return os.path.join(final_dir, safe_name)


def normalize_save_path_chosen_filter(path: str, selected_filter: str) -> Tuple[str, str]:
    """
    Returns (final_path, final_ext_norm). Ensures:
      - appends extension if missing (from chosen filter)
      - avoids double extensions (*.png.png)
      - if user provided a conflicting ext, enforce the chosen filter's default
      - sanitizes the basename (spaces, illegal chars, trailing dots)
    """
    raw_path = (path or "").strip().rstrip(".")
    allowed = _exts_from_filter(selected_filter) or ["png"]  # safe fallback
    default_ext = allowed[0]

    # Split dir + basename (sanitize only the basename)
    directory, base = os.path.split(raw_path)
    if not base:
        base = "untitled"

    # If the user typed something like "name.png" but selected TIFF, fix after sanitization
    base_stem, base_ext = os.path.splitext(base)
    typed = _normalize_ext(base_ext) if base_ext else ""

    def strip_trailing_allowed(stem: str) -> str:
        """Remove repeated extension in stem (e.g. 'foo.png' then + '.png')."""
        lowered = stem.lower()
        for a in allowed:
            suf = "." + a
            if lowered.endswith(suf):
                return stem[:-len(suf)]
        return stem

    base_stem = strip_trailing_allowed(base_stem)

    # Choose final extension
    if not typed:
        final_ext = default_ext
    else:
        final_ext = typed if typed in allowed else default_ext

    # Rebuild name with the chosen extension, then sanitize the WHOLE basename
    basename_target = f"{base_stem}.{final_ext}"
    basename_safe = _sanitize_filename(basename_target, replace_spaces=REPLACE_SPACES_WITH_UNDERSCORES)

    # Final join (create dir if missing)
    final_path = safe_join_dir_and_name(directory, basename_safe)
    return final_path, final_ext


def display_name(doc) -> str:
    """Best-effort title for any doc-like object."""
    # Prefer a method
    for attr in ("display_name", "title", "name"):
        v = getattr(doc, attr, None)
        if callable(v):
            try:
                s = v()
                if isinstance(s, str) and s.strip():
                    return s
            except Exception:
                pass
        elif isinstance(v, str) and v.strip():
            return v

    # Metadata fallbacks
    md = getattr(doc, "metadata", {}) or {}
    if isinstance(md, dict):
        for k in ("display_name", "title", "name", "filename", "basename"):
            s = md.get(k)
            if isinstance(s, str) and s.strip():
                return s

    # Last resort: id snippet
    return f"Document-{id(doc) & 0xFFFF:04X}"


def best_doc_name(doc) -> str:
    """Get the best available name for a document."""
    # Try common attributes in order
    for attr in ("display_name", "name", "title"):
        v = getattr(doc, attr, None)
        if callable(v):
            try:
                v = v()
            except Exception:
                v = None
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Fallback: derive from original path if we have it
    try:
        meta = getattr(doc, "metadata", {}) or {}
        fp = meta.get("file_path")
        if isinstance(fp, str) and fp:
            return os.path.splitext(os.path.basename(fp))[0]
    except Exception:
        pass

    return "untitled"


def doc_looks_like_table(doc) -> bool:
    """Determine if a document represents tabular data rather than an image."""
    md = getattr(doc, "metadata", {}) or {}

    # Explicit type hints from own pipeline
    if str(md.get("doc_type", "")).lower() in {"table", "catalog", "fits_table"}:
        return True
    if str(md.get("fits_hdu_type", "")).lower().endswith("tablehdu"):
        return True
    if str(md.get("hdu_class", "")).lower().endswith("tablehdu"):
        return True

    # FITS header inspection (common with astropy)
    hdr = md.get("original_header") or md.get("fits_header") or {}
    try:
        xt = str(hdr.get("XTENSION", "")).upper()
        if xt in {"TABLE", "BINTABLE", "ASCIITABLE"}:
            return True
    except Exception:
        pass

    # Structural hints from the doc
    if hasattr(doc, "table"):
        return True
    if hasattr(doc, "columns"):
        return True
    if hasattr(doc, "rows") or hasattr(doc, "headers"):
        return True

    # Last resort: no image but we clearly have column metadata
    if getattr(doc, "image", None) is None and isinstance(md.get("columns"), (list, tuple)):
        return True

    return False


def is_alive(obj) -> bool:
    """True if obj is a live Qt wrapper (not deleted)."""
    if obj is None:
        return False
    if sip is not None:
        try:
            return not sip.isdeleted(obj)
        except Exception:
            pass
    # Touch-test: some cheap attribute access; if wrapper is dead this raises RuntimeError
    try:
        getattr(obj, "objectName", None)
        return True
    except RuntimeError:
        return False


def safe_widget(sw) -> Optional[object]:
    """Returns sw.widget() if both subwindow and its widget are alive; else None."""
    try:
        if not is_alive(sw):
            return None
        w = sw.widget()
        return w if is_alive(w) else None
    except Exception:
        return None
