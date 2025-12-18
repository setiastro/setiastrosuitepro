"""
File utilities for Seti Astro Suite Pro.

This module provides centralized file handling utilities to avoid
code duplication across the codebase.

Included:
- Extension normalization
- Filename sanitization
- Path utilities
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, List, Set


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Whether to replace spaces with underscores in filenames
REPLACE_SPACES_WITH_UNDERSCORES = True

# Windows reserved device names
WIN_RESERVED_NAMES: Set[str] = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

# Extension normalization mappings
EXT_ALIASES = {
    "jpeg": "jpg",
    "tiff": "tif",
}

# Allowed bit depths per format
ALLOWED_BIT_DEPTHS = {
    "png":  {"8-bit"},
    "jpg":  {"8-bit"},
    "fits": {"32-bit floating point"},
    "fit":  {"32-bit floating point"},
    "tif":  {"8-bit", "16-bit", "32-bit unsigned", "32-bit floating point"},
    "xisf": {"16-bit", "32-bit unsigned", "32-bit floating point"},
}


# ---------------------------------------------------------------------------
# Extension Normalization
# ---------------------------------------------------------------------------

def normalize_ext(ext: str) -> str:
    """
    Normalize a file extension to its canonical form.
    
    - Converts to lowercase
    - Removes leading dot
    - Converts aliases (jpeg→jpg, tiff→tif)
    
    Args:
        ext: Extension string (with or without leading dot)
        
    Returns:
        Normalized extension without leading dot
        
    Example:
        >>> normalize_ext(".JPEG")
        'jpg'
        >>> normalize_ext("TIFF")
        'tif'
        >>> normalize_ext(".png")
        'png'
    """
    e = ext.lower().lstrip(".")
    return EXT_ALIASES.get(e, e)


# Alias for backward compatibility
_normalize_ext = normalize_ext


# ---------------------------------------------------------------------------
# Filename Sanitization
# ---------------------------------------------------------------------------

def sanitize_filename(
    basename: str,
    replace_spaces: bool = REPLACE_SPACES_WITH_UNDERSCORES
) -> str:
    """
    Sanitize a filename for cross-platform compatibility.
    
    Handles:
    - Collapsed/trimmed whitespace
    - Optional spaces→underscores conversion
    - Illegal characters removed (Windows/macOS/Linux superset)
    - No leading/trailing dots/spaces
    - Windows reserved device names avoided by appending '_'
    
    Args:
        basename: The filename to sanitize (without directory path)
        replace_spaces: Whether to replace spaces with underscores
        
    Returns:
        Sanitized filename
        
    Example:
        >>> sanitize_filename("my file: test?.png")
        'my_file_test.png'
        >>> sanitize_filename("CON.txt")
        'CON_.txt'
    """
    name = (basename or "").strip()
    
    # Split name/ext carefully
    stem, ext = os.path.splitext(name)
    
    # Collapse weird whitespace in stem
    stem = " ".join(stem.split())
    
    # Replace spaces if requested
    if replace_spaces and stem:
        stem = stem.replace(" ", "_")
    
    # Remove illegal characters (Windows/macOS/Linux superset)
    # Illegal: < > : " / \ | ? * and control characters
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', stem)
    ext = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', ext)
    
    # Strip leading/trailing dots and spaces from stem
    stem = stem.strip(". ")
    
    # Handle Windows reserved names
    stem_upper = stem.upper()
    if stem_upper in WIN_RESERVED_NAMES:
        stem = stem + "_"
    
    # Also check stem without extension for reserved names
    stem_base_upper = stem_upper.split(".")[0] if "." in stem_upper else stem_upper
    if stem_base_upper in WIN_RESERVED_NAMES and stem_upper != stem_base_upper:
        stem = stem + "_"
    
    # Combine stem and extension
    if ext:
        return stem + ext
    return stem if stem else "unnamed"


# Alias for backward compatibility
_sanitize_filename = sanitize_filename


# ---------------------------------------------------------------------------
# Path Utilities
# ---------------------------------------------------------------------------

def get_unique_path(path: str) -> str:
    """
    Get a unique file path by appending a number if the file exists.
    
    Args:
        path: Original file path
        
    Returns:
        Unique path (original if doesn't exist, or with _1, _2, etc.)
        
    Example:
        >>> get_unique_path("image.png")  # if exists
        'image_1.png'
    """
    if not os.path.exists(path):
        return path
    
    p = Path(path)
    stem = p.stem
    suffix = p.suffix
    parent = p.parent
    
    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not os.path.exists(new_path):
            return str(new_path)
        counter += 1


def ensure_extension(path: str, default_ext: str) -> str:
    """
    Ensure a file path has an extension.
    
    Args:
        path: File path
        default_ext: Default extension to add if missing
        
    Returns:
        Path with extension
    """
    p = Path(path)
    if not p.suffix:
        ext = default_ext if default_ext.startswith(".") else f".{default_ext}"
        return str(p) + ext
    return path


def get_extension(path: str) -> str:
    """
    Get the normalized extension from a file path.
    
    Args:
        path: File path
        
    Returns:
        Normalized extension without dot
    """
    ext = Path(path).suffix
    return normalize_ext(ext) if ext else ""


def exts_from_filter(selected_filter: str) -> List[str]:
    """
    Extract extensions from a Qt name filter string.
    
    Args:
        selected_filter: Qt filter string like "TIFF (*.tif *.tiff)"
        
    Returns:
        List of normalized extensions
        
    Example:
        >>> exts_from_filter("TIFF (*.tif *.tiff)")
        ['tif']
    """
    exts = [
        m.group(1).lower()
        for m in re.finditer(r"\*\.\s*([A-Za-z0-9]+)", selected_filter)
    ]
    if not exts:
        return []
    
    # Normalize and uniquify while preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for e in exts:
        n = normalize_ext(e)
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


# Alias for backward compatibility
_exts_from_filter = exts_from_filter


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    'REPLACE_SPACES_WITH_UNDERSCORES',
    'WIN_RESERVED_NAMES',
    'EXT_ALIASES',
    'ALLOWED_BIT_DEPTHS',
    
    # Extension normalization
    'normalize_ext',
    '_normalize_ext',
    
    # Filename sanitization
    'sanitize_filename',
    '_sanitize_filename',
    
    # Path utilities
    'get_unique_path',
    'ensure_extension',
    'get_extension',
    'exts_from_filter',
    '_exts_from_filter',
]
