"""
Centralized lazy import system for heavy dependencies.

This module provides a lazy loading mechanism to defer imports of large
libraries until they are actually needed, reducing startup time and
initial memory footprint.
"""
import importlib
import threading
from typing import Any, Optional


class LazyModule:
    """
    Lazy-loading wrapper for a module.
    
    Attributes are forwarded to the actual module once it's imported.
    Thread-safe and caches import failures to avoid repeated attempts.
    """
    
    def __init__(self, module_name: str, optional: bool = False):
        """
        Initialize lazy module wrapper.
        
        Args:
            module_name: Full module path (e.g., 'scipy.ndimage')
            optional: If True, returns None on import failure instead of raising
        """
        self._module_name = module_name
        self._module: Optional[Any] = None
        self._optional = optional
        self._failed = False
        self._lock = threading.Lock()
    
    def _import(self):
        """Import the module if not already imported."""
        if self._module is not None or self._failed:
            return
        
        with self._lock:
            # Double-check after acquiring lock
            if self._module is not None or self._failed:
                return
            
            try:
                self._module = importlib.import_module(self._module_name)
            except Exception as e:
                self._failed = True
                if not self._optional:
                    raise ImportError(
                        f"Failed to import required module '{self._module_name}': {e}"
                    ) from e
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the underlying module."""
        self._import()
        if self._module is None:
            if self._optional:
                return None
            raise AttributeError(
                f"Module '{self._module_name}' not available (import failed)"
            )
        return getattr(self._module, name)
    
    def __call__(self, *args, **kwargs):
        """Allow calling the module itself if it's callable."""
        self._import()
        if self._module is None:
            if self._optional:
                return None
            raise RuntimeError(
                f"Module '{self._module_name}' not available (import failed)"
            )
        return self._module(*args, **kwargs)
    
    @property
    def is_available(self) -> bool:
        """Check if the module is available without triggering import."""
        if self._module is not None:
            return True
        if self._failed:
            return False
        # Try to import to check availability
        self._import()
        return self._module is not None


# ============================================================================
# Lazy importers for heavy dependencies
# ============================================================================

# SciPy modules
lazy_scipy_ndimage = LazyModule('scipy.ndimage')
lazy_scipy_signal = LazyModule('scipy.signal')
lazy_scipy_interpolate = LazyModule('scipy.interpolate')
lazy_scipy_optimize = LazyModule('scipy.optimize')

# OpenCV
lazy_cv2 = LazyModule('cv2', optional=True)

# PyTorch (optional, may not be installed)
lazy_torch = LazyModule('torch', optional=True)

# Astronomy libraries
lazy_astroalign = LazyModule('astroalign', optional=True)
lazy_sep = LazyModule('sep', optional=True)
lazy_reproject = LazyModule('reproject', optional=True)


# ============================================================================
# Lazy imports for photutils and lightkurve (migrated from main files)
# ============================================================================

_photutils_isophote = None
_photutils_lock = threading.Lock()

def get_photutils_isophote():
    """Lazy loader for photutils.isophote module."""
    global _photutils_isophote
    
    if _photutils_isophote is not None:
        return _photutils_isophote if _photutils_isophote is not False else None
    
    with _photutils_lock:
        # Double-check after lock
        if _photutils_isophote is not None:
            return _photutils_isophote if _photutils_isophote is not False else None
        
        try:
            from photutils import isophote as _isophote_module
            _photutils_isophote = _isophote_module
        except Exception:
            _photutils_isophote = False  # Mark as failed
    
    return _photutils_isophote if _photutils_isophote is not False else None


def get_Ellipse():
    """Get photutils.isophote.Ellipse, loading lazily."""
    mod = get_photutils_isophote()
    return mod.Ellipse if mod else None


def get_EllipseGeometry():
    """Get photutils.isophote.EllipseGeometry, loading lazily."""
    mod = get_photutils_isophote()
    return mod.EllipseGeometry if mod else None


def get_build_ellipse_model():
    """Get photutils.isophote.build_ellipse_model, loading lazily."""
    mod = get_photutils_isophote()
    return mod.build_ellipse_model if mod else None


_lightkurve_module = None
_lightkurve_lock = threading.Lock()

def get_lightkurve():
    """Lazy loader for lightkurve module."""
    global _lightkurve_module
    
    if _lightkurve_module is not None:
        return _lightkurve_module if _lightkurve_module is not False else None
    
    with _lightkurve_lock:
        # Double-check after lock
        if _lightkurve_module is not None:
            return _lightkurve_module if _lightkurve_module is not False else None
        
        try:
            import lightkurve as _lk
            _lk.MPLSTYLE = None
            _lightkurve_module = _lk
        except Exception:
            _lightkurve_module = False  # Mark as failed
    
    return _lightkurve_module if _lightkurve_module is not False else None


# ============================================================================
# Convenience function for reproject.reproject_interp
# ============================================================================

def get_reproject_interp():
    """Get reproject.reproject_interp function if available."""
    try:
        mod = lazy_reproject
        if mod.is_available:
            return mod.reproject_interp
    except Exception:
        pass
    return None
