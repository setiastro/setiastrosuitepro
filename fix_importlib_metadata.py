# fix_importlib_metadata.py
# A tiny shim so third-party code that imports the backport keeps working.
try:
    import importlib.metadata as _stdlib_metadata  # Python 3.8+
except Exception:
    _stdlib_metadata = None

try:
    import sys
    if _stdlib_metadata and 'importlib_metadata' not in sys.modules:
        import types
        mod = types.ModuleType('importlib_metadata')
        # Expose the stdlib api under the backport name
        for k in dir(_stdlib_metadata):
            try:
                setattr(mod, k, getattr(_stdlib_metadata, k))
            except Exception:
                pass
        sys.modules['importlib_metadata'] = mod
except Exception:
    pass
