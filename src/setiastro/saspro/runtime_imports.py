# in pro/runtime_imports.py (new helper)
def get_lightkurve():
    import os
    os.environ.setdefault("LIGHTKURVE_STYLE", "default")
    import lightkurve as lk
    lk.MPLSTYLE = None
    return lk
