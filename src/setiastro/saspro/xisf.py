"""Compatibility shim. The XISF reader itself lives in legacy/xisf.py.

This file used to be a second, byte-identical copy of that 1200-line reader,
and the two were imported by different halves of the application:

    saspro/xisf.py          blink_fast_read, doc_manager, header_viewer,
                            exoplanet_detector, wimi, legacy/image_manager
    saspro/legacy/xisf.py   stacking_suite (4 sites), mfdeconv x3,
                            stacking_measure_worker, astrobin_exporter

Keeping two copies in step by hand is a standing trap: a fix applied to one
is simply missing from the other, and nothing errors. That is not
hypothetical -- a readinto() fast path added to the legacy copy did nothing
at all, because the file-open path goes through this one.

legacy/xisf.py is now the single implementation and every in-tree import
points at it directly. This module remains only so that anything outside the
tree -- a user script, a frozen build's hidden imports, an old plugin --
keeps working. New code should import from setiastro.saspro.legacy.xisf.
"""

from setiastro.saspro.legacy.xisf import XISF  # noqa: F401  (the only name anyone imported)
from setiastro.saspro.legacy import xisf as _xisf

__version__ = _xisf.__version__


def __getattr__(name):
    """Forward anything else to the real module (PEP 562), so the shim is
    total rather than a list of names that can drift out of date."""
    return getattr(_xisf, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_xisf)))
