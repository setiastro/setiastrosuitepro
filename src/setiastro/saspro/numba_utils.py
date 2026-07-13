"""Compatibility shim.

The canonical implementation now lives in
``setiastro.saspro.legacy.numba_utils``. This module makes
``setiastro.saspro.numba_utils`` resolve to that exact same module object so
every existing ``import setiastro.saspro.numba_utils`` (and every
``from setiastro.saspro.numba_utils import ...``) keeps working unchanged.

Aliasing via sys.modules — rather than a star-import re-export — is deliberate
and important here: numba_utils reassigns module-level globals at runtime
(NUMBA_PARALLEL_LOCK, _NUMBA_SERIALIZE, _DEBAYER_LOCK) inside
set_numba_parallel_serialize(). A star-import would create *separate* bindings
in this module that would silently go stale after that swap. By making this
name an alias for the one true module, there is only a single set of globals,
so runtime lock swaps are visible through either import path.
"""
import sys
from setiastro.saspro.legacy import numba_utils as _canonical

sys.modules[__name__] = _canonical