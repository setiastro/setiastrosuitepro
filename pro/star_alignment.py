from __future__ import annotations
import warnings

# Deprecation Warning
warnings.warn(
    "The 'pro.star_alignment' module is deprecated and will be removed in a future version. "
    "Please use 'pro.alignment' package structure instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new package structure structure
# Core logic
from pro.alignment.core import (
    _gray2d,
    aa_find_transform_with_backoff
)

# Functions & Helpers
from pro.alignment.functions import (
    run_star_alignment_headless,
    compute_pairs_astroalign,
    handle_shortcut,
    STAR_ALIGN_CID,
    _align_prefs,
    _cap_native_threads_once,
    _find_main_window_from_child,
    _resolve_doc_and_sw_by_ptr,
    _doc_from_sw,
    _warp_like_ref,
    _fmt_doc_title,
    _list_open_docs_fallback,
    _doc_image,
    _active_doc_from_parent,
    _get_image_from_active_view,
    _push_image_to_active_view,
    _cap_points
)

# Mosaic Components
from pro.alignment.mosaic import (
    MosaicMasterDialog,
    MosaicPreviewWindow,
    MosaicSettingsDialog
)

# Stellar Components
from pro.alignment.stellar import (
    StellarAlignmentDialog,
    StarRegistrationThread,
    StarRegistrationWindow,
    StarRegistrationWorker,
)

# Other exports
from pro.alignment.core import (
    IDENTITY_2x3,
    PolyGradientRemoval
)
from skimage.transform import PolynomialTransform


# If consumers imported * from this file, they expect classes and functions.
# The internal helpers starting with _ were technically private but we re-exported some above just in case.