# pro/dnd_mime.py
"""
Custom MIME types used for internal drag-and-drop in Seti Astro Suite (SAS/SASpro).
Keep these as plain strings (Qt expects str MIME type names).
"""

# Existing
MIME_VIEWSTATE   = "application/x-sas-viewstate"    # pan/zoom/scale + autostretch state
MIME_CMD         = "application/x-sas-command"      # function/action command payload
MIME_MASK        = "application/x-sas-mask"         # mask document payload
MIME_ACTION      = "application/x-saspro-action-id" # legacy/action id payload
MIME_ASTROMETRY  = "application/x-sas-astrometry"   # WCS copy payload

# New (for Alt+drag ⧉ to create live link between two views)
MIME_LINKVIEW    = "application/x-sas-link-view"    # view-link handshake (carries source_view_id)

# Optional: a quick set you can reference when accepting DnD
SUPPORTED_MIME_FORMATS = {
    MIME_VIEWSTATE,
    MIME_CMD,
    MIME_MASK,
    MIME_ACTION,
    MIME_ASTROMETRY,
    MIME_LINKVIEW,
}

__all__ = [
    "MIME_VIEWSTATE",
    "MIME_CMD",
    "MIME_MASK",
    "MIME_ACTION",
    "MIME_ASTROMETRY",
    "MIME_LINKVIEW",
    "SUPPORTED_MIME_FORMATS",
]
