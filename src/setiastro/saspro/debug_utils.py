# pro/debug_utils.py (or near save_document)

from astropy.io import fits
import logging

log = logging.getLogger(__name__)

def debug_dump_metadata(meta: dict, context: str = ""):
    """
    Dump all metadata keys and highlight any fits.Header objects.
    """
    if not isinstance(meta, dict):
        log.debug("[MetaDump %s] metadata is not a dict: %r", context, type(meta))
        return

    log.debug("===== METADATA DUMP (%s) =====", context)
    log.debug("keys: %s", ", ".join(sorted(meta.keys())))

    for key, value in meta.items():
        if isinstance(value, fits.Header):
            log.debug("[MetaDump %s] %s -> fits.Header with %d cards", context, key, len(value))
            # If you want the *full* header:
            for card in value.cards:
                log.debug("[MetaDump %s]   %-10s = %r", context, card.keyword, card.value)
        else:
            log.debug("[MetaDump %s] %s -> %r (%s)",
                      context, key, value, type(value).__name__)

    log.debug("===== END METADATA DUMP (%s) =====", context)
