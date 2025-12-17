# pro/headless_utils.py
from __future__ import annotations

def unwrap_docproxy(x, max_depth: int = 8):
    """
    Safely unwrap live/roi/doc proxies to a real ImageDocument when possible.
    - Recurses a few levels.
    - Understands LiveViewDocument (_current/_base) and ROI wrappers (_parent_doc).
    - Never unwraps to None unless input was None.
    """
    if x is None:
        return None

    seen = set()
    y = x

    for _ in range(max_depth):
        if y is None or id(y) in seen:
            break
        seen.add(id(y))

        # LiveViewDocument / similar: prefer its resolver
        cur = getattr(y, "_current", None)
        if callable(cur):
            try:
                z = cur()
                if z is not None and z is not y:
                    y = z
                    continue
            except Exception:
                pass

        # Common doc proxy fields (ordered)
        for attr in (
            "_base", "base",
            "_parent_doc", "parent_doc",
            "base_document", "_base_document",
            "_target", "target",
            "_doc", "doc",
            "_obj", "obj",
            "_proxied", "proxied",
            "_wrapped", "wrapped",
        ):
            try:
                z = getattr(y, attr, None)
            except Exception:
                z = None
            if z is not None and z is not y:
                y = z
                break
        else:
            break

    return y



def normalize_headless_main(main_or_ctx, target_doc=None):
    """
    Returns (main_window, doc, doc_manager)
    Ensures doc + dm are fully unwrapped and ROI-aware.
    """
    ctx = None
    main = main_or_ctx

    if hasattr(main_or_ctx, "app") and hasattr(main_or_ctx, "active_document"):
        ctx = main_or_ctx
        main = getattr(ctx, "app", None)
        if target_doc is None:
            try:
                # Prefer dm.get_active_document() if possible (ROI-aware, real doc type)
                dm0 = getattr(main, "doc_manager", None) or getattr(main, "dm", None)
                dm0 = unwrap_docproxy(dm0)
                if dm0 is not None and hasattr(dm0, "get_active_document"):
                    target_doc = dm0.get_active_document()
                else:
                    target_doc = ctx.active_document()
            except Exception:
                target_doc = None

    doc = unwrap_docproxy(target_doc)

    dm = None
    if main is not None:
        dm = getattr(main, "doc_manager", None) or getattr(main, "dm", None)


    return main, doc, dm
