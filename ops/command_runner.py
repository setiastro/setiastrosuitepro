# ops/command_runner.py
from __future__ import annotations
from typing import Any, Dict, Optional, Callable
import inspect

from ops.commands import get_spec, normalize_cid, CommandSpec

class CommandError(RuntimeError):
    pass


def _merge_defaults(spec: CommandSpec, preset: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(preset or {})
    for ps in spec.presets:
        if ps.key not in out and ps.default is not None:
            out[ps.key] = ps.default
    return out


def _validate_preset(spec: CommandSpec, preset: Dict[str, Any]) -> None:
    # light validation only; don't be annoying for scripts
    for ps in spec.presets:
        if (not ps.optional) and (ps.key not in preset):
            raise CommandError(f"{spec.id}: missing required preset key '{ps.key}'")

        if ps.key not in preset:
            continue

        v = preset[ps.key]
        t = ps.type

        try:
            if t == "float":
                preset[ps.key] = float(v)
            elif t == "int":
                preset[ps.key] = int(v)
            elif t == "bool":
                preset[ps.key] = bool(v)
            elif t == "str":
                preset[ps.key] = str(v)
            elif t == "enum":
                sv = str(v).lower()
                if ps.enum and sv not in [e.lower() for e in ps.enum]:
                    raise CommandError(f"{spec.id}: '{ps.key}' must be one of {ps.enum}, got {v}")
                preset[ps.key] = v
        except CommandError:
            raise
        except Exception:
            raise CommandError(f"{spec.id}: bad value for '{ps.key}': {v!r}")

        # numeric limits
        if ps.type in ("float", "int"):
            fv = float(preset[ps.key])
            if ps.min is not None and fv < ps.min:
                raise CommandError(f"{spec.id}: '{ps.key}' < min {ps.min}")
            if ps.max is not None and fv > ps.max:
                raise CommandError(f"{spec.id}: '{ps.key}' > max {ps.max}")


def _resolve_target_doc(ctx, target_doc=None, on_base: bool=False):
    """
    Resolve a doc for scripts:
    - if target_doc passed: use it
    - else prefer DocManager active doc (ROI-aware)
    - else use ctx.active_document()
    - if on_base True, swap to focused base doc when possible
    """
    doc = target_doc

    if doc is None:
        # ✅ Prefer DocManager (it knows about ROI, LiveViewDocument, etc.)
        app = getattr(ctx, "app", None)
        dm  = getattr(app, "doc_manager", None) if app is not None else None
        if dm is not None:
            try:
                doc = dm.get_active_document()
            except Exception:
                doc = None

        if doc is None:
            try:
                doc = ctx.active_document()
            except Exception:
                doc = None

    if doc is None:
        raise CommandError("No active document to run command on.")

    if on_base:
        app = getattr(ctx, "app", None)
        dm  = getattr(app, "doc_manager", None) if app is not None else None
        if dm is not None:
            try:
                base = dm.get_focused_base_document()
                if base is not None:
                    doc = base
            except Exception:
                pass
        else:
            # fallback to view.base_document if it exists
            try:
                view = ctx.active_view()
                base_doc = getattr(view, "base_document", None)
                if base_doc is not None:
                    doc = base_doc
            except Exception:
                pass

    return doc


def _load_callable(spec: CommandSpec, app_window=None) -> Callable:
    """
    Find the executor for this command. Priority:
    1) spec.callable_name + import_path
    2) spec.headless_method on app_window
    """
    if spec.import_path and spec.callable_name:
        mod = __import__(spec.import_path, fromlist=[spec.callable_name])
        fn = getattr(mod, spec.callable_name, None)
        if callable(fn):
            return fn
        raise CommandError(f"{spec.id}: callable '{spec.callable_name}' not found in {spec.import_path}")

    if app_window and spec.headless_method:
        fn = getattr(app_window, spec.headless_method, None)
        if callable(fn):
            return fn
        raise CommandError(f"{spec.id}: app has no headless method '{spec.headless_method}'")

    raise CommandError(f"{spec.id}: no executor defined in CommandSpec")


def run_command(ctx, command_id: str, preset: Optional[Dict[str, Any]] = None,
                *, target_doc=None, on_base: bool=False, open_ui: bool=False):
    """
    Script entry point:
    - Normalizes CID
    - merges defaults
    - validates preset
    - resolves target doc
    - runs headless executor
    """
    cid = normalize_cid(command_id)
    spec = get_spec(cid)
    if spec is None:
        raise CommandError(f"Unknown command_id '{command_id}' (normalized='{cid}')")

    preset = _merge_defaults(spec, dict(preset or {}))
    _validate_preset(spec, preset)

    # if scripts want UI, allow only if spec has ui_method
    if open_ui:
        if not spec.ui_method:
            raise CommandError(f"{spec.id}: has no UI opener")
        app = getattr(ctx, "app", None)
        m = getattr(app, spec.ui_method, None)
        if not callable(m):
            raise CommandError(f"{spec.id}: UI method '{spec.ui_method}' not found on app")
        return m(preset) if preset else m()

    doc = _resolve_target_doc(ctx, target_doc=target_doc, on_base=on_base)
    app = getattr(ctx, "app", None)

    fn = _load_callable(spec, app_window=app)


    try:
        sig = inspect.signature(fn)
        kwargs = {}

        for name, p in sig.parameters.items():
            lname = name.lower()

            # ignore *args/**kwargs params
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            # ctx / context
            if "ctx" in lname or "context" in lname:
                kwargs[name] = ctx
                continue

            # app / main window
            if lname in ("app", "main", "mw", "window", "app_window") or "main_window" in lname:
                kwargs[name] = app
                continue

            # doc / target_doc
            if lname in ("doc", "document", "target", "target_doc", "active_doc") or "document" in lname:
                kwargs[name] = doc
                continue

            # preset / params
            if lname in ("preset", "params", "options", "settings"):
                kwargs[name] = preset
                continue

        # ✅ First try pure kwargs (best match for your headless via_preset funcs)
        try:
            return fn(**kwargs)
        except TypeError:
            # ---------- safe fallbacks ----------
            # common module-level patterns
            try:
                return fn(app, doc, preset)
            except TypeError:
                try:
                    return fn(doc, preset)
                except TypeError:
                    try:
                        return fn(doc)
                    except TypeError:
                        return fn(preset)

    except Exception as e:
        raise CommandError(f"{spec.id}: executor failed: {e}") from e
