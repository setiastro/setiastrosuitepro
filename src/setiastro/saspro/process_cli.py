# src/setiastro/saspro/process_cli.py
"""
Headless image-processing CLI.

    setiastrosuitepro process list
    setiastrosuitepro process TOOL -i IN -o OUT [tool options]
    setiastrosuitepro process TOOL -i INDIR -o OUTDIR --batch
    setiastrosuitepro process pipeline --steps STEPS.json -i IN -o OUT

Each tool runs the same headless code SASpro's shortcuts and drag-and-drop
presets run, with the same preset keys, so a setting gives the same pixels
here as it does in the app. Files are opened the way File > Open opens them
(including its rescale into [0, 1]) and written by the same saver as
File > Save As.

Written by Franklin Marek
www.setiastro.com
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

IMAGE_EXTS = {".fits", ".fit", ".fts", ".xisf", ".tif", ".tiff",
              ".png", ".jpg", ".jpeg"}

# Output format names as save_document spells them, keyed by what a user types.
FORMATS = {
    "fits": "fits", "fit": "fit", "fts": "fts",
    "tif": "tif", "tiff": "tif",
    "xisf": "xisf",
    "png": "png",
    "jpg": "jpg", "jpeg": "jpg",
}

BIT_DEPTHS = {
    "8": "8-bit",
    "16": "16-bit",
    "32u": "32-bit unsigned",
    "32f": "32-bit floating point",
}

# Formats that hold 32-bit float, which a --save-mask file is always written as.
MASK_EXTS = {".fits", ".fit", ".fts", ".xisf", ".tif", ".tiff"}


# ── Tool table ──────────────────────────────────────────────────────────────

@dataclass
class Param:
    """One tool option. `key` is the preset key the app's own shortcuts use."""
    key: str
    type: str                     # "float" | "int" | "bool" | "choice" | "str" | "json"
    default: Any                  # what the tool uses when the key is absent
    help: str
    choices: list[str] | None = None
    min: float | None = None
    max: float | None = None
    flag: str | None = None       # override the derived --kebab-case flag

    @property
    def option(self) -> str:
        return self.flag or "--" + self.key.replace("_", "-")


@dataclass
class Tool:
    id: str                                   # the app's command id
    name: str
    summary: str
    run: Callable[["_CLIDocument", dict, "_RunContext"], None]
    params: list[Param] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    notes: str = ""
    examples: list[str] = field(default_factory=list)   # set in _EXAMPLES below
    mask: str = ""        # what --save-mask writes; empty: the tool has no mask


TOOLS: dict[str, Tool] = {}
_ALIASES: dict[str, str] = {}


def _register(tool: Tool) -> Tool:
    TOOLS[tool.id] = tool
    for a in [tool.id, *tool.aliases]:
        _ALIASES[a.lower().replace("-", "_")] = tool.id
    return tool


def resolve_tool(name: str) -> Tool | None:
    key = str(name or "").strip().lower().replace("-", "_")
    if key in _ALIASES:
        return TOOLS[_ALIASES[key]]
    # The app's own normaliser knows the long spellings ("statistical stretch").
    try:
        from setiastro.saspro.command_ids import normalize_command_id
        cid = normalize_command_id(name)
        if cid in TOOLS:
            return TOOLS[cid]
    except Exception:
        pass
    return None


# ── Qt / message boxes ──────────────────────────────────────────────────────

class _Messages:
    """
    Every QMessageBox a tool raises, captured instead of shown. A modal box in
    a process with no one to click it would hang the caller forever; and a
    tool that reports a failure only through a warning box (and then returns)
    must still fail the run, so warnings and errors are kept for the verdict.
    """
    def __init__(self):
        self.items: list[tuple[str, str, str]] = []

    def install(self):
        from PyQt6.QtWidgets import QMessageBox
        SB = QMessageBox.StandardButton

        def _make(level, answer):
            def _box(*a, **k):
                title = str(a[1]) if len(a) > 1 else str(k.get("title", ""))
                text = str(a[2]) if len(a) > 2 else str(k.get("text", ""))
                self.items.append((level, title, text))
                print(f"[{level}] {title}: {text}", file=sys.stderr, flush=True)
                return answer
            return staticmethod(_box)

        QMessageBox.information = _make("info", SB.Ok)
        QMessageBox.warning = _make("warning", SB.Ok)
        QMessageBox.critical = _make("error", SB.Ok)
        # A question nobody can answer is declined.
        QMessageBox.question = _make("question", SB.No)

    def problems_since(self, n: int) -> list[str]:
        return [f"{t}: {x}" if t else x
                for lvl, t, x in self.items[n:] if lvl in ("warning", "error")]


MESSAGES = _Messages()


def _ensure_app():
    # Same application identity as the GUI, so the tools read the user's own
    # settings (GPU choice, library folders). Nothing here writes settings.
    from setiastro.saspro.plate_solve_cli import _ensure_qapp
    app = _ensure_qapp()
    MESSAGES.install()
    return app


# ── Document ────────────────────────────────────────────────────────────────

def _document_class():
    """Built on first use so importing this module does not import Qt."""
    global _CLIDocument
    if "_CLIDocument" in globals() and _CLIDocument is not None:
        return _CLIDocument
    import numpy as np
    from setiastro.saspro.doc_manager import ImageDocument, _strip_replay_keys
    from setiastro.saspro.widgets.image_utils import ensure_contiguous

    critical = ("original_header", "fits_header", "wcs_header", "file_meta", "image_meta")

    class _Doc(ImageDocument):
        """
        An ImageDocument with no undo history. The real apply_edit writes a
        full-size snapshot to the swap folder on every edit, which a one-shot
        command line run never needs. Metadata is merged exactly as
        ImageDocument.apply_edit merges it (header-bearing keys survive unless
        replaced with a real value), so headers and WCS reach the output.
        """
        def __init__(self, image, metadata=None):
            super().__init__(image, metadata)
            self.steps_applied: list[str] = []

        def apply_edit(self, new_image, metadata=None, step_name="Edit"):
            if metadata:
                old = dict(self.metadata or {})
                for k, v in dict(_strip_replay_keys(metadata) or {}).items():
                    if k in critical and v is None:
                        continue
                    old[k] = v
                if step_name:
                    old.setdefault("step_name", step_name)
                self.metadata = old
            else:
                self.metadata.setdefault("step_name", step_name)
            try:
                self._sync_strata_from_metadata()
            except Exception:
                pass
            img = np.asarray(new_image, dtype=np.float32)
            if img.size == 0:
                raise ValueError("apply_edit: new image is empty")
            self.image = ensure_contiguous(img)
            self.dirty = True
            self.steps_applied.append(str(step_name or "Edit"))
            self.changed.emit()

        def set_image(self, img, metadata=None, step_name="Edit"):
            self.apply_edit(img, metadata=metadata, step_name=step_name)

    _CLIDocument = _Doc
    return _Doc


_CLIDocument = None


def load_document(path: str):
    """
    Open an image exactly as File > Open does for its main image: the
    metadata-aware loader, WCS attached to the metadata, the header snapshot,
    and the same rescale into [0, 1] (shift up if the minimum is negative,
    then divide by the maximum if it is above 1).
    """
    import os as _os
    from setiastro.saspro.legacy.image_manager import load_image, attach_wcs_to_metadata
    from setiastro.saspro.doc_manager import (
        _normalize_image_01, _snapshot_header_for_metadata, _normalize_ext)

    out = load_image(path, return_metadata=True)
    if not out or out[0] is None:
        raise IOError(f"could not read {path}")
    if len(out) == 5:
        img, header, bit_depth, is_mono, meta = out
    else:
        img, header, bit_depth, is_mono = out
        meta = None
    if meta is None:
        ext = _normalize_ext(_os.path.splitext(path)[1].lower().lstrip("."))
        meta = {"file_path": path, "original_header": header, "bit_depth": bit_depth,
                "is_mono": is_mono, "original_format": ext}
        meta = attach_wcs_to_metadata(meta, header)
    _snapshot_header_for_metadata(meta)
    return _document_class()(_normalize_image_01(img), meta)


def _save_document(doc, path: str, bit_depth: str | None) -> str:
    """File > Save As's own writer: header choice, WCS merge, integer clipping."""
    from setiastro.saspro.doc_manager import DocManager, _ALLOWED_DEPTHS, _normalize_ext
    ext = _normalize_ext(os.path.splitext(path)[1].lower().lstrip("."))
    allowed = list(_ALLOWED_DEPTHS.get(ext, []))
    if bit_depth is None:
        # 32-bit float wherever the format holds it: processing output should
        # not be quantised unless asked. PNG / JPEG only take 8-bit.
        bit_depth = "32-bit floating point" if "32-bit floating point" in allowed else (
            allowed[0] if allowed else None)
    elif allowed and bit_depth not in allowed:
        raise ValueError(f"{ext.upper()} cannot hold {bit_depth}; it takes {', '.join(sorted(allowed))}")
    dm = _RunContext.doc_manager()
    # save_document returns False for "wrote a lower-depth copy" (8/16-bit),
    # which is not a failure; a failed write raises.
    dm.save_document(doc, path, bit_depth_override=bit_depth)
    if not os.path.isfile(path):
        raise IOError(f"could not write {path}")
    return bit_depth


# ── Run context: what a tool may ask of "the app" ───────────────────────────

class _RunContext:
    """
    Stands in for the main window. Tools that take one use it for a parent
    widget, a log line, or the DocManager; documents a tool creates (ABE's
    background model, say) are collected so they can be written too.
    `masks` collects the mask a tool blended with, when --save-mask asked for
    it (None otherwise, so a full-size mask is not kept for nothing).
    """
    _dm = None

    def __init__(self, doc, quiet: bool = False, want_masks: bool = False):
        self.doc = doc
        self.quiet = quiet
        self.extra_docs: list = []
        self.masks: list | None = [] if want_masks else None
        self.warnings: list[str] = []
        self.main = _make_main_stub(self)

    def warn(self, msg: str):
        """Worth knowing, but not a failure: the result was still written."""
        self.warnings.append(msg)
        print(f"   warning: {msg}", file=sys.stderr, flush=True)

    @classmethod
    def doc_manager(cls):
        if cls._dm is None:
            from setiastro.saspro.doc_manager import DocManager
            cls._dm = DocManager()
        return cls._dm

    def log(self, msg: str):
        if not self.quiet:
            print(f"   {msg}", file=sys.stderr, flush=True)


def _make_main_stub(ctx: "_RunContext"):
    from PyQt6.QtWidgets import QWidget

    class _Main(QWidget):
        def __init__(self):
            super().__init__()
            dm = ctx.doc_manager()
            self.doc_manager = dm
            self.docman = dm
            try:
                dm.documentAdded.disconnect()
            except Exception:
                pass
            dm.documentAdded.connect(lambda d: ctx.extra_docs.append(d))

        def _log(self, msg):
            ctx.log(str(msg))

        def _active_doc(self):
            return ctx.doc

        def active_document(self):
            return ctx.doc

        def currentDocument(self):
            return ctx.doc

    return _Main()


# ── Running ─────────────────────────────────────────────────────────────────

class ToolFailed(RuntimeError):
    pass


def run_step(tool: Tool, doc, preset: dict, ctx: _RunContext) -> None:
    """Run one tool on `doc`. Raises ToolFailed with the tool's own reason."""
    import numpy as np
    n_msgs = len(MESSAGES.items)
    n_steps = len(doc.steps_applied)
    before = doc.image
    try:
        tool.run(doc, dict(preset), ctx)
    except ToolFailed:
        raise
    except Exception as e:
        raise ToolFailed(f"{tool.name}: {type(e).__name__}: {e}") from e
    problems = MESSAGES.problems_since(n_msgs)
    changed = len(doc.steps_applied) > n_steps or doc.image is not before
    if not changed:
        why = "; ".join(problems) if problems else "the tool made no change"
        raise ToolFailed(f"{tool.name}: {why}")
    if not np.all(np.isfinite(np.asarray(doc.image, dtype=np.float32))):
        raise ToolFailed(f"{tool.name}: produced non-finite pixels")


def _coerce(p: Param, v):
    if v is None:
        return None
    if p.type in ("float", "int"):
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{p.option}: expected a number, got {v!r}")
        if p.type == "int":
            if f != int(f):
                raise ValueError(f"{p.option}: expected a whole number, got {v!r}")
            v = int(f)
        else:
            v = f
    elif p.type == "bool":
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("1", "true", "yes", "on"):
                v = True
            elif s in ("0", "false", "no", "off"):
                v = False
            else:
                raise ValueError(f"{p.option}: expected true/false, got {v!r}")
        v = bool(v)
    elif p.type == "choice":
        s = str(v).strip()
        match = [c for c in (p.choices or []) if c.lower() == s.lower()]
        if not match:
            raise ValueError(f"{p.option}: must be one of {', '.join(p.choices or [])}; got {v!r}")
        v = match[0]
    elif p.type == "json":
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p.option}: not valid JSON ({e.msg})")
    else:
        v = str(v)
    if p.type in ("float", "int"):
        if p.min is not None and v < p.min:
            raise ValueError(f"{p.option}: {v} is below the minimum {p.min:g}")
        if p.max is not None and v > p.max:
            raise ValueError(f"{p.option}: {v} is above the maximum {p.max:g}")
    return v


def build_preset(tool: Tool, base: dict | None, explicit: dict) -> dict:
    """
    Preset keys from --preset (or a pipeline step), then explicit options on
    top. Known keys are type- and range-checked; unknown keys from a preset
    are passed through untouched, as a shortcut preset would be.
    """
    preset = dict(base or {})
    by_key = {p.key: p for p in tool.params}
    for k, v in list(preset.items()):
        if k in by_key:
            preset[k] = _coerce(by_key[k], v)
    for k, v in explicit.items():
        if v is not None:
            preset[k] = _coerce(by_key[k], v)
    return preset


def _read_json_arg(text: str, what: str):
    """A JSON value given inline or as a path to a .json file."""
    s = (text or "").strip()
    if not s:
        return None
    p = Path(s)
    if not s.startswith(("{", "[")) and p.exists():
        s = p.read_text(encoding="utf-8")
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"{what}: not valid JSON ({e.msg} at line {e.lineno} column {e.colno})")


def _normalise_steps(raw) -> list[tuple[Tool, dict]]:
    """
    Pipeline steps in Function Bundle form: a list of
    {"command_id": "...", "preset": {...}}, or {"steps": [...]} around it.
    "tool" and "id" are accepted for "command_id".
    """
    if isinstance(raw, dict):
        raw = raw.get("steps", raw.get("payload", {}).get("steps") if isinstance(raw.get("payload"), dict) else None)
    if not isinstance(raw, list) or not raw:
        raise ValueError("--steps: expected a non-empty list of {\"command_id\": ..., \"preset\": {...}}")
    out = []
    for i, st in enumerate(raw, 1):
        if not isinstance(st, dict):
            raise ValueError(f"--steps: step {i} is not an object")
        name = st.get("command_id") or st.get("tool") or st.get("id")
        tool = resolve_tool(str(name or ""))
        if tool is None:
            raise ValueError(f"--steps: step {i}: no command-line tool named {name!r} "
                             f"(see 'setiastrosuitepro process list')")
        preset = st.get("preset") or {}
        if not isinstance(preset, dict):
            raise ValueError(f"--steps: step {i}: preset must be an object")
        out.append((tool, build_preset(tool, preset, {})))
    return out


def _save_mask(mask, path: Path) -> None:
    """A --save-mask file: the mask as computed, 32-bit float mono, written by
    the same saver as the output."""
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    m = np.ascontiguousarray(mask, dtype=np.float32)
    doc = _document_class()(m, {"is_mono": True, "bit_depth": "32-bit floating point"})
    _save_document(doc, str(path), "32-bit floating point")


def _process_file(src: Path, dst: Path, steps: list[tuple[Tool, dict]],
                  bit_depth: str | None, quiet: bool, extra_dir: Path | None,
                  seed: int = 0, mask_path: Path | None = None) -> dict:
    import numpy as np
    rec: dict = {"input": str(src), "output": str(dst), "ok": False,
                 "steps": [{"command_id": t.id, "preset": p} for t, p in steps]}
    t0 = time.time()
    try:
        doc = load_document(str(src))
    except Exception as e:
        rec["error"] = f"could not open {src}: {e}"
        rec["io_error"] = True
        return rec
    rec["shape"] = list(doc.image.shape)
    ctx = _RunContext(doc, quiet=quiet, want_masks=mask_path is not None)
    try:
        for tool, preset in steps:
            t1 = time.time()
            ctx.log(f"{tool.name} {json.dumps(preset, default=str)}")
            # Background Neutralization's auto box and star White Balance
            # search with numpy's global RNG; seeding it makes runs repeat.
            np.random.seed(seed)
            run_step(tool, doc, preset, ctx)
            ctx.log(f"{tool.name} done in {time.time() - t1:.1f}s")
    except ToolFailed as e:
        rec["error"] = str(e)
        rec["seconds"] = round(time.time() - t0, 2)
        if ctx.warnings:
            rec["warnings"] = ctx.warnings
        return rec
    if ctx.warnings:
        rec["warnings"] = ctx.warnings
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        rec["bit_depth"] = _save_document(doc, str(dst), bit_depth)
    except Exception as e:
        rec["error"] = f"could not write {dst}: {e}"
        rec["io_error"] = True
        return rec
    if mask_path is not None:
        try:
            if not ctx.masks:
                raise RuntimeError("the tool returned no mask")
            _save_mask(ctx.masks[-1], mask_path)
            ctx.masks = None
            rec["mask_output"] = str(mask_path)
        except Exception as e:
            rec["error"] = f"could not write the mask {mask_path}: {e}"
            rec["io_error"] = True
            return rec
    extras = []
    try:
        src_name = str(doc.display_name() or "")
    except Exception:
        src_name = ""
    for i, xd in enumerate(ctx.extra_docs, 1):
        try:
            name = str((getattr(xd, "metadata", {}) or {}).get("display_name") or f"extra{i}")
            # "<input name>_ABE_BG" -> "ABE_BG"; the output stem already names the file.
            for prefix in (src_name, os.path.splitext(src_name)[0], src.stem):
                if prefix and name.startswith(prefix) and len(name) > len(prefix):
                    name = name[len(prefix):].lstrip("_- ")
                    break
            safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name).strip("_") or f"extra{i}"
            xp = (extra_dir or dst.parent) / f"{dst.stem}_{safe}{dst.suffix}"
            _save_document(xd, str(xp), bit_depth)
            extras.append(str(xp))
        except Exception as e:
            print(f"[process] could not write an extra output: {e}", file=sys.stderr)
    if extras:
        rec["extra_outputs"] = extras
    rec["ok"] = True
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


# ── Command line ────────────────────────────────────────────────────────────

def _common_options(ap: argparse.ArgumentParser, with_preset: bool = True, mask: str = ""):
    io = ap.add_argument_group("input and output")
    io.add_argument("-i", "--input", required=True, metavar="PATH",
                    help="Image to process, or a folder with --batch.")
    io.add_argument("-o", "--output", required=True, metavar="PATH",
                    help="Where to write the result, or the output folder with --batch. "
                         "The format follows the extension (FITS, XISF, TIFF, PNG, JPEG).")
    io.add_argument("--batch", action="store_true",
                    help="Process every image in the input folder into the output folder, "
                         "keeping each file's name.")
    io.add_argument("--format", choices=sorted(set(FORMATS)), metavar="EXT",
                    help="With --batch: output format (fits, xisf, tif, png, jpg). "
                         "Default: each input's own format.")
    io.add_argument("--bit-depth", choices=list(BIT_DEPTHS), metavar="DEPTH",
                    help="Output sample format: 8, 16, 32u or 32f. Default: 32f wherever "
                         "the format holds it (FITS, XISF, TIFF); 8 for PNG and JPEG.")
    if mask:
        io.add_argument("--save-mask", nargs="?", const="", default=None, metavar="PATH",
                        help=f"Also write {mask}, from the same run, as a 32-bit float mono "
                             "image. PATH names the file (.tif, .fits or .xisf); with --batch it "
                             "is a folder. Without PATH the mask goes beside the output as "
                             "NAME_mask, in the output's format (TIFF when that is PNG or JPEG).")
    io.add_argument("--overwrite", action="store_true",
                    help="Replace output files that already exist.")
    io.add_argument("--seed", type=int, default=0, metavar="N",
                    help="Seed for the random search Background Neutralization and star-based "
                         "White Balance use to find a background box, so runs repeat exactly. "
                         "Default: 0.")
    if with_preset:
        io.add_argument("--preset", metavar="JSON",
                        help="Preset as JSON, inline or a .json file: the same keys the app's "
                             "shortcuts and presets use. Options given here override it.")
    out = ap.add_argument_group("reporting")
    out.add_argument("--json", action="store_true",
                     help="Print a machine-readable result on stdout (implies --quiet).")
    out.add_argument("-q", "--quiet", action="store_true",
                     help="Only report errors.")


def _tool_epilog(tool: Tool) -> str:
    parts = []
    if tool.examples:
        parts.append("examples:\n" + "\n".join(f"  {e}" for e in tool.examples))
    if tool.notes:
        parts.append(tool.notes)
    if tool.aliases:
        parts.append("Also callable as: " + ", ".join(tool.aliases) + ".")
    parts.append("Every option can also come from --preset, using the key in brackets.\n"
                 "Batch runs, pipelines and output formats: setiastrosuitepro process --help")
    return "\n\n".join(parts)


def _tool_parser(tool: Tool) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"setiastrosuitepro process {tool.id}",
        description=f"{tool.name}: {tool.summary}",
        epilog=_tool_epilog(tool),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    if tool.params:
        g = ap.add_argument_group(f"{tool.name} options (preset key in brackets)")
        for p in tool.params:
            kw: dict = {"dest": "opt__" + p.key, "default": None}
            rng = ""
            if p.min is not None and p.max is not None:
                rng = f" Range {p.min:g}..{p.max:g}."
            elif p.min is not None:
                rng = f" At least {p.min:g}."
            elif p.max is not None:
                rng = f" At most {p.max:g}."
            if p.type == "bool":
                kw["metavar"] = "true|false"
            elif p.type == "choice":
                kw["metavar"] = "|".join(p.choices or [])
            elif p.type == "json":
                kw["metavar"] = "JSON"
            else:
                kw["metavar"] = p.type.upper() if p.type != "str" else "TEXT"
            d = p.default
            dtxt = ("true" if d else "false") if isinstance(d, bool) else d
            kw["help"] = f"{p.help}{rng} Default: {dtxt}. [{p.key}]"
            g.add_argument(p.option, **kw)
    _common_options(ap, mask=tool.mask)
    return ap


def _jobs(src: Path, dst: Path, batch: bool, fmt: str | None) -> list[tuple[Path, Path]]:
    if batch:
        if not src.is_dir():
            raise ValueError(f"--batch needs a folder, got {src}")
        if dst.exists() and not dst.is_dir():
            raise ValueError(f"--batch needs an output folder, got the file {dst}")
        jobs = []
        for f in sorted(src.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                ext = "." + FORMATS[fmt] if fmt else f.suffix
                jobs.append((f, dst / (f.stem + ext)))
        if not jobs:
            raise ValueError(f"no images found in {src}")
        return jobs
    if src.is_dir():
        raise ValueError("the input is a folder; pass --batch to process it")
    if fmt:
        raise ValueError("--format is for --batch; for one file, give -o the extension you want")
    if dst.suffix.lower() not in IMAGE_EXTS:
        raise ValueError(f"cannot tell the output format from {dst.name!r}; "
                         f"use one of {', '.join(sorted(IMAGE_EXTS))}")
    return [(src, dst)]


def _mask_path(out: Path, opt: str, batch: bool) -> Path:
    """Where --save-mask [opt] writes the mask that goes with the output `out`."""
    if opt and not batch:
        p = Path(opt)
        if p.suffix.lower() not in MASK_EXTS:
            raise ValueError(f"--save-mask: the mask is 32-bit float; name it .tif, .fits or "
                             f".xisf, not {p.name!r}")
        return p
    folder = Path(opt) if opt else out.parent
    if folder.exists() and not folder.is_dir():
        raise ValueError(f"--save-mask with --batch needs a folder, got the file {folder}")
    ext = out.suffix if out.suffix.lower() in MASK_EXTS else ".tif"
    return folder / f"{out.stem}_mask{ext}"


def _run_jobs(args, steps: list[tuple[Tool, dict]]) -> int:
    src, dst = Path(args.input), Path(args.output)
    if not src.exists():
        print(f"ERROR: input does not exist: {src}", file=sys.stderr)
        return 2
    save_mask = getattr(args, "save_mask", None)
    try:
        jobs = _jobs(src, dst, args.batch, args.format)
        masks = ([_mask_path(o, save_mask, args.batch) for _, o in jobs]
                 if save_mask is not None else [None] * len(jobs))
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if save_mask is not None:
        taken = set()
        for i, o in jobs:
            for f in (i, o):
                try:
                    taken.add(f.resolve())
                except Exception:
                    pass
        for m in masks:
            try:
                clash = m.resolve() in taken
            except Exception:
                clash = False
            if clash:
                print(f"ERROR: --save-mask would write over an input or output: {m}", file=sys.stderr)
                return 2
    if not args.overwrite:
        exists = [str(o) for _, o in jobs if o.exists()] + [str(m) for m in masks if m and m.exists()]
        if exists:
            print("ERROR: output already exists (pass --overwrite to replace): "
                  + ", ".join(exists[:5]) + (" ..." if len(exists) > 5 else ""), file=sys.stderr)
            return 2
    for i, o in jobs:
        try:
            if i.resolve() == o.resolve():
                print(f"ERROR: output is the input file: {o}", file=sys.stderr)
                return 2
        except Exception:
            pass

    quiet = bool(args.quiet or args.json)
    bit_depth = BIT_DEPTHS[args.bit_depth] if args.bit_depth else None
    if bit_depth:
        # Refuse up front, not after the tools have run.
        from setiastro.saspro.doc_manager import _ALLOWED_DEPTHS, _normalize_ext
        for _, o in jobs:
            ext = _normalize_ext(o.suffix.lower().lstrip("."))
            allowed = _ALLOWED_DEPTHS.get(ext)
            if allowed and bit_depth not in allowed:
                inv = {v: k for k, v in BIT_DEPTHS.items()}
                print(f"ERROR: {ext.upper()} cannot hold {bit_depth}; --bit-depth for it: "
                      + ", ".join(sorted(inv.get(a, a) for a in allowed)), file=sys.stderr)
                return 2
    _ensure_app()

    import contextlib
    # stdout is the --json channel. The tools print progress chatter of their
    # own, so while they run it goes to stderr (or nowhere, with --quiet).
    # (utf-8 with replace: a tool printing a sigma must never raise mid-run.)
    chatter = open(os.devnull, "w", encoding="utf-8", errors="replace") if quiet else sys.stderr
    results, failed, io_failed = [], 0, 0
    for (path, out), mask_path in zip(jobs, masks):
        if not quiet:
            print(f"=> {path.name}", file=sys.stderr, flush=True)
        with contextlib.redirect_stdout(chatter):
            rec = _process_file(path, out, steps, bit_depth, quiet, None, seed=args.seed,
                                mask_path=mask_path)
        results.append(rec)
        if rec["ok"]:
            if not quiet:
                print(f"   saved {rec['output']} ({rec['bit_depth']}, {rec['seconds']}s)",
                      file=sys.stderr, flush=True)
                if rec.get("mask_output"):
                    print(f"   saved {rec['mask_output']} (mask, 32-bit floating point)",
                          file=sys.stderr, flush=True)
                for x in rec.get("extra_outputs", []):
                    print(f"   saved {x}", file=sys.stderr, flush=True)
        else:
            failed += 1
            io_failed += 1 if rec.get("io_error") else 0
            print(f"   FAILED: {rec.get('error', 'unknown error')}", file=sys.stderr, flush=True)

    if args.json:
        json.dump(results if len(results) > 1 else results[0], sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    elif len(jobs) > 1:
        print(f"{len(jobs) - failed}/{len(jobs)} processed.", file=sys.stderr)
    if failed:
        return 2 if (io_failed == failed and len(jobs) == 1) else 1
    return 0


def _list_tools(stream=sys.stdout) -> int:
    import textwrap
    w = max(len(t.id) for t in TOOLS.values()) if TOOLS else 10
    indent = " " * (w + 4)
    stream.write("Tools (setiastrosuitepro process TOOL --help for each one's options):\n\n")
    for t in sorted(TOOLS.values(), key=lambda t: t.id):
        text = textwrap.wrap(f"{t.name}: {t.summary}", 78 - len(indent))
        stream.write(f"  {t.id:<{w}}  {text[0]}\n")
        for line in text[1:]:
            stream.write(indent + line + "\n")
        if t.aliases:
            for line in textwrap.wrap("aliases: " + ", ".join(t.aliases), 78 - len(indent)):
                stream.write(indent + line + "\n")
    stream.write("\nAlso: setiastrosuitepro process pipeline --steps STEPS.json -i IN -o OUT\n")
    return 0


def _options_overview(width: int = 78) -> str:
    """The options every tool takes, read off the real parser so this text
    cannot drift from what the parser accepts."""
    import textwrap
    ap = argparse.ArgumentParser(add_help=False)
    _common_options(ap)
    col = 24
    lines = []
    for a in ap._actions:
        if not a.option_strings:
            continue
        opts = ", ".join(a.option_strings)
        if a.nargs != 0 and a.metavar:
            opts += f" {a.metavar}"
        text = (a.help or "").replace("%%", "%")
        body = textwrap.wrap(text, width - col)
        if len(opts) <= col - 4:
            lines.append(f"  {opts:<{col - 2}}{body[0] if body else ''}")
            body = body[1:]
        else:
            lines.append(f"  {opts}")
        lines.extend(" " * col + b for b in body)
    return "\n".join(lines)


def _mask_tools_line() -> str:
    names = sorted(t.name for t in TOOLS.values() if t.mask)
    if not names:
        return ""
    return (f"  {' and '.join(names)} also take --save-mask:\n"
            "  the mask they blended with, written from the same run as its own image.\n")


def usage() -> str:
    return f"""usage: setiastrosuitepro process TOOL -i IN -o OUT [options]
       setiastrosuitepro process TOOL -i INDIR -o OUTDIR --batch [options]
       setiastrosuitepro process pipeline --steps STEPS.json -i IN -o OUT [options]
       setiastrosuitepro process list
       setiastrosuitepro process TOOL --help      (also: process help TOOL)

Runs SASpro's own processing tools on image files with no window, using the
same code and preset keys as the app's shortcuts and presets, so a setting
gives the same pixels here as it does in the app.

Images open the way File > Open opens them: 32-bit float, and rescaled into
0..1 (shifted up if anything is negative, then divided by the maximum if it
is above 1). Mono stays mono. Output is 32-bit float unless --bit-depth says
otherwise (PNG and JPEG are 8-bit). FITS headers and WCS are carried through.

OPTIONS EVERY TOOL TAKES
{_options_overview()}

  Each tool's own options: setiastrosuitepro process TOOL --help
{_mask_tools_line()}
PRESETS
  Every tool option has a preset key (shown in brackets in the tool's help),
  the same key the app's shortcuts store. A preset can be given whole with
  --preset, inline or as a .json file; a whole shortcut payload
  ({{"command_id": ..., "preset": {{...}}}}) is accepted too. Options given on
  the command line override the preset.

PIPELINES
  "process pipeline" runs several tools in one pass without writing between
  them. --steps is a list in Function Bundle form, inline or a .json file:
    [{{"command_id": "abe", "preset": {{"degree": 2}}}},
     {{"command_id": "stat_stretch", "preset": {{"target_median": 0.25}}}}]
  Pipelines take the options above except --preset.

EXAMPLES
  Stretch one image:
    setiastrosuitepro process stat_stretch -i stack.fits -o stretched.fits --target-median 0.2

  Every image in a folder, written as 16-bit TIFF:
    setiastrosuitepro process abe -i ./stacks -o ./abe --batch --format tif --bit-depth 16

  Settings from a saved shortcut preset, one of them overridden:
    setiastrosuitepro process curves -i in.tif -o out.tif --preset curves.json --amount 0.3

  Background, colour and stretch in one pass, machine-readable result:
    setiastrosuitepro process pipeline -i stack.fits -o final.tif --steps steps.json --json

EXIT CODES
  0 success, 1 a tool failed, 2 a usage or file error. With --batch, 1 if any
  file failed; the others are still written. --json reports each file.

HELP
  -h, --help, -help, --h, -?, /? and "help" all show help, here and after a
  tool name. "setiastrosuitepro help TOOL" works too.
"""


def _pipeline(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="setiastrosuitepro process pipeline",
        description="Run several tools in one pass, in order, writing only the final image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='STEPS is a list of {"command_id": TOOL, "preset": {...}} objects (Function\n'
               'Bundle form), inline or a .json file. "tool" or "id" may stand for\n'
               '"command_id"; {"steps": [...]} around the list is accepted. Tool names are\n'
               'those of "setiastrosuitepro process list"; preset keys are those in brackets\n'
               'in "setiastrosuitepro process TOOL --help".\n\n'
               'example steps.json:\n'
               '  [{"command_id": "abe", "preset": {"degree": 2}},\n'
               '   {"command_id": "background_neutral"},\n'
               '   {"command_id": "white_balance", "preset": {"mode": "star"}},\n'
               '   {"command_id": "stat_stretch", "preset": {"target_median": 0.22}}]\n\n'
               'example:\n'
               '  setiastrosuitepro process pipeline -i stack.fits -o final.tif --steps steps.json',
    )
    ap.add_argument("--steps", required=True, metavar="JSON",
                    help="The steps, as inline JSON or a path to a .json file.")
    _common_options(ap, with_preset=False)
    args = ap.parse_args(argv)
    try:
        steps = _normalise_steps(_read_json_arg(args.steps, "--steps"))
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return _run_jobs(args, steps)


def _tolerant_streams():
    """
    Tools print characters such as sigma. On Windows a pipe defaults to the
    ANSI code page, which has none, and the UnicodeEncodeError lands inside
    the tool: star-based White Balance caught it and quietly fell back to
    gray world. Replace what a stream cannot encode instead of raising.
    """
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass


def _help_tokens() -> set[str]:
    try:
        from setiastro.saspro.cli import HELP_TOKENS
        return set(HELP_TOKENS)
    except Exception:
        return {"-h", "--help", "--h", "-help", "help", "-?", "/?", "/h"}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _tolerant_streams()
    helps = _help_tokens()
    # Any help spelling after the first word means --help to argparse, which
    # would otherwise read "--h" as an ambiguous abbreviation and "/?" as a
    # stray argument. ("help" stays literal there: it could be a value.)
    argv = [argv[0], *("--help" if a in helps - {"help"} else a for a in argv[1:])] if argv else argv
    if len(argv) == 2 and argv[1] == "help":        # "process TOOL help": nothing else it could mean
        argv[1] = "--help"

    if not argv:
        argv = ["--help"]
    head = argv[0]
    if head in helps:
        rest = [a for a in argv[1:] if a != "--help"]
        if rest:                                   # "process help TOOL"
            return main([rest[0], "--help"])
        sys.stdout.write(usage())
        sys.stdout.write("\n")
        return _list_tools()
    if head.lower() in ("list", "--list", "tools"):
        return _list_tools()
    if head.lower() in ("pipeline", "steps"):
        return _pipeline(argv[1:])
    tool = resolve_tool(head)
    if tool is None:
        print(f"setiastrosuitepro process: no tool named {head!r}\n", file=sys.stderr)
        _list_tools(sys.stderr)
        return 2
    ap = _tool_parser(tool)
    args = ap.parse_args(argv[1:])
    explicit = {p.key: getattr(args, "opt__" + p.key) for p in tool.params}
    try:
        base = _read_json_arg(args.preset, "--preset") if args.preset else None
        if base is not None and not isinstance(base, dict):
            raise ValueError("--preset: expected a JSON object")
        if isinstance(base, dict) and isinstance(base.get("preset"), dict):
            base = base["preset"]          # a whole shortcut payload
        preset = build_preset(tool, base, explicit)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return _run_jobs(args, [(tool, preset)])


# ════════════════════════════════════════════════════════════════════════════
# Tools
# ════════════════════════════════════════════════════════════════════════════
# Each runner calls the code the app's own shortcuts and drops call. Defaults
# shown in --help are the ones that code uses when a key is absent; nothing
# here injects a default unless a comment says so.

def _need_colour(doc, name: str):
    import numpy as np
    a = np.asarray(doc.image)
    if not (a.ndim == 3 and a.shape[2] >= 3):
        raise ToolFailed(f"{name} needs a colour image; this one is mono")


def _check_points(v, what: str, lo=(0.0, 0.0), hi=(1.0, 1.0)):
    if v is None:
        return
    ok = isinstance(v, (list, tuple)) and len(v) >= 2 and all(
        isinstance(q, (list, tuple)) and len(q) == 2
        and all(isinstance(c, (int, float)) for c in q) for q in v)
    if not ok:
        raise ToolFailed(f"{what}: expected a list of [x, y] pairs, e.g. [[0,0],[0.5,0.6],[1,1]]")
    for x, y in v:
        if not (lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]):
            raise ToolFailed(f"{what}: point [{x}, {y}] is outside "
                             f"[{lo[0]:g}..{hi[0]:g}] x [{lo[1]:g}..{hi[1]:g}]")


# ── Stretches ───────────────────────────────────────────────────────────────

def _run_stat_stretch(doc, p, ctx):
    lm = p.get("luma_mode")
    if lm is not None:
        from setiastro.saspro.luminancerecombine import LUMA_PROFILES
        if lm not in LUMA_PROFILES:
            # The dialog would otherwise keep the user's saved mode in silence.
            raise ToolFailed(f"Statistical Stretch: unknown luma mode {lm!r}; one of "
                             + ", ".join(sorted(LUMA_PROFILES)))
    from setiastro.saspro.stat_stretch import apply_stat_stretch_preset_to_doc
    apply_stat_stretch_preset_to_doc(ctx.main, doc, p)


_register(Tool(
    "stat_stretch", "Statistical Stretch",
    "stretch to a target background median, with optional highlight compression.",
    _run_stat_stretch, [
        Param("target_median", "float", 0.25, "Background median after the stretch.", min=0.001, max=0.99),
        Param("linked", "bool", False, "Colour: stretch the channels together (true) or each on its own (false)."),
        Param("normalize", "bool", False, "Scale the result so its maximum is 1."),
        Param("blackpoint_sigma", "float", 5.0, "Black point at the median minus this many noise sigmas.",
              min=0.5, max=10.0),
        Param("no_black_clip", "bool", False, "Use the image minimum as the black point, clipping nothing."),
        Param("hdr_compress", "bool", False, "Compress the highlights."),
        Param("hdr_amount", "float", 0.15, "Highlight compression strength.", min=0.0, max=1.0),
        Param("hdr_knee", "float", 0.75, "Level where highlight compression starts.", min=0.10, max=0.95),
        Param("luma_only", "bool", False, "Colour: stretch luminance only, keeping colour ratios."),
        Param("luma_mode", "str", "rec709", "Luminance weights for --luma-only: rec709, rec601, rec2020, "
              "equal, max, median, snr, or a sensor profile."),
        Param("luma_blend", "float", 0.6, "With --luma-only: 0 = linked stretch, 1 = pure luminance.",
              min=0.0, max=1.0),
        Param("apply_curves", "bool", False, "Add a curves boost after the stretch."),
        Param("curves_boost", "float", 0.2, "Curves boost strength.", min=0.0, max=1.0),
    ], aliases=["statistical_stretch", "statstretch", "stretch"]))


def _run_star_stretch(doc, p, ctx):
    from setiastro.saspro.star_stretch import apply_star_stretch_preset_to_doc
    apply_star_stretch_preset_to_doc(doc, p)


_register(Tool(
    "star_stretch", "Star Stretch",
    "strong non-linear stretch suited to star-only images.",
    _run_star_stretch, [
        Param("stretch_factor", "float", 5.0, "Stretch strength (0 = none).", min=0.0, max=8.0),
        Param("color_boost", "float", 1.0, "Colour saturation (1 = unchanged).", min=0.0, max=2.0),
        Param("scnr_green", "bool", False, "Remove a green cast (green limited to the mean of red and blue)."),
    ], aliases=["starstretch"]))


def _run_levels(doc, p, ctx):
    b, w = float(p.get("black", 0.0)), float(p.get("white", 1.0))
    if w <= b:
        raise ToolFailed(f"Histogram Transform: the white point ({w:g}) must be above the black point ({b:g})")
    from setiastro.saspro.levels_preset import apply_levels_via_preset
    apply_levels_via_preset(None, doc, p)


_register(Tool(
    "levels", "Histogram Transform",
    "black point, midtone and white point (Histogram Transform Pro).",
    _run_levels, [
        Param("black", "float", 0.0, "Shadows clipping point.", min=0.0, max=1.0),
        Param("mid", "float", 0.5, "Midtones balance, applied after the black/white rescale. Below 0.5 "
              "brightens, above darkens.", min=0.0001, max=0.9999),
        Param("white", "float", 1.0, "Highlights clipping point.", min=0.0, max=1.0),
        Param("channel", "choice", "L", "L = luminance (keeps colour ratios), K = every channel, or one of "
              "R, G, B. Ignored for mono.", choices=["L", "K", "R", "G", "B"]),
    ], aliases=["histogram_transform", "histogram_transform_pro", "htp", "histogram"]))


_CURVE_MODES = {"k": "k", "r": "r", "g": "g", "b": "b", "lab-l": "lab_l", "lab-a": "lab_a",
                "lab-b": "lab_b", "chroma": "chroma", "saturation": "sat"}
_CURVE_SHAPES = {"linear", "none", "id", "s", "s_mild", "s_med", "s-curve", "s_strong", "lift_shadows",
                 "crush_shadows", "fade_blacks", "rolloff_highlights", "flatten"}


def _run_curves(doc, p, ctx):
    p = dict(p)
    m = p.get("mode")
    if isinstance(m, str) and m.strip().lower() in _CURVE_MODES:
        p["mode"] = _CURVE_MODES[m.strip().lower()]
    shape = p.get("shape")
    if shape is not None and str(shape).lower() not in _CURVE_SHAPES:
        raise ToolFailed(f"Curves: unknown shape {shape!r}; one of " + ", ".join(sorted(_CURVE_SHAPES)))
    _check_points(p.get("points_norm"), "Curves --points")
    from setiastro.saspro.curves_preset import apply_curves_via_preset
    apply_curves_via_preset(None, doc, p)


_register(Tool(
    "curves", "Curves",
    "a tone curve from a built-in shape or your own points.",
    _run_curves, [
        Param("mode", "str", "k", "Channel the curve acts on: k (every channel), r, g, b, lab-l, lab-a, "
              "lab-b, chroma or saturation. Mono images take the curve whatever this says."),
        Param("shape", "str", "linear", "Built-in curve: linear, s_mild, s_med, s_strong, lift_shadows, "
              "crush_shadows, fade_blacks, rolloff_highlights, flatten."),
        Param("amount", "float", 0.5, "Strength of --shape.", min=0.0, max=1.0),
        Param("points_norm", "json", "none", "Your own curve instead of --shape: [[x, y], ...] in 0..1; "
              "the end points are added if missing.", flag="--points"),
    ], aliases=["curve"],
    notes="A multi-channel curve saved by the Curves tool (its preset has \"kind\": \"curves_multi\") "
          "can be given whole with --preset; --mode/--shape/--points are then ignored."))


def _run_pedestal(doc, p, ctx):
    from setiastro.saspro.pedestal import remove_pedestal
    remove_pedestal(ctx.main, target_doc=doc, preset=p)


_register(Tool(
    "pedestal", "Remove Pedestal",
    "subtract each channel's floor so the darkest pixels sit at zero.",
    _run_pedestal, [
        Param("mode", "choice", "min", "What to subtract: min (the minimum), percentile (a low "
              "percentile, ignoring outliers) or first_nonzero (the smallest value above zero).",
              choices=["min", "percentile", "first_nonzero"]),
        Param("percentile", "float", 0.1, "For --mode percentile: 0.1 means the 0.1th percentile.",
              min=0.0, max=50.0),
        Param("per_channel", "bool", True, "Each channel its own value (true) or one value for all (false)."),
    ], aliases=["remove_pedestal", "pedestal_removal"]))


# ── Background ──────────────────────────────────────────────────────────────

def _run_abe(doc, p, ctx):
    pts = p.get("manual_points")
    if pts is not None:
        import numpy as np
        h, w = np.asarray(doc.image).shape[:2]
        _check_points(pts, "ABE --points", hi=(float(w - 1), float(h - 1)))
    from setiastro.saspro.abe_preset import apply_abe_via_preset
    # The background model, when asked for, arrives as a new document through
    # main.docman.open_array; the run writes it beside the output.
    apply_abe_via_preset(ctx.main, doc, p)


_register(Tool(
    "abe", "Automatic Background Extraction",
    "fit and remove a smooth background gradient.",
    _run_abe, [
        Param("degree", "int", 2, "Polynomial degree of the background model (0 = RBF surface only).",
              min=0, max=6),
        Param("samples", "int", 120, "Background sample points.", min=20, max=100000),
        Param("downsample", "int", 6, "Fit at 1/N resolution.", min=1, max=64),
        Param("patch", "int", 15, "Sample patch size in pixels.", min=5, max=151),
        Param("rbf", "bool", True, "Add an RBF surface for local structure."),
        Param("rbf_smooth", "float", 1.0, "RBF smoothing.", min=0.0),
        Param("correction_mode", "choice", "subtract", "subtract for additive gradients (light pollution), "
              "divide for multiplicative ones (vignetting).", choices=["subtract", "divide"]),
        Param("make_background_doc", "bool", False, "Also write the background model, as "
              "<output name>_<model name>.<ext>.", flag="--save-background"),
        Param("manual_points", "json", "none", "Sample at these [[x, y], ...] image pixels instead of "
              "choosing points automatically.", flag="--points"),
    ], aliases=["automatic_background_extraction", "background_extraction"],
    notes="The multi-scale gradient option (preset key \"multiscale\") can be given through --preset."))


# ── Colour ──────────────────────────────────────────────────────────────────

def _run_satchroma(doc, p, ctx):
    _need_colour(doc, "Saturation / Chroma")
    p = dict(p)
    m = str(p.get("mode", 0)).strip().lower()
    modes = {"0": 0, "saturation": 0, "hsv": 0, "1": 1, "chroma": 1, "lab": 1}
    if m not in modes:
        raise ToolFailed(f"Saturation / Chroma: --mode must be saturation or chroma, got {p.get('mode')!r}")
    p["mode"] = modes[m]
    _check_points(p.get("points"), "Saturation / Chroma --points", hi=(1.0, 3.0))
    from setiastro.saspro.satchroma_tool import apply_satchroma_headless
    if not apply_satchroma_headless(doc, p, None):
        raise ToolFailed("Saturation / Chroma failed (its error is printed above)")


_register(Tool(
    "satchroma", "Saturation / Chroma",
    "hue-selective saturation (HSV) or chroma (Lab) adjustment.",
    _run_satchroma, [
        Param("mode", "str", "saturation", "saturation (HSV) or chroma (Lab)."),
        Param("strength", "float", 1.0, "Scales the whole curve (1 = as given).", min=0.0, max=3.0),
        Param("points", "json", "flat", "Hue curve [[hue, multiplier], ...]: hue 0..1 (0 red, 1/6 yellow, "
              "1/3 green, 1/2 cyan, 2/3 blue, 5/6 magenta, 1 red), multiplier 0..3 (1 = unchanged). "
              "Default: flat at 1."),
    ], aliases=["saturation", "saturation_chroma", "sat_chroma"],
    notes="Mono images are refused: there is no colour to adjust."))


def _run_background_neutral(doc, p, ctx):
    _need_colour(doc, "Background Neutralization")
    if str(p.get("mode") or "auto").lower() == "rect" and p.get("rect_norm") is None:
        raise ToolFailed("Background Neutralization: --mode rect needs --rect [x, y, width, height]")
    r = p.get("rect_norm")
    if r is not None:
        if not (isinstance(r, (list, tuple)) and len(r) == 4
                and all(isinstance(c, (int, float)) and 0.0 <= c <= 1.0 for c in r)):
            raise ToolFailed("Background Neutralization: --rect must be [x, y, width, height], each 0..1")
    from setiastro.saspro.backgroundneutral import apply_background_neutral_to_doc
    apply_background_neutral_to_doc(doc, p)


_register(Tool(
    "background_neutral", "Background Neutralization",
    "make the sky background neutral grey.",
    _run_background_neutral, [
        Param("mode", "choice", "auto", "auto finds a dark background box itself; rect uses --rect.",
              choices=["auto", "rect"]),
        Param("rect_norm", "json", "none", "Background-only box for --mode rect: [x, y, width, height] as "
              "fractions 0..1 of the image.", flag="--rect"),
    ], aliases=["bn", "background_neutralization", "backgroundneutral"]))


def _run_white_balance(doc, p, ctx):
    _need_colour(doc, "White Balance")
    doc.metadata.pop("white_balance_fallback", None)
    from setiastro.saspro.whitebalance import apply_white_balance_to_doc
    apply_white_balance_to_doc(doc, p)
    note = (doc.metadata or {}).pop("white_balance_fallback", None)
    if note:
        ctx.warn(f"White Balance: {note}")


_register(Tool(
    "white_balance", "White Balance",
    "star-based (default), gray-world or manual white balance.",
    _run_white_balance, [
        Param("mode", "choice", "star", "star: make the median star neutral (the background is neutralised "
              "too); auto: gray world; manual: --r-gain/--g-gain/--b-gain.",
              choices=["star", "auto", "manual"]),
        Param("auto_sigma", "bool", True, "Star mode: choose the detection threshold automatically."),
        Param("threshold", "float", 50.0, "Star mode: detection threshold in sigmas, used when "
              "--auto-sigma is false.", min=0.5),
        Param("auto_sigma_min_stars", "int", 1200, "Star mode: stars the automatic threshold aims for.",
              min=1),
        Param("use_color_matrix", "bool", False, "Star mode: also fit a 3x3 colour matrix to the stars."),
        Param("matrix_strength", "float", 0.7, "How much of the colour matrix to apply.", min=0.0, max=1.0),
        Param("r_gain", "float", 1.0, "Manual mode: red gain.", min=0.0),
        Param("g_gain", "float", 1.0, "Manual mode: green gain.", min=0.0),
        Param("b_gain", "float", 1.0, "Manual mode: blue gain.", min=0.0),
    ], aliases=["wb", "whitebalance", "star_white_balance"],
    notes="If star detection fails, the tool falls back to gray world and the run reports a warning."))


def _run_remove_green(doc, p, ctx):
    _need_colour(doc, "Remove Green")
    from setiastro.saspro.remove_green import apply_remove_green_preset_to_doc
    apply_remove_green_preset_to_doc(ctx.main, doc, p)


_register(Tool(
    "remove_green", "Remove Green (SCNR)",
    "limit one channel (green by default) to a neutral level set by the other two.",
    _run_remove_green, [
        Param("amount", "float", 1.0, "How much of the excess to remove.", min=0.0, max=1.0),
        Param("mode", "choice", "avg", "The neutral level: avg (mean of the other two), max (the larger) or "
              "min (the smaller).", choices=["avg", "max", "min"]),
        Param("preserve_lightness", "bool", True, "Rescale so luminance is unchanged."),
        Param("channel", "choice", "G", "Channel to limit: R, G, B, or C, M, Y.",
              choices=["R", "G", "B", "C", "M", "Y"]),
    ], aliases=["scnr", "removegreen"]))


def _run_rgb_align(doc, p, ctx):
    _need_colour(doc, "RGB Align")
    p = dict(p)
    p["new_doc"] = False          # the output file is the aligned image
    from setiastro.saspro.rgbalign import run_rgb_align_headless
    run_rgb_align_headless(ctx.main, doc, p)


_register(Tool(
    "rgb_align", "RGB Align",
    "register the red and blue channels onto green.",
    _run_rgb_align, [
        Param("model", "choice", "edge", "edge (star matching, homography), homography, affine, poly3, poly4, "
              "or planet-centroid (translation only, for planetary discs).",
              choices=["edge", "homography", "affine", "poly3", "poly4", "planet-centroid"]),
        Param("sep_sigma", "float", 3.0, "Lowest SEP sigma for --model edge. Detection steps down "
              "100, 50, 20, 10, 5, 3 and stops at 500 green stars or 100 edge matches.",
              min=0.5, max=50.0),
    ], aliases=["rgbalign", "channel_align"]))


# ── Detail and contrast ─────────────────────────────────────────────────────

def _run_clahe(doc, p, ctx):
    p = dict(p)
    if "precision" in p:
        p["precision"] = int(p["precision"])
    from setiastro.saspro.clahe import apply_clahe_to_doc
    apply_clahe_to_doc(doc, p)


_register(Tool(
    "clahe", "CLAHE",
    "contrast-limited adaptive histogram equalisation.",
    _run_clahe, [
        Param("clip_limit", "float", 2.0, "Contrast limit.", min=0.1, max=20.0),
        Param("tile_px", "int", "not set", "Tile size in pixels; when given, overrides --tile.",
              min=8, max=4096),
        Param("tile", "int", 8, "Tiles per side, when --tile-px is not given.", min=2, max=128),
        Param("blend", "float", 1.0, "Mix with the original (1 = full effect).", min=0.0, max=1.0),
        Param("precision", "choice", "16", "Working precision in bits: 16 or 8.", choices=["16", "8"]),
    ]))


def _run_wavescale_hdr(doc, p, ctx):
    # The app's own runner drives the dialog through a thread and a modal
    # loop; this is the same computation with neither.
    from setiastro.saspro.wavescale_hdr_preset import apply_wavescale_hdr_to_doc
    apply_wavescale_hdr_to_doc(doc, p, mask_out=ctx.masks)


_register(Tool(
    "wavescale_hdr", "WaveScale HDR",
    "compress bright structure across wavelet scales (HDR).",
    _run_wavescale_hdr, [
        Param("n_scales", "int", 5, "Number of wavelet scales.", min=2, max=10),
        Param("compression_factor", "float", 1.5, "Coarse-scale compression.", min=0.1, max=5.0),
        Param("mask_gamma", "float", 5.0, "Gamma on the luminance mask; higher confines the effect to the "
              "brightest areas.", min=0.1, max=10.0),
        Param("amount", "float", 1.0, "Mix with the original (1 = full effect).", min=0.0, max=1.0),
    ], aliases=["hdr", "wavescalehdr"],
    mask="the luminance mask the HDR blend used (the app's HDR Mask window)"))


def _run_wavescale_de(doc, p, ctx):
    from setiastro.saspro.wavescalede_preset import run_wavescalede_via_preset
    run_wavescalede_via_preset(ctx.main, p, target_doc=doc, mask_out=ctx.masks)


_register(Tool(
    "wavescale_dark_enhance", "WaveScale Dark Enhancer",
    "bring out dark structure (dust lanes, dark nebulae) across wavelet scales.",
    _run_wavescale_de, [
        Param("n_scales", "int", 6, "Number of wavelet scales.", min=2, max=10),
        Param("boost_factor", "float", 5.0, "Dark-structure boost.", min=0.1, max=10.0),
        Param("mask_gamma", "float", 1.0, "Gamma on the darkness mask.", min=0.1, max=10.0),
        Param("iterations", "int", 2, "Enhancement passes.", min=1, max=10),
        Param("amount", "float", 1.0, "Mix with the original (1 = full effect).", min=0.0, max=1.0),
    ], aliases=["dark_enhance", "wavescale_de", "wavescalede", "dse", "wavescaledarkenhance"],
    mask="the darkness mask the enhancement used (the app's mask window; with several "
         "iterations, the last pass's)"))


def _run_texture_clarity(doc, p, ctx):
    amounts = [float(p.get(k, 0.0)) for k in ("t_amt", "u_amt", "c_amt")]
    if not any(amounts):
        ctx.warn("Texture & Clarity: --texture, --unsharp and --clarity are all 0, so nothing changes")
    from setiastro.saspro.texture_clarity import texture_clarity_headless
    texture_clarity_headless(
        doc,
        texture_amount=p.get("t_amt", 0.0),
        texture_radius=p.get("t_rad", 1.0),
        unsharp_amount=p.get("u_amt", 0.0),
        unsharp_radius=p.get("u_rad", 2.0),
        unsharp_threshold=p.get("u_thr", 0.0),
        clarity_amount=p.get("c_amt", 0.0),
        # 3.0 is the tool's own default (dialog and preset editor); the drop
        # dispatcher's 1.0 for a missing key looks like a slip.
        clarity_radius=p.get("c_rad", 3.0),
        mask_strength=p.get("mask_str", 1.0),
    )


_register(Tool(
    "texture_clarity", "Texture & Clarity",
    "texture (band-pass), unsharp mask and clarity (local contrast), weighted to the midtones.",
    _run_texture_clarity, [
        Param("t_amt", "float", 0.0, "Texture amount (negative softens).", min=-1.0, max=1.0, flag="--texture"),
        Param("t_rad", "float", 1.0, "Texture radius.", min=0.1, max=10.0, flag="--texture-radius"),
        Param("u_amt", "float", 0.0, "Unsharp mask amount.", min=0.0, max=2.0, flag="--unsharp"),
        Param("u_rad", "float", 2.0, "Unsharp mask radius.", min=0.1, max=10.0, flag="--unsharp-radius"),
        Param("u_thr", "float", 0.0, "Unsharp mask threshold.", min=0.0, max=0.1, flag="--unsharp-threshold"),
        Param("c_amt", "float", 0.0, "Clarity amount (negative softens).", min=-1.0, max=1.0, flag="--clarity"),
        Param("c_rad", "float", 3.0, "Clarity radius.", min=0.1, max=10.0, flag="--clarity-radius"),
        Param("mask_str", "float", 1.0, "Midtone weighting: 1 = midtones only, 0 = everywhere.",
              min=0.0, max=1.0, flag="--midtone-mask"),
    ], aliases=["textureclarity", "texture", "clarity"]))


def _run_morphology(doc, p, ctx):
    from setiastro.saspro.morphology import apply_morphology_to_doc
    apply_morphology_to_doc(doc, p)


_register(Tool(
    "morphology", "Morphological Operations",
    "erosion, dilation, opening or closing with an elliptical kernel.",
    _run_morphology, [
        Param("operation", "choice", "erosion", "erosion, dilation, opening or closing.",
              choices=["erosion", "dilation", "opening", "closing"]),
        Param("kernel", "int", 3, "Kernel diameter in pixels (even sizes round up).", min=1, max=31),
        Param("iterations", "int", 1, "Times the operation is repeated.", min=1, max=10),
    ], aliases=["morph", "morphological"]))


def _run_halo_b_gon(doc, p, ctx):
    from setiastro.saspro.halobgon import apply_halo_b_gon_to_doc
    apply_halo_b_gon_to_doc(None, doc, p)


_register(Tool(
    "halo_b_gon", "Halo-B-Gon",
    "reduce star halos.",
    _run_halo_b_gon, [
        Param("reduction", "int", 0, "Strength: 0 extra low, 1 low, 2 medium, 3 high.", min=0, max=3),
        Param("linear", "bool", False, "The image is linear (not yet stretched)."),
    ], aliases=["halobgon", "halo_reduction", "halo"]))


# ── Repair ──────────────────────────────────────────────────────────────────

def _run_banding_blaster(doc, p, ctx):
    from setiastro.saspro.banding_blaster import apply_banding_blaster_preset_to_doc
    apply_banding_blaster_preset_to_doc(ctx.main, doc, p)


_register(Tool(
    "banding_blaster", "Banding Blaster",
    "reduce row (horizontal) or column (vertical) banding.",
    _run_banding_blaster, [
        Param("direction", "choice", "horizontal", "horizontal: row banding (Canon DSLRs); "
              "vertical: column banding.", choices=["horizontal", "vertical"]),
        Param("amount", "float", 1.0, "Strength; 1 brings each row's median to the image median.",
              min=0.0, max=4.0),
        Param("protect_highlights", "bool", True, "Leave stars and bright cores out of the row medians."),
        Param("sigma", "float", 1.0, "With --protect-highlights: how far above the median a pixel is "
              "left out, in robust sigmas. Lower leaves out more.", min=0.01, max=5.0),
    ], aliases=["banding", "banding_reduction"]))


# ── Examples shown at the foot of each tool's --help ────────────────────────
_P = "setiastrosuitepro process "
_EXAMPLES = {
    "stat_stretch": [
        _P + "stat_stretch -i stack.fits -o stretched.fits --target-median 0.2",
        _P + "stat_stretch -i stack.fits -o stretched.tif --linked true --hdr-compress true --hdr-amount 0.3",
    ],
    "star_stretch": [
        _P + "star_stretch -i stars.tif -o stars_stretched.tif --stretch-factor 4 --scnr-green true",
    ],
    "levels": [
        _P + "levels -i in.tif -o out.tif --black 0.02 --mid 0.4 --white 0.98",
        _P + "levels -i in.tif -o out.tif --mid 0.45 --channel K",
    ],
    "curves": [
        _P + "curves -i in.tif -o out.tif --shape s_med --amount 0.6",
        _P + 'curves -i in.tif -o out.tif --mode saturation --points "[[0,0],[0.5,0.6],[1,1]]"',
    ],
    "pedestal": [
        _P + "pedestal -i stack.fits -o out.fits",
        _P + "pedestal -i stack.fits -o out.fits --mode percentile --percentile 0.1",
    ],
    "abe": [
        _P + "abe -i stack.fits -o abe.fits --degree 3 --samples 200",
        _P + "abe -i stack.fits -o abe.fits --correction-mode divide --save-background true",
    ],
    "satchroma": [
        _P + "satchroma -i in.tif -o out.tif --strength 1.3",
        # boost reds and blues, leave greens alone
        _P + "satchroma -i in.tif -o out.tif --points \"[[0,1.4],[0.33,1],[0.66,1.4],[1,1.4]]\"",
    ],
    "background_neutral": [
        _P + "background_neutral -i stack.fits -o bn.fits",
        _P + "background_neutral -i stack.fits -o bn.fits --mode rect --rect \"[0.05,0.05,0.1,0.1]\"",
    ],
    "white_balance": [
        _P + "white_balance -i stack.fits -o wb.fits",
        _P + "white_balance -i stack.fits -o wb.fits --mode manual --r-gain 0.95 --b-gain 1.08",
    ],
    "remove_green": [
        _P + "remove_green -i in.fits -o out.fits",
        _P + "remove_green -i in.fits -o out.fits --mode max --amount 0.8",
    ],
    "rgb_align": [
        _P + "rgb_align -i stack.fits -o aligned.fits",
        _P + "rgb_align -i jupiter.tif -o aligned.tif --model planet-centroid",
    ],
    "clahe": [
        _P + "clahe -i in.tif -o out.tif --clip-limit 2.5 --tile-px 128",
        _P + "clahe -i in.tif -o out.tif --clip-limit 1.5 --blend 0.5",
    ],
    "wavescale_hdr": [
        _P + "wavescale_hdr -i in.tif -o out.tif --n-scales 6 --compression-factor 2",
        _P + "wavescale_hdr -i in.tif -o out.tif --save-mask --json      (mask: out_mask.tif)",
    ],
    "wavescale_dark_enhance": [
        _P + "wavescale_dark_enhance -i in.tif -o out.tif --boost-factor 4 --iterations 2",
        _P + "wavescale_dark_enhance -i in.fits -o out.fits --save-mask dark_mask.tif",
    ],
    "texture_clarity": [
        _P + "texture_clarity -i in.tif -o out.tif --texture 0.3 --clarity 0.25",
        _P + "texture_clarity -i in.tif -o out.tif --unsharp 0.8 --unsharp-radius 1.5 --midtone-mask 0",
    ],
    "morphology": [
        _P + "morphology -i stars.tif -o smaller.tif --operation erosion --kernel 3",
    ],
    "halo_b_gon": [
        _P + "halo_b_gon -i in.tif -o out.tif --reduction 1",
    ],
    "banding_blaster": [
        _P + "banding_blaster -i stack.fits -o out.fits",
        _P + "banding_blaster -i stack.fits -o out.fits --direction vertical --amount 0.8 --sigma 0.5",
    ],
}
for _id, _ex in _EXAMPLES.items():
    TOOLS[_id].examples = _ex
del _P, _id, _ex


if __name__ == "__main__":
    raise SystemExit(main())
