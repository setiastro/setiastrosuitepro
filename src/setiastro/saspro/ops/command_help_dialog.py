# ops/command_help_dialog.py
from __future__ import annotations
from dataclasses import asdict
from typing import Dict, List, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLineEdit, QTextBrowser, QPushButton, QLabel, QSplitter, QApplication
)

from setiastro.saspro.ops.commands import COMMAND_REGISTRY, CommandSpec, PresetSpec

# ---------------------------------------------------------------------
# Available script libraries (bundled/installed)
# ---------------------------------------------------------------------
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class LibSpec:
    import_name: str                 # what to `import ...`
    pip_name: Optional[str] = None   # if different from import
    note: str = ""                   # short human note
    platforms: str = "all"           # "all" / "non-darwin" / "darwin" / etc.

AVAILABLE_LIBRARIES: Dict[str, List[LibSpec]] = {
    "numeric & scientific": [
        LibSpec("numpy"),
        LibSpec("scipy"),
        LibSpec("pywt", pip_name="pywavelets", note="wavelets"),
        LibSpec("matplotlib"),
        LibSpec("plotly"),
        LibSpec("exifread"),
        LibSpec("lightkurve"),
        LibSpec("oktopus"),
    ],
    "tables & I/O": [
        LibSpec("pandas"),
        LibSpec("tifffile"),
        LibSpec("PIL", pip_name="Pillow", note="use `from PIL import Image`"),
    ],
    "networking & web": [
        LibSpec("requests"),
        LibSpec("astroquery"),
    ],
    "compression": [
        LibSpec("lz4"),
        LibSpec("zstandard", note="import as `import zstandard as zstd`"),
    ],
    "astronomy": [
        LibSpec("astropy"),
        LibSpec("photutils"),
        LibSpec("astroalign"),
        LibSpec("sep"),
        LibSpec("reproject"),
        LibSpec("tzlocal"),
    ],
    "performance": [
        LibSpec("numba"),
    ],
    "neural net": [
        LibSpec("onnx"),
        LibSpec("onnxruntime"),
        LibSpec(
            "onnxruntime_directml",
            pip_name="onnxruntime-directml",
            note="Windows DirectML backend",
            platforms="non-darwin",
        ),
    ],
    "image formats & camera raw": [
        LibSpec("xisf"),
        LibSpec("rawpy"),
    ],
    "utilities": [
        LibSpec("pytz"),
    ],
    "GUI": [
        LibSpec("PyQt6"),
        LibSpec("pyqtgraph"),
    ],
    "system": [
        LibSpec("psutil"),
    ],
    "computer vision": [
        LibSpec("cv2", pip_name="opencv-python", note="OpenCV"),
    ],
}

def render_available_libs_markdown() -> str:
    lines: List[str] = []
    lines.append("## Available Script Libraries")
    lines.append("These libraries are bundled/installed and importable from SAS scripts.\n")

    for group, specs in AVAILABLE_LIBRARIES.items():
        lines.append(f"### {group}")
        for s in specs:
            imp = f"`import {s.import_name}`"
            pip = (
                f"(pip: `{s.pip_name}`)"
                if s.pip_name and s.pip_name != s.import_name
                else ""
            )
            plat = "" if s.platforms == "all" else f"*[{s.platforms} only]*"
            note = f" — {s.note}" if s.note else ""
            tail = " ".join(t for t in (pip, plat) if t)
            tail = f" {tail}" if tail else ""
            lines.append(f"- {imp}{tail}{note}")
        lines.append("")

    return "\n".join(lines)


def _spec_display_title(spec: CommandSpec) -> str:
    # prefer spec.name; fallback to id
    return spec.name or spec.title or spec.id


def _preset_line(ps: PresetSpec) -> str:
    parts = [f"**{ps.key}** ({ps.type})"]
    if ps.default is not None:
        parts.append(f"default={ps.default!r}")
    if ps.min is not None or ps.max is not None:
        parts.append(f"range=[{ps.min},{ps.max}]")
    if ps.enum:
        parts.append(f"enum={ps.enum}")
    if not ps.optional:
        parts.append("**required**")

    head = " — ".join(parts)

    # For markdown: use two spaces + newline to force a line-break under the bullet
    # and indent the description a bit. No <br> or weird unicode spaces.
    if ps.desc:
        return f"{head}  \n    {ps.desc}"
    return head



def _supports_line(spec: CommandSpec) -> str:
    def yn(b): return "✅" if b else "❌"
    return (
        f"Mono {yn(spec.supports_mono)} · "
        f"RGB {yn(spec.supports_rgb)} · "
        f"Linear {yn(spec.supports_linear)} · "
        f"Nonlinear {yn(spec.supports_nonlinear)}"
    )


def render_spec_markdown(cid: str, spec: CommandSpec) -> str:
    call_style = spec.call_style or "ctx.run_command"
    notes = spec.notes or spec.summary or ""
    name = _spec_display_title(spec)

    md = [f"## {name}  (`{cid}`)"]
    md.append(f"**Group:** {spec.group}")
    md.append(f"**Call style:** `{call_style}('{cid}', preset_dict)`")
    if spec.import_path or spec.callable_name:
        md.append(f"**Headless callable:** `{spec.import_path}.{spec.callable_name}`")
    if spec.headless_method:
        md.append(f"**Headless method:** `main_window.{spec.headless_method}(doc, preset)`")
    if spec.ui_method:
        md.append(f"**UI method:** `main_window.{spec.ui_method}(... )`")
    md.append(f"**Supports:** {_supports_line(spec)}")

    if spec.aliases:
        md.append(f"**Aliases:** {', '.join(spec.aliases)}")

    if notes:
        md.append("")
        md.append(notes)

    if spec.presets:
        md.append("")
        md.append("### Presets")
        for ps in spec.presets:
            md.append(f"- {_preset_line(ps)}")
    else:
        md.append("")
        md.append("### Presets")
        md.append("- *(none)*")

    if spec.examples:
        md.append("")
        md.append("### Examples")
        for ex in spec.examples:
            md.append("```python")
            md.append(ex)
            md.append("```")

    return "\n".join(md)

# ---------------------------------------------------------------------
# Script Context (ctx) Help
# ---------------------------------------------------------------------

def render_scripting_quickstart_markdown() -> str:
    return "\n".join([
        "## Scripting Quickstart",
        "User scripts live in your SASpro scripts folder and appear in the Scripts menu.",
        "",
        "### Required entrypoint",
        "Your script must define one of:",
        "- `def run(ctx):` *(preferred)*",
        "- `def main(ctx):` *(fallback)*",
        "",
        "### Minimal example",
        "```python",
        "SCRIPT_NAME  = \"My First Script\"",
        "SCRIPT_GROUP = \"User\"",
        "",
        "def run(ctx):",
        "    ctx.log(\"Hello world\")",
        "```",
        "",
        "### Typical active-image workflow",
        "This operates on the **current active view** (Undo/ROI-safe):",
        "```python",
        "import numpy as np",
        "",
        "def run(ctx):",
        "    img = ctx.get_image()",
        "    if img is None:",
        "        ctx.log(\"No active image\")",
        "        return",
        "",
        "    f = img.astype(np.float32)",
        "    f = np.clip(f, 0.0, 1.0)",
        "",
        "    out = 1.0 - f",
        "    ctx.set_image(out, step_name=\"Invert via Script\")",
        "```",
        "",
        "### File-based workflow (no documents opened)",
        "Use this when you want to process images directly on disk without opening subwindows:",
        "```python",
        "import numpy as np",
        "",
        "def run(ctx):",
        "    img, hdr, bit, mono = ctx.load_image(r\"D:/data/a.fits\")",
        "    out = np.clip(img, 0, 1) ** 0.8",
        "    ctx.save_image(out, r\"D:/data/a_gamma.fits\",",
        "                   original_format=\"fits\",",
        "                   original_header=hdr,",
        "                   is_mono=mono)",
        "```",
        "",
        "### Running built-in operations",
        "All headless/scriptable tools go through `ctx.run_command(...)`.",
        "```python",
        "def run(ctx):",
        "    ctx.run_command('stat_stretch', {'target_median': 0.25})",
        "    ctx.run_command('remove_green', {'amount': 0.7})",
        "```",
        "",
        "### Running Function Bundles from scripts",
        "Function Bundles are saved sequences of commands you manage in the **Function Bundles** dialog.",
        "You can trigger them from scripts using the `function_bundle` command:",
        "",
        "```python",
        "def run(ctx):",
        "    cfg = {",
        "        'bundle_name': 'PreProcess',  # name as shown in Function Bundles dialog",
        "        'inherit_target': True,       # forward the active view / ROI into each step",
        "    }",
        "    ctx.run_command('function_bundle', cfg)",
        "```",
        "",
        "- `bundle_name` (or `name`) must match an existing Function Bundle.",
        "- `inherit_target=True` makes each step run on the same target (active view or ROI).",
        "- Optional: `targets='all_open'` to apply the bundle to **every** open image,",
        "  or `targets=[doc_id1, doc_id2, ...]` to target specific docs (same semantics as drag-and-drop).",
        "- Internally this behaves **exactly like** dropping a Function Bundle chip onto a view.",
        "- The bundled **Run Function Bundle…** script shows a complete picker dialog and then",
        "  calls `ctx.run_command('function_bundle', cfg)` under the hood.",
        "",
        "### Tips",
        "- Scripts operate on the **active view** unless you explicitly target others.",
        "- `ctx.set_image(...)` routes through DocManager so undo + ROI previews stay correct.",
        "- File I/O helpers do **not** open documents unless you call `ctx.open_new_document(...)`.",
        "- Use `ctx.log(...)` to write to the SASpro log and Script Editor output.",
    ])

def render_ctx_api_markdown() -> str:
    return "\n".join([
        "## Script Context (`ctx`) API",
        "Your script receives a `ScriptContext` instance named `ctx`.",
        "",
        "### Logging",
        "- `ctx.log(msg: str)` — write to the SASpro log/output.",
        "```python",
        "ctx.log(\"Starting my script\")",
        "```",
        "",
        "### Main window access",
        "- `ctx.main_window()` — returns the main SASpro window.",
        "```python",
        "mw = ctx.main_window()",
        "mw.update_status(\"hi from script\")",
        "```",
        "",
        "### Active view / document",
        "- `ctx.active_subwindow()` — current QMdiSubWindow or None.",
        "- `ctx.active_view()` — active view widget or None.",
        "- `ctx.active_document()` — active document or None.",
        "- `ctx.base_document()` — base doc for ROI tabbed views.",
        "",
        "### Image data (active view)",
        "- `ctx.get_image()` — returns `doc.image` (usually float32 [0,1]).",
        "- `ctx.set_image(img, step_name='Script')` — commits new image through DocManager (Undo/ROI safe).",
        "",
        "```python",
        "img = ctx.get_image()",
        "if img is not None:",
        "    ctx.set_image(img*0.9, step_name=\"Dim\")",
        "```",
        "",
        "### File I/O (canonical SASpro routes)",
        "These operate on disk **without** opening a document or subwindow.",
        "",
        "- `ctx.load_image(path, return_metadata=False, max_retries=3, wait_seconds=3)`",
        "  - Returns `(img, original_header, bit_depth, is_mono)` by default.",
        "- `ctx.save_image(img, path, original_format=None, bit_depth=None, original_header=None, is_mono=False, image_meta=None, file_meta=None)`",
        "  - `original_format` inferred from suffix if None (fits/tiff/png/jpg/etc).",
        "",
        "Aliases:",
        "- `ctx.open_image(...)` → `ctx.load_image(...)`",
        "- `ctx.write_image(...)` → `ctx.save_image(...)`",
        "",
        "```python",
        "img, hdr, bit, mono = ctx.load_image(r\"D:/data/a.tiff\")",
        "# ...process...",
        "ctx.save_image(img, r\"D:/data/a_out.tiff\",",
        "               original_header=hdr, is_mono=mono)",
        "```",
        "",
        "### Listing and targeting other open views/documents",
        "These helpers let scripts operate on multiple images by name/title/uid.",
        "",
        "- `ctx.list_views()` — list of open views with titles, names, uids, file paths.",
        "- `ctx.list_view_names()` — just the human-visible names.",
        "- `ctx.get_document(name_or_uid)` — fetch an open base document (never ROI wrapper).",
        "- `ctx.get_image_for(name_or_uid)` — ndarray for a specific open view.",
        "- `ctx.set_image_for(name_or_uid, img, step_name='Script')` — update a specific open view.",
        "- `ctx.activate_view(name_or_uid)` — bring a view to front.",
        "",
        "```python",
        "for v in ctx.list_views():",
        "    ctx.log(f\"Open view: {v['name']}  title={v['title']}\")",
        "",
        "doc = ctx.get_document(\"Andromeda\")",
        "if doc:",
        "    ctx.set_image_for(\"Andromeda\", doc.image*1.05, step_name=\"Boost\")",
        "```",
        "",
        "### Creating a new document",
        "- `ctx.open_new_document(img, metadata=None, name=None)` — opens a new view/doc from an array.",
        "```python",
        "out = img ** 0.9",
        "ctx.open_new_document(out, name=\"Gamma\")",
        "```",
        "",
        "### Running commands",
        "- `ctx.run_command(command_id, preset=None, **kwargs)`",
        "",
        "Preferred usage:",
        "```python",
        "ctx.run_command('ghs', {'alpha':1.5, 'beta':1.0})",
        "ctx.run_command('abe', {'degree':2, 'samples':150})",
        "```",
        "",
        "You can also run **Function Bundles**:",
        "```python",
        "cfg = {",
        "    'bundle_name': 'PreProcess',   # Function Bundle name",
        "    'inherit_target': True,        # forward active view / ROI to each step",
        "    # optional: 'targets': 'all_open' ",
        "    # optional: 'targets': [doc_id1, doc_id2, ...],",
        "}",
        "ctx.run_command('function_bundle', cfg)",
        "```",
        "",
        "Notes:",
        "- `command_id` can be any registered id or alias (see left panel).",
        "- `preset` is a dict matching that command’s Presets section.",
        "- Most commands support masks automatically if an active mask exists.",
        "- For `function_bundle`, the command delegates to the same internal path as",
        "  dragging a Function Bundle chip onto a view, so UI and scripts stay in sync.",
        "",
        "### Environment helpers",
        "- `ctx.is_frozen()` — True if running from packaged app (PyInstaller).",
        "",
        "```python",
        "if ctx.is_frozen():",
        "    ctx.log(\"Running from packaged build\")",
        "```",
        "",
        "### Patterns you’ll use a lot",
        "**1) Guard against missing active image**",
        "```python",
        "img = ctx.get_image()",
        "if img is None:",
        "    ctx.log(\"No active image\")",
        "    return",
        "```",
        "",
        "**2) Apply a tool then tweak pixels**",
        "```python",
        "ctx.run_command('stat_stretch', {'target_median':0.25})",
        "img = ctx.get_image()",
        "ctx.set_image(img**0.9, step_name=\"Gamma tweak\")",
        "```",
        "",
        "**3) Disk batch + optional new doc**",
        "```python",
        "img, hdr, bit, mono = ctx.load_image(r\"D:/data/a.fits\")",
        "out = img * 0.95",
        "ctx.save_image(out, r\"D:/data/a_dim.fits\", original_header=hdr)",
        "# only open a view if you want to:",
        "# ctx.open_new_document(out, name=\"a_dim\")",
        "```",
    ])



class CommandHelpDialog(QDialog):
    """
    Dialog that documents all registered COMMAND_REGISTRY commands.
    """
    def __init__(self, parent=None, editor=None):
        super().__init__(parent)
        self.setWindowTitle("Command Help (Scriptable / Headless Ops)")
        self.resize(980, 620)
        self.editor = editor  # optional CodeEditor to insert examples

        root = QVBoxLayout(self)

        # ---- top search bar ----
        top = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search commands… (id, name, group, alias)")
        top.addWidget(QLabel("Search:"))
        top.addWidget(self.search, 1)
        root.addLayout(top)

        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split, 1)

        # ---- left list ----
        self.listw = QListWidget()
        split.addWidget(self.listw)

        # ---- right detail ----
        right = QVBoxLayout()
        right_wrap = QDialog()
        right_wrap.setLayout(right)

        self.detail = QTextBrowser()
        self.detail.setOpenExternalLinks(True)
        #mono = QFont("Consolas")
        #mono.setStyleHint(QFont.StyleHint.Monospace)
        #mono.setPointSize(10)
        #self.detail.setFont(mono)
        right.addWidget(self.detail, 1)

        btnrow = QHBoxLayout()
        self.btn_copy_md = QPushButton("Copy Markdown")
        self.btn_insert_ex = QPushButton("Insert First Example")
        self.btn_close = QPushButton("Close")
        btnrow.addWidget(self.btn_copy_md)
        btnrow.addWidget(self.btn_insert_ex)
        btnrow.addStretch(1)
        btnrow.addWidget(self.btn_close)
        right.addLayout(btnrow)

        split.addWidget(right_wrap)
        split.setStretchFactor(1, 2)

        # ---- wiring ----
        self.btn_close.clicked.connect(self.close)
        self.btn_copy_md.clicked.connect(self.copy_markdown)
        self.btn_insert_ex.clicked.connect(self.insert_first_example)
        self.search.textChanged.connect(self.rebuild_list)
        self.listw.currentItemChanged.connect(self.show_selected)

        self.rebuild_list()
        if self.listw.count():
            self.listw.setCurrentRow(0)

    def rebuild_list(self):
        q = (self.search.text() or "").strip().lower()

        # sort by group then name
        items: List[Tuple[str, CommandSpec]] = sorted(
            COMMAND_REGISTRY.items(),
            key=lambda kv: (kv[1].group or "", _spec_display_title(kv[1]).lower(), kv[0])
        )

        self.listw.blockSignals(True)
        self.listw.clear()

        # ---- synthetic doc item: Available Libraries ----
        synth_items = [
            ("Docs · Scripting Quickstart", "__script_quickstart__",
            "quickstart run(ctx) main(ctx) script entrypoint samples"),

            ("Docs · Script Context (ctx) API", "__ctx_help__",
            "ctx api context get_image set_image run_command log base_document "
            "list_views list_view_names get_document get_image_for set_image_for activate_view "
            "open_new_document "
            "load_image save_image open_image write_image "
            "file io disk batch fits tiff png jpg exr xisf"),

            ("Docs · Available Libraries", "__available_libs__",
            "available libraries libs imports bundled installed"),
        ]

        for title, key, searchable in synth_items:
            if (not q) or (q in searchable.lower()):
                it = QListWidgetItem(title)
                it.setData(Qt.ItemDataRole.UserRole, key)
                self.listw.addItem(it)

        for cid, spec in items:
            searchable = " ".join([
                cid,
                _spec_display_title(spec),
                spec.group or "",
                " ".join(spec.aliases or [])
            ]).lower()

            if q and q not in searchable:
                continue

            it = QListWidgetItem(f"{spec.group} · {_spec_display_title(spec)}")
            it.setData(Qt.ItemDataRole.UserRole, cid)
            self.listw.addItem(it)

        self.listw.blockSignals(False)
        if self.listw.count():
            self.listw.setCurrentRow(0)
        else:
            self.detail.setText("No matches.")

    def show_selected(self, item: QListWidgetItem | None, _prev=None):
        if item is None:
            self.detail.setText("")
            return

        cid = item.data(Qt.ItemDataRole.UserRole)

        # ---- synthetic doc rendering ----
        if cid == "__available_libs__":
            md = render_available_libs_markdown()
            self.detail.setMarkdown(md)
            return

        if cid == "__script_quickstart__":
            md = render_scripting_quickstart_markdown()
            self.detail.setMarkdown(md)
            return

        if cid == "__ctx_help__":
            md = render_ctx_api_markdown()
            self.detail.setMarkdown(md)
            return

        # ---- normal command specs ----
        spec = COMMAND_REGISTRY.get(cid)
        if spec is None:
            self.detail.setText("")
            return

        md = render_spec_markdown(cid, spec)
        # Real markdown rendering (handles headings, bullets, code fences, etc.)
        self.detail.setMarkdown(md)



    def copy_markdown(self):
        item = self.listw.currentItem()
        if item is None:
            return

        cid = item.data(Qt.ItemDataRole.UserRole)

        if cid == "__available_libs__":
            md = render_available_libs_markdown()
        elif cid == "__script_quickstart__":
            md = render_scripting_quickstart_markdown()
        elif cid == "__ctx_help__":
            md = render_ctx_api_markdown()
        else:
            spec = COMMAND_REGISTRY.get(cid)
            if spec is None:
                return
            md = render_spec_markdown(cid, spec)

        cb = QApplication.clipboard()
        cb.setText(md)




    def insert_first_example(self):
        if self.editor is None:
            return
        item = self.listw.currentItem()
        if item is None:
            return
        cid = item.data(Qt.ItemDataRole.UserRole)
        spec = COMMAND_REGISTRY.get(cid)
        if spec is None or not spec.examples:
            return
        ex = spec.examples[0]

        cur = self.editor.textCursor()
        cur.beginEditBlock()
        cur.insertText("\n\n# Example: " + _spec_display_title(spec) + "\n" + ex + "\n")
        cur.endEditBlock()
        self.editor.setTextCursor(cur)
