# ops/command_help_dialog.py
from __future__ import annotations
from dataclasses import asdict
from typing import Dict, List, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLineEdit, QTextBrowser, QPushButton, QLabel, QSplitter
)

from ops.commands import COMMAND_REGISTRY, CommandSpec, PresetSpec

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
    return " — ".join(parts) + (f"<br> {ps.desc}" if ps.desc else "")


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
        "### Typical image workflow",
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
        "### Running built-in operations",
        "All headless/scriptable tools go through `ctx.run_command(...)`.",
        "```python",
        "def run(ctx):",
        "    ctx.run_command('stat_stretch', {'target_median': 0.25})",
        "    ctx.run_command('remove_green', {'amount': 0.7})",
        "```",
        "",
        "### Tips",
        "- Scripts operate on the **active view** unless you explicitly target others.",
        "- `ctx.set_image(...)` routes through DocManager so undo + ROI previews stay correct.",
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
        "",
        "### Image data",
        "- `ctx.get_image()` — returns `doc.image` (usually float32 [0,1]).",
        "- `ctx.set_image(img, step_name='Script')` — commits new image through DocManager.",
        "",
        "```python",
        "img = ctx.get_image()",
        "if img is not None:",
        "    ctx.set_image(img*0.9, step_name=\"Dim\")",
        "```",
        "",
        "### Base document (ROI-aware)",
        "- `ctx.base_document()` — returns base doc for ROI tabbed views.",
        "",
        "Use this if you want to apply something to the parent/base frame:",
        "```python",
        "base = ctx.base_document()",
        "if base:",
        "    base.image *= 0.95",
        "```",
        "",
        "### Listing and targeting other open views/documents",
        "These helpers let scripts operate on multiple images by name (current window title), ",
        "so users can rename views and your script still finds them.",
        "",
        "- `ctx.list_image_views()` — returns a list of `(view_title, document)` for all open image views.",
        "```python",
        "views = ctx.list_image_views()",
        "for title, doc in views:",
        "    ctx.log(f\"Open view: {title}  shape={doc.image.shape}\")",
        "```",
        "",
        "- `ctx.get_document_by_title(title: str)` — returns the document matching a view title (or None).",
        "```python",
        "doc = ctx.get_document_by_title(\"andro2\")",
        "if doc:",
        "    ctx.log(\"Found doc for andro2\")",
        "```",
        "",
        "- `ctx.list_documents()` — returns all open base documents (not ROI previews).",
        "```python",
        "docs = ctx.list_documents()",
        "ctx.log(f\"Open base docs: {len(docs)}\")",
        "```",
        "",
        "### Creating a new document",
        "- `ctx.open_new_document(img, metadata=None, name=\"New Document\")` — opens a new view/doc from an array.",
        "  - `img` should be float32 and shaped like existing docs (H×W or H×W×3).",
        "  - Returns the created document.",
        "```python",
        "out = 0.5 * (img1 + img2)",
        "new_doc = ctx.open_new_document(out, name=\"Average(img1,img2)\")",
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
        "Notes:",
        "- `command_id` can be any registered id or alias (see left panel).",
        "- `preset` is a dict matching that command’s Presets section.",
        "- Most commands support masks automatically if an active mask exists.",
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
        "**3) Luminance workflow**",
        "```python",
        "# Extract L from RGB, edit it, recombine into RGB",
        "ctx.run_command('extract_luminance', {'mode':'rec709'})",
        "# (active view is now L if extract opened it)",
        "L = ctx.get_image()",
        "ctx.set_image(L*1.1, step_name=\"Boost L\")",
        "ctx.run_command('recombine_luminance', {'method':'rec709'})",
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
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(10)
        self.detail.setFont(mono)
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
            ("Docs · Scripting Quickstart", "__script_quickstart__", "quickstart run(ctx) main(ctx) script entrypoint"),
            ("Docs · Script Context (ctx) API", "__ctx_help__",
            "ctx api context get_image set_image run_command log base_document "
            "list_image_views list_documents get_document_by_title open_new_document view titles"),
            ("Docs · Available Libraries", "__available_libs__", "available libraries libs imports bundled installed"),
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
            self.detail.setHtml(md.replace("\n", "<br>"))
            return
        if cid == "__script_quickstart__":
            md = render_scripting_quickstart_markdown()
            self.detail.setHtml(md.replace("\n", "<br>"))
            return

        if cid == "__ctx_help__":
            md = render_ctx_api_markdown()
            self.detail.setHtml(md.replace("\n", "<br>"))
            return
        spec = COMMAND_REGISTRY.get(cid)
        if spec is None:
            self.detail.setText("")
            return

        md = render_spec_markdown(cid, spec)

        # QTextBrowser renders markdown-ish OK if we replace \n with <br>
        html = md.replace("\n", "<br>")
        self.detail.setHtml(html)


    def copy_markdown(self):
        item = self.listw.currentItem()
        if item is None:
            return
        cid = item.data(Qt.ItemDataRole.UserRole)

        if cid == "__available_libs__":
            self.clipboard().setText(render_available_libs_markdown())
            return

        if cid == "__script_quickstart__":
            self.clipboard().setText(render_scripting_quickstart_markdown())
            return

        if cid == "__ctx_help__":
            self.clipboard().setText(render_ctx_api_markdown())
            return

        spec = COMMAND_REGISTRY.get(cid)
        if spec is None:
            return
        md = render_spec_markdown(cid, spec)

        cb = self.clipboard()
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
