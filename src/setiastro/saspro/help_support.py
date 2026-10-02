# src/setiastro/saspro/help_support.py
"""
In-tool documentation support for SASpro.

Goal: put a small "document" button inside each tool dialog that opens a short,
task-focused HTML help page *in the app* -- so users who never visit the wiki
still get guidance right where they are.

Usage (one line per tool):

    from setiastro.saspro.help_support import make_help_button
    ...
    btn_row.addStretch(1)
    btn_row.addWidget(make_help_button("crop", self))

All help pages live in a single folder: ``src/setiastro/saspro/docs``. Add a
matching ``docs/<doc_id>.html`` and it's picked up automatically. (For frozen
builds, ship that folder as ``docs`` -- the resolver also checks
``<_MEIPASS>/docs``.)
"""
from __future__ import annotations

import os
import sys

from PyQt6.QtCore import Qt, QSize, QUrl
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QPen, QColor, QPalette, QImage
from PyQt6.QtWidgets import (
    QToolButton, QDialog, QVBoxLayout, QHBoxLayout, QTextBrowser,
    QPushButton, QWidget,
)

# doc_id -> display title. Unknown ids fall back to a title-cased id.
_DOC_TITLES: dict[str, str] = {
    "crop": "Crop Tool",
    "histogram": "Histogram Viewer",
    "pedestal": "Remove Pedestal",
    "linear_fit": "Linear Fit",
    "stat_stretch": "Statistical Stretch",
    "star_stretch": "Star Stretch",
    "levels": "Levels (Histogram Transform)",
    "curves": "Curves Editor",
    "ghs": "Hyperbolic Stretch",
    "satchroma": "Saturation / Chroma",
    "abe": "Automatic Background Extraction (ADBE)",
    "graxpert": "GraXpert",
    "cosmetic_correction": "Cosmetic Correction",
    "remove_stars": "Remove Stars",
    "add_stars": "Add Stars",
    "background_neutral": "Background Neutralization",
    "white_balance": "White Balance",
    "sfcc": "Spectral Flux Color Calibration (SFCC)",
    "sssc": "Spectrophotometric Standard Star Calibration (SSSC)",
    "remove_green": "Remove Green (SCNR)",
    "convo": "Convolution / Deconvolution",
    "luminance_recombine": "Recombine Luminance",
    "rgb_combination": "RGB Combination",
}


def _doc_title(doc_id: str) -> str:
    if doc_id in _DOC_TITLES:
        return _DOC_TITLES[doc_id]
    nice = doc_id.replace("_", " ").replace("-", " ").strip().title()
    return nice or doc_id


# ---------------------------------------------------------------------------
# doc file resolution -- one canonical folder: saspro/docs (plus _MEIPASS/docs)
# ---------------------------------------------------------------------------
def _docs_dirs() -> list[str]:
    dirs = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")]
    mp = getattr(sys, "_MEIPASS", None)
    if mp:
        dirs.append(os.path.join(mp, "docs"))
    out, seen = [], set()
    for d in dirs:
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def find_doc_html(doc_id: str) -> "str | None":
    for d in _docs_dirs():
        p = os.path.join(d, f"{doc_id}.html")
        if os.path.isfile(p):
            return p
    return None


# ---------------------------------------------------------------------------
# LaTeX math -> inline images (QTextBrowser can't run MathJax, so we pre-render
# each equation to a transparent PNG via matplotlib's mathtext and inline it).
# Authoring: block  <div class="matheq">LaTeX</div>  /  <div class="matheq-hero">…</div>
#            inline  \( LaTeX \)
# matplotlib mathtext is a LaTeX *subset*: use \leq/\geq (not \le/\ge), \frac
# (not \tfrac), no \big, no array/matrix. Matrices stay as monospace .eq blocks.
# ---------------------------------------------------------------------------
_MATH_DPI = 200          # render DPI (then shown at 1/_MATH_SS for crisp supersampling)
_MATH_FS_BLOCK = 15      # mathtext font size (pt) for block equations
_MATH_FS_INLINE = 12     # …and inline ones
_MATH_SS = 2             # supersample factor: display at natural_px / _MATH_SS
_MATH_COLOR = "#e6e6e6"  # doc backgrounds are fixed-dark, so math is always light
_MATH_CACHE: dict = {}
_MATH_DIR = None


def _math_cache_dir() -> str:
    global _MATH_DIR
    if _MATH_DIR is None:
        import tempfile
        _MATH_DIR = tempfile.mkdtemp(prefix="sas_doc_math_")
    return _MATH_DIR


def _render_latex_png(latex: str, color: str, fontsize: int) -> "str | None":
    """Render one LaTeX expression to a transparent PNG; return its path (cached)."""
    latex = (latex or "").strip()
    if not latex:
        return None
    key = (latex, color, _MATH_DPI, fontsize)
    cached = _MATH_CACHE.get(key)
    if cached:
        return cached
    try:
        import hashlib
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h = hashlib.md5(repr(key).encode("utf-8")).hexdigest()[:16]
        path = os.path.join(_math_cache_dir(), f"eq_{h}.png")
        if not os.path.exists(path):
            fig = plt.figure(figsize=(0.01, 0.01))
            fig.text(0, 0, f"${latex}$", color=color, fontsize=fontsize)
            fig.savefig(path, dpi=_MATH_DPI, transparent=True,
                        bbox_inches="tight", pad_inches=0.06)
            plt.close(fig)
        _MATH_CACHE[key] = path
        return path
    except Exception:
        return None


def _mathify_html(html: str, color: str = _MATH_COLOR) -> str:
    """Replace math markers in `html` with <img> tags pointing at rendered PNGs.
    Unrenderable expressions fall back to their raw LaTeX in a <code> span."""
    import re

    def _img(latex: str, inline: bool) -> str:
        fontsize = _MATH_FS_INLINE if inline else _MATH_FS_BLOCK
        path = _render_latex_png(latex, color, fontsize)
        if not path:
            esc = (latex or "").strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            return f"<code>{esc}</code>"
        w = h = 0
        try:
            im = QImage(path)
            w = max(1, im.width() // _MATH_SS)
            h = max(1, im.height() // _MATH_SS)
        except Exception:
            pass
        dim = f' width="{w}" height="{h}"' if w and h else ""
        fn = os.path.basename(path)
        if inline:
            return f'<img src="{fn}"{dim} style="vertical-align:middle;">'
        return f'<div class="eqimg"><img src="{fn}"{dim}></div>'

    # block: <div class="matheq">…</div> and <div class="matheq-hero">…</div>
    def _block(m):
        kind = m.group(1) or ""
        latex = m.group(2)
        inner = _img(latex, inline=False)
        cls = "hero" if "hero" in kind else "eq"
        return f'<div class="{cls}">{inner}</div>'

    html = re.sub(r'<div class="matheq(-hero)?">(.*?)</div>', _block, html, flags=re.S)
    # inline: \( … \)
    html = re.sub(r'\\\((.*?)\\\)', lambda m: _img(m.group(1), inline=True), html, flags=re.S)
    return html


# ---------------------------------------------------------------------------
# the document button
# ---------------------------------------------------------------------------
def _page_glyph_icon(widget, px: int = 18) -> QIcon:
    """A small 'document' glyph painted in the current palette text color, so it
    reads correctly in any theme and stays crisp on HiDPI. No art assets needed."""
    try:
        dpr = float(widget.devicePixelRatioF()) if widget is not None else 1.0
    except Exception:
        dpr = 1.0
    pm = QPixmap(int(px * dpr), int(px * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)

    col = QColor(140, 140, 140)
    if widget is not None:
        try:
            col = widget.palette().color(QPalette.ColorRole.WindowText)
        except Exception:
            pass

    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(col)
    pen.setWidthF(max(1.2, px * 0.08))
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)

    x0, y0 = px * 0.24, px * 0.14
    x1, y1 = px * 0.76, px * 0.86
    fold = px * 0.18
    pts = [(x0, y0), (x1 - fold, y0), (x1, y0 + fold),
           (x1, y1), (x0, y1), (x0, y0)]
    for i in range(len(pts) - 1):
        p.drawLine(int(pts[i][0]), int(pts[i][1]),
                   int(pts[i + 1][0]), int(pts[i + 1][1]))
    p.drawLine(int(x1 - fold), int(y0), int(x1 - fold), int(y0 + fold))
    p.drawLine(int(x1 - fold), int(y0 + fold), int(x1), int(y0 + fold))
    for fy in (0.40, 0.55, 0.70):
        p.drawLine(int(px * 0.34), int(px * fy), int(px * 0.66), int(px * fy))
    p.end()
    return QIcon(pm)


def make_help_button(doc_id: str, parent=None, size: int = 18) -> QToolButton:
    """A small, flat document button that opens this tool's help page.

    Tries a themed resource icon first (``help`` / ``documentation`` / ``doc``),
    otherwise paints a page glyph. Clicking shows the in-app viewer."""
    btn = QToolButton(parent)
    btn.setAutoRaise(True)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
    btn.setIconSize(QSize(size, size))

    icon = None
    try:
        from setiastro.saspro.resources import get_icon_path
        for key in ("help", "documentation", "doc", "question"):
            try:
                p = get_icon_path(key)
            except Exception:
                p = None
            if p and os.path.exists(p):
                ic = QIcon(p)
                if not ic.isNull():
                    icon = ic
                    break
    except Exception:
        pass
    if icon is None:
        icon = _page_glyph_icon(parent, size)
    btn.setIcon(icon)

    btn.setToolTip(btn.tr("Open documentation for {0}").format(_doc_title(doc_id)))
    btn.clicked.connect(lambda _=False: show_tool_doc(doc_id, parent))
    return btn


# ---------------------------------------------------------------------------
# the viewer
# ---------------------------------------------------------------------------
class DocViewerDialog(QDialog):
    """Lightweight in-app HTML help viewer (QTextBrowser-based).

    QTextBrowser renders Qt rich text (a practical subset of HTML/CSS) -- enough
    for headings, lists, tables, inline images and links, with no heavy
    WebEngine dependency."""

    def __init__(self, parent, doc_id: str, html_path):
        super().__init__(parent)
        self._doc_id = doc_id
        title = _doc_title(doc_id)
        self.setWindowTitle(self.tr("Documentation - {0}").format(title))
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass
        self.resize(780, 640)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.browser = QTextBrowser(self)
        self.browser.setOpenExternalLinks(True)
        search = [d for d in _docs_dirs() if os.path.isdir(d)]
        if search:
            self.browser.setSearchPaths(search)  # lets pages use relative <img>
        root.addWidget(self.browser, 1)

        bar = QHBoxLayout()
        bar.setContentsMargins(10, 8, 10, 10)
        bar.addStretch(1)
        self.btn_close = QPushButton(self.tr("Close"))
        self.btn_close.clicked.connect(self.close)
        bar.addWidget(self.btn_close)
        root.addLayout(bar)

        self._load(html_path)

    def _load(self, html_path):
        if html_path and os.path.isfile(html_path):
            try:
                with open(html_path, "r", encoding="utf-8") as fh:
                    raw = fh.read()
                html = _mathify_html(raw, _MATH_COLOR)
                # make the rendered-equation PNGs resolvable by bare filename
                try:
                    paths = list(self.browser.searchPaths())
                    md = _math_cache_dir()
                    if md not in paths:
                        paths.append(md)
                        self.browser.setSearchPaths(paths)
                except Exception:
                    pass
                self.browser.setHtml(html)
                return
            except Exception:
                try:
                    self.browser.setSource(QUrl.fromLocalFile(html_path))
                    return
                except Exception:
                    pass
        title = _doc_title(self._doc_id)
        self.browser.setHtml(
            f"<h2>{title}</h2>"
            f"<p>Documentation for this tool isn't available yet.</p>"
        )


def show_tool_doc(doc_id: str, parent=None) -> DocViewerDialog:
    """Open (or re-raise) the help viewer for a tool. Reuses one viewer per
    parent so repeated clicks don't stack windows."""
    host = parent
    existing = getattr(host, "_sas_doc_viewers", None) if host is not None else None
    if isinstance(existing, dict):
        dlg = existing.get(doc_id)
        if dlg is not None:
            try:
                if dlg.isVisible():
                    dlg.raise_()
                    dlg.activateWindow()
                    return dlg
            except RuntimeError:
                existing.pop(doc_id, None)  # C++ side gone

    dlg = DocViewerDialog(parent, doc_id, find_doc_html(doc_id))
    if host is not None:
        try:
            if not isinstance(getattr(host, "_sas_doc_viewers", None), dict):
                host._sas_doc_viewers = {}
            host._sas_doc_viewers[doc_id] = dlg
            dlg.destroyed.connect(
                lambda *_: host._sas_doc_viewers.pop(doc_id, None)
                if isinstance(getattr(host, "_sas_doc_viewers", None), dict) else None
            )
        except Exception:
            pass
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg