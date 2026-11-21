# src/setiastro/saspro/docs_explorer.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
Documentation explorer: About -> Documentation... (F1).

Every built-in help page (docs/*.html, the pages the tools' help buttons open)
in one window: a contents tree grouped the way the menus are, a search across
the text of every page, back / forward, links between pages, and a button for
the online wiki.

Pages are grouped under the menu their tool lives in (gui/menu_spec.py). A
new page lands there by itself when its file name is the tool's command id, or
"act_<name>" is the tool's action; _DOC_ACTIONS covers the pages named
differently. A page that matches no menu action goes under "Workspace &
General".
"""
from __future__ import annotations

import html as _html
import os
import re

from PyQt6.QtCore import Qt, QSettings, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices, QKeySequence, QShortcut, QTextCursor
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSplitter, QStyle,
    QTextBrowser, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
)

from setiastro.saspro.help_support import (
    _doc_title, _docs_dirs, find_doc_html, load_doc_into,
)

WIKI_URL = "https://github.com/setiastro/setiastrosuitepro/wiki"
GENERAL = "Workspace & General"
_SETTINGS = "docs_explorer"
_ROLE_DOC = Qt.ItemDataRole.UserRole

# Pages whose file name is neither the tool's command id nor "act_<name>".
_DOC_ACTIONS: dict[str, str] = {
    "aberration_ai": "actAberrationAI",
    "cosmic_clarity": "actCosmicUI",
    "cosmic_clarity_satellite": "actCosmicSat",
    "continuum_subtract": "act_contsub",
    "exoplanet": "act_exo_detector",
    "frequency_separation": "act_freqsep",
    "glimr": "act_isophote",
    "luminance_recombine": "act_recombine_luma",
    "mask_creation": "act_create_mask",
    "nb_normalization": "act_narrowband_normalization",
    "nb_rgb_stars": "act_nbtorgb",
    "perfect_palette_picker": "act_ppp",
    "planetary_batch": "act_planetary_stacker",
    "planetary_viewer": "act_planetary_stacker",
    "plate_solver": "act_plate_solve",
    "rgb_combination": "act_rgb_combine",
    "selective_luma": "act_selective_lum",
    "star_alignment": "act_star_align",
    "star_registration": "act_star_register",
    "supernova_asteroid_hunter": "act_supernova_hunter",
    "workflow_assistant": "act_workflows",
}


# ---------------------------------------------------------------------------
# the page catalogue
# ---------------------------------------------------------------------------
def all_doc_ids() -> list[str]:
    """Every help page, by file name without .html."""
    seen: set[str] = set()
    out: list[str] = []
    for d in _docs_dirs():
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for n in names:
            if n.lower().endswith(".html"):
                doc = n[:-5]
                if doc not in seen:
                    seen.add(doc)
                    out.append(doc)
    return out


def _menu_label(text: str) -> str:
    """Menu text to plain text: '&&' is a literal ampersand, '&x' a mnemonic."""
    return text.replace("&&", "\0").replace("&", "").replace("\0", "&")


def doc_tree(main_window, doc_ids) -> list:
    """[(section, [(group or None, [doc_id, ...]), ...]), ...] in menu order.

    Each page appears once, under the first menu (Icon Cheat Sheet order) that
    lists its tool. Without a main window there are no actions to match, so
    every page goes in one alphabetical list."""
    docs = list(doc_ids)
    by_title = lambda d: _doc_title(d).lower()
    if main_window is None:
        return [("All Pages", [(None, sorted(docs, key=by_title))])]

    from setiastro.saspro.gui import menu_spec

    attrs = [a for v in menu_spec.MENU_CONTENTS.values() for a in v]
    attrset = set(attrs)
    by_cid: dict[str, str] = {}
    for a in attrs:
        act = getattr(main_window, a, None)
        if act is None:
            continue
        try:
            cid = str(act.property("command_id") or act.objectName() or "")
        except Exception:
            cid = ""
        if cid:
            by_cid.setdefault(cid, a)

    docs_for_attr: dict[str, list[str]] = {}
    for d in docs:
        a = _DOC_ACTIONS.get(d)
        if a is None and f"act_{d}" in attrset:
            a = f"act_{d}"
        if a is None:
            a = by_cid.get(d)
        if a is not None:
            docs_for_attr.setdefault(a, []).append(d)

    placed: set[str] = set()
    tree = []
    for key in menu_spec.CHEAT_SHEET_ORDER:
        title = menu_spec.MENU_TITLES.get(key, key.replace("_", " ").title())
        contents = menu_spec.MENU_CONTENTS.get(key, [])
        groups = list(menu_spec.SUBMENUS.get(key) or [])
        if groups:
            filed = {a for _label, ga in groups for a in ga}
            rest = [a for a in contents if a not in filed]
            if rest:
                groups.append(("Other", rest))
        else:
            groups = [(None, contents)]
        out_groups = []
        for label, group_attrs in groups:
            ids = []
            for a in group_attrs:
                for d in docs_for_attr.get(a, []):
                    if d not in placed:
                        placed.add(d)
                        ids.append(d)
            if ids:
                out_groups.append((_menu_label(label) if label else None, ids))
        if out_groups:
            tree.append((title, out_groups))

    rest = sorted((d for d in docs if d not in placed), key=by_title)
    if rest:
        tree.insert(0, (GENERAL, [(None, rest)]))
    return tree


_STYLE_RE = re.compile(r"<(style|script)\b.*?</\1\s*>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _plain_text(path: str | None) -> str:
    """A page's visible text, lower-cased, for search."""
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except Exception:
        return ""
    raw = _STYLE_RE.sub(" ", raw)
    return _WS_RE.sub(" ", _html.unescape(_TAG_RE.sub(" ", raw))).lower()


# Contents page, in the help pages' own (fixed-dark) style.
_HOME_STYLE = """
  body { background:#1b1b1b; color:#dcdcdc; font-family:"Segoe UI",sans-serif;
         font-size:13px; line-height:1.55; margin:18px; }
  h1 { color:#3dbf9f; font-size:21px; margin:0 0 4px 0; }
  h2 { color:#3dbf9f; font-size:16px; margin:20px 0 4px 0;
       border-bottom:1px solid #333; padding-bottom:3px; }
  h3 { color:#8fd7c6; font-size:13px; margin:8px 0 2px 0; }
  p { color:#dcdcdc; margin:4px 0; }
  a { color:#88ffcc; text-decoration:none; }
  .sub { color:#9a9a9a; font-size:12px; margin-top:0; }
  .tip { background:#20302c; border-left:3px solid #3dbf9f; padding:8px 12px;
         margin:10px 0; color:#cfe8e0; }
"""


def _home_html(tree, n_pages: int) -> str:
    e = _html.escape
    parts = [
        "<html><head><meta charset='utf-8'><style>", _HOME_STYLE, "</style></head><body>",
        "<h1>SASpro Documentation</h1>",
        f"<p class='sub'>{n_pages} pages: the same ones each tool's help button opens.</p>",
        "<p>Pick a page here or in the list on the left. The search box looks "
        "through the text of every page, not just the titles.</p>",
        f"<p class='tip'>The SASpro wiki on GitHub is the online companion to these "
        f"pages: <a href='{WIKI_URL}'>open the wiki</a>.</p>",
    ]
    for section, groups in tree:
        parts.append(f"<h2>{e(section)}</h2>")
        for label, ids in groups:
            if label:
                parts.append(f"<h3>{e(label)}</h3>")
            links = " &nbsp;&middot;&nbsp; ".join(
                f"<a href='{e(d)}.html'>{e(_doc_title(d))}</a>" for d in ids)
            parts.append(f"<p>{links}</p>")
    parts.append("</body></html>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# the window
# ---------------------------------------------------------------------------
class DocsExplorerDialog(QDialog):
    """Browse all of SASpro's built-in documentation."""

    HOME = ""     # history entry for the contents page

    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("SASpro Documentation"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setModal(False)
        self._ids = all_doc_ids()
        self._tree_data = doc_tree(main_window, self._ids)
        self._text: dict[str, str] = {}      # doc_id -> searchable text, built on first search
        self._items: dict[str, QTreeWidgetItem] = {}
        self._history: list[str] = []
        self._pos = -1
        self._current: str | None = None
        self._query_words: list[str] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(4)
        st = self.style()
        self.btn_back = self._tool_button(st.standardIcon(QStyle.StandardPixmap.SP_ArrowBack),
                                          self.tr("Back (Alt+Left)"), self.go_back)
        self.btn_forward = self._tool_button(st.standardIcon(QStyle.StandardPixmap.SP_ArrowForward),
                                             self.tr("Forward (Alt+Right)"), self.go_forward)
        self.btn_home = self._tool_button(st.standardIcon(QStyle.StandardPixmap.SP_DirHomeIcon),
                                          self.tr("Contents"), self.show_home)
        for b in (self.btn_back, self.btn_forward, self.btn_home):
            top.addWidget(b)
        self.search = QLineEdit(self)
        self.search.setPlaceholderText(self.tr("Search all pages (Ctrl+F)"))
        self.search.setClearButtonEnabled(True)
        top.addWidget(self.search, 1)
        root.addLayout(top)

        self.split = QSplitter(Qt.Orientation.Horizontal, self)
        self.tree = QTreeWidget(self.split)
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumWidth(200)
        self.browser = QTextBrowser(self.split)
        self.browser.setOpenLinks(False)          # every link goes through _on_link
        search_paths = [d for d in _docs_dirs() if os.path.isdir(d)]
        if search_paths:
            self.browser.setSearchPaths(search_paths)
        self.split.setStretchFactor(0, 0)
        self.split.setStretchFactor(1, 1)
        self.split.setSizes([270, 760])
        root.addWidget(self.split, 1)

        bottom = QHBoxLayout()
        self.lbl_count = QLabel(self)
        bottom.addWidget(self.lbl_count)
        bottom.addStretch(1)
        self.btn_wiki = QPushButton(self.tr("Open the Online Wiki..."), self)
        self.btn_wiki.setToolTip(WIKI_URL)
        self.btn_wiki.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(WIKI_URL)))
        bottom.addWidget(self.btn_wiki)
        self.btn_close = QPushButton(self.tr("Close"), self)
        self.btn_close.clicked.connect(self.close)
        bottom.addWidget(self.btn_close)
        root.addLayout(bottom)

        self._build_tree()
        self.tree.currentItemChanged.connect(self._on_tree_current)
        self.browser.anchorClicked.connect(self._on_link)

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(150)
        self._search_timer.timeout.connect(self._apply_filter)
        self.search.textChanged.connect(lambda _t: self._search_timer.start())
        self.search.returnPressed.connect(self._open_first_match)

        QShortcut(QKeySequence("Alt+Left"), self, activated=self.go_back)
        QShortcut(QKeySequence("Alt+Right"), self, activated=self.go_forward)
        QShortcut(QKeySequence(QKeySequence.StandardKey.Find), self,
                  activated=lambda: (self.search.setFocus(), self.search.selectAll()))

        self.resize(1060, 720)
        s = QSettings()
        try:
            g = s.value(f"{_SETTINGS}/geometry")
            if g:
                self.restoreGeometry(g)
            sp = s.value(f"{_SETTINGS}/splitter")
            if sp:
                self.split.restoreState(sp)
        except Exception:
            pass
        last = s.value(f"{_SETTINGS}/last_page", "", type=str) or ""
        if last and last in self._items:
            self.open_doc(last)
        else:
            self.show_home()
        self._update_count(len(self._ids))

    # -- construction ---------------------------------------------------------
    def _tool_button(self, icon, tip, slot) -> QToolButton:
        b = QToolButton(self)
        b.setIcon(icon)
        b.setToolTip(tip)
        b.setAutoRaise(True)
        b.clicked.connect(slot)
        return b

    def _build_tree(self):
        self.tree.clear()
        self._items.clear()
        for section, groups in self._tree_data:
            top = QTreeWidgetItem([section])
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            top.setFlags(top.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.tree.addTopLevelItem(top)
            for label, ids in groups:
                parent = top
                if label:
                    parent = QTreeWidgetItem([label])
                    parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                    top.addChild(parent)
                for d in ids:
                    it = QTreeWidgetItem([_doc_title(d)])
                    it.setData(0, _ROLE_DOC, d)
                    it.setToolTip(0, f"{d}.html")
                    parent.addChild(it)
                    self._items[d] = it
            top.setExpanded(True)

    # -- navigation -----------------------------------------------------------
    def open_doc(self, doc_id: str, anchor: str | None = None, record: bool = True):
        path = find_doc_html(doc_id)
        load_doc_into(self.browser, doc_id, path)
        self._current = doc_id
        it = self._items.get(doc_id)
        if it is not None and self.tree.currentItem() is not it:
            self.tree.blockSignals(True)
            try:
                p = it.parent()
                while p is not None:
                    p.setExpanded(True)
                    p = p.parent()
                self.tree.setCurrentItem(it)
                self.tree.scrollToItem(it)
            finally:
                self.tree.blockSignals(False)
        if record:
            self._push(doc_id)
        if anchor:
            QTimer.singleShot(0, lambda a=anchor: self.browser.scrollToAnchor(a))
        elif self._query_words:
            self._highlight_first(self._query_words)
        try:
            QSettings().setValue(f"{_SETTINGS}/last_page", doc_id)
        except Exception:
            pass

    def show_home(self, record: bool = True):
        self.browser.setHtml(_home_html(self._tree_data, len(self._ids)))
        self._current = self.HOME
        self.tree.blockSignals(True)
        try:
            self.tree.setCurrentItem(None)
            self.tree.clearSelection()
        finally:
            self.tree.blockSignals(False)
        if record:
            self._push(self.HOME)
        try:
            QSettings().setValue(f"{_SETTINGS}/last_page", "")
        except Exception:
            pass

    def _push(self, entry: str):
        if 0 <= self._pos < len(self._history) and self._history[self._pos] == entry:
            self._update_nav()
            return
        del self._history[self._pos + 1:]
        self._history.append(entry)
        self._pos = len(self._history) - 1
        self._update_nav()

    def _go_to(self, pos: int):
        if not (0 <= pos < len(self._history)):
            return
        self._pos = pos
        entry = self._history[pos]
        if entry == self.HOME:
            self.show_home(record=False)
        else:
            self.open_doc(entry, record=False)
        self._update_nav()

    def go_back(self):
        self._go_to(self._pos - 1)

    def go_forward(self):
        self._go_to(self._pos + 1)

    def _update_nav(self):
        self.btn_back.setEnabled(self._pos > 0)
        self.btn_forward.setEnabled(self._pos < len(self._history) - 1)

    def _on_tree_current(self, cur, _prev):
        doc = cur.data(0, _ROLE_DOC) if cur is not None else None
        if doc and doc != self._current:
            self.open_doc(str(doc))

    def _on_link(self, url: QUrl):
        if url.scheme() in ("http", "https", "mailto", "ftp"):
            QDesktopServices.openUrl(url)
            return
        path, frag = url.path(), url.fragment()
        if path.lower().endswith(".html"):
            doc = os.path.splitext(os.path.basename(path))[0]
            if find_doc_html(doc):
                self.open_doc(doc, anchor=frag or None)
                return
        if not path and frag:
            self.browser.scrollToAnchor(frag)
            return
        QDesktopServices.openUrl(url)

    # -- search ---------------------------------------------------------------
    def _ensure_text(self):
        if not self._text:
            self._text = {d: _plain_text(find_doc_html(d)) for d in self._ids}

    def _matches(self, doc: str, words: list[str]) -> bool:
        hay = _doc_title(doc).lower() + " " + self._text.get(doc, "")
        return all(w in hay for w in words)

    def _apply_filter(self):
        words = self.search.text().strip().lower().split()
        self._query_words = words
        if words:
            self._ensure_text()
        shown = 0

        def walk(item) -> bool:
            nonlocal shown
            doc = item.data(0, _ROLE_DOC)
            if doc:
                ok = not words or self._matches(str(doc), words)
                item.setHidden(not ok)
                shown += ok
                return ok
            any_child = False
            for i in range(item.childCount()):
                any_child |= walk(item.child(i))
            item.setHidden(not any_child)
            if words and any_child:
                item.setExpanded(True)
            return any_child

        for i in range(self.tree.topLevelItemCount()):
            walk(self.tree.topLevelItem(i))
        self._update_count(shown)
        if words and self._current:
            self._highlight_first(words)

    def _open_first_match(self):
        """Enter in the search box: open the best match -- a page whose title
        has every word first, then the page that mentions them most."""
        self._search_timer.stop()
        self._apply_filter()
        words = self._query_words
        hits = [d for d, it in self._items.items() if not it.isHidden()]
        if not hits:
            return
        if words:
            def score(d):
                title = _doc_title(d).lower()
                text = self._text.get(d, "")
                return (all(w in title for w in words), sum(text.count(w) for w in words))
            hits.sort(key=score, reverse=True)
        self.open_doc(hits[0])

    def _highlight_first(self, words: list[str]):
        """Select the first place the search shows up on the open page."""
        for w in sorted(words, key=len, reverse=True):
            self.browser.moveCursor(QTextCursor.MoveOperation.Start)
            if self.browser.find(w):
                return

    def _update_count(self, shown: int):
        total = len(self._ids)
        if self._query_words:
            self.lbl_count.setText(self.tr("{0} of {1} pages match").format(shown, total))
        else:
            self.lbl_count.setText(self.tr("{0} pages").format(total))

    # -- persistence ----------------------------------------------------------
    def _save_layout(self):
        try:
            s = QSettings()
            s.setValue(f"{_SETTINGS}/geometry", self.saveGeometry())
            s.setValue(f"{_SETTINGS}/splitter", self.split.saveState())
        except Exception:
            pass

    def hideEvent(self, e):
        self._save_layout()
        super().hideEvent(e)

    def closeEvent(self, e):
        self._save_layout()
        super().closeEvent(e)


# ---------------------------------------------------------------------------
# opening it
# ---------------------------------------------------------------------------
_EXPLORER: DocsExplorerDialog | None = None


def _find_main_window(widget):
    """The SASpro main window above `widget` (the one with a doc manager)."""
    cur = widget
    while cur is not None:
        if getattr(cur, "docman", None) is not None or getattr(cur, "doc_manager", None) is not None:
            return cur
        try:
            cur = cur.parent()
        except Exception:
            return None
    return None


def open_docs_explorer(parent=None, doc_id: str | None = None) -> DocsExplorerDialog:
    """Show the documentation explorer (one per session), optionally at a page."""
    global _EXPLORER
    dlg = _EXPLORER
    try:
        if dlg is not None:
            dlg.isVisible()            # raises RuntimeError once Qt has deleted it
    except RuntimeError:
        dlg = None
    if dlg is None:
        mw = _find_main_window(parent)
        dlg = DocsExplorerDialog(mw, main_window=mw)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        _EXPLORER = dlg

        def _forget(*_):
            global _EXPLORER
            _EXPLORER = None
        dlg.destroyed.connect(_forget)
    if doc_id and find_doc_html(doc_id):
        dlg.open_doc(doc_id)
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg
