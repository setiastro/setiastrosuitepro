"""Rank palette commands using existing action labels and descriptions."""
from difflib import SequenceMatcher
from functools import lru_cache
import re
import unicodedata

from PyQt6 import sip
from PyQt6.QtCore import QSortFilterProxyModel, Qt
from PyQt6.QtGui import QAction, QStandardItem

MAX_QUERY_LENGTH = 256


class _CommandTarget:
    """Keep a Python reference without QVariant converting it to QObject*."""

    def __init__(self, action):
        self.action = action


def command_item(action, title, role):
    """Build a searchable item safe to read after its QAction is destroyed."""
    item = QStandardItem(action.icon(), title)
    item.setEditable(False)
    item.setData(_CommandTarget(action), role)
    item.setToolTip("\n".join(dict.fromkeys(
        text for text in (action.statusTip(), action.toolTip()) if text)))
    return item


def run_command(index, role):
    """Trigger a live, enabled target; stale completions safely do nothing."""
    if not index.isValid():
        return False
    target = index.data(role)
    if not isinstance(target, _CommandTarget):
        return False
    action = target.action
    if (not isinstance(action, QAction) or sip.isdeleted(action)
            or not action.isEnabled()):
        return False
    action.trigger()
    return True


@lru_cache(maxsize=1024)
def _normalize(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    return " ".join(re.findall(
        r"[^\W_]+",
        "".join(c for c in text if not unicodedata.combining(c)),
    ))


def match_score(query, title, description):
    """Lower is better; None means at least one query word did not match.

    Exact names outrank name prefixes/contains, then descriptions, then typos.
    Short queries use literal matches to avoid noisy fuzzy results.
    """
    query = query[:MAX_QUERY_LENGTH]
    normalized = _normalize(query)
    if query.strip() and not normalized:
        return None
    return _match_score(normalized, _normalize(title), _normalize(description))


@lru_cache(maxsize=4096)
def _match_score(query, title, description):
    """Cache normalized, bounded queries rather than arbitrary pasted text."""
    if not query:
        return 0
    if query == title:
        return 0
    if title.startswith(query):
        return 1
    if query in title:
        return 2
    title_words = title.split()
    description_words = description.split()
    scores = []
    for token in dict.fromkeys(query.split()):
        if token in title_words:
            scores.append(3)
        elif any(word.startswith(token) for word in title_words):
            scores.append(4)
        elif token in title:
            scores.append(5)
        elif token in description:
            scores.append(10)
        elif len(token) >= 4:
            similarity = max((SequenceMatcher(None, token, word).ratio()
                              for word in title_words + description_words
                              if abs(len(word) - len(token)) <= 2), default=0)
            if similarity < 0.78:
                return None
            scores.append(20 + (1 - similarity))
        else:
            return None
    return max(scores) + sum(scores) / (100 * len(scores))


class CommandSearchProxy(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._query = ""
        self.setDynamicSortFilter(True)
        self.sort(0)

    def set_query(self, text):
        self._query = text[:MAX_QUERY_LENGTH].strip()
        self.invalidate()
        self.sort(0)

    def _score(self, index):
        title = index.data(Qt.ItemDataRole.DisplayRole) or ""
        description = index.data(Qt.ItemDataRole.ToolTipRole) or ""
        return match_score(self._query, title, description)

    def filterAcceptsRow(self, row, parent):
        return self._score(
            self.sourceModel().index(row, 0, parent)
        ) is not None

    def lessThan(self, left, right):
        left_score, right_score = self._score(left), self._score(right)
        # Qt may compare rows which are about to be filtered out.
        left_score = float("inf") if left_score is None else left_score
        right_score = (
            float("inf") if right_score is None else right_score
        )
        return (left_score, (left.data() or "").casefold(), left.row()) < (
            right_score, (right.data() or "").casefold(), right.row())
