# pro/bundles.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from PyQt6.QtCore import QObject, pyqtSignal

@dataclass
class ViewBundle:
    id: str
    name: str
    doc_uids: List[str] = field(default_factory=list)  # store stable doc UIDs

class BundleManager(QObject):
    changed = pyqtSignal()
    def __init__(self, dm):
        super().__init__()
        self.dm = dm
        self._by_id: Dict[str, ViewBundle] = {}

    def add(self, b: ViewBundle):
        self._by_id[b.id] = b
        self.changed.emit()

    def remove(self, bid: str):
        self._by_id.pop(bid, None)
        self.changed.emit()

    def get(self, bid: str) -> Optional[ViewBundle]:
        return self._by_id.get(bid)

    def all(self) -> list[ViewBundle]:
        return list(self._by_id.values())

    def add_doc(self, bid: str, doc_uid: str):
        b = self._by_id.get(bid)
        if not b: return
        if doc_uid not in b.doc_uids:
            b.doc_uids.append(doc_uid)
            self.changed.emit()

    def remove_doc(self, bid: str, doc_uid: str):
        b = self._by_id.get(bid)
        if not b: return
        if doc_uid in b.doc_uids:
            b.doc_uids.remove(doc_uid)
            self.changed.emit()

    def docs(self, bid: str):
        b = self._by_id.get(bid)
        if not b: return []
        out = []
        for uid in b.doc_uids:
            d = getattr(self.dm, "find_document_by_uid", None)
            if callable(d):
                doc = d(uid)
            else:
                # fallback: linear scan over open docs if you don't have an index
                doc = next((x for x in getattr(self.dm, "documents", []) if getattr(x, "uid", None) == uid), None)
            if doc:
                out.append(doc)
        return out
