# pro/bundles_dock.py
from __future__ import annotations
from PyQt6.QtWidgets import QDockWidget, QListWidget, QListWidgetItem, QMenu, QInputDialog
from PyQt6.QtCore import Qt, QMimeData, QByteArray
import json

from pro.dnd_mime import MIME_CMD  # you already use this

class BundlesDock(QDockWidget):
    def __init__(self, mw, bm, pipelines):
        super().__init__("Bundles", mw)
        self.mw = mw
        self.bm = bm
        self.pipelines = pipelines

        self.list = QListWidget(self)
        self.list.setSelectionMode(self.list.SelectionMode.SingleSelection)
        self.list.setAcceptDrops(True)
        self.list.setDragEnabled(False)
        self.setWidget(self.list)

        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self._refresh()
        bm.changed.connect(self._refresh)

        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._ctx)

    def _refresh(self):
        self.list.clear()
        for b in self.bm.all():
            it = QListWidgetItem(f"{b.name}  ({len(b.doc_uids)} views)")
            it.setData(Qt.ItemDataRole.UserRole, b.id)
            self.list.addItem(it)

    def _ctx(self, pos):
        it = self.list.itemAt(pos)
        m = QMenu(self)
        m.addAction("New Bundle…", self._new_bundle)
        if it:
            bid = it.data(Qt.ItemDataRole.UserRole)
            m.addSeparator()
            m.addAction("Rename…", lambda: self._rename(bid))
            m.addAction("Delete", lambda: self._delete(bid))
            m.addSeparator()
            m.addAction("Run Pipeline…", lambda: self._pick_and_run(bid))
        m.exec(self.list.mapToGlobal(pos))

    def _new_bundle(self):
        name, ok = QInputDialog.getText(self, "New Bundle", "Name:")
        if not ok or not name.strip(): return
        import uuid
        from pro.bundles import ViewBundle
        self.bm.add(ViewBundle(id=uuid.uuid4().hex, name=name.strip()))

    def _rename(self, bid):
        b = self.bm.get(bid); 
        if not b: return
        name, ok = QInputDialog.getText(self, "Rename Bundle", "Name:", text=b.name)
        if ok and name.strip():
            b.name = name.strip()
            self.bm.changed.emit()

    def _delete(self, bid):
        self.bm.remove(bid)

    # --- DnD: add docs or run commands ---
    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(MIME_CMD):
            e.acceptProposedAction(); return
        # Let your explorer/subwindow provide a doc UID mime if you have one; example below:
        if e.mimeData().hasFormat("application/x-saspro-doc-uid"):
            e.acceptProposedAction(); return
        e.ignore()

    def dropEvent(self, e):
        it = self.list.itemAt(e.position().toPoint())
        if not it:
            e.ignore(); return
        bid = it.data(Qt.ItemDataRole.UserRole)

        md: QMimeData = e.mimeData()
        # 1) drop a command/pipeline onto a bundle -> run across all docs in the bundle
        if md.hasFormat(MIME_CMD):
            try:
                payload = json.loads(bytes(md.data(MIME_CMD)).decode("utf-8"))
            except Exception:
                e.ignore(); return
            docs = self.bm.docs(bid)
            if not docs:
                e.ignore(); return
            # Pipeline support via "pipeline:<id>" (see MW patch below)
            self.mw._run_payload_on_docs(payload, docs)
            e.acceptProposedAction(); return

        # 2) drop doc(s) onto a bundle -> add
        if md.hasFormat("application/x-saspro-doc-uid"):
            uid = bytes(md.data("application/x-saspro-doc-uid")).decode("utf-8")
            self.bm.add_doc(bid, uid)
            e.acceptProposedAction(); return

        e.ignore()

    # quick picker to run a pipeline without DnD
    def _pick_and_run(self, bid):
        plist = self.pipelines.all()
        if not plist:
            return
        names = [p.name for p in plist]
        i, ok = QInputDialog.getItem(self, "Run Pipeline", "Pick:", names, 0, False)
        if not ok: return
        p = plist[names.index(i)]
        payload = {"command_id": f"pipeline:{p.id}", "preset": {}}
        self.mw._run_payload_on_docs(payload, self.bm.docs(bid))
