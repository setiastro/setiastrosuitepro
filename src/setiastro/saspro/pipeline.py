# pro/pipeline.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from PyQt6.QtCore import QObject, pyqtSignal, QThread

@dataclass
class PipelineStep:
    command_id: str
    preset: Dict = field(default_factory=dict)

@dataclass
class Pipeline:
    id: str
    name: str
    steps: List[PipelineStep] = field(default_factory=list)

class PipelineStore(QObject):
    changed = pyqtSignal()
    def __init__(self):
        super().__init__()
        self._by_id: dict[str, Pipeline] = {}

    def add(self, p: Pipeline):
        self._by_id[p.id] = p
        self.changed.emit()

    def remove(self, pid: str):
        self._by_id.pop(pid, None)
        self.changed.emit()

    def get(self, pid: str) -> Optional[Pipeline]:
        return self._by_id.get(pid)

    def all(self) -> list[Pipeline]:
        return list(self._by_id.values())

class _PipelineRunner(QObject):
    progress = pyqtSignal(int, int, str)    # doc_index, step_index, message
    finished = pyqtSignal(bool, str)

    def __init__(self, mw, docs: list, pipeline: Pipeline):
        super().__init__()
        self.mw = mw
        self.docs = docs
        self.pipeline = pipeline
        self._cancel = False

    def cancel(self): self._cancel = True

    def _subwindow_for_doc(self, doc):
        # Try to find an open subwindow for doc; if not, open one (same behavior as headless drops)
        for sw in self.mw.mdi.subWindowList():
            try:
                w = sw.widget()
                if getattr(w, "document", None) is doc:
                    return sw
            except Exception:
                pass
        try:
            return self.mw._spawn_subwindow_for(doc)
        except Exception:
            return None

    def run(self):
        ok_all = True
        try:
            for di, doc in enumerate(self.docs):
                if self._cancel: break
                sw = self._subwindow_for_doc(doc)
                if sw is None:
                    ok_all = False
                    self.progress.emit(di, -1, "No subwindow for doc")
                    continue

                for si, step in enumerate(self.pipeline.steps):
                    if self._cancel: break
                    payload = {"command_id": step.command_id, "preset": step.preset or {}}
                    try:
                        # Reuse your existing headless handler
                        self.mw._handle_command_drop(payload, sw)
                        self.progress.emit(di, si, step.command_id)
                    except Exception as e:
                        ok_all = False
                        self.progress.emit(di, si, f"❌ {step.command_id}: {e}")
                        # continue to next step/doc
                        continue
            self.finished.emit(ok_all, "Done" if ok_all else "Finished with errors")
        except Exception as e:
            self.finished.emit(False, f"Pipeline crashed: {e}")

def run_pipeline_async(mw, *, docs: list, pipeline: Pipeline):
    runner = _PipelineRunner(mw, docs, pipeline)
    t = QThread(mw)
    runner.moveToThread(t)

    def _done(ok, msg):
        if hasattr(mw, "update_status"):
            mw.update_status(f"[{pipeline.name}] {'✅' if ok else '⚠️'} {msg}")
        t.quit(); t.wait(); runner.deleteLater(); t.deleteLater()

    def _prog(di, si, msg):
        if hasattr(mw, "update_status"):
            mw.update_status(f"[{pipeline.name}] Doc {di+1}, Step {si+1}: {msg}")

    runner.progress.connect(_prog)
    runner.finished.connect(_done)
    t.started.connect(runner.run)
    t.start()
    return runner, t
