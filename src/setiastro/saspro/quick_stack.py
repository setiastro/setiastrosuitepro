# src/setiastro/saspro/quick_stack.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
Quick Stack: pick frames, press a button, get an image.

A fast look at a set of subs, not a replacement for the Stacking Suite. Frames
are aligned by position, rotation and scale only (the registration delta
pass, meridian-flip check included), calibrated with a master dark and flat if
they are given (or made in memory from single dark and flat frames), put on
the reference's sky level (one offset per channel),
weighted by the stars found in each, and combined as a trimmed mean.
Each filter / exposure / size group in the list becomes its own image and opens
as a new view. No calibrated or aligned frames are written to disk.

This window stays above the main window (it is owned by it), so the views it
opens can be hidden behind it. Each run's stacks are listed in the window with
a preview, and can be brought forward from there.

The CPU engine is quick_stack_worker and the GPU one quick_stack_gpu (both
GUI-free); this module is the window. The GPU is started when the window opens
(torch takes several seconds), so it is usually ready by the time the stack is
wanted, and stacking falls back to the CPU if it cannot be used.
"""
from __future__ import annotations

import html
import os
import threading
import weakref

from PyQt6.QtCore import QCoreApplication, QSettings, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListView,
    QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from setiastro.saspro import quick_stack_worker as engine
from setiastro.saspro.help_support import make_help_button

_SETTINGS = "quick_stack"
_ROLE_PATH = int(Qt.ItemDataRole.UserRole)
_ROLE_INFO = _ROLE_PATH + 1
_ROLE_DOC = _ROLE_PATH + 2          # a weak reference to a stack's document
_ROLE_KEY = _ROLE_PATH + 3          # a group row's FrameGroup.key
_RESULT_ICON = QSize(138, 84)       # the Start panel's thumbnail shape, three quarters of its size

# A group's Dark / Flat list: its first entry, the way out, and "add one".
_AUTO, _NONE, _BROWSE = "\x00auto", "\x00none", "\x00browse"
_BADGE_COLOUR = {"ok": "#3fb950", "warn": "#d29922", "info": "#58a6ff"}

# The Stacking Suite's pipeline output folders (StackingSuiteDialog
# ._PIPELINE_OUTPUT_DIRS): never picked up when a folder is added.
_PIPELINE_OUTPUT_DIRS = frozenset({
    "aligned_images", "calibrated", "master_calibration_files", "master_lights",
    "master_light", "normalized_images",
})

# Threads still winding down after their window closed (a QThread must not be
# destroyed while it runs).
_LIVE_THREADS: set = set()


def _keep_until_finished(thread: QThread) -> None:
    _LIVE_THREADS.add(thread)
    thread.finished.connect(lambda t=thread: _LIVE_THREADS.discard(t))


def collect_frames(paths) -> list:
    """Frame files among `paths`; folders are walked, SASpro output folders skipped."""
    out, seen = [], set()
    for p in paths:
        if os.path.isdir(p):
            for d, subdirs, files in os.walk(p):
                subdirs[:] = sorted(s for s in subdirs if s.casefold() not in _PIPELINE_OUTPUT_DIRS)
                found = [os.path.join(d, f) for f in sorted(files) if engine.is_frame_path(f)]
                out.extend(found)
        elif engine.is_frame_path(p):
            out.append(p)
    uniq = []
    for p in out:
        k = os.path.normcase(os.path.normpath(p))
        if k not in seen:
            seen.add(k)
            uniq.append(os.path.normpath(p))
    return uniq


class _ProbeThread(QThread):
    """Reads headers off the GUI thread (a slow disk must not freeze the window)."""
    probed = pyqtSignal(object)          # list[FrameInfo]
    progressed = pyqtSignal(int, int)

    def __init__(self, paths, known=frozenset()):
        super().__init__()
        self._paths = list(paths)
        self._known = known
        self._stop = threading.Event()

    def cancel(self):
        self._stop.set()

    def run(self):
        from concurrent.futures import ThreadPoolExecutor
        todo = [p for p in collect_frames(self._paths)
                if os.path.normcase(p) not in self._known]
        infos = []
        with ThreadPoolExecutor(8) as ex:
            for i, info in enumerate(ex.map(engine.probe_frame, todo), 1):
                if self._stop.is_set():
                    return
                infos.append(info)
                if i % 50 == 0 or i == len(todo):
                    self.progressed.emit(i, len(todo))
        self.probed.emit(infos)


class _StackThread(QThread):
    log_line = pyqtSignal(str)
    step = pyqtSignal(int, int, str)
    stacked = pyqtSignal(object, object)     # StackResult, FrameGroup
    group_failed = pyqtSignal(str, str)      # group label, message
    all_done = pyqtSignal(int, int)          # groups stacked, groups total

    def __init__(self, groups, ref_path, workers: int = 0, session=None, darks=(), flats=(),
                 chosen=None):
        super().__init__()
        self._groups = list(groups)
        self._ref = ref_path
        self._workers = workers
        self._session = session
        self._darks = list(darks)
        self._flats = list(flats)
        self._chosen = {k: dict(v) for k, v in (chosen or {}).items()}   # group key -> picks made by hand
        self._stop = threading.Event()

    def cancel(self):
        self._stop.set()

    def cancelled(self) -> bool:
        return self._stop.is_set()

    def run(self):
        ok = 0
        n = len(self._groups)
        for gi, g in enumerate(self._groups, 1):
            if self._stop.is_set():
                break
            prefix = f"[{gi}/{n}] " if n > 1 else ""
            self.log_line.emit(f"{prefix}{g.label}: {len(g.frames)} frames")
            ref = self._ref
            if ref:
                info = engine.probe_frame(ref)
                w, h = g.frames[0].width, g.frames[0].height
                if (w and info.width and (w, h) != (info.width, info.height)):
                    self.log_line.emit(f"{prefix}The chosen reference is {info.width} x {info.height}, "
                                       f"these frames are {w} x {h}: using the first frame instead.")
                    ref = None
            masters = None
            if self._darks or self._flats:
                try:
                    masters, notes = engine.pick_masters(g.frames[0], self._darks, self._flats,
                                                         self._chosen.get(g.key))
                except Exception as e:          # never a reason not to stack
                    masters, notes = None, [f"⚠ The masters could not be matched to these frames "
                                            f"({type(e).__name__}: {e}): none was applied."]
                for line in notes:
                    self.log_line.emit(prefix + line)
            try:
                if masters:
                    engine.ensure_made(masters, self._darks + self._flats, self.log_line.emit,
                                       progress=lambda d, t, m, p=prefix: self.step.emit(d, t, p + m),
                                       stopped=self._stop.is_set)
                r = engine.stack_frames([f.path for f in g.frames], ref_path=ref, masters=masters,
                                        workers=self._workers, log=self.log_line.emit,
                                        progress=lambda d, t, m, p=prefix: self.step.emit(d, t, p + m),
                                        should_stop=self._stop.is_set, session=self._session)
                ok += 1
                self.stacked.emit(r, g)
            except InterruptedError:
                break
            except Exception as e:
                import traceback
                self.log_line.emit(traceback.format_exc())
                self.group_failed.emit(g.label, str(e) or type(e).__name__)
        self.all_done.emit(ok, n)


class QuickStackDialog(QDialog):
    def __init__(self, parent=None, doc_manager=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Quick Stack"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setAcceptDrops(True)
        self._docman = doc_manager
        self._infos: list = []              # FrameInfo
        self._darks: list = []              # FrameInfo of the master darks
        self._flats: list = []              # and of the master flats
        self._raw_darks: list = []          # FrameInfo of single dark frames, made into masters
        self._raw_flats: list = []          # and of single flat frames
        self._dark_sets: list = []          # engine.raw_sets of those: stand in the lists as masters
        self._flat_sets: list = []
        self._chosen: dict = {}             # group key -> {"dark"/"flat": path, or None}: picked by hand
        self._cal: dict = {}                # (group key, kind) -> (group, combo, badge) in the list
        self._scan: dict = {}               # what the files being added turned out to be
        self._probe: _ProbeThread | None = None
        self._stack: _StackThread | None = None
        self._queue: list = []              # paths waiting to be probed
        self._session = None                # quick_stack_gpu.GpuSession, once the window is up
        self._closing = False
        self._settings = QSettings()
        self._build()
        self._engine_timer = QTimer(self)
        self._engine_timer.setInterval(400)
        self._engine_timer.timeout.connect(self._show_engine)
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._close_session)
        self.resize(780, 720)
        geo = self._settings.value(f"{_SETTINGS}/geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        self._show_masters()
        self._refresh()

    # -- window ------------------------------------------------------------
    def _build(self):
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        blurb = QLabel(self.tr(
            "A fast look at a set of subs: calibrated if you give it masters, aligned by position, "
            "rotation and scale, matched to the reference's sky level, and combined as a trimmed mean "
            "weighted by the stars found in each frame. No calibrated or aligned files on disk. "
            "Colour cameras are debayered (bilinear)."))
        blurb.setWordWrap(True)
        top.addWidget(blurb, 1)
        top.addWidget(make_help_button("quick_stack", self), 0, Qt.AlignmentFlag.AlignTop)
        root.addLayout(top)

        row = QHBoxLayout()
        self.btn_files = QPushButton(self.tr("Add Files…"))
        self.btn_folder = QPushButton(self.tr("Add Folder…"))
        self.btn_remove = QPushButton(self.tr("Remove Selected"))
        self.btn_clear = QPushButton(self.tr("Clear"))
        for b, fn in ((self.btn_files, self._add_files), (self.btn_folder, self._add_folder),
                      (self.btn_remove, self._remove_selected), (self.btn_clear, self._clear)):
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        self.lbl_count = QLabel("")
        row.addWidget(self.lbl_count)
        root.addLayout(row)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels([self.tr("Frame"), self.tr("Filter"), self.tr("Exposure"),
                                   self.tr("Size")])
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setColumnWidth(0, 420)
        self.tree.setUniformRowHeights(False)       # a group's Dark / Flat rows hold a list
        root.addWidget(self.tree, 2)

        rrow = QHBoxLayout()
        self._row_labels = [QLabel(self.tr("Reference:"))]
        rrow.addWidget(self._row_labels[0])
        self.ed_ref = QLineEdit()
        self.ed_ref.setReadOnly(True)
        self.ed_ref.setPlaceholderText(self.tr("First frame of each group (automatic)"))
        rrow.addWidget(self.ed_ref, 1)
        self.btn_ref = QPushButton(self.tr("Choose…"))
        self.btn_ref_sel = QPushButton(self.tr("Use Selected"))
        self.btn_ref_clear = QPushButton(self.tr("Reset"))
        self.btn_ref.clicked.connect(self._choose_ref)
        self.btn_ref_sel.clicked.connect(self._ref_from_selection)
        self.btn_ref_clear.clicked.connect(lambda: self.ed_ref.clear())
        for b in (self.btn_ref, self.btn_ref_sel, self.btn_ref_clear):
            rrow.addWidget(b)
        root.addLayout(rrow)

        mrow = QHBoxLayout()
        lab = QLabel(self.tr("Masters:"))
        self._row_labels.append(lab)
        mrow.addWidget(lab)
        self.lbl_masters = QLabel()
        mrow.addWidget(self.lbl_masters, 1)
        self.btn_masters = QPushButton(self.tr("Add Masters…"))
        self.btn_masters_clear = QPushButton(self.tr("Clear Masters"))
        self.btn_masters.clicked.connect(lambda: self._browse_masters(None))
        self.btn_masters_clear.clicked.connect(self._clear_masters)
        self._masters_tip = self.tr(
            "Master darks and flats, as many as you have: one flat per filter, darks for each exposure. "
            "Each group in the list shows the dark and flat it will use, picked by size, then filter "
            "(flats) or gain and exposure (darks). Change any group's choice there.\n\n"
            "Single dark and flat frames work too: they are combined into masters in memory "
            "(trimmed mean) the first time a stack needs them, and nothing is written to disk. "
            "Darks are grouped by exposure, gain and size; flats by filter, size and night.\n\n"
            "Masters and frames dropped on this window, or added with the lights, are sorted out by "
            "themselves.")
        for w in (lab, self.btn_masters):
            w.setToolTip(self._masters_tip)
        mrow.addWidget(self.btn_masters)
        mrow.addWidget(self.btn_masters_clear)
        root.insertLayout(root.indexOf(self.tree) + 1, mrow)       # right under the list
        widest = max(lab.sizeHint().width() for lab in self._row_labels)
        for lab in self._row_labels:
            lab.setMinimumWidth(widest)

        self.bar = QProgressBar()
        self.bar.setValue(0)
        root.addWidget(self.bar)
        self.lbl_status = QLabel(self.tr("Add frames to begin. Files and folders can also be dropped here."))
        root.addWidget(self.lbl_status)

        # This run's stacks: they open in the main window, which this window
        # sits on top of, so they are shown here too.
        self.results = QWidget()
        rl = QVBoxLayout(self.results)
        rl.setContentsMargins(0, 0, 0, 0)
        hrow = QHBoxLayout()
        self.lbl_results = QLabel()
        self.lbl_results.setWordWrap(True)
        f = self.lbl_results.font()
        f.setBold(True)
        self.lbl_results.setFont(f)
        hrow.addWidget(self.lbl_results, 1)
        self.btn_show = QPushButton(self.tr("Show in SASpro"))
        self.btn_show.setToolTip(self.tr("Bring the selected stack forward in the main window."))
        self.btn_show.clicked.connect(lambda: self._show_result())
        hrow.addWidget(self.btn_show, 0, Qt.AlignmentFlag.AlignBottom)
        rl.addLayout(hrow)
        self.list_results = QListWidget()
        self.list_results.setViewMode(QListView.ViewMode.IconMode)
        self.list_results.setFlow(QListView.Flow.LeftToRight)
        self.list_results.setWrapping(False)
        self.list_results.setMovement(QListView.Movement.Static)
        self.list_results.setIconSize(_RESULT_ICON)
        self.list_results.setGridSize(QSize(_RESULT_ICON.width() + 36, _RESULT_ICON.height() + 44))
        self.list_results.setWordWrap(True)
        self.list_results.setFixedHeight(_RESULT_ICON.height() + 66)
        self.list_results.itemActivated.connect(self._show_result)
        self.list_results.itemDoubleClicked.connect(self._show_result)
        rl.addWidget(self.list_results)
        self.results.hide()
        root.addWidget(self.results)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        root.addWidget(self.log, 1)

        brow = QHBoxLayout()
        self.btn_run = QPushButton("⚡ " + self.tr("Quick Stack"))
        f = self.btn_run.font()
        f.setBold(True)
        self.btn_run.setFont(f)
        self.btn_cancel = QPushButton(self.tr("Cancel"))
        self.btn_close = QPushButton(self.tr("Close"))
        self.btn_run.clicked.connect(self._run)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_close.clicked.connect(self.close)
        brow.addWidget(self.btn_run)
        brow.addWidget(self.btn_cancel)
        self.lbl_engine = QLabel("")
        self.lbl_engine.setToolTip(self.tr(
            "What Quick Stack will run on. A GPU is used when SASpro's hardware acceleration can reach it "
            "(NVIDIA CUDA, AMD ROCm, Intel XPU, DirectML or Apple Metal); "
            "\"Use hardware acceleration if available\" in Stacking Settings turns it off."))
        brow.addWidget(self.lbl_engine)
        brow.addStretch(1)
        brow.addWidget(self.btn_close)
        root.addLayout(brow)
        self._set_busy(False)

    # -- frame list --------------------------------------------------------
    def _last_dir(self) -> str:
        return self._settings.value(f"{_SETTINGS}/last_dir", "", type=str) or ""

    def _add_files(self):
        exts = " ".join("*" + e for e in engine.FRAME_EXTS)
        files, _ = QFileDialog.getOpenFileNames(
            self, self.tr("Add frames"), self._last_dir(),
            self.tr("Frames ({0});;All Files (*)").format(exts))
        if files:
            self._settings.setValue(f"{_SETTINGS}/last_dir", os.path.dirname(files[0]))
            self._enqueue(files)

    def _add_folder(self):
        d = QFileDialog.getExistingDirectory(self, self.tr("Add a folder of frames"), self._last_dir())
        if d:
            self._settings.setValue(f"{_SETTINGS}/last_dir", d)
            self._enqueue([d])

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths and not self._busy():
            e.acceptProposedAction()
            self._enqueue(paths)

    def _enqueue(self, paths):
        self._queue.extend(paths)
        if self._probe is None:
            self._start_probe()

    def _start_probe(self):
        paths, self._queue = self._queue, []
        known = frozenset(os.path.normcase(i.path) for i in self._infos + self._darks + self._flats
                          + self._raw_darks + self._raw_flats)
        t = _ProbeThread(paths, known)
        t.probed.connect(self._on_probed)
        t.progressed.connect(self._on_probe_progress)
        t.finished.connect(self._probe_finished)
        self._probe = t
        _keep_until_finished(t)
        self.lbl_status.setText(self.tr("Scanning…"))
        t.start()

    # Slots are bound methods, never lambdas: PyQt drops a bound-method
    # connection when its window is deleted, a lambda would outlive it.
    def _on_probe_progress(self, done, total):
        self.lbl_status.setText(self.tr("Reading headers: {0} of {1}…").format(done, total))

    def _on_probed(self, infos):
        """Lights to the list, master darks and flats to the masters, single
        dark and flat frames to be made into masters, and the rest (bias,
        flat-darks) left out and counted."""
        for info in infos:
            where = engine.route_file(info)
            if where == "light":
                self._infos.append(info)
            elif where in ("dark", "flat"):
                (self._darks if where == "dark" else self._flats).append(info)
            elif where in ("dark-frames", "flat-frames"):
                (self._raw_darks if where == "dark-frames" else self._raw_flats).append(info)
            self._scan.setdefault(where, []).append(info.path)

    def _regroup(self):
        """The masters the single dark and flat frames make."""
        self._dark_sets = engine.raw_sets(self._raw_darks, "dark")
        self._flat_sets = engine.raw_sets(self._raw_flats, "flat")

    def _dark_pool(self) -> list:
        return self._darks + self._dark_sets

    def _flat_pool(self) -> list:
        return self._flats + self._flat_sets

    # -- GPU ---------------------------------------------------------------
    def showEvent(self, e):
        super().showEvent(e)
        QTimer.singleShot(0, self._ensure_session)      # after the first paint

    def _ensure_session(self):
        if self._session is None and not self._closing:
            from setiastro.saspro import quick_stack_gpu as gpu
            self._session = gpu.GpuSession()
            self._session.start()
            self._engine_timer.start()
            self._show_engine()

    def _show_engine(self):
        s = self._session
        if s is None:
            return
        if s.state == "ready":
            self.lbl_engine.setText(self.tr("GPU: {0}").format(s.info.get("name", "")))
        elif s.state in ("failed", "closed"):
            self.lbl_engine.setText(self.tr("CPU ({0})").format(s.reason) if s.reason else self.tr("CPU"))
        else:
            self.lbl_engine.setText(self.tr("Starting the GPU…"))
            return
        self._engine_timer.stop()

    def _close_session(self):
        s, self._session = self._session, None
        self._engine_timer.stop()
        if s is not None:
            s.close()

    def _probe_finished(self):
        self._probe = None
        if self._queue:
            self._start_probe()
            return
        scan, self._scan = self._scan, {}
        self._regroup()
        self._show_masters()
        self._refresh()
        if not self._infos:
            self.lbl_status.setText(self.tr("No frames found."))
        self._report_scan(scan)

    def _report_scan(self, scan: dict):
        """Say what the files just added turned out to be, and what was left out and why."""
        n_light, n_dark, n_flat = (len(scan.get(k, ())) for k in ("light", "dark", "flat"))
        n_rdark, n_rflat = (len(scan.get(k, ())) for k in ("dark-frames", "flat-frames"))
        why = {
            "bias": self.tr("{0} bias frame(s): Quick Stack does not use bias (the master dark holds it)."),
            "darkflat": self.tr("{0} flat-dark frame(s): Quick Stack does not use them (a master flat "
                                "should already be calibrated)."),
        }
        left = [(why[k].format(len(scan[k])), scan[k]) for k in why if scan.get(k)]
        if not (n_dark or n_flat or n_rdark or n_rflat or left):
            return                          # only lights: the list says it all
        got = []
        if n_light:
            got.append(self.tr("{0} frame(s)").format(n_light))
        if n_dark:
            got.append(self.tr("{0} master dark(s)").format(n_dark))
        if n_flat:
            got.append(self.tr("{0} master flat(s)").format(n_flat))
        if n_rdark:
            got.append(self.tr("{0} dark frame(s) to make master darks from").format(n_rdark))
        if n_rflat:
            got.append(self.tr("{0} flat frame(s) to make master flats from").format(n_rflat))
        msg = self.tr("Added {0}.").format(", ".join(got)) if got else self.tr("Nothing was added.")
        if left:
            msg += " " + self.tr("Left out {0} file(s): see below.").format(sum(len(p) for _w, p in left))
        self.lbl_status.setText(msg)
        for text, paths in left:
            self.log.appendPlainText(self.tr("Left out {0}").format(text))
            for p in paths[:5]:
                self.log.appendPlainText("    " + p)
            if len(paths) > 5:
                self.log.appendPlainText("    " + self.tr("… and {0} more").format(len(paths) - 5))

    def _selected_paths(self) -> list:
        out = []

        def frames(it):                      # a group or its "N frames" row: all its frames
            p = it.data(0, _ROLE_PATH)
            if p:
                out.append(p)
            for i in range(it.childCount()):
                frames(it.child(i))
        for it in self.tree.selectedItems():
            frames(it)
        return list(dict.fromkeys(out))

    def _remove_selected(self):
        drop = {os.path.normcase(p) for p in self._selected_paths()}
        if drop:
            self._infos = [i for i in self._infos if os.path.normcase(i.path) not in drop]
            if os.path.normcase(self.ed_ref.text()) in drop:
                self.ed_ref.clear()
            self._refresh()

    def _clear(self):
        self._infos = []
        self._chosen.clear()
        self.ed_ref.clear()
        self._refresh()

    def _groups(self) -> list:
        return engine.group_frames(self._infos)

    def _tree_state(self) -> dict:
        """group key -> (group open, its frames open), to put back after a rebuild."""
        out = {}
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            node = top.child(top.childCount() - 1) if top.childCount() else None
            out[top.data(0, _ROLE_KEY)] = (top.isExpanded(), bool(node and node.isExpanded()))
        return out

    def _refresh(self):
        """Rebuild the list: each group, the dark and flat it will use, its frames."""
        was = self._tree_state()
        scroll = self.tree.verticalScrollBar().value()
        self.tree.clear()
        self._cal = {}
        groups = self._groups()
        for g in groups:
            top = QTreeWidgetItem([self.tr("{0}  —  {1} frames").format(g.label, len(g.frames))])
            top.setData(0, _ROLE_KEY, g.key)
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            self.tree.addTopLevelItem(top)
            for kind in ("dark", "flat"):
                row = QTreeWidgetItem(top)
                row.setFlags(Qt.ItemFlag.ItemIsEnabled)          # a control, not a selectable row
                row.setFirstColumnSpanned(True)
                self.tree.setItemWidget(row, 0, self._cal_row(g, kind))
            node = QTreeWidgetItem(top, [self.tr("{0} frames").format(len(g.frames))])
            for info in g.frames:
                w = f"{info.width} x {info.height}" if info.width else ""
                exp = f"{info.exposure:g} s" if info.exposure else ""
                child = QTreeWidgetItem([os.path.basename(info.path), info.filter, exp, w])
                child.setData(0, _ROLE_PATH, info.path)
                child.setToolTip(0, info.path)
                node.addChild(child)
            top_open, frames_open = was.get(g.key, (True, len(groups) == 1))
            top.setExpanded(top_open)
            node.setExpanded(frames_open)
        self.tree.verticalScrollBar().setValue(scroll)
        n = len(self._infos)
        self.lbl_count.setText(self.tr("{0} frames in {1} group(s)").format(n, len(groups)) if n else "")
        if n and not self._busy():
            self.lbl_status.setText(self.tr("Ready."))
        self._set_busy(self._busy())

    # -- a group's dark and flat --------------------------------------------
    def _cal_row(self, g, kind):
        """One group's Dark or Flat: the master it will use (Auto, a loaded
        master, None, or add one) and a badge saying why."""
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(2, 1, 6, 1)
        lay.setSpacing(8)
        names = (self.tr("Dark"), self.tr("Flat"))
        lab = QLabel(names[0] if kind == "dark" else names[1])
        lab.setMinimumWidth(max(lab.fontMetrics().horizontalAdvance(n) for n in names) + 6)
        combo = QComboBox()
        combo.setMinimumContentsLength(34)
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setToolTip(self.tr(
            "Auto picks by size, then gain and exposure (darks) or filter (flats). Pick another master, "
            "or None, for this group only.") if kind == "dark" else self.tr(
            "Auto picks by size, then filter: as written, written another way (O3 for OIII, H-alpha for "
            "Ha), or named in the file name of a flat with no filter card. Pick another master, or None, "
            "for this group only."))
        badge = QLabel()
        badge.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(lab)
        lay.addWidget(combo)
        lay.addWidget(badge, 1)
        self._cal[(g.key, kind)] = (g, combo, badge)
        self._fill_cal(g.key, kind)
        combo.setEnabled(not self._busy())
        combo.activated.connect(lambda i, key=g.key, k=kind: self._on_cal_picked(key, k, i))
        return w

    def _name(self, m, kind) -> str:
        """A master's short name: its file's, or for one made from frames, what it is made from."""
        if not engine.is_made(m):
            return os.path.basename(m.path)
        n = len(m.frames)
        if kind == "dark":
            return self.tr("{0} dark frames, {1:g} s").format(n, m.exposure)
        what = m.filter if m.filter != "Unknown" else ""
        if what and m.date:
            return self.tr("{0} {1} flat frames, {2}").format(n, what, m.date)
        if what or m.date:
            return self.tr("{0} flat frames, {1}").format(n, what or m.date)
        return self.tr("{0} flat frames").format(n)

    def _master_text(self, m, kind) -> str:
        if engine.is_made(m):
            parts = [self._name(m, kind)]
            if kind == "dark" and m.gain:
                parts.append(self.tr("gain {0:g}").format(m.gain))
            if m.width:
                parts.append(f"{m.width} x {m.height}")
            return "  ·  ".join(parts)
        parts = [os.path.basename(m.path)]
        if kind == "dark":
            if m.exposure:
                parts.append(f"{m.exposure:g} s")
            if m.gain:
                parts.append(self.tr("gain {0:g}").format(m.gain))
        else:
            if engine.filter_key(m.filter):
                parts.append(m.filter)
            elif engine.filter_from_name(m.path):
                key = engine.filter_from_name(m.path)
                parts.append(self.tr("{0} (from its name)").format(engine.FILTER_LABEL.get(key, key)))
            else:
                parts.append(self.tr("no filter"))
        if m.width:
            parts.append(f"{m.width} x {m.height}")
        return "  ·  ".join(parts)

    def _fill_cal(self, key, kind):
        g, combo, badge = self._cal[(key, kind)]
        frame = g.frames[0]
        pool = self._dark_pool() if kind == "dark" else self._flat_pool()
        chosen = self._chosen.get(key, {})
        auto = engine.plan_masters(frame, self._dark_pool(), self._flat_pool())[kind]
        combo.blockSignals(True)
        combo.clear()
        if not pool:
            combo.addItem(self.tr("None loaded"), _AUTO)
        else:
            combo.addItem(self.tr("Auto: {0}").format(self._name(auto.master, kind)) if auto.master
                          else self.tr("Auto: none fits"), _AUTO)
            combo.insertSeparator(combo.count())
            for m in pool:
                combo.addItem(self._master_text(m, kind), m.path)
                if not engine.same_size(m, frame):
                    i = combo.count() - 1
                    combo.model().item(i).setEnabled(False)
                    combo.setItemData(i, self.tr("{0} x {1}: these frames are {2} x {3}").format(
                        m.width, m.height, frame.width, frame.height), Qt.ItemDataRole.ToolTipRole)
            combo.addItem(self.tr("None"), _NONE)
        combo.insertSeparator(combo.count())
        combo.addItem(self.tr("Add a master dark or dark frames…") if kind == "dark"
                      else self.tr("Add a master flat or flat frames…"), _BROWSE)
        i = 0
        if kind in chosen:
            i = max(0, combo.findData(_NONE if chosen[kind] is None else chosen[kind]))
        combo.setCurrentIndex(i)
        combo.blockSignals(False)
        self._show_badge(key, kind)

    def _show_badge(self, key, kind):
        g, _combo, badge = self._cal[(key, kind)]
        frame = g.frames[0]
        p = engine.plan_masters(frame, self._dark_pool(), self._flat_pool(), self._chosen.get(key))[kind]
        parts = []
        m = p.master
        if p.state == "auto":
            parts.append(("ok", self.tr("✓ auto")))
        elif p.state == "name":
            parts.append(("info", self.tr("≈ {0} = {1}").format(m.filter, frame.filter)))
        elif p.state == "file":
            parts.append(("info", self.tr("≈ {0}, from its file name").format(frame.filter)))
        elif p.state == "only":
            parts.append(("ok", self.tr("✓ the only flat")))
        elif p.state == "hand":
            parts.append(("info", self.tr("✎ your choice")))
        elif p.state == "hand-none":
            parts.append(("info", self.tr("✎ none")))
        elif p.state == "none":
            if p.why == "size":
                parts.append(("warn", self.tr("⚠ none is {0} x {1}").format(frame.width, frame.height)))
            elif p.why == "exposure":
                parts.append(("warn", self.tr("⚠ the frames give no exposure: choose one")))
            else:
                parts.append(("warn", self.tr("⚠ no {0} flat").format(frame.filter)))
        if m is not None:
            if not engine.same_size(m, frame):
                parts.append(("warn", self.tr("⚠ {0} x {1}, not these frames' size: it will be left out")
                              .format(m.width, m.height)))
            if kind == "dark":
                if m.exposure and frame.exposure and abs(m.exposure - frame.exposure) > 0.05 * frame.exposure:
                    parts.append(("warn", self.tr("⚠ {0:g} s dark, {1:g} s frames").format(
                        m.exposure, frame.exposure)))
                elif frame.exposure and not m.exposure:
                    parts.append(("warn", self.tr("⚠ its exposure is not recorded")))
                if m.gain and frame.gain and m.gain != frame.gain:
                    parts.append(("warn", self.tr("⚠ gain {0:g}, frames {1:g}").format(m.gain, frame.gain)))
            else:
                fk, mk = engine.filter_key(frame.filter), engine.flat_filter_key(m)
                if fk and mk and fk != mk:
                    said = m.filter if engine.filter_key(m.filter) else engine.FILTER_LABEL.get(mk, mk)
                    if p.state == "only":
                        parts = []
                    parts.append(("warn", self.tr("⚠ it is the {0} flat").format(said)))
        if m is not None and engine.is_made(m):
            parts.append(("info", self.tr("made in memory from {0} frames").format(len(m.frames))))
        badge.setText("  ".join(f'<span style="color:{_BADGE_COLOUR[c]}">{html.escape(t)}</span>'
                                for c, t in parts))

    def _on_cal_picked(self, key, kind, index):
        if (key, kind) not in self._cal:
            return
        _g, combo, _badge = self._cal[(key, kind)]
        data = combo.itemData(index)
        if data == _BROWSE:
            # The list is rebuilt after, never inside, the signal of a combo it deletes.
            added = self._browse_masters(kind, rebuild=False)
            if added:
                self._chosen.setdefault(key, {})[kind] = added[0]
            QTimer.singleShot(0, self._refresh)
            return
        ch = self._chosen.setdefault(key, {})
        if data == _NONE:
            ch[kind] = None
        elif data == _AUTO or data is None:
            ch.pop(kind, None)
        else:
            ch[kind] = data
        self._show_badge(key, kind)

    # -- masters -----------------------------------------------------------
    def _browse_masters(self, kind, rebuild: bool = True) -> list:
        """Add masters from a file dialog. kind "dark"/"flat": the files are
        that (a group's "Add a master …"); None: each says what it is."""
        exts = " ".join("*" + e for e in engine.FRAME_EXTS)
        start = self._settings.value(f"{_SETTINGS}/last_master_dir", "", type=str) or self._last_dir()
        title = {"dark": self.tr("Choose a master dark or dark frames"),
                 "flat": self.tr("Choose a master flat or flat frames")}.get(
            kind, self.tr("Choose master darks and flats, or their frames"))
        files, _ = QFileDialog.getOpenFileNames(
            self, title, start, self.tr("Darks and flats ({0});;All Files (*)").format(exts))
        if not files:
            return []
        self._settings.setValue(f"{_SETTINGS}/last_master_dir", os.path.dirname(files[0]))
        return self.add_masters(kind, files, rebuild=rebuild)

    def add_masters(self, kind, files, rebuild: bool = True) -> list:
        """Add masters, or single frames to make them from. kind "dark"/"flat"
        says which they are; None reads it from each (IMAGETYP, else the file
        or folder name) and asks about one that does not say. A file that is
        not a master (integer pixels, no NCOMBINE, no "master" in its name)
        joins the frames its master is made from. Returns, in order, what the
        files are now part of: a master's own path, a frame's set's name."""
        out, left, frames = [], [], []
        known = {os.path.normcase(i.path): i for i in self._darks + self._flats}
        raw_known = {os.path.normcase(i.path) for i in self._raw_darks + self._raw_flats}
        for f in files:
            f = os.path.normpath(f)
            if os.path.normcase(f) in known:
                out.append(known[os.path.normcase(f)].path)
                continue
            if os.path.normcase(f) in raw_known:
                frames.append(f)
                out.append(f)
                continue
            info = engine.probe_frame(f)
            what = kind or engine.route_file(info, as_master=True)
            if what in ("bias", "darkflat"):
                left.append(info.path)
                continue
            if what not in ("dark", "flat"):
                what = self._ask_kind(info)
                if what is None:
                    continue
            if engine.is_master(info):
                (self._darks if what == "dark" else self._flats).append(info)
                known[os.path.normcase(f)] = info
            else:
                (self._raw_darks if what == "dark" else self._raw_flats).append(info)
                raw_known.add(os.path.normcase(f))
                frames.append(info.path)
            out.append(info.path)
            drop = os.path.normcase(f)              # added as a light before: it is calibration now
            self._infos = [i for i in self._infos if os.path.normcase(i.path) != drop]
        self._regroup()
        if frames:                                  # a frame stands for the master its set makes
            set_of = {os.path.normcase(p): s.path for s in self._dark_sets + self._flat_sets
                      for p in s.frames}
            out = [set_of.get(os.path.normcase(p), p) for p in out]
        out = list(dict.fromkeys(out))
        if left:
            self.lbl_status.setText(self.tr("Left out {0} bias / flat-dark file(s): Quick Stack does not "
                                            "use them.").format(len(left)))
        self._show_masters()
        if rebuild:
            self._refresh()
        return out

    def _ask_kind(self, info):
        """'dark', 'flat' or None (skip) for a master that does not say which it is."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(self.tr("Quick Stack"))
        box.setText(self.tr("Is {0} a master dark or a master flat?").format(os.path.basename(info.path)))
        box.setInformativeText(self.tr("It has no IMAGETYP card, and its file name does not say."))
        b_dark = box.addButton(self.tr("Master Dark"), QMessageBox.ButtonRole.AcceptRole)
        b_flat = box.addButton(self.tr("Master Flat"), QMessageBox.ButtonRole.AcceptRole)
        box.addButton(self.tr("Skip"), QMessageBox.ButtonRole.RejectRole)
        box.exec()
        return {b_dark: "dark", b_flat: "flat"}.get(box.clickedButton())

    def _clear_masters(self):
        self._darks.clear()
        self._flats.clear()
        self._raw_darks.clear()
        self._raw_flats.clear()
        self._regroup()
        engine.forget_made()
        for ch in self._chosen.values():            # a "None" choice still holds
            for k in [k for k, v in ch.items() if v is not None]:
                del ch[k]
        self._show_masters()
        self._refresh()

    def _show_masters(self):
        darks, flats = self._dark_pool(), self._flat_pool()
        nd, nf = len(darks), len(flats)
        if not (nd or nf):
            self.lbl_masters.setText(self.tr("none (optional: drop or add master darks and flats, or "
                                             "their frames)"))
            self.lbl_masters.setToolTip(self._masters_tip)
        else:
            self.lbl_masters.setText(self.tr("{0} dark(s)  ·  {1} flat(s): each group shows the ones it "
                                             "uses").format(nd, nf))
            lines = [self._master_text(m, "dark") for m in darks] + \
                    [self._master_text(m, "flat") for m in flats]
            self.lbl_masters.setToolTip("\n".join(lines))
        self.btn_masters_clear.setEnabled(bool(nd or nf) and not self._busy())

    # -- reference ---------------------------------------------------------
    def _choose_ref(self):
        exts = " ".join("*" + e for e in engine.FRAME_EXTS)
        f, _ = QFileDialog.getOpenFileName(
            self, self.tr("Choose a reference frame"), self._last_dir(),
            self.tr("Frames ({0});;All Files (*)").format(exts))
        if f:
            self.ed_ref.setText(os.path.normpath(f))

    def _ref_from_selection(self):
        sel = [p for p in self._selected_paths()]
        if sel:
            self.ed_ref.setText(os.path.normpath(sel[0]))

    # -- running -----------------------------------------------------------
    def _busy(self) -> bool:
        return self._stack is not None

    def _set_busy(self, busy: bool):
        for b in (self.btn_files, self.btn_folder, self.btn_remove, self.btn_clear,
                  self.btn_ref, self.btn_ref_sel, self.btn_ref_clear, self.btn_masters):
            b.setEnabled(not busy)
        self.btn_masters_clear.setEnabled(not busy and bool(self._dark_pool() or self._flat_pool()))
        for _g, combo, _badge in self._cal.values():
            combo.setEnabled(not busy)
        self.btn_run.setEnabled(not busy and len(self._infos) > 1)
        self.btn_cancel.setEnabled(busy)

    def _run(self):
        if self._busy() or self._probe is not None:
            return
        groups = self._groups()
        too_small = [g for g in groups if len(g.frames) < 2]
        groups = [g for g in groups if len(g.frames) >= 2]
        if not groups:
            QMessageBox.information(self, self.tr("Quick Stack"),
                                    self.tr("Add at least two matching frames (same filter, exposure and size)."))
            return
        self.log.clear()
        self.list_results.clear()
        self.results.hide()
        for g in too_small:
            self.log.appendPlainText(self.tr("Skipping {0}: only one frame").format(g.label))
        self._ensure_session()
        t = _StackThread(groups, self.ed_ref.text().strip() or None, session=self._session,
                         darks=self._dark_pool(), flats=self._flat_pool(), chosen=self._chosen)
        t.log_line.connect(self.log.appendPlainText)
        t.step.connect(self._on_step)
        t.stacked.connect(self._on_stacked)
        t.group_failed.connect(self._on_failed)
        t.all_done.connect(self._on_done)
        self._stack = t
        _keep_until_finished(t)
        self._set_busy(True)
        self.bar.setRange(0, 0)
        self.lbl_status.setText(self.tr("Starting…"))
        t.start()

    def _cancel(self):
        if self._stack is not None:
            self._stack.cancel()
            self.lbl_status.setText(self.tr("Cancelling…"))

    def _on_step(self, done, total, msg):
        if self.bar.maximum() != total:
            self.bar.setRange(0, total)
        self.bar.setValue(done)
        self.lbl_status.setText(msg)

    def _on_stacked(self, r, group):
        dm = self._docman
        if dm is None:
            mw = self.parent()
            dm = getattr(mw, "docman", None) or getattr(mw, "doc_manager", None)
        info = group.frames[0]
        parts = ["QuickStack"]
        if info.obj:
            parts.append(info.obj)
        if info.filter and info.filter != "Unknown":
            parts.append(info.filter)
        parts.append(f"{r.used}x{info.exposure:g}s" if info.exposure else f"{r.used}f")
        title = "_".join("".join(c if (c.isalnum() or c in "-+.") else "-" for c in p).strip("-") for p in parts)
        meta = {"source": "QuickStack", "frames_stacked": int(r.used),
                "bit_depth": "32-bit floating point", "is_mono": r.image.ndim == 2,
                "original_header": r.header}
        if dm is None or not hasattr(dm, "open_array"):
            self.log.appendPlainText(self.tr("No document manager is available to open {0}.").format(title))
            return
        try:
            # A finished stack is new data, like a file opened: still linear,
            # so the main window shows it display-stretched, not near-black.
            opening = getattr(dm, "opening_new_data", None)
            if opening is None:
                doc = dm.open_array(r.image, metadata=meta, title=title)
            else:
                with opening():
                    doc = dm.open_array(r.image, metadata=meta, title=title)
        except Exception as e:
            QMessageBox.warning(self, self.tr("Open Failed"),
                                self.tr("Could not open {0}:\n{1}").format(title, e))
            return
        self._add_result(doc, title, r)

    def _add_result(self, doc, title, r):
        """List a stack that has just opened in the main window."""
        name = title
        try:
            name = doc.display_name() or title
        except Exception:
            pass
        h, w = r.image.shape[:2]
        tip = [self.tr("{0}: {1} of {2} frames, {3} x {4}").format(name, r.used, r.total, w, h),
               self.tr("Open in SASpro, not saved yet. Double-click to bring it forward.")]
        if name != title:
            # The window list names the second of two equal titles title_1.
            tip.append(self.tr("A stack named {0} was already open; this one is {1}.").format(title, name))
            self.log.appendPlainText(self.tr("Opened as {0}: a stack named {1} is already open.")
                                     .format(name, title))
        item = QListWidgetItem(self.tr("{0}\n{1} frames").format(name, r.used))
        item.setToolTip("\n".join(tip))
        try:
            item.setData(_ROLE_DOC, weakref.ref(doc))
        except TypeError:
            item.setData(_ROLE_DOC, lambda d=doc: d)
        try:
            from setiastro.saspro.start_panel import render_thumbnail
            qi = render_thumbnail(r.image)
            if qi is not None:
                pm = QPixmap.fromImage(qi)
                pm.setDevicePixelRatio(2.0)          # it is drawn at twice the Start panel's size
                item.setIcon(QIcon(pm))
        except Exception:
            pass
        self.list_results.addItem(item)
        self.list_results.setCurrentItem(item)
        n = self.list_results.count()
        self.lbl_results.setText(
            self.tr("✓ Opened in SASpro as a new image, not saved yet:") if n == 1 else
            self.tr("✓ Opened in SASpro as {0} new images, not saved yet:").format(n))
        self.results.show()

    def _main_window(self):
        p = self.parent()
        while p is not None and not hasattr(p, "mdi"):
            p = p.parent()
        return p

    def _show_result(self, item=None):
        """Bring a stack's view forward in the main window."""
        item = item or self.list_results.currentItem()
        if item is None and self.list_results.count():
            item = self.list_results.item(self.list_results.count() - 1)
        if item is None:
            return
        ref = item.data(_ROLE_DOC)
        doc = ref() if callable(ref) else None
        mw = self._main_window()
        sw = None
        if doc is not None and mw is not None and hasattr(mw, "_find_subwindow_for_doc"):
            try:
                sw = mw._find_subwindow_for_doc(doc)
            except Exception:
                sw = None
        if sw is None:
            self.lbl_status.setText(self.tr("{0} is no longer open in SASpro.")
                                    .format(item.text().split("\n")[0]))
            return
        if mw.isMinimized():
            mw.showNormal()
        mw.raise_()
        mw.activateWindow()
        if sw.isMinimized():
            sw.showNormal()

        def activate():
            try:
                mw.mdi.setActiveSubWindow(sw)
                sw.raise_()
            except RuntimeError:            # closed in the meantime
                pass
        # Now, and on the next tick as the main window does for a new view: a
        # window that is being activated puts back the view that was active
        # when it lost focus.
        activate()
        QTimer.singleShot(0, activate)

    def _on_failed(self, label, msg):
        QMessageBox.warning(self, self.tr("Quick Stack"), f"{label}\n\n{msg}")

    def _on_done(self, ok, total):
        cancelled = self._stack is not None and self._stack.cancelled()
        self._stack = None
        self.bar.setRange(0, 1)
        self.bar.setValue(0 if cancelled else 1)
        self._set_busy(False)
        if cancelled:
            self.lbl_status.setText(self.tr("Cancelled."))
        elif ok:
            self.lbl_status.setText(self.tr("Done: {0} of {1} group(s) stacked. The results are open in "
                                            "SASpro, behind this window.").format(ok, total))
        else:
            self.lbl_status.setText(self.tr("Done: {0} of {1} group(s) stacked.").format(ok, total))

    # -- closing -----------------------------------------------------------
    def _shut_down(self):
        self._settings.setValue(f"{_SETTINGS}/geometry", self.saveGeometry())
        self._closing = True
        for t in (self._probe, self._stack):
            if t is not None:
                t.cancel()          # _LIVE_THREADS keeps it alive until it finishes
        self._close_session()
        engine.forget_made()        # the masters made from frames (memory)

    def reject(self):
        # Esc and Close land here; Qt does not run closeEvent on this path.
        self._shut_down()
        super().reject()

    def closeEvent(self, e):
        self._shut_down()
        super().closeEvent(e)
