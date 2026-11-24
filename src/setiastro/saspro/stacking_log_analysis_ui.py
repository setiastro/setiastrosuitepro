# -----------------------------------------------------------------------------
# SetiAstro Suite Pro - Stacking Log Analysis tab (Beta)
# Copyright (c) 2026 Franklin Marek / www.setiastro.com
# Licensed under GPLv3.
# -----------------------------------------------------------------------------
"""The Log Analysis tab.

Read-only, and deliberately decoupled from live pipeline state: it takes a
file path and nothing else, so it will open last week's log, a log copied off
another machine, or one a user pastes into a support thread. The stacking
directory is only used to populate the picker with a sensible default list.

Parsing lives in ``stacking_log_analysis`` and has no Qt dependency; this
module is presentation only.
"""

from __future__ import annotations

import os
import threading

from PyQt6.QtCore import QT_TRANSLATE_NOOP, QCoreApplication, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QFontDatabase, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QSizePolicy, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from setiastro.saspro.stacking_log_analysis import (
    SEV_ERROR, SEV_WARN, fmt_hms, find_stacking_logs, frame_reconciliation,
    frame_stems, master_outputs, parse_stacking_log,
    rebuild_acquisition_summary, run_stems, slowest_steps,
)

__all__ = ["StackingLogAnalysisTab"]


def _tr(text: str) -> str:
    return QCoreApplication.translate("StackingLogAnalysisUi", text)


def _models_text(models) -> str:
    return ", ".join(f"{name} {n}" for name, n in models) if models else "—"


def _afn_readout(a) -> str:
    """Plain-language account of what AFN measured and did for one group."""
    esc = lambda s: (str(s).replace("&", "&amp;").replace("<", "&lt;")
                     .replace(">", "&gt;"))
    out = [f"<b>{esc(a.group)}</b>"]
    if a.status == "skipped":
        out.append(_tr("<span style='color:#c9913a'>Skipped: {0}.</span> "
                       "This group was integrated exactly as it would be without AFN."
                       ).format(esc(a.skip_reason)))
        return "<br>".join(out)
    if a.status != "ready":
        out.append(_tr("<span style='color:#c9913a'>The log ends before AFN finished "
                       "for this group.</span>"))

    # what it measured
    if a.stars is not None:
        out.append(_tr(
            "<b>Measured:</b> {0} stars (of {1} detected) in "
            "every frame, FWHM {2:.2f} px, aperture radius {3:.1f} px "
            "with a sky annulus of {4} px; sky in a {5} grid "
            "of {6} px cells.").format(
                a.stars, a.detections, a.fwhm, a.aperture,
                esc(a.annulus), esc(a.grid), a.cell))
    elif a.meas_cached:
        out.append(_tr("<b>Measured:</b> reused the cached measurements from an "
                       "earlier run of the same frames."))

    # flip
    if a.balanced:
        out.append(_tr(
            "<b>Meridian flip:</b> frames came from both sides of the flip "
            "({0}). The reference was balanced between them, so a "
            "flat error that flips with the mount cancels instead of being "
            "copied to every frame.").format(esc(a.balanced)))

    # brightness
    const = a.model_share("constant")
    if a.star_gain_median is not None:
        if (const is not None and const >= 0.75
                and a.star_gain_median < 1.0):
            out.append(_tr(
                "<b>Brightness:</b> the frames already agreed. "
                "{0}% of them needed only a single overall "
                "scale, so AFN made almost no spatial correction here."
            ).format(int(round(const * 100))))
        else:
            head = (_tr("<b>Brightness:</b> a small correction.")
                    if a.star_gain_median < 2 else
                    _tr("<b>Brightness:</b> a moderate correction.")
                    if a.star_gain_median < 10 else
                    _tr("<b>Brightness:</b> a large correction."))
            out.append(head + " " + _tr(
                "Frames differed across the "
                "field by a median of {0:.1f}% (most: "
                "{1:.1f}%). The maps predicted stars they were not "
                "fitted to {2:.1f}% better than a single overall "
                "scale; {3:.1f}% for the most-corrected tenth of "
                "frames.").format(a.field_median, a.field_max,
                                  a.star_gain_median, a.star_gain_p90))
        out.append(_tr("&nbsp;&nbsp;Map detail chosen per frame: "
                       "{0}.").format(esc(_models_text(a.bright_models))))

    # sky
    if a.sky_gain_median is not None:
        out.append(_tr(
            "<b>Sky:</b> differences in sky glow between frames were matched; "
            "the sky maps predicted sky they were not fitted to "
            "{0:.1f}% better than one overall offset.").format(a.sky_gain_median))
        out.append(_tr("&nbsp;&nbsp;Map detail chosen per frame: "
                       "{0}.").format(esc(_models_text(a.sky_models))))

    if a.clamped:
        out.append(_tr("{0} map(s) were held to the range their stars "
                       "support where they would otherwise have extrapolated "
                       "(normal near frame edges).").format(a.clamped))
    if a.maps_cached:
        out.append(_tr("The maps themselves were reused from cache."))

    # cost
    bits = []
    if a.seconds is not None:
        bits.append(_tr("took {0}").format(fmt_hms(a.seconds)))
    if a.t_measure is not None:
        bits.append(_tr("measuring {0:.1f} s, fitting {1:.1f} s").format(
            a.t_measure, a.t_fit))
    if a.processes:
        bits.append(_tr("across {0} processes").format(a.processes))
    if bits:
        out.append(_tr("<b>Cost:</b> {0}.").format(", ".join(bits)))
    if a.drizzle:
        out.append(_tr("<b>Drizzle:</b> the same maps were applied as each original "
                       "frame was deposited — {0}.").format(esc(a.drizzle)))
    for n in a.notes:
        out.append(f"<span style='color:#c9913a'>{esc(n)}</span>")
    return "<br>".join(out)

_PHASE_COLORS = [
    "#4F83CC", "#58A55C", "#C9913A", "#9B59B6",
    "#D9534F", "#17A2B8", "#8D6E63", "#7F8C8D",
]


class _NumItem(QTableWidgetItem):
    """Table cell that sorts by a number while displaying formatted text.

    QTableWidgetItem's own comparison looks at the DISPLAY string, so a column
    of "25", "107", "694" would sort as text and put 107 before 25. Holding the
    value separately and overriding __lt__ keeps the formatting and gets a true
    numeric order. Missing values sort to the bottom either way.
    """

    __slots__ = ()

    def __init__(self, text, value):
        super().__init__(text)
        self.setData(Qt.ItemDataRole.UserRole, value)

    def __lt__(self, other):
        a = self.data(Qt.ItemDataRole.UserRole)
        b = (other.data(Qt.ItemDataRole.UserRole)
             if isinstance(other, QTableWidgetItem) else None)
        if a is None and b is None:
            return False
        if a is None:
            return False            # None is "largest": stays at the bottom
        if b is None:
            return True
        try:
            return float(a) < float(b)
        except (TypeError, ValueError):
            return super().__lt__(other)


def _item(text, *, align=None, bold=False, fg=None, sort_value=None):
    it = QTableWidgetItem("" if text is None else str(text))
    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if align is not None:
        it.setTextAlignment(align | Qt.AlignmentFlag.AlignVCenter)
    if bold:
        f = it.font()
        f.setBold(True)
        it.setFont(f)
    if fg is not None:
        it.setForeground(QBrush(QColor(fg)))
    if sort_value is not None:
        it.setData(Qt.ItemDataRole.UserRole, sort_value)
    return it


def _fill(table: QTableWidget, headers, rows, *, stretch_col=None):
    table.clear()
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(list(headers))
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            table.setItem(r, c, cell if isinstance(cell, QTableWidgetItem)
                          else _item(cell))
    table.resizeColumnsToContents()
    if stretch_col is not None and 0 <= stretch_col < len(headers):
        table.horizontalHeader().setSectionResizeMode(
            stretch_col, QHeaderView.ResizeMode.Stretch)


class _PhaseBar(QWidget):
    """One horizontal proportional bar of the run's phases."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._segments = []           # (label, start, dur, color)
        self._total = 0.0
        self.setMinimumHeight(54)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)

    def set_segments(self, segments, total):
        self._segments = list(segments or [])
        self._total = float(total or 0.0)
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        w, h = self.width(), self.height()
        bar_h = 24
        y = 6
        if self._total <= 0 or not self._segments:
            p.setPen(QPen(QColor("#888")))
            p.drawText(0, 0, w, h, Qt.AlignmentFlag.AlignCenter,
                       self.tr("no phase markers in this log"))
            return
        for label, start, dur, color in self._segments:
            if dur is None or dur <= 0:
                continue
            x0 = int(w * (start / self._total))
            x1 = int(w * ((start + dur) / self._total))
            rw = max(1, x1 - x0)
            p.fillRect(x0, y, rw, bar_h, QColor(color))
            if rw > 46:
                p.setPen(QPen(QColor("#ffffff")))
                f = p.font()
                f.setPointSize(max(7, f.pointSize() - 1))
                p.setFont(f)
                p.drawText(x0 + 4, y, rw - 8, bar_h,
                           Qt.AlignmentFlag.AlignVCenter
                           | Qt.AlignmentFlag.AlignLeft,
                           f"{label} {fmt_hms(dur)}")
        p.setPen(QPen(QColor("#999")))
        p.drawText(0, y + bar_h + 2, w, h - bar_h - 8,
                   Qt.AlignmentFlag.AlignLeft,
                   self.tr("total {0}").format(fmt_hms(self._total)))


def _read_frame_headers(runs, roots, cancelled) -> dict:
    """{stem: (path, header)} for the frames a log's runs name, from wherever
    the run's folder still has them. No Qt in here: it runs in a thread,
    because a 700-frame run is several seconds of disk."""
    from concurrent.futures import ThreadPoolExecutor
    from astropy.io import fits

    need = {s for run in runs for s in run_stems(run)}
    if not need:
        return {}

    # Calibrated first: the same cards as the later stages, and the frames
    # people keep. Only these folders are listed, never a whole drive.
    exact, copies = {}, {}
    for root in dict.fromkeys(os.path.normpath(r) for r in roots if r):
        subs = [os.path.join(root, "Calibrated"), root,
                os.path.join(root, "Normalized_Images"),
                os.path.join(root, "Aligned_Images")]
        try:
            subs += [e.path for e in os.scandir(subs[-1]) if e.is_dir()]
        except OSError:
            pass
        for sub in subs:
            try:
                names = os.listdir(sub)
            except OSError:
                continue
            for n in names:
                stem, ext = os.path.splitext(n)
                if ext.lower() not in (".fit", ".fits", ".fts"):
                    continue
                exact.setdefault(stem, os.path.join(sub, n))
                for s in frame_stems(n)[:-1]:     # a later stage of frame s
                    copies.setdefault(s, os.path.join(sub, n))

    paths = {}
    for s in need:
        # the frame itself, what it was made from, or a later copy of it
        stems = [s, *frame_stems(s)]
        hit = next((d[x] for d in (exact, copies) for x in stems if x in d), None)
        if hit is not None:
            paths[s] = hit

    def read(stem):
        if cancelled():
            return stem, None
        try:
            return stem, fits.getheader(paths[stem])
        except Exception:
            return stem, None

    with ThreadPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(read, sorted(paths)))
    return {s: (paths[s], h) for s, h in got if h is not None}


class StackingLogAnalysisTab(QWidget):
    """Pick a stacking log, parse it, and show what the run actually did."""

    _acq_headers = pyqtSignal(int, object)      # (job, _read_frame_headers result)

    def __init__(self, suite=None, parent=None):
        super().__init__(parent)
        self._suite = suite
        self._report = None
        self._build()
        QTimer.singleShot(0, self.refresh_log_list)

    # ── construction ────────────────────────────────────────────────────
    def _build(self):
        root = QVBoxLayout(self)

        banner = QLabel(self.tr(
            "<b>Log Analysis (Beta)</b> — reads a stacking log and "
            "summarises the run. The stacker writes a great deal of free text "
            "and this parses it with pattern rules, so a message it does not "
            "recognise is reported under <i>Coverage</i> rather than silently "
            "dropped."))
        banner.setWordWrap(True)
        banner.setStyleSheet("color:#aaa; padding:2px 0 4px 0;")
        root.addWidget(banner)

        # picker row
        row = QHBoxLayout()
        row.addWidget(QLabel(self.tr("Log:")))
        self.cmb_logs = QComboBox()
        self.cmb_logs.setMinimumWidth(380)
        self.cmb_logs.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_logs.activated.connect(self._on_pick)
        row.addWidget(self.cmb_logs, 1)

        self.btn_rescan = QPushButton(self.tr("Rescan folder"))
        self.btn_rescan.setToolTip(self.tr(
            "Re-list the stacking logs in the current stacking directory."))
        self.btn_rescan.clicked.connect(self.refresh_log_list)
        row.addWidget(self.btn_rescan)

        self.btn_browse = QPushButton(self.tr("Open log…"))
        self.btn_browse.setToolTip(self.tr(
            "Open any stacking log, from any folder or machine."))
        self.btn_browse.clicked.connect(self._on_browse)
        row.addWidget(self.btn_browse)

        self.btn_reload = QPushButton(self.tr("Reload"))
        self.btn_reload.setToolTip(self.tr(
            "Re-read the current log. Useful while a run is still writing it."))
        self.btn_reload.clicked.connect(self._reload)
        row.addWidget(self.btn_reload)
        root.addLayout(row)

        self.lbl_head = QLabel("")
        self.lbl_head.setWordWrap(True)
        self.lbl_head.setStyleSheet("padding:2px 0;")
        root.addWidget(self.lbl_head)

        self.inner = QTabWidget()
        root.addWidget(self.inner, 1)

        self._build_summary()
        self._build_acq_summary()
        self._build_timeline()
        self._build_problems()
        self._build_groups()
        self._build_afn()
        self._build_masters()
        self._build_frames()
        self._build_raw()
        self._build_coverage()
        self.inner.currentChanged.connect(self._ensure_acq)

    def _mk_table(self, sortable=False):
        t = QTableWidget()
        t.setAlternatingRowColors(True)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.verticalHeader().setVisible(False)
        t.setSortingEnabled(sortable)
        return t

    def _build_summary(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        split = QSplitter(Qt.Orientation.Horizontal)

        gb_f = QGroupBox(self.tr("Run settings && facts"))
        lf = QVBoxLayout(gb_f)
        self.tbl_facts = self._mk_table()
        lf.addWidget(self.tbl_facts)
        split.addWidget(gb_f)

        right = QWidget()
        lr = QVBoxLayout(right)
        gb_m = QGroupBox(self.tr("Pipeline stages exercised"))
        lm = QVBoxLayout(gb_m)
        self.lbl_modes = QLabel("")
        self.lbl_modes.setWordWrap(True)
        lm.addWidget(self.lbl_modes)
        lr.addWidget(gb_m)

        gb_r = QGroupBox(self.tr("Frame reconciliation"))
        lrr = QVBoxLayout(gb_r)
        self.tbl_recon = self._mk_table()
        lrr.addWidget(self.tbl_recon)
        lr.addWidget(gb_r, 1)
        split.addWidget(right)
        split.setSizes([520, 520])
        lay.addWidget(split)
        self.inner.addTab(w, self.tr("Summary"))

    def _build_acq_summary(self):
        """The copyable Acquisition Summary from the end of the run: exactly
        as the completion dialog showed it when the stacker logged it
        (SUMMARY_BEGIN in stacking_log_analysis), rebuilt for older logs."""
        w = self._acq_tab = QWidget()
        lay = QVBoxLayout(w)
        top = QHBoxLayout()
        self.lbl_acq = QLabel("")
        self.lbl_acq.setWordWrap(True)
        top.addWidget(self.lbl_acq, 1)
        # Only shown when one log holds several finished runs.
        self.cmb_acq = QComboBox()
        self.cmb_acq.setVisible(False)
        self.cmb_acq.currentIndexChanged.connect(self._show_acq)
        top.addWidget(self.cmb_acq)
        lay.addLayout(top)

        self.txt_acq = QPlainTextEdit()
        self.txt_acq.setReadOnly(True)
        self.txt_acq.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        # The completion dialog's font, so the columns line up the same way.
        self.txt_acq.setFont(
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        lay.addWidget(self.txt_acq, 1)

        row = QHBoxLayout()
        self.btn_acq_copy = QPushButton(self.tr("📋  Copy Summary"))
        self.btn_acq_copy.clicked.connect(self._copy_acq)
        row.addWidget(self.btn_acq_copy)
        row.addStretch(1)
        lay.addLayout(row)
        self._acq = []                  # (clock, text, source) per run
        self._acq_why = ""
        self._acq_todo = None           # report still to be rebuilt
        self._acq_job = 0               # newest header-reading job
        self._acq_headers.connect(self._on_acq_headers)
        self.inner.addTab(w, self.tr("Acquisition Summary"))

    def _ensure_acq(self, *_):
        """Rebuild the summary the first time its tab is looked at: it reads
        frame headers off disk, and the newest log is loaded whenever the
        Stacking Suite opens, which that must not slow."""
        rep = self._acq_todo
        if rep is None or self.inner.currentWidget() is not self._acq_tab:
            return
        self._acq_todo = None
        self._acq_job += 1
        job, runs = self._acq_job, list(rep.runs)
        if not runs:
            self._fill_acq()
            return
        self._acq_why = self.tr("Reading the headers of this run's frames…")
        self._fill_acq()
        # The run's folder, and the log's own (the same one, unless it has
        # since been moved).
        roots = [os.path.dirname(rep.path), rep.directory]
        done = self._acq_headers

        def work():
            try:
                headers = _read_frame_headers(
                    runs, roots, lambda: self._acq_job != job)
            except Exception as e:
                print(f"[log analysis] frame headers: {e!r}")
                headers = {}
            try:
                done.emit(job, (runs, headers))
            except RuntimeError:                 # the suite was closed
                pass

        threading.Thread(target=work, name="acq-headers", daemon=True).start()

    def _on_acq_headers(self, job, result):
        if job != self._acq_job:                 # another log was opened
            return
        runs, headers = result
        self._acq, self._acq_why = [], ""
        try:
            from setiastro.saspro.acquisition_summary import (
                format_acquisition_summary)
            frames = self._frame_records(headers)
            # The suite's own context, so the headings translate the same way.
            tr = self._suite.tr if self._suite is not None else self.tr
            for run in runs:
                data, notes, source = rebuild_acquisition_summary(run, frames)
                if data is None:
                    self._acq_why = notes
                    continue
                text = format_acquisition_summary(data, tr=tr, notes=notes)["text"]
                self._acq.append((run.clock, text,
                                  source + ("+notes" if notes else "")))
        except Exception as e:
            self._acq_why = self.tr("Couldn't rebuild the summary from this "
                                    "log: {0}").format(e)
        self._fill_acq()

    def _frame_records(self, headers) -> dict:
        """{stem: record} for rebuild_acquisition_summary, with the stacker's
        own rules for nights, sessions and group names."""
        suite = self._suite
        if suite is None:
            return {}
        from setiastro.saspro.acquisition_summary import header_fields
        out = {}
        for stem, (path, h) in headers.items():
            try:
                r = header_fields(h)
                axes = [int(h.get(f"NAXIS{i}", 0) or 0) for i in (1, 2, 3)]
                w, ht = ((axes[1], axes[2]) if axes[0] == 3 else (axes[0], axes[1]))
                r["size"] = f"{w}x{ht}"
                r["group"] = suite._sanitize_name(r["filter"])
                r["night"] = suite._night_from_dateobs(r["date_obs"])
                r["session"] = suite._auto_session_from_path(path, h)
                out[stem] = r
            except Exception:
                continue
        return out

    def _fill_acq(self):
        """The run picker, then the summary it points at."""
        self.cmb_acq.blockSignals(True)
        self.cmb_acq.clear()
        for i, (clock, _text, _src) in enumerate(self._acq, start=1):
            self.cmb_acq.addItem(
                self.tr("Run {0}, finished {1}").format(i, clock or "?"))
        self.cmb_acq.setCurrentIndex(len(self._acq) - 1)    # newest
        self.cmb_acq.setVisible(len(self._acq) > 1)
        self.cmb_acq.blockSignals(False)
        self._show_acq()

    def _show_acq(self, *_):
        i = self.cmb_acq.currentIndex()
        if not (0 <= i < len(self._acq)):
            self.txt_acq.setPlainText("")
            self.btn_acq_copy.setEnabled(False)
            if self._acq_todo is not None:       # rebuilt when the tab is opened
                self.lbl_acq.setText("")
                return
            self.lbl_acq.setText(self._acq_why or self.tr(
                "No stacking run finished in this log, so there is no "
                "Acquisition Summary."))
            return
        clock, text, source = self._acq[i]
        self.txt_acq.setPlainText(text)
        self.btn_acq_copy.setEnabled(True)
        if source == "logged":
            msg = self.tr("The summary shown when stacking finished at {0}, "
                          "exactly as it appeared.").format(clock or "?")
        elif source.startswith("frames"):
            msg = self.tr("Rebuilt from this log and the headers of the "
                          "frames it stacked.")
        else:
            msg = self.tr("Rebuilt from this log.")
        if source.endswith("+notes"):
            msg += " " + self.tr("See the note at the bottom.")
        if len(self._acq) > 1:
            msg += " " + self.tr("This log holds {0} finished runs; pick one "
                                 "on the right.").format(len(self._acq))
        self.lbl_acq.setText(msg)

    def _copy_acq(self):
        QApplication.clipboard().setText(self.txt_acq.toPlainText())
        self.btn_acq_copy.setText(self.tr("✓  Copied"))
        QTimer.singleShot(1500, self._reset_acq_copy)

    def _reset_acq_copy(self):
        try:
            self.btn_acq_copy.setText(self.tr("📋  Copy Summary"))
        except RuntimeError:                     # tab closed in the meantime
            pass

    def _build_timeline(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.phase_bar = _PhaseBar()
        lay.addWidget(self.phase_bar)
        lay.addWidget(QLabel(self.tr("<b>Phases</b>")))
        self.tbl_phases = self._mk_table()
        lay.addWidget(self.tbl_phases, 1)
        lay.addWidget(QLabel(self.tr(
            "<b>Longest gaps between log lines</b> — the stacker does not "
            "time every step, so the silence after a line is the best proxy "
            "for what it cost.")))
        self.tbl_gaps = self._mk_table()
        lay.addWidget(self.tbl_gaps, 1)
        self.inner.addTab(w, self.tr("Timeline"))

    def _build_problems(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.lbl_prob = QLabel("")
        self.lbl_prob.setWordWrap(True)
        lay.addWidget(self.lbl_prob)
        self.tbl_problems = self._mk_table()
        lay.addWidget(self.tbl_problems, 2)
        lay.addWidget(QLabel(self.tr(
            "<b>Frames that failed or were skipped</b>")))
        self.tbl_failed = self._mk_table()
        lay.addWidget(self.tbl_failed, 1)
        self.inner.addTab(w, self.tr("Problems"))

    def _build_groups(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.tbl_groups = self._mk_table()
        lay.addWidget(self.tbl_groups, 1)
        self.lbl_groups_note = QLabel("")
        self.lbl_groups_note.setWordWrap(True)
        lay.addWidget(self.lbl_groups_note)
        self.inner.addTab(w, self.tr("Groups"))

    def _build_afn(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.lbl_afn_head = QLabel("")
        self.lbl_afn_head.setWordWrap(True)
        lay.addWidget(self.lbl_afn_head)
        self.tbl_afn = self._mk_table()
        self.tbl_afn.itemSelectionChanged.connect(self._on_afn_pick)
        lay.addWidget(self.tbl_afn, 1)
        self.lbl_afn_detail = QLabel("")
        self.lbl_afn_detail.setWordWrap(True)
        self.lbl_afn_detail.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_afn_detail.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.lbl_afn_detail.setStyleSheet("padding:4px 2px;")
        lay.addWidget(self.lbl_afn_detail, 1)
        self._afn_rows = []
        self.inner.addTab(w, self.tr("Local Normalization"))

    def _on_afn_pick(self):
        rows = sorted({i.row() for i in self.tbl_afn.selectedItems()})
        if rows and rows[0] < len(self._afn_rows):
            self.lbl_afn_detail.setText(_afn_readout(self._afn_rows[rows[0]]))

    def _build_masters(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel(self.tr(
            "<b>Masters this run produced</b> \u2014 calibration masters (darks, flats) and the integrated master lights.")))
        self.tbl_masters = self._mk_table()
        lay.addWidget(self.tbl_masters, 2)
        self.lbl_masters_note = QLabel("")
        self.lbl_masters_note.setWordWrap(True)
        lay.addWidget(self.lbl_masters_note)
        lay.addWidget(QLabel(self.tr(
            "<b>Input groups the stacker found</b> \u2014 from the dark / flat "
            "/ light group summary.")))
        self.tbl_calgroups = self._mk_table()
        lay.addWidget(self.tbl_calgroups, 1)
        self.inner.addTab(w, self.tr("Masters"))

    def _build_frames(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        bar = QHBoxLayout()
        bar.addWidget(QLabel(self.tr("Filter:")))
        self.ed_frame_filter = QLineEdit()
        self.ed_frame_filter.setPlaceholderText(
            self.tr("substring of the frame name…"))
        self.ed_frame_filter.textChanged.connect(self._apply_frame_filter)
        bar.addWidget(self.ed_frame_filter, 1)
        self.lbl_frames_stats = QLabel("")
        bar.addWidget(self.lbl_frames_stats)
        lay.addLayout(bar)
        self.tbl_frames = self._mk_table(sortable=True)
        lay.addWidget(self.tbl_frames, 1)
        lay.addWidget(QLabel(self.tr(
            "Click a column header to sort — lowest star count or highest "
            "FWHM first is the quick way to find the frames that hurt.")))
        self.inner.addTab(w, self.tr("Frames"))

    def _build_raw(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        bar = QHBoxLayout()
        bar.addWidget(QLabel(self.tr("Search:")))
        self.ed_search = QLineEdit()
        self.ed_search.setPlaceholderText(self.tr("substring…"))
        self.ed_search.textChanged.connect(self._apply_raw_filter)
        bar.addWidget(self.ed_search, 1)
        self.chk_problems_only = QCheckBox(self.tr("Warnings && errors only"))
        self.chk_problems_only.toggled.connect(self._apply_raw_filter)
        bar.addWidget(self.chk_problems_only)
        self.lbl_raw_count = QLabel("")
        bar.addWidget(self.lbl_raw_count)
        lay.addLayout(bar)
        self.txt_raw = QPlainTextEdit()
        self.txt_raw.setReadOnly(True)
        self.txt_raw.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        f = QFont("Consolas")
        f.setStyleHint(QFont.StyleHint.Monospace)
        f.setPointSize(9)
        self.txt_raw.setFont(f)
        lay.addWidget(self.txt_raw, 1)
        self.inner.addTab(w, self.tr("Raw log"))

    def _build_coverage(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.lbl_cov = QLabel("")
        self.lbl_cov.setWordWrap(True)
        lay.addWidget(self.lbl_cov)
        lay.addWidget(QLabel(self.tr(
            "<b>Unrecognised message shapes</b> — lines no rule matched. "
            "Numbers, paths and filenames are collapsed so repeats group "
            "together. A familiar message appearing here means the stacker's "
            "wording changed and the matching rule needs updating.")))
        self.tbl_unmatched = self._mk_table()
        lay.addWidget(self.tbl_unmatched, 1)
        self.lbl_parse_errors = QLabel("")
        self.lbl_parse_errors.setWordWrap(True)
        lay.addWidget(self.lbl_parse_errors)
        self.inner.addTab(w, self.tr("Coverage"))

    # ── log discovery / loading ─────────────────────────────────────────
    def _stacking_dir(self) -> str:
        d = getattr(self._suite, "stacking_directory", "") or ""
        return d if isinstance(d, str) else ""

    def refresh_log_list(self):
        cur = self.cmb_logs.currentData()
        self.cmb_logs.blockSignals(True)
        self.cmb_logs.clear()
        d = self._stacking_dir()
        found = find_stacking_logs(d) if d else []
        if not found:
            self.cmb_logs.addItem(
                self.tr("(no stacking logs found — use “Open log…”)"),
                None)
        for path, mtime, size in found:
            import datetime as _dt
            when = _dt.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            self.cmb_logs.addItem(
                f"{os.path.basename(path)}   —   {when}, {size/1e6:.2f} MB",
                path)
        self.cmb_logs.blockSignals(False)
        if cur:
            i = self.cmb_logs.findData(cur)
            if i >= 0:
                self.cmb_logs.setCurrentIndex(i)
                return
        if found:
            self.cmb_logs.setCurrentIndex(0)
            self.load_log(found[0][0])

    def _on_pick(self, *_):
        p = self.cmb_logs.currentData()
        if p:
            self.load_log(p)

    def _on_browse(self):
        start = self._stacking_dir() or os.path.expanduser("~")
        p, _ = QFileDialog.getOpenFileName(
            self, self.tr("Open stacking log"), start,
            self.tr("Stacking logs (*.txt);;All files (*)"))
        if p:
            i = self.cmb_logs.findData(p)
            if i < 0:
                self.cmb_logs.addItem(os.path.basename(p), p)
                i = self.cmb_logs.count() - 1
            self.cmb_logs.setCurrentIndex(i)
            self.load_log(p)

    def _reload(self):
        if self._report is not None and self._report.path:
            self.load_log(self._report.path)
        else:
            self.refresh_log_list()

    def load_log(self, path: str):
        """Parse and display one log. Reports trouble inline, never modally.

        A modal warning here blocks the UI thread waiting for a click. That
        turns any defect in this read-only tab into an apparently frozen
        Stacking Suite, which is far worse than the defect -- so failures
        are shown in the header and the tab stays usable.
        """
        self._load_error = ""
        try:
            rep = parse_stacking_log(path)
        except Exception as e:
            self._report = None
            self.lbl_head.setText(self.tr(
                "<b>{0}</b><br>"
                "<span style='color:#c0392b'>Could not read this log: "
                "{1}: {2}</span>").format(
                    os.path.basename(path), type(e).__name__, e))
            return
        self._report = rep
        try:
            self._populate(rep)
        except Exception as e:
            import traceback
            self._load_error = traceback.format_exc()
            self.lbl_head.setText(self.tr(
                "<b>{0}</b> — parsed "
                "{1} line(s), but the display failed.<br>"
                "<span style='color:#c0392b'>{2}: {3}</span>"
                "<br><span style='color:#888'>Raw log and Coverage still "
                "work; please report this with the log.</span>").format(
                    os.path.basename(path), rep.n_lines, type(e).__name__, e))
            try:
                self._raw_lines = [(l.text, l.clock, l.severity)
                                   for l in rep.lines]
                self._apply_raw_filter()
            except Exception:
                pass

    # ── population ──────────────────────────────────────────────────────
    def _populate(self, rep):
        incomplete = "" if rep.complete else self.tr(
            "  —  <b style='color:#c9913a'>no close marker: this run was "
            "interrupted or is still writing</b>")
        self.lbl_head.setText(self.tr(
            "<b>{0}</b>{1}<br>"
            "started {2} &nbsp;&nbsp; "
            "duration <b>{3}</b> &nbsp;&nbsp; "
            "{4} lines &nbsp;&nbsp; "
            "<span style='color:#c0392b'>{5} error(s)</span>, "
            "<span style='color:#c9913a'>{6} warning(s)</span>"
            "<br><span style='color:#888'>{7}</span>").format(
                os.path.basename(rep.path), incomplete, rep.started or '?',
                fmt_hms(rep.duration), rep.n_lines, len(rep.errors),
                len(rep.warnings), rep.directory or ''))

        # --- acquisition summary (first: a failure further down must not
        # leave the previous log's summary on screen to be copied)
        self._acq = [(clock, text, "logged") for clock, text in rep.summaries]
        self._acq_why = ""
        self._acq_job += 1                       # drop any read still running
        self._acq_todo = None if self._acq else rep
        self._fill_acq()
        self._ensure_acq()

        # --- summary
        _fill(self.tbl_facts, (self.tr("Setting"), self.tr("Value")),
              [(_item(k, bold=True), _item(v)) for k, v in rep.facts.items()],
              stretch_col=1)
        self.lbl_modes.setText(
            ("  •  ".join(rep.modes)) if rep.modes
            else self.tr("(none detected)"))

        rrows = []
        for label, n, delta, note in frame_reconciliation(rep):
            bad = delta is not None and delta < 0
            rrows.append((
                _item(label, bold=True),
                _item(n, align=Qt.AlignmentFlag.AlignRight),
                _item("" if delta in (None, 0) else f"{delta:+d}",
                      align=Qt.AlignmentFlag.AlignRight,
                      fg="#c0392b" if bad else None),
                _item(note, fg="#c0392b" if bad else None),
            ))
        for filt, n, tot, thr, med, sig in rep.filter_rejections:
            rrows.append((
                _item(self.tr("   rejected [{0}]").format(filt)),
                _item(f"{n}/{tot}", align=Qt.AlignmentFlag.AlignRight),
                _item(""),
                _item(self.tr(
                    "score threshold {0:.4f} (median {1:.4f}, "
                    "σ {2:.4f})").format(thr, med, sig),
                    fg="#c0392b" if n else None),
            ))
        _fill(self.tbl_recon,
              (self.tr("Stage"), self.tr("Frames"), self.tr("Δ"),
               self.tr("Note")), rrows, stretch_col=3)

        # --- timeline
        total = rep.duration or 0.0
        segs = []
        prows = []
        first = min((l.t for l in rep.lines if l.t is not None), default=0.0)
        if rep.phases and rep.phases[0].t_start > first:
            lead = rep.phases[0].t_start - first
            segs.append((self.tr("setup"), 0.0, lead, "#555e68"))
            prows.append((_item(self.tr("(setup / loading)")),
                          _item("0:00", align=Qt.AlignmentFlag.AlignRight),
                          _item(fmt_hms(lead), align=Qt.AlignmentFlag.AlignRight),
                          _item(self._pct(lead, total),
                                align=Qt.AlignmentFlag.AlignRight)))
        for i, ph in enumerate(rep.phases):
            segs.append((ph.name.split()[0], ph.t_start - first, ph.duration,
                         _PHASE_COLORS[i % len(_PHASE_COLORS)]))
            prows.append((
                _item(ph.name, bold=True),
                _item(fmt_hms(ph.t_start - first),
                      align=Qt.AlignmentFlag.AlignRight),
                _item(fmt_hms(ph.duration),
                      align=Qt.AlignmentFlag.AlignRight),
                _item(self._pct(ph.duration, total),
                      align=Qt.AlignmentFlag.AlignRight),
            ))
        self.phase_bar.set_segments(segs, total)
        _fill(self.tbl_phases,
              (self.tr("Phase"), self.tr("Starts at"), self.tr("Duration"),
               self.tr("Share")), prows, stretch_col=0)

        _fill(self.tbl_gaps,
              (self.tr("Gap"), self.tr("At"), self.tr("Line that preceded it")),
              [(_item(fmt_hms(g), align=Qt.AlignmentFlag.AlignRight),
                _item(clock), _item(txt[:170]))
               for g, clock, txt in slowest_steps(rep)], stretch_col=2)

        # --- problems
        if rep.problems:
            self.lbl_prob.setText(self.tr(
                "<b>{0} error(s)</b> and "
                "<b>{1} warning(s)</b> out of {2} "
                "lines.").format(len(rep.errors), len(rep.warnings),
                                 rep.n_lines))
        else:
            self.lbl_prob.setText(self.tr(
                "No warnings or errors were logged for this run."))
        _fill(self.tbl_problems,
              (self.tr("Time"), self.tr("Severity"), self.tr("Line"),
               self.tr("Log line #")),
              [(_item(p.clock),
                _item(p.severity.upper(), bold=True,
                      fg="#c0392b" if p.severity == SEV_ERROR else "#c9913a"),
                _item(p.text),
                _item(p.lineno, align=Qt.AlignmentFlag.AlignRight))
               for p in rep.problems], stretch_col=2)

        frows = [(_item(os.path.basename(n)), _item(self.tr("failed")),
                  _item(why)) for n, why in rep.failed_frames]
        frows += [(_item(os.path.basename(n)), _item(self.tr("skipped")),
                   _item(why)) for n, why in rep.skipped_frames]
        _fill(self.tbl_failed,
              (self.tr("Frame"), self.tr("Outcome"), self.tr("Reason")),
              frows, stretch_col=2)

        # --- groups
        grows = []
        for name, gi in rep.groups.items():
            contrib = ("—" if gi.prepass_contributing is None
                       else f"{gi.prepass_contributing}/{gi.prepass_total}")
            short = (gi.prepass_total is not None
                     and gi.files_after_filter is not None
                     and gi.prepass_total < gi.files_after_filter)
            grows.append((
                _item(name, bold=True),
                _item(gi.files_after_filter, align=Qt.AlignmentFlag.AlignRight),
                _item(contrib, align=Qt.AlignmentFlag.AlignRight,
                      fg="#c0392b" if short else None),
                _item(gi.soft_rejected if gi.soft_rejected else "",
                      align=Qt.AlignmentFlag.AlignRight),
                _item(gi.stacked, align=Qt.AlignmentFlag.AlignRight),
                _item(gi.prepass_canvas),
                _item(fmt_hms(gi.prepass_seconds),
                      align=Qt.AlignmentFlag.AlignRight),
                _item(fmt_hms(gi.integration_seconds),
                      align=Qt.AlignmentFlag.AlignRight),
                _item("" if gi.t_reduce is None else f"{gi.t_reduce:.0f}s",
                      align=Qt.AlignmentFlag.AlignRight),
                _item("" if gi.io_seconds is None else f"{gi.io_seconds:.0f}s",
                      align=Qt.AlignmentFlag.AlignRight),
                _item(gi.algorithm),
                _item("" if gi.weight_mean is None else
                      f"{gi.weight_min:g} – {gi.weight_max:g} "
                      f"(mean {gi.weight_mean:g})"),
                _item(os.path.basename(gi.master_path) if gi.master_path else ""),
            ))
        _fill(self.tbl_groups,
              (self.tr("Group"), self.tr("Files"), self.tr("Contributing"),
               self.tr("Soft-rej"), self.tr("Stacked"), self.tr("Canvas"),
               self.tr("Prepass"), self.tr("Integrate"), self.tr("GPU reduce"),
               self.tr("I/O"), self.tr("Rejection"), self.tr("Weights"),
               self.tr("Master")), grows, stretch_col=0)
        shorts = [n for n, gi in rep.groups.items()
                  if gi.prepass_total is not None
                  and gi.files_after_filter is not None
                  and gi.prepass_total < gi.files_after_filter]
        self.lbl_groups_note.setText(
            self.tr("⚠️ Fewer frames reached the prepass than the "
                    "tree held for: {0} — check Problems "
                    "for frames that failed alignment.").format(', '.join(shorts))
            if shorts else "")

        # --- local normalization (Aperture Flux Normalization)
        self._fill_afn(rep)

        # --- masters (calibration masters AND integrated master lights)
        mrows = []
        _outs = master_outputs(rep)
        for b in _outs:
            mrows.append((
                _item(b.kind.upper(), bold=True),
                _item(b.key),
                _item(b.frames, align=Qt.AlignmentFlag.AlignRight),
                _item(b.session),
                _item(b.temp),
                _item(b.method),
                _item(b.tiles, align=Qt.AlignmentFlag.AlignRight),
                _item(fmt_hms(b.seconds), align=Qt.AlignmentFlag.AlignRight),
                _item(b.dark_used),
                _item(os.path.basename(b.path) if b.path else
                      self.tr("(not saved)"),
                      fg=None if b.path else "#c0392b"),
                _item("" if b.solved is None else
                      (self.tr("solved") if b.solved else self.tr("FAILED")),
                      bold=not b.solved and b.solved is not None,
                      fg=None if b.solved is not False else "#c0392b"),
            ))
        _fill(self.tbl_masters,
              (self.tr("Kind"), self.tr("Group"), self.tr("Frames"),
               self.tr("Session"), self.tr("Temp"), self.tr("Reducer"),
               self.tr("Tiles"), self.tr("Time"), self.tr("Dark used"),
               self.tr("Output"), self.tr("Plate solve")),
              mrows, stretch_col=1)
        unsaved = [b for b in _outs if not b.path]
        total_s = sum(b.seconds or 0.0 for b in _outs)
        if _outs:
            kinds = {}
            for b in _outs:
                kinds[b.kind] = kinds.get(b.kind, 0) + 1
            breakdown = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
            note = self.tr(
                "{0} master(s) ({1}), "
                "{2} input frame(s), "
                "{3} of build time.").format(
                    len(_outs), breakdown,
                    sum(b.frames or 0 for b in _outs), fmt_hms(total_s))
            _solv = [b for b in _outs if b.solved is not None]
            if _solv:
                _ok = sum(1 for b in _solv if b.solved)
                note += self.tr(
                    "  Plate solve: {0} of {1} master(s) solved.").format(
                        _ok, len(_solv))
            if unsaved:
                note += self.tr(
                    "  \u26a0\ufe0f {0} started but no output "
                    "was logged \u2014 the run may have been "
                    "interrupted.").format(len(unsaved))
        else:
            note = self.tr("This run produced no masters.")
        self.lbl_masters_note.setText(note)

        _fill(self.tbl_calgroups,
              (self.tr("Group"), self.tr("Session"), self.tr("Files"),
               self.tr("Integration")),
              [(_item(lbl), _item(sess),
                _item(n, align=Qt.AlignmentFlag.AlignRight), _item(dur))
               for lbl, sess, n, dur in rep.cal_groups], stretch_col=0)

        # --- frames
        # Keep the DATA, not QTableWidgetItems: _fill() calls table.clear(),
        # which deletes the items it owns, so cached items would be dangling
        # C++ objects on the next filter pass.
        self._frame_stats = list(rep.frame_stats)
        self._apply_frame_filter()

        # --- raw
        self._raw_lines = [(l.text, l.clock, l.severity) for l in rep.lines]
        self._apply_raw_filter()

        # --- coverage
        pct = rep.coverage
        color = "#58A55C" if pct >= 97 else ("#c9913a" if pct >= 90 else "#c0392b")
        content = rep.content_lines
        unknown = max(0, content - rep.n_matched)
        skipped = rep.n_lines - content
        extra = (self.tr(" {0} blank/progress line(s) are not counted.").format(skipped)
                 if skipped else "")
        self.lbl_cov.setText(
            self.tr("Rules matched <b style='color:{0}'>{1:.1f}%</b> of "
                    "{2} content line(s) ({3} recognised, "
                    "{4} not).{5} Unrecognised lines are still shown "
                    "in full under <i>Raw log</i> — nothing is discarded; "
                    "they just do not feed the tables above.").format(
                        color, pct, content, rep.n_matched, unknown, extra))
        _fill(self.tbl_unmatched,
              (self.tr("Count"), self.tr("Message shape")),
              [(_item(n, align=Qt.AlignmentFlag.AlignRight), _item(sh))
               for sh, n in rep.unmatched_shapes.most_common(400)],
              stretch_col=1)
        self.lbl_parse_errors.setText(
            self.tr("⚠️ {0} line(s) matched a "
                    "rule but could not be interpreted: "
                    "{1}").format(len(rep.parse_errors), rep.parse_errors[0])
            if rep.parse_errors else "")

    def _fill_afn(self, rep):
        groups = list(getattr(rep, "afn", {}).values())
        self._afn_rows = groups
        if not groups:
            _fill(self.tbl_afn, (self.tr("Group"),), [])
            self.lbl_afn_head.setText(self.tr(
                "Aperture Flux Normalization was not used in this run. It "
                "matches every registered frame to the whole set, region by "
                "region, before integration. Turn it on in Stacking Settings "
                "→ <b>Local Normalization (Aperture Flux)</b>."))
            self.lbl_afn_detail.setText("")
            return

        ready = [a for a in groups if a.status == "ready"]
        secs = sum(a.seconds for a in groups if a.seconds is not None)
        self.lbl_afn_head.setText(self.tr(
            "<b>Aperture Flux Normalization</b> — {0} of "
            "{1} group(s) normalized, {2} in total. "
            "Each frame gets two maps: a <b>brightness</b> map from its stars "
            "(dimming: cloud, haze, dew, flats) and a <b>sky</b> map from its "
            "background (glow: light pollution, moon, airglow). "
            "<i>Held-out</i> figures score each map on stars or sky it was "
            "not fitted to. Select a row for details.").format(
                len(ready), len(groups), fmt_hms(secs)))

        def num(v, fmt):
            return "—" if v is None else fmt.format(v)

        rows = []
        R = Qt.AlignmentFlag.AlignRight
        for a in groups:
            st = {"ready": self.tr("normalized"), "skipped": self.tr("skipped"),
                  "running": self.tr("incomplete")}.get(a.status, a.status)
            fg = None if a.status == "ready" else "#c9913a"
            rows.append((
                _item(a.group, bold=True),
                _item(num(a.frames, "{}"), align=R),
                _item(st, fg=fg),
                _item((self.tr("cached") if a.stars is None and a.meas_cached
                       else num(a.stars, "{}")), align=R),
                _item(a.balanced or "—", align=R),
                _item(("—" if a.field_median is None else
                       f"{a.field_median:.1f}% / {a.field_max:.1f}%"), align=R),
                _item(("—" if a.star_gain_median is None else
                       f"{a.star_gain_median:.1f}% / {a.star_gain_p90:.1f}%"), align=R),
                _item(num(a.sky_gain_median, "{:.1f}%"), align=R),
                _item(fmt_hms(a.seconds), align=R),
            ))
        _fill(self.tbl_afn,
              (self.tr("Group"), self.tr("Frames"), self.tr("Status"),
               self.tr("Stars"), self.tr("Flip balance"),
               self.tr("Brightness across field\n(median / max)"),
               self.tr("Held-out stars better\n(median / 90th)"),
               self.tr("Held-out sky\nbetter"), self.tr("Time")),
              rows, stretch_col=0)
        self.tbl_afn.selectRow(0)
        self.lbl_afn_detail.setText(_afn_readout(groups[0]))

    @staticmethod
    def _pct(part, total):
        if not total or part is None:
            return ""
        return f"{100.0 * part / total:.1f}%"

    # ── filters ─────────────────────────────────────────────────────────
    # Column spec for the Frames table: (header, attribute, format).
    # Headers are display text only (attribute names are the keys); they are
    # marked here and translated where the table is filled.
    _FRAME_COLS = (
        (QT_TRANSLATE_NOOP("StackingLogAnalysisTab", "Stars"),      "star_count", "{:g}"),
        (QT_TRANSLATE_NOOP("StackingLogAnalysisTab", "Ecc"),        "ecc",        "{:.4f}"),
        (QT_TRANSLATE_NOOP("StackingLogAnalysisTab", "Bg"),         "bg",         "{:.5f}"),
        ("FWHM",       "fwhm",       "{:.2f}"),
        (QT_TRANSLATE_NOOP("StackingLogAnalysisTab", "Sharp"),      "sharp",      "{:.4f}"),
        ("SNR",        "snr",        "{:.3f}"),
        ("ExpScale",   "exp_scale",  "{:.3f}"),
        (QT_TRANSLATE_NOOP("StackingLogAnalysisTab", "Weight"),     "weight",     "{:.3f}"),
        (QT_TRANSLATE_NOOP("StackingLogAnalysisTab", "Normalized"), "normalized", "{:.4f}"),
    )

    def _apply_frame_filter(self):
        stats = getattr(self, "_frame_stats", [])
        needle = (self.ed_frame_filter.text() or "").strip().lower()
        keep = []
        for fs in stats:
            if needle and needle not in (fs.name or "").lower():
                continue
            cells = [_item(os.path.basename(fs.name or ""))]
            for _hdr, attr, fmtstr in self._FRAME_COLS:
                v = getattr(fs, attr, None)
                it = _NumItem("" if v is None else fmtstr.format(v),
                              None if v is None else float(v))
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                    | Qt.AlignmentFlag.AlignVCenter)
                cells.append(it)
            keep.append(cells)
        self.tbl_frames.setSortingEnabled(False)
        _fill(self.tbl_frames,
              [self.tr("Frame")]
              + [QCoreApplication.translate("StackingLogAnalysisTab", h)
                 for h, _a, _f in self._FRAME_COLS],
              keep, stretch_col=0)
        self.tbl_frames.setSortingEnabled(True)
        self.lbl_frames_stats.setText(
            self.tr("{0} of {1} frame(s)").format(len(keep), len(stats)))

    def _apply_raw_filter(self):
        lines = getattr(self, "_raw_lines", [])
        needle = (self.ed_search.text() or "").strip().lower()
        probs = self.chk_problems_only.isChecked()
        out = []
        for text, clock, sev in lines:
            if probs and sev not in (SEV_ERROR, SEV_WARN):
                continue
            if needle and needle not in text.lower():
                continue
            out.append(f"[{clock}] {text}" if clock else text)
        # Guard against pathological logs locking the widget up.
        cap = 40000
        clipped = len(out) > cap
        self.txt_raw.setPlainText("\n".join(out[:cap]))
        self.lbl_raw_count.setText(
            self.tr("{0} line(s) — showing first {1}").format(len(out), cap)
            if clipped else self.tr("{0} line(s)").format(len(out)))
