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

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QSizePolicy, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from setiastro.saspro.stacking_log_analysis import (
    SEV_ERROR, SEV_WARN, fmt_hms, find_stacking_logs, frame_reconciliation,
    master_outputs, parse_stacking_log, slowest_steps,
)

__all__ = ["StackingLogAnalysisTab"]


def _models_text(models) -> str:
    return ", ".join(f"{name} {n}" for name, n in models) if models else "—"


def _afn_readout(a) -> str:
    """Plain-language account of what AFN measured and did for one group."""
    esc = lambda s: (str(s).replace("&", "&amp;").replace("<", "&lt;")
                     .replace(">", "&gt;"))
    out = [f"<b>{esc(a.group)}</b>"]
    if a.status == "skipped":
        out.append(f"<span style='color:#c9913a'>Skipped: {esc(a.skip_reason)}.</span> "
                   "This group was integrated exactly as it would be without AFN.")
        return "<br>".join(out)
    if a.status != "ready":
        out.append("<span style='color:#c9913a'>The log ends before AFN finished "
                   "for this group.</span>")

    # what it measured
    if a.stars is not None:
        out.append(
            f"<b>Measured:</b> {a.stars} stars (of {a.detections} detected) in "
            f"every frame, FWHM {a.fwhm:.2f} px, aperture radius {a.aperture:.1f} px "
            f"with a sky annulus of {esc(a.annulus)} px; sky in a {esc(a.grid)} grid "
            f"of {a.cell} px cells.")
    elif a.meas_cached:
        out.append("<b>Measured:</b> reused the cached measurements from an "
                   "earlier run of the same frames.")

    # flip
    if a.balanced:
        out.append(
            f"<b>Meridian flip:</b> frames came from both sides of the flip "
            f"({esc(a.balanced)}). The reference was balanced between them, so a "
            f"flat error that flips with the mount cancels instead of being "
            f"copied to every frame.")

    # brightness
    const = a.model_share("constant")
    if a.star_gain_median is not None:
        if (const is not None and const >= 0.75
                and a.star_gain_median < 1.0):
            out.append(
                f"<b>Brightness:</b> the frames already agreed. "
                f"{int(round(const * 100))}% of them needed only a single overall "
                f"scale, so AFN made almost no spatial correction here.")
        else:
            size = ("a small" if a.star_gain_median < 2 else
                    "a moderate" if a.star_gain_median < 10 else "a large")
            out.append(
                f"<b>Brightness:</b> {size} correction. Frames differed across the "
                f"field by a median of {a.field_median:.1f}% (most: "
                f"{a.field_max:.1f}%). The maps predicted stars they were not "
                f"fitted to {a.star_gain_median:.1f}% better than a single overall "
                f"scale; {a.star_gain_p90:.1f}% for the most-corrected tenth of "
                f"frames.")
        out.append(f"&nbsp;&nbsp;Map detail chosen per frame: "
                   f"{esc(_models_text(a.bright_models))}.")

    # sky
    if a.sky_gain_median is not None:
        out.append(
            f"<b>Sky:</b> differences in sky glow between frames were matched; "
            f"the sky maps predicted sky they were not fitted to "
            f"{a.sky_gain_median:.1f}% better than one overall offset.")
        out.append(f"&nbsp;&nbsp;Map detail chosen per frame: "
                   f"{esc(_models_text(a.sky_models))}.")

    if a.clamped:
        out.append(f"{a.clamped} map(s) were held to the range their stars "
                   f"support where they would otherwise have extrapolated "
                   f"(normal near frame edges).")
    if a.maps_cached:
        out.append("The maps themselves were reused from cache.")

    # cost
    bits = []
    if a.seconds is not None:
        bits.append(f"took {fmt_hms(a.seconds)}")
    if a.t_measure is not None:
        bits.append(f"measuring {a.t_measure:.1f} s, fitting {a.t_fit:.1f} s")
    if a.processes:
        bits.append(f"across {a.processes} processes")
    if bits:
        out.append("<b>Cost:</b> " + ", ".join(bits) + ".")
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
                   self.tr(f"total {fmt_hms(self._total)}"))


class StackingLogAnalysisTab(QWidget):
    """Pick a stacking log, parse it, and show what the run actually did."""

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
        self._build_timeline()
        self._build_problems()
        self._build_groups()
        self._build_afn()
        self._build_masters()
        self._build_frames()
        self._build_raw()
        self._build_coverage()

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

        gb_f = QGroupBox(self.tr("Run settings & facts"))
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
        self.chk_problems_only = QCheckBox(self.tr("Warnings & errors only"))
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
            self.lbl_head.setText(
                f"<b>{os.path.basename(path)}</b><br>"
                f"<span style='color:#c0392b'>Could not read this log: "
                f"{type(e).__name__}: {e}</span>")
            return
        self._report = rep
        try:
            self._populate(rep)
        except Exception as e:
            import traceback
            self._load_error = traceback.format_exc()
            self.lbl_head.setText(
                f"<b>{os.path.basename(path)}</b> — parsed "
                f"{rep.n_lines} line(s), but the display failed.<br>"
                f"<span style='color:#c0392b'>{type(e).__name__}: {e}</span>"
                f"<br><span style='color:#888'>Raw log and Coverage still "
                f"work; please report this with the log.</span>")
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
        self.lbl_head.setText(
            f"<b>{os.path.basename(rep.path)}</b>{incomplete}<br>"
            f"started {rep.started or '?'} &nbsp;&nbsp; "
            f"duration <b>{fmt_hms(rep.duration)}</b> &nbsp;&nbsp; "
            f"{rep.n_lines} lines &nbsp;&nbsp; "
            f"<span style='color:#c0392b'>{len(rep.errors)} error(s)</span>, "
            f"<span style='color:#c9913a'>{len(rep.warnings)} warning(s)</span>"
            f"<br><span style='color:#888'>{rep.directory or ''}</span>")

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
                _item(self.tr(f"   rejected [{filt}]")),
                _item(f"{n}/{tot}", align=Qt.AlignmentFlag.AlignRight),
                _item(""),
                _item(self.tr(
                    f"score threshold {thr:.4f} (median {med:.4f}, "
                    f"σ {sig:.4f})"),
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
                f"<b>{len(rep.errors)} error(s)</b> and "
                f"<b>{len(rep.warnings)} warning(s)</b> out of {rep.n_lines} "
                f"lines."))
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
                    f"tree held for: {', '.join(shorts)} — check Problems "
                    f"for frames that failed alignment.") if shorts else "")

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
                f"{len(_outs)} master(s) ({breakdown}), "
                f"{sum(b.frames or 0 for b in _outs)} input frame(s), "
                f"{fmt_hms(total_s)} of build time.")
            _solv = [b for b in _outs if b.solved is not None]
            if _solv:
                _ok = sum(1 for b in _solv if b.solved)
                note += self.tr(
                    f"  Plate solve: {_ok} of {len(_solv)} master(s) solved.")
            if unsaved:
                note += self.tr(
                    f"  \u26a0\ufe0f {len(unsaved)} started but no output "
                    f"was logged \u2014 the run may have been "
                    f"interrupted.")
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
        extra = (self.tr(f" {skipped} blank/progress line(s) are not counted.")
                 if skipped else "")
        self.lbl_cov.setText(
            self.tr(f"Rules matched <b style='color:{color}'>{pct:.1f}%</b> of "
                    f"{content} content line(s) ({rep.n_matched} recognised, "
                    f"{unknown} not).{extra} Unrecognised lines are still shown "
                    f"in full under <i>Raw log</i> — nothing is discarded; "
                    f"they just do not feed the tables above."))
        _fill(self.tbl_unmatched,
              (self.tr("Count"), self.tr("Message shape")),
              [(_item(n, align=Qt.AlignmentFlag.AlignRight), _item(sh))
               for sh, n in rep.unmatched_shapes.most_common(400)],
              stretch_col=1)
        self.lbl_parse_errors.setText(
            self.tr(f"⚠️ {len(rep.parse_errors)} line(s) matched a "
                    f"rule but could not be interpreted: "
                    f"{rep.parse_errors[0]}") if rep.parse_errors else "")

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
            f"<b>Aperture Flux Normalization</b> — {len(ready)} of "
            f"{len(groups)} group(s) normalized, {fmt_hms(secs)} in total. "
            "Each frame gets two maps: a <b>brightness</b> map from its stars "
            "(dimming: cloud, haze, dew, flats) and a <b>sky</b> map from its "
            "background (glow: light pollution, moon, airglow). "
            "<i>Held-out</i> figures score each map on stars or sky it was "
            "not fitted to. Select a row for details."))

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
                _item(("cached" if a.stars is None and a.meas_cached
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
    _FRAME_COLS = (
        ("Stars",      "star_count", "{:g}"),
        ("Ecc",        "ecc",        "{:.4f}"),
        ("Bg",         "bg",         "{:.5f}"),
        ("FWHM",       "fwhm",       "{:.2f}"),
        ("Sharp",      "sharp",      "{:.4f}"),
        ("SNR",        "snr",        "{:.3f}"),
        ("ExpScale",   "exp_scale",  "{:.3f}"),
        ("Weight",     "weight",     "{:.3f}"),
        ("Normalized", "normalized", "{:.4f}"),
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
              [self.tr("Frame")] + [self.tr(h) for h, _a, _f in self._FRAME_COLS],
              keep, stretch_col=0)
        self.tbl_frames.setSortingEnabled(True)
        self.lbl_frames_stats.setText(
            self.tr(f"{len(keep)} of {len(stats)} frame(s)"))

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
            self.tr(f"{len(out)} line(s)"
                    + (f" — showing first {cap}" if clipped else "")))
