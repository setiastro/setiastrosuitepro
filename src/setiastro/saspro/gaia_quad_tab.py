#src/setiastro/saspro/gaia_quad_tab.py
#!/usr/bin/env python3
# ======================================================
#  gaia_quad_tab.py  —  "Blind Solve Index" tab for the Gaia DR3 Library
#
#  Builds the quad-hash layers LOCALLY from the Gaia catalogs already on disk.
#  Nothing is downloaded here. If the user has the Bright (G < 17) astrometric
#  tier — the one marked Recommended — every layer that matters is buildable,
#  because layers 2-7 need nothing fainter than G ~ 13.6 and that all lives in
#  gaia_astro_lt15.sqlite.
#
#  Written by Franklin Marek
#  www.setiastro.com
# ======================================================

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal as _Signal
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSpinBox, QVBoxLayout,
    QWidget,
)

from setiastro.saspro.gaia_quad_builder import default_workers as _default_workers
from setiastro.saspro.gaia_quad_index import (
    QUAD_LAYER_DEFS, QuadLayerDef, fmt_size, read_saq_header,
    get_quad_library_dir, set_quad_library_dir, refresh_quad_library,
    close_quad_library,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Worker
# ─────────────────────────────────────────────────────────────────────────────

class QuadBuildWorker(QThread):
    layer_progress = _Signal(int, float, str)      # layer index, 0..1, message
    layer_done = _Signal(int, bool, str)           # layer index, ok, message
    all_done = _Signal(int, int)                   # built, failed
    cancelled = _Signal()

    def __init__(self, layers: List[QuadLayerDef], out_dir: Path,
                 astro_dir: Optional[Path], xp_dir: Optional[Path],
                 workers: int = 1, parent=None):
        super().__init__(parent)
        self._layers = list(layers)
        self._out = Path(out_dir)
        self._astro = astro_dir
        self._xp = xp_dir
        self._workers = max(1, int(workers))
        self._cancel = False
        self._current = -1

    def cancel(self):
        self._cancel = True

    def run(self):
        from setiastro.saspro.gaia_quad_builder import build_layer, open_source

        source, info = open_source(self._astro, self._xp)
        if source is None:
            self.all_done.emit(0, len(self._layers))
            for d in self._layers:
                self.layer_done.emit(d.index, False, "No Gaia catalogue installed.")
            return

        built = failed = 0
        try:
            for d in self._layers:
                if self._cancel:
                    self.cancelled.emit()
                    return
                self._current = d.index
                if source.depth() + 1e-6 < d.gmag_limit:
                    failed += 1
                    self.layer_done.emit(
                        d.index, False,
                        f"Needs stars to G ≤ {d.gmag_limit:.1f}; the installed "
                        f"catalogue only reaches G ≤ {source.depth():.1f}.")
                    continue
                try:
                    stats = build_layer(
                        d, self._out / d.filename, source,
                        progress=lambda f, m, i=d.index: self.layer_progress.emit(i, f, m),
                        cancel=lambda: self._cancel,
                        workers=self._workers)
                    built += 1
                    self.layer_done.emit(
                        d.index, True,
                        f"{stats['n_quads']:,} quads · "
                        f"{fmt_size(stats['bytes'])} · {stats['seconds']:.0f}s")
                except RuntimeError as e:
                    if "cancelled" in str(e):
                        self.cancelled.emit()
                        return
                    failed += 1
                    self.layer_done.emit(d.index, False, str(e))
                except Exception as e:
                    failed += 1
                    self.layer_done.emit(d.index, False, f"{type(e).__name__}: {e}")
        finally:
            try:
                source.close()
            except Exception:
                pass
        self.all_done.emit(built, failed)


# ─────────────────────────────────────────────────────────────────────────────
#  Row
# ─────────────────────────────────────────────────────────────────────────────

class _QuadRowWidget(QFrame):
    build_requested = _Signal(int)
    remove_requested = _Signal(int)

    def __init__(self, d: QuadLayerDef, parent=None):
        super().__init__(parent)
        self._d = d
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame { border:1px solid #2a2a3e; border-radius:6px; "
                           "background:#12121f; }")

        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(12)

        self._dot = QLabel("●")
        self._dot.setFixedWidth(16)
        self._dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._dot)

        self._chk = QCheckBox()
        self._chk.setChecked(d.recommended)
        self._chk.setToolTip("Include this layer when you press "
                             "“Build Selected”.")
        # Qt draws an unstyled indicator through the stylesheet engine as soon
        # as any ancestor carries a stylesheet -- and this row sets one. On a
        # dark background that indicator comes out invisible, so the box has
        # to be drawn explicitly.
        self._chk.setStyleSheet("""
            QCheckBox { background: transparent; border: none; spacing: 0px; }
            QCheckBox::indicator {
                width: 15px; height: 15px;
                border: 1px solid #5b6680; border-radius: 3px;
                background: #0a0a18;
            }
            QCheckBox::indicator:hover { border-color: #8b97b5; }
            QCheckBox::indicator:checked {
                background: #4a7cf0; border: 1px solid #6f9bff;
            }
            QCheckBox::indicator:checked:hover { background: #5d8bf7; }
            QCheckBox::indicator:disabled {
                border-color: #2f3442; background: #14141f;
            }
            QCheckBox::indicator:checked:disabled { background: #2b3a5c; }
        """)
        outer.addWidget(self._chk)

        info = QVBoxLayout()
        info.setSpacing(3)

        trow = QHBoxLayout()
        t = QLabel(d.label)
        t.setStyleSheet("font-weight:bold; font-size:13px; color:#e0e0ff; border:none;")
        trow.addWidget(t)
        if d.recommended:
            b = QLabel("Recommended")
            b.setStyleSheet("font-size:10px; color:#44cc88; font-weight:bold; "
                            "border:1px solid #44cc88; border-radius:3px; padding:1px 5px;")
            trow.addWidget(b)
        trow.addStretch()
        info.addLayout(trow)

        desc = QLabel(f"Solves {d.field_label}  ·  needs stars to G ≤ "
                      f"{d.gmag_limit:.1f}  ·  about {d.est_quads:,} quads, "
                      f"{fmt_size(d.est_bytes)}"
                      + (f"  ·  builds in {d.build_note}" if d.build_note else ""))
        desc.setStyleSheet("font-size:11px; color:#888; border:none;")
        desc.setWordWrap(True)
        info.addWidget(desc)

        self._warn = QLabel(f"⚠  {d.warning}")
        self._warn.setStyleSheet("font-size:11px; color:#ddaa44; border:none;")
        self._warn.setWordWrap(True)
        self._warn.setVisible(bool(d.warning))
        info.addWidget(self._warn)

        self._stats = QLabel("")
        self._stats.setStyleSheet("font-size:11px; color:#6688aa; border:none;")
        self._stats.setWordWrap(True)
        info.addWidget(self._stats)
        outer.addLayout(info, stretch=1)

        self._prog = QProgressBar()
        self._prog.setFixedWidth(200)
        self._prog.setVisible(False)
        self._prog.setTextVisible(True)
        self._prog.setStyleSheet("""
            QProgressBar { border:1px solid #333; border-radius:3px;
                background:#0a0a18; color:#aaa; text-align:center; font-size:9px; }
            QProgressBar::chunk { background:#4466cc; border-radius:2px; }
        """)
        outer.addWidget(self._prog)

        br = QHBoxLayout()
        br.setSpacing(6)
        self._btn_build = QPushButton("Build")
        self._btn_build.setFixedWidth(80)
        self._btn_build.setStyleSheet(self._style("#224488", "#3366cc"))
        self._btn_build.clicked.connect(lambda: self.build_requested.emit(d.index))
        self._btn_remove = QPushButton("Remove")
        self._btn_remove.setFixedWidth(80)
        self._btn_remove.setStyleSheet(self._style("#442222", "#883333"))
        self._btn_remove.clicked.connect(lambda: self.remove_requested.emit(d.index))
        br.addWidget(self._btn_build)
        br.addWidget(self._btn_remove)
        outer.addLayout(br)

    @staticmethod
    def _style(bg, hover):
        return (f"QPushButton {{ background:{bg}; color:#ddd; border:1px solid #444; "
                f"border-radius:4px; padding:4px 8px; font-size:11px; }}"
                f"QPushButton:hover {{ background:{hover}; }}"
                f"QPushButton:disabled {{ color:#555; background:#1a1a2e; }}")

    @property
    def selected(self) -> bool:
        return self._chk.isChecked()

    def set_state(self, built: bool, buildable: bool, detail: str, size: int = 0,
                  hdr: Optional[dict] = None):
        if built:
            self._dot.setStyleSheet("color:#44cc66; font-size:14px;")
            # Once a layer exists, report what it actually holds rather than
            # the estimate above it -- the estimate is a projection, the
            # header is the truth.
            if hdr:
                self._stats.setText(
                    f"Built  ·  {hdr['n_quads']:,} quads from "
                    f"{hdr['n_stars']:,} stars  ·  {fmt_size(size)} on disk"
                    f"  ·  epoch {hdr['epoch']:.1f}")
            else:
                self._stats.setText(f"Built  ·  {fmt_size(size)} on disk"
                                    if size else "Built")
            self._btn_build.setText("Rebuild")
            self._btn_build.setEnabled(buildable)
            self._btn_remove.setVisible(True)
        else:
            self._btn_remove.setVisible(False)
            self._btn_build.setText("Build")
            self._btn_build.setEnabled(buildable)
            if buildable:
                self._dot.setStyleSheet("color:#555; font-size:14px;")
                self._stats.setText("Not built")
            else:
                self._dot.setStyleSheet("color:#aa4444; font-size:14px;")
                self._stats.setText(detail)
        self._chk.setEnabled(buildable)

    def set_busy(self, busy: bool):
        self._btn_build.setEnabled(not busy)
        self._btn_remove.setEnabled(not busy)
        self._chk.setEnabled(not busy)
        self._prog.setVisible(busy)
        if not busy:
            self._prog.setValue(0)

    def set_progress(self, frac: float, msg: str):
        self._prog.setVisible(True)
        self._prog.setMaximum(1000)
        self._prog.setValue(int(frac * 1000))
        self._prog.setFormat(f"{frac*100:.0f}%")
        self._stats.setText(msg)


# ─────────────────────────────────────────────────────────────────────────────
#  Tab
# ─────────────────────────────────────────────────────────────────────────────

class BlindSolveIndexTab(QWidget):
    library_changed = _Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: Dict[int, _QuadRowWidget] = {}
        self._worker: Optional[QuadBuildWorker] = None
        self._build_ui()
        self.refresh()

    # ── ui ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(8)

        info = QLabel(
            "Blind solving finds the field from the star pattern alone — no RA/Dec, "
            "no field of view, no hints. These index layers are <b>built locally "
            "from the Gaia data you already downloaded</b>; nothing extra comes off "
            "the wire.<br><br>"
            "A layer is tied to one angular scale and solves fields from about 2.5× "
            "to 8× its quad size, so the layers overlap. Build the Recommended set "
            "and you cover roughly 20′ to 25° fields — everything most imaging rigs "
            "produce — from the Bright (G&nbsp;&lt;&nbsp;17) astrometric tier alone."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#778; font-size:11px; margin-bottom:4px;")
        root.addWidget(info)

        self._src_lbl = QLabel("")
        self._src_lbl.setWordWrap(True)
        self._src_lbl.setStyleSheet("font-size:11px; color:#6688aa;")
        root.addWidget(self._src_lbl)

        drow = QHBoxLayout()
        drow.addWidget(QLabel("Index folder:"))
        self._dir_lbl = QLabel("")
        self._dir_lbl.setStyleSheet("color:#6688aa; font-size:11px; font-family:monospace;")
        self._dir_lbl.setWordWrap(True)
        drow.addWidget(self._dir_lbl, stretch=1)
        b_change = QPushButton("Change Location…")
        b_change.setFixedWidth(130)
        b_change.clicked.connect(self._change_location)
        drow.addWidget(b_change)
        b_open = QPushButton("Open Folder")
        b_open.setFixedWidth(100)
        b_open.clicked.connect(self._open_folder)
        drow.addWidget(b_open)
        root.addLayout(drow)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border:none; background:transparent; }")
        cont = QWidget()
        cont.setStyleSheet("background:transparent;")
        self._list = QVBoxLayout(cont)
        self._list.setSpacing(8)
        for d in QUAD_LAYER_DEFS:
            row = _QuadRowWidget(d)
            row.build_requested.connect(lambda i: self._start([i]))
            row.remove_requested.connect(self._remove)
            self._list.addWidget(row)
            self._rows[d.index] = row
        self._list.addStretch()
        scroll.setWidget(cont)
        root.addWidget(scroll, stretch=1)

        brow = QHBoxLayout()
        brow.addWidget(QLabel("Reader processes:"))
        self._spin_workers = QSpinBox()
        self._spin_workers.setRange(1, 16)
        self._spin_workers.setValue(_default_workers())
        self._spin_workers.setToolTip(
            "Parallel readers for the catalogue scan.\n\n"
            "Reading the Gaia tables is about half the build time, and each\n"
            "worker reduces its own slice before anything is handed back, so\n"
            "extra readers cost little memory. Threads were measured to be\n"
            "SLOWER than one reader (the GIL is held while sqlite builds row\n"
            "tuples), so these are processes.\n\n"
            "Set to 1 if your drive is the bottleneck rather than the CPU.")
        self._spin_workers.valueChanged.connect(self._save_workers)
        brow.addWidget(self._spin_workers)
        brow.addSpacing(12)
        self._status = QLabel("")
        self._status.setStyleSheet("color:#668; font-size:11px;")
        self._status.setWordWrap(True)
        brow.addWidget(self._status, stretch=1)
        self._btn_all = QPushButton("Build Selected")
        self._btn_all.setFixedWidth(130)
        self._btn_all.clicked.connect(self._build_selected)
        brow.addWidget(self._btn_all)
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setFixedWidth(90)
        self._btn_cancel.setVisible(False)
        self._btn_cancel.clicked.connect(self._cancel)
        brow.addWidget(self._btn_cancel)
        root.addLayout(brow)

    # ── state ────────────────────────────────────────────────────────────

    def _source_dirs(self):
        from setiastro.saspro.gaia_database import (
            get_astro_library_dir, get_library_dir)
        try:
            a = get_astro_library_dir()
        except Exception:
            a = None
        try:
            x = get_library_dir()
        except Exception:
            x = None
        return a, x

    def refresh(self):
        from setiastro.saspro.gaia_quad_builder import open_source

        d = get_quad_library_dir()
        self._dir_lbl.setText(str(d))

        astro, xp = self._source_dirs()
        source, info = open_source(astro, xp)
        if source is not None:
            source.close()

        depth = info.depth
        if depth <= 0:
            self._src_lbl.setText(
                "<span style='color:#cc8844;'>No Gaia catalogue installed. "
                "Install the <b>Bright (G &lt; 17)</b> tier on the Astrometry tab "
                "— that single tier covers every Recommended layer here.</span>")
        else:
            n_ok = sum(1 for x in QUAD_LAYER_DEFS if depth >= x.gmag_limit - 1e-6)
            self._src_lbl.setText(
                f"Source: {info.detail}  ·  {n_ok} of {len(QUAD_LAYER_DEFS)} "
                f"layers buildable")

        built = 0
        tot_quads = tot_stars = tot_bytes = 0
        for x in QUAD_LAYER_DEFS:
            p = d / x.filename
            is_built = p.exists()
            built += is_built
            can = depth >= x.gmag_limit - 1e-6
            detail = (f"Needs stars to G ≤ {x.gmag_limit:.1f} — install a deeper "
                      f"tier (have G ≤ {depth:.1f})" if depth > 0 else
                      "No Gaia catalogue installed")
            size = p.stat().st_size if is_built else 0
            # The header read here is the one the row already needs, so the
            # totals cost nothing extra.
            hdr = read_saq_header(p) if is_built else None
            if is_built:
                tot_bytes += size
                if hdr:
                    tot_quads += int(hdr.get("n_quads", 0) or 0)
                    tot_stars += int(hdr.get("n_stars", 0) or 0)
            self._rows[x.index].set_state(is_built, can, detail, size, hdr)

        if built:
            self._status.setText(
                f"{built}/{len(QUAD_LAYER_DEFS)} layers built  ·  "
                f"{tot_quads:,} quads from {tot_stars:,} index stars  ·  "
                f"{fmt_size(tot_bytes)} on disk  ·  blind solving is active")
        else:
            self._status.setText(
                f"0/{len(QUAD_LAYER_DEFS)} layers built  ·  blind solving is "
                f"unavailable until at least one layer is built")

    # ── build ────────────────────────────────────────────────────────────

    def _build_selected(self):
        want = [d for d in QUAD_LAYER_DEFS if self._rows[d.index].selected]
        if not want:
            QMessageBox.information(self, "Blind Solve Index",
                                    "Tick at least one layer to build.")
            return
        self._start([d.index for d in want])

    def _start(self, indices: List[int]):
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "Blind Solve Index",
                                "A build is already running.")
            return
        defs = [d for d in QUAD_LAYER_DEFS if d.index in set(indices)]
        heavy = [d for d in defs if d.warning]
        if heavy:
            names = "\n".join(f"  • {d.label}: {d.warning}" for d in heavy)
            if QMessageBox.question(
                    self, "Long build",
                    f"These layers take a long time and a lot of memory:\n\n{names}\n\n"
                    f"Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return

        astro, xp = self._source_dirs()
        out = get_quad_library_dir()
        nw = int(self._spin_workers.value())
        for d in defs:
            self._rows[d.index].set_busy(True)
        self._btn_all.setEnabled(False)
        self._btn_cancel.setVisible(True)
        self._status.setText(f"Building {len(defs)} layer(s)…")

        self._worker = QuadBuildWorker(defs, out, astro, xp,
                                       workers=nw, parent=None)
        self._worker.layer_progress.connect(self._on_progress)
        self._worker.layer_done.connect(self._on_layer_done)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.start()

    def _on_progress(self, index: int, frac: float, msg: str):
        row = self._rows.get(index)
        if row is not None:
            row.set_progress(frac, msg)

    def _on_layer_done(self, index: int, ok: bool, msg: str):
        row = self._rows.get(index)
        if row is not None:
            row.set_busy(False)
        if not ok:
            self._status.setText(f"Layer {index} failed: {msg}")

    def _finish(self):
        for r in self._rows.values():
            r.set_busy(False)
        self._btn_all.setEnabled(True)
        self._btn_cancel.setVisible(False)
        if self._worker is not None:
            self._worker.wait(3000)
        self._worker = None
        refresh_quad_library()
        self.refresh()
        self.library_changed.emit()

    def _on_all_done(self, built: int, failed: int):
        self._finish()
        self._status.setText(
            f"Built {built} layer(s)" + (f", {failed} failed" if failed else "")
            + ".  Blind solving will now be tried automatically when an image "
              "has no usable RA/Dec.")

    def _on_cancelled(self):
        self._finish()
        self._status.setText("Build cancelled. Layers already finished are usable.")

    def _cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self._status.setText("Cancelling…")

    # ── housekeeping ─────────────────────────────────────────────────────

    def _remove(self, index: int):
        from setiastro.saspro.gaia_quad_index import LAYER_BY_INDEX
        d = LAYER_BY_INDEX[index]
        p = get_quad_library_dir() / d.filename
        if not p.exists():
            return
        if QMessageBox.question(
                self, "Remove Layer",
                f"Delete {d.label}?\n\n{fmt_size(p.stat().st_size)} will be freed. "
                f"You can rebuild it at any time from the Gaia data you already have.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return
        # close, not refresh: refresh() closes the library and immediately
        # reopens it, which re-maps the very file we are about to delete.
        close_quad_library()
        try:
            p.unlink()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not delete {p.name}:\n{e}")
        refresh_quad_library()
        self.refresh()
        self.library_changed.emit()

    def _change_location(self):
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "Blind Solve Index",
                                "Wait for the build to finish first.")
            return
        cur = get_quad_library_dir()
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose Blind Solve Index Folder", str(cur))
        if not chosen:
            return
        dest = Path(chosen)
        if dest == cur:
            return
        present = [d.filename for d in QUAD_LAYER_DEFS if (cur / d.filename).exists()]
        move = False
        if present:
            move = QMessageBox.question(
                self, "Move Existing Layers",
                f"Move {len(present)} built layer(s) to the new folder?\n\n"
                f"Choosing No just points SASpro at the new folder and leaves the "
                f"files where they are.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) == QMessageBox.StandardButton.Yes

        # Same reason as Remove: a mapped file cannot be moved on Windows,
        # and refresh() would re-map these from the old folder right before
        # shutil.move touches them.
        close_quad_library()
        set_quad_library_dir(dest)
        if move:
            dest.mkdir(parents=True, exist_ok=True)
            for fn in present:
                self._status.setText(f"Moving {fn}…")
                QApplication.processEvents()
                try:
                    shutil.move(str(cur / fn), str(dest / fn))
                except Exception as e:
                    QMessageBox.warning(self, "Move Failed", f"{fn}:\n{e}")
                    break
        refresh_quad_library()
        self.refresh()
        self.library_changed.emit()

    def _open_folder(self):
        import subprocess
        d = str(get_quad_library_dir())
        if sys.platform == "win32":
            os.startfile(d)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", d])
        else:
            subprocess.Popen(["xdg-open", d])

    def _save_workers(self, v: int):
        try:
            from PyQt6.QtCore import QSettings
            QSettings().setValue("gaia_quad_library/workers", int(v))
        except Exception:
            pass

    def shutdown(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(5000)
        self._worker = None