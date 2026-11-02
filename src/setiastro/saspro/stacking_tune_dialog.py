# ──────────────────────────────────────────────────────────────────────────────
#  src/setiastro/saspro/stacking_tune_dialog.py — SetiAstroSuite Pro
#  Copyright (C) 2026 Franklin Marek / SetiAstro
#  Licensed under the GNU General Public License v3.0 or later.
# ──────────────────────────────────────────────────────────────────────────────
"""
"Optimize…" for the stacking settings: measure this machine instead of guessing.

Two independent questions, measured separately because one answer does not
imply the other:

  Registration  which star detector, and how many PROCESSES may hold a CUDA
                context at once. Measured on a 3070 Ti at 12 MP: GPU 4 workers
                45.1 f/s against SEP 32 workers 30.7 f/s, and 16 GPU workers
                wedged the card outright.

  Integration   how many full-width rows go into ONE rejection call in the
                parent process (`stacking/chunk_height`). The shipped default
                is 16, chosen to match the old 256x256 tile, which on a 12 MP
                frame is ~188 tiny GPU launches per pass.

The work runs on a QThread so the dialog stays responsive and Cancel works;
`stacking_worker_tune` does the measuring and owns the watchdog that kills a
configuration which thrashes rather than fails.

Wiring (see sa_settings_tune_patch.py, which does this automatically):

    from setiastro.saspro.stacking_tune_dialog import install_optimize_button
    install_optimize_button(self, hw_row, self.chunkHeightSpinBox)
"""
from __future__ import annotations

import os

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

__all__ = [
    "LAST_FRAMES",
    "LAST_H",
    "LAST_W",
    "TuneWorker",
    "StackingTuneDialog",
    "install_optimize_button",
    "make_align_workers_row",
]


# What the optimizer was last run with. Under stacking/align/* like the rest
# of the shared alignment prefs.
LAST_H = "stacking/align/tune_last_h"
LAST_W = "stacking/align/tune_last_w"
LAST_FRAMES = "stacking/align/tune_last_frames"
LAST_DETECT = "stacking/align/tune_last_detect"
LAST_BAND = "stacking/align/tune_last_band"
LAST_WARP = "stacking/align/tune_last_warp"


def _load_last_inputs() -> dict:
    """The optimizer's last inputs, or {} when it has never been run."""
    out = {}
    try:
        from PyQt6.QtCore import QSettings
        s = QSettings()

        def _int(key):
            v = s.value(key, None)
            if v in (None, ""):
                return None
            try:
                n = int(v)
            except Exception:
                return None
            return n if n > 0 else None

        def _bool(key, default=True):
            v = s.value(key, None)
            if v is None:
                return default
            if isinstance(v, bool):
                return v
            return str(v).strip().lower() not in ("0", "false", "no", "off")

        out["h"] = _int(LAST_H)
        out["w"] = _int(LAST_W)
        out["frames"] = _int(LAST_FRAMES)
        out["detect"] = _bool(LAST_DETECT)
        out["band"] = _bool(LAST_BAND)
        out["warp"] = _bool(LAST_WARP)
    except Exception:
        pass
    return out


def _save_last_inputs(h, w, frames, detect, band, warp=True) -> None:
    """Remember the inputs so they are prefilled next time. Best effort."""
    try:
        from PyQt6.QtCore import QSettings
        s = QSettings()
        s.setValue(LAST_H, int(h))
        s.setValue(LAST_W, int(w))
        s.setValue(LAST_FRAMES, int(frames))
        s.setValue(LAST_DETECT, bool(detect))
        s.setValue(LAST_BAND, bool(band))
        s.setValue(LAST_WARP, bool(warp))
        s.sync()
    except Exception:
        pass


class TuneWorker(QObject):
    """Runs the sweeps off the GUI thread. Kept free of widget references."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, frame_shape, n_frames, do_detect=True, do_band=True,
                 do_warp=True, channels=1, ram_cap_rows=None):
        super().__init__()
        self._shape = frame_shape
        self._nframes = int(n_frames)
        self._do_detect = bool(do_detect)
        self._do_band = bool(do_band)
        self._do_warp = bool(do_warp)
        self._channels = int(channels)
        self._ram_cap = ram_cap_rows
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _cancelled(self):
        return self._cancel

    def run(self):
        out = {"detection": None, "band": None, "warp": None, "error": None}
        try:
            from setiastro.saspro import stacking_worker_tune as T
            if self._do_detect:
                out["detection"] = T.tune(
                    self._shape,
                    status_cb=self.progress.emit,
                    cancel_cb=self._cancelled,
                    save=True,
                )
            if self._do_band and not self._cancel:
                self.progress.emit("")
                out["band"] = T.tune_band_height(
                    self._shape, self._nframes, self._channels,
                    status_cb=self.progress.emit,
                    cancel_cb=self._cancelled,
                    save=True,
                    ram_cap_rows=self._ram_cap,
                )
            if self._do_warp and not self._cancel:
                self.progress.emit("")
                out["warp"] = T.tune_warp(
                    self._shape,
                    status_cb=self.progress.emit,
                    cancel_cb=self._cancelled,
                    save=True,
                )
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"
            self.progress.emit(f"⚠️ Tuning failed: {out['error']}")
        out["cancelled"] = self._cancel
        self.finished.emit(out)


class StackingTuneDialog(QDialog):
    """
    Modal-ish optimizer window: pick the frame size, run, watch, apply.

    `frame_shape` should be the user's real sensor size — the answer depends on
    it. 16 GPU workers was fine at 2048x2048 and wedged the same card at
    3000x4000, so tuning at the wrong size gives a wrong answer.
    """

    def __init__(self, parent=None, frame_shape=(3000, 4000), n_frames=24,
                 band_spinbox: QSpinBox | None = None, ram_cap_rows=None):
        super().__init__(parent)
        self.setWindowTitle("Optimize stacking for this computer")
        self.setMinimumSize(760, 620)
        self._band_spin = band_spinbox
        self._ram_cap = ram_cap_rows
        self._thread = None
        self._worker = None
        self._result = None
        self._measured = {}          # backend -> (workers, fps), from a Run

        # What was used last time wins over the caller's guess. Typing a
        # 46.8 MP sensor and a 64-frame depth once should be enough; the
        # caller's provider is only a starting point for a first run.
        last = _load_last_inputs()
        h0 = last.get("h") or int(frame_shape[0])
        w0 = last.get("w") or int(frame_shape[1])
        n0 = last.get("frames") or int(n_frames)

        form = QFormLayout()
        self.h_spin = QSpinBox(); self.h_spin.setRange(64, 65536)
        self.w_spin = QSpinBox(); self.w_spin.setRange(64, 65536)
        self.h_spin.setValue(int(h0))
        self.w_spin.setValue(int(w0))
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("H")); size_row.addWidget(self.h_spin)
        size_row.addWidget(QLabel("W")); size_row.addWidget(self.w_spin)
        size_row.addStretch(1)
        from PyQt6.QtWidgets import QWidget
        size_w = QWidget(); size_w.setLayout(size_row)
        form.addRow("Frame size:", size_w)

        # If the caller can see real lights and they disagree with what is in
        # the boxes, say so rather than overriding — a remembered value was a
        # deliberate choice, but a changed sensor is worth flagging.
        self.size_hint = QLabel("")
        self.size_hint.setEnabled(False)
        try:
            dh, dw = int(frame_shape[0]), int(frame_shape[1])
            if (dh, dw) != (3000, 4000) and (dh, dw) != (int(h0), int(w0)):
                self.size_hint.setText(
                    f"your loaded lights are {dw}x{dh} — tuning at a "
                    f"different size will give a different answer"
                )
                size_row.insertWidget(4, self.size_hint)
        except Exception:
            pass

        self.n_spin = QSpinBox(); self.n_spin.setRange(4, 256)
        self.n_spin.setValue(int(n0))
        self.n_spin.setToolTip(
            "Stack depth for the integration test. Use something close to a "
            "typical session; deeper stacks shift the band-height answer."
        )
        form.addRow("Frames in stack:", self.n_spin)

        self.cb_detect = QCheckBox("Star detection (registration workers)")
        self.cb_band = QCheckBox("Integration band height (rows per pass)")
        self.cb_warp = QCheckBox("Anti-ring warp: GPU or CPU")
        self.cb_warp.setToolTip(
            "Finalize warps each frame in a pool of worker processes. One of "
            "them can use the GPU instead, so the question is whether the "
            "card beats a SINGLE worker at its share of the cores — not "
            "whether it beats the whole pool.\n\n"
            "Measured on a 3070 Ti beside a 36-core Xeon the two are within a "
            "few percent and the CPU keeps it. A strong card with few cores "
            "is the case where this turns on."
        )
        self.cb_detect.setChecked(bool(last.get("detect", True)))
        self.cb_band.setChecked(bool(last.get("band", True)))
        self.cb_warp.setChecked(bool(last.get("warp", True)))
        form.addRow("Measure:", self.cb_detect)
        form.addRow("", self.cb_band)
        form.addRow("", self.cb_warp)

        self.note = QLabel(
            "Measures this machine rather than guessing. Takes a few minutes; "
            "the GPU will be busy. A configuration that overloads the card is "
            "abandoned automatically rather than left to thrash."
        )
        self.note.setWordWrap(True)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        try:
            from PyQt6.QtGui import QFont
            self.log.setFont(QFont("Consolas" if os.name == "nt" else "Monospace", 9))
        except Exception:
            pass

        self.btn_run = QPushButton("Run")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_apply = QPushButton("Save && Close")
        self.btn_close = QPushButton("Close")
        self.btn_apply.setDefault(True)
        self.btn_apply.setToolTip(
            "Write the settings above. A Run already saves what it measured, "
            "so this is for keeping your own edits on top of it."
        )
        self.btn_cancel.setEnabled(False)
        btns = QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_apply)
        btns.addWidget(self.btn_close)

        self.results = self._build_results()

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(self.note)
        root.addWidget(self.log, 1)
        root.addWidget(self.results)
        root.addLayout(btns)

        self.btn_run.clicked.connect(self._start)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_apply.clicked.connect(self._apply_and_close)
        self.btn_close.clicked.connect(self.close)

        self._load_results()
        self._append(self._hw_line())


    # ── results: what was measured, and what will actually be used ──────────
    # Separate widgets rather than log text, because a measurement the user
    # cannot override is a measurement they have to live with. The optimizer
    # is right most of the time; "most" is why these exist. A card that is
    # fine in every benchmark and unstable in a long run, a sensor the sweep
    # was not run at, a backend that regresses in a driver update — all of
    # them need a way out that is not editing QSettings by hand.
    def _build_results(self):
        box = QGroupBox("Settings — measured by Run, editable here")
        form = QFormLayout(box)

        self.cmb_detect = QComboBox()
        self.cmb_detect.addItem("GPU (torch)", True)
        self.cmb_detect.addItem("CPU (SEP)", False)
        self.cmb_detect.setToolTip(
            "Which star detector the registration solve pass uses.\n"
            "Writes stacking/align/gpu_detect."
        )
        form.addRow("Star detector:", self.cmb_detect)

        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, 64)
        self.spin_workers.setSpecialValueText("Auto")
        self.spin_workers.setToolTip(
            "Processes the solve pass may use. 0 = Auto.\n"
            "Each GPU worker carries its own CUDA context, so too many "
            "thrash the card instead of going faster.\n"
            "Writes stacking/align/workers."
        )
        self.lbl_workers = QLabel("")
        self.lbl_workers.setEnabled(False)
        wrow = QHBoxLayout()
        wrow.setContentsMargins(0, 0, 0, 0)
        wrow.addWidget(self.spin_workers)
        wrow.addWidget(self.lbl_workers, 1)
        ww = QWidget(); ww.setLayout(wrow)
        form.addRow("Alignment workers:", ww)

        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(1, 8192)
        self.spin_rows.setToolTip(
            "Full-width rows handed to one rejection call during "
            "integration.\nWrites stacking/chunk_height."
        )
        form.addRow("Rows per pass:", self.spin_rows)

        self.cmb_warp = QComboBox()
        self.cmb_warp.addItem("CPU (OpenCV)", False)
        self.cmb_warp.addItem("GPU (torch)", True)
        self.cmb_warp.setToolTip(
            "Where the anti-ring warp runs during finalize.\n\n"
            "Finalize is a pool, and one GPU slot replaces one worker — so "
            "the card has to beat a single worker, not the whole pool. On a "
            "strong CPU it usually will not.\n"
            "Writes stacking/align/gpu_warp."
        )
        form.addRow("Anti-ring warp:", self.cmb_warp)

        self.spin_slots = QSpinBox()
        self.spin_slots.setRange(1, 16)
        self.spin_slots.setToolTip(
            "How many finalize workers may use the card at once. Each one "
            "holds its own CUDA context and working set; oversubscribing an "
            "8 GB card does not error on Windows, it pages over PCIe and "
            "hangs.\nWrites stacking/align/gpu_warp_slots."
        )
        self.lbl_slots = QLabel("")
        self.lbl_slots.setEnabled(False)
        srow = QHBoxLayout()
        srow.setContentsMargins(0, 0, 0, 0)
        srow.addWidget(self.spin_slots)
        srow.addWidget(self.lbl_slots, 1)
        sw = QWidget(); sw.setLayout(srow)
        form.addRow("GPU warp slots:", sw)

        self.cmb_warp.currentIndexChanged.connect(self._warp_changed)
        self.cmb_detect.currentIndexChanged.connect(self._detect_changed)
        return box

    def _warp_changed(self):
        on = bool(self.cmb_warp.currentData())
        self.spin_slots.setEnabled(on)
        self.lbl_slots.setText("" if on else "(warp is on the CPU)")

    def _detect_changed(self):
        """Offer the worker count measured for THIS backend, not the winner's.

        6 GPU workers and 32 SEP workers were both measured on the same box.
        Switching the detector and keeping the other's number is a worse
        setting than either, and silently so.
        """
        key = "gpu" if self.cmb_detect.currentData() else "cpu"
        got = self._measured.get(key)
        if not got:
            self.lbl_workers.setText("")
            return
        nw, fps = got
        self.spin_workers.setValue(int(nw))
        self.lbl_workers.setText(f"measured best for {key.upper()}: "
                                 f"{nw} ({fps:.1f} frames/s)")

    def _load_results(self):
        """Fill the controls from QSettings — the dialog is usable without a run."""
        try:
            from PyQt6.QtCore import QSettings
            from setiastro.saspro import stacking_worker_tune as T
            s = QSettings()

            def _b(key, default):
                v = s.value(key, default)
                if isinstance(v, bool):
                    return v
                if v in (None, ""):
                    return default
                return str(v).strip().lower() not in ("0", "false", "no", "off")

            def _i(key, default):
                v = s.value(key, default)
                try:
                    return int(v)
                except Exception:
                    return default

            self.cmb_detect.setCurrentIndex(
                0 if _b("stacking/align/gpu_detect", True) else 1)
            self.spin_workers.setValue(_i(T.KEY_WORKERS_MANUAL, 0))
            self.spin_rows.setValue(max(1, _i(T.BAND_KEY, 16)))
            self.cmb_warp.setCurrentIndex(
                1 if _b(T.KEY_GPU_WARP, False) else 0)
            self.spin_slots.setValue(max(1, _i(T.KEY_GPU_WARP_SLOTS, 1)))
        except Exception:
            pass
        self._warp_changed()

    def _apply_results(self) -> bool:
        """Write the controls to QSettings. Returns True if it stuck."""
        try:
            from PyQt6.QtCore import QSettings
            from setiastro.saspro import stacking_worker_tune as T
            s = QSettings()
            s.setValue("stacking/align/gpu_detect",
                       bool(self.cmb_detect.currentData()))
            s.setValue(T.KEY_WORKERS_MANUAL, int(self.spin_workers.value()))
            s.setValue(T.BAND_KEY, int(self.spin_rows.value()))
            s.setValue(T.KEY_GPU_WARP, bool(self.cmb_warp.currentData()))
            s.setValue(T.KEY_GPU_WARP_SLOTS, int(self.spin_slots.value()))
            # the same guard tune_band_height uses, so a hand-set band height
            # is not eaten by the one-time migration either
            try:
                T._mark_band_schema_current(s, None)
            except Exception:
                pass
            s.sync()
        except Exception as e:
            self._append(f"⚠️ Could not save: {type(e).__name__}: {e}")
            return False
        if self._band_spin is not None:
            try:
                self._band_spin.setValue(int(self.spin_rows.value()))
            except Exception:
                pass
        return True

    def _apply_and_close(self):
        if self._apply_results():
            self._append("")
            self._append("Settings saved.")
        self.close()

    # ── helpers ─────────────────────────────────────────────────────────────
    def _hw_line(self):
        try:
            from setiastro.saspro.stacking_worker_tune import hardware_stamp
            return f"{hardware_stamp()}   ({os.cpu_count()} cores)"
        except Exception:
            return f"{os.cpu_count()} cores"

    def _append(self, line):
        self.log.appendPlainText(line)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── run / cancel ────────────────────────────────────────────────────────
    def _start(self):
        if self._thread is not None:
            return
        if not (self.cb_detect.isChecked() or self.cb_band.isChecked()
                or self.cb_warp.isChecked()):
            self._append("Nothing selected to measure.")
            return

        self._remember()
        self.log.clear()
        self._append(self._hw_line())
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_close.setEnabled(False)
        self.btn_apply.setEnabled(False)
        self.results.setEnabled(False)
        for wdg in (self.h_spin, self.w_spin, self.n_spin,
                    self.cb_detect, self.cb_band, self.cb_warp):
            wdg.setEnabled(False)

        self._worker = TuneWorker(
            (self.h_spin.value(), self.w_spin.value()),
            self.n_spin.value(),
            do_detect=self.cb_detect.isChecked(),
            do_band=self.cb_band.isChecked(),
            do_warp=self.cb_warp.isChecked(),
            ram_cap_rows=self._ram_cap,
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._append,
                                      Qt.ConnectionType.QueuedConnection)
        self._worker.finished.connect(self._done,
                                      Qt.ConnectionType.QueuedConnection)
        self._thread.start()

    def _cancel(self):
        if self._worker is not None:
            self._append("⏹ Cancelling at the next batch boundary…")
            self._worker.cancel()
        self.btn_cancel.setEnabled(False)

    def _done(self, result):
        self._result = result
        try:
            self._thread.quit()
            self._thread.wait(5000)
        except Exception:
            pass
        self._thread = None
        self._worker = None

        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_close.setEnabled(True)
        self.btn_apply.setEnabled(True)
        self.results.setEnabled(True)
        for wdg in (self.h_spin, self.w_spin, self.n_spin,
                    self.cb_detect, self.cb_band, self.cb_warp):
            wdg.setEnabled(True)

        # Remember what each backend managed, so switching the detector in
        # the panel can offer that backend's own number instead of the
        # winner's.
        det0 = result.get("detection") or {}
        for key in ("gpu", "cpu"):
            good = [r for r in (det0.get(key) or [])
                    if not r.get("aborted") and r.get("fps", 0) > 0]
            if good:
                best = max(good, key=lambda r: r["fps"])
                self._measured[key] = (int(best["workers"]),
                                       float(best["fps"]))
        # the sweeps wrote their verdicts straight to QSettings, so the panel
        # has to re-read rather than keep showing what was there before
        self._load_results()

        if result.get("cancelled"):
            self._append("")
            self._append("Cancelled — nothing was changed.")
            return
        if result.get("error"):
            return

        self._append("")
        det = result.get("detection") or {}
        band = result.get("band") or {}
        if det.get("backend"):
            self._append(
                f"Registration: {det['backend'].upper()} detector, "
                f"{det['workers']} worker(s)  ({det['fps']:.1f} frames/s)"
            )
        if band.get("rows"):
            self._append(
                f"Integration:  {band['rows']} rows per pass  "
                f"({band['mpx_s']:.1f} Mpx/s)"
            )
            if self._band_spin is not None:
                try:
                    self._band_spin.setValue(int(band["rows"]))
                except Exception:
                    pass
        warp = result.get("warp") or {}
        if warp.get("t_gpu") is not None:
            tg, tc = warp["t_gpu"], warp["t_cpu"]
            self._append(
                f"Warp:         {'GPU' if warp['use_gpu'] else 'CPU'}  "
                f"(card {tg:.2f}s vs one worker {tc:.2f}s; "
                f"{warp['gain']:.2f}x finalize throughput)"
            )
        elif warp:
            self._append("Warp:         CPU (no GPU warp available to measure)")
        if (not det.get("backend") and not band.get("rows")
                and not warp.get("t_gpu")):
            self._append("No usable result; settings left unchanged.")
        else:
            self._append("")
            self._append("Saved. Change anything above and use Save & Close "
                         "to keep your own values instead.")
            self._append("Re-run this if you change GPU or sensor.")

    @property
    def result(self):
        return self._result

    def _remember(self):
        _save_last_inputs(self.h_spin.value(), self.w_spin.value(),
                          self.n_spin.value(), self.cb_detect.isChecked(),
                          self.cb_band.isChecked(), self.cb_warp.isChecked())

    def closeEvent(self, ev):
        # Remember even when the window is closed without running: the point
        # is not to retype a 46.8 MP sensor, and that is just as true after a
        # cancel as after a completed sweep.
        self._remember()
        if self._thread is not None:
            self._cancel()
            try:
                self._thread.quit()
                self._thread.wait(10000)
            except Exception:
                pass
        super().closeEvent(ev)


def install_optimize_button(parent_dialog, layout, band_spinbox=None,
                            frame_shape_provider=None, on_finished=None):
    """
    Add an "Optimize…" button to `layout`, opening the tuner.

    `frame_shape_provider` is an optional callable returning (H, W) so the
    dialog opens pre-filled with the user's real sensor size. Returns the
    button, or None if it could not be created (never raises: a settings dialog
    must still open if this fails).
    """
    try:
        btn = QPushButton("Optimize…")
        btn.setToolTip(
            "Measure this computer and set the star-detection backend, the "
            "registration worker count, and the rows per pass."
        )

        def _open():
            shape = (3000, 4000)
            if frame_shape_provider is not None:
                try:
                    got = frame_shape_provider()
                    if got and len(got) >= 2 and int(got[0]) > 0:
                        shape = (int(got[0]), int(got[1]))
                except Exception:
                    pass
            dlg = StackingTuneDialog(parent_dialog, frame_shape=shape,
                                     band_spinbox=band_spinbox)
            dlg.exec()
            # The optimizer writes its result straight to QSettings, so any
            # control already on screen is showing a stale value until it is
            # told to re-read. Called after exec() returns, i.e. once the
            # window is closed, whether the run finished or was cancelled.
            if on_finished is not None:
                try:
                    on_finished()
                except Exception:
                    pass

        btn.clicked.connect(_open)
        layout.addWidget(btn)
        return btn
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Shared "Alignment workers" control
#
# One factory, used by BOTH the Stacking Suite settings page and the Star
# Registration window, writing one QSettings key — the same arrangement
# `stacking/align/kernel` already uses, so a user's choice follows them between
# the two dialogs instead of each one keeping its own idea of the number.
#
# 0 means Auto. The spin box shows what Auto currently resolves to, so the
# number in front of the user is always the number the solve pass will use.
# ─────────────────────────────────────────────────────────────────────────────
def make_align_workers_row(parent, *, band_spinbox=None,
                           frame_shape_provider=None, with_optimize=True):
    """
    Build the row: [spin box] [Auto resolves to N] [Optimize…].

    Returns (container_widget, spinbox). The spin box reads and writes
    `stacking/align/workers` (0 = Auto). Never raises — a settings page must
    still open if any of this fails; on failure it returns (None, None).
    """
    try:
        from PyQt6.QtWidgets import QWidget

        spin = QSpinBox()
        spin.setRange(0, 64)
        spin.setSpecialValueText("Auto")
        spin.setToolTip(
            "Processes the star-alignment solve pass may use.\n\n"
            "Auto measures nothing: it uses the optimizer's result if you "
            "have run it, otherwise a value derived from your VRAM.\n\n"
            "This matters more with the GPU detector than with SEP. Each "
            "worker carries its own CUDA context, so too many will thrash the "
            "card instead of going faster — on an 8 GB card at 12 MP, 8 "
            "workers was the measured optimum and 16 wedged it outright. SEP "
            "is pure CPU and scales to the full core count."
        )

        hint = QLabel("")
        hint.setEnabled(False)

        def _effective():
            try:
                from setiastro.saspro.stacking_worker_tune import (
                    resolve_align_workers, stamp_is_current,
                )
                mp_ = None
                if frame_shape_provider is not None:
                    try:
                        h, w = frame_shape_provider()[:2]
                        mp_ = (float(h) * float(w)) / 1e6
                    except Exception:
                        mp_ = None
                n, why = resolve_align_workers(mp_)
                stale = stamp_is_current() is False
                return n, why, stale
            except Exception:
                return None, "", False

        def _refresh():
            n, why, stale = _effective()
            if spin.value() == 0:
                hint.setText(f"Auto → {n} ({why})" if n else "")
            elif stale:
                hint.setText("measured on different hardware — re-run Optimize")
            else:
                hint.setText("")

        def _store(v):
            try:
                from PyQt6.QtCore import QSettings
                from setiastro.saspro.stacking_worker_tune import (
                    KEY_WORKERS_MANUAL,
                )
                QSettings().setValue(KEY_WORKERS_MANUAL, int(v))
            except Exception:
                pass
            _refresh()

        try:
            from PyQt6.QtCore import QSettings
            from setiastro.saspro.stacking_worker_tune import KEY_WORKERS_MANUAL
            cur = QSettings().value(KEY_WORKERS_MANUAL, 0)
            spin.setValue(int(cur) if cur not in (None, "") else 0)
        except Exception:
            spin.setValue(0)

        spin.valueChanged.connect(_store)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(spin)
        row.addWidget(hint, 1)

        if with_optimize:
            def _after_tune():
                # the optimizer writes the measured number into the same key,
                # so pull it back into the spin box the user is looking at
                try:
                    from PyQt6.QtCore import QSettings
                    from setiastro.saspro.stacking_worker_tune import (
                        KEY_WORKERS_MANUAL,
                    )
                    v = QSettings().value(KEY_WORKERS_MANUAL, 0)
                    spin.blockSignals(True)
                    spin.setValue(int(v) if v not in (None, "") else 0)
                    spin.blockSignals(False)
                except Exception:
                    pass
                _refresh()

            install_optimize_button(parent, row, band_spinbox=band_spinbox,
                                    frame_shape_provider=frame_shape_provider,
                                    on_finished=_after_tune)

        _refresh()
        w = QWidget()
        w.setLayout(row)
        return w, spin
    except Exception:
        return None, None