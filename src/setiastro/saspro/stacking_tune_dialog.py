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
    QCheckBox, QDialog, QFormLayout, QHBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QSpinBox, QVBoxLayout,
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
    except Exception:
        pass
    return out


def _save_last_inputs(h, w, frames, detect, band) -> None:
    """Remember the inputs so they are prefilled next time. Best effort."""
    try:
        from PyQt6.QtCore import QSettings
        s = QSettings()
        s.setValue(LAST_H, int(h))
        s.setValue(LAST_W, int(w))
        s.setValue(LAST_FRAMES, int(frames))
        s.setValue(LAST_DETECT, bool(detect))
        s.setValue(LAST_BAND, bool(band))
        s.sync()
    except Exception:
        pass


class TuneWorker(QObject):
    """Runs the sweeps off the GUI thread. Kept free of widget references."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, frame_shape, n_frames, do_detect=True, do_band=True,
                 channels=1, ram_cap_rows=None):
        super().__init__()
        self._shape = frame_shape
        self._nframes = int(n_frames)
        self._do_detect = bool(do_detect)
        self._do_band = bool(do_band)
        self._channels = int(channels)
        self._ram_cap = ram_cap_rows
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _cancelled(self):
        return self._cancel

    def run(self):
        out = {"detection": None, "band": None, "error": None}
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
        self.setMinimumSize(720, 480)
        self._band_spin = band_spinbox
        self._ram_cap = ram_cap_rows
        self._thread = None
        self._worker = None
        self._result = None

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
        self.cb_detect.setChecked(bool(last.get("detect", True)))
        self.cb_band.setChecked(bool(last.get("band", True)))
        form.addRow("Measure:", self.cb_detect)
        form.addRow("", self.cb_band)

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
        self.btn_close = QPushButton("Close")
        self.btn_cancel.setEnabled(False)
        btns = QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_close)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(self.note)
        root.addWidget(self.log, 1)
        root.addLayout(btns)

        self.btn_run.clicked.connect(self._start)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_close.clicked.connect(self.close)

        self._append(self._hw_line())

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
        if not (self.cb_detect.isChecked() or self.cb_band.isChecked()):
            self._append("Nothing selected to measure.")
            return

        self._remember()
        self.log.clear()
        self._append(self._hw_line())
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_close.setEnabled(False)
        for wdg in (self.h_spin, self.w_spin, self.n_spin,
                    self.cb_detect, self.cb_band):
            wdg.setEnabled(False)

        self._worker = TuneWorker(
            (self.h_spin.value(), self.w_spin.value()),
            self.n_spin.value(),
            do_detect=self.cb_detect.isChecked(),
            do_band=self.cb_band.isChecked(),
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
        for wdg in (self.h_spin, self.w_spin, self.n_spin,
                    self.cb_detect, self.cb_band):
            wdg.setEnabled(True)

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
            # push the measured value into the settings dialog's own control so
            # the user sees it land, and so Save writes it like any manual edit
            if self._band_spin is not None:
                try:
                    self._band_spin.setValue(int(band["rows"]))
                    self._append("              (applied to 'Rows per pass')")
                except Exception:
                    pass
        if not det.get("backend") and not band.get("rows"):
            self._append("No usable result; settings left unchanged.")
        else:
            self._append("")
            self._append("Saved. Re-run this if you change GPU or sensor.")

    @property
    def result(self):
        return self._result

    def _remember(self):
        _save_last_inputs(self.h_spin.value(), self.w_spin.value(),
                          self.n_spin.value(), self.cb_detect.isChecked(),
                          self.cb_band.isChecked())

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