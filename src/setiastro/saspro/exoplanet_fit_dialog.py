# src/setiastro/saspro/exoplanet_fit_dialog.py
# Copyright (c) Franklin Marek / SetiAstro (www.setiastro.com). GPLv3.
"""
Transit fit dialog for the exoplanet tool -- the EXOTIC-style workflow:

  1. priors from the NASA Exoplanet Archive (by name or by the target's
     position), limb darkening from Claret (2011) for the filter;
  2. selection: every aperture radius x a family of comparison sets (the
     weighted ensemble, its best-n subsets, and each of the best single
     comps), each scored by EXOTIC's metric, std(residuals)/median, from a
     least-squares transit + airmass fit;
  3. MCMC on the winner, errors scaled by reduced chi^2 and red-noise beta;
  4. AAVSO Exoplanet Database report, CSV, plot.

All the numerics live in exoplanet_transit / exoplanet_photometry; this file
is the Qt shell plus `run_fit_job`, which is Qt-free so it can be tested.
"""
from __future__ import annotations

import datetime as _dt
import math
import traceback
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, QTimer, QSettings, pyqtSignal, QCoreApplication
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QSpinBox, QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from setiastro.saspro import exoplanet_transit as xfit
from setiastro.saspro import exoplanet_photometry as xphot
from setiastro.saspro.widgets.factory_reset import (
    make_reset_button, snapshot_controls, restore_controls)


def _tr(text: str) -> str:
    return QCoreApplication.translate("ExoplanetFitDialog", text)


# ═════════════════════════ Qt-free fit orchestration ══════════════════════

def run_fit_job(job: dict, progress=None, cancelled=None) -> dict:
    """
    Aperture x comparison-set selection, then the full fit.

    job keys: flux, err (n_ap, n_s, n_f); valid (n_ap, n_s, n_f) bool;
    radii; target; t (n_f); airmass (n_f); priors; options; ap_mode
    ('auto' or int); comp_mode ('auto' | 'ensemble'); max_comps; exclude.
    """
    say = progress or (lambda p, m: None)
    stop = cancelled or (lambda: False)
    F, E, V = job["flux"], job["err"], job["valid"]
    target = int(job["target"])
    t = np.asarray(job["t"], dtype=np.float64)
    X = np.asarray(job["airmass"], dtype=np.float64)
    P = job.get("side")
    P = np.asarray(P, dtype=np.float64) if P is not None and len(P) == t.size else None
    pr: xfit.TransitPriors = job["priors"]
    opt: xfit.FitOptions = job["options"]
    n_ap, n_s, n_f = F.shape
    max_comps = int(job.get("max_comps", 10))
    exclude = [target] + [int(i) for i in job.get("exclude", ()) if int(i) != target]
    sat_ok = np.asarray(job.get("eligible", np.ones(n_s, bool)), bool)

    aps = list(range(n_ap)) if job.get("ap_mode", "auto") == "auto" else [int(job["ap_mode"])]
    trials = []
    frame_noise = {}
    for k in aps:
        ens = xphot.build_ensemble(F[k], E[k], V[k], max_comps=max(max_comps, 8),
                                   exclude=exclude, eligible=sat_ok)
        comps = list(ens.comps)
        w = list(ens.weights)
        if not comps:
            continue
        # Per-frame noise beyond the CCD equation, measured from this
        # aperture's comparison ensemble; every trial set inherits it.
        frame_noise[k] = xphot.frame_excess_noise(F[k], E[k], V[k], ens.comps, ens.weights,
                                                  t=t)
        sets = []
        if job.get("comp_mode", "auto") == "auto":
            for n in sorted({1, 2, 3, 5, 8, len(comps)}):
                if 1 <= n <= len(comps):
                    sets.append((f"best {n}" if n < len(comps) else f"ensemble ({n})",
                                 comps[:n], w[:n]))
            for j in range(1, min(5, len(comps))):
                sets.append((f"single #{comps[j]}", [comps[j]], [1.0]))
        else:
            sets.append((f"ensemble ({len(comps)})", comps, w))
        trials.append((k, sets))
    n_trials = sum(len(s) for _, s in trials)
    if n_trials == 0:
        raise RuntimeError("No usable comparison stars for this target.")

    scores = []
    done = 0
    for k, sets in trials:
        for label, cs, ws in sets:
            if stop():
                raise RuntimeError("Cancelled.")
            rel, rerr, ok = xphot.target_lightcurve(F[k], E[k], V[k], target, cs, ws,
                                                    frame_noise=frame_noise.get(k))
            if np.count_nonzero(ok) >= max(20, int(0.5 * n_f)):
                try:
                    sc, _th, _f = xfit.fit_quick(t[ok], rel[ok], rerr[ok], X[ok], pr, opt,
                                                 side=(P[ok] if P is not None else None))
                except Exception:
                    sc = float("inf")
                scores.append({"ap": k, "radius": float(job["radii"][k]), "label": label,
                               "comps": [int(c) for c in cs], "weights": [float(x) for x in ws],
                               "score": float(sc), "n": int(np.count_nonzero(ok))})
            done += 1
            say(int(30 * done / n_trials), _tr("Selecting aperture/comparisons {0}/{1}").format(done, n_trials))
    good = [s for s in scores if np.isfinite(s["score"])]
    if not good:
        raise RuntimeError("Every aperture/comparison combination failed to fit.")
    best = min(good, key=lambda s: s["score"])

    k = best["ap"]
    fn = frame_noise.get(k)
    rel, rerr, ok = xphot.target_lightcurve(F[k], E[k], V[k], target, best["comps"], best["weights"],
                                            frame_noise=fn)
    best["frame_noise_ppt"] = (float(np.median(fn[ok]) * 1e3)
                               if fn is not None and np.any(ok) else 0.0)

    def _p(p, m):
        say(30 + int(0.7 * p), m)

    res = xfit.fit_transit(t[ok], rel[ok], rerr[ok], X[ok], pr, opt,
                           progress=_p, cancelled=stop,
                           side=(P[ok] if P is not None else None))
    return {"result": res, "best": best, "scores": scores,
            "rel": rel, "err": rerr, "ok": ok, "t": t, "airmass": X}


# ═══════════════════════════════ workers ══════════════════════════════════

_LIVE_THREADS: set = set()


def _start_thread(th: QThread):
    """
    Keep a module-level reference to every running worker and only ever drop
    references to threads that have already finished.  Dropping the last
    Python reference to a running QThread (parent=None) is the classic
    "QThread: Destroyed while thread is still running" fatal.
    """
    for old in list(_LIVE_THREADS):
        try:
            if old.isFinished():
                _LIVE_THREADS.discard(old)
        except RuntimeError:
            _LIVE_THREADS.discard(old)
    _LIVE_THREADS.add(th)
    th.start()


class _FitWorker(QThread):
    # NB: no custom `finished` signal -- that shadows QThread.finished.
    progress_msg = pyqtSignal(int, str)
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, job: dict):
        super().__init__(None)
        self.job = job
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            out = run_fit_job(self.job, progress=lambda p, m: self.progress_msg.emit(int(p), str(m)),
                              cancelled=lambda: self._cancel)
            self.result_ready.emit(out)
        except Exception as e:
            if self._cancel:
                self.failed.emit("Cancelled.")
            else:
                traceback.print_exc()
                self.failed.emit(f"{type(e).__name__}: {e}")


class _CallWorker(QThread):
    """Run one blocking call (archive / VizieR lookup) off the GUI thread."""
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__(None)
        self._fn, self._args, self._kw = fn, args, kwargs

    def run(self):
        try:
            self.result_ready.emit(self._fn(*self._args, **self._kw))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


# ════════════════════════════════ dialog ══════════════════════════════════

def _dspin(lo, hi, dec, val=0.0, step=None):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setValue(val)
    s.setSingleStep(step if step is not None else 10 ** (-max(0, dec - 2)))
    s.setKeyboardTracking(False)
    return s


class TransitFitDialog(QDialog):
    """
    ctx (built by ExoPlanetWindow._transit_context):
      target, label, ra, dec, jd_utc, exptime_s, site, airmass, filter_kw,
      object_name, binning, camera_hint, flux, err, valid, radii, best_ap,
      eligible, exclude, max_comps, star_xy (ref positions), wcs_to_radec
      (callable x,y -> (ra,dec) or None), screen (metrics dict or None),
      rel, rel_err, rel_ok (preview curve)
    """

    def __init__(self, parent, ctx: dict):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle(self.tr("Transit Fit — {0}").format(ctx.get('label', self.tr('target'))))
        self.resize(1250, 820)
        self._worker = None
        self._out = None            # last run_fit_job output
        self._t = None              # time array used for the fit
        self._time_kind = "JD_UTC"
        self._lookup_rows = []

        root = QHBoxLayout(self)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)

        # ── target / planet ───────────────────────────────────────────
        g_t = QGroupBox(self.tr("Target && planet"))
        ft = QFormLayout(g_t)
        ra, dec = ctx.get("ra"), ctx.get("dec")
        pos = (f"RA {ra:.5f}°, Dec {dec:+.5f}°" if ra is not None and dec is not None
               else self.tr("no WCS position"))
        self.target_lbl = QLabel(f"{ctx.get('label', '')}   ({pos})")
        self.target_lbl.setWordWrap(True)
        ft.addRow(self.target_lbl)
        prow = QHBoxLayout()
        self.planet_edit = QLineEdit(ctx.get("object_name", "") or "")
        self.planet_edit.setPlaceholderText(self.tr("Planet or host name, e.g. HAT-P-32 b"))
        prow.addWidget(self.planet_edit, 1)
        self.lookup_btn = QPushButton(self.tr("Look up"))
        self.lookup_btn.setToolTip(self.tr("NASA Exoplanet Archive (pscomppars) by planet or host name"))
        self.lookup_btn.clicked.connect(self._lookup_name)
        prow.addWidget(self.lookup_btn)
        self.cone_btn = QPushButton(self.tr("At position"))
        self.cone_btn.setToolTip(self.tr("Known planets within 20\" of this star's plate-solved position"))
        self.cone_btn.clicked.connect(self._lookup_position)
        self.cone_btn.setEnabled(ra is not None and dec is not None)
        prow.addWidget(self.cone_btn)
        ft.addRow(self.tr("Planet:"), prow)
        self.lookup_status = QLabel("")
        self.lookup_status.setWordWrap(True)
        self.lookup_status.setStyleSheet("color: #8fb8de;")
        ft.addRow(self.lookup_status)
        self.dip_btn = QPushButton(self.tr("Unknown planet: start from detected dip"))
        self.dip_btn.setToolTip(self.tr(
            "Fill Tc, depth and duration from the screening box search. The\n"
            "period is nominal (3 d) and a/R* is set free, so only Tc, depth\n"
            "and duration are meaningful."))
        self.dip_btn.clicked.connect(self._fill_from_dip)
        sc = ctx.get("screen") or {}
        self.dip_btn.setEnabled(bool(sc) and np.isfinite(sc.get("dip_t", np.nan)))
        ft.addRow(self.dip_btn)

        # One night per fit -- EXOTIC's model too: one light curve, one Tc,
        # one AAVSO report.  Each night has its own systematics.
        n_f = len(ctx["jd_utc"])
        S = ctx.get("session_idx")
        self._S = (np.asarray(S, dtype=int) if S is not None and len(S) == n_f
                   else np.zeros(n_f, dtype=int))
        self._sessions = list(ctx.get("sessions") or []) or [
            {"index": 0, "label": self.tr("All frames — {0} frames").format(n_f), "start_jd": float("nan")}]
        self.night_combo = QComboBox()
        for s in self._sessions:
            self.night_combo.addItem(s["label"], int(s["index"]))
        self._night_from_user = ctx.get("active_session") is not None
        start = ctx.get("active_session")
        if start is None:
            dip_t = float(sc.get("dip_t", np.nan)) if sc else float("nan")
            if np.isfinite(dip_t):
                start = self._session_of_jd(float(ctx["jd_utc"][0]) + dip_t)
            else:
                start = int(np.bincount(self._S).argmax())
        i0 = self.night_combo.findData(int(start))
        self.night_combo.setCurrentIndex(i0 if i0 >= 0 else 0)
        self.night_combo.currentIndexChanged.connect(self._on_night_changed)
        self.night_lbl = QLabel(self.tr("Night:"))
        ft.addRow(self.night_lbl, self.night_combo)
        self.night_lbl.setVisible(len(self._sessions) > 1)
        self.night_combo.setVisible(len(self._sessions) > 1)
        lv.addWidget(g_t)

        # ── ephemeris & system ─────────────────────────────────────────
        g_p = QGroupBox(self.tr("Priors"))
        fp = QFormLayout(g_p)
        self.per = _dspin(0.05, 5000.0, 7, 3.0, 0.0001)
        self.per_e = _dspin(0.0, 10.0, 7, 0.0, 1e-6)
        self.tc = _dspin(0.0, 3.0e6, 6, 0.0, 0.001)
        self.tc_e = _dspin(0.0, 10.0, 6, 0.0, 0.0001)
        self.rp = _dspin(0.001, 0.7, 5, 0.1, 0.001)
        self.rp_e = _dspin(0.0, 0.5, 5, 0.0, 0.001)
        self.ars = _dspin(1.1, 500.0, 3, 10.0, 0.1)
        self.ars_e = _dspin(0.0, 100.0, 3, 0.0, 0.1)
        self.inc = _dspin(50.0, 90.0, 3, 90.0, 0.1)
        self.inc_e = _dspin(0.0, 20.0, 3, 0.0, 0.1)
        self.ecc = _dspin(0.0, 0.95, 4, 0.0, 0.01)
        self.omega = _dspin(0.0, 360.0, 2, 90.0, 1.0)
        for lbl, a, b in ((self.tr("Period [d]"), self.per, self.per_e),
                          (self.tr("Tc ref [BJD_TDB]"), self.tc, self.tc_e),
                          ("Rp/R*", self.rp, self.rp_e),
                          ("a/R*", self.ars, self.ars_e),
                          (self.tr("Inclination [°]"), self.inc, self.inc_e)):
            h = QHBoxLayout()
            h.addWidget(a, 3)
            h.addWidget(QLabel("±"))
            h.addWidget(b, 2)
            fp.addRow(lbl + ":", h)
        h = QHBoxLayout()
        h.addWidget(self.ecc)
        h.addWidget(QLabel("ω [°]"))
        h.addWidget(self.omega)
        fp.addRow(self.tr("Eccentricity:"), h)
        lv.addWidget(g_p)

        # ── limb darkening ─────────────────────────────────────────────
        g_l = QGroupBox(self.tr("Filter && limb darkening (quadratic)"))
        fl = QFormLayout(g_l)
        self.filter_combo = QComboBox()
        for code, (desc, lo, hi, band) in xfit.FILTERS.items():
            approx = " (LD ≈ " + band + ")" if code in xfit.APPROXIMATE_LD_BANDS else ""
            self.filter_combo.addItem(f"{code} — {desc}{approx}", code)
        guess = xfit.guess_filter_code(ctx.get("filter_kw", "") or "")
        fi = self.filter_combo.findData(guess) if guess else -1
        self.filter_combo.setCurrentIndex(fi if fi >= 0 else 0)
        fl.addRow(self.tr("Filter (header: {0}):").format(ctx.get('filter_kw') or '—'), self.filter_combo)
        self.teff = _dspin(0.0, 50000.0, 0, 0.0, 50.0)
        self.teff.setSpecialValueText(self.tr("unknown"))
        self.logg = _dspin(0.0, 6.0, 2, 4.5, 0.1)
        self.feh = _dspin(-5.0, 1.0, 2, 0.0, 0.1)
        h = QHBoxLayout()
        h.addWidget(QLabel("Teff"))
        h.addWidget(self.teff)
        h.addWidget(QLabel("log g"))
        h.addWidget(self.logg)
        h.addWidget(QLabel("[Fe/H]"))
        h.addWidget(self.feh)
        fl.addRow(self.tr("Star:"), h)
        self.u1 = _dspin(-1.0, 2.0, 4, 0.465, 0.01)
        self.u2 = _dspin(-1.0, 2.0, 4, 0.257, 0.01)
        h = QHBoxLayout()
        h.addWidget(QLabel("u1"))
        h.addWidget(self.u1)
        h.addWidget(QLabel("u2"))
        h.addWidget(self.u2)
        self.ld_btn = QPushButton("Claret 2011")
        self.ld_btn.setToolTip(self.tr("Quadratic coefficients for this star and filter from\n"
                               "Claret & Bloemen 2011 (VizieR J/A+A/529/A75)."))
        self.ld_btn.clicked.connect(self._lookup_ld)
        h.addWidget(self.ld_btn)
        fl.addRow(self.tr("Coefficients:"), h)
        self.ld_src = QLabel(self.tr("Default: solar-type V band (Claret 2011, Teff 5750, log g 4.5)."))
        self.ld_src.setWordWrap(True)
        self.ld_src.setStyleSheet("color: #8fb8de;")
        fl.addRow(self.ld_src)
        lv.addWidget(g_l)

        # ── fit options ────────────────────────────────────────────────
        g_f = QGroupBox(self.tr("Fit"))
        ff = QFormLayout(g_f)
        self.sys_combo = QComboBox()
        for key, lab in xfit.SYSTEMATICS_LABELS.items():
            self.sys_combo.addItem(lab, key)
        self.free_inc = QCheckBox(self.tr("Inclination (±5°, Gaussian prior if ± given)"))
        self.free_inc.setChecked(True)
        self.free_ars = QCheckBox(self.tr("a/R* (Gaussian prior if ± given)"))
        self.free_ld = QCheckBox(self.tr("Limb darkening (σ=0.1 prior)"))
        ff.addRow(self.tr("Systematics:"), self.sys_combo)
        ff.addRow(self.tr("Also free:"), self.free_inc)
        ff.addRow("", self.free_ars)
        ff.addRow("", self.free_ld)
        self.ap_combo = QComboBox()
        self.ap_combo.addItem(self.tr("Auto (lowest residual scatter)"), "auto")
        for k, r in enumerate(ctx["radii"]):
            self.ap_combo.addItem(f"r = {r:.2f} px", k)
        ff.addRow(self.tr("Aperture:"), self.ap_combo)
        self.comp_combo = QComboBox()
        self.comp_combo.addItem(self.tr("Auto (ensemble subsets + single comps)"), "auto")
        self.comp_combo.addItem(self.tr("Full weighted ensemble"), "ensemble")
        ff.addRow(self.tr("Comparisons:"), self.comp_combo)
        self.clip = _dspin(0.0, 20.0, 1, 4.0, 0.5)
        self.clip.setSpecialValueText(self.tr("off"))
        ff.addRow(self.tr("Outlier clip [σ]:"), self.clip)
        self.steps = QSpinBox()
        self.steps.setRange(300, 50000)
        self.steps.setSingleStep(500)
        self.steps.setValue(2000)
        ff.addRow(self.tr("MCMC steps:"), self.steps)
        self.use_beta = QCheckBox(self.tr("Inflate errors by red-noise β"))
        self.use_beta.setChecked(True)
        ff.addRow("", self.use_beta)
        lv.addWidget(g_f)

        self.time_lbl = QLabel("")
        self.time_lbl.setWordWrap(True)
        self.time_lbl.setStyleSheet("color: #c8c8c8;")
        lv.addWidget(self.time_lbl)

        brow = QHBoxLayout()
        self.run_btn = QPushButton(self.tr("Run Fit"))
        self.run_btn.setStyleSheet("font-weight: bold;")
        self.run_btn.clicked.connect(self._run_or_cancel)
        brow.addWidget(self.run_btn)
        # Reset covers the Fit options; priors, limb darkening and the
        # filter describe the target (archive / Claret lookups) and stay.
        self._factory = snapshot_controls(g_f)
        self.reset_btn = make_reset_button(self)
        self.reset_btn.setToolTip(self.tr("Reset the Fit options to their factory defaults"))
        self.reset_btn.clicked.connect(self._reset_to_factory)
        brow.addWidget(self.reset_btn)
        self.aavso_btn = QPushButton(self.tr("AAVSO Report…"))
        self.aavso_btn.clicked.connect(self._export_aavso)
        self.csv_btn = QPushButton("CSV…")
        self.csv_btn.clicked.connect(self._export_csv)
        self.png_btn = QPushButton(self.tr("Save Plot…"))
        self.png_btn.setToolTip(self.tr("Graphical: PNG screenshots of both panels.\n"
                                "Scientific: one figure as 300 dpi PNG, PDF or SVG."))
        self.png_btn.clicked.connect(self._export_png)
        for b in (self.aavso_btn, self.csv_btn, self.png_btn):
            b.setEnabled(False)
            brow.addWidget(b)
        lv.addLayout(brow)
        self.prog = QProgressBar()
        self.prog.setRange(0, 100)
        self.prog.setVisible(False)
        self.prog_lbl = QLabel("")
        lv.addWidget(self.prog)
        lv.addWidget(self.prog_lbl)
        lv.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setMinimumWidth(470)

        # ── plots + results ────────────────────────────────────────────
        right = QSplitter(Qt.Orientation.Vertical)
        self.p_lc = pg.PlotWidget(title=self.tr("Light curve"))
        self.p_lc.showGrid(True, True, alpha=0.25)
        self.p_lc.setLabel("left", self.tr("Relative flux"))
        self.p_res = pg.PlotWidget(title=self.tr("Residuals"))
        self.p_res.showGrid(True, True, alpha=0.25)
        self.p_res.setLabel("left", "ppt")
        self.p_res.setXLink(self.p_lc)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        f = self.summary.font()
        f.setFamily("Consolas")
        self.summary.setFont(f)
        pg_pair = QSplitter(Qt.Orientation.Vertical)
        pg_pair.addWidget(self.p_lc)
        pg_pair.addWidget(self.p_res)
        pg_pair.setSizes([430, 170])
        self._pg_pair = pg_pair
        self.sci_plot = None            # matplotlib "Scientific" figure, built on first use
        self.plot_stack = QStackedWidget()
        self.plot_stack.addWidget(pg_pair)
        ps = QSettings()
        srow = QHBoxLayout()
        srow.setContentsMargins(0, 0, 0, 0)
        srow.addWidget(QLabel(self.tr("Plot:")))
        self.plot_style_combo = QComboBox()
        self.plot_style_combo.addItem(self.tr("Graphical"), "graphical")
        self.plot_style_combo.addItem(self.tr("Scientific"), "scientific")
        self.plot_style_combo.setToolTip(self.tr(
            "Scientific: the publication figure -- detrended flux with error bars,\n"
            "binned points and the model, residuals underneath."))
        self.plot_bin_combo = QComboBox()
        for lab, mins in ((self.tr("No bins"), 0), (self.tr("5-min bins"), 5), (self.tr("10-min bins"), 10),
                          (self.tr("15-min bins"), 15), (self.tr("30-min bins"), 30)):
            self.plot_bin_combo.addItem(lab, mins)
        for cb, key, default in ((self.plot_style_combo, "Exoplanet/plot_style", "graphical"),
                                 (self.plot_bin_combo, "Exoplanet/plot_bin_min", 10)):
            want = str(ps.value(key, default))
            for j in range(cb.count()):
                if str(cb.itemData(j)) == want:
                    cb.setCurrentIndex(j)
                    break
        srow.addWidget(self.plot_style_combo)
        srow.addSpacing(8)
        srow.addWidget(self.plot_bin_combo)
        srow.addStretch(1)
        plot_area = QWidget()
        pa = QVBoxLayout(plot_area)
        pa.setContentsMargins(0, 0, 0, 0)
        pa.setSpacing(2)
        pa.addLayout(srow)
        pa.addWidget(self.plot_stack, 1)
        right.addWidget(plot_area)
        right.addWidget(self.summary)
        right.setSizes([600, 220])
        self.plot_style_combo.currentIndexChanged.connect(self._on_plot_options_changed)
        self.plot_bin_combo.currentIndexChanged.connect(self._on_plot_options_changed)

        root.addWidget(scroll, 0)
        root.addWidget(right, 1)

        self._setup_times()
        self._sync_plot_style()
        self._plot_preview()
        if ra is not None and dec is not None:
            QTimer.singleShot(50, self._lookup_position)

    # ------------------------------------------------------------- time base

    def _setup_times(self):
        ctx = self.ctx
        jd = np.asarray(ctx["jd_utc"], dtype=np.float64)
        ra, dec, site = ctx.get("ra"), ctx.get("dec"), ctx.get("site")
        self._t_all, self._time_kind = jd, "JD_UTC"
        why = (self.tr("no target coordinates") if ra is None or dec is None
               else self.tr("some frames have no timestamp"))
        info = {}
        if ra is not None and dec is not None and np.all(np.isfinite(jd)):
            try:
                self._t_all = xfit.bjd_tdb(jd, ra, dec, site, info=info)
                self._time_kind = "BJD_TDB"
            except Exception as e:
                why = self.tr("the barycentric correction failed ({0})").format(xfit.first_paragraph(e))
        self._apply_night()
        how = ctx.get("time_how", "")
        if info.get("geocentric_fallback"):
            site_txt = self.tr("geocentric (≤21 ms): the site correction failed "
                               "({0})").format(info['geocentric_fallback'])
        elif site:
            site_txt = self.tr("site {0:.4f}°, {1:.4f}° from {2}").format(site[0], site[1], site[3])
        else:
            site_txt = self.tr("no site in header — geocentric (≤21 ms)")
        if self._time_kind == "BJD_TDB":
            self.time_lbl.setText(self.tr("Times: BJD_TDB at mid-exposure ({0}); {1}.").format(how, site_txt))
            # A planet lookup can supply coordinates after a JD_UTC start.
            self.time_lbl.setStyleSheet("")
        else:
            fix = (self.tr(" Plate-solve the frames or look the planet up to get them.")
                   if ra is None or dec is None else "")
            self.time_lbl.setText(self.tr(
                "Times: JD_UTC mid-exposure — {0}, so no barycentric correction. "
                "AAVSO and timing databases require BJD_TDB.{1}").format(why, fix))
            self.time_lbl.setStyleSheet("color: #ffb347;")

    # ------------------------------------------------------------- nights

    def _session_of_jd(self, jd_utc: float) -> int:
        jd = np.asarray(self.ctx["jd_utc"], dtype=np.float64)
        return int(self._S[int(np.nanargmin(np.abs(jd - float(jd_utc))))])

    def _apply_night(self):
        """Frames of the selected night -> self._mask, self._t."""
        k = self.night_combo.currentData()
        self._mask = (self._S == int(k)) if k is not None else np.ones(self._S.size, bool)
        self._t = self._t_all[self._mask]

    def _select_night(self, k: int):
        """Switch night programmatically (doesn't count as the user's choice)."""
        i = self.night_combo.findData(int(k))
        if i >= 0 and i != self.night_combo.currentIndex():
            user = self._night_from_user
            self.night_combo.setCurrentIndex(i)      # -> _on_night_changed
            self._night_from_user = user

    def _on_night_changed(self, _i: int = 0):
        self._night_from_user = True
        self._apply_night()
        # A fit belongs to the night it was run on.
        if self._out is not None:
            self._out = None
            for b in (self.aavso_btn, self.csv_btn, self.png_btn):
                b.setEnabled(False)
            self.summary.setPlainText(self.tr("Night changed — run the fit for this night."))
        self._plot_preview()

    def _night_with_transit(self, pr: xfit.TransitPriors):
        """Index of the night whose frames cover the predicted transit, or None."""
        best, best_d = None, None
        for s in self._sessions:
            k = int(s["index"])
            t = self._t_all[self._S == k]
            if t.size < 3:
                continue
            pred, _sig, _ep = pr.predicted_tc(float(np.median(t)))
            if not np.isfinite(pred):
                return None
            if t.min() - 0.02 <= pred <= t.max() + 0.02:
                d = abs(pred - 0.5 * (t.min() + t.max()))
                if best_d is None or d < best_d:
                    best, best_d = k, d
        return best

    # ------------------------------------------------------------- lookups

    def _set_busy_lookup(self, busy: bool, msg: str = ""):
        for b in (self.lookup_btn, self.cone_btn, self.ld_btn):
            b.setEnabled(not busy if b is not self.cone_btn
                         else (not busy and self.ctx.get("ra") is not None))
        # A fit started mid-lookup would run on the old priors / limb
        # darkening -- and on an eccentric orbit a wrong LD law shifts Tc
        # (measured: 46 s on HAT-P-32 b's archive e=0.16 geometry).
        if not (self._worker is not None and self._worker.isRunning()):
            self.run_btn.setEnabled(not busy)
            self.run_btn.setText(self.tr("Run Fit") if not busy else self.tr("Run Fit (waiting for lookup…)"))
        if msg:
            self.lookup_status.setText(msg)

    def _call(self, fn, ok_slot, *args, **kw):
        w = _CallWorker(fn, *args, **kw)
        w.result_ready.connect(ok_slot)
        w.failed.connect(self._lookup_failed)
        _start_thread(w)

    def _lookup_failed(self, msg: str):
        self._set_busy_lookup(False, self.tr("Lookup failed: {0}").format(msg))

    def _lookup_name(self):
        name = self.planet_edit.text().strip()
        if not name:
            return
        self._set_busy_lookup(True, self.tr("Querying NASA Exoplanet Archive for “{0}”…").format(name))
        self._call(xfit.nea_lookup_name, self._rows_arrived, name)

    def _lookup_position(self):
        ra, dec = self.ctx.get("ra"), self.ctx.get("dec")
        if ra is None or dec is None:
            return
        self._set_busy_lookup(True, self.tr("Looking for known planets at this position…"))
        self._call(xfit.nea_lookup_position, self._rows_arrived, ra, dec, 20.0)

    def _rows_arrived(self, rows):
        self._set_busy_lookup(False)
        rows = list(rows or [])
        if not rows:
            sc = self.ctx.get("screen") or {}
            self.lookup_status.setText(
                self.tr("No known planet found. Enter the ephemeris by hand, or start from the detected dip.")
                if sc else self.tr("No known planet found. Enter the ephemeris by hand."))
            return
        row = rows[0]
        if len(rows) > 1:
            names = [str(r.get("pl_name")) for r in rows]
            pick, ok = QInputDialog.getItem(self, self.tr("Planet"), self.tr("Several planets match — pick one:"),
                                            names, 0, False)
            if not ok:
                return
            row = rows[names.index(pick)]
        p = xfit.priors_from_nea_row(row)
        self._apply_priors(p)
        if self.ctx.get("ra") is None and np.isfinite(xfit._num(row, "ra")):
            # No WCS: the archive position is good enough for the barycentric
            # correction (an arcminute error moves BJD by < 0.1 s).
            self.ctx["ra"], self.ctx["dec"] = xfit._num(row, "ra"), xfit._num(row, "dec")
            self._setup_times()
        moved = ""
        if len(self._sessions) > 1 and not self._night_from_user:
            k = self._night_with_transit(p)
            if k is not None and k != self.night_combo.currentData():
                self._select_night(k)
                moved = self.tr("\nSwitched to {0} — it covers the transit.").format(self.night_combo.currentText())
        pred, sig, epoch = p.predicted_tc(float(np.nanmedian(self._t)))
        msg = f"{p.name}: P = {p.per:.6f} d, Rp/R* = {p.rp:.4f}, a/R* = {p.ars:.3f}, i = {p.inc:.2f}°"
        if np.isfinite(pred):
            msg += self.tr("\nPredicted Tc this night: {0:.5f} ± {1:.1f} min "
                           "(epoch {2}).").format(pred, sig * 1440, epoch)
            if not (self._t.min() - 0.1 <= pred <= self._t.max() + 0.1):
                msg += self.tr("  ⚠ outside this night's observations!")
        msg += moved
        self.lookup_status.setText(msg)
        if np.isfinite(p.teff):
            self._lookup_ld()

    def _apply_priors(self, p: xfit.TransitPriors):
        self.planet_edit.setText(p.name)
        self._host = p.host
        for spin, v in ((self.per, p.per), (self.per_e, p.per_err), (self.tc, p.tc_ref),
                        (self.tc_e, p.tc_ref_err), (self.rp, p.rp), (self.rp_e, p.rp_err),
                        (self.ars, p.ars), (self.ars_e, p.ars_err), (self.inc, p.inc),
                        (self.inc_e, p.inc_err), (self.ecc, p.ecc), (self.omega, p.omega)):
            if v is not None and np.isfinite(v):
                spin.setValue(float(v))
        if np.isfinite(p.teff):
            self.teff.setValue(float(p.teff))
        if np.isfinite(p.logg):
            self.logg.setValue(float(p.logg))
        if np.isfinite(p.feh):
            self.feh.setValue(float(p.feh))
        self.free_ars.setChecked(False)
        self._prior_source = p.source

    def _lookup_ld(self):
        code = self.filter_combo.currentData()
        band = xfit.FILTERS.get(code, (None, None, None, None))[3]
        teff = self.teff.value()
        if not band or teff <= 0:
            self.ld_src.setText(self.tr("Need a stellar Teff and a standard filter for a Claret lookup."))
            return
        self._set_busy_lookup(True)
        self.ld_src.setText(self.tr("Querying Claret 2011 ({0}) on VizieR…").format(band))
        w = _CallWorker(xfit.claret_quadratic_ld, teff, self.logg.value(), self.feh.value(), band)
        w.result_ready.connect(self._ld_arrived)
        w.failed.connect(lambda m: (self._set_busy_lookup(False),
                                    self.ld_src.setText(self.tr("LD lookup failed ({0}); "
                                                                "using the values shown.").format(m))))
        _start_thread(w)

    def _ld_arrived(self, res):
        self._set_busy_lookup(False)
        u1, u2, desc = res
        self.u1.setValue(u1)
        self.u2.setValue(u2)
        code = self.filter_combo.currentData()
        approx = (self.tr(" — approximate: no Claret passband for this filter")
                  if code in xfit.APPROXIMATE_LD_BANDS else "")
        self.ld_src.setText(desc + approx)

    def _fill_from_dip(self):
        sc = self.ctx.get("screen") or {}
        if not sc:
            return
        jd0 = float(self.ctx["jd_utc"][0])
        # dip_t is in days since the first frame of the whole set (UTC); shift
        # onto the fit time base and fit the night the dip is in.
        tc_utc = jd0 + float(sc["dip_t"])
        tc = float(np.interp(tc_utc, self.ctx["jd_utc"], self._t_all))
        self._select_night(self._session_of_jd(tc_utc))
        depth = max(float(sc.get("dip_depth_ppt", 10.0)) / 1e3, 1e-4)
        dur_d = max(float(sc.get("dip_dur_h", 2.0)) / 24.0, 0.02)
        rp = math.sqrt(depth)
        self.per.setValue(3.0)
        self.per_e.setValue(0.0)
        self.tc.setValue(tc)
        self.tc_e.setValue(dur_d)
        self.rp.setValue(rp)
        self.rp_e.setValue(0.0)
        self.ars.setValue(max(1.5, xfit.ars_from_duration(3.0, dur_d, rp, 0.0)))
        self.ars_e.setValue(0.0)
        self.inc.setValue(90.0)
        self.inc_e.setValue(0.0)
        self.ecc.setValue(0.0)
        self.free_ars.setChecked(True)
        self.free_inc.setChecked(True)
        self.lookup_status.setText(self.tr(
            "Started from the detected dip. Period is nominal (3 d) with a/R* free: "
            "only Tc, depth and duration are meaningful."))

    # -------------------------------------------------------------- fit

    def _priors(self) -> xfit.TransitPriors:
        return xfit.TransitPriors(
            name=self.planet_edit.text().strip(), host=getattr(self, "_host", ""),
            per=self.per.value(), per_err=self.per_e.value(),
            tc_ref=self.tc.value() if self.tc.value() > 0 else float("nan"),
            tc_ref_err=self.tc_e.value(), rp=self.rp.value(), rp_err=self.rp_e.value(),
            ars=self.ars.value(), ars_err=self.ars_e.value(), inc=self.inc.value(),
            inc_err=self.inc_e.value(), ecc=self.ecc.value(), omega=self.omega.value(),
            u1=self.u1.value(), u2=self.u2.value(), teff=self.teff.value() or float("nan"),
            logg=self.logg.value(), feh=self.feh.value(),
            source=getattr(self, "_prior_source", "manual"), ld_source=self.ld_src.text())

    def _options(self) -> xfit.FitOptions:
        return xfit.FitOptions(
            systematics=self.sys_combo.currentData(), free_inc=self.free_inc.isChecked(),
            free_ars=self.free_ars.isChecked(), free_ld=self.free_ld.isChecked(),
            clip_sigma=self.clip.value(), mcmc_steps=self.steps.value(),
            exptime_s=float(self.ctx.get("exptime_s") or 0.0),
            use_beta=self.use_beta.isChecked())

    def _reset_to_factory(self):
        if self._worker is not None and self._worker.isRunning():
            return                      # a fit is running
        restore_controls(self._factory)

    def _run_or_cancel(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self.prog_lbl.setText(self.tr("Cancelling…"))
            return
        pr = self._priors()
        if not np.isfinite(pr.tc_ref):
            pr.tc_ref = float("nan")
            sc = self.ctx.get("screen") or {}
            if not (sc and np.isfinite(sc.get("dip_t", np.nan))):
                QMessageBox.information(self, self.tr("Transit Fit"),
                                        self.tr("Enter a reference Tc (or look up the planet) first."))
                return
        ctx = self.ctx
        m = self._mask
        job = {
            "flux": ctx["flux"][:, :, m], "err": ctx["err"][:, :, m],
            "valid": ctx["valid"][:, :, m],
            "radii": ctx["radii"], "target": ctx["target"], "t": self._t,
            "airmass": np.asarray(ctx["airmass"])[m], "priors": pr, "options": self._options(),
            "ap_mode": self.ap_combo.currentData(), "comp_mode": self.comp_combo.currentData(),
            "max_comps": ctx.get("max_comps", 10), "exclude": ctx.get("exclude", ()),
            "eligible": ctx.get("eligible"),
            "side": (np.asarray(ctx["side_idx"])[m] if ctx.get("side_idx") is not None
                     and len(ctx["side_idx"]) == m.size else None),
        }
        if not np.isfinite(pr.tc_ref):
            # unknown ephemeris: centre the Tc window on the observed night
            job["options"].tc_window = (float(self._t.min()), float(self._t.max()))
        self._worker = _FitWorker(job)
        self._worker.progress_msg.connect(self._on_progress)
        self._worker.result_ready.connect(self._on_result)
        self._worker.failed.connect(self._on_failed)
        self.run_btn.setText(self.tr("Cancel"))
        self.prog.setVisible(True)
        self.prog.setValue(0)
        _start_thread(self._worker)

    def _on_progress(self, p: int, msg: str):
        self.prog.setValue(int(p))
        self.prog_lbl.setText(msg)

    def _on_failed(self, msg: str):
        self.run_btn.setText(self.tr("Run Fit"))
        self.prog.setVisible(False)
        self.prog_lbl.setText(self.tr("Cancelled.") if msg == "Cancelled." else msg)
        if msg != "Cancelled.":
            QMessageBox.warning(self, self.tr("Transit Fit"), msg)

    def _on_result(self, out: dict):
        self.run_btn.setText(self.tr("Run Fit"))
        self.prog.setVisible(False)
        res: xfit.FitResult = out["result"]
        if not res.ok:
            self.prog_lbl.setText(res.message)
            QMessageBox.warning(self, self.tr("Transit Fit"), res.message)
            return
        self._out = out
        self.prog_lbl.setText(self.tr("Fit complete."))
        for b in (self.aavso_btn, self.csv_btn, self.png_btn):
            b.setEnabled(True)
        self._plot_result()
        self.summary.setPlainText(self._summary_text())
        par = self.parent()
        if par is not None and hasattr(par, "_remember_transit_fit"):
            try:
                par._remember_transit_fit(self.ctx["target"], out)
            except Exception:
                pass

    # ------------------------------------------------------------- plotting

    def _plot_style(self) -> str:
        return str(self.plot_style_combo.currentData())

    def _bin_minutes(self) -> int:
        try:
            return int(self.plot_bin_combo.currentData())
        except (TypeError, ValueError):
            return 10

    def _sync_plot_style(self):
        if self._plot_style() == "scientific":
            if self.sci_plot is None:
                try:
                    from setiastro.saspro.exoplanet_sciplot import SciPlotWidget
                    self.sci_plot = SciPlotWidget(self)
                    self.plot_stack.addWidget(self.sci_plot)
                except Exception as e:
                    print(f"[Transit fit] scientific plot unavailable: {e}")
                    self.plot_style_combo.blockSignals(True)
                    self.plot_style_combo.setCurrentIndex(0)
                    self.plot_style_combo.blockSignals(False)
                    self.plot_stack.setCurrentWidget(self._pg_pair)
                    return
            self.plot_stack.setCurrentWidget(self.sci_plot)
        else:
            self.plot_stack.setCurrentWidget(self._pg_pair)

    def _on_plot_options_changed(self, _i: int = 0):
        ps = QSettings()
        ps.setValue("Exoplanet/plot_style", self._plot_style())
        ps.setValue("Exoplanet/plot_bin_min", self._bin_minutes())
        self._sync_plot_style()
        if self._out:
            self._plot_result()
        else:
            self._plot_preview()

    def _model_curve(self, res):
        tf = np.linspace(res.t.min(), res.t.max(), 1500)
        g = lambda n, d: res.best.get(n, res.fixed.get(n, d))
        pr = res.priors
        mf = xfit.transit_model(tf, res.best["tc"], pr.per, g("rp", pr.rp), g("ars", pr.ars),
                                g("inc", pr.inc), g("u1", pr.u1), g("u2", pr.u2),
                                pr.ecc, pr.omega)
        return tf, mf

    def _night_title(self) -> str:
        k = self.night_combo.currentData()
        if k is not None:
            for s in self._sessions:
                if int(s["index"]) == int(k):
                    return s.get("date") or ""
        if len(self._sessions) == 1:
            return self._sessions[0].get("date") or ""
        return ""

    def _draw_sci(self):
        """The Scientific figure for the current state (preview or fit)."""
        if self._plot_style() != "scientific" or self.sci_plot is None:
            return
        from setiastro.saspro import exoplanet_sciplot as sciplot
        fig = self.sci_plot.figure
        bmin = self._bin_minutes()
        bin_label = self.tr("{0}-min bins").format(bmin) if bmin else ""
        sub = r"$_\mathrm{TDB}$" if self._time_kind == "BJD_TDB" else r"$_\mathrm{UTC}$"
        tk = "BJD" if self._time_kind == "BJD_TDB" else "JD"
        if self._out:
            res: xfit.FitResult = self._out["result"]
            pr = res.priors
            tc = res.best["tc"]
            sysm = res.systematics
            x = (res.t - tc) * 24.0
            det = res.y / sysm
            derr = res.yerr / sysm
            tf, mf = self._model_curve(res)
            info = []
            if "rp" in res.median:
                info.append(rf"$R_p/R_\star$ = {res.median['rp']:.4f} ± "
                            f"{0.5 * (res.err_lo['rp'] + res.err_hi['rp']):.4f}")
            tce = 0.5 * (res.err_lo["tc"] + res.err_hi["tc"])
            info.append(rf"$T_c$ = {res.median['tc']:.5f} ± {tce:.5f} {tk}{sub}")
            d = res.derived
            if "oc_min" in d:
                m, lo, hi = d["oc_min"]
                info.append(f"O − C = {m:+.2f} ± {0.5 * (lo + hi):.2f} min")
            st = res.stats
            info.append(self.tr("σ = {0:.2f} ppt per {1:.1f}-min point").format(st['rms_ppt'], st['cadence_min']))
            pred = d.get("tc_pred")
            pred_x = (pred - tc) * 24.0 if pred is not None and np.isfinite(pred) else None
            parts = [pr.name or self.ctx.get("label", self.tr("Transit")), self._night_title(),
                     str(self.filter_combo.currentData() or "")]
            sciplot.draw_transit_fit(
                fig, x=x, y=det, err=derr, used=res.used, model_x=(tf - tc) * 24.0,
                model_y=mf, resid=(res.y - res.model) / sysm * 1e3, resid_err=derr * 1e3,
                bin_w=bmin / 60.0, bin_label=bin_label, xlabel=self.tr("Time from mid-transit (h)"),
                title="  ·  ".join(p for p in parts if p), info=info, pred_x=pred_x)
        else:
            rel = self.ctx.get("rel")
            if rel is None or self._t is None or self._t.size == 0:
                sciplot.draw_message(fig, self.tr("No light curve"))
                self.sci_plot.refresh()
                return
            rel = np.asarray(rel, dtype=np.float64)[self._mask]
            ok = np.asarray(self.ctx.get("rel_ok"), bool)[self._mask] & np.isfinite(rel)
            err = self.ctx.get("rel_err")
            err = np.asarray(err, dtype=np.float64)[self._mask] if err is not None else None
            off = float(np.floor(np.nanmin(self._t)))
            S = self._S[self._mask]
            series = [{"label": self.ctx.get("label", ""), "x": (self._t - off)[ok], "y": rel[ok],
                       "err": (err[ok] if err is not None else None), "group": S[ok]}]
            parts = [self.ctx.get("label", ""), self._night_title(), self.tr("screening ensemble")]
            sciplot.draw_light_curves(fig, series, bin_w=bmin / 1440.0, bin_label=bin_label,
                                      xlabel=self.tr("{0}{1} − {2:.0f}  (mid-exposure)").format(tk, sub, off),
                                      title="  ·  ".join(p for p in parts if p))
        self.sci_plot.refresh()

    def _plot_preview(self):
        rel = self.ctx.get("rel")
        ok = self.ctx.get("rel_ok")
        if rel is None:
            return
        rel = np.asarray(rel)[self._mask]
        ok = np.asarray(ok, bool)[self._mask]
        t = self._t
        if t.size == 0:
            return
        h = (t - t[0]) * 24.0
        m = ok & np.isfinite(rel)
        self.p_lc.clear()
        self.p_res.clear()
        self.p_lc.plot(h[m], rel[m], pen=None, symbol="o", symbolSize=4,
                       symbolBrush=pg.mkBrush(150, 190, 255, 140), symbolPen=None)
        bw = self._bin_minutes() / 60.0
        err = self.ctx.get("rel_err")
        err = np.asarray(err, dtype=np.float64)[self._mask] if err is not None else None
        tb, yb, eb = (xphot.bin_curve(h[m], rel[m], bw, e=(err[m] if err is not None else None))
                      if bw > 0 else (np.array([]),) * 3)
        if tb.size:
            self.p_lc.addItem(pg.ErrorBarItem(x=tb, y=yb, height=2 * eb,
                                              pen=pg.mkPen((255, 210, 80), width=1)))
            self.p_lc.plot(tb, yb, pen=None, symbol="o", symbolSize=7,
                           symbolBrush=pg.mkBrush(255, 210, 80), symbolPen=None)
        self.p_lc.setLabel("bottom", self.tr("Hours since first frame ({0})").format(self._time_kind))
        self.p_lc.setTitle(self.tr("Light curve (screening ensemble) — run the fit to optimise"))
        self._draw_sci()

    def _plot_result(self):
        out = self._out
        res: xfit.FitResult = out["result"]
        tc = res.best["tc"]
        h = (res.t - tc) * 24.0
        det = res.y / res.systematics
        used = res.used
        self.p_lc.clear()
        self.p_res.clear()
        self.p_lc.addLegend(offset=(10, 10))
        self.p_lc.plot(h[used], det[used], pen=None, symbol="o", symbolSize=4,
                       symbolBrush=pg.mkBrush(150, 190, 255, 130), symbolPen=None,
                       name=self.tr("detrended"))
        if np.any(~used):
            self.p_lc.plot(h[~used], det[~used], pen=None, symbol="x", symbolSize=8,
                           symbolPen=pg.mkPen((255, 80, 80)), name=self.tr("clipped"))
        bmin = self._bin_minutes()
        bw = bmin / 60.0
        derr = res.yerr / res.systematics
        tb, yb, eb = (xphot.bin_curve(h[used], det[used], bw, e=derr[used])
                      if bw > 0 else (np.array([]),) * 3)
        if tb.size:
            self.p_lc.addItem(pg.ErrorBarItem(x=tb, y=yb, height=2 * eb,
                                              pen=pg.mkPen((255, 210, 80), width=1)))
            self.p_lc.plot(tb, yb, pen=None, symbol="o", symbolSize=7,
                           symbolBrush=pg.mkBrush(255, 210, 80), symbolPen=None,
                           name=self.tr("{0}-min bins").format(bmin))
        pr = res.priors
        tf, mf = self._model_curve(res)
        self.p_lc.plot((tf - tc) * 24.0, mf, pen=pg.mkPen((255, 90, 90), width=2),
                       name=self.tr("transit model"))
        pred = res.derived.get("tc_pred")
        if pred is not None and np.isfinite(pred):
            self.p_lc.addLine(x=(pred - tc) * 24.0,
                              pen=pg.mkPen((180, 180, 180), style=Qt.PenStyle.DashLine))
        self.p_lc.setLabel("bottom", self.tr("Hours from fitted mid-transit"))
        self.p_lc.setTitle(self.tr("{0} — detrended ({1})").format(pr.name or self.tr("Transit"), self._time_kind))
        r = (res.y - res.model) * 1e3
        self.p_res.plot(h[used], r[used], pen=None, symbol="o", symbolSize=4,
                        symbolBrush=pg.mkBrush(150, 190, 255, 130), symbolPen=None)
        tb, rb, eb = (xphot.bin_curve(h[used], r[used], bw, e=res.yerr[used] * 1e3)
                      if bw > 0 else (np.array([]),) * 3)
        if tb.size:
            self.p_res.plot(tb, rb, pen=None, symbol="o", symbolSize=6,
                            symbolBrush=pg.mkBrush(255, 210, 80), symbolPen=None)
        self.p_res.addLine(y=0.0, pen=pg.mkPen((200, 200, 200), style=Qt.PenStyle.DashLine))
        self.p_res.setLabel("bottom", self.tr("Hours from fitted mid-transit"))
        self._draw_sci()

    # -------------------------------------------------------------- summary

    def _summary_text(self) -> str:
        out = self._out
        res: xfit.FitResult = out["result"]
        best = out["best"]
        pr = res.priors
        d = res.derived
        st = res.stats
        L = []

        def pm(name, scale=1.0, fmt="{:.5f}"):
            if name in res.median:
                m = res.median[name] * scale
                lo, hi = res.err_lo[name] * scale, res.err_hi[name] * scale
                return f"{fmt.format(m)} -{fmt.format(lo)} +{fmt.format(hi)}"
            v = res.fixed.get(name)
            return (fmt.format(v * scale) + self.tr(" (fixed)")) if v is not None else "—"

        L.append(self.tr("{0} — SASpro transit fit ({1}, mid-exposure)").format(pr.name or self.tr("Transit"), self._time_kind))
        L.append("")
        tce = 0.5 * (res.err_lo["tc"] + res.err_hi["tc"])
        L.append(f"Tc        = {res.median['tc']:.6f} ± {tce:.6f}  (± {tce * 1440:.2f} min)")
        if "oc_min" in d:
            m, lo, hi = d["oc_min"]
            L.append(self.tr("O–C       = {0:+.2f} ± {1:.2f} min   (epoch {2}, "
                             "predicted {3:.5f} ± {4:.2f} min)").format(
                m, 0.5 * (lo + hi), d['epoch'], d['tc_pred'], d['oc_pred_sigma_min']))
            if self._time_kind != "BJD_TDB":
                # BJD_TDB - JD_UTC runs from -7.2 to +9.5 min (light time
                # +-8.3 min, plus TDB - UTC = 69.2 s).
                L.append(self.tr("            ⚠ JD_UTC times against a BJD_TDB ephemeris: "
                                 "this O–C can be off by up to 9.5 min"))
        L.append(f"Rp/R*     = {pm('rp')}")
        dm, dl, dh = d["depth_rp2"]
        L.append(self.tr("(Rp/R*)²  = {0:.2f} ± {1:.2f} ppt    "
                         "model depth (with LD) = {2:.2f} ppt").format(
            dm * 1e3, 0.5 * (dl + dh) * 1e3, d['depth_model'] * 1e3))
        L.append(f"inc       = {pm('inc', fmt='{:.2f}')} °")
        L.append(f"a/R*      = {pm('ars', fmt='{:.3f}')}")
        bm, bl, bh = d["b"]
        tm, tl, th = d["t14_h"]
        L.append(f"b         = {bm:.3f} ± {0.5 * (bl + bh):.3f}      T14 = {tm:.3f} ± {0.5 * (tl + th):.3f} h")
        if "u1" in res.median:
            L.append(f"u1, u2    = {pm('u1', fmt='{:.3f}')}, {pm('u2', fmt='{:.3f}')}")
        else:
            L.append(self.tr("u1, u2    = {0:.3f}, {1:.3f} (fixed)").format(pr.u1, pr.u2))
        a1, a1l, a1h = d["a1"]
        a2, a2l, a2h = d["a2"]
        if "fs" in res.median:
            L.append(self.tr("Flip step = {0:+.2f} ± "
                             "{1:.2f} ppt "
                             "(far pier side vs near; meridian flip inside the night)").format(
                res.median['fs'] * 1e3, 0.5 * (res.err_lo['fs'] + res.err_hi['fs']) * 1e3))
        if "am" in res.median:
            L.append(self.tr("Airmass   a1 = {0:.5f} ± {1:.5f}   "
                             "a2 = {2:+.5f} ± {3:.5f}   (EXOTIC convention)").format(
                a1, 0.5 * (a1l + a1h), a2, 0.5 * (a2l + a2h)))
        # tau is in days: c1 per day, c2 per day^2.
        for name, scale, unit, lab in (("c1", 1e3 / 24.0, "ppt/h", self.tr("Slope")),
                                       ("c2", 1e3 / 576.0, "ppt/h²", self.tr("Curvature"))):
            if name in res.median:
                L.append(f"{lab:<9} = {res.median[name] * scale:+.3f} ± "
                         f"{0.5 * (res.err_lo[name] + res.err_hi[name]) * scale:.3f} {unit}")
        L.append("")
        L.append(self.tr("Residual RMS = {0:.2f} ppt/point ({1:.2f} min cadence), "
                         "{2:.2f} ppt per 10 min").format(st['rms_ppt'], st['cadence_min'], st['rms10_ppt']))
        L.append(self.tr("χ²_red (photometric errors) = {0:.2f} → errors × {1:.2f};"
                         "  red-noise β = {2:.2f}").format(st['chi2_red_raw'], st['err_scale'], st['beta'])
                 + ("" if self.use_beta.isChecked() else self.tr(" (not applied)")))
        fnp = float(best.get("frame_noise_ppt", 0.0) or 0.0)
        L.append(self.tr("Point errors: CCD equation ⊕ per-frame noise measured from the comparison "
                         "stars (median {0:.2f} ppt: scintillation, cloud, seeing)").format(fnp)
                 if fnp > 0 else self.tr("Point errors: CCD equation (comparison stars showed no excess)"))
        L.append(self.tr("Points: {0} used, {1} clipped"
                         "   MCMC: {2} walkers × {3} steps, "
                         "acceptance {4:.2f}").format(st['n_used'], st['n_clipped'], st['walkers'],
                                                      st['steps'], st['acceptance']))
        _sb = int(self.ctx.get("soft_bin", 2) or 2)
        L.append(self.tr("Aperture r = {0:.2f} px ({1}×{1}-binned grid); comparison: {2} "
                         "→ stars {3}{4}").format(best['radius'], _sb, best['label'], best['comps'][:12],
                                                  ' …' if len(best['comps']) > 12 else ''))
        L.append(self.tr("Selection: {0} aperture/comparison combinations scored, "
                         "best std/median = {1:.5f}").format(len(out['scores']), best['score']))
        sys_lab = xfit.SYSTEMATICS_LABELS.get(res.options.systematics)
        sa = st.get("sys_auto")
        if sa:
            L.append(self.tr("Systematics: {0} — chosen automatically (lowest BIC; "
                             "within ΔBIC 2 the model with fewer terms wins)").format(sys_lab))
            parts = []
            for r in sa["rows"]:
                nm = xfit.SYSTEMATICS_SHORT.get(r["key"], r["key"])
                if np.isfinite(r["dbic"]):
                    parts.append(f"{nm} {r['dbic']:+.1f}" + (" ✓" if r["chosen"] else ""))
                else:
                    parts.append(f"{nm} ({r['note']})")
            L.append("  ΔBIC: " + " · ".join(parts))
        else:
            L.append(self.tr("Systematics: {0}").format(sys_lab))
        L.append(self.tr("Limb darkening: u1={0:.4f}, u2={1:.4f} — {2}").format(pr.u1, pr.u2, pr.ld_source))
        L.append(self.tr("Priors: {0}").format(getattr(self, '_prior_source', 'manual')))
        for n in res.notes:
            L.append(self.tr("Note: {0}").format(n))
        return "\n".join(L)

    # -------------------------------------------------------------- exports

    def _comp_star_xc(self):
        out = self._out
        comps = out["best"]["comps"]
        if not comps:
            return None
        c = int(comps[0])
        xy = self.ctx["star_xy"][c]
        radec = None
        fn = self.ctx.get("wcs_to_radec")
        if callable(fn):
            try:
                radec = fn(float(xy[0]), float(xy[1]))
            except Exception:
                radec = None
        return {"ra": f"{radec[0]:.6f}" if radec else "", "dec": f"{radec[1]:+.6f}" if radec else "",
                "x": f"{xy[0]:.2f}", "y": f"{xy[1]:.2f}"}

    def _export_aavso(self):
        if not self._out:
            return
        res: xfit.FitResult = self._out["result"]
        if self._time_kind != "BJD_TDB":
            if QMessageBox.question(
                    self, self.tr("AAVSO Report"),
                    self.tr("These times are JD_UTC, not BJD_TDB. The AAVSO Exoplanet Database "
                            "expects BJD_TDB. Write the report anyway?")) != QMessageBox.StandardButton.Yes:
                return
        settings = QSettings()
        code, ok = QInputDialog.getText(self, "AAVSO", self.tr("Observer code:"),
                                        QLineEdit.EchoMode.Normal,
                                        settings.value("AAVSO/observer_code", "", type=str))
        if not ok:
            return
        code = code.strip().upper()
        settings.setValue("AAVSO/observer_code", code)
        sec, ok = QInputDialog.getText(self, "AAVSO", self.tr("Secondary observer codes (optional):"),
                                       QLineEdit.EchoMode.Normal,
                                       settings.value("AAVSO/secondary_codes", "", type=str))
        if not ok:
            return
        settings.setValue("AAVSO/secondary_codes", sec.strip())
        cams = ["CCD", "CMOS", "DSLR"]
        cam, ok = QInputDialog.getItem(self, "AAVSO", self.tr("Camera type:"), cams,
                                       cams.index(self.ctx.get("camera_hint", "CCD"))
                                       if self.ctx.get("camera_hint", "CCD") in cams else 0, False)
        if not ok:
            return
        best = self._out["best"]
        st = res.stats
        notes_default = (
            f"SASpro differential aperture photometry; {best['label']} comparison "
            f"(inverse-variance weighted); aperture r={best['radius']:.2f} px on "
            f"{int(self.ctx.get('soft_bin', 2) or 2)}x{int(self.ctx.get('soft_bin', 2) or 2)} software-binned "
            f"frames; errors x{st['err_scale']:.2f} (chi2) x{st['beta_used']:.2f} (beta); "
            f"quadratic LD u1={res.priors.u1:.3f} u2={res.priors.u2:.3f}"
            + ("" if res.options.systematics == "airmass" else
               f"; systematics model: {xfit.SYSTEMATICS_LABELS.get(res.options.systematics)} "
               "(DETREND_2 is that model)"))
        notes, ok = QInputDialog.getMultiLineText(self, "AAVSO", self.tr("Notes:"), notes_default)
        if not ok:
            return
        planet = res.priors.name or self.planet_edit.text().strip() or "unknown"
        star = getattr(self, "_host", "") or (planet[:-2] if planet[-2:-1] == " " else planet)
        text = xfit.aavso_exoplanet_report(
            res, obscode=code, secondary=sec.strip(), star_name=star, planet_name=planet,
            filter_code=self.filter_combo.currentData(),
            exposure_s=float(self.ctx.get("exptime_s") or -1),
            binning=self.ctx.get("binning", "1x1"), camera=cam,
            comp_star=self._comp_star_xc(), notes=notes.replace("\n", " "),
            date_type=self._time_kind)
        date = _dt.date.today().strftime("%Y-%m-%d")
        default = f"AAVSO_{planet.replace(' ', '')}_{date}.txt"
        path, _ = QFileDialog.getSaveFileName(self, self.tr("Save AAVSO Exoplanet Report"), default,
                                              "Text (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        QMessageBox.information(self, self.tr("AAVSO Report"), self.tr("Wrote\n{0}").format(path))

    def _export_csv(self):
        if not self._out:
            return
        res: xfit.FitResult = self._out["result"]
        path, _ = QFileDialog.getSaveFileName(self, self.tr("Save light curve"), "transit_fit.csv",
                                              "CSV (*.csv)")
        if not path:
            return
        import pandas as pd
        df = pd.DataFrame({
            self._time_kind: res.t, "rel_flux": res.y, "err": res.yerr,
            "airmass": res.airmass, "model": res.model, "transit": res.transit,
            "systematics": res.systematics, "detrended": res.y / res.systematics,
            "residual": res.y - res.model, "used": res.used.astype(int),
        })
        df.to_csv(path, index=False, float_format="%.8f")
        QMessageBox.information(self, "CSV", self.tr("Wrote\n{0}").format(path))

    def _export_png(self):
        if not self._out:
            return
        if self._plot_style() == "scientific" and self.sci_plot is not None:
            path, filt = QFileDialog.getSaveFileName(
                self, self.tr("Save plot"), "transit_fit.png",
                "PNG image, 300 dpi (*.png);;PDF (*.pdf);;SVG (*.svg)")
            if not path:
                return
            if not path.lower().endswith((".png", ".pdf", ".svg")):
                path += ".pdf" if "PDF" in filt else ".svg" if "SVG" in filt else ".png"
            try:
                self.sci_plot.save(path)
            except Exception as e:
                QMessageBox.warning(self, self.tr("Plot"), self.tr("Could not save the plot:\n{0}").format(e))
                return
            QMessageBox.information(self, self.tr("Plot"), self.tr("Wrote\n{0}").format(path))
            return
        path, _ = QFileDialog.getSaveFileName(self, self.tr("Save plot"), "transit_fit.png", "PNG (*.png)")
        if not path:
            return
        self.p_lc.grab().save(path)
        base = path[:-4] if path.lower().endswith(".png") else path
        self.p_res.grab().save(base + "_residuals.png")
        QMessageBox.information(self, self.tr("Plot"), self.tr("Wrote\n{0}\n{1}_residuals.png").format(path, base))

    # -------------------------------------------------------------- lifetime

    def closeEvent(self, ev):
        self._cancel_worker()
        super().closeEvent(ev)

    def reject(self):
        self._cancel_worker()
        super().reject()

    def _cancel_worker(self):
        w = self._worker
        if w is not None:
            try:
                if w.isRunning():
                    w.cancel()
                    for sig, slot in ((w.result_ready, self._on_result),
                                      (w.failed, self._on_failed),
                                      (w.progress_msg, self._on_progress)):
                        try:
                            sig.disconnect(slot)
                        except Exception:
                            pass
            except RuntimeError:
                pass
