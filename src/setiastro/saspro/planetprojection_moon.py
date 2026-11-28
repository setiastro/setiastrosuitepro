# src/setiastro/saspro/planetprojection_moon.py
# Copyright (c) Franklin Marek / SetiAstro (www.setiastro.com). GPLv3.
"""
Moon (true relief) for the 3D Projection tool.

The image is registered to the lunar globe by lunar_solver.solve_moon (capture
time + site -> ephemeris; solved blind when the date is unknown) and given real
heights from one of two sources:

  * LOLA       -- NASA's LRO laser-altimeter terrain, downloaded on demand
  * this image -- shadows + shading near the terminator (experimental; scored
                  against LOLA so the user sees how well it did)

The result is an interactive 3D explorer (lunar_relief_viewer: one offline HTML
file, WebGL) with the image laid on the terrain and optionally the rest of the
globe; the mesh can also be saved as PLY + OBJ for Blender / MeshLab.
"""
from __future__ import annotations

import datetime as _dt
import math
import os
import tempfile
import webbrowser
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QDateTime, QEvent, QObject, QSettings, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDateTimeEdit, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QMessageBox, QProgressDialog, QPushButton, QSpinBox,
    QWidget)

_SETTINGS = "planet_projection/moon_"
_LIVE_WORKERS: set = set()      # keep running workers alive if the dialog goes away
_LOLA16_KM = 2 * math.pi * 1737.4 / 360.0 / 16.0     # ~1.9 km per LDEM_16 cell


def _tr(text: str) -> str:
    from PyQt6.QtCore import QCoreApplication
    return QCoreApplication.translate("PlanetProjectionDialog", text)


# ---------------------------------------------------------------------------
# Header helpers (fallbacks when lunar_solver.capture_time is unavailable)
# ---------------------------------------------------------------------------

def _header_of(doc):
    meta = getattr(doc, "metadata", None) or {}
    for k in ("original_header", "fits_header", "header"):
        h = meta.get(k)
        if h is not None:
            return h
    return None


def _hget(hdr, key):
    try:
        return hdr.get(key) if hdr is not None else None
    except Exception:
        return None


def _parse_angle(v):
    """Decimal or sexagesimal ('30 24 00', '-90:09:00') degrees."""
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        pass
    try:
        s = str(v).replace(":", " ").split()
        sign = -1.0 if s[0].strip().startswith("-") else 1.0
        parts = [abs(float(x)) for x in s[:3]] + [0.0, 0.0]
        return sign * (parts[0] + parts[1] / 60.0 + parts[2] / 3600.0)
    except Exception:
        return None


def stored_solution(doc, image=None):
    """The Lunar Solver's placement stored with this image (the live object, or
    the LUN* header cards of a saved file); None when there is none or the image
    has been resized / cropped since it was solved."""
    meta = getattr(doc, "metadata", None) or {}
    try:
        from setiastro.saspro import lunar_solver as LS
    except Exception:
        return None
    img = image if image is not None else getattr(doc, "image", None)
    if img is None:
        return None
    # the live object, else the LUN* cards wherever the document keeps its header
    # (a reopened FITS / TIFF / PNG / JPEG holds it as fits_header, XISF as properties)
    return LS.solution_from_metadata(meta, np.asarray(img).shape[:2])


def _solution_sig(sol):
    """Identifies a stored placement (cache key; tells a re-solve apart)."""
    if sol is None:
        return None
    return (str(sol.utc), round(float(sol.cx), 3), round(float(sol.cy), 3), round(float(sol.radius_px), 3),
            round(float(sol.theta), 7), int(sol.parity), round(float(sol.rms_px), 4))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def set_form_rows_visible(form: QFormLayout, fields, visible: bool):
    """Show/hide whole QFormLayout rows by their field widget or layout
    (QFormLayout.setRowVisible needs Qt 6.4; PyQt6 is unpinned)."""
    for f in fields:
        if hasattr(form, "setRowVisible"):
            try:
                form.setRowVisible(f, bool(visible))
                continue
            except TypeError:
                pass
        for r in range(form.rowCount()):
            it = (form.itemAt(r, QFormLayout.ItemRole.FieldRole)
                  or form.itemAt(r, QFormLayout.ItemRole.SpanningRole))
            if it is None or (it.widget() is not f and it.layout() is not f):
                continue
            lab = form.itemAt(r, QFormLayout.ItemRole.LabelRole)
            if lab is not None and lab.widget() is not None:
                lab.widget().setVisible(visible)
            if it.widget() is not None:
                it.widget().setVisible(visible)
            else:
                lay = it.layout()
                for i in range(lay.count()):
                    w = lay.itemAt(i).widget()
                    if w is not None:
                        w.setVisible(visible)
            break


def build_moon_box(dlg) -> tuple:
    """Create the 'Moon relief' group box; widgets are attached to `dlg`.
    Returns (box, widgets_to_exclude_from_factory_reset)."""
    box = QGroupBox(_tr("Moon relief"))
    form = QFormLayout(box)
    dlg._moon_form = form

    # shown only when the image carries a Lunar Solver placement (MoonRelief.refresh_stored)
    dlg.moon_chk_use_solved = QCheckBox(_tr("Use the Lunar Solver's placement stored with this image"))
    dlg.moon_chk_use_solved.setToolTip(_tr(
        "This image was already placed on the Moon by Tools > Lunar Solver, so no capture time\n"
        "is needed here. Untick to place it again from the capture time below."))
    form.addRow(dlg.moon_chk_use_solved)

    t_row = QWidget()
    t_lay = QHBoxLayout(t_row)
    t_lay.setContentsMargins(0, 0, 0, 0)
    dlg.moon_dt = QDateTimeEdit()
    dlg.moon_dt.setDisplayFormat("yyyy-MM-dd  HH:mm:ss")
    dlg.moon_dt.setCalendarPopup(True)
    dlg.moon_dt.setDateTime(QDateTime.currentDateTimeUtc())
    dlg.moon_dt.setToolTip(_tr("Capture time in UTC (mid-exposure). It sets the Sun angle and the lunar libration."))
    t_lay.addWidget(dlg.moon_dt, 1)
    t_lay.addWidget(QLabel("UTC"))
    dlg.moon_btn_time = QPushButton(_tr("From image"))
    dlg.moon_btn_time.setToolTip(_tr("Read the capture time from the header, EXIF or a SharpCap file name."))
    t_lay.addWidget(dlg.moon_btn_time)
    form.addRow(_tr("Captured:"), t_row)

    dlg.moon_chk_unknown = QCheckBox(_tr("Date unknown - solve it from the image (slower)"))
    form.addRow("", dlg.moon_chk_unknown)
    dlg.moon_lbl_time_src = QLabel("")
    dlg.moon_lbl_time_src.setWordWrap(True)
    form.addRow("", dlg.moon_lbl_time_src)

    s_row = QWidget()
    s_lay = QHBoxLayout(s_row)
    s_lay.setContentsMargins(0, 0, 0, 0)
    dlg.moon_chk_site = QCheckBox(_tr("Site"))
    dlg.moon_chk_site.setToolTip(_tr("Your observing location. Without it the libration is fitted from the image."))
    s_lay.addWidget(dlg.moon_chk_site)
    dlg.moon_lat = QDoubleSpinBox()
    dlg.moon_lat.setRange(-90.0, 90.0)
    dlg.moon_lat.setDecimals(4)
    dlg.moon_lat.setSuffix(" °N")
    dlg.moon_lon = QDoubleSpinBox()
    dlg.moon_lon.setRange(-180.0, 180.0)
    dlg.moon_lon.setDecimals(4)
    dlg.moon_lon.setSuffix(" °E")
    dlg.moon_elev = QSpinBox()
    dlg.moon_elev.setRange(-500, 6000)
    dlg.moon_elev.setSuffix(" m")
    for w in (dlg.moon_lat, dlg.moon_lon, dlg.moon_elev):
        s_lay.addWidget(w)
    form.addRow("", s_row)

    dlg.moon_cmb_source = QComboBox()
    dlg.moon_cmb_source.addItems([
        _tr("LOLA terrain (NASA laser altimeter)"),
        _tr("Derived from this image (shadows + shading, experimental)"),
    ])
    dlg.moon_cmb_source.setToolTip(_tr(
        "Where the heights come from.\n"
        "LOLA: NASA's measured terrain, draped with your image.\n"
        "Derived: measured from your own image - shadow lengths with the Sun angle for the\n"
        "capture time, plus slope from shading near the terminator. Works best on a linear\n"
        "(unstretched) image with the terminator in frame. It is scored against LOLA."))
    form.addRow(_tr("Heights:"), dlg.moon_cmb_source)

    dlg.moon_spin_exag = QDoubleSpinBox()
    dlg.moon_spin_exag.setRange(1.0, 20.0)
    dlg.moon_spin_exag.setSingleStep(0.5)
    dlg.moon_spin_exag.setValue(5.0)
    dlg.moon_spin_exag.setToolTip(_tr("Starting height exaggeration (the viewer has a slider for it).\n"
                                      "Lunar relief is ~0.3% of the radius; x5-x10 makes it easy to see."))
    dlg.moon_spin_vertices = QSpinBox()
    dlg.moon_spin_vertices.setRange(50_000, 2_000_000)
    dlg.moon_spin_vertices.setSingleStep(50_000)
    dlg.moon_spin_vertices.setValue(600_000)
    dlg.moon_spin_vertices.setToolTip(_tr("Terrain mesh detail. The picture itself is always shown at full\n"
                                          "resolution; more vertices give finer terrain but a heavier page."))
    ex_row = QWidget()
    ex_lay = QHBoxLayout(ex_row)
    ex_lay.setContentsMargins(0, 0, 0, 0)
    ex_lay.addWidget(dlg.moon_spin_exag)
    ex_lay.addWidget(QLabel(_tr("Terrain vertices:")))
    ex_lay.addWidget(dlg.moon_spin_vertices)
    form.addRow(_tr("Exaggeration (x):"), ex_row)

    dlg.moon_chk_globe = QCheckBox(_tr("Show the rest of the Moon around the image"))
    dlg.moon_chk_globe.setChecked(True)
    dlg.moon_chk_globe.setToolTip(_tr("Completes the globe from NASA LOLA terrain and Clementine albedo, lit by\n"
                                      "the same Sun, so the image sits on a whole Moon (it can be hidden in the viewer)."))
    form.addRow("", dlg.moon_chk_globe)

    sv_row = QWidget()
    sv_lay = QHBoxLayout(sv_row)
    sv_lay.setContentsMargins(0, 0, 0, 0)
    dlg.moon_btn_save_view = QPushButton(_tr("Save 3D Viewer…"))
    dlg.moon_btn_save_view.setToolTip(_tr("Save the explorer as one HTML file. It opens offline in any browser\n"
                                          "and can be shared."))
    dlg.moon_btn_save_mesh = QPushButton(_tr("Save Mesh…"))
    dlg.moon_btn_save_mesh.setToolTip(_tr("Save the textured terrain mesh (PLY + OBJ) for Blender / MeshLab."))
    for b in (dlg.moon_btn_save_view, dlg.moon_btn_save_mesh):
        b.setEnabled(False)
        sv_lay.addWidget(b)
    sv_lay.addStretch(1)
    form.addRow("", sv_row)

    dlg.moon_lbl_status = QLabel("")
    dlg.moon_lbl_status.setWordWrap(True)
    dlg.moon_lbl_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    form.addRow(dlg.moon_lbl_status)

    def _sync():
        solved = dlg.moon_chk_use_solved.isChecked() and not dlg.moon_chk_use_solved.isHidden()
        dlg.moon_chk_unknown.setEnabled(not solved)
        dlg.moon_chk_site.setEnabled(not solved)
        dlg.moon_dt.setEnabled(not solved and not dlg.moon_chk_unknown.isChecked())
        dlg.moon_btn_time.setEnabled(not solved and not dlg.moon_chk_unknown.isChecked())
        for w in (dlg.moon_lat, dlg.moon_lon, dlg.moon_elev):
            w.setEnabled(not solved and dlg.moon_chk_site.isChecked())
    dlg._moon_sync = _sync
    dlg.moon_chk_unknown.toggled.connect(_sync)
    dlg.moon_chk_site.toggled.connect(_sync)
    dlg.moon_chk_use_solved.toggled.connect(_sync)
    set_form_rows_visible(form, (dlg.moon_chk_use_solved,), False)
    _sync()
    # capture time, site and the stored placement are data about this image,
    # not tool parameters
    exclude = (dlg.moon_dt, dlg.moon_chk_unknown, dlg.moon_chk_site, dlg.moon_lat, dlg.moon_lon,
               dlg.moon_elev, dlg.moon_chk_use_solved)
    return box, exclude


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class _Cancelled(Exception):
    pass


class _MoonWorker(QThread):
    progress = pyqtSignal(float, str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, job):
        super().__init__(None)            # parent=None: never owned by the dialog
        self._job = job
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            res = self._job(lambda f, m="": self.progress.emit(float(f), str(m)),
                            lambda: self._cancel)
            if self._cancel:
                return
            self.succeeded.emit(res)
        except _Cancelled:
            pass
        except Exception as ex:
            if self._cancel or "cancel" in str(ex).lower():
                return
            self.failed.emit(_error_text(ex))
        finally:
            self._job = None


def _error_text(ex) -> str:
    from setiastro.saspro import lunar_relief as LR
    if LR.is_network_error(ex):
        return _tr("Could not download the lunar terrain data: {0}.\n\nCheck the internet connection and "
                   "Generate again. Each download is cached, so it is needed only once per area."
                   ).format(LR.network_reason(ex))
    return str(ex) or ex.__class__.__name__


def _doc_title(doc) -> str:
    dn = getattr(doc, "display_name", None)
    try:
        t = dn() if callable(dn) else dn
    except Exception:
        t = None
    return str(t) if t else "Moon"


def _to01(img):
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0
    a = a.astype(np.float32, copy=False)
    mx = float(np.nanmax(a)) if a.size else 1.0
    return np.clip(a / mx if mx > 1.0 else a, 0.0, 1.0)


class MoonRelief(QObject):
    """Controller owned by PlanetProjectionDialog for the Moon (true relief) type."""

    def __init__(self, dlg):
        super().__init__(dlg)
        self.dlg = dlg
        self._worker = None
        self._progress = None
        self._last = None           # the last finished result (for the Save buttons)
        self._cache = {}            # (image key, when, site) -> LunarSolution
        self._derived = {}          # same key -> (DerivedSurface, score)
        self._stored = None         # the Lunar Solver's placement stored with the image
        self._stored_sig = None
        self._time_note = ""        # where the capture time came from (shown when not using _stored)
        dlg.moon_btn_time.clicked.connect(self.prefill_time)
        dlg.moon_chk_use_solved.toggled.connect(self._show_time_note)
        dlg.moon_btn_save_view.clicked.connect(self._save_viewer)
        dlg.moon_btn_save_mesh.clicked.connect(self._save_mesh)
        self._load_site()
        self.prefill_time(silent=True)
        self.refresh_stored()
        # solving in the Lunar Solver while this dialog is open: pick it up on return
        dlg.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if obj is self.dlg and ev.type() == QEvent.Type.WindowActivate and not self.busy():
            try:
                self.refresh_stored()
            except RuntimeError:
                pass
        return False

    # -- the Lunar Solver's stored placement ----------------------------------
    def refresh_stored(self):
        """Re-read the placement stored with the image; returns it or None."""
        dlg = self.dlg
        sol = stored_solution(getattr(dlg, "doc", None), dlg.image)
        sig = _solution_sig(sol)
        if sig != self._stored_sig:
            self._stored, self._stored_sig = sol, sig
            set_form_rows_visible(dlg._moon_form, (dlg.moon_chk_use_solved,), sol is not None)
            dlg.moon_chk_use_solved.setChecked(sol is not None)
            if sol is not None and sol.date_known and dlg.moon_chk_unknown.isChecked():
                # the solver knows the date the image did not carry: keep it for a re-solve
                try:
                    when = _dt.datetime.fromisoformat(str(sol.utc).strip().replace("Z", "")[:19])
                    dlg.moon_dt.setDateTime(QDateTime(when.year, when.month, when.day, when.hour,
                                                      when.minute, when.second))
                    dlg.moon_chk_unknown.setChecked(False)
                    self._time_note = _tr("Capture time from the Lunar Solver's solution.")
                except Exception:
                    pass
            dlg._moon_sync()
            self._show_time_note()
        return self._stored

    def _use_stored(self) -> bool:
        return self._stored is not None and self.dlg.moon_chk_use_solved.isChecked()

    def _show_time_note(self, *_):
        dlg = self.dlg
        sol = self._stored
        if not self._use_stored():
            dlg.moon_lbl_time_src.setText(self._time_note)
            return
        if sol.date_known:
            when = str(sol.utc).replace("T", " ")[:19] + " UTC"
        else:
            when = _tr("date solved blind, colongitude {0:.1f}°").format(float(sol.colongitude))
        mode = {"disc": _tr("full disc"), "closeup": _tr("close-up"),
                "blind": _tr("blind search")}.get(sol.mode, sol.mode)
        dlg.moon_lbl_time_src.setText(_tr("Placed by the Lunar Solver ({0}, {1:.2f} px RMS; {2}).")
                                      .format(mode, float(sol.rms_px), when))

    # -- prefill --------------------------------------------------------------
    def _load_site(self):
        dlg = self.dlg
        hdr = _header_of(getattr(dlg, "doc", None))
        lat = _parse_angle(_hget(hdr, "SITELAT"))
        lon = _parse_angle(_hget(hdr, "SITELONG"))
        elev = _parse_angle(_hget(hdr, "SITEELEV"))
        s = QSettings()
        if lat is None or lon is None:
            lat = s.value(_SETTINGS + "lat", None, type=float) if s.contains(_SETTINGS + "lat") else None
            lon = s.value(_SETTINGS + "lon", None, type=float) if s.contains(_SETTINGS + "lon") else None
            elev = s.value(_SETTINGS + "elev", 0, type=float) if s.contains(_SETTINGS + "elev") else 0.0
        if lat is not None and lon is not None:
            dlg.moon_lat.setValue(float(lat))
            dlg.moon_lon.setValue(float(lon))
            dlg.moon_elev.setValue(int(round(elev or 0.0)))
            dlg.moon_chk_site.setChecked(True)

    def prefill_time(self, silent=False):
        dlg = self.dlg
        doc = getattr(dlg, "doc", None)
        meta = getattr(doc, "metadata", None) or {}
        when, src = None, ""
        try:
            from setiastro.saspro.lunar_solver import capture_time
            when, src = capture_time(meta, meta.get("file_path"))
        except ImportError:
            hdr = _header_of(doc)
            v = _hget(hdr, "DATE-OBS") or _hget(hdr, "DATE-BEG")
            if v:
                try:
                    when = _dt.datetime.fromisoformat(str(v).strip().replace("Z", ""))
                    src = "DATE-OBS"
                except Exception:
                    when = None
        except Exception as ex:
            src = str(ex)
        if when is not None:
            if when.tzinfo is not None:
                when = when.astimezone(_dt.timezone.utc).replace(tzinfo=None)
            dlg.moon_dt.setDateTime(QDateTime(when.year, when.month, when.day, when.hour,
                                              when.minute, when.second))
            dlg.moon_chk_unknown.setChecked(False)
            src = (src or "").strip().rstrip(".")
            if src[:5].lower() == "from ":
                src = src[5:]
            self._time_note = _tr("Capture time from {0}.").format(src or _tr("the image"))
        elif silent:
            # opening the tool: never guess "now" -- solve the date from the image
            dlg.moon_chk_unknown.setChecked(True)
            self._time_note = _tr("No capture time found in the image, so the date will be solved from "
                                  "the image (slower). Untick 'Date unknown' to enter it in UTC.")
        else:
            QMessageBox.information(dlg, _tr("Lunar Relief"),
                                    _tr("No capture time found in this image. Enter it in UTC, "
                                        "or tick 'Date unknown' to solve it from the image."))
            self._time_note = _tr("No capture time found in the image - enter it (UTC) "
                                  "or tick 'Date unknown'.")
        self._show_time_note()

    # -- generate -------------------------------------------------------------
    def busy(self) -> bool:
        return self._worker is not None

    def generate(self):
        dlg = self.dlg
        if self.busy():
            return
        if dlg.image is None:
            QMessageBox.information(dlg, _tr("Lunar Relief"), _tr("No image loaded."))
            return
        import importlib.util
        if importlib.util.find_spec("setiastro.saspro.lunar_solver") is None:
            QMessageBox.information(dlg, _tr("Lunar Relief"),
                                    _tr("The lunar solver is not available in this build, so the image "
                                        "cannot be placed on the globe yet."))
            return
        if not self._download_consent():
            return
        self.refresh_stored()
        use_stored = self._use_stored()
        if dlg.moon_chk_site.isChecked() and not use_stored:
            s = QSettings()
            s.setValue(_SETTINGS + "lat", float(dlg.moon_lat.value()))
            s.setValue(_SETTINGS + "lon", float(dlg.moon_lon.value()))
            s.setValue(_SETTINGS + "elev", float(dlg.moon_elev.value()))

        qd = dlg.moon_dt.dateTime()
        when = None
        if not dlg.moon_chk_unknown.isChecked():
            d, t = qd.date(), qd.time()
            when = _dt.datetime(d.year(), d.month(), d.day(), t.hour(), t.minute(), t.second())
        site = ((float(dlg.moon_lat.value()), float(dlg.moon_lon.value()), float(dlg.moon_elev.value()))
                if dlg.moon_chk_site.isChecked() else None)
        img01 = _to01(dlg.image)
        if img01.ndim == 3 and img01.shape[2] > 3:
            img01 = img01[..., :3]
        img_key = dlg._current_image_key(np.asarray(dlg.image))
        if use_stored:
            when = site = None
            key = (img_key, "lunar_solver") + self._stored_sig
        else:
            key = (img_key, when.isoformat() if when else "", site)
        params = dict(
            key=key, when=when, site=site, stored=use_stored,
            derived=dlg.moon_cmb_source.currentIndex() == 1,
            exag=float(dlg.moon_spin_exag.value()), globe=bool(dlg.moon_chk_globe.isChecked()),
            max_vertices=int(dlg.moon_spin_vertices.value()), title=_doc_title(getattr(dlg, "doc", None)))
        cached_sol = self._stored if use_stored else self._cache.get(key)
        cached_derived = self._derived.get(key)

        def job(progress, cancel):
            return _run_job(img01, params, cached_sol, cached_derived, progress, cancel)

        self._start_worker(job, _tr("Placing the image on the lunar globe..."))

    def _start_worker(self, job, label):
        dlg = self.dlg
        w = _MoonWorker(job)
        w.progress.connect(self._on_progress)
        w.succeeded.connect(self._on_done)
        w.failed.connect(self._on_failed)
        w.finished.connect(self._on_finished)
        self._worker = w
        _LIVE_WORKERS.add(w)
        self._progress = QProgressDialog(label, _tr("Cancel"), 0, 1000, dlg)
        self._progress.setWindowTitle(_tr("Lunar Relief"))
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setMinimumDuration(0)
        self._progress.canceled.connect(w.cancel)
        self._progress.show()
        dlg.btn_generate.setEnabled(False)
        for b in (dlg.moon_btn_save_view, dlg.moon_btn_save_mesh):
            b.setEnabled(False)
        w.start()

    def _download_consent(self) -> bool:
        s = QSettings()
        if s.value(_SETTINGS + "download_ok", False, type=bool):
            return True
        from setiastro.saspro.lunar_relief import LolaDEM
        r = QMessageBox.question(
            self.dlg, _tr("Lunar Relief - terrain data"),
            _tr("Lunar Relief uses NASA LRO terrain and albedo data, downloaded once and cached in\n"
                "{0}\n\n"
                "About 35 MB up front, then ~12 MB for each 4° band of latitude that your close-ups\n"
                "cover (whole-disc work uses the 35 MB set only).\n\nDownload when needed?")
            .format(str(LolaDEM.default_dir())))
        if r != QMessageBox.StandardButton.Yes:
            return False
        s.setValue(_SETTINGS + "download_ok", True)
        return True

    def _on_progress(self, f, msg):
        if self._progress is not None:
            self._progress.setValue(int(max(0.0, min(1.0, f)) * 1000))
            if msg:
                self._progress.setLabelText(msg)

    def _on_failed(self, msg):
        if self.sender() is not self._worker:
            return
        QMessageBox.warning(self.dlg, _tr("Lunar Relief"), msg)

    def _on_finished(self):
        w = self.sender()
        _LIVE_WORKERS.discard(w)
        if w is not self._worker:
            return                      # a stale worker must not release the current one
        self._worker = None
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        try:
            self.dlg._update_enable()
            for b in (self.dlg.moon_btn_save_view, self.dlg.moon_btn_save_mesh):
                b.setEnabled(self._last is not None)
        except RuntimeError:
            pass

    def shutdown(self):
        """Dialog closing: stop quietly; the worker stays referenced until it ends."""
        if self._worker is not None:
            self._worker.cancel()
        if self._progress is not None:
            self._progress.close()
            self._progress = None

    def _on_done(self, res):
        if self.sender() is not self._worker:
            return
        dlg = self.dlg
        if res.get("kind") == "mesh":
            dlg.moon_lbl_status.setText(_status_text(self._last["sol"], self._last) + "\n" +
                                        _tr("Mesh saved: {0}.ply and .obj").format(os.path.basename(res["stem"])))
            return
        sol = res["sol"]
        self._cache[res["key"]] = sol
        if res.get("surf") is not None:
            self._derived[res["key"]] = (res["surf"], res.get("score"))
        dlg.moon_lbl_status.setText(_status_text(sol, res))
        if not res.get("stored") and not sol.date_known and getattr(sol, "geometry", None) is not None:
            dlg.moon_lbl_time_src.setText(_tr("Date solved blind from the image: colongitude {0:.1f}°.")
                                          .format(float(sol.colongitude)))
        self._last = res
        dlg.moon_btn_save_view.setEnabled(True)
        dlg.moon_btn_save_mesh.setEnabled(True)
        try:
            path = _write_temp_viewer(res["html"], res["title"])
        except OSError as ex:
            QMessageBox.warning(dlg, _tr("Lunar Relief"),
                                _tr("Could not write the 3D viewer:\n{0}\n\nUse Save 3D Viewer to put it "
                                    "somewhere else.").format(ex))
            return
        if not webbrowser.open(Path(path).as_uri()):
            QMessageBox.information(dlg, _tr("Lunar Relief"),
                                    _tr("Open this file in your web browser:\n{0}").format(path))

    def _save_viewer(self):
        res, dlg = self._last, self.dlg
        if res is None:
            return
        fn, _ = QFileDialog.getSaveFileName(dlg, _tr("Save 3D Viewer As"), _safe_name(res["title"]) + "_3d.html",
                                            _tr("HTML Files (*.html)"))
        if not fn:
            return
        if not fn.lower().endswith((".html", ".htm")):
            fn += ".html"
        try:
            with open(fn, "w", encoding="utf-8") as f:
                f.write(res["html"])
        except OSError as ex:
            QMessageBox.warning(dlg, _tr("Lunar Relief"), _tr("Could not save the viewer:\n{0}").format(ex))

    def _save_mesh(self):
        res, dlg = self._last, self.dlg
        if res is None or self.busy():
            return
        fn, _ = QFileDialog.getSaveFileName(dlg, _tr("Save Mesh As"), _safe_name(res["title"]) + "_3d.ply",
                                            _tr("PLY mesh (*.ply)"))
        if not fn:
            return
        stem = os.path.splitext(fn)[0]
        exag = float(dlg.moon_spin_exag.value())

        def job(progress, cancel):            # a big OBJ takes seconds to write: not on the GUI thread
            from setiastro.saspro import lunar_relief as LR
            from setiastro.saspro import lunar_relief_viewer as LRV
            g, cam = res["grid"], res["sol"].camera
            progress(0.05, _tr("Building the mesh..."))
            tex = LRV.display_image(res["image"], g)
            mesh = LR.build_mesh_from_points(tex, g.uv, g.X, g.valid, cam, exag, source=g.source)
            if cancel():
                raise _Cancelled()
            progress(0.3, _tr("Writing {0}...").format(os.path.basename(stem) + ".ply"))
            LR.export_ply(stem + ".ply", mesh)
            progress(0.45, _tr("Writing {0}...").format(os.path.basename(stem) + ".obj"))
            LR.export_obj(stem + ".obj", mesh, tex)
            progress(1.0, "")
            return dict(kind="mesh", stem=stem)

        self._start_worker(job, _tr("Saving the mesh..."))


def _safe_name(title: str) -> str:
    stem = os.path.splitext(os.path.basename(str(title or "moon")))[0]
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in stem)[:80] or "moon"


def _write_temp_viewer(html: str, title: str) -> str:
    """The viewer goes to a SASpro folder in the temp dir (one file per image name,
    replaced on each Generate; files older than a week are swept)."""
    import time
    d = Path(tempfile.gettempdir()) / "SASpro_LunarRelief"
    d.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for old in d.glob("*.html"):
        try:
            if now - old.stat().st_mtime > 7 * 86400:
                old.unlink()
        except OSError:
            pass
    path = d / (_safe_name(title) + "_3d.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return str(path)


def _status_text(sol, res) -> str:
    head = (_tr("Placed by the Lunar Solver ({0}): {1:.2f} px rms over {2} matches; colongitude {3:.1f}°; "
                "{4:.2f} km/px.") if res.get("stored") else
            _tr("Placed on the globe ({0}): {1:.2f} px rms over {2} matches; colongitude {3:.1f}°; "
                "{4:.2f} km/px."))
    lines = [head.format(sol.mode, float(sol.rms_px), int(sol.n_matches), float(sol.colongitude),
                         float(sol.km_per_px))]
    surf, score = res.get("surf"), res.get("score")
    if surf is not None and getattr(sol, "sun_approximate", False):
        lines.append(_tr("The date was solved blind and the Sun angle is only approximate, so the "
                         "derived heights are too. Enter the capture time and solve again."))
    if surf is not None:
        la, lo = surf.centre_latlon
        lines.append(_tr("Derived area: {0:.0f} km square around {1:.1f}°{2}, {3:.1f}°{4}; "
                         "{5} shadows anchor it.").format(2 * surf.half_km, abs(la), "N" if la >= 0 else "S",
                                                          abs(lo), "E" if lo >= 0 else "W", surf.n_shadows))
        if score:
            lines.append(_tr("Check against LOLA: local relief corr {0:.2f} at {1:.0%} amplitude; "
                             "slope toward the Sun corr {2:.2f}.").format(score["relief_corr"],
                                                                         score["relief_amplitude"],
                                                                         score["sun_slope_corr"]))
            if score["relief_corr"] < 0.35:
                lines.append(_tr("Low agreement: shading only reads true slopes from a linear, unsharpened "
                                 "stack. Try the raw stacker output."))
        if res.get("data_note"):
            lines.append(res["data_note"])
    if res.get("dem_note"):
        lines.append(res["dem_note"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The work (worker thread; no Qt)
# ---------------------------------------------------------------------------

def _run_job(img01, p, cached_sol, cached_derived, progress, cancel):
    from setiastro.saspro import lunar_relief as LR
    from setiastro.saspro import lunar_solver as LS

    def check():
        if cancel():
            raise _Cancelled()

    def sub(a, b):
        return lambda f, m="": progress(a + (b - a) * float(f), m)

    def dl(a, b):
        """LolaDEM reports (bytes got, bytes total), not (fraction, message)."""
        def _p(got, total):
            progress(a + (b - a) * min(1.0, got / max(total, 1)),
                     _tr("Downloading LOLA terrain for this area ({0} of {1} MB)...")
                     .format(int(got) >> 20, max(1, int(total) >> 20)))
        return _p

    lola = LR.LolaDEM(LR.LolaDEM.default_dir())
    LS.ensure_data(progress=sub(0.0, 0.1), cancel=cancel)
    check()
    sol = cached_sol
    if sol is None:
        progress(0.1, _tr("Placing the image on the lunar globe..."))
        sol = LS.solve_moon(img01, p["when"], p["site"], progress=sub(0.1, 0.45), cancel=cancel)
    check()
    cam, geom = sol.camera, sol.geometry
    H, W = img01.shape[:2]
    mono = img01.mean(axis=2) if img01.ndim == 3 else img01
    res = dict(key=p["key"], sol=sol, surf=None, score=None, stored=bool(p.get("stored")))

    surf = score = None
    if p["derived"]:
        levels = np.unique(np.round(mono[::7, ::7] * 65535.0)).size
        if levels < 1024:
            res["data_note"] = _tr("This image has only {0} brightness levels (8-bit or processed data), "
                                   "which limits shading-derived slopes.").format(levels)
        if cached_derived is not None:
            surf, score = cached_derived
        else:
            progress(0.45, _tr("Measuring shadows and shading..."))
            surf = LR.derive_surface(mono, cam, geom, progress=sub(0.45, 0.85), cancel=cancel)
            check()
            try:
                C = LR.sph_to_body(*surf.centre_latlon)
                lat0 = LR.body_to_latlon(C)[0]
                pad = surf.half_km / 30.0 + 1.0
                band = lola.band64(lat0 - pad, lat0 + pad, progress=dl(0.85, 0.9), cancel=cancel)
                score = LR.score_against_lola(surf, geom, band)
                score.pop("lola_heights", None)
            except Exception:
                score = None
        res["surf"], res["score"] = surf, score

    def dem_for(spacing_km, x0=0, y0=0, x1=W, y1=H, cam_=cam):
        """LDEM_16 when the sampling is coarser than its cells, else the LDEM_64 band."""
        if spacing_km >= _LOLA16_KM:
            return lola.global16(progress=dl(0.45, 0.88), cancel=cancel)
        span = LR.footprint_latitudes(cam_, (y1 - y0, x1 - x0))
        if span is None:
            raise ValueError(_tr("No lunar surface in this image."))
        try:
            return lola.band64(*span, progress=dl(0.45, 0.88), cancel=cancel)
        except Exception as ex:
            # offline: the global terrain is cached (the solver needs it), so use it
            # rather than fail, and say so -- coarser, still true heights
            if cancel() or not LR.is_network_error(ex) or not lola.has_global16():
                raise
            res["dem_note"] = _tr("The detailed LOLA terrain for this area could not be downloaded ({0}), so "
                                  "the coarser 1.9 km global terrain was used. Generate again when online for "
                                  "full detail.").format(LR.network_reason(ex))
            return lola.global16(cancel=cancel)

    from setiastro.saspro import lunar_relief_viewer as LRV
    if surf is not None:
        progress(0.9, _tr("Building the 3D terrain..."))
        grid = LR.relief_grid(cam, (H, W), surf=surf, max_vertices=p["max_vertices"])
        heights = _tr("heights derived from this image (shadows + shading)")
    else:
        n_disk = max(1.0, math.pi * float(getattr(sol, "radius_px", max(H, W) / 2.0)) ** 2)
        n_px = min(float(H * W), n_disk)
        spacing = float(sol.km_per_px) * max(1.0, math.sqrt(n_px / p["max_vertices"]))
        dem = dem_for(spacing)          # may download first
        check()
        progress(0.9, _tr("Building the 3D terrain..."))
        grid = LR.relief_grid(cam, (H, W), dem=dem, max_vertices=p["max_vertices"])
        heights = _tr("NASA LOLA terrain, {0:g} px/deg").format(float(dem.ppd))
    check()
    dem16 = albedo = None
    if p.get("globe") and lola.has_global16():
        dem16 = lola.global16(cancel=cancel)
        albedo = LRV.load_albedo(lola.cache_dir)
    check()
    progress(0.95, _tr("Preparing the 3D viewer..."))
    when = (str(sol.utc).replace("T", " ")[:16] + " UTC" if sol.date_known else
            _tr("date unknown (colongitude {0:.1f}°)").format(float(sol.colongitude)))
    sub = "  ·  ".join([heights, when, _tr("{0:.2f} km/px").format(float(sol.km_per_px)),
                        _tr("placed to {0:.2f} px RMS").format(float(sol.rms_px))])
    notes = [res.get("dem_note", "")]
    if surf is not None and score and score["relief_corr"] < 0.35:
        notes.append(_tr("Derived heights agree poorly with LOLA here (corr {0:.2f}).").format(score["relief_corr"]))
    html = LRV.viewer_html(grid, img01, cam, title=p["title"], subtitle=sub, exaggeration=p["exag"],
                           dem16=dem16, albedo=albedo, notes=notes)
    check()
    res.update(html=html, grid=grid, image=img01, exag=p["exag"], title=p["title"])
    progress(1.0, "")
    return res
