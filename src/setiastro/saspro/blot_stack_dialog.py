"""Stacking Suite controls for native detector registration and blot rejection."""
from pathlib import Path

from astropy.io import fits
from PyQt6.QtCore import QSettings, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from .blot_spcc import SpectralCalibration
from .blot_stack import StackOptions, stack
from .gaia_database import get_library_dir
from .spectral_curves import builtin_curves_path, list_curves, user_curves_path


class StackThread(QThread):
    message = pyqtSignal(str)

    def __init__(self, paths, reference, output, options, calibration, parent):
        super().__init__(parent)
        self.args = paths, reference, output, options, calibration
        self.error = None
        self.cancelled = False

    def check(self):
        if self.isInterruptionRequested():
            raise InterruptedError

    def run(self):
        try:
            stack(*self.args, log=self.message.emit, check=self.check)
        except InterruptedError:
            self.cancelled = True
        except Exception as exc:
            self.error = str(exc)


class BlotStackDialog(QDialog):
    def __init__(self, groups, reference=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Drizzle → Blot → SPCC → ImageMM"))
        self.resize(610, 640)
        self.groups = dict(groups)
        self.preferred_reference = reference
        self.worker = None
        self.settings = QSettings()
        self.controls = {}
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        form = self._tab("Registration and Blot")
        self.group = QComboBox()
        self.group.addItems(list(groups))
        choose = QPushButton(self.tr("Choose detector lights…"))
        choose.clicked.connect(self._choose_lights)
        row = QHBoxLayout()
        row.addWidget(self.group, 1)
        row.addWidget(choose)
        form.addRow(self.tr("Calibrated, unregistered lights"), row)
        self.reference = QComboBox()
        form.addRow(self.tr("Registration reference"), self.reference)
        self.group.currentTextChanged.connect(self._group_changed)
        self._group_changed(self.group.currentText())
        self._check(form, "align_colors", "Align colors independently", True)
        self._number(form, "scale", "Drizzle scale", 1., .1, 8., 2)
        aperture = self._number(form, "aperture_mm", "Automatic scale · aperture (mm)", 0., 0., 20000., 1)
        aperture.setSpecialValueText(self.tr("Use manual scale"))
        aperture.valueChanged.connect(lambda n: self.controls["scale"].setEnabled(n == 0))
        self.controls["scale"].setEnabled(aperture.value() == 0)
        self._number(form, "sampling", "Input arcsec/pixel", 0., 0., 1000., 3).setSpecialValueText(self.tr("From reference"))
        self._number(form, "pixfrac", "Pixel fraction", 0., 0., 2., 3).setSpecialValueText(self.tr("Automatic"))
        self._combo(form, "kernel", "Drizzle kernel", ["Square", "Disk", "Gaussian", "Point"], "Square")
        self._number(form, "primary", "Primary rejection SNR", 5., .1, 100., 1)
        self._number(form, "adjacent", "Adjacent rejection SNR", 4., .1, 100., 1)

        form = self._tab("SPCC")
        self.spcc = self._check(form, "spcc", "Calibrate the clean RGB seed with Gaia XP", True)
        form.addRow(QLabel(self.tr("SPCC requires a plate-solved reference and installed Gaia XP spectra.")))
        self.catalog = QLineEdit(self.settings.value("blot/catalog", str(get_library_dir())))
        browse = QPushButton(self.tr("Browse…"))
        browse.clicked.connect(self._choose_catalog)
        row = QHBoxLayout()
        row.addWidget(self.catalog)
        row.addWidget(browse)
        form.addRow(self.tr("Gaia XP library folder"), row)
        builtin = builtin_curves_path()
        curves = list_curves(builtin, user_curves_path())
        self.filters = {}
        self.sensors = {}
        for curve in curves:
            library = self.sensors if curve.kind == "SENSOR" else self.filters
            library.setdefault(curve.name, (curve.path, curve.name))
        with fits.open(builtin, memmap=False) as hdus:
            self.whites = {h.name: (builtin, h.name) for h in hdus if h.header.get("CTYPE") == "SED"}
        for key, label, choices, sfcc_key, default in (
            ("red", "Red passband", self.filters, "RFilter", "BAADER_R"),
            ("green", "Green passband", self.filters, "GFilter", "BAADER_G"),
            ("blue", "Blue passband", self.filters, "BFilter", "BAADER_B"),
            ("sensor", "Sensor QE", self.sensors, "Sensor", "100%_QE_CURVE"),
            ("white", "White reference", self.whites, "WhiteReference", "A0V"),
            ("common", "Common optical filter", ["(None)", *sorted(self.filters)], "LPFilter", "(None)"),
        ):
            saved = self.settings.value("SFCC/" + sfcc_key, default)
            self._combo(form, key, label, list(choices), saved if saved in choices else default)
        form = self._tab("ImageMM")
        self._number(form, "iterations", "Maximum iterations", 7, 0, 500)
        self._number(form, "minimum", "Minimum iterations", 3, 1, 500)
        self._number(form, "kappa", "Update clip", 1.1, 1., 10., 3)
        self._number(form, "huber_delta", "Huber delta (negative = RMS multiplier)", -2., -100., 100., 3)
        self._number(form, "strength", "Huber strength (0 = L2)", .85, 0., 1., 2)
        self._number(form, "fallback_fwhm", "Fallback FWHM (output pixels)", 3., .1, 100., 2)
        self._number(form, "tolerance", "Relative convergence tolerance", .0005, 0., 1., 5)
        self._check(form, "luminance", "Refine luminance and preserve seed chroma", False)
        self._check(form, "protect_stars", "Protect star cores", True)
        self._check(form, "noise_map", "Estimate local noise", True)
        self._check(form, "save_iterations", "Save seed and iteration planes", False)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.run_button = self.buttons.addButton(self.tr("Stack…"), QDialogButtonBox.ButtonRole.ActionRole)
        self.run_button.clicked.connect(self._start)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def _tab(self, title):
        tab = QWidget()
        self.tabs.addTab(tab, self.tr(title))
        return QFormLayout(tab)

    def _check(self, form, key, label, default):
        control = QCheckBox(self.tr(label))
        control.setChecked(self.settings.value("blot/"+key, default, type=bool))
        form.addRow(control)
        self.controls[key] = control
        return control

    def _number(self, form, key, label, default, low, high, precision=None):
        if precision is None:
            control = QSpinBox()
            dtype = int
        else:
            control = QDoubleSpinBox()
            control.setDecimals(precision)
            control.setSingleStep(10. ** -min(precision, 3))
            dtype = float
        control.setRange(low, high)
        control.setValue(self.settings.value("blot/"+key, default, type=dtype))
        form.addRow(self.tr(label), control)
        self.controls[key] = control
        return control

    def _combo(self, form, key, label, choices, default):
        control = QComboBox()
        control.addItems(choices)
        control.setCurrentText(default)
        control.setCurrentText(self.settings.value("blot/"+key, default))
        form.addRow(self.tr(label), control)
        self.controls[key] = control

    def _choose_catalog(self):
        directory = QFileDialog.getExistingDirectory(self, self.tr("Gaia XP library"), self.catalog.text())
        if directory:
            self.catalog.setText(directory)

    def _choose_lights(self):
        paths, _ = QFileDialog.getOpenFileNames(self, self.tr("Select unregistered detector lights"), "",
                                                self.tr("Detector lights (*.fit *.fits *.fts *.fit.fz *.fits.fz *.xisf)"))
        if paths:
            label = self.tr("Selected detector lights")
            self.groups[label] = paths
            if self.group.findText(label) < 0:
                self.group.addItem(label)
            self.group.setCurrentText(label)
            self._group_changed(label)

    def _group_changed(self, label):
        self.reference.clear()
        for path in self.groups.get(label, []):
            self.reference.addItem(Path(path).name, str(path))
            if self.preferred_reference and Path(path) == Path(self.preferred_reference):
                self.reference.setCurrentIndex(self.reference.count()-1)

    def _start(self):
        paths = self.groups.get(self.group.currentText(), [])
        if len(paths) < 3:
            QMessageBox.warning(self, self.windowTitle(), self.tr("Select at least three unregistered lights."))
            return
        values = {}
        for key, widget in self.controls.items():
            if isinstance(widget, QCheckBox):
                values[key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                values[key] = widget.currentText()
            else:
                values[key] = widget.value()
        calibration = None
        if self.spcc.isChecked():
            directory = Path(self.catalog.text())
            if not any(directory.glob("gaia_xp_*.sqlite")):
                QMessageBox.warning(self, self.windowTitle(), self.tr("Choose a folder with installed SAS Gaia XP spectra."))
                return
            calibration = SpectralCalibration(tuple(self.filters[values[c]] for c in ("red", "green", "blue")),
                                              self.sensors[values["sensor"]], self.whites[values["white"]],
                                              directory, self.filters.get(values["common"]))
        if values["strength"] and values["huber_delta"] == 0:
            QMessageBox.warning(self, self.windowTitle(), self.tr("Choose a nonzero Huber delta or set strength to zero."))
            return
        output, _ = QFileDialog.getSaveFileName(self, self.tr("Save ImageMM master"),
                                               str(Path(paths[0]).parent / "ImageMM.fit"),
                                               self.tr("FITS image (*.fit *.fits)"))
        if not output:
            return
        for key, value in values.items():
            self.settings.setValue("blot/"+key, value)
        self.settings.setValue("blot/catalog", self.catalog.text())
        options = StackOptions(**{k: values[k] for k in StackOptions.__dataclass_fields__})
        self.log.clear()
        self.tabs.setEnabled(False)
        self.run_button.setEnabled(False)
        self.buttons.button(QDialogButtonBox.StandardButton.Close).setText(self.tr("Cancel"))
        self.worker = StackThread(paths, self.reference.currentData(), output, options, calibration, self)
        self.worker.message.connect(self.log.appendPlainText)
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def _finished(self):
        worker = self.worker
        self.worker = None
        self.tabs.setEnabled(True)
        self.run_button.setEnabled(True)
        self.buttons.button(QDialogButtonBox.StandardButton.Close).setText(self.tr("Close"))
        if worker.error:
            self.log.appendPlainText(worker.error)
            QMessageBox.warning(self, self.windowTitle(), worker.error)
        elif worker.cancelled:
            self.log.appendPlainText(self.tr("Stack cancelled."))
        worker.deleteLater()

    def reject(self):
        if self.worker is not None:
            self.worker.requestInterruption()
            return
        super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.requestInterruption()
            event.ignore()
        else:
            super().closeEvent(event)
