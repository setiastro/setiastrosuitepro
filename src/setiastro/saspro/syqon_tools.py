# src/setiastro/saspro/syqon_tools.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from PyQt6.QtCore import Qt, QSettings, QUrl, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QStackedWidget, QWidget, QFormLayout, QGroupBox, QMessageBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QFileDialog, QProgressBar, QSlider
)
from PyQt6.QtGui import QIcon, QDesktopServices, QPixmap

from setiastro.saspro.resources import starnet_path

from setiastro.saspro.syqon_paths import syqon_prism_model_path
from setiastro.saspro.denoise_engines.syqon_prism_engine import (
    prism_denoise_rgb01,
    clear_prism_models_cache,
)
from setiastro.saspro.starless_engines.syqon_nafnet_engine import (
    clear_axiom_models_cache,
)
from setiastro.saspro.remove_stars import (
    SyQonStarlessDialog,SyQonLivePreviewWindow, 
    _ProcDialog,
    _mtf_params_unlinked,
    _apply_mtf_unlinked_rgb,
    _invert_mtf_unlinked_rgb,
)
from setiastro.saspro.resources import starnet_path, get_icons
_SYQON_BUY_URL_PRISM_MINI = "https://github.com/setiastro/setiastrosuitepro/releases/download/benchmarkFIT/prism_mini"   # replace with exact URL when you have it
_SYQON_BUY_URL_PRISM_DEEP = "https://syqon.it/prism"   # replace with exact URL when you have it

def _syqon_prism_buy_url_for(model_kind: str) -> str:
    mk = (model_kind or "prism_mini").lower().strip()
    return _SYQON_BUY_URL_PRISM_DEEP if mk == "prism_deep" else _SYQON_BUY_URL_PRISM_MINI

def _blend_result_with_mask(processed: np.ndarray, orig: np.ndarray, doc) -> np.ndarray:
    """
    Blend processed result with orig using doc's active mask.
    orig must already match the shape/dtype of processed.
    Returns processed unchanged if no mask is active or mask is invalid.
    """
    mid = getattr(doc, "active_mask_id", None) or None
    if not mid:
        return processed

    layer = getattr(doc, "masks", {}).get(mid)
    if layer is None:
        return processed

    m = np.asarray(getattr(layer, "data", None))
    if m is None or m.size == 0:
        return processed

    m = m.astype(np.float32)
    if m.dtype.kind in "ui":
        imax = float(np.iinfo(m.dtype).max)
        m = m / imax if imax > 0 else m
    else:
        mx = float(m.max()) if m.size else 0.0
        if mx > 1.0:
            m = m / mx
        elif mx == 0.0:
            return processed  # blank mask = no effect

    m = np.clip(m, 0.0, 1.0)

    # resample mask if needed
    mh, mw = m.shape[:2]
    oh, ow = processed.shape[:2]
    if (mh, mw) != (oh, ow):
        yi = np.linspace(0, mh - 1, oh).astype(np.int32)
        xi = np.linspace(0, mw - 1, ow).astype(np.int32)
        m = m[yi][:, xi]

    out = processed.astype(np.float32)
    src = orig.astype(np.float32)

    if out.ndim == 3 and out.shape[2] >= 3:
        m = m[..., None]
        if src.ndim == 2:
            src = np.stack([src] * 3, axis=-1)
    elif out.ndim == 2:
        if src.ndim == 3:
            src = src[..., 0]

    result = m * out + (1.0 - m) * src
    if not np.isfinite(result).all():
        return processed  # safety net
    return result.astype(np.float32)

class SyQonToolsDialog(QDialog):
    def __init__(self, parent, docman, get_active_doc_callable, icon: QIcon | None = None):
        super().__init__(parent)
        self.docman = docman
        self.get_active_doc = get_active_doc_callable

        self.setWindowTitle("SyQon Tools")
        self.setMinimumSize(640, 520)
        if icon is not None:
            self.setWindowIcon(icon)

        self.settings = QSettings()

        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Tool family:"))
        self.cmb_family = QComboBox(self)
        self.cmb_family.addItem("Starless", userData="starless")
        self.cmb_family.addItem("Denoise", userData="denoise")
        self.cmb_family.addItem("Sharpening", userData="sharpening")
        top.addWidget(self.cmb_family, 1)
        lay.addLayout(top)

        self.stack = QStackedWidget(self)
        lay.addWidget(self.stack, 1)

        self.page_starless = _SyQonStarlessHubPage(self)
        self.page_denoise = _SyQonDenoiseHubPage(self)
        self.page_sharpen = _SyQonSharpenHubPage(self)

        self.stack.addWidget(self.page_starless)
        self.stack.addWidget(self.page_denoise)
        self.stack.addWidget(self.page_sharpen)

        btns = QHBoxLayout()
        self.btn_launch = QPushButton("Open Tool", self)
        self.btn_close = QPushButton("Close", self)
        self.btn_launch.clicked.connect(self._launch_selected_tool)
        self.btn_close.clicked.connect(self.reject)
        btns.addStretch(1)
        btns.addWidget(self.btn_launch)
        btns.addWidget(self.btn_close)
        lay.addLayout(btns)

        self.cmb_family.currentIndexChanged.connect(self._on_family_changed)

        # restore last-used family
        saved_family = str(self.settings.value("syqon/tools/last_family", "starless", type=str) or "starless")
        idx = self.cmb_family.findData(saved_family)
        if idx < 0:
            idx = self.cmb_family.findData("starless")
        if idx >= 0:
            self.cmb_family.setCurrentIndex(idx)

        self._sync_page()
        self.cmb_family.currentIndexChanged.connect(self._on_family_changed)
        self.cmb_family.currentIndexChanged.connect(self._maybe_embed_starless)
        self._sync_page()
        self._maybe_embed_starless()
        
    def _maybe_embed_starless(self, *_):
        if self.cmb_family.currentData() != "starless":
            return
        doc = self.get_active_doc()
        if doc is None or getattr(doc, "image", None) is None:
            return
        self.page_starless.ensure_embedded(self.parent(), doc)

    def _on_family_changed(self, *_):
        self.settings.setValue("syqon/tools/last_family", str(self.cmb_family.currentData() or "starless"))
        self._sync_page()

    def _sync_page(self):
        key = self.cmb_family.currentData()
        if key == "starless":
            self.stack.setCurrentWidget(self.page_starless)
            self.btn_launch.setVisible(False)
        elif key == "denoise":
            self.stack.setCurrentWidget(self.page_denoise)
            self.btn_launch.setVisible(True)
            self.btn_launch.setText("Process")
        else:
            self.stack.setCurrentWidget(self.page_sharpen)
            self.btn_launch.setVisible(True)
            self.btn_launch.setText("Process")

    def _launch_selected_tool(self):
        doc = self.get_active_doc()
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "SyQon Tools", "No active image.")
            return

        key = self.cmb_family.currentData()

        if key == "starless":
            # Already embedded — the Process button inside SyQonStarlessDialog handles it
            self.page_starless.ensure_embedded(self.parent(), doc)
            return

        if key == "denoise":
            self.page_denoise.process_document(doc, self.parent())
            return
        else:
            self.page_sharpen.process_document(doc, self.parent())

    def closeEvent(self, ev):
        try:
            page = getattr(self, "page_denoise", None)
            if page is not None:
                thr = getattr(page, "proc_thr", None)
                if thr is not None and thr.isRunning():
                    reply = QMessageBox.question(
                        self,
                        "SyQon Prism Running",
                        "Denoise is still processing.\n\n"
                        "Cancel it and close? (May take a moment to finish the current tile.)",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No,
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        ev.ignore()
                        return
                    thr.cancel()
                    if not thr.wait(10000):
                        thr.terminate()
                        thr.wait(2000)
                    page.proc_thr = None
        except Exception:
            pass
        super().closeEvent(ev)

class _SyQonStarlessHubPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._embedded_dlg = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._lay = lay

    def ensure_embedded(self, main, doc):
        """
        Called when the user switches to the Starless tab.
        Embeds SyQonStarlessDialog as a child widget (not shown as a dialog).
        Rebuilds it if the doc changed.
        """
        from setiastro.saspro.remove_stars import SyQonStarlessDialog

        if self._embedded_dlg is not None:
            # clear old one
            self._lay.removeWidget(self._embedded_dlg)
            self._embedded_dlg.setParent(None)
            self._embedded_dlg.deleteLater()
            self._embedded_dlg = None

        dlg = SyQonStarlessDialog(main, doc, parent=self)
        # Make it behave as an embedded widget, not a top-level dialog
        dlg.setWindowFlags(Qt.WindowType.Widget)
        dlg.setModal(False)

        # Hide the Close button since we're embedded — user closes via SyQon Tools
        try:
            dlg.btn_close.hide()
        except Exception:
            pass

        self._embedded_dlg = dlg
        self._lay.addWidget(dlg)
        dlg.show()
class _SyQonDenoiseHubPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.proc_thr = None
        self.doc = None
        self.main = None
        self._scale_factor = 1.0
        self._orig_was_mono = False
        self._do_mtf = False
        self._mtf_params = None

        self.settings = QSettings()
        self.icons = get_icons()

        lay = QVBoxLayout(self)

        # --- Prism logo ---
        self.lbl_logo = QLabel(self)
        self.lbl_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        try:
            pm = QPixmap(self.icons.SYQON_PRISM)
            if not pm.isNull():
                self.lbl_logo.setPixmap(
                    pm.scaledToWidth(260, Qt.TransformationMode.SmoothTransformation)
                )
                lay.addWidget(self.lbl_logo)
        except Exception:
            pass

        box = QGroupBox("SyQon Prism Denoise", self)
        form = QFormLayout(box)

        self.cmb_model = QComboBox(self)
        self.cmb_model.addItem("Prism Mini (free)", userData="prism_mini")
        self.cmb_model.addItem("Prism Deep (paid)", userData="prism_deep")
        self.cmb_model.currentIndexChanged.connect(self._on_model_changed)
        form.addRow("Model:", self.cmb_model)

        self.lbl_model_path = QLabel("", self)
        self.lbl_model_path.setWordWrap(True)
        form.addRow("Installed model path:", self.lbl_model_path)

        btn_row = QHBoxLayout()
        self.btn_buy = QPushButton("Get Model Here...", self)
        self.btn_install = QPushButton("Install Downloaded Model...", self)
        self.btn_remove = QPushButton("Remove Model", self)
        self.btn_clear_cache = QPushButton("Clear AI Cache", self)

        self.btn_buy.clicked.connect(self._open_buy_page)
        self.btn_install.clicked.connect(self._install_model)
        self.btn_remove.clicked.connect(self._remove_model)
        self.btn_clear_cache.clicked.connect(self._clear_ai_cache)

        btn_row.addWidget(self.btn_buy)
        btn_row.addWidget(self.btn_install)
        btn_row.addWidget(self.btn_remove)
        btn_row.addWidget(self.btn_clear_cache)
        self.chk_live_preview = QCheckBox("Live tile preview (slower)", self)
        self.chk_live_preview.setChecked(False)
        lay.addWidget(self.chk_live_preview)

        self.preview_win = None
        self.chk_live_preview.toggled.connect(self._toggle_live_preview_window)
        btn_wrap = QWidget(self)
        btn_wrap.setLayout(btn_row)
        form.addRow("", btn_wrap)

        self.lbl_status = QLabel("", self)
        self.lbl_status.setWordWrap(True)
        form.addRow("", self.lbl_status)

        self.spin_tile = QSpinBox(self)
        self.spin_tile.setRange(128, 2048)
        self.spin_tile.setValue(int(self.settings.value("syqon/prism_tile_size", 512)))
        form.addRow("Tile size:", self.spin_tile)

        self.spin_overlap = QSpinBox(self)
        self.spin_overlap.setRange(16, 512)
        self.spin_overlap.setValue(int(self.settings.value("syqon/prism_overlap", 64)))
        form.addRow("Overlap:", self.spin_overlap)

        # --- NEW: edge padding (for tile context) ---
        self.spin_pad = QSpinBox(self)
        self.spin_pad.setRange(0, 2048)
        # default: at least overlap, with a little cushion (old behavior)
        default_pad = max(int(self.settings.value("syqon/prism_overlap", 64)), 32)
        self.spin_pad.setValue(int(self.settings.value("syqon/prism_pad", default_pad)))
        form.addRow("Edge pad (px):", self.spin_pad)

        pad_help = QLabel(
            "Pads edges before tiling (reflect) to prevent border artifacts.\n"
            "Typical: overlap or overlap+32. 0 disables padding."
        )
        pad_help.setWordWrap(True)
        pad_help.setStyleSheet("color:#888;")
        form.addRow("", pad_help)

        self.chk_mtf = QCheckBox("Apply temporary stretch for model (recommended)", self)
        self.chk_mtf.setChecked(bool(self.settings.value("syqon/prism_use_mtf", True, type=bool)))
        form.addRow("", self.chk_mtf)

        self.spin_mtf_median = QDoubleSpinBox(self)
        self.spin_mtf_median.setRange(0.01, 0.50)
        self.spin_mtf_median.setSingleStep(0.01)
        self.spin_mtf_median.setDecimals(3)
        self.spin_mtf_median.setValue(float(self.settings.value("syqon/prism_mtf_target_median", 0.10)))
        form.addRow("Temp stretch target median:", self.spin_mtf_median)
        saved_strength = float(self.settings.value("syqon/prism_strength", 0.85))

        self.sld_strength = QSlider(Qt.Orientation.Horizontal, self)
        self.sld_strength.setRange(0, 100)
        self.sld_strength.setSingleStep(1)
        self.sld_strength.setPageStep(5)
        self.sld_strength.setValue(int(round(saved_strength * 100.0)))

        self.spin_strength = QDoubleSpinBox(self)
        self.spin_strength.setRange(0.0, 1.0)
        self.spin_strength.setSingleStep(0.01)
        self.spin_strength.setDecimals(2)
        self.spin_strength.setValue(saved_strength)

        self._syncing_strength = False

        def _slider_to_spin(v: int):
            if self._syncing_strength:
                return
            self._syncing_strength = True
            self.spin_strength.setValue(v / 100.0)
            self._syncing_strength = False

        def _spin_to_slider(v: float):
            if self._syncing_strength:
                return
            self._syncing_strength = True
            self.sld_strength.setValue(int(round(v * 100.0)))
            self._syncing_strength = False

        self.sld_strength.valueChanged.connect(_slider_to_spin)
        self.spin_strength.valueChanged.connect(_spin_to_slider)

        strength_row = QWidget(self)
        strength_lay = QHBoxLayout(strength_row)
        strength_lay.setContentsMargins(0, 0, 0, 0)
        strength_lay.setSpacing(8)
        strength_lay.addWidget(self.sld_strength, 1)
        strength_lay.addWidget(self.spin_strength)

        form.addRow("Strength:", strength_row)

        self.chk_mtf.toggled.connect(lambda on: self.spin_mtf_median.setEnabled(bool(on)))
        self.spin_mtf_median.setEnabled(self.chk_mtf.isChecked())
        self.chk_amp = QCheckBox("Use AMP (mixed precision)", self)
        self.chk_amp.setChecked(bool(self.settings.value("syqon/prism_use_amp", False, type=bool)))
        form.addRow("", self.chk_amp)

        lay.addWidget(box)
        self.pbar = QProgressBar(self)
        self.pbar.setRange(0, 100)
        self.pbar.setValue(0)
        self.pbar.setVisible(False)
        lay.addWidget(self.pbar)
        info = QLabel(
            "Prism Mini and Prism Deep use SyQon’s denoise models. "
            "Install the downloaded model file here after obtaining it from SyQon."
        )
        info.setWordWrap(True)
        lay.addWidget(info)
        lay.addStretch(1)
        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self._cancel_processing)
        lay.addWidget(self.btn_cancel)
        saved_kind = str(self.settings.value("syqon/prism_model_kind", "prism_mini"))
        idx = self.cmb_model.findData(saved_kind)
        if idx >= 0:
            self.cmb_model.setCurrentIndex(idx)

        self._refresh_state()

    def _toggle_live_preview_window(self, on: bool):
            on = bool(on)
            if not on:
                try:
                    if self.preview_win is not None:
                        self.preview_win.hide()
                except Exception:
                    pass
                return

            if self.preview_win is None:
                self.preview_win = SyQonLivePreviewWindow(
                    parent=self,
                    title="SyQon Prism Live Preview",
                    icon=self.window().windowIcon(),
                )
                self.preview_win.finished.connect(
                    lambda *_: self.chk_live_preview.setChecked(False)
                )

            self.preview_win.show()
            self.preview_win.raise_()
            self.preview_win.activateWindow()

    def _on_worker_preview(self, qimg):
        try:
            if not self.chk_live_preview.isChecked():
                return
            if qimg is None:
                return
            if self.preview_win is None:
                self._toggle_live_preview_window(True)
            if self.preview_win is not None and self.preview_win.isVisible():
                self.preview_win.set_frame(qimg)
        except Exception:
            pass

    def _cancel_processing(self):
        thr = getattr(self, "proc_thr", None)
        if thr is not None and thr.isRunning():
            thr.cancel()
            self.btn_cancel.setEnabled(False)
            self.lbl_status.setText("Cancelling… finishing current tile.")

    def model_kind(self) -> str:
        return str(self.cmb_model.currentData() or "prism_mini")

    def _model_dst_path(self):
        return syqon_prism_model_path(self.model_kind())

    def _have_model(self) -> bool:
        try:
            return self._installed_model_path() is not None
        except Exception:
            return False


    def _on_model_changed(self, *_):
        self.settings.setValue("syqon/prism_model_kind", self.model_kind())
        self._refresh_state()

    def _refresh_state(self):
        mk = self.model_kind()
        installed_path = self._installed_model_path()

        src_path = str(self.settings.value(self._src_model_settings_key(), "", type=str) or "")

        if installed_path is not None:
            self.lbl_model_path.setText(str(installed_path))
        else:
            self.lbl_model_path.setText("Not installed")
            self.settings.remove(self._installed_model_settings_key())

        if mk == "prism_deep":
            expected = "the Prism Deep model file named prism_sas395.pt"
        else:
            expected = "the Prism Mini model file"

        if installed_path is not None:
            msg = "Ready (Prism model installed)."
            if src_path:
                msg += f"\nLast installed from: {src_path}"
            self.lbl_status.setText(msg)
            self.btn_remove.setEnabled(True)
        else:
            self.lbl_status.setText(
                "Prism model is not installed.\n\n"
                "1) Click 'Get Model Here...' to obtain it from SyQon.\n"
                "2) Click 'Install Downloaded Model...' and select:\n"
                f"   {expected}"
            )
            self.btn_remove.setEnabled(False)

        self.btn_buy.setEnabled(True)
        self.btn_install.setEnabled(True)


    def _model_base_dir(self) -> Path:
        return syqon_prism_model_path(self.model_kind()).parent

    def _installed_model_settings_key(self) -> str:
        return f"syqon/prism_model_installed_path/{self.model_kind()}"

    def _src_model_settings_key(self) -> str:
        return f"syqon/prism_model_src_path/{self.model_kind()}"

    def _installed_model_path(self) -> Path | None:
        # 1) prefer remembered installed path
        p = str(self.settings.value(self._installed_model_settings_key(), "", type=str) or "").strip()
        if p:
            pp = Path(p)
            if pp.exists() and pp.is_file():
                return pp

        # 2) fallback for older installs that used canonical path
        legacy = self._model_dst_path()
        if legacy.exists() and legacy.is_file():
            return legacy

        return None


    def _open_buy_page(self):
        url = _syqon_prism_buy_url_for(self.model_kind())
        if not url:
            QMessageBox.information(
                self,
                "SyQon Prism",
                "The purchase/download URL is not configured yet."
            )
            return
        QDesktopServices.openUrl(QUrl(url))

    def _install_model(self):
        mk = self.model_kind()

        if mk == "prism_deep":
            expected_desc = "Prism Deep model file"
            required_name = "prism_sas395.pt"
        else:
            expected_desc = "Prism Mini model file"
            required_name = None

        src_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {expected_desc}",
            "",
            "PyTorch model (*.pt);;All Files (*)"
        )
        if not src_path:
            return

        src = Path(src_path)
        if not src.exists():
            QMessageBox.warning(self, "SyQon Prism", "Selected file does not exist.")
            return

        # Prism Deep: only accept the exact required filename
        if required_name is not None and src.name != required_name:
            QMessageBox.warning(
                self,
                "SyQon Prism",
                f"Invalid Prism Deep model selected.\n\n"
                f"Expected filename:\n{required_name}\n\n"
                f"Selected filename:\n{src.name}"
            )
            return

        base_dir = self._model_base_dir()
        base_dir.mkdir(parents=True, exist_ok=True)
        dst = base_dir / src.name


        try:
            import shutil
            shutil.copy2(str(src), str(dst))
        except Exception as e:
            QMessageBox.critical(self, "SyQon Prism", f"Failed to install model:\n{e}")
            self._refresh_state()
            return

        self.settings.setValue(self._src_model_settings_key(), str(src))
        self.settings.setValue(self._installed_model_settings_key(), str(dst))

        self._refresh_state()

    def _remove_model(self):
        mk = self.model_kind()
        dst = self._installed_model_path()
        if dst is None:
            self.settings.remove(self._src_model_settings_key())
            self.settings.remove(self._installed_model_settings_key())
            self._refresh_state()
            return


        reply = QMessageBox.question(
            self,
            "Remove Model",
            "Remove the installed Prism model from disk?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            dst.unlink(missing_ok=True)
        except Exception:
            try:
                if dst.exists():
                    os.remove(str(dst))
            except Exception:
                pass

        self.settings.remove(self._src_model_settings_key())
        self.settings.remove(self._installed_model_settings_key())

        self._refresh_state()
            
    
    def _set_busy(self, busy: bool):
        self.btn_buy.setEnabled(not busy)
        self.btn_install.setEnabled(not busy)
        self.btn_remove.setEnabled((not busy) and self._have_model())
        self.btn_clear_cache.setEnabled(not busy)

        self.cmb_model.setEnabled(not busy)
        self.spin_tile.setEnabled(not busy)
        self.spin_overlap.setEnabled(not busy)
        self.spin_pad.setEnabled(not busy)
        self.chk_mtf.setEnabled(not busy)
        self.spin_strength.setEnabled(not busy)
        self.spin_mtf_median.setEnabled((not busy) and self.chk_mtf.isChecked())
        self.chk_amp.setEnabled(not busy)
        self.chk_live_preview.setEnabled((not busy))
        self.pbar.setVisible(busy)
        self.btn_cancel.setVisible(busy)
        self.btn_cancel.setEnabled(busy)
        if busy:
            self.pbar.setRange(0, 100)
            self.pbar.setValue(0)


    def _on_worker_progress(self, pct: int, stage: str):
        self.pbar.setValue(int(pct))
        if stage:
            self.lbl_status.setText(stage)


    def _on_worker_finished(self, denoised_s, info: dict, err: str):
        # If proc_thr was cleared by closeEvent, we're already shutting down
        if getattr(self, "proc_thr", None) is None and err == "__cancelled__":
            return

        if err == "__cancelled__":
            self._set_busy(False)
            self.lbl_status.setText("Cancelled.")
            self.proc_thr = None
            self._refresh_state()
            return
        try:
            if self.preview_win is not None:
                self.preview_win.hide()
        except Exception:
            pass
        if err:
            self._set_busy(False)
            self.proc_thr = None
            QMessageBox.critical(self, "SyQon Prism", err)
            self._refresh_state()
            return

        try:
            if hasattr(self.main, "_log"):
                self.main._log(
                    f"SyQon Prism backend: device={info.get('device')} "
                    f"torch={info.get('torch_version')} file={info.get('torch_file')}"
                )
        except Exception:
            pass

        scale_factor = self._scale_factor
        orig_was_mono = self._orig_was_mono
        do_mtf = bool(self._do_mtf)
        mtf_params = self._mtf_params

        if denoised_s.ndim == 2:
            denoised_s = np.stack([denoised_s] * 3, axis=-1)

        denoised_lin = denoised_s
        if do_mtf and mtf_params is not None:
            denoised_lin = _invert_mtf_unlinked_rgb(denoised_s, mtf_params)

        if scale_factor > 1.01:
            denoised_lin = np.clip(denoised_lin * scale_factor, 0.0, 1.0).astype(np.float32, copy=False)

        strength = float(np.clip(self.spin_strength.value(), 0.0, 1.0))

        orig = np.asarray(self.doc.image).astype(np.float32, copy=False)
        orig = np.nan_to_num(orig, nan=0.0, posinf=0.0, neginf=0.0)

        if orig_was_mono:
            if orig.ndim == 3:
                if orig.shape[2] == 1:
                    orig_base = orig[..., 0]
                else:
                    orig_base = orig.mean(axis=2)
            else:
                orig_base = orig

            den_base = denoised_lin.mean(axis=2).astype(np.float32, copy=False)
            final_to_apply = ((1.0 - strength) * orig_base + strength * den_base).astype(np.float32, copy=False)
            orig_for_mask = orig_base
        else:
            if orig.ndim == 2:
                orig_rgb = np.stack([orig] * 3, axis=-1)
            elif orig.ndim == 3 and orig.shape[2] == 1:
                orig_rgb = np.repeat(orig, 3, axis=2)
            else:
                orig_rgb = orig[..., :3]

            final_to_apply = ((1.0 - strength) * orig_rgb + strength * denoised_lin).astype(np.float32, copy=False)
            orig_for_mask = orig_rgb

        final_to_apply = np.clip(final_to_apply, 0.0, 1.0).astype(np.float32, copy=False)
        orig_for_mask = orig_base if orig_was_mono else orig_rgb
        final_to_apply = _blend_result_with_mask(final_to_apply, orig_for_mask, self.doc)

        meta = {
            "step_name": "Denoised",
            "bit_depth": "32-bit floating point",
            "is_mono": bool(orig_was_mono),
            "masked": bool(getattr(self.doc, "active_mask_id", None)),
            "mask_id": getattr(self.doc, "active_mask_id", None) or None,
            "mask_blend": "m*out+(1-m)*src",            
            "replay_last": {
                "op": "syqon_prism",
                "params": {
                    "model_kind": self.model_kind(),
                    "tile_size": int(self.spin_tile.value()),
                    "overlap": int(self.spin_overlap.value()),
                    "pad": int(self.spin_pad.value()),
                    "strength": float(self.spin_strength.value()),
                    "model_path": str(self._model_dst_path()),
                    "use_mtf": bool(do_mtf),
                    "mtf_target_median": float(self.spin_mtf_median.value()),
                    "use_amp": bool(self.chk_amp.isChecked()),
                    "label": "Denoise (SyQon Prism)",
                }
            }
        }

        self.doc.apply_edit(final_to_apply, metadata=meta, step_name="Denoised")

        try:
            self.main._last_headless_command = {
                "command_id": "syqon_prism",
                "preset": {
                    "model_kind": self.model_kind(),
                    "tile_size": int(self.spin_tile.value()),
                    "overlap": int(self.spin_overlap.value()),
                    "pad": int(self.spin_pad.value()),
                    "strength": float(self.spin_strength.value()),
                    "model_path": str(self._model_dst_path()),
                    "use_mtf": bool(do_mtf),
                    "mtf_target_median": float(self.spin_mtf_median.value()),
                    "use_amp": bool(self.chk_amp.isChecked()),
                },
            }
        except Exception:
            pass

        self.proc_thr = None
        self._set_busy(False)
        self.lbl_status.setText("Complete!")
        self._refresh_state()


    def process_document(self, doc, main):
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.warning(self, "SyQon Prism", "No active image.")
            return

        if not self._have_model():
            QMessageBox.warning(self, "SyQon Prism", "Model is not installed. Install it first.")
            return

        self.doc = doc
        self.main = main

        self.settings.setValue("syqon/prism_tile_size", int(self.spin_tile.value()))
        self.settings.setValue("syqon/prism_overlap", int(self.spin_overlap.value()))
        self.settings.setValue("syqon/prism_use_amp", bool(self.chk_amp.isChecked()))
        self.settings.setValue("syqon/prism_use_mtf", bool(self.chk_mtf.isChecked()))
        self.settings.setValue("syqon/prism_pad", int(self.spin_pad.value()))
        self.settings.setValue("syqon/prism_strength", float(self.spin_strength.value()))
        self.settings.setValue("syqon/prism_mtf_target_median", float(self.spin_mtf_median.value()))
        self.settings.setValue("syqon/prism_model_kind", self.model_kind())

        src = np.asarray(self.doc.image).astype(np.float32, copy=False)
        orig_was_mono = (src.ndim == 2) or (src.ndim == 3 and src.shape[2] == 1)

        x = np.nan_to_num(src, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        scale_factor = float(np.max(x)) if x.size else 1.0
        if scale_factor > 1.01:
            x01 = np.clip(x / scale_factor, 0.0, 1.0)
        else:
            x01 = np.clip(x, 0.0, 1.0)

        if x01.ndim == 2:
            xrgb = np.stack([x01] * 3, axis=-1)
        elif x01.ndim == 3 and x01.shape[2] == 1:
            xrgb = np.repeat(x01, 3, axis=2)
        else:
            xrgb = x01[..., :3]

        do_mtf = bool(self.chk_mtf.isChecked())
        if do_mtf:
            mtf_target = float(self.spin_mtf_median.value())
            mtf_params = _mtf_params_unlinked(xrgb, shadows_clipping=-2.8, targetbg=mtf_target)
            x_for_net = _apply_mtf_unlinked_rgb(xrgb, mtf_params)
        else:
            mtf_params = None
            x_for_net = xrgb

        self._scale_factor = scale_factor
        self._orig_was_mono = orig_was_mono
        self._do_mtf = do_mtf
        self._mtf_params = mtf_params

        installed = self._installed_model_path()
        if installed is None:
            QMessageBox.warning(self, "SyQon Prism", "Model is not installed. Install it first.")
            self._set_busy(False)
            self._refresh_state()
            return

        ckpt_path = str(installed)


        self._set_busy(True)
        self.lbl_status.setText("Processing...")

        try:
            if self.proc_thr is not None and self.proc_thr.isRunning():
                self.proc_thr.cancel()
                self.proc_thr.wait(200)
        except Exception:
            pass

        self.proc_thr = _SyQonPrismProcessThread(
            x_for_net=x_for_net,
            ckpt_path=ckpt_path,
            model_kind=self.model_kind(),
            tile=int(self.spin_tile.value()),
            overlap=int(self.spin_overlap.value()),
            pad=int(self.spin_pad.value()),
            use_amp=bool(self.chk_amp.isChecked()),
            amp_dtype="fp16",
            live_preview=bool(self.chk_live_preview.isChecked()),           # ← ADD
            preview_src_rgb01=(x_for_net if self.chk_live_preview.isChecked() else None),  # ← ADD
            preview_max_dim=900,                                             # ← ADD
            preview_emit_ms=120,                                             # ← ADD
            parent=self,
        )
        self.proc_thr.progress.connect(self._on_worker_progress)
        self.proc_thr.preview.connect(self._on_worker_preview)              # ← ADD
        self.proc_thr.finished.connect(self._on_worker_finished)
        self.proc_thr.start()

    def _clear_ai_cache(self):
        errors = []

        try:
            clear_prism_models_cache(aggressive=True, status_cb=print)
        except Exception as e:
            errors.append(f"Prism: {type(e).__name__}: {e}")

        try:
            clear_axiom_models_cache(aggressive=True, status_cb=print)
        except Exception as e:
            errors.append(f"Axiom: {type(e).__name__}: {e}")

        if errors:
            QMessageBox.warning(
                self,
                "SyQon Tools",
                "AI cache clear completed with some issues:\n\n" + "\n".join(errors)
            )
            return

        QMessageBox.information(
            self,
            "SyQon Tools",
            "AI cache cleared.\n\nNext Prism or Axiom run will take longer because models must reload."
        )

    def closeEvent(self, ev):
        try:
            thr = getattr(self, "proc_thr", None)
            if thr is not None and thr.isRunning():
                thr.cancel()
                if not thr.wait(10000):
                    thr.terminate()
                    thr.wait(2000)
                self.proc_thr = None
        except Exception:
            pass
        super().closeEvent(ev)


_SYQON_PARALLAX_BUY_URL = "https://syqon.it/parallax"
 

class _SyQonSharpenHubPage(QWidget):
 
    def __init__(self, parent=None):
        super().__init__(parent)
 
        self.proc_thr       = None
        self.doc            = None
        self.main           = None
        self._orig_was_mono = False
        self._scale_factor  = 1.0
        self._do_mtf        = False
        self._mtf_params    = None
        self._mtf_linked    = False
        self.settings       = QSettings()
 
        lay = QVBoxLayout(self)
 
        # ── Logo ──
        self.lbl_logo = QLabel(self)
        self.lbl_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        try:
            from setiastro.saspro.resources import get_icons
            pm = QPixmap(get_icons().SYQON_PARALLAX)
            if not pm.isNull():
                self.lbl_logo.setPixmap(pm.scaledToWidth(260, Qt.TransformationMode.SmoothTransformation))
                lay.addWidget(self.lbl_logo)
        except Exception:
            pass
 
        # ── Single group box — mirrors Prism's "SyQon Prism Denoise" box ──
        box  = QGroupBox("SyQon Parallax", self)
        form = QFormLayout(box)
 
        # Mode selector — chooses which model triplet to use for processing
        self.cmb_mode = QComboBox(self)
        self.cmb_mode.addItem("Classic",    userData="classic")
        self.cmb_mode.addItem("Aesthetics", userData="aesthetics")
        saved_mode = str(self.settings.value("syqon/parallax_mode", "classic", type=str) or "classic").lower()
        idx = self.cmb_mode.findData(saved_mode)
        if idx >= 0:
            self.cmb_mode.setCurrentIndex(idx)
        self.cmb_mode.currentIndexChanged.connect(
            lambda *_: self.settings.setValue("syqon/parallax_mode", self.mode())
        )
        form.addRow("Mode:", self.cmb_mode)

        # Six install rows in a 2-column grid: Classic left, Aesthetics right
        from PyQt6.QtWidgets import QGridLayout

        models_wrap = QWidget(self)
        grid = QGridLayout(models_wrap)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(4)

        def _hdr(text):
            l = QLabel(text, self)
            l.setStyleSheet("color:#8e8e93; font-size:10px; font-weight:bold;")
            return l

        def _build_row(attr_lbl, row_label, setting_key, canonical_name):
            saved = str(self.settings.value(setting_key, "", type=str) or "")
            installed = bool(saved and Path(saved).exists())

            status_lbl = QLabel("✔  " + row_label if installed else "✘  " + row_label, self)
            status_lbl.setStyleSheet(
                "color: #2ed573; font-weight: bold;" if installed else "color: #ff6b6b; font-weight: bold;"
            )
            status_lbl.setToolTip(saved if installed else "Not installed")
            setattr(self, attr_lbl, status_lbl)

            btn_i = QPushButton("Install…", self)
            btn_r = QPushButton("Remove", self)
            btn_i.setFixedWidth(72)
            btn_r.setFixedWidth(62)
            btn_i.clicked.connect(self._make_install_fn(setting_key, status_lbl, canonical_name))
            btn_r.clicked.connect(self._make_remove_fn(setting_key, status_lbl, row_label))

            row_w = QWidget(self)
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(6)
            row_l.addWidget(status_lbl, 1)
            row_l.addWidget(btn_i)
            row_l.addWidget(btn_r)
            return row_w

        classic_rows = [
            ("_correction_lbl",       "Correction",     "syqon/parallax_correction_path",             "parallax_correction.pth"),
            ("_star_reduce_lbl",      "Star reduction", "syqon/parallax_star_reduce_path",            "parallax_star_reduction.pth"),
            ("_sharpen_lbl",          "Sharpen",        "syqon/parallax_sharpen_path",                "parallax_sharpen.pth"),
        ]
        aesth_rows = [
            ("_correction_aesth_lbl", "Correction",     "syqon/parallax_correction_aesthetics_path",  "parallax_correction_aesthetics.pth"),
            ("_star_reduce_aesth_lbl","Star reduction", "syqon/parallax_star_reduce_aesthetics_path", "parallax_star_reduction_aesthetics.pth"),
            ("_sharpen_aesth_lbl",    "Sharpen",        "syqon/parallax_sharpen_aesthetics_path",     "parallax_sharpen_aesthetics.pth"),
        ]

        grid.addWidget(_hdr("Classic models"),    0, 0)
        grid.addWidget(_hdr("Aesthetics models"), 0, 1)
        for i, (c, a) in enumerate(zip(classic_rows, aesth_rows), start=1):
            grid.addWidget(_build_row(*c), i, 0)
            grid.addWidget(_build_row(*a), i, 1)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        form.addRow(models_wrap)
 
        # Get Models Here + Clear AI Cache row
        ar = QHBoxLayout()
        self.btn_buy = QPushButton("Get Models Here…", self)
        self.btn_buy.clicked.connect(self._open_buy_page)
        self.btn_clear_cache = QPushButton("Clear AI Cache", self)
        self.btn_clear_cache.clicked.connect(self._clear_ai_cache)
        ar.addWidget(self.btn_buy)
        ar.addWidget(self.btn_clear_cache)
        ar.addStretch(1)
        aw = QWidget(self); aw.setLayout(ar)
        form.addRow("", aw)
 
        # Status label
        self.lbl_status = QLabel("", self)
        self.lbl_status.setWordWrap(True)
        form.addRow("", self.lbl_status)
 
        # Processing stages — compact single row for correction + star reduction
        stages_row = QWidget(self)
        stages_lay = QHBoxLayout(stages_row)
        stages_lay.setContentsMargins(0, 0, 0, 0)
        stages_lay.setSpacing(16)
 
        self.chk_correction = QCheckBox("Aberration correction", self)
        self.chk_correction.setChecked(bool(self.settings.value("syqon/parallax_correct", True, type=bool)))
        stages_lay.addWidget(self.chk_correction)
 
        star_lbl = QLabel("Star reduction (0=off, 1–10):", self)
        self.spin_star = QSpinBox(self)
        self.spin_star.setRange(0, 10)
        self.spin_star.setValue(int(self.settings.value("syqon/parallax_star_level", 3)))
        self.spin_star.setFixedWidth(60)
        self.spin_star.setToolTip(
            "0 = disabled\n"
            "1–6 = direct model strength\n"
            "7–10 = hybrid neural-morphological star reduction (model run at "
            "safe level 5, plus escalating morphological erosion of flagged "
            "star cores; 10 = stars nearly removed)"
        )
        stages_lay.addWidget(star_lbl)
        stages_lay.addWidget(self.spin_star)
        stages_lay.addStretch(1)
 
        form.addRow("Stages:", stages_row)
 
        # Sharpen strength — slider + spinbox on same row like Prism's Strength
        saved_sharpen = float(self.settings.value("syqon/parallax_sharpen_alpha", 0.7))
 
        self.sld_sharpen = QSlider(Qt.Orientation.Horizontal, self)
        self.sld_sharpen.setRange(0, 200)
        self.sld_sharpen.setValue(int(round(saved_sharpen * 100.0)))

        self.spin_sharpen = QDoubleSpinBox(self)
        self.spin_sharpen.setRange(0.0, 2.0)
        self.spin_sharpen.setSingleStep(0.01)
        self.spin_sharpen.setDecimals(2)
        self.spin_sharpen.setValue(saved_sharpen)
        self.spin_sharpen.setToolTip(
            "0% = off, 100% = full model-strength sharpen, 200% = double "
            "strength.\n"
            "Above 100% extrapolates beyond the model's output (more "
            "aggressive; higher risk of clipping/halos in highlights and "
            "shadows — result is still clamped to [0,1])."
        )
        self.spin_sharpen.setFixedWidth(62)
 
        self._syncing_sharpen = False
 
        def _slider_to_spin(v):
            if self._syncing_sharpen: return
            self._syncing_sharpen = True
            self.spin_sharpen.setValue(v / 100.0)
            self._syncing_sharpen = False
 
        def _spin_to_slider(v):
            if self._syncing_sharpen: return
            self._syncing_sharpen = True
            self.sld_sharpen.setValue(int(round(v * 100.0)))
            self._syncing_sharpen = False
 
        self.sld_sharpen.valueChanged.connect(_slider_to_spin)
        self.spin_sharpen.valueChanged.connect(_spin_to_slider)
 
        sharpen_row = QWidget(self)
        sharpen_lay = QHBoxLayout(sharpen_row)
        sharpen_lay.setContentsMargins(0, 0, 0, 0)
        sharpen_lay.setSpacing(8)
        sharpen_lay.addWidget(self.sld_sharpen, 1)
        sharpen_lay.addWidget(self.spin_sharpen)
 
        form.addRow("Sharpen strength:", sharpen_row)
 
        # Tiling
        self.spin_tile = QSpinBox(self)
        self.spin_tile.setRange(128, 2048)
        self.spin_tile.setValue(int(self.settings.value("syqon/parallax_tile", 512)))
        form.addRow("Tile size:", self.spin_tile)
 
        self.spin_overlap = QSpinBox(self)
        self.spin_overlap.setRange(8, 512)
        self.spin_overlap.setValue(int(self.settings.value("syqon/parallax_overlap", 64)))
        form.addRow("Overlap:", self.spin_overlap)
 
        self.spin_pad = QSpinBox(self)
        self.spin_pad.setRange(0, 2048)
        self.spin_pad.setValue(int(self.settings.value("syqon/parallax_pad", 96)))
        form.addRow("Edge pad (px):", self.spin_pad)
 
        pad_help = QLabel("Pads edges before tiling (reflect) to prevent border artifacts.\nTypical: overlap or overlap+32. 0 disables padding.")
        pad_help.setWordWrap(True)
        pad_help.setStyleSheet("color:#888;")
        form.addRow("", pad_help)
 
        # Stretch
        self.chk_mtf = QCheckBox("Apply temporary stretch for model (recommended)", self)
        self.chk_mtf.setChecked(bool(self.settings.value("syqon/parallax_use_mtf", True, type=bool)))
        form.addRow("", self.chk_mtf)
 
        self.spin_mtf = QDoubleSpinBox(self)
        self.spin_mtf.setRange(0.01, 0.50)
        self.spin_mtf.setSingleStep(0.01)
        self.spin_mtf.setDecimals(3)
        self.spin_mtf.setValue(float(self.settings.value("syqon/parallax_mtf_target", 0.15)))
        form.addRow("Temp stretch target median:", self.spin_mtf)
 
        self.chk_mtf_linked = QCheckBox("Linked stretch (preserves star colors — use if already color calibrated)", self)
        self.chk_mtf_linked.setChecked(bool(self.settings.value("syqon/parallax_mtf_linked", False, type=bool)))
        self.chk_mtf_linked.setToolTip(
            "Unlinked stretch applies independent per-channel blackpoint which can shift star colors.\n"
            "Linked stretch uses one combined stretch for all channels, preserving star colors."
        )
        form.addRow("", self.chk_mtf_linked)
 
        self.chk_mtf.toggled.connect(lambda on: (self.spin_mtf.setEnabled(bool(on)), self.chk_mtf_linked.setEnabled(bool(on))))
        self.spin_mtf.setEnabled(self.chk_mtf.isChecked())
        self.chk_mtf_linked.setEnabled(self.chk_mtf.isChecked())
 
        lay.addWidget(box)
 
        # Progress bar
        self.pbar = QProgressBar(self)
        self.pbar.setRange(0, 100)
        self.pbar.setVisible(False)
        lay.addWidget(self.pbar)
 
        lay.addStretch(1)
 
        # Cancel button
        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self._cancel)
        lay.addWidget(self.btn_cancel)
 
    def mode(self) -> str:
        try:
            return str(self.cmb_mode.currentData() or "classic").lower()
        except Exception:
            return "classic"

    def _key_for(self, variant: str) -> str:
        base = {
            "correction":  "syqon/parallax_correction_path",
            "star_reduce": "syqon/parallax_star_reduce_path",
            "sharpen":     "syqon/parallax_sharpen_path",
        }[variant]
        if self.mode() == "aesthetics":
            return base.replace("_path", "_aesthetics_path")
        return base

    # ------------------------------------------------------------------
    # Install / remove
    # ------------------------------------------------------------------
 
    def _make_install_fn(self, setting_key: str, status_lbl: QLabel, canonical_name: str):
        # Derive display name from canonical_name for the label
        names = {
            "parallax_correction.pth":    "Correction model",
            "parallax_star_reduction.pth":"Star reduction model",
            "parallax_sharpen.pth":       "Sharpen model",
        }
        display = names.get(canonical_name, canonical_name)
 
        def _fn():
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Parallax Model File", "",
                "PyTorch model (*.pth *.pt);;All Files (*)"
            )
            if not path:
                return
            src = Path(path)
            if not src.exists():
                QMessageBox.warning(self, "Parallax", "Selected file does not exist.")
                return
            from setiastro.saspro.syqon_paths import syqon_family_dir
            dst = syqon_family_dir("sharpen") / canonical_name
            try:
                import shutil
                shutil.copy2(str(src), str(dst))
            except Exception as e:
                QMessageBox.critical(self, "Parallax", f"Failed to install model:\n{e}")
                return
            self.settings.setValue(setting_key, str(dst))
            status_lbl.setText("✔  " + display)
            status_lbl.setStyleSheet("color: #2ed573; font-weight: bold;")
            status_lbl.setToolTip(str(dst))
            self.lbl_status.setText(display + " installed.")
        return _fn
 
    def _make_remove_fn(self, setting_key: str, status_lbl: QLabel, display: str):
        def _fn():
            p = str(self.settings.value(setting_key, "", type=str) or "")
            if not p:
                return
            reply = QMessageBox.question(
                self, "Remove Model", f"Remove the installed {display} from disk?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
            self.settings.remove(setting_key)
            status_lbl.setText("✘  " + display)
            status_lbl.setStyleSheet("color: #ff6b6b; font-weight: bold;")
            status_lbl.setToolTip("Not installed")
            self.lbl_status.setText(display + " removed.")
        return _fn
 
    def _open_buy_page(self):
        QDesktopServices.openUrl(QUrl(_SYQON_PARALLAX_BUY_URL))
 
    def _clear_ai_cache(self):
        try:
            from setiastro.saspro.sharpen_engines.syqon_parallax_engine import clear_parallax_models_cache
            clear_parallax_models_cache(aggressive=True, status_cb=print)
        except Exception as e:
            QMessageBox.warning(self, "Parallax", f"Cache clear issue:\n{e}")
            return
        QMessageBox.information(self, "Parallax", "AI cache cleared.")
 
    def _model_path(self, key: str) -> str | None:
        p = str(self.settings.value(key, "", type=str) or "")
        return p if (p and Path(p).exists()) else None
 
    def correction_model_path(self)  -> str | None: return self._model_path(self._key_for("correction"))
    def star_reduce_model_path(self) -> str | None: return self._model_path(self._key_for("star_reduce"))
    def sharpen_model_path(self)     -> str | None: return self._model_path(self._key_for("sharpen"))
 
    # ------------------------------------------------------------------
    # Busy state
    # ------------------------------------------------------------------
 
    def _set_busy(self, busy: bool):
        for w in (self.btn_buy, self.btn_clear_cache, self.spin_tile, self.spin_overlap,
                  self.spin_pad, self.chk_correction, self.spin_star, self.spin_sharpen,
                  self.chk_mtf):
            w.setEnabled(not busy)
        self.spin_mtf.setEnabled((not busy) and self.chk_mtf.isChecked())
        self.pbar.setVisible(busy)
        self.btn_cancel.setVisible(busy)
        self.btn_cancel.setEnabled(busy)
        if busy:
            self.pbar.setValue(0)
 
    def _cancel(self):
        thr = getattr(self, "proc_thr", None)
        if thr is not None and thr.isRunning():
            thr.cancel()
            self.btn_cancel.setEnabled(False)
            self.lbl_status.setText("Cancelling…")
 
    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
 
    def _on_progress(self, pct: int, stage: str):
        self.pbar.setValue(int(pct))
        if stage:
            self.lbl_status.setText(stage)
 
    def _on_finished(self, result_s, info: dict, err: str):
        if err == "__cancelled__":
            self._set_busy(False)
            self.lbl_status.setText("Cancelled.")
            self.proc_thr = None
            return
        if err:
            self._set_busy(False)
            self.proc_thr = None
            QMessageBox.critical(self, "SyQon Parallax", err)
            return
        try:
            from setiastro.saspro.remove_stars import _invert_mtf_unlinked_rgb
 
            result = np.asarray(result_s, dtype=np.float32)
            if result.ndim == 2:
                result = np.stack([result] * 3, axis=-1)
 
            if self._do_mtf and self._mtf_params is not None:
                from setiastro.saspro.remove_stars import (
                    _invert_mtf_unlinked_rgb, _invert_mtf_linked_rgb,
                )
                if getattr(self, "_mtf_linked", False):
                    result = _invert_mtf_linked_rgb(result, self._mtf_params)
                else:
                    result = _invert_mtf_unlinked_rgb(result, self._mtf_params)
 
            if self._scale_factor > 1.01:
                result = np.clip(result * self._scale_factor, 0.0, 1.0).astype(np.float32)
 
            if self._orig_was_mono:
                final = result.mean(axis=2).astype(np.float32)
            else:
                final = result[..., :3]
 
            final = np.clip(final, 0.0, 1.0).astype(np.float32)
 
            meta = {
                "step_name": "Parallax",
                "bit_depth": "32-bit floating point",
                "is_mono":   bool(self._orig_was_mono),
                "replay_last": {
                    "op": "syqon_parallax",
                    "params": {
                        "correct":       bool(self.chk_correction.isChecked()),
                        "star_level":    int(self.spin_star.value()),
                        "sharpen_alpha": float(self.spin_sharpen.value()),
                        "tile":          int(self.spin_tile.value()),
                        "overlap":       int(self.spin_overlap.value()),
                        "pad":           int(self.spin_pad.value()),
                        "use_mtf":       bool(self.chk_mtf.isChecked()),
                        "mtf_target":    float(self.spin_mtf.value()),
                    }
                }
            }
            self.doc.apply_edit(final, metadata=meta, step_name="Parallax")
            self.lbl_status.setText("Complete!")
        except Exception as e:
            QMessageBox.critical(self, "SyQon Parallax", f"Failed to apply result:\n{e}")
        finally:
            self._set_busy(False)
            self.proc_thr = None
 
    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
 
    def process_document(self, doc, main):
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.warning(self, "SyQon Parallax", "No active image.")
            return
 
        do_correct    = bool(self.chk_correction.isChecked())
        star_level    = int(self.spin_star.value())
        sharpen_alpha = float(self.spin_sharpen.value())
 
        if not do_correct and star_level == 0 and sharpen_alpha <= 0.0:
            QMessageBox.warning(self, "SyQon Parallax", "Enable at least one processing stage.")
            return
        if do_correct and self.correction_model_path() is None:
            QMessageBox.warning(self, "SyQon Parallax", "Correction model not installed.")
            return
        if star_level > 0 and self.star_reduce_model_path() is None:
            QMessageBox.warning(self, "SyQon Parallax", "Star reduction model not installed.")
            return
        if sharpen_alpha > 0.0 and self.sharpen_model_path() is None:
            QMessageBox.warning(self, "SyQon Parallax", "Sharpen model not installed.")
            return
 
        self.doc  = doc
        self.main = main
 
        self.settings.setValue("syqon/parallax_correct",       do_correct)
        self.settings.setValue("syqon/parallax_star_level",    star_level)
        self.settings.setValue("syqon/parallax_sharpen_alpha", sharpen_alpha)
        self.settings.setValue("syqon/parallax_tile",          int(self.spin_tile.value()))
        self.settings.setValue("syqon/parallax_overlap",       int(self.spin_overlap.value()))
        self.settings.setValue("syqon/parallax_pad",           int(self.spin_pad.value()))
        self.settings.setValue("syqon/parallax_use_mtf",       bool(self.chk_mtf.isChecked()))
        self.settings.setValue("syqon/parallax_mtf_target",    float(self.spin_mtf.value()))
        mode = self.mode()
        self.settings.setValue("syqon/parallax_mode", mode)        
 
        src = np.asarray(doc.image, dtype=np.float32)
        self._orig_was_mono = (src.ndim == 2) or (src.ndim == 3 and src.shape[2] == 1)
        x = np.nan_to_num(src, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self._scale_factor = float(np.max(x)) if x.size else 1.0
        x01 = np.clip(x / self._scale_factor if self._scale_factor > 1.01 else x, 0.0, 1.0)
 
        if x01.ndim == 2:
            xrgb = np.stack([x01] * 3, axis=-1)
        elif x01.ndim == 3 and x01.shape[2] == 1:
            xrgb = np.repeat(x01, 3, axis=2)
        else:
            xrgb = x01[..., :3]
 
        if bool(self.chk_mtf.isChecked()):
            from setiastro.saspro.remove_stars import (
                _mtf_params_unlinked, _apply_mtf_unlinked_rgb,
                _mtf_params_linked,   _apply_mtf_linked_rgb,
            )
            mtf_target  = float(self.spin_mtf.value())
            use_linked  = bool(self.chk_mtf_linked.isChecked())
            self.settings.setValue("syqon/parallax_mtf_linked", use_linked)
            if use_linked:
                self._mtf_params = _mtf_params_linked(xrgb, targetbg=mtf_target)
                x_for_net        = _apply_mtf_linked_rgb(xrgb, self._mtf_params)
            else:
                self._mtf_params = _mtf_params_unlinked(xrgb, targetbg=mtf_target)
                x_for_net        = _apply_mtf_unlinked_rgb(xrgb, self._mtf_params)
            self._mtf_linked = use_linked
            self._do_mtf     = True
        else:
            x_for_net        = xrgb
            self._do_mtf     = False
            self._mtf_params = None
            self._mtf_linked = False
 
        self._set_busy(True)
        self.lbl_status.setText("Processing…")
 
        if self.proc_thr is not None and self.proc_thr.isRunning():
            self.proc_thr.cancel()
            self.proc_thr.wait(200)
 
        self.proc_thr = _SyQonParallaxProcessThread(
            x_for_net       = x_for_net,
            do_correct      = do_correct,
            correction_path = self.correction_model_path() or "",
            star_level      = star_level,
            star_path       = self.star_reduce_model_path() or "",
            sharpen_alpha   = sharpen_alpha,
            sharpen_path    = self.sharpen_model_path() or "",
            tile            = int(self.spin_tile.value()),
            overlap         = int(self.spin_overlap.value()),
            pad             = int(self.spin_pad.value()),
            parent          = self,
            mode=mode,
        )
        self.proc_thr.progress.connect(self._on_progress)
        self.proc_thr.finished.connect(self._on_finished)
        self.proc_thr.start()
 
 
class _SyQonParallaxProcessThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, dict, str)
 
    def __init__(self, x_for_net, *, do_correct, correction_path, star_level,
                 star_path, sharpen_alpha, sharpen_path, tile, overlap, pad, parent=None, mode="classic"):
        super().__init__(parent)
        self.x_for_net       = np.asarray(x_for_net, dtype=np.float32)
        self.do_correct      = bool(do_correct)
        self.correction_path = str(correction_path)
        self.star_level      = int(np.clip(star_level, 0, 10))
        self.star_path       = str(star_path)
        self.sharpen_alpha   = float(np.clip(sharpen_alpha, 0.0, 2.0))
        self.sharpen_path    = str(sharpen_path)
        self.tile            = int(tile)
        self.overlap         = int(overlap)
        self.pad             = int(pad)
        self._cancel         = False
        self.mode = str(mode or "classic").lower()
 
    def cancel(self): self._cancel = True
 
    def run(self):
        from setiastro.saspro.sharpen_engines.syqon_parallax_engine import (
            parallax_correction_rgb01,
            parallax_star_reduce_rgb01,
            parallax_sharpen_rgb01,
        )
        info    = {}
        current = self.x_for_net.copy()
 
        try:
            n = ((1 if self.do_correct else 0) +
                 (1 if self.star_level > 0 else 0) +
                 (1 if self.sharpen_alpha > 0 else 0))
            s = 0
 
            def _cb(name):
                def _fn(done, total, detail):
                    if self._cancel:
                        raise InterruptedError("Cancelled")
                    self.progress.emit(int(100 * done / max(total, 1)),
                                       f"{name} — {detail}")
                return _fn
 
            if self.do_correct:
                s += 1
                current, inf = parallax_correction_rgb01(
                    current, self.correction_path,
                    tile=self.tile, overlap=self.overlap, pad=self.pad,
                    use_gpu=True, prefer_dml=True,
                    mode=self.mode,
                    progress_cb=_cb(f"[{s}/{n}] Aberration Correction"),
                )
                info["correction"] = inf
 
            if self.star_level > 0:
                s += 1
                current, inf = parallax_star_reduce_rgb01(
                    current, self.star_path, self.star_level,
                    tile=self.tile, overlap=self.overlap, pad=self.pad,
                    use_gpu=True, prefer_dml=True,
                    mode=self.mode,
                    progress_cb=_cb(f"[{s}/{n}] Star Reduction L{self.star_level}"),
                )
                info["star_reduce"] = inf
 
            if self.sharpen_alpha > 0.0:
                s += 1
                current, inf = parallax_sharpen_rgb01(
                    current, self.sharpen_path, self.sharpen_alpha,
                    tile=self.tile, overlap=self.overlap, pad=self.pad,
                    use_gpu=True, prefer_dml=True,
                    mode=self.mode,
                    progress_cb=_cb(f"[{s}/{n}] Sharpening α={self.sharpen_alpha:.2f}"),
                )
                info["sharpen"] = inf
 
            self.finished.emit(current, info, "")
 
        except InterruptedError:
            self.finished.emit(None, info, "__cancelled__")
        except Exception as e:
            import traceback
            info["traceback"] = traceback.format_exc()
            self.finished.emit(None, info, str(e))


class _SyQonPrismProcessThread(QThread):
    progress = pyqtSignal(int, str)          # percent, stage
    preview  = pyqtSignal(object)            # QImage or None  ← ADD
    finished = pyqtSignal(object, dict, str) # denoised, info, err

    def __init__(
        self,
        x_for_net: np.ndarray,
        ckpt_path: str,
        model_kind: str,
        tile: int,
        overlap: int,
        pad: int = 0,
        use_amp: bool = False,
        amp_dtype: str = "fp16",
        *,                                   # ← ADD everything below
        live_preview: bool = False,
        preview_src_rgb01: np.ndarray | None = None,
        preview_max_dim: int = 900,
        preview_emit_ms: int = 120,
        parent=None,
    ):
        super().__init__(parent)
        self.x_for_net = np.asarray(x_for_net, dtype=np.float32)
        self.ckpt_path = str(ckpt_path)
        self.tile = int(tile)
        self.overlap = int(overlap)
        self.use_amp = bool(use_amp)
        ad = (amp_dtype or "fp16").lower().strip()
        self.amp_dtype = ad if ad in ("fp16", "bf16") else "fp16"
        self._cancel = False
        self.model_kind = str(model_kind or "prism_mini")
        self.pad = int(pad)
        self.live_preview = bool(live_preview)           # ← ADD
        self.preview_src = preview_src_rgb01             # ← ADD
        self.preview_max_dim = int(preview_max_dim)      # ← ADD
        self.preview_emit_ms = int(preview_emit_ms)      # ← ADD

    def _pad_rgb_reflect(self, img_rgb01: np.ndarray, pad: int) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Reflect-pad an RGB float image (H,W,3). Returns (padded, (h,w)).
        """
        x = np.asarray(img_rgb01, dtype=np.float32)
        if x.ndim != 3 or x.shape[2] != 3:
            raise ValueError(f"_pad_rgb_reflect expects (H,W,3), got {x.shape}")

        h, w = x.shape[:2]
        p = int(max(0, pad))
        if p == 0:
            return x, (h, w)

        # Reflect is usually best for astro edges; symmetric also works.
        padded = np.pad(x, ((p, p), (p, p), (0, 0)), mode="reflect")
        return padded.astype(np.float32, copy=False), (h, w)


    def _unpad_rgb(self, img_rgb01: np.ndarray, orig_hw: tuple[int, int], pad: int) -> np.ndarray:
        """
        Crop padded RGB back to original size.
        """
        x = np.asarray(img_rgb01, dtype=np.float32)
        h, w = (int(orig_hw[0]), int(orig_hw[1]))
        p = int(max(0, pad))
        if p == 0:
            return x[:h, :w, :]

        return x[p:p + h, p:p + w, :].astype(np.float32, copy=False)

    def cancel(self):
        self._cancel = True

    def run(self):
        import time
        info = {
            "engine": "syqon_prism",
            "use_amp_requested": bool(self.use_amp),
            "amp_dtype_requested": str(self.amp_dtype),
        }
        try:
            def _prog(done, total, stage):
                if self._cancel:
                    raise InterruptedError("Cancelled")
                pct = int(100.0 * float(done) / max(float(total), 1.0))
                self.progress.emit(pct, str(stage or ""))

            model_variant = "deep" if self.model_kind == "prism_deep" else "free"
            info["model_kind"] = self.model_kind
            info["model_variant_requested"] = model_variant

            pad = int(max(0, getattr(self, "pad", 0)))
            info["pad_px"] = pad

            x_in, orig_hw = self._pad_rgb_reflect(self.x_for_net, pad)

            # --- live preview setup ---
            preview_buf = None
            scale_x = scale_y = 1.0
            last_emit = 0.0

            if self.live_preview and self.preview_src is not None:
                try:
                    import cv2 as _cv2
                except ImportError:
                    _cv2 = None

                # Use x_in (padded) so tile coords match the preview buffer exactly
                src = np.asarray(x_in, dtype=np.float32)
                if src.ndim == 2:
                    src = np.stack([src] * 3, axis=-1)
                elif src.ndim == 3 and src.shape[2] == 1:
                    src = np.repeat(src, 3, axis=2)
                else:
                    src = src[..., :3]

                H, W = src.shape[:2]
                md = max(H, W)
                if md > self.preview_max_dim:
                    s = self.preview_max_dim / float(md)
                    Hd = max(1, int(round(H * s)))
                    Wd = max(1, int(round(W * s)))
                else:
                    Hd, Wd = H, W

                if _cv2 is not None and (Hd != H or Wd != W):
                    preview_buf = _cv2.resize(src, (Wd, Hd), interpolation=_cv2.INTER_AREA).astype(np.float32, copy=False)
                else:
                    preview_buf = src[:Hd, :Wd, :].copy()

                scale_x = float(preview_buf.shape[1]) / float(W)
                scale_y = float(preview_buf.shape[0]) / float(H)

                from setiastro.saspro.remove_stars import _rgb01_to_qimage
                self.preview.emit(_rgb01_to_qimage(preview_buf))
                last_emit = time.time()

            def _tile_cb(y0, x0, ph, pw, tile_rgb01):
                nonlocal preview_buf, last_emit
                if preview_buf is None:
                    return
                if self._cancel:
                    raise InterruptedError("Cancelled")

                try:
                    import cv2 as _cv2
                except ImportError:
                    _cv2 = None

                from setiastro.saspro.remove_stars import _rgb01_to_qimage

                dx0 = int(round(x0 * scale_x))
                dy0 = int(round(y0 * scale_y))
                dx1 = int(round((x0 + pw) * scale_x))
                dy1 = int(round((y0 + ph) * scale_y))
                dx1 = max(dx0 + 1, dx1)
                dy1 = max(dy0 + 1, dy1)

                dh = min(preview_buf.shape[0], dy1) - dy0
                dw = min(preview_buf.shape[1], dx1) - dx0
                if dh <= 0 or dw <= 0:
                    return

                if _cv2 is not None and (dw != pw or dh != ph):
                    tile_disp = _cv2.resize(tile_rgb01, (dw, dh), interpolation=_cv2.INTER_AREA)
                else:
                    tile_disp = tile_rgb01[:dh, :dw, :]

                preview_buf[dy0:dy0 + dh, dx0:dx0 + dw, :] = tile_disp.astype(np.float32, copy=False)

                now = time.time()
                if (now - last_emit) * 1000.0 >= float(self.preview_emit_ms):
                    self.preview.emit(_rgb01_to_qimage(preview_buf))
                    last_emit = now

            denoised_p, engine_info = prism_denoise_rgb01(
                x_in,
                ckpt_path=self.ckpt_path,
                tile=self.tile,
                overlap=self.overlap,
                use_gpu=True,
                prefer_dml=True,
                use_amp=self.use_amp,
                amp_dtype=self.amp_dtype,
                model_variant=model_variant,
                progress_cb=_prog,
                tile_cb=(_tile_cb if preview_buf is not None else None),
            )

            denoised = self._unpad_rgb(denoised_p, orig_hw, pad)

            if engine_info:
                try:
                    info.update(dict(engine_info))
                except Exception:
                    info["engine_info"] = engine_info

            self.finished.emit(denoised, info, "")

        except InterruptedError:
            self.finished.emit(None, info, "__cancelled__")
        except Exception as e:
            import traceback
            info["traceback"] = traceback.format_exc()
            self.finished.emit(None, info, str(e))

def run_syqon_tools_via_preset(main, doc, preset: dict | None = None):
    """
    Public headless entrypoint for SyQon tools.
    Used by command-drop / shortcuts / bundles.
    """
    preset = dict(preset or {})
    family = str(preset.get("family", "denoise") or "denoise").strip().lower()

    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.information(main, "SyQon Tools", "No active image.")
        return

    if family == "denoise":
        _run_syqon_prism_headless(main, doc, preset)
        return

    if family == "starless":
        # starless is owned by remove_stars_preset.py, not here
        from setiastro.saspro.remove_stars_preset import run_remove_stars_via_preset

        run_remove_stars_via_preset(main, doc, {
            "tool": "syqon",
            "model_kind": str(preset.get("starless_model_kind", "nadir") or "nadir"),
            "tile_size": int(preset.get("starless_tile_size", 512)),
            "overlap": int(preset.get("starless_overlap", 64)),
            "make_stars": bool(preset.get("starless_make_stars", True)),
            "pad_edges": bool(preset.get("starless_pad_edges", True)),
            "pad_pixels": int(preset.get("starless_pad_pixels", 128)),
            "stars_extract": str(preset.get("starless_stars_extract", "subtract")),
        })
        return

    _run_syqon_parallax_headless(main, doc, preset)         

def _run_syqon_parallax_headless(main, doc, preset: dict | None = None):
    """
    Headless Parallax runner — mirrors _run_syqon_prism_headless exactly.
    Called by run_syqon_tools_via_preset when family="sharpening".
    """
    import numpy as np
    from setiastro.saspro.remove_stars import (
        _mtf_params_unlinked, _apply_mtf_unlinked_rgb, _invert_mtf_unlinked_rgb,
    )
    from setiastro.saspro.sharpen_engines.syqon_parallax_engine import (
        parallax_correction_rgb01,
        parallax_star_reduce_rgb01,
        parallax_sharpen_rgb01,
    )
    #from setiastro.saspro.syqon_paths import (
    #    syqon_parallax_correction_model_path,
    #    syqon_parallax_star_reduction_model_path,
    #    syqon_parallax_sharpen_model_path,
    #)
 
    preset = dict(preset or {})
 
    do_correct    = bool(preset.get("parallax_correct",       True))
    star_level    = int(preset.get("parallax_star_level",     3))
    sharpen_alpha = float(preset.get("parallax_sharpen_alpha", 0.7))
    tile          = int(preset.get("parallax_tile_size",      512))
    overlap       = int(preset.get("parallax_overlap",        64))
    pad           = int(preset.get("parallax_pad",            96))
    use_mtf       = bool(preset.get("parallax_use_mtf",       True))
    mtf_target    = float(preset.get("parallax_mtf_target",   0.15))
    mode = str(preset.get("parallax_mode", "classic") or "classic").lower()
    settings = QSettings()
    _keys = {
        "correction":  "syqon/parallax_correction_path",
        "star_reduce": "syqon/parallax_star_reduce_path",
        "sharpen":     "syqon/parallax_sharpen_path",
    }
    if mode == "aesthetics":
        _keys = {k: v.replace("_path", "_aesthetics_path") for k, v in _keys.items()}

    def _installed(variant):
        p = str(settings.value(_keys[variant], "", type=str) or "")
        return p if (p and Path(p).exists()) else None
    # Validate model paths
    if do_correct and _installed("correction") is None:
        QMessageBox.warning(main, "SyQon Parallax",
                            f"Correction model ({mode}) not installed. Install it from the SyQon Tools hub.")
        return
    if star_level > 0 and _installed("star_reduce") is None:
        QMessageBox.warning(main, "SyQon Parallax",
                            f"Star reduction model ({mode}) not installed. Install it from the SyQon Tools hub.")
        return
    if sharpen_alpha > 0.0 and _installed("sharpen") is None:
        QMessageBox.warning(main, "SyQon Parallax",
                            f"Sharpen model ({mode}) not installed. Install it from the SyQon Tools hub.")
        return
 
    if not do_correct and star_level == 0 and sharpen_alpha <= 0.0:
        QMessageBox.warning(main, "SyQon Parallax",
                            "No stages enabled in preset. Enable at least one.")
        return
 
    # Prepare image
    src = np.asarray(doc.image, dtype=np.float32)
    orig_was_mono = (src.ndim == 2) or (src.ndim == 3 and src.shape[2] == 1)
 
    x = np.nan_to_num(src, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    scale_factor = float(np.max(x)) if x.size else 1.0
    x01 = np.clip(x / scale_factor if scale_factor > 1.01 else x, 0.0, 1.0)
 
    if x01.ndim == 2:
        xrgb = np.stack([x01] * 3, axis=-1)
    elif x01.ndim == 3 and x01.shape[2] == 1:
        xrgb = np.repeat(x01, 3, axis=2)
    else:
        xrgb = x01[..., :3]
 
    if use_mtf:
        mtf_params = _mtf_params_unlinked(xrgb, shadows_clipping=-2.8, targetbg=mtf_target)
        x_for_net  = _apply_mtf_unlinked_rgb(xrgb, mtf_params)
    else:
        mtf_params = None
        x_for_net  = xrgb
 
    dlg = _ProcDialog(main, title="SyQon Parallax Progress")
    dlg.append_text("Starting SyQon Parallax…\\n")
 
    # Build thread
    from setiastro.saspro.syqon_tools import _SyQonParallaxProcessThread
    thr = _SyQonParallaxProcessThread(
        x_for_net       = x_for_net,
        do_correct      = do_correct,
        correction_path = _installed("correction") or "",
        star_level      = star_level,
        star_path       = _installed("star_reduce") or "",
        sharpen_alpha   = sharpen_alpha,
        sharpen_path    = _installed("sharpen") or "",
        tile            = tile,
        overlap         = overlap,
        pad             = pad,
        parent          = dlg,
        mode            = mode,
    )
 
    def _on_prog(pct: int, stage: str):
        dlg.set_progress(pct, 100, stage)
 
    def _on_done(result_s, info: dict, err: str):
        if err:
            QMessageBox.critical(main, "SyQon Parallax", err)
            dlg.close()
            return
 
        result = np.asarray(result_s, dtype=np.float32)
        if result.ndim == 2:
            result = np.stack([result] * 3, axis=-1)
 
        if use_mtf and mtf_params is not None:
            result = _invert_mtf_unlinked_rgb(result, mtf_params)
 
        if scale_factor > 1.01:
            result = np.clip(result * scale_factor, 0.0, 1.0).astype(np.float32)
 
        if orig_was_mono:
            final = result.mean(axis=2).astype(np.float32)
        else:
            final = result[..., :3]
 
        final = np.clip(final, 0.0, 1.0).astype(np.float32)
 
        meta = {
            "step_name": "Parallax",
            "bit_depth": "32-bit floating point",
            "is_mono":   bool(orig_was_mono),
            "replay_last": {
                "op": "syqon_parallax",
                "params": {
                    "correct":       do_correct,
                    "star_level":    star_level,
                    "sharpen_alpha": sharpen_alpha,
                    "tile":          tile,
                    "overlap":       overlap,
                    "pad":           pad,
                    "use_mtf":       use_mtf,
                    "mtf_target":    mtf_target,
                }
            }
        }
        doc.apply_edit(final, metadata=meta, step_name="Parallax")
 
        try:
            if hasattr(main, "_log"):
                main._log("SyQon Parallax (headless)")
        except Exception:
            pass
 
        dlg.close()
 
    thr.progress.connect(_on_prog)
    thr.finished.connect(_on_done)
    dlg.cancel_button.clicked.connect(lambda: thr.cancel())
    dlg.show()
    thr.start()
    dlg.exec()


def _run_syqon_prism_headless(main, doc, preset: dict | None = None):
    preset = dict(preset or {})

    model_kind = str(preset.get("prism_model_kind", "prism_mini") or "prism_mini").strip().lower()
    tile = int(preset.get("prism_tile_size", 512))
    overlap = int(preset.get("prism_overlap", 64))
    pad = int(preset.get("prism_pad", 64))
    strength = float(preset.get("prism_strength", 0.85))
    use_mtf = bool(preset.get("prism_use_mtf", True))
    mtf_target = float(preset.get("prism_mtf_target_median", 0.10))
    use_amp = bool(preset.get("prism_use_amp", False))

    model_path = syqon_prism_model_path(model_kind)
    if not model_path.exists():
        QMessageBox.warning(main, "SyQon Prism", "Model is not installed. Install it first from the SyQon Tools hub.")
        return

    src = np.asarray(doc.image).astype(np.float32, copy=False)
    orig_was_mono = (src.ndim == 2) or (src.ndim == 3 and src.shape[2] == 1)

    x = np.nan_to_num(src, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    scale_factor = float(np.max(x)) if x.size else 1.0
    if scale_factor > 1.01:
        x01 = np.clip(x / scale_factor, 0.0, 1.0)
    else:
        x01 = np.clip(x, 0.0, 1.0)

    if x01.ndim == 2:
        xrgb = np.stack([x01] * 3, axis=-1)
    elif x01.ndim == 3 and x01.shape[2] == 1:
        xrgb = np.repeat(x01, 3, axis=2)
    else:
        xrgb = x01[..., :3]

    if use_mtf:
        mtf_params = _mtf_params_unlinked(xrgb, shadows_clipping=-2.8, targetbg=mtf_target)
        x_for_net = _apply_mtf_unlinked_rgb(xrgb, mtf_params)
    else:
        mtf_params = None
        x_for_net = xrgb

    dlg = _ProcDialog(main, title="SyQon Prism Progress")
    dlg.append_text("Starting SyQon Prism…\n")

    thr = _SyQonPrismProcessThread(
        x_for_net=x_for_net,
        ckpt_path=str(model_path),
        model_kind=model_kind,
        tile=tile,
        overlap=overlap,
        pad=pad,
        use_amp=use_amp,
        amp_dtype="fp16",
        parent=dlg,
    )

    def _on_prog(pct: int, stage: str):
        dlg.set_progress(pct, 100, stage)

    def _on_done(denoised_s, info: dict, err: str):
        if err:
            QMessageBox.critical(main, "SyQon Prism", err)
            dlg.close()
            return

        denoised_s = np.asarray(denoised_s, dtype=np.float32)
        if denoised_s.ndim == 2:
            denoised_s = np.stack([denoised_s] * 3, axis=-1)

        denoised_lin = denoised_s
        if use_mtf and mtf_params is not None:
            denoised_lin = _invert_mtf_unlinked_rgb(denoised_s, mtf_params)

        if scale_factor > 1.01:
            denoised_lin = np.clip(denoised_lin * scale_factor, 0.0, 1.0).astype(np.float32, copy=False)

        orig = np.asarray(doc.image).astype(np.float32, copy=False)
        orig = np.nan_to_num(orig, nan=0.0, posinf=0.0, neginf=0.0)

        if orig_was_mono:
            if orig.ndim == 3:
                if orig.shape[2] == 1:
                    orig_base = orig[..., 0]
                else:
                    orig_base = orig.mean(axis=2)
            else:
                orig_base = orig

            den_base = denoised_lin.mean(axis=2).astype(np.float32, copy=False)
            final_to_apply = ((1.0 - strength) * orig_base + strength * den_base).astype(np.float32, copy=False)
            orig_for_mask = orig_base
        else:
            if orig.ndim == 2:
                orig_rgb = np.stack([orig] * 3, axis=-1)
            elif orig.ndim == 3 and orig.shape[2] == 1:
                orig_rgb = np.repeat(orig, 3, axis=2)
            else:
                orig_rgb = orig[..., :3]

            final_to_apply = ((1.0 - strength) * orig_rgb + strength * denoised_lin).astype(np.float32, copy=False)
            orig_for_mask = orig_rgb

        final_to_apply = np.clip(final_to_apply, 0.0, 1.0).astype(np.float32, copy=False)
        orig_for_mask = orig_base if orig_was_mono else orig_rgb
        final_to_apply = _blend_result_with_mask(final_to_apply, orig_for_mask, doc)

        meta = {
            "step_name": "Denoised",
            "bit_depth": "32-bit floating point",
            "is_mono": bool(orig_was_mono),
            "masked": bool(getattr(doc, "active_mask_id", None)),
            "mask_id": getattr(doc, "active_mask_id", None) or None,
            "mask_blend": "m*out+(1-m)*src",            
            "replay_last": {
                "op": "syqon_prism",
                "params": {
                    "model_kind": model_kind,
                    "tile_size": tile,
                    "overlap": overlap,
                    "pad": pad,
                    "strength": strength,
                    "model_path": str(model_path),
                    "use_mtf": use_mtf,
                    "mtf_target_median": mtf_target,
                    "use_amp": use_amp,
                    "label": "Denoise (SyQon Prism)",
                }
            }
        }

        doc.apply_edit(final_to_apply, metadata=meta, step_name="Denoised")

        try:
            if hasattr(main, "_log"):
                main._log("SyQon Prism Denoise (headless)")
        except Exception:
            pass

        dlg.close()

    thr.progress.connect(_on_prog)
    thr.finished.connect(_on_done)
    dlg.cancel_button.clicked.connect(lambda: thr.cancel())
    dlg.show()
    thr.start()
    dlg.exec()    


def run_syqon_prism_via_preset(main, doc, preset: dict | None = None):
    preset = dict(preset or {})
    preset["family"] = "denoise"
    return run_syqon_tools_via_preset(main, doc, preset)


def run_syqon_starless_via_preset(main, doc, preset: dict | None = None):
    preset = dict(preset or {})
    preset["family"] = "starless"
    return run_syqon_tools_via_preset(main, doc, preset)    