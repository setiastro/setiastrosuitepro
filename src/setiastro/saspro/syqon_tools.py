# src/setiastro/saspro/syqon_tools.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from PyQt6.QtCore import Qt, QSettings, QUrl, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QStackedWidget, QWidget, QFormLayout, QGroupBox, QMessageBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QFileDialog, QProgressBar
)
from PyQt6.QtGui import QIcon, QDesktopServices

from setiastro.saspro.resources import starnet_path

from setiastro.saspro.syqon_paths import syqon_prism_model_path
from setiastro.saspro.denoise_engines.syqon_prism_engine import prism_denoise_rgb01
from setiastro.saspro.remove_stars import (
    SyQonStarlessDialog,
    _mtf_params_unlinked,
    _apply_mtf_unlinked_rgb,
    _invert_mtf_unlinked_rgb,
)

_SYQON_BUY_URL_PRISM_MINI = "https://syqon.it/"   # replace with exact URL when you have it
_SYQON_BUY_URL_PRISM_DEEP = "https://syqon.it/"   # replace with exact URL when you have it

def _syqon_prism_buy_url_for(model_kind: str) -> str:
    mk = (model_kind or "prism_mini").lower().strip()
    return _SYQON_BUY_URL_PRISM_DEEP if mk == "prism_deep" else _SYQON_BUY_URL_PRISM_MINI

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

        self.cmb_family.currentIndexChanged.connect(self._sync_page)
        self._sync_page()

    def _sync_page(self):
        key = self.cmb_family.currentData()
        if key == "starless":
            self.stack.setCurrentWidget(self.page_starless)
            self.btn_launch.setText("Open Starless Tool")
        elif key == "denoise":
            self.stack.setCurrentWidget(self.page_denoise)
            self.btn_launch.setText("Process")
        else:
            self.stack.setCurrentWidget(self.page_sharpen)
            self.btn_launch.setText("Open Sharpening Tool")

    def _launch_selected_tool(self):
        doc = self.get_active_doc()
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "SyQon Tools", "No active image.")
            return

        key = self.cmb_family.currentData()

        if key == "starless":
            model_kind = self.page_starless.model_kind()
            if not hasattr(self, "_child_dialogs"):
                self._child_dialogs = []

            dlg = SyQonStarlessDialog(self.parent(), doc, parent=self.parent(), icon=self.windowIcon())
            idx = dlg.cmb_model.findData(model_kind)
            if idx >= 0:
                dlg.cmb_model.setCurrentIndex(idx)

            dlg.setModal(False)
            dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

            self._child_dialogs.append(dlg)
            dlg.destroyed.connect(lambda *_ , d=dlg: self._child_dialogs.remove(d) if d in self._child_dialogs else None)

            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
            return

        if key == "denoise":
            self.page_denoise.process_document(doc, self.parent())
            return

        QMessageBox.information(
            self,
            "SyQon Sharpening",
            "Parallax / sharpening models are not available yet."
        )

class _SyQonStarlessHubPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)

        box = QGroupBox("SyQon Starless", self)
        form = QFormLayout(box)

        self.cmb_model = QComboBox(self)
        self.cmb_model.addItem("Nadir", userData="nadir")
        self.cmb_model.addItem("AxiomV2", userData="axiomv2")
        form.addRow("Model:", self.cmb_model)

        info = QLabel(
            "Use SyQon’s star-removal models. This opens the full Starless dialog "
            "with all current settings and live preview support."
        )
        info.setWordWrap(True)

        lay.addWidget(box)
        lay.addWidget(info)
        lay.addStretch(1)

    def model_kind(self) -> str:
        return str(self.cmb_model.currentData() or "nadir")

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

        lay = QVBoxLayout(self)

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

        self.btn_buy.clicked.connect(self._open_buy_page)
        self.btn_install.clicked.connect(self._install_model)
        self.btn_remove.clicked.connect(self._remove_model)

        btn_row.addWidget(self.btn_buy)
        btn_row.addWidget(self.btn_install)
        btn_row.addWidget(self.btn_remove)

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
        self.chk_mtf = QCheckBox("Apply temporary stretch for model (recommended)", self)
        self.chk_mtf.setChecked(bool(self.settings.value("syqon/prism_use_mtf", True, type=bool)))
        form.addRow("", self.chk_mtf)

        self.spin_mtf_median = QDoubleSpinBox(self)
        self.spin_mtf_median.setRange(0.01, 0.50)
        self.spin_mtf_median.setSingleStep(0.01)
        self.spin_mtf_median.setDecimals(3)
        self.spin_mtf_median.setValue(float(self.settings.value("syqon/prism_mtf_target_median", 0.10)))
        form.addRow("Temp stretch target median:", self.spin_mtf_median)
        self.spin_strength = QDoubleSpinBox(self)
        self.spin_strength.setRange(0.0, 1.0)
        self.spin_strength.setSingleStep(0.05)
        self.spin_strength.setDecimals(2)
        self.spin_strength.setValue(float(self.settings.value("syqon/prism_strength", 0.85)))
        form.addRow("Strength:", self.spin_strength)
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

        saved_kind = str(self.settings.value("syqon/prism_model_kind", "prism_mini"))
        idx = self.cmb_model.findData(saved_kind)
        if idx >= 0:
            self.cmb_model.setCurrentIndex(idx)

        self._refresh_state()

    def model_kind(self) -> str:
        return str(self.cmb_model.currentData() or "prism_mini")

    def _model_dst_path(self):
        return syqon_prism_model_path(self.model_kind())

    def _have_model(self) -> bool:
        try:
            p = self._model_dst_path()
            return p.exists() and p.is_file()
        except Exception:
            return False

    def _on_model_changed(self, *_):
        self.settings.setValue("syqon/prism_model_kind", self.model_kind())
        self._refresh_state()

    def _refresh_state(self):
        mk = self.model_kind()
        dst = self._model_dst_path()
        self.lbl_model_path.setText(str(dst))

        if mk == "prism_deep":
            expected = "the Prism Deep model file"
        else:
            expected = "the Prism Mini model file"

        if self._have_model():
            self.lbl_status.setText("Ready (Prism model installed).")
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
        else:
            expected_desc = "Prism Mini model file"

        src_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {expected_desc}",
            "",
            "All Files (*)"
        )
        if not src_path:
            return

        src = Path(src_path)
        if not src.exists():
            QMessageBox.warning(self, "SyQon Prism", "Selected file does not exist.")
            return

        dst = self._model_dst_path()
        dst.parent.mkdir(parents=True, exist_ok=True)

        try:
            import shutil
            shutil.copy2(str(src), str(dst))
        except Exception as e:
            QMessageBox.critical(self, "SyQon Prism", f"Failed to install model:\n{e}")
            self._refresh_state()
            return

        self.settings.setValue(f"syqon/prism_model_src_path/{mk}", str(src))
        self.settings.setValue(f"syqon/prism_model_installed_path/{mk}", str(dst))
        self._refresh_state()

    def _remove_model(self):
        dst = self._model_dst_path()
        if not dst.exists():
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

        self._refresh_state()            
    
    def _set_busy(self, busy: bool):
        self.btn_buy.setEnabled(not busy)
        self.btn_install.setEnabled(not busy)
        self.btn_remove.setEnabled((not busy) and self._have_model())

        self.cmb_model.setEnabled(not busy)
        self.spin_tile.setEnabled(not busy)
        self.spin_overlap.setEnabled(not busy)
        self.chk_mtf.setEnabled(not busy)
        self.spin_strength.setEnabled(not busy)
        self.spin_mtf_median.setEnabled((not busy) and self.chk_mtf.isChecked())
        self.chk_amp.setEnabled(not busy)

        self.pbar.setVisible(busy)
        if busy:
            self.pbar.setRange(0, 100)
            self.pbar.setValue(0)


    def _on_worker_progress(self, pct: int, stage: str):
        self.pbar.setValue(int(pct))
        if stage:
            self.lbl_status.setText(stage)


    def _on_worker_finished(self, denoised_s, info: dict, err: str):
        if err:
            self._set_busy(False)
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
        else:
            if orig.ndim == 2:
                orig_rgb = np.stack([orig] * 3, axis=-1)
            elif orig.ndim == 3 and orig.shape[2] == 1:
                orig_rgb = np.repeat(orig, 3, axis=2)
            else:
                orig_rgb = orig[..., :3]

            final_to_apply = ((1.0 - strength) * orig_rgb + strength * denoised_lin).astype(np.float32, copy=False)

        final_to_apply = np.clip(final_to_apply, 0.0, 1.0).astype(np.float32, copy=False)

        meta = {
            "step_name": "Denoised",
            "bit_depth": "32-bit floating point",
            "is_mono": bool(orig_was_mono),
            "replay_last": {
                "op": "syqon_prism",
                "params": {
                    "model_kind": self.model_kind(),
                    "tile_size": int(self.spin_tile.value()),
                    "overlap": int(self.spin_overlap.value()),
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
                    "strength": float(self.spin_strength.value()),
                    "model_path": str(self._model_dst_path()),
                    "use_mtf": bool(do_mtf),
                    "mtf_target_median": float(self.spin_mtf_median.value()),
                    "use_amp": bool(self.chk_amp.isChecked()),
                },
            }
        except Exception:
            pass

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

        ckpt_path = str(self._model_dst_path())

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
            tile=int(self.spin_tile.value()),
            overlap=int(self.spin_overlap.value()),
            use_amp=bool(self.chk_amp.isChecked()),
            amp_dtype="fp16",
            parent=self,
        )
        self.proc_thr.progress.connect(self._on_worker_progress)
        self.proc_thr.finished.connect(self._on_worker_finished)
        self.proc_thr.start()

    def closeEvent(self, ev):
        try:
            if self.proc_thr is not None and self.proc_thr.isRunning():
                self.proc_thr.cancel()
                self.proc_thr.wait(500)
        except Exception:
            pass
        super().closeEvent(ev)

class _SyQonSharpenHubPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)

        box = QGroupBox("SyQon Sharpening", self)
        form = QFormLayout(box)

        self.lbl = QLabel("Parallax / future sharpening tools are not available yet.")
        self.lbl.setWordWrap(True)
        lay.addWidget(box)
        lay.addWidget(self.lbl)
        lay.addStretch(1)    

class _SyQonPrismProcessThread(QThread):
    progress = pyqtSignal(int, str)          # percent, stage
    finished = pyqtSignal(object, dict, str) # denoised, info, err

    def __init__(
        self,
        x_for_net: np.ndarray,
        ckpt_path: str,
        tile: int,
        overlap: int,
        use_amp: bool = False,
        amp_dtype: str = "fp16",
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

    def cancel(self):
        self._cancel = True

    def run(self):
        info = {
            "engine": "syqon_prism",
            "use_amp_requested": bool(self.use_amp),
            "amp_dtype_requested": str(self.amp_dtype),
        }

        try:
            if self._cancel:
                raise RuntimeError("Cancelled")

            def _prog(done, total, stage):
                if self._cancel:
                    raise RuntimeError("Cancelled")
                pct = int(100.0 * float(done) / max(float(total), 1.0))
                self.progress.emit(pct, str(stage or ""))

            denoised, engine_info = prism_denoise_rgb01(
                self.x_for_net,
                ckpt_path=self.ckpt_path,
                tile=self.tile,
                overlap=self.overlap,
                use_gpu=True,
                prefer_dml=True,
                use_amp=self.use_amp,
                amp_dtype=self.amp_dtype,
                progress_cb=_prog,
            )

            if engine_info:
                try:
                    info.update(dict(engine_info))
                except Exception:
                    info["engine_info"] = engine_info

            self.finished.emit(denoised, info, "")
            return

        except Exception as e:
            import traceback
            info["traceback"] = traceback.format_exc()
            self.finished.emit(None, info, str(e))
