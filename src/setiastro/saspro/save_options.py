# pro/save_options.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout, QCheckBox,
    QRadioButton, QButtonGroup, QWidget
)

# Allowed bit depths per output format (what your saver actually supports)
_BIT_DEPTHS = {
    "png":  ["8-bit"],
    "jpg":  ["8-bit"],
    "fits": ["8-bit", "16-bit", "32-bit unsigned", "32-bit floating point"],
    "fit":  ["8-bit", "16-bit", "32-bit unsigned", "32-bit floating point"],
    "tif":  ["8-bit", "16-bit", "32-bit unsigned", "32-bit floating point"],
    "xisf": ["16-bit", "32-bit unsigned", "32-bit floating point"],
}

_TIFF_COMP = ["None", "LZW", "ZIP (Deflate)"]


def _norm_ext(target_ext: str) -> str:
    raw_ext = (target_ext or "").lower().strip()
    if "." in raw_ext:
        raw_ext = raw_ext.split(".")[-1]

    if raw_ext in ("fit", "fits", "fz", "fits.gz", "fit.gz"):
        return "fits"
    if raw_ext in ("tif", "tiff"):
        return "tif"
    if raw_ext in ("jpg", "jpeg"):
        return "jpg"
    return raw_ext


class ExportDialog(QDialog):
    """
    "Export"-style options dialog.

    Returns:
      - selected_bit_depth(): str
      - selected_jpeg_quality(): Optional[int]
      - export_options(): dict   (dpi, tiff_compression, resize_mode, etc.)
    """
    def __init__(
        self,
        parent,
        target_ext: str,
        current_bit_depth: str | None,
        current_jpeg_quality: int | None = None,
        *,
        settings=None,   # optional QSettings for persistence
    ):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Export"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)

        self._settings = settings
        self._ext = _norm_ext(target_ext)

        allowed_depths = _BIT_DEPTHS.get(self._ext, ["32-bit floating point"])

        # ---- layout root ----
        lay = QVBoxLayout(self)

        # -----------------------------
        # 1) Format summary (nice UX)
        # -----------------------------
        fmt_lbl = QLabel(self.tr(f"Export format: {self._ext.upper()}"))
        fmt_lbl.setWordWrap(True)
        lay.addWidget(fmt_lbl)

        # -----------------------------
        # 2) Core options box
        # -----------------------------
        core_box = QGroupBox(self.tr("Export settings"))
        core_form = QFormLayout(core_box)

        # Bit depth
        self.combo_depth = QComboBox(self)
        self.combo_depth.addItems(allowed_depths)
        if current_bit_depth in allowed_depths:
            self.combo_depth.setCurrentText(current_bit_depth)
        core_form.addRow(self.tr("Bit depth"), self.combo_depth)

        # Embed profile (placeholder – wire later if/when you support ICC)
        self.chk_embed_icc = QCheckBox(self.tr("Embed ICC profile (if available)"))
        self.chk_embed_icc.setChecked(True)
        core_form.addRow(self.tr("Color"), self.chk_embed_icc)

        lay.addWidget(core_box)

        # -----------------------------
        # 3) Format-specific panels
        # -----------------------------
        self.panel_jpg = self._build_jpg_panel(current_jpeg_quality)
        self.panel_tif = self._build_tif_panel()
        lay.addWidget(self.panel_jpg)
        lay.addWidget(self.panel_tif)

        # -----------------------------
        # 4) Resize panel (applies to all; ok if you don't implement yet)
        # -----------------------------
        self.panel_resize = self._build_resize_panel()
        lay.addWidget(self.panel_resize)

        # -----------------------------
        # Buttons
        # -----------------------------
        btn_ok = QPushButton(self.tr("Export"))
        btn_cancel = QPushButton(self.tr("Cancel"))
        btn_ok.clicked.connect(self._on_accept)
        btn_cancel.clicked.connect(self.reject)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        lay.addLayout(row)

        # ---- show/hide panels based on format ----
        self._apply_format_visibility()

        # ---- load persisted defaults (optional) ----
        self._load_settings()

    # ------------------------------------------------------------------
    # UI builders
    # ------------------------------------------------------------------
    def _build_jpg_panel(self, current_jpeg_quality: int | None) -> QWidget:
        box = QGroupBox(self.tr("JPEG options"))
        form = QFormLayout(box)

        self.jpeg_quality_spin = QSpinBox(self)
        self.jpeg_quality_spin.setRange(1, 100)
        default_q = int(current_jpeg_quality) if current_jpeg_quality is not None else 95
        self.jpeg_quality_spin.setValue(max(1, min(100, default_q)))
        form.addRow(self.tr("Quality (1–100)"), self.jpeg_quality_spin)

        # future: subsampling toggle etc.
        return box

    def _build_tif_panel(self) -> QWidget:
        box = QGroupBox(self.tr("TIFF options"))
        form = QFormLayout(box)

        self.tiff_dpi_spin = QSpinBox(self)
        self.tiff_dpi_spin.setRange(1, 2400)
        self.tiff_dpi_spin.setValue(300)
        form.addRow(self.tr("DPI"), self.tiff_dpi_spin)

        self.tiff_comp_combo = QComboBox(self)
        self.tiff_comp_combo.addItems(_TIFF_COMP)
        self.tiff_comp_combo.setCurrentText("LZW")
        form.addRow(self.tr("Compression"), self.tiff_comp_combo)

        return box

    def _build_resize_panel(self) -> QWidget:
        box = QGroupBox(self.tr("Resize (optional)"))
        form = QFormLayout(box)

        self.rb_resize_none = QRadioButton(self.tr("No resize"))
        self.rb_resize_percent = QRadioButton(self.tr("Scale by percent"))
        self.rb_resize_long_edge = QRadioButton(self.tr("Set long edge (px)"))
        self.rb_resize_none.setChecked(True)

        self.resize_group = QButtonGroup(self)
        self.resize_group.addButton(self.rb_resize_none, 0)
        self.resize_group.addButton(self.rb_resize_percent, 1)
        self.resize_group.addButton(self.rb_resize_long_edge, 2)

        # percent
        self.resize_percent_spin = QSpinBox(self)
        self.resize_percent_spin.setRange(1, 400)
        self.resize_percent_spin.setValue(100)

        # long edge
        self.resize_long_edge_spin = QSpinBox(self)
        self.resize_long_edge_spin.setRange(16, 200000)
        self.resize_long_edge_spin.setValue(4096)

        # enable/disable inputs based on selection
        self.resize_group.idClicked.connect(self._update_resize_enable)
        self._update_resize_enable()

        form.addRow(self.rb_resize_none)
        rowp = QHBoxLayout()
        rowp.addWidget(self.rb_resize_percent)
        rowp.addStretch(1)
        rowp.addWidget(QLabel(self.tr("Percent:")))
        rowp.addWidget(self.resize_percent_spin)
        w = QWidget()
        w.setLayout(rowp)
        form.addRow(w)

        rowe = QHBoxLayout()
        rowe.addWidget(self.rb_resize_long_edge)
        rowe.addStretch(1)
        rowe.addWidget(QLabel(self.tr("Long edge:")))
        rowe.addWidget(self.resize_long_edge_spin)
        w2 = QWidget()
        w2.setLayout(rowe)
        form.addRow(w2)

        return box

    # ------------------------------------------------------------------
    # Behavior
    # ------------------------------------------------------------------
    def _apply_format_visibility(self):
        self.panel_jpg.setVisible(self._ext == "jpg")
        self.panel_tif.setVisible(self._ext == "tif")

    def _update_resize_enable(self):
        mode = self.resize_group.checkedId()
        self.resize_percent_spin.setEnabled(mode == 1)
        self.resize_long_edge_spin.setEnabled(mode == 2)

    def _on_accept(self):
        # persist settings if present
        self._save_settings()
        self.accept()

    # ------------------------------------------------------------------
    # Settings persistence (optional)
    # ------------------------------------------------------------------
    def _k(self, suffix: str) -> str:
        # per-format persistence
        return f"export/{self._ext}/{suffix}"

    def _load_settings(self):
        s = self._settings
        if s is None:
            return
        try:
            bd = s.value(self._k("bit_depth"), "", type=str) or ""
            if bd:
                idx = self.combo_depth.findText(bd)
                if idx >= 0:
                    self.combo_depth.setCurrentIndex(idx)

            embed = s.value(self._k("embed_icc"), True, type=bool)
            self.chk_embed_icc.setChecked(bool(embed))

            if self._ext == "jpg":
                q = s.value(self._k("jpeg_quality"), 95, type=int)
                self.jpeg_quality_spin.setValue(max(1, min(100, int(q))))

            if self._ext == "tif":
                dpi = s.value(self._k("dpi"), 300, type=int)
                self.tiff_dpi_spin.setValue(max(1, int(dpi)))
                comp = s.value(self._k("tiff_comp"), "LZW", type=str)
                idx = self.tiff_comp_combo.findText(comp)
                if idx >= 0:
                    self.tiff_comp_combo.setCurrentIndex(idx)

            rmode = s.value(self._k("resize_mode"), "none", type=str)
            if rmode == "percent":
                self.rb_resize_percent.setChecked(True)
            elif rmode == "long_edge":
                self.rb_resize_long_edge.setChecked(True)
            else:
                self.rb_resize_none.setChecked(True)

            rp = s.value(self._k("resize_percent"), 100, type=int)
            self.resize_percent_spin.setValue(max(1, int(rp)))
            re = s.value(self._k("resize_long_edge"), 4096, type=int)
            self.resize_long_edge_spin.setValue(max(16, int(re)))

            self._update_resize_enable()
        except Exception:
            pass

    def _save_settings(self):
        s = self._settings
        if s is None:
            return
        try:
            s.setValue(self._k("bit_depth"), self.combo_depth.currentText())
            s.setValue(self._k("embed_icc"), bool(self.chk_embed_icc.isChecked()))

            if self._ext == "jpg":
                s.setValue(self._k("jpeg_quality"), int(self.jpeg_quality_spin.value()))

            if self._ext == "tif":
                s.setValue(self._k("dpi"), int(self.tiff_dpi_spin.value()))
                s.setValue(self._k("tiff_comp"), self.tiff_comp_combo.currentText())

            mode = self.resize_group.checkedId()
            if mode == 1:
                s.setValue(self._k("resize_mode"), "percent")
            elif mode == 2:
                s.setValue(self._k("resize_mode"), "long_edge")
            else:
                s.setValue(self._k("resize_mode"), "none")

            s.setValue(self._k("resize_percent"), int(self.resize_percent_spin.value()))
            s.setValue(self._k("resize_long_edge"), int(self.resize_long_edge_spin.value()))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def selected_bit_depth(self) -> str:
        return self.combo_depth.currentText()

    def selected_jpeg_quality(self) -> Optional[int]:
        if self._ext != "jpg":
            return None
        return int(self.jpeg_quality_spin.value())

    def export_options(self) -> Dict[str, Any]:
        """
        Options consumed by docman.save_document(..., export_opts=...)
        and then legacy_save_image(..., export_opts=...)
        """
        opts: Dict[str, Any] = {}

        # common
        opts["embed_icc"] = bool(self.chk_embed_icc.isChecked())

        # TIFF
        if self._ext == "tif":
            opts["dpi"] = int(self.tiff_dpi_spin.value())
            comp = (self.tiff_comp_combo.currentText() or "").strip().lower()
            if comp.startswith("lzw"):
                opts["tiff_compression"] = "lzw"
            elif comp.startswith("zip") or "deflate" in comp:
                opts["tiff_compression"] = "deflate"
            else:
                opts["tiff_compression"] = None

        # Resize
        mode = self.resize_group.checkedId()
        if mode == 1:
            opts["resize_mode"] = "percent"
            opts["resize_percent"] = int(self.resize_percent_spin.value())
        elif mode == 2:
            opts["resize_mode"] = "long_edge"
            opts["resize_long_edge"] = int(self.resize_long_edge_spin.value())
        else:
            opts["resize_mode"] = "none"

        return opts
