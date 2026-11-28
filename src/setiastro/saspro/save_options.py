#saspro/save_options.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout, QCheckBox,
    QRadioButton, QButtonGroup, QWidget, QMessageBox
)
from setiastro.saspro.color_space_manager import (
    COLOR_SPACES,
    DEFAULT_RENDERING_INTENT,
    RENDERING_INTENT_PERCEPTUAL,
    RENDERING_INTENT_RELATIVE,
    get_softproof_black_point_compensation_from_settings,
    get_softproof_rendering_intent_from_settings,
    get_color_space_options,
    get_profile_search_directories,
    get_icc_key_from_color_space_key,
    get_key_from_icc_key,
    get_working_color_space_from_settings,
    is_profile_available,
)

# Allowed bit depths per output format (what your saver actually supports)
_BIT_DEPTHS = {
    "png":  ["8-bit"],
    "jpg":  ["8-bit"],
    "fits": ["8-bit", "16-bit", "32-bit unsigned", "32-bit floating point"],
    "fit":  ["8-bit", "16-bit", "32-bit unsigned", "32-bit floating point"],
    "tif":  ["8-bit", "16-bit", "32-bit unsigned", "32-bit floating point"],
    "xisf": ["16-bit", "32-bit unsigned", "32-bit floating point"],
    "psb":  ["16-bit", "32-bit floating point"], 
    "webp": ["8-bit"],
    # SYQ Core 1: u8 / u16 / f32 (SASpro is f32-internal; no u32 or f64 export).
    "syq":  ["8-bit", "16-bit", "32-bit floating point"],
}

_TIFF_COMP = ["None", "LZW", "ZIP (Deflate)"]

# FITS row order choices (state / export_opts values). "source" = the
# document's own: bottom-up when it came from a bottom-up FITS (Siril's).
ROW_ORDER_SOURCE = "source"
_ROW_ORDERS = (ROW_ORDER_SOURCE, "top-down", "bottom-up")


def _norm_ext(target_ext: str) -> str:
    raw_ext = (target_ext or "").lower().strip()
    if "." in raw_ext:
        raw_ext = raw_ext.split(".")[-1]

    if raw_ext in ("fit", "fits", "fts", "fz", "fits.gz", "fit.gz"):
        return "fits"
    if raw_ext in ("tif", "tiff"):
        return "tif"
    if raw_ext in ("jpg", "jpeg"):
        return "jpg"
    return raw_ext


# Extra bit-depth entry offered by tools that save copies of an open document
# (Checkpoint Save / Export): write at whatever depth the document carries.
KEEP_DOC_DEPTH = "Same as document"


def _tiff_comp_value(text) -> Optional[str]:
    comp = (str(text or "")).strip().lower()
    if comp.startswith("lzw"):
        return "lzw"
    if comp.startswith("zip") or "deflate" in comp:
        return "deflate"
    return None


def export_opts_from_state(target_ext: str, st: Dict[str, Any]) -> Dict[str, Any]:
    """
    The export_opts dict for docman.save_document / legacy save_image, built
    from a panel state (see ExportOptionsPanel.state()). Pure function so
    headless savers (shortcuts, Function Bundles) produce exactly what the
    dialog would.
    """
    ext = _norm_ext(target_ext)
    st = dict(st or {})
    opts: Dict[str, Any] = {}

    # common
    opts["embed_icc"] = bool(st.get("embed_icc", True))
    opts["icc_color_space"] = st.get("icc_color_space") or "P3"
    opts["rendering_intent"] = st.get("rendering_intent") or DEFAULT_RENDERING_INTENT
    opts["black_point_compensation"] = bool(st.get("black_point_compensation", True))

    # TIFF
    if ext == "tif":
        opts["dpi"] = int(st.get("dpi", 300))
        opts["tiff_compression"] = _tiff_comp_value(st.get("tiff_comp", "LZW"))

    # FITS (.fit / .fits / .fts / .fz). Absent = the document's own row order.
    if ext == "fits":
        ro = str(st.get("fits_row_order") or ROW_ORDER_SOURCE).strip().lower()
        if ro in ("top-down", "bottom-up"):
            opts["fits_row_order"] = ro

    # Resize
    mode = str(st.get("resize_mode") or "none")
    if mode == "percent":
        opts["resize_mode"] = "percent"
        opts["resize_percent"] = int(st.get("resize_percent", 100))
    elif mode == "long_edge":
        opts["resize_mode"] = "long_edge"
        opts["resize_long_edge"] = int(st.get("resize_long_edge", 4096))
    else:
        opts["resize_mode"] = "none"

    return opts


def jpeg_quality_from_state(target_ext: str, st: Dict[str, Any]) -> Optional[int]:
    if _norm_ext(target_ext) != "jpg":
        return None
    try:
        return max(1, min(100, int((st or {}).get("jpeg_quality", 95))))
    except Exception:
        return 95


def bit_depth_from_state(st: Dict[str, Any]) -> Optional[str]:
    """The chosen bit depth, or None for 'Same as document'."""
    bd = (st or {}).get("bit_depth")
    if not bd or bd == KEEP_DOC_DEPTH:
        return None
    return str(bd)


class ExportOptionsPanel(QWidget):
    """
    The per-format save options (bit depth, colour management, JPEG / TIFF
    settings, resize) as an embeddable widget. Save As shows it inside
    ExportDialog; Checkpoint Save / Export embeds the same panel, so both
    always offer the same options for a format.

    `keep_depth_option=True` adds a leading "Same as document" bit depth
    (selected_bit_depth() then returns None for it).

    `source_bottom_up`: how the one document being saved stores its rows,
    shown on the "Same as source file" row order. None when the panel may
    save several documents (Checkpoint / Export).
    `row_order_option=False` hides the row order for writers that ignore it
    (FITS bundles).
    """
    def __init__(
        self,
        target_ext: str,
        current_bit_depth: str | None = None,
        current_jpeg_quality: int | None = None,
        *,
        settings=None,
        keep_depth_option: bool = False,
        source_bottom_up: bool | None = None,
        row_order_option: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._settings = settings
        self._keep_depth = bool(keep_depth_option)
        self._row_order_option = bool(row_order_option)
        self._ext = _norm_ext(target_ext)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        # -----------------------------
        # Core options box
        # -----------------------------
        core_box = QGroupBox(self.tr("Export settings"))
        core_form = QFormLayout(core_box)

        # Bit depth
        self.combo_depth = QComboBox(self)
        self._fill_depths(current_bit_depth)
        core_form.addRow(self.tr("Bit depth"), self.combo_depth)

        # Embed profile
        self.chk_embed_icc = QCheckBox(self.tr("Embed ICC profile (if available)"))
        self.chk_embed_icc.setChecked(True)
        core_form.addRow(self.tr("Color"), self.chk_embed_icc)

        # Colour space to tag on export. Default to the working display space,
        # while still allowing per-format overrides below.
        self.combo_color_space = QComboBox(self)
        for info in get_color_space_options():
            self.combo_color_space.addItem(self._color_space_label(info.key), info.icc_key)
            idx = self.combo_color_space.count() - 1
            self.combo_color_space.setItemData(idx, self._color_space_description(info.key), Qt.ItemDataRole.ToolTipRole)
        self.combo_color_space.setToolTip(self.tr(
            "Select the ICC profile used to tag exported images. The default "
            "matches the working color space selected in Preferences."))
        self.chk_embed_icc.toggled.connect(self.combo_color_space.setEnabled)
        core_form.addRow(self.tr("Colour space"), self.combo_color_space)

        self.combo_rendering_intent = QComboBox(self)
        self.combo_rendering_intent.addItem(self.tr("Relative Colorimetric"), RENDERING_INTENT_RELATIVE)
        self.combo_rendering_intent.addItem(self.tr("Perceptual"), RENDERING_INTENT_PERCEPTUAL)
        intent_idx = self.combo_rendering_intent.findData(get_softproof_rendering_intent_from_settings(self._settings))
        if intent_idx < 0:
            intent_idx = self.combo_rendering_intent.findData(DEFAULT_RENDERING_INTENT)
        if intent_idx >= 0:
            self.combo_rendering_intent.setCurrentIndex(intent_idx)
        core_form.addRow(self.tr("Rendering intent"), self.combo_rendering_intent)

        self.chk_black_point_compensation = QCheckBox(self.tr("Black-point compensation"))
        self.chk_black_point_compensation.setChecked(get_softproof_black_point_compensation_from_settings(self._settings))
        core_form.addRow("", self.chk_black_point_compensation)

        default_key = get_working_color_space_from_settings(self._settings)
        default_icc = get_icc_key_from_color_space_key(default_key)
        default_idx = self.combo_color_space.findData(default_icc)
        if default_idx >= 0:
            self.combo_color_space.setCurrentIndex(default_idx)

        lay.addWidget(core_box)

        # -----------------------------
        # Format-specific panels
        # -----------------------------
        self.panel_jpg = self._build_jpg_panel(current_jpeg_quality)
        self.panel_tif = self._build_tif_panel()
        self.panel_fits = self._build_fits_panel(source_bottom_up)
        lay.addWidget(self.panel_jpg)
        lay.addWidget(self.panel_tif)
        lay.addWidget(self.panel_fits)

        # -----------------------------
        # Resize panel (applies to all formats)
        # -----------------------------
        self.panel_resize = self._build_resize_panel()
        lay.addWidget(self.panel_resize)

        self._apply_format_visibility()

    # ------------------------------------------------------------------
    # UI builders
    # ------------------------------------------------------------------
    def _fill_depths(self, prefer: str | None):
        allowed = list(_BIT_DEPTHS.get(self._ext, ["32-bit floating point"]))
        # "Same as document" only means something when the format has a
        # choice: PNG / JPEG / WebP are 8-bit only, so show just that.
        if self._keep_depth and len(allowed) > 1:
            allowed = [KEEP_DOC_DEPTH] + allowed
        self.combo_depth.blockSignals(True)
        try:
            self.combo_depth.clear()
            self.combo_depth.addItems(allowed)
            if prefer in allowed:
                self.combo_depth.setCurrentText(prefer)
        finally:
            self.combo_depth.blockSignals(False)

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

    def _build_fits_panel(self, source_bottom_up: bool | None) -> QWidget:
        box = QGroupBox(self.tr("FITS options"))
        form = QFormLayout(box)

        if source_bottom_up is None:
            same = self.tr("Same as source file")
        elif source_bottom_up:
            same = self.tr("Same as source file (bottom-up)")
        else:
            same = self.tr("Same as source file (top-down)")
        self.fits_row_order_combo = QComboBox(self)
        self.fits_row_order_combo.addItem(same, ROW_ORDER_SOURCE)
        self.fits_row_order_combo.addItem(self.tr("Top-down"), "top-down")
        self.fits_row_order_combo.addItem(self.tr("Bottom-up"), "bottom-up")
        self.fits_row_order_combo.setToolTip(self.tr(
            "The order the image rows are stored in. SASpro and PixInsight read both,\n"
            "using the ROWORDER keyword. Siril shows top-down files upside down, so\n"
            "use Bottom-up for files going to Siril.\n"
            "Same as source file: bottom-up when the image came from a bottom-up\n"
            "FITS (Siril's own files), top-down otherwise."))
        form.addRow(self.tr("Row order"), self.fits_row_order_combo)
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
    def target_ext(self) -> str:
        return self._ext

    def set_format(self, target_ext: str):
        """Re-point the panel at another output format: bit depths and the
        format-specific boxes follow; other values are kept."""
        self._ext = _norm_ext(target_ext)
        self._fill_depths(self.combo_depth.currentText())
        self._apply_format_visibility()

    def _apply_format_visibility(self):
        self.panel_jpg.setVisible(self._ext == "jpg")
        self.panel_tif.setVisible(self._ext == "tif")
        self.panel_fits.setVisible(self._ext == "fits" and self._row_order_option)

    def _update_resize_enable(self):
        mode = self.resize_group.checkedId()
        self.resize_percent_spin.setEnabled(mode == 1)
        self.resize_long_edge_spin.setEnabled(mode == 2)

    def _color_space_label(self, key: str) -> str:
        if key == "DisplayP3":
            return self.tr("Display P3")
        if key == "AdobeRGB":
            return self.tr("Adobe RGB (1998)")
        if key == "ProPhotoRGB":
            return self.tr("ProPhoto RGB")
        if key == "sRGB":
            return self.tr("sRGB")
        return str(key)

    def _color_space_description(self, key: str) -> str:
        if key == "DisplayP3":
            return self.tr("Wide-gamut color space with D65 white point and P3 primaries.")
        if key == "AdobeRGB":
            return self.tr("Wide-gamut photography color space using Adobe RGB (1998) primaries.")
        if key == "ProPhotoRGB":
            return self.tr("Very wide-gamut color space for professional color workflows.")
        if key == "sRGB":
            return self.tr("Standard RGB color space for web sharing and broad compatibility.")
        return str(key)

    def missing_profile_message(self, key: str) -> str:
        color_key = get_key_from_icc_key(key)
        info = COLOR_SPACES.get(color_key)
        names = ", ".join(info.profile_names[:4]) if info is not None else str(key)
        dirs = "\n".join(f"- {path}" for path in get_profile_search_directories())
        return self.tr(
            "{profile} ICC profile was not found.\n\n"
            "Expected names include: {names}\n\n"
            "SASpro checks these locations:\n{dirs}\n\n"
            "Display and export tagging will fall back to sRGB until the profile is installed."
        ).format(profile=self._color_space_label(color_key), names=names, dirs=dirs)

    def warn_if_profile_missing(self, parent=None):
        if self.chk_embed_icc.isChecked() and not is_profile_available(self.combo_color_space.currentData()):
            QMessageBox.warning(
                parent or self,
                self.tr("Color profile not found"),
                self.missing_profile_message(self.combo_color_space.currentData()),
            )

    # ------------------------------------------------------------------
    # Full widget state (for presets that are saved without the dialog)
    # ------------------------------------------------------------------
    def state(self) -> Dict[str, Any]:
        mode = self.resize_group.checkedId()
        return {
            "bit_depth": self.combo_depth.currentText(),
            "embed_icc": bool(self.chk_embed_icc.isChecked()),
            "icc_color_space": self.combo_color_space.currentData(),
            "rendering_intent": self.combo_rendering_intent.currentData(),
            "black_point_compensation": bool(self.chk_black_point_compensation.isChecked()),
            "jpeg_quality": int(self.jpeg_quality_spin.value()),
            "dpi": int(self.tiff_dpi_spin.value()),
            "tiff_comp": self.tiff_comp_combo.currentText(),
            "fits_row_order": self.fits_row_order_combo.currentData(),
            "resize_mode": {1: "percent", 2: "long_edge"}.get(mode, "none"),
            "resize_percent": int(self.resize_percent_spin.value()),
            "resize_long_edge": int(self.resize_long_edge_spin.value()),
        }

    def set_state(self, st: Dict[str, Any] | None):
        st = dict(st or {})
        try:
            bd = st.get("bit_depth")
            if bd:
                idx = self.combo_depth.findText(str(bd))
                if idx >= 0:
                    self.combo_depth.setCurrentIndex(idx)
            if "embed_icc" in st:
                self.chk_embed_icc.setChecked(bool(st["embed_icc"]))
            if st.get("icc_color_space"):
                cs = get_icc_key_from_color_space_key(get_key_from_icc_key(str(st["icc_color_space"])))
                idx = self.combo_color_space.findData(cs)
                if idx >= 0:
                    self.combo_color_space.setCurrentIndex(idx)
            self.combo_color_space.setEnabled(self.chk_embed_icc.isChecked())
            if st.get("rendering_intent"):
                idx = self.combo_rendering_intent.findData(st["rendering_intent"])
                if idx >= 0:
                    self.combo_rendering_intent.setCurrentIndex(idx)
            if "black_point_compensation" in st:
                self.chk_black_point_compensation.setChecked(bool(st["black_point_compensation"]))
            if "jpeg_quality" in st:
                self.jpeg_quality_spin.setValue(max(1, min(100, int(st["jpeg_quality"]))))
            if "dpi" in st:
                self.tiff_dpi_spin.setValue(max(1, int(st["dpi"])))
            if st.get("tiff_comp"):
                idx = self.tiff_comp_combo.findText(str(st["tiff_comp"]))
                if idx >= 0:
                    self.tiff_comp_combo.setCurrentIndex(idx)
            if st.get("fits_row_order") in _ROW_ORDERS:
                idx = self.fits_row_order_combo.findData(st["fits_row_order"])
                if idx >= 0:
                    self.fits_row_order_combo.setCurrentIndex(idx)
            mode = st.get("resize_mode")
            if mode == "percent":
                self.rb_resize_percent.setChecked(True)
            elif mode == "long_edge":
                self.rb_resize_long_edge.setChecked(True)
            elif mode is not None:
                self.rb_resize_none.setChecked(True)
            if "resize_percent" in st:
                self.resize_percent_spin.setValue(max(1, int(st["resize_percent"])))
            if "resize_long_edge" in st:
                self.resize_long_edge_spin.setValue(max(16, int(st["resize_long_edge"])))
            self._update_resize_enable()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Settings persistence (Save As remembers per-format defaults)
    # The FITS row order is deliberately NOT remembered: it starts at "Same
    # as source file" every time, so each image follows its own file.
    # ------------------------------------------------------------------
    def _k(self, suffix: str) -> str:
        # per-format persistence
        return f"export/{self._ext}/{suffix}"

    def load_settings(self):
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

            cs = s.value(
                self._k("icc_color_space"),
                get_icc_key_from_color_space_key(get_working_color_space_from_settings(s)),
                type=str,
            ) or "P3"
            cs = get_icc_key_from_color_space_key(get_key_from_icc_key(cs))
            _cs_idx = self.combo_color_space.findData(cs)
            if _cs_idx >= 0:
                self.combo_color_space.setCurrentIndex(_cs_idx)
            self.combo_color_space.setEnabled(self.chk_embed_icc.isChecked())

            intent = s.value(self._k("rendering_intent"), get_softproof_rendering_intent_from_settings(s), type=str)
            intent_idx = self.combo_rendering_intent.findData(intent)
            if intent_idx >= 0:
                self.combo_rendering_intent.setCurrentIndex(intent_idx)

            bpc = s.value(self._k("black_point_compensation"), get_softproof_black_point_compensation_from_settings(s), type=bool)
            self.chk_black_point_compensation.setChecked(bool(bpc))

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

    def save_settings(self):
        s = self._settings
        if s is None:
            return
        try:
            s.setValue(self._k("bit_depth"), self.combo_depth.currentText())
            s.setValue(self._k("embed_icc"), bool(self.chk_embed_icc.isChecked()))
            s.setValue(self._k("icc_color_space"), self.combo_color_space.currentData())
            s.setValue(self._k("rendering_intent"), self.combo_rendering_intent.currentData())
            s.setValue(self._k("black_point_compensation"), bool(self.chk_black_point_compensation.isChecked()))

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
    def selected_bit_depth(self) -> Optional[str]:
        """The chosen depth; None when 'Same as document' is selected."""
        return bit_depth_from_state({"bit_depth": self.combo_depth.currentText()})

    def selected_jpeg_quality(self) -> Optional[int]:
        if self._ext != "jpg":
            return None
        return int(self.jpeg_quality_spin.value())

    def export_options(self) -> Dict[str, Any]:
        """
        Options consumed by docman.save_document(..., export_opts=...)
        and then legacy_save_image(..., export_opts=...)
        """
        return export_opts_from_state(self._ext, self.state())


class ExportDialog(QDialog):
    """
    "Export"-style options dialog (Save As and friends). The options live in
    ExportOptionsPanel, shared with Checkpoint Save / Export.

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
        source_bottom_up: bool | None = None,   # see ExportOptionsPanel
        row_order_option: bool = True,
    ):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Export"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)

        self._settings = settings
        self._ext = _norm_ext(target_ext)

        # ---- layout root ----
        lay = QVBoxLayout(self)

        # Format summary
        fmt_lbl = QLabel(self.tr("Export format: {0}").format(self._ext.upper()))
        fmt_lbl.setWordWrap(True)
        lay.addWidget(fmt_lbl)

        # All options (bit depth, colour, JPEG/TIFF, resize)
        self.panel = ExportOptionsPanel(
            self._ext, current_bit_depth, current_jpeg_quality,
            settings=settings, source_bottom_up=source_bottom_up,
            row_order_option=row_order_option, parent=self,
        )
        lay.addWidget(self.panel)
        # Older code reached into the dialog for these widgets.
        self.combo_depth = self.panel.combo_depth
        self.jpeg_quality_spin = self.panel.jpeg_quality_spin

        # Buttons
        btn_ok = QPushButton(self.tr("Export"))
        btn_cancel = QPushButton(self.tr("Cancel"))
        btn_ok.clicked.connect(self._on_accept)
        btn_cancel.clicked.connect(self.reject)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        lay.addLayout(row)

        # ---- load persisted defaults (optional) ----
        self.panel.load_settings()

    def _on_accept(self):
        self.panel.warn_if_profile_missing(self)
        # persist settings if present
        self.panel.save_settings()
        self.accept()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def selected_bit_depth(self) -> str:
        return self.panel.combo_depth.currentText()

    def selected_jpeg_quality(self) -> Optional[int]:
        return self.panel.selected_jpeg_quality()

    def export_options(self) -> Dict[str, Any]:
        return self.panel.export_options()
