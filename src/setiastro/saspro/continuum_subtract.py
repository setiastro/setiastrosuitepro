# pro/continuum_subtract.py
from __future__ import annotations
from setiastro.saspro.main_helpers import non_blocking_sleep
import os
import numpy as np

# Optional deps used by the processing threads
try:
    import cv2
except Exception:
    cv2 = None

try:
    import pywt
except Exception:
    pywt = None

from PyQt6.QtCore import (
    Qt, QSize, QPoint, QEvent, QThread, pyqtSignal, QTimer,
    QCoreApplication
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QScrollArea, QDialog, QInputDialog, QFileDialog,
    QMessageBox, QCheckBox, QApplication, QMainWindow, QCheckBox
)
from PyQt6.QtGui import (
    QPixmap, QImage, QCursor, QWheelEvent
)

# register QImage for cross-thread signals
#qRegisterMetaType(QImage)

from .doc_manager import ImageDocument  # add this import
from setiastro.saspro.legacy.image_manager import load_image as legacy_load_image, save_image as legacy_save_image  # CHANGED
from setiastro.saspro.imageops.stretch import stretch_mono_image, stretch_color_image
from setiastro.saspro.imageops.starbasedwhitebalance import apply_star_based_white_balance
from setiastro.saspro.legacy.numba_utils import apply_curves_numba

from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
# At the top of continuum_subtract.py, with the other imports:
try:
    from setiastro.saspro.cosmicclarity_headless import run_cosmicclarity_on_array as _cc_denoise
except Exception:
    _cc_denoise = None

def apply_curves_adjustment(image, target_median, curves_boost):
    """
    Original signature unchanged, but now uses a Numba helper
    to do the pixel-by-pixel interpolation.

    'image' can be 2D (H,W) or 3D (H,W,3).
    """
    # Build the curve array as before
    curve = [
        [0.0, 0.0],
        [0.5 * target_median, 0.5 * target_median],
        [target_median, target_median],
        [
            (1/4 * (1 - target_median) + target_median),
            np.power((1/4 * (1 - target_median) + target_median), (1 - curves_boost))
        ],
        [
            (3/4 * (1 - target_median) + target_median),
            np.power(np.power((3/4 * (1 - target_median) + target_median), (1 - curves_boost)), (1 - curves_boost))
        ],
        [1.0, 1.0]
    ]
    # Convert to arrays
    xvals = np.array([p[0] for p in curve], dtype=np.float32)
    yvals = np.array([p[1] for p in curve], dtype=np.float32)

    # Ensure 'image' is float32
    image_32 = image.astype(np.float32, copy=False)

    # Now apply the piecewise linear function in Numba
    adjusted_image = apply_curves_numba(image_32, xvals, yvals)
    return adjusted_image

class ContinuumSubtractTab(QWidget):
    def __init__(self, doc_manager, document=None, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.doc_manager = doc_manager
        self.initUI()
        self._threads = []
        # — initialize every loadable image to None —
        self.ha_image    = None
        self.sii_image   = None
        self.oiii_image  = None
        self.red_image   = None
        self.green_image = None
        self.osc_image   = None
        # NEW: composite HaO3 / S2O3 "source" images (optional)
        self.hao3_image          = None
        self.s2o3_image          = None
        self.hao3_starless_image = None
        self.s2o3_starless_image = None

        # NEW: OIII components extracted from composites (for averaging)
        self._o3_from_hao3          = None
        self._o3_from_s2o3          = None
        self._o3_from_hao3_starless = None
        self._o3_from_s2o3_starless = None
        self.filename = None
        self.is_mono = True
        self.combined_image = None
        self.processing_thread = None
        self.original_header = None
        self._clickable_images = {}
        self._wb_diag_entries = []   # list of dicts built in _onOneResult
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception as e:
            print(f"ContinuumSubtractTab init error: {e}")

    def initUI(self):
        self.spinnerLabel = QLabel("")
        self.spinnerLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spinnerLabel.setStyleSheet("color:#999; font-style:italic;")
        self.spinnerLabel.hide()

        # starless image slots
        self.ha_starless_image    = None
        self.sii_starless_image   = None
        self.oiii_starless_image  = None
        self.red_starless_image   = None
        self.green_starless_image = None
        self.osc_starless_image   = None

        self.statusLabel = QLabel("")
        self.statusLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        from PyQt6.QtWidgets import QGridLayout, QSizePolicy

        main_layout = QVBoxLayout()
        top_cols = QHBoxLayout()
        top_cols.setSpacing(8)

        # ── helper: build one filter block (starry + starless rows) ──────────
        def _filter_block(tag, attr, parent_layout, is_composite=False):
            """
            Adds two rows to parent_layout (a QVBoxLayout):
            row 1: [tag lbl] [filename lbl] [View btn] [File btn]   ← starry
            row 2: [+sl lbl] [filename lbl] [View btn] [File btn]   ← starless
            Wires View/File buttons to loadImage().
            Stores refs as self.<attr>Label, self.<attr>StarlessLabel etc.
            """
            for starless in (False, True):
                row = QHBoxLayout()
                row.setSpacing(4)
                row.setContentsMargins(0, 0, 0, 0)

                tag_lbl = QLabel("+starless" if starless else tag)
                tag_lbl.setFixedWidth(56)
                tag_lbl.setStyleSheet(
                    "font-size:10px; color: palette(placeholderText);" if starless
                    else "font-size:11px; font-weight:500;"
                )
                row.addWidget(tag_lbl)

                fname_lbl = QLabel("no file")
                fname_lbl.setStyleSheet("font-size:11px; font-style:italic; color: palette(placeholderText);")
                fname_lbl.setSizePolicy(
                    QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
                )
                fname_lbl.setMinimumWidth(0)
                row.addWidget(fname_lbl, 1)

                suffix = " (Starless)" if starless else ""
                channel = f"{tag}{suffix}"

                btn_view = QPushButton("View")
                btn_file = QPushButton("File")
                for b in (btn_view, btn_file):
                    b.setFixedWidth(36)
                    b.setStyleSheet("font-size:10px; padding: 1px 4px;")
                btn_view.clicked.connect(lambda _, c=channel: self._load_from_view(c))
                btn_file.clicked.connect(lambda _, c=channel: self._load_from_file(c))
                row.addWidget(btn_view)
                row.addWidget(btn_file)

                parent_layout.addLayout(row)

                # store refs
                attr_name = f"{attr}{'Starless' if starless else ''}Label"
                setattr(self, attr_name, fname_lbl)

            # thin divider after each block (caller adds it)

        # ── Narrowband group ─────────────────────────────────────────────────
        nb_group = QGroupBox("Narrowband Filters")
        nb_l = QVBoxLayout()
        nb_l.setSpacing(3)
        nb_l.setContentsMargins(8, 8, 8, 8)

        for tag, attr in [("Ha","ha"),("SII","sii"),("OIII","oiii"),("HaO3","hao3"),("S2O3","s2o3")]:
            _filter_block(tag, attr, nb_l)
            sep = QWidget(); sep.setFixedHeight(1)
            sep.setStyleSheet("background: palette(dark);")
            nb_l.addWidget(sep)

        nb_l.addStretch(1)
        nb_group.setLayout(nb_l)

        # ── Middle column: Continuum + Processing Options ─────────────────────
        mid_col = QVBoxLayout()
        mid_col.setSpacing(8)

        cont_group = QGroupBox("Continuum Sources")
        cont_l = QVBoxLayout()
        cont_l.setSpacing(3)
        cont_l.setContentsMargins(8, 8, 8, 8)

        for tag, attr in [("Red","red"),("Green","green"),("OSC","osc")]:
            _filter_block(tag, attr, cont_l)
            sep = QWidget(); sep.setFixedHeight(1)
            sep.setStyleSheet("background: palette(mid);")
            cont_l.addWidget(sep)

        cont_l.addStretch(1)
        cont_group.setLayout(cont_l)
        mid_col.addWidget(cont_group)

        # ── Processing Options ────────────────────────────────────────────────
        opts_group = QGroupBox("Processing Options")
        opts_l = QVBoxLayout()
        opts_l.setSpacing(2)
        opts_l.setContentsMargins(8, 8, 8, 8)

        self.linear_output_checkbox = QCheckBox("Output linear only")
        opts_l.addWidget(self.linear_output_checkbox)

        self.denoise_checkbox = QCheckBox("Denoise with Cosmic Clarity")
        self.denoise_checkbox.setChecked(True)
        self.denoise_checkbox.setToolTip(
            "Runs Cosmic Clarity denoise on the linear continuum-subtracted image "
            "before any non-linear stretch."
        )
        opts_l.addWidget(self.denoise_checkbox)

        # strength row (indented)
        str_row = QHBoxLayout()
        str_row.setContentsMargins(19, 0, 0, 0)
        str_row.setSpacing(6)
        self.denoise_strength_label = QLabel("Strength")
        self.denoise_strength_label.setStyleSheet("font-size:11px;")
        self.denoise_strength_btn = QPushButton("0.90")
        self.denoise_strength_btn.setFixedWidth(46)
        self.denoise_strength_btn.setStyleSheet("font-size:11px; padding:1px 4px;")
        self.denoise_strength_btn.clicked.connect(self._change_denoise_strength)
        self.denoise_strength = 0.90
        str_row.addWidget(self.denoise_strength_label)
        str_row.addWidget(self.denoise_strength_btn)
        str_row.addStretch(1)
        opts_l.addLayout(str_row)

        # wire enable/disable
        def _sync_denoise(checked):
            self.denoise_strength_btn.setEnabled(checked)
            self.denoise_strength_label.setEnabled(checked)
        self.denoise_checkbox.toggled.connect(_sync_denoise)
        _sync_denoise(True)

        # separator
        sep2 = QWidget(); sep2.setFixedHeight(1)
        sep2.setStyleSheet("background: palette(mid);")
        opts_l.addWidget(sep2)
        opts_l.addSpacing(2)

        self.stretch_checkbox = QCheckBox("Auto-stretch")
        self.stretch_checkbox.setChecked(True)
        self.stretch_checkbox.setToolTip(
            "Stretch the linear result to non-linear. "
            "When unchecked, always outputs linear regardless of the output linear checkbox."
        )
        opts_l.addWidget(self.stretch_checkbox)

        # stretch sub-options widget (indented, disabled when stretch off)
        self.stretch_subopts = QWidget()
        sub_l = QVBoxLayout(self.stretch_subopts)
        sub_l.setContentsMargins(19, 0, 0, 0)
        sub_l.setSpacing(2)

        self.median_sub_checkbox = QCheckBox("Median background subtract")
        self.median_sub_checkbox.setChecked(True)
        self.median_sub_checkbox.setToolTip(
            "Subtracts 0.7× median after stretch. "
            "Uncheck to keep a cleaner background."
        )
        sub_l.addWidget(self.median_sub_checkbox)

        self.curves_checkbox = QCheckBox("Curves boost")
        self.curves_checkbox.setChecked(False)
        self.curves_checkbox.setToolTip("Applies a gentle S-curve boost after stretching.")
        sub_l.addWidget(self.curves_checkbox)

        self.no_black_clip_checkbox = QCheckBox("No black clip")
        self.no_black_clip_checkbox.setChecked(True)
        self.no_black_clip_checkbox.setToolTip(
            "Uses image minimum as black point instead of median − σ. "
            "Preserves faint background signal."
        )
        sub_l.addWidget(self.no_black_clip_checkbox)

        opts_l.addWidget(self.stretch_subopts)

        def _sync_stretch(checked):
            self.stretch_subopts.setEnabled(checked)
        self.stretch_checkbox.toggled.connect(_sync_stretch)
        _sync_stretch(True)

        # separator + Advanced collapsed row
        sep3 = QWidget(); sep3.setFixedHeight(1)
        sep3.setStyleSheet("background: palette(mid);")
        opts_l.addWidget(sep3)
        opts_l.addSpacing(2)

        # defaults
        self.threshold_value = 5.0
        self.q_factor        = 0.80
        self.summary_gamma   = 0.6

        adv_hdr = QHBoxLayout()
        adv_hdr.setContentsMargins(0, 0, 0, 0)
        self.advanced_btn = QPushButton("Advanced ▸")
        self.advanced_btn.setFlat(True)
        self.advanced_btn.setStyleSheet("font-size:11px; text-align:left;")
        self.advanced_btn.clicked.connect(self._toggle_advanced)

        self.adv_summary_label = QLabel(
            f"Q {self.q_factor:.2f}   WB σ {self.threshold_value:.1f}"
        )
        self.adv_summary_label.setStyleSheet("font-size:10px; color: palette(placeholderText);")
        adv_hdr.addWidget(self.advanced_btn)
        adv_hdr.addStretch(1)
        adv_hdr.addWidget(self.adv_summary_label)
        opts_l.addLayout(adv_hdr)

        self.advanced_panel = QWidget()
        adv_l = QVBoxLayout(self.advanced_panel)
        adv_l.setContentsMargins(4, 0, 0, 0)
        adv_l.setSpacing(4)

        thr_row = QHBoxLayout()
        self.threshold_label = QLabel(f"WB star detect threshold: {self.threshold_value:.1f}")
        self.threshold_label.setStyleSheet("font-size:11px;")
        self.threshold_btn = QPushButton("Change…")
        self.threshold_btn.setStyleSheet("font-size:11px;")
        self.threshold_btn.clicked.connect(self._change_threshold)
        thr_row.addWidget(self.threshold_label)
        thr_row.addWidget(self.threshold_btn)
        adv_l.addLayout(thr_row)

        q_row = QHBoxLayout()
        self.q_label = QLabel(f"Continuum Q factor: {self.q_factor:.2f}")
        self.q_label.setStyleSheet("font-size:11px;")
        self.q_btn = QPushButton("Change…")
        self.q_btn.setStyleSheet("font-size:11px;")
        self.q_btn.clicked.connect(self._change_q)
        q_row.addWidget(self.q_label)
        q_row.addWidget(self.q_btn)
        adv_l.addLayout(q_row)

        self.advanced_panel.setVisible(False)
        opts_l.addWidget(self.advanced_panel)

        opts_group.setLayout(opts_l)
        mid_col.addWidget(opts_group, 1)

        # ── WB Diagnostics (right column) ────────────────────────────────────


        # ── Assemble top columns ──────────────────────────────────────────────
        top_cols.addWidget(nb_group, 2)

        mid_widget = QWidget()
        mid_widget.setLayout(mid_col)
        top_cols.addWidget(mid_widget, 2)

        # ── Bottom row ────────────────────────────────────────────────────────
        bottom_row = QHBoxLayout()
        self.execute_button = QPushButton("Execute")
        self.execute_button.clicked.connect(self.startContinuumSubtraction)
        self.clear_button = QPushButton("Clear Images")
        self.clear_button.clicked.connect(self.clear_loaded_images)

        bottom_row.addWidget(self.execute_button)
        bottom_row.addWidget(self.clear_button)
        bottom_row.addWidget(self.spinnerLabel)
        bottom_row.addWidget(self.statusLabel, 1)
        # In bottom_row, after clear_button:
        self.wb_diag_button = QPushButton("WB Diagnostics…")
        self.wb_diag_button.clicked.connect(self._show_wb_diagnostics)
        self.wb_diag_button.setEnabled(False)  # enabled after first result
        bottom_row.addWidget(self.wb_diag_button)
        main_layout.addLayout(top_cols)
        main_layout.addLayout(bottom_row)
        self.setLayout(main_layout)
        self.installEventFilter(self)

    def _apply_loaded_image(self, channel: str, image, header, bit_depth, is_mono, name_or_path):
        """Common handler after loading from either View or File."""
        label_text = str(name_or_path) if name_or_path else "From View"
        try:
            if isinstance(name_or_path, str) and os.path.isabs(name_or_path):
                label_text = os.path.basename(name_or_path)
        except Exception:
            pass

        is_starless = "(Starless)" in channel
        base = channel.replace(" (Starless)", "")

        if base == "Ha":
            if is_starless:
                self.ha_starless_image = image
                self.haStarlessLabel.setText(label_text)
            else:
                self.ha_image = image
                self.haLabel.setText(label_text)

        elif base == "SII":
            if is_starless:
                self.sii_starless_image = image
                self.siiStarlessLabel.setText(label_text)
            else:
                self.sii_image = image
                self.siiLabel.setText(label_text)

        elif base == "OIII":
            if is_starless:
                self.oiii_starless_image = image
                self.oiiiStarlessLabel.setText(label_text)
            else:
                self.oiii_image = image
                self.oiiiLabel.setText(label_text)

        elif base == "Red":
            if is_starless:
                self.red_starless_image = image
                self.redStarlessLabel.setText(label_text)
            else:
                self.red_image = image
                self.redLabel.setText(label_text)

        elif base == "Green":
            if is_starless:
                self.green_starless_image = image
                self.greenStarlessLabel.setText(label_text)
            else:
                self.green_image = image
                self.greenLabel.setText(label_text)

        elif base == "OSC":
            if is_starless:
                self.osc_starless_image = image
                self.oscStarlessLabel.setText(label_text)
            else:
                self.osc_image = image
                self.oscLabel.setText(label_text)

        elif base == "HaO3":
            if is_starless:
                self.hao3_starless_image = image
                self.hao3StarlessLabel.setText(label_text)
            else:
                self.hao3_image = image
                self.hao3Label.setText(label_text)

            if not (isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] >= 2):
                QMessageBox.warning(self, "HaO3 Load",
                    "HaO3 expects a 3-channel color image (R=Ha, G=OIII). "
                    "Loaded image is not 3-channel; cannot extract Ha/OIII.")
                return

            img32 = image.astype(np.float32, copy=False)
            ha_from_r = img32[..., 0]
            o3_from_g = img32[..., 1]

            if is_starless:
                self.ha_starless_image = ha_from_r
                self.haStarlessLabel.setText(label_text + " [R → Ha (starless)]")
                self._o3_from_hao3_starless = o3_from_g
                self._update_oiii_from_composites(starless=True)
            else:
                self.ha_image = ha_from_r
                self.haLabel.setText(label_text + " [R → Ha]")
                self._o3_from_hao3 = o3_from_g
                self._update_oiii_from_composites(starless=False)

        elif base == "S2O3":
            if is_starless:
                self.s2o3_starless_image = image
                self.s2o3StarlessLabel.setText(label_text)
            else:
                self.s2o3_image = image
                self.s2o3Label.setText(label_text)

            if not (isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] >= 2):
                QMessageBox.warning(self, "S2O3 Load",
                    "S2O3 expects a 3-channel color image (R=SII, G=OIII). "
                    "Loaded image is not 3-channel; cannot extract SII/OIII.")
                return

            img32 = image.astype(np.float32, copy=False)
            s2_from_r = img32[..., 0]
            o3_from_g = img32[..., 1]

            if is_starless:
                self.sii_starless_image = s2_from_r
                self.siiStarlessLabel.setText(label_text + " [R → SII (starless)]")
                self._o3_from_s2o3_starless = o3_from_g
                self._update_oiii_from_composites(starless=True)
            else:
                self.sii_image = s2_from_r
                self.siiLabel.setText(label_text + " [R → SII]")
                self._o3_from_s2o3 = o3_from_g
                self._update_oiii_from_composites(starless=False)

        else:
            QMessageBox.critical(self, "Error", f"Unknown channel '{channel}'.")
            return

        self.original_header = header
        self.is_mono         = is_mono

    def _load_from_view(self, channel: str):
        result = self.loadImageFromView(channel)
        if result:
            self._apply_loaded_image(channel, *result)

    def _load_from_file(self, channel: str):
        result = self.loadImageFromFile(channel)
        if result:
            self._apply_loaded_image(channel, *result)

    def _change_denoise_strength(self):
        val, ok = QInputDialog.getDouble(
            self,
            "Denoise Strength",
            "Cosmic Clarity denoise strength (0.0–1.0):",
            self.denoise_strength,
            0.0, 1.0, 2
        )
        if ok:
            self.denoise_strength = float(val)
            self.denoise_strength_btn.setText(f"{self.denoise_strength:.2f}")

    def _toggle_advanced(self):
        show = not self.advanced_panel.isVisible()
        self.advanced_panel.setVisible(show)
        self.advanced_btn.setText("Advanced ▾" if show else "Advanced ▸")
        self.adv_summary_label.setVisible(not show)

    def _change_q(self):
        val, ok = QInputDialog.getDouble(
            self,
            "Continuum Q Factor",
            "Q (scale of broadband subtraction, typical 0.6–1.0):",
            self.q_factor,
            0.10, 2.00, 2
        )
        if ok:
            self.q_factor = float(val)
            self.q_label.setText(f"Continuum Q factor: {self.q_factor:.2f}")

        self.adv_summary_label.setText(f"Q {self.q_factor:.2f}   WB σ {self.threshold_value:.1f}")

    def _change_threshold(self):
        val, ok = QInputDialog.getDouble(
            self,
            "WB Threshold",
            "Sigma threshold for star detection:",
            self.threshold_value,
            0.5, 50.0, 1
        )
        if ok:
            self.threshold_value = float(val)
            self.threshold_label.setText(f"WB star detect threshold: {self.threshold_value:.1f}")

        self.adv_summary_label.setText(f"Q {self.q_factor:.2f}   WB σ {self.threshold_value:.1f}")


    def _main_window(self) -> QMainWindow | None:
        # 1) explicit parent the tool may have been created with
        mw = self.parent_window
        if mw and hasattr(mw, "mdi"):
            return mw
        # 2) walk up the parent chain
        p = self.parent()
        while p is not None:
            if hasattr(p, "mdi"):
                return p  # main window
            p = p.parent()
        # 3) search top-level widgets
        for w in QApplication.topLevelWidgets():
            if hasattr(w, "mdi"):
                return w
        return None

    def _iter_open_docs(self):
        """Yield (doc, title) for all open subwindows."""
        mw = self._main_window()
        if not mw or not hasattr(mw, "mdi"):
            return []
        out = []
        for sw in mw.mdi.subWindowList():
            w = sw.widget()
            d = getattr(w, "document", None)
            if d is not None:
                out.append((d, sw.windowTitle()))
        return out


    def refresh(self):
        if self.image_manager:
            # You might have a way to retrieve the current image and metadata.
            # For example, if your image_manager stores the current image,
            # you could do something like:
            return



    def clear_loaded_images(self):
        for attr in (
            "ha_image","sii_image","oiii_image","red_image","green_image","osc_image",
            "ha_starless_image","sii_starless_image","oiii_starless_image",
            "red_starless_image","green_starless_image","osc_starless_image",
            # NEW composite attrs
            "hao3_image","s2o3_image",
            "hao3_starless_image","s2o3_starless_image"
        ):
            setattr(self, attr, None)

        # Reset NB labels
        self.haLabel.setText("No Ha")
        self.siiLabel.setText("No SII")
        self.oiiiLabel.setText("No OIII")
        # NEW composite labels
        self.hao3Label.setText("No HaO3")
        self.s2o3Label.setText("No S2O3")
        self._wb_diag_entries = []
        if hasattr(self, "wb_diag_button"):
            self.wb_diag_button.setEnabled(False)
        # Reset continuum labels
        self.redLabel.setText("No Red")
        self.greenLabel.setText("No Green")
        self.oscLabel.setText("No OSC")

        self.haStarlessLabel.setText("No Ha (starless)")
        self.siiStarlessLabel.setText("No SII (starless)")
        self.oiiiStarlessLabel.setText("No OIII (starless)")
        self.redStarlessLabel.setText("No Red (starless)")
        self.greenStarlessLabel.setText("No Green (starless)")
        self.oscStarlessLabel.setText("No OSC (starless)")

        # NEW: clear OIII-from-composite caches
        self._o3_from_hao3 = None
        self._o3_from_s2o3 = None
        self._o3_from_hao3_starless = None
        self._o3_from_s2o3_starless = None

        self.combined_image = None
        self.statusLabel.setText("All loaded images cleared.")


    # --- NEW: helper to combine OIII from HaO3 and S2O3 ---
    def _update_oiii_from_composites(self, starless: bool):
        """
        Average all available composite-derived green channels into a single OIII NB image.
        Only averages HaO3+S2O3 (does NOT touch any manually loaded OIII).
        """
        sources = []
        labels = []

        if starless:
            if self._o3_from_hao3_starless is not None:
                sources.append(self._o3_from_hao3_starless)
                labels.append("HaO3")
            if self._o3_from_s2o3_starless is not None:
                sources.append(self._o3_from_s2o3_starless)
                labels.append("S2O3")
        else:
            if self._o3_from_hao3 is not None:
                sources.append(self._o3_from_hao3)
                labels.append("HaO3")
            if self._o3_from_s2o3 is not None:
                sources.append(self._o3_from_s2o3)
                labels.append("S2O3")

        if not sources:
            return

        try:
            combo = np.mean(np.stack(sources, axis=0), axis=0).astype(np.float32, copy=False)
        except ValueError:
            # shape mismatch – fall back to last one
            combo = sources[-1]

        if starless:
            self.oiii_starless_image = combo
            base_label = "OIII from " + "+".join(labels) + " (starless)"
            self.oiiiStarlessLabel.setText(base_label)
        else:
            self.oiii_image = combo
            base_label = "OIII from " + "+".join(labels)
            self.oiiiLabel.setText(base_label)


    def _collect_open_documents(self):
        # kept for compatibility with callers; returns only docs
        return [d for d, _ in self._iter_open_docs()]

    def _select_document_via_dropdown(self, title: str):
        items = self._iter_open_docs()
        if not items:
            QMessageBox.information(self, f"Select View — {title}", "No open views/documents found.")
            return None

        # default to active view if present
        mw = self._main_window()
        active_doc = None
        if mw and mw.mdi.activeSubWindow():
            active_doc = getattr(mw.mdi.activeSubWindow().widget(), "document", None)

        if len(items) == 1:
            return items[0][0]

        names = [t for _, t in items]
        default_index = next((i for i, (d, _) in enumerate(items) if d is active_doc), 0)

        choice, ok = QInputDialog.getItem(
            self, f"Select View — {title}", "Choose:", names, default_index, False
        )
        if not ok:
            return None
        return items[names.index(choice)][0]

    def _image_from_doc(self, doc):
        """(np.ndarray, header, bit_depth, is_mono, file_path) from an ImageDocument."""
        arr = getattr(doc, "image", None)
        if arr is None:
            QMessageBox.warning(self, "No image", "Selected view has no image.")
            return None
        meta = getattr(doc, "metadata", {}) or {}
        header = meta.get("original_header") or meta.get("fits_header") or meta.get("header")
        bit_depth = meta.get("bit_depth", "Unknown")
        is_mono = False
        try:
            import numpy as np
            is_mono = isinstance(arr, np.ndarray) and (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1))
        except Exception:
            pass
        return arr, header, bit_depth, is_mono, meta.get("file_path")

    def loadImageFromView(self, channel: str):
        doc = self._select_document_via_dropdown(channel)
        if not doc:
            return None
        res = self._image_from_doc(doc)
        if not res:
            return None

        img, header, bit_depth, is_mono, _ = res

        # Build a human-friendly name for the label (view/subwindow title)
        title = ""
        try:
            title = doc.display_name()
        except Exception:
            mw = self._main_window()
            if mw and mw.mdi.activeSubWindow():
                title = mw.mdi.activeSubWindow().windowTitle()

        # Return with the "path" field set to the title so the caller can label it
        return img, header, bit_depth, is_mono, title


    def loadImageFromFile(self, channel: str):
        file_filter = "Images (*.png *.tif *.tiff *.fits *.fit *.xisf)"
        path, _ = QFileDialog.getOpenFileName(self, f"Select {channel} Image", "", file_filter)
        if not path:
            return None
        try:
            image, header, bit_depth, is_mono = legacy_load_image(path)  # ← use the alias
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load {channel} image:\n{e}")
            return None
        return image, header, bit_depth, is_mono, path

    def startContinuumSubtraction(self):
        # STARRED (with stars) continuum channels
        cont_red_starry   = self.red_image   if self.red_image   is not None else (self.osc_image[..., 0] if self.osc_image is not None else None)
        cont_green_starry = self.green_image if self.green_image is not None else (self.osc_image[..., 1] if self.osc_image is not None else None)

        # STARLESS continuum channels
        cont_red_starless   = self.red_starless_image   if self.red_starless_image   is not None else (self.osc_starless_image[..., 0] if self.osc_starless_image is not None else None)
        cont_green_starless = self.green_starless_image if self.green_starless_image is not None else (self.osc_starless_image[..., 1] if self.osc_starless_image is not None else None)

        # Build tasks per NB filter
        pairs = []
        def add_pair(name, nb_starry, cont_starry, nb_starless, cont_starless):
            has_starry   = (nb_starry is not None and cont_starry is not None)
            has_starless = (nb_starless is not None and cont_starless is not None)
            if has_starry or has_starless:
                pairs.append({
                    "name": name,
                    "nb": nb_starry,
                    "cont": cont_starry,
                    "nb_sl": nb_starless,
                    "cont_sl": cont_starless,
                    "starless_only": (has_starless and not has_starry),
                })

        add_pair("Ha",   self.ha_image,   cont_red_starry,   self.ha_starless_image,   cont_red_starless)
        add_pair("SII",  self.sii_image,  cont_red_starry,   self.sii_starless_image,  cont_red_starless)
        add_pair("OIII", self.oiii_image, cont_green_starry, self.oiii_starless_image, cont_green_starless)

        if not pairs:
            self.statusLabel.setText("Load at least one NB + matching continuum channel (or OSC).")
            return
        mw = self._main_window()

        denoise_linear    = self.denoise_checkbox.isChecked()
        denoise_strength  = self.denoise_strength
        do_stretch        = self.stretch_checkbox.isChecked()
        do_median_sub     = self.median_sub_checkbox.isChecked()
        do_curves         = self.curves_checkbox.isChecked()
        no_black_clip     = self.no_black_clip_checkbox.isChecked()
        self.showSpinner()
        self._threads = []
        self._results = []
        self._pushed_results = False
        self._pending = 0

        # How many result signals do we expect in total?
        self._expected_results = sum(
            (1 if p["nb"]    is not None and p["cont"]    is not None else 0) +
            (1 if p["nb_sl"] is not None and p["cont_sl"] is not None else 0)
            for p in pairs
        )

        for p in pairs:
            t = ContinuumProcessingThread(
                p["nb"], p["cont"], self.linear_output_checkbox.isChecked(),
                starless_nb=p["nb_sl"], starless_cont=p["cont_sl"], starless_only=p["starless_only"],
                threshold=self.threshold_value, summary_gamma=self.summary_gamma, q_factor=self.q_factor,
                denoise_linear=denoise_linear, denoise_strength=denoise_strength,
                do_stretch=do_stretch, do_median_sub=do_median_sub,
                do_curves=do_curves, no_black_clip=no_black_clip,
            )
            name = p["name"]  # avoid late binding in lambdas

            if p["nb"] is not None and p["cont"] is not None:
                self._pending += 1
                t.processing_complete.connect(
                    lambda img, stars, overlay, raw, after, n=f"{name} (starry)":
                        self._onOneResult(n, img, stars, overlay, raw, after)
                )

            if p["nb_sl"] is not None and p["cont_sl"] is not None:
                self._pending += 1
                t.processing_complete_starless.connect(
                    lambda img, stars, overlay, raw, after, n=f"{name} (starless)":
                        self._onOneResult(n, img, stars, overlay, raw, after)
                )

            t.status_update.connect(self.update_status_label)
            self._threads.append(t)
            t.start()


    def _onOneResult(self, filt, img, star_count, overlay_qimg, raw_pixels, after_pixels):
        self._results.append({
            "filter": filt, "image": img, "stars": star_count,
            "overlay": overlay_qimg, "raw": raw_pixels, "after": after_pixels
        })

        # store full-res pixmaps for the diagnostics dialog
        make_scatter = (
            isinstance(raw_pixels, np.ndarray) and
            raw_pixels.ndim == 2 and raw_pixels.shape[1] >= 2 and
            raw_pixels.shape[0] >= 3 and cv2 is not None
        )

        scatter_pix = None
        if make_scatter:
            nb_flux   = raw_pixels[:, 0].astype(np.float32, copy=False)
            cont_flux = raw_pixels[:, 1].astype(np.float32, copy=False)
            h, w = 256, 256
            scatter_img = np.ones((h, w, 3), np.uint8) * 255
            try:
                m, c = np.polyfit(cont_flux, nb_flux, 1)
                x0 = int(0.0 * (w-1)); y0 = int((1 - float(np.clip(c, 0, 1))) * (h-1))
                x1 = int(1.0 * (w-1)); y1 = int((1 - float(np.clip(m+c, 0, 1))) * (h-1))
                cv2.line(scatter_img, (x0, y0), (x1, y1), (0, 0, 255), 2)
            except Exception:
                pass
            xs = (np.clip(cont_flux, 0, 1) * (w-1)).astype(int)
            ys = ((1 - np.clip(nb_flux, 0, 1)) * (h-1)).astype(int)
            for x, y in zip(xs, ys):
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(scatter_img, (x, y), 2, (255, 0, 0), -1)
            cv2.line(scatter_img, (0, h-1), (w-1, h-1), (0,0,0), 1)
            cv2.line(scatter_img, (0, 0), (0, h-1), (0,0,0), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            ((tw,_),_) = cv2.getTextSize("BB Flux", font, 0.5, 1)
            cv2.putText(scatter_img, "BB Flux", ((w-tw)//2, h-5), font, 0.5, (0,0,0), 1, cv2.LINE_AA)
            for i, ch in enumerate("NB Flux"):
                cv2.putText(scatter_img, ch, (2, 15+i*15), font, 0.5, (0,0,0), 1, cv2.LINE_AA)
            qscatter = QImage(scatter_img.data, w, h, 3*w, QImage.Format.Format_RGB888).copy()
            scatter_pix = QPixmap.fromImage(qscatter)

        overlay_pix = QPixmap.fromImage(overlay_qimg)

        sub = "recipe applied from starry" if "(starless)" in filt else f"{star_count} stars detected"
        self._wb_diag_entries.append({
            "filter":      filt,
            "subtitle":    sub,
            "scatter_pix": scatter_pix,   # full 256×256, or None
            "overlay_pix": overlay_pix,   # full res
        })

        if hasattr(self, "wb_diag_button"):
            self.wb_diag_button.setEnabled(True)

        if (not getattr(self, "_pushed_results", False)
                and len(self._results) == getattr(self, "_expected_results", 0)):
            self._pushed_results = True
            self.hideSpinner()
            self._pushResultsToDocs(self._results)

    def _show_wb_diagnostics(self):
        from PyQt6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
            QScrollArea, QLabel, QPushButton, QSizePolicy
        )

        dlg = QDialog(self)
        dlg.setWindowTitle("WB Diagnostics")
        dlg.resize(100, 100)

        outer = QVBoxLayout(dlg)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # scrollable grid area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(10)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        THUMB = 256
        CARD_W = THUMB * 2 + 30  # two thumbs + padding
        COLS = 3

        for idx, entry in enumerate(self._wb_diag_entries):
            row, col = divmod(idx, COLS)

            card = QGroupBox()
            card.setTitle(entry["filter"])
            card.setFixedWidth(CARD_W)
            card_l = QVBoxLayout(card)
            card_l.setSpacing(4)
            card_l.setContentsMargins(6, 14, 6, 6)

            sub_lbl = QLabel(entry["subtitle"])
            sub_lbl.setStyleSheet("font-size:10px; color: palette(placeholderText);")
            card_l.addWidget(sub_lbl)

            thumbs = QHBoxLayout()
            thumbs.setSpacing(6)

            for pix, label_text in [
                (entry["scatter_pix"], "scatter"),
                (entry["overlay_pix"], "overlay"),
            ]:
                cell = QVBoxLayout()
                cell.setSpacing(2)
                img_lbl = QLabel()
                img_lbl.setFixedSize(THUMB, THUMB)
                img_lbl.setStyleSheet(
                    "background: palette(base); border: 0.5px solid palette(mid);"
                )
                img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                if pix is not None:
                    img_lbl.setPixmap(
                        pix.scaled(THUMB, THUMB,
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
                    )
                    img_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
                    # clicking enlarges via existing _showEnlarged
                    _p = pix  # capture
                    img_lbl.mousePressEvent = lambda e, p=_p: self._showEnlarged(p)
                else:
                    img_lbl.setText("n/a")
                cell.addWidget(img_lbl)
                cap = QLabel(label_text)
                cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cap.setStyleSheet("font-size:10px; color: palette(placeholderText);")
                cell.addWidget(cap)
                thumbs.addLayout(cell)

            card_l.addLayout(thumbs)
            grid.addWidget(card, row, col)

        # pad remaining cells so grid doesn't stretch last card
        total = len(self._wb_diag_entries)
        remainder = total % COLS
        if remainder:
            for c in range(remainder, COLS):
                spacer = QWidget()
                grid.addWidget(spacer, total // COLS, c)

        wrapper = QWidget()
        wrapper_l = QHBoxLayout(wrapper)
        wrapper_l.setContentsMargins(0, 0, 0, 0)
        wrapper_l.addWidget(container)
        wrapper_l.addStretch(1)
        scroll.setWidget(wrapper)

        def _resize_to_content():
            n = len(self._wb_diag_entries)
            if n == 0:
                return
            cols = min(n, COLS)
            rows = (n + COLS - 1) // COLS
            # card width + spacing between cards + margins
            w = cols * CARD_W + (cols - 1) * 10 + 24
            # card height: groupbox title + subtitle + thumbs + captions + padding
            card_h = 30 + 16 + THUMB + 20 + 20
            h = rows * card_h + (rows - 1) * 10 + 24 + 40  # +40 for close button row
            # clamp to screen
            screen = QApplication.primaryScreen().availableGeometry()
            w = min(w, screen.width() - 80)
            h = min(h, screen.height() - 80)
            dlg.resize(w, h)

        _resize_to_content()

        outer.addWidget(scroll, 1)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        close_row.addWidget(btn_close)
        outer.addLayout(close_row)

        dlg.exec()

    def eventFilter(self, source, event):
        # catch mouse releases on any of our clickable labels
        if event.type() == QEvent.Type.MouseButtonRelease and source in self._clickable_images:
            pix = self._clickable_images[source]
            self._showEnlarged(pix)
            return True
        return super().eventFilter(source, event)

    def _showEnlarged(self, pixmap: QPixmap):
        """
        Detail View dialog with themed zoom controls, autostretch, and 'push to document'.
        Uses ZoomableGraphicsView + QGraphicsScene (consistent with CLAHE).
        """
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QImage, QPixmap
        from PyQt6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QMessageBox,
            QLabel, QPushButton, QGraphicsScene, QGraphicsPixmapItem
        )


        from setiastro.saspro.widgets.graphics_views import ZoomableGraphicsView
        from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
        from setiastro.saspro.imageops.stretch import stretch_color_image, stretch_mono_image

        # ---------- helpers ----------
        def _float01_from_qimage(qimg: QImage) -> np.ndarray:
            """QImage -> float32 [0..1]. Handles Grayscale8 and RGB888; converts others to RGB888."""
            if qimg is None or qimg.isNull():
                return np.zeros((1, 1), dtype=np.float32)

            fmt = qimg.format()
            if fmt not in (QImage.Format.Format_Grayscale8, QImage.Format.Format_RGB888):
                qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
                fmt = QImage.Format.Format_RGB888

            h = qimg.height()
            w = qimg.width()
            bpl = qimg.bytesPerLine()

            ptr = qimg.bits()
            ptr.setsize(h * bpl)
            buf = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))

            if fmt == QImage.Format.Format_Grayscale8:
                return (buf[:, :w].astype(np.float32) / 255.0).clip(0.0, 1.0)

            # RGB888
            rgb = buf[:, :w * 3].reshape((h, w, 3)).astype(np.float32) / 255.0
            return rgb.clip(0.0, 1.0)

        def _qimage_from_float01(arr: np.ndarray) -> QImage:
            """float32 [0..1] -> QImage (RGB888 or Grayscale8), deep-copied."""
            a = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)

            if a.ndim == 2:
                u8 = (a * 255.0 + 0.5).astype(np.uint8, copy=False)
                h, w = u8.shape
                q = QImage(u8.data, w, h, w, QImage.Format.Format_Grayscale8)
                return q.copy()

            if a.ndim == 3 and a.shape[2] == 1:
                return _qimage_from_float01(a[..., 0])

            if a.ndim == 3 and a.shape[2] == 3:
                u8 = (a * 255.0 + 0.5).astype(np.uint8, copy=False)
                h, w, _ = u8.shape
                q = QImage(u8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
                return q.copy()

            # fallback
            raise ValueError(f"Unexpected image shape: {a.shape}")

        def _pixmap_from_float01(arr: np.ndarray) -> QPixmap:
            return QPixmap.fromImage(_qimage_from_float01(arr))

        # ---------- dialog ----------
        dlg = QDialog(self)
        dlg.setWindowTitle("Detail View")
        dlg.resize(980, 820)

        outer = QVBoxLayout(dlg)

        # Convert input pixmap -> float01 working buffer
        try:
            base_qimg = pixmap.toImage()
            current_arr = _float01_from_qimage(base_qimg)
        except Exception:
            current_arr = np.zeros((1, 1), dtype=np.float32)

        # Ensure "display" is always RGB for the pixmap item (GraphicsView)
        def _ensure_rgb(a: np.ndarray) -> np.ndarray:
            a = np.asarray(a, dtype=np.float32)
            if a.ndim == 2:
                return np.stack([a, a, a], axis=-1)
            if a.ndim == 3 and a.shape[2] == 1:
                return np.repeat(a, 3, axis=2)
            return a

        # Graphics view
        scene = QGraphicsScene(dlg)
        view = ZoomableGraphicsView(scene)
        view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        pix_item = QGraphicsPixmapItem()
        scene.addItem(pix_item)
        outer.addWidget(view, stretch=1)

        def _set_scene_from_arr(arr01: np.ndarray):
            rgb = _ensure_rgb(arr01)
            pm = _pixmap_from_float01(rgb)
            pix_item.setPixmap(pm)
            scene.setSceneRect(pix_item.boundingRect())

        _set_scene_from_arr(current_arr)

        # Toolbar row (themed zoom helper)
        row = QHBoxLayout()

        btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        btn_zoom_in = themed_toolbtn("zoom-in", "Zoom In")
        btn_zoom_1to1 = themed_toolbtn("zoom-original", "1:1 (100%)")
        btn_zoom_fit = themed_toolbtn("zoom-fit-best", "Fit to Preview")

        btn_zoom_out.clicked.connect(view.zoom_out)
        btn_zoom_in.clicked.connect(view.zoom_in)
        btn_zoom_1to1.clicked.connect(view.one_to_one if hasattr(view, "one_to_one") else (lambda: None))
        btn_zoom_fit.clicked.connect(lambda: view.fit_to_item(pix_item))

        row.addWidget(btn_zoom_out)
        row.addWidget(btn_zoom_in)
        row.addWidget(btn_zoom_1to1)
        row.addWidget(btn_zoom_fit)

        row.addStretch(1)

        btn_autostretch = QPushButton("Autostretch")
        row.addWidget(btn_autostretch)

        btn_push = QPushButton("Push to New Document")
        row.addWidget(btn_push)

        btn_close = QPushButton("Close")
        row.addWidget(btn_close)

        outer.addLayout(row)

        # Actions
        def _do_autostretch():
            nonlocal current_arr
            try:
                a = np.asarray(current_arr, dtype=np.float32)

                # Autostretch in-place, respecting mono vs RGB
                if a.ndim == 2:
                    stretched = stretch_mono_image(a, target_median=0.25)
                    current_arr = np.clip(stretched, 0.0, 1.0).astype(np.float32, copy=False)
                elif a.ndim == 3 and a.shape[2] == 1:
                    stretched = stretch_mono_image(a[..., 0], target_median=0.25)
                    current_arr = np.clip(stretched, 0.0, 1.0).astype(np.float32, copy=False)
                else:
                    stretched = stretch_color_image(a, target_median=0.25, linked=False)
                    current_arr = np.clip(stretched, 0.0, 1.0).astype(np.float32, copy=False)

                _set_scene_from_arr(current_arr)
                # keep current zoom, just refresh pixmap
            except Exception as e:
                QMessageBox.warning(dlg, "Detail View", f"Autostretch failed:\n{e}")

        def _do_push_to_doc():
            dm = getattr(self, "doc_manager", None)
            mw = self._main_window() if hasattr(self, "_main_window") else None

            if dm is None or mw is None or not hasattr(mw, "_spawn_subwindow_for"):
                QMessageBox.critical(dlg, "Detail View", "Cannot create document: missing DocManager or MainWindow.")
                return

            try:
                img = np.asarray(current_arr, dtype=np.float32)

                # Preserve mono where appropriate
                is_mono = (img.ndim == 2) or (img.ndim == 3 and img.shape[2] == 1)

                counter = getattr(self, "_detail_doc_counter", 0) + 1
                self._detail_doc_counter = counter
                name = f"DetailView_{counter}"

                meta = {
                    "display_name": name,
                    "file_path": name,
                    "bit_depth": "32-bit floating point",
                    "is_mono": bool(is_mono),
                    "original_header": getattr(self, "original_header", None),
                    "source": "Continuum Subtract — Detail View",
                }

                doc = dm.create_document(img, metadata=meta, name=name)
                mw._spawn_subwindow_for(doc)

                try:
                    if hasattr(self, "statusLabel") and self.statusLabel is not None:
                        self.statusLabel.setText(f"Pushed detail view → '{name}'.")
                except Exception:
                    pass

            except Exception as e:
                QMessageBox.critical(dlg, "Detail View", f"Failed to create document:\n{e}")

        btn_autostretch.clicked.connect(_do_autostretch)
        btn_push.clicked.connect(_do_push_to_doc)
        btn_close.clicked.connect(dlg.accept)

        # Initial fit
        try:
            view.fit_to_item(pix_item)
        except Exception:
            pass

        dlg.exec()


    def _onThreadFinished(self):
        self._pending -= 1
        if self._pending == 0:
            self.hideSpinner()
            self._pushResultsToDocs(self._results) 

    def _pushResultsToDocs(self, results):
        dm = getattr(self, "doc_manager", None)
        mw = self._main_window()
        if dm is None or mw is None or not hasattr(mw, "_spawn_subwindow_for"):
            QMessageBox.critical(self, "Continuum Subtract",
                                "Cannot create documents: missing DocManager or MainWindow.")
            return

        created = 0
        for entry in results:
            filt = entry["filter"]
            img  = np.asarray(entry["image"], dtype=np.float32)  # keep everything float32
            name = f"{filt}_ContSub"

            meta = {
                "display_name": name,                 # nice title in the UI
                "file_path": name,                    # placeholder path until user saves
                "bit_depth": "32-bit floating point",
                "is_mono": (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)),
                "original_header": self.original_header,
                "source": "Continuum Subtract",
            }

            try:
                # create a proper ImageDocument and register it
                doc = dm.create_document(img, metadata=meta, name=name)
                # show it as an MDI subwindow
                mw._spawn_subwindow_for(doc)
                created += 1
            except Exception as e:
                QMessageBox.critical(self, "Continuum Subtract",
                                    f"Failed to create document '{name}':\n{e}")

        self.statusLabel.setText(f"Created {created} document(s).")


    def _onThreadFinished(self):
        self._pending -= 1
        # do NOT push here if you already push in _onOneResult

    def update_status_label(self, message):
        self.statusLabel.setText(message)

    def showSpinner(self):
        self.spinnerLabel.setText("Processing…")
        self.spinnerLabel.show()
        if hasattr(self, "execute_button"):
            self.execute_button.setEnabled(False)

    def hideSpinner(self):
        self.spinnerLabel.hide()
        self.spinnerLabel.clear()
        if hasattr(self, "execute_button"):
            self.execute_button.setEnabled(True)



class ContinuumProcessingThread(QThread):
    processing_complete = pyqtSignal(np.ndarray, int, QImage, np.ndarray, np.ndarray)
    processing_complete_starless = pyqtSignal(np.ndarray, int, QImage, np.ndarray, np.ndarray)
    status_update = pyqtSignal(str)

    def __init__(self, nb_image, continuum_image, output_linear, *,
                starless_nb=None, starless_cont=None, starless_only=False,
                threshold: float = 5.0, summary_gamma: float = 0.6, q_factor: float = 0.8,
                denoise_linear: bool = False, denoise_strength: float = 0.90,
                do_stretch: bool = True, do_median_sub: bool = True,
                do_curves: bool = True, no_black_clip: bool = False):
        super().__init__()
        self.nb_image = nb_image
        self.continuum_image = continuum_image
        self.output_linear = output_linear
        self.starless_nb = starless_nb
        self.starless_cont = starless_cont
        self.starless_only = starless_only
        self.background_reference = None
        self._recipe = None  # learned from starry pass

        # user knobs
        self.threshold = float(threshold)
        self.summary_gamma = float(summary_gamma)
        self.q_factor = float(q_factor)

        # NEW: Cosmic Clarity integration
        self.denoise_linear   = bool(denoise_linear)
        self.denoise_strength = float(denoise_strength)
        self.do_stretch       = bool(do_stretch)
        self.do_median_sub    = bool(do_median_sub)
        self.do_curves        = bool(do_curves)
        self.no_black_clip    = bool(no_black_clip)

    # ---------- small helpers ----------
    @staticmethod
    def _to_mono(img):
        a = np.asarray(img)
        if a.ndim == 3:
            if a.shape[2] == 3:
                return a[..., 0]  # use R channel for NB/cont slots when color
            if a.shape[2] == 1:
                return a[..., 0]
        return a

    @staticmethod
    def _as_rgb(nb, cont):
        r = np.asarray(nb,   dtype=np.float32)
        g = np.asarray(cont, dtype=np.float32)
        if r.ndim == 3: r = r[..., 0]
        if g.ndim == 3: g = g[..., 0]
        if r.dtype.kind in "ui":
            r = r / (255.0 if r.dtype == np.uint8 else 65535.0)
        if g.dtype.kind in "ui":
            g = g / (255.0 if g.dtype == np.uint8 else 65535.0)
        b = g
        return np.stack([r, g, b], axis=-1).astype(np.float32, copy=False)

    @staticmethod
    def _fit_ab(x, y):
        x = x.reshape(-1).astype(np.float32)
        y = y.reshape(-1).astype(np.float32)
        N = min(x.size, 100_000)
        if x.size > N:
            idx = np.random.choice(x.size, N, replace=False)
            x = x[idx]; y = y[idx]
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(a), float(b)

    @staticmethod
    def _qimage_from_uint8(rgb_uint8: np.ndarray) -> QImage:
        """Create a deep-copied QImage from an HxWx3 uint8 array."""
        h, w = rgb_uint8.shape[:2]
        return QImage(rgb_uint8.data, w, h, 3*w, QImage.Format.Format_RGB888).copy()

    @staticmethod
    def _nonlinear_finalize(lin_img: np.ndarray) -> np.ndarray:
        """Stretch → subtract pedestal → curves, returned as float32 in [0,1]."""
        target_median = 0.25
        stretched = stretch_color_image(lin_img, target_median, True, False)
        final = stretched - 0.7 * np.median(stretched)
        final = np.clip(final, 0, 1)
        return apply_curves_adjustment(final, np.median(final), 0.5).astype(np.float32, copy=False)

    # ---------- BG neutral: return pedestals (no in-place surprises) ----------
    def _compute_bg_pedestal(self, rgb):
        height, width, _ = rgb.shape
        num_boxes, box_size, iterations = 200, 25, 25

        boxes = [(np.random.randint(0, height - box_size),
                  np.random.randint(0, width - box_size)) for _ in range(num_boxes)]
        best = np.full(num_boxes, np.inf, dtype=np.float32)

        for _ in range(iterations):
            for i, (y, x) in enumerate(boxes):
                if y + box_size <= height and x + box_size <= width:
                    patch = rgb[y:y+box_size, x:x+box_size]
                    med = np.median(patch) if patch.size else np.inf
                    best[i] = min(best[i], med)
                    sv = []
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            yy, xx = y + dy*box_size, x + dx*box_size
                            if 0 <= yy < height - box_size and 0 <= xx < width - box_size:
                                p2 = rgb[yy:yy+box_size, xx:xx+box_size]
                                if p2.size:
                                    sv.append(np.median(p2))
                    if sv:
                        k = int(np.argmin(sv))
                        y += (k // 3 - 1) * box_size
                        x += (k % 3  - 1) * box_size
                        boxes[i] = (y, x)

        # pick darkest
        darkest = np.inf; ref = None
        for y, x in boxes:
            if y + box_size <= height and x + box_size <= width:
                patch = rgb[y:y+box_size, x:x+box_size]
                med = np.median(patch) if patch.size else np.inf
                if med < darkest:
                    darkest, ref = med, patch

        ped = np.zeros(3, dtype=np.float32)
        if ref is not None:
            self.background_reference = np.median(ref.reshape(-1, 3), axis=0)
            chan_meds = np.median(rgb, axis=(0, 1))
            # pedestal to lift channels toward their own median
            ped = np.maximum(0.0, chan_meds - self.background_reference)

            # specifically lift G/B if below R reference
            r_ref = float(self.background_reference[0])
            for ch in (1, 2):
                if self.background_reference[ch] < r_ref:
                    ped[ch] += (r_ref - self.background_reference[ch])
        return ped

    @staticmethod
    def _apply_pedestal(rgb, ped):
        return np.clip(rgb + ped.reshape(1,1,3), 0.0, 1.0)

    @staticmethod
    def _normalize_red_to_green(rgb):
        r = rgb[...,0]; g = rgb[...,1]
        mad_r = float(np.mean(np.abs(r - np.mean(r))))
        mad_g = float(np.mean(np.abs(g - np.mean(g))))
        med_r = float(np.median(r))
        med_g = float(np.median(g))
        g_gain = (mad_g / max(mad_r, 1e-9))
        g_offs = (-g_gain * med_r + med_g)
        rgb2 = rgb.copy()
        rgb2[...,0] = np.clip(r * g_gain + g_offs, 0.0, 1.0)
        return rgb2, g_gain, g_offs

    def _linear_subtract(self, rgb, Q, green_median):
        r = rgb[...,0]; g = rgb[...,1]
        return np.clip(r - Q * (g - green_median), 0.0, 1.0)

    def _nonlinear_finalize_with_opts(self, lin_img: np.ndarray) -> np.ndarray:
        """Stretch → optional median sub → optional curves, honouring user checkboxes."""
        if not self.do_stretch:
            return np.clip(lin_img, 0.0, 1.0).astype(np.float32, copy=False)

        target_median = 0.25
        stretched = stretch_color_image(
            lin_img, target_median, True, False,
            no_black_clip=self.no_black_clip
        )

        if self.do_median_sub:
            stretched = stretched - 0.7 * np.median(stretched)
            stretched = np.clip(stretched, 0.0, 1.0)

        if self.do_curves:
            stretched = apply_curves_adjustment(stretched, np.median(stretched), 0.5)

        return np.clip(stretched, 0.0, 1.0).astype(np.float32, copy=False)

    # ---------- main ----------
    def run(self):
        try:
            # STARLESS-ONLY early exit
            if (self.nb_image is None or self.continuum_image is None) and self.starless_only:
                self._run_starless_only()
                return

            recipe = None

            # ----- starry pass (learn recipe) -----
            if self.nb_image is not None and self.continuum_image is not None:
                rgb = self._as_rgb(self.nb_image, self.continuum_image)

                self.status_update.emit("Performing background neutralization...")
                ped = self._compute_bg_pedestal(rgb)
                rgb = self._apply_pedestal(rgb, ped)

                self.status_update.emit("Normalizing red to green…")
                rgb, g_gain, g_offs = self._normalize_red_to_green(rgb)

                self.status_update.emit("Performing star-based white balance…")
                balanced_rgb, star_count, star_overlay, raw_star_pixels, after_star_pixels = \
                    apply_star_based_white_balance(
                        rgb, threshold=self.threshold, autostretch=False,
                        reuse_cached_sources=True, return_star_colors=True
                    )

                # per-channel affine fit to reproduce WB later
                wb_a = np.zeros(3, np.float32)
                wb_b = np.zeros(3, np.float32)
                for c in range(3):
                    a, b = self._fit_ab(rgb[..., c], balanced_rgb[..., c])
                    wb_a[c], wb_b[c] = a, b

                green_med = float(np.median(balanced_rgb[..., 1]))
                Q = self.q_factor
                linear_image = self._linear_subtract(balanced_rgb, Q, green_med)
                if self.denoise_linear:
                    self.status_update.emit("Denoising continuum-subtracted image (Cosmic Clarity)…")
                    linear_image = self._denoise_linear_image(linear_image)
                # --- NEW: gamma brighten overlay for the summary ---
                g = max(self.summary_gamma, 1e-6)
                overlay_gamma = np.power(np.clip(star_overlay, 0.0, 1.0), g)
                overlay_uint8 = (overlay_gamma * 255).astype(np.uint8)
                qimg = self._qimage_from_uint8(overlay_uint8)

                if self.output_linear:
                    self.processing_complete.emit(
                        np.clip(linear_image, 0, 1), int(star_count), qimg,
                        np.array(raw_star_pixels), np.array(after_star_pixels)
                    )
                else:
                    self.status_update.emit("Linear → Non-linear stretch…")
                    final = self._nonlinear_finalize_with_opts(linear_image)
                    self.processing_complete.emit(
                        final, int(star_count), qimg,
                        np.array(raw_star_pixels), np.array(after_star_pixels)
                    )

                # learned recipe + fit data (reused for starless)
            recipe = {
                "pedestal": ped,
                "rnorm_gain": g_gain,
                "rnorm_offs": g_offs,
                "wb_a": wb_a,
                "wb_b": wb_b,
                "Q": Q,
                "green_median": green_med,

                # store raw overlay + star stats for reuse
                "overlay_uint8": overlay_uint8,
                "fit_star_count": int(star_count),
                "fit_raw": np.array(raw_star_pixels, copy=True),
                "fit_after": np.array(after_star_pixels, copy=True),
            }

            # ----- starless paired pass (apply recipe) -----
            if self.starless_nb is not None and self.starless_cont is not None:
                if recipe is not None:
                    rgb = self._as_rgb(self.starless_nb, self.starless_cont)
                    # apply starry recipe exactly
                    rgb = self._apply_pedestal(rgb, recipe["pedestal"])
                    r = rgb[..., 0]
                    rgb[..., 0] = np.clip(r * recipe["rnorm_gain"] + recipe["rnorm_offs"], 0.0, 1.0)
                    for c in range(3):
                        rgb[..., c] = np.clip(rgb[..., c] * recipe["wb_a"][c] + recipe["wb_b"][c], 0.0, 1.0)

                    lin = self._linear_subtract(rgb, recipe["Q"], recipe["green_median"])
                    if self.denoise_linear:
                        self.status_update.emit("Denoising starless continuum-subtracted image (Cosmic Clarity)…")
                        lin = self._denoise_linear_image(lin)
                    # reuse gamma-bright overlay & fit info from the starry pass
                    # rebuild overlay & make fresh copies of arrays for the starless emit
                    overlay_uint8 = np.array(recipe["overlay_uint8"], copy=True)
                    fit_qimg = self._qimage_from_uint8(overlay_uint8)
                    fit_count = int(recipe["fit_star_count"])
                    fit_raw   = np.array(recipe["fit_raw"], copy=True)
                    fit_after = np.array(recipe["fit_after"], copy=True)

                    if self.output_linear:
                        self.processing_complete_starless.emit(
                            np.clip(lin, 0, 1), fit_count, fit_qimg, fit_raw, fit_after
                        )
                    else:
                        self.status_update.emit("Linear → Non-linear stretch (starless)…")
                        final = self._nonlinear_finalize_with_opts(lin)
                        self.processing_complete_starless.emit(
                            final, fit_count, fit_qimg, fit_raw, fit_after
                        )

                elif self.starless_only:
                    pass  # handled in _run_starless_only
        except Exception as e:
            try:
                self.status_update.emit(f"Continuum subtraction failed: {e}")
            except Exception:
                pass

    # ----- starless-only path (no WB; same math you had) -----
    def _run_starless_only(self):
        rgb = self._as_rgb(self.starless_nb, self.starless_cont)

        self.status_update.emit("Performing background neutralization…")
        ped = self._compute_bg_pedestal(rgb)
        rgb = self._apply_pedestal(rgb, ped)

        self.status_update.emit("Normalizing red to green…")
        rgb, _, _ = self._normalize_red_to_green(rgb)

        green_med = float(np.median(rgb[..., 1]))
        lin = self._linear_subtract(rgb, 0.9, green_med)
        if self.denoise_linear:
            self.status_update.emit("Denoising starless continuum-subtracted image (Cosmic Clarity)…")
            lin = self._denoise_linear_image(lin)
        # Blank overlay & empty star lists (no star detection in starless-only path)
        h, w = lin.shape[:2]
        blank = np.zeros((h, w, 3), np.uint8)
        qimg  = self._qimage_from_uint8(blank)
        empty = np.empty((0, 2), float)

        if self.output_linear:
            self.processing_complete_starless.emit(np.clip(lin, 0, 1), 0, qimg, empty, empty)
            return

        self.status_update.emit("Linear → Non-linear stretch…")
        final = self._nonlinear_finalize_with_opts(lin)
        self.processing_complete_starless.emit(final, 0, qimg, empty, empty)

    def _denoise_linear_image(self, img: np.ndarray) -> np.ndarray:
        try:
            if img is None:
                return img

            if _cc_denoise is None:
                self.status_update.emit("Cosmic Clarity denoise unavailable (import failed).")
                return img

            preset = {
                "mode": "denoise",
                "denoise_luma": self.denoise_strength,
                "denoise_color": self.denoise_strength,
                "denoise_mode": "full",
                "separate_channels": False,
                "denoise_lite": False,
                "denoise_walking": False,
                "gpu": True,
                "chunk_size": 256,
                "overlap": 64,
                "temp_stretch": False,
                "target_median": 0.25,
            }

            arr = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
            result = _cc_denoise(arr, preset)
            return np.clip(np.asarray(result, dtype=np.float32), 0.0, 1.0)

        except Exception as e:
            try:
                self.status_update.emit(f"Cosmic Clarity denoise failed: {e}")
            except Exception:
                pass
            return img