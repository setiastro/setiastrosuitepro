from __future__ import annotations

# =========================
# SASpro Script Metadata
# Creator: DarkEnergy
# Version: 1.0
# =========================
SCRIPT_NAME     = "Dual-Filter NB Extractor (WP, Full)"
SCRIPT_GROUP    = "Narrowband"
SCRIPT_SHORTCUT = ""   # optional

import json
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QMessageBox, QGroupBox, QGridLayout, QDoubleSpinBox, QCheckBox, QSpinBox,
    QFileDialog,
)
from PyQt6.QtCore import Qt


# ---------------------------------------------------------------------
# Utility: ensure float32, 0–1 range
# ---------------------------------------------------------------------
def _to_float01(img) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype not in (np.float32, np.float64):
        a = a.astype(np.float32, copy=False)
    else:
        a = a.astype(np.float32, copy=False)

    if a.size == 0:
        return a

    # If data looks like 0–255 or 16-bit, normalize
    vmax = float(np.nanmax(a))
    if np.isfinite(vmax) and vmax > 1.5:
        a = a / vmax

    # Clip to sensible range
    return np.clip(a, 0.0, 1.0)


# ---------------------------------------------------------------------
# Core math: coefficient matrices and LS decoders
# ---------------------------------------------------------------------
def build_matrix_A(c: dict[str, float]) -> np.ndarray:
    """Build the 6x3 coefficient matrix A for the full C1+C2 model."""
    return np.array([
        [c["r_h1"], c["r_o1"], 0.0],        # R1
        [c["g_h1"], c["g_o1"], 0.0],        # G1
        [c["b_h1"], c["b_o1"], 0.0],        # B1
        [0.0,       c["r_o2"], c["r_s2"]],  # R2
        [0.0,       c["g_o2"], c["g_s2"]],  # G2
        [0.0,       c["b_o2"], c["b_s2"]],  # B2
    ], dtype=np.float32)


def build_matrix_A_c1(c: dict[str, float]) -> np.ndarray:
    """Build the 3x2 coefficient matrix A for the C1-only (Ha+OIII) model."""
    return np.array([
        [c["r_h1"], c["r_o1"]],  # R1
        [c["g_h1"], c["g_o1"]],  # G1
        [c["b_h1"], c["b_o1"]],  # B1
    ], dtype=np.float32)


def compute_decoder_B(A: np.ndarray) -> np.ndarray:
    """
    Compute least-squares decoder matrix:  B = (A^T A)^-1 A^T

    A : (N, M)
    B : (M, N)
    """
    AT = A.T
    ATA = AT @ A
    ATA_inv = np.linalg.inv(ATA)
    B = ATA_inv @ AT
    return B


def extract_channel_unweighted(
    r1: np.ndarray,
    g1: np.ndarray,
    b1: np.ndarray,
    r2: np.ndarray,
    g2: np.ndarray,
    b2: np.ndarray,
    decoder_row: np.ndarray,
) -> np.ndarray:
    """
    Fast unweighted extraction for a single channel in the 6-channel model.

    decoder_row : shape (6,)
    """
    coeffs = decoder_row.astype(np.float32)
    return (coeffs[0] * r1 + coeffs[1] * g1 + coeffs[2] * b1 +
            coeffs[3] * r2 + coeffs[4] * g2 + coeffs[5] * b2)


def extract_channel_unweighted_c1(
    r1: np.ndarray,
    g1: np.ndarray,
    b1: np.ndarray,
    decoder_row: np.ndarray,
) -> np.ndarray:
    """
    Fast unweighted extraction for a single channel in the C1-only 3-channel model.

    decoder_row : shape (3,)
    """
    coeffs = decoder_row.astype(np.float32)
    return coeffs[0] * r1 + coeffs[1] * g1 + coeffs[2] * b1


def fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    """
    Fit slope using least squares: y = slope * x
    (mask invalid & x>0, require >=10 points)
    """
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if np.sum(valid) < 10:
        return 0.0

    xv = x[valid]
    yv = y[valid]

    num = float(np.sum(xv * yv))
    den = float(np.sum(xv * xv))
    if den > 0.0:
        return num / den
    return 0.0


def renormalize_gb(g: float, b: float, r_fixed: float, target: float) -> tuple[float, float]:
    """
    Renormalize green and blue coefficients to sum to 'target'
    when red is fixed to r_fixed.
    """
    s = g + b
    if s <= 0.0:
        return g, b
    scale = (target - r_fixed) / s
    return g * scale, b * scale


def compute_stats(data: np.ndarray) -> dict[str, float]:
    """Simple stats summary for logging or inspection."""
    return {
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "negative_fraction": float(np.sum(data < 0) / data.size),
    }


# ---------------------------------------------------------------------
# Generic vectorized weighted regression backend
# ---------------------------------------------------------------------
def _weighted_regression_all(
    observations: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """
    Generic vectorized weighted regression for K components.

    observations : (H, W, J)
    A            : (J, K)

    Returns
    -------
    X : (H, W, K)
        Estimated component images.
    """
    h, w, J = observations.shape
    if A.shape[0] != J:
        raise ValueError(f"Incompatible shapes: observations has {J} channels, A has {A.shape[0]} rows")

    obs_flat = observations.reshape(-1, J).astype(np.float32)

    # Compute weights (same rules as original, but vectorized)
    max_signals = np.max(obs_flat, axis=1, keepdims=True)
    max_signals = np.maximum(max_signals, 1e-6)
    obs_normalized = obs_flat / max_signals

    weights = np.sqrt(obs_normalized + 0.001)
    mask_low = obs_normalized < 0.01
    mask_high = obs_normalized > 0.95
    weights[mask_low] *= 0.1
    weights[mask_high] *= 0.1
    weights = np.maximum(weights, 0.01)

    # A is (J,K). For each pixel n, we want:
    #   ATWA_n = A^T W_n A  (KxK)
    #   ATWy_n = A^T W_n y_n (K,)
    ATWA = np.einsum("nj,jk,jl->nkl", weights, A, A, optimize=True)
    ATWy = np.einsum("nj,jk,nj->nk", weights, A, obs_flat, optimize=True)

    # Try batched solve; if NumPy is too old (SAS Pro embedded), fall back
    # to batched inverse.
    try:
        X = np.linalg.solve(ATWA, ATWy)  # (N,K)
    except Exception:
        eye = np.eye(A.shape[1], dtype=np.float32)
        ATWA_reg = ATWA + 1e-6 * eye
        inv_ATWA = np.linalg.inv(ATWA_reg)          # (N,K,K)
        X = np.einsum("nij,nj->ni", inv_ATWA, ATWy, optimize=True)

    X = X.reshape(h, w, A.shape[1])
    return X


def _extract_channels_weighted_vec(
    observations: np.ndarray,
    A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted extraction for the full 6x3 model (Ha, OIII, SII).
    """
    X = _weighted_regression_all(observations, A)
    ha = X[..., 0]
    oiii = X[..., 1]
    sii = X[..., 2]
    return ha, oiii, sii


def _extract_c1_weighted_vec(
    observations: np.ndarray,
    A_c1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted extraction for the C1-only 3x2 model (Ha, OIII).
    """
    X = _weighted_regression_all(observations, A_c1)
    ha = X[..., 0]
    oiii = X[..., 1]
    return ha, oiii


# ---------------------------------------------------------------------
# Full extraction (self-cal + weighted/unweighted) for C1+C2
# ---------------------------------------------------------------------
def run_extraction_full(
    img_c1: np.ndarray,
    img_c2: np.ndarray,
    coeffs: dict[str, float],
    iterations: int = 0,
    clip_negative: bool = True,
    use_weighted: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, float]], dict[str, float]]:
    """
    Full extraction with optional self-calibration and weighted regression
    using both C1 (Ha+OIII) and C2 (SII+OIII) filters.
    """
    # RGB split
    c1_r = img_c1[..., 0].astype(np.float32)
    c1_g = img_c1[..., 1].astype(np.float32)
    c1_b = img_c1[..., 2].astype(np.float32)

    c2_r = img_c2[..., 0].astype(np.float32)
    c2_g = img_c2[..., 1].astype(np.float32)
    c2_b = img_c2[..., 2].astype(np.float32)

    c = dict(coeffs)  # copy so we can update during self-cal

    # Self-calibration iterations
    for _ in range(int(max(0, iterations))):
        A_iter = build_matrix_A(c)
        B_iter = compute_decoder_B(A_iter)

        # provisional OIII
        oiii_temp = extract_channel_unweighted(
            c1_r, c1_g, c1_b, c2_r, c2_g, c2_b, B_iter[1]
        )

        # Slopes C1-G, C1-B, C2-G, C2-B vs provisional OIII
        g1_slope = fit_slope(oiii_temp.flatten(), c1_g.flatten())
        b1_slope = fit_slope(oiii_temp.flatten(), c1_b.flatten())
        g2_slope = fit_slope(oiii_temp.flatten(), c2_g.flatten())
        b2_slope = fit_slope(oiii_temp.flatten(), c2_b.flatten())

        # Renormalize to keep total OIII response consistent (target=1.0)
        g1_norm, b1_norm = renormalize_gb(g1_slope, b1_slope, c["r_o1"], 1.0)
        g2_norm, b2_norm = renormalize_gb(g2_slope, b2_slope, c["r_o2"], 1.0)

        c["g_o1"] = g1_norm
        c["b_o1"] = b1_norm
        c["g_o2"] = g2_norm
        c["b_o2"] = b2_norm

    # Final extraction
    A = build_matrix_A(c)

    if use_weighted:
        # Weighted extraction using the vectorized backend
        obs = np.stack([c1_r, c1_g, c1_b, c2_r, c2_g, c2_b], axis=2)
        ha, oiii, sii = _extract_channels_weighted_vec(obs, A)
    else:
        # Simple global decoder B (very fast)
        B = compute_decoder_B(A)
        ha = extract_channel_unweighted(c1_r, c1_g, c1_b, c2_r, c2_g, c2_b, B[0])
        oiii = extract_channel_unweighted(c1_r, c1_g, c1_b, c2_r, c2_g, c2_b, B[1])
        sii = extract_channel_unweighted(c1_r, c1_g, c1_b, c2_r, c2_g, c2_b, B[2])

    if clip_negative:
        ha = np.clip(ha, 0.0, None)
        oiii = np.clip(oiii, 0.0, None)
        sii = np.clip(sii, 0.0, None)

    stats = {
        "ha": compute_stats(ha),
        "oiii": compute_stats(oiii),
        "sii": compute_stats(sii),
    }

    return ha, oiii, sii, stats, c


# ---------------------------------------------------------------------
# C1-only extraction (Ha+OIII filter only)
# ---------------------------------------------------------------------
def run_extraction_c1_only(
    img_c1: np.ndarray,
    coeffs: dict[str, float],
    clip_negative: bool = True,
    use_weighted: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]], dict[str, float]]:
    """
    Extraction using only the C1 (Ha+OIII) filter to produce Ha and OIII
    narrowband images. No self-calibration is applied here.
    """
    c1_r = img_c1[..., 0].astype(np.float32)
    c1_g = img_c1[..., 1].astype(np.float32)
    c1_b = img_c1[..., 2].astype(np.float32)

    c = dict(coeffs)

    A_c1 = build_matrix_A_c1(c)

    if use_weighted:
        obs = np.stack([c1_r, c1_g, c1_b], axis=2)
        ha, oiii = _extract_c1_weighted_vec(obs, A_c1)
    else:
        B_c1 = compute_decoder_B(A_c1)  # shape (2,3)
        ha = extract_channel_unweighted_c1(c1_r, c1_g, c1_b, B_c1[0])
        oiii = extract_channel_unweighted_c1(c1_r, c1_g, c1_b, B_c1[1])

    if clip_negative:
        ha = np.clip(ha, 0.0, None)
        oiii = np.clip(oiii, 0.0, None)

    stats = {
        "ha": compute_stats(ha),
        "oiii": compute_stats(oiii),
    }

    return ha, oiii, stats, c


# ---------------------------------------------------------------------
# Coefficient UI widget embedded in the dialog
# ---------------------------------------------------------------------
class _CoeffEditor:
    """
    Manages QDoubleSpinBoxes for the coefficient set used in the dual-filter model.

    Keys:
        r_h1, g_h1, b_h1,
        r_o1, g_o1, b_o1,
        r_s2, g_s2, b_s2,
        r_o2, g_o2, b_o2
    """

    # Default coefficients (Space Koala)
    DEFAULT_COEFFS = {
       "r_h1": 1.0,   "g_h1": 0.0,   "b_h1": 0.0,
       "r_o1": 0.0,   "g_o1": 0.25,  "b_o1": 0.25,
       "r_s2": 1.0,   "g_s2": 0.0,   "b_s2": 0.0,
       "r_o2": 0.0,   "g_o2": 0.25,  "b_o2": 0.25,
    }

    PRESETS = {
        0: ("Custom", None),
        1: ("ASI533MC + Askar C1/C2", {
        "r_h1": 0.65, "g_h1": 0.12, "b_h1": 0.03,
        "r_o1": 0.03, "g_o1": 0.75, "b_o1": 0.45,
        "r_s2": 0.55, "g_s2": 0.2, "b_s2": 0.05,
        "r_o2": 0.03, "g_o2": 0.75, "b_o2": 0.45,
        }),
        2: ("Space Koala", {
            "r_h1": 1.0,   "g_h1": 0.0,   "b_h1": 0.0,
            "r_o1": 0.0,   "g_o1": 0.25,  "b_o1": 0.25,
            "r_s2": 1.0,   "g_s2": 0.0,   "b_s2": 0.0,
            "r_o2": 0.0,   "g_o2": 0.25,  "b_o2": 0.25,
        }),
        3: ("Cuiv", {
            "r_h1": 1.0,   "g_h1": 0.0,   "b_h1": 0.0,
            "r_o1": 0.0,   "g_o1": 0.35,  "b_o1": 0.15,
            "r_s2": 1.0,   "g_s2": 0.0,   "b_s2": 0.0,
            "r_o2": 0.0,   "g_o2": 0.35,  "b_o2": 0.15,
        }),
    }

    def __init__(self, parent_layout: QVBoxLayout):
        self.spin_boxes: dict[str, QDoubleSpinBox] = {}

        # Preset selector
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.combo_preset = QComboBox()
        for idx in sorted(self.PRESETS.keys()):
            label = self.PRESETS[idx][0]
            self.combo_preset.addItem(label, idx)
        preset_row.addWidget(self.combo_preset, 1)
        parent_layout.addLayout(preset_row)

        # Coefficients group
        gbox = QGroupBox("Extraction Coefficients")
        grid = QGridLayout()
        gbox.setLayout(grid)

        labels = [
            ("Filter 1 (Ha + OIII)", [
                "r_h1", "g_h1", "b_h1",
                "r_o1", "g_o1", "b_o1",
            ]),
            ("Filter 2 (SII + OIII)", [
                "r_s2", "g_s2", "b_s2",
                "r_o2", "g_o2", "b_o2",
            ]),
        ]

        row = 0
        for header, keys in labels:
            lbl = QLabel(f"<b>{header}</b>")
            lbl.setTextFormat(Qt.TextFormat.RichText)
            grid.addWidget(lbl, row, 0, 1, 2)
            row += 1
            for key in keys:
                grid.addWidget(QLabel(key + ":"), row, 0)
                spin = QDoubleSpinBox()
                spin.setRange(0.0, 2.0)
                spin.setDecimals(5)
                spin.setSingleStep(0.01)
                spin.setValue(self.DEFAULT_COEFFS.get(key, 0.0))
                grid.addWidget(spin, row, 1)
                self.spin_boxes[key] = spin
                row += 1

        parent_layout.addWidget(gbox)

        # Buttons for reset / save / load
        rrow = QHBoxLayout()
        rrow.addStretch(1)
        self.btn_reset = QPushButton("Reset to Defaults")
        self.btn_save = QPushButton("Save Weights…")
        self.btn_load = QPushButton("Load Weights…")
        rrow.addWidget(self.btn_load)
        rrow.addWidget(self.btn_save)
        rrow.addWidget(self.btn_reset)
        parent_layout.addLayout(rrow)

        # wire up preset + reset (save/load wired in dialog)
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        self.btn_reset.clicked.connect(self.reset_to_defaults)

    # ---- API ----
    def get_coefficients(self) -> dict[str, float]:
        return {k: sb.value() for k, sb in self.spin_boxes.items()}

    def set_coefficients(self, coeffs: dict[str, float]) -> None:
        for k, v in coeffs.items():
            sb = self.spin_boxes.get(k)
            if sb is not None:
                sb.setValue(float(v))

    def reset_to_defaults(self) -> None:
        self.set_coefficients(self.DEFAULT_COEFFS)
        # switch preset combo back to match defaults (ASI533/Askar)
        self.combo_preset.setCurrentIndex(1)

    # ---- persistence helpers ----
    def save_coefficients_via_dialog(self, parent: QDialog) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            parent,
            "Save Regression Weights",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if not filename:
            return
        try:
            data = self.get_coefficients()
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            QMessageBox.critical(parent, "Save Weights", f"Failed to save weights:\n{e}")

    def load_coefficients_via_dialog(self, parent: QDialog) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            parent,
            "Load Regression Weights",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if not filename:
            return
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("JSON root must be an object mapping coefficient names to values.")
            for k, v in data.items():
                if k in self.spin_boxes:
                    self.spin_boxes[k].setValue(float(v))
            # set preset to "Custom"
            self.combo_preset.setCurrentIndex(0)
        except Exception as e:
            QMessageBox.critical(parent, "Load Weights", f"Failed to load weights:\n{e}")

    # ---- internal ----
    def _on_preset_changed(self, idx: int) -> None:
        preset_idx = self.combo_preset.itemData(idx)  # our numeric key
        _, coeffs = self.PRESETS.get(preset_idx, (None, None))
        if coeffs is not None:
            self.set_coefficients(coeffs)


# ---------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------
class NBExtractorDialog(QDialog):
    """
    SASpro UI wrapper around the full dual-filter extraction.

    - Lets user pick two open documents as Filter 1 (Ha+OIII) and Filter 2 (SII+OIII)
    - Coefficient editor with presets and save/load
    - Iteration count for self-calibration
    - Optional weighted extraction
    - Option to use Filter-1-only (Ha+OIII) mode
    - Clip negative toggle
    - Creates new Ha/OIII/SII documents
    """
    def __init__(self, ctx):
        super().__init__(parent=ctx.app)
        self.ctx = ctx
        self.setWindowTitle("Dual-Filter NB Extractor (WP, Full)")
        self.resize(720, 460)

        self._title_to_doc: dict[str, object] = {}

        root = QVBoxLayout(self)

        # ----------------- C1/C2 selection -----------------
        row_c1 = QHBoxLayout()
        row_c1.addWidget(QLabel("Filter 1 (Ha+OIII):"))
        self.combo_c1 = QComboBox()
        row_c1.addWidget(self.combo_c1, 1)
        root.addLayout(row_c1)

        row_c2 = QHBoxLayout()
        row_c2.addWidget(QLabel("Filter 2 (SII+OIII):"))
        self.combo_c2 = QComboBox()
        row_c2.addWidget(self.combo_c2, 1)
        root.addLayout(row_c2)

        brow_top = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh List")
        brow_top.addStretch(1)
        brow_top.addWidget(self.btn_refresh)
        root.addLayout(brow_top)

        # ----------------- Coefficients -----------------
        self.coeff_editor = _CoeffEditor(root)

        # connect save/load buttons
        self.coeff_editor.btn_save.clicked.connect(
            lambda: self.coeff_editor.save_coefficients_via_dialog(self)
        )
        self.coeff_editor.btn_load.clicked.connect(
            lambda: self.coeff_editor.load_coefficients_via_dialog(self)
        )

        # ----------------- Options -----------------
        opt_group = QGroupBox("Extraction Options")
        opt_layout = QGridLayout()
        opt_group.setLayout(opt_layout)

        # Self-calibration iterations (only used in dual-filter mode)
        opt_layout.addWidget(QLabel("Self-calibration iterations:"), 0, 0)
        self.spn_iter = QSpinBox()
        self.spn_iter.setRange(0, 10)
        self.spn_iter.setValue(0)
        opt_layout.addWidget(self.spn_iter, 0, 1)

        # Weighted LS checkbox
        self.chk_weighted = QCheckBox("Use weighted regression (robust)")
        self.chk_weighted.setChecked(False)
        opt_layout.addWidget(self.chk_weighted, 1, 0, 1, 2)

        # Clip negatives
        self.chk_clip_neg = QCheckBox("Clip negative pixels to 0")
        self.chk_clip_neg.setChecked(True)
        opt_layout.addWidget(self.chk_clip_neg, 2, 0, 1, 2)

        # C1-only mode
        self.chk_c1_only = QCheckBox("Use Filter 1 (Ha+OIII) only; ignore Filter 2 and SII")
        self.chk_c1_only.setChecked(False)
        opt_layout.addWidget(self.chk_c1_only, 3, 0, 1, 2)

        root.addWidget(opt_group)

        # ----------------- Bottom buttons -----------------
        brow = QHBoxLayout()
        brow.addStretch(1)
        self.btn_extract = QPushButton("Extract \u2192 New Docs")
        self.btn_close = QPushButton("Close")
        brow.addWidget(self.btn_extract)
        brow.addWidget(self.btn_close)
        root.addLayout(brow)

        # Wiring
        self.btn_refresh.clicked.connect(self._populate_docs)
        self.btn_extract.clicked.connect(self._do_extract)
        self.btn_close.clicked.connect(self.reject)

        # initial populate
        self._populate_docs()

    # ----------------- document listing -----------------
    def _populate_docs(self):
        self.combo_c1.clear()
        self.combo_c2.clear()
        self._title_to_doc.clear()

        try:
            views = self.ctx.list_image_views()
        except Exception:
            views = []

        for title, doc in views:
            key = title
            # disambiguate duplicate titles if needed
            if key in self._title_to_doc:
                try:
                    uid = getattr(doc, "uid", "")[:6]
                    key = f"{title} [{uid}]"
                except Exception:
                    n = 2
                    while f"{title} ({n})" in self._title_to_doc:
                        n += 1
                    key = f"{title} ({n})"

            self._title_to_doc[key] = doc
            self.combo_c1.addItem(key)
            self.combo_c2.addItem(key)

        if self.combo_c1.count() == 0:
            self.combo_c1.addItem("<no image views>")
            self.combo_c2.addItem("<no image views>")
            self.btn_extract.setEnabled(False)
        else:
            self.btn_extract.setEnabled(True)

    def _resolve_doc(self, combo: QComboBox, label: str):
        key = combo.currentText()
        doc = self._title_to_doc.get(key)
        if doc is None:
            raise RuntimeError(f"Please select a valid document for {label}.")
        if getattr(doc, "image", None) is None:
            raise RuntimeError(f"Selected {label} document has no image.")
        return key, doc

    # ----------------- extraction workflow -----------------
    def _do_extract(self):
        try:
            key_c1, doc_c1 = self._resolve_doc(self.combo_c1, "C1")
            img_c1 = _to_float01(doc_c1.image)

            # Ensure at least 3 channels for C1
            if img_c1.ndim == 2:
                img_c1 = img_c1[..., None]
            if img_c1.shape[2] == 1:
                img_c1 = np.repeat(img_c1, 3, axis=2)
            if img_c1.shape[2] != 3:
                raise RuntimeError(f"C1 image must be RGB or mono: got shape {img_c1.shape}")

            coeffs = self.coeff_editor.get_coefficients()
            iterations = int(self.spn_iter.value())
            use_weighted = bool(self.chk_weighted.isChecked())
            clip_negative = bool(self.chk_clip_neg.isChecked())
            c1_only = bool(self.chk_c1_only.isChecked())

            if c1_only:
                # C1-only mode: ignore C2 and SII, no self-cal iterations.
                self.ctx.log(
                    "[NBExtractor_Wp] starting extraction (Filter-1-only mode)\n"
                    f"  C1 = {key_c1}\n"
                    f"  shape = {img_c1.shape}\n"
                    f"  weighted = {use_weighted}, clip_negative = {clip_negative}\n"
                    "  note: self-calibration iterations are ignored in C1-only mode."
                )

                ha, oiii, stats, final_coeffs = run_extraction_c1_only(
                    img_c1,
                    coeffs,
                    clip_negative=clip_negative,
                    use_weighted=use_weighted,
                )

                # Log stats
                try:
                    self.ctx.log("[NBExtractor_Wp] statistics (Filter-1-only):")
                    for ch_name, ch_stats in stats.items():
                        self.ctx.log(
                            f"  {ch_name}: "
                            f"min={ch_stats['min']:.4g}, "
                            f"max={ch_stats['max']:.4g}, "
                            f"mean={ch_stats['mean']:.4g}, "
                            f"median={ch_stats['median']:.4g}, "
                            f"std={ch_stats['std']:.4g}, "
                            f"neg_frac={ch_stats['negative_fraction']:.4g}"
                        )
                except Exception:
                    pass

                base = f"NBExtract({key_c1})"

                # Create Ha / OIII docs (mono images)
                try:
                    self.ctx.open_new_document(
                        ha.astype(np.float32),
                        metadata={"channel": "Ha", "coefficients": final_coeffs},
                        name=f"{base} [Ha]"
                    )
                except Exception as e:
                    self.ctx.log(f"[NBExtractor_Wp] Failed to create Ha doc: {e}")

                try:
                    self.ctx.open_new_document(
                        oiii.astype(np.float32),
                        metadata={"channel": "OIII", "coefficients": final_coeffs},
                        name=f"{base} [OIII]"
                    )
                except Exception as e:
                    self.ctx.log(f"[NBExtractor_Wp] Failed to create OIII doc: {e}")

                QMessageBox.information(
                    self, "NB Extractor",
                    "Ha and OIII channels created as new documents (C1-only mode).\n"
                    "See log for extraction statistics."
                )
                self.accept()
                return

            # Dual-filter mode (C1 + C2) ------------------------------------
            key_c2, doc_c2 = self._resolve_doc(self.combo_c2, "C2")
            img_c2 = _to_float01(doc_c2.image)

            if img_c2.ndim == 2:
                img_c2 = img_c2[..., None]
            if img_c2.shape[2] == 1:
                img_c2 = np.repeat(img_c2, 3, axis=2)

            if img_c1.shape != img_c2.shape or img_c1.shape[2] != 3:
                raise RuntimeError(
                    "Images must be aligned RGB with identical shape.\n"
                    f"C1: {img_c1.shape}, C2: {img_c2.shape}"
                )

            self.ctx.log(
                "[NBExtractor_Wp] starting extraction (dual-filter mode)\n"
                f"  C1 = {key_c1}\n"
                f"  C2 = {key_c2}\n"
                f"  shape = {img_c1.shape}\n"
                f"  iterations = {iterations}, weighted = {use_weighted}, "
                f"clip_negative = {clip_negative}"
            )

            ha, oiii, sii, stats, final_coeffs = run_extraction_full(
                img_c1, img_c2, coeffs,
                iterations=iterations,
                clip_negative=clip_negative,
                use_weighted=use_weighted,
            )

            # Log stats
            try:
                self.ctx.log("[NBExtractor_Wp] statistics:")
                for ch_name, ch_stats in stats.items():
                    self.ctx.log(
                        f"  {ch_name}: "
                        f"min={ch_stats['min']:.4g}, "
                        f"max={ch_stats['max']:.4g}, "
                        f"mean={ch_stats['mean']:.4g}, "
                        f"median={ch_stats['median']:.4g}, "
                        f"std={ch_stats['std']:.4g}, "
                        f"neg_frac={ch_stats['negative_fraction']:.4g}"
                    )
            except Exception:
                pass

            # Build base name for outputs
            base = f"NBExtract({key_c1}, {key_c2})"

            # Create Ha / OIII / SII docs (mono images)
            try:
                self.ctx.open_new_document(
                    ha.astype(np.float32),
                    metadata={"channel": "Ha", "coefficients": final_coeffs},
                    name=f"{base} [Ha]"
                )
            except Exception as e:
                self.ctx.log(f"[NBExtractor_Wp] Failed to create Ha doc: {e}")

            try:
                self.ctx.open_new_document(
                    oiii.astype(np.float32),
                    metadata={"channel": "OIII", "coefficients": final_coeffs},
                    name=f"{base} [OIII]"
                )
            except Exception as e:
                self.ctx.log(f"[NBExtractor_Wp] Failed to create OIII doc: {e}")

            try:
                self.ctx.open_new_document(
                    sii.astype(np.float32),
                    metadata={"channel": "SII", "coefficients": final_coeffs},
                    name=f"{base} [SII]"
                )
            except Exception as e:
                self.ctx.log(f"[NBExtractor_Wp] Failed to create SII doc: {e}")

            QMessageBox.information(
                self, "NB Extractor",
                "Ha, OIII and SII channels created as new documents.\n"
                "See log for extraction statistics."
            )
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "NB Extractor", f"Extraction failed:\n{e}")


# ---------------------------------------------------------------------
# SASpro entry point
# ---------------------------------------------------------------------
def run(ctx):
    """
    SASpro entry point.
    """
    dlg = NBExtractorDialog(ctx)
    dlg.exec()
