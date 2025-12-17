#!/usr/bin/env python3
##############################################
# VeraLux — HyperMetric Stretch
# Photometric Hyperbolic Stretch Engine
# Author: Riccardo Paterniti (2025)
# Contact: info@veralux.space
##############################################

# (c) 2025 Riccardo Paterniti
# VeraLux — HyperMetric Stretch
# SPDX-License-Identifier: GPL-3.0-or-later
# Version 1.1.0 (Architecture Upgrade)
##############################################

#!/usr/bin/env python3
# VeraLux — HyperMetric Stretch (Merged & Fixed for SASpro / SETI Astro Suite)
# Full, working script with GUI + headless support
# Entrypoint: run(ctx)
# Version: merged fix + expanded sensors

import sys, traceback
import numpy as np
import math

# Try import PyQt6 for GUI; if unavailable script runs headless
try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                                 QWidget, QLabel, QDoubleSpinBox, QSlider,
                                 QPushButton, QGroupBox, QProgressBar,
                                 QComboBox, QRadioButton, QButtonGroup, QCheckBox)
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    _HAS_QT = True
except Exception:
    _HAS_QT = False

VERSION = "1.1.0 "

# -----------------------------
# Parameters (tweak if desired)
# -----------------------------
PROCESSING_MODE = 'ready_to_use'   # 'ready_to_use' or 'scientific'
DEFAULT_PROFILE = 'Rec.709 (Recommended)'
TARGET_BACKGROUND = 0.20
INITIAL_LOG_D = 2.0
INITIAL_PROTECT_B = 6.0
INITIAL_CONVERGENCE = 3.5
USE_AUTO_SOLVER = True

# Sensor profiles (expanded: IMX family, Canon approximations, Sestar S50/S30)
SENSOR_PROFILES = {
    "Rec.709 (Recommended)": {
        'weights': (0.2126, 0.7152, 0.0722),
        'description': "ITU-R BT.709 standard for sRGB/HDTV",
        'info': "Default choice."
    },

    # Sony IMX family — approximated weights used in astro cameras
    "Sony IMX571 (ASI2600/QHY268)": {
        'weights': (0.2944, 0.5021, 0.2035),
        'description': "Sony IMX571 — ASI2600 / QHY268 (approx)",
        'info': "IMX family (APS-C/fullframe variants)"
    },
    "Sony IMX455 (ASI6200/QHY600)": {
        'weights': (0.2987, 0.5001, 0.2013),
        'description': "Sony IMX455 — ASI6200 / QHY600 (approx)",
        'info': "Full frame reference"
    },
    "Sony IMX533 (ASI533)": {
        'weights': (0.2910, 0.5072, 0.2018),
        'description': "Sony IMX533 — (approx)",
        'info': "Square-ish 1\" style sensors"
    },
    "Sony IMX294 (ASI294)": {
        'weights': (0.3068, 0.5008, 0.1925),
        'description': "Sony IMX294 — (approx)",
        'info': "Popular 4/3\" style sensor"
    },

    # Canon DSLR / Mirrorless (generic approximations)
    "Canon EOS (Generic DSLR)": {
        'weights': (0.2627, 0.6770, 0.0603),
        'description': "Generic Canon DSLR (Bayer tuned approx)",
        'info': "Good default for Canon Bayer cameras"
    },
    "Canon EOS 5D Mark IV (approx)": {
        'weights': (0.2630, 0.6765, 0.0605),
        'description': "Canon 5D Mark IV (approx)",
        'info': "Approximate"
    },
    "Canon EOS R (approx)": {
        'weights': (0.2625, 0.6775, 0.0600),
        'description': "Canon EOS R (approx)",
        'info': "Approximate"
    },

    # Sestar custom sensors (user requested)
    "Sestar S50": {
        'weights': (0.2800, 0.5400, 0.1800),
        'description': "Sestar S50 (user model)",
        'info': "User-specified Sestar profile"
    },
    "Sestar S30": {
        'weights': (0.3000, 0.5000, 0.2000),
        'description': "Sestar S30 (user model)",
        'info': "User-specified Sestar profile"
    },

    # Narrowband presets
    "Narrowband HOO": {
        'weights': (0.5000, 0.2500, 0.2500),
        'description': "HOO palette",
        'info': "H-alpha = R, OIII = G+B"
    },
    "Narrowband SHO": {
        'weights': (0.3333, 0.3400, 0.3267),
        'description': "SHO palette",
        'info': "SII=R, Hα=G, OIII=B"
    }
}

# -----------------------------
# Core engine
# -----------------------------
class VeraLuxCore:
    @staticmethod
    def normalize_input(img_data):
        # Remove NaN/Inf defensively and convert to float32
        img = np.nan_to_num(img_data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if np.issubdtype(img_data.dtype, np.integer):
            info = np.iinfo(img_data.dtype)
            denom = float(info.max) if info.max > 0 else 1.0
            return img / denom
        # floating
        if img.size == 0:
            return img
        current_max = float(np.nanmax(img))
        if current_max <= 1.0 + 1e-5:
            return img
        if current_max <= 255.0:
            return img / 255.0
        if current_max <= 65535.0:
            return img / 65535.0
        return img / 4294967295.0

    @staticmethod
    def calculate_anchor(data_norm):
        # percentiles on subsample to speed up large images
        if data_norm.size == 0:
            return 0.0
        if data_norm.ndim == 3:
            floors = []
            stride = max(1, data_norm.size // 500000)
            for c in range(data_norm.shape[0]):
                channel_flat = data_norm[c].ravel()[::stride]
                floors.append(np.percentile(channel_flat, 0.5))
            anchor = max(0.0, min(floors) - 0.00025)
        else:
            stride = max(1, data_norm.size // 200000)
            anchor = max(0.0, np.percentile(data_norm.ravel()[::stride], 0.5) - 0.00025)
        return float(anchor)

    @staticmethod
    def extract_luminance(data_norm, anchor, weights):
        r_w, g_w, b_w = weights
        img_anchored = np.maximum(data_norm - anchor, 0.0)
        if img_anchored.ndim == 3 and img_anchored.shape[0] == 3:
            L_anchored = (r_w * img_anchored[0] +
                          g_w * img_anchored[1] +
                          b_w * img_anchored[2])
        elif img_anchored.ndim == 3 and img_anchored.shape[0] == 1:
            L_anchored = img_anchored[0]
            img_anchored = img_anchored[0]
        else:
            L_anchored = img_anchored
        return L_anchored, img_anchored

    @staticmethod
    def ghs_stretch(data, D, b, SP=0.0):
        D = max(D, 0.1)
        b = max(b, 0.1)
        term1 = np.arcsinh(D * (data - SP) + b)
        term2 = np.arcsinh(b)
        denom = np.arcsinh(D * (1.0 - SP) + b) - term2
        if denom == 0:
            denom = 1e-12
        return (term1 - term2) / denom

    @staticmethod
    def solve_log_d(luma_sample, target_median, b_val):
        # Binary search in log space to find log_D such that median maps to target
        median_in = np.median(luma_sample)
        if median_in < 1e-9:
            return 2.0
        low_log, high_log = 0.0, 7.0
        for _ in range(40):
            mid_log = (low_log + high_log) / 2.0
            mid_D = 10.0 ** mid_log
            test_val = VeraLuxCore.ghs_stretch(median_in, mid_D, b_val)
            if abs(test_val - target_median) < 1e-4:
                return mid_log
            if test_val < target_median:
                low_log = mid_log
            else:
                high_log = mid_log
        return (low_log + high_log) / 2.0

    @staticmethod
    def apply_mtf(data, m):
        term1 = (m - 1.0) * data
        term2 = (2.0 * m - 1.0) * data - m
        with np.errstate(divide='ignore', invalid='ignore'):
            res = term1 / term2
        return np.nan_to_num(res, nan=0.0, posinf=1.0, neginf=0.0)

# -----------------------------
# Helpers (stable)
# -----------------------------
def safe_std(arr):
    if arr.size == 0:
        return 0.0
    var = float(np.nanvar(arr))
    return math.sqrt(var) if var > 0.0 else 0.0

def adaptive_output_scaling(img_data, working_space="Rec.709 (Recommended)",
                            target_bg=0.20, progress_callback=None):
    if progress_callback:
        progress_callback("Adaptive Scaling: Analyzing Dynamic Range...")
    luma_r, luma_g, luma_b = SENSOR_PROFILES.get(working_space, SENSOR_PROFILES[DEFAULT_PROFILE])['weights']
    is_rgb = (img_data.ndim == 3 and (img_data.shape[0] == 3 or img_data.shape[2] == 3))
    if is_rgb:
        if img_data.shape[0] == 3:
            R, G, B = img_data[0], img_data[1], img_data[2]
        else:
            R, G, B = img_data[...,0], img_data[...,1], img_data[...,2]
        L_raw = luma_r * R + luma_g * G + luma_b * B
    else:
        L_raw = img_data
    median_L = float(np.nanmedian(L_raw))
    std_L = float(safe_std(L_raw))
    min_L = float(np.nanmin(L_raw))
    global_floor = max(min_L, median_L - 2.7 * std_L)
    PEDESTAL = 0.001
    TARGET_SOFT = 0.98; TARGET_HARD = 1.0
    if is_rgb:
        stride = max(1, R.size // 500000)
        soft_r = np.percentile(R.ravel()[::stride], 99.0)
        soft_g = np.percentile(G.ravel()[::stride], 99.0)
        soft_b = np.percentile(B.ravel()[::stride], 99.0)
        soft_ceil = max(soft_r, soft_g, soft_b)
        hard_r = np.percentile(R.ravel()[::stride], 99.99)
        hard_g = np.percentile(G.ravel()[::stride], 99.99)
        hard_b = np.percentile(B.ravel()[::stride], 99.99)
        hard_ceil = max(hard_r, hard_g, hard_b)
    else:
        stride = max(1, L_raw.size // 200000)
        soft_ceil = np.percentile(L_raw.ravel()[::stride], 99.0)
        hard_ceil = np.percentile(L_raw.ravel()[::stride], 99.99)
    if soft_ceil <= global_floor: soft_ceil = global_floor + 1e-6
    if hard_ceil <= soft_ceil: hard_ceil = soft_ceil + 1e-6
    scale_contrast = (TARGET_SOFT - PEDESTAL) / (soft_ceil - global_floor + 1e-9)
    scale_safety = (TARGET_HARD - PEDESTAL) / (hard_ceil - global_floor + 1e-9)
    final_scale = min(scale_contrast, scale_safety)
    if progress_callback:
        mode = "PROTECTION" if scale_safety < scale_contrast else "CONTRAST"
        progress_callback(f"Expansion Mode: {mode} (Scale: {final_scale:.3f})")
    def expand_channel(c):
        return np.clip((c - global_floor) * final_scale + PEDESTAL, 0.0, 1.0)
    if is_rgb:
        if img_data.shape[0] == 3:
            img_data[0] = expand_channel(R)
            img_data[1] = expand_channel(G)
            img_data[2] = expand_channel(B)
        else:
            img_data[...,0] = expand_channel(R)
            img_data[...,1] = expand_channel(G)
            img_data[...,2] = expand_channel(B)
        L = luma_r * img_data[0] + luma_g * img_data[1] + luma_b * img_data[2]
    else:
        img_data = expand_channel(L_raw)
        L = img_data
    current_bg = float(np.nanmedian(L))
    if current_bg > 0.0 and current_bg < 1.0 and abs(current_bg - target_bg) > 1e-3:
        if progress_callback: progress_callback(f"Applying MTF (Bg: {current_bg:.3f} -> {target_bg})")
        x = current_bg; y = target_bg
        denom = (x * (2.0 * y - 1.0) - y)
        m = 1.0
        if abs(denom) >= 1e-12:
            m = (x * (y - 1.0)) / denom
        if is_rgb:
            if img_data.shape[0] == 3:
                img_data[0] = VeraLuxCore.apply_mtf(img_data[0], m)
                img_data[1] = VeraLuxCore.apply_mtf(img_data[1], m)
                img_data[2] = VeraLuxCore.apply_mtf(img_data[2], m)
            else:
                img_data[...,0] = VeraLuxCore.apply_mtf(img_data[...,0], m)
                img_data[...,1] = VeraLuxCore.apply_mtf(img_data[...,1], m)
                img_data[...,2] = VeraLuxCore.apply_mtf(img_data[...,2], m)
        else:
            img_data = VeraLuxCore.apply_mtf(img_data, m)
    return img_data

def apply_ready_to_use_soft_clip(img_data, threshold=0.98, rolloff=2.0, progress_callback=None):
    if progress_callback: progress_callback(f"Final Polish: Soft-clip > {threshold:.2f}")
    def soft_clip_channel(c, thresh, roll):
        mask = c > thresh
        result = c.copy()
        if np.any(mask):
            t = (c[mask] - thresh) / (1.0 - thresh + 1e-9)
            t = np.clip(t, 0.0, 1.0)
            f = 1.0 - np.power(1.0 - t, roll)
            result[mask] = thresh + (1.0 - thresh) * f
        return np.clip(result, 0.0, 1.0)
    if img_data.ndim == 3:
        for i in range(img_data.shape[0]):
            img_data[i] = soft_clip_channel(img_data[i], threshold, rolloff)
    else:
        img_data = soft_clip_channel(img_data, threshold, rolloff)
    return img_data

def process_veralux_v6(img_data, log_D, protect_b, convergence_power,
                       working_space=DEFAULT_PROFILE,
                       processing_mode="ready_to_use",
                       target_bg=None,
                       progress_callback=None):
    if progress_callback: progress_callback("Normalization & Analysis...")
    img = VeraLuxCore.normalize_input(img_data)
    if img.ndim == 3 and img.shape[2] == 3 and img.shape[0] != 3:
        img = img.transpose(2, 0, 1)
    luma_weights = SENSOR_PROFILES.get(working_space, SENSOR_PROFILES[DEFAULT_PROFILE])['weights']
    is_rgb = (img.ndim == 3 and img.shape[0] == 3)
    if progress_callback: progress_callback("Calculating Anchor...")
    anchor = VeraLuxCore.calculate_anchor(img)
    if progress_callback: progress_callback(f"Extracting Luminance ({working_space})...")
    L_anchored, img_anchored = VeraLuxCore.extract_luminance(img, anchor, luma_weights)
    epsilon = 1e-9
    L_safe = L_anchored + epsilon
    if is_rgb:
        with np.errstate(divide='ignore', invalid='ignore'):
            r_ratio = np.nan_to_num(img_anchored[0] / L_safe, nan=0.0, posinf=0.0, neginf=0.0)
            g_ratio = np.nan_to_num(img_anchored[1] / L_safe, nan=0.0, posinf=0.0, neginf=0.0)
            b_ratio = np.nan_to_num(img_anchored[2] / L_safe, nan=0.0, posinf=0.0, neginf=0.0)
    if progress_callback: progress_callback(f"Stretching (Log D={log_D:.3f})...")
    L_str = VeraLuxCore.ghs_stretch(L_anchored, 10.0 ** log_D, protect_b)
    L_str = np.clip(L_str, 0.0, 1.0)
    if progress_callback: progress_callback("Dynamic Color Convergence...")
    if is_rgb:
        final = np.zeros_like(img)
        k = np.power(L_str, convergence_power)
        r_final = r_ratio * (1.0 - k) + 1.0 * k
        g_final = g_ratio * (1.0 - k) + 1.0 * k
        b_final = b_ratio * (1.0 - k) + 1.0 * k
        final[0] = L_str * r_final
        final[1] = L_str * g_final
        final[2] = L_str * b_final
    else:
        final = L_str
    final = final * (1.0 - 0.005) + 0.005
    final = np.clip(final, 0.0, 1.0).astype(np.float32)
    if processing_mode == "ready_to_use":
        if progress_callback: progress_callback("Ready-to-Use: Star-Safe Expansion...")
        effective_bg = 0.20 if target_bg is None else float(target_bg)
        final = adaptive_output_scaling(final, working_space, effective_bg, progress_callback)
        if progress_callback: progress_callback("Ready-to-Use: Polish...")
        final = apply_ready_to_use_soft_clip(final, 0.98, 2.0, progress_callback)
        if progress_callback: progress_callback("Output ready!")
    else:
        if progress_callback: progress_callback("Scientific mode: Raw output")
    return final

# -----------------------------
# GUI threads (if Qt present)
# -----------------------------
if _HAS_QT:
    class AutoSolverThread(QThread):
        result_ready = pyqtSignal(float)
        def __init__(self, data, target, b_val, luma_weights):
            super().__init__()
            self.data = data; self.target = target; self.b_val = b_val; self.luma_weights = luma_weights
        def run(self):
            try:
                img_norm = VeraLuxCore.normalize_input(self.data)
                if img_norm.ndim == 3 and img_norm.shape[2] == 3:
                    lum = (img_norm[...,0]*self.luma_weights[0] + img_norm[...,1]*self.luma_weights[1] + img_norm[...,2]*self.luma_weights[2]).ravel()
                elif img_norm.ndim == 3 and img_norm.shape[0] == 3:
                    lum = (img_norm[0]*self.luma_weights[0] + img_norm[1]*self.luma_weights[1] + img_norm[2]*self.luma_weights[2]).ravel()
                else:
                    lum = img_norm.ravel()
                if lum.size > 100000:
                    idx = np.random.choice(lum.size, 100000, replace=False)
                    sample = lum[idx]
                else:
                    sample = lum
                valid = sample[sample > 1e-9]
                if len(valid) == 0:
                    self.result_ready.emit(2.0); return
                res = VeraLuxCore.solve_log_d(valid, self.target, self.b_val)
                self.result_ready.emit(res)
            except Exception:
                self.result_ready.emit(2.0)

    class ProcessingThread(QThread):
        finished = pyqtSignal(object); progress = pyqtSignal(str)
        def __init__(self, img, D, b, conv, working_space, processing_mode, target_bg):
            super().__init__()
            self.img = img; self.D = D; self.b = b; self.conv = conv
            self.working_space = working_space; self.processing_mode = processing_mode; self.target_bg = target_bg
        def run(self):
            try:
                res = process_veralux_v6(self.img, self.D, self.b, self.conv, self.working_space, self.processing_mode, self.target_bg, self.progress.emit)
                self.finished.emit(res)
            except Exception as e:
                self.progress.emit(f"Error: {e}")
                self.finished.emit(None)

else:
    AutoSolverThread = None
    ProcessingThread = None

# -----------------------------
# GUI (Qt) — adapted for SASpro
# -----------------------------
if _HAS_QT:
    class VeraLuxInterface:
        def __init__(self, host_ctx, qt_app=None):
            self.siril = host_ctx
            self.app = qt_app
            self.linear_cache = None

            # Basic window
            self.window = QMainWindow()
            self.window.setWindowTitle(f"VeraLux v{VERSION}")
            try:
                if self.app:
                    self.app.setStyle("Fusion")
            except:
                pass

            central = QWidget()
            self.window.setCentralWidget(central)
            layout = QVBoxLayout(central)
            layout.setSpacing(8)

            head_title = QLabel(f"VeraLux HyperMetric Stretch v{VERSION}")
            head_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            head_title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #88aaff;")
            layout.addWidget(head_title)

            subhead = QLabel("Requirement: Linear Data • Color Calibration (SPCC) Applied")
            subhead.setAlignment(Qt.AlignmentFlag.AlignCenter)
            subhead.setStyleSheet("font-size: 9pt; color: #999999; font-style: italic; margin-bottom: 5px;")
            layout.addWidget(subhead)

            # Top row: mode + sensor
            top_row = QHBoxLayout()
            grp_mode = QGroupBox("0. Processing Mode")
            l_mode = QVBoxLayout(grp_mode)
            self.radio_ready = QRadioButton("Ready-to-Use (Aesthetic)")
            self.radio_scientific = QRadioButton("Scientific (Preserve)")
            self.radio_ready.setChecked(True)
            self.mode_group = QButtonGroup()
            self.mode_group.addButton(self.radio_ready, 0)
            self.mode_group.addButton(self.radio_scientific, 1)
            l_mode.addWidget(self.radio_ready); l_mode.addWidget(self.radio_scientific)
            self.label_mode_info = QLabel(); self.label_mode_info.setWordWrap(True)
            self.update_mode_info()
            self.radio_ready.toggled.connect(self.update_mode_info)
            l_mode.addWidget(self.label_mode_info)
            top_row.addWidget(grp_mode)

            grp_space = QGroupBox("1. Sensor Calibration")
            l_space = QVBoxLayout(grp_space)
            l_combo = QHBoxLayout()
            l_combo.addWidget(QLabel("Sensor Profile:"))
            self.combo_profile = QComboBox()
            for profile_name in SENSOR_PROFILES.keys():
                self.combo_profile.addItem(profile_name)
            self.combo_profile.setCurrentText(DEFAULT_PROFILE)
            self.combo_profile.currentTextChanged.connect(self.update_profile_info)
            l_combo.addWidget(self.combo_profile)
            l_space.addLayout(l_combo)
            self.label_profile_info = QLabel(""); self.label_profile_info.setWordWrap(True)
            l_space.addWidget(self.label_profile_info)
            top_row.addWidget(grp_space)
            layout.addLayout(top_row)

            # Stretch controls
            grp_combined = QGroupBox("2. Stretch Engine & Calibration")
            l_combined = QVBoxLayout(grp_combined)
            l_calib = QHBoxLayout()
            l_calib.addWidget(QLabel("Target Background:"))
            self.spin_target = QDoubleSpinBox(); self.spin_target.setRange(0.05, 0.50); self.spin_target.setValue(0.20); self.spin_target.setSingleStep(0.01)
            l_calib.addWidget(self.spin_target)
            self.slide_target = QSlider(Qt.Orientation.Horizontal)
            self.slide_target.setRange(5, 50); self.slide_target.setValue(20)
            self.slide_target.valueChanged.connect(lambda v: self.spin_target.setValue(v/100.0))
            self.spin_target.valueChanged.connect(lambda v: self.slide_target.setValue(int(v*100)))
            l_calib.addWidget(self.slide_target)
            self.btn_auto = QPushButton("⚡ Auto-Calculate Log D"); self.btn_auto.clicked.connect(self.run_solver)
            l_calib.addWidget(self.btn_auto)
            l_combined.addLayout(l_calib)
            l_manual = QHBoxLayout()
            l_manual.addWidget(QLabel("Log D:"))
            self.spin_d = QDoubleSpinBox(); self.spin_d.setRange(0.0, 7.0); self.spin_d.setValue(2.0); self.spin_d.setDecimals(2); self.spin_d.setSingleStep(0.1)
            self.slide_d = QSlider(Qt.Orientation.Horizontal); self.slide_d.setRange(0,700); self.slide_d.setValue(200)
            self.slide_d.valueChanged.connect(lambda v: self.spin_d.setValue(v/100.0))
            self.spin_d.valueChanged.connect(lambda v: self.slide_d.setValue(int(v*100)))
            l_manual.addWidget(self.spin_d); l_manual.addWidget(self.slide_d)
            l_manual.addSpacing(15)
            l_manual.addWidget(QLabel("Protect b:"))
            self.spin_b = QDoubleSpinBox(); self.spin_b.setRange(0.1, 15.0); self.spin_b.setValue(6.0); self.spin_b.setSingleStep(0.1)
            l_manual.addWidget(self.spin_b)
            l_combined.addLayout(l_manual)
            layout.addWidget(grp_combined)

            grp_phys = QGroupBox("3. Physics & Convergence")
            l_phys = QVBoxLayout(grp_phys)
            l_conv = QHBoxLayout()
            l_conv.addWidget(QLabel("Star Core Recovery (White Point):"))
            self.spin_conv = QDoubleSpinBox(); self.spin_conv.setRange(1.0, 10.0); self.spin_conv.setValue(3.5)
            l_conv.addWidget(self.spin_conv)
            l_phys.addLayout(l_conv)
            layout.addWidget(grp_phys)

            self.progress = QProgressBar(); self.progress.setTextVisible(True)
            layout.addWidget(self.progress)
            self.status = QLabel("Ready. Please cache input first."); self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.status)

            btns = QHBoxLayout()
            self.chk_ontop = QCheckBox("Always on top"); self.chk_ontop.setChecked(True); self.chk_ontop.toggled.connect(self.toggle_ontop)
            btns.addWidget(self.chk_ontop)
            b_reset = QPushButton("Default Settings"); b_reset.clicked.connect(self.set_defaults); btns.addWidget(b_reset)
            b_reload = QPushButton("Reload Input"); b_reload.clicked.connect(self.cache_input); btns.addWidget(b_reload)
            b_proc = QPushButton("PROCESS"); b_proc.clicked.connect(self.run_process); btns.addWidget(b_proc)
            b_close = QPushButton("Close"); b_close.clicked.connect(self.window.close); btns.addWidget(b_close)
            layout.addLayout(btns)

            self.update_profile_info(DEFAULT_PROFILE)
            self.window.show()
            self.center_window()
            try:
                self.cache_input()
            except Exception:
                pass

        def update_mode_info(self):
            if self.radio_ready.isChecked():
                text = ("✓ Star-Safe Expansion\n✓ Linked MTF Stretch\n✓ Soft-clip highlights\n✓ Ready for export")
            else:
                text = ("✓ Pure GHS stretch (1.0)\n✓ Manual tone mapping\n✓ Lossless data\n✓ Accurate for scientific")
            self.label_mode_info.setText(text)

        def update_profile_info(self, profile_name):
            if profile_name in SENSOR_PROFILES:
                profile = SENSOR_PROFILES[profile_name]
                weights = profile['weights']
                text = f"Weights: R={weights[0]:.4f}, G={weights[1]:.4f}, B={weights[2]:.4f}\n{profile.get('info','')}"
                self.label_profile_info.setText(text)

        def set_defaults(self):
            self.spin_d.setValue(2.0); self.spin_b.setValue(6.0); self.spin_target.setValue(0.20)
            self.spin_conv.setValue(3.5); self.combo_profile.setCurrentText(DEFAULT_PROFILE)
            self.radio_ready.setChecked(True); self.chk_ontop.setChecked(True)
            self.status.setText("Settings reset to defaults.")

        def center_window(self):
            if self.app:
                screen = self.app.primaryScreen()
                if screen:
                    frame_geo = self.window.frameGeometry()
                    frame_geo.moveCenter(screen.availableGeometry().center())
                    self.window.move(frame_geo.topLeft())

        def toggle_ontop(self, checked):
            pos = self.window.pos()
            if checked:
                self.window.setWindowFlags(self.window.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
            else:
                self.window.setWindowFlags(self.window.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
            self.window.show(); self.window.move(pos)

        # Host integration
        def cache_input(self):
            try:
                if hasattr(self.siril, "connect") and not getattr(self.siril, "connected", True):
                    self.siril.connect()
            except:
                pass
            self.status.setText("Caching Linear Data...")
            if self.app:
                self.app.processEvents()
            try:
                # Prefer documented ctx.get_image()
                try:
                    self.linear_cache = self.siril.get_image()
                except Exception:
                    # fallback to older API names if present
                    if hasattr(self.siril, "get_image_pixeldata"):
                        self.linear_cache = self.siril.get_image_pixeldata()
                    else:
                        self.linear_cache = None
                if self.linear_cache is None:
                    self.status.setText("Error: No image open.")
                else:
                    self.status.setText("Input Cached.")
                    try: self.siril.log("VeraLux: Input Cached.")
                    except: pass
            except Exception as e:
                self.status.setText("Cache error.")
                try: self.siril.log(f"Cache error: {e}")
                except: pass

        def run_solver(self):
            if self.linear_cache is None:
                self.status.setText("No cached image.")
                return
            # Note: btn_auto exists only when GUI built; guard if missing
            try:
                getattr(self, 'btn_auto', None).setEnabled(False)
            except:
                pass
            self.status.setText("Solving..."); self.progress.setRange(0,0)
            tgt = self.spin_target.value(); b = self.spin_b.value(); ws = self.combo_profile.currentText()
            try:
                if _HAS_QT:
                    self.solver = AutoSolverThread(self.linear_cache, tgt, b, SENSOR_PROFILES[ws]['weights'])
                    self.solver.result_ready.connect(self.apply_solver_result)
                    self.solver.start()
                else:
                    arr_norm = VeraLuxCore.normalize_input(self.linear_cache)
                    sample = arr_norm.ravel()
                    res = VeraLuxCore.solve_log_d(sample, tgt, b)
                    self.apply_solver_result(res)
            except Exception as e:
                self.status.setText("Solver error");
                try: self.siril.log(f"Solver error: {e}")
                except: pass
                try:
                    getattr(self, 'btn_auto', None).setEnabled(True)
                except:
                    pass

        def apply_solver_result(self, log_d):
            self.spin_d.setValue(log_d); self.progress.setRange(0,100); self.progress.setValue(100)
            try:
                getattr(self, 'btn_auto', None).setEnabled(True)
            except:
                pass
            self.status.setText(f"Solved: Log D = {log_d:.2f}")
            try: self.siril.log(f"VeraLux Solver: Log D={log_d:.2f}")
            except: pass

        def run_process(self):
            if self.linear_cache is None:
                self.status.setText("No cached image.")
                return
            try:
                try:
                    if hasattr(self.siril, "undo_save_state"):
                        self.siril.undo_save_state(f"VeraLux v{VERSION} Stretch")
                except:
                    pass
                D = float(self.spin_d.value()); b = float(self.spin_b.value()); conv = float(self.spin_conv.value())
                ws = self.combo_profile.currentText(); t_bg = float(self.spin_target.value())
                mode = "ready_to_use" if self.radio_ready.isChecked() else "scientific"
                self.status.setText("Processing..."); self.progress.setRange(0,0)
                img_copy = self.linear_cache.copy()
                if _HAS_QT:
                    self.worker = ProcessingThread(img_copy, D, b, conv, ws, mode, t_bg)
                    self.worker.progress.connect(self.status.setText)
                    self.worker.finished.connect(self.finish_process)
                    self.worker.start()
                else:
                    res = process_veralux_v6(img_copy, D, b, conv, ws, mode, t_bg, None)
                    self.finish_process(res)
            except Exception as e:
                traceback.print_exc()
                try: self.siril.log(f"Processing error: {e}")
                except: pass
                self.status.setText("Processing error")

        def finish_process(self, result_img):
            self.progress.setRange(0,100); self.progress.setValue(100); self.status.setText("Complete.")
            if result_img is None:
                self.status.setText("Processing failed.")
                try: self.siril.log("VeraLux: processing returned None")
                except: pass
                return
            # Ensure layout matches host expectations: convert CHW->HWC if necessary
            try:
                out = result_img
                if out.ndim == 3 and out.shape[0] == 3:
                    out = out.transpose(1,2,0)
                out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
                out = np.clip(out, 0.0, 1.0).astype(np.float32)
                # Set image via documented API
                try:
                    self.siril.set_image(out, step_name=f"VeraLux v{VERSION}")
                except Exception:
                    # fallback: other API names
                    if hasattr(self.siril, "set_image_for"):
                        try:
                            self.siril.set_image_for(self.siril.active_view(), out, step_name=f"VeraLux v{VERSION}")
                        except:
                            raise
                    else:
                        raise
                try:
                    self.siril.log(f"VeraLux v{VERSION}: Processing applied")
                except:
                    pass
            except Exception as e:
                try:
                    np.save("veralux_output.npy", result_img)
                    try: self.siril.log(f"Failed to set image in host: {e}. Saved to veralux_output.npy")
                    except: pass
                except Exception:
                    try: self.siril.log(f"Failed to set image and save: {e}")
                    except: pass

# -----------------------------
# Headless/simple script entrypoint
# -----------------------------
class ProgressLogger:
    def __init__(self, ctx): self.ctx = ctx
    def __call__(self, txt):
        try:
            self.ctx.log(str(txt))
        except Exception:
            print(txt)

def run(ctx):
    """
    Main entrypoint for SASpro scripts. If PyQt available a GUI will open;
    otherwise the script runs headless and replaces the active image.
    """
    try:
        # If Qt available and called interactively, build GUI
        if _HAS_QT:
            app = QApplication.instance()
            if not app:
                app = QApplication(sys.argv)
            gui = VeraLuxInterface(ctx, app)
            # Show GUI and return immediately (SASpro keeps script reference)
            gui.window.show()
            # Keep reference to prevent GC
            ctx._veralux_gui_ref = gui  # attach to context to keep alive
            return

        # Headless path: operate on active image
        ctx.log("VeraLux (headless) — starting")
        img = ctx.get_image()
        if img is None:
            ctx.log("No active image.")
            return
        logger = ProgressLogger(ctx)
        working_space = DEFAULT_PROFILE
        mode = PROCESSING_MODE
        b_val = INITIAL_PROTECT_B
        conv = INITIAL_CONVERGENCE
        target_bg = TARGET_BACKGROUND
        log_d = INITIAL_LOG_D

        # Auto-solver (if enabled)
        if USE_AUTO_SOLVER:
            try:
                logger("Auto-solver: estimating Log D...")
                img_norm = VeraLuxCore.normalize_input(img)
                if img_norm.ndim == 3 and img_norm.shape[2] == 3 and img_norm.shape[0] != 3:
                    img_norm = img_norm.transpose(2,0,1)
                luma_w = SENSOR_PROFILES.get(working_space, SENSOR_PROFILES[DEFAULT_PROFILE])['weights']
                L_anchored, _ = VeraLuxCore.extract_luminance(img_norm, VeraLuxCore.calculate_anchor(img_norm), luma_w)
                valid = L_anchored[L_anchored > 1e-7]
                if valid.size > 0:
                    n = valid.size
                    if n > 100000:
                        idx = np.random.choice(n, 100000, replace=False)
                        sample = valid.ravel()[idx]
                    else:
                        sample = valid.ravel()
                    best_log = VeraLuxCore.solve_log_d(sample, target_bg, b_val)
                    log_d = best_log
                logger(f"Auto-solver result: Log D = {log_d:.3f}")
            except Exception as e:
                logger(f"Auto-solver failed: {e}")
                log_d = INITIAL_LOG_D

        # Process
        out = process_veralux_v6(img, log_d, b_val, conv, working_space, mode, target_bg, logger)

        # Convert CHW->HWC if needed
        if out.ndim == 3 and out.shape[0] == 3:
            out = out.transpose(1,2,0)

        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        # Commit to active document via ctx.set_image (document-manager safe)
        try:
            ctx.set_image(out, step_name=f"VeraLux v{VERSION}")
            logger("Output committed to active document.")
        except Exception as e:
            # fallback: save to a file and log
            try:
                np.save("veralux_output.npy", out)
                ctx.log(f"Failed to set image: {e}. Saved to veralux_output.npy")
            except Exception as e2:
                ctx.log(f"Failed to set image and failed to save output: {e2}")
                raise

        try:
            ctx.run_command('stat', {})
        except Exception:
            pass

        logger("Processing complete.")
    except Exception as e:
        traceback.print_exc()
        try:
            ctx.log(f"VeraLux run error: {e}")
        except:
            print(f"VeraLux run error: {e}")

# Backward-compat
def main(ctx): return run(ctx)

if __name__ == "__main__":
    print("VeraLux SASpro script. Use run(ctx) inside SASpro / SETI Astro Suite.")
