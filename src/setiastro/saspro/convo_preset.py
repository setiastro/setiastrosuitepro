# pro/convo_preset.py
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QDialogButtonBox, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QCheckBox
)
from PyQt6.QtCore import Qt

# Reuse widgets/utilities from convo.py
from .convo import (
    ConvoDeconvoDialog, FloatSliderWithEdit,
    make_elliptical_gaussian_psf, van_cittert_deconv, larson_sekanina
)

# ---------------------------- Preset Editor Dialog ----------------------------
class ConvoPresetDialog(QDialog):
    """
    One dialog for all Convo/Deconvo presets (including TV).
    Produces a JSON-safe dict you can stash on a shortcut.
    """
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Convolution / Deconvolution — Preset")
        p = dict(initial or {})
        op = p.get("op", "convolution")

        root = QVBoxLayout(self)

        # --- top: operation selector ---
        # Legacy "tv" presets map onto the "denoise" op (TV Chambolle method).
        if op == "tv":
            op = "denoise"
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Operation:"))
        self.op_combo = QComboBox()
        self.op_combo.addItems(["convolution", "deconvolution", "denoise"])
        self.op_combo.setCurrentText(op if op in ("convolution", "deconvolution", "denoise") else "convolution")
        op_row.addWidget(self.op_combo); op_row.addStretch()
        root.addLayout(op_row)

        # --- stacked parameter forms (we'll toggle visibility) ---
        self.form_conv = QFormLayout()
        self.conv_radius   = FloatSliderWithEdit(minimum=0.1, maximum=200.0, step=0.1, initial=float(p.get("radius", 5.0)), suffix=" px")
        self.conv_kurtosis = FloatSliderWithEdit(minimum=0.1, maximum=10.0,  step=0.1, initial=float(p.get("kurtosis", 2.0)), suffix="σ")
        self.conv_aspect   = FloatSliderWithEdit(minimum=0.1, maximum=10.0,  step=0.1, initial=float(p.get("aspect", 1.0)))
        self.conv_rotation = FloatSliderWithEdit(minimum=0.0, maximum=360.0, step=1.0, initial=float(p.get("rotation", 0.0)), suffix="°")
        self.conv_strength = FloatSliderWithEdit(minimum=0.0, maximum=1.0,   step=0.01, initial=float(p.get("strength", 1.0)))
        self.form_conv.addRow("Radius:", self.conv_radius)
        self.form_conv.addRow("Kurtosis (σ):", self.conv_kurtosis)
        self.form_conv.addRow("Aspect Ratio:", self.conv_aspect)
        self.form_conv.addRow("Rotation:", self.conv_rotation)
        self.form_conv.addRow("Strength:", self.conv_strength)

        self.form_deconv = QFormLayout()
        self.deconv_algo = QComboBox()
        self.deconv_algo.addItems(["Richardson-Lucy", "Wiener", "Larson-Sekanina", "Van Cittert"])
        self.deconv_algo.setCurrentText(p.get("algo", "Richardson-Lucy"))
        self.form_deconv.addRow("Algorithm:", self.deconv_algo)

        # RL/Wiener PSF params
        self.psf_radius   = FloatSliderWithEdit(minimum=0.1, maximum=100.0, step=0.1, initial=float(p.get("psf_radius", 3.0)), suffix=" px")
        self.psf_kurtosis = FloatSliderWithEdit(minimum=0.1, maximum=10.0,  step=0.1, initial=float(p.get("psf_kurtosis", 2.0)), suffix="σ")
        self.psf_aspect   = FloatSliderWithEdit(minimum=0.1, maximum=10.0,  step=0.1, initial=float(p.get("psf_aspect", 1.0)))
        self.psf_rot      = FloatSliderWithEdit(minimum=0.0, maximum=360.0, step=1.0,   initial=float(p.get("psf_rotation", 0.0)), suffix="°")
        self.form_deconv.addRow("PSF Radius:", self.psf_radius)
        self.form_deconv.addRow("PSF Kurtosis:", self.psf_kurtosis)
        self.form_deconv.addRow("PSF Aspect:", self.psf_aspect)
        self.form_deconv.addRow("PSF Rotation:", self.psf_rot)

        # RL options
        self.rl_iter      = FloatSliderWithEdit(minimum=1, maximum=200, step=1, initial=float(p.get("rl_iter", 30)))
        self.rl_reg       = QComboBox(); self.rl_reg.addItems(["None (Plain R–L)", "Tikhonov (L2)", "Total Variation (TV)"])
        self.rl_reg.setCurrentText(p.get("rl_reg", "None (Plain R–L)"))
        self.rl_clip      = QCheckBox("De-ring (bilateral)"); self.rl_clip.setChecked(bool(p.get("rl_dering", True)))
        self.rl_l_only    = QCheckBox("L* only"); self.rl_l_only.setChecked(bool(p.get("luminance_only", True)))
        self.form_deconv.addRow("RL Iterations:", self.rl_iter)
        self.form_deconv.addRow("RL Regularization:", self.rl_reg)
        self.form_deconv.addRow("", self.rl_clip)
        self.form_deconv.addRow("", self.rl_l_only)

        # Wiener options
        self.wiener_nsr   = FloatSliderWithEdit(minimum=0.0, maximum=1.0, step=0.001, initial=float(p.get("wiener_nsr", 0.01)))
        self.wiener_reg   = QComboBox(); self.wiener_reg.addItems(["None (Classical Wiener)", "Tikhonov (L2)"])
        self.wiener_reg.setCurrentText(p.get("wiener_reg", "None (Classical Wiener)"))
        self.wiener_dering= QCheckBox("De-ring pass"); self.wiener_dering.setChecked(bool(p.get("wiener_dering", True)))
        self.form_deconv.addRow("Wiener NSR:", self.wiener_nsr)
        self.form_deconv.addRow("Wiener Regularization:", self.wiener_reg)
        self.form_deconv.addRow("", self.wiener_dering)

        # Larson–Sekanina
        self.ls_rstep     = FloatSliderWithEdit(minimum=0.0, maximum=50.0, step=0.1, initial=float(p.get("ls_rstep", 0.0)), suffix=" px")
        self.ls_astep     = FloatSliderWithEdit(minimum=0.1, maximum=360.0, step=0.1, initial=float(p.get("ls_astep", 1.0)), suffix="°")
        self.ls_operator  = QComboBox(); self.ls_operator.addItems(["Divide", "Subtract"]); self.ls_operator.setCurrentText(p.get("ls_operator", "Divide"))
        self.ls_blend     = QComboBox(); self.ls_blend.addItems(["SoftLight", "Screen"]);   self.ls_blend.setCurrentText(p.get("ls_blend", "SoftLight"))
        self.form_deconv.addRow("LS Radial Step:", self.ls_rstep)
        self.form_deconv.addRow("LS Angular Step:", self.ls_astep)
        self.form_deconv.addRow("LS Operator:", self.ls_operator)
        self.form_deconv.addRow("Blend:", self.ls_blend)

        # Van Cittert
        self.vc_iter      = FloatSliderWithEdit(minimum=1, maximum=1000, step=1, initial=float(p.get("vc_iter", 10)))
        self.vc_relax     = FloatSliderWithEdit(minimum=0.0, maximum=1.0, step=0.01, initial=float(p.get("vc_relax", 0.0)))
        self.form_deconv.addRow("VC Iterations:", self.vc_iter)
        self.form_deconv.addRow("VC Relaxation:", self.vc_relax)

        # Strength (applies to all ops)
        self.deconv_strength = FloatSliderWithEdit(minimum=0.0, maximum=1.0, step=0.01, initial=float(p.get("strength", 1.0)))
        self.form_deconv.addRow("Strength:", self.deconv_strength)

        # Classical Denoise (op="denoise") — method selector + per-method rows
        self.form_tv = QFormLayout()
        self.dn_algo = QComboBox()
        self.dn_algo.addItems([
            "TV Chambolle", "Non-Local Means", "Bilateral",
            "Gaussian", "Median", "Wavelet",
        ])
        self.dn_algo.setCurrentText(str(p.get("denoise_algo", "TV Chambolle")))
        self.form_tv.addRow("Method:", self.dn_algo)

        self.dn_lum_only = QCheckBox("L* only (color — preserves chroma)")
        self.dn_lum_only.setChecked(bool(p.get("lum_only", True)))
        self.form_tv.addRow("", self.dn_lum_only)

        # TV Chambolle
        self.tv_weight = FloatSliderWithEdit(minimum=0.0, maximum=1.0, step=0.01, initial=float(p.get("tv_weight", 0.10)))
        self.tv_iter   = FloatSliderWithEdit(minimum=1, maximum=100, step=1, initial=float(p.get("tv_iter", 10)))
        self.tv_multi  = QCheckBox("Multi-channel"); self.tv_multi.setChecked(bool(p.get("tv_multichannel", True)))
        self.row_tv_weight = self.form_tv.addRow("TV Weight:", self.tv_weight) or None
        self.form_tv.addRow("TV Iterations:", self.tv_iter)
        self.form_tv.addRow("", self.tv_multi)

        # Non-Local Means
        self.nlm_h     = FloatSliderWithEdit(minimum=0.001, maximum=0.30, step=0.001, initial=float(p.get("nlm_h", 0.015)))
        self.nlm_patch = FloatSliderWithEdit(minimum=3, maximum=15, step=2, initial=float(p.get("nlm_patch", 5)), suffix=" px")
        self.nlm_dist  = FloatSliderWithEdit(minimum=3, maximum=21, step=2, initial=float(p.get("nlm_dist", 7)), suffix=" px")
        self.form_tv.addRow("NLM h (strength):", self.nlm_h)
        self.form_tv.addRow("NLM Patch size:", self.nlm_patch)
        self.form_tv.addRow("NLM Search dist:", self.nlm_dist)

        # Bilateral
        self.bil_d     = FloatSliderWithEdit(minimum=1, maximum=25, step=2, initial=float(p.get("bil_d", 9)), suffix=" px")
        self.bil_sc    = FloatSliderWithEdit(minimum=1.0, maximum=200.0, step=1.0, initial=float(p.get("bil_sigma_color", 50.0)))
        self.bil_ss    = FloatSliderWithEdit(minimum=1.0, maximum=200.0, step=1.0, initial=float(p.get("bil_sigma_space", 50.0)))
        self.form_tv.addRow("Bilateral Diameter:", self.bil_d)
        self.form_tv.addRow("Bilateral Sigma colour:", self.bil_sc)
        self.form_tv.addRow("Bilateral Sigma space:", self.bil_ss)

        # Gaussian
        self.gau_sigma = FloatSliderWithEdit(minimum=0.1, maximum=10.0, step=0.1, initial=float(p.get("gau_sigma", 1.0)), suffix=" px")
        self.form_tv.addRow("Gaussian Sigma:", self.gau_sigma)

        # Median
        self.med_size  = FloatSliderWithEdit(minimum=1, maximum=15, step=2, initial=float(p.get("med_size", 3)), suffix=" px")
        self.form_tv.addRow("Median Kernel size:", self.med_size)

        # Wavelet
        self.wav_sigma   = FloatSliderWithEdit(minimum=0.0, maximum=0.30, step=0.001, initial=float(p.get("wav_sigma", 0.0)))
        self.wav_wavelet = QComboBox(); self.wav_wavelet.addItems(["db1","db2","db4","db8","sym4","sym8","coif1","bior1.3"])
        self.wav_wavelet.setCurrentText(str(p.get("wav_wavelet", "db2")))
        self.wav_mode    = QComboBox(); self.wav_mode.addItems(["soft","hard"])
        self.wav_mode.setCurrentText(str(p.get("wav_mode", "soft")))
        self.form_tv.addRow("Wavelet Noise σ (0=auto):", self.wav_sigma)
        self.form_tv.addRow("Wavelet:", self.wav_wavelet)
        self.form_tv.addRow("Wavelet Threshold mode:", self.wav_mode)

        # Strength (all methods)
        self.tv_strength = FloatSliderWithEdit(minimum=0.0, maximum=1.0, step=0.01, initial=float(p.get("strength", 1.0)))
        self.form_tv.addRow("Strength:", self.tv_strength)

        # Per-method row visibility. Map each method to the widgets it owns.
        self._dn_row_map = {
            "TV Chambolle":    [self.tv_weight, self.tv_iter, self.tv_multi],
            "Non-Local Means": [self.nlm_h, self.nlm_patch, self.nlm_dist],
            "Bilateral":       [self.bil_d, self.bil_sc, self.bil_ss],
            "Gaussian":        [self.gau_sigma],
            "Median":          [self.med_size],
            "Wavelet":         [self.wav_sigma, self.wav_wavelet, self.wav_mode],
        }

        def _dn_toggle():
            sel = self.dn_algo.currentText()
            for algo, widgets in self._dn_row_map.items():
                show = (algo == sel)
                for w in widgets:
                    w.setVisible(show)
                    lbl = self.form_tv.labelForField(w)
                    if lbl is not None:
                        lbl.setVisible(show)
        self.dn_algo.currentTextChanged.connect(lambda _: _dn_toggle())
        self._dn_toggle = _dn_toggle  # keep a handle so _toggle() can call it

        # containers to show/hide
        self.box_conv = _wrap_form(self.form_conv)
        self.box_decv = _wrap_form(self.form_deconv)
        self.box_tv   = _wrap_form(self.form_tv)
        root.addWidget(self.box_conv)
        root.addWidget(self.box_decv)
        root.addWidget(self.box_tv)

        def _toggle():
            v = self.op_combo.currentText()
            self.box_conv.setVisible(v == "convolution")
            self.box_decv.setVisible(v == "deconvolution")
            self.box_tv.setVisible(v == "denoise")
            if v == "denoise":
                self._dn_toggle()   # collapse to the selected method's rows
        self.op_combo.currentTextChanged.connect(lambda _: _toggle())
        _toggle()

        # buttons
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def result_dict(self) -> dict:
        op = self.op_combo.currentText()
        if op == "convolution":
            return {
                "op": "convolution",
                "radius": self.conv_radius.value(),
                "kurtosis": self.conv_kurtosis.value(),
                "aspect": self.conv_aspect.value(),
                "rotation": self.conv_rotation.value(),
                "strength": self.conv_strength.value(),
            }
        if op == "deconvolution":
            return {
                "op": "deconvolution",
                "algo": self.deconv_algo.currentText(),
                "psf_radius": self.psf_radius.value(),
                "psf_kurtosis": self.psf_kurtosis.value(),
                "psf_aspect": self.psf_aspect.value(),
                "psf_rotation": self.psf_rot.value(),
                "rl_iter": self.rl_iter.value(),
                "rl_reg": self.rl_reg.currentText(),
                "rl_dering": bool(self.rl_clip.isChecked()),
                "luminance_only": bool(self.rl_l_only.isChecked()),
                "wiener_nsr": self.wiener_nsr.value(),
                "wiener_reg": self.wiener_reg.currentText(),
                "wiener_dering": bool(self.wiener_dering.isChecked()),
                "ls_rstep": self.ls_rstep.value(),
                "ls_astep": self.ls_astep.value(),
                "ls_operator": self.ls_operator.currentText(),
                "ls_blend": self.ls_blend.currentText(),
                "vc_iter": self.vc_iter.value(),
                "vc_relax": self.vc_relax.value(),
                "strength": self.deconv_strength.value(),
                # optional center for LS (x,y) — if omitted we’ll use image center
                # "center": [x, y],
            }
        # denoise
        algo = self.dn_algo.currentText()
        d = {
            "op": "denoise",
            "denoise_algo": algo,
            "lum_only": bool(self.dn_lum_only.isChecked()),
            "strength": self.tv_strength.value(),
        }
        if algo == "TV Chambolle":
            d["tv_weight"]       = self.tv_weight.value()
            d["tv_iter"]         = int(round(self.tv_iter.value()))
            d["tv_multichannel"] = bool(self.tv_multi.isChecked())
        elif algo == "Non-Local Means":
            d["nlm_h"]     = self.nlm_h.value()
            d["nlm_patch"] = int(round(self.nlm_patch.value()))
            d["nlm_dist"]  = int(round(self.nlm_dist.value()))
        elif algo == "Bilateral":
            d["bil_d"]           = int(round(self.bil_d.value()))
            d["bil_sigma_color"] = self.bil_sc.value()
            d["bil_sigma_space"] = self.bil_ss.value()
        elif algo == "Gaussian":
            d["gau_sigma"] = self.gau_sigma.value()
        elif algo == "Median":
            d["med_size"] = int(round(self.med_size.value()))
        elif algo == "Wavelet":
            d["wav_sigma"]   = self.wav_sigma.value()
            d["wav_wavelet"] = self.wav_wavelet.currentText()
            d["wav_mode"]    = self.wav_mode.currentText()
        return d


def _wrap_form(form: QFormLayout):
    from PyQt6.QtWidgets import QWidget, QVBoxLayout
    w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(0,0,0,0); l.addLayout(form)
    return w


# ---------------------------- Headless Apply ----------------------------
def apply_convo_via_preset(main_window, doc, preset: dict):
    """
    Headless executor for Convolution/Deconvolution/TV using the same kernels/flows
    as the dialog. Applies result to `doc` via doc_manager.
    """
    import numpy as np
    from skimage.color import rgb2lab, lab2rgb
    from skimage.restoration import denoise_tv_chambolle

    dm = getattr(main_window, "doc_manager", None) or getattr(main_window, "dm", None)
    if dm is None or doc is None or getattr(doc, "image", None) is None:
        return

    # ⚠️ You can keep or drop this; it no longer matters for the apply step.
    try:
        if hasattr(dm, "set_active_document"):
            dm.set_active_document(doc)
    except Exception:
        pass

    img = np.asarray(doc.image).astype(np.float32, copy=False)
    p = dict(preset or {})
    op = p.get("op", "convolution")

    # Create a dialog instance to reuse its helpers (no UI shown)
    d = ConvoDeconvoDialog(doc_manager=dm, parent=main_window)

    def _blend(a, b, s):
        s = float(max(0.0, min(1.0, s)))
        return np.clip(b * s + a * (1.0 - s), 0.0, 1.0).astype(np.float32)

    if op == "convolution":
        psf = make_elliptical_gaussian_psf(
            float(p.get("radius", 5.0)),
            float(p.get("kurtosis", 2.0)),
            float(p.get("aspect", 1.0)),
            float(p.get("rotation", 0.0)),
        ).astype(np.float32)
        out = d._convolve_color(img, psf)
        out = _blend(img, out, float(p.get("strength", 1.0)))

    elif op == "deconvolution":
        algo = p.get("algo", "Richardson-Lucy")
        if algo in ("Richardson-Lucy", "Wiener"):
            psf = make_elliptical_gaussian_psf(
                float(p.get("psf_radius", 3.0)),
                float(p.get("psf_kurtosis", 2.0)),
                float(p.get("psf_aspect", 1.0)),
                float(p.get("psf_rotation", 0.0)),
            ).astype(np.float32)

        if algo == "Richardson-Lucy":
            iters = int(round(float(p.get("rl_iter", 30))))
            reg   = p.get("rl_reg", "None (Plain R–L)")
            clipf = bool(p.get("rl_dering", True))
            lum_only = bool(p.get("luminance_only", True))
            if lum_only and img.ndim == 3 and img.shape[2] == 3:
                lab = rgb2lab(img); L = (lab[...,0] / 100.0).astype(np.float32)
                Ld = d._richardson_lucy_color(L, psf, iterations=iters, reg_type=reg, clip_flag=clipf)
                lab[...,0] = np.clip(Ld * 100.0, 0.0, 100.0)
                tmp = lab2rgb(lab.astype(np.float32)).astype(np.float32)
                out = np.clip(tmp, 0.0, 1.0)
            else:
                out = d._richardson_lucy_color(img, psf, iterations=iters, reg_type=reg, clip_flag=clipf)
            out = _blend(img, out, float(p.get("strength", 1.0)))

        elif algo == "Wiener":
            nsr   = float(p.get("wiener_nsr", 0.01))
            reg   = p.get("wiener_reg", "None (Classical Wiener)")
            dering= bool(p.get("wiener_dering", True))
            lum_only = bool(p.get("luminance_only", True))
            if lum_only and img.ndim == 3 and img.shape[2] == 3:
                lab = rgb2lab(img); L = (lab[...,0] / 100.0).astype(np.float32)
                Ld = d._wiener_deconv_with_kernel(L, psf, nsr, reg, dering)
                lab[...,0] = np.clip(Ld * 100.0, 0.0, 100.0)
                tmp = lab2rgb(lab.astype(np.float32)).astype(np.float32)
                out = np.clip(tmp, 0.0, 1.0)
            else:
                out = d._wiener_deconv_with_kernel(img, psf, nsr, reg, dering)
                out = np.clip(out, 0.0, 1.0)
            out = _blend(img, out, float(p.get("strength", 1.0)))


        elif algo == "Larson-Sekanina":
            H, W = img.shape[:2]
            cxy = p.get("center", [W/2, H/2])
            cx = float(cxy[0]); cy = float(cxy[1])

            B = larson_sekanina(
                image=img,
                center=(cy, cx),  # (y,x)
                radial_step=float(p.get("ls_rstep", 0.0)),
                angular_step_deg=float(p.get("ls_astep", 1.0)),
                operator=p.get("ls_operator", "Divide")
            )

            A = img
            if A.ndim == 3 and A.shape[2] == 3:
                # ✅ FIX: repeat into channel axis
                B_rgb = np.repeat(B[..., None], 3, axis=2)
                A_rgb = A
            else:
                B_rgb = B[..., None]
                A_rgb = A[..., None]

            blend_mode = p.get("ls_blend", "SoftLight")
            if blend_mode == "Screen":
                C = (A_rgb + B_rgb - (A_rgb * B_rgb))
            else:  # SoftLight
                C = (1 - 2 * B_rgb) * (A_rgb ** 2) + 2 * B_rgb * A_rgb

            out = np.clip(C, 0.0, 1.0)
            out = out[..., 0] if img.ndim == 2 else out
            out = _blend(img, out, float(p.get("strength", 1.0)))

        elif algo == "Van Cittert":
            iters = int(round(float(p.get("vc_iter", 10))))
            relax = float(p.get("vc_relax", 0.0))
            if img.ndim == 3 and img.shape[2] == 3:
                out = np.stack([van_cittert_deconv(img[...,c], iters, relax) for c in range(3)], axis=2).astype(np.float32)
            else:
                out = van_cittert_deconv(img, iters, relax).astype(np.float32)
            out = np.clip(out, 0.0, 1.0)
            out = _blend(img, out, float(p.get("strength", 1.0)))
        else:
            return  # unknown algo

    elif op in ("denoise", "tv"):
        # Reuse the dialog's own six-method denoise engine so headless == interactive.
        # Legacy "tv" presets carry no denoise_algo -> TV Chambolle.
        algo = str(p.get("denoise_algo", "TV Chambolle"))
        try:
            d.denoise_algo_combo.setCurrentText(algo)
        except Exception:
            pass
        try:
            d.denoise_lum_only_chk.setChecked(bool(p.get("lum_only", True)))
        except Exception:
            pass

        # Seed the selected method's parameter widgets (FloatSliderWithEdit stores true floats).
        if algo == "TV Chambolle":
            if "tv_weight" in p:       d.tv_weight_slider.setValue(float(p["tv_weight"]))
            if "tv_iter" in p:         d.tv_iter_slider.setValue(float(p["tv_iter"]))
            if "tv_multichannel" in p: d.tv_multichannel_checkbox.setChecked(bool(p["tv_multichannel"]))
        elif algo == "Non-Local Means":
            if "nlm_h" in p:     d.nlm_h_slider.setValue(float(p["nlm_h"]))
            if "nlm_patch" in p: d.nlm_patch_slider.setValue(float(p["nlm_patch"]))
            if "nlm_dist" in p:  d.nlm_dist_slider.setValue(float(p["nlm_dist"]))
        elif algo == "Bilateral":
            if "bil_d" in p:           d.bil_d_slider.setValue(float(p["bil_d"]))
            if "bil_sigma_color" in p: d.bil_sigma_color_slider.setValue(float(p["bil_sigma_color"]))
            if "bil_sigma_space" in p: d.bil_sigma_space_slider.setValue(float(p["bil_sigma_space"]))
        elif algo == "Gaussian":
            if "gau_sigma" in p: d.gau_sigma_slider.setValue(float(p["gau_sigma"]))
        elif algo == "Median":
            if "med_size" in p: d.med_size_slider.setValue(float(p["med_size"]))
        elif algo == "Wavelet":
            if "wav_sigma" in p:   d.wav_sigma_slider.setValue(float(p["wav_sigma"]))
            if "wav_wavelet" in p: d.wav_wavelet_combo.setCurrentText(str(p["wav_wavelet"]))
            if "wav_mode" in p:    d.wav_mode_combo.setCurrentText(str(p["wav_mode"]))

        denoised = d._apply_classical_denoise(img.astype(np.float32))
        out = _blend(img, np.clip(denoised, 0.0, 1.0), float(p.get("strength", 1.0)))

    else:
        return

    meta = dict(getattr(doc, "metadata", {}) or {})
    meta["source"] = "ConvoDeconvo"

    try:
        if hasattr(doc, "apply_edit"):
            # Let Document handle full vs ROI, history, etc.
            doc.apply_edit(
                out.astype(np.float32, copy=False),
                metadata=meta,
                step_name="Convo/Deconvo (preset)",
            )
        else:
            # Fallback for legacy paths
            if hasattr(dm, "set_active_document"):
                dm.set_active_document(doc)
            dm.update_active_document(
                out.astype(np.float32, copy=False),
                metadata=meta,
                step_name="Convo/Deconvo (preset)",
            )
    except Exception:
        # Re-raise so replay_last_action_on_base can show the warning
        raise

def run_convo_via_preset(main, doc_or_preset=None, preset: dict | None = None, *, target_doc=None):
    """
    Headless Convo/Deconvo/TV entrypoint for CommandSpec + Replay.

    Supports BOTH call shapes:
      1) New CommandRunner shape:
            run_convo_via_preset(main, target_doc, preset)
      2) Legacy shape:
            run_convo_via_preset(main, preset_dict, target_doc=doc)
            run_convo_via_preset(main, preset_dict)
    """

    from PyQt6.QtWidgets import QMessageBox

    # ---- Interpret arguments for backward compat / new executor ----
    if preset is None and isinstance(doc_or_preset, dict):
        # Legacy: (main, preset_dict, target_doc=?)
        p = dict(doc_or_preset or {})
        doc = target_doc
    else:
        # New executor: (main, doc, preset_dict)
        p = dict(preset or {})
        doc = target_doc if target_doc is not None else doc_or_preset

    # Resolve active doc if still None
    if doc is None:
        d = getattr(main, "_active_doc", None)
        doc = d() if callable(d) else d

    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.warning(main, "Convolution / Deconvolution", "Load an image first.")
        return

    # ---- Record for Replay ----
    try:
        remember = getattr(main, "remember_last_headless_command", None)
        if remember is None:
            remember = getattr(main, "_remember_last_headless_command", None)

        if callable(remember):
            # IMPORTANT: store canonical id that exists in registry
            remember("convo", p, description="Convolution / Deconvolution")
        else:
            setattr(main, "_last_headless_command", {
                "command_id": "convo",
                "preset": dict(p),
            })
    except Exception:
        pass

    apply_convo_via_preset(main, doc, p)

def open_convo_with_preset(main_window, preset: dict | None = None):
    from PyQt6.QtGui import QIcon

    # Resolve doc: active MDI subwindow first, then docman fallback.
    doc = None
    try:
        sw = main_window.mdi.activeSubWindow()
        if sw is not None:
            doc = getattr(sw.widget(), "document", None)
    except Exception:
        doc = None
    dm = getattr(main_window, "doc_manager", None) or getattr(main_window, "dm", None) \
         or getattr(main_window, "docman", None)
    if doc is None and dm is not None:
        doc = (dm.get_active_document() if hasattr(dm, "get_active_document")
               else getattr(dm, "active_document", None))
    if doc is None or getattr(doc, "image", None) is None:
        return None

    # Constructor is (doc_manager, parent, doc) — NOT (parent, doc).
    dlg = ConvoDeconvoDialog(doc_manager=dm, parent=main_window, doc=doc)
    try:
        from setiastro.saspro.resources import convoicon_path
        dlg.setWindowIcon(QIcon(convoicon_path))
    except Exception:
        pass

    # Controls have no on-show reset -> seed before show. LS center defers via showEvent.
    try:
        dlg.seed_from_preset(preset or {})
    except Exception:
        pass

    try:
        main_window._convo_dialog = dlg   # retain against GC (WA_DeleteOnClose)
    except Exception:
        pass

    dlg.show(); dlg.raise_(); dlg.activateWindow()
    return dlg