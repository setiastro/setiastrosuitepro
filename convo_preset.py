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
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Operation:"))
        self.op_combo = QComboBox()
        self.op_combo.addItems(["convolution", "deconvolution", "tv"])
        self.op_combo.setCurrentText(op if op in ("convolution", "deconvolution", "tv") else "convolution")
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

        # TV Denoise
        self.form_tv = QFormLayout()
        self.tv_weight = FloatSliderWithEdit(minimum=0.0, maximum=1.0, step=0.01, initial=float(p.get("tv_weight", 0.10)))
        self.tv_iter   = FloatSliderWithEdit(minimum=1, maximum=100, step=1, initial=float(p.get("tv_iter", 10)))
        self.tv_multi  = QCheckBox("Multi-channel"); self.tv_multi.setChecked(bool(p.get("tv_multichannel", True)))
        self.tv_strength = FloatSliderWithEdit(minimum=0.0, maximum=1.0, step=0.01, initial=float(p.get("strength", 1.0)))
        self.form_tv.addRow("TV Weight:", self.tv_weight)
        self.form_tv.addRow("TV Iterations:", self.tv_iter)
        self.form_tv.addRow("", self.tv_multi)
        self.form_tv.addRow("Strength:", self.tv_strength)

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
            self.box_tv.setVisible(v == "tv")
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
        # tv
        return {
            "op": "tv",
            "tv_weight": self.tv_weight.value(),
            "tv_iter": int(self.tv_iter.value()),
            "tv_multichannel": bool(self.tv_multi.isChecked()),
            "strength": self.tv_strength.value(),
        }


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
            cx = float(p.get("center", [W/2, H/2])[0]); cy = float(p.get("center", [W/2, H/2])[1])
            B = larson_sekanina(
                image=img,
                center=(cy, cx),  # (y,x)
                radial_step=float(p.get("ls_rstep", 0.0)),
                angular_step_deg=float(p.get("ls_astep", 1.0)),
                operator=p.get("ls_operator", "Divide")
            )
            # blend modes from dialog
            A = img
            if A.ndim == 3 and A.shape[2] == 3:
                B_rgb, A_rgb = np.repeat(B[...,None], 3, 0), A
            else:
                B_rgb, A_rgb = B[...,None], A[...,None]
            blend_mode = p.get("ls_blend", "SoftLight")
            C = (A_rgb + B_rgb - (A_rgb * B_rgb)) if blend_mode == "Screen" else ((1 - 2 * B_rgb) * (A_rgb**2) + 2 * B_rgb * A_rgb)
            out = np.clip(C, 0.0, 1.0); out = out[...,0] if img.ndim == 2 else out
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

    elif op == "tv":
        from skimage.restoration import denoise_tv_chambolle
        weight = float(p.get("tv_weight", 0.10))
        max_iter = int(p.get("tv_iter", 10))
        multich = bool(p.get("tv_multichannel", True))
        if img.ndim == 3 and multich:
            out = denoise_tv_chambolle(img.astype(np.float32), weight=weight, max_num_iter=max_iter, channel_axis=-1).astype(np.float32)
        elif img.ndim == 3 and img.shape[2] == 3:
            chans = [denoise_tv_chambolle(img[...,c].astype(np.float32), weight=weight, max_num_iter=max_iter, channel_axis=None) for c in range(3)]
            out = np.stack(chans, axis=2).astype(np.float32)
        else:
            out = denoise_tv_chambolle(img.astype(np.float32), weight=weight, max_num_iter=max_iter, channel_axis=None).astype(np.float32)
        out = _blend(img, np.clip(out, 0.0, 1.0), float(p.get("strength", 1.0)))

    else:
        return

    meta = dict(getattr(doc, "metadata", {}) or {})
    meta["source"] = "ConvoDeconvo"
    dm.update_active_document(out, metadata=meta, step_name="Convo/Deconvo (preset)")
