# pro/whitebalance.py
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QWidget, QGroupBox,
    QGridLayout, QSlider, QCheckBox, QPushButton, QMessageBox, QDoubleSpinBox
)

# imageops
from setiastro.saspro.imageops.starbasedwhitebalance import apply_star_based_white_balance
from setiastro.saspro.imageops.stretch import stretch_color_image

# Shared utilities
from setiastro.saspro.widgets.image_utils import (
    to_float01 as _to_float01,
    extract_mask_from_document as _active_mask_array_from_doc
)

from matplotlib import pyplot as plt                # NEW
from matplotlib.patches import Circle               # NEW
from matplotlib.ticker import MaxNLocator           # NEW
# ----------------------------
# Core WB implementations
# ----------------------------
def plot_star_color_ratios_comparison(raw_pixels: np.ndarray, after_pixels: np.ndarray):
    """
    Replicates the SASv2 diagnostic plot: star color ratios before/after WB,
    with an RGB background grid, best-fit line, and neutral markers.
    Expects Nx3 arrays of star RGB samples in [0,1] (or any common scale).
    """
    def compute_ratios(pixels: np.ndarray):
        eps = 1e-8
        rb = pixels[:, 0] / (pixels[:, 2] + eps)  # R/B
        gb = pixels[:, 1] / (pixels[:, 2] + eps)  # G/B
        return rb, gb

    rb_before, gb_before = compute_ratios(raw_pixels)
    rb_after,  gb_after  = compute_ratios(after_pixels)

    # Optional: keep only finite points
    def _finite(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        return x[m], y[m]
    rb_before, gb_before = _finite(rb_before, gb_before)
    rb_after,  gb_after  = _finite(rb_after,  gb_after)

    # Plot bounds + background grid
    rmin, rmax = 0.5, 2.0
    gmin, gmax = 0.5, 2.0
    res = 200

    rb_vals = np.linspace(rmin, rmax, res)
    gb_vals = np.linspace(gmin, gmax, res)
    rb_grid, gb_grid = np.meshgrid(rb_vals, gb_vals)
    rgb_image = np.stack([rb_grid, gb_grid, np.ones_like(rb_grid)], axis=-1)
    rgb_image /= np.maximum(rgb_image.max(axis=2, keepdims=True), 1e-8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    def plot_panel(ax, rb_data, gb_data, title):
        ax.imshow(rgb_image, extent=(rmin, rmax, gmin, gmax), origin='lower', aspect='auto')
        ax.scatter(rb_data, gb_data, alpha=0.6, edgecolors='k', label="Stars")

        if rb_data.size >= 2:
            m, b = np.polyfit(rb_data, gb_data, 1)
            xs = np.linspace(rmin, rmax, 100)
            ax.plot(xs, m * xs + b, 'r--', label=f"Best Fit\ny = {m:.2f}x + {b:.2f}")

        ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
        ax.axvline(1.0, color='gray', linestyle=':', linewidth=1)
        ax.add_patch(Circle((1.0, 1.0), 0.2, fill=False, edgecolor='blue', linestyle='--', linewidth=1.5))
        ax.text(1.03, 1.17, "Neutral Region", color='blue', fontsize=9)

        ax.set_xlim(rmin, rmax); ax.set_ylim(gmin, gmax)
        ax.set_title(f"{title} White Balance")
        ax.set_xlabel("Red / Blue Ratio"); ax.set_ylabel("Green / Blue Ratio")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True); ax.legend()

    plot_panel(ax1, rb_before, gb_before, "Before")
    plot_panel(ax2, rb_after,  gb_after,  "After")

    plt.suptitle("Star Color Ratios with RGB Mapping", fontsize=14)
    plt.tight_layout()
    plt.show(block=False)

def apply_manual_white_balance(img: np.ndarray, r_gain: float, g_gain: float, b_gain: float) -> np.ndarray:
    """Simple per-channel gain, clipped to [0,1]."""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Manual WB requires RGB image.")
    out = _to_float01(img).copy()
    gains = np.array([r_gain, g_gain, b_gain], dtype=np.float32).reshape((1, 1, 3))
    out = np.clip(out * gains, 0.0, 1.0)
    return out.astype(np.float32, copy=False)


def apply_auto_white_balance(img: np.ndarray) -> np.ndarray:
    """
    Gray-world auto WB: scale each channel so its mean equals the overall mean.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Auto WB requires RGB image.")
    rgb = _to_float01(img)
    means = np.mean(rgb, axis=(0, 1))
    overall = float(np.mean(means))
    means = np.where(means <= 1e-8, 1e-8, means)
    scale = (overall / means).reshape((1, 1, 3))
    out = np.clip(rgb * scale, 0.0, 1.0)
    return out.astype(np.float32, copy=False)


# --------------------------------
# Headless entry point for DnD
# --------------------------------
def apply_white_balance_to_doc(doc, preset: Optional[Dict] = None):
    """
    Preset schema:
      {
        "mode": "star" | "manual" | "auto",   # default "star"
        # star mode:
        "threshold": float (default 50),
        "reuse_cached_sources": bool (default True),
        # manual mode:
        "r_gain": float (default 1.0), "g_gain": float (default 1.0), "b_gain": float (default 1.0)
      }
    """
    import numpy as np

    p = dict(preset or {})
    mode = (p.get("mode") or "star").lower()

    base = np.asarray(doc.image).astype(np.float32, copy=False)
    if base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("White Balance requires an RGB image.")

    base_n = _to_float01(base)

    try:
        if mode == "manual":
            r = float(p.get("r_gain", 1.0))
            g = float(p.get("g_gain", 1.0))
            b = float(p.get("b_gain", 1.0))
            out = apply_manual_white_balance(base_n, r, g, b)

        elif mode == "auto":
            out = apply_auto_white_balance(base_n)

        else:  # "star"
            thr = float(p.get("threshold", 50.0))
            reuse = bool(p.get("reuse_cached_sources", True))
            out, _count, _overlay = apply_star_based_white_balance(
                base_n, threshold=thr, autostretch=False,
                reuse_cached_sources=reuse, return_star_colors=False
            )
    except Exception as e:
        # Fallback: if SEP missing or star detection fails, try Auto WB
        if mode == "star":
            try:
                out = apply_auto_white_balance(base_n)
            except Exception:
                raise e
        else:
            raise

    # Destination-mask blend (if any)
    m = _active_mask_array_from_doc(doc)
    if m is not None:
        if out.ndim == 3:
            m3 = np.repeat(m[..., None], 3, axis=2).astype(np.float32, copy=False)
        else:
            m3 = m.astype(np.float32, copy=False)
        base_for_blend = _to_float01(np.asarray(doc.image).astype(np.float32, copy=False))
        out = base_for_blend * (1.0 - m3) + out * m3

    doc.apply_edit(
        out.astype(np.float32, copy=False),
        metadata={"step_name": "White Balance", "preset": p},
        step_name="White Balance",
    )


# -------------------------
# Interactive dialog (UI)
# -------------------------
class WhiteBalanceDialog(QDialog):
    def __init__(self, parent, doc, icon: QIcon | None = None):
        super().__init__(parent)
        self._main = parent
        self.doc = doc

        # Connect to active document change signal
        if hasattr(self._main, "currentDocumentChanged"):
            self._main.currentDocumentChanged.connect(self._on_active_doc_changed)
        if icon:
            self.setWindowIcon(icon)
        self.setWindowTitle(self.tr("White Balance"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.resize(900, 600)

        self._build_ui()
        self._wire_events()

        # default to Star-Based, like SASv2
        self.type_combo.setCurrentText("Star-Based")
        self._update_mode_widgets()
        # kick off a first detection preview
        QTimer.singleShot(200, self._update_star_preview)

    # ---- UI construction ------------------------------------------------
    def _build_ui(self):
        self.main_layout = QVBoxLayout(self)

        # Type selector
        row = QHBoxLayout()
        row.addWidget(QLabel(self.tr("White Balance Type:")))
        self.type_combo = QComboBox()
        self.type_combo.addItems([self.tr("Star-Based"), self.tr("Manual"), self.tr("Auto")])
        row.addWidget(self.type_combo); row.addStretch()
        self.main_layout.addLayout(row)

        # Manual group
        self.manual_group = QGroupBox(self.tr("Manual Gains"))
        g = QGridLayout(self.manual_group)
        self.r_spin = QDoubleSpinBox(); self._cfg_gain(self.r_spin, 1.0)
        self.g_spin = QDoubleSpinBox(); self._cfg_gain(self.g_spin, 1.0)
        self.b_spin = QDoubleSpinBox(); self._cfg_gain(self.b_spin, 1.0)
        g.addWidget(QLabel(self.tr("Red gain:")),   0, 0); g.addWidget(self.r_spin, 0, 1)
        g.addWidget(QLabel(self.tr("Green gain:")), 1, 0); g.addWidget(self.g_spin, 1, 1)
        g.addWidget(QLabel(self.tr("Blue gain:")),  2, 0); g.addWidget(self.b_spin, 2, 1)
        self.main_layout.addWidget(self.manual_group)

        # Star-based controls + preview
        self.star_group = QGroupBox(self.tr("Star-Based Settings"))
        sg = QVBoxLayout(self.star_group)
        # threshold slider
        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel(self.tr("Detection threshold (σ):")))
        self.thr_slider = QSlider(Qt.Orientation.Horizontal)
        self.thr_slider.setMinimum(1); self.thr_slider.setMaximum(100)
        self.thr_slider.setValue(50); self.thr_slider.setTickInterval(10)
        self.thr_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.thr_label = QLabel("50")
        thr_row.addWidget(self.thr_slider); thr_row.addWidget(self.thr_label)
        sg.addLayout(thr_row)

        self.chk_reuse = QCheckBox(self.tr("Reuse cached star detections")); self.chk_reuse.setChecked(True)
        sg.addWidget(self.chk_reuse)

        self.chk_autostretch_overlay = QCheckBox(self.tr("Autostretch overlay preview")); self.chk_autostretch_overlay.setChecked(True)
        sg.addWidget(self.chk_autostretch_overlay)

        # star count + image preview
        self.star_count = QLabel(self.tr("Detecting stars..."))
        sg.addWidget(self.star_count)

        self.preview = QLabel(); self.preview.setMinimumSize(640, 360)
        self.preview.setStyleSheet("border: 1px solid #333;")
        sg.addWidget(self.preview)
        self.main_layout.addWidget(self.star_group)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_apply = QPushButton(self.tr("Apply"))
        self.btn_cancel = QPushButton(self.tr("Cancel"))
        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(self.btn_cancel)
        self.main_layout.addLayout(btn_row)

        self.setLayout(self.main_layout)

        # debounce timer for star preview
        self._debounce = QTimer(self); self._debounce.setSingleShot(True); self._debounce.setInterval(600)
        self._debounce.timeout.connect(self._update_star_preview)

    def _cfg_gain(self, box: QDoubleSpinBox, val: float):
        box.setRange(0.5, 1.5); box.setDecimals(3); box.setSingleStep(0.01); box.setValue(val)

    # ---- events ---------------------------------------------------------
    def _wire_events(self):
        self.type_combo.currentTextChanged.connect(self._update_mode_widgets)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_apply.clicked.connect(self._on_apply)

        self.thr_slider.valueChanged.connect(lambda v: (self.thr_label.setText(str(v)), self._debounce.start()))
        self.chk_autostretch_overlay.toggled.connect(lambda _=None: self._debounce.start())
        self.chk_reuse.toggled.connect(lambda _=None: self._debounce.start())

    def _update_mode_widgets(self):
        t = self.type_combo.currentText()
        self.manual_group.setVisible(t == "Manual")
        self.star_group.setVisible(t == "Star-Based")

    # ---- active document change ------------------------------------
    def _on_active_doc_changed(self, doc):
        """Called when user clicks a different image window."""
        if doc is None or getattr(doc, "image", None) is None:
            return
        self.doc = doc
        # Force fresh star detection by temporarily disabling cache
        old_reuse = self.chk_reuse.isChecked()
        self.chk_reuse.setChecked(False)
        self._update_star_preview()
        self.chk_reuse.setChecked(old_reuse)

    # ---- preview --------------------------------------------------------
    def _update_star_preview(self):
        if self.type_combo.currentText() != "Star-Based":
            return
        try:
            img = _to_float01(np.asarray(self.doc.image))
            thr = float(self.thr_slider.value())
            reuse = bool(self.chk_reuse.isChecked())
            auto = bool(self.chk_autostretch_overlay.isChecked())

            _, count, overlay = apply_star_based_white_balance(
                img, threshold=thr, autostretch=auto,
                reuse_cached_sources=reuse, return_star_colors=False
            )
            self.star_count.setText(self.tr("Detected {0} stars.").format(count))
            # to pixmap
            h, w, _ = overlay.shape
            qimg = QImage((overlay * 255).astype(np.uint8).data, w, h, 3 * w, QImage.Format.Format_RGB888)
            pm = QPixmap.fromImage(qimg).scaled(self.preview.width(), self.preview.height(),
                                                Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation)
            self.preview.setPixmap(pm)
        except Exception as e:
            self.star_count.setText(self.tr("Detection failed."))
            self.preview.clear()

    # ---- apply ----------------------------------------------------------
    def _on_apply(self):
        mode = self.type_combo.currentText()

        # Find the main window that carries the replay machinery
        main = self.parent()
        while main is not None and not hasattr(main, "replay_last_action_on_base"):
            main = main.parent()

        def _record_preset_for_replay(preset: dict):
            """Helper to stash the WB preset on the main window for Replay Last Action."""
            if main is None:
                return
            try:
                main._last_headless_command = {
                    "command_id": "white_balance",
                    "preset": preset,
                }
                if hasattr(main, "_log"):
                    mode_str = str(preset.get("mode", "star")).lower()
                    if mode_str == "manual":
                        r = float(preset.get("r_gain", 1.0))
                        g = float(preset.get("g_gain", 1.0))
                        b = float(preset.get("b_gain", 1.0))
                        main._log(
                            f"[Replay] Recorded White Balance preset "
                            f"(mode=manual, R={r:.3f}, G={g:.3f}, B={b:.3f})"
                        )
                    elif mode_str == "auto":
                        main._log("[Replay] Recorded White Balance preset (mode=auto)")
                    else:
                        thr = float(preset.get("threshold", 50.0))
                        reuse = bool(preset.get("reuse_cached_sources", True))
                        main._log(
                            f"[Replay] Recorded White Balance preset "
                            f"(mode=star, threshold={thr:.1f}, "
                            f"reuse={'yes' if reuse else 'no'})"
                        )
            except Exception:
                # Logging/recording must never break the dialog
                pass

        try:
            if mode == "Manual":
                preset = {
                    "mode": "manual",
                    "r_gain": float(self.r_spin.value()),
                    "g_gain": float(self.g_spin.value()),
                    "b_gain": float(self.b_spin.value()),
                }

                # Record for Replay Last Action
                _record_preset_for_replay(preset)

                # Use the headless helper so doc metadata is consistent
                apply_white_balance_to_doc(self.doc, preset)
                # Dialog stays open - refresh document for next operation
                self._finish_and_close()
                return

            elif mode == "Auto":
                preset = {"mode": "auto"}

                # Record for Replay Last Action
                _record_preset_for_replay(preset)

                apply_white_balance_to_doc(self.doc, preset)
                # Dialog stays open - refresh document for next operation
                self._finish_and_close()
                return

            else:  # --- Star-Based: compute here so we can plot like SASv2 ---
                thr   = float(self.thr_slider.value())
                reuse = bool(self.chk_reuse.isChecked())

                preset = {
                    "mode": "star",
                    "threshold": thr,
                    "reuse_cached_sources": reuse,
                }

                base = _to_float01(
                    np.asarray(self.doc.image).astype(np.float32, copy=False)
                )

                # Ask for star colors so we can plot
                result = apply_star_based_white_balance(
                    base,
                    threshold=thr,
                    autostretch=False,
                    reuse_cached_sources=reuse,
                    return_star_colors=True,
                )

                # Expected: (out, count, overlay, raw_colors, after_colors)
                if len(result) < 5:
                    raise RuntimeError(
                        "Star-based WB did not return star color arrays. "
                        "Ensure return_star_colors=True is supported."
                    )

                out, star_count, _overlay, raw_colors, after_colors = result

                # Optional destination-mask blend, same as headless path
                m = _active_mask_array_from_doc(self.doc)
                if m is not None:
                    m3 = np.repeat(m[..., None], 3, axis=2).astype(np.float32, copy=False)
                    base_for_blend = _to_float01(
                        np.asarray(self.doc.image).astype(np.float32, copy=False)
                    )
                    out = base_for_blend * (1.0 - m3) + out * m3

                # Commit to the document, including the preset in metadata
                self.doc.apply_edit(
                    out.astype(np.float32, copy=False),
                    metadata={"step_name": "White Balance", "preset": preset},
                    step_name="White Balance",
                )

                # Record for Replay Last Action (using the same preset we just stored in metadata)
                _record_preset_for_replay(preset)

                # 🔬 Show the same diagnostic plot SASv2 shows
                if (
                    raw_colors is not None
                    and after_colors is not None
                    and raw_colors.size
                    and after_colors.size
                ):
                    plot_star_color_ratios_comparison(raw_colors, after_colors)

                QMessageBox.information(
                    self,
                    self.tr("White Balance"),
                    self.tr("Star-Based WB applied.\nDetected {0} stars.").format(int(star_count)),
                )
                # Dialog stays open - refresh document for next operation
                self._finish_and_close()
                return

        except Exception as e:
            QMessageBox.critical(self, self.tr("White Balance"), self.tr("Failed to apply White Balance:\n{0}").format(e))

    def _refresh_document_from_active(self):
        """
        Refresh the dialog's document reference to the currently active document.
        This allows reusing the same dialog on different images.
        """
        try:
            main = self.parent()
            if main and hasattr(main, "_active_doc"):
                new_doc = main._active_doc()
                if new_doc is not None and new_doc is not self.doc:
                    self.doc = new_doc
        except Exception:
            pass

    def _finish_and_close(self):
        """
        Close this dialog after a successful apply.
        Use accept() so it behaves like a successful completion.
        """
        try:
            self.accept()
        except Exception:
            self.close()

    def closeEvent(self, ev):
        try:
            if hasattr(self._main, "currentDocumentChanged"):
                self._main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
        except Exception:
            pass
        super().closeEvent(ev)
