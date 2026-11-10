# saspro/hdr_composition.py
"""HDR composition of aligned linear frames with different exposure times.

The longest exposure is the base. Saturated pixels are replaced with the
matching pixels of each shorter exposure, scaled by a robust linear fit
measured on the unsaturated overlap. Restored pixels stay above the
saturation threshold on purpose: later frames must not treat that recovered
signal as clipped.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QDoubleSpinBox, QSpinBox, QCheckBox,
    QDialogButtonBox, QApplication, QMessageBox,
)

from setiastro.saspro.help_support import make_help_button


class HDRCompositionError(ValueError):
    """The frames cannot be composed with the current settings."""


_EXP_KEYS = ("EXPTIME", "EXPOSURE", "EXPOSURETIME", "XPOSURE", "EXP-TIME", "EXP_TIME")
_ROLE_DOC = Qt.ItemDataRole.UserRole
_ROLE_EXP = Qt.ItemDataRole.UserRole + 1


def _as_float(value) -> float | None:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, dict):
        value = value.get("value", value.get("Value"))
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out) or out <= 0.0:
        return None
    return out


def _header_value(header, key: str):
    if header is None:
        return None
    if isinstance(header, dict):
        if key in header:
            return header[key]
        fits_kw = header.get("FITSKeywords") or header.get("fits_keywords")
        if isinstance(fits_kw, dict) and key in fits_kw:
            entry = fits_kw[key]
            if isinstance(entry, list) and entry:
                entry = entry[0]
            if isinstance(entry, dict):
                return entry.get("value", entry.get("Value"))
            return entry
        return None
    try:
        if key in header:
            return header[key]
    except Exception:
        return None
    return None


def exposure_seconds(doc) -> float | None:
    """Exposure time in seconds from document metadata or its FITS/XISF header."""
    meta = getattr(doc, "metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    for key in ("EXPTIME", "EXPOSURE", "exptime", "exposure"):
        found = _as_float(meta.get(key))
        if found is not None:
            return found
    header = None
    for key in ("original_header", "fits_header", "wcs_header", "header"):
        if meta.get(key) is not None:
            header = meta.get(key)
            break
    if header is None:
        return None
    for key in _EXP_KEYS:
        found = _as_float(_header_value(header, key))
        if found is not None:
            return found
    return None


def _as_hwc(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim == 3 and arr.shape[2] in (1, 3):
        return arr
    raise HDRCompositionError(
        f"Unsupported image shape {tuple(arr.shape)}. Use a mono or RGB image."
    )


def _disk(radius: int) -> np.ndarray:
    radius = int(radius)
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y) <= (radius * radius)


def _feather(saturated: np.ndarray, growth: int, smoothness: float) -> np.ndarray:
    """Mask that is 1 on the saturated core and ramps to 0 outside it.

    The ramp is outward only. A core smaller than ``smoothness`` is still
    fully replaced; the seam sits in the pixels around it.
    """
    region = np.asarray(saturated, dtype=bool)
    if int(growth) > 0:
        region = binary_dilation(region, structure=_disk(int(growth)))
    if not np.any(region):
        return np.zeros(region.shape, dtype=np.float32)
    if float(smoothness) <= 0.0:
        return region.astype(np.float32)
    # Distance to the nearest inside pixel. Inside the region this is 0.
    outside = distance_transform_edt(~region)
    t = np.clip(outside / float(smoothness), 0.0, 1.0).astype(np.float32, copy=False)
    ramp = 1.0 - t
    # Smoothstep on the way down, still exactly 1 on the whole grown core.
    return ramp * ramp * (3.0 - 2.0 * ramp)


def _sample_valid(valid: np.ndarray, x_plane: np.ndarray, y_plane: np.ndarray,
                  max_n: int = 400_000) -> tuple[np.ndarray, np.ndarray]:
    count = int(valid.sum())
    if count <= max_n:
        return x_plane[valid], y_plane[valid]
    step = int(np.ceil(np.sqrt(count / float(max_n))))
    sub = valid[::step, ::step]
    return x_plane[::step, ::step][sub], y_plane[::step, ::step][sub]


def _robust_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """Sigma-clipped least squares for y = a x + b. Returns None when the fit is unusable."""
    if x.size < 64:
        return None
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if float(np.nanmax(xs) - np.nanmin(xs)) < 1e-5:
        return None
    sel = np.ones(xs.shape[0], dtype=bool)
    a = b = 0.0
    for _ in range(5):
        x_fit = xs[sel]
        y_fit = ys[sel]
        if x_fit.size < 64:
            return None
        x_mean = float(x_fit.mean())
        y_mean = float(y_fit.mean())
        xd = x_fit - x_mean
        denom = float(np.dot(xd, xd))
        if denom < 1e-20:
            return None
        a = float(np.dot(xd, y_fit - y_mean) / denom)
        b = y_mean - a * x_mean
        if not np.isfinite(a) or not np.isfinite(b) or a <= 0.0:
            return None
        resid = ys - (a * xs + b)
        chosen = resid[sel]
        med = float(np.median(chosen))
        mad = float(np.median(np.abs(chosen - med))) * 1.4826
        if mad < 1e-8:
            break
        updated = np.abs(resid - med) <= (3.0 * mad)
        if int(updated.sum()) < 64 or int(updated.sum()) == int(sel.sum()):
            break
        sel = updated
    return float(a), float(b)


def _clipped_median(values: np.ndarray) -> float:
    sel = np.ones(values.shape[0], dtype=bool)
    med = float(np.median(values))
    for _ in range(4):
        chosen = values[sel]
        if chosen.size < 16:
            break
        med = float(np.median(chosen))
        mad = float(np.median(np.abs(chosen - med))) * 1.4826
        if mad < 1e-8:
            break
        updated = np.abs(values - med) <= (3.0 * mad)
        if int(updated.sum()) < 16:
            break
        sel = updated
    return med


def _fit_scale(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float] | None:
    """Scale from the signal above sky, so a sky-dominated frame still yields a slope."""
    if xs.size < 64:
        return None
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    faint = xs <= float(np.percentile(xs, 40))
    if int(faint.sum()) < 16:
        return _robust_line(xs, ys)
    x_sky = float(np.median(xs[faint]))
    y_sky = float(np.median(ys[faint]))
    span = float(np.percentile(xs, 99) - x_sky)
    if span > 1e-5:
        signal = xs > (x_sky + 0.15 * span)
        if int(signal.sum()) >= 64:
            denom = xs[signal] - x_sky
            ratios = (ys[signal] - y_sky) / np.maximum(denom, 1e-8)
            ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
            if ratios.size >= 64:
                scale = _clipped_median(ratios)
                if np.isfinite(scale) and scale > 0.0:
                    return float(scale), float(y_sky - scale * x_sky)
    return _robust_line(xs, ys)


def _fit_planes(long_plane: np.ndarray, short_plane: np.ndarray,
                valid: np.ndarray, short_ceiling: float) -> tuple[float, float] | None:
    usable = np.asarray(valid, dtype=bool)
    usable &= np.isfinite(long_plane) & np.isfinite(short_plane)
    usable &= short_plane < float(short_ceiling)
    usable &= short_plane > -0.05
    # Shared zero borders (registration margins) carry no scale information.
    usable &= ~((np.abs(short_plane) < 1e-6) & (np.abs(long_plane) < 1e-6))
    if int(usable.sum()) < 64:
        return None
    xs, ys = _sample_valid(usable, short_plane, long_plane)
    return _fit_scale(xs, ys)


def compose_hdr(
    images: list[np.ndarray],
    *,
    threshold: float = 0.8,
    growth: int = 1,
    smoothness: float = 7.0,
    names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Compose ``images`` from longest exposure to shortest.

    Returns ``(image, mask, steps)``. ``image`` is linear and may exceed 1.
    ``mask`` is the maximum replacement weight. Each step records per-channel
    ``scales`` / ``offsets`` (``None`` when that channel had nothing to replace).
    """
    if len(images) < 2:
        raise HDRCompositionError("HDR composition needs at least two images.")
    threshold = float(threshold)
    if not (0.0 < threshold < 1.5):
        raise HDRCompositionError("Binarizing threshold must be between 0 and 1.5.")

    planes = [_as_hwc(img) for img in images]
    shape = planes[0].shape
    for plane in planes[1:]:
        if plane.shape != shape:
            raise HDRCompositionError(
                f"Images must be the same size and type. Got {tuple(shape)} and {tuple(plane.shape)}."
            )

    labels = list(names) if names is not None else [f"Image {i + 1}" for i in range(len(planes))]
    if len(labels) != len(planes):
        labels = [f"Image {i + 1}" for i in range(len(planes))]

    current = planes[0].copy()
    saturated = current >= threshold
    combined = np.zeros(current.shape[:2], dtype=np.float32)
    nch = int(current.shape[2])
    steps: list[dict] = []

    for plane, name in zip(planes[1:], labels[1:]):
        scales: list[float | None] = []
        offsets: list[float | None] = []
        step_mask = np.zeros(current.shape[:2], dtype=np.float32)
        for channel in range(nch):
            sat = saturated[..., channel]
            if not np.any(sat):
                scales.append(None)
                offsets.append(None)
                continue
            fit = _fit_planes(
                current[..., channel], plane[..., channel], ~sat, threshold,
            )
            if fit is None:
                ch_name = "L" if nch == 1 else "RGB"[channel]
                raise HDRCompositionError(
                    f"Could not measure a scale factor for {name} ({ch_name}). "
                    "The unsaturated overlap is too small — lower the binarizing "
                    "threshold, or check that the frames are linear and registered."
                )
            scale, offset = fit
            soft = _feather(sat, int(growth), float(smoothness))
            long_ch = current[..., channel]
            filled = scale * plane[..., channel] + offset
            filled = np.where(np.isfinite(filled), filled, long_ch)
            current[..., channel] = long_ch * (1.0 - soft) + filled * soft
            step_mask = np.maximum(step_mask, soft)
            scales.append(float(scale))
            offsets.append(float(offset))
            # A pixel stays saturated only when the shorter frame is clipped there too.
            # Restored signal above `threshold` is real and must be left alone.
            saturated[..., channel] = sat & (plane[..., channel] >= threshold)
        combined = np.maximum(combined, step_mask)
        replaced = float(step_mask.mean()) if step_mask.size else 0.0
        steps.append({
            "name": name,
            "scales": scales,
            "offsets": offsets,
            "replaced": replaced,
        })

    image = current[..., 0] if nch == 1 else current
    return image.astype(np.float32, copy=False), combined, steps


def _doc_name(doc) -> str:
    try:
        name = doc.display_name()
    except Exception:
        name = None
    return name or "Untitled"


def _format_steps(steps: list[dict], nch: int) -> str:
    labels = ["L"] if nch == 1 else ["R", "G", "B"]
    lines = []
    for step in steps:
        parts = []
        for index, label in enumerate(labels):
            scale = step["scales"][index] if index < len(step["scales"]) else None
            offset = step["offsets"][index] if index < len(step["offsets"]) else None
            if scale is None or offset is None:
                parts.append(f"{label} unchanged")
            else:
                parts.append(f"{label} ×{scale:.3f} {offset:+.4f}")
        lines.append(
            f"{step['name']}: {', '.join(parts)} — {100.0 * float(step['replaced']):.2f}% replaced"
        )
    return "\n".join(lines)


class HDRCompositionDialog(QDialog):
    """Pick two or more open linear views. The top of the list is the longest exposure."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self.setWindowTitle("HDR Composition")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass
        self.mw = main_window
        self.dm = getattr(main_window, "doc_manager", None) or getattr(main_window, "dm", None)
        self._settings = QSettings()

        root = QVBoxLayout(self)
        root.addWidget(QLabel(
            "Longest exposure at the top. Frames must already be registered, "
            "the same size, and still linear."
        ))

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list.setMinimumHeight(180)
        root.addWidget(self.list, 1)

        row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_sort = QPushButton("Sort by exposure")
        self.btn_up = QPushButton("Up")
        self.btn_down = QPushButton("Down")
        for button in (self.btn_refresh, self.btn_sort, self.btn_up, self.btn_down):
            row.addWidget(button)
        row.addStretch(1)
        root.addLayout(row)

        form = QFormLayout()
        self.sp_threshold = QDoubleSpinBox()
        self.sp_threshold.setRange(0.05, 1.20)
        self.sp_threshold.setDecimals(3)
        self.sp_threshold.setSingleStep(0.01)
        self.sp_threshold.setValue(0.800)
        self.sp_threshold.setToolTip(
            "Pixels at or above this value in the long exposure are treated as saturated "
            "and replaced. Lower it when cores clip below 0.8."
        )
        form.addRow("Binarizing threshold:", self.sp_threshold)

        self.sp_growth = QSpinBox()
        self.sp_growth.setRange(0, 64)
        self.sp_growth.setValue(1)
        self.sp_growth.setToolTip(
            "Dilate the saturated mask by this many pixels so blooming just outside "
            "the clip is replaced too."
        )
        form.addRow("Mask growth (px):", self.sp_growth)

        self.sp_smooth = QDoubleSpinBox()
        self.sp_smooth.setRange(0.0, 64.0)
        self.sp_smooth.setDecimals(1)
        self.sp_smooth.setSingleStep(0.5)
        self.sp_smooth.setValue(7.0)
        self.sp_smooth.setToolTip(
            "Width, in pixels, of the blend between the long exposure and the "
            "scaled short exposure. The core of the mask stays fully replaced."
        )
        form.addRow("Mask smoothness (px):", self.sp_smooth)

        self.chk_mask = QCheckBox("Also open the replacement mask")
        form.addRow(self.chk_mask)
        root.addLayout(form)

        self.lbl_status = QLabel(
            "The result stays linear and can exceed 1. Autostretch shows the recovered cores."
        )
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        buttons = QDialogButtonBox()
        self.btn_apply = buttons.addButton("Apply", QDialogButtonBox.ButtonRole.AcceptRole)
        self.btn_close = buttons.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        self.btn_help = make_help_button("hdr_composition", self)
        buttons.addButton(self.btn_help, QDialogButtonBox.ButtonRole.ActionRole)
        self.btn_close.clicked.connect(self.reject)
        self.btn_apply.clicked.connect(self._apply)
        root.addWidget(buttons)

        self.btn_refresh.clicked.connect(self._reload)
        self.btn_sort.clicked.connect(self._sort_by_exposure)
        self.btn_up.clicked.connect(lambda: self._move(-1))
        self.btn_down.clicked.connect(lambda: self._move(1))

        self._restore()
        self._reload()
        for widget in (self.sp_threshold, self.sp_growth, self.sp_smooth):
            widget.valueChanged.connect(lambda *_: self._save())
        self.chk_mask.toggled.connect(lambda *_: self._save())

    def _k(self, key: str) -> str:
        return f"hdr_composition/{key}"

    def _restore(self):
        s = self._settings
        try:
            self.sp_threshold.setValue(float(s.value(self._k("threshold"), 0.8)))
            self.sp_growth.setValue(int(s.value(self._k("growth"), 1)))
            self.sp_smooth.setValue(float(s.value(self._k("smoothness"), 7.0)))
            self.chk_mask.setChecked(s.value(self._k("open_mask"), False, type=bool))
        except Exception:
            pass

    def _save(self):
        s = self._settings
        s.setValue(self._k("threshold"), float(self.sp_threshold.value()))
        s.setValue(self._k("growth"), int(self.sp_growth.value()))
        s.setValue(self._k("smoothness"), float(self.sp_smooth.value()))
        s.setValue(self._k("open_mask"), bool(self.chk_mask.isChecked()))

    def _open_docs(self) -> list:
        docs = []
        if self.dm is not None:
            docs = list(getattr(self.dm, "_docs", []) or [])
        if not docs and hasattr(self.mw, "_list_open_docs"):
            try:
                docs = list(self.mw._list_open_docs())
            except Exception:
                docs = []
        out = []
        for doc in docs:
            image = getattr(doc, "image", None)
            if image is None:
                continue
            try:
                _as_hwc(image)
            except HDRCompositionError:
                continue
            out.append(doc)
        return out

    def _reload(self):
        previous = []
        for row in range(self.list.count()):
            item = self.list.item(row)
            previous.append((
                id(item.data(_ROLE_DOC)),
                item.checkState() == Qt.CheckState.Checked,
            ))
        checked = {doc_id: flag for doc_id, flag in previous}
        order = [doc_id for doc_id, _flag in previous]

        docs = self._open_docs()
        by_id = {id(doc): doc for doc in docs}
        ordered = [by_id[doc_id] for doc_id in order if doc_id in by_id]
        ordered.extend(doc for doc in docs if id(doc) not in checked)

        self.list.clear()
        for doc in ordered:
            exp = exposure_seconds(doc)
            suffix = f"{exp:.1f} s" if exp is not None else "exposure unknown"
            item = QListWidgetItem(f"{_doc_name(doc)}  —  {suffix}")
            item.setData(_ROLE_DOC, doc)
            item.setData(_ROLE_EXP, exp)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            known = id(doc) in checked
            item.setCheckState(
                Qt.CheckState.Checked if (not known or checked[id(doc)]) else Qt.CheckState.Unchecked
            )
            self.list.addItem(item)
        if self.list.count() and not any(
            self.list.item(i).data(_ROLE_EXP) is not None for i in range(self.list.count())
        ):
            return
        if not previous:
            self._sort_by_exposure()

    def _sort_by_exposure(self):
        items = [self.list.takeItem(0) for _ in range(self.list.count())]

        def sort_key(item: QListWidgetItem):
            exp = item.data(_ROLE_EXP)
            unknown = exp is None
            return (unknown, -(float(exp) if exp is not None else 0.0))

        for item in sorted(items, key=sort_key):
            self.list.addItem(item)

    def _move(self, delta: int):
        row = self.list.currentRow()
        target = row + int(delta)
        if row < 0 or target < 0 or target >= self.list.count():
            return
        item = self.list.takeItem(row)
        self.list.insertItem(target, item)
        self.list.setCurrentRow(target)

    def _checked_docs(self) -> list:
        docs = []
        for row in range(self.list.count()):
            item = self.list.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                doc = item.data(_ROLE_DOC)
                if doc is not None and getattr(doc, "image", None) is not None:
                    docs.append(doc)
        return docs

    def _apply(self):
        docs = self._checked_docs()
        if len(docs) < 2:
            QMessageBox.information(self, "HDR Composition", "Check at least two images.")
            return
        images = []
        for doc in docs:
            images.append(np.asarray(doc.image))
        names = [_doc_name(doc) for doc in docs]
        self._save()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result, mask, steps = compose_hdr(
                images,
                threshold=float(self.sp_threshold.value()),
                growth=int(self.sp_growth.value()),
                smoothness=float(self.sp_smooth.value()),
                names=names,
            )
        except HDRCompositionError as exc:
            QMessageBox.warning(self, "HDR Composition", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "HDR Composition", f"Failed:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._publish(docs[0], result, mask, steps)

    def _publish(self, base, result: np.ndarray, mask: np.ndarray, steps: list[dict]):
        if self.dm is None or not hasattr(self.dm, "create_document"):
            QMessageBox.warning(self, "HDR Composition", "No document manager.")
            return
        meta = dict(getattr(base, "metadata", None) or {})
        meta.pop("file_path", None)
        meta.pop("file_name", None)
        header = meta.get("original_header")
        if hasattr(header, "copy"):
            try:
                header = header.copy()
                if hasattr(header, "add_history"):
                    header.add_history("SASpro HDR Composition")
                meta["original_header"] = header
            except Exception:
                pass
        elif isinstance(header, dict):
            meta["original_header"] = dict(header)
        nch = 1 if result.ndim == 2 else int(result.shape[2])
        meta["display_name"] = "HDR Composition"
        meta["bit_depth"] = "32-bit floating point"
        meta["is_mono"] = result.ndim == 2
        meta["source"] = "HDR Composition"
        doc = self.dm.create_document(result, metadata=meta, name="HDR Composition")
        if hasattr(self.mw, "_spawn_subwindow_for"):
            self.mw._spawn_subwindow_for(doc)
        if self.chk_mask.isChecked():
            mask_meta = {
                "display_name": "HDR Composition Mask",
                "bit_depth": "32-bit floating point",
                "is_mono": True,
                "source": "HDR Composition mask",
            }
            mask_doc = self.dm.create_document(
                mask, metadata=mask_meta, name="HDR Composition Mask",
            )
            if hasattr(self.mw, "_spawn_subwindow_for"):
                self.mw._spawn_subwindow_for(mask_doc)
        summary = _format_steps(steps, nch)
        self.lbl_status.setText(summary or "Nothing to replace at this threshold.")
        if hasattr(self.mw, "_log"):
            self.mw._log("HDR Composition → new view 'HDR Composition'\n" + summary)
