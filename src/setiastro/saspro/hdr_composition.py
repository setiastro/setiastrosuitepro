# saspro/hdr_composition.py
"""HDR composition of aligned linear frames with different exposure times.

The longest exposure is the base. Saturated pixels are replaced with the
matching pixels of each shorter exposure, scaled by a robust linear fit
measured on the unsaturated overlap. Restored pixels stay above the
saturation threshold on purpose: later frames must not treat that recovered
signal as clipped.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QDoubleSpinBox, QSpinBox, QCheckBox,
    QComboBox, QDialogButtonBox, QApplication, QMessageBox,
)

from setiastro.saspro.help_support import make_help_button


class HDRCompositionError(ValueError):
    """The frames cannot be composed with the current settings."""


_EXP_KEYS = ("EXPTIME", "EXPOSURE", "EXPOSURETIME", "XPOSURE", "EXP-TIME", "EXP_TIME")
# High-pass correlation of a real match (a clipped long exposure against its
# short one) sits near 0.9. A one-pixel miss on sharp stars is only a little
# lower, so the chosen shift has to beat "leave them where they are" as well.
_ALIGN_MIN_SCORE = 0.75
_ALIGN_MARGIN = 0.04
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


def exposure_order_warning(docs) -> str | None:
    """Why the current order may not be longest-first, or None when it is safe."""
    rows = [(_doc_name(doc), exposure_seconds(doc)) for doc in docs]
    if len(rows) < 2:
        return None
    unknown = [name for name, exp in rows if exp is None]
    known = [(name, exp) for name, exp in rows if exp is not None]
    lines = []
    if unknown:
        listed = ", ".join(unknown)
        lines.append(
            f"No exposure time was found for {listed}. "
            "Sort leaves those views at the bottom, so the longest exposure may not be first."
        )
    if known:
        longest_name, longest = max(known, key=lambda row: row[1])
        _first_name, first_exp = rows[0]
        if first_exp is None or first_exp < longest:
            lines.append(
                f"{longest_name} is the longest exposure with a time in the header. "
                "The top of the list has to be the longest exposure."
            )
    if not lines:
        return None
    return " ".join(lines)


def _as_hwc(image: np.ndarray) -> np.ndarray:
    """Mono or RGB as HxWxC. Accepts a bare plane, a trailing channel, or CHW."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        if arr.shape[1] > 4 and arr.shape[2] > 4:
            arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[2] in (1, 3):
        return arr
    raise HDRCompositionError(
        f"Unsupported image shape {tuple(np.asarray(image).shape)}. Use a mono or RGB image."
    )


def _pixel_size(shape_hw: tuple[int, int]) -> str:
    height, width = int(shape_hw[0]), int(shape_hw[1])
    return f"{width}×{height}"


def _channel_name(channels: int) -> str:
    return "mono" if int(channels) == 1 else "RGB"


def _luma(plane: np.ndarray) -> np.ndarray:
    if plane.shape[2] == 1:
        return plane[..., 0]
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return plane[..., 0] * weights[0] + plane[..., 1] * weights[1] + plane[..., 2] * weights[2]


def _phase_shift(base: np.ndarray, other: np.ndarray) -> tuple[float, float]:
    """Translation of ``other`` relative to ``base``.

    ``base[y, x]`` lines up with ``other[y + dy, x + dx]``.
    A Hann window is not applied: it pulls the peak off a bright object
    that does not fill the frame.
    """
    a = np.ascontiguousarray(base, dtype=np.float64)
    b = np.ascontiguousarray(other, dtype=np.float64)
    try:
        import cv2
        (dx, dy), _response = cv2.phaseCorrelate(a, b)
        return float(dx), float(dy)
    except Exception:
        fa = np.fft.rfft2(a)
        fb = np.fft.rfft2(b)
        cross = fa * np.conj(fb)
        cross /= np.maximum(np.abs(cross), 1e-12)
        corr = np.fft.irfft2(cross, s=a.shape)
        peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
        dy = float(peak[0])
        dx = float(peak[1])
        if dy > a.shape[0] / 2:
            dy -= a.shape[0]
        if dx > a.shape[1] / 2:
            dx -= a.shape[1]
        return dx, dy


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    step = max(1, int(np.ceil(max(a.shape) / 512.0)))
    aa = np.asarray(a[::step, ::step], dtype=np.float64).ravel()
    bb = np.asarray(b[::step, ::step], dtype=np.float64).ravel()
    if aa.size < 16 or bb.size != aa.size:
        return 0.0
    aa -= aa.mean()
    bb -= bb.mean()
    denom = float(np.sqrt(np.dot(aa, aa) * np.dot(bb, bb)))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def _estimate_shift(base: np.ndarray, other: np.ndarray) -> tuple[int, int]:
    """Integer pixel shift. Coarse on a downsampled copy, then refined at full scale."""
    height = min(int(base.shape[0]), int(other.shape[0]))
    width = min(int(base.shape[1]), int(other.shape[1]))
    if height < 32 or width < 32:
        return 0, 0
    sample_b = np.asarray(base[:height, :width])
    sample_o = np.asarray(other[:height, :width])
    step = max(1, int(np.ceil(max(height, width) / 1024.0)))
    dx_f, dy_f = _phase_shift(sample_b[::step, ::step], sample_o[::step, ::step])
    dx = int(round(dx_f * step))
    dy = int(round(dy_f * step))
    if step == 1:
        return dx, dy

    y0 = max(0, -dy)
    x0 = max(0, -dx)
    y1 = min(height, int(sample_o.shape[0]) - dy)
    x1 = min(width, int(sample_o.shape[1]) - dx)
    if y1 - y0 < 64 or x1 - x0 < 64:
        return dx, dy
    side_y = min(768, y1 - y0)
    side_x = min(768, x1 - x0)
    patch_b = sample_b[y0:y0 + side_y, x0:x0 + side_x]
    patch_o = sample_o[y0 + dy:y0 + dy + side_y, x0 + dx:x0 + dx + side_x]
    rdx, rdy = _phase_shift(patch_b, patch_o)
    return dx + int(round(rdx)), dy + int(round(rdy))


def _overlap_box(base_hw: tuple[int, int], other_hw: tuple[int, int],
                 dx: int, dy: int) -> tuple[int, int, int, int] | None:
    """Base-image rectangle whose pixels exist in both frames.

    ``base[y, x]`` corresponds to ``other[y + dy, x + dx]``.
    """
    base_h, base_w = base_hw
    other_h, other_w = other_hw
    y0 = max(0, -int(dy))
    x0 = max(0, -int(dx))
    y1 = min(int(base_h), int(other_h) - int(dy))
    x1 = min(int(base_w), int(other_w) - int(dx))
    if y1 <= y0 or x1 <= x0:
        return None
    return x0, y0, x1, y1


def _highpass(image: np.ndarray) -> np.ndarray:
    """Remove the sky so a smooth background cannot pass as an alignment."""
    data = np.asarray(image, dtype=np.float32)
    return data - gaussian_filter(data, sigma=8.0)


def _align_score(base: np.ndarray, other: np.ndarray, dx: int, dy: int) -> float:
    box = _overlap_box(base.shape[:2], other.shape[:2], dx, dy)
    if box is None:
        return -1.0
    x0, y0, x1, y1 = box
    if y1 - y0 < 32 or x1 - x0 < 32:
        return -1.0
    sample_base = base[y0:y1, x0:x1]
    sample_other = other[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return _pearson(_highpass(sample_base), _highpass(sample_other))


def _choose_shift(base: np.ndarray, other: np.ndarray) -> tuple[int, int] | None:
    """Integer shift of ``other`` relative to ``base``, or None when it is not safe.

    A registered pair that differs by a pixel of padding scores best at (0, 0).
    A real crop offset has to beat that, not merely clear a low correlation bar:
    a one-pixel miss on a nebula still correlates well and would rebuild each
    star core from its neighbour.
    """
    guess_x, guess_y = _estimate_shift(base, other)
    zero = _align_score(base, other, 0, 0)
    best = (0, 0)
    best_score = zero
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cand = (int(guess_x) + ox, int(guess_y) + oy)
            if cand == (0, 0):
                continue
            score = _align_score(base, other, cand[0], cand[1])
            if score > best_score:
                best, best_score = cand, score
    if best != (0, 0) and best_score < zero + _ALIGN_MARGIN:
        best, best_score = (0, 0), zero
    if best_score < _ALIGN_MIN_SCORE:
        return None
    return best


def _match_planes(planes: list[np.ndarray], names: list[str]
                  ) -> tuple[list[np.ndarray], str | None, tuple[int, int]]:
    """One channel count, and one pixel grid.

    Different crop windows can change the size and slide the content. The shift
    is measured, and only the pixels that actually correspond are kept.
    The third value is the origin of that overlap in the base image, so the
    plate solution can move with the pixels.
    """
    channels = {int(plane.shape[2]) for plane in planes}
    if len(channels) != 1:
        detail = ", ".join(
            f"{name} is {_channel_name(plane.shape[2])} {_pixel_size(plane.shape[:2])}"
            for name, plane in zip(names, planes)
        )
        raise HDRCompositionError(
            "These views do not have the same channels. " + detail + "."
        )

    if len({plane.shape[:2] for plane in planes}) == 1:
        return planes, None, (0, 0)

    detail = ", ".join(
        f"{name} is {_pixel_size(plane.shape[:2])}" for name, plane in zip(names, planes)
    )
    base = planes[0]
    shifts: list[tuple[int, int]] = [(0, 0)]
    for plane, name in zip(planes[1:], names[1:]):
        chosen = _choose_shift(_luma(base), _luma(plane))
        if chosen is None:
            raise HDRCompositionError(
                "These views are not the same size and their pixels do not line up: "
                + detail + ". Register them, then crop both with the same pixel rectangle."
            )
        dx, dy = chosen
        shifts.append((int(dx), int(dy)))

    boxes = []
    for plane, (dx, dy) in zip(planes, shifts):
        box = _overlap_box(base.shape[:2], plane.shape[:2], dx, dy)
        if box is None:
            raise HDRCompositionError(
                "These views do not overlap after alignment: " + detail + "."
            )
        boxes.append(box)
    x0 = max(box[0] for box in boxes)
    y0 = max(box[1] for box in boxes)
    x1 = min(box[2] for box in boxes)
    y1 = min(box[3] for box in boxes)
    if x1 - x0 < 32 or y1 - y0 < 32:
        raise HDRCompositionError(
            "These views barely overlap: " + detail + "."
        )

    aligned = []
    for plane, (dx, dy) in zip(planes, shifts):
        aligned.append(plane[y0 + dy:y1 + dy, x0 + dx:x1 + dx])

    shift_bits = []
    for name, (dx, dy) in zip(names[1:], shifts[1:]):
        if dx or dy:
            shift_bits.append(f"{name} is offset by Δx={dx:+d}, Δy={dy:+d} px from {names[0]}")
    shift_text = ("; ".join(shift_bits) + ". ") if shift_bits else "No offset detected. "
    note = (
        f"Views were not the same size ({detail}). {shift_text}"
        f"Used the overlapping {_pixel_size((y1 - y0, x1 - x0))} area."
    )
    return aligned, note, (int(x0), int(y0))


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
    labels = list(names) if names is not None else [f"Image {i + 1}" for i in range(len(planes))]
    if len(labels) != len(planes):
        labels = [f"Image {i + 1}" for i in range(len(planes))]
    planes, size_note, crop_origin = _match_planes(planes, labels)

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
    if size_note and steps:
        steps[0]["size_note"] = size_note
        steps[0]["crop_origin"] = (int(crop_origin[0]), int(crop_origin[1]))
        steps[0]["crop_size"] = (int(planes[0].shape[1]), int(planes[0].shape[0]))

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
    note = next((step.get("size_note") for step in steps if step.get("size_note")), None)
    if note:
        lines.append(str(note))
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


def _copied_header(header):
    if header is None:
        return None
    if isinstance(header, dict):
        return copy.deepcopy(header)
    if hasattr(header, "copy"):
        try:
            return header.copy()
        except Exception:
            return header
    return header


def _shift_keyword_map(header, x0: int, y0: int, width: int, height: int) -> None:
    for key, delta in (("CRPIX1", x0), ("CRPIX2", y0)):
        if key not in header:
            continue
        try:
            header[key] = float(header[key]) - float(delta)
        except (TypeError, ValueError):
            pass
    try:
        header["NAXIS1"] = int(width)
        header["NAXIS2"] = int(height)
    except Exception:
        pass


def _shift_xisf_keywords(keywords, x0: int, y0: int, width: int, height: int) -> None:
    def entry_for(key):
        entry = keywords.get(key)
        if isinstance(entry, list) and entry:
            entry = entry[0]
        if isinstance(entry, dict) and "value" in entry:
            return entry
        return None

    for key, delta in (("CRPIX1", x0), ("CRPIX2", y0)):
        entry = entry_for(key)
        if entry is None:
            continue
        try:
            entry["value"] = float(entry["value"]) - float(delta)
        except (TypeError, ValueError):
            pass
    for key, value in (("NAXIS1", int(width)), ("NAXIS2", int(height))):
        entry = entry_for(key)
        if entry is not None:
            entry["value"] = value


def crop_header(header, x0: int, y0: int, width: int, height: int):
    """Header for a crop whose origin in the parent image is ``(x0, y0)``.

    CRPIX moves with the origin. CRVAL and the CD matrix stay. The input is
    left unchanged.
    """
    copied = _copied_header(header)
    if isinstance(copied, dict):
        keywords = copied.get("FITSKeywords") or copied.get("fits_keywords")
        if isinstance(keywords, dict):
            _shift_xisf_keywords(keywords, x0, y0, width, height)
        else:
            _shift_keyword_map(copied, x0, y0, width, height)
        return copied
    if copied is not None and hasattr(copied, "keys"):
        _shift_keyword_map(copied, x0, y0, width, height)
    return copied


def publish_hdr_composition(
    main_window,
    dm,
    base,
    result: np.ndarray,
    mask: np.ndarray,
    steps: list[dict],
    *,
    open_mask: bool = False,
) -> str | None:
    """Create the HDR result document (and optional mask). Returns the summary log."""
    if dm is None or not hasattr(dm, "create_document"):
        return None
    meta = dict(getattr(base, "metadata", None) or {})
    meta.pop("file_path", None)
    meta.pop("file_name", None)
    crop = None
    if steps and "crop_origin" in steps[0] and "crop_size" in steps[0]:
        origin = steps[0]["crop_origin"]
        size = steps[0]["crop_size"]
        crop = (int(origin[0]), int(origin[1]), int(size[0]), int(size[1]))
    for key in ("original_header", "fits_header", "wcs_header", "header"):
        header = meta.get(key)
        if header is None:
            continue
        if crop is None:
            meta[key] = _copied_header(header)
        else:
            meta[key] = crop_header(header, crop[0], crop[1], crop[2], crop[3])
    header = meta.get("original_header")
    if hasattr(header, "add_history"):
        try:
            header.add_history("SASpro HDR Composition")
        except Exception:
            pass
    nch = 1 if result.ndim == 2 else int(result.shape[2])
    meta["display_name"] = "HDR Composition"
    meta["bit_depth"] = "32-bit floating point"
    meta["is_mono"] = result.ndim == 2
    meta["source"] = "HDR Composition"
    doc = dm.create_document(result, metadata=meta, name="HDR Composition")
    if hasattr(main_window, "_spawn_subwindow_for"):
        main_window._spawn_subwindow_for(doc)
    if open_mask:
        mask_meta = {
            "display_name": "HDR Composition Mask",
            "bit_depth": "32-bit floating point",
            "is_mono": True,
            "source": "HDR Composition mask",
        }
        mask_doc = dm.create_document(
            mask, metadata=mask_meta, name="HDR Composition Mask",
        )
        if hasattr(main_window, "_spawn_subwindow_for"):
            main_window._spawn_subwindow_for(mask_doc)
    summary = _format_steps(steps, nch)
    if hasattr(main_window, "_log"):
        try:
            main_window._log("HDR Composition → new view 'HDR Composition'\n" + summary)
        except Exception:
            pass
    return summary


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
            "Add the views to combine. Longest exposure at the top. "
            "Frames must already be registered and still linear. "
            "A frame a pixel or two larger is kept where the pixels still line up."
        ))

        pick = QHBoxLayout()
        self.cb_available = QComboBox()
        self.cb_available.setMinimumWidth(220)
        self.btn_add = QPushButton("Add")
        self.btn_remove = QPushButton("Remove")
        self.btn_refresh = QPushButton("Refresh")
        pick.addWidget(self.cb_available, 1)
        pick.addWidget(self.btn_add)
        pick.addWidget(self.btn_remove)
        pick.addWidget(self.btn_refresh)
        root.addLayout(pick)

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list.setMinimumHeight(180)
        root.addWidget(self.list, 1)

        row = QHBoxLayout()
        self.btn_sort = QPushButton("Sort by exposure")
        self.btn_up = QPushButton("Up")
        self.btn_down = QPushButton("Down")
        for button in (self.btn_sort, self.btn_up, self.btn_down):
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

        drag_row = QHBoxLayout()
        try:
            from setiastro.saspro.shortcuts import PresetDragHandle
            from setiastro.saspro.resources import hdrcomposition_path
            from PyQt6.QtGui import QIcon
            self.preset_drag_handle = PresetDragHandle(
                "hdr_composition", self.get_preset,
                icon=QIcon(hdrcomposition_path),
                tooltip="Drag onto a view to run HDR Composition headlessly, "
                        "or onto the shortcut canvas to save these settings.",
                parent=self,
            )
            drag_row.addWidget(self.preset_drag_handle)
        except Exception:
            self.preset_drag_handle = None
        drag_row.addStretch(1)
        drag_row.addWidget(buttons)
        root.addLayout(drag_row)

        self.btn_add.clicked.connect(self._add_selected)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_refresh.clicked.connect(self._refresh_available)
        self.btn_sort.clicked.connect(self._sort_by_exposure)
        self.btn_up.clicked.connect(lambda: self._move(-1))
        self.btn_down.clicked.connect(lambda: self._move(1))

        self._restore()
        self._refresh_available()
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

    def _chosen_ids(self) -> set[int]:
        ids = set()
        for row in range(self.list.count()):
            doc = self.list.item(row).data(_ROLE_DOC)
            if doc is not None:
                ids.add(id(doc))
        return ids

    def _refresh_available(self):
        """Refill the picker. Views already added stay out of it."""
        chosen = self._chosen_ids()
        current = self.cb_available.currentData()
        self.cb_available.blockSignals(True)
        self.cb_available.clear()
        for doc in self._open_docs():
            if id(doc) in chosen:
                continue
            exp = exposure_seconds(doc)
            suffix = f"{exp:.1f} s" if exp is not None else "exposure unknown"
            self.cb_available.addItem(f"{_doc_name(doc)}  —  {suffix}", userData=doc)
        if current is not None:
            for index in range(self.cb_available.count()):
                if self.cb_available.itemData(index) is current:
                    self.cb_available.setCurrentIndex(index)
                    break
        self.cb_available.blockSignals(False)
        self._drop_closed_views()

    def _drop_closed_views(self):
        open_ids = {id(doc) for doc in self._open_docs()}
        for row in range(self.list.count() - 1, -1, -1):
            doc = self.list.item(row).data(_ROLE_DOC)
            if doc is None or id(doc) not in open_ids or getattr(doc, "image", None) is None:
                self.list.takeItem(row)

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_available()

    def _add_doc(self, doc) -> None:
        if doc is None or getattr(doc, "image", None) is None:
            return
        if id(doc) in self._chosen_ids():
            return
        exp = exposure_seconds(doc)
        suffix = f"{exp:.1f} s" if exp is not None else "exposure unknown"
        item = QListWidgetItem(f"{_doc_name(doc)}  —  {suffix}")
        item.setData(_ROLE_DOC, doc)
        item.setData(_ROLE_EXP, exp)
        self.list.addItem(item)
        self.list.setCurrentItem(item)

    def _add_selected(self):
        doc = self.cb_available.currentData()
        if doc is None:
            return
        self._add_doc(doc)
        self._refresh_available()

    def _remove_selected(self):
        row = self.list.currentRow()
        if row < 0:
            return
        self.list.takeItem(row)
        if self.list.count():
            self.list.setCurrentRow(min(row, self.list.count() - 1))
        self._refresh_available()

    def _sort_by_exposure(self):
        items = [self.list.takeItem(0) for _ in range(self.list.count())]

        def sort_key(item: QListWidgetItem):
            exp = item.data(_ROLE_EXP)
            unknown = exp is None
            return (unknown, -(float(exp) if exp is not None else 0.0))

        for item in sorted(items, key=sort_key):
            self.list.addItem(item)
        unknown = [
            self.list.item(row).text().split("  —  ")[0]
            for row in range(self.list.count())
            if self.list.item(row).data(_ROLE_EXP) is None
        ]
        if unknown:
            listed = ", ".join(unknown)
            self.lbl_status.setText(
                f"No exposure time was found for {listed}. "
                "Those views were left at the bottom. Move the longest exposure to the top."
            )

    def _move(self, delta: int):
        row = self.list.currentRow()
        target = row + int(delta)
        if row < 0 or target < 0 or target >= self.list.count():
            return
        item = self.list.takeItem(row)
        self.list.insertItem(target, item)
        self.list.setCurrentRow(target)

    def _chosen_docs(self) -> list:
        docs = []
        for row in range(self.list.count()):
            doc = self.list.item(row).data(_ROLE_DOC)
            if doc is not None and getattr(doc, "image", None) is not None:
                docs.append(doc)
        return docs

    def _apply(self):
        docs = self._chosen_docs()
        if len(docs) < 2:
            QMessageBox.information(self, "HDR Composition", "Add at least two views.")
            return
        warning = exposure_order_warning(docs)
        if warning:
            answer = QMessageBox.question(
                self, "HDR Composition",
                warning + "\n\nApply anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
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

        summary = publish_hdr_composition(
            self.mw, self.dm, docs[0], result, mask, steps,
            open_mask=bool(self.chk_mask.isChecked()),
        )
        if summary is None:
            QMessageBox.warning(self, "HDR Composition", "No document manager.")
            return
        self.lbl_status.setText(summary or "Nothing to replace at this threshold.")
        try:
            remember = getattr(self.mw, "remember_last_headless_command", None)
            if remember is None:
                remember = getattr(self.mw, "_remember_last_headless_command", None)
            if callable(remember):
                remember(
                    "hdr_composition",
                    self.current_preset(),
                    description="HDR Composition",
                )
        except Exception:
            pass

    def current_preset(self) -> dict:
        docs = self._chosen_docs()
        return {
            "threshold": float(self.sp_threshold.value()),
            "growth": int(self.sp_growth.value()),
            "smoothness": float(self.sp_smooth.value()),
            "open_mask": bool(self.chk_mask.isChecked()),
            "doc_titles": [_doc_name(doc) for doc in docs],
            "doc_ids": [id(doc) for doc in docs],
        }

    def get_preset(self) -> dict:
        """Alias for PresetDragHandle / shortcut capture."""
        return self.current_preset()

    def apply_preset(self, preset: dict | None = None) -> None:
        """Seed controls (and optionally the view list) from a preset dict."""
        from setiastro.saspro.hdr_composition_preset import hdr_composition_params_from_preset

        params = hdr_composition_params_from_preset(preset)
        self.sp_threshold.setValue(params["threshold"])
        self.sp_growth.setValue(params["growth"])
        self.sp_smooth.setValue(params["smoothness"])
        self.chk_mask.setChecked(params["open_mask"])
        titles = list(params.get("doc_titles") or [])
        if not titles:
            return
        open_by_title = {_doc_name(doc): doc for doc in self._open_docs()}
        self.list.clear()
        for title in titles:
            doc = open_by_title.get(str(title).strip())
            if doc is not None:
                self._add_doc(doc)
        self._refresh_available()
        self._save()
