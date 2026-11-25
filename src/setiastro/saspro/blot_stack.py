"""Two-pass detector-space rejection for a registered, coverage-weighted ImageMM stack."""
from __future__ import annotations

from dataclasses import dataclass
import errno
import math
import mmap
import os
from pathlib import Path
import re
import shutil
import tempfile
import warnings

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.special import ndtri
import sep

from . import astroalign
from .bayer_utils import detect_bayer_pattern
from .legacy.numba_utils import drizzle_deposit_kernel_mono_proj


@dataclass(frozen=True)
class StackOptions:
    scale: float = 1.
    aperture_mm: float = 0.  # Positive selects automatic scale.
    sampling: float = 0.     # arcsec/pixel, or read from reference
    pixfrac: float = 0.      # zero selects automatic width
    kernel: str = "Square"
    align_colors: bool = True
    primary: float | None = None
    adjacent: float | None = None
    iterations: int = 7
    minimum: int = 3
    kappa: float = 1.1
    huber_delta: float = -2.
    strength: float = .85
    tolerance: float = .0005
    fallback_fwhm: float = 3.
    noise_map: bool = True
    protect_stars: bool = True
    luminance: bool = False
    save_iterations: bool = False

    @property
    def blot_thresholds(self) -> tuple[float, float] | None:
        if self.primary is None and self.adjacent is None:
            return None
        # A caller overriding just one threshold keeps the other manual default.
        primary = 5. if self.primary is None else self.primary
        adjacent = 4. if self.adjacent is None else self.adjacent
        if not all(math.isfinite(v) and v > 0 for v in (primary, adjacent)):
            raise ValueError("Blot SNR thresholds must be finite and positive.")
        return primary, adjacent


def read_detector(path):
    """Keep FITS detector pixels, CFA phase and WCS in their on-disk orientation."""
    if Path(path).suffix.lower() == ".xisf":
        from .legacy.image_manager import load_image
        data, hdr, _, _ = load_image(str(path))
        if data is None:
            raise ValueError(f"Could not load {path}")
        data = np.asarray(data)
        if data.ndim == 3:
            data = np.moveaxis(data, -1, 0)
    else:
        data, hdr = fits.getdata(path, header=True, memmap=False)
        if np.issubdtype(data.dtype, np.integer):
            # Match SAS's fractional full-scale units while preserving signed
            # sky and relative throughput between exposures.
            data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    if hdr.get("SASDZ") or hdr.get("CFADRIZ") or hdr.get("SAS_RSMP") or hdr.get("BLOTSTK"):
        raise ValueError(f"{Path(path).name} is already resampled. Select its unregistered detector light.")
    if data.ndim == 2:
        data = data[None, :, :]
    if data.ndim != 3 or data.shape[0] not in (1, 3):
        raise ValueError(f"Expected a mono, CFA or RGB light in {path}")
    data = np.ascontiguousarray(data, dtype=np.float32)
    pattern = None
    if len(data) == 1:
        token = str(hdr.get("BAYERPAT", "")).strip().upper()
        if len(token) == 36 and set(token) == set("RGB"):
            pattern = np.array(["RGB".index(c) for c in token]).reshape(6, 6)
            pattern = np.roll(pattern, (-int(hdr.get("YBAYROFF", 0)),
                                        -int(hdr.get("XBAYROFF", 0))), axis=(0, 1))
        else:
            bayer = detect_bayer_pattern(hdr)
            if bayer:
                pattern = np.array(["RGB".index(c) for c in bayer]).reshape(2, 2)
    return data, hdr, pattern


def sampling_geometry(options, hdr, pattern, count):
    scale = options.scale
    if options.aperture_mm:
        sampling = options.sampling
        if sampling <= 0:
            wcs = WCS(hdr, naxis=2)
            if wcs.has_celestial:
                sampling = np.mean(proj_plane_pixel_scales(wcs.celestial)) * 3600
            elif hdr.get("FOCALLEN", 0) and hdr.get("XPIXSZ", 0):
                sampling = 206.265 * float(hdr["XPIXSZ"]) / float(hdr["FOCALLEN"])
        if sampling <= 0 or not np.isfinite(sampling):
            raise ValueError("Automatic scale needs a reference pixel scale or an input arcsec/pixel value.")
        scale = np.clip(2 * sampling * options.aperture_mm / (206265 * 1.22 * .00065628), .1, 8.)
    density = np.ones(3) if pattern is None else np.bincount(pattern.ravel(), minlength=3) / pattern.size
    ratio = np.sqrt(density.min() / density)
    width = options.pixfrac or np.clip(np.sqrt(20 / (count * density.min())), .125 / ratio.min(), 2.)
    return float(scale), width * ratio


def scaled_header(header, scale):
    from .legacy.image_manager import _scale_wcs_in_header
    hdr = _scale_wcs_in_header(header.copy(), scale)
    if any(key.startswith("PC1_") for key in hdr) and "CD1_1" not in hdr:
        for key in ("CDELT1", "CDELT2"):
            if key not in hdr:
                hdr[key] = 1. / scale
    for key in ("PIXSCALE", "SECPIX", "SECPIX1", "SECPIX2"):
        if isinstance(hdr.get(key), (float, int)):
            hdr[key] /= scale
    for key in list(hdr):
        # SIP polynomials express distortion in pixels relative to CRPIX.
        sip = re.fullmatch(r"(?:A|B|AP|BP)_(\d+)_(\d+)", key)
        if sip:
            hdr[key] *= scale ** (1 - int(sip[1]) - int(sip[2]))
        if key in ("BSCALE", "BZERO", "BLANK", "CHECKSUM", "DATASUM", "DATAMIN", "DATAMAX",
                   "BAYERPAT", "BAYERPATN", "BAYER_PATTERN", "BAYERPATTERN", "CFAPATTERN",
                   "CFA_PATTERN", "COLORFILTERARRAY", "XBAYROFF", "YBAYROFF", "BAYER_EFF", "CFA"):
            del hdr[key]
    return hdr


def channel_data(data, pattern, c):
    plane = data[0 if pattern is not None else c]
    valid = np.isfinite(plane) & (plane != 0)
    if pattern is not None:
        size = len(pattern)
        for y, x in np.argwhere(pattern != c):
            valid[y::size, x::size] = False
    return plane, valid


def detection_proxy(data, pattern, c):
    plane, valid = channel_data(data, pattern, c)
    filled = np.where(valid, plane, 0.)
    if pattern is None:
        return filled
    # Interpolate this color only, solely for centroid detection.
    weights = ndi.gaussian_filter(valid.astype(np.float32), 1.)
    return ndi.gaussian_filter(filled, 1.) / np.maximum(weights, 1e-10)


def match(source, target, projective):
    _, (src, dst) = astroalign.find_transform(source, target, detector="sep", max_control_points=100)
    if projective and len(src) >= 10:
        matrix, keep = cv2.findHomography(src, dst, cv2.RANSAC, 1.5)
    else:
        affine, keep = cv2.estimateAffine2D(src, dst, ransacReprojThreshold=1.5)
        matrix = np.vstack((affine, [0, 0, 1])) if affine is not None else None
    if matrix is None or keep.sum() < 6 or not np.isfinite(matrix).all():
        raise ValueError("Registration needs six matched stars.")
    return matrix


def register(data, pattern, reference, is_reference, align_colors, channels):
    guide = 1 if channels == 3 else 0
    proxy = detection_proxy(data, pattern, guide)
    frame = np.eye(3) if is_reference else match(proxy, reference, True)
    transforms = []
    for c in range(channels):
        matrix = frame
        if align_colors and c != guide:
            matrix = frame @ match(detection_proxy(data, pattern, c), proxy, False)
        transforms.append(matrix)
    return transforms


def deposit(plane, valid, matrix, shape, scale, width, kernel):
    # SAS kernel footprints use pixel edges at integer coordinates.
    edge_matrix = matrix.copy()
    edge_matrix[:2] += .5 * matrix[2]
    science = np.zeros(shape, np.float32)
    weight = np.zeros_like(science)
    code = {"Square": 3, "Gaussian": 2, "Disk": 1, "Point": 0}[kernel]
    drizzle_deposit_kernel_mono_proj(np.where(valid, plane, 0.), edge_matrix, science,
                                     weight, scale, 0. if kernel == "Point" else width,
                                     1., code, width / 2.35482)
    np.divide(science, weight, out=science, where=weight > 0)
    return science, weight


def robust_stats(values):
    if not values.size:
        raise ValueError("A detector channel has no usable samples.")
    median = np.median(values)
    return float(median), float(1.4826 * np.median(np.abs(values - median)))


def median_model(cache, check, *, automatic):
    frames, channels, _, h, w = cache.shape
    model = np.empty((channels, 2, h, w), np.float32)
    rows = max(1, (32 << 20) // (frames * w * 8))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        for c in range(channels):
            for y in range(0, h, rows):
                check()
                science = cache[:, c, 0, y:y+rows]
                center, noise = model[c, :, y:y+rows]
                center[:] = np.nanmedian(science, axis=0)
                if automatic:
                    count = np.count_nonzero(np.isfinite(science), axis=0)
                    center[count < 2] = np.nan
                    noise[:] = 1.4826 * np.nanmedian(np.abs(science - center), axis=0)
                    # With just two exposures use the detector residual noise alone.
                    noise[count == 2] = 0.
                else:
                    noise[:] = np.nanmedian(cache[:, c, 1, y:y+rows], axis=0)
    return model


def blot(model, error, matrix, shape, scale):
    # Work in strips to keep double-precision coordinate buffers off the output grid.
    result = np.full((2, *shape), np.nan, np.float32)
    valid = np.isfinite(model) & np.isfinite(error)
    planes = [np.where(valid, p, 0.) for p in (model, error)]
    support = valid.astype(np.float32)
    x = np.arange(shape[1], dtype=float)
    for y0 in range(0, shape[0], 128):
        y = np.arange(y0, min(y0 + 128, shape[0]), dtype=float)[:, None]
        denom = matrix[2, 0]*x + matrix[2, 1]*y + matrix[2, 2]
        coords = [((row[0]*x + row[1]*y + row[2]) / denom + .5)*scale - .5
                  for row in (matrix[1], matrix[0])]
        def sample(p):
            return ndi.map_coordinates(p, coords, order=1, mode="grid-constant", cval=0.)
        weight = sample(support)
        for out, plane in zip(result, planes):
            np.divide(sample(plane), weight, out=out[y0:y0+len(y)], where=weight > 1e-6)
    return result


def detector_flags(data, pattern, models, matrices, scale, options, log=lambda message: None):
    primary, adjacent = [], []
    thresholds = options.blot_thresholds
    for c, ((model, error), matrix) in enumerate(zip(models, matrices)):
        native, valid = channel_data(data, pattern, c)
        predicted, sigma = blot(model, error, matrix, native.shape, scale)
        valid &= np.isfinite(predicted)
        n = np.count_nonzero(valid)
        if n == 0:
            primary.append(valid)
            adjacent.append(valid)
            continue
        if thresholds is None:
            offset, floor = robust_stats(native[valid] - predicted[valid])
            predicted += offset
            sigma = np.maximum(sigma, floor)
            # Allow at most half an expected upper-tail Gaussian crossing per channel.
            primary_snr = max(3., -ndtri(.5 / n))
            adjacent_snr = .8 * primary_snr
            log(f"Blot channel {c+1}: residual noise {floor:.6g}, "
                f"primary {primary_snr:.2f} sigma, adjacent {adjacent_snr:.2f} sigma")
        else:
            primary_snr, adjacent_snr = thresholds
            predicted += np.median(native[valid]) - np.median(predicted[valid])
        # Missing neighbors contribute zero gradient. Evaluate the same color model
        # at all native sites, including adjacent sites of a different CFA color.
        derivative = np.zeros_like(predicted)
        for axis in (0, 1):
            diff = np.nan_to_num(np.abs(np.diff(predicted, axis=axis)), nan=0.)
            grad = np.moveaxis(derivative, axis, 0)
            diff = np.moveaxis(diff, axis, 0)
            np.maximum(grad[1:], diff, out=grad[1:])
            np.maximum(grad[:-1], diff, out=grad[:-1])
        residual = native - predicted
        sigma = np.maximum(sigma, np.finfo(np.float32).eps * np.abs(native))
        primary.append(valid & (residual > 1.2 * derivative + primary_snr * sigma))
        adjacent.append(valid & (residual > .7 * derivative + adjacent_snr * sigma))
    if pattern is not None:
        neighbors = ndi.maximum_filter(np.logical_or.reduce(primary), size=3)
        return [a & neighbors for a in adjacent]
    return [a & ndi.maximum_filter(p, size=3) for p, a in zip(primary, adjacent)]


def stars(plane, valid):
    image = np.ascontiguousarray(np.where(valid, plane, 0.), dtype=np.float32)
    mask = np.ascontiguousarray(~valid)
    if not valid.any():
        return []
    bkg = sep.Background(image, mask=mask)
    return sep.extract(image - bkg.back(), max(bkg.globalrms * 5., 1e-12), mask=mask)


def refine(seed, cache, gains, offsets, psfs, noise, options, log, check, snapshot):
    result = seed.copy()
    if not options.iterations:
        return result
    for c, (gain, offset) in enumerate(zip(gains, offsets)):
        base = seed[c]
        pedestal = max(0., -float(base.min()))
        for frame in cache:
            obs, cov = frame[c]
            check()
            valid = (cov > 0) & np.isfinite(cov) & np.isfinite(obs)
            lowest = float(np.min(obs, where=valid, initial=np.inf))
            pedestal = max(pedestal, -(gain * lowest + offset))
        pedestal += 1e-7
        current = base + pedestal
        detail = base - ndi.gaussian_filter(base, 1.)
        floor = np.where((detail > 3 * np.sqrt(np.mean(detail**2))) & (base > 0),
                         current * .5, 0.) if options.protect_stars else 0.
        for iteration in range(1, options.iterations + 1):
            numerator = np.zeros_like(base)
            denominator = np.zeros_like(base)
            for j, frame in enumerate(cache):
                check()
                obs, cov = frame[c]
                valid = (cov > 0) & np.isfinite(cov) & np.isfinite(obs)
                obs = np.where(valid, obs, 0.)
                cov = np.where(valid, cov, 0.)

                def blur(plane):
                    return ndi.gaussian_filter(plane, psfs[j, c] / 2.35482, mode="constant")
                prediction = blur(current)
                residual = obs - (prediction - offset - pedestal) / gain
                residual2 = cov * residual**2
                variance = max(float(residual2.sum(dtype=float) / max(cov.sum(dtype=float), 1e-20)),
                               noise[j, c]**2, 1e-12)
                weight = cov.copy()
                if options.strength:
                    delta = options.huber_delta
                    threshold = -delta * np.sqrt(variance) if delta < 0 else delta
                    robust = np.minimum(1., threshold / np.maximum(abs(residual), 1e-20))
                    weight *= 1. - options.strength * (1. - robust)
                if options.noise_map:
                    local = blur(residual2) / np.maximum(blur(cov), 1e-20)
                    weight /= np.maximum(local, max(noise[j, c]**2, .1 * variance))
                elif noise[j, c] > 0:
                    weight /= noise[j, c]**2
                weight /= gain**2
                numerator += blur(weight * (gain * obs + offset + pedestal))
                denominator += blur(weight * prediction)
            ratio = np.divide(numerator, denominator, out=np.ones_like(base), where=denominator > 1e-20)
            updated = np.maximum(current * np.clip(ratio, 1/options.kappa, options.kappa), floor)
            change = np.abs(updated-current).sum(dtype=float) / max(current.sum(dtype=float), 1e-20)
            if not np.isfinite(change):
                raise ValueError("ImageMM produced a non-finite update.")
            current = updated
            snapshot(c, iteration, current - pedestal)
            log(f"ImageMM channel {c+1}, iteration {iteration}, relative change {change:.5g}")
            if iteration >= options.minimum and change <= options.tolerance:
                break
        result[c] = current - pedestal
    return result


def stack(paths, reference, output, options, calibration=None, log=print, check=lambda: None):
    """Write one master atomically. All native detector flags are frame-local."""
    paths = [Path(p).resolve() for p in paths]
    reference = Path(reference).resolve()
    output = Path(output).resolve()
    thresholds = options.blot_thresholds
    if reference not in paths or len(set(paths)) != len(paths) or len(paths) < 3:
        raise ValueError("Select at least three distinct detector lights, including the reference.")
    if output in paths or (output.exists() and any(output.samefile(p) for p in paths)):
        raise ValueError("The output must be different from every input light.")
    data, hdr, pattern = read_detector(reference)
    channels = 3 if pattern is not None else len(data)
    if calibration and (channels != 3 or not WCS(hdr, naxis=2).has_celestial):
        raise ValueError("SPCC needs RGB or CFA data and a plate-solved reference.")
    native_shape = data.shape
    ref_proxy = detection_proxy(data, pattern, 1 if channels == 3 else 0)
    frames = []
    for path in paths:
        check()
        data, frame_header, cfa = read_detector(path)
        compatible = (cfa is None) == (pattern is None)
        if compatible and cfa is not None:
            compatible = np.array_equal(np.bincount(cfa.ravel()), np.bincount(pattern.ravel()))
        if data.shape != native_shape or not compatible:
            raise ValueError(f"Detector dimensions or color pattern differ in {path.name}.")
        try:
            matrices = register(data, cfa, ref_proxy, path == reference, options.align_colors, channels)
        except (ValueError, astroalign.MaxIterError, cv2.error) as exc:
            if path == reference:
                raise ValueError(f"Reference registration failed: {exc}") from exc
            log(f"Skipping {path.name}: {exc}")
            continue
        frames.append((path, matrices, frame_header.get("FWHM")))
        log(f"Registered {path.name}")
    if len(frames) < 3:
        raise ValueError("Blot requires at least three successfully registered lights.")
    del data, ref_proxy
    scale, widths = sampling_geometry(options, hdr, pattern, len(frames))
    shape = tuple(max(1, round(n * scale)) for n in native_shape[-2:])
    header = scaled_header(hdr, scale)
    header["NCOMBINE"] = (len(frames), "Number of registered detector exposures")
    header["BLOTSTK"] = (True, "Detector-space blot stack")
    header.add_history(f"Drizzle {options.kernel}, scale {scale:.6g}, widths {widths[:channels]}")
    snr = "automatic" if thresholds is None else f"{thresholds[0]:g}/{thresholds[1]:g}"
    header.add_history(f"Blot SNR {snr}, {len(frames)} registered lights")
    log(f"Blot SNR {snr} (bright outliers only)")
    log(f"Drizzle grid {shape[1]} x {shape[0]}, scale {scale:.4g}, widths {widths[:channels]}")
    cache_shape = (len(frames), channels, 2, *shape)
    size = math.prod(cache_shape) * 4
    if shutil.disk_usage(output.parent).free < size + channels * math.prod(shape) * 4:
        raise ValueError(f"Insufficient space beside the output for the {size / 2**30:.2f} GiB drizzle cache.")
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".blot-") as scratch:
        temp = Path(scratch)
        with (temp / "observations").open("w+b") as stream:
            stream.truncate(size)
            # Reserve blocks before mapping. An out-of-space page fault can
            # otherwise terminate the application while filling a sparse cache.
            if hasattr(os, "posix_fallocate"):
                try:
                    os.posix_fallocate(stream.fileno(), 0, size)
                except OSError as exc:
                    if exc.errno not in (errno.EOPNOTSUPP, errno.ENOSYS):
                        raise
            with mmap.mmap(stream.fileno(), size) as mapping:
                cache = np.ndarray(cache_shape, np.float32, buffer=mapping)
                result = _stack_registered(frames, reference, scale, widths, cache,
                                           header, options, calibration, temp, log, check)
                del cache
        check()
        fits.writeto(temp / "master.fits", result[0] if channels == 1 else result,
                     header, output_verify="silentfix")
        if options.save_iterations:
            # A fresh directory for each run. Keep partial runs private.
            dest = Path(tempfile.mkdtemp(prefix="imagemm-", dir=output.parent))
            try:
                for plane in (temp / "iterations").iterdir():
                    plane.rename(dest / plane.name)
                os.replace(temp / "master.fits", output)
            except BaseException:
                shutil.rmtree(dest)
                raise
            log(f"Saved ImageMM iteration planes to {dest}")
        else:
            os.replace(temp / "master.fits", output)
    log(f"Saved {output}")


def _stack_registered(frames, reference, scale, widths, cache, header,
                      options, calibration, temp, log, check):
    count, channels, _, h, w = cache.shape
    sky = np.empty((count, channels))
    noise = np.empty_like(sky)
    psfs = np.empty_like(sky)
    for j, (path, matrices, _) in enumerate(frames):
        data, _, pattern = read_detector(path)
        for c, matrix in enumerate(matrices):
            check()
            plane, valid = channel_data(data, pattern, c)
            sky[j, c], noise[j, c] = robust_stats(plane[valid])
            sci, cov = deposit(plane, valid, matrix, (h, w), scale, widths[c], options.kernel)
            positive = cov[cov > 0]
            if not positive.size:
                raise ValueError(f"No overlap for {path.name}, channel {c+1}.")
            mean, _, _ = sigma_clipped_stats(positive, sigma=3)
            good = (cov > 0) & (cov >= .7 * mean)
            cache[j, c, 0] = np.where(good, sci - sky[j, c], np.nan)
            cache[j, c, 1] = np.where(good, noise[j, c], np.nan)
        log(f"First drizzle {j+1}/{count}: {path.name}")
    model = median_model(cache, check, automatic=options.blot_thresholds is None)
    seed = np.zeros((channels, h, w), np.float32)
    total = np.zeros_like(seed)
    ref_sky = sky[next(j for j, f in enumerate(frames) if f[0] == reference)]
    for j, (path, matrices, frame_fwhm) in enumerate(frames):
        check()
        data, _, pattern = read_detector(path)
        flags = detector_flags(data, pattern, model, matrices, scale, options,
                               log=lambda message: log(f"{path.name}: {message}"))
        for c, matrix in enumerate(matrices):
            check()
            plane, valid = channel_data(data, pattern, c)
            valid &= ~flags[c]
            sci, cov = deposit(plane, valid, matrix, (h, w), scale, widths[c], options.kernel)
            covered = cov > 0
            sci[covered] += ref_sky[c] - sky[j, c]
            cache[j, c, 0], cache[j, c, 1] = sci, cov
            seed[c] += sci * cov
            total[c] += cov
            if covered.any():
                _, noise[j, c] = robust_stats(sci[covered])
            detections = stars(sci, covered)
            measured = [2.35482 * np.sqrt(s["a"] * s["b"]) for s in detections if s["flag"] == 0]
            fallback = float(frame_fwhm or 0.) * scale
            psfs[j, c] = np.median(measured) if len(measured) >= 3 else fallback
            if not 0 < psfs[j, c] <= 100:
                psfs[j, c] = options.fallback_fwhm
        log(f"Clean drizzle {j+1}/{count}: {path.name}, {sum(np.count_nonzero(f) for f in flags)} detector samples rejected")
        del flags
    del model
    np.divide(seed, total, out=seed, where=total > 0)
    gains, offsets = np.ones(channels), np.zeros(channels)
    luminance = np.array([.2126, .7152, .0722])
    if calibration is not None:
        log("SPCC: measuring Gaia XP stars on the clean seed")
        gains, offsets, luminance, apertures = calibration.fit(seed, total > 0, header, check)
        header.add_history(f"SPCC {len(apertures)} Gaia XP stars, gains {gains}, offsets {offsets}")
        header.add_history(f"SPCC fitted luminance weights {luminance}")
        seed = (seed * gains[:, None, None] + offsets[:, None, None]).astype(np.float32)
        seed[total == 0] = 0.
        before = calibration.ratios(seed, apertures, total > 0)
    if options.save_iterations:
        (temp / "iterations").mkdir()

    def snapshot(c, i, plane):
        if options.save_iterations:
            fits.writeto(temp / "iterations" / f"ch{c+1}-iter{i:03}.fits", plane, header)
    if channels == 3 and options.luminance and calibration is None:
        # A luminance observation needs simultaneous coverage in all colors of
        # this exposure. Reuse its first cache plane now that RGB is consumed.
        luminance = luminance.astype(np.float32)
        for frame in cache:
            check()
            frame[0, 0] = np.tensordot(luminance, frame[:, 0], axes=1)
            frame[0, 1] = frame[:, 1].min(axis=0)
        lum_seed = np.tensordot(luminance, seed, axes=1)[None]
        snapshot(0, 0, lum_seed[0])
        lum_psf = np.sqrt(psfs**2 @ luminance)[:, None]
        lum_noise = np.sqrt(noise**2 @ luminance**2)[:, None]
        lum = refine(lum_seed, cache[:, :1], gains[:1], offsets[:1], lum_psf,
                     lum_noise, options, log, check, snapshot)
        result = seed + np.where(total.min(axis=0) > 0, lum-lum_seed, 0.)
    else:
        for c, plane in enumerate(seed):
            snapshot(c, 0, plane)
        result = refine(seed, cache, gains, offsets, psfs, noise, options, log, check, snapshot)
        if channels == 3 and options.luminance:
            delta = np.tensordot(luminance, result-seed, axes=1).astype(np.float32)
            result = seed + np.where(np.all(total > 0, axis=0), delta, 0.)
    result[total == 0] = 0.
    if calibration is not None:
        after = calibration.ratios(result, apertures, total > 0)
        valid = np.isfinite(before).all(axis=1) & np.isfinite(after).all(axis=1)
        if valid.sum() >= 3:
            drift = 100 * np.median(after[valid] / before[valid] - 1, axis=0)
            label = "Warning: " if (np.abs(drift) > 5).any() else ""
            message = f"{label}SPCC aperture drift R/G {drift[0]:+.2f}%, B/G {drift[1]:+.2f}% ({valid.sum()} stars)"
        else:
            message = "SPCC aperture drift unavailable: fewer than three common stars"
        log(message)
        header.add_history(message)
    header.add_history(f"ImageMM {'luminance' if options.luminance else 'per-channel'}, star floor {options.protect_stars}, local noise {options.noise_map}")
    header.add_history(f"ImageMM max iterations {options.iterations}, kappa {options.kappa}, tolerance {options.tolerance}")
    return result
