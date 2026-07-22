"""
Emerald's Experimental spatially aware mask
Adaptive masks for galaxies, nebulae, and other diffuse structures.
"""

from __future__ import annotations

import cv2
import numpy as np


_AUTO_SENSITIVITIES = (0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0)


def _odd(value: float, minimum: int = 3) -> int:
    """Return an odd integer suitable for an OpenCV kernel."""
    size = max(minimum, int(round(value)))
    return size if size % 2 else size + 1


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    """Return the median and a robust standard-deviation estimate."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, max(1.4826 * mad, 1e-7)


def _tile_robust_scale(
    values: np.ndarray,
    *,
    tile_fraction: float = 0.06,
    percentile: float = 55.0,
) -> float:
    """Estimate noise from mostly flat tiles instead of source-filled pixels.

    A frame-wide MAD treats real nebulosity as noise when an object occupies a
    substantial part of the image. Per-tile MADs measure the small-scale sky
    variation instead, while a middle percentile is conservative in gradients
    and avoids selecting a single unusually smooth tile.
    """
    height, width = values.shape
    tile_size = max(24, int(round(min(height, width) * tile_fraction)))
    scales: list[float] = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = values[y : y + tile_size, x : x + tile_size]
            if tile.size < 64:
                continue
            _, scale = _robust_location_scale(tile)
            if np.isfinite(scale) and scale > 1e-7:
                scales.append(scale)

    if not scales:
        return _robust_location_scale(values)[1]
    return max(float(np.percentile(scales, percentile)), 1e-7)


def _multiscale_structure_evidence(
    brightness: np.ndarray,
    *,
    keep_polarities: bool = True,
    working_limit: int | None = 2000,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    tuple[float, ...],
]:
    """Return noise-normalized bright, dark, and two-sided structure evidence.

    Difference-of-Gaussian bands act like a compact starlet decomposition:
    coherent wisps, dust boundaries, shells, and galaxy arms remain significant
    at one or more adjacent spatial scales while pixel noise does not.  The
    coarsest frame-scale background is deliberately absent from this measure,
    so smooth gradients cannot masquerade as structure.
    """
    original_height, original_width = brightness.shape
    work_scale = min(
        1.0,
        float(working_limit) / max(original_height, original_width)
        if working_limit
        else 1.0,
    )
    if work_scale < 1.0:
        source = cv2.resize(
            np.asarray(brightness, dtype=np.float32),
            (
                max(16, int(round(original_width * work_scale))),
                max(16, int(round(original_height * work_scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        source = np.asarray(brightness, dtype=np.float32)
    height, width = source.shape
    short_side = min(height, width)

    # Star suppression is intentionally not applied to these native-scale
    # bands.  Large grayscale openings are both expensive on Ultra images and
    # indistinguishable from narrow nebular filaments.  The opt-in mode instead
    # prevents stellar seeds on the bounded coarse pass and punches compact
    # peaks from the finished mask.
    scales = (
        max(0.75, short_side * 0.0012),
        max(1.50, short_side * 0.0030),
        max(3.00, short_side * 0.0060),
        max(6.00, short_side * 0.0120),
        max(12.0, short_side * 0.0240),
        max(24.0, short_side * 0.0480),
    )
    previous = cv2.GaussianBlur(source, (0, 0), sigmaX=scales[0])
    positive = np.full(source.shape, -np.inf, dtype=np.float32)
    negative = np.full(source.shape, -np.inf, dtype=np.float32)
    fine_activity: np.ndarray | None = None

    for index, sigma in enumerate(scales[1:]):
        current = cv2.GaussianBlur(source, (0, 0), sigmaX=sigma)
        band = cv2.subtract(previous, current)
        previous = current
        centre = float(np.median(band))
        scale = _tile_robust_scale(band)
        band -= centre
        band /= scale
        if index == 0:
            # The finest band is useful later for a native outline, but by
            # itself mostly measures stars and pixel noise.  Requiring a
            # slightly broader scale keeps the structure support coherent.
            fine_activity = np.abs(band)
            continue
        np.maximum(positive, band, out=positive)
        np.negative(band, out=band)
        np.maximum(negative, band, out=negative)

    # A clean sky notch or a single-scale noise halo can have a strong coarse
    # coefficient but no fine evidence. Real dust lanes, wisps, and nebular
    # boundaries persist at both scales. Smooth positive emission is handled by
    # the separately seeded faint-evidence branch below.
    assert fine_activity is not None
    if keep_polarities:
        textured_positive = positive.copy()
        textured_positive[fine_activity < 0.75] = -1.0e6
        textured_negative = negative.copy()
        textured_negative[fine_activity < 0.75] = -1.0e6
        activity = np.maximum(textured_positive, textured_negative)
        if activity.shape != brightness.shape:
            output_size = (original_width, original_height)
            positive = cv2.resize(
                positive, output_size, interpolation=cv2.INTER_LINEAR
            )
            negative = cv2.resize(
                negative, output_size, interpolation=cv2.INTER_LINEAR
            )
            activity = cv2.resize(
                activity, output_size, interpolation=cv2.INTER_LINEAR
            )
        return positive, negative, activity, positive, scales

    # Production consumes both ungated positive evidence for source validation
    # and fine-gated context evidence. Reuse the negative-polarity buffer for
    # context so this does not retain another native-resolution float image.
    negative[fine_activity < 0.75] = -1.0e6
    np.maximum(
        negative,
        positive,
        out=negative,
        where=fine_activity >= 0.75,
    )
    if negative.shape != brightness.shape:
        negative = cv2.resize(
            negative,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        )
    return None, None, negative, positive, scales


def _grow_multiscale_context(
    body: np.ndarray,
    activity: np.ndarray,
    source_activity: np.ndarray,
    detail_contrast: np.ndarray,
    *,
    sensitivity: float,
    min_area: int,
    source_min_area: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Extend a diffuse body along nearby coherent bright or dark structure."""
    if not np.any(body):
        return body, np.zeros_like(body), 0.0, 0.0, 0.0, 0.0

    # Unlike the old hard support floor, this remains responsive throughout the
    # full UI range.  Area qualification and proximity prevent the low maximum-
    # sensitivity threshold from percolating through an unrelated noisy sky.
    support_threshold = max(1.50, 4.25 / (sensitivity**0.33))
    seed_threshold = support_threshold + 1.35
    context_min_area = max(8, min_area)

    # The broad positive detector is intentionally permissive, but correlated
    # sky can also make a large coarse island. Require every source component
    # to contain a sizeable, much stronger multiscale core before allowing any
    # faint growth around it.
    if source_activity.shape != body.shape:
        source_height, source_width = source_activity.shape
        source_body = cv2.resize(
            body,
            (source_width, source_height),
            interpolation=cv2.INTER_NEAREST,
        )
        scaled_source_area = max(
            8,
            int(round(source_min_area * source_activity.size / body.size)),
        )
        source_seeds = _keep_large_components(
            source_activity >= seed_threshold + 1.50,
            scaled_source_area,
        )
        source_body = _components_touching_seeds(
            source_body,
            source_seeds,
            scaled_source_area,
        )
        source_body = cv2.resize(
            source_body,
            (body.shape[1], body.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        body = ((body > 0) & (source_body > 0)).astype(np.uint8)
        del source_body
    else:
        source_seeds = _keep_large_components(
            source_activity >= seed_threshold + 1.50,
            source_min_area,
        )
        body = _components_touching_seeds(
            body,
            source_seeds,
            source_min_area,
        )
    del source_seeds
    if not np.any(body):
        return (
            body,
            np.zeros_like(body),
            seed_threshold,
            support_threshold,
            0.0,
            0.0,
        )

    short_side = min(body.shape)
    context_seeds = _keep_large_components(
        activity >= seed_threshold,
        context_min_area,
    )
    coherent = _components_touching_seeds(
        activity >= support_threshold,
        context_seeds,
        context_min_area,
    )
    anchor_size = _odd(short_side * 0.012, 3)
    anchors = cv2.dilate(
        body,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (anchor_size, anchor_size)
        ),
    )
    anchored = _components_touching_seeds(
        coherent,
        anchors,
        min_area=1,
    )
    del context_seeds, coherent

    # A smooth, extremely low-surface-brightness wisp may have no strong DoG
    # coefficient. Average at a small structure scale, normalize that response
    # by the tile noise, then use seeded hysteresis. The earlier raw 0.25-sigma
    # growth percolated through ordinary sky at maximum sensitivity.
    faint_evidence = cv2.GaussianBlur(
        np.asarray(detail_contrast, dtype=np.float32),
        (0, 0),
        sigmaX=max(1.0, short_side * 0.004),
    )
    faint_centre = float(np.median(faint_evidence))
    faint_scale = _tile_robust_scale(faint_evidence)
    faint_evidence -= faint_centre
    faint_evidence /= faint_scale
    faint_support_threshold = max(1.25, 3.4 / np.sqrt(sensitivity))
    faint_seed_threshold = faint_support_threshold + 1.50
    faint_seeds = _keep_large_components(
        faint_evidence >= faint_seed_threshold,
        context_min_area,
    )
    faint_coherent = _components_touching_seeds(
        faint_evidence >= faint_support_threshold,
        faint_seeds,
        context_min_area,
    )
    faint_anchored = _components_touching_seeds(
        faint_coherent,
        anchors,
        min_area=1,
    )
    np.maximum(anchored, faint_anchored, out=anchored)
    del faint_evidence, faint_seeds, faint_coherent, faint_anchored, anchors

    # Sensitivity controls both statistical significance and how far a coherent
    # branch may be associated with the object.  This admits the faint/dark
    # outskirts without allowing a long chain of field noise to cross the frame.
    reach_fraction = float(np.clip(0.018 + 0.0065 * sensitivity, 0.022, 0.15))
    distance = cv2.distanceTransform(
        1 - (body > 0).astype(np.uint8), cv2.DIST_L2, 5
    )
    context = (
        (anchored > 0)
        & (body == 0)
        & (distance <= short_side * reach_fraction)
    ).astype(np.uint8)
    return (
        np.maximum(body, context),
        context,
        seed_threshold,
        support_threshold,
        faint_support_threshold,
        reach_fraction,
    )


def _compact_peak_mask(brightness: np.ndarray, min_area: int) -> np.ndarray:
    """Locate small high-frequency peaks for optional star suppression."""
    short_side = min(brightness.shape)
    smooth = cv2.GaussianBlur(
        np.asarray(brightness, dtype=np.float32),
        (0, 0),
        sigmaX=max(1.25, short_side * 0.006),
    )
    top_hat = np.asarray(brightness, dtype=np.float32) - smooth
    centre = float(np.median(top_hat))
    threshold = centre + 5.0 * _tile_robust_scale(top_hat)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (top_hat >= threshold).astype(np.uint8), connectivity=8
    )
    kept_labels = np.zeros(count, dtype=bool)
    maximum_area = max(
        12, min(2 * min_area, int(round(brightness.size * 0.002)))
    )
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area > maximum_area or min(width, height) <= 0:
            continue
        aspect = width / height
        if 0.40 <= aspect <= 2.50:
            kept_labels[label] = True

    compact = kept_labels[labels].astype(np.uint8)

    if np.any(compact):
        dilation_size = _odd(short_side * 0.006, 3)
        compact = cv2.dilate(
            compact,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilation_size, dilation_size)
            ),
        )
    return compact


def _keep_large_components(
    candidate: np.ndarray,
    min_area: int,
    border_margin: int = 0,
    max_border_contact_fraction: float = 0.18,
) -> np.ndarray:
    """Keep large candidate regions with limited frame-edge contact."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    kept = np.zeros(candidate.shape, dtype=np.uint8)
    height, width = candidate.shape
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < min_area:
            continue
        if border_margin > 0:
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            if (
                x <= border_margin
                or y <= border_margin
                or x + w >= width - border_margin
                or y + h >= height - border_margin
            ):
                # A real nebula may leave the frame through a small part of an
                # edge. Illumination gradients tend to run along much more of
                # the frame, so reject those without rejecting every edge object.
                region = labels == label
                top_contact = np.count_nonzero(region[0, :])
                bottom_contact = np.count_nonzero(region[-1, :])
                left_contact = np.count_nonzero(region[:, 0])
                right_contact = np.count_nonzero(region[:, -1])
                border_contact = (
                    top_contact
                    + bottom_contact
                    + left_contact
                    + right_contact
                )
                frame_perimeter = max(1, 2 * (height + width))
                component_area = int(stats[label, cv2.CC_STAT_AREA])
                touches_horizontal_edge = (
                    y <= border_margin or y + h >= height - border_margin
                )
                touches_vertical_edge = (
                    x <= border_margin or x + w >= width - border_margin
                )
                shallow_edge_band = (
                    (touches_horizontal_edge and h < 0.18 * height)
                    or (touches_vertical_edge and w < 0.18 * width)
                )
                # Fixed-pattern or illumination bands usually fill nearly all
                # of the edge spanned by their bounding box. Requiring that
                # contact distinguishes them from a broad, irregular nebula
                # that merely exits through part of the same edge.
                edge_span_contact = max(
                    top_contact / max(1, w),
                    bottom_contact / max(1, w),
                    left_contact / max(1, h),
                    right_contact / max(1, h),
                )
                elongated_edge_band = edge_span_contact > 0.82 and (
                    (
                        touches_horizontal_edge
                        and w > 0.45 * width
                        and h < 0.30 * height
                    )
                    or (
                        touches_vertical_edge
                        and h > 0.45 * height
                        and w < 0.30 * width
                    )
                )
                # Catch both a broad strip along the frame and a thin fragment
                # whose area is dominated by contact with one edge.
                if (
                    shallow_edge_band
                    or elongated_edge_band
                    or border_contact / frame_perimeter
                    > max_border_contact_fraction
                    or border_contact / component_area > 0.08
                ):
                    continue
        region = labels == label
        kept[region] = 1
    return kept


def _fill_structure_envelope(
    body: np.ndarray,
    fill_fraction: float,
    working_limit: int = 512,
) -> np.ndarray:
    """Close broad concavities without sacrificing the native mask boundary."""
    if fill_fraction <= 0 or not np.any(body):
        return body

    height, width = body.shape
    scale = min(1.0, float(working_limit) / max(height, width))
    if scale < 1.0:
        work = cv2.resize(
            body,
            (max(16, round(width * scale)), max(16, round(height * scale))),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        work = body

    closing_size = _odd(min(work.shape) * fill_fraction, 3)
    envelope = cv2.morphologyEx(
        work,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_size, closing_size)
        ),
    )
    if envelope.shape != body.shape:
        envelope = cv2.resize(
            envelope, (width, height), interpolation=cv2.INTER_NEAREST
        )
    return np.maximum(body, envelope.astype(body.dtype, copy=False))


def _components_touching_seeds(
    support: np.ndarray,
    seeds: np.ndarray,
    min_area: int,
    border_margin: int = 0,
) -> np.ndarray:
    """Keep supported regions that contain a confident diffuse-emission seed."""
    count, labels, _, _ = cv2.connectedComponentsWithStats(
        support.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros(support.shape, dtype=np.uint8)

    seeded_labels = np.unique(labels[seeds.astype(bool)])
    seeded_labels = seeded_labels[seeded_labels != 0]
    if seeded_labels.size == 0:
        return np.zeros(support.shape, dtype=np.uint8)

    selected = np.isin(labels, seeded_labels).astype(np.uint8)
    return _keep_large_components(
        selected,
        min_area,
        border_margin=border_margin,
    )


def _grow_structure_context(
    body: np.ndarray,
    detail_contrast: np.ndarray,
    support_threshold: float,
    fill_fraction: float,
) -> np.ndarray:
    """Grow into nearby faint/dark structure without leaking across the sky."""
    if fill_fraction <= 0 or not np.any(body):
        return body

    body_binary = (body > 0).astype(np.uint8)
    short_side = min(body.shape)
    radius = max(1.0, short_side * fill_fraction)
    distance = cv2.distanceTransform(1 - body_binary, cv2.DIST_L2, 5)

    # Envelope fill is also a contextual control: larger values progressively
    # admit locally dark dust and sub-background wisps near a detected object.
    # The distance gate is essential; a negative global threshold would join
    # ordinary noisy sky to the object at high sensitivity.
    context_threshold = max(-0.75, support_threshold - 18.0 * fill_fraction)
    context = (distance <= radius) & (detail_contrast >= context_threshold)
    growth_support = body_binary.astype(bool) | context
    return _components_touching_seeds(
        growth_support,
        body_binary,
        min_area=1,
    )


def _weight_structure_mask(
    body: np.ndarray,
    detail_contrast: np.ndarray,
    profile: str,
    min_area: int = 1,
) -> np.ndarray:
    """Convert a detected body to a uniform or brightness-weighted mask."""
    body_float = (body > 0).astype(np.float32)
    if profile == "extended" or not np.any(body_float):
        return body_float

    # Suppress compact peaks twice: a median removes hot/stellar cores, then a
    # rolling grayscale opening estimates the brightness without structures
    # smaller than roughly two percent of the frame. Fine response is capped
    # against that envelope rather than replaced by it, so broad spiral-arm
    # and clump modulation remains visible.
    response = cv2.medianBlur(np.asarray(detail_contrast, dtype=np.float32), 5)
    compact_size = _odd(min(body.shape) * 0.022, 5)
    opened = cv2.morphologyEx(
        response,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (compact_size, compact_size)
        ),
    )
    compact_residual = np.maximum(response - opened, 0.0)
    residual_values = compact_residual[body_float > 0]
    allowance = max(0.15, float(np.percentile(residual_values, 70.0)))
    response = np.minimum(response, opened + allowance)

    # Percentiles make the weighting adaptive to both a bright galaxy core and
    # low-contrast linear data.
    values = response[body_float > 0]
    low, high = np.percentile(values, (10.0, 92.0))
    strength = np.clip(
        (response - float(low)) / max(float(high - low), 1e-7),
        0.0,
        1.0,
    )
    strength = strength * strength * (3.0 - 2.0 * strength)

    # In the weighted profile, Minimum area also qualifies distinct bright
    # islands before their lower-weight surroundings are restored. This makes
    # detached stars disappear progressively instead of remaining until the
    # entire connected support network is rejected.
    bright_islands = strength >= 0.15
    qualified_bright = _keep_large_components(
        bright_islands,
        min_area,
    )
    retained_body = _components_touching_seeds(
        body_float > 0,
        qualified_bright,
        min_area=1,
    ).astype(np.float32)
    # Retain a small amount of the detected envelope so the mask does not
    # acquire brittle holes, while still allowing inter-arm regions to darken.
    return retained_body * (0.05 + 0.95 * strength)


def _detection_stages(
    brightness: np.ndarray,
    *,
    sensitivity: float,
    min_area_fraction: float,
    background_fraction: float = 0.35,
    reject_stars: bool = False,
    include_diagnostics: bool = True,
) -> dict[str, np.ndarray | float | int]:
    """Detect diffuse emission, then recover nearby two-sided structure."""
    height, width = brightness.shape
    short_side = min(height, width)

    # Seeds and the very broad background contain no native-pixel detail.
    # Bounding just this branch makes Ultra practical while its final support
    # map and outline still run at the requested/native resolution.
    seed_scale = min(1.0, 700.0 / max(height, width))
    if seed_scale < 1.0:
        seed_width = max(16, int(round(width * seed_scale)))
        seed_height = max(16, int(round(height * seed_scale)))
        seed_source = cv2.resize(
            brightness,
            (seed_width, seed_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        seed_source = brightness
    seed_short_side = min(seed_source.shape)

    rejection_size = _odd(seed_short_side * 0.018, 5)
    if reject_stars:
        star_rejected = cv2.morphologyEx(
            seed_source,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (rejection_size, rejection_size)
            ),
        )
    else:
        star_rejected = seed_source
    coarse = cv2.GaussianBlur(
        star_rejected, (0, 0), sigmaX=max(2.0, seed_short_side * 0.009)
    )
    local_background = cv2.GaussianBlur(
        coarse,
        (0, 0),
        sigmaX=max(12.0, seed_short_side * background_fraction),
    )
    coarse_prominence = coarse - local_background
    coarse_sky, coarse_sigma = _robust_location_scale(coarse_prominence)
    coarse_contrast = (coarse_prominence - coarse_sky) / coarse_sigma

    if seed_scale < 1.0:
        def restore_seed_stage(stage: np.ndarray) -> np.ndarray:
            return cv2.resize(
                stage, (width, height), interpolation=cv2.INTER_LINEAR
            )

        local_background = restore_seed_stage(local_background)
        coarse_contrast = restore_seed_stage(coarse_contrast)
        if include_diagnostics:
            star_rejected = restore_seed_stage(star_rejected)
            coarse = restore_seed_stage(coarse)
            coarse_prominence = restore_seed_stage(coarse_prominence)

    # A much finer denoised image supplies the actual outline. Compact stars
    # remain here, but they cannot enter the result unless connected to a
    # large seeded structure, and isolated stellar components fail min-area.
    detail = cv2.GaussianBlur(
        brightness, (0, 0), sigmaX=max(0.75, short_side * 0.0012)
    )
    detail_prominence = detail - local_background
    detail_sky, detail_sigma = _robust_location_scale(detail_prominence)
    detail_contrast = (detail_prominence - detail_sky) / detail_sigma

    # Sensitivity changes both levels smoothly. Square-root scaling prevents
    # the former sensitivity=20 threshold (0.25 sigma) from turning much of a
    # noisy sky into one percolating component.
    sensitivity_scale = np.sqrt(sensitivity)
    seed_threshold = max(1.15, 4.6 / sensitivity_scale)
    support_threshold = max(0.45, 3.15 / sensitivity_scale)
    seeds = coarse_contrast >= seed_threshold
    support = detail_contrast >= support_threshold

    min_area = max(1, int(height * width * min_area_fraction))
    # Lower pixel thresholds need stronger spatial evidence. This is a
    # statistical qualification rather than a morphology operation: maximum
    # sensitivity no longer promotes small patches of correlated background,
    # while a user can still lower Minimum area for a genuinely compact target.
    qualification_area = max(
        min_area,
        int(round(min_area * np.sqrt(max(1.0, sensitivity)))),
    )
    border_margin = max(2, int(round(short_side * 0.025)))
    # Qualify high-confidence islands before faint support can connect them.
    # Applying minimum area only after hysteresis made the control behave like
    # an all-or-nothing switch once a star field touched the main object.
    qualified_seeds = _keep_large_components(
        seeds,
        max(8, qualification_area),
        border_margin=border_margin,
    )
    base_body = _components_touching_seeds(
        support,
        qualified_seeds,
        qualification_area,
        border_margin=border_margin,
    )
    if not include_diagnostics and not np.any(base_body):
        return {
            "detail_contrast": detail_contrast,
            "body": base_body,
            "support_threshold": support_threshold,
            "minimum_area_pixels": min_area,
            "qualification_area_pixels": qualification_area,
        }
    if not include_diagnostics:
        del (
            seed_source,
            star_rejected,
            coarse,
            local_background,
            coarse_prominence,
            coarse_contrast,
            detail,
            detail_prominence,
            seeds,
            qualified_seeds,
            support,
        )
    (
        structure_positive,
        structure_negative,
        structure_activity,
        structure_source_activity,
        structure_scales,
    ) = _multiscale_structure_evidence(
        brightness,
        keep_polarities=include_diagnostics,
    )
    (
        body,
        structure_context,
        structure_seed_threshold,
        structure_support_threshold,
        faint_support_threshold,
        structure_reach_fraction,
    ) = _grow_multiscale_context(
        base_body,
        structure_activity,
        structure_source_activity,
        detail_contrast,
        sensitivity=sensitivity,
        min_area=min_area,
        source_min_area=qualification_area,
    )
    result: dict[str, np.ndarray | float | int] = {
        "detail_contrast": detail_contrast,
        "body": body,
        "support_threshold": support_threshold,
        "minimum_area_pixels": min_area,
        "qualification_area_pixels": qualification_area,
    }
    if not include_diagnostics:
        return result

    assert structure_positive is not None
    assert structure_negative is not None
    result.update(
        {
            "star_rejected": star_rejected,
            "coarse": coarse,
            "detail": detail,
            "local_background": local_background,
            "coarse_prominence": coarse_prominence,
            "detail_prominence": detail_prominence,
            "coarse_contrast": coarse_contrast,
            "seeds": seeds,
            "qualified_seeds": qualified_seeds,
            "support": support,
            "base_body": base_body,
            "structure_positive": structure_positive,
            "structure_negative": structure_negative,
            "structure_activity": structure_activity,
            "structure_context": structure_context,
            "seed_threshold": seed_threshold,
            "rejection_size": rejection_size,
            "reject_stars": int(reject_stars),
            "structure_scales": np.asarray(
                structure_scales, dtype=np.float32
            ),
            "structure_seed_threshold": structure_seed_threshold,
            "structure_support_threshold": structure_support_threshold,
            "faint_support_threshold": faint_support_threshold,
            "structure_reach_fraction": structure_reach_fraction,
        }
    )
    return result


def diffuse_structure_mask(
    image: np.ndarray,
    *,
    sensitivity: float = 1.0,
    feather_fraction: float = 0.0075,
    min_area_fraction: float = 0.001,
    envelope_fraction: float = 0.0,
    profile: str = "extended",
    reject_stars: bool = False,
    _processing_limit: int | None = 1400,
) -> np.ndarray:
    """Build a soft mask for interesting extended bright and dark structure.

    Parameters are resolution-independent so the same settings work for the
    downsampled live preview and the full-resolution document.

    Args:
        image: Mono or RGB floating-point image, normally scaled to ``[0, 1]``.
        sensitivity: Higher values retain fainter connected structure.
        feather_fraction: Feather sigma as a fraction of the shorter image side.
        min_area_fraction: Smallest accepted component as a fraction of the frame.
        envelope_fraction: Reach for nearby faint/dark structure and inward
            gaps, as a fraction of the shorter image side. Zero disables it.
        profile: ``"extended"`` for a uniform protection mask, or
            ``"brightness_weighted"`` to emphasize brighter internal detail.
        reject_stars: Suppress PSF-scale peaks before structure detection. This
            is optional because starless input usually gives a cleaner result.

    Returns:
        A ``float32`` two-dimensional mask in ``[0, 1]``.
    """
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive")
    if profile not in {"extended", "brightness_weighted"}:
        raise ValueError("profile must be 'extended' or 'brightness_weighted'")
    if feather_fraction < 0 or min_area_fraction < 0 or envelope_fraction < 0:
        raise ValueError(
            "feather_fraction, min_area_fraction, and envelope_fraction "
            "cannot be negative"
        )

    source = np.asarray(image, dtype=np.float32)
    if source.ndim not in (2, 3):
        raise ValueError("image must be a mono or RGB array")
    if source.ndim == 3 and source.shape[2] < 3:
        raise ValueError("color images must have at least three channels")

    # Low and Medium intentionally bound the working image for interactive
    # speed. Ultra passes ``None`` and preserves native-scale boundary evidence.
    original_height, original_width = source.shape[:2]
    longest_side = max(original_height, original_width)
    if _processing_limit and longest_side > _processing_limit:
        scale = float(_processing_limit) / float(longest_side)
        work_width = max(16, int(round(original_width * scale)))
        work_height = max(16, int(round(original_height * scale)))
        reduced = cv2.resize(
            source, (work_width, work_height), interpolation=cv2.INTER_AREA
        )
        reduced_mask = diffuse_structure_mask(
            reduced,
            sensitivity=sensitivity,
            feather_fraction=feather_fraction,
            min_area_fraction=min_area_fraction,
            envelope_fraction=envelope_fraction,
            profile=profile,
            reject_stars=reject_stars,
            _processing_limit=None,
        )
        restored = cv2.resize(
            reduced_mask,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        )
        peak = float(restored.max())
        if peak > 0:
            restored /= peak
        return np.clip(restored, 0.0, 1.0).astype(np.float32, copy=False)

    finite = np.isfinite(source)
    if not np.all(finite):
        fill = float(np.nanmedian(source)) if np.any(finite) else 0.0
        source = np.nan_to_num(source, nan=fill, posinf=fill, neginf=fill)

    # Normalization only compensates for data outside SASpro's usual [0, 1]
    # document range. It does not stretch normally scaled linear image data.
    low = float(np.percentile(source, 0.1))
    high = float(np.percentile(source, 99.9))
    if low < 0.0 or high > 1.0:
        source = np.clip((source - low) / max(high - low, 1e-7), 0.0, 1.0)
    else:
        source = np.clip(source, 0.0, 1.0)

    if source.ndim == 3:
        rgb = source[..., :3]
        luminance = (
            0.2126 * rgb[..., 0]
            + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]
        )
        # A small max-channel contribution retains coloured emission nebulae.
        brightness = 0.72 * luminance + 0.28 * np.max(rgb, axis=2)
        del rgb, luminance
    else:
        brightness = source
    del source

    height, width = brightness.shape
    short_side = min(height, width)
    if short_side < 16:
        return np.zeros((height, width), dtype=np.float32)

    stages = _detection_stages(
        brightness,
        sensitivity=sensitivity,
        min_area_fraction=min_area_fraction,
        reject_stars=reject_stars,
        include_diagnostics=False,
    )
    body = np.asarray(stages["body"], dtype=np.uint8)
    body = _grow_structure_context(
        body,
        np.asarray(stages["detail_contrast"]),
        float(stages["support_threshold"]),
        envelope_fraction,
    )
    body = _fill_structure_envelope(body, envelope_fraction)

    weighted_body = _weight_structure_mask(
        body,
        np.asarray(stages["detail_contrast"]),
        profile,
        int(stages["minimum_area_pixels"]),
    )
    feather_sigma = max(0.0, short_side * feather_fraction)
    if feather_sigma > 0:
        mask = cv2.GaussianBlur(
            weighted_body, (0, 0), sigmaX=max(0.5, feather_sigma)
        )
    else:
        mask = weighted_body

    # Punch opt-in stellar holes after the broad object feather. Otherwise a
    # feather wider than a PSF fills the hole back in, especially on large
    # Ultra images.
    if reject_stars:
        mask = mask.copy()
        mask[
            _compact_peak_mask(
                brightness, int(stages["minimum_area_pixels"])
            ).astype(bool)
        ] = 0.0

    peak = float(mask.max())
    if peak > 0:
        mask /= peak
    return np.clip(mask, 0.0, 1.0).astype(np.float32, copy=False)


def suggest_diffuse_structure_settings(
    image: np.ndarray,
) -> dict[str, float | str | bool]:
    """Estimate editable diffuse-mask settings from structure stability.

    The estimator chooses the lowest sensitivity that recovers most of the
    safe high-sensitivity coverage. This avoids selecting maximum sensitivity
    merely because it can absorb a little more noise. Geometry then determines
    whether a compact object benefits from brightness weighting; irregular
    objects keep the native multiscale outline instead of an automatic fill.
    """
    source = np.asarray(image)
    if source.ndim not in (2, 3):
        raise ValueError("image must be a mono or RGB array")
    if source.ndim == 3 and source.shape[2] < 3:
        raise ValueError("color images must have at least three channels")

    height, width = source.shape[:2]
    scale = min(1.0, 700.0 / max(height, width))
    if scale < 1.0:
        source = cv2.resize(
            source,
            (max(16, round(width * scale)), max(16, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    trials: list[tuple[float, np.ndarray, float]] = []
    for sensitivity in _AUTO_SENSITIVITIES:
        mask = diffuse_structure_mask(
            source,
            sensitivity=sensitivity,
            feather_fraction=0.0,
            min_area_fraction=0.001,
            envelope_fraction=0.0,
            profile="extended",
            _processing_limit=None,
        )
        coverage = float(np.mean(mask >= 0.5))
        trials.append((sensitivity, mask, coverage))

    # Ignore percolated trials. If all non-empty trials exceed the safe limit,
    # retain the least aggressive result rather than returning an empty mask.
    safe_trials = [trial for trial in trials if 0.0 < trial[2] <= 0.52]
    if not safe_trials:
        nonempty = [trial for trial in trials if trial[2] > 0.0]
        if not nonempty:
            return {
                "sensitivity": 1.0,
                "feather_fraction": 0.01,
                "min_area_fraction": 0.001,
                "envelope_fraction": 0.0,
                "profile": "extended",
                "reject_stars": False,
            }
        safe_trials = [nonempty[0]]

    # A compact source can keep acquiring unrelated nearby texture as the
    # context radius grows, so its maximum non-percolated coverage is not a
    # useful plateau.  Recognize compact geometry in the conservative trial and
    # bound Auto to the stable part of that object's growth curve.
    probe_binary = (safe_trials[0][1] >= 0.5).astype(np.uint8)
    probe_count, probe_labels, probe_stats, _ = cv2.connectedComponentsWithStats(
        probe_binary, 8
    )
    probe_bbox_fraction = 1.0
    probe_solidity = 0.0
    probe_extent = 0.0
    probe_dominance = 0.0
    if probe_count > 1:
        probe_label = 1 + int(
            np.argmax(probe_stats[1:, cv2.CC_STAT_AREA])
        )
        probe_area = int(probe_stats[probe_label, cv2.CC_STAT_AREA])
        probe_bbox_area = int(
            probe_stats[probe_label, cv2.CC_STAT_WIDTH]
            * probe_stats[probe_label, cv2.CC_STAT_HEIGHT]
        )
        probe_bbox_fraction = float(
            probe_bbox_area / probe_binary.size
        )
        probe_extent = probe_area / max(1, probe_bbox_area)
        probe_dominance = probe_area / max(1, np.count_nonzero(probe_binary))
        probe_region = (probe_labels == probe_label).astype(np.uint8)
        contours, _ = cv2.findContours(
            probe_region,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        hull_area = max(
            (cv2.contourArea(cv2.convexHull(contour)) for contour in contours),
            default=0.0,
        )
        probe_solidity = probe_area / max(1.0, hull_area)
    probe_compact = (
        safe_trials[0][2] < 0.13
        and probe_bbox_fraction < 0.22
        and probe_extent > 0.48
        and probe_solidity > 0.82
        and probe_dominance > 0.90
    )
    if probe_compact:
        compact_limit = min(0.24, max(0.12, safe_trials[0][2] * 1.60))
        compact_trials = [
            trial for trial in safe_trials if trial[2] <= compact_limit
        ]
        if compact_trials:
            safe_trials = compact_trials

    plateau_coverage = safe_trials[-1][2]
    target_coverage = 0.82 * plateau_coverage
    selected = next(
        (trial for trial in safe_trials if trial[2] >= target_coverage),
        safe_trials[-1],
    )
    sensitivity, _, coverage = selected

    compact = probe_compact
    profile = "brightness_weighted" if compact else "extended"
    # Irregularity is now handled by the native multiscale context pass.
    # Automatically closing low-solidity objects rounded away the very dust
    # lanes, notches, and wisps that Ultra is intended to retain.
    envelope_fraction = 0.0
    min_area_fraction = float(np.clip(coverage * 0.015, 0.001, 0.01))
    feather_fraction = 0.005 if compact else 0.0075

    return {
        "sensitivity": float(sensitivity),
        "feather_fraction": feather_fraction,
        "min_area_fraction": min_area_fraction,
        "envelope_fraction": envelope_fraction,
        "profile": profile,
        "reject_stars": False,
    }
