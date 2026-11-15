#!/usr/bin/env python3
"""Measure lights the way Blink pre-flag does, and check grades / suspects.

Star count, FWHM, eccentricity, noise, patchiness and high-end trail come
from SEP on the stretched frame at half resolution. The sky is the
pre-stretch background. Grades and suspects come from ``quality_grades``.

Usage:
    PYTHONPATH=src:venv/lib/python3.14/site-packages \\
        /usr/bin/python3.14 ops/check_preflag.py /path/to/lights
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from setiastro.saspro.stacking_suite import (
    _STAR_COLLAPSE_FRAC,
    _STAR_REF_MIN,
    _STAR_USABLE,
    _group_reference,
    _object_name_from_header,
    _quality_group_key,
    format_quality_reason,
    quality_grades,
)
from setiastro.saspro.blink_comparator_pro import (
    _blink_bin_and_size,
    _blink_exposure_label,
    _blink_field_deg,
    _blink_hdr_get,
    _blink_pointing_deg,
)


def _saved_load_scale() -> int:
    """Blink's saved 1:N view. The star count depends on it."""
    conf = Path.home() / ".config" / "SetiAstro" / "Seti Astro Suite Pro.conf"
    if not conf.is_file():
        return 1
    in_blink = False
    for line in conf.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_blink = stripped == "[blink]"
            continue
        if in_blink and stripped.startswith("load_scale="):
            try:
                return max(1, int(stripped.split("=", 1)[1]))
            except ValueError:
                return 1
    return 1


def _measure(path: str, scale: int = 1) -> dict:
    """One frame, matching Blink at the given load scale.

    Scale 4 or 8 uses the same strided reader as Blink. The sky is the
    pre-stretch background of that view. Stars, FWHM and eccentricity are
    SEP on the stretched view at half of that scale, then FWHM is put back
    in full-resolution pixels.
    """
    import cv2
    import sep
    from setiastro.saspro.blink_fast_read import read_downsampled
    from setiastro.saspro.imageops.stretch import stretch_color_image, stretch_mono_image

    scale = max(1, int(scale))
    image = None
    header = None
    needs_debayer = False
    if scale > 1:
        try:
            fast = read_downsampled(path, scale, cfa_mode="color")
        except Exception:
            fast = None
        if fast is not None:
            image, header, _bit_depth, needs_debayer = fast

    if image is None:
        from setiastro.saspro.legacy.image_manager import load_image
        with contextlib.redirect_stdout(io.StringIO()):
            image, header, _bit_depth, is_mono = load_image(path)
        if image is None or np.size(image) == 0:
            raise RuntimeError(f"empty image: {path}")
        if scale > 1:
            h, w = np.asarray(image).shape[:2]
            image = cv2.resize(
                np.asarray(image),
                (max(1, w // scale), max(1, h // scale)),
                interpolation=cv2.INTER_AREA,
            )
        needs_debayer = bool(is_mono and header is not None and _blink_hdr_get(header, "BAYERPAT"))

    if image is None or np.size(image) == 0:
        raise RuntimeError(f"empty image: {path}")
    header = {} if header is None else header
    if needs_debayer:
        from setiastro.saspro.blink_comparator_pro import BlinkTab
        image = BlinkTab.debayer_image(image, path, header)

    plane = np.asarray(image, dtype=np.float32)
    color = plane.ndim == 3 and plane.shape[2] >= 3
    if color:
        mono = plane.mean(axis=2)
    else:
        mono = plane
    if not np.isfinite(mono).all():
        mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)

    # Pre-stretch sky (full load-scale view); SEP metrics use measure_ds below.
    sky = float(sep.Background(mono).globalback)
    if color:
        stretched = stretch_color_image(plane, 0.25, linked=False, no_black_clip=True)
    else:
        stretched = stretch_mono_image(mono, 0.25, no_black_clip=True)
    stored = np.clip(np.asarray(stretched, dtype=np.float32), 0.0, 1.0)
    stored = (stored * 65535.0).astype(np.uint16).astype(np.float32) / 65535.0
    if stored.ndim == 3:
        stored = stored.mean(axis=2)

    h0, w0 = stored.shape[:2]
    ds = cv2.resize(
        stored,
        (max(16, int(w0 // 2)), max(16, int(h0 // 2))),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32, copy=False)

    from setiastro.saspro.blink_measure_worker import measure_ds
    res = measure_ds(ds)
    if res is None:
        fwhm, ecc, stars = 30.0, 1.0, 0
        noise = patch = trail = float("nan")
    else:
        fwhm_ds, ecc, stars, noise, patch, trail = res
        if not math.isfinite(float(fwhm_ds)):
            fwhm, ecc, stars = 30.0, 1.0, int(stars or 0)
        else:
            fwhm = float(fwhm_ds) * 2.0 * scale
            ecc = float(ecc)
            stars = int(stars)

    ra, dec = _blink_pointing_deg(header)
    bin_v, size = _blink_bin_and_size(header)
    return {
        "path": path,
        "ok": bool(math.isfinite(sky)),
        "sky": float(sky),
        "stars": int(stars),
        "ecc": float(ecc) if math.isfinite(float(ecc)) else float("nan"),
        "fwhm": float(fwhm),
        "noise": float(noise) if math.isfinite(float(noise)) else float("nan"),
        "patch": float(patch) if math.isfinite(float(patch)) else float("nan"),
        "trail": float(trail) if math.isfinite(float(trail)) else float("nan"),
        "obj": _object_name_from_header(header),
        "filt": str(_blink_hdr_get(header, "FILTER") or ""),
        "exposure": _blink_exposure_label(header),
        "bin": None if bin_v is None else int(bin_v),
        "size": None if size is None else [int(size[0]), int(size[1])],
        "ra": None if ra is None else float(ra),
        "dec": None if dec is None else float(dec),
        "field": None if _blink_field_deg(header) is None else float(_blink_field_deg(header)),
    }


def _fits_under(root: Path) -> list[str]:
    exts = {".fits", ".fit", ".fts", ".fz"}
    out = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if Path(name).suffix.lower() in exts:
                out.append(os.path.join(dirpath, name))
    out.sort()
    return out


def _cache_path(root: Path, scale: int) -> Path:
    safe = str(root.resolve()).replace("/", "_")
    return Path("/tmp") / f"saspro-preflag-s{scale}-{safe}.json"


def _load_cache(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _fresh(path: str, cached: dict) -> bool:
    row = cached.get(path)
    if not row:
        return False
    # Invalidate rows measured before noise/patch/trail were stored.
    for key in ("noise", "patch", "trail"):
        if key not in row:
            return False
    try:
        return row.get("mtime") == os.path.getmtime(path)
    except OSError:
        return False


def _check(samples: list[dict], results: list[dict]) -> list[str]:
    """Sanity: collapse flags use the high-end reference; mild dips are grade-only."""
    problems = []
    groups: dict = {}
    for i, sample in enumerate(samples):
        groups.setdefault(_quality_group_key(sample), []).append(i)

    for key, idxs in groups.items():
        measured = [i for i in idxs if samples[i].get("ok")]
        if len(measured) < 4:
            for i in measured:
                code = (results[i].get("verdict") or {}).get("code")
                if code in ("few_stars", "no_stars"):
                    problems.append(f"{Path(samples[i]['path']).name}: star verdict in a group of {len(measured)}")
            continue
        ref = _group_reference([samples[i] for i in measured])
        star_ref = ref["star_ref"]
        collapse_at = (
            max(float(_STAR_USABLE), _STAR_COLLAPSE_FRAC * float(star_ref))
            if np.isfinite(star_ref) else float(_STAR_USABLE)
        )
        for i in measured:
            stars = int(samples[i]["stars"])
            code = (results[i].get("verdict") or {}).get("code")
            name = Path(samples[i]["path"]).name
            grade = results[i].get("grade")
            if code == "no_stars" and stars > 0:
                problems.append(f"{name}: no_stars with {stars} detections")
            if code == "few_stars" and not (0 < stars < collapse_at):
                problems.append(
                    f"{name}: few_stars with {stars} detections (collapse at {collapse_at:.0f})"
                )
            if (
                code is None
                and ref["stable"]
                and np.isfinite(star_ref)
                and star_ref >= _STAR_REF_MIN
                and stars >= collapse_at
                and np.isfinite(grade)
                and grade > 95
                and stars < 0.5 * star_ref
            ):
                problems.append(
                    f"{name}: {stars} stars (~{100 * stars / star_ref:.0f}% of high end) still grade {grade:.0f}"
                )
    return problems


def _report(samples: list[dict], results: list[dict]) -> int:
    problems = _check(samples, results)
    groups: dict = {}
    for i, sample in enumerate(samples):
        groups.setdefault(_quality_group_key(sample), []).append(i)

    print()
    print(f"{'group':<28} {'n':>4} {'g50':>5} {'g10':>5} {'flag':>5}")
    total_flag = 0
    for key in sorted(groups, key=lambda k: (str(k[0]), str(k[1]), str(k[2]))):
        idxs = groups[key]
        measured = [i for i in idxs if samples[i].get("ok")]
        grades = np.asarray(
            [results[i].get("grade", float("nan")) for i in measured], dtype=np.float64)
        grades = grades[np.isfinite(grades)]
        flagged = sum(1 for i in idxs if results[i].get("verdict"))
        total_flag += flagged
        label = f"{key[1] or '-'} {key[2] or '-'}"
        g50 = f"{np.median(grades):.0f}" if grades.size else "-"
        g10 = f"{np.percentile(grades, 10):.0f}" if grades.size else "-"
        print(f"{label:<28} {len(measured):4d} {g50:>5} {g10:>5} {flagged:5d}")

    print()
    print(f"Suspects flagged: {total_flag} / {len(samples)}")
    print()
    shown = 0
    ordered = sorted(
        range(len(samples)),
        key=lambda i: (
            0 if results[i].get("verdict") else 1,
            results[i].get("grade") if np.isfinite(results[i].get("grade", float("nan"))) else 999,
        ),
    )
    for i in ordered:
        sample, result = samples[i], results[i]
        verdict, note = format_quality_reason(result)
        grade = result.get("grade", float("nan"))
        if not verdict and not note and not (np.isfinite(grade) and grade < 50):
            continue
        mark = "FLAG" if verdict else ("note" if note else "low")
        text = verdict or note or ""
        gtxt = f"{grade:5.0f}" if np.isfinite(grade) else "  nan"
        patch = sample.get("patch", float("nan"))
        trail = sample.get("trail", float("nan"))
        ptxt = f"{patch:.3f}" if np.isfinite(patch) else "  nan"
        ttxt = f"{trail:.2f}" if np.isfinite(trail) else " nan"
        print(
            f"  {mark:<4} g={gtxt}  stars {sample['stars']:5d}  "
            f"FWHM {sample['fwhm']:5.2f}  ecc {sample['ecc']:.2f}  "
            f"trail {ttxt}  patch {ptxt}  sky {sample['sky']:.4f}  {text}"
        )
        print(f"       {Path(sample['path']).name}")
        shown += 1
        if shown >= 50:
            print("  ...")
            break
    if not shown:
        print("  No verdict, note, or low grade.")

    print()
    if problems:
        print(f"{len(problems)} check(s) failed:")
        for line in problems:
            print(f"  {line}")
        return 1
    print("Checks passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--scale", type=int, default=0, help="Blink load scale. 0 uses the saved setting")
    parser.add_argument("--limit", type=int, default=0, help="measure at most this many frames")
    args = parser.parse_args()
    scale = args.scale if args.scale > 0 else _saved_load_scale()
    print(f"Blink load scale 1:{scale}")
    root = args.folder
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    paths = _fits_under(root)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"no FITS in {root}", file=sys.stderr)
        return 2

    cache_file = _cache_path(root, scale)
    cache = _load_cache(cache_file)
    pending = [p for p in paths if not _fresh(p, cache)]
    print(f"{len(paths)} frames, {len(pending)} to measure, {len(paths) - len(pending)} cached")

    if pending:
        done = 0

        def _take(path, row):
            nonlocal done
            done += 1
            row["mtime"] = os.path.getmtime(path)
            cache[path] = row
            print(
                f"[{done}/{len(pending)}] {row['filt'] or '-':<3} {row['stars']:5d} stars  "
                f"FWHM {row['fwhm']:5.2f}  {Path(path).name}",
                flush=True,
            )
            if done % 5 == 0:
                cache_file.write_text(json.dumps(cache))

        if args.jobs <= 1:
            for path in pending:
                try:
                    _take(path, _measure(path, scale))
                except Exception as exc:
                    done += 1
                    print(f"[{done}/{len(pending)}] FAIL {Path(path).name}: {exc}", flush=True)
        else:
            ctx = mp.get_context("spawn")
            exe = sys.executable
            if not os.path.basename(exe).startswith("python"):
                exe = "/usr/bin/python3.14" if os.path.isfile("/usr/bin/python3.14") else exe
            ctx.set_executable(exe)
            with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool:
                futures = {pool.submit(_measure, p, scale): p for p in pending}
                for fut in as_completed(futures):
                    path = futures[fut]
                    try:
                        _take(path, fut.result())
                    except Exception as exc:
                        done += 1
                        print(f"[{done}/{len(pending)}] FAIL {Path(path).name}: {exc}", flush=True)
        cache_file.write_text(json.dumps(cache))

    samples = []
    missing = []
    for path in paths:
        row = cache.get(path)
        if not row:
            missing.append(path)
            continue
        sample = {k: row[k] for k in (
            "path", "ok", "sky", "stars", "ecc", "fwhm", "obj", "filt",
            "exposure", "bin", "size", "ra", "dec", "field",
        )}
        sample["noise"] = row.get("noise", float("nan"))
        sample["patch"] = row.get("patch", float("nan"))
        sample["trail"] = row.get("trail", float("nan"))
        if isinstance(sample["size"], list):
            sample["size"] = tuple(sample["size"])
        samples.append(sample)
    if missing:
        print(f"{len(missing)} frames had no measurement", file=sys.stderr)
        return 1

    results = quality_grades(samples)
    return _report(samples, results)


if __name__ == "__main__":
    raise SystemExit(main())
