# -----------------------------------------------------------------------------
# SetiAstro Suite Pro - Acquisition Summary
# Copyright (c) 2026 Franklin Marek / www.setiastro.com
# Licensed under GPLv3.
# -----------------------------------------------------------------------------
"""The Acquisition Summary: the copyable block shown when stacking finishes.

One layout for both places it appears -- the completion dialog, tallied from
the frames' FITS headers (StackingSuiteDialog._build_acquisition_summary), and
Log Analysis, rebuilt from a stacking log that predates logging it -- so the
two can't drift apart. Qt-free; pass ``tr`` to translate the headings.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict

__all__ = ["aggregate_records", "blank_filter_bucket", "blank_night_bucket",
           "format_acquisition_summary", "fmt_duration", "fmt_exp",
           "header_fields"]


def fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as e.g. '5h 42m 00s' / '42m 10s' / '45s'."""
    try:
        s = float(seconds or 0.0)
    except Exception:
        s = 0.0
    if s <= 0:
        return "0s"
    hrs = int(s // 3600)
    rem = s - hrs * 3600
    mins = int(rem // 60)
    secs = int(round(rem - mins * 60))
    # roll-over if rounding bumped us past 60
    if secs == 60:
        secs = 0; mins += 1
    if mins == 60:
        mins = 0; hrs += 1
    if hrs:
        return f"{hrs}h {mins:02d}m {secs:02d}s"
    if mins:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"


def fmt_exp(x: float) -> str:
    """Format a single exposure length."""
    try:
        x = float(x)
    except Exception:
        x = 0.0
    if abs(x - round(x)) < 1e-3:
        return f"{int(round(x))}s"
    return f"{x:.2f}s"


def header_fields(h) -> dict:
    """What the summary takes from one frame's FITS header."""
    exp = h.get("EXPTIME", h.get("EXPOSURE", 0.0))
    try:
        exp = float(exp)
    except Exception:
        exp = 0.0
    return {
        "filter": str(h.get("FILTER", "Unknown") or "Unknown").strip() or "Unknown",
        "exposure": exp,
        "date_obs": str(h.get("DATE-OBS", "") or "").strip(),
        "object": str(h.get("OBJECT", "") or "").strip(),
    }


def blank_filter_bucket() -> dict:
    return {"count": 0, "integration": 0.0, "exposures": defaultdict(int)}


def blank_night_bucket() -> dict:
    return {"count": 0, "integration": 0.0, "filters": defaultdict(int)}


def aggregate_records(records) -> dict:
    """Per-frame records -> the tallies format_acquisition_summary prints.

    Each record is a dict with 'filter', 'exposure' (s), 'night', 'object' and
    'set'. Sets are listed in the order they first appear.
    """
    records = list(records)
    by_filter = defaultdict(blank_filter_bucket)
    by_set: "OrderedDict[str, dict]" = OrderedDict()
    by_night = defaultdict(blank_night_bucket)

    for r in records:
        b = by_filter[r["filter"]]
        b["count"] += 1
        b["integration"] += r["exposure"]
        b["exposures"][round(r["exposure"], 3)] += 1

        buckets = by_set.get(r["set"])
        if buckets is None:
            buckets = by_set[r["set"]] = defaultdict(blank_filter_bucket)
        b = buckets[r["filter"]]
        b["count"] += 1
        b["integration"] += r["exposure"]
        b["exposures"][round(r["exposure"], 3)] += 1

        n = by_night[r["night"]]
        n["count"] += 1
        n["integration"] += r["exposure"]
        n["filters"][r["filter"]] += 1

    return {
        "objects": sorted({r["object"] for r in records if r["object"]}),
        "total_frames": len(records),
        "total_integration": sum(r["exposure"] for r in records),
        "by_filter": by_filter,
        "by_set": by_set,
        "by_night": by_night,
    }


def format_acquisition_summary(data: dict, *, skipped: int = 0,
                               title_suffix: str = "", tr=None,
                               notes=()) -> dict:
    """Lay the tallies out as the copyable summary text.

    ``data`` is aggregate_records' result (or the same shape, with
    'nights_unknown' set when no frame's night is known). ``notes`` are plain
    sentences added at the end. Returns {'text', 'totals'}.
    """
    tr = tr or (lambda s: s)
    objects = data["objects"]
    total_frames = data["total_frames"]
    total_integration = data["total_integration"]
    by_filter = data["by_filter"]
    by_set = data["by_set"]
    by_night = data["by_night"]

    # Date-sorted, with "Unknown date" (if any) last.
    known_nights = sorted(k for k in by_night.keys() if k != "Unknown date")
    all_nights = known_nights + (["Unknown date"] if "Unknown date" in by_night else [])

    def _exps_breakdown(info):
        parts = []
        for exp in sorted(info["exposures"].keys()):
            n = info["exposures"][exp]
            parts.append(f"{n} × {fmt_exp(exp)}")
        return ", ".join(parts)

    # Pad filter names so columns line up — monospace, so this prints clean.
    filt_width = max((len(k) for k in by_filter), default=4)
    filt_width = max(filt_width, 4)

    lines = []
    heading = tr("Acquisition Summary")
    if title_suffix:
        heading = f"{heading} — {title_suffix}"
    lines.append(heading)
    lines.append("─" * max(32, len(heading)))
    lines.append("")

    if objects:
        label = tr("Target(s):")
        lines.append(f"{label:<20}{', '.join(objects)}")

    nights_label = tr("Nights of data:")
    if data.get("nights_unknown"):
        lines.append(f"{nights_label:<20}{tr('(not in this log)')}")
    elif known_nights:
        lines.append(f"{nights_label:<20}{len(known_nights)}  "
                     f"({', '.join(known_nights)})")
    else:
        lines.append(f"{nights_label:<20}{tr('(no DATE-OBS in headers)')}")

    lines.append(f"{tr('Total frames:'):<20}{total_frames}")
    lines.append(f"{tr('Total integration:'):<20}{fmt_duration(total_integration)}")
    lines.append("")

    # By filter
    lines.append(tr("── By filter ──"))
    for f in sorted(by_filter.keys(), key=lambda s: s.lower()):
        info = by_filter[f]
        lines.append(
            f"  {f:<{filt_width}}  "
            f"{info['count']:>4} frames,  "
            f"{fmt_duration(info['integration']):>12}   "
            f"({_exps_breakdown(info)})"
        )
    lines.append("")

    # By set (only if more than one set has frames)
    if len(by_set) > 1:
        lines.append(tr("── By set ──"))
        for s, buckets in by_set.items():
            set_total = sum(info["integration"] for info in buckets.values())
            set_count = sum(info["count"] for info in buckets.values())
            lines.append(
                f"  {s}:  {set_count} frames,  "
                f"{fmt_duration(set_total)}"
            )
            for f in sorted(buckets.keys(), key=lambda x: x.lower()):
                info = buckets[f]
                lines.append(
                    f"     {f:<{filt_width}} "
                    f"{info['count']:>4} × ({_exps_breakdown(info)})  =  "
                    f"{fmt_duration(info['integration'])}"
                )
        lines.append("")

    # By night
    if not data.get("nights_unknown"):
        lines.append(tr("── By night ──"))
        for n in all_nights:
            info = by_night[n]
            parts = sorted(info["filters"].items(), key=lambda kv: kv[0].lower())
            filt_breakdown = ", ".join(f"{k} ×{v}" for k, v in parts)
            lines.append(
                f"  {n}:  {info['count']:>4} frames,  "
                f"{fmt_duration(info['integration']):>12}   "
                f"({filt_breakdown})"
            )
        lines.append("")

    if skipped:
        lines.append(tr(
            "(Note: FITS header could not be read for {0} file(s); "
            "those frames are counted but have 'Unknown' metadata.)"
        ).format(skipped))

    for note in notes:
        lines.append(f"({note})")

    return {
        "text": "\n".join(lines),
        "totals": {
            "frames": total_frames,
            "integration_s": total_integration,
            "nights": len(known_nights),
            "filters": len(by_filter),
        },
    }
