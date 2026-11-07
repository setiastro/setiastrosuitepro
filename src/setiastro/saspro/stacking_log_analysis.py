# -----------------------------------------------------------------------------
# SetiAstro Suite Pro - Stacking Log Analysis (Beta)
# Copyright (c) 2026 Franklin Marek / www.setiastro.com
# Licensed under GPLv3.
# -----------------------------------------------------------------------------
"""Parse and summarise SASpro stacking logs.

A stacking run writes one timestamped text log per session. On a 770-frame set
that is ~11,000 lines, and the two lines that actually explain a failure are
buried in the middle of it. This module turns that stream back into structure:
phases and their durations, per-group results, per-frame quality statistics,
and -- first of all -- the handful of warnings and errors.

Why regex and not a structured sidecar
--------------------------------------
There are years of existing logs and they have no sidecar. A parser that only
understands logs written after today would be useless for exactly the runs you
want to look back at, so the text IS the format and this parser has to cope.

The cost of that choice is that the rules are pinned to free-text status
strings which change as the stacker changes. Mitigation is ``RULES`` below: one
declarative table of (id, regex) with handlers dispatched by id, so adapting to
a new or reworded message is adding a row, not editing a parser. The second
mitigation is coverage accounting -- every line that matches nothing is counted
and bucketed by shape, and the UI reports it. A rule that silently stops
matching shows up as a coverage drop instead of a quietly empty field.

The parser is deliberately permissive: unknown lines are kept (never dropped),
a truncated or still-being-written log parses fine, and no single bad line can
abort a run report.
"""

from __future__ import annotations

import os
import re
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field

__all__ = [
    "StackingRunReport", "parse_stacking_log", "find_stacking_logs",
    "LogLine", "GroupInfo", "FrameStat", "MasterBuild", "Phase", "Problem",
    "master_outputs", "frame_reconciliation", "slowest_steps", "fmt_hms",
    "SEV_ERROR", "SEV_WARN", "SEV_INFO",
]

# ── severity ────────────────────────────────────────────────────────────────
SEV_ERROR = "error"
SEV_WARN = "warn"
SEV_INFO = "info"

_ERROR_GLYPHS = ("❌", "\U0001f6d1", "⛔")           # X, stop, no-entry
_WARN_GLYPHS = ("⚠",)                                    # warning sign
_ERROR_WORDS = re.compile(
    r"^\s*(?:error|failed|failure|traceback|exception)\b"
    r"|\b(?:raised|aborted)\b", re.I)
_WARN_WORDS = re.compile(r"^\s*(?:warn(?:ing)?)\b", re.I)

# Lines that merely *contain* the word but are routine bookkeeping. Without
# this, "0 errors" style progress chatter floods the Problems panel.
_NOT_A_PROBLEM = re.compile(
    r"no outliers|passed self-test|Cosmetic:\s*(?:True|False)"
    r"|rejected_px|rejected\s*=\s*0\b|,\s*0\s+rejected", re.I)


def _severity(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return SEV_INFO
    head = t[:2]
    if t[0] in _ERROR_GLYPHS or head[:1] in _ERROR_GLYPHS:
        return SEV_ERROR
    if t[0] in _WARN_GLYPHS:
        return SEV_WARN
    if _NOT_A_PROBLEM.search(t):
        return SEV_INFO
    if _ERROR_WORDS.search(t):
        return SEV_ERROR
    if _WARN_WORDS.search(t):
        return SEV_WARN
    return SEV_INFO


# Rules whose matched line should outrank what its leading glyph implies.
# The "no entry" glyph is used both for routine per-sub rejections (hundreds
# per run, not problems) and for frames genuinely dropped from the output, so
# severity cannot be decided from the glyph alone.
SEVERITY_OVERRIDES = {
    "finalize_rejected": SEV_WARN,
    "offgrid_drop": SEV_WARN,
    "geom_audit_bad": SEV_ERROR,
    "cancelled": SEV_WARN,
}


# ── records ─────────────────────────────────────────────────────────────────
@dataclass
class LogLine:
    idx: int                      # 0-based index into the parsed line list
    lineno: int                   # 1-based line number in the file
    t: float | None               # seconds since first timestamp
    clock: str                    # "HH:MM:SS" or ""
    text: str
    severity: str = SEV_INFO
    rule: str = ""                # id of the rule that matched, "" if none
    transient: bool = False       # was a \r progress-overwrite line


@dataclass
class Problem:
    t: float | None
    clock: str
    text: str
    severity: str
    lineno: int


@dataclass
class Phase:
    name: str
    t_start: float
    t_end: float | None = None

    @property
    def duration(self) -> float | None:
        if self.t_end is None or self.t_start is None:
            return None
        return max(0.0, self.t_end - self.t_start)


@dataclass
class GroupInfo:
    name: str
    files_after_filter: int | None = None
    prepass_files: int | None = None
    prepass_canvas: str = ""
    prepass_contributing: int | None = None
    prepass_total: int | None = None
    prepass_seconds: float | None = None
    stacked: int | None = None
    algorithm: str = ""
    weight_min: float | None = None
    weight_max: float | None = None
    weight_mean: float | None = None
    integration_started: float | None = None
    integration_done: float | None = None
    tiles: int | None = None
    soft_rejected: int | None = None
    master_path: str = ""
    t_read: float | None = None
    t_upload: float | None = None
    t_reduce: float | None = None
    t_download: float | None = None
    plate_solved: bool | None = None
    plate_solve_note: str = ""
    drizzle_path: str = ""
    drizzle_scale: float | None = None
    drizzle_pixfrac: float | None = None
    drizzle_kernel: str = ""
    drizzle_frames: int | None = None
    drizzle_canvas: str = ""

    @property
    def io_seconds(self):
        vals = [v for v in (self.t_read, self.t_upload, self.t_download)
                if v is not None]
        return sum(vals) if vals else None

    @property
    def integration_seconds(self) -> float | None:
        if self.integration_started is None or self.integration_done is None:
            return None
        return max(0.0, self.integration_done - self.integration_started)


@dataclass
class MasterBuild:
    """One master dark/flat build: what went in, how, and what came out."""

    kind: str = ""                 # "dark" | "flat"
    key: str = ""                  # "0s (8288x5644)" / "0.0s (8288x5644) [L]"
    frames: int | None = None
    session: str = ""
    temp: str = ""
    canvas: str = ""
    method: str = ""
    tiles: int | None = None
    dark_used: str = ""
    t_start: float | None = None
    t_end: float | None = None
    path: str = ""
    solved: bool | None = None
    solve_note: str = ""

    @property
    def seconds(self):
        if self.t_start is None or self.t_end is None:
            return None
        return max(0.0, self.t_end - self.t_start)


@dataclass
class FrameStat:
    name: str
    star_count: float | None = None
    ecc: float | None = None
    bg: float | None = None
    fwhm: float | None = None
    sharp: float | None = None
    snr: float | None = None
    exp_scale: float | None = None
    weight: float | None = None
    normalized: float | None = None


@dataclass
class StackingRunReport:
    path: str = ""
    started: str = ""
    closed: str = ""
    directory: str = ""
    complete: bool = False            # saw the "# Log closed" marker

    lines: list = field(default_factory=list)          # list[LogLine]
    phases: list = field(default_factory=list)         # list[Phase]
    problems: list = field(default_factory=list)       # list[Problem]
    groups: "OrderedDict[str, GroupInfo]" = field(default_factory=OrderedDict)
    frame_stats: list = field(default_factory=list)    # list[FrameStat]
    master_builds: list = field(default_factory=list)  # list[MasterBuild]
    cal_groups: list = field(default_factory=list)     # (label, session, files, dur)
    filter_rejections: list = field(default_factory=list)  # (filt, n, tot, thr, med, sigma)
    _cur_master: object = None                         # parser scratch
    _cur_int_group: str = ""                           # parser scratch
    _cur_drizzle_group: str = ""                       # parser scratch

    facts: "OrderedDict[str, str]" = field(default_factory=OrderedDict)
    counters: Counter = field(default_factory=Counter)
    modes: list = field(default_factory=list)          # detected pipeline modes

    masters_dark: list = field(default_factory=list)
    masters_flat: list = field(default_factory=list)
    lights_added: list = field(default_factory=list)
    rejected_subs: list = field(default_factory=list)  # (name, score)
    flips: list = field(default_factory=list)          # (name, degrees)
    align_deltas: list = field(default_factory=list)   # (name, dx, dy)
    calib_pairs: list = field(default_factory=list)    # (frame, dark, flat, cosmetic)
    written_frames: list = field(default_factory=list)
    failed_frames: list = field(default_factory=list)  # (name, reason)
    skipped_frames: list = field(default_factory=list)  # (name, reason)

    # coverage / honesty accounting
    n_lines: int = 0
    n_stamped: int = 0
    n_matched: int = 0
    unmatched_shapes: Counter = field(default_factory=Counter)
    parse_errors: list = field(default_factory=list)

    @property
    def duration(self) -> float | None:
        ts = [l.t for l in self.lines if l.t is not None]
        return (max(ts) - min(ts)) if ts else None

    @property
    def content_lines(self) -> int:
        """Lines that actually carried something to recognise."""
        return max(0, self.n_lines - self.counters.get("blank", 0)
                   - self.counters.get("transient", 0))

    @property
    def coverage(self) -> float:
        """Share of CONTENT lines a rule understood.

        Blank bodies and transient progress-overwrite chatter are excluded
        from the denominator: they carry nothing to recognise, so counting
        them would understate how well the rules actually fit the log.
        """
        return 100.0 * self.n_matched / max(1, self.content_lines)

    @property
    def errors(self) -> list:
        return [p for p in self.problems if p.severity == SEV_ERROR]

    @property
    def warnings(self) -> list:
        return [p for p in self.problems if p.severity == SEV_WARN]

    def group(self, name: str) -> GroupInfo:
        g = self.groups.get(name)
        if g is None:
            g = GroupInfo(name=name)
            self.groups[name] = g
        return g


# ── helpers ─────────────────────────────────────────────────────────────────
_TS = re.compile(r"^\[(\d{1,2}):(\d{2}):(\d{2})\]\s*(.*)$")
_HDR = re.compile(r"^#\s*([A-Za-z ]+?):\s*(.*)$")

_NUM = r"([-+]?\d+(?:\.\d+)?)"


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _shape(t: str) -> str:
    """Collapse a line to a comparable 'shape' for coverage bucketing."""
    s = t
    s = re.sub(r"[A-Za-z]:[\\/][^\s,;]+", "<path>", s)
    s = re.sub(r"[\w\-.+]+\.(?:fits?|fts|xisf|tiff?|png|jpe?g|cr[23]|nef|arw|dng|sasd|npy|txt)\b",
               "<file>", s, flags=re.I)
    s = re.sub(r"\d+\s*[x×]\s*\d+", "<WxH>", s)
    s = re.sub(r"[-+]?\d+\.\d+", "<f>", s)
    s = re.sub(r"\b\d+\b", "<n>", s)
    s = re.sub(r"'[^']{0,90}'", "'<g>'", s)
    return re.sub(r"\s+", " ", s).strip()[:160]


# ── rule table ──────────────────────────────────────────────────────────────
# (id, regex). Order matters only in that the FIRST match classifies the line;
# a line can still feed several facts if its handler says so. Add a row here
# when the stacker gains or rewords a message -- no parser surgery needed.
RULES: list = [
    # run / environment
    ("log_opened",      re.compile(r"Log file opened:\s*(?P<path>.+)$")),

    # --- round 6: plate solving the finished masters -------------------
    ("plate_solve_go",  re.compile(
        r"Plate solving\s+'(?P<g>[^']+)'\s+master\s*\((?P<chain>[^)]*)\)")),
    ("plate_solve_ok",  re.compile(
        r"Plate solve succeeded for\s+'(?P<g>[^']+)'\s+master")),
    ("plate_solve_bad", re.compile(
        r"Plate solve failed for\s+'(?P<g>[^']+)'\s+master:\s*(?P<why>.+?)\s*$")),
    ("preflight",       re.compile(r"Worker preflight OK\s*\((?P<env>[^)]*)\)")),
    ("precision",       re.compile(r"Internal precision:\s*(?P<v>\S+)")),
    ("accel",           re.compile(r"Hardware acceleration:\s*(?P<v>\w+)")),
    ("settings_saved",  re.compile(r"Saved stacking settings")),
    ("free_disk",       re.compile(r"Free on target drive:\s*" + _NUM + r"\s*(?P<unit>[KMGT]i?B)")),
    ("est_disk",        re.compile(r"Estimated disk for this step:\s*~?" + _NUM + r"\s*GB")),

    # masters / inputs
    ("master_dark",     re.compile(r"Added Master DARK\s*->\s*(?P<path>.+?)\s+under\s+(?P<key>.+?)\s+with metadata")),
    ("master_flat",     re.compile(r"Added Master FLAT\s*->\s*(?P<path>.+?)\s+under\s+(?P<key>.+?)\s+with metadata")),
    ("light_added",     re.compile(r"Added\s+(?P<name>\S+)\s+as LIGHT")),
    ("tree_snapshot",   re.compile(r"Tree snapshot\s*→\s*groups:\s*(?P<groups>\d+),\s*leaves seen:\s*(?P<leaves>\d+),\s*paths kept:\s*(?P<kept>\d+)")),
    ("group_files",     re.compile(r"Group\s+'(?P<g>[^']+)'\s+has\s+(?P<n>\d+)\s+file\(s\) after filtering")),
    ("group_listed",    re.compile(r"^⤴\s*(?P<g>.+?):\s*(?P<n>\d+)\s+files\s*$")),
    ("total_frames",    re.compile(r"Found\s+(?P<n>\d+)\s+total frames")),

    # phases
    ("phase_start",     re.compile(r"Phase:\s*(?P<name>.+?)\s+starting")),
    ("phase_done",      re.compile(r"Phase:\s*(?P<name>.+?)\s+complete\.?\s*$")),

    # calibration
    ("calib_pair",      re.compile(r"^\U0001f527\s*(?P<frame>\S+)\s*->\s*Dark:\s*(?P<dark>.+?)\s*\|\s*Flat:\s*(?P<flat>.+?)\s*\|\s*Cosmetic:\s*(?P<cos>\w+)")),
    ("calib_group",     re.compile(r"Calibrating group:\s*(?P<g>.+)$")),
    ("calib_saved",     re.compile(r"Saved:\s*(?P<name>\S+)\s*\((?P<i>\d+)/(?P<n>\d+)\)")),
    ("calib_progress",  re.compile(r"Progress:\s*(?P<g>.+?)\s*—\s*(?P<i>\d+)/(?P<n>\d+)\s*frames")),
    ("pipeline_start",  re.compile(r"Calibration frames ready\. Starting pipeline on\s*(?P<n>\d+)")),
    ("satmask",         re.compile(r"^\[satmask\]")),
    ("debayer",         re.compile(r"Debayering\s+(?P<name>\S+)")),

    # measurement / weighting
    ("measured_n",      re.compile(r"Measured\s+(?P<i>\d+)/(?P<n>\d+)\s+frames")),
    ("measure_done",    re.compile(r"All chunks complete! Measured\s+(?P<n>\d+)\s+frames total")),
    ("measure_procs",   re.compile(r"Measuring\s+(?P<n>\d+)\s+frames across\s+(?P<p>\d+)\s+processes")),
    ("frame_stat",      re.compile(
        r"^\U0001f4c2\s*(?P<name>\S+)\s*→\s*StarCount=(?P<stars>[-+\d.]+)"
        r"(?:.*?Ecc=(?P<ecc>[-+\d.]+))?(?:.*?Bg=(?P<bg>[-+\d.]+))?"
        r"(?:.*?FWHM~(?P<fwhm>[-+\d.]+))?(?:.*?Sharp=(?P<sharp>[-+\d.]+))?"
        r"(?:.*?SNR=(?P<snr>[-+\d.]+))?(?:.*?ExpScale=(?P<exps>[-+\d.]+))?"
        r"(?:.*?Weight=(?P<w>[-+\d.]+))?(?:.*?Normalized=(?P<norm>[-+\d.]+))?")),
    ("weights_global",  re.compile(r"Frame weights computed:\s*min=" + _NUM + r",\s*max=" + _NUM + r",\s*mean=" + _NUM)),
    ("weights_group",   re.compile(r"Weight range for\s+'(?P<g>[^']+)':\s*min=" + _NUM + r"\s*max=" + _NUM + r"\s*mean=" + _NUM)),
    ("weights_mode",    re.compile(r"Frame Weights Debug Log\s*\(mode:\s*(?P<mode>[^)]+)\)")),
    ("sub_reject_sum",  re.compile(r"Sub rejection:\s*(?P<n>\d+)\s*/\s*(?P<tot>\d+)\s*subs rejected across\s*(?P<g>\d+)\s*filter group")),
    ("sub_reject_one",  re.compile(r"Rejected sub \(score outlier\):\s*(?P<name>\S+)\s*\(score\s*" + _NUM + r"\)")),
    ("no_outliers",     re.compile(r"no outliers across any filter\s*—\s*all\s*(?P<n>\d+)\s*subs kept")),

    # reference / geometry
    ("ref_select",      re.compile(r"Auto-selected reference:\s*(?P<name>.+?)\s*\(score=" + _NUM + r"\)")),
    ("ref_select2",     re.compile(r"Auto-selected reference:\s*(?P<name>.+)$")),
    ("ref_from_stamp",  re.compile(r"Reference from SAS_REF stamp:\s*(?P<name>.+)$")),
    ("ref_borderless",  re.compile(r"Reference = borderless frame[^:]*:\s*(?P<name>.+)$")),
    ("ref_align",       re.compile(r"Reference for alignment(?:\s*\([^)]*\))?:\s*(?P<path>.+)$")),
    ("ref_scale",       re.compile(r"Reference pixel scale:\s*" + _NUM)),
    ("ref_pa",          re.compile(r"Reference PA:\s*" + _NUM)),
    ("ref_median",      re.compile(r"Reference \(debayered\) median:\s*" + _NUM)),
    ("target_scale",    re.compile(r"Target pixel scale[:\s]*(?P<v>.+)$")),
    ("binning",         re.compile(r"Binning summary\s*→\s*target=(?P<v>\S+)\s*\((?P<range>[^)]*)\)")),
    ("canvas_change",   re.compile(r"Normalize (?:canvas|target shape)\s*(?P<from>\d+×\d+)\s*→\s*(?P<to>\d+×\d+)")),
    ("pixscale_norm",   re.compile(r"Pixel-scale normalize .*?size\s*(?P<from>\d+×\d+)\s*→\s*(?P<to>\d+×\d+)")),
    ("bin_resample",    re.compile(r"Resampled for binning\s*(?P<from>\S+)\s*→\s*(?P<to>\S+)")),
    ("geom_audit",      re.compile(r"Geometry audit\s+'(?P<g>[^']+)':\s*(?P<rest>.+)$")),
    ("offgrid_drop",    re.compile(r"not on the reference grid")),
    ("ext_prereg",      re.compile(r"Externally pre-registered set")),

    # normalization
    ("norm_chunk",      re.compile(r"Normalizing chunk\s*(?P<i>\d+)/(?P<n>\d+)")),
    ("norm_load",       re.compile(r"Loading\s+(?P<n>\d+)\s+images in parallel for normalization")),
    ("norm_updated",    re.compile(r"Updated self\.light_files to use debayered")),
    ("abe",             re.compile(r"ABE\s+(?P<i>\d+)/(?P<n>\d+)")),

    # registration
    ("reg_started",     re.compile(r"Image Registration Started")),
    ("align_model",     re.compile(r"Alignment model\s*=\s*(?P<v>\S+)")),
    ("align_procs",     re.compile(r"Using\s+(?P<n>\d+)\s+processes for stellar alignment")),
    ("align_progress",  re.compile(r"Aligning stars…\s*\((?P<i>\d+)/(?P<n>\d+)\)")),
    ("refine_pass",     re.compile(r"Refinement Pass\s*(?P<i>\d+)/(?P<n>\d+)")),
    ("aa_delta",        re.compile(r"Astroalign delta for\s+(?P<name>\S+)\s*\([^)]*\):\s*dx=" + _NUM + r",\s*dy=" + _NUM)),
    ("aa_tile",         re.compile(r"^\[AA tile")),
    ("ransac",          re.compile(r"^\[RANSAC\]")),
    ("finalize_dbg",    re.compile(r"^\[finalize\]")),
    ("distortion",      re.compile(r"Distortion Correction on\s+(?P<name>\S+):\s*warp=(?P<warp>\S+)")),
    ("wrote_frame",     re.compile(r"Wrote\s+(?P<name>\S+)\s*\[(?P<warp>[^\]]+)\]")),
    ("flip",            re.compile(r"(?P<deg>\d+)° rotate \(PA Δ≈[^)]*\):\s*(?P<name>\S+)")),
    ("flip_plain",      re.compile(r"^Flipping Image")),
    ("reg_complete",    re.compile(r"Registration complete\. Valid frames:\s*(?P<ok>\d+)/(?P<tot>\d+)")),
    ("align_summary",   re.compile(r"Alignment summary:\s*(?P<ok>\d+)\s+succeeded,\s*(?P<bad>\d+)\s+rejected")),
    ("within_tol",      re.compile(r"Within tolerance\s*\(≤\s*" + _NUM + r"px\):\s*(?P<n>\d+)\s*frame")),
    ("transform_saved", re.compile(r"Transform file saved as\s+(?P<name>.+?)(?:\s*\(v2\))?\s*$")),
    ("reg_precheck",    re.compile(r"reg-precheck:\s*(?P<dirs>\d+)\s*aligned dir")),
    ("pass_shifts",     re.compile(r"Pass\s+(?P<i>\d+)\s+delta shifts")),
    ("skip_reg",        re.compile(r"Skip[- ]Registration|skip registration", re.I)),

    # integration
    ("rej_prepass_all", re.compile(r"Rejection prepass:\s*(?P<g>\d+)\s*group\(s\),\s*(?P<n>\d+)\s*aligned frame")),
    ("rej_prepass_one", re.compile(r"\[(?P<i>\d+)/(?P<n>\d+)\]\s*Rejection prepass for\s+'(?P<g>[^']+)'\s*with\s*(?P<files>\d+)\s*file")),
    ("rej_backend",     re.compile(r"Rejection GPU backend passed self-test")),
    ("int_start",       re.compile(r"Starting integration for group\s+'(?P<g>[^']+)'\s*with\s*(?P<n>\d+)\s*files")),
    ("int_done",        re.compile(r"Integration complete for group\s+'(?P<g>[^']+)'")),
    ("int_algo",        re.compile(r"Stacking group\s+'(?P<g>[^']+)'\s*with\s*(?P<algo>.+?)\s*$")),
    ("int_tiles",       re.compile(r"Integrating\s+(?P<n>\d+)\s+tiles\s*\((?P<how>[^)]*)\)")),
    ("tile_progress",   re.compile(r"^\U0001f527\s*Tile\s+(?P<i>\d+)/(?P<n>\d+)")),
    # Three historical formats; most specific first.
    ("prepass_soft",    re.compile(
        r"Prepass\s+'(?P<g>[^']+)':\s*" r"(?P<canvas>\d+\s*[x×]\s*\d+)" r",\s*"
        r"(?P<contrib>\d+)/(?P<tot>\d+)\s*frame\(s\) contributing\s*"
        r"\((?P<soft>\d+)\s*soft-rejected\),\s*" + _NUM + r"s")),
    ("prepass_result",  re.compile(
        r"Prepass\s+'(?P<g>[^']+)':\s*" r"(?P<canvas>\d+\s*[x×]\s*\d+)" r",\s*"
        r"(?P<contrib>\d+)/(?P<tot>\d+)\s*frame\(s\) contributing,\s*" + _NUM + r"s")),
    ("prepass_plain",   re.compile(
        r"Prepass\s+'(?P<g>[^']+)':\s*" r"(?P<canvas>\d+\s*[x×]\s*\d+)" r",\s*"
        r"(?P<tot>\d+)\s*frame\(s\),\s*" + _NUM + r"s")),
    ("group_stacked",   re.compile(r"^•\s*(?P<g>.+?):\s*(?P<n>\d+)\s+stacked(?:\s*\+\s*drizzle)?\s*$")),
    ("finalize_prepass", re.compile(r"Post-align finalize from prepass:\s*(?P<g>\d+)\s*group\(s\),\s*(?P<n>\d+)")),
    ("finalize_one",    re.compile(r"\[(?P<i>\d+)/(?P<n>\d+)\]\s*Finalizing\s+'(?P<g>[^']+)'\s*from prepass")),
    ("finalize_time",   re.compile(r"Finalize from prepass done in\s*" + _NUM + r"s")),
    ("post_align_done", re.compile(r"Post-alignment complete")),
    ("drizzle_off",     re.compile(r"Drizzle disabled")),
    # Specific drizzle rules MUST precede the generic drizzle_on
    # catch-all below, or it swallows them and the data is lost.
    ("drizzle_cfg",     re.compile(
        r"Drizzle cfg\s*→\s*scale=(?P<scale>[\d.]+)×?,\s*pixfrac=(?P<pf>[\d.]+),\s*kernel=(?P<kern>\S+)")),
    ("drizzle_group",   re.compile(
        r"Drizzle for\s+'(?P<g>[^']+)'\s+at\s+(?P<scale>[\d.]+)×?\s*\((?:drop|pixfrac)=(?P<drop>[\d.]+)\)\s*using\s+(?P<n>\d+)\s+frame")),
    ("drizzle_stack",   re.compile(
        r"Drizzle stacking for group\s+'(?P<g>[^']+)'\s+with\s+(?P<n>\d+)\s+rejection tiles")),
    ("drizzle_bbox",    re.compile(
        r"bbox\(ref\)→reg:.*?canvas\s+(?P<canvas>\d+\s*[x×]\s*\d+)\s*@\s*(?P<scale>[\d.]+)")),
    ("drizzle_saved",   re.compile(
        r"Drizzle\s*\(original\)\s*saved:\s*(?P<path>.+?)\s*$")),
    ("drizzle_master_line", re.compile(
        r"^\s*•\s*(?P<g>.+?)\s*→\s*(?P<path>\S.*?\.(?:fits?|fts|xisf))\s*$")),
    ("drizzle_masters_hdr", re.compile(r"Drizzle masters saved")),
    ("drizzle_warp",    re.compile(
        r"Drizzle uses\s+(?P<kind>\S+)\s+for\s+(?P<name>\S+)")),
    ("drizzle_kern_use", re.compile(r"Using\s+(?P<kern>\S+)\s+drizzle")),
    ("drizzle_accum",   re.compile(r"Drizzle: accumulating frames")),
    ("drizzle_dict",    re.compile(r"Drizzle dict:")),
    ("drizzle_on",      re.compile(r"[Dd]rizzl\w+", re.I)),
    ("mfdeconv",        re.compile(r"MFDeconv", re.I)),
    ("masters_opened",  re.compile(r"Opened\s+(?P<n>\d+)\s+master\(s\) in SASpro")),
    ("summary_headers", re.compile(r"Reading headers from\s+(?P<n>\d+)\s+frame\(s\) for the summary")),
    ("hdr_read",        re.compile(r"Read\s+(?P<i>\d+)/(?P<n>\d+)\s+calibrated frame headers")),
    ("sets_assigned",   re.compile(r"Auto-assigned targets to Sets:\s*(?P<v>.+?)\.")),
    ("tidying",         re.compile(r"Doing a little tidying up")),
    ("released",        re.compile(r"Releasing previous set runtime buffers")),
    ("cancelled",       re.compile(r"Registration cancelled|Stacking cancelled|halted", re.I)),

    # --- per-frame failures (round 2: from corpus survey) ---------------
    ("aa_failed",       re.compile(r"Error:\s*Astroalign failed for\s+(?P<name>\S+?):\s*(?P<why>.+)$")),
    ("aa_timeout",      re.compile(r"Error:\s*Astroalign timeout for\s+(?P<name>\S+?)\s*\((?P<why>[^)]*)\)")),
    ("finalize_error",  re.compile(r"Finalize error\s+(?P<name>\S+?):\s*(?P<why>.+)$")),
    ("gpu_calib_fail",  re.compile(r"GPU calibration failed for\s+(?P<name>\S+?):\s*(?P<why>.+)$")),
    ("align_skipped",   re.compile(r"Skipping\s+(?P<name>\S+?)\s*\u2014\s*(?P<why>.*?alignment failed.*?)(?:\.|$)")),
    ("failed_bare",     re.compile(r"^\u274c\s*(?P<name>[\w\-.+]+\.(?:fits?|fts|xisf))\s*$")),

    # --- calibration detail --------------------------------------------
    ("frame_added",     re.compile(r"Added\s+(?P<name>\S+)\s+as\s+(?P<kind>DARK|FLAT|BIAS|FLATDARK)\b")),
    ("dark_sub",        re.compile(r"Dark Subtracted:\s*(?P<name>\S+)")),
    ("flat_applied",    re.compile(r"Flat Applied:\s*(?P<name>\S+)")),
    ("cosmetic_bayer",  re.compile(r"Cosmetic Correction Applied for Bayer Pattern\s*\((?P<pat>[^)]*)\)")),
    ("cosmetic_mono",   re.compile(r"Cosmetic Correction Applied\s*\((?P<how>[^)]*)\)")),
    ("sat_running",     re.compile(r"Running Satellite Trail Removal")),
    ("sat_applied",     re.compile(r"Satellite Trail Removal Applied(?:\s*\((?P<how>[^)]*)\))?")),
    ("master_debug",    re.compile(r"DEBUG: Master Files Stored")),
    ("before_saving",   re.compile(r"Before saving:\s*min\s*=\s*" + _NUM + r",\s*max\s*=\s*" + _NUM)),

    # --- masks / measurement -------------------------------------------
    ("mask_saved",      re.compile(r"(?:Saved mask|Mask saved):\s*(?P<name>\S+)\s*\((?P<n>\d+)\s*(?:rejected\s*)?px\)")),
    ("starmask_dense",  re.compile(r"Star mask: dense field")),
    ("starmask_none",   re.compile(r"Star mask: no sources found")),
    ("variance_map",    re.compile(r"Variance map:")),
    ("measured_orig",   re.compile(r"Measured\s+(?P<i>\d+)/(?P<n>\d+)\s+originals")),
    ("coverage_comp",   re.compile(r"Coverage computed\s+(?P<i>\d+)/(?P<n>\d+)")),
    ("measure_chunk",   re.compile(r"Measuring chunk\s+(?P<i>\d+)/(?P<n>\d+)")),
    ("filter_target",   re.compile(r"Filter\s+'(?P<f>[^']+)'\s*target\s*\u2190\s*(?P<name>\S+)\s*\(score=" + _NUM + r"\)")),

    # --- background extraction -----------------------------------------
    ("abe_config",      re.compile(r"Gradient removal \(ABE[^)]*\):\s*(?P<cfg>.+)$")),
    ("abe_done",        re.compile(r"Gradient removal complete:\s*(?P<ok>\d+)\s*ok,\s*(?P<skip>\d+)\s*skipped,\s*(?P<err>\d+)\s*error")),

    # --- round 3: outputs, reference selection, runtime config ----------
    ("master_saved",    re.compile(r"Saved integrated image[^']*for\s+'(?P<g>[^']+)':\s*(?P<path>.+)$")),
    ("master_hdr_fix",  re.compile(r"Header corrected for\s*(?P<f>\S+)\s*(?:upscale|drizzle scale) on\s+'(?P<g>[^']+)'\s*master")),
    ("dark_loaded",     re.compile(r"Dark loaded:\s*(?P<name>\S+)")),
    ("flat_loaded",     re.compile(r"Flat loaded:\s*(?P<name>\S+)")),
    ("weight_mode2",    re.compile(r"Weighting mode:\s*(?P<mode>.+?)\s*$")),
    ("ref_selecting",   re.compile(r"Selecting reference frame")),
    ("ref_autoaccept",  re.compile(r"Auto-accept measured reference")),
    ("ref_minmed",      re.compile(r"Reference min=" + _NUM + r",\s*normalized-median=" + _NUM)),
    ("auto_thresh",     re.compile(r"Auto detecting threshold on reference")),
    ("auto_thresh2",    re.compile(r"Auto threshold: writing\s*(?P<v>\S+)\s*to settings")),
    ("prev_parallel",   re.compile(r"Loading\s+(?P<n>\d+)\s+previews in parallel")),
    ("global_means",    re.compile(r"Measuring global means in parallel")),
    ("finalize_procs",  re.compile(r"Finalizing aligned outputs with\s*(?P<n>\d+)\s*processes")),
    ("finalize_cfg",    re.compile(r"Finalize:\s*(?P<w>\d+)\s*workers?\s*x\s*(?P<t>\d+)\s*cv2 threads")),
    ("chunk_size",      re.compile(r"Using chunk size\s*(?P<v>\S+)")),
    ("rowband",         re.compile(r"Row-band mode:\s*(?P<rows>\d+)\s*rows")),
    ("hdr_bg_read",     re.compile(r"Reading headers of\s*(?P<n>\d+)\s*calibrated frame")),
    ("gpu_transfer",    re.compile(r"GPU transfer mode:\s*(?P<v>.+?)\s*$")),
    ("disk_estimate",   re.compile(r"(?P<what>Normalized frames|Registered frames|Calibrated frames)\s*\(×\s*(?P<n>\d+),[^)]*\):\s*~?" + _NUM + r"\s*GB")),
    ("starmask_stats",  re.compile(r"Star mask:\s*thresh=(?P<thresh>\S+)\s*\|\s*detected=(?P<det>\d+)\s*\|\s*kept=(?P<kept>\d+)")),

    # --- round 4: master dark / master flat creation --------------------
    ("mdark_begin",     re.compile(r"Starting Master Dark Creation")),
    ("mflat_begin",     re.compile(r"Starting Master Flat Creation")),
    ("master_discover", re.compile(
        r"Discovered\s+(?P<n>\d+)\s+(?P<what>grouped exposures|flat groups|light groups)"
        r"\s*\((?P<elig>\d+)\s*eligible to stack\)\s*[\u2014-]\s*(?P<files>\d+)\s*files total")),
    ("cal_groups_hdr",  re.compile(r"(?P<kind>Dark|Flat|Light) groups summary")),
    ("cal_group_line",  re.compile(
        r"^\u2022\s*(?P<g>.+?)\s*\|\s*session:\s*(?P<sess>\S+)\s*"
        r"\u2192\s*(?P<n>\d+)\s*files,\s*(?P<dur>.+?)\s*$")),
    ("total_tiles",     re.compile(r"Total tiles to process:\s*(?P<n>\d+)")),
    ("progress_init",   re.compile(r"Progress initialized:\s*(?P<n>\d+)\s*tiles")),
    ("mdark_group",     re.compile(
        r"Processing\s+(?P<n>\d+)\s+darks for\s+(?P<key>.+?)\s+in session\s+"
        r"'(?P<sess>[^']*)'\s*at\s*(?P<temp>[-+\d.]+)\s*C")),
    ("mflat_group",     re.compile(
        r"Processing\s+(?P<n>\d+)\s+flats for\s+(?P<key>.+?)\s+in session\s+"
        r"'(?P<sess>[^']*)'")),
    ("master_method",   re.compile(r"(?P<m>GPU .+?)\s+selected for\s+(?P<n>\d+)\s+frames")),
    ("flat_reducer",    re.compile(r"GPU reducer for flats\s*[\u2014-]\s*(?P<m>.+?)\s*$")),
    ("flat_norm",       re.compile(r"Normalizing\s+(?P<n>\d+)\s+flats by")),
    ("flat_dark",       re.compile(r"This flat group: dark\s*\u2192\s*(?P<name>\S+)")),
    ("temp_memmap",     re.compile(r"Creating temp memmap:\s*(?P<name>\S+)")),
    ("group_tiles",     re.compile(r"(?P<n>\d+)\s+tiles to process for this group")),
    ("tile_proc_start", re.compile(r"Starting tile processing for group\s+'(?P<g>[^']+)'")),
    ("mdark_saved",     re.compile(r"Master Dark saved(?:\s+and added to UI)?:\s*(?P<path>.+?)\s*$")),
    ("mflat_saved",     re.compile(r"Master Flat (?:queued|saved)(?:\s+and added to UI)?:\s*(?P<path>.+?)\s*$")),
    ("mflats_done",     re.compile(r"All master flats saved")),
    ("preload_darks",   re.compile(r"Pre-loading\s+(?P<n>\d+)\s+unique master dark")),
    ("cal_precheck",    re.compile(r"cal-precheck:\s*(?P<ok>\d+)/(?P<tot>\d+)\s*usable twins")),
    ("load_masters_gpu", re.compile(r"Loading master calibration frames to GPU")),
    ("calib_complete",  re.compile(r"Calibration Complete")),
    ("checked_file",    re.compile(r"^\u2713\s*(?P<name>\S+\.(?:fits?|fts|xisf))\s*$")),

    # --- round 5: integration timing, rejection detail, audit output ----
    ("int_timing",      re.compile(
        r"Integration timing:\s*read wait=" + _NUM + r"s[,\s]*"
        r"upload staging=" + _NUM + r"s[,\s]*"
        r"reduction=\S+\s*" + _NUM + r"s[,\s]*"
    # Named groups for the numbers too. This rule already has three named
    # groups ahead of them, so positional m.group(3..5) silently read
    # filt/n/tot instead of the statistics -- the UI showed the frame
    # total where the score threshold belongs.
        r"download/wait=" + _NUM + r"s")),
    ("filter_reject",   re.compile(
        r"^\u2022\s*\[(?P<filt>[^\]]+)\]\s*rejected\s*(?P<n>\d+)/(?P<tot>\d+)\s*"
        r"threshold=(?P<thr>[-+]?\d+(?:\.\d+)?)\s*\(median=(?P<med>[-+]?\d+(?:\.\d+)?),\s*"
        r"\u03c3=(?P<sig>[-+]?\d+(?:\.\d+)?)\)")),
    ("finalize_rejected", re.compile(
        r"Rejected\s+(?P<n>\d+)\s+frame\(s\) whose aligned output wasn't written")),
    ("exposure_scaling", re.compile(
        r"Exposure scaling\s+(?P<state>ON|OFF)\b[^;]*;\s*(?P<n>\d+)/(?P<tot>\d+)\s*frame")),
    ("exposure_scaling2", re.compile(r"Exposure scaling\s+(?P<state>ON|OFF)\b")),
    ("worker_count",    re.compile(r"Worker count\s+(?P<from>\d+)\s*->\s*(?P<to>\d+)")),
    ("geom_audit_src",  re.compile(r"source folders:\s*(?P<v>.+?)\s*$")),
    ("geom_audit_bad",  re.compile(
        r"^\u274c\s*\[(?P<idx>\d+)\]\s*(?P<hw>\d+\s*[x\u00d7]\s*\d+)\s+(?P<name>.+?)\s*$")),

    # --- round 7: cancellation, exclusions, drizzle bookkeeping --------
    ("restored_tabs",   re.compile(r"Restored tab file lists")),
    ("cancel_request",  re.compile(r"Cancel requested")),
    ("int_cancelled",   re.compile(
        r"Integration cancelled for group\s+'(?P<g>[^']+)'")),
    ("cancelled_user",  re.compile(r"^Cancelled by user")),
    ("stopped_user",    re.compile(r"Stacking stopped by user")),
    ("excluded_frames", re.compile(
        r"Excluding\s+(?P<n>\d+)\s+removed frame")),
    ("reg_twins",       re.compile(
        r"reg-precheck:\s*(?P<ok>\d+)/(?P<tot>\d+)\s*usable twins")),
    ("rejection_map",   re.compile(r"Saved rejection map to\s*(?P<path>.+?)\s*$")),
    ("sasd_loaded",     re.compile(r"SASD v2: loaded\s+(?P<n>\d+)\s+transform")),
    ("sample_list",     re.compile(r"^(?:originals needed|sasd FILEs)\s*\(sample\)")),
]

# Pipeline-mode fingerprints: rule ids (or regexes) whose presence means the
# run exercised that part of the stacker. Drives the "modes detected" readout.
MODE_RULES: list = [
    ("Calibration",          ("calib_pair", "calib_group", "calib_saved")),
    ("Cosmetic correction",  ("cosmetic_bayer", "cosmetic_mono",
                              "__re__", re.compile(r"Cosmetic:\s*True"))),
    ("Satellite trails",     ("satmask", "sat_running", "sat_applied")),
    ("Dark/flat stacking",   ("frame_added",)),
    ("Master dark creation", ("mdark_begin", "mdark_group")),
    ("Master flat creation", ("mflat_begin", "mflat_group")),
    ("Debayer / OSC",        ("debayer",)),
    ("Measurement",          ("frame_stat", "measure_done")),
    ("Sub rejection",        ("sub_reject_one", "sub_reject_sum")),
    ("Pixel-scale normalize", ("pixscale_norm", "target_scale")),
    ("Background extraction", ("abe", "abe_config", "abe_done")),
    ("Star alignment",       ("align_model", "aa_delta")),
    ("Meridian flip",        ("flip", "flip_plain")),
    ("Skip registration",    ("skip_reg", "ext_prereg")),
    ("Registration sets",    ("sets_assigned",)),
    ("Drizzle",              ("drizzle_on",)),
    ("MFDeconv",             ("mfdeconv",)),
    ("GPU acceleration",     ("__re__", re.compile(r"\[GPU\]|GPU backend|GPU,"))),
]


# ── parser ──────────────────────────────────────────────────────────────────
def _read_text(path: str) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as fh:
                return fh.read()
        except (UnicodeDecodeError, LookupError):
            continue
        except OSError:
            raise
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def parse_stacking_log(path: str, *, keep_transient: bool = False
                       ) -> StackingRunReport:
    """Parse one stacking log into a :class:`StackingRunReport`.

    Never raises on malformed content: a line that defeats its own handler is
    recorded in ``report.parse_errors`` and parsing continues.
    """
    rep = StackingRunReport(path=path)
    try:
        text = _read_text(path)
    except OSError as e:
        rep.parse_errors.append(f"could not read file: {e}")
        return rep

    t0 = None
    prev_abs = None
    day = 0
    last_t = None
    last_clock = ""
    matched_ids = set()

    for lineno, raw in enumerate(text.split("\n"), start=1):
        line = raw.rstrip("\n")
        if not line.strip():
            continue

        # framed header block
        m = _HDR.match(line)
        if m:
            key, val = m.group(1).strip(), m.group(2).strip()
            kl = key.lower()
            if kl.startswith("start"):
                rep.started = val
            elif kl.startswith("director"):
                rep.directory = val
            elif kl.startswith("log closed"):
                rep.closed = val
                rep.complete = True
            else:
                rep.facts.setdefault(key, val)
            continue
        if line.startswith("#"):
            continue

        transient = line.startswith("\r") or "\r" in line[:3]
        body = line.lstrip("\r")

        m = _TS.match(body)
        if m:
            h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
            abs_s = h * 3600 + mi * 60 + s
            if prev_abs is not None and abs_s + 1 < prev_abs - 3600:
                day += 1                      # crossed midnight
            prev_abs = abs_s
            abs_s += day * 86400
            if t0 is None:
                t0 = abs_s
            t = float(abs_s - t0)
            clock = f"{m.group(1).zfill(2)}:{m.group(2)}:{m.group(3)}"
            txt = m.group(4)
            rep.n_stamped += 1
            last_t, last_clock = t, clock
        else:
            # Continuation / unstamped line (the integration summary block is
            # written this way). Attach it to the last known time so it still
            # places on the timeline.
            t, clock, txt = last_t, last_clock, body

        if not (txt or "").strip():
            # A timestamped line with no body carries nothing; counting it as
            # unrecognised would understate coverage.
            rep.n_lines += 1
            rep.counters["blank"] += 1
            continue

        if transient and not keep_transient:
            # progress-overwrite chatter; count it but don't retain
            rep.n_lines += 1
            rep.counters["transient"] += 1
            continue

        ll = LogLine(idx=len(rep.lines), lineno=lineno, t=t, clock=clock,
                     text=txt, severity=_severity(txt), transient=transient)
        rep.lines.append(ll)
        rep.n_lines += 1

        # classify + extract
        hit = None
        for rid, rx in RULES:
            mm = rx.search(txt)
            if mm:
                hit = (rid, mm)
                break
        if hit is None:
            rep.unmatched_shapes[_shape(txt)] += 1
        else:
            rid, mm = hit
            ll.rule = rid
            sev = SEVERITY_OVERRIDES.get(rid)
            if sev is not None:
                ll.severity = sev
            rep.n_matched += 1
            # Every matched line is counted under its rule id. A handler must
            # therefore NOT also increment counters[<its own rule id>] -- that
            # double-counts. Use a distinct key when a handler needs its own
            # tally.
            rep.counters[rid] += 1
            matched_ids.add(rid)
            try:
                _apply(rep, rid, mm, ll)
            except Exception as e:           # never let one line kill the run
                rep.parse_errors.append(f"line {lineno} [{rid}]: {e!r}")

        if ll.severity in (SEV_ERROR, SEV_WARN):
            rep.problems.append(Problem(t=t, clock=clock, text=txt,
                                        severity=ll.severity, lineno=lineno))

    # close the last open phase
    end_t = max((l.t for l in rep.lines if l.t is not None), default=None)
    for ph in rep.phases:
        if ph.t_end is None:
            ph.t_end = end_t

    rep.modes = _detect_modes(rep, matched_ids, text)
    return rep


def _open_phase(rep: StackingRunReport, name: str, t) -> None:
    """Start a phase, closing whatever was still open."""
    for ph in rep.phases:
        if ph.t_end is None:
            ph.t_end = t
    rep.phases.append(Phase(name=name, t_start=(0.0 if t is None else t)))


def _close_phase(rep: StackingRunReport, name: str, t) -> None:
    """Close the newest open phase, preferring one whose name matches."""
    for ph in reversed(rep.phases):
        if ph.t_end is None and ph.name.lower() == (name or "").lower():
            ph.t_end = t
            return
    for ph in reversed(rep.phases):
        if ph.t_end is None:
            ph.t_end = t
            return


def _apply(rep: StackingRunReport, rid: str, m, ll: LogLine) -> None:
    """Fold one matched line into the report."""
    g = m.groupdict() if m.groupdict() else {}

    # ---- facts (single-value readouts) ----
    if rid == "preflight":
        rep.facts["Worker environment"] = g.get("env", "")
    elif rid == "precision":
        rep.facts["Internal precision"] = g.get("v", "")
    elif rid == "accel":
        rep.facts["Hardware acceleration"] = g.get("v", "")
    elif rid == "align_model":
        rep.facts["Alignment model"] = g.get("v", "")
    elif rid == "weights_mode":
        rep.facts["Weighting mode"] = g.get("mode", "")
    elif rid == "binning":
        rep.facts["Binning target"] = g.get("v", "")
        rep.facts["Binning observed"] = g.get("range", "")
    elif rid == "target_scale":
        rep.facts["Target pixel scale"] = g.get("v", "")
    elif rid == "ref_scale":
        rep.facts["Reference pixel scale"] = (m.group(1) or "") + '"/px'
    elif rid == "ref_pa":
        rep.facts["Reference PA"] = (m.group(1) or "") + "°"
    elif rid == "canvas_change":
        rep.facts["Normalize canvas"] = f"{g.get('from')} → {g.get('to')}"
    elif rid == "bin_resample":
        rep.facts["Binning resample"] = f"{g.get('from')} → {g.get('to')}"
    elif rid in ("ref_select", "ref_select2", "ref_from_stamp", "ref_borderless"):
        rep.facts["Reference frame"] = (g.get("name") or "").strip()
        if rid == "ref_select":
            try:
                rep.facts["Reference score"] = m.group(2)
            except Exception:
                pass
    elif rid == "ref_align":
        rep.facts["Reference (alignment)"] = os.path.basename(
            (g.get("path") or "").strip())
    elif rid == "transform_saved":
        rep.facts["Transform file"] = os.path.basename((g.get("name") or "").strip())
    elif rid == "sets_assigned":
        rep.facts["Registration sets"] = g.get("v", "")
    elif rid == "free_disk":
        val = f"{m.group(1)} {g.get('unit') or 'GB'}"
        rep.facts.setdefault("Free disk at start", val)
        rep.facts["Free disk (last seen)"] = val
    elif rid == "log_opened":
        rep.facts.setdefault("Log path", (g.get("path") or "").strip())
    elif rid == "weights_global":
        rep.facts["Frame weights"] = (f"min={m.group(1)} max={m.group(2)} "
                                      f"mean={m.group(3)}")
    elif rid == "align_summary":
        rep.facts["Alignment summary"] = (f"{g.get('ok')} succeeded, "
                                          f"{g.get('bad')} rejected")
        rep.counters["aligned_ok"] = _i(g.get("ok")) or 0
        rep.counters["aligned_bad"] = _i(g.get("bad")) or 0
    elif rid == "reg_complete":
        rep.facts["Registration result"] = f"{g.get('ok')}/{g.get('tot')} valid"
        rep.counters["reg_valid"] = _i(g.get("ok")) or 0
        rep.counters["reg_total"] = _i(g.get("tot")) or 0
    elif rid == "tree_snapshot":
        rep.counters["tree_groups"] = _i(g.get("groups")) or 0
        rep.counters["tree_leaves"] = _i(g.get("leaves")) or 0
        rep.counters["tree_kept"] = _i(g.get("kept")) or 0
    elif rid == "total_frames":
        rep.counters["frames_found"] = _i(g.get("n")) or 0
    elif rid == "pipeline_start":
        rep.counters["frames_pipeline"] = _i(g.get("n")) or 0
        # Light calibration has no "Phase:" marker of its own, so without a
        # synthetic phase here its minutes land in the unexplained block
        # before the first marker (22% of a real 70-minute run).
        _open_phase(rep, "Calibration", ll.t)
    elif rid == "measure_done":
        rep.counters["frames_measured"] = _i(g.get("n")) or 0
    elif rid == "summary_headers":
        rep.counters["frames_summary"] = _i(g.get("n")) or 0
    elif rid == "rej_prepass_all":
        rep.counters["prepass_groups"] = _i(g.get("g")) or 0
        rep.counters["prepass_frames"] = _i(g.get("n")) or 0
    elif rid == "sub_reject_sum":
        rep.counters["subs_rejected"] = _i(g.get("n")) or 0
        rep.counters["subs_total"] = _i(g.get("tot")) or 0
    elif rid == "no_outliers":
        rep.counters["subs_rejected"] = 0
        rep.counters["subs_total"] = _i(g.get("n")) or 0
    elif rid == "masters_opened":
        rep.counters["masters_opened"] = _i(g.get("n")) or 0

    # ---- phases ----
    elif rid == "phase_start":
        _open_phase(rep, (g.get("name") or "").strip(), ll.t)
    elif rid == "phase_done":
        _close_phase(rep, (g.get("name") or "").strip().rstrip("."), ll.t)

    # ---- groups ----
    elif rid == "group_files":
        rep.group(g["g"]).files_after_filter = _i(g.get("n"))
    elif rid == "group_listed":
        rep.group((g.get("g") or "").strip()).files_after_filter = _i(g.get("n"))
    elif rid == "rej_prepass_one":
        gi = rep.group(g["g"])
        gi.prepass_files = _i(g.get("files"))
    elif rid in ("prepass_result", "prepass_soft", "prepass_plain"):
        gi = rep.group(g["g"])
        gi.prepass_canvas = (g.get("canvas") or "").replace(" ", "")
        gi.prepass_total = _i(g.get("tot"))
        # "N frame(s)" with no split means all of them contributed
        gi.prepass_contributing = (_i(g.get("contrib"))
                                   if g.get("contrib") is not None
                                   else gi.prepass_total)
        gi.soft_rejected = _i(g.get("soft"))
        gi.prepass_seconds = _f(m.group(m.lastindex))
    elif rid == "group_stacked":
        rep.group((g.get("g") or "").strip()).stacked = _i(g.get("n"))
    elif rid == "int_algo":
        rep.group(g["g"]).algorithm = (g.get("algo") or "").strip()
    elif rid == "int_start":
        gi = rep.group(g["g"])
        gi.integration_started = ll.t
        rep._cur_int_group = g["g"]
        if gi.files_after_filter is None:
            gi.files_after_filter = _i(g.get("n"))
    elif rid == "int_done":
        rep.group(g["g"]).integration_done = ll.t
        rep._cur_int_group = ""
    elif rid == "int_tiles":
        rep.counters["tiles_total"] += _i(g.get("n")) or 0
        if rep._cur_int_group:
            rep.group(rep._cur_int_group).tiles = _i(g.get("n"))
    elif rid == "weights_group":
        gi = rep.group(g["g"])
        gi.weight_min, gi.weight_max, gi.weight_mean = (
            _f(m.group(2)), _f(m.group(3)), _f(m.group(4)))

    # ---- per-frame records ----
    elif rid == "frame_stat":
        rep.frame_stats.append(FrameStat(
            name=g.get("name") or "",
            star_count=_f(g.get("stars")), ecc=_f(g.get("ecc")),
            bg=_f(g.get("bg")), fwhm=_f(g.get("fwhm")),
            sharp=_f(g.get("sharp")), snr=_f(g.get("snr")),
            exp_scale=_f(g.get("exps")), weight=_f(g.get("w")),
            normalized=_f(g.get("norm")),
        ))
    elif rid == "sub_reject_one":
        rep.rejected_subs.append(((g.get("name") or ""), _f(m.group(m.lastindex))))
    elif rid == "light_added":
        rep.lights_added.append(g.get("name") or "")
    elif rid == "master_dark":
        rep.masters_dark.append((os.path.basename((g.get("path") or "").strip()),
                                 (g.get("key") or "").strip()))
    elif rid == "master_flat":
        rep.masters_flat.append((os.path.basename((g.get("path") or "").strip()),
                                 (g.get("key") or "").strip()))
    elif rid == "calib_pair":
        rep.calib_pairs.append((g.get("frame") or "", g.get("dark") or "",
                                g.get("flat") or "", g.get("cos") or ""))
    elif rid == "calib_saved":
        rep.counters["frames_calibrated"] = max(
            rep.counters.get("frames_calibrated", 0), _i(g.get("i")) or 0)
        rep.counters["frames_calibrated_total"] = _i(g.get("n")) or 0
    elif rid == "flip":
        rep.flips.append((g.get("name") or "", _i(g.get("deg"))))
    elif rid == "aa_delta":
        rep.align_deltas.append((g.get("name") or "",
                                 _f(m.group(2)), _f(m.group(3))))
    elif rid == "wrote_frame":
        rep.written_frames.append(g.get("name") or "")
    elif rid == "finalize_time":
        rep.counters["finalize_seconds"] = int(_f(m.group(1)) or 0)

    # ---- round 2: per-frame failures ----
    elif rid in ("aa_failed", "aa_timeout", "finalize_error", "gpu_calib_fail"):
        rep.failed_frames.append(((g.get("name") or "").strip(),
                                  (g.get("why") or "").strip()[:200]))
    elif rid == "failed_bare":
        # A bare "X <file>" line: the companion to a preceding reason line.
        # Record it only if we have not already logged that frame, so the
        # pair does not double-count in the Problems panel.
        nm = (g.get("name") or "").strip()
        if not any(nm == f[0] for f in rep.failed_frames):
            rep.failed_frames.append((nm, ""))
    elif rid == "align_skipped":
        rep.skipped_frames.append(((g.get("name") or "").strip(),
                                   (g.get("why") or "").strip()[:200]))

    # ---- round 2: calibration detail ----
    elif rid == "frame_added":
        kind = (g.get("kind") or "").upper()
        rep.counters[f"added_{kind.lower()}"] += 1
    elif rid == "cosmetic_bayer":
        rep.facts.setdefault("Cosmetic correction", f"Bayer {g.get('pat')}")
    elif rid == "cosmetic_mono":
        rep.facts.setdefault("Cosmetic correction", g.get("how") or "yes")
    elif rid == "sat_applied":
        rep.facts.setdefault("Satellite trail removal",
                             g.get("how") or "applied")
    elif rid == "mask_saved":
        rep.counters["masks_saved"] += 1
        rep.counters["mask_rejected_px"] += _i(g.get("n")) or 0
    elif rid == "filter_target":
        rep.facts[f"Norm target [{g.get('f')}]"] = (g.get("name") or "").strip()

    # ---- round 2: background extraction ----
    elif rid == "abe_config":
        rep.facts["Gradient removal"] = (g.get("cfg") or "").strip()[:140]
    elif rid == "abe_done":
        rep.counters["abe_ok"] = _i(g.get("ok")) or 0
        rep.counters["abe_skipped"] = _i(g.get("skip")) or 0
        rep.counters["abe_errors"] = _i(g.get("err")) or 0

    # ---- round 3 ----
    elif rid == "master_saved":
        rep.group(g["g"]).master_path = (g.get("path") or "").strip()
    elif rid == "weight_mode2":
        rep.facts.setdefault("Weighting mode", (g.get("mode") or "").strip())
    elif rid == "gpu_transfer":
        rep.facts["GPU transfer mode"] = (g.get("v") or "").strip()[:120]
    elif rid == "auto_thresh2":
        rep.facts["Auto threshold"] = (g.get("v") or "").strip()
    elif rid == "chunk_size":
        rep.facts.setdefault("Integration chunk", (g.get("v") or "").strip())
    elif rid == "rowband":
        rep.facts["Row-band mode"] = f"{g.get('rows')} rows"
    elif rid == "finalize_cfg":
        rep.facts["Finalize workers"] = f"{g.get('w')} x {g.get('t')} cv2 threads"
    elif rid == "disk_estimate":
        rep.facts[f"Disk estimate: {(g.get('what') or '').strip()}"] = \
            (m.group(m.lastindex) or "") + " GB"
    elif rid == "dark_loaded":
        rep.counters["darks_loaded"] += 1
    elif rid == "flat_loaded":
        rep.counters["flats_loaded"] += 1

    # ---- round 4: master creation ----
    elif rid == "mdark_begin":
        _open_phase(rep, "Master Dark Creation", ll.t)
    elif rid == "mflat_begin":
        _open_phase(rep, "Master Flat Creation", ll.t)
    elif rid == "calib_complete":
        _close_phase(rep, "Calibration", ll.t)
    elif rid == "master_discover":
        what = (g.get("what") or "").replace(" groups", "").replace(
            "grouped exposures", "dark")
        rep.facts[f"Discovered ({what})"] = (
            f"{g.get('n')} group(s), {g.get('elig')} eligible, "
            f"{g.get('files')} file(s)")
    elif rid == "cal_group_line":
        rep.cal_groups.append(((g.get("g") or "").strip(),
                               (g.get("sess") or "").strip(),
                               _i(g.get("n")), (g.get("dur") or "").strip()))
    elif rid in ("mdark_group", "mflat_group"):
        mb = MasterBuild(
            kind=("dark" if rid == "mdark_group" else "flat"),
            key=(g.get("key") or "").strip(),
            frames=_i(g.get("n")),
            session=(g.get("sess") or "").strip(),
            temp=((g.get("temp") or "").strip() + "C"
                  if g.get("temp") else ""),
            t_start=ll.t,
        )
        km = re.search(r"\((\d+\s*[x\u00d7]\s*\d+)\)", mb.key)
        if km:
            mb.canvas = km.group(1).replace(" ", "")
        rep.master_builds.append(mb)
        rep._cur_master = mb
    elif rid in ("master_method", "flat_reducer"):
        mb = rep._cur_master
        if mb is not None and not mb.method:
            mb.method = (g.get("m") or "").strip()
        else:
            rep.facts.setdefault("Master reducer", (g.get("m") or "").strip())
    elif rid == "flat_dark":
        if rep._cur_master is not None:
            rep._cur_master.dark_used = os.path.basename(
                (g.get("name") or "").strip())
    elif rid == "group_tiles":
        if rep._cur_master is not None:
            rep._cur_master.tiles = _i(g.get("n"))
    elif rid in ("mdark_saved", "mflat_saved"):
        path = (g.get("path") or "").strip()
        mb = rep._cur_master
        if mb is not None and not mb.path:
            mb.path = path
            mb.t_end = ll.t
        elif mb is not None and mb.t_end is None:
            mb.t_end = ll.t
        rep.counters["masters_built"] = len(
            [b for b in rep.master_builds if b.path])
    elif rid == "total_tiles":
        rep.facts["Total tiles"] = g.get("n") or ""
    elif rid == "preload_darks":
        rep.facts["Master darks pre-loaded"] = g.get("n") or ""
    elif rid == "int_timing":
        # The timing line is emitted just BEFORE "Integration complete", so it
        # describes the group currently being integrated -- the one opened by
        # int_start -- not the last one that finished. Attaching it to the
        # finished group shifts every timing by one group, which silently
        # mislabels the slowest group in the run.
        gi = rep.groups.get(rep._cur_int_group) if rep._cur_int_group else None
        if gi is None:
            # Fall back to the FIRST group still untimed, in log order.
            for cand in rep.groups.values():
                if cand.t_reduce is None:
                    gi = cand
                    break
        if gi is not None:
            gi.t_read = _f(m.group(1))
            gi.t_upload = _f(m.group(2))
            gi.t_reduce = _f(m.group(3))
            gi.t_download = _f(m.group(4))
    elif rid == "filter_reject":
        rep.filter_rejections.append((
            (g.get("filt") or "").strip(), _i(g.get("n")), _i(g.get("tot")),
            _f(g.get("thr")), _f(g.get("med")), _f(g.get("sig"))))
    elif rid == "finalize_rejected":
        rep.counters["finalize_rejected"] = _i(g.get("n")) or 0
    elif rid in ("exposure_scaling", "exposure_scaling2"):
        v = g.get("state") or ""
        if g.get("n") is not None:
            v += f" \u2014 {g.get('n')}/{g.get('tot')} frame(s) down-weighted"
        rep.facts["Exposure scaling"] = v
    elif rid == "worker_count":
        rep.facts["Worker count"] = f"{g.get('from')} \u2192 {g.get('to')}"
    elif rid == "geom_audit_src":
        rep.facts.setdefault("Geometry audit sources", "")
        prev = rep.facts["Geometry audit sources"]
        cur = (g.get("v") or "").strip()
        if cur not in prev:
            rep.facts["Geometry audit sources"] = (
                f"{prev}; {cur}" if prev else cur)[:300]
    elif rid == "geom_audit_bad":
        rep.failed_frames.append((
            (g.get("name") or "").strip(),
            f"off the reference grid ({(g.get('hw') or '').replace(' ', '')})"))
    elif rid == "drizzle_cfg":
        rep.facts["Drizzle"] = (f"scale {g.get('scale')}x, pixfrac "
                                f"{g.get('pf')}, kernel {g.get('kern')}")
    elif rid == "drizzle_group":
        gi = rep.group(g["g"])
        gi.drizzle_scale = _f(g.get("scale"))
        gi.drizzle_pixfrac = _f(g.get("drop"))
        gi.drizzle_frames = _i(g.get("n"))
        rep._cur_drizzle_group = g["g"]
    elif rid == "drizzle_stack":
        rep._cur_drizzle_group = g["g"]
    elif rid == "drizzle_kern_use":
        if rep._cur_drizzle_group:
            rep.group(rep._cur_drizzle_group).drizzle_kernel = (
                g.get("kern") or "")
    elif rid == "drizzle_bbox":
        # The drizzled canvas is only stated here, on the first frame of
        # each group, so it has to be attached to whichever group is
        # currently drizzling rather than looked up by name.
        if rep._cur_drizzle_group:
            gi = rep.group(rep._cur_drizzle_group)
            gi.drizzle_canvas = (g.get("canvas") or "").replace(" ", "")
            if gi.drizzle_scale is None:
                gi.drizzle_scale = _f(g.get("scale"))
    elif rid == "drizzle_saved":
        if rep._cur_drizzle_group:
            rep.group(rep._cur_drizzle_group).drizzle_path = (
                g.get("path") or "").strip()
            rep._cur_drizzle_group = ""
    elif rid == "drizzle_master_line":
        gi = rep.group((g.get("g") or "").strip())
        if not gi.drizzle_path:
            gi.drizzle_path = (g.get("path") or "").strip()
    elif rid == "excluded_frames":
        rep.counters["frames_excluded"] = _i(g.get("n")) or 0
    elif rid == "reg_twins":
        rep.facts["Registered twins"] = f"{g.get('ok')}/{g.get('tot')} usable"
    elif rid == "sasd_loaded":
        rep.facts["SASD transforms loaded"] = g.get("n") or ""
    elif rid == "int_cancelled":
        rep.group(g["g"]).integration_done = ll.t
        rep.counters["groups_cancelled"] += 1
    elif rid in ("cancel_request", "cancelled_user", "stopped_user"):
        rep.facts["Run outcome"] = "cancelled by user"
    elif rid == "plate_solve_go":
        rep.facts.setdefault("Plate solve chain", (g.get("chain") or "").strip())
    elif rid == "plate_solve_ok":
        gi = rep.group(g["g"])
        gi.plate_solved = True
        gi.plate_solve_note = ""
    elif rid == "plate_solve_bad":
        gi = rep.group(g["g"])
        gi.plate_solved = False
        gi.plate_solve_note = (g.get("why") or "").strip()[:300]
    elif rid == "cal_precheck":
        rep.facts["Calibration twins"] = f"{g.get('ok')}/{g.get('tot')} usable"
        rep.counters["cal_twins_ok"] = _i(g.get("ok")) or 0
        rep.counters["cal_twins_total"] = _i(g.get("tot")) or 0


def _detect_modes(rep: StackingRunReport, matched_ids: set, text: str) -> list:
    out = []
    for label, spec in MODE_RULES:
        hit = False
        for s in spec:
            if s == "__re__":
                continue
            if isinstance(s, str):
                if s in matched_ids:
                    hit = True
                    break
        if not hit and "__re__" in spec:
            rx = spec[spec.index("__re__") + 1]
            hit = bool(rx.search(text))
        if hit:
            out.append(label)
    return out


# ── log discovery ───────────────────────────────────────────────────────────
def find_stacking_logs(directory: str) -> list:
    """Stacking logs in *directory*, newest first, as (path, mtime, size)."""
    out = []
    try:
        for f in os.listdir(directory or ""):
            if not f.lower().endswith(".txt"):
                continue
            if "stacking_log" not in f.lower():
                continue
            p = os.path.join(directory, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            out.append((p, st.st_mtime, st.st_size))
    except OSError:
        return []
    out.sort(key=lambda r: r[1], reverse=True)
    return out


def frame_reconciliation(rep) -> list:
    """Follow the frame count through the pipeline and name what fell out.

    A run that silently loses frames between two stages is the failure mode
    worth catching: the count difference is visible immediately, whereas the
    reason is one line somewhere in eleven thousand. Each returned row is
    ``(stage, count, delta, note)`` where delta compares to the previous known
    stage and note explains a loss when it can.
    """
    c = rep.counters
    chain = [
        ("Frames found",        c.get("frames_found")),
        ("Calibrated",          c.get("frames_calibrated_total")),
        ("Measured",            c.get("frames_measured")),
        ("Kept in tree",        c.get("tree_kept")),
        ("Registered (valid)",  c.get("reg_valid")),
        ("Prepass input",       c.get("prepass_frames")),
    ]
    stacked = sum(g.stacked for g in rep.groups.values()
                  if g.stacked is not None)
    if stacked:
        chain.append(("Stacked", stacked))

    rows = []
    prev_label, prev = None, None
    for label, n in chain:
        if n is None:
            continue
        delta = None if prev is None else (n - prev)
        note = ""
        if delta is not None and delta < 0:
            lost = -delta
            note = f"{lost} frame(s) lost since {prev_label}"
            names = [f[0] for f in rep.failed_frames] +                     [f[0] for f in rep.skipped_frames]
            # de-dup while preserving order
            seen, uniq = set(), []
            for nm in names:
                if nm and nm not in seen:
                    seen.add(nm)
                    uniq.append(nm)
            if uniq:
                shown = ", ".join(os.path.basename(x) for x in uniq[:3])
                more = f" (+{len(uniq)-3} more)" if len(uniq) > 3 else ""
                note += f": {shown}{more}"
        elif delta is not None and delta > 0:
            note = f"+{delta} vs {prev_label}"
        rows.append((label, n, delta, note))
        prev_label, prev = label, n

    if rep.counters.get("subs_rejected"):
        rows.append(("Subs rejected (score outlier)",
                     rep.counters["subs_rejected"], None,
                     ", ".join(os.path.basename(n)
                               for n, _s in rep.rejected_subs[:3])))
    return rows


def master_outputs(rep) -> list:
    """Every master this run produced, as MasterBuild rows, in time order.

    Master darks and flats are built by their own pipeline and already arrive
    as MasterBuild records. A master LIGHT is the output of integrating a
    group, so its facts live on the GroupInfo instead -- but from the user's
    side it is the same kind of thing: an output file with inputs, a method
    and a duration. Synthesising a row for it keeps "what did this run
    produce?" answerable in one place.
    """
    out = list(rep.master_builds)
    for name, gi in (rep.groups or {}).items():
        if not gi.master_path:
            continue
        out.append(MasterBuild(
            kind="light",
            key=name,
            frames=gi.stacked if gi.stacked is not None else gi.prepass_contributing,
            canvas=gi.prepass_canvas,
            method=gi.algorithm,
            tiles=gi.tiles,
            t_start=gi.integration_started,
            t_end=gi.integration_done,
            path=gi.master_path,
            solved=gi.plate_solved,
            solve_note=gi.plate_solve_note,
        ))
        if gi.drizzle_path:
            # Drizzle writes a SECOND master per group at its own scale;
            # it is a distinct output file and deserves its own row.
            out.append(MasterBuild(
                kind="drizzle",
                key=name,
                frames=gi.drizzle_frames,
                canvas=gi.drizzle_canvas,
                method=" ".join(x for x in (
                    gi.drizzle_kernel,
                    (f"{gi.drizzle_scale:g}x" if gi.drizzle_scale else ""),
                    (f"pixfrac {gi.drizzle_pixfrac:g}"
                     if gi.drizzle_pixfrac else "")) if x),
                t_start=gi.integration_started,
                t_end=gi.integration_done,
                path=gi.drizzle_path,
                solved=gi.plate_solved,
            ))
    out.sort(key=lambda b: (b.t_start is None, b.t_start or 0.0))
    return out


def slowest_steps(rep, limit: int = 12) -> list:
    """Largest gaps between consecutive log lines, as (gap_s, clock, text).

    The log has no explicit step durations outside the phase markers, so the
    silence before a line is the best available measure of what it cost.
    """
    out = []
    lines = [l for l in rep.lines if l.t is not None]
    for i in range(len(lines) - 1):
        gap = lines[i + 1].t - lines[i].t
        if gap > 0:
            out.append((gap, lines[i].clock, lines[i].text))
    out.sort(key=lambda r: r[0], reverse=True)
    return out[:limit]


def fmt_hms(seconds) -> str:
    """Seconds -> compact H:MM:SS / M:SS, '—' when unknown."""
    if seconds is None:
        return "—"
    try:
        s = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "—"
    sign = "-" if s < 0 else ""
    s = abs(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{sign}{h}:{m:02d}:{sec:02d}"
    return f"{sign}{m}:{sec:02d}"
