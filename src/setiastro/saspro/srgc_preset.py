# saspro/srgc_preset.py — headless Survey Reference Gradient Correction
# -----------------------------------------------------------------------------
# The replay/bundle half of survey_gradient.py: run SRGC with no dialog, from a
# dropped shortcut, a function bundle, or ctx.run_command('srgc', {...}).
#
# WHY A SURVEY PRESET IS SERIALISABLE AND A USER REFERENCE IS NOT
# ---------------------------------------------------------------
# A preset has to mean the same thing when it is replayed next week on another
# image. A survey name does: it names a published dataset that is fetched by
# the target's own sky coordinates, so the same preset on a different frame
# correctly pulls a different patch of sky. A user-supplied reference does not
# — it is one array in one session, and there is no name a later run could
# resolve back to it. So the dialog's grip records the SURVEY settings even
# when "my own image" is selected, and says so in its docstring rather than
# quietly emitting a shortcut that would do something different.
#
# NETWORK, AND WHY THAT IS FINE HERE
# ----------------------------------
# This is the one headless command in SASpro that needs the internet. The
# connect pre-flight in survey_gradient caps a dead host at CONNECT_TIMEOUT
# rather than the OS's ~21 s, so a drop onto an image while offline fails in
# seconds with a real message instead of hanging the UI.
# -----------------------------------------------------------------------------
from __future__ import annotations

import numpy as np

from .survey_gradient import (
    SURVEYS,
    DEFAULT_SURVEY,
    SurveyFetchError,
    survey_gradient_model,
    apply_survey_gradient,
    reset_dead_hosts,
)


def _doc_image_float01(doc) -> np.ndarray | None:
    """The document's pixels as float32, scaled to [0,1] if they were integer."""
    img = getattr(doc, "image", None)
    if img is None:
        return None
    a = np.asarray(img)
    if a.size == 0:
        return None
    if np.issubdtype(a.dtype, np.integer):
        return a.astype(np.float32) / float(np.iinfo(a.dtype).max)
    return a.astype(np.float32, copy=False)


def _header_of(doc):
    meta = getattr(doc, "metadata", None) or {}
    return (meta.get("wcs_header") or meta.get("original_header")
            or meta.get("header"))


def _channels_for(name: str) -> list:
    """Channel spec for a survey name, falling back to the default."""
    info = SURVEYS.get(str(name)) or SURVEYS[DEFAULT_SURVEY]
    return list(info.get("channels") or [])


def srgc_params_from_preset(preset: dict | None = None) -> dict:
    """
    Normalise and clamp a preset into the full parameter set.

    Every bound here mirrors the dialog's own spin-box ranges, so a
    hand-written preset cannot ask for something the UI would refuse.
    """
    p = dict(preset or {})
    survey = str(p.get("survey") or p.get("name") or DEFAULT_SURVEY)
    if survey not in SURVEYS:
        survey = DEFAULT_SURVEY
    return {
        "survey":         survey,
        "work_long_side": int(np.clip(int(p.get("work_long_side", 1024)), 256, 4096)),
        "degree":         int(np.clip(int(p.get("degree", 1)), 0, 4)),
        "rbf":            bool(p.get("rbf", True)),
        "samples":        int(np.clip(int(p.get("samples", 200)), 20, 4000)),
        "patch":          int(np.clip(int(p.get("patch", 15)), 5, 151)),
        "exclude":        int(np.clip(int(p.get("exclude", 50)), 0, 95)),
        "protect_k":      float(np.clip(float(p.get("protect_k", 3.0)), 0.5, 20.0)),
        # 0-150: above 100 deliberately over-corrects (see the dialog).
        "strength":       int(np.clip(int(p.get("strength", 100)), 0, 150)),
    }


def apply_srgc_via_preset(main_window, doc, preset: dict | None = None):
    """
    Run SRGC headlessly on `doc` and commit the result.

    Raises SurveyFetchError with a message worth showing when the field is not
    plate-solved, the survey cannot be reached, or the reference carries no
    structure. The caller decides how to surface that — the drop handler shows
    a message box, a bundle run logs it.
    """
    params = srgc_params_from_preset(preset)

    img = _doc_image_float01(doc)
    if img is None:
        raise SurveyFetchError("Target document has no image data.")
    hdr = _header_of(doc)
    if hdr is None:
        raise SurveyFetchError(
            "This image has no WCS. SRGC fetches its reference by sky "
            "coordinates, so plate-solve it first.\n\n"
            "Open the SRGC dialog if you want it to offer to solve for you — "
            "a blind solve can take minutes and is never started from a "
            "dropped preset without asking."
        )

    # A host that was down for an earlier drop may be up now, and a dropped
    # preset is a fresh user gesture, same as reopening the dialog.
    reset_dead_hosts()

    log = getattr(main_window, "_log", None)

    def _progress(msg: str):
        if callable(log):
            try:
                log(f"[SRGC] {msg}")
            except Exception:
                pass

    res = survey_gradient_model(
        img, hdr,
        channels=_channels_for(params["survey"]),
        work_long_side=params["work_long_side"],
        degree=params["degree"],
        use_rbf=params["rbf"],
        num_samples=params["samples"],
        patch_size=params["patch"],
        exclusion_fraction=params["exclude"] / 100.0,
        protect_k=params["protect_k"],
        progress_cb=_progress,
    )

    strength = params["strength"] / 100.0
    out = apply_survey_gradient(img, res["gradient"], strength)

    step = f"Survey Gradient ({params['survey']}, {params['strength']}%)"
    meta = {
        "step_name": "SurveyGradient",
        "survey_gradient": {
            "survey": params["survey"],
            "channels": [c["hips"] if isinstance(c, dict) else c
                         for c in _channels_for(params["survey"])],
            "work_long_side": params["work_long_side"],
            "degree": params["degree"],
            "rbf": params["rbf"],
            "strength": strength,
            "coverage": float(res["coverage"]),
            "coeffs": [[float(a), float(b)] for a, b in res["coeffs"]],
            "headless": True,
        },
    }
    doc.apply_edit(out.astype(np.float32, copy=False),
                   step_name=step, metadata=meta)

    # Remember for Replay, exactly as abe_preset does.
    try:
        remember = getattr(main_window, "remember_last_headless_command", None)
        if remember is None:
            remember = getattr(main_window, "_remember_last_headless_command", None)
        if callable(remember):
            remember("srgc", dict(params),
                     description="Survey Reference Gradient Correction")
            if callable(log):
                try:
                    log(f"[Replay] SRGC headless stored: command_id='srgc', "
                        f"preset_keys={list(params.keys())}")
                except Exception:
                    pass
    except Exception:
        # Never let bookkeeping sink a correction that already applied.
        pass

    share = (res.get("diag") or {}).get("reference_share", 0.0)
    if callable(log):
        try:
            log(f"[SRGC] {step} coverage={res['coverage'] * 100:.1f}%  "
                f"reference explains {share * 100:.0f}% of the gradient")
        except Exception:
            pass
    return res


def open_srgc_with_preset(main_window, preset: dict | None = None):
    """
    Open the SRGC dialog with a preset already applied to its controls.

    Used when a shortcut is double-clicked rather than dropped on an image —
    the user wants to see and adjust the settings, not fire them.
    """
    from .survey_gradient import open_survey_gradient

    doc = None
    for attr in ("docman", "doc_manager"):
        dm = getattr(main_window, attr, None)
        if dm is not None:
            try:
                doc = dm.get_active_document()
            except Exception:
                doc = None
            if doc is not None:
                break
    if doc is None or getattr(doc, "image", None) is None:
        raise SurveyFetchError("Open an image first.")

    dlg = open_survey_gradient(main_window, doc)
    p = srgc_params_from_preset(preset)
    try:
        if p["survey"] in SURVEYS:
            dlg.cmb_survey.setCurrentText(p["survey"])
        dlg.sp_work.setValue(p["work_long_side"])
        dlg.sp_degree.setValue(p["degree"])
        dlg.chk_rbf.setChecked(p["rbf"])
        dlg.sp_samples.setValue(p["samples"])
        dlg.sp_patch.setValue(p["patch"])
        dlg.sp_exclude.setValue(p["exclude"])
        dlg.sp_protect.setValue(p["protect_k"])
        dlg.sp_strength.setValue(p["strength"])
    except Exception:
        pass
    return dlg
