# saspro/hdr_composition_preset.py — headless HDR Composition
# -----------------------------------------------------------------------------
# Replay / shortcut / command-drop half of hdr_composition.py. Runs with no
# dialog from a dropped shortcut, a function bundle, or
# ctx.run_command('hdr_composition', {...}).
#
# Multi-view resolution mirrors Image Combine: an ordered list of titles (and
# optionally session-local doc_ids) names the frames longest→shortest. When a
# drop targets a view and the preset has no list, that view is taken as the
# longest exposure and the remaining open views are filled by exposure time.
# -----------------------------------------------------------------------------
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QLineEdit, QMessageBox, QSpinBox, QVBoxLayout,
)

from setiastro.saspro.hdr_composition import (
    HDRCompositionError,
    compose_hdr,
    exposure_seconds,
    exposure_seconds,
    publish_hdr_composition,
)


def hdr_composition_params_from_preset(preset: dict | None = None) -> dict:
    """Normalise and clamp a preset into the full parameter set."""
    p = dict(preset or {})
    threshold = float(p.get("threshold", 0.8))
    growth = int(p.get("growth", 1))
    smoothness = float(p.get("smoothness", 7.0))
    titles = p.get("doc_titles") or p.get("titles") or []
    if isinstance(titles, str):
        titles = [t.strip() for t in titles.split(",") if t.strip()]
    else:
        titles = [str(t).strip() for t in list(titles) if str(t).strip()]
    ids = p.get("doc_ids") or []
    try:
        ids = [int(i) for i in list(ids)]
    except Exception:
        ids = []
    return {
        "threshold": float(min(1.20, max(0.05, threshold))),
        "growth": int(min(64, max(0, growth))),
        "smoothness": float(min(64.0, max(0.0, smoothness))),
        "open_mask": bool(p.get("open_mask", False)),
        "doc_titles": titles,
        "doc_ids": ids,
    }


def _doc_name(doc) -> str:
    try:
        name = doc.display_name()
    except Exception:
        name = None
    if not name:
        name = getattr(doc, "name", None) or getattr(doc, "title", None)
    return str(name or "Untitled").strip()


def _list_open_docs(main_window) -> list:
    if hasattr(main_window, "_list_open_docs"):
        try:
            docs = list(main_window._list_open_docs() or [])
            if docs:
                return docs
        except Exception:
            pass
    for attr in ("doc_manager", "docman", "dm"):
        dm = getattr(main_window, attr, None)
        if dm is None:
            continue
        docs = list(getattr(dm, "_docs", []) or [])
        if docs:
            return docs
    return []


def _sort_by_exposure(docs: list) -> list:
    """Longest exposure first; unknown times stay at the end."""

    def key(doc):
        exp = exposure_seconds(doc)
        return (exp is None, -(float(exp) if exp is not None else 0.0))

    return sorted(docs, key=key)


def resolve_hdr_docs(main_window, preset: dict | None = None, *, target_doc=None) -> list:
    """
    Resolve an ordered longest→shortest document list for headless apply.

    Preference order:
      1. ``doc_ids`` still alive in this session
      2. ``doc_titles`` matched against open views
      3. ``target_doc`` as the longest, plus remaining open views by exposure
      4. all open views sorted by exposure
    """
    params = hdr_composition_params_from_preset(preset)
    open_docs = [
        doc for doc in _list_open_docs(main_window)
        if getattr(doc, "image", None) is not None
    ]
    by_id = {id(doc): doc for doc in open_docs}
    by_title: dict[str, object] = {}
    for doc in open_docs:
        by_title.setdefault(_doc_name(doc), doc)

    resolved: list = []
    seen: set[int] = set()

    def _add(doc) -> None:
        if doc is None:
            return
        key = id(doc)
        if key in seen:
            return
        if getattr(doc, "image", None) is None:
            return
        seen.add(key)
        resolved.append(doc)

    for doc_id in params["doc_ids"]:
        _add(by_id.get(doc_id))
    if len(resolved) < 2:
        resolved.clear()
        seen.clear()
        for title in params["doc_titles"]:
            _add(by_title.get(title))

    if len(resolved) < 2:
        resolved.clear()
        seen.clear()
        if target_doc is not None:
            _add(target_doc)
            rest = [doc for doc in open_docs if id(doc) not in seen]
            for doc in _sort_by_exposure(rest):
                _add(doc)
        else:
            for doc in _sort_by_exposure(open_docs):
                _add(doc)

    return resolved


def apply_hdr_composition_via_preset(
    main_window,
    preset: dict | None = None,
    *,
    target_doc=None,
    ask_order: bool = False,
):
    """
    Run HDR Composition headlessly and publish a new view.

    Returns ``(result_doc_summary, steps)`` on success. Raises
    ``HDRCompositionError`` (or shows a message box when ``main_window`` is a
    live UI) when the frames cannot be resolved or composed.
    """
    params = hdr_composition_params_from_preset(preset)
    docs = resolve_hdr_docs(main_window, params, target_doc=target_doc)
    if len(docs) < 2:
        raise HDRCompositionError(
            "HDR Composition needs at least two open views. "
            "Name them in doc_titles (longest first), or open two registered frames."
        )

    # No order question any more: compose_hdr ranks the frames by their own
    # measured exposure, so a dropped preset or a bundle run no longer needs
    # somebody present to confirm that the list is the right way up.
    images = [doc.image for doc in docs]
    names = [_doc_name(doc) for doc in docs]
    result, mask, steps = compose_hdr(
        images,
        threshold=params["threshold"],
        growth=params["growth"],
        smoothness=params["smoothness"],
        names=names,
        exposures=[exposure_seconds(doc) for doc in docs],
    )
    order = list((steps[0].get("order") if steps else None) or range(len(docs)))
    if hasattr(main_window, "_log"):
        try:
            main_window._log(
                "[HDR Composition] composed longest-first: "
                + ", ".join(names[i] for i in order)
            )
        except Exception:
            pass

    dm = (
        getattr(main_window, "doc_manager", None)
        or getattr(main_window, "docman", None)
        or getattr(main_window, "dm", None)
    )
    summary = publish_hdr_composition(
        main_window, dm, docs[order[0]] if order else docs[0], result, mask,
        steps, open_mask=params["open_mask"],
    )
    if summary is None:
        raise HDRCompositionError("No document manager.")

    # Persist titles so replay works across sessions (ids do not).
    store = dict(params)
    store["doc_titles"] = names
    store["doc_ids"] = [id(doc) for doc in docs]
    try:
        remember = getattr(main_window, "remember_last_headless_command", None)
        if remember is None:
            remember = getattr(main_window, "_remember_last_headless_command", None)
        if callable(remember):
            remember("hdr_composition", store, description="HDR Composition")
    except Exception:
        pass
    return summary, steps


def open_hdr_composition_with_preset(main_window, preset: dict | None = None):
    """Open the live dialog with controls (and views) seeded from ``preset``."""
    from setiastro.saspro.hdr_composition import HDRCompositionDialog
    from setiastro.saspro.resources import hdrcomposition_path
    from PyQt6.QtGui import QIcon
    from PyQt6.QtCore import Qt as _Qt

    dlg = HDRCompositionDialog(main_window)
    dlg.setAttribute(_Qt.WidgetAttribute.WA_DeleteOnClose, True)
    try:
        dlg.setWindowIcon(QIcon(hdrcomposition_path))
    except Exception:
        pass
    try:
        dlg.apply_preset(preset)
    except Exception:
        pass
    dlg.resize(560, 520)
    dlg.show()
    return dlg


class HDRCompositionPresetDialog(QDialog):
    """Small form used by Edit Preset… on a desktop shortcut."""

    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("HDR Composition Preset")
        p = hdr_composition_params_from_preset(initial)

        form = QFormLayout()
        self.sp_threshold = QDoubleSpinBox()
        self.sp_threshold.setRange(0.05, 1.20)
        self.sp_threshold.setDecimals(3)
        self.sp_threshold.setSingleStep(0.01)
        self.sp_threshold.setValue(p["threshold"])
        form.addRow("Binarizing threshold:", self.sp_threshold)

        self.sp_growth = QSpinBox()
        self.sp_growth.setRange(0, 64)
        self.sp_growth.setValue(p["growth"])
        form.addRow("Mask growth (px):", self.sp_growth)

        self.sp_smooth = QDoubleSpinBox()
        self.sp_smooth.setRange(0.0, 64.0)
        self.sp_smooth.setDecimals(1)
        self.sp_smooth.setSingleStep(0.5)
        self.sp_smooth.setValue(p["smoothness"])
        form.addRow("Mask smoothness (px):", self.sp_smooth)

        self.chk_mask = QCheckBox("Also open the replacement mask")
        self.chk_mask.setChecked(p["open_mask"])
        form.addRow(self.chk_mask)

        self.ed_titles = QLineEdit(", ".join(p["doc_titles"]))
        self.ed_titles.setPlaceholderText("Longest first: Long 300s, Short 30s")
        form.addRow("View titles:", self.ed_titles)

        root = QVBoxLayout(self)
        root.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def result_dict(self) -> dict:
        raw = self.ed_titles.text().strip()
        titles = [t.strip() for t in raw.split(",") if t.strip()] if raw else []
        return hdr_composition_params_from_preset({
            "threshold": float(self.sp_threshold.value()),
            "growth": int(self.sp_growth.value()),
            "smoothness": float(self.sp_smooth.value()),
            "open_mask": bool(self.chk_mask.isChecked()),
            "doc_titles": titles,
        })
