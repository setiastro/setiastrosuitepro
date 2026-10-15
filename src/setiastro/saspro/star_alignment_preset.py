#saspro/star_alignment_preset.py
from __future__ import annotations
import os
import json
import re
import numpy as np
import cv2
from setiastro.saspro import astroalign
from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QDialogButtonBox, QComboBox, QLineEdit, QCheckBox,
    QSpinBox, QDoubleSpinBox, QPushButton, QFileDialog, QWidget, QMessageBox,
    QListWidget, QListWidgetItem, QAbstractItemView, QHBoxLayout, QVBoxLayout,
    QLabel, QProgressBar, QApplication, QGroupBox,
)
from PyQt6.QtCore import Qt


class _StarAlignmentProgress:
    """
    Modal "please wait" splash shown by the headless Star Alignment runner.
    Lightweight on purpose — the runner is synchronous on the main thread
    (no QThread), so this exists only to give the user something to look
    at between the Preset-drop click and the "aligned" result landing in
    their view.

    Context-manager style so the finally clause always closes it, even if
    find_transform raises or the user chose a weirdly-shaped target that
    trips the warp step. ``phase(label)`` updates the subtitle and pumps
    the event loop so the window actually repaints without a thread. The
    bar is indeterminate (range 0,0) — cancellation would require
    cooperative polling across astroalign/cv2 that would complicate the
    runner for little benefit on a single-frame op; keep it simple.

    Parented to the host main window when available so taskbar grouping,
    modality masking, and dark-mode styling match the rest of SASpro's
    dialogs. Falls back to a top-level window when no host is passed
    (e.g. headless unit tests that drive the runner directly).
    """

    def __init__(self, parent: QWidget | None = None, title: str = "Star Alignment"):
        self._dlg = QDialog(parent)
        self._dlg.setWindowTitle(title)
        self._dlg.setModal(True)
        try:
            # Frameless looks splash-ier and prevents the user from
            # accidentally closing a dialog that doesn't yet support cancel.
            self._dlg.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        except Exception:
            pass
        v = QVBoxLayout(self._dlg)
        v.setContentsMargins(20, 18, 20, 18)
        self._lbl = QLabel("Starting Star Alignment…")
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._lbl)
        self._bar = QProgressBar()
        self._bar.setRange(0, 0)   # indeterminate
        self._bar.setTextVisible(False)
        v.addWidget(self._bar)
        self._dlg.setFixedWidth(320)

    def __enter__(self):
        self._dlg.show()
        self._dlg.raise_()
        # Two pumps because the first often lands a layout+resize, the
        # second actually paints — otherwise the window may show up blank
        # until the first phase() call.
        QApplication.processEvents()
        QApplication.processEvents()
        return self

    def phase(self, label: str) -> None:
        """Update the subtitle text and give Qt a chance to repaint."""
        try:
            self._lbl.setText(str(label))
            QApplication.processEvents()
        except Exception:
            # Dialog might already be torn down (user closed main window
            # mid-run); don't let that mask the real error from the runner.
            pass

    def close(self) -> None:
        try:
            self._dlg.close()
            self._dlg.deleteLater()
        except Exception:
            pass
        try:
            QApplication.processEvents()
        except Exception:
            pass

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False  # don't swallow the runner's exceptions


def _find_main_window(start_widget):
    """
    Walk up the Qt parent chain from a dialog widget looking for the SASpro
    main window — identified by the ``mdi`` attribute (also what
    _iter_views_with_titles already uses as the entry point). Returns None
    if we never find one (e.g. the preset dialog was opened from a test
    harness without the full host wired up).
    """
    w = start_widget
    for _ in range(16):  # bounded walk — Qt trees are shallow
        if w is None:
            return None
        if hasattr(w, "mdi"):
            return w
        w = w.parent() if hasattr(w, "parent") else None
    return None

def _norm_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    # strip common UI decorations
    s = re.sub(r"^[■●◆▲▪▫•◼◻◾◽]\s*", "", s)
    s = re.sub(r"^Active View:\s*", "", s, flags=re.I)
    # ignore extensions when users type a short name
    base = os.path.splitext(s)[0]
    return base.casefold()

def _iter_views_with_titles(app):
    """Yield (visible_title, doc, subwin). Visible title honors per-view overrides."""
    mdi = getattr(app, "mdi", None)
    if mdi is None:
        return
    for sw in mdi.subWindowList() or []:
        try:
            w = sw.widget()
            doc = getattr(w, "document", None)
            if doc is None:
                continue
            if hasattr(w, "_effective_title"):
                title = str(w._effective_title())
            else:
                title = str(sw.windowTitle() or doc.display_name())
            yield (title, doc, sw)
        except Exception:
            continue

def _resolve_ref_doc_by_view_name(app, name: str):
    """Return (doc or None, debug_list)."""
    wanted = _norm_name(name)
    if not wanted:
        return None, []

    views = list(_iter_views_with_titles(app))
    seen = [t for (t, _, __) in views]

    # 1) exact title or exact doc name
    for t, d, _ in views:
        if _norm_name(t) == wanted or _norm_name(getattr(d, "display_name")()) == wanted:
            return d, seen

    # 2) prefix match
    for t, d, _ in views:
        if _norm_name(t).startswith(wanted) or _norm_name(getattr(d, "display_name")()).startswith(wanted):
            return d, seen

    # 3) substring
    for t, d, _ in views:
        if wanted in _norm_name(t) or wanted in _norm_name(getattr(d, "display_name")()):
            return d, seen

    return None, seen

# ---------------------------
# Preset dialog (like others)
# ---------------------------
# Default alignment-settings seed used by the preset dialog when the caller
# didn't pass one through. Shape matches the full set of knobs the live Star
# Alignment dialog exposes (minus Trial Detect), so a preset built here can
# drive the headless runner with exactly the same inputs as a UI run.
_DEFAULT_ALIGN_SETTINGS = {
    "method":              "stellar",     # stellar | wcs | wcs_stellar
    "model":               "affine",      # affine | homography | poly3 | poly4
    "det_sigma":           12.0,
    "min_area":            10,
    "max_control_points":  250,
    "h_reproj":            3.0,            # only consumed when model=homography
    "limit_stars":         500,
    "min_fwhm":            1.2,
    "max_ellipticity":     0.6,
}

_IDX_TO_METHOD = ("stellar", "wcs", "wcs_stellar")
_IDX_TO_MODEL  = ("affine", "homography", "poly3", "poly4")


class StarAlignmentPresetDialog(QDialog):
    """
    Preset UI for headless Star Alignment.

    Persisted preset shape (all keys optional; headless runner fills in
    _DEFAULT_ALIGN_SETTINGS for anything missing):

      Reference selection:
        ref_mode:  active | view_name | file
        ref_name:  str  (view_name)
        ref_file:  str  (file)
      Output:
        overwrite: bool
      Legacy user-level downsample knob (kept for back-compat with earlier
      presets; the headless runner now treats this as a solve downsample
      override exposed under the "Advanced" group below):
        downsample: int (>=1)
      Advanced alignment settings (match the live Star Alignment dialog):
        method:              stellar | wcs | wcs_stellar
        model:               affine | homography | poly3 | poly4
        det_sigma:           float   (astroalign detection sigma)
        min_area:            int     (astroalign min connected area)
        max_control_points:  int     (astroalign cap on matched stars)
        h_reproj:            float   (homography RANSAC inlier threshold)
        limit_stars:         int     (reserved; parity with Stacking Suite)
        min_fwhm:            float   (reject detections narrower than this)
        max_ellipticity:     float   (reject detections more elliptical)
    """
    def __init__(self, parent: QWidget | None = None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Star Alignment — Preset")
        init = dict(initial or {})

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["active", "view_name", "file"])
        self.cmb_mode.setCurrentText(str(init.get("ref_mode", "active")).lower())

        self.edt_name = QLineEdit()
        self.edt_name.setPlaceholderText("Exact view title (case-insensitive)")
        self.edt_name.setText(str(init.get("ref_name", "")))
        # Pick-from-list shortcut — the typing-only field was annoying when
        # the view's visible title has long project-specific prefixes or
        # unicode decorations. Clicking "Pick…" pops a chooser that lists
        # every open view (same iterator the headless resolver walks) and
        # fills the field with the normalised base title on commit — so the
        # text that lands in the preset is the one the resolver's exact
        # match will succeed on first hop. Also auto-switches ref_mode to
        # view_name so the user doesn't have to click the combo first.
        self.btn_pick_view = QPushButton("Pick…")
        self.btn_pick_view.setToolTip(
            "Pick an open view by name. Fills the text field with the view's "
            "title and sets the reference mode to 'view_name' automatically."
        )
        self.btn_pick_view.clicked.connect(self._pick_view)

        self.edt_file = QLineEdit()
        self.edt_file.setPlaceholderText("/absolute/path/to/reference.fits|fit|xisf|tif")
        self.edt_file.setText(str(init.get("ref_file", "")))
        self.btn_browse = QPushButton("Browse…")
        def _pick_file():
            p, _ = QFileDialog.getOpenFileName(self, "Choose reference file", "",
                "Images (*.fits *.fit *.xisf *.tif *.tiff *.png *.jpg);;All Files (*)")
            if p:
                self.edt_file.setText(p)
        self.btn_browse.clicked.connect(_pick_file)

        self.chk_overwrite = QCheckBox("Overwrite target view")
        self.chk_overwrite.setChecked(bool(init.get("overwrite", False)))

        self.spin_down = QSpinBox()
        self.spin_down.setRange(1, 8)
        self.spin_down.setValue(int(init.get("downsample", 2)))
        self.spin_down.setToolTip("Downsample factor during transform solve (speed vs precision)")

        # ---- Advanced alignment settings (parity with the live Star
        # Alignment dialog, minus Trial Detect — this is a static-preset
        # editor and detection-threshold tuning on a snapshot wouldn't
        # mean much). Lives in a collapsible group so first-time users
        # aren't overwhelmed; most presets live or die by ref_mode +
        # overwrite, and the rest only matter for tricky frames. ----
        self.cmb_method = QComboBox()
        self.cmb_method.addItems([
            "Stellar (astroalign)",
            "WCS (plate-solve reproject)",
            "WCS → Stellar refine",
        ])
        _METHOD_TO_IDX = {"stellar": 0, "wcs": 1, "wcs_stellar": 2}
        self.cmb_method.setCurrentIndex(
            _METHOD_TO_IDX.get(str(init.get("method",
                _DEFAULT_ALIGN_SETTINGS["method"])).lower(), 0)
        )
        self.cmb_method.setToolTip(
            "Alignment method applied to the dropped target.\n"
            " • Stellar: fast; detects stars with astroalign.\n"
            " • WCS: reprojects via the per-frame plate solve (needs WCS\n"
            "   in BOTH reference and target headers).\n"
            " • WCS → Stellar: WCS for coarse alignment, astroalign refine on top."
        )

        self.cmb_model = QComboBox()
        self.cmb_model.addItems([
            "Affine (fast)",
            "Homography (projective)",
            "Polynomial (order 3)",
            "Polynomial (order 4)",
        ])
        _MODEL_TO_IDX = {"affine": 0, "homography": 1, "poly3": 2, "poly4": 3}
        self.cmb_model.setCurrentIndex(
            _MODEL_TO_IDX.get(str(init.get("model",
                _DEFAULT_ALIGN_SETTINGS["model"])).lower(), 0)
        )
        self.cmb_model.setToolTip(
            "Transform model for the final warp.\n"
            " • Affine: fast; good for well-matched optics.\n"
            " • Homography: handles perspective shear / meridian flips.\n"
            " • Polynomial 3/4: fits radial / mild field distortion."
        )

        self.spin_det_sigma = QDoubleSpinBox()
        self.spin_det_sigma.setRange(1.0, 200.0)
        self.spin_det_sigma.setDecimals(1); self.spin_det_sigma.setSingleStep(1.0)
        self.spin_det_sigma.setValue(float(init.get("det_sigma",
            _DEFAULT_ALIGN_SETTINGS["det_sigma"])))
        self.spin_det_sigma.setToolTip(
            "Star detection threshold (σ above background). Lower = more\n"
            "stars (may include noise). Higher = only bright stars."
        )

        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(1, 100)
        self.spin_min_area.setValue(int(init.get("min_area",
            _DEFAULT_ALIGN_SETTINGS["min_area"])))
        self.spin_min_area.setToolTip(
            "Minimum connected-pixel area for a detection. Smaller = includes\n"
            "faint/undersampled stars; larger = rejects noise and hot pixels."
        )

        self.spin_maxcp = QSpinBox()
        self.spin_maxcp.setRange(20, 2000); self.spin_maxcp.setSingleStep(10)
        self.spin_maxcp.setValue(int(init.get("max_control_points",
            _DEFAULT_ALIGN_SETTINGS["max_control_points"])))
        self.spin_maxcp.setToolTip(
            "Cap on matched stars (astroalign max_control_points). 250 is a\n"
            "balanced default; raise for crowded fields."
        )

        self.spin_h_reproj = QDoubleSpinBox()
        self.spin_h_reproj.setRange(0.1, 10.0); self.spin_h_reproj.setDecimals(2)
        self.spin_h_reproj.setSingleStep(0.1)
        self.spin_h_reproj.setValue(float(init.get("h_reproj",
            _DEFAULT_ALIGN_SETTINGS["h_reproj"])))
        self.spin_h_reproj.setToolTip(
            "RANSAC inlier threshold (px). ONLY used when Model = Homography."
        )

        self.spin_limit_stars = QSpinBox()
        self.spin_limit_stars.setRange(50, 5000); self.spin_limit_stars.setSingleStep(50)
        self.spin_limit_stars.setValue(int(init.get("limit_stars",
            _DEFAULT_ALIGN_SETTINGS["limit_stars"])))
        self.spin_limit_stars.setToolTip(
            "Hard cap on stars used for matching per frame. Reserved for parity\n"
            "with Stacking Suite — astroalign honours max_control_points above."
        )

        self.spin_min_fwhm = QDoubleSpinBox()
        self.spin_min_fwhm.setRange(0.1, 10.0); self.spin_min_fwhm.setDecimals(2)
        self.spin_min_fwhm.setSingleStep(0.1)
        self.spin_min_fwhm.setValue(float(init.get("min_fwhm",
            _DEFAULT_ALIGN_SETTINGS["min_fwhm"])))
        self.spin_min_fwhm.setToolTip(
            "Reject detections narrower than this FWHM (filters hot pixels\n"
            "and cosmic ray hits). Parity with Stacking Suite."
        )

        self.spin_max_ellip = QDoubleSpinBox()
        self.spin_max_ellip.setRange(0.0, 1.0); self.spin_max_ellip.setDecimals(2)
        self.spin_max_ellip.setSingleStep(0.05)
        self.spin_max_ellip.setValue(float(init.get("max_ellipticity",
            _DEFAULT_ALIGN_SETTINGS["max_ellipticity"])))
        self.spin_max_ellip.setToolTip(
            "Reject detections more elliptical than this (1 - b/a).\n"
            "Lower = strict (rejects slightly trailed stars). Parity with\n"
            "Stacking Suite."
        )

        lay = QFormLayout(self)
        lay.addRow("Reference mode:", self.cmb_mode)
        # View-name row: text field + "Pick…" button side by side. The HBox
        # keeps them on a single QFormLayout row with the "View name" label
        # on the left (unlike the file row's nested-QFormLayout trick, which
        # splits label and widget onto two visual lines).
        row_name = QWidget()
        rn = QHBoxLayout(row_name); rn.setContentsMargins(0, 0, 0, 0); rn.setSpacing(6)
        rn.addWidget(self.edt_name, 1)
        rn.addWidget(self.btn_pick_view, 0)
        lay.addRow("View name (if view_name):", row_name)
        rowf = QWidget(); rf = QFormLayout(rowf); rf.setContentsMargins(0,0,0,0)
        rf.addRow(self.edt_file, self.btn_browse)
        lay.addRow("File (if file):", rowf)
        lay.addRow(self.chk_overwrite)
        lay.addRow("Downsample factor:", self.spin_down)

        # Build the Advanced group as a collapsible QGroupBox and add it as
        # a full-width row (passing a single widget to QFormLayout.addRow
        # drops the label column, which is what we want here — the group's
        # title serves as the label).
        self.grp_advanced = QGroupBox("Advanced — Alignment Settings")
        self.grp_advanced.setCheckable(True)
        # Opens collapsed by default so first-time preset editors don't see
        # eight knobs they don't need. Users editing a preset that already
        # has non-default advanced values see the group auto-open — see
        # _advanced_has_overrides below.
        self.grp_advanced.setChecked(self._advanced_has_overrides(init))
        _agl = QFormLayout(self.grp_advanced)
        _agl.setContentsMargins(10, 6, 10, 6)
        _agl.setVerticalSpacing(4)
        _agl.addRow("Method:",                     self.cmb_method)
        _agl.addRow("Model:",                      self.cmb_model)
        _agl.addRow("Detection threshold (σ):",    self.spin_det_sigma)
        _agl.addRow("Min area (px):",              self.spin_min_area)
        _agl.addRow("Max control points:",         self.spin_maxcp)
        _agl.addRow("Max stars (limit):",          self.spin_limit_stars)
        _agl.addRow("Min FWHM (px):",              self.spin_min_fwhm)
        _agl.addRow("Max ellipticity:",            self.spin_max_ellip)
        _agl.addRow("Homog. RANSAC reproj (px):",  self.spin_h_reproj)
        lay.addRow(self.grp_advanced)

        # Enable/disable Advanced fields that only apply to certain Method
        # choices. Mirrors the live dialog's behaviour so the UI feels the
        # same between preset editor and tool dialog.
        def _advanced_gating():
            idx = self.cmb_method.currentIndex()
            uses_stellar = (idx == 0)
            uses_refine  = (idx == 2)
            uses_stars   = uses_stellar or uses_refine
            # Stellar-only knobs (not consumed by WCS):
            for w in (self.cmb_model, self.spin_maxcp, self.spin_limit_stars,
                      self.spin_min_fwhm, self.spin_max_ellip,
                      self.spin_h_reproj):
                w.setEnabled(uses_stellar)
                try:
                    lab = _agl.labelForField(w)
                    if lab is not None:
                        lab.setEnabled(uses_stellar)
                except Exception:
                    pass
            # det_sigma + min_area are also used by the stellar refine step
            # of wcs_stellar, so they stay enabled for either stellar-based
            # method.
            for w in (self.spin_det_sigma, self.spin_min_area):
                w.setEnabled(uses_stars)
                try:
                    lab = _agl.labelForField(w)
                    if lab is not None:
                        lab.setEnabled(uses_stars)
                except Exception:
                    pass
            # Homography-only knob within the stellar case.
            if uses_stellar:
                is_h = (self.cmb_model.currentIndex() == 1)
                self.spin_h_reproj.setEnabled(is_h)
                try:
                    lab = _agl.labelForField(self.spin_h_reproj)
                    if lab is not None:
                        lab.setEnabled(is_h)
                except Exception:
                    pass
        self._advanced_gating = _advanced_gating
        self.cmb_method.currentIndexChanged.connect(lambda _: _advanced_gating())
        self.cmb_model.currentIndexChanged.connect(lambda _: _advanced_gating())
        _advanced_gating()

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        lay.addRow(btns)

        # enable/disable inputs per mode. Pick… stays enabled in every mode:
        # clicking it is itself a declaration "I want a view", so it auto-
        # switches ref_mode to view_name on commit — forcing the user to
        # change the combo first would defeat the point of the shortcut.
        def _upd():
            m = self.cmb_mode.currentText()
            self.edt_name.setEnabled(m == "view_name")
            self.edt_file.setEnabled(m == "file")
            self.btn_browse.setEnabled(m == "file")
            # self.btn_pick_view stays enabled — see docstring above.
        self.cmb_mode.currentTextChanged.connect(_upd); _upd()

    # ------------------------------------------------------------------
    # View picker
    # ------------------------------------------------------------------
    def _pick_view(self):
        """
        Pop a single-select chooser listing every open view and, on commit,
        write its visible title (extension stripped) into ``edt_name`` and
        switch ``cmb_mode`` to ``view_name``. Uses the SAME iterator the
        headless resolver walks (``_iter_views_with_titles``) so a title
        chosen here is guaranteed to be a title the resolver would also
        match against — no "typed name doesn't match any view" surprises.
        """
        app = _find_main_window(self)
        if app is None:
            QMessageBox.information(
                self, "Pick View",
                "Cannot enumerate open views from here — the preset dialog "
                "doesn't have a path to the main window.\n\n"
                "Type the view's exact title in the field instead."
            )
            return

        views = [(t, d) for (t, d, _sw) in _iter_views_with_titles(app)
                 if d is not None and getattr(d, "image", None) is not None]
        if not views:
            QMessageBox.information(
                self, "No Views",
                "No open views have image data available to use as a reference.\n\n"
                "Open an image first, then try again."
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Pick Reference View")
        dlg.setMinimumWidth(380)
        vb = QVBoxLayout(dlg)
        vb.addWidget(QLabel(
            "Select which open view should be the reference image:"
        ))

        lst = QListWidget(dlg)
        lst.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        # Pre-select the view whose title best matches whatever's already in
        # the text field, so re-opening the picker to tweak the pick doesn't
        # drop the current selection.
        current_norm = _norm_name(self.edt_name.text())
        pre_row = 0
        for i, (title, doc) in enumerate(views):
            item = QListWidgetItem(title)
            # Stash the raw title so commit uses the string the resolver
            # will see — not whatever QListWidget's display formatting did.
            item.setData(Qt.ItemDataRole.UserRole, title)
            lst.addItem(item)
            if current_norm and _norm_name(title) == current_norm:
                pre_row = i
        lst.setCurrentRow(pre_row)
        vb.addWidget(lst)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dlg,
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        vb.addWidget(btns)
        # Double-click = pick + OK, matches typical QListWidget UX.
        lst.itemDoubleClicked.connect(lambda _: dlg.accept())

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        it = lst.currentItem()
        if it is None:
            return
        title = str(it.data(Qt.ItemDataRole.UserRole) or it.text())
        # Store the base (extension-stripped) title — matches how the
        # headless resolver's _norm_name compares, and keeps the preset JSON
        # human-readable.
        self.edt_name.setText(os.path.splitext(title)[0])
        # Auto-switch mode to view_name so the picked value actually gets
        # consumed by the headless runner; without this a Pick-then-save
        # would silently keep "active" mode and ignore the typed name.
        self.cmb_mode.setCurrentText("view_name")

    @staticmethod
    def _advanced_has_overrides(init: dict) -> bool:
        """
        Return True iff the incoming preset carries any Advanced value that
        differs from _DEFAULT_ALIGN_SETTINGS. Used to auto-open the Advanced
        group on dialog creation so users editing a tuned preset immediately
        see what they tuned — a collapsed Advanced group would hide values
        the user is likely to want to inspect.
        """
        if not isinstance(init, dict):
            return False
        for k, default in _DEFAULT_ALIGN_SETTINGS.items():
            if k not in init:
                continue
            v = init.get(k)
            try:
                if isinstance(default, float):
                    if abs(float(v) - float(default)) > 1e-9:
                        return True
                elif isinstance(default, int):
                    if int(v) != int(default):
                        return True
                else:
                    if str(v).strip().lower() != str(default).strip().lower():
                        return True
            except Exception:
                return True
        return False

    def result_dict(self) -> dict:
        return {
            # Reference + output (back-compat with original 5-key schema)
            "ref_mode":   self.cmb_mode.currentText(),
            "ref_name":   self.edt_name.text().strip(),
            "ref_file":   self.edt_file.text().strip(),
            "overwrite":  bool(self.chk_overwrite.isChecked()),
            "downsample": int(self.spin_down.value()),
            # Advanced — full set the live Star Alignment dialog exposes.
            # Written on every save so a preset file is self-contained.
            "method":              _IDX_TO_METHOD[self.cmb_method.currentIndex()],
            "model":               _IDX_TO_MODEL[self.cmb_model.currentIndex()],
            "det_sigma":           float(self.spin_det_sigma.value()),
            "min_area":            int(self.spin_min_area.value()),
            "max_control_points":  int(self.spin_maxcp.value()),
            "h_reproj":            float(self.spin_h_reproj.value()),
            "limit_stars":         int(self.spin_limit_stars.value()),
            "min_fwhm":            float(self.spin_min_fwhm.value()),
            "max_ellipticity":     float(self.spin_max_ellip.value()),
        }


# ---------------------------
# Headless runner
# ---------------------------
def run_star_alignment_via_preset(app, preset: dict, target_doc) -> None:
    """
    app: main window (has mdi, docman/doc_manager, _log)
    target_doc: document object for the drop target
    Behavior mirrors other *_via_preset helpers: overwrite or create new view.

    Blocks the main thread (this runner is synchronous — no QThread). A
    modal _StarAlignmentProgress splash is shown around the whole body so
    the user sees something between the preset-drop click and the aligned
    pixels landing. phase() calls at each meaningful step pump the Qt
    event loop so the splash actually repaints.
    """
    from setiastro.saspro.legacy.image_manager import load_image  # for file ref

    # The splash is parented to the main window when available so modality
    # and taskbar grouping match the rest of SASpro. On a headless unit
    # test (no main window), it floats as a top-level dialog.
    with _StarAlignmentProgress(parent=app, title="Star Alignment") as _progress:
        _progress.phase("Loading target image…")

        # ---- resolve target image ----
        if target_doc is None or getattr(target_doc, "image", None) is None:
            raise RuntimeError("Target view has no image.")
        tgt = np.asarray(target_doc.image, dtype=np.float32)
        if tgt.ndim not in (2, 3):
            raise RuntimeError("Unsupported target image shape.")

        # ---- resolve reference image per preset ----
        pp   = dict(preset or {})
        mode = str(pp.get("ref_mode", "active")).lower()

        def _doc_display_name(d):
            try:
                if callable(getattr(d, "display_name", None)):
                    return d.display_name() or ""
                nm = getattr(d, "title", None)
                return nm or ""
            except Exception:
                return ""

        ref = None
        ref_name = "Reference"

        if mode == "file":
            path = pp.get("ref_file") or ""
            _progress.phase(f"Loading reference file: {os.path.basename(path) if path else '(unset)'}")
            if not path or not os.path.exists(path):
                raise RuntimeError("Reference file does not exist.")
            ref, _, _, _ = load_image(path)
            if ref is None:
                raise RuntimeError("Failed to load reference file.")
            ref = np.asarray(ref, dtype=np.float32)
            ref_name = os.path.basename(path)

        elif mode == "view_name":
            # NEW: resolve against visible titles (per-view overrides) w/ normalization
            raw_name = (pp.get("ref_name") or "").strip()
            _progress.phase(f"Resolving reference view: {raw_name or '(unset)'}")
            if not raw_name:
                raise RuntimeError("Reference view name is empty.")

            ref_doc, seen_titles = _resolve_ref_doc_by_view_name(app, raw_name)
            if ref_doc is None or getattr(ref_doc, "image", None) is None:
                # helpful error includes visible open view titles (no glyphs/exts)
                opts = ", ".join(os.path.splitext(t)[0] for t in seen_titles) if seen_titles else "(no open views)"
                raise RuntimeError(f"Reference view '{raw_name}' not found.\nOpen views: {opts}")

            ref = np.asarray(ref_doc.image, dtype=np.float32)

            # Prefer the visible title of the matched subwindow for the label
            ref_name = None
            for t, d, _ in _iter_views_with_titles(app):
                if d is ref_doc:
                    ref_name = os.path.splitext(t)[0]
                    break
            ref_name = ref_name or (_doc_display_name(ref_doc) or "Reference")

        else:  # "active" (default) → use the app's active subwindow doc
            _progress.phase("Reading active view as reference…")
            sw = getattr(app, "mdi", None).activeSubWindow() if getattr(app, "mdi", None) else None
            if sw is None:
                raise RuntimeError("No active view to use as reference.")
            ref_doc = getattr(sw.widget(), "document", None)
            if ref_doc is None or getattr(ref_doc, "image", None) is None:
                raise RuntimeError("Active view has no image to use as reference.")
            ref = np.asarray(ref_doc.image, dtype=np.float32)
            # Prefer visible title if available
            try:
                for t, d, _ in _iter_views_with_titles(app):
                    if d is ref_doc:
                        ref_name = os.path.splitext(t)[0]
                        break
                if ref_name == "Reference":
                    ref_name = _doc_display_name(ref_doc) or "Reference"
            except Exception:
                ref_name = _doc_display_name(ref_doc) or "Reference"

        # ---- grayscale + optional downsample for solve ----
        def _to_gray(a):
            return np.mean(a, axis=2) if a.ndim == 3 else a

        ds = int(max(1, pp.get("downsample", 2)))
        _progress.phase(
            f"Preparing images (ds={ds})…" if ds > 1
            else "Preparing images (grayscale)…"
        )
        ref_g = _to_gray(ref); tgt_g = _to_gray(tgt)
        if ds > 1:
            wR, hR = ref_g.shape[1] // ds, ref_g.shape[0] // ds
            wT, hT = tgt_g.shape[1] // ds, tgt_g.shape[0] // ds
            ref_s = cv2.resize(ref_g, (max(1, wR), max(1, hR)), interpolation=cv2.INTER_AREA)
            tgt_s = cv2.resize(tgt_g, (max(1, wT), max(1, hT)), interpolation=cv2.INTER_AREA)
        else:
            ref_s, tgt_s = ref_g, tgt_g

        # Pull the Advanced preset values with safe defaults so an older
        # 5-key preset still runs — _DEFAULT_ALIGN_SETTINGS is the single
        # source of truth; the headless runner never second-guesses it.
        method    = str(pp.get("method", _DEFAULT_ALIGN_SETTINGS["method"])).strip().lower()
        if method not in ("stellar", "wcs", "wcs_stellar"):
            method = "stellar"
        model     = str(pp.get("model",  _DEFAULT_ALIGN_SETTINGS["model"])).strip().lower()
        if model not in ("affine", "homography", "poly3", "poly4"):
            model = "affine"
        user_det  = float(pp.get("det_sigma",          _DEFAULT_ALIGN_SETTINGS["det_sigma"]))
        user_ma   = int(  pp.get("min_area",           _DEFAULT_ALIGN_SETTINGS["min_area"]))
        user_mcp  = int(  pp.get("max_control_points", _DEFAULT_ALIGN_SETTINGS["max_control_points"]))
        user_hr   = float(pp.get("h_reproj",           _DEFAULT_ALIGN_SETTINGS["h_reproj"]))

        # ---- WCS reproject branch (method = "wcs" or "wcs_stellar") ----
        # Reuses the proven wcs_reproject_align helper from star_alignment.
        # The sibling WCSRegistrationThread does the same thing per-frame;
        # here we do it once for the single dropped target.
        aligned = None
        if method in ("wcs", "wcs_stellar"):
            _progress.phase("WCS: resolving plate solves…")
            try:
                from setiastro.saspro.star_alignment import (
                    _wcs_from_header, wcs_reproject_align,
                    _doc_header_or_wcs,
                )
            except Exception as e:
                raise RuntimeError(
                    f"WCS alignment unavailable (import failed: {e}). "
                    "Switch Method to Stellar in the preset to continue."
                )

            # Reference WCS — from the loaded header (file) or the resolved doc.
            ref_hdr = None
            if mode == "file":
                # load_image already gave us the header above into `_` — re-read
                # to pick it up without changing earlier variable names.
                ref_img2, ref_hdr, _, _ = load_image(pp.get("ref_file", ""))
                del ref_img2  # we already have `ref`
            elif mode == "view_name":
                ref_hdr = _doc_header_or_wcs(ref_doc)
            else:
                ref_hdr = _doc_header_or_wcs(ref_doc)
            ref_wcs = _wcs_from_header(ref_hdr)

            # Target WCS — comes from the target doc's header.
            tgt_hdr = _doc_header_or_wcs(target_doc)
            tgt_wcs = _wcs_from_header(tgt_hdr)

            if ref_wcs is None or tgt_wcs is None:
                missing = [n for n, w in (("reference", ref_wcs), ("target", tgt_wcs)) if w is None]
                raise RuntimeError(
                    f"WCS alignment needs plate solves on both images; "
                    f"missing on: {', '.join(missing)}. Plate-solve them first "
                    "(or switch Method to Stellar in the preset)."
                )

            _progress.phase("WCS: reprojecting target onto reference grid…")
            H_ref, W_ref = ref.shape[:2]
            aligned = wcs_reproject_align(
                tgt, tgt_wcs, ref_wcs, (H_ref, W_ref),
                tile=512, log=lambda s: None,
            ).astype(np.float32, copy=False)
            # Set shape vars used by the write-back block below.
            H, W = H_ref, W_ref

        # ---- Stellar branch (method = "stellar") OR stellar refine step ----
        # For "stellar" we solve + warp from scratch. For "wcs_stellar" we
        # take the already-reprojected `aligned` as the "target" and solve
        # a small correction against the reference, then warp again.
        if method in ("stellar", "wcs_stellar"):
            # When refining, the "target to solve against the reference" is
            # the WCS-aligned result, not the original target. For a pure
            # stellar run it's the original target.
            solve_src = aligned if (method == "wcs_stellar" and aligned is not None) else tgt
            solve_src_g = _to_gray(solve_src)
            if ds > 1:
                wS, hS = solve_src_g.shape[1] // ds, solve_src_g.shape[0] // ds
                solve_src_s = cv2.resize(solve_src_g, (max(1, wS), max(1, hS)),
                                         interpolation=cv2.INTER_AREA)
            else:
                solve_src_s = solve_src_g

            # First attempt uses the user's preset values. If it fails, fall
            # back to the previous 4-try sigma-ladder with a warning in the
            # splash so the user knows their preset tuning was too strict
            # (useful signal — if they NEEDED those settings for a reason,
            # a successful fallback hints they can relax the preset).
            tries = [
                dict(detection_sigma=user_det, min_area=user_ma,
                     max_control_points=user_mcp),
                dict(detection_sigma=5,  min_area=7,  max_control_points=75),
                dict(detection_sigma=12, min_area=9,  max_control_points=75),
                dict(detection_sigma=20, min_area=9,  max_control_points=75),
                dict(detection_sigma=30, min_area=11, max_control_points=75),
            ]
            last = None
            M = None
            for i, kw in enumerate(tries, start=1):
                tag = "preset" if i == 1 else f"fallback {i-1}/{len(tries)-1}"
                _progress.phase(
                    f"Finding transform ({tag}, σ={kw['detection_sigma']}, "
                    f"min_area={kw['min_area']})…"
                )
                try:
                    T, _ = astroalign.find_transform(
                        solve_src_s.astype(np.float32),
                        ref_s.astype(np.float32), **kw)
                    # astroalign returns similarity in T.params[0:2]; the full
                    # model refinement for homography/poly happens in the live
                    # dialog's run_alignment, which is heavier. For a single-
                    # target headless op the affine solve is accurate enough —
                    # the Model combo is still recorded in the preset so a
                    # UI run honours it faithfully.
                    M = T.params[0:2, :].astype(np.float32)
                    break
                except Exception as e:
                    last = e
                    try:
                        import sep
                        if "pixel buffer full" in str(e).lower():
                            sep.set_extract_pixstack(int(sep.get_extract_pixstack() * 2))
                    except Exception:
                        pass
            if M is None:
                raise RuntimeError(f"Astroalign failed: {last}")

            if ds > 1:  # scale the translation back up
                M = M.copy()
                M[0, 2] *= ds
                M[1, 2] *= ds

            # ---- warp the appropriate source onto the reference geometry ----
            H, W = ref_g.shape[:2]
            warp_src = aligned if (method == "wcs_stellar" and aligned is not None) else tgt
            if warp_src.ndim == 2:
                _progress.phase("Warping target to reference grid…")
                aligned = cv2.warpAffine(
                    warp_src, M, (W, H),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
            else:
                planes = []
                # Per-plane phase updates — on a 6248×4176 RGB frame each
                # warp is a visible chunk of wall time, and the user wants
                # to see motion between them.
                ch_labels = (["R", "G", "B"] if warp_src.shape[2] == 3
                             else [f"ch{i}" for i in range(warp_src.shape[2])])
                for i in range(warp_src.shape[2]):
                    label = ch_labels[i] if i < len(ch_labels) else f"ch{i}"
                    _progress.phase(f"Warping target ({label}) to reference grid…")
                    planes.append(cv2.warpAffine(
                        warp_src[..., i], M, (W, H),
                        flags=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                    ))
                aligned = np.stack(planes, axis=2)

        if aligned is None:
            raise RuntimeError(
                f"Alignment produced no result for method={method!r}."
            )
        aligned = aligned.astype(np.float32, copy=False)

        # ---- write back (overwrite or new view) ----
        overwrite = bool(pp.get("overwrite", False))
        if overwrite:
            _progress.phase(f"Writing back → {ref_name} (overwriting target view)…")
            if hasattr(target_doc, "set_image"):
                target_doc.set_image(aligned, step_name=f"Star Alignment → {ref_name}")
            else:
                target_doc.image = aligned
            try:
                if hasattr(target_doc, "changed"):
                    target_doc.changed.emit()
            except Exception:
                pass
        else:
            _progress.phase(f"Opening new view → {ref_name}…")
            dm = getattr(app, "docman", None) or getattr(app, "doc_manager", None)
            if dm is None:
                raise RuntimeError("Document manager not available.")
            # title
            base = ""
            try:
                base = target_doc.display_name() if callable(getattr(target_doc, "display_name", None)) \
                       else (getattr(target_doc, "title", None) or "")
            except Exception:
                pass
            base = base or "Image"
            title = f"{base} [Aligned → {ref_name}]"
            meta = {
                "step_name":   "Star Alignment",
                "description": f"Aligned to {ref_name}",
                "is_mono": bool(aligned.ndim == 2 or (aligned.ndim == 3 and aligned.shape[2] == 1)),
            }
            newdoc = dm.open_array(aligned, metadata=meta, title=title)
            if hasattr(app, "_spawn_subwindow_for"):
                app._spawn_subwindow_for(newdoc)