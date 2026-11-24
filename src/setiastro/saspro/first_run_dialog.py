# ============================================================================
# setiastro/saspro/first_run_dialog.py
#
# When the welcome is due, and the tip of the day. The welcome itself is the
# setup window, setup_center.py.
#
# Wiring (gui_entry.py _kick_updates_after_splash):
#
#   from setiastro.saspro.first_run_dialog import maybe_show_first_run_dialog
#   QTimer.singleShot(200, lambda: maybe_show_first_run_dialog(win))
#
# ============================================================================

from PyQt6.QtCore import Qt, QSettings, QCoreApplication, QT_TRANSLATE_NOOP


def _tr(text: str) -> str:
    return QCoreApplication.translate("FirstRunDialog", text)


# ── Settings keys ────────────────────────────────────────────────────────────
_SETTINGS_KEY = "ui/first_run_dialog_shown_v1"      # seen on 1.x
WELCOME_KEY = "ui/welcome_shown_v2"                 # seen on 2.0 or later

SASPRO_TIPS = [
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Send all panels to the Secondary Dock Window via the Views menu — perfect for dual monitor setups."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Command Search finds literally anything in SASpro — open it first, then just start typing."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Right-click the menu bar to show and hide panels instantly."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Use the dotted handle at the start of any toolbar to drag it anywhere or dock it vertically."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Did you know you can create data Tables in SASpro? Check the File menu."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Alt+drag the View icon onto another image window to link their views together."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Press 'A' to toggle the display stretch on and off."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Shift+A makes the current display stretch permanent — bakes it into the image."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 You can export your icon shortcuts to save a custom toolset — like a narrowband set or planet processing set."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Alt+drag any function icon off the toolbar and drop it onto an image to run it instantly."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Hold Spacebar and click on an image to see pixel values and WCS coordinates at that point."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Ctrl+Shift+I quickly inverts your active mask."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 You can paste images directly into SASpro with Ctrl+V."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Create an ROI to preview how a tool will affect just a small region before committing."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Don't forget about History Explorer — every edit is tracked and reversible."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 The Blink Comparator lets you rapidly compare multiple images to spot differences."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Right-click on tree widgets, images, and panels — there are often extra options lurking there."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 SASpro remembers your panel positions and sizes — float them anywhere and they'll be there next time."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Right-click on most tree widgets in the Stacking Suite for extra options and session assignment."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 The wrench icon in the Stacking Suite opens all stacking settings including rejection algorithm tuning."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 You can drag the Stacking Suite log window to a second monitor and watch a run live."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 ABE (Automatic Background Extraction) supports manual sample point placement for tricky images."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 The Execution Monitor shows a live breakdown of every stacking phase with elapsed times."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Plate solving is under Analysis — SASpro tries ASTAP first with astrometry.net as fallback."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 What's In My Image (WIMI) shows every catalogued object in your field of view."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 What's In My Sky tells you what's observable from your location right now."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 The Pixel Math tool supports full expressions — add, subtract, and combine images with formulas."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 When you save a project, all your edits are preserved including the full undo/redo stacks!"),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 You can undo and redo edits all the way back to the original using the History stack."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Masks can be created from stars, luminance, ranges, colors, or drawn manually."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Drag the View icon off to a blank area of the canvas to duplicate your image into a new window."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 The Levels tool supports per-channel adjustments with live histogram feedback."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Some say Frank is actually an AI from the future sent back to help astrophotographers. We can neither confirm nor deny this. 🛸"),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 SASpro supports FITS, TIFF, XISF, JPG, PNG, Camera RAW, and even PDF as input formats."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 You can assign custom keyboard shortcuts to almost any function in SASpro."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 The Pipeline tab in the Stacking Suite lets you run a full stack with minimal setup."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 The mouse wheel zooms in and out on your image."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Did you know SASpro can do planetary stacking?"),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Use View Bundles to Batch process hundreds or thousands of images in one go!"),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Automate your entire processing workflow with Function Bundles and View Bundles — save and replay your full process in one click."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Did you know SASpro can project any galaxy to be viewed top-down? Check it out under the Functions menu."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Easily convert a mono image to a 3-channel color image under the Edit menu."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Curious about keyboard shortcuts? There's a full cheat sheet under the Shortcuts menu."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Bored? Check under the About menu. You're welcome. 🙂"),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Batch convert files from one format to another using Batch File Conversion under Header Mods & Misc."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Use Image Peeker to analyze your image and check if your sensor is too close to your reducer."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 The Dither Analysis tool visualizes your guiding drift and dither patterns after a stacking run."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 You can annotate your images with compass overlays, scale bars, and object labels directly from WIMI."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 The Stacking Suite supports comet stacking — it can track the comet and the stars separately and blend them."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 ABE supports exclusion polygons — draw around galaxies or nebulae so they don't affect the background model."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 You can right-click the toolbar to reset it if icons go missing or get rearranged."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 The Blink Comparator grades every frame against the rest of its target, filter and exposure. Sort by Grade, or press Pre-flag quality to mark the obvious rejects for review."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Star Alignment in the Stacking Suite has a Trial Detect button — use it to tune your star detection threshold before committing to a full run."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 SASpro supports session-based flat and light grouping in the Stacking Suite — right-click frames to assign them to sessions."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 You can override the master dark or flat for individual light groups in the Stacking Suite via right-click."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 The Resource Monitor shows live CPU, RAM, and GPU usage — find it under the Views menu."),    
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 You can batch crop all your open views at the same time from the crop tool."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 The Header Viewer shows all FITS keywords for any open image."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 You can open multiple images at once — or drag and drop multiple files directly onto SASpro."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 The Icon Cheat Sheet under Help shows every toolbar icon with its function at a glance."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 Press F1 (About → Documentation) to browse and search every SASpro help page."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 DarkStar separates stars from nebulosity — find it under Remove Stars."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 You can link zoom and pan between windows using the View Sync controls."),
    QT_TRANSLATE_NOOP("FirstRunDialog", "💡 The Pipeline tab gives a live summary of your loaded calibration files before you commit to a full stack."),
]

def _is_v2(main_window) -> bool:
    import os
    from setiastro.saspro import workspace_layout as wl
    return (os.environ.get(wl.FORCE_ENV) == "1"
            or wl.version_major(getattr(main_window, "_version", "")) >= wl.NEW_LAYOUT_MAJOR)


def welcome_due(main_window) -> bool:
    """The welcome hasn't been seen yet. Everyone sees it once on 2.0, however
    long they have used 1.x."""
    s = QSettings()
    if s.value(WELCOME_KEY, False, type=bool):
        return False
    return _is_v2(main_window) or not s.value(_SETTINGS_KEY, False, type=bool)


def mark_welcome_shown(main_window) -> None:
    s = QSettings()
    s.setValue(_SETTINGS_KEY, True)
    if _is_v2(main_window):
        s.setValue(WELCOME_KEY, True)
    s.sync()


def maybe_show_first_run_dialog(main_window) -> None:
    """Call this after the splash is gone. The welcome if it is due, otherwise
    a reminder in the corner if something is still to be set up."""
    from PyQt6.QtCore import QTimer
    try:
        from setiastro.saspro import setup_center
        if welcome_due(main_window):
            setup_center.open_setup_center(main_window)
        else:
            QTimer.singleShot(1500, lambda: setup_center.show_reminder(main_window))
    except Exception as e:
        print(f"[setup] the welcome failed to open: {e!r}")


def maybe_show_tip_of_day(main_window) -> None:
    """Show a rotating tip bar at the bottom of the main window, once per session."""
    from PyQt6.QtCore import QSettings
    s = QSettings()
    if s.value("tips/disabled", False, type=bool):
        return
    # Not while the welcome is up
    if welcome_due(main_window):
        return
    from PyQt6.QtCore import QTimer
    QTimer.singleShot(1500, lambda: _show_tip_bar(main_window))


def _show_tip_bar(main_window) -> None:
    import random
    from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton
    from PyQt6.QtCore import QSettings, QTimer

    s = QSettings()
    if s.value("tips/disabled", False, type=bool):
        return

    # Check if update dialog is showing
    if getattr(main_window, "_update_dialog_shown", False):
        QTimer.singleShot(2000, lambda: _show_tip_bar(main_window))
        return

    # One thing at a time: no tip while the First Steps tour is on screen.
    try:
        from setiastro.saspro.first_steps_tour import tour_is_showing
        if tour_is_showing(main_window):
            return
    except Exception:
        pass

    # Pick a tip not seen recently
    seen = list(s.value("tips/recently_seen", [], type=list) or [])
    available = [t for t in SASPRO_TIPS if t not in seen]
    if not available:
        seen = []
        available = SASPRO_TIPS[:]

    tip = random.choice(available)
    seen.append(tip)
    s.setValue("tips/recently_seen", seen[-10:])
    s.sync()

    # Build the bar
    bar = QWidget(main_window)
    bar.setObjectName("tip_bar")
    bar.setStyleSheet("""
        QWidget#tip_bar {
            background-color: rgba(20, 20, 40, 230);
            border-top: 1px solid #2a2a4a;
        }
    """)

    h = QHBoxLayout(bar)
    h.setContentsMargins(12, 5, 8, 5)
    h.setSpacing(10)

    lbl = QLabel(_tr(tip))
    lbl.setStyleSheet("color: #cccccc; font-size: 13px; font-weight: 500; background: transparent; border: none;")
    lbl.setWordWrap(False)
    h.addWidget(lbl, 1)

    btn_dontshow = QPushButton(_tr("Don't show tips"))
    btn_dontshow.setStyleSheet("""
        QPushButton {
            background: transparent;
            color: #555588;
            border: none;
            font-size: 10px;
            text-decoration: underline;
        }
        QPushButton:hover { color: #8888bb; }
    """)
    btn_dontshow.setFixedHeight(20)
    btn_dontshow.setCursor(Qt.CursorShape.PointingHandCursor)

    btn_dismiss = QPushButton("✕")
    btn_dismiss.setFixedSize(20, 20)
    btn_dismiss.setStyleSheet("""
        QPushButton {
            background: transparent;
            color: #555577;
            border: none;
            font-size: 11px;
        }
        QPushButton:hover { color: #aaaacc; }
    """)
    btn_dismiss.setCursor(Qt.CursorShape.PointingHandCursor)

    h.addWidget(btn_dontshow)
    h.addWidget(btn_dismiss)

    def _dismiss():
        try:
            bar.setVisible(False)
            bar.deleteLater()
        except RuntimeError:
            pass

    def _dont_show():
        s.setValue("tips/disabled", True)
        s.sync()
        _dismiss()

    btn_dismiss.clicked.connect(_dismiss)
    btn_dontshow.clicked.connect(_dont_show)

    # Position it at the bottom of the window, above the status bar
    try:
        bar.setFixedHeight(36)

        def _place_bottom():
            from PyQt6.QtWidgets import QDockWidget
            r = main_window.rect()

            sb = main_window.statusBar()
            sb_h = sb.height() if (sb is not None and sb.isVisible()) else 0

            # Sum heights of dock widgets currently docked in the BOTTOM area
            # (skip floating ones and hidden ones — they take no bottom space).
            dock_h = 0
            try:
                for dock in main_window.findChildren(QDockWidget):
                    if not dock.isVisible() or dock.isFloating():
                        continue
                    if main_window.dockWidgetArea(dock) == Qt.DockWidgetArea.BottomDockWidgetArea:
                        dock_h = max(dock_h, dock.height())
            except Exception:
                dock_h = 0

            bar.setFixedWidth(r.width())
            bar.move(0, r.height() - sb_h - dock_h - bar.height())

        _place_bottom()
        bar.raise_()
        bar.show()
        main_window._tip_bar = bar

        # Auto-dismiss after 20 seconds
        QTimer.singleShot(20000, lambda: _safe_dismiss())

        def _safe_dismiss():
            try:
                if bar.isVisible():
                    _dismiss()
            except RuntimeError:
                pass  # already deleted, no problem

        # Keep it pinned to the bottom on resize
        original_resize = main_window.resizeEvent
        def _patched_resize(event):
            try:
                original_resize(event)
                if bar.isVisible():
                    _place_bottom()
                    bar.raise_()
            except Exception:
                pass
        main_window.resizeEvent = _patched_resize

    except Exception:
        pass
