# src/setiastro/saspro/gui_entry.py
from __future__ import annotations

"""
GUI entry point for Seti Astro Suite Pro.

This module contains the heavy GUI startup path (splash, runtime bootstrap,
PyQt6 imports, main window creation). It must NOT be imported from the CLI path
unless the user is launching the GUI.
"""

import sys
import os
from pathlib import Path
# -*- coding: utf-8 -*-
"""
Seti Astro Suite Pro - Main Entry Point Module

This module contains the main application entry point logic.
It can be executed directly via `python -m setiastro.saspro` or
called via the `main()` function when invoked as an entry point.
"""

# Show splash screen IMMEDIATELY before any heavy imports

_STARTUP_PROFILE = os.environ.get("SASPRO_STARTUP_PROFILE", "").strip().lower() in ("1","true","yes","on")
_app_icon_obj = None
_app_icon_path = ""

def _is_wayland_session() -> bool:
    # Best-effort detection that does NOT require a QApplication instance
    xdg = (os.environ.get("XDG_SESSION_TYPE") or "").lower()
    if xdg == "wayland":
        return True
    # QT can still run wayland even if XDG_SESSION_TYPE isn't set
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    return False

def _allow_window_opacity_effects() -> bool:
    # User override: allow opacity effects even on Wayland if they want
    if os.environ.get("SASPRO_ALLOW_OPACITY", "").strip().lower() in ("1","true","yes","on"):
        return True
    # Default: disable opacity effects on Wayland
    return not _is_wayland_session()


from pathlib import Path

if sys.platform.startswith("win"):
    exe_dir = Path(sys.executable).resolve().parent
    internal = exe_dir / "_internal"
    if internal.is_dir():
        try:
            os.add_dll_directory(str(internal))
        except Exception:
            pass
        os.environ["PATH"] = str(internal) + os.pathsep + os.environ.get("PATH", "")
        


# ---- Linux Qt stability guard (must run BEFORE any PyQt6 import) ----
# Default behavior: DO NOT override Wayland.
# If a user needs the "safe" path, they can opt-in by setting:
#   SASPRO_QT_SAFE=1
#
# This avoids punishing all Wayland users for one bad driver/Qt stack.
if sys.platform.startswith("linux"):
    if os.environ.get("SASPRO_QT_SAFE", "").strip() in ("1", "true", "yes", "on"):
        # Prefer X11/xcb unless user explicitly set a platform plugin
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

        # Prefer software GL unless user explicitly set something else
        os.environ.setdefault("QT_OPENGL", "software")
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

from PyQt6.QtCore import QCoreApplication

# Global variables for splash screen and app
_splash = None
_app = None

# Flag to track if splash was initialized
_splash_initialized = False

# Flag to ensure heavy imports/bootstrap happens only once
_imports_bootstrapped = False

def _set_windows_appusermodelid(app_id: str) -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass

from setiastro.saspro.versioning import get_app_version
_EARLY_VERSION = get_app_version("setiastrosuitepro")

VERSION = _EARLY_VERSION

def _collect_open_paths(argv: list[str] | None) -> list[str]:
    if not argv:
        return []
    out: list[str] = []
    for a in argv:
        if not a or a.startswith("-"):
            continue
        # strip surrounding quotes if any
        s = a.strip().strip('"').strip("'")
        try:
            p = Path(s)
        except Exception:
            continue
        if p.exists() and p.is_file():
            out.append(str(p))
        else:
            print(f"[startup] ignoring arg (not a file): {p}  exists={p.exists()} is_file={p.is_file()}")

    return out

def _init_splash():
    """Initialize the splash screen. Safe to call multiple times."""
    global _splash, _app, _splash_initialized

    if _splash_initialized:
        return

    # --- Windows: set AppUserModelID as early as possible ---
    if sys.platform.startswith("win"):
        _set_windows_appusermodelid("SetiAstro.SetiAstroSuitePro")

    # Minimal imports for splash screen
    from PyQt6.QtWidgets import QApplication, QWidget
    from PyQt6.QtCore import Qt, QCoreApplication, QRect, QPropertyAnimation, QEasingCurve
    from PyQt6.QtGui import (
        QGuiApplication, QIcon, QPixmap, QColor, QPainter, QFont, QLinearGradient
    )
    import time

    # If we're forcing software OpenGL, do it *before* QApplication is created.
    if sys.platform.startswith("linux"):
        if os.environ.get("SASPRO_QT_SAFE", "").strip().lower() in ("1", "true", "yes", "on"):
            if os.environ.get("QT_OPENGL", "").lower() == "software":
                try:
                    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL, True)
                except Exception:
                    pass

    # Set application attributes before creating QApplication
    try:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass

    # Keep these exactly (QSettings/defaults depend on org/app names)
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, False)
    QCoreApplication.setOrganizationName("SetiAstro")
    QCoreApplication.setOrganizationDomain("setiastrosuite.pro")
    QCoreApplication.setApplicationName("Seti Astro Suite Pro")

    # ---------------------------------------------------------------------
    # Path resolution helpers (NO QIcon/QPixmap usage before QApplication)
    # ---------------------------------------------------------------------
    def _find_resource_paths_legacy() -> tuple[str, str, str | None]:
        """
        Returns (app_icon_path, splash_logo_path, startup_bg_path) using filesystem only.
        app_icon_path prefers ICO on Windows.
        splash_logo_path prefers PNG for large splash rendering.
        """
        if hasattr(sys, "_MEIPASS"):
            base = sys._MEIPASS
        else:
            try:
                import setiastro
                package_dir = os.path.dirname(os.path.abspath(setiastro.__file__))
                package_parent = os.path.dirname(package_dir)
                images_dir_installed = os.path.join(package_parent, "images")
                if os.path.exists(images_dir_installed):
                    base = package_parent
                else:
                    base = os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(
                                os.path.dirname(os.path.abspath(__file__))
                            )
                        )
                    )
            except Exception:
                base = os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(os.path.abspath(__file__))
                        )
                    )
                )

        images_dir = os.path.join(base, "images")

        # App icon (taskbar/tray/window icon): prefer ICO on Windows
        if sys.platform.startswith("win"):
            app_icon_candidates = [
                os.path.join(images_dir, "astrosuitepro.ico"),
                os.path.join(images_dir, "astrosuite.ico"),
                os.path.join(images_dir, "astrosuitepro.png"),
                os.path.join(images_dir, "astrosuite.png"),
            ]
        else:
            app_icon_candidates = [
                os.path.join(images_dir, "astrosuitepro.png"),
                os.path.join(images_dir, "astrosuitepro.ico"),
                os.path.join(images_dir, "astrosuite.png"),
                os.path.join(images_dir, "astrosuite.ico"),
            ]

        # Splash logo (large display): prefer PNG always
        splash_logo_candidates = [
            os.path.join(images_dir, "astrosuitepro.png"),
            os.path.join(images_dir, "astrosuite.png"),
            os.path.join(images_dir, "astrosuitepro.ico"),
            os.path.join(images_dir, "astrosuite.ico"),
        ]

        # Startup background (optional)
        startup_bg_candidates = [
            os.path.join(images_dir, "startup_background.png"),
            os.path.join(images_dir, "startup_bg.png"),
            os.path.join(images_dir, "splash_background.png"),
        ]

        app_icon_path = ""
        for p in app_icon_candidates:
            if os.path.exists(p):
                app_icon_path = p
                break

        splash_logo_path = ""
        for p in splash_logo_candidates:
            if os.path.exists(p):
                splash_logo_path = p
                break

        startup_bg_path = None
        for p in startup_bg_candidates:
            if os.path.exists(p):
                startup_bg_path = p
                break

        return app_icon_path, splash_logo_path, startup_bg_path

    def _resolve_early_paths() -> tuple[str, str, str | None]:
        """
        Returns (app_icon_path, splash_logo_path, startup_bg_path)
        using path existence only (NO Qt image/icon creation yet).
        """
        app_icon_path = ""
        splash_logo_path = ""
        startup_bg_path = None

        # Prefer centralized resources resolver if available
        try:
            from setiastro.saspro.resources import icon_path as res_icon_path, background_startup_path

            # app icon path from resources
            if res_icon_path and os.path.exists(res_icon_path):
                app_icon_path = res_icon_path

            # splash logo should prefer PNG for rendering
            # If resources.icon_path is ICO on Windows, try same stem .png first.
            if app_icon_path:
                root, ext = os.path.splitext(app_icon_path)
                png_try = root + ".png"
                if os.path.exists(png_try):
                    splash_logo_path = png_try
                else:
                    splash_logo_path = app_icon_path

            # startup background
            if background_startup_path and os.path.exists(background_startup_path):
                startup_bg_path = background_startup_path

        except Exception:
            pass

        # Legacy fallbacks
        legacy_app_icon, legacy_splash_logo, legacy_bg = _find_resource_paths_legacy()

        if not app_icon_path:
            app_icon_path = legacy_app_icon

        if not splash_logo_path:
            splash_logo_path = legacy_splash_logo or app_icon_path

        if not startup_bg_path:
            startup_bg_path = legacy_bg

        # Windows: for app icon only, prefer .ico sibling if current app icon is .png
        if sys.platform.startswith("win") and app_icon_path:
            root, ext = os.path.splitext(app_icon_path)
            if ext.lower() == ".png":
                ico_try = root + ".ico"
                if os.path.exists(ico_try):
                    app_icon_path = ico_try

        return app_icon_path, splash_logo_path, startup_bg_path

    # Resolve paths BEFORE QApplication and BEFORE any widgets (filesystem only)
    _early_app_icon_path, _early_splash_logo_path, _startup_bg_path = _resolve_early_paths()

    # ---------------------------
    # Create QApplication (Qt image/icon creation allowed after this)
    # ---------------------------
    _app = QApplication(sys.argv)

    try:
        _app.setQuitOnLastWindowClosed(True)
    except Exception:
        pass

    # Linux startup diagnostics (kept from your original working flow)
    if sys.platform.startswith("linux"):
        try:
            print("Qt platform:", _app.platformName())
            print("QuitOnLastWindowClosed:", _app.quitOnLastWindowClosed())
            print("XDG_SESSION_TYPE:", os.environ.get("XDG_SESSION_TYPE"))
            print("QT_QPA_PLATFORM:", os.environ.get("QT_QPA_PLATFORM"))
            print("QT_OPENGL:", os.environ.get("QT_OPENGL"))
        except Exception:
            pass


    # ------------------------------------------------------------------
    # IMPORTANT: Use the SAME SIMPLE APP ICON INIT FLOW as the version
    # that worked for the Windows taskbar icon.
    # ------------------------------------------------------------------
    global _app_icon_obj, _app_icon_path
    _app_icon_path = _early_app_icon_path
    _app_icon_obj = QIcon()
    try:
        if _early_app_icon_path and os.path.exists(_early_app_icon_path):
            _app_icon_obj = QIcon(_early_app_icon_path)
            if _app_icon_obj.isNull() and sys.platform.startswith("win"):
                # fallback to png sibling if ico failed
                root, _ext = os.path.splitext(_early_app_icon_path)
                png_try = root + ".png"
                if os.path.exists(png_try):
                    _app_icon_obj = QIcon(png_try)

            if not _app_icon_obj.isNull():
                _app.setWindowIcon(_app_icon_obj)
            else:
                print(f"[startup] WARNING: Qt failed to load app icon '{_early_app_icon_path}'")
    except Exception as e:
        print(f"[startup] WARNING: setWindowIcon failed: {e!r}")

    # =========================================================================
    # PhotoshopStyleSplash - Custom splash screen widget
    # =========================================================================
    class _EarlySplash(QWidget):
        """
        A modern, Photoshop-style splash screen shown immediately on startup.
        """
        def __init__(self, logo_path: str):
            super().__init__()
            self._version = _EARLY_VERSION
            self._build = ""
            self.current_message = QCoreApplication.translate("Splash", "Starting...")
            self.progress_value = 0

            self.setWindowFlags(
                Qt.WindowType.SplashScreen |
                Qt.WindowType.FramelessWindowHint |
                Qt.WindowType.WindowStaysOnTopHint
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

            self.splash_width = 600
            self.splash_height = 400
            self.setFixedSize(self.splash_width, self.splash_height)

            screen = QGuiApplication.primaryScreen()
            if screen:
                screen_geo = screen.availableGeometry()
                x = (screen_geo.width() - self.splash_width) // 2 + screen_geo.x()
                y = (screen_geo.height() - self.splash_height) // 2 + screen_geo.y()
                self.move(x, y)

            self.logo_pixmap = self._load_logo(logo_path)

            self.bg_image_pixmap = QPixmap()
            if _startup_bg_path:
                bg_pm = QPixmap(_startup_bg_path)
                if not bg_pm.isNull():
                    self.bg_image_pixmap = bg_pm
                else:
                    print(f"[startup] WARNING: splash background failed to load: {_startup_bg_path}")

            self.title_font = QFont("Segoe UI", 28, QFont.Weight.Bold)
            self.subtitle_font = QFont("Segoe UI", 11)
            self.message_font = QFont("Segoe UI", 9)
            self.copyright_font = QFont("Segoe UI", 8)

        def _load_logo(self, path: str) -> QPixmap:
            """Load splash logo. Prefer PNG path passed in; fallback robustly."""
            if not path or not os.path.exists(path):
                return QPixmap()

            ext = os.path.splitext(path)[1].lower()

            # For splash logo, PNG is preferred. ICO can work, but some ICOs look wrong when enlarged.
            if ext == ".ico":
                ic = QIcon(path)
                pm = ic.pixmap(256, 256)
                if pm.isNull():
                    pm = ic.pixmap(128, 128)
                if pm.isNull():
                    pm = QPixmap(path)
            else:
                pm = QPixmap(path)
                if pm.isNull():
                    pm = QIcon(path).pixmap(256, 256)

            if not pm.isNull():
                pm = pm.scaled(
                    180, 180,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            else:
                print(f"[startup] WARNING: splash logo failed to load: {path}")

            return pm

        def setMessage(self, message: str):
            self.current_message = message
            self.repaint()
            if _app:
                _app.processEvents()

        def setProgress(self, value: int):
            target = max(0, min(100, int(value)))
            start = float(self.progress_value)

            if target <= start or (target - start) < 1:
                self.progress_value = target
                self.repaint()
                if _app:
                    _app.processEvents()
                return

            steps = 15
            dt = 0.005
            for i in range(1, steps + 1):
                t = i / steps
                factor = -t * (t - 2)  # quadratic ease-out
                cur = start + (target - start) * factor
                self.progress_value = cur
                self.repaint()
                if _app:
                    _app.processEvents()
                time.sleep(dt)

            self.progress_value = target
            self.repaint()
            if _app:
                _app.processEvents()

        def setBuildInfo(self, version: str, build: str):
            self._version = version or _EARLY_VERSION
            self._build = build
            self.repaint()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

            w, h = self.splash_width, self.splash_height

            gradient = QLinearGradient(0, 0, 0, h)
            gradient.setColorAt(0.0, QColor(15, 15, 25))
            gradient.setColorAt(0.5, QColor(25, 25, 45))
            gradient.setColorAt(1.0, QColor(10, 10, 20))
            painter.fillRect(0, 0, w, h, gradient)

            if not self.bg_image_pixmap.isNull():
                temp = QPixmap(w, h)
                temp.fill(Qt.GlobalColor.transparent)

                ptmp = QPainter(temp)
                ptmp.setRenderHint(QPainter.RenderHint.Antialiasing)
                ptmp.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

                scaled = self.bg_image_pixmap.scaled(
                    w, h,
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation
                )

                sx = (w - scaled.width()) // 2
                sy = (h - scaled.height()) // 2
                ptmp.drawPixmap(sx, sy, scaled)

                ptmp.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
                fade_gradient = QLinearGradient(0, 0, 0, h)
                fade_gradient.setColorAt(0.0, QColor(0, 0, 0, 255))
                fade_gradient.setColorAt(0.5, QColor(0, 0, 0, 255))
                fade_gradient.setColorAt(1.0, QColor(0, 0, 0, 0))
                ptmp.fillRect(0, 0, w, h, fade_gradient)
                ptmp.end()

                painter.save()
                painter.setOpacity(0.25)
                painter.drawPixmap(0, 0, temp)
                painter.restore()

            painter.setPen(QColor(60, 60, 80))
            painter.drawRect(0, 0, w - 1, h - 1)

            if not self.logo_pixmap.isNull():
                logo_x = (w - self.logo_pixmap.width()) // 2
                logo_y = 40
                painter.drawPixmap(logo_x, logo_y, self.logo_pixmap)

            painter.setFont(self.title_font)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(QRect(0, 230, w, 40), Qt.AlignmentFlag.AlignCenter, "Seti Astro Suite Pro")

            painter.setFont(self.subtitle_font)
            painter.setPen(QColor(180, 180, 200))
            subtitle_text = QCoreApplication.translate("Splash", "Version {0}").format(self._version)
            if self._build:
                if self._build == "dev":
                    subtitle_text += QCoreApplication.translate("Splash", "  •  Running locally from source code")
                else:
                    subtitle_text += QCoreApplication.translate("Splash", "  •  Build {0}").format(self._build)
            painter.drawText(QRect(0, 270, w, 25), Qt.AlignmentFlag.AlignCenter, subtitle_text)

            bar_margin = 50
            bar_height = 4
            bar_y = h - 70
            bar_width = w - (bar_margin * 2)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(40, 40, 60))
            painter.drawRoundedRect(bar_margin, bar_y, bar_width, bar_height, 2, 2)

            if self.progress_value > 0:
                fill_width = int(bar_width * float(self.progress_value) / 100.0)
                fill_width = max(0, min(bar_width, fill_width))
                if fill_width > 0:
                    bar_gradient = QLinearGradient(bar_margin, 0, bar_margin + bar_width, 0)
                    bar_gradient.setColorAt(0.0, QColor(80, 140, 220))
                    bar_gradient.setColorAt(1.0, QColor(140, 180, 255))
                    painter.setBrush(bar_gradient)
                    painter.drawRoundedRect(bar_margin, bar_y, fill_width, bar_height, 2, 2)

            painter.setFont(self.message_font)
            painter.setPen(QColor(150, 150, 180))
            painter.drawText(
                QRect(bar_margin, bar_y + 10, bar_width, 20),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self.current_message
            )

            painter.setFont(self.copyright_font)
            painter.setPen(QColor(100, 100, 130))
            painter.drawText(
                QRect(0, h - 30, w, 20),
                Qt.AlignmentFlag.AlignCenter,
                "© 2024-2026 Franklin Marek (Seti Astro)  •  All Rights Reserved"
            )

            painter.end()

        def finish(self):
            self.hide()
            self.close()
            self.deleteLater()

        def start_fade_out(self, on_finished=None):
            if not _allow_window_opacity_effects():
                self.finish()
                if callable(on_finished):
                    on_finished()
                return

            self._anim = QPropertyAnimation(self, b"windowOpacity")
            self._anim.setDuration(1000)
            self._anim.setStartValue(1.0)
            self._anim.setEndValue(0.0)
            self._anim.setEasingCurve(QEasingCurve.Type.OutQuad)

            def _done():
                self.finish()
                if callable(on_finished):
                    on_finished()

            self._anim.finished.connect(_done)
            self._anim.start()

        def start_fade_in(self):
            if not _allow_window_opacity_effects():
                self.setWindowOpacity(1.0)
                return
            self.setWindowOpacity(0.0)
            self._anim = QPropertyAnimation(self, b"windowOpacity")
            self._anim.setDuration(800)
            self._anim.setStartValue(0.0)
            self._anim.setEndValue(1.0)
            self._anim.setEasingCurve(QEasingCurve.Type.InQuad)
            self._anim.start()

    # Create splash AFTER app icon is set (and use splash-logo path, not app icon path)
    _splash = _EarlySplash(_early_splash_logo_path)

    # Set splash window icon too (shell/task switcher help)
    try:
        if not _app_icon_obj.isNull():
            _splash.setWindowIcon(_app_icon_obj)
    except Exception:
        pass

    _splash.start_fade_in()
    _splash.show()

    # Allow fade-in to progress before heavy imports start
    t_start = time.time()
    while time.time() - t_start < 0.85:
        _app.processEvents()
        try:
            if _allow_window_opacity_effects() and _splash.windowOpacity() >= 0.99:
                break
        except Exception:
            break
        time.sleep(0.01)

    _splash.setMessage(QCoreApplication.translate("Splash", "Initializing Python runtime..."))
    _splash.setProgress(2)
    _app.processEvents()

    # Load translation BEFORE any other widgets are created
    try:
        from setiastro.saspro.i18n import load_language, get_translations_dir  # keep same import shape as original
        _ = get_translations_dir  # avoid lint complaints if unused
        ok = load_language(app=_app)
    except Exception as e:
        print("i18n load failed:", repr(e))

    _splash_initialized = True

# =============================================================================
# Now proceed with all the heavy imports (splash is visible)
# =============================================================================

# Helper to update splash during imports
def _update_splash(msg: str, progress: int):
    global _splash
    if _splash is not None:
        _splash.setMessage(msg)
        _splash.setProgress(progress)

def _bootstrap_imports():
    """
    Heavy imports + runtime bootstrap.
    This must NOT run at module import time, only when main() is called.
    """
    global _imports_bootstrapped
    if _imports_bootstrapped:
        return
    _imports_bootstrapped = True

    # Make sure splash exists before we start doing heavy work
    if not _splash_initialized:
        _init_splash()

    _update_splash(QCoreApplication.translate("Splash", "Loading PyTorch runtime..."), 5)

    from setiastro.saspro.runtime_torch import (
        add_runtime_to_sys_path,
        _ban_shadow_torch_paths,
        _purge_bad_torch_from_sysmodules,
    )

    add_runtime_to_sys_path(status_cb=lambda *_: None)
    _ban_shadow_torch_paths(status_cb=lambda *_: None)
    _purge_bad_torch_from_sysmodules(status_cb=lambda *_: None)
    _update_splash(QCoreApplication.translate("Splash", "Preparing AI runtime cache..."), 7)
    try:
        from setiastro.saspro.runtime_torch import prewarm_torch_cache
        prewarm_torch_cache(
            status_cb=lambda *_: None,   # keep console clean during splash
            require_torchaudio=True,
            ensure_venv=True,
            ensure_numpy=False,
            validate_marker=True,
        )
    except Exception:
        pass
    _update_splash(QCoreApplication.translate("Splash", "Loading standard libraries..."), 10)

    # ----------------------------------------
    # Standard library imports (consolidated)
    # ----------------------------------------
    import importlib
    import json
    import logging
    import math
    import multiprocessing
    import os
    import re
    import sys
    import threading
    import time
    import traceback
    import warnings
    import webbrowser

    from collections import defaultdict
    from datetime import datetime
    from decimal import getcontext
    from io import BytesIO
    from itertools import combinations
    from math import isnan
    from pathlib import Path
    from typing import Dict, List, Optional, Set, Tuple
    from urllib.parse import quote, quote_plus

    _update_splash(QCoreApplication.translate("Splash", "Loading NumPy..."), 15)

    # ----------------------------------------
    # Third-party imports
    # ----------------------------------------
    # numpy deferred to modules that need it


    # Image libraries (tifffile, XISF) deferred


    _update_splash(QCoreApplication.translate("Splash", "Configuring matplotlib..."), 25)
    from setiastro.saspro.config_bootstrap import ensure_mpl_config_dir
    _MPL_CFG_DIR = ensure_mpl_config_dir()

    # Apply metadata patches for frozen builds
    from setiastro.saspro.metadata_patcher import apply_metadata_patches
    apply_metadata_patches()

    warnings.filterwarnings(
        "ignore",
        message=r"Call to deprecated function \(or staticmethod\) _destroy\.",
        category=DeprecationWarning,
    )

    # Prevent lightkurve from applying mpl styles automatically
    os.environ["LIGHTKURVE_STYLE"] = "default"

    # ----------------------------------------
    # Matplotlib configuration (HARD NO-TeX)
    # ----------------------------------------
    import matplotlib

    # Backend selection must happen before pyplot is imported anywhere.
    try:
        matplotlib.use("QtAgg", force=True)
    except TypeError:
        # Older mpl may not have force=
        matplotlib.use("QtAgg")

    # Force MPL to NOT use LaTeX (prevents texmanager subprocess calls)
    try:
        matplotlib.rcParams["text.usetex"] = False
        matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
    except Exception:
        pass

    # If any style got applied earlier (rare), this reasserts our policy.
    def _force_mpl_no_tex():
        try:
            matplotlib.rcParams["text.usetex"] = False
            matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
            matplotlib.rcParams["font.family"] = "DejaVu Sans"
        except Exception:
            pass

    # HARD disable TeX (do NOT auto-enable even if latex exists)
    _force_mpl_no_tex()

    # Configure stdout encoding
    if (sys.stdout is not None) and (hasattr(sys.stdout, "reconfigure")):
        sys.stdout.reconfigure(encoding='utf-8')

    # --- Lazy imports for heavy dependencies (performance optimization) ---
    # photutils: loaded on first use
    global _photutils_isophote
    _photutils_isophote = None

    def _get_photutils_isophote():
        global _photutils_isophote
        if _photutils_isophote is None:
            try:
                from photutils import isophote as _isophote_module
                _photutils_isophote = _isophote_module
            except Exception:
                _photutils_isophote = False
        return _photutils_isophote if _photutils_isophote else None

    def get_Ellipse():
        mod = _get_photutils_isophote()
        return mod.Ellipse if mod else None

    def get_EllipseGeometry():
        mod = _get_photutils_isophote()
        return mod.EllipseGeometry if mod else None

    def get_build_ellipse_model():
        mod = _get_photutils_isophote()
        return mod.build_ellipse_model if mod else None

    global _lightkurve_module
    _lightkurve_module = None

    def get_lightkurve():
        global _lightkurve_module
        if _lightkurve_module is None:
            try:
                import lightkurve as _lk
                # prevent any implicit style behavior
                _lk.MPLSTYLE = None
                _lightkurve_module = _lk
            except Exception:
                _lightkurve_module = False

            # Re-assert no-TeX after any potential style meddling
            _force_mpl_no_tex()

        return _lightkurve_module if _lightkurve_module else None


    _update_splash(QCoreApplication.translate("Splash", "Loading UI utilities..."), 30)

    from setiastro.saspro.widgets.common_utilities import (
        AboutDialog,
        ProjectSaveWorker as _ProjectSaveWorker,
        DECOR_GLYPHS,
        _strip_ui_decorations,
        install_crash_handlers,
    )

    # reproject deferred


    # OpenCV deferred


    _update_splash(QCoreApplication.translate("Splash", "Loading PyQt6 components..."), 45)

    from PyQt6 import sip

    from PyQt6.QtWidgets import (QDialog, QApplication, QMainWindow, QWidget, QHBoxLayout, QFileDialog, QMessageBox, QSizePolicy, QToolBar, QPushButton, QAbstractItemDelegate,
        QLineEdit, QMenu, QListWidget, QListWidgetItem, QSplashScreen, QDockWidget, QListView, QCompleter, QMdiArea, QMdiSubWindow, QWidgetAction, QAbstractItemView,
        QInputDialog, QVBoxLayout, QLabel, QCheckBox, QProgressBar, QProgressDialog, QGraphicsItem, QTabWidget, QTableWidget, QHeaderView, QTableWidgetItem, QToolButton, QPlainTextEdit
    )

    from PyQt6.QtGui import (QPixmap, QColor, QIcon, QKeySequence, QShortcut, QGuiApplication, QStandardItemModel, QStandardItem, QAction, QPalette, QBrush, QActionGroup, QDesktopServices, QFont, QTextCursor
    )

    from PyQt6.QtCore import (Qt, pyqtSignal, QTimer, QSize, QSignalBlocker, QModelIndex, QThread, QUrl, QSettings, QEvent, QByteArray, QObject,
        QPropertyAnimation, QEasingCurve
    )


    from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply

    global BUILD_TIMESTAMP
    try:
        from setiastro.saspro._generated.build_info import BUILD_TIMESTAMP
    except Exception:
        BUILD_TIMESTAMP = "dev"

    _update_splash(QCoreApplication.translate("Splash", "Loading resources..."), 50)

    from setiastro.saspro.resources import (
        icon_path, windowslogo_path, # green_path, neutral_path, whitebalance_path,
        #morpho_path, clahe_path, starnet_path, staradd_path, LExtract_path,
        #LInsert_path, slot0_path, slot1_path, slot2_path, slot3_path, slot4_path,
        #rgbcombo_path, rgbextract_path, copyslot_path, graxperticon_path,
        #cropicon_path, openfile_path, abeicon_path, undoicon_path, redoicon_path,
        #blastericon_path, hdr_path, invert_path, fliphorizontal_path,
        #flipvertical_path, rotateclockwise_path, rotatecounterclockwise_path,
        #rotate180_path, maskcreate_path, maskapply_path, maskremove_path,
        #slot5_path, slot6_path, slot7_path, slot8_path, slot9_path, pixelmath_path,
        #histogram_path, mosaic_path, rescale_path, staralign_path, mask_path,
        #platesolve_path, psf_path, supernova_path, starregistration_path,
        #stacking_path, pedestal_icon_path, starspike_path, aperture_path,
        #jwstpupil_path, signature_icon_path, livestacking_path, hrdiagram_path,
        #convoicon_path, spcc_icon_path, sasp_data_path, exoicon_path, peeker_icon,
        #dse_icon_path, astrobin_filters_csv_path, isophote_path, statstretch_path,
        #starstretch_path, curves_path, disk_path, uhs_path, blink_path, ppp_path,
        #nbtorgb_path, freqsep_path, contsub_path, halo_path, cosmic_path,
        #satellite_path, imagecombine_path, wrench_path, eye_icon_path,
        #disk_icon_path, nuke_path, hubble_path, collage_path, annotated_path,
        #colorwheel_path, font_path, csv_icon_path, spinner_path, wims_path,
        #wimi_path, linearfit_path, debayer_path, aberration_path,
        #functionbundles_path, viewbundles_path, selectivecolor_path, rgbalign_path,
    )

    _update_splash(QCoreApplication.translate("Splash", "Configuring Qt message handler..."), 55)

    from PyQt6.QtCore import qInstallMessageHandler, QtMsgType

    def _qt_msg_handler(mode, ctx, msg):
        lvl = {
            QtMsgType.QtDebugMsg:    logging.DEBUG,
            QtMsgType.QtInfoMsg:     logging.INFO,
            QtMsgType.QtWarningMsg:  logging.WARNING,
            QtMsgType.QtCriticalMsg: logging.ERROR,
            QtMsgType.QtFatalMsg:    logging.CRITICAL,
        }.get(mode, logging.ERROR)
        logging.log(lvl, "Qt: %s (%s:%s)", msg, getattr(ctx, "file", "?"), getattr(ctx, "line", -1))

    qInstallMessageHandler(_qt_msg_handler)

    _update_splash(QCoreApplication.translate("Splash", "Loading MDI widgets..."), 60)

    from setiastro.saspro.mdi_widgets import (
        MdiArea, ViewLinkController, ConsoleListWidget, QtLogStream, _DocProxy,
        ROLE_ACTION as _ROLE_ACTION,
    )

    from setiastro.saspro.main_helpers import (
        safe_join_dir_and_name as _safe_join_dir_and_name,
        normalize_save_path_chosen_filter as _normalize_save_path_chosen_filter,
        display_name as _display_name,
        best_doc_name as _best_doc_name,
        doc_looks_like_table as _doc_looks_like_table,
        is_alive as _is_alive,
        safe_widget as _safe_widget,
    )

    from setiastro.saspro.file_utils import (
        _normalize_ext,
        _sanitize_filename,
        _exts_from_filter,
        REPLACE_SPACES_WITH_UNDERSCORES as _REPLACE_SPACES_WITH_UNDERSCORES,
        WIN_RESERVED_NAMES as _WIN_RESERVED,
    )

    _update_splash(QCoreApplication.translate("Splash", "Loading main window module..."), 65)

    from setiastro.saspro.gui.main_window import AstroSuiteProMainWindow

    _update_splash(QCoreApplication.translate("Splash", "Modules loaded, finalizing..."), 70)

    # Export things main() already relies on as globals (min disruption)
    globals().update({
        "logging": logging,
        "multiprocessing": multiprocessing,
        "warnings": warnings,
        "QIcon": QIcon,
        "QMessageBox": QMessageBox,
        "QPropertyAnimation": QPropertyAnimation,
        "QEasingCurve": QEasingCurve,
        "windowslogo_path": windowslogo_path,
        "icon_path": icon_path,
        "install_crash_handlers": install_crash_handlers,
        "AstroSuiteProMainWindow": AstroSuiteProMainWindow,
        "BUILD_TIMESTAMP": BUILD_TIMESTAMP,
        "_force_mpl_no_tex": _force_mpl_no_tex,
    })



def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for Seti Astro Suite Pro.
    
    This function can be called from:
    - The package entry point (setiastrosuitepro command)
    - Direct import and call
    - When running as a module: python -m setiastro.saspro
    """
    global _splash, _app, _splash_initialized
    from PyQt6.QtCore import QTimer

    open_paths = _collect_open_paths(argv)
    
    # Initialize splash if not already done
    if not _splash_initialized:
        _init_splash()

    _bootstrap_imports()
    # Update splash with build info now that we have VERSION and BUILD_TIMESTAMP
    if _splash:
        _splash.setBuildInfo(VERSION, BUILD_TIMESTAMP)
        _splash.setMessage(QCoreApplication.translate("Splash", "Setting up logging..."))
        _splash.setProgress(72)
    
    # --- Logging (catch unhandled exceptions to a file) ---
    import tempfile
    from pathlib import Path
 
    # Cross-platform log file location
    def get_log_file_path():
        """Get appropriate log file path for the current platform."""
        
        if hasattr(sys, '_MEIPASS'):
            # Running in PyInstaller bundle - use platform-appropriate user directory
            if sys.platform.startswith('win'):
                # Windows: %APPDATA%\SetiAstroSuitePro\logs\
                log_dir = Path(os.path.expandvars('%APPDATA%')) / 'SetiAstroSuitePro' / 'logs'
            elif sys.platform.startswith('darwin'):
                # macOS: ~/Library/Logs/SetiAstroSuitePro/
                log_dir = Path.home() / 'Library' / 'Logs' / 'SetiAstroSuitePro'
            else:
                # Linux: ~/.local/share/SetiAstroSuitePro/logs/
                log_dir = Path.home() / '.local' / 'share' / 'SetiAstroSuitePro' / 'logs'
            
            # Create directory if it doesn't exist
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / 'saspro.log'
            except (OSError, PermissionError):
                # Fallback to temp directory if user directory fails
                log_file = Path(tempfile.gettempdir()) / 'saspro.log'
        else:
            # Development mode - use logs folder in project
            log_dir = Path('logs')
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / 'saspro.log'
            except (OSError, PermissionError):
                log_file = Path('saspro.log')
        
        return str(log_file)
    
    # Configure logging with cross-platform path
    log_file_path = get_log_file_path()

    try:
        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filemode='a'  # Append mode
        )
        logging.info(f"Logging to: {log_file_path}")
        logging.info(f"Platform: {sys.platform}")
        logging.info(f"PyInstaller bundle: {hasattr(sys, '_MEIPASS')}")
    except Exception as e:
        # Ultimate fallback - console only logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        print(f"Warning: Could not write to log file {log_file_path}: {e}")
        print("Using console-only logging")
        

    # Setup crash handlers and app icon
    if _splash:
        _splash.setMessage(QCoreApplication.translate("Splash", "Installing crash handlers..."))
        _splash.setProgress(75)
    install_crash_handlers(_app) 
    try:
        if _app_icon_obj is not None and not _app_icon_obj.isNull():
            _app.setWindowIcon(_app_icon_obj)
    except Exception:
        pass


    # --- Windows exe / multiprocessing friendly ---
    if _splash:
        _splash.setMessage(QCoreApplication.translate("Splash", "Configuring multiprocessing..."))
        _splash.setProgress(78)
    try:
        multiprocessing.freeze_support()
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            # Already set in this interpreter
            pass
    except Exception:
        logging.exception("Multiprocessing init failed (continuing).")

    try:
        if _splash:
            _splash.setMessage(QCoreApplication.translate("Splash", "Loading image manager..."))
            _splash.setProgress(80)
        from setiastro.saspro.legacy.image_manager import ImageManager
        
        if _splash:
            _splash.setMessage(QCoreApplication.translate("Splash", "Suppressing warnings..."))
            _splash.setProgress(82)
        from matplotlib import MatplotlibDeprecationWarning
        warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

        if _splash:
            _splash.setMessage(QCoreApplication.translate("Splash", "Creating image manager..."))
            _splash.setProgress(85)
        imgr = ImageManager(max_slots=100)
        
        if _splash:
            _splash.setMessage(QCoreApplication.translate("Splash", "Building main window..."))
            _splash.setProgress(90)
        win = AstroSuiteProMainWindow(
            image_manager=imgr,
            version=VERSION,
            build_timestamp=BUILD_TIMESTAMP,
        )
        try:
            if _app_icon_obj is not None and not _app_icon_obj.isNull():
                win.setWindowIcon(_app_icon_obj)
        except Exception:
            pass
        def _kick_updates_after_splash():
            try:
                win.raise_()
                win.activateWindow()
                if win.settings.value("updates/check_on_startup", True, type=bool):
                    QTimer.singleShot(1000, win.check_for_updates_startup)
            except Exception:
                pass

        if _splash:
            _splash.setMessage(QCoreApplication.translate("Splash", "Showing main window..."))
            _splash.setProgress(95)
        
        # --- Smooth Transition: App Fade In + Splash Fade Out ---
        # MITIGATION: Prevent "White Flash" on startup
        # 1. Force a dark background immediately so if opacity lags, it's dark not white
        win.setStyleSheet("QMainWindow { background-color: #0F0F19; }")
        win.winId()

        if _allow_window_opacity_effects():
            win.setWindowOpacity(0.0)

        win.show()
        QTimer.singleShot(0, lambda: win.setWindowIcon(_app.windowIcon()))
        
        if open_paths:
            def _open_cli_paths():
                dm = getattr(win, "docman", None) or getattr(win, "doc_manager", None)
                if dm is None:
                    print("[startup open] No doc manager on main window")
                    return
                for p in open_paths:
                    try:
                        dm.open_path(p)
                    except Exception as e:
                        print("[startup open] failed:", p, e)

            QTimer.singleShot(0, _open_cli_paths)

        if _allow_window_opacity_effects():
            anim_app = QPropertyAnimation(win, b"windowOpacity")
            anim_app.setDuration(1200)
            anim_app.setStartValue(0.0)
            anim_app.setEndValue(1.0)
            anim_app.setEasingCurve(QEasingCurve.Type.OutQuad)

            def _on_fade_in_finished():
                win.setStyleSheet("")
                if hasattr(win, "on_fade_in_complete"):
                    win.on_fade_in_complete()

            anim_app.finished.connect(_on_fade_in_finished)
            anim_app.start()
        else:
            # No opacity animation on Wayland: just show immediately and clear the temp stylesheet
            win.setStyleSheet("")
            if hasattr(win, "on_fade_in_complete"):
                win.on_fade_in_complete()
        

        # Start background Numba warmup after UI is visible
        try:
            from setiastro.saspro.numba_warmup import start_background_warmup
            start_background_warmup()
        except Exception:
            pass  # Non-critical if warmup fails

        if _splash:
            _splash.setMessage(QCoreApplication.translate("Splash", "Ready!"))
            _splash.setProgress(100)
            _app.processEvents()
            
            # Small delay to ensure "Ready!" is seen briefly before fade starts
            import time
            time.sleep(0.1)
            
            # 2. Animate Splash Fade Out
            # Note: We do NOT use finish() directly here. The animation calls it when done.
            _splash.start_fade_out(on_finished=_kick_updates_after_splash)
            
            # NOTE: We keep a reference to _splash (global) so it doesn't get GC'd during animation.
            # It will deleteLater() itself.
        
        if BUILD_TIMESTAMP == "dev":
            build_label = "running from local source code"
        else:
            build_label = f"build {BUILD_TIMESTAMP}"

        print(f"Seti Astro Suite Pro v{VERSION} ({build_label}) up and running!")
        sys.exit(_app.exec())

    except Exception:
        import traceback
        if _splash:
            try:
                _splash.hide()
                _splash.close()
                _splash.deleteLater()
            except Exception:
                pass
        tb = traceback.format_exc()
        logging.error("Unhandled exception occurred\n%s", tb)
        msg = QMessageBox(None)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle(QCoreApplication.translate("Main", "Application Error"))
        msg.setText(QCoreApplication.translate("Main", "An unexpected error occurred."))
        msg.setInformativeText(tb.splitlines()[-1] if tb else "See details.")
        msg.setDetailedText(tb)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
        sys.exit(1)


# When run as a module, execute main()
if __name__ == "__main__":
    main()