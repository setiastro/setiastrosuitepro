"""Register Qt's native Help menu with AppKit (no PyObjC dependency).

PyQt does not expose QMenu::toNSMenu(). Find the menu by its current title
in NSApplication.mainMenu instead, then use the public helpMenu property.
AppKit supplies search and menu-location highlighting itself.
"""
from __future__ import annotations

import ctypes
import logging
import sys

# Preferences checkbox, macOS only. Off puts the menus back inside the
# window as in 1.24.6: a notched MacBook has no room in the macOS menu bar
# for all of SASpro's menus, and macOS cuts their titles short.
NATIVE_MENU_BAR_KEY = "ui/native_menu_bar"


def use_native_menu_bar(settings) -> bool:
    """True when the menus go in the macOS menu bar (the macOS default)."""
    if sys.platform != "darwin":
        return False
    try:
        return bool(settings.value(NATIVE_MENU_BAR_KEY, True, type=bool))
    except Exception:
        return True


def register_help_menu(title: str) -> bool:
    """Call on the GUI thread after Qt has attached its native menu bar."""
    if sys.platform != "darwin":
        return False
    from PyQt6.QtGui import QGuiApplication
    if QGuiApplication.platformName() != "cocoa":
        return False  # e.g. an offscreen Qt test has no initialized AppKit app
    try:
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        # Separate typed callables: changing objc_msgSend's signature in place
        # would be unsafe if another caller used a different return type.
        address = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        obj = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        )(address)
        count = ctypes.CFUNCTYPE(
            ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p,
        )(address)
        at = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long,
        )(address)
        string = ctypes.CFUNCTYPE(
            ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p,
        )(address)
        set_menu = ctypes.CFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        )(address)
        sel = objc.sel_registerName
        app = obj(
            objc.objc_getClass(b"NSApplication"), sel(b"sharedApplication"),
        )
        main = obj(app, sel(b"mainMenu"))
        if not main:
            return False
        for index in range(count(main, sel(b"numberOfItems"))):
            item = at(main, sel(b"itemAtIndex:"), index)
            menu = obj(item, sel(b"submenu"))
            if not menu:
                continue
            label = obj(menu, sel(b"title"))
            raw = string(label, sel(b"UTF8String")) if label else None
            if raw is not None and raw.decode("utf-8") == title:
                if obj(app, sel(b"helpMenu")) != menu:
                    set_menu(app, sel(b"setHelpMenu:"), menu)
                return True
    except (OSError, AttributeError, UnicodeError):
        logging.getLogger(__name__).exception(
            "Could not register the native Help menu"
        )
    return False
