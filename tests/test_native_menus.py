"""Menu construction without loading the image-processing/runtime stack.

These tests exercise QAction wiring, not AppKit's search field. The latter
requires a manual test on macOS with the cocoa platform plugin.
"""
from types import SimpleNamespace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from PyQt6.QtGui import QAction
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidgetAction

from setiastro.saspro.gui.macos_help import register_help_menu

# The mixins package eagerly imports all other GUI mixins, including the
# scientific runtime. Load this actual source module independently so the
# focused suite can run with Qt alone on each operating system.
_spec = spec_from_file_location(
    "menu_mixin_qa",
    Path(__file__).parents[1]
    / "src/setiastro/saspro/gui/mixins/menu_mixin.py",
)
menu_mixin = module_from_spec(_spec)
_spec.loader.exec_module(menu_mixin)


class MenuWindow(menu_mixin.MenuMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.group = None
        self.palette_requests = 0
        self.scriptman = SimpleNamespace(rebuild_menu=lambda menu: None)
        self.shortcuts = SimpleNamespace(clear=lambda: None)

    def _rebuild_recent_menus(self):
        pass

    def _show_command_palette(self):
        self.palette_requests += 1

    def __getattr__(self, name):
        # Supply the host's existing actions/callbacks, retaining actual Qt
        # objects and the production menu-building and synchronization code.
        if (
            name.startswith("act_")
            or name.startswith("actCosmic")
            or name in ("actAberrationAI", "actSyQonTools")
        ):
            action = QAction(name, self)
            setattr(self, name, action)
            return action
        if name.startswith("_") or name == "update_undo_redo_action_labels":
            return lambda *args: None
        raise AttributeError(name)

    def _current_group_of_active(self):
        return self.group

    def _set_group_for_active(self, group):
        self.group = group


def _menu(window, title):
    return next(
        a.menu() for a in window.menuBar().actions() if a.text() == title
    )


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_help_registration_skips_non_cocoa_platforms(qapp):
    if qapp.platformName() == "cocoa":
        pytest.skip("This check covers offscreen/non-Cocoa Qt platforms")
    assert register_help_menu("Help") is False


@pytest.mark.parametrize("platform", ["darwin", "win32", "linux"])
def test_menus_and_roles(qapp, monkeypatch, platform):
    monkeypatch.setattr(menu_mixin, "sys", SimpleNamespace(platform=platform))
    window = MenuWindow()
    window._init_menubar()
    help_menu = _menu(window, "&Help" if platform == "darwin" else "&About")
    assert window.act_docs in help_menu.actions()
    about = next(a for a in help_menu.actions() if a.text() == "About...")
    assert about.menuRole() == QAction.MenuRole.AboutRole
    preferences = next(
        a for a in _menu(window, "&Settings").actions()
        if a.text() == "Preferences..."
    )
    assert preferences.menuRole() == QAction.MenuRole.PreferencesRole

    search = next(
        a for a in help_menu.actions() if a.text() == "Search Commands..."
    )
    assert search.shortcut().toString() == "Ctrl+/"
    search.trigger()
    assert window.palette_requests == 1
    window.show()
    window.activateWindow()
    window.setFocus()
    qapp.processEvents()
    QTest.keySequence(window, search.shortcut())
    qapp.processEvents()
    assert window.palette_requests == 2

    view = _menu(window, "&View")
    if platform == "darwin":
        assert not any(isinstance(a, QWidgetAction) for a in view.actions())
        assert window.act_link_group in view.actions()
        groups = next(
            a.menu() for a in view.actions() if a.text() == "Link Group"
        )
    else:
        assert window._link_btn is not None
        groups = window._link_btn.menu()
    group_b = next(a for a in groups.actions() if a.text() == "Group B")
    group_b.trigger()
    assert window.group == "B"
    groups.aboutToShow.emit()
    assert group_b.isChecked()
    assert sum(a.isChecked() for a in groups.actions()) == 1
    groups.actions()[0].trigger()
    assert window.group is None
    window.close()


@pytest.mark.parametrize('focus_kind', ['edit', 'text', 'spin', 'image'])
@pytest.mark.parametrize('keys', ['Ctrl+/', 'Ctrl+Num+/'])
def test_palette_shortcut_from_child_widgets_after_menu_rebuild(
        qapp, focus_kind, keys):
    from PyQt6.QtGui import QKeySequence
    from PyQt6.QtWidgets import (
        QLineEdit, QTextEdit, QSpinBox, QMdiArea, QLabel,
    )

    window = MenuWindow()
    window._init_menubar()
    action = window._command_palette_action
    if focus_kind == 'image':
        mdi = QMdiArea(window)
        window.setCentralWidget(mdi)
        sub = mdi.addSubWindow(QLabel('Synthetic image'))
        sub.show()
        target = sub.widget()
    else:
        cls = {'edit': QLineEdit, 'text': QTextEdit, 'spin': QSpinBox}
        target = cls[focus_kind](window)
        window.setCentralWidget(target)
    window.show()
    window.activateWindow()
    target.setFocus()
    qapp.processEvents()
    QTest.keySequence(target, QKeySequence(keys))
    assert window.palette_requests == 1
    window.menuBar().clear()
    window._init_menubar()
    assert window._command_palette_action is action
    target.setFocus()
    qapp.processEvents()
    QTest.keySequence(target, QKeySequence(keys))
    assert window.palette_requests == 2
    window.close()


@pytest.mark.parametrize('layout', ['00000409', '00000415'])
def test_windows_native_slash_keys(qapp, layout):
    """Exercise Windows key translation, including Polish (Programmers).

    QTest.keySequence bypasses virtual-key translation. SendInput supplies
    real VK_OEM_2 / VK_DIVIDE messages, only while this test owns focus.
    """
    import sys
    if sys.platform != 'win32' or qapp.platformName() != 'windows':
        pytest.skip('Requires the native Windows Qt platform plugin')
    import ctypes
    from ctypes import wintypes
    from PyQt6.QtCore import QObject, QEvent
    from PyQt6.QtGui import QKeySequence, QShortcut
    from PyQt6.QtWidgets import QLineEdit

    class KeyboardInput(ctypes.Structure):
        _fields_ = [
            ('vk', wintypes.WORD), ('scan', wintypes.WORD),
            ('flags', wintypes.DWORD), ('time', wintypes.DWORD),
            ('extra', ctypes.c_size_t),
        ]

    class MouseInput(ctypes.Structure):
        _fields_ = [
            ('x', wintypes.LONG), ('y', wintypes.LONG),
            ('data', wintypes.DWORD), ('flags', wintypes.DWORD),
            ('time', wintypes.DWORD), ('extra', ctypes.c_size_t),
        ]

    class HardwareInput(ctypes.Structure):
        _fields_ = [('msg', wintypes.DWORD), ('low', wintypes.WORD),
                    ('high', wintypes.WORD)]

    class InputData(ctypes.Union):
        _fields_ = [('keyboard', KeyboardInput), ('mouse', MouseInput),
                    ('hardware', HardwareInput)]

    class Input(ctypes.Structure):
        _fields_ = [('kind', wintypes.DWORD), ('data', InputData)]

    user32 = ctypes.WinDLL('user32', use_last_error=True)
    user32.SendInput.argtypes = [
        wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int]
    user32.SendInput.restype = wintypes.UINT
    user32.GetKeyboardLayout.argtypes = [wintypes.DWORD]
    user32.GetKeyboardLayout.restype = wintypes.HANDLE
    user32.LoadKeyboardLayoutW.argtypes = [wintypes.LPCWSTR, wintypes.UINT]
    user32.LoadKeyboardLayoutW.restype = wintypes.HANDLE
    user32.ActivateKeyboardLayout.argtypes = [wintypes.HANDLE, wintypes.UINT]
    user32.ActivateKeyboardLayout.restype = wintypes.HANDLE
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.GetForegroundWindow.restype = wintypes.HWND
    previous_layout = user32.GetKeyboardLayout(0)
    keyboard_layout = user32.LoadKeyboardLayoutW(layout, 0)
    assert keyboard_layout, 'Windows keyboard layout unavailable'
    window = MenuWindow()
    window._init_menubar()
    edit = QLineEdit(window)
    window.setCentralWidget(edit)
    events = []

    class Recorder(QObject):
        def eventFilter(self, obj, event):
            if event.type() in (QEvent.Type.KeyPress,
                                QEvent.Type.ShortcutOverride):
                events.append((event.key(), event.modifiers().value,
                               event.nativeVirtualKey()))
            return False

    recorder = Recorder(edit)
    edit.installEventFilter(recorder)
    baseline_calls = []
    baseline = QShortcut(QKeySequence('Ctrl+Shift+P'), window)
    baseline.activated.connect(lambda: baseline_calls.append(True))
    window.show()
    window.activateWindow()
    edit.setFocus()
    hwnd = int(window.winId())
    user32.SetForegroundWindow(hwnd)
    QTest.qWait(100)

    def press(virtual_keys):
        assert user32.GetForegroundWindow() == hwnd, 'Test window lost focus'
        inputs = []
        for key in virtual_keys:
            flags = 1 if key == 0x6F else 0  # extended VK_DIVIDE
            inputs.append(Input(1, InputData(
                keyboard=KeyboardInput(key, 0, flags, 0, 0))))
        for key in reversed(virtual_keys):
            flags = 3 if key == 0x6F else 2  # KEYEVENTF_KEYUP
            inputs.append(Input(1, InputData(
                keyboard=KeyboardInput(key, 0, flags, 0, 0))))
        packet = (Input * len(inputs))(*inputs)
        assert user32.SendInput(len(inputs), packet, ctypes.sizeof(Input)) == (
            len(inputs))
        QTest.qWait(100)

    try:
        assert user32.ActivateKeyboardLayout(keyboard_layout, 0)
        press([0x11, 0x10, 0x50])  # Ctrl+Shift+P, known working baseline
        assert baseline_calls == [True], events
        press([0x11, 0xBF])  # Ctrl+OEM slash
        assert window.palette_requests == 1, events
        press([0x11, 0x6F])  # Ctrl+keypad divide
        assert window.palette_requests == 2, events
    finally:
        user32.ActivateKeyboardLayout(previous_layout, 0)
        window.close()
