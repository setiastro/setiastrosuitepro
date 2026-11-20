from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import (
    QKeyEvent, QKeySequence, QStandardItem, QStandardItemModel,
)
from PyQt6.QtWidgets import QApplication, QMainWindow

from setiastro.saspro.gui import command_palette


def test_command_slash_toggles_popup_without_consuming_slash():
    app = QApplication.instance() or QApplication([])
    parent = QMainWindow()
    model = QStandardItemModel(parent)
    model.appendRow(QStandardItem("Curves Editor"))
    executed = []
    palette = command_palette.CommandPalette(parent, model, executed.append)
    palette.show()
    slash = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Slash,
        Qt.KeyboardModifier.NoModifier, "/",
    )
    app.sendEvent(palette.edit, slash)
    assert palette.edit.text() == "/"
    assert palette.isVisible()
    for keys in command_palette.PALETTE_SHORTCUTS:
        palette.show()
        combination = QKeySequence(keys)[0]
        toggle = QKeyEvent(
            QEvent.Type.KeyPress, combination.key(),
            combination.keyboardModifiers(),
        )
        app.sendEvent(palette.edit, toggle)
        assert not palette.isVisible()
    assert executed == []
    parent.close()


def test_paste_limit_and_recovery():
    app = QApplication.instance() or QApplication([])
    parent = QMainWindow()
    model = QStandardItemModel(parent)
    item = QStandardItem('ADBE')
    item.setToolTip('Automatic Dynamic Background Extraction')
    model.appendRow(item)
    palette = command_palette.CommandPalette(parent, model, lambda idx: None)
    parent.show()
    palette.popup_centered()
    clipboard = app.clipboard()
    previous = clipboard.text()
    try:
        clipboard.setText('automtic ' * 125_000)
        palette.edit.paste()
        assert len(palette.edit.text()) == command_palette.MAX_QUERY_LENGTH
        palette.edit.setText('automatic')
        assert palette.proxy.rowCount() == 1
        assert palette.list.currentIndex().data() == 'ADBE'
        palette.edit.setText('😀' * 100_000)
        assert len(palette.edit.text()) <= command_palette.MAX_QUERY_LENGTH
        assert palette.proxy.rowCount() == 0
        palette.edit.clear()
        assert palette.proxy.rowCount() == 1
    finally:
        clipboard.setText(previous)
        parent.close()


def test_navigation_empty_results_model_changes_and_one_execution():
    from PyQt6.QtTest import QTest

    app = QApplication.instance() or QApplication([])
    parent = QMainWindow()
    model = QStandardItemModel(parent)
    for name in ('Alpha', 'Beta', 'Gamma'):
        model.appendRow(QStandardItem(name))
    executed = []
    palette = command_palette.CommandPalette(parent, model, executed.append)
    parent.show()
    palette.popup_centered()
    QTest.keyClick(palette.edit, Qt.Key.Key_Down)
    assert palette.list.currentIndex().data() == 'Beta'
    QTest.keyClick(palette.edit, Qt.Key.Key_PageDown)
    assert palette.list.currentIndex().data() == 'Gamma'
    QTest.keyClick(palette.edit, Qt.Key.Key_PageUp)
    assert palette.list.currentIndex().data() == 'Alpha'
    palette.edit.setText('no results')
    QTest.keyClick(palette.edit, Qt.Key.Key_Return)
    assert executed == []
    assert palette.isVisible()
    palette.edit.clear()
    model.clear()
    model.appendRow(QStandardItem('New Command'))
    assert palette.list.currentIndex().data() == 'New Command'
    repeat = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return,
                       Qt.KeyboardModifier.NoModifier, '', True)
    app.sendEvent(palette.edit, repeat)
    assert executed == []
    QTest.keyClick(palette.edit, Qt.Key.Key_Return)
    QTest.keyClick(palette.edit, Qt.Key.Key_Return)
    assert len(executed) == 1
    parent.close()


def test_popup_fits_small_screen_and_long_labels():
    from types import SimpleNamespace
    from PyQt6.QtCore import QRect

    app = QApplication.instance() or QApplication([])
    available = QRect(10, 20, 320, 180)

    class SmallScreenWindow(QMainWindow):
        def screen(self):
            return SimpleNamespace(availableGeometry=lambda: available)

    parent = SmallScreenWindow()
    parent.resize(1200, 900)
    model = QStandardItemModel(parent)
    model.appendRow(QStandardItem('Very long title ' * 1000))
    palette = command_palette.CommandPalette(parent, model, lambda idx: None)
    parent.show()
    palette.popup_centered()
    app.processEvents()
    assert available.contains(palette.geometry())
    assert palette.list.horizontalScrollBarPolicy() == (
        Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    parent.close()


def test_popup_is_hidden_before_opening_a_modal_tool():
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QDialog

    app = QApplication.instance() or QApplication([])
    parent = QMainWindow()
    model = QStandardItemModel(parent)
    model.appendRow(QStandardItem('Modal tool'))
    tool = QDialog(parent)
    tool.setModal(True)
    states = []

    def run(index):
        states.append(palette.isVisible())
        tool.show()

    palette = command_palette.CommandPalette(parent, model, run)
    parent.show()
    palette.popup_centered()
    QTest.keyClick(palette.edit, Qt.Key.Key_Return)
    app.processEvents()
    assert states == [False]
    assert tool.isVisible()
    tool.close()
    parent.close()


def test_command_uses_current_mdi_view_without_caching_its_document():
    from PyQt6.QtGui import QAction
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QMdiArea, QWidget
    from setiastro.saspro.gui.command_search import command_item, run_command

    app = QApplication.instance() or QApplication([])
    parent = QMainWindow()
    mdi = QMdiArea(parent)
    parent.setCentralWidget(mdi)
    documents = [object(), object()]
    views = []
    for document in documents:
        view = QWidget()
        view.document = document
        subwindow = mdi.addSubWindow(view)
        subwindow.show()
        views.append(subwindow)
    executed = []
    action = QAction('Context tool', parent)
    action.triggered.connect(lambda: executed.append(
        mdi.activeSubWindow().widget().document))
    role = Qt.ItemDataRole.UserRole + 1
    model = QStandardItemModel(parent)
    model.appendRow(command_item(action, action.text(), role))
    palette = command_palette.CommandPalette(
        parent, model, lambda index: run_command(index, role))
    parent.show()
    try:
        for selected in (0, 1, 0):
            mdi.setActiveSubWindow(views[selected])
            parent.activateWindow()
            app.processEvents()
            palette.popup_centered()
            palette.edit.setText('context')
            # Taking keyboard focus must not change the selected image view.
            assert mdi.activeSubWindow() is views[selected]
            QTest.keyClick(palette.edit, Qt.Key.Key_Return)
            assert executed[-1] is documents[selected]
        # Menu/toolbar activation resolves the same document.
        mdi.setActiveSubWindow(views[1])
        action.trigger()
        assert executed == [documents[0], documents[1], documents[0],
                            documents[1]]
    finally:
        parent.close()
