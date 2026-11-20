import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import QApplication, QMainWindow

from setiastro.saspro.gui.command_palette import CommandPalette
from setiastro.saspro.gui.command_search import match_score


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("query", [
    "ADBE", "automatic", "automtic", "background automatic",
    "  AUTOMATIC (dynamic) ",
])
def test_full_name_and_typo_match_adbe(query):
    assert match_score(
        query, "ADBE", "Automatic (Dynamic) Background Extraction",
    ) is not None


def test_unrelated_and_short_queries_do_not_fuzzy_match():
    assert match_score(
        "galaxy", "ADBE", "Automatic Background Extraction",
    ) is None
    assert match_score("ax", "ADBE", "Automatic Background Extraction") is None
    assert match_score(
        "automatic galaxy", "ADBE", "Automatic Background Extraction",
    ) is None


def test_exact_name_beats_description_and_typo():
    exact = match_score("curves", "Curves", "")
    contains = match_score("curves", "RGB Curves Editor", "")
    description = match_score("curves", "Other Tool", "Edit curves")
    typo = match_score("curves", "Curvs", "")
    assert exact < contains < description < typo
    assert match_score("neutralization", "Neutralization", "") == 0
    assert match_score("cafe", "Café", "") == 0


def test_palette_rank_mapping_empty_results_and_model_updates(qapp):
    parent = QMainWindow()
    model = QStandardItemModel(parent)
    model.appendRow(QStandardItem("Background Tool"))
    adbe = QStandardItem("ADBE")
    adbe.setToolTip("Automatic (Dynamic) Background Extraction")
    model.appendRow(adbe)
    model.appendRow(QStandardItem("Automatic"))
    executed = []
    palette = CommandPalette(
        parent, model, lambda index: executed.append(index.row()),
    )
    parent.show()
    palette.popup_centered()
    palette.edit.setText("automatic")
    assert palette.proxy.index(0, 0).data() == "Automatic"
    palette.edit.setText("automtic background")
    assert palette.proxy.rowCount() == 1
    assert palette.list.currentIndex().data() == "ADBE"
    QApplication.sendEvent(palette.edit, QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Return,
        Qt.KeyboardModifier.NoModifier,
    ))
    assert executed == [1]  # original source row, not the ranked result row
    palette.edit.setText("no such command anywhere")
    assert palette.proxy.rowCount() == 0
    QApplication.sendEvent(palette.edit, QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Return,
        Qt.KeyboardModifier.NoModifier,
    ))
    assert executed == [1]
    palette.edit.setText("automatic")
    adbe.setToolTip("Different description")
    assert palette.proxy.rowCount() == 1
    palette.edit.clear()
    assert palette.proxy.rowCount() == 3
    assert [palette.proxy.index(row, 0).data() for row in range(3)] == [
        "ADBE", "Automatic", "Background Tool",
    ]
    parent.close()


@pytest.mark.parametrize('query,title', [
    ('TŁO', 'Tło'), ('CAFÉ', 'Cafe'), ('cafe\u0301', 'Café'),
    ('背景', '背景'), ('Кривые', 'Кривые'), ('منحنيات', 'منحنيات'),
    ('  curves\n\t editor  ', 'Curves Editor'),
])
def test_unicode_and_whitespace(query, title):
    assert match_score(query, title, '') is not None


@pytest.mark.parametrize('query', ['/', '???', '😀', '\u200b', '\u0301'])
def test_non_word_queries_do_not_select_arbitrary_commands(query):
    assert match_score(query, 'Delete All', '') is None


def test_long_input_is_bounded_before_caching():
    from setiastro.saspro.gui.command_search import (
        MAX_QUERY_LENGTH, _match_score,
    )
    query = 'x' * MAX_QUERY_LENGTH
    _match_score.cache_clear()
    expected = match_score(query, 'Curves', '')
    assert match_score(query + 'y' * 1_000_000, 'Curves', '') == expected
    assert _match_score.cache_info().currsize == 1


def test_destroyed_and_disabled_qactions_are_safe(qapp):
    from PyQt6.QtCore import QCoreApplication
    from PyQt6.QtGui import QAction
    from setiastro.saspro.gui.command_search import command_item, run_command

    parent = QMainWindow()
    model = QStandardItemModel(parent)
    action = QAction('Dynamic tool', parent)
    action.setStatusTip('Full tool name')
    calls = []
    action.triggered.connect(lambda: calls.append(True))
    role = Qt.ItemDataRole.UserRole + 1
    model.appendRow(command_item(action, action.text(), role))
    index = model.index(0, 0)
    assert index.data(Qt.ItemDataRole.ToolTipRole) == (
        'Full tool name\nDynamic tool')
    assert run_command(index, role)
    action.setEnabled(False)
    assert not run_command(index, role)
    action.setEnabled(True)
    action.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    # Reading QVariant data itself used to dereference a deleted QObject*.
    assert not run_command(index, role)
    assert calls == [True]
    model.clear()
    assert not run_command(model.index(0, 0), role)
    parent.close()
