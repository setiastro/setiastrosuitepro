# pro/widgets/themed_buttons.py
from __future__ import annotations
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QToolButton

def themed_toolbtn(icon_name: str, tip: str, *, on_click=None) -> QToolButton:
    b = QToolButton()
    b.setIcon(QIcon.fromTheme(icon_name))
    b.setToolTip(tip)
    b.setAutoRaise(True)  # nice flat toolbar look
    if on_click is not None:
        b.clicked.connect(on_click)
    return b
