# pro/widgets/spinboxes.py
"""
Custom spinbox widgets for Seti Astro Suite Pro.

Provides enhanced spinbox widgets with consistent styling and behavior.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, QToolButton
)
from PyQt6.QtGui import QIntValidator, QDoubleValidator


class CustomSpinBox(QWidget):
    """
    A custom integer spin box widget with up/down buttons.
    
    Emits valueChanged(int) when the value changes.
    
    Usage:
        spin = CustomSpinBox(minimum=0, maximum=100, initial=50, step=1)
        spin.valueChanged.connect(my_handler)
    """
    valueChanged = pyqtSignal(int)

    def __init__(
        self,
        minimum: int = 0,
        maximum: int = 100,
        initial: int = 0,
        step: int = 1,
        parent: QWidget | None = None
    ):
        super().__init__(parent)
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self._value = int(initial)

        # Line edit for value display/entry
        self.lineEdit = QLineEdit(str(initial))
        self.lineEdit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.lineEdit.setValidator(QIntValidator(self.minimum, self.maximum, self))
        self.lineEdit.editingFinished.connect(self._on_editing_finished)

        # Up/down buttons
        self.upButton = QToolButton()
        self.upButton.setText("▲")
        self.upButton.setAutoRepeat(True)
        self.upButton.setAutoRepeatInterval(50)
        self.upButton.setAutoRepeatDelay(300)
        self.upButton.clicked.connect(self._increase_value)
        
        self.downButton = QToolButton()
        self.downButton.setText("▼")
        self.downButton.setAutoRepeat(True)
        self.downButton.setAutoRepeatInterval(50)
        self.downButton.setAutoRepeatDelay(300)
        self.downButton.clicked.connect(self._decrease_value)

        # Layout buttons vertically
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.upButton)
        button_layout.addWidget(self.downButton)
        button_layout.setSpacing(0)
        button_layout.setContentsMargins(0, 0, 0, 0)

        # Main horizontal layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.lineEdit)
        main_layout.addLayout(button_layout)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)

        self._update_button_states()

    def value(self) -> int:
        """Qt-style getter."""
        return int(self._value)

    def setValue(self, val: int) -> None:
        val = int(val)
        val = max(int(self.minimum), min(int(self.maximum), val))
        if val != self._value:
            self._value = val
            self.lineEdit.setText(str(val))
            self.valueChanged.emit(val)
            self._update_button_states()

    def setMinimum(self, minimum: int) -> None:
        """Set the minimum value."""
        self.minimum = minimum
        self.lineEdit.setValidator(QIntValidator(self.minimum, self.maximum, self))
        if self._value < minimum:
            self.setValue(minimum)
        self._update_button_states()

    def setMaximum(self, maximum: int) -> None:
        """Set the maximum value."""
        self.maximum = maximum
        self.lineEdit.setValidator(QIntValidator(self.minimum, self.maximum, self))
        if self._value > maximum:
            self.setValue(maximum)
        self._update_button_states()

    def setRange(self, minimum: int, maximum: int) -> None:
        """Set both minimum and maximum values."""
        self.minimum = minimum
        self.maximum = maximum
        self.lineEdit.setValidator(QIntValidator(self.minimum, self.maximum, self))
        self.setValue(max(minimum, min(maximum, self._value)))

    def setSingleStep(self, step: int) -> None:
        """Set the step value for up/down buttons."""
        self.step = step

    def _on_editing_finished(self) -> None:
        """Handle manual text entry."""
        try:
            val = int(self.lineEdit.text())
            self.setValue(val)
        except ValueError:
            self.lineEdit.setText(str(self._value))

    def _increase_value(self) -> None:
        """Increase value by step."""
        self.setValue(self._value + self.step)

    def _decrease_value(self) -> None:
        """Decrease value by step."""
        self.setValue(self._value - self.step)

    def value(self) -> int:
        """
        Qt-compatible getter (QSpinBox uses value()).

        Note: we also have @property value for convenience,
        but code that expects QSpinBox calls value().
        """
        return self._value

    def _update_button_states(self) -> None:
        self.upButton.setEnabled(self._value < self.maximum)
        self.downButton.setEnabled(self._value > self.minimum)


class CustomDoubleSpinBox(QWidget):
    """
    A custom double (float) spin box widget with up/down buttons.
    
    Emits valueChanged(float) when the value changes.
    
    Usage:
        spin = CustomDoubleSpinBox(minimum=0.0, maximum=1.0, initial=0.5, step=0.1, decimals=2)
        spin.valueChanged.connect(my_handler)
    """
    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        minimum: float = 0.0,
        maximum: float = 100.0,
        initial: float = 0.0,
        step: float = 1.0,
        decimals: int = 2,
        parent: QWidget | None = None
    ):
        super().__init__(parent)
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self.decimals = decimals
        self._value = float(initial)

        # Line edit for value display/entry
        self.lineEdit = QLineEdit(f"{initial:.{decimals}f}")
        self.lineEdit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.lineEdit.setValidator(QDoubleValidator(self.minimum, self.maximum, decimals, self))
        self.lineEdit.editingFinished.connect(self._on_editing_finished)

        # Up/down buttons
        self.upButton = QToolButton()
        self.upButton.setText("▲")
        self.upButton.setAutoRepeat(True)
        self.upButton.setAutoRepeatInterval(50)
        self.upButton.setAutoRepeatDelay(300)
        self.upButton.clicked.connect(self._increase_value)
        
        self.downButton = QToolButton()
        self.downButton.setText("▼")
        self.downButton.setAutoRepeat(True)
        self.downButton.setAutoRepeatInterval(50)
        self.downButton.setAutoRepeatDelay(300)
        self.downButton.clicked.connect(self._decrease_value)

        # Layout buttons vertically
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.upButton)
        button_layout.addWidget(self.downButton)
        button_layout.setSpacing(0)
        button_layout.setContentsMargins(0, 0, 0, 0)

        # Main horizontal layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.lineEdit)
        main_layout.addLayout(button_layout)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)

        self._update_button_states()

    def value(self) -> float:
        """Qt-style getter."""
        return float(self._value)

    def setValue(self, val: float) -> None:
        val = float(val)
        val = max(float(self.minimum), min(float(self.maximum), val))
        if abs(val - self._value) > 1e-10:
            self._value = val
            self.lineEdit.setText(f"{val:.{self.decimals}f}")
            self.valueChanged.emit(val)
            self._update_button_states()

    def setMinimum(self, minimum: float) -> None:
        """Set the minimum value."""
        self.minimum = minimum
        self.lineEdit.setValidator(QDoubleValidator(self.minimum, self.maximum, self.decimals, self))
        if self._value < minimum:
            self.setValue(minimum)
        self._update_button_states()

    def value(self) -> float:
        """
        Qt-compatible getter (QDoubleSpinBox uses value()).

        Note: we also have @property value for convenience,
        but code that expects QDoubleSpinBox calls value().
        """
        return self._value

    def setMaximum(self, maximum: float) -> None:
        """Set the maximum value."""
        self.maximum = maximum
        self.lineEdit.setValidator(QDoubleValidator(self.minimum, self.maximum, self.decimals, self))
        if self._value > maximum:
            self.setValue(maximum)
        self._update_button_states()

    def setRange(self, minimum: float, maximum: float) -> None:
        """Set both minimum and maximum values."""
        self.minimum = minimum
        self.maximum = maximum
        self.lineEdit.setValidator(QDoubleValidator(self.minimum, self.maximum, self.decimals, self))
        self.setValue(max(minimum, min(maximum, self._value)))

    def setSingleStep(self, step: float) -> None:
        """Set the step value for up/down buttons."""
        self.step = step

    def setDecimals(self, decimals: int) -> None:
        """Set the number of decimal places."""
        self.decimals = decimals
        self.lineEdit.setText(f"{self._value:.{decimals}f}")
        self.lineEdit.setValidator(QDoubleValidator(self.minimum, self.maximum, decimals, self))

    def _on_editing_finished(self) -> None:
        """Handle manual text entry."""
        try:
            val = float(self.lineEdit.text())
            self.setValue(val)
        except ValueError:
            self.lineEdit.setText(f"{self._value:.{self.decimals}f}")

    def _increase_value(self) -> None:
        """Increase value by step."""
        self.setValue(self._value + self.step)

    def _decrease_value(self) -> None:
        """Decrease value by step."""
        self.setValue(self._value - self.step)

    def _update_button_states(self) -> None:
        """Enable/disable buttons at limits."""
        self.upButton.setEnabled(self._value < self.maximum - 1e-10)
        self.downButton.setEnabled(self._value > self.minimum + 1e-10)
