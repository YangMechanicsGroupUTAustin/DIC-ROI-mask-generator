"""Labeled number input with optional unit suffix.

Used for numeric parameters like iterations, kernel size, etc.
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.icons import get_icon
from gui.theme import Colors, Fonts


class NumberInput(QWidget):
    """Labeled number input with optional unit suffix."""

    value_changed = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        default: float = 0,
        min_val: float = 0,
        max_val: float = 99999,
        step: float = 1,
        decimals: int = 1,
        unit: str = "",
        icon_name: str = "",
        tooltip: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        if tooltip:
            self.setToolTip(tooltip)
        self._decimals = decimals

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Label row (icon + label text)
        label_row = QHBoxLayout()
        label_row.setContentsMargins(0, 0, 0, 0)
        label_row.setSpacing(6)

        if icon_name:
            icon_label = QLabel()
            pixmap = get_icon(icon_name, Colors.TEXT_MUTED, 12).pixmap(12, 12)
            icon_label.setPixmap(pixmap)
            icon_label.setFixedSize(12, 12)
            label_row.addWidget(icon_label)

        text_label = QLabel(label)
        text_label.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-size: {Fonts.SIZE_SM}px; "
            f"background: transparent;"
        )
        label_row.addWidget(text_label)
        label_row.addStretch()
        layout.addLayout(label_row)

        # Input row (spinbox + optional unit)
        input_row = QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(4)

        if decimals > 0:
            self._spinbox = QDoubleSpinBox()
            self._spinbox.setDecimals(decimals)
            self._spinbox.setSingleStep(step)
            self._spinbox.setRange(min_val, max_val)
            self._spinbox.setValue(default)
            self._spinbox.valueChanged.connect(self._on_value_changed)
        else:
            self._spinbox = QSpinBox()
            self._spinbox.setSingleStep(int(step))
            self._spinbox.setRange(int(min_val), int(max_val))
            self._spinbox.setValue(int(default))
            self._spinbox.valueChanged.connect(self._on_value_changed)

        input_row.addWidget(self._spinbox)

        if unit:
            unit_label = QLabel(unit)
            unit_label.setStyleSheet(
                f"color: {Colors.TEXT_DIM}; font-size: {Fonts.SIZE_SM}px; "
                f"background: transparent;"
            )
            input_row.addWidget(unit_label)

        layout.addLayout(input_row)

    def value(self) -> float:
        """Return the current value."""
        return float(self._spinbox.value())

    def set_value(self, value: float) -> None:
        """Set the current value."""
        self._spinbox.blockSignals(True)
        if self._decimals > 0:
            self._spinbox.setValue(value)
        else:
            self._spinbox.setValue(int(value))
        self._spinbox.blockSignals(False)

    def _on_value_changed(self, value) -> None:
        self.value_changed.emit(float(value))
