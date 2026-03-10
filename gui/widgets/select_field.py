"""Labeled combo box with optional icon prefix.

Used for dropdown selections like model, device, format, etc.
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from gui.icons import get_icon
from gui.theme import Colors, Fonts


class SelectField(QWidget):
    """Labeled combo box with icon prefix."""

    value_changed = pyqtSignal(str)

    def __init__(
        self,
        label: str,
        options: list[str],
        default: str = "",
        icon_name: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

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

        # Combo box
        self._combo = QComboBox()
        self._combo.addItems(options)
        if default and default in options:
            self._combo.setCurrentText(default)
        self._combo.currentTextChanged.connect(self.value_changed.emit)
        layout.addWidget(self._combo)

    def value(self) -> str:
        """Return the currently selected value."""
        return self._combo.currentText()

    def set_value(self, value: str) -> None:
        """Set the selected value."""
        index = self._combo.findText(value)
        if index >= 0:
            self._combo.setCurrentIndex(index)

    def set_options(self, options: list[str]) -> None:
        """Replace all options in the combo box."""
        current = self._combo.currentText()
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItems(options)
        # Try to restore previous selection
        restored_index = self._combo.findText(current)
        if restored_index >= 0:
            self._combo.setCurrentIndex(restored_index)
        self._combo.blockSignals(False)
        # Emit if selection changed
        if self._combo.currentText() != current:
            self.value_changed.emit(self._combo.currentText())
