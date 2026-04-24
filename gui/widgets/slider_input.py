"""Labeled slider with gradient track and live value display.

Used for threshold, confidence, and similar continuous parameters.
"""

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gui.theme import Colors, Fonts


class SliderInput(QWidget):
    """Labeled slider with gradient track and live value display."""

    value_changed = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        default: float = 0,
        min_val: float = 0,
        max_val: float = 1,
        step: float = 0.01,
        decimals: int = 2,
        tooltip: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        if tooltip:
            self.setToolTip(tooltip)
        self._min_val = min_val
        self._max_val = max_val
        self._step = step
        self._decimals = decimals

        # Compute integer range for slider
        self._num_steps = max(1, int(round((max_val - min_val) / step)))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Top row: label + value
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)

        text_label = QLabel(label)
        text_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent;"
        )
        top_row.addWidget(text_label)
        top_row.addStretch()

        self._value_label = QLabel()
        self._value_label.setStyleSheet(
            f"color: {Colors.PRIMARY_HOVER}; font-family: '{Fonts.MONO}'; "
            f"font-size: {Fonts.SIZE_MD}px; font-weight: bold; "
            f"background: transparent;"
        )
        top_row.addWidget(self._value_label)
        layout.addLayout(top_row)

        # Slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, self._num_steps)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(max(1, self._num_steps // 10))
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider)

        # Set initial value
        self.set_value(default)

    def value(self) -> float:
        """Return the current float value."""
        return self._min_val + (self._slider.value() * self._step)

    def set_value(self, value: float) -> None:
        """Set the current value."""
        clamped = max(self._min_val, min(self._max_val, value))
        int_val = int(round((clamped - self._min_val) / self._step))
        self._slider.blockSignals(True)
        self._slider.setValue(int_val)
        self._slider.blockSignals(False)
        self._update_label()

    def _on_slider_changed(self, _int_value: int) -> None:
        self._update_label()
        self.value_changed.emit(self.value())

    def _update_label(self) -> None:
        self._value_label.setText(f"{self.value():.{self._decimals}f}")
