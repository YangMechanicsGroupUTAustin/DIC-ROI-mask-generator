"""Collapsible accordion section for the sidebar.

Click the header (icon + title + chevron) to expand/collapse
the content area. The chevron rotates to indicate state.
"""

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.icons import get_icon
from gui.theme import Colors, Fonts


class _HeaderWidget(QFrame):
    """Clickable header bar for the collapsible section."""

    clicked = pyqtSignal()

    def __init__(self, title: str, icon_name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            f"QFrame {{ background: {Colors.BG_LIGHT}; border: none; "
            f"border-radius: 6px; padding: 6px 10px; }}"
            f"QFrame:hover {{ background: rgba(255, 255, 255, 0.04); }}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Optional leading icon
        if icon_name:
            icon_label = QLabel()
            pixmap = get_icon(icon_name, Colors.TEXT_MUTED, 14).pixmap(14, 14)
            icon_label.setPixmap(pixmap)
            icon_label.setFixedSize(14, 14)
            layout.addWidget(icon_label)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-size: {Fonts.SIZE_SM}px; "
            f"font-weight: 600; background: transparent;"
        )
        layout.addWidget(title_label)
        layout.addStretch()

        # Chevron indicator
        self._chevron_label = QLabel()
        self._update_chevron(True)
        self._chevron_label.setFixedSize(14, 14)
        layout.addWidget(self._chevron_label)

    def _update_chevron(self, is_open: bool) -> None:
        icon_name = "chevron-down" if is_open else "chevron-right"
        pixmap = get_icon(icon_name, Colors.TEXT_DIM, 14).pixmap(14, 14)
        self._chevron_label.setPixmap(pixmap)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class CollapsibleSection(QWidget):
    """Collapsible sidebar section with animated toggle."""

    toggled = pyqtSignal(bool)  # True = open

    def __init__(
        self,
        title: str,
        icon_name: str = "",
        default_open: bool = True,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._is_open = default_open

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header
        self._header = _HeaderWidget(title, icon_name, self)
        self._header.clicked.connect(self._toggle)
        main_layout.addWidget(self._header)

        # Content container
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(4, 8, 4, 4)
        self._content_layout.setSpacing(8)
        self._content.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        main_layout.addWidget(self._content)

        # Apply initial state
        self._apply_state()

    def add_widget(self, widget: QWidget) -> None:
        """Add a widget to the content area."""
        self._content_layout.addWidget(widget)

    def add_layout(self, layout: QLayout) -> None:
        """Set the content layout."""
        # Remove existing content layout items first
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        # Add the new layout as a nested layout
        self._content_layout.addLayout(layout)

    def set_open(self, is_open: bool) -> None:
        """Set the section open/closed state."""
        if self._is_open != is_open:
            self._is_open = is_open
            self._apply_state()
            self.toggled.emit(self._is_open)

    def is_open(self) -> bool:
        """Return whether the section is currently open."""
        return self._is_open

    def _toggle(self) -> None:
        self.set_open(not self._is_open)

    def _apply_state(self) -> None:
        self._content.setVisible(self._is_open)
        self._header._update_chevron(self._is_open)
