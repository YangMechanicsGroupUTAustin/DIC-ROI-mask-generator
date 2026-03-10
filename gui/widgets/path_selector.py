"""Directory path selector with label and Browse button.

Used for selecting input/output directories.
Supports both Browse button selection and direct path pasting/typing.
"""

import os

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.icons import get_icon
from gui.theme import Colors, Fonts


class PathSelector(QWidget):
    """Directory path selector with label and Browse button."""

    path_changed = pyqtSignal(str)

    def __init__(
        self,
        label: str,
        placeholder: str = "/path/to/directory...",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Label
        text_label = QLabel(label)
        text_label.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-size: {Fonts.SIZE_SM}px; "
            f"background: transparent;"
        )
        layout.addWidget(text_label)

        # Input row (line edit + browse button)
        input_row = QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(4)

        self._line_edit = QLineEdit()
        self._line_edit.setPlaceholderText(placeholder)
        self._line_edit.editingFinished.connect(self._on_editing_finished)
        input_row.addWidget(self._line_edit)

        browse_btn = QPushButton()
        browse_btn.setIcon(get_icon("folder-open", Colors.TEXT_SECONDARY, 14))
        browse_btn.setToolTip("Browse...")
        browse_btn.setFixedWidth(36)
        browse_btn.clicked.connect(self._browse)
        input_row.addWidget(browse_btn)

        layout.addLayout(input_row)

    def path(self) -> str:
        """Return the current path."""
        return self._line_edit.text()

    def set_path(self, path: str) -> None:
        """Set the path programmatically."""
        self._line_edit.setText(path)
        self.path_changed.emit(path)

    def _on_editing_finished(self) -> None:
        """Validate and emit path when user finishes editing (Enter or focus lost)."""
        path = self._line_edit.text().strip().strip('"')
        if path and os.path.isdir(path):
            self.path_changed.emit(path)

    def _browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            self._line_edit.text() or "",
        )
        if directory:
            self.set_path(directory)
