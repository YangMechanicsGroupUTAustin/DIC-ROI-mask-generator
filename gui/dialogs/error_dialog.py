"""Styled error dialog with dark theme.

Displays error messages with optional details,
copy-to-clipboard functionality, and an OK button.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gui.icons import get_pixmap
from gui.theme import Colors, Fonts


class ErrorDialog(QDialog):
    """Dark-themed error dialog with copy support."""

    def __init__(
        self,
        title: str,
        message: str,
        details: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(450)
        self.setMinimumHeight(200)
        self.setStyleSheet(
            f"QDialog {{ background: {Colors.BG_MEDIUM}; "
            f"color: {Colors.TEXT_PRIMARY}; }}"
        )

        self._full_text = message
        if details:
            self._full_text += f"\n\n{details}"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(16)

        # --- Header row: icon + title ---
        header = QHBoxLayout()
        header.setSpacing(12)

        icon_label = QLabel()
        icon_label.setPixmap(get_pixmap("circle", Colors.DANGER, 24))
        icon_label.setFixedSize(24, 24)
        icon_label.setStyleSheet("background: transparent;")
        header.addWidget(icon_label)

        title_label = QLabel(title)
        title_label.setStyleSheet(
            f"color: {Colors.DANGER}; font-size: {Fonts.SIZE_LG}px; "
            f"font-weight: 600; background: transparent;"
        )
        header.addWidget(title_label)
        header.addStretch()

        layout.addLayout(header)

        # --- Message ---
        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent;"
        )
        layout.addWidget(msg_label)

        # --- Details (scrollable) ---
        if details:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setMaximumHeight(200)
            scroll.setStyleSheet(
                f"QScrollArea {{ background: {Colors.BG_DARKEST}; "
                f"border: 1px solid {Colors.BORDER}; border-radius: 6px; }}"
            )

            details_label = QLabel(details)
            details_label.setWordWrap(True)
            details_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            details_label.setStyleSheet(
                f"color: {Colors.TEXT_MUTED}; font-family: '{Fonts.MONO}'; "
                f"font-size: {Fonts.SIZE_SM}px; padding: 8px; "
                f"background: {Colors.BG_DARKEST};"
            )
            scroll.setWidget(details_label)
            layout.addWidget(scroll)

        layout.addStretch()

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_btn.clicked.connect(self._copy_to_clipboard)
        btn_layout.addWidget(copy_btn)

        ok_btn = QPushButton("OK")
        ok_btn.setProperty("cssClass", "btn-primary")
        ok_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        ok_btn.setMinimumWidth(80)
        ok_btn.clicked.connect(self.accept)
        ok_btn.setDefault(True)
        btn_layout.addWidget(ok_btn)

        layout.addLayout(btn_layout)

    def _copy_to_clipboard(self) -> None:
        """Copy the full error text to the system clipboard."""
        clipboard = QGuiApplication.clipboard()
        if clipboard:
            clipboard.setText(self._full_text)

    @staticmethod
    def show_error(
        title: str,
        message: str,
        details: str = "",
        parent: QWidget | None = None,
    ) -> None:
        """Show an error dialog and block until dismissed."""
        dialog = ErrorDialog(title, message, details, parent)
        dialog.exec()
