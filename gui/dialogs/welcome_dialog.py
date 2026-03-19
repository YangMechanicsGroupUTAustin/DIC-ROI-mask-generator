"""Welcome / onboarding dialog for first-time users.

Shows a brief workflow overview with numbered steps.
Remembers dismissal via a simple flag file so it only shows once.
"""

import os
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
)

from gui.theme import Colors, Fonts

# Flag file location: ~/.sam2studio/welcome_dismissed
_FLAG_DIR = Path.home() / ".sam2studio"
_FLAG_FILE = _FLAG_DIR / "welcome_dismissed"


def should_show_welcome() -> bool:
    """Return True if the welcome dialog should be shown."""
    return not _FLAG_FILE.exists()


def mark_welcome_dismissed() -> None:
    """Mark the welcome dialog as dismissed."""
    _FLAG_DIR.mkdir(parents=True, exist_ok=True)
    _FLAG_FILE.write_text("1")


class WelcomeDialog(QDialog):
    """First-time onboarding dialog showing the basic workflow."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to SAM2 Studio")
        self.setMinimumWidth(520)
        self.setStyleSheet(
            f"QDialog {{ background: {Colors.BG_MEDIUM}; "
            f"color: {Colors.TEXT_PRIMARY}; }}"
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(32, 24, 32, 24)

        # Title
        title = QLabel("Welcome to SAM2 Studio")
        title.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-size: 22px; "
            f"font-weight: 700; background: transparent;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Mask Generator for DIC & ROI Recognition")
        subtitle.setStyleSheet(
            f"color: {Colors.TEXT_DIM}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent;"
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        # Steps
        steps = [
            (
                "1. Select Input Directory",
                "Choose a folder containing your image sequence "
                "(TIFF, PNG, JPG, BMP supported).",
            ),
            (
                "2. Add Annotation Points",
                "Use the Draw tool (D) to place foreground (green) and "
                "background (red) points. Press Space to toggle mode.",
            ),
            (
                "3. Start Processing",
                "Press Ctrl+Enter or click the Start button. "
                "SAM2 will propagate your annotations across all frames.",
            ),
            (
                "4. Review & Correct",
                "Navigate frames with arrow keys. Add correction points "
                "if needed and re-propagate from that frame.",
            ),
            (
                "5. Post-Process (Optional)",
                "Apply spatial or temporal smoothing to refine mask boundaries. "
                "Use preprocessing for image enhancement before SAM2.",
            ),
        ]

        for step_title, step_desc in steps:
            step_label = QLabel(
                f"<b style='color: {Colors.PRIMARY};'>{step_title}</b>"
                f"<br/>"
                f"<span style='color: {Colors.TEXT_SECONDARY};'>{step_desc}</span>"
            )
            step_label.setWordWrap(True)
            step_label.setStyleSheet(
                f"font-size: {Fonts.SIZE_BASE}px; "
                f"padding: 8px 12px; "
                f"background: {Colors.BG_DARK}; "
                f"border-radius: 8px; "
                f"border-left: 3px solid {Colors.PRIMARY};"
            )
            layout.addWidget(step_label)

        # Tip
        tip = QLabel(
            f"<span style='color: {Colors.TEXT_DIM};'>"
            "Press F1 at any time to see all keyboard shortcuts."
            "</span>"
        )
        tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tip.setStyleSheet("background: transparent;")
        layout.addWidget(tip)

        # Don't show again checkbox
        self._dont_show = QCheckBox("Don't show this again")
        self._dont_show.setChecked(True)
        self._dont_show.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; "
            f"font-size: {Fonts.SIZE_SM}px; "
            f"background: transparent;"
        )
        layout.addWidget(self._dont_show)

        # OK button
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btn_box.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Get Started"
        )
        btn_box.accepted.connect(self._on_accept)
        layout.addWidget(btn_box)

    def _on_accept(self) -> None:
        if self._dont_show.isChecked():
            mark_welcome_dismissed()
        self.accept()
