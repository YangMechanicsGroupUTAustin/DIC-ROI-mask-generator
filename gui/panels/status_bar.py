"""Bottom status bar with device info and timing.

Shows status indicator, device name, VRAM usage, and elapsed time
matching the Figma status-bar reference.
"""

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QWidget,
)

from gui.icons import get_icon
from gui.theme import Colors, Fonts


# Status level colors
_STATUS_COLORS = {
    "ready": Colors.SUCCESS,
    "processing": Colors.WARNING,
    "error": Colors.DANGER,
}


def _create_divider() -> QFrame:
    """Create a small vertical divider."""
    d = QFrame()
    d.setFixedWidth(1)
    d.setFixedHeight(12)
    d.setStyleSheet(f"background: {Colors.BORDER};")
    return d


class StatusBar(QWidget):
    """Bottom status bar with device info, VRAM monitor, and timer."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.setFixedHeight(28)
        self.setStyleSheet(
            f"StatusBar {{ background: {Colors.BG_DARK}; "
            f"border-top: 1px solid {Colors.BORDER}; }}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(8)

        # --- Status indicator ---
        self._status_dot = QLabel()
        self._status_dot.setFixedSize(8, 8)
        self._set_dot_color(Colors.SUCCESS)
        layout.addWidget(self._status_dot)

        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet(
            f"color: {Colors.TEXT_DIM}; font-size: {Fonts.SIZE_SM}px; "
            f"background: transparent;"
        )
        layout.addWidget(self._status_label)

        layout.addWidget(_create_divider())

        # --- Device info ---
        cpu_icon = QLabel()
        cpu_icon.setPixmap(
            get_icon("cpu", Colors.TEXT_DIM, 12).pixmap(12, 12)
        )
        cpu_icon.setFixedSize(12, 12)
        cpu_icon.setStyleSheet("background: transparent;")
        layout.addWidget(cpu_icon)

        self._device_label = QLabel("CPU")
        self._device_label.setStyleSheet(
            f"color: {Colors.TEXT_DIM}; font-size: {Fonts.SIZE_SM}px; "
            f"background: transparent;"
        )
        layout.addWidget(self._device_label)

        layout.addWidget(_create_divider())

        # --- VRAM usage ---
        hdd_icon = QLabel()
        hdd_icon.setPixmap(
            get_icon("hard-drive", Colors.TEXT_DIM, 12).pixmap(12, 12)
        )
        hdd_icon.setFixedSize(12, 12)
        hdd_icon.setStyleSheet("background: transparent;")
        layout.addWidget(hdd_icon)

        self._vram_label = QLabel("VRAM: 0.0 / 0.0 GB")
        self._vram_label.setStyleSheet(
            f"color: {Colors.TEXT_DIM}; font-size: {Fonts.SIZE_SM}px; "
            f"background: transparent;"
        )
        layout.addWidget(self._vram_label)

        self._vram_bar = QProgressBar()
        self._vram_bar.setFixedWidth(64)
        self._vram_bar.setFixedHeight(4)
        self._vram_bar.setRange(0, 100)
        self._vram_bar.setValue(0)
        self._vram_bar.setTextVisible(False)
        layout.addWidget(self._vram_bar)

        # --- Processing progress (hidden until processing starts) ---
        self._progress_container = QWidget()
        self._progress_container.setStyleSheet("background: transparent;")
        progress_layout = QHBoxLayout(self._progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(6)

        progress_layout.addWidget(_create_divider())

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedWidth(100)
        self._progress_bar.setFixedHeight(4)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        progress_layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet(
            f"color: {Colors.TEXT_DIM}; font-size: {Fonts.SIZE_SM}px; "
            f"background: transparent;"
        )
        progress_layout.addWidget(self._progress_label)

        self._progress_container.setVisible(False)
        layout.addWidget(self._progress_container)

        # Spacer
        layout.addStretch()

        # --- Elapsed time ---
        clock_icon = QLabel()
        clock_icon.setPixmap(
            get_icon("clock", Colors.TEXT_DIM, 12).pixmap(12, 12)
        )
        clock_icon.setFixedSize(12, 12)
        clock_icon.setStyleSheet("background: transparent;")
        layout.addWidget(clock_icon)

        self._time_label = QLabel("Elapsed: 00:00:00")
        self._time_label.setStyleSheet(
            f"color: {Colors.TEXT_DIM}; font-size: {Fonts.SIZE_SM}px; "
            f"background: transparent;"
        )
        layout.addWidget(self._time_label)

        # Timer
        self._elapsed_seconds = 0
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

    # --- Public API ---

    def set_status(self, text: str, level: str = "ready") -> None:
        """Set status text and indicator color.

        Args:
            text: Status message to display.
            level: One of 'ready', 'processing', 'error'.
        """
        self._status_label.setText(text)
        color = _STATUS_COLORS.get(level, Colors.SUCCESS)
        self._set_dot_color(color)

    def set_device_info(self, name: str) -> None:
        """Set the device name display."""
        self._device_label.setText(name)

    def set_vram_usage(self, used_gb: float, total_gb: float) -> None:
        """Update VRAM usage display and progress bar."""
        self._vram_label.setText(f"VRAM: {used_gb:.1f} / {total_gb:.1f} GB")
        if total_gb > 0:
            percent = int((used_gb / total_gb) * 100)
            self._vram_bar.setValue(min(100, percent))
        else:
            self._vram_bar.setValue(0)

    def set_processing_progress(
        self, current: int, total: int, eta_text: str = "",
    ) -> None:
        """Update the processing progress bar and ETA label.

        Args:
            current: Number of items processed so far.
            total: Total number of items.
            eta_text: Formatted ETA string (e.g. "~30s remaining").
        """
        if total <= 0:
            # Indeterminate progress (e.g. model loading)
            self._progress_bar.setRange(0, 0)
            self._progress_label.setText(eta_text or "Initializing...")
            self._progress_container.setVisible(True)
            return

        percent = int((current / total) * 100)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(min(100, percent))
        label = f"Frame {current}/{total}"
        if eta_text:
            label += f" \u2022 {eta_text}"
        self._progress_label.setText(label)
        self._progress_container.setVisible(True)

    def hide_processing_progress(self) -> None:
        """Hide the processing progress bar."""
        self._progress_container.setVisible(False)
        self._progress_bar.setValue(0)
        self._progress_label.setText("")

    def start_timer(self) -> None:
        """Start the elapsed time timer."""
        self._timer.start()

    def stop_timer(self) -> None:
        """Stop the elapsed time timer."""
        self._timer.stop()

    def reset_timer(self) -> None:
        """Reset elapsed time to zero."""
        self._timer.stop()
        self._elapsed_seconds = 0
        self._update_time_display()

    # --- Private methods ---

    def _set_dot_color(self, color: str) -> None:
        self._status_dot.setStyleSheet(
            f"QLabel {{ background: {color}; "
            f"border-radius: 4px; border: none; }}"
        )

    def _tick(self) -> None:
        self._elapsed_seconds += 1
        self._update_time_display()

    def _update_time_display(self) -> None:
        hours = self._elapsed_seconds // 3600
        minutes = (self._elapsed_seconds % 3600) // 60
        seconds = self._elapsed_seconds % 60
        self._time_label.setText(f"Elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}")
