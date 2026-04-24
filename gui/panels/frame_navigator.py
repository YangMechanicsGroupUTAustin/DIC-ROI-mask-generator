"""Bottom frame navigation bar.

Contains frame range controls, prev/next navigation,
and a preview slider matching the Figma FrameNavigator reference.
"""

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QWidget,
)

from gui.icons import get_icon
from gui.theme import Colors, Fonts


def _create_divider() -> QFrame:
    """Create a vertical divider."""
    d = QFrame()
    d.setFixedWidth(1)
    d.setFixedHeight(24)
    d.setStyleSheet(f"background: {Colors.BORDER};")
    return d


def _create_nav_spinbox(value: int = 1, max_val: int = 9999) -> QSpinBox:
    """Create a small styled spin box for frame range input."""
    sb = QSpinBox()
    sb.setRange(1, max_val)
    sb.setValue(value)
    sb.setFixedWidth(60)
    sb.setAlignment(Qt.AlignmentFlag.AlignCenter)
    sb.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
    return sb


def _create_uppercase_label(text: str) -> QLabel:
    """Create a small uppercase label."""
    label = QLabel(text)
    label.setStyleSheet(
        f"color: {Colors.TEXT_DIM}; font-size: {Fonts.SIZE_SM}px; "
        f"background: transparent; text-transform: uppercase; "
        f"letter-spacing: 1px;"
    )
    return label


class FrameNavigator(QWidget):
    """Bottom frame navigation bar."""

    frame_changed = pyqtSignal(int)        # 1-based
    start_frame_changed = pyqtSignal(int)
    end_frame_changed = pyqtSignal(int)
    mark_frame_toggled = pyqtSignal(int)   # toggle bookmark on frame
    jump_to_prev_mark = pyqtSignal()
    jump_to_next_mark = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.setFixedHeight(48)
        self.setStyleSheet(
            f"FrameNavigator {{ background: {Colors.BG_MEDIUM}; "
            f"border-top: 1px solid {Colors.BORDER}; }}"
        )

        self._total_frames = 1
        self._current_frame = 1

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(8)

        # --- Frame range ---
        layout.addWidget(_create_uppercase_label("START"))
        self._start_spin = _create_nav_spinbox(1)
        self._start_spin.setToolTip("First frame to process (1-based)")
        self._start_spin.valueChanged.connect(self._on_start_changed)
        layout.addWidget(self._start_spin)

        layout.addWidget(_create_uppercase_label("END"))
        self._end_spin = _create_nav_spinbox(1)
        self._end_spin.setToolTip("Last frame to process (1-based)")
        self._end_spin.valueChanged.connect(self._on_end_changed)
        layout.addWidget(self._end_spin)

        # Divider
        layout.addWidget(_create_divider())

        # --- Navigation ---
        prev_btn = QPushButton()
        prev_btn.setIcon(get_icon("chevron-left", Colors.TEXT_SECONDARY, 16))
        prev_btn.setFixedSize(28, 28)
        prev_btn.setFlat(True)
        prev_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        prev_btn.setToolTip("Previous frame [Left Arrow]")
        prev_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; border-radius: 4px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.05); }"
        )
        prev_btn.clicked.connect(self._prev_frame)
        layout.addWidget(prev_btn)

        # Frame display
        frame_display = QWidget()
        frame_display.setStyleSheet(
            f"QWidget {{ background: {Colors.BG_DARK}; "
            f"border: 1px solid {Colors.BORDER}; border-radius: 8px; }}"
        )
        frame_display_layout = QHBoxLayout(frame_display)
        frame_display_layout.setContentsMargins(10, 2, 10, 2)
        frame_display_layout.setSpacing(4)

        frame_prefix = QLabel("Frame")
        frame_prefix.setStyleSheet(
            f"color: {Colors.TEXT_DIM}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent; border: none;"
        )
        frame_display_layout.addWidget(frame_prefix)

        self._frame_num_label = QLabel("1")
        self._frame_num_label.setStyleSheet(
            f"color: {Colors.PRIMARY}; font-family: '{Fonts.MONO}'; "
            f"font-size: {Fonts.SIZE_MD}px; background: transparent; border: none;"
        )
        frame_display_layout.addWidget(self._frame_num_label)

        slash_label = QLabel("/")
        slash_label.setStyleSheet(
            f"color: {Colors.TEXT_DIM}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent; border: none;"
        )
        frame_display_layout.addWidget(slash_label)

        self._frame_total_label = QLabel("1")
        self._frame_total_label.setStyleSheet(
            f"color: {Colors.TEXT_DIM}; font-family: '{Fonts.MONO}'; "
            f"font-size: {Fonts.SIZE_BASE}px; background: transparent; border: none;"
        )
        frame_display_layout.addWidget(self._frame_total_label)

        layout.addWidget(frame_display)

        next_btn = QPushButton()
        next_btn.setIcon(get_icon("chevron-right", Colors.TEXT_SECONDARY, 16))
        next_btn.setFixedSize(28, 28)
        next_btn.setFlat(True)
        next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        next_btn.setToolTip("Next frame [Right Arrow]")
        next_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; border-radius: 4px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.05); }"
        )
        next_btn.clicked.connect(self._next_frame)
        layout.addWidget(next_btn)

        # Divider
        layout.addWidget(_create_divider())

        # --- Preview slider ---
        layout.addWidget(_create_uppercase_label("PREVIEW"))

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setToolTip("Drag to navigate frames quickly")
        self._slider.setRange(1, 1)
        self._slider.setValue(1)
        self._slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider, 1)

        # Divider
        layout.addWidget(_create_divider())

        # --- Bookmark controls ---
        # Previous marked frame
        prev_mark_btn = QPushButton()
        prev_mark_btn.setIcon(get_icon("skip-back", Colors.TEXT_DIM, 14))
        prev_mark_btn.setFixedSize(28, 28)
        prev_mark_btn.setFlat(True)
        prev_mark_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        prev_mark_btn.setToolTip("Previous marked frame")
        prev_mark_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; border-radius: 4px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.05); }"
        )
        prev_mark_btn.clicked.connect(self.jump_to_prev_mark.emit)
        layout.addWidget(prev_mark_btn)

        # Toggle mark on current frame
        self._mark_btn = QPushButton()
        self._mark_btn.setIcon(get_icon("bookmark", Colors.TEXT_DIM, 14))
        self._mark_btn.setFixedSize(28, 28)
        self._mark_btn.setFlat(True)
        self._mark_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._mark_btn.setToolTip("Toggle bookmark on current frame (M)")
        self._mark_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; border-radius: 4px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.05); }"
        )
        self._mark_btn.clicked.connect(self._toggle_mark)
        layout.addWidget(self._mark_btn)

        # Next marked frame
        next_mark_btn = QPushButton()
        next_mark_btn.setIcon(get_icon("skip-forward", Colors.TEXT_DIM, 14))
        next_mark_btn.setFixedSize(28, 28)
        next_mark_btn.setFlat(True)
        next_mark_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        next_mark_btn.setToolTip("Next marked frame")
        next_mark_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; border-radius: 4px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.05); }"
        )
        next_mark_btn.clicked.connect(self.jump_to_next_mark.emit)
        layout.addWidget(next_mark_btn)

        # Mark count label
        self._mark_count_label = QLabel("")
        self._mark_count_label.setStyleSheet(
            f"color: {Colors.TEXT_DIM}; font-size: {Fonts.SIZE_SM}px; "
            f"background: transparent;"
        )
        layout.addWidget(self._mark_count_label)

        self._is_current_marked = False

    # --- Public API ---

    def set_total_frames(self, total: int) -> None:
        """Set the total number of frames."""
        self._total_frames = max(1, total)
        self._frame_total_label.setText(str(self._total_frames))
        self._end_spin.setMaximum(self._total_frames)
        self._end_spin.setValue(self._total_frames)
        self._start_spin.setMaximum(self._total_frames)
        self._slider.setRange(1, self._total_frames)

    def set_current_frame(self, frame: int) -> None:
        """Set the current frame (1-based)."""
        frame = max(1, min(self._total_frames, frame))
        self._current_frame = frame
        self._frame_num_label.setText(str(frame))
        self._slider.blockSignals(True)
        self._slider.setValue(frame)
        self._slider.blockSignals(False)

    def get_frame_range(self) -> tuple[int, int]:
        """Get (start, end) frame range."""
        return self._start_spin.value(), self._end_spin.value()

    def update_mark_state(self, marked_frames: set[int]) -> None:
        """Update bookmark visual state for current frame and count."""
        self._is_current_marked = self._current_frame in marked_frames
        if self._is_current_marked:
            self._mark_btn.setIcon(get_icon("bookmark", Colors.PRIMARY, 14))
            self._mark_btn.setStyleSheet(
                f"QPushButton {{ background: {Colors.PRIMARY_BG}; border: none; border-radius: 4px; }}"
                f"QPushButton:hover {{ background: rgba(99,102,241,0.2); }}"
            )
        else:
            self._mark_btn.setIcon(get_icon("bookmark", Colors.TEXT_DIM, 14))
            self._mark_btn.setStyleSheet(
                "QPushButton { background: transparent; border: none; border-radius: 4px; }"
                "QPushButton:hover { background: rgba(255,255,255,0.05); }"
            )
        count = len(marked_frames)
        self._mark_count_label.setText(f"{count}" if count else "")

    # --- Private slots ---

    def _prev_frame(self) -> None:
        if self._current_frame > 1:
            self.set_current_frame(self._current_frame - 1)
            self.frame_changed.emit(self._current_frame)

    def _next_frame(self) -> None:
        if self._current_frame < self._total_frames:
            self.set_current_frame(self._current_frame + 1)
            self.frame_changed.emit(self._current_frame)

    def _on_slider_changed(self, value: int) -> None:
        self._current_frame = value
        self._frame_num_label.setText(str(value))
        self.frame_changed.emit(value)

    def _on_start_changed(self, value: int) -> None:
        self.start_frame_changed.emit(value)

    def _on_end_changed(self, value: int) -> None:
        self.end_frame_changed.emit(value)

    def _toggle_mark(self) -> None:
        self.mark_frame_toggled.emit(self._current_frame)
