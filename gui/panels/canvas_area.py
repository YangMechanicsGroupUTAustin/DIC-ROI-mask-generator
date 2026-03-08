"""Three-panel canvas container with zoom controls.

Contains Original Frame, Mask Preview, and Overlay panels
with a synchronized zoom bar.
"""

import numpy as np
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.icons import get_icon
from gui.panels.canvas_panel import CanvasPanel
from gui.theme import Colors, Fonts


def _create_zoom_divider() -> QFrame:
    """Create a small vertical divider for the zoom bar."""
    d = QFrame()
    d.setFixedWidth(1)
    d.setFixedHeight(16)
    d.setStyleSheet(f"background: {Colors.BORDER};")
    return d


class CanvasArea(QWidget):
    """Three-panel canvas container with zoom controls."""

    # Forward signals from the original (interactive) panel
    point_added = pyqtSignal(float, float)
    point_moved = pyqtSignal(int, float, float)
    point_removed = pyqtSignal(int)
    load_images_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Zoom bar ---
        zoom_bar = QWidget()
        zoom_bar.setFixedHeight(32)
        zoom_bar.setStyleSheet(
            f"QWidget {{ background: rgba(15, 17, 23, 0.5); }}"
        )
        zoom_layout = QHBoxLayout(zoom_bar)
        zoom_layout.setContentsMargins(12, 0, 12, 0)
        zoom_layout.setSpacing(4)

        # "3 Panels" label
        layers_label = QLabel()
        layers_label.setPixmap(
            get_icon("layers", Colors.TEXT_DIM, 14).pixmap(14, 14)
        )
        layers_label.setFixedSize(14, 14)
        layers_label.setStyleSheet("background: transparent;")
        zoom_layout.addWidget(layers_label)

        panels_label = QLabel("3 Panels")
        panels_label.setStyleSheet(
            f"color: {Colors.TEXT_DIM}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent;"
        )
        zoom_layout.addWidget(panels_label)

        zoom_layout.addStretch()

        # Zoom out
        zoom_out_btn = QPushButton()
        zoom_out_btn.setIcon(get_icon("zoom-out", Colors.TEXT_DIM, 14))
        zoom_out_btn.setFixedSize(28, 28)
        zoom_out_btn.setFlat(True)
        zoom_out_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        zoom_out_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; border-radius: 4px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.05); }"
        )
        zoom_out_btn.clicked.connect(self._zoom_out)
        zoom_layout.addWidget(zoom_out_btn)

        # Zoom label
        self._zoom_label = QLabel("100%")
        self._zoom_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-family: '{Fonts.MONO}'; "
            f"font-size: {Fonts.SIZE_BASE}px; background: transparent; "
            f"min-width: 40px;"
        )
        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        zoom_layout.addWidget(self._zoom_label)

        # Zoom in
        zoom_in_btn = QPushButton()
        zoom_in_btn.setIcon(get_icon("zoom-in", Colors.TEXT_DIM, 14))
        zoom_in_btn.setFixedSize(28, 28)
        zoom_in_btn.setFlat(True)
        zoom_in_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        zoom_in_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; border-radius: 4px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.05); }"
        )
        zoom_in_btn.clicked.connect(self._zoom_in)
        zoom_layout.addWidget(zoom_in_btn)

        zoom_layout.addWidget(_create_zoom_divider())

        # Reset zoom
        reset_btn = QPushButton()
        reset_btn.setIcon(get_icon("rotate-ccw", Colors.TEXT_DIM, 14))
        reset_btn.setFixedSize(28, 28)
        reset_btn.setFlat(True)
        reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; border-radius: 4px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.05); }"
        )
        reset_btn.clicked.connect(self._reset_zoom)
        zoom_layout.addWidget(reset_btn)

        # Grid toggle
        grid_btn = QPushButton()
        grid_btn.setIcon(get_icon("grid", Colors.TEXT_DIM, 14))
        grid_btn.setFixedSize(28, 28)
        grid_btn.setFlat(True)
        grid_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        grid_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; border-radius: 4px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.05); }"
        )
        zoom_layout.addWidget(grid_btn)

        main_layout.addWidget(zoom_bar)

        # --- Canvas panels ---
        panels_widget = QWidget()
        panels_layout = QHBoxLayout(panels_widget)
        panels_layout.setContentsMargins(8, 8, 8, 8)
        panels_layout.setSpacing(8)

        # Original Frame (interactive, stretch=2)
        self._original = CanvasPanel(
            "Original Frame", badge="Input", is_interactive=True
        )
        self._original.point_added.connect(self.point_added.emit)
        self._original.point_moved.connect(self.point_moved.emit)
        self._original.point_removed.connect(self.point_removed.emit)
        self._original.load_images_requested.connect(
            self.load_images_requested.emit
        )
        self._original.zoom_changed.connect(self._on_zoom_changed)
        panels_layout.addWidget(self._original, 2)

        # Mask Preview (stretch=1)
        self._mask = CanvasPanel("Mask Preview", badge="Segmentation")
        panels_layout.addWidget(self._mask, 1)

        # Overlay (stretch=1)
        self._overlay = CanvasPanel("Overlay", badge="Result")
        panels_layout.addWidget(self._overlay, 1)

        main_layout.addWidget(panels_widget, 1)

        self._current_zoom = 100

    # --- Public API ---

    def set_original_image(self, image: np.ndarray) -> None:
        """Set the original frame image."""
        self._original.set_image(image)

    def set_mask_image(self, mask: np.ndarray) -> None:
        """Set the mask preview image.

        Accepts either a 2D grayscale or 3D RGB array.
        """
        if mask.ndim == 2:
            # Convert binary/grayscale mask to RGB for display
            rgb = np.stack([mask, mask, mask], axis=-1)
            self._mask.set_image(rgb)
        else:
            self._mask.set_image(mask)

    def set_overlay_image(self, overlay: np.ndarray) -> None:
        """Set the overlay result image."""
        self._overlay.set_image(overlay)

    def set_annotation_points(
        self, points: list, labels: list
    ) -> None:
        """Update annotation points on the original frame panel."""
        self._original.set_points(points, labels)

    def set_active_tool(self, tool: str) -> None:
        """Set the active tool on the interactive panel."""
        self._original.set_active_tool(tool)

    def clear_all(self) -> None:
        """Clear all three panels."""
        self._original.clear_image()
        self._mask.clear_image()
        self._overlay.clear_image()

    # --- Private slots ---

    def _on_zoom_changed(self, percent: int) -> None:
        self._current_zoom = percent
        self._zoom_label.setText(f"{percent}%")

    def _zoom_in(self) -> None:
        new_zoom = min(400, self._current_zoom + 25)
        self._set_all_zoom(new_zoom)

    def _zoom_out(self) -> None:
        new_zoom = max(25, self._current_zoom - 25)
        self._set_all_zoom(new_zoom)

    def _reset_zoom(self) -> None:
        self._original.fit_to_view()
        self._mask.fit_to_view()
        self._overlay.fit_to_view()
        # The zoom_changed signal from _original will update the label

    def _set_all_zoom(self, percent: int) -> None:
        self._current_zoom = percent
        self._zoom_label.setText(f"{percent}%")
        self._original.set_zoom(percent)
        self._mask.set_zoom(percent)
        self._overlay.set_zoom(percent)
