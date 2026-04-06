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
    QSplitter,
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
    point_selection_toggled = pyqtSignal(int)
    delete_selected_requested = pyqtSignal()
    load_images_requested = pyqtSignal()

    # Shape drawing signals (forwarded from original panel)
    shape_confirmed = pyqtSignal(str, str, tuple)  # mode, shape_type, points
    shape_drawing_cancelled = pyqtSignal()

    # Mask brush / eraser signals (forwarded from the mask panel)
    brush_stroke_begun = pyqtSignal(float, float)
    brush_stroke_continued = pyqtSignal(float, float)
    brush_stroke_ended = pyqtSignal()

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
        zoom_out_btn.setToolTip("Zoom out (or scroll down)")
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
        zoom_in_btn.setToolTip("Zoom in (or scroll up)")
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
        reset_btn.setToolTip("Fit to window [Ctrl+1]")
        reset_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; border-radius: 4px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.05); }"
        )
        reset_btn.clicked.connect(self._reset_zoom)
        zoom_layout.addWidget(reset_btn)

        # Grid toggle
        self._grid_btn = QPushButton()
        self._grid_btn.setIcon(get_icon("grid", Colors.TEXT_DIM, 14))
        self._grid_btn.setFixedSize(28, 28)
        self._grid_btn.setFlat(True)
        self._grid_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._grid_btn.setToolTip("Toggle grid overlay")
        self._grid_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; border-radius: 4px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.05); }"
        )
        self._grid_btn.clicked.connect(self._toggle_grid)
        zoom_layout.addWidget(self._grid_btn)

        zoom_layout.addWidget(_create_zoom_divider())

        # A/B comparison toggle
        self._compare_btn = QPushButton("A/B")
        self._compare_btn.setFixedSize(36, 28)
        self._compare_btn.setFlat(True)
        self._compare_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._compare_btn.setToolTip("Toggle A/B comparison on overlay panel")
        self._compare_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; border-radius: 4px; "
            f"color: {Colors.TEXT_DIM}; font-size: {Fonts.SIZE_SM}px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: rgba(255,255,255,0.05); }}"
        )
        self._compare_btn.clicked.connect(self._toggle_ab_compare)
        zoom_layout.addWidget(self._compare_btn)

        main_layout.addWidget(zoom_bar)

        # --- Canvas panels (resizable splitter) ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setContentsMargins(8, 8, 8, 8)
        splitter.setHandleWidth(6)
        splitter.setStyleSheet(
            "QSplitter::handle {"
            f"  background: {Colors.BORDER};"
            "  border-radius: 2px;"
            "  margin: 8px 0px;"
            "}"
            "QSplitter::handle:hover {"
            f"  background: {Colors.PRIMARY};"
            "}"
        )

        # Original Frame
        self._original = CanvasPanel(
            "Original Frame", badge="Input", is_interactive=True
        )
        self._original.point_added.connect(self.point_added.emit)
        self._original.point_moved.connect(self.point_moved.emit)
        self._original.point_removed.connect(self.point_removed.emit)
        self._original.point_selection_toggled.connect(
            self.point_selection_toggled.emit
        )
        self._original.delete_selected_requested.connect(
            self.delete_selected_requested.emit
        )
        self._original.load_images_requested.connect(
            self.load_images_requested.emit
        )
        self._original.zoom_changed.connect(
            lambda pct: self._on_panel_zoom_changed(self._original, pct)
        )
        self._original.shape_confirmed.connect(self.shape_confirmed.emit)
        self._original.shape_drawing_cancelled.connect(
            self.shape_drawing_cancelled.emit
        )
        splitter.addWidget(self._original)

        # Mask Preview
        self._mask = CanvasPanel("Mask Preview", badge="Segmentation")
        self._mask.zoom_changed.connect(
            lambda pct: self._on_panel_zoom_changed(self._mask, pct)
        )
        # Forward brush signals so MainWindow can wire them to the
        # ManualEditController without reaching into the panel directly.
        self._mask.brush_stroke_begun.connect(self.brush_stroke_begun.emit)
        self._mask.brush_stroke_continued.connect(
            self.brush_stroke_continued.emit
        )
        self._mask.brush_stroke_ended.connect(self.brush_stroke_ended.emit)
        splitter.addWidget(self._mask)

        # Overlay
        self._overlay = CanvasPanel("Overlay", badge="Result")
        self._overlay.zoom_changed.connect(
            lambda pct: self._on_panel_zoom_changed(self._overlay, pct)
        )
        splitter.addWidget(self._overlay)

        # Equal initial proportions (1:1:1)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)

        main_layout.addWidget(splitter, 1)

        self._splitter = splitter
        self._current_zoom = 100
        self._ab_active = False
        self._grid_active = False
        self._maximized_panel: CanvasPanel | None = None
        self._cached_overlay: np.ndarray | None = None
        self._cached_original: np.ndarray | None = None

        # Connect maximize signals from each panel
        self._original.maximize_toggled.connect(self._toggle_panel_maximize)
        self._mask.maximize_toggled.connect(self._toggle_panel_maximize)
        self._overlay.maximize_toggled.connect(self._toggle_panel_maximize)

        # Pan synchronization across panels
        self._syncing_pan = False
        self._original.pan_changed.connect(
            lambda h, v: self._sync_pan_from(self._original, h, v)
        )
        self._mask.pan_changed.connect(
            lambda h, v: self._sync_pan_from(self._mask, h, v)
        )
        self._overlay.pan_changed.connect(
            lambda h, v: self._sync_pan_from(self._overlay, h, v)
        )

    # --- Public API ---

    def set_original_image(self, image: np.ndarray) -> None:
        """Set the original frame image."""
        self._cached_original = image
        self._original.set_image(image)
        if self._ab_active:
            self._overlay.set_image(image)

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
        self._cached_overlay = overlay
        if not self._ab_active:
            self._overlay.set_image(overlay)

    def set_annotation_points(
        self, points: list, labels: list
    ) -> None:
        """Update annotation points on the original frame panel."""
        self._original.set_points(points, labels)

    def set_active_tool(self, tool: str) -> None:
        """Set the active tool on the interactive panel."""
        self._original.set_active_tool(tool)

    # --- Mask edit (Step 0) API ---

    def set_mask_edit_mode(self, enabled: bool, tool: str = "brush") -> None:
        """Enable or disable mask brush editing on the Mask Preview panel.

        When enabled, the mask panel activates `brush`/`eraser` tools that
        emit `brush_stroke_*` signals. When disabled, the mask panel
        reverts to the non-interactive default ("select").
        """
        if enabled:
            self._mask.set_active_tool(tool)
        else:
            self._mask.set_active_tool("select")

    def set_mask_brush_tool(self, tool: str) -> None:
        """Switch between 'brush' and 'eraser' on the Mask Preview panel."""
        if tool not in ("brush", "eraser"):
            return
        self._mask.set_active_tool(tool)

    def set_mask_brush_radius(self, radius: int) -> None:
        """Set the mask brush/eraser radius in image pixels."""
        self._mask.set_brush_radius(radius)

    def clear_all(self) -> None:
        """Clear all three panels."""
        self._original.clear_image()
        self._mask.clear_image()
        self._overlay.clear_image()

    # --- Shape drawing API ---

    def enter_shape_draw_mode(self, mode: str, shape_type: str) -> None:
        """Enter shape drawing mode on the original panel."""
        self._original.enter_shape_draw_mode(mode, shape_type)

    def exit_shape_draw_mode(self) -> None:
        """Cancel any active shape drawing."""
        self._original.exit_shape_draw_mode()

    @property
    def is_shape_drawing(self) -> bool:
        """True if shape drawing is in progress."""
        return self._original.is_shape_drawing

    def add_confirmed_shape(
        self, mode: str, shape_type: str, points: tuple, index: int,
    ) -> None:
        """Add a confirmed shape overlay to the original panel."""
        self._original.add_confirmed_shape(mode, shape_type, points, index)

    def remove_confirmed_shape(self, index: int) -> None:
        """Remove a confirmed shape overlay by index."""
        self._original.remove_confirmed_shape(index)

    def clear_confirmed_shapes(self) -> None:
        """Remove all confirmed shape overlays."""
        self._original.clear_confirmed_shapes()

    def highlight_shape(self, index: int) -> None:
        """Highlight a specific shape on the canvas."""
        self._original.highlight_shape(index)

    def get_selected_point_indices(self) -> list[int]:
        """Return indices of selected annotation points."""
        return self._original.get_selected_indices()

    def clear_point_selection(self) -> None:
        """Deselect all annotation points."""
        self._original.clear_point_selection()

    # --- Private slots ---

    def _on_panel_zoom_changed(self, source: CanvasPanel, percent: int) -> None:
        """Sync zoom from one panel's wheel event to the others."""
        if self._syncing_pan:
            return
        self._current_zoom = percent
        self._zoom_label.setText(f"{percent}%")
        self._syncing_pan = True
        try:
            for panel in (self._original, self._mask, self._overlay):
                if panel is not source:
                    panel.set_zoom(percent)
        finally:
            self._syncing_pan = False

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

    def reset_zoom(self) -> None:
        """Public: reset zoom to 100% for all panels."""
        self._set_all_zoom(100)

    def fit_in_view(self) -> None:
        """Public: fit all panels to their view area."""
        self._reset_zoom()

    def _toggle_ab_compare(self) -> None:
        """Toggle A/B comparison mode on the overlay panel."""
        self._ab_active = not self._ab_active
        if self._ab_active:
            self._compare_btn.setStyleSheet(
                f"QPushButton {{ background: {Colors.PRIMARY_BG}; border: none; "
                f"border-radius: 4px; color: {Colors.PRIMARY}; "
                f"font-size: {Fonts.SIZE_SM}px; font-weight: bold; }}"
                f"QPushButton:hover {{ background: rgba(99,102,241,0.2); }}"
            )
            if self._cached_original is not None:
                self._overlay.set_image(self._cached_original)
        else:
            self._compare_btn.setStyleSheet(
                f"QPushButton {{ background: transparent; border: none; "
                f"border-radius: 4px; color: {Colors.TEXT_DIM}; "
                f"font-size: {Fonts.SIZE_SM}px; font-weight: bold; }}"
                f"QPushButton:hover {{ background: rgba(255,255,255,0.05); }}"
            )
            if self._cached_overlay is not None:
                self._overlay.set_image(self._cached_overlay)

    def _sync_pan_from(self, source: CanvasPanel, h_val: int, v_val: int) -> None:
        """Sync scrollbar positions from source panel to the other two."""
        if self._syncing_pan:
            return
        self._syncing_pan = True
        try:
            for panel in (self._original, self._mask, self._overlay):
                if panel is not source:
                    panel.sync_scroll(h_val, v_val)
        finally:
            self._syncing_pan = False

    def _toggle_grid(self) -> None:
        """Toggle grid overlay on all three panels."""
        self._grid_active = not self._grid_active
        for panel in (self._original, self._mask, self._overlay):
            panel.set_grid_visible(self._grid_active)
        # Update button style
        if self._grid_active:
            self._grid_btn.setStyleSheet(
                f"QPushButton {{ background: {Colors.PRIMARY_BG}; border: none; "
                f"border-radius: 4px; }}"
                f"QPushButton:hover {{ background: rgba(99,102,241,0.2); }}"
            )
        else:
            self._grid_btn.setStyleSheet(
                "QPushButton { background: transparent; border: none; border-radius: 4px; }"
                "QPushButton:hover { background: rgba(255,255,255,0.05); }"
            )

    def _toggle_panel_maximize(self, panel: CanvasPanel) -> None:
        """Maximize a single panel (hide others) or restore all three."""
        panels = [self._original, self._mask, self._overlay]
        if self._maximized_panel is panel:
            # Restore all panels
            for p in panels:
                p.setVisible(True)
            self._maximized_panel = None
            panel._max_btn.setIcon(
                get_icon("maximize", Colors.TEXT_DIM, 14)
            )
            panel._max_btn.setToolTip("Maximize this panel")
        else:
            # Maximize this panel, hide others
            for p in panels:
                p.setVisible(p is panel)
            if self._maximized_panel is not None:
                self._maximized_panel._max_btn.setIcon(
                    get_icon("maximize", Colors.TEXT_DIM, 14)
                )
                self._maximized_panel._max_btn.setToolTip(
                    "Maximize this panel"
                )
            self._maximized_panel = panel
            panel._max_btn.setIcon(
                get_icon("minimize", Colors.TEXT_DIM, 14)
            )
            panel._max_btn.setToolTip("Restore all panels")

    def _set_all_zoom(self, percent: int) -> None:
        self._current_zoom = percent
        self._zoom_label.setText(f"{percent}%")
        self._original.set_zoom(percent)
        self._mask.set_zoom(percent)
        self._overlay.set_zoom(percent)
