"""Single image viewer panel with optional annotation support.

Features zoom/pan via QGraphicsView, annotation point overlay,
and a placeholder state when no image is loaded.
"""

from typing import Optional

import numpy as np
from PyQt6.QtCore import pyqtSignal, QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.icons import get_icon, get_pixmap
from gui.theme import Colors, Fonts


# Annotation point radius as fraction of image's longest dimension
_POINT_RADIUS_FRACTION = 0.005  # 0.5%
_POINT_RADIUS_DEFAULT = 5.0    # fallback before image is loaded
_POINT_COLORS = {
    1: QColor(Colors.FG_POINT),  # Foreground (emerald)
    0: QColor(Colors.BG_POINT),  # Background (rose)
}


class _AnnotationPointItem(QGraphicsEllipseItem):
    """A draggable annotation point drawn on the image."""

    def __init__(self, x: float, y: float, label: int, index: int, radius: float):
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        self.setPos(x, y)
        self.index = index
        self.label = label

        color = _POINT_COLORS.get(label, _POINT_COLORS[1])
        self.setBrush(QBrush(color))
        pen_width = max(1.0, radius * 0.3)
        self.setPen(QPen(QColor("white"), pen_width))
        self.setZValue(100)
        self.setFlags(
            QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )


class _ImageView(QGraphicsView):
    """Custom QGraphicsView with mouse wheel zoom and pan support."""

    # Signals emitted in image-space coordinates
    point_clicked = pyqtSignal(float, float)     # image x, y
    point_erased = pyqtSignal(int)               # point index
    zoom_changed = pyqtSignal(int)               # zoom percentage

    def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None):
        super().__init__(scene, parent)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setResizeAnchor(
            QGraphicsView.ViewportAnchor.AnchorViewCenter
        )
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setStyleSheet(
            f"QGraphicsView {{ background: {Colors.BG_DARKEST}; border: none; }}"
        )
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._is_interactive = False
        self._active_tool = "select"
        self._is_panning = False
        self._pan_start = QPointF()
        self._zoom_factor = 1.0

    def set_interactive(self, enabled: bool) -> None:
        """Enable or disable annotation interaction."""
        self._is_interactive = enabled

    def set_active_tool(self, tool: str) -> None:
        """Set the active tool: 'select', 'draw', 'erase'."""
        self._active_tool = tool
        if tool == "draw":
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif tool == "erase":
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom with mouse wheel."""
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom_factor *= factor
        self._zoom_factor = max(0.1, min(20.0, self._zoom_factor))
        self.scale(factor, factor)
        self.zoom_changed.emit(int(self._zoom_factor * 100))

    def mousePressEvent(self, event) -> None:
        # Middle button or Space+Left for panning
        if event.button() == Qt.MouseButton.MiddleButton:
            self._start_pan(event)
            return

        if self._is_interactive and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())

            if self._active_tool == "erase":
                # Check if clicking on an annotation point
                item = self.scene().itemAt(scene_pos, self.transform())
                if isinstance(item, _AnnotationPointItem):
                    self.point_erased.emit(item.index)
                    return

            if self._active_tool == "draw":
                # Check we're not clicking on an existing point
                item = self.scene().itemAt(scene_pos, self.transform())
                if not isinstance(item, _AnnotationPointItem):
                    self.point_clicked.emit(scene_pos.x(), scene_pos.y())
                    return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._is_panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._is_panning:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        super().mouseReleaseEvent(event)

    def fit_to_view(self) -> None:
        """Fit the scene content to the view bounds."""
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        # Compute new zoom factor from the transform
        transform = self.transform()
        self._zoom_factor = transform.m11()
        self.zoom_changed.emit(int(self._zoom_factor * 100))

    def set_zoom(self, percent: int) -> None:
        """Set absolute zoom level."""
        new_factor = percent / 100.0
        scale_by = new_factor / self._zoom_factor
        self._zoom_factor = new_factor
        self.scale(scale_by, scale_by)
        self.zoom_changed.emit(percent)

    def get_zoom(self) -> int:
        """Get current zoom percentage."""
        return int(self._zoom_factor * 100)

    def _start_pan(self, event) -> None:
        self._is_panning = True
        self._pan_start = event.pos()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)


class CanvasPanel(QWidget):
    """Single image viewer panel with optional annotation support."""

    point_added = pyqtSignal(float, float)       # image-space x, y
    point_moved = pyqtSignal(int, float, float)   # index, new_x, new_y
    point_removed = pyqtSignal(int)               # index
    load_images_requested = pyqtSignal()
    zoom_changed = pyqtSignal(int)                # zoom percentage

    def __init__(
        self,
        title: str,
        badge: str = "",
        is_interactive: bool = False,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._title_text = title
        self._badge_text = badge
        self._is_interactive = is_interactive
        self._show_overlay = True
        self._annotation_items: list[_AnnotationPointItem] = []
        self._has_image = False
        self._point_radius = _POINT_RADIUS_DEFAULT

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setStyleSheet(
            f"CanvasPanel {{ background: {Colors.BG_MEDIUM}; "
            f"border: 1px solid {Colors.BORDER}; border-radius: 10px; }}"
        )

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Header ---
        header = QWidget()
        header.setFixedHeight(32)
        header.setStyleSheet(
            f"QWidget {{ background: {Colors.BG_LIGHT}; "
            f"border-bottom: 1px solid {Colors.BORDER}; "
            f"border-top-left-radius: 10px; border-top-right-radius: 10px; }}"
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 0, 6, 0)
        header_layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent; border: none;"
        )
        header_layout.addWidget(title_label)

        if badge:
            badge_label = QLabel(badge)
            badge_label.setStyleSheet(
                f"color: {Colors.PRIMARY}; font-size: {Fonts.SIZE_XS}px; "
                f"background: {Colors.PRIMARY_BG}; "
                f"border: 1px solid {Colors.PRIMARY_BORDER}; "
                f"border-radius: 8px; padding: 1px 6px;"
            )
            header_layout.addWidget(badge_label)

        header_layout.addStretch()

        # Eye toggle
        self._eye_btn = QPushButton()
        self._eye_btn.setIcon(get_icon("eye", Colors.TEXT_DIM, 14))
        self._eye_btn.setFixedSize(24, 24)
        self._eye_btn.setFlat(True)
        self._eye_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._eye_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; }"
            "QPushButton:hover { background: rgba(255,255,255,0.05); border-radius: 4px; }"
        )
        self._eye_btn.clicked.connect(self._toggle_visibility)
        header_layout.addWidget(self._eye_btn)

        # Maximize button
        max_btn = QPushButton()
        max_btn.setIcon(get_icon("maximize", Colors.TEXT_DIM, 14))
        max_btn.setFixedSize(24, 24)
        max_btn.setFlat(True)
        max_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        max_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; }"
            "QPushButton:hover { background: rgba(255,255,255,0.05); border-radius: 4px; }"
        )
        header_layout.addWidget(max_btn)

        main_layout.addWidget(header)

        # --- Content stack: placeholder vs image view ---
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)

        # Graphics scene + view
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self._view = _ImageView(self._scene, self)
        self._view.set_interactive(is_interactive)
        self._view.point_clicked.connect(self._on_point_clicked)
        self._view.point_erased.connect(self.point_removed.emit)
        self._view.zoom_changed.connect(self.zoom_changed.emit)

        self._content_layout.addWidget(self._view)

        # Placeholder
        self._placeholder = self._create_placeholder()
        self._content_layout.addWidget(self._placeholder)

        main_layout.addWidget(self._content, 1)

        # Show placeholder initially
        self._view.setVisible(False)
        self._placeholder.setVisible(True)

    # --- Public API ---

    def set_image(self, image: np.ndarray) -> None:
        """Set displayed image (RGB uint8 ndarray or grayscale)."""
        if image.ndim == 2:
            # Grayscale -> RGB
            h, w = image.shape
            qimg = QImage(
                image.data.tobytes(), w, h, w, QImage.Format.Format_Grayscale8
            ).copy()
        else:
            h, w, ch = image.shape
            bytes_per_line = ch * w
            if ch == 3:
                qimg = QImage(
                    image.data.tobytes(), w, h, bytes_per_line,
                    QImage.Format.Format_RGB888,
                ).copy()
            else:
                qimg = QImage(
                    image.data.tobytes(), w, h, bytes_per_line,
                    QImage.Format.Format_RGBA8888,
                ).copy()

        pixmap = QPixmap.fromImage(qimg)

        # Save view state before modifying the scene so we can restore it
        # and prevent zoom/pan jumps when the image is replaced while
        # annotation points are present (e.g. during processing or preview).
        preserve_view = self._has_image and bool(self._annotation_items)
        if preserve_view:
            saved_transform = self._view.transform()
            saved_hval = self._view.horizontalScrollBar().value()
            saved_vval = self._view.verticalScrollBar().value()

        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))

        # Update point radius based on image dimensions
        longest = max(pixmap.width(), pixmap.height())
        self._point_radius = longest * _POINT_RADIUS_FRACTION

        self._has_image = True
        self._view.setVisible(True)
        self._placeholder.setVisible(False)

        if preserve_view:
            # Restore the user's zoom/pan position unchanged
            self._view.setTransform(saved_transform)
            self._view.horizontalScrollBar().setValue(saved_hval)
            self._view.verticalScrollBar().setValue(saved_vval)
        elif not self._annotation_items:
            self._view.fit_to_view()

    def set_points(
        self, points: list[list[float]], labels: list[int]
    ) -> None:
        """Update annotation point overlay."""
        self._clear_annotation_items()
        for i, (pt, lbl) in enumerate(zip(points, labels)):
            item = _AnnotationPointItem(pt[0], pt[1], lbl, i, self._point_radius)
            self._scene.addItem(item)
            self._annotation_items.append(item)

    def set_active_tool(self, tool: str) -> None:
        """Set active tool: 'select', 'draw', 'erase'."""
        self._view.set_active_tool(tool)
        # Disable point dragging unless select tool
        for item in self._annotation_items:
            item.setFlag(
                QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable,
                tool == "select",
            )

    def fit_to_view(self) -> None:
        """Fit image to view bounds."""
        self._view.fit_to_view()

    def set_zoom(self, percent: int) -> None:
        """Set zoom level."""
        self._view.set_zoom(percent)

    def get_zoom(self) -> int:
        """Get current zoom percentage."""
        return self._view.get_zoom()

    def clear_image(self) -> None:
        """Show placeholder."""
        self._pixmap_item.setPixmap(QPixmap())
        self._clear_annotation_items()
        self._has_image = False
        self._view.setVisible(False)
        self._placeholder.setVisible(True)

    def set_crosshair_visible(self, visible: bool) -> None:
        """Show/hide crosshair cursor."""
        if visible:
            self._view.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._view.setCursor(Qt.CursorShape.ArrowCursor)

    # --- Private methods ---

    def _create_placeholder(self) -> QWidget:
        """Create the placeholder widget shown when no image is loaded."""
        placeholder = QWidget()
        placeholder.setStyleSheet(
            f"QWidget {{ background: {Colors.BG_DARKEST}; }}"
        )
        layout = QVBoxLayout(placeholder)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)

        # Icon
        icon_label = QLabel()
        icon_label.setPixmap(get_pixmap("image", Colors.TEXT_DIM, 40))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(icon_label)

        # Text
        text = QLabel("No image loaded")
        text.setStyleSheet(
            f"color: {Colors.TEXT_DIM}; font-size: {Fonts.SIZE_MD}px; "
            f"background: transparent; border: none;"
        )
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(text)

        subtext = QLabel("Load images from the input directory to begin")
        subtext.setStyleSheet(
            f"color: {Colors.TEXT_DIM}; font-size: {Fonts.SIZE_SM}px; "
            f"background: transparent; border: none;"
        )
        subtext.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtext)

        # Load button (only for interactive panels)
        if self._is_interactive:
            load_btn = QPushButton("Load Images")
            load_btn.setProperty("cssClass", "btn-accent")
            load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            load_btn.clicked.connect(self.load_images_requested.emit)
            layout.addWidget(
                load_btn, 0, Qt.AlignmentFlag.AlignCenter
            )

        return placeholder

    def _clear_annotation_items(self) -> None:
        """Remove all annotation point graphics items."""
        for item in self._annotation_items:
            self._scene.removeItem(item)
        self._annotation_items.clear()

    def _toggle_visibility(self) -> None:
        """Toggle overlay visibility."""
        self._show_overlay = not self._show_overlay
        icon_name = "eye" if self._show_overlay else "eye-off"
        self._eye_btn.setIcon(get_icon(icon_name, Colors.TEXT_DIM, 14))
        # Toggle annotation visibility
        for item in self._annotation_items:
            item.setVisible(self._show_overlay)

    def _on_point_clicked(self, x: float, y: float) -> None:
        """Handle click on canvas when draw tool is active."""
        # Only emit if click is within the image bounds
        pixmap = self._pixmap_item.pixmap()
        if pixmap and not pixmap.isNull():
            if 0 <= x <= pixmap.width() and 0 <= y <= pixmap.height():
                self.point_added.emit(x, y)
