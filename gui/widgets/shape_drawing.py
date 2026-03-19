"""Interactive shape drawing tools for the canvas.

Provides a controller that manages rect/circle/polygon drawing with
preview rendering and auto-confirm on release.

State machine:  IDLE -> DRAWING -> IDLE
  - DRAWING: user is actively drawing (drag for rect/circle, click for polygon)
  - Shape is confirmed automatically on mouse release (rect/circle) or
    double-click (polygon). No manual Enter required.
"""

from __future__ import annotations

import logging
from enum import Enum, auto

logger = logging.getLogger("sam2studio.shape_drawing")

from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
)

# --- Visual constants ---

_ADD_FILL = QColor(255, 255, 255, 60)
_ADD_BORDER = QColor(76, 175, 80)       # green
_CUT_FILL = QColor(0, 0, 0, 80)
_CUT_BORDER = QColor(244, 67, 54)       # red

# Minimum drag distance to create a shape (pixels in scene coords)
_MIN_SHAPE_SIZE = 3.0

# Distance threshold for deduplicating polygon points on double-click.
# Needs to be generous because scene coords can differ significantly
# between two clicks depending on zoom level.
_DOUBLE_CLICK_DEDUP_DIST = 15.0

# Z-values for layering
_Z_CONFIRMED = 50
_Z_PREVIEW = 150
_Z_TRACKING = 151
_Z_MARKER = 152


class DrawState(Enum):
    """Shape drawing state machine states."""

    IDLE = auto()
    DRAWING = auto()


def _colors_for_mode(mode: str) -> tuple[QColor, QColor]:
    """Return (fill, border) colors for add or cut mode."""
    if mode == "add":
        return QColor(_ADD_FILL), QColor(_ADD_BORDER)
    return QColor(_CUT_FILL), QColor(_CUT_BORDER)


class ShapeDrawController(QObject):
    """Manages interactive shape drawing on a QGraphicsScene.

    Shapes are confirmed automatically when the user releases the mouse
    (rect/circle) or double-clicks to close a polygon. No Enter key needed.

    Signals:
        shape_confirmed(str, str, tuple): mode, shape_type, points
        drawing_cancelled(): shape drawing was cancelled
    """

    shape_confirmed = pyqtSignal(str, str, tuple)
    drawing_cancelled = pyqtSignal()

    def __init__(
        self, scene: QGraphicsScene, parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._scene = scene
        self._state = DrawState.IDLE

        self._mode: str = ""
        self._shape_type: str = ""

        # Drawing state
        self._start_point: QPointF | None = None
        self._current_points: list[QPointF] = []

        # Graphics items for preview
        self._preview_item: QGraphicsItem | None = None
        self._tracking_line: QGraphicsLineItem | None = None
        self._point_markers: list[QGraphicsEllipseItem] = []

        # Confirmed shape overlays: index -> (item, mode)
        self._confirmed_items: dict[int, tuple[QGraphicsItem, str]] = {}

    @property
    def state(self) -> DrawState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state != DrawState.IDLE

    # --- Public API ---

    def enter_draw_mode(self, mode: str, shape_type: str) -> None:
        """Start a new shape drawing session."""
        self.cancel()
        self._mode = mode
        self._shape_type = shape_type
        self._state = DrawState.DRAWING
        self._start_point = None
        self._current_points.clear()

    def cancel(self) -> None:
        """Cancel current drawing and clean up all temporary items."""
        was_active = self._state != DrawState.IDLE
        self._cleanup_temp_items()
        self._state = DrawState.IDLE
        self._start_point = None
        self._current_points.clear()
        if was_active:
            self.drawing_cancelled.emit()

    # --- Event handlers (return True if consumed) ---

    def handle_mouse_press(self, scene_pos: QPointF) -> bool:
        if self._state != DrawState.DRAWING:
            return False

        if self._shape_type in ("rect", "circle"):
            self._start_point = scene_pos
            return True

        if self._shape_type == "polygon":
            self._current_points.append(scene_pos)
            self._add_polygon_marker(scene_pos)
            self._update_polygon_preview()
            return True

        return False

    def handle_mouse_move(self, scene_pos: QPointF) -> bool:
        if self._state != DrawState.DRAWING:
            return False

        if self._shape_type in ("rect", "circle") and self._start_point:
            self._update_shape_preview(scene_pos)
            return True

        if self._shape_type == "polygon" and self._current_points:
            self._update_tracking_line(scene_pos)
            return True

        return False

    def handle_mouse_release(self, scene_pos: QPointF) -> bool:
        if self._state != DrawState.DRAWING:
            return False

        if self._shape_type in ("rect", "circle"):
            if not self._start_point:
                return True  # Consume event even without start point

            # Check minimum size
            dx = scene_pos.x() - self._start_point.x()
            dy = scene_pos.y() - self._start_point.y()
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist < _MIN_SHAPE_SIZE:
                # Too small, stay in drawing state for retry
                self._start_point = None
                self._remove_preview()
                return True

            self._update_shape_preview(scene_pos)
            self._confirm_and_finish()
            return True

        return False

    def handle_double_click(self, scene_pos: QPointF) -> bool:
        if self._state != DrawState.DRAWING:
            return False

        if self._shape_type == "polygon" and len(self._current_points) >= 3:
            # The press event of the double-click already added a duplicate
            # point; remove it before closing.
            if len(self._current_points) >= 2:
                last = self._current_points[-1]
                prev = self._current_points[-2]
                dist = (
                    (last.x() - prev.x()) ** 2
                    + (last.y() - prev.y()) ** 2
                ) ** 0.5
                if dist < _DOUBLE_CLICK_DEDUP_DIST:
                    self._current_points.pop()
                    if self._point_markers:
                        self._scene.removeItem(self._point_markers.pop())

            if len(self._current_points) >= 3:
                self._update_polygon_preview()
                self._confirm_and_finish()
                return True

        return False

    def handle_key_press(self, key: int) -> bool:
        if self._state == DrawState.IDLE:
            return False

        if key == Qt.Key.Key_Escape:
            self.cancel()
            return True

        return False

    # --- Confirmed shape overlay management ---

    def add_confirmed_shape(
        self, mode: str, shape_type: str, points: tuple, index: int,
    ) -> QGraphicsItem | None:
        """Add a confirmed shape overlay to the scene."""
        fill, border = _colors_for_mode(mode)
        fill.setAlpha(40)
        pen = QPen(border, 2, Qt.PenStyle.DashLine)

        item: QGraphicsItem | None = None
        if shape_type == "rect":
            x1, y1, x2, y2 = points
            item = QGraphicsRectItem(QRectF(x1, y1, x2 - x1, y2 - y1))
        elif shape_type == "circle":
            cx, cy, r = points
            item = QGraphicsEllipseItem(cx - r, cy - r, 2 * r, 2 * r)
        elif shape_type == "polygon":
            poly = QPolygonF([QPointF(x, y) for x, y in points])
            item = QGraphicsPolygonItem(poly)

        if item is not None:
            item.setBrush(QBrush(fill))
            item.setPen(pen)
            item.setZValue(_Z_CONFIRMED)
            item.setData(0, index)
            self._scene.addItem(item)
            self._confirmed_items[index] = (item, mode)

        return item

    def remove_confirmed_shape(self, index: int) -> None:
        """Remove a confirmed shape overlay by index."""
        entry = self._confirmed_items.pop(index, None)
        if entry is not None:
            self._scene.removeItem(entry[0])

    def clear_confirmed_shapes(self) -> None:
        """Remove all confirmed shape overlays."""
        for item, _mode in self._confirmed_items.values():
            self._scene.removeItem(item)
        self._confirmed_items.clear()

    def highlight_shape(self, index: int) -> None:
        """Highlight a confirmed shape (thicker border)."""
        for idx, (item, mode) in self._confirmed_items.items():
            if idx == index:
                _fill, border = _colors_for_mode(mode)
                pen = QPen(border, 3, Qt.PenStyle.SolidLine)
            else:
                _fill, border = _colors_for_mode(mode)
                pen = QPen(border, 2, Qt.PenStyle.DashLine)
            item.setPen(pen)

    # --- Private: preview rendering ---

    def _update_shape_preview(self, current: QPointF) -> None:
        """Update rect/circle preview item during drag."""
        fill, border = _colors_for_mode(self._mode)
        pen = QPen(border, 2, Qt.PenStyle.DashLine)

        self._remove_preview()

        if self._shape_type == "rect":
            rect = QRectF(self._start_point, current).normalized()
            item = QGraphicsRectItem(rect)
        else:  # circle
            dx = current.x() - self._start_point.x()
            dy = current.y() - self._start_point.y()
            radius = (dx ** 2 + dy ** 2) ** 0.5
            cx, cy = self._start_point.x(), self._start_point.y()
            item = QGraphicsEllipseItem(
                cx - radius, cy - radius, 2 * radius, 2 * radius,
            )

        item.setBrush(QBrush(fill))
        item.setPen(pen)
        item.setZValue(_Z_PREVIEW)
        self._scene.addItem(item)
        self._preview_item = item

    def _update_polygon_preview(self) -> None:
        """Update polygon preview from accumulated points."""
        self._remove_preview()

        if len(self._current_points) < 2:
            return

        fill, border = _colors_for_mode(self._mode)
        pen = QPen(border, 2, Qt.PenStyle.DashLine)

        poly = QPolygonF(self._current_points)
        item = QGraphicsPolygonItem(poly)
        item.setBrush(QBrush(fill))
        item.setPen(pen)
        item.setZValue(_Z_PREVIEW)
        self._scene.addItem(item)
        self._preview_item = item

    def _update_tracking_line(self, cursor: QPointF) -> None:
        """Update the dotted line from last polygon point to cursor."""
        if self._tracking_line:
            self._scene.removeItem(self._tracking_line)
            self._tracking_line = None

        if not self._current_points:
            return

        _, border = _colors_for_mode(self._mode)
        pen = QPen(border, 1, Qt.PenStyle.DotLine)
        last = self._current_points[-1]
        line = QGraphicsLineItem(
            last.x(), last.y(), cursor.x(), cursor.y(),
        )
        line.setPen(pen)
        line.setZValue(_Z_TRACKING)
        self._scene.addItem(line)
        self._tracking_line = line

    def _add_polygon_marker(self, pos: QPointF) -> None:
        """Add a small dot at a polygon vertex."""
        _, border = _colors_for_mode(self._mode)
        r = 4.0
        marker = QGraphicsEllipseItem(
            pos.x() - r, pos.y() - r, 2 * r, 2 * r,
        )
        marker.setBrush(QBrush(border))
        marker.setPen(QPen(Qt.PenStyle.NoPen))
        marker.setZValue(_Z_MARKER)
        self._scene.addItem(marker)
        self._point_markers.append(marker)

    # --- Private: confirm and finalize ---

    def _confirm_and_finish(self) -> None:
        """Compute final points, emit signal, and return to IDLE."""
        points = self._compute_final_points()
        if points is not None:
            self.shape_confirmed.emit(self._mode, self._shape_type, points)
        else:
            logger.warning(
                "Shape confirm failed: could not compute final points "
                "(shape_type=%s, preview=%r)",
                self._shape_type, type(self._preview_item).__name__,
            )

        self._cleanup_temp_items()
        self._state = DrawState.IDLE
        self._start_point = None
        self._current_points.clear()

    def _compute_final_points(self) -> tuple | None:
        """Compute final shape coordinates from the current preview."""
        if self._shape_type == "rect":
            if isinstance(self._preview_item, QGraphicsRectItem):
                rect = self._preview_item.rect()
                return (
                    int(rect.left()), int(rect.top()),
                    int(rect.right()), int(rect.bottom()),
                )
        elif self._shape_type == "circle":
            if isinstance(self._preview_item, QGraphicsEllipseItem):
                rect = self._preview_item.rect()
                return (
                    int(rect.center().x()),
                    int(rect.center().y()),
                    int(rect.width() / 2),
                )
        elif self._shape_type == "polygon":
            if self._current_points:
                return tuple(
                    (int(p.x()), int(p.y())) for p in self._current_points
                )
        return None

    # --- Private: cleanup ---

    def _remove_preview(self) -> None:
        if self._preview_item:
            self._scene.removeItem(self._preview_item)
            self._preview_item = None

    def _cleanup_temp_items(self) -> None:
        """Remove all temporary drawing items (preview, markers, tracking)."""
        self._remove_preview()
        if self._tracking_line:
            self._scene.removeItem(self._tracking_line)
            self._tracking_line = None
        for m in self._point_markers:
            self._scene.removeItem(m)
        self._point_markers.clear()
