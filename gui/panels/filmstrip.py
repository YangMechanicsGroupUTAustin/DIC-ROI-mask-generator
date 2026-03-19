"""Horizontal filmstrip of frame thumbnails for quick navigation.

Displays small preview thumbnails of loaded frames in a scrollable
horizontal strip. Clicking a thumbnail navigates to that frame.
The current frame is highlighted with a colored border.
"""

from PyQt6.QtCore import pyqtSignal, QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QWidget,
)

from gui.theme import Colors, Fonts

_THUMB_SIZE = 64


class Filmstrip(QWidget):
    """Horizontal filmstrip of frame thumbnails."""

    frame_selected = pyqtSignal(int)  # 1-based frame index

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedHeight(_THUMB_SIZE + 24)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.setStyleSheet(
            f"Filmstrip {{ background: {Colors.BG_DARK}; "
            f"border-top: 1px solid {Colors.BORDER}; }}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        self._list = QListWidget()
        self._list.setFlow(QListWidget.Flow.LeftToRight)
        self._list.setWrapping(False)
        self._list.setViewMode(QListWidget.ViewMode.IconMode)
        self._list.setIconSize(QSize(_THUMB_SIZE, _THUMB_SIZE))
        self._list.setSpacing(2)
        self._list.setMovement(QListWidget.Movement.Static)
        self._list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        self._list.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._list.setStyleSheet(
            f"QListWidget {{ background: {Colors.BG_DARK}; border: none; }}"
            f"QListWidget::item {{ padding: 2px; border: 2px solid transparent; border-radius: 4px; }}"
            f"QListWidget::item:selected {{ border: 2px solid {Colors.PRIMARY}; "
            f"background: {Colors.PRIMARY_BG}; }}"
        )
        self._list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self._list, 1)

        self._image_files: list[str] = []
        self._thumbnails_loaded: set[int] = set()

    def set_image_files(self, files: list[str]) -> None:
        """Set the list of image file paths and create placeholder items."""
        self._list.blockSignals(True)
        self._list.clear()
        self._image_files = files
        self._thumbnails_loaded.clear()

        placeholder = QPixmap(_THUMB_SIZE, _THUMB_SIZE)
        placeholder.fill(QColor(Colors.BG_MEDIUM))

        for i, fpath in enumerate(files):
            item = QListWidgetItem()
            item.setIcon(QIcon(placeholder))
            item.setSizeHint(QSize(_THUMB_SIZE + 4, _THUMB_SIZE + 4))
            item.setToolTip(f"Frame {i + 1}")
            self._list.addItem(item)

        self._list.blockSignals(False)

        # Load thumbnails for visible range
        self._load_visible_thumbnails()

    def set_current_frame(self, frame: int) -> None:
        """Highlight and scroll to the given frame (1-based)."""
        idx = frame - 1
        if 0 <= idx < self._list.count():
            self._list.blockSignals(True)
            self._list.setCurrentRow(idx)
            self._list.scrollToItem(
                self._list.item(idx),
                QAbstractItemView.ScrollHint.PositionAtCenter,
            )
            self._list.blockSignals(False)
            self._load_visible_thumbnails()

    def update_mark_indicators(self, marked_frames: set[int]) -> None:
        """Update visual indicators for marked/bookmarked frames."""
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is None:
                continue
            frame_num = i + 1
            if frame_num in marked_frames:
                item.setToolTip(f"Frame {frame_num} [marked]")
            else:
                item.setToolTip(f"Frame {frame_num}")

    # --- Private ---

    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            self.frame_selected.emit(row + 1)

    def _load_visible_thumbnails(self) -> None:
        """Lazily load thumbnails for items near the viewport."""
        if not self._image_files:
            return

        # Determine visible range
        viewport = self._list.viewport()
        if viewport is None:
            return

        first_visible = 0
        last_visible = min(len(self._image_files) - 1, 20)

        # Try to find actual visible range
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is None:
                continue
            rect = self._list.visualItemRect(item)
            if rect.intersects(viewport.rect()):
                if first_visible == 0 and i > 0:
                    first_visible = i
                last_visible = i

        # Load with buffer
        start = max(0, first_visible - 5)
        end = min(len(self._image_files), last_visible + 10)

        for i in range(start, end):
            if i in self._thumbnails_loaded:
                continue
            self._load_thumbnail(i)

    def _load_thumbnail(self, index: int) -> None:
        """Load and set a thumbnail for the given index."""
        import cv2
        from core.image_processing import imread_safe

        fpath = self._image_files[index]
        img = imread_safe(fpath, cv2.IMREAD_COLOR)
        if img is None:
            return

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to thumbnail
        h, w = img.shape[:2]
        scale = _THUMB_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        thumb = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        qimg = QImage(
            thumb.data.tobytes(), new_w, new_h, new_w * 3,
            QImage.Format.Format_RGB888,
        ).copy()

        item = self._list.item(index)
        if item is not None:
            item.setIcon(QIcon(QPixmap.fromImage(qimg)))
            self._thumbnails_loaded.add(index)
