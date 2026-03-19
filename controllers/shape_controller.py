"""Shape overlay controller for preprocessing regions.

Manages the list of shape overlays (add/cut regions) that are composited
into the preprocessing pipeline.  Emits signals so the GUI can update
the sidebar shape list and canvas visual overlays.
"""

from dataclasses import replace

from PyQt6.QtCore import QObject, pyqtSignal

from core.preprocessing import PreprocessingConfig, ShapeOverlay


class ShapeController(QObject):
    """Owns the shape-overlay list and exposes a signal-based API.

    Keeps MainWindow free of shape bookkeeping logic.
    """

    # Emitted after shapes change so the preview can refresh
    shapes_changed = pyqtSignal()  # trigger preview refresh

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._overlays: list[ShapeOverlay] = []

    # --- Public API ---

    @property
    def overlays(self) -> list[ShapeOverlay]:
        return list(self._overlays)

    def add_shape(self, mode: str, shape_type: str, points: tuple) -> int:
        """Add a shape overlay and return its index."""
        overlay = ShapeOverlay(mode=mode, shape_type=shape_type, points=points)
        index = len(self._overlays)
        self._overlays.append(overlay)
        self.shapes_changed.emit()
        return index

    def remove_shape(self, index: int) -> bool:
        """Remove a shape at the given index. Returns False if out of range."""
        if index < 0 or index >= len(self._overlays):
            return False
        self._overlays.pop(index)
        self.shapes_changed.emit()
        return True

    def clear(self) -> None:
        """Remove all shape overlays."""
        self._overlays.clear()
        self.shapes_changed.emit()

    def inject_shapes(self, config: PreprocessingConfig) -> PreprocessingConfig:
        """Return a copy of *config* with current shapes injected."""
        if not self._overlays:
            return config
        return replace(config, shape_overlays=tuple(self._overlays))
