"""Central application state with PyQt6 signals for reactivity.

Single source of truth for all shared state.
GUI panels observe state changes via signals.
Controllers modify state through methods.
"""

import os
from enum import Enum
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal


class AppState(QObject):
    """Central application state manager."""

    # --- State Machine ---
    class State(Enum):
        INIT = "init"
        ANNOTATING = "annotating"
        PROCESSING = "processing"
        REVIEWING = "reviewing"
        CORRECTION = "correction"
        POST_PROCESSING = "post_processing"

    # --- Signals ---
    state_changed = pyqtSignal(str)               # State enum value
    input_dir_changed = pyqtSignal(str)
    output_dir_changed = pyqtSignal(str)
    image_files_changed = pyqtSignal(list)         # list of file paths
    current_frame_changed = pyqtSignal(int)        # 1-based index
    frame_range_changed = pyqtSignal(int, int)     # start, end (1-based)
    points_changed = pyqtSignal(list, list)        # points [[x,y],...], labels [0|1,...]
    device_changed = pyqtSignal(str)
    model_changed = pyqtSignal(str)
    threshold_changed = pyqtSignal(float)
    format_changed = pyqtSignal(str)
    processing_progress = pyqtSignal(int, int, str)  # current, total, message
    processing_finished = pyqtSignal()
    processing_error = pyqtSignal(str)
    current_images_changed = pyqtSignal()          # original/mask/overlay updated
    vram_updated = pyqtSignal(float, float)        # used_gb, total_gb
    status_message = pyqtSignal(str, str)          # message, level (ready/processing/error/warning)

    # Model registry -- maps display name to (config_yaml, checkpoint_filename)
    MODEL_REGISTRY: dict[str, tuple[str, str]] = {
        "SAM2 Hiera Large":     ("sam2.1_hiera_l.yaml",   "sam2.1_hiera_large.pt"),
        "SAM2 Hiera Base Plus": ("sam2.1_hiera_b+.yaml",  "sam2.1_hiera_base_plus.pt"),
        "SAM2 Hiera Small":     ("sam2.1_hiera_s.yaml",   "sam2.1_hiera_small.pt"),
        "SAM2 Hiera Tiny":      ("sam2.1_hiera_t.yaml",   "sam2.1_hiera_tiny.pt"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        # State machine
        self._state = self.State.INIT

        # File paths
        self._input_dir = ""
        self._output_dir = ""
        self._image_files: list[str] = []

        # Frame navigation
        self._current_frame = 1    # 1-based
        self._start_frame = 1      # 1-based
        self._end_frame = 1        # 1-based

        # Annotations
        self._points: list[list[float]] = []   # [[x, y], ...]
        self._labels: list[int] = []            # [1=fg, 0=bg, ...]

        # Model config
        self._device = "cpu"
        self._model_name = "SAM2 Hiera Large"
        self._threshold = 0.0
        self._intermediate_format = "JPEG (fast)"
        self._force_reprocess = False

        # Display images (numpy arrays, RGB uint8)
        self._current_original: Optional[np.ndarray] = None
        self._current_mask: Optional[np.ndarray] = None
        self._current_overlay: Optional[np.ndarray] = None

        # SAM2 runtime (managed by processing controller)
        self.predictor = None
        self.inference_state = None

    # --- Properties with signal emission ---

    @property
    def state(self) -> 'AppState.State':
        return self._state

    def set_state(self, new_state: 'AppState.State') -> None:
        if self._state != new_state:
            self._state = new_state
            self.state_changed.emit(new_state.value)

    @property
    def input_dir(self) -> str:
        return self._input_dir

    def set_input_dir(self, path: str) -> None:
        if self._input_dir != path:
            self._input_dir = path
            self.input_dir_changed.emit(path)
            # Auto-discover images
            if os.path.isdir(path):
                from core.image_processing import get_image_files
                files = get_image_files(path)
                self.set_image_files(files)

    @property
    def output_dir(self) -> str:
        return self._output_dir

    def set_output_dir(self, path: str) -> None:
        if self._output_dir != path:
            self._output_dir = path
            self.output_dir_changed.emit(path)

    @property
    def image_files(self) -> list[str]:
        return self._image_files

    def set_image_files(self, files: list[str]) -> None:
        self._image_files = files
        total = len(files)
        if total > 0:
            self._start_frame = 1
            self._end_frame = total
            self._current_frame = 1
            self.frame_range_changed.emit(1, total)
            self.current_frame_changed.emit(1)
        self.image_files_changed.emit(files)

    @property
    def current_frame(self) -> int:
        return self._current_frame

    def set_current_frame(self, frame: int) -> None:
        total = len(self._image_files)
        frame = max(1, min(frame, total)) if total > 0 else 1
        if self._current_frame != frame:
            self._current_frame = frame
            self.current_frame_changed.emit(frame)

    @property
    def start_frame(self) -> int:
        return self._start_frame

    @property
    def end_frame(self) -> int:
        return self._end_frame

    def set_frame_range(self, start: int, end: int) -> None:
        total = len(self._image_files)
        start = max(1, min(start, total)) if total > 0 else 1
        end = max(start, min(end, total)) if total > 0 else 1
        changed = (self._start_frame != start) or (self._end_frame != end)
        self._start_frame = start
        self._end_frame = end
        if changed:
            self.frame_range_changed.emit(start, end)

    @property
    def points(self) -> list[list[float]]:
        return self._points

    @property
    def labels(self) -> list[int]:
        return self._labels

    def set_points(self, points: list[list[float]], labels: list[int]) -> None:
        self._points = list(points)
        self._labels = list(labels)
        self.points_changed.emit(self._points, self._labels)

    def add_point(self, x: float, y: float, label: int) -> None:
        self._points.append([x, y])
        self._labels.append(label)
        self.points_changed.emit(self._points, self._labels)

    def remove_point(self, index: int) -> tuple[list[float], int]:
        """Remove point at index. Returns (point, label) for undo."""
        point = self._points.pop(index)
        label = self._labels.pop(index)
        self.points_changed.emit(self._points, self._labels)
        return point, label

    def move_point(self, index: int, new_x: float, new_y: float) -> tuple[float, float]:
        """Move point. Returns old (x, y) for undo."""
        old = self._points[index].copy()
        self._points[index] = [new_x, new_y]
        self.points_changed.emit(self._points, self._labels)
        return old[0], old[1]

    def clear_points(self) -> tuple[list[list[float]], list[int]]:
        """Clear all points. Returns snapshot for undo."""
        old_points = list(self._points)
        old_labels = list(self._labels)
        self._points = []
        self._labels = []
        self.points_changed.emit(self._points, self._labels)
        return old_points, old_labels

    @property
    def device(self) -> str:
        return self._device

    def set_device(self, device: str) -> None:
        if self._device != device:
            self._device = device
            self.device_changed.emit(device)

    @property
    def model_name(self) -> str:
        return self._model_name

    def set_model_name(self, name: str) -> None:
        if self._model_name != name:
            self._model_name = name
            self.model_changed.emit(name)

    @property
    def threshold(self) -> float:
        return self._threshold

    def set_threshold(self, value: float) -> None:
        if self._threshold != value:
            self._threshold = value
            self.threshold_changed.emit(value)

    @property
    def intermediate_format(self) -> str:
        return self._intermediate_format

    def set_intermediate_format(self, fmt: str) -> None:
        if self._intermediate_format != fmt:
            self._intermediate_format = fmt
            self.format_changed.emit(fmt)

    @property
    def force_reprocess(self) -> bool:
        return self._force_reprocess

    def set_force_reprocess(self, value: bool) -> None:
        self._force_reprocess = value

    # --- Display images ---

    @property
    def current_original(self) -> Optional[np.ndarray]:
        return self._current_original

    @property
    def current_mask(self) -> Optional[np.ndarray]:
        return self._current_mask

    @property
    def current_overlay(self) -> Optional[np.ndarray]:
        return self._current_overlay

    def set_display_images(
        self,
        original: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        overlay: Optional[np.ndarray] = None,
    ) -> None:
        if original is not None:
            self._current_original = original
        if mask is not None:
            self._current_mask = mask
        if overlay is not None:
            self._current_overlay = overlay
        self.current_images_changed.emit()

    # --- Model helpers ---

    def get_model_config(self) -> tuple[str, str]:
        """Return (config_yaml_name, checkpoint_filename) for current model."""
        return self.MODEL_REGISTRY.get(
            self._model_name,
            ("sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
        )

    @property
    def total_frames(self) -> int:
        return len(self._image_files)
