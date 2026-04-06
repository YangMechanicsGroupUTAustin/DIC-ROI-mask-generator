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

from core.preprocessing import PreprocessingConfig


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
    preprocessing_changed = pyqtSignal()           # preprocessing config updated
    marked_frames_changed = pyqtSignal(set)         # set of 1-based frame indices
    vram_updated = pyqtSignal(float, float)        # used_gb, total_gb
    status_message = pyqtSignal(str, str)          # message, level (ready/processing/error/warning)
    # --- Early-Frame Refinement (Phase C) ---
    refine_enabled_changed = pyqtSignal(bool)
    refine_anchor_changed = pyqtSignal(int)        # 1-based anchor frame
    refine_overwrite_changed = pyqtSignal(int)     # K: count of earliest frames

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
        self._mask_output_format = "TIFF (default)"
        self._force_reprocess = False

        # Preprocessing
        self._preprocessing_config = PreprocessingConfig()

        # Marked/bookmarked frames (1-based indices)
        self._marked_frames: set[int] = set()

        # Overlay display settings
        self._overlay_alpha = 0.4
        self._overlay_color = (255, 0, 0)  # RGB red

        # Display images (numpy arrays, RGB uint8)
        self._current_original: Optional[np.ndarray] = None
        self._current_mask: Optional[np.ndarray] = None
        self._current_overlay: Optional[np.ndarray] = None

        # SAM2 runtime (managed by processing controller)
        self.predictor = None
        self.inference_state = None

        # Early-Frame Refinement settings (Phase C) -- opt-in post-stage
        # that reverse-propagates from a user-chosen anchor to overwrite
        # the K worst early frames.
        self._refine_enabled = False
        self._refine_anchor_frame = 1        # 1-based; recomputed on load
        self._refine_overwrite_count = 1     # K; recomputed on load

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
            # Validate and auto-discover images
            if not os.path.isdir(path):
                self.status_message.emit(
                    f"Directory does not exist: {path}", "error",
                )
                return
            if not os.access(path, os.R_OK):
                self.status_message.emit(
                    f"Directory is not readable: {path}", "error",
                )
                return
            from core.image_processing import get_image_files
            files = get_image_files(path)
            if not files:
                self.status_message.emit(
                    "No supported images found in directory", "warning",
                )
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
        # Recompute refine defaults for the new sequence length. Does NOT
        # touch refine_enabled -- that's a user preference, orthogonal to
        # anchor/K.
        self._reset_refine_defaults()
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
    def mask_output_format(self) -> str:
        return self._mask_output_format

    def set_mask_output_format(self, fmt: str) -> None:
        self._mask_output_format = fmt

    @property
    def force_reprocess(self) -> bool:
        return self._force_reprocess

    def set_force_reprocess(self, value: bool) -> None:
        self._force_reprocess = value

    @property
    def preprocessing_config(self) -> PreprocessingConfig:
        return self._preprocessing_config

    def set_preprocessing_config(self, config: PreprocessingConfig) -> None:
        self._preprocessing_config = config
        self.preprocessing_changed.emit()

    # --- Marked frames ---

    @property
    def marked_frames(self) -> set[int]:
        return self._marked_frames

    def toggle_marked_frame(self, frame: int) -> None:
        """Toggle bookmark on a frame. Returns new state."""
        if frame in self._marked_frames:
            self._marked_frames.discard(frame)
        else:
            self._marked_frames.add(frame)
        self.marked_frames_changed.emit(set(self._marked_frames))

    def clear_marked_frames(self) -> None:
        """Remove all bookmarks."""
        self._marked_frames.clear()
        self.marked_frames_changed.emit(set(self._marked_frames))

    def next_marked_frame(self, current: int) -> int | None:
        """Return the next marked frame after current, or None."""
        above = sorted(f for f in self._marked_frames if f > current)
        return above[0] if above else None

    def prev_marked_frame(self, current: int) -> int | None:
        """Return the previous marked frame before current, or None."""
        below = sorted((f for f in self._marked_frames if f < current), reverse=True)
        return below[0] if below else None

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

    def clear_display_images(self) -> None:
        """Clear all cached display images (original, mask, overlay)."""
        self._current_original = None
        self._current_mask = None
        self._current_overlay = None

    # --- Model helpers ---

    def get_model_config(self) -> tuple[str, str]:
        """Return (config_yaml_name, checkpoint_filename) for current model."""
        return self.MODEL_REGISTRY.get(
            self._model_name,
            ("sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
        )

    @property
    def overlay_alpha(self) -> float:
        return self._overlay_alpha

    def set_overlay_alpha(self, alpha: float) -> None:
        self._overlay_alpha = max(0.0, min(1.0, alpha))

    @property
    def overlay_color(self) -> tuple[int, int, int]:
        return self._overlay_color

    def set_overlay_color(self, color: tuple[int, int, int]) -> None:
        self._overlay_color = color

    @property
    def total_frames(self) -> int:
        return len(self._image_files)

    # --- Early-Frame Refinement (Phase C) ---------------------------------

    @property
    def refine_enabled(self) -> bool:
        return self._refine_enabled

    @property
    def refine_anchor_frame(self) -> int:
        """1-based anchor frame from which reverse propagation starts."""
        return self._refine_anchor_frame

    @property
    def refine_overwrite_count(self) -> int:
        """K: number of earliest frames whose masks will be overwritten."""
        return self._refine_overwrite_count

    def set_refine_enabled(self, value: bool) -> None:
        value = bool(value)
        if self._refine_enabled != value:
            self._refine_enabled = value
            self.refine_enabled_changed.emit(value)

    def set_refine_anchor_frame(self, value: int) -> None:
        """Set the anchor frame (1-based) with clamping.

        Lower bound is 2 when there are at least 2 images, so that at least
        one earlier frame exists to overwrite. Upper bound is total_frames.
        Cascades: if the new anchor is below the current overwrite count K,
        K is dragged down to fit (K <= anchor is required by the core).
        """
        total = len(self._image_files)
        if total >= 2:
            lo, hi = 2, total
        elif total == 1:
            lo, hi = 1, 1
        else:
            # No images loaded yet -- refuse to budge the anchor beyond
            # the initial default (1).
            lo, hi = 1, 1
        value = max(lo, min(int(value), hi))
        if self._refine_anchor_frame != value:
            self._refine_anchor_frame = value
            self.refine_anchor_changed.emit(value)
        # Cascade K to satisfy K <= anchor.
        if self._refine_overwrite_count > value:
            new_k = max(1, value)
            if new_k != self._refine_overwrite_count:
                self._refine_overwrite_count = new_k
                self.refine_overwrite_changed.emit(new_k)

    def set_refine_overwrite_count(self, value: int) -> None:
        """Set the overwrite count K with clamping to [1, anchor]."""
        anchor = self._refine_anchor_frame
        lo, hi = 1, max(1, anchor)
        value = max(lo, min(int(value), hi))
        if self._refine_overwrite_count != value:
            self._refine_overwrite_count = value
            self.refine_overwrite_changed.emit(value)

    def _reset_refine_defaults(self) -> None:
        """Recompute default anchor/K for the current sequence length.

        Policy:
            anchor = min(10, total_frames)   (or 1 if no images)
            K      = max(1, min(3, anchor-1))
                     -- prefer 3, but default keeps K strictly below anchor
                     -- so the anchor frame's own mask stays intact by default.

        Leaves refine_enabled untouched -- that's a user preference.
        """
        total = len(self._image_files)
        if total <= 0:
            new_anchor = 1
            new_k = 1
        else:
            new_anchor = min(10, total)
            # For anchor=1 (total=1), (anchor-1)=0 -> max(1,0)=1.
            new_k = max(1, min(3, new_anchor - 1))

        if self._refine_anchor_frame != new_anchor:
            self._refine_anchor_frame = new_anchor
            self.refine_anchor_changed.emit(new_anchor)
        if self._refine_overwrite_count != new_k:
            self._refine_overwrite_count = new_k
            self.refine_overwrite_changed.emit(new_k)
