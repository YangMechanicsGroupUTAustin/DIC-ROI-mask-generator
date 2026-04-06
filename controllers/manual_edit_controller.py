"""Manual mask edit controller (Step 0 of postprocessing).

Holds the in-memory mask for the currently edited frame, applies brush
and eraser strokes via `core.manual_edit`, and maintains a patch-based
undo/redo stack that is reset on every frame change.

Persistence rules
-----------------
* Edits for a frame are held entirely in memory while that frame is
  loaded. They are only written to disk when:
    - `save_frame_if_dirty()` is called (on frame change or Step 0 Done)
    - and the mask was actually modified since load
* Writes always go to the `manual_edited/` subdir — never overwrite the
  chain input directory (e.g. `masks/`).
* On load, the controller prefers an existing `manual_edited/<name>.png`
  over the source dir, so previous manual edits persist across sessions.

Undo/redo
---------
* Each completed stroke produces ONE undo entry: the pre-stroke pixels
  inside the stroke's union bbox (patch-based, much smaller than a full
  copy of the mask).
* The cap is `_UNDO_CAP` entries per frame. Loading a new frame clears
  both stacks — undo never crosses frame boundaries.
"""

import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QRect, pyqtSignal

from core.image_processing import get_image_files, imread_safe, imwrite_safe
from core.manual_edit import BBox, paint_dot, paint_stroke

logger = logging.getLogger("sam2studio.manual_edit_controller")

UndoEntry = Tuple[BBox, np.ndarray]  # (bbox, pre-state patch)


class ManualEditController(QObject):
    """Owns the working mask for Step 0 and exposes a signal-based API."""

    # Dirty bbox in image coordinates; listeners use it to re-render overlay.
    mask_modified = pyqtSignal(QRect)
    # (can_undo, can_redo) — emitted whenever either flag changes.
    undo_state_changed = pyqtSignal(bool, bool)

    _UNDO_CAP: int = 20

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)

        self._mask: Optional[np.ndarray] = None
        self._frame_idx: Optional[int] = None
        self._source_dir: Optional[str] = None
        self._edit_dir: Optional[str] = None
        self._file_name: Optional[str] = None
        self._dirty: bool = False

        self._undo: list[UndoEntry] = []
        self._redo: list[UndoEntry] = []

        # Transient per-stroke state.
        self._stroke_active: bool = False
        self._stroke_snapshot: Optional[np.ndarray] = None
        self._stroke_bbox: Optional[BBox] = None
        self._stroke_radius: int = 1
        self._stroke_value: int = 255
        self._last_stroke_point: Optional[Tuple[float, float]] = None

    # ------------------------------------------------------------------ props

    @property
    def current_mask(self) -> Optional[np.ndarray]:
        return self._mask

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def can_undo(self) -> bool:
        return len(self._undo) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo) > 0

    # -------------------------------------------------------------- load/save

    def load_frame(
        self, frame_idx: int, source_dir: str, edit_dir: str
    ) -> bool:
        """Load frame `frame_idx` from `edit_dir` (if present) or `source_dir`.

        Clears undo/redo state for the previous frame. Returns True on
        success, False if the frame could not be located or decoded.
        """
        files = get_image_files(source_dir)
        if not files or frame_idx < 0 or frame_idx >= len(files):
            logger.warning(
                "load_frame: frame_idx %d out of range for %s",
                frame_idx, source_dir,
            )
            return False

        basename = os.path.basename(files[frame_idx])
        edit_path = os.path.join(edit_dir, basename)

        if os.path.exists(edit_path):
            mask = imread_safe(edit_path, cv2.IMREAD_GRAYSCALE)
            source_desc = "manual_edited"
        else:
            mask = imread_safe(files[frame_idx], cv2.IMREAD_GRAYSCALE)
            source_desc = "source"

        if mask is None:
            logger.error("Failed to decode mask at %s", edit_path)
            return False

        self._mask = mask
        self._frame_idx = frame_idx
        self._source_dir = source_dir
        self._edit_dir = edit_dir
        self._file_name = basename
        self._dirty = False
        self._undo.clear()
        self._redo.clear()
        self._reset_stroke_state()
        self._emit_undo_state()
        logger.debug(
            "load_frame: frame=%d source=%s basename=%s",
            frame_idx, source_desc, basename,
        )
        return True

    def save_frame_if_dirty(self) -> bool:
        """Write the current frame to `manual_edited/` if modified.

        Returns True iff a write happened. Safe to call when no frame is
        loaded (returns False).
        """
        if not self._dirty or self._mask is None:
            return False
        if not self._edit_dir or not self._file_name:
            return False

        os.makedirs(self._edit_dir, exist_ok=True)
        path = os.path.join(self._edit_dir, self._file_name)
        ok = imwrite_safe(path, self._mask)
        if ok:
            self._dirty = False
        else:
            logger.error("Failed to save manual edit to %s", path)
        return ok

    # ---------------------------------------------------------------- strokes

    def begin_stroke(
        self, x: float, y: float, radius: int, is_eraser: bool
    ) -> None:
        """Start a new brush or eraser stroke at (x, y)."""
        if self._mask is None:
            return

        # Snapshot the mask once so end_stroke can derive a cheap patch
        # slice without tracking per-point history.
        self._stroke_snapshot = self._mask.copy()
        self._stroke_bbox = None
        self._stroke_radius = max(1, int(radius))
        self._stroke_value = 0 if is_eraser else 255
        self._stroke_active = True

        bbox = paint_dot(
            self._mask, x, y, self._stroke_radius, self._stroke_value
        )
        self._merge_bbox(bbox)
        self._last_stroke_point = (float(x), float(y))
        self._emit_modified(bbox)

    def continue_stroke(self, x: float, y: float) -> None:
        """Extend the active stroke to (x, y) by drawing a thick segment."""
        if not self._stroke_active or self._mask is None:
            return
        if self._last_stroke_point is None:
            return

        bbox = paint_stroke(
            self._mask,
            [self._last_stroke_point, (float(x), float(y))],
            self._stroke_radius,
            self._stroke_value,
        )
        self._merge_bbox(bbox)
        self._last_stroke_point = (float(x), float(y))
        self._emit_modified(bbox)

    def end_stroke(self) -> None:
        """Finalize the active stroke and push one undo entry."""
        if not self._stroke_active:
            return

        self._stroke_active = False
        snapshot = self._stroke_snapshot
        bbox = self._stroke_bbox
        self._stroke_snapshot = None
        self._stroke_bbox = None
        self._last_stroke_point = None

        # A stroke that modified no pixels (e.g. a dot fully outside the
        # image) produces no undo entry and no dirty flag change.
        if bbox is None or snapshot is None:
            return

        x0, y0, x1, y1 = bbox
        pre_patch = snapshot[y0:y1, x0:x1].copy()
        self._push_undo((bbox, pre_patch))
        self._redo.clear()
        self._dirty = True
        self._emit_undo_state()

    # -------------------------------------------------------------- undo/redo

    def undo(self) -> None:
        if not self._undo or self._mask is None:
            return
        bbox, stored = self._undo.pop()
        x0, y0, x1, y1 = bbox
        current_patch = self._mask[y0:y1, x0:x1].copy()
        self._mask[y0:y1, x0:x1] = stored
        self._redo.append((bbox, current_patch))
        # The mask still differs from the on-disk copy (unless this
        # undo happens to restore the exact loaded state), so keep the
        # dirty flag true to force a re-save.
        self._dirty = True
        self._emit_modified(bbox)
        self._emit_undo_state()

    def redo(self) -> None:
        if not self._redo or self._mask is None:
            return
        bbox, stored = self._redo.pop()
        x0, y0, x1, y1 = bbox
        current_patch = self._mask[y0:y1, x0:x1].copy()
        self._mask[y0:y1, x0:x1] = stored
        self._undo.append((bbox, current_patch))
        self._dirty = True
        self._emit_modified(bbox)
        self._emit_undo_state()

    # --------------------------------------------------------------- internal

    def _reset_stroke_state(self) -> None:
        self._stroke_active = False
        self._stroke_snapshot = None
        self._stroke_bbox = None
        self._last_stroke_point = None

    def _push_undo(self, entry: UndoEntry) -> None:
        self._undo.append(entry)
        if len(self._undo) > self._UNDO_CAP:
            # Drop the oldest entry so the stack stays bounded per-frame.
            self._undo.pop(0)

    def _merge_bbox(self, bbox: Optional[BBox]) -> None:
        if bbox is None:
            return
        if self._stroke_bbox is None:
            self._stroke_bbox = bbox
            return
        x0a, y0a, x1a, y1a = self._stroke_bbox
        x0b, y0b, x1b, y1b = bbox
        self._stroke_bbox = (
            min(x0a, x0b), min(y0a, y0b),
            max(x1a, x1b), max(y1a, y1b),
        )

    def _emit_modified(self, bbox: Optional[BBox]) -> None:
        if bbox is None:
            return
        x0, y0, x1, y1 = bbox
        self.mask_modified.emit(QRect(x0, y0, x1 - x0, y1 - y0))

    def _emit_undo_state(self) -> None:
        self.undo_state_changed.emit(self.can_undo, self.can_redo)
