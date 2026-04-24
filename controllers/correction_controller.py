"""State machine for the range-based mask correction workflow.

Owns the anchor-frame / range / point-gating logic that main_window used
to handle inline. Pure controller — no numpy, no SAM2, no file I/O.

Concepts
--------
* **anchor frame**: the frame where correction points live. Locked when
  the user drops the first point, cleared on `on_enter_correction` /
  `on_exit_correction` / `clear_anchor`.
* **range** `[range_start, range_end]` (1-based, inclusive): the frames
  that will be re-propagated over on Apply. Must contain the anchor.
* **total frames**: clamp for `set_range`. Tracked separately so the
  controller never returns a range the UI can't display.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal


class CorrectionController(QObject):
    """Non-GUI state machine for the Add-Correction / Re-run-Range flow."""

    # Emits Optional[int] — None when cleared, int (1-based) when locked.
    anchor_changed = pyqtSignal(object)
    # Emits (range_start, range_end), 1-based inclusive.
    range_changed = pyqtSignal(int, int)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._anchor: Optional[int] = None
        self._range_start: int = 1
        self._range_end: int = 1
        self._total_frames: int = 1

    # --------------------------------------------------------------- props

    @property
    def anchor_frame(self) -> Optional[int]:
        return self._anchor

    @property
    def range_start(self) -> int:
        return self._range_start

    @property
    def range_end(self) -> int:
        return self._range_end

    @property
    def total_frames(self) -> int:
        return self._total_frames

    # ---------------------------------------------------------- lifecycle

    def on_enter_correction(
        self, current_frame: int, total_frames: int
    ) -> None:
        """Seed the default range to `[current, current]` and clear anchor.

        `current_frame` / `total_frames` are 1-based, matching the rest of
        the app (current_frame state, sidebar, etc.).
        """
        self._total_frames = max(1, int(total_frames))
        self._set_anchor(None)
        self._set_range(int(current_frame), int(current_frame))

    def on_exit_correction(self) -> None:
        """Clear anchor. Range is left as-is (next enter will reseed)."""
        self._set_anchor(None)

    # ----------------------------------------------------------- anchor

    def try_add_point(self, current_frame: int) -> bool:
        """Request to add a point on `current_frame`. Returns True on accept.

        Rules:
        * If no anchor: lock anchor to `current_frame` and accept.
        * If anchor == current_frame: accept (another point on the same anchor).
        * Otherwise: reject (anchor is locked, points can't land elsewhere).
        """
        if self._anchor is None:
            self._set_anchor(int(current_frame))
            return True
        return int(current_frame) == self._anchor

    def clear_anchor(self) -> None:
        """Unlock the anchor, allowing the user to re-pick on another frame."""
        self._set_anchor(None)

    # ------------------------------------------------------------- range

    def set_range(self, start: int, end: int) -> None:
        """Set `[start, end]`, clamped to `[1, total_frames]`.

        Rejected (no-op) when:
        * `start > end`
        * anchor is locked and would fall outside the new range
        """
        start = max(1, int(start))
        end = min(int(end), self._total_frames)
        if start > end:
            return
        if self._anchor is not None and not (start <= self._anchor <= end):
            return
        self._set_range(start, end)

    def set_total_frames(self, total: int) -> None:
        """Update the clamp bound; adjust current range to stay in-bounds."""
        self._total_frames = max(1, int(total))
        new_start = min(self._range_start, self._total_frames)
        new_end = min(self._range_end, self._total_frames)
        if new_start > new_end:
            new_start = new_end
        if (new_start, new_end) != (self._range_start, self._range_end):
            self._set_range(new_start, new_end)

    # --------------------------------------------------------- gating

    def can_apply(self, points_count: int) -> bool:
        """True iff a real correction run is currently launch-able."""
        if points_count <= 0:
            return False
        if self._anchor is None:
            return False
        if not (self._range_start <= self._anchor <= self._range_end):
            return False
        return True

    # ---------------------------------------------------- internal setters

    def _set_anchor(self, value: Optional[int]) -> None:
        if self._anchor == value:
            return
        self._anchor = value
        self.anchor_changed.emit(value)

    def _set_range(self, start: int, end: int) -> None:
        if (start, end) == (self._range_start, self._range_end):
            return
        self._range_start = start
        self._range_end = end
        self.range_changed.emit(start, end)
