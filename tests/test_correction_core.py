"""Unit tests for MaskGenerator correction-range logic (Phase A).

These tests use a `FakePredictor` that captures calls to
`add_new_points_or_box`, `propagate_in_video`, and `reset_state` so we can
assert the dispatching logic without loading a real SAM2 model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import torch


# ----------------------------------------------------------------- fake SAM2

@dataclass
class _AddPointsCall:
    frame_idx: int
    obj_id: int
    points: np.ndarray
    labels: np.ndarray


@dataclass
class _PropagateCall:
    start_frame_idx: int | None
    max_frame_num_to_track: int | None
    reverse: bool


@dataclass
class FakePredictor:
    """Minimal stand-in for SAM2VideoPredictor used by MaskGenerator."""

    total_frames: int = 20
    add_points_calls: list[_AddPointsCall] = field(default_factory=list)
    propagate_calls: list[_PropagateCall] = field(default_factory=list)
    reset_state_calls: int = 0

    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx: int,
        obj_id: int,
        points: np.ndarray,
        labels: np.ndarray,
    ):
        self.add_points_calls.append(
            _AddPointsCall(
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=np.asarray(points).copy(),
                labels=np.asarray(labels).copy(),
            )
        )
        # Mimic SAM2 return: (frame_idx, obj_ids, mask_logits)
        dummy_logits = torch.zeros((1, 1, 10, 10))
        return frame_idx, [obj_id], dummy_logits

    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ):
        self.propagate_calls.append(
            _PropagateCall(
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
                reverse=reverse,
            )
        )

        if start_frame_idx is None:
            start_frame_idx = 0
        num = self.total_frames
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num

        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            order = list(range(start_frame_idx, end_frame_idx - 1, -1))
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num - 1
            )
            order = list(range(start_frame_idx, end_frame_idx + 1))

        for fi in order:
            # Generate a small tensor that thresholds to a single "on" pixel
            # so the caller's `(logits > 0).cpu().numpy()` produces something.
            mask = torch.ones((1, 1, 10, 10))
            yield fi, [1], mask

    def reset_state(self, inference_state):
        self.reset_state_calls += 1


# ------------------------------------------------------------------ fixtures

@pytest.fixture
def fake_predictor():
    return FakePredictor(total_frames=20)


@pytest.fixture
def mask_gen(fake_predictor):
    """MaskGenerator with its predictor/state pre-populated by a fake."""
    from core.mask_generator import MaskGenerator

    mg = MaskGenerator()
    mg._predictor = fake_predictor
    mg._inference_state = object()  # opaque placeholder — fake ignores it
    return mg


# ------------------------------------------------- original-points tracking

class TestOriginalPointsTracking:
    def test_add_original_points_records_and_forwards(self, mask_gen, fake_predictor):
        pts = np.array([[10.0, 20.0]], dtype=np.float32)
        lbl = np.array([1], dtype=np.int32)

        mask_gen.add_original_points(frame_idx=0, points=pts, labels=lbl)

        # Forwarded to SAM2
        assert len(fake_predictor.add_points_calls) == 1
        assert fake_predictor.add_points_calls[0].frame_idx == 0

        # Tracked internally
        assert hasattr(mask_gen, "_original_conditioning")
        assert len(mask_gen._original_conditioning) == 1
        entry = mask_gen._original_conditioning[0]
        assert entry[0] == 0             # frame_idx
        assert entry[3] == 1             # obj_id (default)
        np.testing.assert_array_equal(entry[1], pts)
        np.testing.assert_array_equal(entry[2], lbl)

    def test_add_correction_does_not_touch_originals(self, mask_gen, fake_predictor):
        pts_orig = np.array([[10.0, 20.0]], dtype=np.float32)
        pts_corr = np.array([[30.0, 40.0]], dtype=np.float32)
        lbl = np.array([1], dtype=np.int32)

        mask_gen.add_original_points(frame_idx=0, points=pts_orig, labels=lbl)
        mask_gen.add_correction(frame_idx=5, points=pts_corr, labels=lbl)

        # Both forwarded to SAM2
        assert len(fake_predictor.add_points_calls) == 2
        # Only one entry in originals
        assert len(mask_gen._original_conditioning) == 1
        np.testing.assert_array_equal(
            mask_gen._original_conditioning[0][1], pts_orig
        )

    def test_multiple_originals_across_frames(self, mask_gen):
        pts_a = np.array([[10.0, 10.0]], dtype=np.float32)
        pts_b = np.array([[20.0, 20.0]], dtype=np.float32)
        lbl = np.array([1], dtype=np.int32)

        mask_gen.add_original_points(frame_idx=0, points=pts_a, labels=lbl)
        mask_gen.add_original_points(frame_idx=3, points=pts_b, labels=lbl)

        assert [e[0] for e in mask_gen._original_conditioning] == [0, 3]


# -------------------------------------------------------------- reset_corrections

class TestResetCorrections:
    def test_reset_calls_reset_state_then_replays_originals(
        self, mask_gen, fake_predictor
    ):
        pts = np.array([[10.0, 20.0]], dtype=np.float32)
        lbl = np.array([1], dtype=np.int32)
        mask_gen.add_original_points(frame_idx=0, points=pts, labels=lbl)

        # Simulate a correction that dirtied the state
        corr_pts = np.array([[50.0, 60.0]], dtype=np.float32)
        mask_gen.add_correction(frame_idx=5, points=corr_pts, labels=lbl)

        baseline = len(fake_predictor.add_points_calls)  # 2
        assert baseline == 2

        mask_gen.reset_corrections()

        # reset_state was called
        assert fake_predictor.reset_state_calls == 1

        # Originals were replayed: one more add_points_calls entry,
        # matching the original frame_idx/points (NOT the correction).
        assert len(fake_predictor.add_points_calls) == baseline + 1
        replay = fake_predictor.add_points_calls[-1]
        assert replay.frame_idx == 0
        np.testing.assert_array_equal(replay.points, pts)

    def test_reset_noop_when_no_originals(self, mask_gen, fake_predictor):
        # No originals tracked — still safe to call
        mask_gen.reset_corrections()
        # reset_state should still fire (it cheaply clears any corrections)
        assert fake_predictor.reset_state_calls == 1

    def test_reset_preserves_originals_list_across_calls(
        self, mask_gen, fake_predictor
    ):
        pts = np.array([[10.0, 20.0]], dtype=np.float32)
        lbl = np.array([1], dtype=np.int32)
        mask_gen.add_original_points(frame_idx=0, points=pts, labels=lbl)

        mask_gen.reset_corrections()
        mask_gen.reset_corrections()

        assert len(mask_gen._original_conditioning) == 1
        assert fake_predictor.reset_state_calls == 2


# ---------------------------------------------------------------- propagate_range

class TestPropagateRange:
    def _capture(self, mask_gen):
        """Run propagate_range and record every (frame_idx) that was written."""
        written: list[int] = []
        mask_gen.propagate_range(
            anchor_frame_idx=self._anchor,
            range_start=self._start,
            range_end=self._end,
            threshold=0.0,
            frame_callback=lambda fi, mask: written.append(fi),
            stop_check=None,
        )
        return written

    # --- Single-frame case --------------------------------------------------

    def test_single_frame_range(self, mask_gen, fake_predictor):
        self._anchor, self._start, self._end = 5, 5, 5
        written = self._capture(mask_gen)

        assert written == [5]
        # Only a single forward propagate call (reverse would add a 2nd)
        assert len(fake_predictor.propagate_calls) == 1
        call = fake_predictor.propagate_calls[0]
        assert call.reverse is False
        assert call.start_frame_idx == 5
        assert call.max_frame_num_to_track == 0

    # --- Anchor at range start → forward only -------------------------------

    def test_forward_only_when_anchor_at_start(self, mask_gen, fake_predictor):
        self._anchor, self._start, self._end = 10, 10, 15
        written = self._capture(mask_gen)

        assert written == [10, 11, 12, 13, 14, 15]
        assert len(fake_predictor.propagate_calls) == 1
        assert fake_predictor.propagate_calls[0].reverse is False
        assert fake_predictor.propagate_calls[0].max_frame_num_to_track == 5

    # --- Anchor at range end → reverse only ---------------------------------

    def test_reverse_only_when_anchor_at_end(self, mask_gen, fake_predictor):
        self._anchor, self._start, self._end = 10, 5, 10
        written = self._capture(mask_gen)

        # Anchor (10) yielded once by the forward-leg singleton + reverse
        # leg writes 9,8,7,6,5
        assert sorted(written) == [5, 6, 7, 8, 9, 10]
        # Anchor is written exactly once even though SAM2 yields it in both
        assert written.count(10) == 1

        # Both legs should have been invoked (fwd with max=0, rev with max=5)
        assert len(fake_predictor.propagate_calls) == 2
        fwd = [c for c in fake_predictor.propagate_calls if not c.reverse]
        rev = [c for c in fake_predictor.propagate_calls if c.reverse]
        assert len(fwd) == 1 and len(rev) == 1
        assert fwd[0].max_frame_num_to_track == 0
        assert rev[0].max_frame_num_to_track == 5

    # --- Anchor in middle → bidirectional -----------------------------------

    def test_bidirectional_writes_anchor_once(self, mask_gen, fake_predictor):
        self._anchor, self._start, self._end = 10, 8, 13
        written = self._capture(mask_gen)

        assert sorted(written) == [8, 9, 10, 11, 12, 13]
        assert written.count(10) == 1
        assert len(fake_predictor.propagate_calls) == 2

    def test_bidirectional_wide_range(self, mask_gen, fake_predictor):
        self._anchor, self._start, self._end = 10, 0, 19
        written = self._capture(mask_gen)

        assert sorted(written) == list(range(20))
        assert written.count(10) == 1

    # --- Validation ---------------------------------------------------------

    def test_anchor_before_range_raises(self, mask_gen):
        with pytest.raises(ValueError):
            mask_gen.propagate_range(
                anchor_frame_idx=2,
                range_start=5,
                range_end=10,
            )

    def test_anchor_after_range_raises(self, mask_gen):
        with pytest.raises(ValueError):
            mask_gen.propagate_range(
                anchor_frame_idx=15,
                range_start=5,
                range_end=10,
            )

    def test_empty_range_raises(self, mask_gen):
        with pytest.raises(ValueError):
            mask_gen.propagate_range(
                anchor_frame_idx=5,
                range_start=10,
                range_end=5,  # end < start
            )

    # --- stop_check short-circuits ------------------------------------------

    def test_stop_check_halts_propagation(self, mask_gen, fake_predictor):
        calls = {"n": 0}

        def stop_after_two():
            calls["n"] += 1
            return calls["n"] > 2

        written: list[int] = []
        mask_gen.propagate_range(
            anchor_frame_idx=0,
            range_start=0,
            range_end=10,
            threshold=0.0,
            frame_callback=lambda fi, mask: written.append(fi),
            stop_check=stop_after_two,
        )
        # Only a handful of frames make it through before the stop triggers
        assert len(written) <= 3
