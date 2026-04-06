"""Unit tests for MaskGenerator.refine_early_frames (Early-Frame Refinement).

These tests use a FakePredictor that captures calls to `reset_state`,
`add_new_mask`, `add_new_points_or_box`, and `propagate_in_video` so we can
assert the dispatching logic without loading a real SAM2 model.

Semantics under test:
- reset_state is called first (wipes any prior conditioning on the state).
- The anchor mask is injected via add_new_mask at `anchor_frame_idx`.
- propagate_in_video is invoked in reverse from `anchor_frame_idx` all the
  way to frame 0, even when `overwrite_count < anchor_frame_idx`.
- frame_callback is invoked only for frames with index < overwrite_count,
  i.e. the earliest K frames. Frames in (overwrite_count, anchor_frame_idx]
  are still processed internally for memory-bank context, but their masks
  are discarded.
- After the refine pass, the original conditioning points are replayed
  (so subsequent correction runs start from a clean baseline).
- Parameter validation rejects anchor < 1, overwrite_count < 1, and
  overwrite_count > anchor_frame_idx.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import torch


# ----------------------------------------------------------------- fake SAM2

@dataclass
class _AddMaskCall:
    frame_idx: int
    obj_id: int
    mask_shape: tuple


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
    add_mask_calls: list[_AddMaskCall] = field(default_factory=list)
    add_points_calls: list[_AddPointsCall] = field(default_factory=list)
    propagate_calls: list[_PropagateCall] = field(default_factory=list)
    reset_state_calls: int = 0

    def reset_state(self, inference_state):
        self.reset_state_calls += 1

    def add_new_mask(
        self,
        inference_state,
        frame_idx: int,
        obj_id: int,
        mask,
    ):
        # Accept tensor, numpy, or anything with a shape attribute
        if hasattr(mask, "shape"):
            shape = tuple(mask.shape)
        else:
            shape = ()
        self.add_mask_calls.append(
            _AddMaskCall(
                frame_idx=int(frame_idx),
                obj_id=int(obj_id),
                mask_shape=shape,
            )
        )
        dummy_logits = torch.zeros((1, 1, 10, 10))
        return frame_idx, [obj_id], dummy_logits

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
                frame_idx=int(frame_idx),
                obj_id=int(obj_id),
                points=np.asarray(points).copy(),
                labels=np.asarray(labels).copy(),
            )
        )
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
            mask = torch.ones((1, 1, 10, 10))
            yield fi, [1], mask


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
    mg._inference_state = object()
    return mg


@pytest.fixture
def anchor_mask():
    """A small binary mask used as the refine anchor."""
    m = np.zeros((32, 32), dtype=np.uint8)
    m[10:20, 10:20] = 1
    return m


# ================================================================== Phase A

class TestRefineEarlyFrames:
    """Tests for `MaskGenerator.refine_early_frames`."""

    # ------------------------------------------------------ reset_state first
    def test_reset_state_called_first(
        self, mask_gen, fake_predictor, anchor_mask,
    ):
        mask_gen.refine_early_frames(
            anchor_frame_idx=10,
            anchor_mask=anchor_mask,
            overwrite_count=3,
        )
        assert fake_predictor.reset_state_calls >= 1

    # ----------------------------------------------------- add_new_mask call
    def test_add_new_mask_called_with_anchor_and_obj_id(
        self, mask_gen, fake_predictor, anchor_mask,
    ):
        mask_gen.refine_early_frames(
            anchor_frame_idx=10,
            anchor_mask=anchor_mask,
            overwrite_count=3,
        )
        assert len(fake_predictor.add_mask_calls) == 1
        call = fake_predictor.add_mask_calls[0]
        assert call.frame_idx == 10
        assert call.obj_id == 1

    # ------------------------------------ propagate reverse from anchor to 0
    def test_propagate_in_video_called_reverse_from_anchor(
        self, mask_gen, fake_predictor, anchor_mask,
    ):
        mask_gen.refine_early_frames(
            anchor_frame_idx=10,
            anchor_mask=anchor_mask,
            overwrite_count=3,
        )
        # Expect exactly one reverse-propagation call that walks anchor -> 0
        refine_calls = [
            c for c in fake_predictor.propagate_calls
            if c.reverse and c.start_frame_idx == 10
        ]
        assert len(refine_calls) == 1
        assert refine_calls[0].max_frame_num_to_track == 10

    # -------------------- only earliest K frames are written to frame_callback
    def test_frame_callback_only_called_for_frames_lt_overwrite_count(
        self, mask_gen, fake_predictor, anchor_mask,
    ):
        written: list[int] = []
        mask_gen.refine_early_frames(
            anchor_frame_idx=10,
            anchor_mask=anchor_mask,
            overwrite_count=3,
            frame_callback=lambda fi, m: written.append(int(fi)),
        )
        # Earliest 3 frames: 0, 1, 2
        assert sorted(written) == [0, 1, 2]

    # ------------------------------------------- anchor frame is NOT saved
    def test_anchor_frame_itself_not_saved(
        self, mask_gen, fake_predictor, anchor_mask,
    ):
        written: list[int] = []
        mask_gen.refine_early_frames(
            anchor_frame_idx=10,
            anchor_mask=anchor_mask,
            overwrite_count=10,  # would include 0..9 but NOT 10
            frame_callback=lambda fi, m: written.append(int(fi)),
        )
        assert 10 not in written
        assert sorted(written) == list(range(10))

    # ------------------------------------------ stop_check short-circuits
    def test_stop_check_short_circuits_loop(
        self, mask_gen, fake_predictor, anchor_mask,
    ):
        calls = {"n": 0}

        def stop_after_two():
            calls["n"] += 1
            return calls["n"] > 2

        written: list[int] = []
        mask_gen.refine_early_frames(
            anchor_frame_idx=10,
            anchor_mask=anchor_mask,
            overwrite_count=10,
            frame_callback=lambda fi, m: written.append(int(fi)),
            stop_check=stop_after_two,
        )
        # Should bail out well before completing all 10 frames
        assert len(written) < 10

    # ----------------------------------------- progress_callback reports count
    def test_progress_callback_reports_written_count(
        self, mask_gen, fake_predictor, anchor_mask,
    ):
        progress_log: list[tuple[int, int]] = []
        mask_gen.refine_early_frames(
            anchor_frame_idx=10,
            anchor_mask=anchor_mask,
            overwrite_count=3,
            progress_callback=lambda done, total: progress_log.append(
                (int(done), int(total))
            ),
        )
        assert len(progress_log) == 3
        # Monotonically increasing done counts, total = 3 for all
        assert [p[0] for p in progress_log] == [1, 2, 3]
        assert all(p[1] == 3 for p in progress_log)

    # -------------------------------------------- threshold applied to logits
    def test_threshold_applied_to_logits(
        self, mask_gen, fake_predictor, anchor_mask,
    ):
        # FakePredictor yields torch.ones(1,1,10,10); with threshold=2.0 the
        # binary mask should be all-zero (1 > 2 is False everywhere).
        captured: list[np.ndarray] = []

        def cb(fi, m):
            captured.append(m.copy())

        mask_gen.refine_early_frames(
            anchor_frame_idx=10,
            anchor_mask=anchor_mask,
            overwrite_count=3,
            threshold=2.0,
            frame_callback=cb,
        )
        assert len(captured) == 3
        for m in captured:
            assert (m == 0).all()

        # And with threshold=0.0 (default behavior), masks are non-empty.
        mask_gen2 = mask_gen
        mask_gen2._predictor = FakePredictor(total_frames=20)
        mask_gen2._inference_state = object()
        captured2: list[np.ndarray] = []

        mask_gen2.refine_early_frames(
            anchor_frame_idx=10,
            anchor_mask=anchor_mask,
            overwrite_count=3,
            threshold=0.0,
            frame_callback=lambda fi, m: captured2.append(m.copy()),
        )
        assert len(captured2) == 3
        for m in captured2:
            assert m.any()

    # ----------------------------------------------- validation errors
    def test_raises_when_overwrite_count_exceeds_anchor(
        self, mask_gen, anchor_mask,
    ):
        with pytest.raises(ValueError):
            mask_gen.refine_early_frames(
                anchor_frame_idx=5,
                anchor_mask=anchor_mask,
                overwrite_count=6,  # > anchor
            )

    def test_raises_when_overwrite_count_equals_anchor_plus_one(
        self, mask_gen, anchor_mask,
    ):
        # Edge: K == anchor is allowed (overwrite [0..anchor-1]),
        # K == anchor + 1 is not.
        with pytest.raises(ValueError):
            mask_gen.refine_early_frames(
                anchor_frame_idx=5,
                anchor_mask=anchor_mask,
                overwrite_count=6,
            )

    def test_overwrite_count_equal_to_anchor_is_allowed(
        self, mask_gen, fake_predictor, anchor_mask,
    ):
        # K == anchor means overwrite frames [0, anchor-1] — all frames
        # strictly before the anchor.
        written: list[int] = []
        mask_gen.refine_early_frames(
            anchor_frame_idx=5,
            anchor_mask=anchor_mask,
            overwrite_count=5,
            frame_callback=lambda fi, m: written.append(int(fi)),
        )
        assert sorted(written) == [0, 1, 2, 3, 4]

    def test_raises_when_overwrite_count_is_zero(
        self, mask_gen, anchor_mask,
    ):
        with pytest.raises(ValueError):
            mask_gen.refine_early_frames(
                anchor_frame_idx=10,
                anchor_mask=anchor_mask,
                overwrite_count=0,
            )

    def test_raises_when_anchor_lt_1(self, mask_gen, anchor_mask):
        # Anchor 0 would mean "no earlier frames exist", refine is a no-op.
        with pytest.raises(ValueError):
            mask_gen.refine_early_frames(
                anchor_frame_idx=0,
                anchor_mask=anchor_mask,
                overwrite_count=1,
            )

    # ----------------------- reset_corrections replays originals after refine
    def test_reset_corrections_replayed_after_refine(
        self, mask_gen, fake_predictor, anchor_mask,
    ):
        pts = np.array([[10.0, 20.0]], dtype=np.float32)
        lbl = np.array([1], dtype=np.int32)
        mask_gen.add_original_points(frame_idx=0, points=pts, labels=lbl)

        # Forward propagation already happened; the original point call count
        # is 1. After refine, the original point must have been replayed so
        # that the inference state is clean for any follow-up Re-run Range.
        baseline_adds = len(fake_predictor.add_points_calls)
        assert baseline_adds == 1

        mask_gen.refine_early_frames(
            anchor_frame_idx=10,
            anchor_mask=anchor_mask,
            overwrite_count=3,
        )

        # The original entry should appear among add_points_calls a second
        # time, indicating reset_corrections was invoked after refine.
        assert len(fake_predictor.add_points_calls) == baseline_adds + 1
        replay = fake_predictor.add_points_calls[-1]
        assert replay.frame_idx == 0
        np.testing.assert_array_equal(replay.points, pts)
