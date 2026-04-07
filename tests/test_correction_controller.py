"""Unit tests for controllers.correction_controller (Phase B)."""

import sys

import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def controller(qapp):
    from controllers.correction_controller import CorrectionController
    return CorrectionController()


class SignalSpy:
    """Collect signal emissions into a list."""

    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, *args):
        self.calls.append(args)


# ----------------------------------------------------------------- init state

class TestInitialState:
    def test_anchor_is_none(self, controller):
        assert controller.anchor_frame is None

    def test_default_range_is_1_to_1(self, controller):
        assert controller.range_start == 1
        assert controller.range_end == 1

    def test_cannot_apply_when_empty(self, controller):
        assert controller.can_apply(points_count=0) is False
        assert controller.can_apply(points_count=3) is False  # no anchor yet


# ----------------------------------------------------------------- enter/exit

class TestEnterExit:
    def test_on_enter_sets_default_range_to_current_frame(self, controller):
        controller.on_enter_correction(current_frame=7, total_frames=20)
        assert controller.range_start == 7
        assert controller.range_end == 7

    def test_on_enter_emits_range_changed(self, controller):
        spy = SignalSpy()
        controller.range_changed.connect(spy)
        controller.on_enter_correction(current_frame=5, total_frames=20)
        assert (5, 5) in spy.calls

    def test_on_enter_resets_anchor(self, controller):
        # Simulate a lingering anchor from a previous session
        controller.on_enter_correction(current_frame=5, total_frames=20)
        controller.try_add_point(current_frame=5)
        assert controller.anchor_frame == 5

        controller.on_enter_correction(current_frame=10, total_frames=20)
        assert controller.anchor_frame is None

    def test_on_exit_clears_anchor(self, controller):
        controller.on_enter_correction(current_frame=5, total_frames=20)
        controller.try_add_point(current_frame=5)
        controller.on_exit_correction()
        assert controller.anchor_frame is None

    def test_on_exit_emits_anchor_changed(self, controller):
        spy = SignalSpy()
        controller.anchor_changed.connect(spy)

        controller.on_enter_correction(current_frame=5, total_frames=20)
        controller.try_add_point(current_frame=5)   # locks to 5
        spy.calls.clear()
        controller.on_exit_correction()

        # anchor went from 5 → None
        assert (None,) in spy.calls


# ----------------------------------------------------------------- try_add_point

class TestTryAddPoint:
    def test_first_point_locks_anchor_to_current_frame(self, controller):
        controller.on_enter_correction(current_frame=8, total_frames=20)
        assert controller.try_add_point(current_frame=8) is True
        assert controller.anchor_frame == 8

    def test_first_point_emits_anchor_changed(self, controller):
        spy = SignalSpy()
        controller.anchor_changed.connect(spy)
        controller.on_enter_correction(current_frame=8, total_frames=20)
        controller.try_add_point(current_frame=8)
        assert (8,) in spy.calls

    def test_subsequent_point_on_anchor_frame_accepted(self, controller):
        controller.on_enter_correction(current_frame=8, total_frames=20)
        controller.try_add_point(current_frame=8)
        assert controller.try_add_point(current_frame=8) is True
        assert controller.anchor_frame == 8

    def test_point_on_non_anchor_frame_rejected(self, controller):
        controller.on_enter_correction(current_frame=8, total_frames=20)
        controller.try_add_point(current_frame=8)
        assert controller.try_add_point(current_frame=10) is False
        # Anchor unchanged
        assert controller.anchor_frame == 8

    def test_first_point_on_different_frame_than_default_is_ok(self, controller):
        """Entering on frame 5 but the user navigated to 7 before clicking."""
        controller.on_enter_correction(current_frame=5, total_frames=20)
        assert controller.try_add_point(current_frame=7) is True
        assert controller.anchor_frame == 7


# ----------------------------------------------------------------- clear anchor

class TestClearAnchor:
    def test_clear_anchor_allows_new_anchor_on_different_frame(self, controller):
        controller.on_enter_correction(current_frame=8, total_frames=20)
        controller.try_add_point(current_frame=8)
        controller.clear_anchor()
        assert controller.anchor_frame is None
        # Now a new anchor can be picked
        assert controller.try_add_point(current_frame=12) is True
        assert controller.anchor_frame == 12

    def test_clear_anchor_emits_signal(self, controller):
        controller.on_enter_correction(current_frame=8, total_frames=20)
        controller.try_add_point(current_frame=8)
        spy = SignalSpy()
        controller.anchor_changed.connect(spy)
        controller.clear_anchor()
        assert (None,) in spy.calls


# ----------------------------------------------------------------- set_range

class TestSetRange:
    def test_set_range_clamps_to_total_frames(self, controller):
        controller.on_enter_correction(current_frame=5, total_frames=20)
        controller.set_range(start=0, end=25)
        assert controller.range_start == 1
        assert controller.range_end == 20

    def test_set_range_rejects_start_greater_than_end(self, controller):
        controller.on_enter_correction(current_frame=5, total_frames=20)
        before = (controller.range_start, controller.range_end)
        controller.set_range(start=10, end=5)
        # Unchanged
        assert (controller.range_start, controller.range_end) == before

    def test_set_range_emits_signal_on_change(self, controller):
        controller.on_enter_correction(current_frame=5, total_frames=20)
        spy = SignalSpy()
        controller.range_changed.connect(spy)
        controller.set_range(start=3, end=10)
        assert (3, 10) in spy.calls

    def test_set_range_noop_when_unchanged(self, controller):
        controller.on_enter_correction(current_frame=5, total_frames=20)
        controller.set_range(start=3, end=10)
        spy = SignalSpy()
        controller.range_changed.connect(spy)
        controller.set_range(start=3, end=10)
        assert spy.calls == []

    def test_set_range_with_locked_anchor_outside_is_rejected(self, controller):
        controller.on_enter_correction(current_frame=5, total_frames=20)
        controller.try_add_point(current_frame=5)  # anchor locked at 5
        before = (controller.range_start, controller.range_end)
        # Range [7, 10] would not contain anchor=5 → reject
        controller.set_range(start=7, end=10)
        assert (controller.range_start, controller.range_end) == before

    def test_set_range_with_locked_anchor_inside_is_accepted(self, controller):
        controller.on_enter_correction(current_frame=5, total_frames=20)
        controller.try_add_point(current_frame=5)
        controller.set_range(start=3, end=10)
        assert controller.range_start == 3
        assert controller.range_end == 10


# ----------------------------------------------------------------- can_apply

class TestCanApply:
    def test_can_apply_requires_points(self, controller):
        controller.on_enter_correction(current_frame=5, total_frames=20)
        controller.try_add_point(current_frame=5)
        assert controller.can_apply(points_count=0) is False
        assert controller.can_apply(points_count=1) is True

    def test_can_apply_requires_anchor(self, controller):
        controller.on_enter_correction(current_frame=5, total_frames=20)
        # Never called try_add_point → no anchor
        assert controller.can_apply(points_count=5) is False

    def test_can_apply_requires_anchor_inside_range(self, controller):
        # Construct a degenerate state where anchor somehow drifts outside
        # the range (shouldn't happen via normal flow, but be defensive).
        controller.on_enter_correction(current_frame=5, total_frames=20)
        controller.try_add_point(current_frame=5)
        # Force the range to bypass set_range's guard (white-box)
        controller._range_start = 10
        controller._range_end = 15
        assert controller.can_apply(points_count=3) is False


# ----------------------------------------------------------------- total frames

class TestTotalFrames:
    def test_set_total_frames_clamps_current_range(self, controller):
        controller.on_enter_correction(current_frame=5, total_frames=20)
        controller.set_range(start=15, end=18)
        controller.set_total_frames(10)
        assert controller.range_end <= 10
