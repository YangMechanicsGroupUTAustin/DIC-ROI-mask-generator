"""Unit tests for controllers.manual_edit_controller (Phase B)."""

import os
import sys
import tempfile

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def tmp_dirs():
    """Create source-mask + manual-edited tmp dirs seeded with 3 frames."""
    with tempfile.TemporaryDirectory() as base:
        src = os.path.join(base, "masks")
        edit = os.path.join(base, "manual_edited")
        os.makedirs(src, exist_ok=True)

        # Seed 3 identifiable frames in the source dir.
        for i in range(3):
            frame = np.full((50, 50), (i + 1) * 50, dtype=np.uint8)
            cv2.imwrite(os.path.join(src, f"frame_{i:03d}.png"), frame)

        yield src, edit


@pytest.fixture
def controller(qapp):
    from controllers.manual_edit_controller import ManualEditController
    return ManualEditController()


class TestLoadFrame:
    def test_load_frame_seeds_from_masks_if_no_manual_edit(
        self, controller, tmp_dirs
    ):
        src, edit = tmp_dirs
        controller.load_frame(0, src, edit)

        assert controller.current_mask is not None
        # Seeded value for frame 0 was (0+1)*50 = 50
        assert int(controller.current_mask[10, 10]) == 50

    def test_load_frame_prefers_manual_edited_if_exists(
        self, controller, tmp_dirs
    ):
        src, edit = tmp_dirs
        os.makedirs(edit, exist_ok=True)
        distinctive = np.full((50, 50), 200, dtype=np.uint8)
        cv2.imwrite(os.path.join(edit, "frame_000.png"), distinctive)

        controller.load_frame(0, src, edit)
        assert int(controller.current_mask[10, 10]) == 200

    def test_load_new_frame_clears_undo_stack(self, controller, tmp_dirs):
        src, edit = tmp_dirs
        controller.load_frame(0, src, edit)
        controller.begin_stroke(10, 10, radius=3, is_eraser=False)
        controller.end_stroke()
        assert controller.can_undo is True

        # Navigating to a different frame wipes per-frame history.
        controller.load_frame(1, src, edit)
        assert controller.can_undo is False
        assert controller.can_redo is False


class TestStrokeApplication:
    def test_apply_stroke_marks_dirty_and_modifies_mask(
        self, controller, tmp_dirs
    ):
        src, edit = tmp_dirs
        controller.load_frame(0, src, edit)
        before = controller.current_mask.copy()

        # Brush: erase into the background-filled frame by writing 255.
        controller.begin_stroke(25, 25, radius=5, is_eraser=False)
        controller.continue_stroke(30, 30)
        controller.end_stroke()

        assert controller.is_dirty is True
        assert not np.array_equal(controller.current_mask, before)
        # Point inside the stroke should be 255 (brush writes foreground)
        assert int(controller.current_mask[25, 25]) == 255

    def test_eraser_clears_mask_region(self, controller, tmp_dirs):
        src, edit = tmp_dirs
        # Seed frame 2 (pixel value 150) and erase a hole in it.
        controller.load_frame(2, src, edit)
        assert int(controller.current_mask[25, 25]) == 150

        controller.begin_stroke(25, 25, radius=4, is_eraser=True)
        controller.end_stroke()

        assert int(controller.current_mask[25, 25]) == 0


class TestUndoRedo:
    def test_undo_restores_previous_state(self, controller, tmp_dirs):
        src, edit = tmp_dirs
        controller.load_frame(0, src, edit)
        original = controller.current_mask.copy()

        controller.begin_stroke(25, 25, radius=4, is_eraser=False)
        controller.end_stroke()
        assert not np.array_equal(controller.current_mask, original)

        controller.undo()
        assert np.array_equal(controller.current_mask, original)
        assert controller.can_undo is False
        assert controller.can_redo is True

    def test_undo_twice_then_redo_once(self, controller, tmp_dirs):
        src, edit = tmp_dirs
        controller.load_frame(0, src, edit)
        original = controller.current_mask.copy()

        # Stroke 1
        controller.begin_stroke(10, 10, radius=3, is_eraser=False)
        controller.end_stroke()
        after1 = controller.current_mask.copy()

        # Stroke 2
        controller.begin_stroke(40, 40, radius=3, is_eraser=False)
        controller.end_stroke()

        controller.undo()  # back to after1
        assert np.array_equal(controller.current_mask, after1)

        controller.undo()  # back to original
        assert np.array_equal(controller.current_mask, original)

        controller.redo()  # forward to after1
        assert np.array_equal(controller.current_mask, after1)

    def test_undo_stack_capped_at_20(self, controller, tmp_dirs):
        src, edit = tmp_dirs
        controller.load_frame(0, src, edit)

        # Apply 25 distinct strokes.
        for i in range(25):
            controller.begin_stroke(i + 1, i + 1, radius=2, is_eraser=False)
            controller.end_stroke()

        # Only the last 20 strokes should be undoable.
        undo_count = 0
        while controller.can_undo:
            controller.undo()
            undo_count += 1
            if undo_count > 25:
                pytest.fail("Undo stack did not cap at 20 entries")
        assert undo_count == 20

    def test_new_stroke_clears_redo_stack(self, controller, tmp_dirs):
        src, edit = tmp_dirs
        controller.load_frame(0, src, edit)

        controller.begin_stroke(10, 10, radius=3, is_eraser=False)
        controller.end_stroke()
        controller.undo()
        assert controller.can_redo is True

        # A new stroke invalidates the redo history.
        controller.begin_stroke(30, 30, radius=3, is_eraser=False)
        controller.end_stroke()
        assert controller.can_redo is False


class TestSave:
    def test_save_writes_to_manual_edited_subdir(self, controller, tmp_dirs):
        src, edit = tmp_dirs
        controller.load_frame(0, src, edit)
        controller.begin_stroke(25, 25, radius=4, is_eraser=False)
        controller.end_stroke()

        assert controller.save_frame_if_dirty() is True
        assert os.path.exists(os.path.join(edit, "frame_000.png"))

        # Reloading the frame should pick up the manual_edited version.
        controller.load_frame(0, src, edit)
        assert int(controller.current_mask[25, 25]) == 255

    def test_save_noop_if_not_dirty(self, controller, tmp_dirs):
        src, edit = tmp_dirs
        controller.load_frame(0, src, edit)

        assert controller.save_frame_if_dirty() is False
        assert not os.path.exists(os.path.join(edit, "frame_000.png"))

    def test_save_clears_dirty_flag(self, controller, tmp_dirs):
        src, edit = tmp_dirs
        controller.load_frame(0, src, edit)
        controller.begin_stroke(25, 25, radius=4, is_eraser=False)
        controller.end_stroke()

        controller.save_frame_if_dirty()
        assert controller.is_dirty is False


class TestSignals:
    def test_mask_modified_emitted_after_stroke(self, controller, tmp_dirs):
        src, edit = tmp_dirs
        controller.load_frame(0, src, edit)

        events = []
        controller.mask_modified.connect(lambda rect: events.append(rect))

        controller.begin_stroke(25, 25, radius=4, is_eraser=False)
        controller.end_stroke()

        assert len(events) >= 1  # emitted at least once during the stroke

    def test_undo_state_changed_emits_after_stroke(self, controller, tmp_dirs):
        src, edit = tmp_dirs
        controller.load_frame(0, src, edit)

        states = []
        controller.undo_state_changed.connect(
            lambda u, r: states.append((u, r))
        )

        controller.begin_stroke(25, 25, radius=4, is_eraser=False)
        controller.end_stroke()

        # After the stroke the controller must report can_undo=True.
        assert any(s[0] is True for s in states)
