"""Tests for bug fixes: path reset, new project flow, and new features.

Covers:
- Bug #1: New Project resets all state including paths
- Bug #2: Re-running on same images after New Project works correctly
- Feature: Marked frames
- Feature: Project save/load
- Feature: Contour export
- Feature: Batch processing
"""
import json
import os
import sys
import tempfile

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    return app


@pytest.fixture
def state(qapp):
    from controllers.app_state import AppState
    return AppState()


@pytest.fixture
def sample_images(tmp_path):
    """Create a temp directory with sample PNG images."""
    for i in range(5):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"frame_{i:03d}.png"), img)
    return str(tmp_path)


@pytest.fixture
def sample_masks(tmp_path):
    """Create a temp directory with sample binary mask files."""
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    for i in range(5):
        mask = np.zeros((64, 64), dtype=np.uint8)
        # Draw a white rectangle as a "mask"
        mask[10:50, 10:50] = 255
        cv2.imwrite(str(mask_dir / f"frame_{i:03d}.tiff"), mask)
    return str(tmp_path)


# ==============================================================================
# Bug #1 + #2: New Project state reset
# ==============================================================================


class TestNewProjectReset:
    """Verify _new_project() properly resets ALL state."""

    def test_input_dir_reset(self, state, sample_images):
        """After setting input_dir then clearing, re-setting same dir must work."""
        state.set_input_dir(sample_images)
        assert state.input_dir == sample_images
        assert len(state.image_files) == 5

        # Simulate _new_project reset
        state.set_image_files([])
        state._input_dir = ""
        state._output_dir = ""

        assert state.input_dir == ""
        assert state.output_dir == ""
        assert len(state.image_files) == 0

        # Re-enter the SAME directory — must re-discover images
        state.set_input_dir(sample_images)
        assert state.input_dir == sample_images
        assert len(state.image_files) == 5, (
            "BUG #2: set_input_dir guard prevented re-discovery!"
        )

    def test_output_dir_reset(self, state):
        """Output dir must be resettable."""
        state.set_output_dir("/some/path")
        assert state.output_dir == "/some/path"

        state._output_dir = ""
        assert state.output_dir == ""

        # Re-set same path must work
        state.set_output_dir("/some/path")
        assert state.output_dir == "/some/path"

    def test_marked_frames_cleared(self, state):
        """New project must clear marked frames."""
        state._image_files = ["a"] * 10
        state.toggle_marked_frame(1)
        state.toggle_marked_frame(5)
        assert len(state.marked_frames) == 2

        state.clear_marked_frames()
        assert len(state.marked_frames) == 0

    def test_points_cleared(self, state):
        """New project must clear all points."""
        state.add_point(10.0, 20.0, 1)
        state.add_point(30.0, 40.0, 0)
        state.set_points([], [])
        assert len(state.points) == 0
        assert len(state.labels) == 0

    def test_full_new_project_cycle(self, state, sample_images):
        """Full cycle: setup → new project → re-setup with same paths."""
        # Setup
        state.set_input_dir(sample_images)
        state.set_output_dir(sample_images)
        state.add_point(10.0, 20.0, 1)
        state.toggle_marked_frame(1)

        initial_file_count = len(state.image_files)
        assert initial_file_count == 5

        # New project reset (mirrors MainWindow._new_project)
        state.set_points([], [])
        state.set_image_files([])
        state._input_dir = ""
        state._output_dir = ""
        state.clear_marked_frames()
        from controllers.app_state import AppState
        state.set_state(AppState.State.INIT)

        # Verify clean state
        assert state.input_dir == ""
        assert state.output_dir == ""
        assert len(state.points) == 0
        assert len(state.marked_frames) == 0
        assert len(state.image_files) == 0

        # Re-setup with same paths
        state.set_input_dir(sample_images)
        state.set_output_dir(sample_images)
        assert len(state.image_files) == initial_file_count


# ==============================================================================
# Feature: Marked Frames
# ==============================================================================


class TestMarkedFrames:
    def test_toggle_mark(self, state):
        state._image_files = ["a"] * 10
        state.toggle_marked_frame(3)
        assert 3 in state.marked_frames
        state.toggle_marked_frame(3)
        assert 3 not in state.marked_frames

    def test_next_marked_frame(self, state):
        state._image_files = ["a"] * 20
        state.toggle_marked_frame(5)
        state.toggle_marked_frame(10)
        state.toggle_marked_frame(15)

        assert state.next_marked_frame(1) == 5
        assert state.next_marked_frame(5) == 10
        assert state.next_marked_frame(10) == 15
        assert state.next_marked_frame(15) is None

    def test_prev_marked_frame(self, state):
        state._image_files = ["a"] * 20
        state.toggle_marked_frame(5)
        state.toggle_marked_frame(10)

        assert state.prev_marked_frame(15) == 10
        assert state.prev_marked_frame(10) == 5
        assert state.prev_marked_frame(5) is None

    def test_clear_marked_frames(self, state):
        state.toggle_marked_frame(1)
        state.toggle_marked_frame(2)
        state.clear_marked_frames()
        assert len(state.marked_frames) == 0

    def test_marked_frames_signal(self, state):
        received = []
        state.marked_frames_changed.connect(lambda s: received.append(s))
        state.toggle_marked_frame(3)
        assert len(received) == 1
        assert 3 in received[0]


# ==============================================================================
# Feature: Project Save/Load
# ==============================================================================


class TestProjectSaveLoad:
    def test_save_and_load(self, state, tmp_path, sample_images):
        """Round-trip: save project → load → verify all fields."""
        from core.project import save_project, load_project

        state.set_input_dir(sample_images)
        state.set_output_dir(str(tmp_path))
        state.set_model_name("SAM2 Hiera Small")
        state.set_device("cpu")
        state.set_threshold(0.5)
        state.add_point(10.0, 20.0, 1)
        state.add_point(30.0, 40.0, 0)
        state.toggle_marked_frame(2)

        filepath = str(tmp_path / "test.s2proj")
        save_project(filepath, state)

        # Verify file exists and is valid JSON
        assert os.path.isfile(filepath)
        with open(filepath, "r") as f:
            data = json.load(f)

        assert data["version"] == "1.0"
        assert data["paths"]["input_dir"] == sample_images
        assert data["model"]["name"] == "SAM2 Hiera Small"
        assert data["model"]["threshold"] == 0.5
        assert len(data["annotation"]["points"]) == 2
        assert data["frames"]["marked"] == [2]

    def test_load_and_apply(self, state, tmp_path, sample_images):
        """Load a saved project and apply to fresh state."""
        from controllers.app_state import AppState
        from core.project import save_project, load_project, apply_project_to_state

        # Setup and save
        state.set_input_dir(sample_images)
        state.set_output_dir(str(tmp_path))
        state.set_threshold(0.75)
        state.add_point(5.0, 5.0, 1)
        state.toggle_marked_frame(3)

        filepath = str(tmp_path / "test2.s2proj")
        save_project(filepath, state)

        # Fresh state
        new_state = AppState()
        project = load_project(filepath)
        apply_project_to_state(project, new_state)

        assert new_state.input_dir == sample_images
        assert new_state.output_dir == str(tmp_path)
        assert new_state.threshold == 0.75
        assert len(new_state.points) == 1
        assert 3 in new_state.marked_frames

    def test_invalid_project_file(self, tmp_path):
        """Loading an invalid project file should raise ValueError."""
        from core.project import load_project

        filepath = str(tmp_path / "bad.s2proj")
        with open(filepath, "w") as f:
            json.dump({"invalid": True}, f)

        with pytest.raises(ValueError, match="missing 'paths'"):
            load_project(filepath)


# ==============================================================================
# Feature: Contour Export
# ==============================================================================


class TestContourExport:
    def test_extract_contours(self):
        """Extract contours from a binary mask."""
        from core.contour_export import extract_contours

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255  # White square

        contours = extract_contours(mask)
        assert len(contours) >= 1
        # The contour should have roughly 4 points (corners)
        total_pts = sum(len(c) for c in contours)
        assert total_pts >= 4

    def test_export_contour_png(self, tmp_path):
        """Export contours as PNG."""
        from core.contour_export import export_contour_png

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 30:70] = 255

        out_path = str(tmp_path / "contour.png")
        ok = export_contour_png(mask, out_path)
        assert ok
        assert os.path.isfile(out_path)

        # Verify the PNG has content (not all black)
        result = cv2.imread(out_path)
        assert result is not None
        assert result.sum() > 0  # Not all zeros

    def test_export_contour_svg(self, tmp_path):
        """Export contours as SVG."""
        from core.contour_export import export_contour_svg

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 30:70] = 255

        out_path = str(tmp_path / "contour.svg")
        ok = export_contour_svg(mask, out_path)
        assert ok
        assert os.path.isfile(out_path)

        # Verify SVG content
        with open(out_path, "r") as f:
            content = f.read()
        assert "<svg" in content
        assert "<path" in content
        assert 'fill="none"' in content

    def test_batch_export(self, sample_masks):
        """Batch export contours for all masks in a directory."""
        from core.contour_export import batch_export_contours

        mask_dir = os.path.join(sample_masks, "masks")
        out_dir = os.path.join(sample_masks, "contours_png")

        progress_calls = []
        count = batch_export_contours(
            mask_dir, out_dir, fmt="PNG",
            progress_callback=lambda c, t, m: progress_calls.append((c, t)),
        )

        assert count == 5
        assert len(progress_calls) == 5
        assert os.listdir(out_dir)

    def test_batch_export_svg(self, sample_masks):
        """Batch SVG export."""
        from core.contour_export import batch_export_contours

        mask_dir = os.path.join(sample_masks, "masks")
        out_dir = os.path.join(sample_masks, "contours_svg")

        count = batch_export_contours(mask_dir, out_dir, fmt="SVG")
        assert count == 5

    def test_empty_mask_no_contours(self, tmp_path):
        """Empty mask produces no contours."""
        from core.contour_export import extract_contours, export_contour_png

        mask = np.zeros((50, 50), dtype=np.uint8)
        contours = extract_contours(mask)
        assert len(contours) == 0

        ok = export_contour_png(mask, str(tmp_path / "empty.png"))
        assert not ok  # No contours to export


# ==============================================================================
# Feature: Overlay settings
# ==============================================================================


class TestOverlaySettings:
    def test_overlay_alpha_clamping(self, state):
        state.set_overlay_alpha(0.5)
        assert state.overlay_alpha == 0.5

        state.set_overlay_alpha(-0.5)
        assert state.overlay_alpha == 0.0

        state.set_overlay_alpha(2.0)
        assert state.overlay_alpha == 1.0

    def test_overlay_color(self, state):
        state.set_overlay_color((0, 128, 255))
        assert state.overlay_color == (0, 128, 255)


# ==============================================================================
# Feature: Mask output format
# ==============================================================================


class TestMaskOutputFormat:
    def test_default_format(self, state):
        assert state.mask_output_format == "TIFF (default)"

    def test_set_format(self, state):
        state.set_mask_output_format("PNG (lossless)")
        assert state.mask_output_format == "PNG (lossless)"


# ==============================================================================
# AppState signal integrity
# ==============================================================================


class TestAppStateSignals:
    """Ensure signals fire correctly for all new state properties."""

    def test_marked_frames_signal_emitted(self, state):
        signals = []
        state.marked_frames_changed.connect(lambda s: signals.append(s))
        state.toggle_marked_frame(1)
        state.toggle_marked_frame(2)
        state.toggle_marked_frame(1)  # Remove
        assert len(signals) == 3
        assert 1 in signals[0]
        assert 2 in signals[1]
        assert 1 not in signals[2]

    def test_preprocessing_signal(self, state):
        from core.preprocessing import PreprocessingConfig
        signals = []
        state.preprocessing_changed.connect(lambda: signals.append(True))
        state.set_preprocessing_config(
            PreprocessingConfig(brightness=10)
        )
        assert len(signals) == 1

    def test_frame_range_signal(self, state):
        state._image_files = ["a"] * 20
        signals = []
        state.frame_range_changed.connect(
            lambda s, e: signals.append((s, e))
        )
        state.set_frame_range(5, 15)
        assert signals == [(5, 15)]

    def test_no_duplicate_signal_on_same_value(self, state):
        """Setting same device twice should emit only once."""
        signals = []
        state.device_changed.connect(lambda d: signals.append(d))
        state.set_device("cuda")
        state.set_device("cuda")  # Same value
        assert len(signals) == 1


# ==============================================================================
# Preprocessing config
# ==============================================================================


class TestPreprocessingConfig:
    def test_builtin_presets_exist(self):
        from core.preprocessing import BUILTIN_PRESETS
        assert len(BUILTIN_PRESETS) >= 7
        assert "None (identity)" in BUILTIN_PRESETS
        assert "DIC Microscopy" in BUILTIN_PRESETS

    def test_identity_preset(self):
        from core.preprocessing import BUILTIN_PRESETS
        identity = BUILTIN_PRESETS["None (identity)"]
        assert identity.is_identity()

    def test_dic_preset_not_identity(self):
        from core.preprocessing import BUILTIN_PRESETS
        dic = BUILTIN_PRESETS["DIC Microscopy"]
        assert not dic.is_identity()

    def test_frozen_immutability(self):
        from core.preprocessing import PreprocessingConfig
        config = PreprocessingConfig(brightness=10)
        with pytest.raises(AttributeError):
            config.brightness = 20
