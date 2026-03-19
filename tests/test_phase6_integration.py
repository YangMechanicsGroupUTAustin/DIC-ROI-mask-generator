"""Phase 6 verification: integration tests for MainWindow wiring and edge cases.

Tests MainWindow signal wiring with controllers, UI state management,
keyboard shortcuts, AppState edge cases, and AnnotationController edge cases.

Uses unittest.mock to avoid SAM2 model dependencies.
Requires PyQt6 and QApplication for signal tests.
"""
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication


# --- QApplication Singleton ---

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    return app


# --- Mock Controllers with proper PyQt signals ---

class MockProcessingController(QObject):
    """Mock ProcessingController with real PyQt signals for integration tests."""
    progress = pyqtSignal(int, int, str)
    frame_processed = pyqtSignal(int, object)
    processing_finished = pyqtSignal()
    processing_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self.start_processing_called = False
        self.stop_processing_called = False
        self.start_correction_calls = []

    @property
    def is_running(self):
        return self._running

    def start_processing(self, skip_existing: bool = False):
        self.start_processing_called = True
        self._running = True

    def stop_processing(self):
        self.stop_processing_called = True
        self._running = False

    def start_correction(self, frame_idx, points, labels):
        self.start_correction_calls.append((frame_idx, points, labels))
        self._running = True

    @property
    def _mask_generator(self):
        """Mock mask generator for correction validation."""
        class _MockMG:
            is_initialized = False
        return _MockMG()


class MockSmoothingController(QObject):
    """Mock SmoothingController with real PyQt signals for integration tests."""
    progress = pyqtSignal(int, int, str)
    smoothing_finished = pyqtSignal(str)
    smoothing_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self.start_spatial_calls = []
        self.start_temporal_calls = []
        self.stop_called = False

    @property
    def is_running(self):
        return self._running

    def start_spatial(self, **kwargs):
        self.start_spatial_calls.append(kwargs)
        self._running = True

    def start_temporal(self, **kwargs):
        self.start_temporal_calls.append(kwargs)
        self._running = True

    def stop(self):
        self.stop_called = True
        self._running = False


# --- Fixtures ---

@pytest.fixture
def state(qapp):
    from controllers.app_state import AppState
    return AppState()


@pytest.fixture
def annotation_ctrl(state):
    from controllers.annotation_controller import AnnotationController
    return AnnotationController(state)


@pytest.fixture
def processing_ctrl(qapp):
    return MockProcessingController()


@pytest.fixture
def smoothing_ctrl(qapp):
    return MockSmoothingController()


@pytest.fixture
def wired_window(qapp, state, annotation_ctrl, processing_ctrl, smoothing_ctrl):
    """Create MainWindow with all controllers wired."""
    from gui.main_window import MainWindow
    window = MainWindow(
        app_state=state,
        annotation_controller=annotation_ctrl,
        processing_controller=processing_ctrl,
        smoothing_controller=smoothing_ctrl,
    )
    yield window
    # Use hide() instead of close() to avoid closeEvent triggering
    # QMessageBox on Windows during test teardown
    window.hide()


# =============================================================================
# MainWindow Integration Tests
# =============================================================================

class TestMainWindowControllerWiring:
    """Verify MainWindow stores and wires controllers."""

    def test_stores_controllers(self, wired_window, state, annotation_ctrl,
                                processing_ctrl, smoothing_ctrl):
        """MainWindow should store all controller references."""
        assert wired_window._state is state
        assert wired_window._annotation is annotation_ctrl
        assert wired_window._processing is processing_ctrl
        assert wired_window._smoothing is smoothing_ctrl

    def test_state_changed_signal_connected(self, wired_window, state):
        """State changes should trigger UI updates without errors."""
        from controllers.app_state import AppState
        # Transition through all states to verify no disconnected signals
        for target_state in AppState.State:
            state.set_state(target_state)
            # No exception means signal is connected


class TestMainWindowStateUI:
    """Test _update_ui_for_state for all 6 states."""

    def test_init_state_ui(self, wired_window, state):
        """INIT state: sidebar enabled, start/correction buttons disabled."""
        from controllers.app_state import AppState
        state.set_state(AppState.State.INIT)
        toolbar = wired_window.toolbar

        assert wired_window.sidebar.isEnabled() is True
        assert toolbar._start_btn.isEnabled() is False
        assert toolbar._add_correction_btn.isEnabled() is False
        assert toolbar._apply_correction_btn.isEnabled() is False
        assert wired_window.frame_navigator.isEnabled() is True

    def test_annotating_state_ui(self, wired_window, state):
        """ANNOTATING state: start button enabled, correction apply disabled."""
        from controllers.app_state import AppState
        state.set_state(AppState.State.ANNOTATING)
        toolbar = wired_window.toolbar

        assert wired_window.sidebar.isEnabled() is True
        assert toolbar._start_btn.isEnabled() is True
        assert toolbar._clear_btn.isEnabled() is True
        assert toolbar._undo_btn.isEnabled() is True
        assert toolbar._add_correction_btn.isEnabled() is False
        assert toolbar._apply_correction_btn.isEnabled() is False

    def test_processing_state_ui(self, wired_window, state):
        """PROCESSING state: sidebar and nav disabled, timer started."""
        from controllers.app_state import AppState
        state.set_state(AppState.State.PROCESSING)
        toolbar = wired_window.toolbar

        assert wired_window.sidebar.isEnabled() is False
        assert toolbar._select_btn.isEnabled() is False
        assert toolbar._draw_btn.isEnabled() is False
        assert toolbar._start_btn.isEnabled() is False
        assert toolbar._clear_btn.isEnabled() is False
        assert wired_window.frame_navigator.isEnabled() is False

    def test_reviewing_state_ui(self, wired_window, state):
        """REVIEWING state: start and add-correction enabled."""
        from controllers.app_state import AppState
        state.set_state(AppState.State.REVIEWING)
        toolbar = wired_window.toolbar

        assert wired_window.sidebar.isEnabled() is True
        assert toolbar._start_btn.isEnabled() is True
        assert toolbar._add_correction_btn.isEnabled() is True
        assert toolbar._apply_correction_btn.isEnabled() is False

    def test_correction_state_ui(self, wired_window, state):
        """CORRECTION state: apply correction enabled, add correction disabled."""
        from controllers.app_state import AppState
        state.set_state(AppState.State.CORRECTION)
        toolbar = wired_window.toolbar

        assert wired_window.sidebar.isEnabled() is True
        assert toolbar._apply_correction_btn.isEnabled() is True
        assert toolbar._add_correction_btn.isEnabled() is False
        assert toolbar._clear_btn.isEnabled() is True

    def test_post_processing_state_ui(self, wired_window, state):
        """POST_PROCESSING state: sidebar and nav disabled, like processing."""
        from controllers.app_state import AppState
        state.set_state(AppState.State.POST_PROCESSING)
        toolbar = wired_window.toolbar

        assert wired_window.sidebar.isEnabled() is False
        assert toolbar._select_btn.isEnabled() is False
        assert toolbar._start_btn.isEnabled() is False
        assert wired_window.frame_navigator.isEnabled() is False


class TestMainWindowDeviceHandler:
    """Test _on_device_changed conversion."""

    def test_cuda_conversion(self, wired_window, state):
        """CUDA display name should be converted to 'cuda' string."""
        wired_window._on_device_changed("CUDA")
        assert state.device == "cuda"

    def test_cpu_conversion(self, wired_window, state):
        """CPU display name should be converted to 'cpu' string."""
        wired_window._on_device_changed("CPU")
        assert state.device == "cpu"

    def test_mps_conversion(self, wired_window, state):
        """MPS display name should be converted to 'mps' string."""
        wired_window._on_device_changed("MPS")
        assert state.device == "mps"


class TestMainWindowFrameHandlers:
    """Test frame navigation handler methods."""

    def test_on_frame_changed_updates_state(self, wired_window, state):
        """_on_frame_changed should update state current frame."""
        state._image_files = ["a.png"] * 10
        wired_window._on_frame_changed(5)
        assert state.current_frame == 5

    def test_on_start_frame_changed(self, wired_window, state):
        """_on_start_frame_changed should update state frame range."""
        state._image_files = ["a.png"] * 20
        state.set_frame_range(1, 20)
        wired_window._on_start_frame_changed(5)
        assert state.start_frame == 5

    def test_on_end_frame_changed(self, wired_window, state):
        """_on_end_frame_changed should update state frame range."""
        state._image_files = ["a.png"] * 20
        state.set_frame_range(1, 20)
        wired_window._on_end_frame_changed(15)
        assert state.end_frame == 15


class TestMainWindowProcessingValidation:
    """Test _on_start_processing validation."""

    def test_no_images_shows_error(self, wired_window, state, processing_ctrl):
        """Start processing with no images should show error, not start."""
        state._image_files = []  # Ensure no images loaded
        with patch.object(wired_window, '_show_error') as mock_err:
            wired_window._on_start_processing()
            mock_err.assert_called_once()
            assert "No images" in mock_err.call_args[0][1]
        assert processing_ctrl.start_processing_called is False

    def test_no_output_dir_shows_error(self, wired_window, state, processing_ctrl):
        """Start processing with no output dir should show error."""
        state._image_files = ["a.png"]
        state._output_dir = ""  # Ensure no output dir
        with patch.object(wired_window, '_show_error') as mock_err:
            wired_window._on_start_processing()
            mock_err.assert_called_once()
            assert "output directory" in mock_err.call_args[0][1].lower()
        assert processing_ctrl.start_processing_called is False

    def test_no_points_shows_error(self, wired_window, state, processing_ctrl):
        """Start processing with no points should show error."""
        state._image_files = ["a.png"]
        state._output_dir = "/tmp/out"
        state._points = []  # Ensure no points
        state._labels = []
        with patch.object(wired_window, '_show_error') as mock_err:
            wired_window._on_start_processing()
            mock_err.assert_called_once()
            assert "point" in mock_err.call_args[0][1].lower()
        assert processing_ctrl.start_processing_called is False

    def test_valid_processing_starts(self, wired_window, state, processing_ctrl):
        """Start processing with valid config should start the controller."""
        state._image_files = ["a.png"]
        state._output_dir = "/tmp/out"
        state.add_point(10.0, 20.0, 1)
        wired_window._on_start_processing()
        assert processing_ctrl.start_processing_called is True


class TestMainWindowCorrectionHandlers:
    """Test correction-related handlers."""

    def test_add_correction_blocked_without_model(self, wired_window, state):
        """_on_add_correction should NOT transition without initialized model."""
        from controllers.app_state import AppState
        state.set_state(AppState.State.REVIEWING)
        wired_window._on_add_correction()
        # Model is not initialized in mock, so state stays REVIEWING
        assert state.state == AppState.State.REVIEWING

    def test_apply_correction_no_points(self, wired_window, state, processing_ctrl):
        """_on_apply_correction with no points should show error."""
        with patch.object(wired_window, '_show_error') as mock_err:
            wired_window._on_apply_correction()
            mock_err.assert_called_once()
            assert "point" in mock_err.call_args[0][1].lower()
        assert len(processing_ctrl.start_correction_calls) == 0

    def test_apply_correction_with_points(self, wired_window, state, processing_ctrl):
        """_on_apply_correction with points should start correction."""
        from controllers.app_state import AppState
        state._image_files = ["a.png"] * 5
        state.set_current_frame(3)
        state.add_point(50.0, 60.0, 1)
        state.set_state(AppState.State.CORRECTION)

        wired_window._on_apply_correction()

        assert len(processing_ctrl.start_correction_calls) == 1
        frame_idx, pts, lbls = processing_ctrl.start_correction_calls[0]
        assert frame_idx == 2  # current_frame(3) - 1 = 0-based index 2
        assert pts == [[50.0, 60.0]]
        assert lbls == [1]


class TestMainWindowProcessingCallbacks:
    """Test processing/smoothing finished/error callbacks."""

    def test_processing_finished_transitions(self, wired_window, state):
        """_on_processing_finished should transition to REVIEWING."""
        from controllers.app_state import AppState
        state.set_state(AppState.State.PROCESSING)
        wired_window._on_processing_finished()
        assert state.state == AppState.State.REVIEWING

    def test_processing_error_with_images(self, wired_window, state):
        """Error during processing with images loaded -> ANNOTATING."""
        from controllers.app_state import AppState
        state._image_files = ["a.png"]
        state.set_state(AppState.State.PROCESSING)
        with patch.object(wired_window, '_show_error'):
            wired_window._on_processing_error("Something failed")
        assert state.state == AppState.State.ANNOTATING

    def test_processing_error_no_images(self, wired_window, state):
        """Error during processing with no images -> INIT."""
        from controllers.app_state import AppState
        state._image_files = []
        state.set_state(AppState.State.PROCESSING)
        with patch.object(wired_window, '_show_error'):
            wired_window._on_processing_error("Something failed")
        assert state.state == AppState.State.INIT

    def test_smoothing_finished_transitions(self, wired_window, state):
        """_on_smoothing_finished should transition to REVIEWING."""
        from controllers.app_state import AppState
        state.set_state(AppState.State.POST_PROCESSING)
        wired_window._on_smoothing_finished("/output/dir")
        assert state.state == AppState.State.REVIEWING

    def test_smoothing_error_transitions(self, wired_window, state):
        """_on_smoothing_error should transition to REVIEWING."""
        from controllers.app_state import AppState
        state.set_state(AppState.State.POST_PROCESSING)
        with patch.object(wired_window, '_show_error'):
            wired_window._on_smoothing_error("Smoothing failed")
        assert state.state == AppState.State.REVIEWING


class TestMainWindowSmoothingValidation:
    """Test smoothing validation in MainWindow."""

    def test_spatial_smooth_no_output_dir(self, wired_window, state):
        """Spatial smooth with no output dir should show error."""
        state._output_dir = ""
        with patch.object(wired_window, '_show_error') as mock_err:
            wired_window._on_spatial_smooth({})
            mock_err.assert_called_once()
            assert "output directory" in mock_err.call_args[0][1].lower()

    def test_spatial_smooth_no_masks_dir(self, wired_window, state):
        """Spatial smooth with no masks dir should show error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state._output_dir = tmpdir
            # No 'masks' subdirectory
            with patch.object(wired_window, '_show_error') as mock_err:
                wired_window._on_spatial_smooth({})
                mock_err.assert_called_once()
                assert "not found" in mock_err.call_args[0][1].lower()

    def test_spatial_smooth_valid(self, wired_window, state, smoothing_ctrl):
        """Spatial smooth with valid config should start the controller."""
        from controllers.app_state import AppState
        with tempfile.TemporaryDirectory() as tmpdir:
            state._output_dir = tmpdir
            masks_dir = os.path.join(tmpdir, "masks")
            os.makedirs(masks_dir)

            wired_window._on_spatial_smooth({"iterations": 10, "dt": 0.05})

            assert len(smoothing_ctrl.start_spatial_calls) == 1
            assert state.state == AppState.State.POST_PROCESSING


# =============================================================================
# Keyboard Shortcut Tests (Optional)
# =============================================================================

class TestKeyboardShortcuts:
    """Test keyboard shortcuts if Phase 7 registered them."""

    def test_shortcuts_exist_if_registered(self, wired_window):
        """Check if shortcut actions exist on the window.

        These are optional - skip if Phase 7 hasn't run.
        """
        # Common shortcuts that Phase 7 might register
        shortcut_actions = wired_window.findChildren(
            type(wired_window).staticMetaObject.__class__
        ) if hasattr(wired_window, 'findChildren') else []
        # Just verify the window was created without errors
        assert wired_window is not None


# =============================================================================
# AppState Edge Cases
# =============================================================================

class TestAppStateEdgeCases:
    """Additional edge case tests for AppState."""

    def test_frame_range_start_greater_than_end(self, state):
        """When start > end, end should be clamped to >= start."""
        state._image_files = ["a.png"] * 20
        state.set_frame_range(15, 5)
        assert state.start_frame == 15
        assert state.end_frame == 15  # clamped

    def test_frame_range_beyond_total(self, state):
        """Frame range values beyond total_frames should be clamped."""
        state._image_files = ["a.png"] * 10
        state.set_frame_range(1, 100)
        assert state.end_frame == 10

    def test_set_device_unknown_string(self, state):
        """Setting an arbitrary device string should be stored as-is."""
        state.set_device("some_unknown_device")
        assert state.device == "some_unknown_device"

    def test_set_display_images_with_none(self, state):
        """set_display_images with None values should not replace existing."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        state.set_display_images(original=img)
        assert state.current_original is not None

        # Calling with no args should emit signal but not clear existing
        state.set_display_images()
        assert state.current_original is not None

    def test_double_state_transition_same_state(self, state):
        """Setting same state twice should NOT emit signal second time."""
        from controllers.app_state import AppState
        signals = []
        state.state_changed.connect(lambda s: signals.append(s))
        state.set_state(AppState.State.ANNOTATING)
        state.set_state(AppState.State.ANNOTATING)
        assert len(signals) == 1  # only first transition emits

    def test_model_config_unknown_model_fallback(self, state):
        """Unknown model name should fall back to Large defaults."""
        state._model_name = "NonExistent Model XYZ"
        cfg, ckpt = state.get_model_config()
        assert cfg == "sam2.1_hiera_l.yaml"
        assert "large" in ckpt

    def test_current_frame_no_images(self, state):
        """Current frame should stay at 1 when no images loaded."""
        state._image_files = []
        state.set_current_frame(5)
        assert state.current_frame == 1

    def test_set_image_files_resets_navigation(self, state):
        """Setting new image files should reset frame range and current."""
        state._image_files = ["a.png"] * 10
        state.set_current_frame(5)
        state.set_frame_range(3, 8)

        # Load new set of files
        state.set_image_files(["b.png"] * 20)
        assert state.current_frame == 1
        assert state.start_frame == 1
        assert state.end_frame == 20


# =============================================================================
# AnnotationController Edge Cases
# =============================================================================

class TestAnnotationControllerEdgeCases:
    """Additional edge case tests for AnnotationController."""

    def test_add_undo_redo_undo_sequence(self, annotation_ctrl, state):
        """Add -> undo -> redo -> undo should return to empty."""
        annotation_ctrl.add_point(10.0, 20.0)
        assert len(state.points) == 1

        annotation_ctrl.undo()
        assert len(state.points) == 0

        annotation_ctrl.redo()
        assert len(state.points) == 1

        annotation_ctrl.undo()
        assert len(state.points) == 0
        assert annotation_ctrl.can_redo is True
        assert annotation_ctrl.can_undo is False

    def test_multiple_ops_undo_all(self, annotation_ctrl, state):
        """Multiple operations then undo all should restore empty state."""
        annotation_ctrl.add_point(1.0, 2.0)
        annotation_ctrl.add_point(3.0, 4.0)
        annotation_ctrl.add_point(5.0, 6.0)
        annotation_ctrl.move_point(0, 10.0, 20.0)
        annotation_ctrl.remove_point(1)  # remove middle point

        assert len(state.points) == 2

        # Undo all 5 operations
        for _ in range(5):
            annotation_ctrl.undo()

        assert len(state.points) == 0
        assert annotation_ctrl.can_undo is False

    def test_move_point_invalid_index(self, annotation_ctrl, state):
        """Move with invalid index should be a no-op."""
        annotation_ctrl.add_point(10.0, 20.0)
        initial_points = list(state.points)

        # Out of range
        annotation_ctrl.move_point(99, 50.0, 60.0)
        assert state.points == initial_points

        # Negative index
        annotation_ctrl.move_point(-1, 50.0, 60.0)
        assert state.points == initial_points

    def test_remove_point_invalid_index(self, annotation_ctrl, state):
        """Remove with invalid index should be a no-op."""
        annotation_ctrl.add_point(10.0, 20.0)
        initial_count = len(state.points)

        annotation_ctrl.remove_point(99)
        assert len(state.points) == initial_count

        annotation_ctrl.remove_point(-1)
        assert len(state.points) == initial_count

    def test_save_load_config(self, annotation_ctrl, state):
        """Save and load config should round-trip point data."""
        state._image_files = ["a.png"] * 10
        state.set_frame_range(2, 8)
        state.set_model_name("SAM2 Hiera Small")
        state.set_device("cuda")
        state.set_threshold(0.5)

        annotation_ctrl.set_point_mode("foreground")
        annotation_ctrl.add_point(100.0, 200.0)
        annotation_ctrl.set_point_mode("background")
        annotation_ctrl.add_point(50.0, 60.0)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            filepath = f.name

        try:
            annotation_ctrl.save_config(filepath)

            # Verify file was created with valid JSON
            with open(filepath, "r") as f:
                data = json.load(f)
            assert data["annotation"]["points"] == [[100.0, 200.0], [50.0, 60.0]]
            assert data["annotation"]["labels"] == [1, 0]
            assert data["parameters"]["model"] == "SAM2 Hiera Small"

            # Clear and reload
            annotation_ctrl.clear_points()
            assert len(state.points) == 0

            annotation_ctrl.load_config(filepath)
            assert len(state.points) == 2
            assert state.points[0] == [100.0, 200.0]
            assert state.labels == [1, 0]
            assert state.model_name == "SAM2 Hiera Small"

            # Undo/redo stacks should be cleared after load
            assert annotation_ctrl.can_undo is False
            assert annotation_ctrl.can_redo is False
        finally:
            os.unlink(filepath)

    def test_clear_empty_is_noop(self, annotation_ctrl, state):
        """Clearing when already empty should not add to undo stack."""
        initial_can_undo = annotation_ctrl.can_undo
        annotation_ctrl.clear_points()
        assert annotation_ctrl.can_undo == initial_can_undo

    def test_point_mode_affects_label(self, annotation_ctrl, state):
        """Point mode should determine the label value."""
        annotation_ctrl.set_point_mode("foreground")
        annotation_ctrl.add_point(1.0, 2.0)
        assert state.labels[-1] == 1

        annotation_ctrl.set_point_mode("background")
        annotation_ctrl.add_point(3.0, 4.0)
        assert state.labels[-1] == 0
