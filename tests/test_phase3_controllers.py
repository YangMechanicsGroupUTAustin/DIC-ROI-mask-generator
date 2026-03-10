"""Phase 3 verification: controllers and utilities."""
import sys
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
def annotation_ctrl(state):
    from controllers.annotation_controller import AnnotationController
    return AnnotationController(state)


# --- AppState Tests ---

class TestAppState:
    def test_initial_state(self, state):
        from controllers.app_state import AppState
        assert state.state == AppState.State.INIT
        assert state.input_dir == ""
        assert state.total_frames == 0
        assert state.current_frame == 1

    def test_state_transitions(self, state):
        from controllers.app_state import AppState
        signals = []
        state.state_changed.connect(lambda s: signals.append(s))
        state.set_state(AppState.State.ANNOTATING)
        assert state.state == AppState.State.ANNOTATING
        assert signals == ["annotating"]

    def test_frame_clamping(self, state):
        state._image_files = ["a.png"] * 10
        state.set_current_frame(0)
        assert state.current_frame == 1
        state.set_current_frame(100)
        assert state.current_frame == 10

    def test_frame_range_validation(self, state):
        state._image_files = ["a.png"] * 20
        state.set_frame_range(5, 15)
        assert state.start_frame == 5
        assert state.end_frame == 15
        state.set_frame_range(15, 5)  # start > end
        assert state.start_frame == 15
        assert state.end_frame == 15  # end clamped to >= start

    def test_points_management(self, state):
        state.add_point(10.0, 20.0, 1)
        state.add_point(30.0, 40.0, 0)
        assert len(state.points) == 2
        assert state.labels == [1, 0]

        old_pt, old_lbl = state.remove_point(0)
        assert old_pt == [10.0, 20.0]
        assert old_lbl == 1
        assert len(state.points) == 1

    def test_move_point(self, state):
        state.add_point(10.0, 20.0, 1)
        old_x, old_y = state.move_point(0, 50.0, 60.0)
        assert old_x == 10.0
        assert old_y == 20.0
        assert state.points[0] == [50.0, 60.0]

    def test_clear_points(self, state):
        state.add_point(1.0, 2.0, 1)
        state.add_point(3.0, 4.0, 0)
        old_pts, old_lbls = state.clear_points()
        assert len(old_pts) == 2
        assert len(state.points) == 0

    def test_set_display_images(self, state):
        import numpy as np
        signals = []
        state.current_images_changed.connect(lambda: signals.append(True))
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        state.set_display_images(original=img)
        assert state.current_original is not None
        assert len(signals) == 1

    def test_model_config(self, state):
        cfg, ckpt = state.get_model_config()
        assert cfg == "sam2.1_hiera_l.yaml"
        assert "large" in ckpt

    def test_device_signal(self, state):
        signals = []
        state.device_changed.connect(lambda d: signals.append(d))
        state.set_device("CUDA")
        assert signals == ["CUDA"]
        # Same value should not re-emit
        state.set_device("CUDA")
        assert len(signals) == 1


# --- AnnotationController Tests ---

class TestAnnotationController:
    def test_add_point(self, annotation_ctrl, state):
        annotation_ctrl.set_point_mode("foreground")
        annotation_ctrl.add_point(100.0, 200.0)
        assert len(state.points) == 1
        assert state.labels == [1]

    def test_undo_redo(self, annotation_ctrl, state):
        annotation_ctrl.add_point(10.0, 20.0)
        annotation_ctrl.add_point(30.0, 40.0)
        assert len(state.points) == 2

        annotation_ctrl.undo()
        assert len(state.points) == 1
        assert state.points[0] == [10.0, 20.0]

        annotation_ctrl.redo()
        assert len(state.points) == 2

    def test_undo_empty(self, annotation_ctrl, state):
        # Should not crash
        annotation_ctrl.undo()
        assert len(state.points) == 0

    def test_redo_empty(self, annotation_ctrl, state):
        annotation_ctrl.redo()
        assert len(state.points) == 0

    def test_clear_undo(self, annotation_ctrl, state):
        annotation_ctrl.add_point(1.0, 2.0)
        annotation_ctrl.add_point(3.0, 4.0)
        annotation_ctrl.clear_points()
        assert len(state.points) == 0
        annotation_ctrl.undo()
        assert len(state.points) == 2

    def test_move_undo(self, annotation_ctrl, state):
        annotation_ctrl.add_point(10.0, 20.0)
        annotation_ctrl.move_point(0, 50.0, 60.0)
        assert state.points[0] == [50.0, 60.0]
        annotation_ctrl.undo()
        assert state.points[0] == [10.0, 20.0]

    def test_remove_undo(self, annotation_ctrl, state):
        annotation_ctrl.add_point(10.0, 20.0)
        annotation_ctrl.add_point(30.0, 40.0)
        annotation_ctrl.remove_point(0)
        assert len(state.points) == 1
        annotation_ctrl.undo()
        assert len(state.points) == 2
        assert state.points[0] == [10.0, 20.0]

    def test_redo_cleared_on_new_action(self, annotation_ctrl, state):
        annotation_ctrl.add_point(1.0, 2.0)
        annotation_ctrl.undo()
        assert annotation_ctrl.can_redo
        annotation_ctrl.add_point(5.0, 6.0)
        assert not annotation_ctrl.can_redo

    def test_can_undo_redo_signals(self, annotation_ctrl, state):
        undo_signals = []
        redo_signals = []
        annotation_ctrl.can_undo_changed.connect(lambda v: undo_signals.append(v))
        annotation_ctrl.can_redo_changed.connect(lambda v: redo_signals.append(v))
        annotation_ctrl.add_point(1.0, 2.0)
        assert undo_signals[-1] is True
        assert redo_signals[-1] is False

    def test_max_history(self, annotation_ctrl, state):
        for i in range(150):
            annotation_ctrl.add_point(float(i), float(i))
        assert len(annotation_ctrl._undo_stack) == annotation_ctrl.MAX_HISTORY


# --- DeviceManager Tests ---

class TestDeviceManager:
    def test_detect_devices(self):
        from utils.device_manager import DeviceManager
        devices = DeviceManager.detect_available_devices()
        assert "CPU" in devices
        assert isinstance(devices, list)

    def test_device_string(self):
        from utils.device_manager import DeviceManager
        assert DeviceManager.get_device_string("CUDA") == "cuda"
        assert DeviceManager.get_device_string("CPU") == "cpu"
        assert DeviceManager.get_device_string("MPS") == "mps"
        assert DeviceManager.get_device_string("Unknown") == "cpu"

    def test_gpu_name(self):
        from utils.device_manager import DeviceManager
        name = DeviceManager.get_gpu_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_vram_usage(self):
        from utils.device_manager import DeviceManager
        used, total = DeviceManager.get_vram_usage()
        assert isinstance(used, float)
        assert isinstance(total, float)
        assert used >= 0
        assert total >= 0

    def test_torch_version(self):
        from utils.device_manager import DeviceManager
        version = DeviceManager.get_torch_version()
        assert "." in version

    def test_empty_cache(self):
        from utils.device_manager import DeviceManager
        # Should not crash
        DeviceManager.empty_cache()


# --- Logging Tests ---

class TestLogging:
    def test_setup_logging(self):
        import logging
        import tempfile
        import os
        from utils.logging_config import setup_logging, get_memory_handler
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(log_dir=tmpdir)
            logger.info("Test message")
            handler = get_memory_handler()
            assert handler is not None
            msgs = handler.get_messages()
            assert any("Test message" in m for m in msgs)
            # Check log file created
            log_files = [f for f in os.listdir(tmpdir) if f.endswith(".log")]
            assert len(log_files) >= 1
            # Close all handlers to release file locks (required on Windows)
            for h in logger.handlers[:]:
                h.close()
                logger.removeHandler(h)
