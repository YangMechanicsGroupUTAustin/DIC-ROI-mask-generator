"""Phase 2 verification: all GUI panels load and basic functionality works."""
import sys
import pytest
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    return app


class TestSidebar:
    def test_creation(self, qapp):
        from gui.panels.sidebar import Sidebar
        sidebar = Sidebar()
        assert sidebar is not None
        assert sidebar.minimumWidth() >= 250

    def test_signals_defined(self, qapp):
        from gui.panels.sidebar import Sidebar
        sidebar = Sidebar()
        # Verify all expected signals exist
        assert hasattr(sidebar, 'input_dir_changed')
        assert hasattr(sidebar, 'output_dir_changed')
        assert hasattr(sidebar, 'device_changed')
        assert hasattr(sidebar, 'model_changed')
        assert hasattr(sidebar, 'threshold_changed')
        assert hasattr(sidebar, 'spatial_smooth_requested')
        assert hasattr(sidebar, 'temporal_smooth_requested')


class TestToolbar:
    def test_creation(self, qapp):
        from gui.panels.toolbar import Toolbar
        toolbar = Toolbar()
        assert toolbar is not None

    def test_tool_change_signal(self, qapp):
        from gui.panels.toolbar import Toolbar
        toolbar = Toolbar()
        signals = []
        toolbar.tool_changed.connect(lambda t: signals.append(t))
        # The toolbar should have a way to change tool programmatically
        toolbar.set_tool("draw")
        toolbar.set_tool("erase")
        toolbar.set_tool("select")

    def test_mode_change(self, qapp):
        from gui.panels.toolbar import Toolbar
        toolbar = Toolbar()
        toolbar.set_mode("background")
        toolbar.set_mode("foreground")

    def test_processing_toggle(self, qapp):
        from gui.panels.toolbar import Toolbar
        toolbar = Toolbar()
        toolbar.set_processing(True)
        toolbar.set_processing(False)


class TestCanvasPanel:
    def test_creation(self, qapp):
        from gui.panels.canvas_panel import CanvasPanel
        panel = CanvasPanel("Test Panel", badge="Test")
        assert panel is not None

    def test_set_image(self, qapp):
        from gui.panels.canvas_panel import CanvasPanel
        panel = CanvasPanel("Test", is_interactive=True)
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[:, :, 1] = 128  # Green image
        panel.set_image(img)

    def test_set_points(self, qapp):
        from gui.panels.canvas_panel import CanvasPanel
        panel = CanvasPanel("Test", is_interactive=True)
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        panel.set_image(img)
        panel.set_points([[50.0, 50.0], [100.0, 30.0]], [1, 0])

    def test_zoom(self, qapp):
        from gui.panels.canvas_panel import CanvasPanel
        panel = CanvasPanel("Test")
        panel.set_zoom(200)
        assert panel.get_zoom() == 200

    def test_placeholder(self, qapp):
        from gui.panels.canvas_panel import CanvasPanel
        panel = CanvasPanel("Test", badge="Input")
        # Should show placeholder when no image set
        panel.clear_image()

    def test_grayscale_image(self, qapp):
        from gui.panels.canvas_panel import CanvasPanel
        panel = CanvasPanel("Test")
        img = np.zeros((100, 200), dtype=np.uint8)
        panel.set_image(img)

    def test_active_tool(self, qapp):
        from gui.panels.canvas_panel import CanvasPanel
        panel = CanvasPanel("Test", is_interactive=True)
        panel.set_active_tool("draw")
        panel.set_active_tool("erase")
        panel.set_active_tool("select")


class TestCanvasArea:
    def test_creation(self, qapp):
        from gui.panels.canvas_area import CanvasArea
        area = CanvasArea()
        assert area is not None

    def test_set_images(self, qapp):
        from gui.panels.canvas_area import CanvasArea
        area = CanvasArea()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        area.set_original_image(img)
        area.set_mask_image(mask)

    def test_set_overlay(self, qapp):
        from gui.panels.canvas_area import CanvasArea
        area = CanvasArea()
        overlay = np.zeros((100, 100, 3), dtype=np.uint8)
        area.set_overlay_image(overlay)

    def test_set_tool(self, qapp):
        from gui.panels.canvas_area import CanvasArea
        area = CanvasArea()
        area.set_active_tool("draw")


class TestFrameNavigator:
    def test_creation(self, qapp):
        from gui.panels.frame_navigator import FrameNavigator
        nav = FrameNavigator()
        assert nav is not None

    def test_set_total_frames(self, qapp):
        from gui.panels.frame_navigator import FrameNavigator
        nav = FrameNavigator()
        nav.set_total_frames(120)
        start, end = nav.get_frame_range()
        assert end == 120

    def test_frame_navigation(self, qapp):
        from gui.panels.frame_navigator import FrameNavigator
        nav = FrameNavigator()
        nav.set_total_frames(100)
        nav.set_current_frame(50)

    def test_frame_range(self, qapp):
        from gui.panels.frame_navigator import FrameNavigator
        nav = FrameNavigator()
        nav.set_total_frames(200)
        start, end = nav.get_frame_range()
        assert start == 1
        assert end == 200


class TestStatusBar:
    def test_creation(self, qapp):
        from gui.panels.status_bar import StatusBar
        bar = StatusBar()
        assert bar is not None

    def test_set_status(self, qapp):
        from gui.panels.status_bar import StatusBar
        bar = StatusBar()
        bar.set_status("Processing", "processing")
        bar.set_status("Ready", "ready")
        bar.set_status("Error", "error")

    def test_set_device(self, qapp):
        from gui.panels.status_bar import StatusBar
        bar = StatusBar()
        bar.set_device_info("CUDA: NVIDIA RTX 4090")

    def test_set_vram(self, qapp):
        from gui.panels.status_bar import StatusBar
        bar = StatusBar()
        bar.set_vram_usage(2.4, 24.0)

    def test_timer(self, qapp):
        from gui.panels.status_bar import StatusBar
        bar = StatusBar()
        bar.start_timer()
        bar.stop_timer()
        bar.reset_timer()


class TestErrorDialog:
    def test_creation(self, qapp):
        from gui.dialogs.error_dialog import ErrorDialog
        dialog = ErrorDialog("Test Error", "Something went wrong", "Details here")
        assert dialog is not None

    def test_creation_no_details(self, qapp):
        from gui.dialogs.error_dialog import ErrorDialog
        dialog = ErrorDialog("Test Error", "Something went wrong")
        assert dialog is not None


class TestMainWindow:
    def test_creation(self, qapp):
        from gui.main_window import MainWindow
        window = MainWindow()
        assert window is not None
        assert window.windowTitle() == "DIC Mask Generator"

    def test_panels_accessible(self, qapp):
        from gui.main_window import MainWindow
        window = MainWindow()
        assert window.sidebar is not None
        assert window.toolbar is not None
        assert window.canvas_area is not None
        assert window.frame_navigator is not None
        assert window.status_bar_widget is not None
