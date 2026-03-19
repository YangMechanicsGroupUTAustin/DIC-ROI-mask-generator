"""SAM2 Studio v2.0 -- Entry Point.

Creates all controllers, configures device detection and VRAM monitoring,
then launches the main window with full signal wiring.
"""

import atexit
import sys
import os
import logging

# Set environment before any torch imports
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication

from controllers.app_state import AppState
from controllers.annotation_controller import AnnotationController
from controllers.preview_controller import PreviewController
from controllers.processing_controller import ProcessingController
from controllers.shape_controller import ShapeController
from controllers.smoothing_controller import SmoothingController
from core.mask_generator import MaskGenerator
from gui.main_window import MainWindow
from gui.theme import generate_stylesheet
from utils.device_manager import DeviceManager

logger = logging.getLogger("sam2studio")

# VRAM polling interval in milliseconds
VRAM_POLL_INTERVAL_MS = 5000


def _configure_logging() -> None:
    """Set up application-wide logging."""
    log_format = (
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _detect_and_configure_device(
    state: AppState, sidebar, status_bar
) -> None:
    """Detect available devices and configure sidebar + status bar."""
    devices = DeviceManager.detect_available_devices()
    default_device = devices[0] if devices else "CPU"

    sidebar.set_device_options(devices, default_device)

    device_str = DeviceManager.get_device_string(default_device)
    state.set_device(device_str)

    gpu_name = DeviceManager.get_gpu_name()
    status_bar.set_device_info(gpu_name)

    logger.info(
        f"Detected devices: {devices}, default: {default_device} "
        f"({gpu_name})"
    )


def _create_vram_timer(state: AppState, status_bar) -> QTimer:
    """Create a timer that polls VRAM usage every VRAM_POLL_INTERVAL_MS."""
    timer = QTimer()
    timer.setInterval(VRAM_POLL_INTERVAL_MS)

    def poll_vram():
        try:
            used, total = DeviceManager.get_vram_usage()
            status_bar.set_vram_usage(used, total)
            state.vram_updated.emit(used, total)
        except Exception:
            pass  # VRAM polling is best-effort

    timer.timeout.connect(poll_vram)
    timer.start()
    return timer


def main():
    """Application entry point."""
    _configure_logging()
    logger.info("Starting SAM2 Studio v2.0")

    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("SAM2 Studio")
    app.setApplicationVersion("2.0.0")
    app.setStyleSheet(generate_stylesheet())

    # --- Create controllers ---
    state = AppState()
    mask_generator = MaskGenerator()
    annotation_controller = AnnotationController(state)
    processing_controller = ProcessingController(state, mask_generator)
    smoothing_controller = SmoothingController()
    shape_controller = ShapeController()
    preview_controller = PreviewController(state, shape_controller)

    # Ensure GPU memory is released on exit
    atexit.register(mask_generator.cleanup)

    # --- Create main window with controllers ---
    window = MainWindow(
        app_state=state,
        annotation_controller=annotation_controller,
        processing_controller=processing_controller,
        smoothing_controller=smoothing_controller,
        shape_controller=shape_controller,
        preview_controller=preview_controller,
    )

    # --- Device detection and VRAM monitoring ---
    _detect_and_configure_device(
        state, window.sidebar, window.status_bar_widget
    )

    # Keep reference to timer so it doesn't get garbage-collected
    vram_timer = _create_vram_timer(state, window.status_bar_widget)

    # Initial VRAM reading
    try:
        used, total = DeviceManager.get_vram_usage()
        window.status_bar_widget.set_vram_usage(used, total)
    except Exception:
        pass

    window.show()

    # Show welcome dialog for first-time users
    from gui.dialogs.welcome_dialog import should_show_welcome, WelcomeDialog
    if should_show_welcome():
        welcome = WelcomeDialog(window)
        welcome.exec()

    logger.info("SAM2 Studio ready")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
