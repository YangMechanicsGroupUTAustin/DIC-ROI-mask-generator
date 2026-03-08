"""Main application window assembling all GUI panels.

Layout matches the Figma App.tsx reference:
  QVBoxLayout:
    QHBoxLayout:
      Sidebar (fixed 300px)
      QVBoxLayout:
        Toolbar (fixed 56px)
        CanvasArea (expand)
        FrameNavigator (fixed 48px)
    StatusBar (fixed 28px)

When controllers are provided, wires all signals between panels,
controllers, and AppState for full application functionality.
"""

import logging
import os
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from gui.icons import get_icon
from gui.panels.canvas_area import CanvasArea
from gui.panels.frame_navigator import FrameNavigator
from gui.panels.sidebar import Sidebar
from gui.panels.status_bar import StatusBar
from gui.panels.toolbar import Toolbar
from gui.theme import Colors

logger = logging.getLogger("sam2studio.main_window")


class MainWindow(QMainWindow):
    """Main application window for SAM2 Studio.

    When instantiated without controllers (e.g. in tests), builds the
    layout only. When controllers are provided, wires all signals.
    """

    def __init__(
        self,
        app_state=None,
        annotation_controller=None,
        processing_controller=None,
        smoothing_controller=None,
    ):
        super().__init__()

        self._state = app_state
        self._annotation = annotation_controller
        self._processing = processing_controller
        self._smoothing = smoothing_controller

        self.setWindowTitle("SAM2 Studio")
        self.setMinimumSize(1200, 700)

        # Application icon
        self.setWindowIcon(get_icon("brain", Colors.PRIMARY, 32))

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # --- Main content row: sidebar + right content ---
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Sidebar
        self._sidebar = Sidebar()
        content_layout.addWidget(self._sidebar)

        # Right column: toolbar + canvas + frame nav
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self._toolbar = Toolbar()
        right_layout.addWidget(self._toolbar)

        self._canvas_area = CanvasArea()
        right_layout.addWidget(self._canvas_area, 1)

        self._frame_navigator = FrameNavigator()
        right_layout.addWidget(self._frame_navigator)

        content_layout.addLayout(right_layout, 1)

        root_layout.addLayout(content_layout, 1)

        # Status bar
        self._status_bar = StatusBar()
        root_layout.addWidget(self._status_bar)

        # Wire signals if controllers are provided
        if self._state is not None:
            self._connect_signals()
            self._register_shortcuts()
            self._update_ui_for_state(self._state.state.value)

        # Start maximized
        self.showMaximized()

    # --- Window event overrides ---

    def closeEvent(self, event) -> None:
        """Warn user if processing is still running before closing."""
        if self._processing is not None and self._processing.is_running:
            reply = QMessageBox.question(
                self,
                "Processing Running",
                "Processing is still running. Stop and exit?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._processing.stop_processing()
                if (
                    self._smoothing is not None
                    and self._smoothing.is_running
                ):
                    self._smoothing.stop()
                event.accept()
            else:
                event.ignore()
                return
        event.accept()

    # --- Properties for external access ---

    @property
    def sidebar(self) -> Sidebar:
        """Access the sidebar panel."""
        return self._sidebar

    @property
    def toolbar(self) -> Toolbar:
        """Access the toolbar panel."""
        return self._toolbar

    @property
    def canvas_area(self) -> CanvasArea:
        """Access the canvas area panel."""
        return self._canvas_area

    @property
    def frame_navigator(self) -> FrameNavigator:
        """Access the frame navigator panel."""
        return self._frame_navigator

    @property
    def status_bar_widget(self) -> StatusBar:
        """Access the status bar widget.

        Named status_bar_widget to avoid conflict with QMainWindow.statusBar().
        """
        return self._status_bar

    # --- Signal Wiring ---

    def _connect_signals(self) -> None:
        """Wire all signals between panels, controllers, and state."""
        state = self._state
        annotation = self._annotation
        processing = self._processing
        smoothing = self._smoothing

        # --- Sidebar signals ---
        self._sidebar.input_dir_changed.connect(self._on_input_dir_changed)
        self._sidebar.output_dir_changed.connect(state.set_output_dir)
        self._sidebar.device_changed.connect(self._on_device_changed)
        self._sidebar.model_changed.connect(state.set_model_name)
        self._sidebar.format_changed.connect(state.set_intermediate_format)
        self._sidebar.threshold_changed.connect(state.set_threshold)

        if smoothing is not None:
            self._sidebar.spatial_smooth_requested.connect(
                self._on_spatial_smooth
            )
            self._sidebar.temporal_smooth_requested.connect(
                self._on_temporal_smooth
            )

        # --- Toolbar signals ---
        self._toolbar.tool_changed.connect(self._canvas_area.set_active_tool)

        if annotation is not None:
            self._toolbar.mode_changed.connect(annotation.set_point_mode)
            self._toolbar.clear_requested.connect(annotation.clear_points)
            self._toolbar.undo_requested.connect(annotation.undo)
            self._toolbar.redo_requested.connect(annotation.redo)
            self._toolbar.save_requested.connect(self._on_save_config)
            self._toolbar.load_requested.connect(self._on_load_config)
            annotation.can_undo_changed.connect(
                self._toolbar._undo_btn.setEnabled
            )
            annotation.can_redo_changed.connect(
                self._toolbar._redo_btn.setEnabled
            )

        if processing is not None:
            self._toolbar.start_processing_requested.connect(
                self._on_start_processing
            )
            self._toolbar.stop_processing_requested.connect(
                self._on_stop_processing
            )

        self._toolbar.add_correction_requested.connect(
            self._on_add_correction
        )
        self._toolbar.apply_correction_requested.connect(
            self._on_apply_correction
        )
        self._toolbar.force_reprocess_changed.connect(
            state.set_force_reprocess
        )

        # --- Canvas area signals ---
        if annotation is not None:
            self._canvas_area.point_added.connect(annotation.add_point)
            self._canvas_area.point_moved.connect(annotation.move_point)
            self._canvas_area.point_removed.connect(annotation.remove_point)

        self._canvas_area.load_images_requested.connect(
            self._on_load_images_requested
        )

        # --- Frame navigator signals ---
        self._frame_navigator.frame_changed.connect(self._on_frame_changed)
        self._frame_navigator.start_frame_changed.connect(
            self._on_start_frame_changed
        )
        self._frame_navigator.end_frame_changed.connect(
            self._on_end_frame_changed
        )

        # --- AppState signals ---
        state.state_changed.connect(self._update_ui_for_state)
        state.points_changed.connect(self._canvas_area.set_annotation_points)
        state.current_images_changed.connect(self._on_current_images_changed)
        state.image_files_changed.connect(self._on_image_files_changed)
        state.current_frame_changed.connect(
            self._frame_navigator.set_current_frame
        )
        state.frame_range_changed.connect(self._on_frame_range_changed)
        state.device_changed.connect(
            lambda dev: self._status_bar.set_device_info(dev)
        )
        state.vram_updated.connect(self._status_bar.set_vram_usage)
        state.status_message.connect(self._status_bar.set_status)

        # --- Processing controller signals ---
        if processing is not None:
            processing.progress.connect(self._on_processing_progress)
            processing.frame_processed.connect(self._on_frame_processed)
            processing.processing_finished.connect(
                self._on_processing_finished
            )
            processing.processing_error.connect(self._on_processing_error)

        # --- Smoothing controller signals ---
        if smoothing is not None:
            smoothing.progress.connect(self._on_smoothing_progress)
            smoothing.smoothing_finished.connect(self._on_smoothing_finished)
            smoothing.smoothing_error.connect(self._on_smoothing_error)

    # --- Keyboard Shortcuts ---

    def _register_shortcuts(self) -> None:
        """Register keyboard shortcuts for tools, navigation, and actions."""
        # Tool shortcuts
        QShortcut(QKeySequence("V"), self).activated.connect(
            lambda: self._activate_tool("select")
        )
        QShortcut(QKeySequence("D"), self).activated.connect(
            lambda: self._activate_tool("draw")
        )
        QShortcut(QKeySequence("E"), self).activated.connect(
            lambda: self._activate_tool("erase")
        )

        # Toggle foreground/background mode
        QShortcut(QKeySequence("Space"), self).activated.connect(
            self._toggle_point_mode
        )

        # Undo / Redo
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(
            self._shortcut_undo
        )
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(
            self._shortcut_redo
        )
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self).activated.connect(
            self._shortcut_redo
        )

        # Save / Load config
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(
            self._on_save_config
        )
        QShortcut(QKeySequence("Ctrl+O"), self).activated.connect(
            self._on_load_config
        )

        # Frame navigation
        QShortcut(QKeySequence("Left"), self).activated.connect(
            self._frame_navigator._prev_frame
        )
        QShortcut(QKeySequence("Right"), self).activated.connect(
            self._frame_navigator._next_frame
        )
        QShortcut(QKeySequence("Home"), self).activated.connect(
            self._go_to_first_frame
        )
        QShortcut(QKeySequence("End"), self).activated.connect(
            self._go_to_last_frame
        )

        # Processing
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(
            self._on_start_processing
        )
        QShortcut(QKeySequence("Escape"), self).activated.connect(
            self._shortcut_escape
        )

    def _activate_tool(self, tool: str) -> None:
        """Activate a tool via shortcut: update toolbar and canvas."""
        self._toolbar.set_tool(tool)
        self._canvas_area.set_active_tool(tool)

    def _toggle_point_mode(self) -> None:
        """Toggle between foreground and background point mode."""
        if self._annotation is not None:
            new_mode = (
                "background"
                if self._annotation.point_mode == "foreground"
                else "foreground"
            )
            self._annotation.set_point_mode(new_mode)
            self._toolbar.set_mode(new_mode)

    def _shortcut_undo(self) -> None:
        """Undo the last annotation action."""
        if self._annotation is not None:
            self._annotation.undo()

    def _shortcut_redo(self) -> None:
        """Redo the last undone annotation action."""
        if self._annotation is not None:
            self._annotation.redo()

    def _go_to_first_frame(self) -> None:
        """Navigate to the first frame."""
        nav = self._frame_navigator
        if nav._total_frames >= 1:
            nav.set_current_frame(1)
            nav.frame_changed.emit(1)

    def _go_to_last_frame(self) -> None:
        """Navigate to the last frame."""
        nav = self._frame_navigator
        if nav._total_frames >= 1:
            nav.set_current_frame(nav._total_frames)
            nav.frame_changed.emit(nav._total_frames)

    def _shortcut_escape(self) -> None:
        """Handle Escape: stop processing or cancel correction."""
        from controllers.app_state import AppState

        if self._state is None:
            return

        current = self._state.state
        if current == AppState.State.PROCESSING:
            self._on_stop_processing()
        elif current == AppState.State.POST_PROCESSING:
            self._on_stop_processing()
        elif current == AppState.State.CORRECTION:
            self._state.set_state(AppState.State.REVIEWING)

    # --- State-dependent UI enablement ---

    def _update_ui_for_state(self, state_value: str) -> None:
        """Enable/disable UI elements based on application state."""
        from controllers.app_state import AppState

        is_init = state_value == AppState.State.INIT.value
        is_annotating = state_value == AppState.State.ANNOTATING.value
        is_processing = state_value == AppState.State.PROCESSING.value
        is_reviewing = state_value == AppState.State.REVIEWING.value
        is_correction = state_value == AppState.State.CORRECTION.value
        is_post_processing = (
            state_value == AppState.State.POST_PROCESSING.value
        )

        is_busy = is_processing or is_post_processing

        # Sidebar: enabled unless processing
        self._sidebar.setEnabled(not is_busy)

        # Toolbar: tool buttons
        self._toolbar._select_btn.setEnabled(not is_busy)
        self._toolbar._draw_btn.setEnabled(not is_busy)
        self._toolbar._erase_btn.setEnabled(not is_busy)
        self._toolbar._fg_btn.setEnabled(not is_busy)
        self._toolbar._bg_btn.setEnabled(not is_busy)

        # Action buttons
        self._toolbar._clear_btn.setEnabled(
            is_annotating or is_reviewing or is_correction
        )
        self._toolbar._undo_btn.setEnabled(
            is_annotating or is_reviewing or is_correction
        )
        self._toolbar._redo_btn.setEnabled(
            is_annotating or is_reviewing or is_correction
        )
        self._toolbar._save_btn.setEnabled(not is_init and not is_busy)
        self._toolbar._load_btn.setEnabled(not is_busy)

        # Processing buttons
        self._toolbar._start_btn.setEnabled(
            is_annotating or is_reviewing
        )
        self._toolbar.set_processing(is_busy)

        # Correction buttons
        self._toolbar._add_correction_btn.setEnabled(is_reviewing)
        self._toolbar._apply_correction_btn.setEnabled(is_correction)
        self._toolbar._force_reprocess.setEnabled(not is_busy)

        # Frame navigator
        self._frame_navigator.setEnabled(not is_busy)

        # Status bar timer
        if is_busy:
            self._status_bar.start_timer()
        else:
            self._status_bar.stop_timer()

        # Status message
        status_messages = {
            AppState.State.INIT.value: ("Ready", "ready"),
            AppState.State.ANNOTATING.value: (
                "Annotating — add points to define mask", "ready"
            ),
            AppState.State.PROCESSING.value: ("Processing...", "processing"),
            AppState.State.REVIEWING.value: (
                "Review masks — add corrections if needed", "ready"
            ),
            AppState.State.CORRECTION.value: (
                "Correction mode — adjust points and apply", "ready"
            ),
            AppState.State.POST_PROCESSING.value: (
                "Post-processing...", "processing"
            ),
        }
        msg, level = status_messages.get(state_value, ("Ready", "ready"))
        self._status_bar.set_status(msg, level)

    # --- Sidebar handlers ---

    def _on_input_dir_changed(self, path: str) -> None:
        """Handle input directory change: load images and display first frame."""
        try:
            self._state.set_input_dir(path)
            if self._state.image_files:
                self._load_current_frame()
                self._state.set_state(
                    self._state.State.ANNOTATING
                )
                logger.info(
                    f"Loaded {len(self._state.image_files)} images "
                    f"from {path}"
                )
        except Exception as e:
            logger.exception("Failed to load input directory")
            self._show_error(
                "Input Directory Error",
                f"Failed to load images from: {path}",
                str(e),
            )

    def _on_device_changed(self, device_display: str) -> None:
        """Convert display device name to torch string and update state."""
        from utils.device_manager import DeviceManager

        device_str = DeviceManager.get_device_string(device_display)
        self._state.set_device(device_str)
        self._status_bar.set_device_info(
            DeviceManager.get_gpu_name()
            if device_display != "CPU"
            else "CPU"
        )

    # --- Toolbar handlers ---

    def _on_start_processing(self) -> None:
        """Start the mask generation pipeline."""
        if self._processing is None:
            return

        if not self._state.image_files:
            self._show_error(
                "Cannot Start",
                "No images loaded. Select an input directory first.",
            )
            return

        if not self._state.output_dir:
            self._show_error(
                "Cannot Start",
                "No output directory selected. Set one in the sidebar.",
            )
            return

        if not self._state.points:
            self._show_error(
                "Cannot Start",
                "No annotation points. Add at least one point.",
            )
            return

        self._status_bar.reset_timer()
        self._state.set_state(self._state.State.PROCESSING)
        self._processing.start_processing()

    def _on_stop_processing(self) -> None:
        """Stop the current processing operation."""
        if self._processing is not None and self._processing.is_running:
            self._processing.stop_processing()
        if self._smoothing is not None and self._smoothing.is_running:
            self._smoothing.stop()

    def _on_add_correction(self) -> None:
        """Enter correction state for mid-sequence adjustments."""
        if self._state is not None:
            self._state.set_state(self._state.State.CORRECTION)

    def _on_apply_correction(self) -> None:
        """Apply correction from the current frame with current points."""
        if self._processing is None:
            return

        if not self._state.points:
            self._show_error(
                "Cannot Apply Correction",
                "No annotation points. Add correction points first.",
            )
            return

        # current_frame is 1-based, correction needs 0-based
        frame_idx = self._state.current_frame - 1

        self._status_bar.reset_timer()
        self._state.set_state(self._state.State.PROCESSING)
        self._processing.start_correction(
            frame_idx=frame_idx,
            points=list(self._state.points),
            labels=list(self._state.labels),
        )

    def _on_save_config(self) -> None:
        """Save annotation config to file via QFileDialog."""
        if self._annotation is None:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Annotation Config",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if filepath:
            try:
                self._annotation.save_config(filepath)
                self._status_bar.set_status(
                    f"Config saved: {os.path.basename(filepath)}", "ready"
                )
            except Exception as e:
                logger.exception("Failed to save config")
                self._show_error(
                    "Save Error",
                    "Failed to save annotation config.",
                    str(e),
                )

    def _on_load_config(self) -> None:
        """Load annotation config from file via QFileDialog."""
        if self._annotation is None:
            return

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Annotation Config",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if filepath:
            try:
                self._annotation.load_config(filepath)
                self._status_bar.set_status(
                    f"Config loaded: {os.path.basename(filepath)}", "ready"
                )
            except Exception as e:
                logger.exception("Failed to load config")
                self._show_error(
                    "Load Error",
                    "Failed to load annotation config.",
                    str(e),
                )

    # --- Canvas area handlers ---

    def _on_load_images_requested(self) -> None:
        """Open a directory dialog to select input images."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Input Directory"
        )
        if dir_path:
            self._sidebar.set_input_path(dir_path)
            self._on_input_dir_changed(dir_path)

    # --- Frame navigator handlers ---

    def _on_frame_changed(self, frame: int) -> None:
        """Handle frame navigation: update state and load the frame image."""
        self._state.set_current_frame(frame)
        self._load_current_frame()

    def _on_start_frame_changed(self, start: int) -> None:
        """Handle start frame spinbox change."""
        _, end = self._frame_navigator.get_frame_range()
        self._state.set_frame_range(start, end)

    def _on_end_frame_changed(self, end: int) -> None:
        """Handle end frame spinbox change."""
        start, _ = self._frame_navigator.get_frame_range()
        self._state.set_frame_range(start, end)

    # --- AppState signal handlers ---

    def _on_current_images_changed(self) -> None:
        """Update all three canvas panels from AppState display images."""
        if self._state.current_original is not None:
            self._canvas_area.set_original_image(
                self._state.current_original
            )
        if self._state.current_mask is not None:
            self._canvas_area.set_mask_image(self._state.current_mask)
        if self._state.current_overlay is not None:
            self._canvas_area.set_overlay_image(self._state.current_overlay)

    def _on_image_files_changed(self, files: list) -> None:
        """Update frame navigator when image list changes."""
        self._frame_navigator.set_total_frames(len(files))

    def _on_frame_range_changed(self, start: int, end: int) -> None:
        """Sync frame range spinboxes when state changes."""
        nav = self._frame_navigator
        nav._start_spin.blockSignals(True)
        nav._end_spin.blockSignals(True)
        nav._start_spin.setValue(start)
        nav._end_spin.setValue(end)
        nav._start_spin.blockSignals(False)
        nav._end_spin.blockSignals(False)

    # --- Processing controller handlers ---

    def _on_processing_progress(
        self, current: int, total: int, message: str
    ) -> None:
        """Update status bar with processing progress."""
        self._status_bar.set_status(message, "processing")

    def _on_frame_processed(
        self, frame_idx: int, mask, overlay
    ) -> None:
        """Update display when a frame is processed."""
        # Show the most recently processed frame
        if self._state is not None:
            self._state.set_display_images(mask=mask, overlay=overlay)

    def _on_processing_finished(self) -> None:
        """Handle processing completion."""
        self._state.set_state(self._state.State.REVIEWING)
        self._status_bar.set_status("Processing complete", "ready")
        # Reload current frame to show final results
        self._load_current_frame()

    def _on_processing_error(self, error_msg: str) -> None:
        """Handle processing error."""
        was_processing = (
            self._state.state == self._state.State.PROCESSING
        )
        if was_processing and self._state.image_files:
            self._state.set_state(self._state.State.ANNOTATING)
        elif was_processing:
            self._state.set_state(self._state.State.INIT)
        self._show_error("Processing Error", error_msg)

    # --- Smoothing handlers ---

    def _on_spatial_smooth(self, params: dict) -> None:
        """Start spatial smoothing with parameters from the sidebar."""
        if self._smoothing is None or self._state is None:
            return

        output_dir = self._state.output_dir
        if not output_dir:
            self._show_error(
                "Cannot Smooth",
                "No output directory set.",
            )
            return

        masks_dir = os.path.join(output_dir, "masks")
        if not os.path.isdir(masks_dir):
            self._show_error(
                "Cannot Smooth",
                f"Masks directory not found: {masks_dir}\n"
                "Run processing first.",
            )
            return

        smooth_output = os.path.join(output_dir, "mask_spatial_smoothing")

        self._status_bar.reset_timer()
        self._state.set_state(self._state.State.POST_PROCESSING)
        self._smoothing.start_spatial(
            input_dir=masks_dir,
            output_dir=smooth_output,
            num_iterations=params.get("iterations", 50),
            dt=params.get("dt", 0.1),
            kappa=params.get("kappa", 30.0),
            option=params.get("option", 1),
        )

    def _on_temporal_smooth(self, params: dict) -> None:
        """Start temporal smoothing with parameters from the sidebar."""
        if self._smoothing is None or self._state is None:
            return

        output_dir = self._state.output_dir
        if not output_dir:
            self._show_error(
                "Cannot Smooth",
                "No output directory set.",
            )
            return

        masks_dir = os.path.join(output_dir, "masks")
        if not os.path.isdir(masks_dir):
            self._show_error(
                "Cannot Smooth",
                f"Masks directory not found: {masks_dir}\n"
                "Run processing first.",
            )
            return

        smooth_output = os.path.join(output_dir, "mask_temporal_smoothing")

        self._status_bar.reset_timer()
        self._state.set_state(self._state.State.POST_PROCESSING)
        self._smoothing.start_temporal(
            input_dir=masks_dir,
            output_dir=smooth_output,
            sigma=params.get("sigma", 2.0),
            num_neighbors=params.get("neighbors", 2),
            variance_threshold=params.get("variance_threshold"),
        )

    def _on_smoothing_progress(
        self, current: int, total: int, message: str
    ) -> None:
        """Update status bar with smoothing progress."""
        self._status_bar.set_status(message, "processing")

    def _on_smoothing_finished(self, output_dir: str) -> None:
        """Handle smoothing completion."""
        self._state.set_state(self._state.State.REVIEWING)
        self._status_bar.set_status(
            f"Smoothing complete: {os.path.basename(output_dir)}", "ready"
        )

    def _on_smoothing_error(self, error_msg: str) -> None:
        """Handle smoothing error."""
        self._state.set_state(self._state.State.REVIEWING)
        self._show_error("Smoothing Error", error_msg)

    # --- Helper methods ---

    def _load_current_frame(self) -> None:
        """Load the current frame image from disk and update display."""
        if self._state is None:
            return

        files = self._state.image_files
        frame = self._state.current_frame
        if not files or frame < 1 or frame > len(files):
            return

        try:
            from core.image_processing import load_image_as_rgb

            image_path = files[frame - 1]  # current_frame is 1-based
            image = load_image_as_rgb(image_path)
            if image is not None:
                self._state.set_display_images(original=image)

                # Also try to load existing mask and overlay
                self._load_existing_mask_for_frame(frame - 1)
        except Exception as e:
            logger.exception(f"Failed to load frame {frame}")

    def _load_existing_mask_for_frame(self, frame_idx: int) -> None:
        """Try to load an existing mask for a frame and create overlay."""
        if self._state is None or not self._state.output_dir:
            return

        mask_dir = os.path.join(self._state.output_dir, "masks")
        mask_path = os.path.join(mask_dir, f"mask_{frame_idx:06d}.tiff")

        if not os.path.exists(mask_path):
            return

        try:
            import cv2
            from core.image_processing import create_overlay

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None and self._state.current_original is not None:
                overlay = create_overlay(
                    self._state.current_original, mask
                )
                self._state.set_display_images(mask=mask, overlay=overlay)
        except Exception as e:
            logger.warning(f"Could not load mask for frame {frame_idx}: {e}")

    def _show_error(
        self,
        title: str,
        message: str,
        details: str = "",
    ) -> None:
        """Show an error dialog."""
        from gui.dialogs.error_dialog import ErrorDialog

        ErrorDialog.show_error(title, message, details, parent=self)
