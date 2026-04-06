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
import time

import numpy as np
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gui.icons import get_icon
from gui.panels.canvas_area import CanvasArea
from gui.panels.filmstrip import Filmstrip
from gui.panels.frame_navigator import FrameNavigator
from gui.panels.sidebar import Sidebar
from gui.panels.status_bar import StatusBar
from gui.panels.toolbar import Toolbar
from gui.theme import Colors, Fonts

logger = logging.getLogger("sam2studio.main_window")


class MainWindow(QMainWindow):
    """Main application window for DIC Mask Generator.

    When instantiated without controllers (e.g. in tests), builds the
    layout only. When controllers are provided, wires all signals.
    """

    def __init__(
        self,
        app_state=None,
        annotation_controller=None,
        processing_controller=None,
        smoothing_controller=None,
        shape_controller=None,
        preview_controller=None,
    ):
        super().__init__()

        self._state = app_state
        self._annotation = annotation_controller
        self._processing = processing_controller
        self._smoothing = smoothing_controller
        self._shape_ctrl = shape_controller
        self._preview_ctrl = preview_controller

        # Previous tool before entering shape draw mode (to restore on cancel)
        self._tool_before_shape: str = "select"

        # Active mask directory for display (relative to output_dir)
        # Changes when smoothing outputs to a separate directory
        self._active_mask_subdir: str = "masks"

        # Track whether spatial smoothing was run in this session
        self._spatial_ran_this_session: bool = False

        self.setWindowTitle("DIC Mask Generator")
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

        self._filmstrip = Filmstrip()
        right_layout.addWidget(self._filmstrip)

        self._frame_navigator = FrameNavigator()
        right_layout.addWidget(self._frame_navigator)

        content_layout.addLayout(right_layout, 1)

        root_layout.addLayout(content_layout, 1)

        # Status bar
        self._status_bar = StatusBar()
        root_layout.addWidget(self._status_bar)

        # Menu bar
        self._create_menu_bar()

        # Wire signals if controllers are provided
        if self._state is not None:
            self._connect_signals()
            self._register_shortcuts()
            self._update_ui_for_state(self._state.state.value)

        # Start maximized
        self.showMaximized()

        # Restore last-used paths from QSettings
        self._restore_settings()

    # --- Window event overrides ---

    def closeEvent(self, event) -> None:
        """Save settings and warn user if processing is still running."""
        try:
            self._save_settings()
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
        except RuntimeError:
            # Qt objects may be partially destroyed during app shutdown
            pass
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

    # --- Menu Bar ---

    def _create_menu_bar(self) -> None:
        """Build the application menu bar."""
        menubar = self.menuBar()
        menubar.setStyleSheet(
            f"QMenuBar {{ background: {Colors.BG_DARK}; "
            f"color: {Colors.TEXT_SECONDARY}; "
            f"font-size: 12px; padding: 2px 0; }}"
            f"QMenuBar::item:selected {{ background: {Colors.BG_LIGHT}; }}"
            f"QMenu {{ background: {Colors.BG_MEDIUM}; "
            f"color: {Colors.TEXT_PRIMARY}; border: 1px solid {Colors.BORDER}; }}"
            f"QMenu::item:selected {{ background: {Colors.PRIMARY}; }}"
        )

        # --- File menu ---
        file_menu = menubar.addMenu("&File")

        new_project = QAction("New Project", self)
        new_project.setShortcut(QKeySequence("Ctrl+N"))
        new_project.triggered.connect(self._new_project)
        file_menu.addAction(new_project)

        file_menu.addSeparator()

        open_input = QAction("Open Input Directory...", self)
        open_input.setShortcut(QKeySequence("Ctrl+Shift+O"))
        open_input.triggered.connect(self._menu_open_input)
        file_menu.addAction(open_input)

        open_output = QAction("Set Output Directory...", self)
        open_output.triggered.connect(self._menu_open_output)
        file_menu.addAction(open_output)

        file_menu.addSeparator()

        self._recent_menu = QMenu("Recent Projects", self)
        file_menu.addMenu(self._recent_menu)
        self._recent_menu.aboutToShow.connect(self._build_recent_menu)

        file_menu.addSeparator()

        save_project = QAction("Save Project...", self)
        save_project.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_project.triggered.connect(self._on_save_project)
        file_menu.addAction(save_project)

        load_project = QAction("Open Project...", self)
        load_project.setShortcut(QKeySequence("Ctrl+Shift+P"))
        load_project.triggered.connect(self._on_load_project)
        file_menu.addAction(load_project)

        file_menu.addSeparator()

        save_config = QAction("Save Config...", self)
        save_config.setShortcut(QKeySequence("Ctrl+S"))
        save_config.triggered.connect(self._on_save_config)
        file_menu.addAction(save_config)

        load_config = QAction("Load Config...", self)
        load_config.setShortcut(QKeySequence("Ctrl+O"))
        load_config.triggered.connect(self._on_load_config)
        file_menu.addAction(load_config)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # --- Edit menu ---
        edit_menu = menubar.addMenu("&Edit")

        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        undo_action.triggered.connect(self._shortcut_undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut(QKeySequence("Ctrl+Y"))
        redo_action.triggered.connect(self._shortcut_redo)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        clear_action = QAction("Clear All Points", self)
        clear_action.triggered.connect(
            lambda: self._annotation.clear_points() if self._annotation else None
        )
        edit_menu.addAction(clear_action)

        edit_menu.addSeparator()

        goto_frame = QAction("Go to Frame...", self)
        goto_frame.setShortcut(QKeySequence("Ctrl+G"))
        goto_frame.triggered.connect(self._show_jump_to_frame)
        edit_menu.addAction(goto_frame)

        # --- View menu ---
        view_menu = menubar.addMenu("&View")

        reset_zoom = QAction("Reset Zoom", self)
        reset_zoom.setShortcut(QKeySequence("Ctrl+0"))
        reset_zoom.triggered.connect(self._canvas_area.reset_zoom)
        view_menu.addAction(reset_zoom)

        fit_window = QAction("Fit to Window", self)
        fit_window.setShortcut(QKeySequence("Ctrl+1"))
        fit_window.triggered.connect(self._canvas_area.fit_in_view)
        view_menu.addAction(fit_window)

        view_menu.addSeparator()

        overlay_settings = QAction("Overlay Settings...", self)
        overlay_settings.triggered.connect(self._show_overlay_settings)
        view_menu.addAction(overlay_settings)

        # --- Tools menu ---
        tools_menu = menubar.addMenu("&Tools")

        export_contour_png = QAction("Export Contours as PNG...", self)
        export_contour_png.triggered.connect(
            lambda: self._export_contours("PNG")
        )
        tools_menu.addAction(export_contour_png)

        export_contour_svg = QAction("Export Contours as SVG...", self)
        export_contour_svg.triggered.connect(
            lambda: self._export_contours("SVG")
        )
        tools_menu.addAction(export_contour_svg)

        tools_menu.addSeparator()

        batch_action = QAction("Batch Processing...", self)
        batch_action.triggered.connect(self._show_batch_dialog)
        tools_menu.addAction(batch_action)

        # --- Help menu ---
        help_menu = menubar.addMenu("&Help")

        shortcuts_action = QAction("Keyboard Shortcuts", self)
        shortcuts_action.setShortcut(QKeySequence("F1"))
        shortcuts_action.triggered.connect(self._show_shortcuts_dialog)
        help_menu.addAction(shortcuts_action)

        help_menu.addSeparator()

        about_action = QAction("About DIC Mask Generator", self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)

    def _new_project(self) -> None:
        """Reset all state for a fresh project."""
        if self._state is None:
            return
        # Confirm if there are existing annotations
        if self._state.points:
            reply = QMessageBox.question(
                self, "New Project",
                "Current annotations will be lost. Continue?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Stop any running processing
        if self._processing and self._processing.is_running:
            self._processing.stop_processing()

        # Reset ALL state including paths (bypass setters to avoid
        # triggering image discovery or error messages on empty strings)
        self._state.set_points([], [])
        self._state.set_image_files([])
        self._state._input_dir = ""
        self._state._output_dir = ""
        self._state.clear_marked_frames()
        self._state.clear_display_images()
        self._active_mask_subdir = "masks"
        self._state.set_state(self._state.State.INIT)

        # Clear UI
        self._sidebar.set_input_path("")
        self._sidebar.set_output_path("")
        self._sidebar.switch_to_processing()
        self._canvas_area.clear_all()
        self._filmstrip.set_image_files([])
        if self._shape_ctrl:
            self._shape_ctrl.clear()
            self._sidebar.clear_shape_entries()
        if self._annotation:
            self._annotation._undo_stack.clear()
            self._annotation._redo_stack.clear()
            self._annotation._emit_state()

        self._status_bar.set_status("New project — select input directory", "ready")

    def _menu_open_input(self) -> None:
        """Open input directory via file dialog."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Input Directory",
        )
        if path:
            self._sidebar.set_input_path(path)
            self._on_input_dir_changed(path)

    def _menu_open_output(self) -> None:
        """Set output directory via file dialog."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory",
        )
        if path:
            self._sidebar.set_output_path(path)
            if self._state:
                self._state.set_output_dir(path)

    # --- Settings persistence (QSettings) ---

    _MAX_RECENT = 8
    _SETTINGS_ORG = "SAM2Studio"
    _SETTINGS_APP = "SAM2Studio"

    def _get_settings(self) -> QSettings:
        return QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)

    def _save_settings(self) -> None:
        """Persist input/output paths and recent projects."""
        settings = self._get_settings()
        if self._state:
            if self._state.input_dir:
                settings.setValue("paths/last_input", self._state.input_dir)
            if self._state.output_dir:
                settings.setValue("paths/last_output", self._state.output_dir)
            # Add to recent projects list
            if self._state.input_dir:
                recent = settings.value("paths/recent_projects", [], list)
                entry = self._state.input_dir
                if entry in recent:
                    recent.remove(entry)
                recent.insert(0, entry)
                settings.setValue(
                    "paths/recent_projects", recent[: self._MAX_RECENT],
                )

    def _restore_settings(self) -> None:
        """Restore path display from previous session.

        Only populates the sidebar path fields for convenience —
        does NOT load images or change application state.  The user
        must explicitly click Browse or press Enter to load.
        """
        if self._state is None:
            return
        settings = self._get_settings()
        last_input = settings.value("paths/last_input", "")
        last_output = settings.value("paths/last_output", "")
        if last_input and os.path.isdir(last_input):
            self._sidebar.set_input_path(last_input, emit_signal=False)
        if last_output:
            self._sidebar.set_output_path(last_output, emit_signal=False)

    def _build_recent_menu(self) -> None:
        """Rebuild the 'Recent Projects' submenu."""
        if not hasattr(self, "_recent_menu"):
            return
        self._recent_menu.clear()
        settings = self._get_settings()
        recent = settings.value("paths/recent_projects", [], list)
        if not recent:
            no_items = QAction("(no recent projects)", self)
            no_items.setEnabled(False)
            self._recent_menu.addAction(no_items)
            return
        for path in recent:
            action = QAction(path, self)
            action.triggered.connect(
                lambda checked, p=path: self._open_recent_project(p),
            )
            self._recent_menu.addAction(action)
        self._recent_menu.addSeparator()
        clear = QAction("Clear Recent", self)
        clear.triggered.connect(self._clear_recent_projects)
        self._recent_menu.addAction(clear)

    def _open_recent_project(self, path: str) -> None:
        """Open a recent project by path."""
        if not os.path.isdir(path):
            QMessageBox.warning(
                self, "Not Found",
                f"Directory no longer exists:\n{path}",
            )
            return
        self._sidebar.set_input_path(path)
        self._on_input_dir_changed(path)

    def _clear_recent_projects(self) -> None:
        """Clear the recent projects list."""
        settings = self._get_settings()
        settings.setValue("paths/recent_projects", [])
        self._build_recent_menu()

    def _show_shortcuts_dialog(self) -> None:
        """Show keyboard shortcuts reference dialog."""
        shortcuts = [
            ("Tool Selection", [
                ("V", "Select tool"),
                ("D", "Draw tool"),
                ("E", "Erase tool"),
                ("Space", "Toggle foreground / background"),
            ]),
            ("Editing", [
                ("Ctrl+Z", "Undo"),
                ("Ctrl+Y / Ctrl+Shift+Z", "Redo"),
                ("Ctrl+Click", "Toggle point selection"),
                ("Delete", "Delete selected points"),
            ]),
            ("Navigation", [
                ("Left / Right Arrow", "Previous / Next frame"),
                ("Home / End", "First / Last frame"),
                ("Ctrl+G", "Jump to frame"),
                ("M", "Toggle bookmark on current frame"),
            ]),
            ("File", [
                ("Ctrl+Shift+S", "Save project"),
                ("Ctrl+Shift+P", "Open project"),
                ("Ctrl+S", "Save annotation config"),
                ("Ctrl+O", "Load annotation config"),
                ("Ctrl+Shift+O", "Open input directory"),
                ("Ctrl+Q", "Quit"),
            ]),
            ("Processing", [
                ("Ctrl+Enter", "Start processing"),
                ("Escape", "Stop / Cancel"),
            ]),
            ("View", [
                ("Ctrl+0", "Reset zoom"),
                ("Ctrl+1", "Fit to window"),
                ("Scroll Wheel", "Zoom in / out"),
                ("Middle-click drag", "Pan"),
            ]),
        ]

        dialog = QDialog(self)
        dialog.setWindowTitle("Keyboard Shortcuts")
        dialog.setMinimumSize(420, 500)
        dialog.setStyleSheet(
            f"QDialog {{ background: {Colors.BG_MEDIUM}; "
            f"color: {Colors.TEXT_PRIMARY}; }}"
        )

        layout = QVBoxLayout(dialog)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        content = QWidget()
        content_layout = QVBoxLayout(content)

        for section_name, keys in shortcuts:
            header = QLabel(f"<b>{section_name}</b>")
            header.setStyleSheet(
                f"color: {Colors.PRIMARY}; font-size: 14px; "
                f"margin-top: 12px; background: transparent;"
            )
            content_layout.addWidget(header)
            for key, desc in keys:
                row = QLabel(
                    f"<span style='color: {Colors.TEXT_PRIMARY}; "
                    f"font-family: monospace; background: {Colors.BG_DARK}; "
                    f"padding: 2px 6px; border-radius: 3px;'>"
                    f"{key}</span>  {desc}"
                )
                row.setStyleSheet(
                    f"color: {Colors.TEXT_SECONDARY}; font-size: 13px; "
                    f"padding: 3px 0; background: transparent;"
                )
                content_layout.addWidget(row)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btn_box.accepted.connect(dialog.accept)
        layout.addWidget(btn_box)

        dialog.exec()

    def _show_about_dialog(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About DIC Mask Generator",
            "<h3>DIC Mask Generator v2.0</h3>"
            "<p>Mask Generator for DIC & ROI Recognition</p>"
            "<p>Built with PyQt6 + Meta SAM2</p>"
            "<p>Powered by Segment Anything Model 2</p>",
        )

    def _show_overlay_settings(self) -> None:
        """Show overlay transparency and color settings dialog."""
        from PyQt6.QtWidgets import QColorDialog, QSlider, QSpinBox

        if self._state is None:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Overlay Settings")
        dlg.setMinimumWidth(320)
        layout = QVBoxLayout(dlg)

        # Opacity slider
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Opacity:"))
        opacity_slider = QSlider(Qt.Orientation.Horizontal)
        opacity_slider.setRange(0, 100)
        opacity_slider.setValue(int(self._state.overlay_alpha * 100))
        opacity_label = QLabel(f"{int(self._state.overlay_alpha * 100)}%")
        opacity_slider.valueChanged.connect(
            lambda v: opacity_label.setText(f"{v}%"),
        )
        opacity_row.addWidget(opacity_slider)
        opacity_row.addWidget(opacity_label)
        layout.addLayout(opacity_row)

        # Color picker button
        current_color = self._state.overlay_color
        color_btn = QPushButton(
            f"Color: ({current_color[0]}, {current_color[1]}, {current_color[2]})"
        )
        chosen_color = list(current_color)

        from PyQt6.QtWidgets import QPushButton as _QPB

        def pick_color():
            from PyQt6.QtGui import QColor
            c = QColorDialog.getColor(
                QColor(*current_color), self, "Overlay Color",
            )
            if c.isValid():
                chosen_color[:] = [c.red(), c.green(), c.blue()]
                color_btn.setText(
                    f"Color: ({c.red()}, {c.green()}, {c.blue()})",
                )

        color_btn.clicked.connect(pick_color)
        layout.addWidget(color_btn)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
        )
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        layout.addWidget(btn_box)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._state.set_overlay_alpha(opacity_slider.value() / 100.0)
            self._state.set_overlay_color(tuple(chosen_color))
            # Refresh current frame to apply new settings
            self._load_current_frame()

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
        self._sidebar.mask_format_changed.connect(
            state.set_mask_output_format,
        )
        self._sidebar.threshold_changed.connect(state.set_threshold)

        self._sidebar.preprocessing_preview_requested.connect(
            self._on_preview_preprocessing
        )
        self._sidebar.save_preprocessed_requested.connect(
            self._on_save_preprocessed
        )
        self._sidebar.shape_draw_requested.connect(
            self._on_shape_draw_requested
        )
        self._sidebar.shape_removed.connect(self._on_shape_removed)
        self._sidebar.shape_selected.connect(
            self._canvas_area.highlight_shape
        )

        # Shape controller: refresh preview whenever shapes change
        if self._shape_ctrl is not None:
            self._shape_ctrl.shapes_changed.connect(
                self._refresh_preview_with_shapes
            )

        if smoothing is not None:
            self._sidebar.spatial_smooth_requested.connect(
                self._on_spatial_smooth
            )
            self._sidebar.spatial_preview_requested.connect(
                self._on_spatial_preview
            )
            self._sidebar.temporal_smooth_requested.connect(
                self._on_temporal_smooth
            )
            self._sidebar.refresh_stats_requested.connect(
                self._on_refresh_stats
            )
            self._sidebar.mask_view_changed.connect(
                self._on_mask_view_changed
            )
            self._sidebar.panel_switched.connect(
                self._on_sidebar_panel_switched
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

        self._canvas_area.delete_selected_requested.connect(
            self._on_delete_selected_points
        )

        self._canvas_area.load_images_requested.connect(
            self._on_load_images_requested
        )
        self._canvas_area.shape_confirmed.connect(self._on_shape_confirmed)
        self._canvas_area.shape_drawing_cancelled.connect(
            self._on_shape_drawing_cancelled
        )

        # --- Frame navigator signals ---
        self._frame_navigator.frame_changed.connect(self._on_frame_changed)
        self._frame_navigator.start_frame_changed.connect(
            self._on_start_frame_changed
        )
        self._frame_navigator.end_frame_changed.connect(
            self._on_end_frame_changed
        )
        self._frame_navigator.mark_frame_toggled.connect(
            self._on_toggle_mark_frame
        )

        # --- Filmstrip signals ---
        self._filmstrip.frame_selected.connect(self._on_frame_changed)
        self._frame_navigator.jump_to_prev_mark.connect(
            self._on_jump_to_prev_mark
        )
        self._frame_navigator.jump_to_next_mark.connect(
            self._on_jump_to_next_mark
        )

        # --- AppState signals ---
        state.state_changed.connect(self._update_ui_for_state)
        state.points_changed.connect(self._canvas_area.set_annotation_points)
        state.current_images_changed.connect(self._on_current_images_changed)
        state.image_files_changed.connect(self._on_image_files_changed)
        state.current_frame_changed.connect(
            self._frame_navigator.set_current_frame
        )
        state.current_frame_changed.connect(
            self._filmstrip.set_current_frame
        )
        state.frame_range_changed.connect(self._on_frame_range_changed)
        state.device_changed.connect(
            lambda dev: self._status_bar.set_device_info(dev)
        )
        state.vram_updated.connect(self._status_bar.set_vram_usage)
        state.status_message.connect(self._status_bar.set_status)
        state.marked_frames_changed.connect(
            self._frame_navigator.update_mark_state
        )
        state.marked_frames_changed.connect(
            self._filmstrip.update_mark_indicators
        )

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

        # Ctrl+Shift+Z as alternative redo (Ctrl+Z and Ctrl+Y are
        # already registered via menu bar QActions — duplicating them
        # here would create ambiguous shortcuts and neither would fire).
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self).activated.connect(
            self._shortcut_redo
        )

        # Frame navigation (no menu equivalents)
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

        # Processing (no menu equivalents)
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(
            self._on_start_processing
        )
        QShortcut(QKeySequence("Escape"), self).activated.connect(
            self._shortcut_escape
        )

        # Mark frame
        QShortcut(QKeySequence("M"), self).activated.connect(
            lambda: self._on_toggle_mark_frame(self._state.current_frame)
        )

    def _activate_tool(self, tool: str) -> None:
        """Activate a tool via shortcut: update toolbar and canvas.

        Also transfers keyboard focus to the canvas so the very first
        click registers immediately (avoids the "first click lost to
        focus transfer" issue common with QGraphicsView).
        """
        self._toolbar.set_tool(tool)
        self._canvas_area.set_active_tool(tool)
        # Ensure the interactive canvas view has focus so mouse events
        # are processed without needing a focus-acquiring click first.
        self._canvas_area._original._view.setFocus()

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

    def _show_jump_to_frame(self) -> None:
        """Show a dialog to jump to a specific frame (Ctrl+G)."""
        if self._state is None or not self._state.image_files:
            return

        from PyQt6.QtWidgets import QInputDialog

        total = len(self._state.image_files)
        frame, ok = QInputDialog.getInt(
            self, "Go to Frame",
            f"Frame number (1 - {total}):",
            self._state.current_frame, 1, total,
        )
        if ok:
            self._state.set_current_frame(frame)
            self._frame_navigator.set_current_frame(frame)

    def _shortcut_escape(self) -> None:
        """Handle Escape: cancel shape drawing, stop processing, or cancel correction."""
        # Shape drawing takes priority
        if self._canvas_area.is_shape_drawing:
            self._canvas_area.exit_shape_draw_mode()
            return

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

        # Check for existing masks in output directory
        skip_existing = False
        output_dir = self._state.output_dir
        mask_dir = os.path.join(output_dir, "masks")
        if os.path.isdir(mask_dir):
            existing = [
                f for f in os.listdir(mask_dir)
                if f.lower().endswith((".tiff", ".tif", ".png"))
            ]
            if existing:
                action = self._ask_overwrite_action(len(existing), mask_dir)
                if action == "cancel":
                    return
                if action == "new_subfolder":
                    from datetime import datetime
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_dir = os.path.join(output_dir, f"run_{ts}")
                    os.makedirs(new_dir, exist_ok=True)
                    self._state.set_output_dir(new_dir)
                    self._sidebar.set_output_path(new_dir)
                elif action == "resume":
                    skip_existing = True
                # action == "overwrite" → proceed with existing dir

        # Capture current preprocessing config from sidebar (with shapes)
        preprocessing_config = self._get_config_with_shapes()
        self._state.set_preprocessing_config(preprocessing_config)

        self._status_bar.reset_timer()
        self._processing_start_time = time.monotonic()
        self._state.set_state(self._state.State.PROCESSING)
        self._processing.start_processing(skip_existing=skip_existing)

    def _on_stop_processing(self) -> None:
        """Stop the current processing operation."""
        if self._processing is not None and self._processing.is_running:
            self._processing.stop_processing()
            self._status_bar.set_status("Stopping...", "processing")
        if self._smoothing is not None and self._smoothing.is_running:
            self._smoothing.stop()

    def _on_add_correction(self) -> None:
        """Enter correction state for mid-sequence adjustments."""
        if self._state is None:
            return
        # Block if SAM2 model is not loaded
        if self._processing is not None:
            from core.mask_generator import MaskGenerator
            mg = self._processing._mask_generator
            if not mg.is_initialized:
                self._show_error(
                    "Cannot Enter Correction",
                    "SAM2 model is not loaded.\n"
                    "Run full processing first to initialize the model.",
                )
                return
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
        self._processing_start_time = time.monotonic()
        self._state.set_state(self._state.State.PROCESSING)
        self._processing.start_correction(
            frame_idx=frame_idx,
            points=list(self._state.points),
            labels=list(self._state.labels),
        )

    def _on_save_project(self) -> None:
        """Save full project state to a .s2proj file."""
        if self._state is None:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Project",
            os.path.join(self._state.output_dir or "", "project.s2proj"),
            "DIC Mask Generator Project (*.s2proj)",
        )
        if not filepath:
            return

        from core.project import save_project
        try:
            save_project(filepath, self._state)
            self._state.status_message.emit(
                f"Project saved to {filepath}", "ready"
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _on_load_project(self) -> None:
        """Load a project from a .s2proj file."""
        if self._state is None:
            return

        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "",
            "DIC Mask Generator Project (*.s2proj)",
        )
        if not filepath:
            return

        from core.project import load_project, apply_project_to_state
        try:
            project = load_project(filepath)
            # Reset state first
            self._new_project()
            apply_project_to_state(project, self._state)

            # Update sidebar with restored paths
            if self._state.input_dir:
                self._sidebar.set_input_path(self._state.input_dir)
            if self._state.output_dir:
                self._sidebar.set_output_path(self._state.output_dir)

            # Load the first frame
            if self._state.image_files:
                self._load_current_frame()

            self._state.status_message.emit(
                f"Project loaded from {filepath}", "ready"
            )
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

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
                shapes = (
                    self._shape_ctrl.overlays if self._shape_ctrl else None
                )
                self._annotation.save_config(filepath, shapes=shapes)
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
                saved_shapes = self._annotation.load_config(filepath)
                # Restore shapes into ShapeController
                if self._shape_ctrl and saved_shapes:
                    self._shape_ctrl.clear()
                    for s in saved_shapes:
                        self._shape_ctrl.add_shape(
                            s["mode"], s["shape_type"], s["points"],
                        )
                    self._sidebar.rebuild_shape_list(
                        self._shape_ctrl.overlays,
                    )
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
        # Update bookmark visual for new frame
        self._frame_navigator.update_mark_state(self._state.marked_frames)

    def _on_start_frame_changed(self, start: int) -> None:
        """Handle start frame spinbox change."""
        _, end = self._frame_navigator.get_frame_range()
        self._state.set_frame_range(start, end)

    def _on_end_frame_changed(self, end: int) -> None:
        """Handle end frame spinbox change."""
        start, _ = self._frame_navigator.get_frame_range()
        self._state.set_frame_range(start, end)

    def _on_delete_selected_points(self) -> None:
        """Delete all selected annotation points at once."""
        if self._annotation is None:
            return
        selected = self._canvas_area.get_selected_point_indices()
        if not selected:
            return
        # Remove in reverse order to preserve indices
        for idx in sorted(selected, reverse=True):
            self._annotation.remove_point(idx)
        self._canvas_area.clear_point_selection()

    def _on_toggle_mark_frame(self, frame: int) -> None:
        """Toggle bookmark on the given frame."""
        self._state.toggle_marked_frame(frame)

    def _on_jump_to_prev_mark(self) -> None:
        """Jump to the previous marked frame."""
        prev_f = self._state.prev_marked_frame(self._state.current_frame)
        if prev_f is not None:
            self._state.set_current_frame(prev_f)
            self._load_current_frame()
        else:
            self._state.status_message.emit("No previous marked frame", "warning")

    def _on_jump_to_next_mark(self) -> None:
        """Jump to the next marked frame."""
        next_f = self._state.next_marked_frame(self._state.current_frame)
        if next_f is not None:
            self._state.set_current_frame(next_f)
            self._load_current_frame()
        else:
            self._state.status_message.emit("No next marked frame", "warning")

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
        """Update frame navigator and filmstrip when image list changes."""
        self._frame_navigator.set_total_frames(len(files))
        self._filmstrip.set_image_files(files)

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
        """Update status bar with processing progress and ETA."""
        self._status_bar.set_status(message, "processing")

        # Calculate ETA based on elapsed time and progress
        eta_text = ""
        start_time = getattr(self, "_processing_start_time", None)
        if start_time is not None and current > 0 and total > 0:
            elapsed = time.monotonic() - start_time
            rate = elapsed / current
            remaining = rate * (total - current)
            if remaining < 60:
                eta_text = f"~{int(remaining)}s remaining"
            elif remaining < 3600:
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                eta_text = f"~{mins}m {secs}s remaining"
            else:
                hrs = int(remaining // 3600)
                mins = int((remaining % 3600) // 60)
                eta_text = f"~{hrs}h {mins}m remaining"

        self._status_bar.set_processing_progress(current, total, eta_text)

    def _on_frame_processed(self, frame_idx: int, mask) -> None:
        """Update all three panels during propagation.

        Loads display-sized original from the directory SAM2 used (preprocessed
        if available, otherwise converted). Creates overlay at display resolution.
        """
        if self._state is None:
            return

        import os
        import cv2
        from core.image_processing import load_image_as_rgb, create_overlay

        output_dir = self._state.output_dir
        fmt = "jpeg" if "jpeg" in self._state.intermediate_format.lower() else "png"
        ext = ".jpg" if fmt == "jpeg" else ".png"
        filename = f"{frame_idx:06d}{ext}"

        # Prefer preprocessed (what SAM2 actually sees), fall back to converted
        preprocessed_path = os.path.join(
            output_dir, f"preprocessed_{fmt}", filename,
        )
        converted_path = os.path.join(
            output_dir, f"converted_{fmt}", filename,
        )
        display_path = (
            preprocessed_path if os.path.exists(preprocessed_path)
            else converted_path
        )

        display_original = None
        if os.path.exists(display_path):
            display_original = load_image_as_rgb(display_path)

        if display_original is not None:
            # Upscale display image to match mask resolution (original res)
            # so scene rect stays consistent and annotation points remain valid
            mh, mw = mask.shape[:2]
            dh, dw = display_original.shape[:2]
            if (dh, dw) != (mh, mw):
                display_original = cv2.resize(
                    display_original, (mw, mh),
                    interpolation=cv2.INTER_LINEAR,
                )
            overlay = create_overlay(
                display_original, mask,
                alpha=self._state.overlay_alpha,
                color=self._state.overlay_color,
            )
            self._state.set_display_images(
                original=display_original, mask=mask, overlay=overlay,
            )
        else:
            self._state.set_display_images(mask=mask)

    def _on_processing_finished(self) -> None:
        """Handle processing completion."""
        self._state.set_state(self._state.State.REVIEWING)
        self._active_mask_subdir = "masks"  # Fresh results in masks/
        self._status_bar.set_status("Processing complete", "ready")
        self._status_bar.hide_processing_progress()
        # Reload current frame to show final results
        self._load_current_frame()
        # Auto-export overlays if enabled
        self._auto_export_overlays()
        # Save processing summary
        self._save_processing_summary()
        # Offer cleanup of intermediate files
        self._offer_cleanup()
        # Prompt user to enter post-processing mode
        self._prompt_postprocessing()

    def _prompt_postprocessing(self) -> None:
        """Ask user if they want to switch to the post-processing panel."""
        reply = QMessageBox.question(
            self,
            "Processing Complete",
            "Segmentation complete. Would you like to enter\n"
            "post-processing mode for mask smoothing?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._sidebar.switch_to_postprocessing()

    def _on_processing_error(self, error_msg: str) -> None:
        """Handle processing error."""
        was_processing = (
            self._state.state == self._state.State.PROCESSING
        )
        if was_processing and self._state.image_files:
            self._state.set_state(self._state.State.ANNOTATING)
        elif was_processing:
            self._state.set_state(self._state.State.INIT)
        self._status_bar.hide_processing_progress()
        self._show_error("Processing Error", error_msg)

    # --- Preprocessing preview handlers ---

    def _on_preview_preprocessing(self, config) -> None:
        """Delegate preprocessing preview to PreviewController."""
        if self._preview_ctrl is not None:
            config = self._preview_ctrl.get_config_with_shapes(config)
            if config.is_identity():
                self._load_current_frame()
                return
            self._preview_ctrl.apply_preview(config)

    # --- Save Preprocessed handler ---

    def _on_save_preprocessed(self, config) -> None:
        """Start standalone preprocessing save workflow."""
        if self._processing is None:
            return

        if self._shape_ctrl is not None:
            config = self._shape_ctrl.inject_shapes(config)

        if config.is_identity():
            self._show_error(
                "No Preprocessing",
                "No preprocessing operations are enabled.\n"
                "Adjust preprocessing settings before saving.",
            )
            return

        if not self._state.image_files:
            self._show_error(
                "Cannot Save",
                "No images loaded. Select an input directory first.",
            )
            return

        if not self._state.output_dir:
            self._show_error(
                "Cannot Save",
                "No output directory selected. Set one in the sidebar.",
            )
            return

        self._status_bar.reset_timer()
        self._state.set_state(self._state.State.POST_PROCESSING)
        self._processing.start_save_preprocessed(config)

    # --- Shape drawing handlers ---

    def _on_shape_draw_requested(self, mode: str, shape_type: str) -> None:
        """Enter shape drawing mode on canvas when requested from sidebar."""
        # Remember current tool to restore after drawing completes/cancels
        tool_names = {0: "select", 1: "draw", 2: "erase"}
        checked_id = self._toolbar._tool_group.checkedId()
        self._tool_before_shape = tool_names.get(checked_id, "select")
        self._canvas_area.enter_shape_draw_mode(mode, shape_type)

    def _on_shape_confirmed(
        self, mode: str, shape_type: str, points: tuple,
    ) -> None:
        """Handle a completed shape from the canvas."""
        if self._shape_ctrl is None:
            return

        index = self._shape_ctrl.add_shape(mode, shape_type, points)

        # Update sidebar shape list
        self._sidebar.add_shape_entry(mode, shape_type)

        # Add visual overlay on canvas
        self._canvas_area.add_confirmed_shape(mode, shape_type, points, index)

        # Restore previous tool
        self._activate_tool(self._tool_before_shape)

    def _on_shape_removed(self, index: int) -> None:
        """Handle shape removal from sidebar."""
        if self._shape_ctrl is None:
            return

        if not self._shape_ctrl.remove_shape(index):
            return

        # Rebuild canvas overlays (indices shifted)
        self._canvas_area.clear_confirmed_shapes()
        for i, shape in enumerate(self._shape_ctrl.overlays):
            self._canvas_area.add_confirmed_shape(
                shape.mode, shape.shape_type, shape.points, i,
            )

        # Rebuild sidebar list
        self._sidebar.rebuild_shape_list(self._shape_ctrl.overlays)

    def _on_shape_drawing_cancelled(self) -> None:
        """Restore previous tool state when shape drawing is cancelled."""
        self._activate_tool(self._tool_before_shape)

    def _get_config_with_shapes(self, config=None):
        """Build preprocessing config with current shape overlays included."""
        if config is None:
            config = self._sidebar.get_preprocessing_config()
        if self._shape_ctrl is not None:
            config = self._shape_ctrl.inject_shapes(config)
        return config

    def _refresh_preview_with_shapes(self) -> None:
        """Trigger preprocessing preview refresh with current shapes."""
        config = self._get_config_with_shapes()
        self._on_preview_preprocessing(config)

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

        if params.get("replace_originals"):
            smooth_output = masks_dir  # write back in-place
        else:
            smooth_output = os.path.join(output_dir, "mask_spatial_smoothing")

        self._spatial_ran_this_session = True
        self._status_bar.reset_timer()
        self._smoothing_start_time = time.monotonic()
        self._state.set_state(self._state.State.POST_PROCESSING)
        self._smoothing.start_spatial(
            input_dir=masks_dir,
            output_dir=smooth_output,
            num_iterations=params.get("iterations", 50),
            dt=params.get("dt", 0.1),
            kappa=params.get("kappa", 30.0),
            option=params.get("option", 1),
            gaussian_sigma=params.get("gaussian_sigma", 2.0),
        )

    def _on_spatial_preview(self, params: dict) -> None:
        """Open the interactive spatial smoothing preview dialog."""
        if self._state is None or not self._state.output_dir:
            return

        masks_dir = os.path.join(self._state.output_dir, "masks")
        if not os.path.isdir(masks_dir):
            self._show_error("Cannot Preview", "No masks directory found.")
            return

        from gui.dialogs.spatial_preview_dialog import SpatialPreviewDialog

        frame_idx = max(0, self._state.current_frame - 1)
        preset_name = self._sidebar._spatial_strength.value()

        dlg = SpatialPreviewDialog(
            masks_dir=masks_dir,
            initial_frame=frame_idx,
            initial_preset=preset_name,
            parent=self,
        )
        dlg.exec()

    def _on_temporal_smooth(self, params: dict) -> None:
        """Start temporal smoothing with parameters from the sidebar.

        Before starting, checks whether spatial smoothing has been run
        and warns the user accordingly:
        - No spatial folder at all → warn, offer to proceed on raw masks
        - Spatial folder exists from a previous session → ask whether to use it
        - Spatial ran this session → chain automatically (no warning)
        """
        if self._smoothing is None or self._state is None:
            return

        output_dir = self._state.output_dir
        if not output_dir:
            self._show_error(
                "Cannot Smooth",
                "No output directory set.",
            )
            return

        spatial_dir = os.path.join(output_dir, "mask_spatial_smoothing")
        masks_dir = os.path.join(output_dir, "masks")
        spatial_exists = (
            os.path.isdir(spatial_dir) and bool(os.listdir(spatial_dir))
        )

        if not os.path.isdir(masks_dir):
            self._show_error(
                "Cannot Smooth",
                f"No mask directory found in:\n{output_dir}\n"
                "Run processing first.",
            )
            return

        # --- Determine input source with user confirmation ---
        if self._spatial_ran_this_session and spatial_exists:
            # Best case: spatial was run this session, chain automatically
            input_dir = spatial_dir
        elif spatial_exists and not self._spatial_ran_this_session:
            # Spatial folder exists from a previous session
            reply = QMessageBox.question(
                self,
                "Use Previous Spatial Smoothing?",
                "Spatial smoothing has not been run in this session,\n"
                "but results from a previous session were found.\n\n"
                "Use previous spatial smoothing results?\n\n"
                "  Yes  \u2192  Chain from previous spatial results\n"
                "  No   \u2192  Run on raw masks (skip spatial)",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return
            input_dir = (
                spatial_dir
                if reply == QMessageBox.StandardButton.Yes
                else masks_dir
            )
        else:
            # No spatial results at all
            reply = QMessageBox.warning(
                self,
                "Spatial Smoothing Not Run",
                "Step 1 (Spatial Smoothing) has not been run yet.\n\n"
                "Recommended workflow:\n"
                "  1. Apply Spatial Smoothing first\n"
                "  2. Then apply Temporal Smoothing\n\n"
                "Proceed with temporal smoothing on raw masks anyway?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            input_dir = masks_dir

        if params.get("replace_originals"):
            smooth_output = masks_dir
        else:
            smooth_output = os.path.join(output_dir, "mask_temporal_smoothing")

        self._status_bar.reset_timer()
        self._smoothing_start_time = time.monotonic()
        self._state.set_state(self._state.State.POST_PROCESSING)
        self._smoothing.start_temporal(
            input_dir=input_dir,
            output_dir=smooth_output,
            sigma=params.get("sigma", 2.0),
            num_neighbors=params.get("neighbors", 2),
            variance_threshold=params.get("variance_threshold"),
            temporal_sigma=params.get("temporal_sigma", 0),
        )

    def _on_smoothing_progress(
        self, current: int, total: int, message: str
    ) -> None:
        """Update status bar with smoothing progress and ETA."""
        self._status_bar.set_status(message, "processing")

        # Calculate ETA based on elapsed time
        eta_text = ""
        start_time = getattr(self, "_smoothing_start_time", None)
        if start_time is not None and current > 0 and total > 0:
            elapsed = time.monotonic() - start_time
            rate = elapsed / current
            remaining = rate * (total - current)
            if remaining < 60:
                eta_text = f"~{int(remaining)}s remaining"
            elif remaining < 3600:
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                eta_text = f"~{mins}m {secs}s remaining"
            else:
                hrs = int(remaining // 3600)
                mins = int((remaining % 3600) // 60)
                eta_text = f"~{hrs}h {mins}m remaining"

        self._status_bar.set_processing_progress(current, total, eta_text)

    def _on_smoothing_finished(self, output_dir: str) -> None:
        """Handle smoothing completion.

        Switches the active mask display directory to the smoothed
        output so the user can immediately see the results.
        Also updates the Mask View selector with available options.
        """
        self._state.set_state(self._state.State.REVIEWING)

        # Determine which mask subdir to display
        if self._state.output_dir and output_dir.startswith(
            self._state.output_dir
        ):
            # Extract relative subdir (e.g. "mask_spatial_smoothing")
            self._active_mask_subdir = os.path.relpath(
                output_dir, self._state.output_dir,
            )
        else:
            # Replace-in-place mode → still "masks"
            self._active_mask_subdir = "masks"

        # Update the mask view selector in sidebar
        view_label = self._mask_subdir_to_label(self._active_mask_subdir)
        self._sidebar.add_mask_view_option(view_label)
        self._sidebar.set_mask_view(view_label)

        # Update temporal source indicator
        self._update_temporal_source_label()

        dir_name = os.path.basename(output_dir)
        self._status_bar.set_status(
            f"Smoothing complete — viewing: {dir_name}", "ready",
        )
        self._status_bar.hide_processing_progress()
        # Reload current frame to show smoothed masks
        self._load_current_frame()

    def _on_smoothing_error(self, error_msg: str) -> None:
        """Handle smoothing error."""
        self._state.set_state(self._state.State.REVIEWING)
        self._status_bar.hide_processing_progress()
        self._show_error("Smoothing Error", error_msg)

    def _on_mask_view_changed(self, subdir: str) -> None:
        """Switch display to a different mask directory."""
        self._active_mask_subdir = subdir
        self._load_current_frame()

    @staticmethod
    def _mask_subdir_to_label(subdir: str) -> str:
        """Convert mask subdir name to a human-readable label."""
        mapping = {
            "masks": "Original (masks/)",
            "mask_spatial_smoothing": "Spatial Smoothed",
            "mask_temporal_smoothing": "Temporal Smoothed",
        }
        return mapping.get(subdir, subdir)

    def _update_temporal_source_label(self) -> None:
        """Update the temporal smoothing input source label in sidebar."""
        if self._state is None or not self._state.output_dir:
            self._sidebar.update_temporal_source_label("masks/", is_chained=False)
            return
        spatial_dir = os.path.join(
            self._state.output_dir, "mask_spatial_smoothing",
        )
        if os.path.isdir(spatial_dir) and os.listdir(spatial_dir):
            self._sidebar.update_temporal_source_label(
                "mask_spatial_smoothing/ (chained)", is_chained=True,
            )
        else:
            self._sidebar.update_temporal_source_label(
                "masks/", is_chained=False,
            )

    def _on_sidebar_panel_switched(self, panel: str) -> None:
        """Refresh dynamic labels when user switches sidebar tabs."""
        if panel == "postprocessing":
            self._update_temporal_source_label()

    def _on_refresh_stats(self) -> None:
        """Compute and display mask statistics."""
        if self._state is None or not self._state.output_dir:
            self._sidebar.update_mask_statistics(
                "Area: no output directory", "Consistency: —", "Anomalies: —",
            )
            return

        import cv2
        from core.image_processing import imread_safe

        mask_dir = os.path.join(
            self._state.output_dir, self._active_mask_subdir,
        )
        if not os.path.isdir(mask_dir):
            self._sidebar.update_mask_statistics(
                "Area: no masks found", "Consistency: —", "Anomalies: —",
            )
            return

        # Collect mask files sorted by name
        mask_files = sorted([
            f for f in os.listdir(mask_dir)
            if f.lower().endswith((".tiff", ".tif", ".png"))
        ])
        if not mask_files:
            self._sidebar.update_mask_statistics(
                "Area: no masks found", "Consistency: —", "Anomalies: —",
            )
            return

        # Calculate per-frame area percentage
        areas = []
        for mf in mask_files:
            mask = imread_safe(
                os.path.join(mask_dir, mf), cv2.IMREAD_GRAYSCALE,
            )
            if mask is None:
                areas.append(0.0)
                continue
            total_pixels = mask.shape[0] * mask.shape[1]
            mask_pixels = int((mask > 127).sum())
            areas.append(100.0 * mask_pixels / total_pixels if total_pixels > 0 else 0.0)

        # Area summary
        avg_area = sum(areas) / len(areas) if areas else 0.0
        min_area = min(areas) if areas else 0.0
        max_area = max(areas) if areas else 0.0
        area_text = (
            f"Area: avg {avg_area:.1f}% | "
            f"min {min_area:.1f}% | max {max_area:.1f}% "
            f"({len(mask_files)} frames)"
        )

        # Frame-to-frame consistency (mean absolute change)
        changes = []
        for i in range(1, len(areas)):
            changes.append(abs(areas[i] - areas[i - 1]))
        avg_change = sum(changes) / len(changes) if changes else 0.0
        consistency_text = f"Consistency: avg Δ {avg_change:.2f}%/frame"

        # Anomaly detection (frames with >3x average change)
        anomaly_threshold = max(avg_change * 3.0, 1.0)
        anomalies = [
            i + 1 for i, c in enumerate(changes) if c > anomaly_threshold
        ]
        if anomalies:
            if len(anomalies) > 5:
                shown = ", ".join(str(a) for a in anomalies[:5])
                anomaly_text = (
                    f"Anomalies: {len(anomalies)} frames "
                    f"(Δ>{anomaly_threshold:.1f}%): {shown}..."
                )
            else:
                shown = ", ".join(str(a) for a in anomalies)
                anomaly_text = (
                    f"Anomalies: {len(anomalies)} frames "
                    f"(Δ>{anomaly_threshold:.1f}%): {shown}"
                )
        else:
            anomaly_text = "Anomalies: none detected"

        self._sidebar.update_mask_statistics(
            area_text, consistency_text, anomaly_text,
        )

    # --- Helper methods ---

    def _load_current_frame(self) -> None:
        """Load the current frame image from disk and update display.

        Also updates the preview cache for real-time preprocessing slider
        feedback.  If preprocessing is active, delegates to
        ``_on_preview_preprocessing`` so that frame-switch and slider-change
        share the exact same rendering path.
        """
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
                # Update preview cache for this frame
                if self._preview_ctrl is not None:
                    self._preview_ctrl.cache_frame(image)

                # Apply preprocessing through the same path as slider
                # changes so the result is always consistent.
                config = self._get_config_with_shapes()
                if not config.is_identity():
                    self._on_preview_preprocessing(config)
                else:
                    self._state.set_display_images(original=image)

                # Also try to load existing mask and overlay
                self._load_existing_mask_for_frame(frame - 1)
        except Exception as e:
            logger.exception(f"Failed to load frame {frame}")

    def _load_existing_mask_for_frame(self, frame_idx: int) -> None:
        """Try to load an existing mask for a frame and create overlay.

        Searches the active mask directory (defaults to ``masks/``,
        switches to smoothed output dir after smoothing).
        Supports both ``.tiff`` and ``.png`` extensions.
        Clears the mask/overlay panels if no mask is found.
        """
        if self._state is None or not self._state.output_dir:
            return

        mask_dir = os.path.join(
            self._state.output_dir, self._active_mask_subdir,
        )

        # Try both extensions — TIFF first, then PNG
        mask_path = None
        for ext in (".tiff", ".tif", ".png"):
            candidate = os.path.join(mask_dir, f"mask_{frame_idx:06d}{ext}")
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is None:
            # Clear stale mask/overlay so the display doesn't show
            # leftover data from a previously viewed frame or directory.
            self._canvas_area._mask.clear_image()
            self._canvas_area._overlay.clear_image()
            return

        try:
            import cv2
            from core.image_processing import create_overlay, imread_safe

            mask = imread_safe(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None and self._state.current_original is not None:
                overlay = create_overlay(
                    self._state.current_original, mask,
                    alpha=self._state.overlay_alpha,
                    color=self._state.overlay_color,
                )
                self._state.set_display_images(mask=mask, overlay=overlay)
            else:
                self._canvas_area._mask.clear_image()
                self._canvas_area._overlay.clear_image()
        except Exception as e:
            logger.warning(f"Could not load mask for frame {frame_idx}: {e}")

    def _auto_export_overlays(self) -> None:
        """Export overlay images if the sidebar checkbox is enabled."""
        if not self._sidebar.export_overlays_enabled:
            return
        if self._state is None or not self._state.output_dir:
            return

        import cv2
        from core.image_processing import (
            create_overlay, imread_safe, imwrite_safe,
        )

        mask_dir = os.path.join(self._state.output_dir, "masks")
        overlay_dir = os.path.join(self._state.output_dir, "overlays")
        if not os.path.isdir(mask_dir):
            return
        os.makedirs(overlay_dir, exist_ok=True)

        exported = 0
        for i, img_path in enumerate(self._state.image_files):
            # Find corresponding mask (try tiff then png)
            mask_path = None
            for ext in (".tiff", ".png", ".tif"):
                candidate = os.path.join(mask_dir, f"mask_{i:06d}{ext}")
                if os.path.exists(candidate):
                    mask_path = candidate
                    break
            if mask_path is None:
                continue

            original = imread_safe(img_path)
            mask = imread_safe(mask_path, cv2.IMREAD_GRAYSCALE)
            if original is None or mask is None:
                continue

            # Convert BGR to RGB if needed
            if original.ndim == 3 and original.shape[2] == 3:
                original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            else:
                original_rgb = original

            overlay = create_overlay(original_rgb, mask)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            out_path = os.path.join(overlay_dir, f"overlay_{i:06d}.png")
            imwrite_safe(out_path, overlay_bgr)
            exported += 1

        if exported > 0:
            self._status_bar.set_status(
                f"Exported {exported} overlay images", "ready",
            )
            logger.info(f"Auto-exported {exported} overlays to {overlay_dir}")

    def _save_processing_summary(self) -> None:
        """Save a processing summary JSON after processing completes."""
        if self._state is None or not self._state.output_dir:
            return

        import json
        from datetime import datetime

        output_dir = self._state.output_dir
        mask_dir = os.path.join(output_dir, "masks")
        mask_count = 0
        if os.path.isdir(mask_dir):
            mask_count = len([
                f for f in os.listdir(mask_dir)
                if f.lower().endswith((".tiff", ".tif", ".png"))
            ])

        duration = 0.0
        if hasattr(self, "_processing_start_time"):
            duration = time.monotonic() - self._processing_start_time

        summary = {
            "timestamp": datetime.now().isoformat(),
            "model": self._state.model_name,
            "device": self._state.device,
            "threshold": self._state.threshold,
            "mask_output_format": self._state.mask_output_format,
            "intermediate_format": self._state.intermediate_format,
            "input_directory": self._state.input_dir,
            "output_directory": output_dir,
            "total_input_frames": len(self._state.image_files),
            "masks_generated": mask_count,
            "frame_range": [self._state.start_frame, self._state.end_frame],
            "annotation_points": len(self._state.points),
            "processing_duration_seconds": round(duration, 2),
            "preprocessing": {
                "enabled": not self._state.preprocessing_config.is_identity(),
            },
        }

        summary_path = os.path.join(output_dir, "processing_summary.json")
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Processing summary saved to {summary_path}")
        except OSError as e:
            logger.warning(f"Failed to save processing summary: {e}")

    def _show_batch_dialog(self) -> None:
        """Show the batch multi-directory processing dialog."""
        if self._state is None:
            return

        if not self._state.points:
            QMessageBox.warning(
                self, "No Annotations",
                "Add annotation points before batch processing.",
            )
            return

        from gui.dialogs.batch_dialog import BatchProcessingDialog

        dialog = BatchProcessingDialog(
            default_output=self._state.output_dir, parent=self,
        )
        dialog.batch_start.connect(self._start_batch_processing)
        dialog.exec()

    def _start_batch_processing(self, dirs: list[tuple[str, str]]) -> None:
        """Process multiple directories sequentially.

        Args:
            dirs: List of (input_dir, output_dir) tuples.
        """
        if self._state is None or self._processing is None:
            return

        self._batch_queue = list(dirs)
        self._batch_total = len(dirs)
        self._batch_completed = 0
        self._process_next_batch()

    def _process_next_batch(self) -> None:
        """Process the next directory in the batch queue."""
        if not hasattr(self, "_batch_queue") or not self._batch_queue:
            # All done
            self._state.status_message.emit(
                f"Batch complete: {self._batch_completed}/{self._batch_total} directories",
                "ready",
            )
            return

        input_dir, output_dir = self._batch_queue.pop(0)
        self._batch_completed += 1

        self._state.status_message.emit(
            f"Batch {self._batch_completed}/{self._batch_total}: {os.path.basename(input_dir)}",
            "processing",
        )

        # Set paths and trigger processing
        self._state.set_input_dir(input_dir)
        self._state.set_output_dir(output_dir)
        self._sidebar.set_input_path(input_dir)
        self._sidebar.set_output_path(output_dir)

        # Disconnect/reconnect the batch chain
        try:
            self._processing.processing_finished.disconnect(
                self._process_next_batch
            )
        except (TypeError, RuntimeError):
            pass
        self._processing.processing_finished.connect(
            self._process_next_batch
        )

        # Start processing for this directory
        self._processing.start_processing()

    def _export_contours(self, fmt: str) -> None:
        """Export mask contours as PNG or SVG files."""
        if self._state is None or not self._state.output_dir:
            QMessageBox.warning(
                self, "No Output", "No output directory set."
            )
            return

        mask_dir = os.path.join(self._state.output_dir, "masks")
        if not os.path.isdir(mask_dir):
            QMessageBox.warning(
                self, "No Masks",
                "No masks directory found. Run processing first.",
            )
            return

        from core.contour_export import batch_export_contours

        contour_dir = os.path.join(
            self._state.output_dir, f"contours_{fmt.lower()}"
        )
        self._state.status_message.emit(
            f"Exporting contours as {fmt}...", "processing"
        )

        def on_progress(cur, total, msg):
            self._state.processing_progress.emit(cur, total, msg)

        count = batch_export_contours(
            mask_dir, contour_dir, fmt=fmt, thickness=2,
            progress_callback=on_progress,
        )
        self._state.status_message.emit(
            f"Exported {count} contour files to {contour_dir}", "ready"
        )

    def _offer_cleanup(self) -> None:
        """Offer to clean up intermediate files after processing."""
        if self._state is None or not self._state.output_dir:
            return

        import shutil

        output_dir = self._state.output_dir
        intermediate_dirs = []
        for name in ("converted_jpeg", "converted_png",
                      "preprocessed_jpeg", "preprocessed_png",
                      "preprocessed"):
            d = os.path.join(output_dir, name)
            if os.path.isdir(d):
                intermediate_dirs.append(d)

        if not intermediate_dirs:
            return

        # Calculate total size
        total_bytes = 0
        for d in intermediate_dirs:
            for root, _dirs, files in os.walk(d):
                for f in files:
                    try:
                        total_bytes += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
        size_mb = total_bytes / (1024 * 1024)

        dir_names = ", ".join(os.path.basename(d) for d in intermediate_dirs)
        reply = QMessageBox.question(
            self,
            "Clean Up Intermediate Files",
            f"Delete intermediate files ({size_mb:.1f} MB)?\n\n"
            f"Directories: {dir_names}\n\n"
            "These can be regenerated if needed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            for d in intermediate_dirs:
                try:
                    shutil.rmtree(d)
                except OSError as e:
                    logger.warning(f"Failed to remove {d}: {e}")
            self._status_bar.set_status(
                f"Cleaned up {size_mb:.1f} MB of intermediate files", "ready"
            )

    def _ask_overwrite_action(self, count: int, mask_dir: str) -> str:
        """Ask user what to do when output directory already has masks.

        Returns 'cancel', 'overwrite', 'new_subfolder', or 'resume'.
        """
        msg = QMessageBox(self)
        msg.setWindowTitle("Existing Masks Detected")
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText(
            f"The output directory already contains {count} mask file(s).\n\n"
            f"Directory: {mask_dir}"
        )
        msg.setInformativeText("How would you like to proceed?")

        overwrite_btn = msg.addButton(
            "Overwrite All", QMessageBox.ButtonRole.DestructiveRole,
        )
        resume_btn = msg.addButton(
            "Resume (keep existing)", QMessageBox.ButtonRole.ActionRole,
        )
        subfolder_btn = msg.addButton(
            "New Subfolder", QMessageBox.ButtonRole.AcceptRole,
        )
        cancel_btn = msg.addButton(QMessageBox.StandardButton.Cancel)
        msg.setDefaultButton(cancel_btn)

        msg.exec()
        clicked = msg.clickedButton()
        if clicked == overwrite_btn:
            return "overwrite"
        if clicked == resume_btn:
            return "resume"
        if clicked == subfolder_btn:
            return "new_subfolder"
        return "cancel"

    def _show_error(
        self,
        title: str,
        message: str,
        details: str = "",
    ) -> None:
        """Show an error dialog."""
        from gui.dialogs.error_dialog import ErrorDialog

        ErrorDialog.show_error(title, message, details, parent=self)
