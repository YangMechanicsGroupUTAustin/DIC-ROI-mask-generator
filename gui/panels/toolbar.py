"""Top toolbar with drawing tools and action buttons.

Contains tool selection, mode toggle, action buttons,
and processing controls matching the Figma toolbar reference.
"""

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from gui.icons import get_icon
from gui.theme import Colors, Fonts
from gui.widgets.tool_button import ToolButton


def _create_divider() -> QFrame:
    """Create a vertical divider line."""
    divider = QFrame()
    divider.setFixedWidth(1)
    divider.setFixedHeight(28)
    divider.setStyleSheet(f"background: {Colors.BORDER};")
    return divider


def _create_tool_group_container() -> QWidget:
    """Create a dark rounded container for tool groups."""
    container = QWidget()
    container.setStyleSheet(
        f"QWidget {{ background: {Colors.BG_DARK}; "
        f"border-radius: 10px; "
        f"border: 1px solid {Colors.BORDER_SUBTLE}; }}"
    )
    return container


class _ModeButton(QPushButton):
    """Toggle button with a colored dot indicator for foreground/background mode."""

    def __init__(
        self,
        label: str,
        dot_color: str,
        active_bg: str,
        active_text: str,
        active_border: str,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._label = label
        self._dot_color = dot_color
        self._active_bg = active_bg
        self._active_text = active_text
        self._active_border = active_border

        self.setText(label)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(30)
        self._apply_style(False)
        self.toggled.connect(self._apply_style)

    def _apply_style(self, checked: bool) -> None:
        if checked:
            self.setStyleSheet(
                f"QPushButton {{"
                f"  background: {self._active_bg};"
                f"  color: {self._active_text};"
                f"  border: 1px solid {self._active_border};"
                f"  border-radius: 8px;"
                f"  padding: 4px 10px;"
                f"  font-size: {Fonts.SIZE_MD}px;"
                f"}}"
                f"QPushButton:hover {{"
                f"  background: {self._active_bg};"
                f"}}"
            )
        else:
            self.setStyleSheet(
                f"QPushButton {{"
                f"  background: transparent;"
                f"  color: {Colors.TEXT_DIM};"
                f"  border: 1px solid transparent;"
                f"  border-radius: 8px;"
                f"  padding: 4px 10px;"
                f"  font-size: {Fonts.SIZE_MD}px;"
                f"}}"
                f"QPushButton:hover {{"
                f"  color: {Colors.TEXT_SECONDARY};"
                f"  background: rgba(255, 255, 255, 0.03);"
                f"}}"
            )


class Toolbar(QWidget):
    """Top toolbar with drawing tools and action buttons."""

    # Signals
    tool_changed = pyqtSignal(str)       # "select", "draw", "erase"
    mode_changed = pyqtSignal(str)       # "foreground", "background"
    clear_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()
    save_requested = pyqtSignal()
    load_requested = pyqtSignal()
    add_correction_requested = pyqtSignal()
    apply_correction_requested = pyqtSignal()
    start_processing_requested = pyqtSignal()
    stop_processing_requested = pyqtSignal()
    force_reprocess_changed = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.setFixedHeight(56)
        self.setStyleSheet(
            f"Toolbar {{ background: {Colors.BG_MEDIUM}; "
            f"border-bottom: 1px solid {Colors.BORDER}; }}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(4)

        # --- Tool group (Select | Draw | Erase) ---
        tool_container = _create_tool_group_container()
        tool_layout = QHBoxLayout(tool_container)
        tool_layout.setContentsMargins(4, 4, 4, 4)
        tool_layout.setSpacing(2)

        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)

        self._select_btn = ToolButton(
            "cursor", "Select", "V", checkable=True
        )
        self._select_btn.setToolTip("Select / Move Points [V]\nDrag annotation points to reposition them")
        self._draw_btn = ToolButton(
            "pencil", "Draw Points", "D", checkable=True
        )
        self._draw_btn.setToolTip("Draw Points [D]\nClick on image to add annotation points")
        self._erase_btn = ToolButton(
            "eraser", "Erase", "E", checkable=True
        )
        self._erase_btn.setToolTip("Erase Points [E]\nClick on a point to remove it")

        self._tool_group.addButton(self._select_btn, 0)
        self._tool_group.addButton(self._draw_btn, 1)
        self._tool_group.addButton(self._erase_btn, 2)
        self._select_btn.setChecked(True)

        tool_layout.addWidget(self._select_btn)
        tool_layout.addWidget(self._draw_btn)
        tool_layout.addWidget(self._erase_btn)

        layout.addWidget(tool_container)

        self._tool_group.idClicked.connect(self._on_tool_clicked)

        # Divider
        layout.addWidget(_create_divider())

        # --- Mode toggle (Foreground | Background) ---
        mode_container = _create_tool_group_container()
        mode_layout = QHBoxLayout(mode_container)
        mode_layout.setContentsMargins(4, 4, 4, 4)
        mode_layout.setSpacing(2)

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)

        self._fg_btn = _ModeButton(
            "Foreground",
            dot_color=Colors.SUCCESS,
            active_bg=Colors.SUCCESS_BG,
            active_text=Colors.SUCCESS,
            active_border=Colors.SUCCESS_BORDER,
        )
        self._fg_btn.setToolTip(
            "Foreground Mode [Space to toggle]\n"
            "Next point marks area to INCLUDE in mask"
        )
        self._bg_btn = _ModeButton(
            "Background",
            dot_color=Colors.ROSE,
            active_bg=Colors.ROSE_BG,
            active_text=Colors.ROSE,
            active_border=Colors.ROSE_BORDER,
        )
        self._bg_btn.setToolTip(
            "Background Mode [Space to toggle]\n"
            "Next point marks area to EXCLUDE from mask"
        )

        self._mode_group.addButton(self._fg_btn, 0)
        self._mode_group.addButton(self._bg_btn, 1)
        self._fg_btn.setChecked(True)

        mode_layout.addWidget(self._fg_btn)
        mode_layout.addWidget(self._bg_btn)

        layout.addWidget(mode_container)

        self._mode_group.idClicked.connect(self._on_mode_clicked)

        # Divider
        layout.addWidget(_create_divider())

        # --- Actions: Clear | Undo | Redo ---
        self._clear_btn = ToolButton("trash", "Clear", variant="danger")
        self._clear_btn.setToolTip("Clear All Points\nRemove all annotation points from the image")
        self._clear_btn.clicked.connect(self.clear_requested.emit)
        layout.addWidget(self._clear_btn)

        self._undo_btn = ToolButton("undo", "Undo", "Ctrl+Z")
        self._undo_btn.setToolTip("Undo [Ctrl+Z]\nUndo last annotation action")
        self._undo_btn.clicked.connect(self.undo_requested.emit)
        layout.addWidget(self._undo_btn)

        self._redo_btn = ToolButton("redo", "Redo", "Ctrl+Y")
        self._redo_btn.setToolTip("Redo [Ctrl+Y]\nRedo last undone action")
        self._redo_btn.clicked.connect(self.redo_requested.emit)
        layout.addWidget(self._redo_btn)

        # Divider
        layout.addWidget(_create_divider())

        # --- File ops: Save | Load ---
        self._save_btn = ToolButton("save", "Save", "Ctrl+S")
        self._save_btn.setToolTip(
            "Save Config [Ctrl+S]\n"
            "Save annotation points and settings to a JSON file\n"
            "for later reuse"
        )
        self._save_btn.clicked.connect(self.save_requested.emit)
        layout.addWidget(self._save_btn)

        self._load_btn = ToolButton("upload", "Load", "Ctrl+O")
        self._load_btn.setToolTip(
            "Load Config [Ctrl+O]\n"
            "Load previously saved annotation points and settings\n"
            "from a JSON file"
        )
        self._load_btn.clicked.connect(self.load_requested.emit)
        layout.addWidget(self._load_btn)

        # Spacer
        layout.addStretch()

        # --- Mask Correction (2-step workflow) ---
        self._add_correction_btn = ToolButton(
            "plus-circle", "Fix Mask"
        )
        self._add_correction_btn.setToolTip(
            "Step 1: Fix Mask on Current Frame\n"
            "Navigate to a frame with an inaccurate mask, then\n"
            "click this to add correction points.\n"
            "Use Foreground/Background to mark include/exclude areas.\n"
            "When done, click 'Apply & Propagate' (step 2)."
        )
        self._add_correction_btn.clicked.connect(
            self.add_correction_requested.emit
        )
        layout.addWidget(self._add_correction_btn)

        self._apply_correction_btn = ToolButton(
            "check-circle", "Apply & Propagate", variant="success"
        )
        self._apply_correction_btn.setToolTip(
            "Step 2: Apply & Propagate\n"
            "Re-generate masks from the corrected frame\n"
            "forward through all remaining frames.\n"
            "Only available after adding correction points."
        )
        self._apply_correction_btn.clicked.connect(
            self.apply_correction_requested.emit
        )
        layout.addWidget(self._apply_correction_btn)

        # Divider
        layout.addWidget(_create_divider())

        # Force re-convert (separate from correction workflow)
        self._force_reprocess = QCheckBox("Re-convert")
        self._force_reprocess.setToolTip(
            "Force Re-convert Images\n"
            "Re-converts all source images to the intermediate\n"
            "format before processing. Only needed if you changed\n"
            "the source images since the last run."
        )
        self._force_reprocess.setStyleSheet(
            f"QCheckBox {{ color: {Colors.TEXT_DIM}; "
            f"font-size: {Fonts.SIZE_SM}px; background: transparent; }}"
        )
        self._force_reprocess.toggled.connect(self.force_reprocess_changed.emit)
        layout.addWidget(self._force_reprocess)

        # Start / Stop processing
        self._start_btn = QPushButton("Start Processing")
        self._start_btn.setIcon(get_icon("play", "white", 16))
        self._start_btn.setProperty("cssClass", "btn-primary")
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_btn.setMinimumWidth(140)
        self._start_btn.setMinimumHeight(36)
        self._start_btn.setToolTip(
            "Start Processing [Ctrl+Enter]\n"
            "Convert images, load model, and propagate masks\n"
            "through all frames using your annotation points.\n"
            "Previously converted images and loaded models\n"
            "are reused automatically."
        )
        self._start_btn.clicked.connect(self.start_processing_requested.emit)
        layout.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setIcon(get_icon("stop", "white", 14))
        self._stop_btn.setProperty("cssClass", "btn-danger")
        self._stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_btn.setMinimumWidth(80)
        self._stop_btn.setMinimumHeight(36)
        self._stop_btn.setToolTip(
            "Stop Processing [Escape]\n"
            "Stop the current operation. Masks already saved\n"
            "will be kept. You can restart processing later."
        )
        self._stop_btn.setStyleSheet(
            f"QPushButton {{ background: {Colors.DANGER}; color: white; "
            f"border-radius: 8px; padding: 6px 16px; "
            f"font-size: {Fonts.SIZE_MD}px; }}"
            f"QPushButton:hover {{ background: #dc2626; }}"
        )
        self._stop_btn.clicked.connect(self.stop_processing_requested.emit)
        self._stop_btn.setVisible(False)
        layout.addWidget(self._stop_btn)

        self._is_processing = False

    # --- Public API ---

    def set_processing(self, is_processing: bool) -> None:
        """Toggle between Start and Stop button."""
        self._is_processing = is_processing
        self._start_btn.setVisible(not is_processing)
        self._stop_btn.setVisible(is_processing)

    def set_tool(self, tool: str) -> None:
        """Programmatically set the active tool."""
        tool_map = {"select": 0, "draw": 1, "erase": 2}
        btn_id = tool_map.get(tool, 0)
        buttons = {0: self._select_btn, 1: self._draw_btn, 2: self._erase_btn}
        buttons[btn_id].setChecked(True)

    def set_mode(self, mode: str) -> None:
        """Programmatically set the active mode."""
        if mode == "foreground":
            self._fg_btn.setChecked(True)
        else:
            self._bg_btn.setChecked(True)

    # --- Private slots ---

    def _on_tool_clicked(self, btn_id: int) -> None:
        tool_names = {0: "select", 1: "draw", 2: "erase"}
        self.tool_changed.emit(tool_names.get(btn_id, "select"))

    def _on_mode_clicked(self, btn_id: int) -> None:
        mode_names = {0: "foreground", 1: "background"}
        self.mode_changed.emit(mode_names.get(btn_id, "foreground"))
