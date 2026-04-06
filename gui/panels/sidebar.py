"""Left sidebar with configuration panels.

Contains collapsible sections for file paths, model configuration,
spatial smoothing, and temporal smoothing.
"""

from PyQt6.QtCore import QTimer, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui.icons import get_pixmap
from gui.theme import Colors, Fonts
from gui.widgets.collapsible_section import CollapsibleSection
from gui.widgets.hover_popup_button import HoverPopupButton
from gui.widgets.number_input import NumberInput
from gui.widgets.path_selector import PathSelector
from gui.widgets.select_field import SelectField
from gui.widgets.slider_input import SliderInput


def _get_pytorch_version() -> str:
    """Return the installed PyTorch version string."""
    try:
        import torch
        return torch.__version__
    except ImportError:
        return "N/A"


def _create_logo_header() -> QWidget:
    """Create the logo header widget with gradient icon and title."""
    header = QWidget()
    header.setStyleSheet(
        f"QWidget {{ border-bottom: 1px solid {Colors.BORDER}; }}"
    )

    layout = QHBoxLayout(header)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(10)

    # Gradient icon background with brain icon
    icon_container = QLabel()
    icon_container.setFixedSize(32, 32)
    icon_container.setPixmap(get_pixmap("brain", "white", 18))
    icon_container.setAlignment(Qt.AlignmentFlag.AlignCenter)
    icon_container.setStyleSheet(
        "QLabel {"
        "  background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
        f"    stop:0 {Colors.PRIMARY}, stop:1 #9333ea);"
        "  border-radius: 8px;"
        "}"
    )
    layout.addWidget(icon_container)

    # Title + subtitle
    text_layout = QVBoxLayout()
    text_layout.setContentsMargins(0, 0, 0, 0)
    text_layout.setSpacing(0)

    title = QLabel("DIC Mask Generator")
    title.setStyleSheet(
        f"color: white; font-size: {Fonts.SIZE_LG}px; "
        f"font-weight: 600; background: transparent; border: none;"
    )
    text_layout.addWidget(title)

    subtitle = QLabel("Mask Generator v2.0")
    subtitle.setStyleSheet(
        f"color: {Colors.TEXT_DIM}; font-size: {Fonts.SIZE_SM}px; "
        f"background: transparent; border: none;"
    )
    text_layout.addWidget(subtitle)

    layout.addLayout(text_layout)
    layout.addStretch()

    return header


def _create_accent_button(text: str) -> QPushButton:
    """Create a full-width accent-styled button."""
    btn = QPushButton(text)
    btn.setProperty("cssClass", "btn-accent")
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setMinimumHeight(36)
    return btn


class Sidebar(QWidget):
    """Left sidebar with configuration panels."""

    # Signals
    input_dir_changed = pyqtSignal(str)
    output_dir_changed = pyqtSignal(str)
    device_changed = pyqtSignal(str)
    model_changed = pyqtSignal(str)
    format_changed = pyqtSignal(str)
    mask_format_changed = pyqtSignal(str)
    threshold_changed = pyqtSignal(float)
    preprocessing_preview_requested = pyqtSignal(object)  # PreprocessingConfig
    save_preprocessed_requested = pyqtSignal(object)      # PreprocessingConfig
    shape_draw_requested = pyqtSignal(str, str)            # mode, shape_type
    shape_removed = pyqtSignal(int)                        # shape index
    shape_selected = pyqtSignal(int)                       # shape index
    spatial_smooth_requested = pyqtSignal(dict)
    spatial_preview_requested = pyqtSignal(dict)
    temporal_smooth_requested = pyqtSignal(dict)
    pp_step_advanced = pyqtSignal(int)  # step index that was just completed/skipped
    refresh_stats_requested = pyqtSignal()
    mask_view_changed = pyqtSignal(str)  # subdir name: "masks", "mask_spatial_smoothing", etc.
    panel_switched = pyqtSignal(str)  # "processing" or "postprocessing"

    # --- Step 0 manual edit signals ---
    manual_tool_changed = pyqtSignal(str)       # "brush" or "eraser"
    manual_brush_size_changed = pyqtSignal(int)  # radius in pixels
    manual_undo_requested = pyqtSignal()
    manual_redo_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.setMinimumWidth(280)
        self.setMaximumWidth(320)
        self.setFixedWidth(300)
        self.setStyleSheet(
            f"Sidebar {{ background: {Colors.BG_MEDIUM}; "
            f"border-right: 1px solid {Colors.BORDER}; }}"
        )

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Logo header
        main_layout.addWidget(_create_logo_header())

        # --- Panel toggle buttons ---
        tab_bar = QWidget()
        tab_bar.setStyleSheet(
            f"QWidget {{ border-bottom: 1px solid {Colors.BORDER}; "
            f"background: {Colors.BG_DARK}; }}"
        )
        tab_layout = QHBoxLayout(tab_bar)
        tab_layout.setContentsMargins(4, 4, 4, 4)
        tab_layout.setSpacing(2)

        self._tab_processing = QPushButton("Processing")
        self._tab_processing.setCursor(Qt.CursorShape.PointingHandCursor)
        self._tab_processing.setCheckable(True)
        self._tab_processing.setChecked(True)
        self._tab_processing.clicked.connect(self.switch_to_processing)

        self._tab_postprocessing = QPushButton("Post-Processing")
        self._tab_postprocessing.setCursor(Qt.CursorShape.PointingHandCursor)
        self._tab_postprocessing.setCheckable(True)
        self._tab_postprocessing.setChecked(False)
        self._tab_postprocessing.clicked.connect(self.switch_to_postprocessing)

        self._update_tab_styles()

        tab_layout.addWidget(self._tab_processing)
        tab_layout.addWidget(self._tab_postprocessing)
        main_layout.addWidget(tab_bar)

        # --- Stacked widget for panel switching ---
        self._panel_stack = QStackedWidget()
        self._panel_stack.setStyleSheet("background: transparent;")

        # ===== PAGE 1: Processing =====
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(8, 8, 8, 8)
        scroll_layout.setSpacing(8)

        # --- File Paths section ---
        file_section = CollapsibleSection(
            "File Paths", icon_name="folder-open", default_open=True
        )
        self._input_path = PathSelector("Input Directory", "/path/to/input...")
        self._input_path.path_changed.connect(self.input_dir_changed.emit)
        file_section.add_widget(self._input_path)

        self._output_path = PathSelector("Output Directory", "/path/to/output...")
        self._output_path.path_changed.connect(self.output_dir_changed.emit)
        file_section.add_widget(self._output_path)

        scroll_layout.addWidget(file_section)

        # --- Preprocessing section (before model config — logically first) ---
        preprocess_section = CollapsibleSection(
            "Preprocessing", icon_name="sliders", default_open=False
        )

        # Custom frames checkbox + input
        pp_custom_row = QWidget()
        pp_custom_layout = QHBoxLayout(pp_custom_row)
        pp_custom_layout.setContentsMargins(0, 0, 0, 0)
        pp_custom_layout.setSpacing(6)

        self._pp_custom_frames_check = QCheckBox("Custom frames")
        self._pp_custom_frames_check.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent;"
        )
        self._pp_custom_frames_check.setToolTip(
            "Process only selected frames instead of the full range.\n"
            "Enter comma-separated frame numbers or ranges (e.g. 1-10, 15, 20-30)."
        )
        self._pp_custom_frames_check.stateChanged.connect(self._on_pp_changed)
        pp_custom_layout.addWidget(self._pp_custom_frames_check)

        self._pp_custom_frames_input = QLineEdit()
        self._pp_custom_frames_input.setPlaceholderText("1-10, 15, 20-30")
        self._pp_custom_frames_input.setToolTip(
            "Specify frames to process: comma-separated numbers or ranges.\n"
            "Example: 1-10, 15, 20-30"
        )
        self._pp_custom_frames_input.setEnabled(False)
        self._pp_custom_frames_input.setStyleSheet(
            f"QLineEdit {{ background: {Colors.BG_DARK}; color: {Colors.TEXT_PRIMARY}; "
            f"border: 1px solid {Colors.BORDER}; border-radius: 4px; "
            f"padding: 2px 6px; font-size: {Fonts.SIZE_BASE}px; }}"
            f"QLineEdit:disabled {{ color: {Colors.TEXT_DIM}; }}"
        )
        self._pp_custom_frames_input.textChanged.connect(self._on_pp_changed)
        self._pp_custom_frames_check.toggled.connect(
            self._pp_custom_frames_input.setEnabled
        )
        pp_custom_layout.addWidget(self._pp_custom_frames_input)

        preprocess_section.add_widget(pp_custom_row)

        # ── [Tone ▼] popup: Gain, Brightness, Contrast, Clip, CLAHE ──
        self._tone_popup = HoverPopupButton("Tone", icon_name="sun")

        self._pp_gain = SliderInput(
            "Gain", default=1.0, min_val=0.1, max_val=5.0,
            step=0.01, decimals=2,
            tooltip="Multiplicative gain applied to pixel intensity.\n1.0 = no change. >1.0 brightens, <1.0 darkens.",
        )
        self._pp_gain.value_changed.connect(self._on_pp_changed)
        self._tone_popup.add_popup_widget(self._pp_gain)

        self._pp_brightness = SliderInput(
            "Brightness", default=0, min_val=-255, max_val=255,
            step=1, decimals=0,
            tooltip="Additive brightness offset.\n0 = no change. Positive = brighter, negative = darker.",
        )
        self._pp_brightness.value_changed.connect(self._on_pp_changed)
        self._tone_popup.add_popup_widget(self._pp_brightness)

        self._pp_contrast = SliderInput(
            "Contrast", default=1.0, min_val=0.0, max_val=5.0,
            step=0.01, decimals=2,
            tooltip="Contrast scaling factor.\n1.0 = no change. >1.0 increases contrast, <1.0 decreases.",
        )
        self._pp_contrast.value_changed.connect(self._on_pp_changed)
        self._tone_popup.add_popup_widget(self._pp_contrast)

        self._pp_clip_min = SliderInput(
            "Clip Min", default=0, min_val=0, max_val=254,
            step=1, decimals=0,
            tooltip="Minimum intensity clamp. Pixel values below this are set to 0.",
        )
        self._pp_clip_min.value_changed.connect(self._on_pp_changed)
        self._tone_popup.add_popup_widget(self._pp_clip_min)

        self._pp_clip_max = SliderInput(
            "Clip Max", default=255, min_val=1, max_val=255,
            step=1, decimals=0,
            tooltip="Maximum intensity clamp. Pixel values above this are set to 255.",
        )
        self._pp_clip_max.value_changed.connect(self._on_pp_changed)
        self._tone_popup.add_popup_widget(self._pp_clip_max)

        # CLAHE (contrast enhancement → belongs in Tone)
        self._pp_clahe_check = QCheckBox("CLAHE (Adaptive Histogram Eq.)")
        self._pp_clahe_check.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent;"
        )
        self._pp_clahe_check.setToolTip(
            "Contrast Limited Adaptive Histogram Equalization.\n"
            "Enhances local contrast — useful for low-contrast microscopy images."
        )
        self._pp_clahe_check.stateChanged.connect(self._on_pp_changed)
        self._tone_popup.add_popup_widget(self._pp_clahe_check)

        pp_clahe_grid = QWidget()
        pp_clahe_layout = QGridLayout(pp_clahe_grid)
        pp_clahe_layout.setContentsMargins(0, 0, 0, 0)
        pp_clahe_layout.setSpacing(8)

        self._pp_clahe_clip = NumberInput(
            "CLAHE Clip", default=2.0, min_val=0.1, max_val=40.0,
            step=0.5, decimals=1, icon_name="gauge",
            tooltip="CLAHE clip limit. Higher values allow more contrast enhancement.\nLow values (1-3) give subtle results; high values (10+) are aggressive.",
        )
        self._pp_clahe_clip.value_changed.connect(self._on_pp_changed)
        pp_clahe_layout.addWidget(self._pp_clahe_clip, 0, 0)

        self._pp_clahe_tile = NumberInput(
            "Tile Size", default=8, min_val=2, max_val=32,
            step=1, decimals=0, icon_name="hash",
            tooltip="CLAHE grid tile size (NxN pixels).\nSmaller tiles give more localized contrast enhancement.",
        )
        self._pp_clahe_tile.value_changed.connect(self._on_pp_changed)
        pp_clahe_layout.addWidget(self._pp_clahe_tile, 0, 1)

        self._tone_popup.add_popup_widget(pp_clahe_grid)

        preprocess_section.add_widget(self._tone_popup)

        # ── [Smoothing ▼] popup: Gaussian, Bilateral, Median, Box, NLM, Diffusion ──
        self._smooth_popup = HoverPopupButton("Smoothing", icon_name="waves")

        self._pp_gaussian = SliderInput(
            "Gaussian Sigma", default=0.0, min_val=0.0, max_val=10.0,
            step=0.1, decimals=1,
            tooltip="Standard deviation of Gaussian blur.\n0 = off. Higher values produce stronger smoothing.",
        )
        self._pp_gaussian.value_changed.connect(self._on_pp_changed)
        self._smooth_popup.add_popup_widget(self._pp_gaussian)

        self._pp_bilateral_check = QCheckBox("Bilateral (Edge-Preserving)")
        self._pp_bilateral_check.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent;"
        )
        self._pp_bilateral_check.setToolTip(
            "Bilateral filter smooths while preserving edges.\n"
            "Good for reducing noise without blurring boundaries."
        )
        self._pp_bilateral_check.stateChanged.connect(self._on_pp_changed)
        self._smooth_popup.add_popup_widget(self._pp_bilateral_check)

        self._pp_median = SliderInput(
            "Median Ksize (0=off)", default=0, min_val=0, max_val=31,
            step=1, decimals=0,
            tooltip="Median filter kernel size (must be odd).\n0 = off. Effective at removing salt-and-pepper noise.",
        )
        self._pp_median.value_changed.connect(self._on_pp_changed)
        self._smooth_popup.add_popup_widget(self._pp_median)

        self._pp_box = SliderInput(
            "Box Ksize", default=0, min_val=0, max_val=31,
            step=1, decimals=0,
            tooltip="Box (averaging) filter kernel size.\n0 = off. Uniform smoothing — simple but fast.",
        )
        self._pp_box.value_changed.connect(self._on_pp_changed)
        self._smooth_popup.add_popup_widget(self._pp_box)

        self._pp_nlm_check = QCheckBox("NLM Denoise (Slow)")
        self._pp_nlm_check.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent;"
        )
        self._pp_nlm_check.setToolTip(
            "Non-Local Means denoising. High quality but computationally slow.\n"
            "Best for images with significant Gaussian noise."
        )
        self._pp_nlm_check.stateChanged.connect(self._on_pp_changed)
        self._smooth_popup.add_popup_widget(self._pp_nlm_check)

        self._pp_nlm_h = SliderInput(
            "NLM Strength", default=10.0, min_val=1.0, max_val=40.0,
            step=1.0, decimals=0,
            tooltip="Filter strength (h parameter).\nHigher values remove more noise but may blur details.",
        )
        self._pp_nlm_h.value_changed.connect(self._on_pp_changed)
        self._smooth_popup.add_popup_widget(self._pp_nlm_h)

        self._pp_diffusion_check = QCheckBox("Anisotropic Diffusion")
        self._pp_diffusion_check.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent;"
        )
        self._pp_diffusion_check.setToolTip(
            "Perona-Malik anisotropic diffusion for edge-preserving smoothing.\n"
            "Smooths homogeneous regions while preserving strong edges."
        )
        self._pp_diffusion_check.stateChanged.connect(self._on_pp_changed)
        self._smooth_popup.add_popup_widget(self._pp_diffusion_check)

        pp_diff_grid = QWidget()
        pp_diff_layout = QGridLayout(pp_diff_grid)
        pp_diff_layout.setContentsMargins(0, 0, 0, 0)
        pp_diff_layout.setSpacing(8)

        self._pp_diff_iter = NumberInput(
            "Iterations", default=10, min_val=1, max_val=100,
            step=1, decimals=0, icon_name="repeat",
            tooltip="Number of diffusion iterations.\nMore iterations = stronger smoothing effect.",
        )
        self._pp_diff_iter.value_changed.connect(self._on_pp_changed)
        pp_diff_layout.addWidget(self._pp_diff_iter, 0, 0)

        self._pp_diff_kappa = NumberInput(
            "Kappa", default=30.0, min_val=1.0, max_val=200.0,
            step=5.0, decimals=0, icon_name="activity",
            tooltip="Edge sensitivity (conductance coefficient).\nLow kappa preserves weaker edges; high kappa only preserves strong edges.",
        )
        self._pp_diff_kappa.value_changed.connect(self._on_pp_changed)
        pp_diff_layout.addWidget(self._pp_diff_kappa, 0, 1)

        self._pp_diff_option = SelectField(
            "Mode", options=["Option 1 (edges)", "Option 2 (regions)"],
            tooltip="Diffusion function type.\nOption 1: favors high-contrast edges.\nOption 2: favors wide regions over thin edges.",
        )
        self._pp_diff_option.value_changed.connect(self._on_pp_changed)
        pp_diff_layout.addWidget(self._pp_diff_option, 1, 0, 1, 2)

        self._smooth_popup.add_popup_widget(pp_diff_grid)

        preprocess_section.add_widget(self._smooth_popup)

        # ── [Binarize ▼] popup: Threshold, Invert ──
        self._binarize_popup = HoverPopupButton("Binarize", icon_name="toggle-left")

        self._pp_threshold_check = QCheckBox("Binary Threshold")
        self._pp_threshold_check.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent;"
        )
        self._pp_threshold_check.setToolTip(
            "Convert image to binary (black/white) using a threshold.\n"
            "Pixels above the threshold become white; below become black."
        )
        self._pp_threshold_check.stateChanged.connect(self._on_pp_changed)
        self._binarize_popup.add_popup_widget(self._pp_threshold_check)

        self._pp_threshold_val = SliderInput(
            "Threshold Value", default=127, min_val=0, max_val=255,
            step=1, decimals=0,
            tooltip="Intensity cutoff for binarization (0-255).\nOnly used when method is 'Fixed'. Otsu computes automatically.",
        )
        self._pp_threshold_val.value_changed.connect(self._on_pp_changed)
        self._binarize_popup.add_popup_widget(self._pp_threshold_val)

        self._pp_threshold_method = SelectField(
            "Threshold Method",
            options=["Fixed", "Otsu", "Adaptive"],
            default="Fixed",
            tooltip="Fixed: use manual threshold value.\nOtsu: auto-compute optimal threshold.\nAdaptive: local region-based threshold.",
        )
        self._pp_threshold_method.value_changed.connect(self._on_pp_changed)
        self._binarize_popup.add_popup_widget(self._pp_threshold_method)

        self._pp_invert_check = QCheckBox("Invert")
        self._pp_invert_check.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent;"
        )
        self._pp_invert_check.setToolTip(
            "Invert pixel intensities (255 - value).\n"
            "Useful when foreground is darker than background."
        )
        self._pp_invert_check.stateChanged.connect(self._on_pp_changed)
        self._binarize_popup.add_popup_widget(self._pp_invert_check)

        preprocess_section.add_widget(self._binarize_popup)

        # ── [Morphology ▼] popup: Operation, Kernel, Iterations, Fill Holes ──
        self._morph_popup = HoverPopupButton("Morphology", icon_name="layers")

        self._pp_morph_check = QCheckBox("Morphology")
        self._pp_morph_check.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent;"
        )
        self._pp_morph_check.setToolTip(
            "Apply morphological operations to the image.\n"
            "Common uses: close gaps, remove small noise, extract edges."
        )
        self._pp_morph_check.stateChanged.connect(self._on_pp_changed)
        self._morph_popup.add_popup_widget(self._pp_morph_check)

        self._pp_morph_op = SelectField(
            "Operation",
            options=[
                "Dilate", "Erode", "Open", "Close",
                "Gradient", "Top-hat", "Black-hat",
            ],
            default="Close",
            tooltip="Dilate: expand bright regions. Erode: shrink bright regions.\nOpen: remove small bright spots. Close: fill small dark gaps.\nGradient: edge detection. Top/Black-hat: extract small features.",
        )
        self._pp_morph_op.value_changed.connect(self._on_pp_changed)
        self._morph_popup.add_popup_widget(self._pp_morph_op)

        pp_morph_grid = QWidget()
        pp_morph_grid_layout = QGridLayout(pp_morph_grid)
        pp_morph_grid_layout.setContentsMargins(0, 0, 0, 0)
        pp_morph_grid_layout.setSpacing(8)

        self._pp_morph_kernel = NumberInput(
            "Kernel Size", default=3, min_val=3, max_val=51,
            step=2, decimals=0, icon_name="hash",
            tooltip="Structuring element size (must be odd).\nLarger kernels produce stronger morphological effects.",
        )
        self._pp_morph_kernel.value_changed.connect(self._on_pp_changed)
        pp_morph_grid_layout.addWidget(self._pp_morph_kernel, 0, 0)

        self._pp_morph_iter = NumberInput(
            "Iterations", default=1, min_val=1, max_val=20,
            step=1, decimals=0, icon_name="hash",
            tooltip="Number of times the morphological operation is applied.\nMore iterations amplify the effect.",
        )
        self._pp_morph_iter.value_changed.connect(self._on_pp_changed)
        pp_morph_grid_layout.addWidget(self._pp_morph_iter, 0, 1)

        self._morph_popup.add_popup_widget(pp_morph_grid)

        self._pp_fill_holes_check = QCheckBox("Fill Holes")
        self._pp_fill_holes_check.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent;"
        )
        self._pp_fill_holes_check.setToolTip(
            "Fill enclosed dark holes within bright regions.\n"
            "Applied after morphological operations."
        )
        self._pp_fill_holes_check.stateChanged.connect(self._on_pp_changed)
        self._morph_popup.add_popup_widget(self._pp_fill_holes_check)

        preprocess_section.add_widget(self._morph_popup)

        # ── [Add ▼] [Cut ▼] shape buttons ──
        from PyQt6.QtWidgets import QMenu

        shape_btn_row = QWidget()
        shape_btn_layout = QHBoxLayout(shape_btn_row)
        shape_btn_layout.setContentsMargins(0, 0, 0, 0)
        shape_btn_layout.setSpacing(6)

        self._add_shape_btn = QPushButton("Add \u25BC")
        self._add_shape_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._add_shape_btn.setToolTip(
            "Draw a shape to fill with white (foreground).\n"
            "Choose Rect, Circle, or Polygon from the dropdown."
        )
        self._add_shape_btn.setStyleSheet(
            f"QPushButton {{ background: {Colors.SUCCESS_BG}; "
            f"color: {Colors.SUCCESS}; border: 1px solid {Colors.SUCCESS_BORDER}; "
            f"border-radius: 6px; padding: 4px 12px; "
            f"font-size: {Fonts.SIZE_SM}px; }}"
            f"QPushButton:hover {{ background: rgba(16, 185, 129, 0.30); }}"
        )
        add_menu = QMenu(self._add_shape_btn)
        add_menu.addAction("Rect", lambda: self._emit_shape("add", "rect"))
        add_menu.addAction("Circle", lambda: self._emit_shape("add", "circle"))
        add_menu.addAction("Polygon", lambda: self._emit_shape("add", "polygon"))
        self._add_shape_btn.setMenu(add_menu)
        shape_btn_layout.addWidget(self._add_shape_btn)

        self._cut_shape_btn = QPushButton("Cut \u25BC")
        self._cut_shape_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cut_shape_btn.setToolTip(
            "Draw a shape to fill with black (background).\n"
            "Choose Rect, Circle, or Polygon from the dropdown."
        )
        self._cut_shape_btn.setStyleSheet(
            f"QPushButton {{ background: {Colors.DANGER_BG}; "
            f"color: {Colors.DANGER}; border: 1px solid {Colors.DANGER_BORDER}; "
            f"border-radius: 6px; padding: 4px 12px; "
            f"font-size: {Fonts.SIZE_SM}px; }}"
            f"QPushButton:hover {{ background: rgba(239, 68, 68, 0.20); }}"
        )
        cut_menu = QMenu(self._cut_shape_btn)
        cut_menu.addAction("Rect", lambda: self._emit_shape("cut", "rect"))
        cut_menu.addAction("Circle", lambda: self._emit_shape("cut", "circle"))
        cut_menu.addAction("Polygon", lambda: self._emit_shape("cut", "polygon"))
        self._cut_shape_btn.setMenu(cut_menu)
        shape_btn_layout.addWidget(self._cut_shape_btn)

        shape_btn_layout.addStretch()
        preprocess_section.add_widget(shape_btn_row)

        # Shape list (dynamic, hidden when empty)
        self._shape_list_container = QWidget()
        self._shape_list_layout = QVBoxLayout(self._shape_list_container)
        self._shape_list_layout.setContentsMargins(0, 0, 0, 0)
        self._shape_list_layout.setSpacing(2)
        self._shape_list_container.setVisible(False)
        self._shape_entries: list[QWidget] = []
        preprocess_section.add_widget(self._shape_list_container)

        # Built-in presets dropdown
        self._builtin_preset_select = SelectField(
            "Preset Library",
            options=[
                "None (identity)", "DIC Microscopy", "Fluorescence",
                "Phase Contrast", "Brightfield",
                "High Noise (Denoise)", "Edge Enhancement",
            ],
            default="None (identity)",
            tooltip="Built-in preprocessing presets for common imaging modalities.\nSelect a preset to auto-configure all preprocessing parameters.",
        )
        self._builtin_preset_select.value_changed.connect(
            self._on_builtin_preset_changed,
        )
        preprocess_section.add_widget(self._builtin_preset_select)

        # Preset save / load row
        preset_row = QWidget()
        preset_layout = QHBoxLayout(preset_row)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(4)

        self._save_preset_btn = QPushButton("Save Preset")
        self._save_preset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._save_preset_btn.setToolTip("Save current preprocessing settings to a JSON file.")
        self._save_preset_btn.setStyleSheet(
            f"QPushButton {{ background: {Colors.BG_LIGHT}; "
            f"color: {Colors.TEXT_SECONDARY}; border: 1px solid {Colors.BORDER}; "
            f"border-radius: 6px; padding: 6px 0; font-size: {Fonts.SIZE_SM}px; }}"
            f"QPushButton:hover {{ background: {Colors.BG_INPUT}; "
            f"color: {Colors.TEXT_PRIMARY}; }}"
        )
        self._save_preset_btn.clicked.connect(self._on_save_preset)
        preset_layout.addWidget(self._save_preset_btn)

        self._load_preset_btn = QPushButton("Load Preset")
        self._load_preset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._load_preset_btn.setToolTip("Load preprocessing settings from a JSON file.")
        self._load_preset_btn.setStyleSheet(
            f"QPushButton {{ background: {Colors.BG_LIGHT}; "
            f"color: {Colors.TEXT_SECONDARY}; border: 1px solid {Colors.BORDER}; "
            f"border-radius: 6px; padding: 6px 0; font-size: {Fonts.SIZE_SM}px; }}"
            f"QPushButton:hover {{ background: {Colors.BG_INPUT}; "
            f"color: {Colors.TEXT_PRIMARY}; }}"
        )
        self._load_preset_btn.clicked.connect(self._on_load_preset)
        preset_layout.addWidget(self._load_preset_btn)

        preprocess_section.add_widget(preset_row)

        # Save Preprocessed button
        self._save_preprocessed_btn = _create_accent_button("Save Preprocessed")
        self._save_preprocessed_btn.setToolTip(
            "Apply current preprocessing to all frames and save to output directory.\n"
            "Preprocessed images are used as SAM2 input during processing."
        )
        self._save_preprocessed_btn.clicked.connect(self._on_save_preprocessed)
        preprocess_section.add_widget(self._save_preprocessed_btn)

        scroll_layout.addWidget(preprocess_section)

        # --- Model Configuration section ---
        model_section = CollapsibleSection(
            "Model Configuration", icon_name="settings", default_open=True
        )

        self._device_select = SelectField(
            "Device",
            options=["CUDA", "CPU", "MPS"],
            default="CUDA",
            icon_name="cpu",
            tooltip="CUDA: NVIDIA GPU (~100x faster). CPU: universal fallback.\nMPS: Apple Silicon GPU. CUDA is strongly recommended.",
        )
        self._device_select.value_changed.connect(self.device_changed.emit)
        model_section.add_widget(self._device_select)

        self._model_select = SelectField(
            "Model",
            options=[
                "SAM2 Hiera Large",
                "SAM2 Hiera Base Plus",
                "SAM2 Hiera Small",
                "SAM2 Hiera Tiny",
            ],
            default="SAM2 Hiera Large",
            icon_name="brain",
            tooltip="Large: best quality (~900 MB, ~1.5 GB VRAM).\nBase Plus: balanced quality/speed.\nSmall: faster, moderate quality.\nTiny: fastest (~150 MB), good for quick tests.",
        )
        self._model_select.value_changed.connect(self.model_changed.emit)
        model_section.add_widget(self._model_select)

        self._format_select = SelectField(
            "Intermediate Format",
            options=["JPEG (fast)", "PNG (lossless)"],
            default="JPEG (fast)",
            icon_name="zap",
        )
        self._format_select.setToolTip(
            "Format used to convert source images for SAM2.\n"
            "JPEG is faster; PNG is lossless but slower."
        )
        self._format_select.value_changed.connect(self.format_changed.emit)
        model_section.add_widget(self._format_select)

        self._mask_format_select = SelectField(
            "Mask Output Format",
            options=["TIFF (default)", "PNG (lossless)"],
            default="TIFF (default)",
            icon_name="image",
        )
        self._mask_format_select.setToolTip(
            "Format for saved mask files.\n"
            "TIFF is the default; PNG is lossless and smaller."
        )
        self._mask_format_select.value_changed.connect(
            self.mask_format_changed.emit,
        )
        model_section.add_widget(self._mask_format_select)

        self._export_overlay_check = QCheckBox("Auto-export overlay images")
        self._export_overlay_check.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent;"
        )
        self._export_overlay_check.setToolTip(
            "After processing, save overlay images (original + mask)\n"
            "to an 'overlays' subdirectory."
        )
        model_section.add_widget(self._export_overlay_check)

        self._threshold_slider = SliderInput(
            "Mask Threshold",
            default=0.0,
            min_val=-5.0,
            max_val=5.0,
            step=0.01,
            decimals=2,
        )
        self._threshold_slider.setToolTip(
            "Logit threshold for mask binarization.\n"
            "Higher = stricter (smaller mask),\n"
            "Lower = more permissive (larger mask).\n"
            "Default 0.0 works well for most cases."
        )
        self._threshold_slider.value_changed.connect(
            self.threshold_changed.emit
        )
        model_section.add_widget(self._threshold_slider)

        scroll_layout.addWidget(model_section)

        # End of processing page — add spacer and finalize
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        self._panel_stack.addWidget(scroll)  # index 0 = Processing

        # ===== PAGE 2: Post-Processing =====
        pp_scroll = QScrollArea()
        pp_scroll.setWidgetResizable(True)
        pp_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        pp_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        pp_scroll_content = QWidget()
        pp_scroll_layout = QVBoxLayout(pp_scroll_content)
        pp_scroll_layout.setContentsMargins(8, 8, 8, 8)
        pp_scroll_layout.setSpacing(8)

        # --- Mask View Selector ---
        self._mask_view_select = SelectField(
            "Viewing Masks From",
            options=["Original (masks/)"],
            default="Original (masks/)",
            tooltip=(
                "Switch between viewing original SAM2 masks,\n"
                "spatially smoothed masks, or temporally smoothed masks.\n"
                "New options appear after you apply smoothing."
            ),
        )
        self._mask_view_select.value_changed.connect(
            self._on_mask_view_changed
        )
        pp_scroll_layout.addWidget(self._mask_view_select)

        # ── Step state management ──
        # 0 = Manual Edit, 1 = Spatial, 2 = Temporal
        self._pp_step_sections: list[CollapsibleSection] = []
        self._pp_current_step: int = 0

        # --- Step 0: Manual Edit ---
        self._step0_section = CollapsibleSection(
            "Step 0 \u00b7 Manual Edit", icon_name="pencil", default_open=True,
        )

        step0_info = QLabel(
            "Paint or erase directly on the Mask Preview.\n"
            "Edits save to manual_edited/ on frame change."
        )
        step0_info.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px; "
            f"background: transparent; padding: 2px 0;"
        )
        step0_info.setWordWrap(True)
        self._step0_section.add_widget(step0_info)

        # ── Tool selector: Brush / Eraser (mutually exclusive) ──
        step0_tools = QWidget()
        step0_tools_layout = QHBoxLayout(step0_tools)
        step0_tools_layout.setContentsMargins(0, 4, 0, 0)
        step0_tools_layout.setSpacing(6)

        self._manual_brush_btn = QPushButton("Brush")
        self._manual_brush_btn.setCheckable(True)
        self._manual_brush_btn.setChecked(True)
        self._manual_brush_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._manual_brush_btn.setProperty("cssClass", "btn-tool")
        self._manual_brush_btn.setToolTip("Paint mask foreground (255).")
        self._manual_brush_btn.clicked.connect(
            lambda: self._on_manual_tool_selected("brush")
        )
        step0_tools_layout.addWidget(self._manual_brush_btn)

        self._manual_eraser_btn = QPushButton("Eraser")
        self._manual_eraser_btn.setCheckable(True)
        self._manual_eraser_btn.setChecked(False)
        self._manual_eraser_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._manual_eraser_btn.setProperty("cssClass", "btn-tool")
        self._manual_eraser_btn.setToolTip("Erase mask to background (0).")
        self._manual_eraser_btn.clicked.connect(
            lambda: self._on_manual_tool_selected("eraser")
        )
        step0_tools_layout.addWidget(self._manual_eraser_btn)

        self._step0_section.add_widget(step0_tools)

        # ── Brush size slider (1–100 px, default 10) ──
        self._manual_brush_size = SliderInput(
            "Brush Size",
            default=10,
            min_val=1,
            max_val=100,
            step=1,
            decimals=0,
            tooltip="Brush radius in pixels. Wheel reserved for canvas zoom.",
        )
        self._manual_brush_size.value_changed.connect(
            self._on_manual_brush_size_changed
        )
        self._step0_section.add_widget(self._manual_brush_size)

        # ── Undo / Redo row (disabled until first stroke) ──
        step0_history = QWidget()
        step0_history_layout = QHBoxLayout(step0_history)
        step0_history_layout.setContentsMargins(0, 4, 0, 0)
        step0_history_layout.setSpacing(6)

        self._manual_undo_btn = QPushButton("Undo")
        self._manual_undo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._manual_undo_btn.setProperty("cssClass", "btn-tool")
        self._manual_undo_btn.setEnabled(False)
        self._manual_undo_btn.setToolTip("Undo last stroke (per-frame, up to 20).")
        self._manual_undo_btn.clicked.connect(self.manual_undo_requested.emit)
        step0_history_layout.addWidget(self._manual_undo_btn)

        self._manual_redo_btn = QPushButton("Redo")
        self._manual_redo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._manual_redo_btn.setProperty("cssClass", "btn-tool")
        self._manual_redo_btn.setEnabled(False)
        self._manual_redo_btn.setToolTip("Redo previously undone stroke.")
        self._manual_redo_btn.clicked.connect(self.manual_redo_requested.emit)
        step0_history_layout.addWidget(self._manual_redo_btn)

        self._step0_section.add_widget(step0_history)

        # ── Done / Skip row (existing navigation) ──
        step0_btns = QWidget()
        step0_btn_layout = QHBoxLayout(step0_btns)
        step0_btn_layout.setContentsMargins(0, 4, 0, 0)
        step0_btn_layout.setSpacing(8)
        self._step0_done_btn = QPushButton("Done")
        self._step0_done_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._step0_done_btn.setProperty("cssClass", "btn-success")
        self._step0_done_btn.setToolTip("Mark manual editing as complete.")
        self._step0_done_btn.clicked.connect(lambda: self._advance_step(0))
        self._step0_skip_btn = QPushButton("Skip")
        self._step0_skip_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._step0_skip_btn.setToolTip("Skip manual editing.")
        self._step0_skip_btn.clicked.connect(lambda: self._advance_step(0))
        step0_btn_layout.addWidget(self._step0_done_btn)
        step0_btn_layout.addWidget(self._step0_skip_btn)
        self._step0_section.add_widget(step0_btns)

        pp_scroll_layout.addWidget(self._step0_section)
        self._pp_step_sections.append(self._step0_section)

        # --- Step 1: Spatial Smoothing ---
        self._step1_section = CollapsibleSection(
            "Step 1 \u00b7 Spatial Smoothing", icon_name="waves", default_open=True,
        )

        self._spatial_strength = SelectField(
            "Smoothing Strength",
            options=[
                "Light (5 iterations)",
                "Moderate (10 iterations)",
                "Standard (20 iterations)",
                "Strong (50 iterations)",
            ],
            default="Standard (20 iterations)",
            tooltip=(
                "Light: 5 iterations, minimal edge cleanup\n"
                "Moderate: 10 iterations, mild smoothing\n"
                "Standard: 20 iterations, balanced smoothing\n"
                "Strong: 50 iterations, aggressive smoothing"
            ),
        )
        self._step1_section.add_widget(self._spatial_strength)

        self._spatial_replace_check = QCheckBox("Replace original masks")
        self._spatial_replace_check.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent;"
        )
        self._spatial_replace_check.setToolTip(
            "Write smoothed masks back to the masks/ directory,\n"
            "replacing the originals instead of creating a new folder."
        )
        self._step1_section.add_widget(self._spatial_replace_check)

        # Preview + Apply buttons side by side
        step1_btns = QWidget()
        step1_btn_layout = QHBoxLayout(step1_btns)
        step1_btn_layout.setContentsMargins(0, 0, 0, 0)
        step1_btn_layout.setSpacing(8)

        self._spatial_preview_btn = QPushButton("Preview")
        self._spatial_preview_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._spatial_preview_btn.setStyleSheet(
            f"QPushButton {{ background: {Colors.BG_LIGHT}; "
            f"color: {Colors.TEXT_SECONDARY}; border: 1px solid {Colors.BORDER}; "
            f"border-radius: 6px; padding: 6px 0; font-size: {Fonts.SIZE_SM}px; }}"
            f"QPushButton:hover {{ background: {Colors.BG_INPUT}; "
            f"color: {Colors.TEXT_PRIMARY}; }}"
        )
        self._spatial_preview_btn.setToolTip(
            "Preview smoothing on the current frame only (instant).\n"
            "Shows before/after comparison without processing all frames."
        )
        self._spatial_preview_btn.clicked.connect(self._on_spatial_preview)
        step1_btn_layout.addWidget(self._spatial_preview_btn)

        self._spatial_btn = _create_accent_button("Apply to All")
        self._spatial_btn.setToolTip(
            "Apply smoothing to all mask frames.\n"
            "Smooths jagged mask boundaries while preserving overall shape."
        )
        self._spatial_btn.clicked.connect(self._on_spatial_smooth)
        step1_btn_layout.addWidget(self._spatial_btn)

        self._step1_section.add_widget(step1_btns)

        # Done / Skip row
        step1_nav = QWidget()
        step1_nav_layout = QHBoxLayout(step1_nav)
        step1_nav_layout.setContentsMargins(0, 4, 0, 0)
        step1_nav_layout.setSpacing(8)
        self._step1_done_btn = QPushButton("Done")
        self._step1_done_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._step1_done_btn.setProperty("cssClass", "btn-success")
        self._step1_done_btn.setToolTip("Mark spatial smoothing as complete.")
        self._step1_done_btn.clicked.connect(lambda: self._advance_step(1))
        self._step1_skip_btn = QPushButton("Skip")
        self._step1_skip_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._step1_skip_btn.setToolTip("Skip spatial smoothing.")
        self._step1_skip_btn.clicked.connect(lambda: self._advance_step(1))
        step1_nav_layout.addWidget(self._step1_done_btn)
        step1_nav_layout.addWidget(self._step1_skip_btn)
        self._step1_section.add_widget(step1_nav)

        pp_scroll_layout.addWidget(self._step1_section)
        self._pp_step_sections.append(self._step1_section)

        # --- Step 2: Temporal Smoothing ---
        self._step2_section = CollapsibleSection(
            "Step 2 \u00b7 Temporal Smoothing", icon_name="sliders", default_open=True,
        )

        # Input source indicator
        self._temporal_source_label = QLabel("Input: masks/")
        self._temporal_source_label.setStyleSheet(
            f"color: {Colors.TEXT_DIM}; font-size: {Fonts.SIZE_SM}px; "
            f"background: transparent; padding: 2px 0; font-style: italic;"
        )
        self._temporal_source_label.setToolTip(
            "Shows which mask directory temporal smoothing will read from.\n"
            "If spatial smoothing results exist, temporal will chain from them."
        )
        self._step2_section.add_widget(self._temporal_source_label)

        self._temporal_strength = SelectField(
            "Smoothing Strength",
            options=[
                "Repair Bad Frames Only",
                "Repair + Light Smoothing",
                "Repair + Standard Smoothing",
                "Repair + Strong Smoothing",
            ],
            default="Repair Bad Frames Only",
            tooltip=(
                "Repair Bad Frames Only: correct bad frames, preserve all dynamics\n"
                "Repair + Light Smoothing: fix bad frames + subtle temporal smoothing\n"
                "Repair + Standard Smoothing: fix bad frames + moderate temporal smoothing\n"
                "Repair + Strong Smoothing: fix bad frames + aggressive smoothing (may flatten rapid changes)"
            ),
        )
        self._step2_section.add_widget(self._temporal_strength)

        self._temporal_replace_check = QCheckBox("Replace original masks")
        self._temporal_replace_check.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent;"
        )
        self._temporal_replace_check.setToolTip(
            "Write smoothed masks back to the masks/ directory,\n"
            "replacing the originals instead of creating a new folder."
        )
        self._step2_section.add_widget(self._temporal_replace_check)

        self._temporal_btn = _create_accent_button("Apply Temporal Smooth")
        self._temporal_btn.setToolTip(
            "Apply 3D Gaussian filter across the temporal dimension.\n"
            "Fixes outlier frames and ensures smooth transitions."
        )
        self._temporal_btn.clicked.connect(self._on_temporal_smooth)
        self._step2_section.add_widget(self._temporal_btn)

        pp_scroll_layout.addWidget(self._step2_section)
        self._pp_step_sections.append(self._step2_section)

        # Apply initial step state
        self._refresh_step_visuals()

        # --- Mask Statistics section ---
        stats_section = CollapsibleSection(
            "Mask Statistics", icon_name="activity", default_open=False,
        )

        self._stats_area_label = QLabel("Area: —")
        self._stats_area_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent; padding: 4px 0;"
        )
        self._stats_area_label.setWordWrap(True)
        stats_section.add_widget(self._stats_area_label)

        self._stats_consistency_label = QLabel("Consistency: —")
        self._stats_consistency_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent; padding: 4px 0;"
        )
        self._stats_consistency_label.setWordWrap(True)
        stats_section.add_widget(self._stats_consistency_label)

        self._stats_anomaly_label = QLabel("Anomalies: —")
        self._stats_anomaly_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent; padding: 4px 0;"
        )
        self._stats_anomaly_label.setWordWrap(True)
        stats_section.add_widget(self._stats_anomaly_label)

        self._refresh_stats_btn = QPushButton("Refresh Statistics")
        self._refresh_stats_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh_stats_btn.setToolTip(
            "Recalculate mask area, frame-to-frame consistency,\n"
            "and anomaly detection for the active mask directory."
        )
        self._refresh_stats_btn.setStyleSheet(
            f"QPushButton {{ background: {Colors.BG_LIGHT}; "
            f"color: {Colors.TEXT_SECONDARY}; border: 1px solid {Colors.BORDER}; "
            f"border-radius: 6px; padding: 6px 0; font-size: {Fonts.SIZE_SM}px; }}"
            f"QPushButton:hover {{ background: {Colors.BG_INPUT}; "
            f"color: {Colors.TEXT_PRIMARY}; }}"
        )
        self._refresh_stats_btn.clicked.connect(
            self.refresh_stats_requested.emit,
        )
        stats_section.add_widget(self._refresh_stats_btn)

        pp_scroll_layout.addWidget(stats_section)

        # Spacer to push footer down on post-processing page
        pp_scroll_layout.addStretch()

        pp_scroll.setWidget(pp_scroll_content)
        self._panel_stack.addWidget(pp_scroll)  # index 1 = Post-Processing

        main_layout.addWidget(self._panel_stack, 1)

        # Debounce timer for preprocessing preview (500ms)
        self._pp_debounce_timer = QTimer(self)
        self._pp_debounce_timer.setSingleShot(True)
        self._pp_debounce_timer.setInterval(500)
        self._pp_debounce_timer.timeout.connect(self._emit_pp_preview)

        # Version footer
        version_label = QLabel(
            f"DIC Mask Generator v2.0.0 | PyTorch {_get_pytorch_version()}"
        )
        version_label.setStyleSheet(
            f"color: {Colors.TEXT_DIM}; font-size: {Fonts.SIZE_SM}px; "
            f"padding: 8px 12px; border-top: 1px solid {Colors.BORDER}; "
            f"background: transparent;"
        )
        main_layout.addWidget(version_label)

    # --- Panel switching ---

    def _update_tab_styles(self) -> None:
        """Update toggle button styles based on active state."""
        active_style = (
            f"QPushButton {{ background: {Colors.PRIMARY}; "
            f"color: {Colors.TEXT_PRIMARY}; border: none; "
            f"border-radius: 4px; padding: 6px 12px; "
            f"font-size: {Fonts.SIZE_SM}px; font-weight: 600; }}"
        )
        inactive_style = (
            f"QPushButton {{ background: transparent; "
            f"color: {Colors.TEXT_DIM}; border: none; "
            f"border-radius: 4px; padding: 6px 12px; "
            f"font-size: {Fonts.SIZE_SM}px; }}"
            f"QPushButton:hover {{ color: {Colors.TEXT_SECONDARY}; "
            f"background: {Colors.BG_LIGHT}; }}"
        )
        self._tab_processing.setStyleSheet(
            active_style if self._tab_processing.isChecked() else inactive_style
        )
        self._tab_postprocessing.setStyleSheet(
            active_style if self._tab_postprocessing.isChecked() else inactive_style
        )

    def switch_to_processing(self) -> None:
        """Switch sidebar to the processing panel."""
        self._panel_stack.setCurrentIndex(0)
        self._tab_processing.setChecked(True)
        self._tab_postprocessing.setChecked(False)
        self._update_tab_styles()
        self.panel_switched.emit("processing")

    def switch_to_postprocessing(self) -> None:
        """Switch sidebar to the post-processing panel."""
        self._panel_stack.setCurrentIndex(1)
        self._tab_processing.setChecked(False)
        self._tab_postprocessing.setChecked(True)
        self._update_tab_styles()
        self.panel_switched.emit("postprocessing")

    @property
    def is_postprocessing_active(self) -> bool:
        """Whether the post-processing panel is currently shown."""
        return self._panel_stack.currentIndex() == 1

    # --- Public API ---

    def set_device_options(self, options: list[str], default: str = "") -> None:
        """Update available device options (called by DeviceManager)."""
        self._device_select.set_options(options)
        if default:
            self._device_select.set_value(default)

    def update_mask_statistics(
        self, area_text: str, consistency_text: str, anomaly_text: str,
    ) -> None:
        """Update mask statistics labels."""
        self._stats_area_label.setText(area_text)
        self._stats_consistency_label.setText(consistency_text)
        self._stats_anomaly_label.setText(anomaly_text)

    @property
    def export_overlays_enabled(self) -> bool:
        """Whether auto-export overlay images is checked."""
        return self._export_overlay_check.isChecked()

    def set_input_path(self, path: str, emit_signal: bool = True) -> None:
        """Set input directory path."""
        self._input_path.set_path(path, emit_signal=emit_signal)

    def set_output_path(self, path: str, emit_signal: bool = True) -> None:
        """Set output directory path."""
        self._output_path.set_path(path, emit_signal=emit_signal)

    def get_preprocessing_config(self):
        """Build PreprocessingConfig from current sidebar values."""
        from core.preprocessing import PreprocessingConfig

        # Map morphology display name to config string
        morph_map = {
            "dilate": "dilate", "erode": "erode", "open": "open",
            "close": "close", "gradient": "gradient",
            "top-hat": "tophat", "black-hat": "blackhat",
        }
        morph_op = "none"
        if self._pp_morph_check.isChecked():
            morph_op = morph_map.get(
                self._pp_morph_op.value().lower(), "none",
            )

        custom_frames = ""
        if self._pp_custom_frames_check.isChecked():
            custom_frames = self._pp_custom_frames_input.text().strip()

        # Diffusion option: 1 or 2
        diff_option = 1 if "1" in self._pp_diff_option.value() else 2

        return PreprocessingConfig(
            gain=self._pp_gain.value(),
            brightness=int(self._pp_brightness.value()),
            contrast=self._pp_contrast.value(),
            clip_min=int(self._pp_clip_min.value()),
            clip_max=int(self._pp_clip_max.value()),
            clahe_enabled=self._pp_clahe_check.isChecked(),
            clahe_clip_limit=self._pp_clahe_clip.value(),
            clahe_tile_size=int(self._pp_clahe_tile.value()),
            gaussian_sigma=self._pp_gaussian.value(),
            bilateral_enabled=self._pp_bilateral_check.isChecked(),
            bilateral_d=9,
            bilateral_sigma_color=75.0,
            bilateral_sigma_space=75.0,
            median_ksize=int(self._pp_median.value()),
            box_ksize=int(self._pp_box.value()),
            nlm_enabled=self._pp_nlm_check.isChecked(),
            nlm_h=self._pp_nlm_h.value(),
            diffusion_enabled=self._pp_diffusion_check.isChecked(),
            diffusion_iterations=int(self._pp_diff_iter.value()),
            diffusion_kappa=self._pp_diff_kappa.value(),
            diffusion_dt=0.1,
            diffusion_option=diff_option,
            threshold_enabled=self._pp_threshold_check.isChecked(),
            threshold_value=int(self._pp_threshold_val.value()),
            threshold_method=self._pp_threshold_method.value().lower(),
            invert=self._pp_invert_check.isChecked(),
            morphology_op=morph_op,
            morphology_kernel_size=int(self._pp_morph_kernel.value()),
            morphology_iterations=int(self._pp_morph_iter.value()),
            fill_holes=self._pp_fill_holes_check.isChecked(),
            custom_frames=custom_frames,
        )

    def set_preprocessing_config(self, config) -> None:
        """Restore all preprocessing UI controls from a PreprocessingConfig.

        Blocks signals while updating to avoid triggering redundant previews,
        then fires a single preview update at the end.
        """
        # Block individual widget signals during batch update
        widgets = [
            self._pp_gain, self._pp_brightness, self._pp_contrast,
            self._pp_clip_min, self._pp_clip_max,
            self._pp_clahe_check, self._pp_clahe_clip, self._pp_clahe_tile,
            self._pp_gaussian, self._pp_bilateral_check,
            self._pp_median, self._pp_box,
            self._pp_nlm_check, self._pp_nlm_h,
            self._pp_diffusion_check, self._pp_diff_iter,
            self._pp_diff_kappa, self._pp_diff_option,
            self._pp_threshold_check, self._pp_threshold_val,
            self._pp_threshold_method, self._pp_invert_check,
            self._pp_morph_check, self._pp_morph_op,
            self._pp_morph_kernel, self._pp_morph_iter,
            self._pp_fill_holes_check,
        ]
        for w in widgets:
            w.blockSignals(True)

        # --- Tone ---
        self._pp_gain.set_value(config.gain)
        self._pp_brightness.set_value(config.brightness)
        self._pp_contrast.set_value(config.contrast)
        self._pp_clip_min.set_value(config.clip_min)
        self._pp_clip_max.set_value(config.clip_max)
        self._pp_clahe_check.setChecked(config.clahe_enabled)
        self._pp_clahe_clip.set_value(config.clahe_clip_limit)
        self._pp_clahe_tile.set_value(config.clahe_tile_size)

        # --- Smoothing ---
        self._pp_gaussian.set_value(config.gaussian_sigma)
        self._pp_bilateral_check.setChecked(config.bilateral_enabled)
        self._pp_median.set_value(config.median_ksize)
        self._pp_box.set_value(config.box_ksize)
        self._pp_nlm_check.setChecked(config.nlm_enabled)
        self._pp_nlm_h.set_value(config.nlm_h)
        self._pp_diffusion_check.setChecked(config.diffusion_enabled)
        self._pp_diff_iter.set_value(config.diffusion_iterations)
        self._pp_diff_kappa.set_value(config.diffusion_kappa)
        option_text = (
            "Option 1 (edges)" if config.diffusion_option == 1
            else "Option 2 (regions)"
        )
        self._pp_diff_option.set_value(option_text)

        # --- Binarize ---
        self._pp_threshold_check.setChecked(config.threshold_enabled)
        self._pp_threshold_val.set_value(config.threshold_value)
        method_map = {"fixed": "Fixed", "otsu": "Otsu", "adaptive": "Adaptive"}
        self._pp_threshold_method.set_value(
            method_map.get(config.threshold_method, "Fixed"),
        )
        self._pp_invert_check.setChecked(config.invert)

        # --- Morphology ---
        morph_display = {
            "none": "", "dilate": "Dilate", "erode": "Erode",
            "open": "Open", "close": "Close", "gradient": "Gradient",
            "tophat": "Top-hat", "blackhat": "Black-hat",
        }
        has_morph = config.morphology_op != "none"
        self._pp_morph_check.setChecked(has_morph)
        if has_morph:
            self._pp_morph_op.set_value(
                morph_display.get(config.morphology_op, "Dilate"),
            )
        self._pp_morph_kernel.set_value(config.morphology_kernel_size)
        self._pp_morph_iter.set_value(config.morphology_iterations)
        self._pp_fill_holes_check.setChecked(config.fill_holes)

        # Unblock signals
        for w in widgets:
            w.blockSignals(False)

        # Fire a single update
        self._update_popup_active_states()
        new_config = self.get_preprocessing_config()
        self.preprocessing_preview_requested.emit(new_config)

    # --- Private slots ---

    def add_shape_entry(self, mode: str, shape_type: str) -> None:
        """Add a shape entry to the shape list."""
        idx = len(self._shape_entries)
        label_text = f"{mode.capitalize()} {shape_type.capitalize()} #{idx + 1}"

        row = QWidget()
        row.setStyleSheet(
            f"QWidget {{ background: {Colors.BG_INPUT}; "
            f"border-radius: 6px; }}"
        )
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(8, 4, 4, 4)
        row_layout.setSpacing(6)

        # Colored dot indicator
        dot_color = Colors.SUCCESS if mode == "add" else Colors.DANGER
        dot = QLabel("\u25CF")
        dot.setStyleSheet(
            f"color: {dot_color}; font-size: 10px; background: transparent;"
        )
        row_layout.addWidget(dot)

        label = QPushButton(label_text)
        label.setStyleSheet(
            f"QPushButton {{ color: {Colors.TEXT_SECONDARY}; "
            f"background: transparent; border: none; text-align: left; "
            f"font-size: {Fonts.SIZE_SM}px; padding: 2px; }}"
            f"QPushButton:hover {{ color: {Colors.TEXT_PRIMARY}; }}"
        )
        capture_idx = idx
        label.clicked.connect(lambda: self.shape_selected.emit(capture_idx))
        row_layout.addWidget(label, 1)

        del_btn = QPushButton("\u00D7")
        del_btn.setFixedSize(20, 20)
        del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        del_btn.setStyleSheet(
            f"QPushButton {{ color: {Colors.TEXT_SECONDARY}; background: transparent; "
            f"border: none; font-size: 14px; border-radius: 4px; }}"
            f"QPushButton:hover {{ color: {Colors.DANGER}; "
            f"background: {Colors.DANGER_BG}; }}"
        )
        del_btn.clicked.connect(lambda: self._on_remove_shape(capture_idx))
        row_layout.addWidget(del_btn)

        self._shape_entries.append(row)
        self._shape_list_layout.addWidget(row)
        self._shape_list_container.setVisible(True)

    def clear_shape_entries(self) -> None:
        """Remove all shape entries from the list."""
        for entry in self._shape_entries:
            self._shape_list_layout.removeWidget(entry)
            entry.deleteLater()
        self._shape_entries.clear()
        self._shape_list_container.setVisible(False)

    def _on_remove_shape(self, index: int) -> None:
        """Handle shape removal from the list."""
        if 0 <= index < len(self._shape_entries):
            self.shape_removed.emit(index)

    def rebuild_shape_list(self, shapes: list) -> None:
        """Rebuild the shape list from a list of ShapeOverlay objects."""
        self.clear_shape_entries()
        for shape in shapes:
            self.add_shape_entry(shape.mode, shape.shape_type)

    def _emit_shape(self, mode: str, shape_type: str) -> None:
        """Emit shape draw request signal."""
        self.shape_draw_requested.emit(mode, shape_type)

    def _update_popup_active_states(self) -> None:
        """Update popup button active indicators based on current values."""
        # Tone: active if any value differs from default (includes CLAHE)
        tone_active = (
            self._pp_gain.value() != 1.0
            or int(self._pp_brightness.value()) != 0
            or self._pp_contrast.value() != 1.0
            or int(self._pp_clip_min.value()) != 0
            or int(self._pp_clip_max.value()) != 255
            or self._pp_clahe_check.isChecked()
        )
        self._tone_popup.set_active(tone_active)

        # Smoothing: active if any filter is enabled
        smooth_active = (
            self._pp_gaussian.value() > 0.0
            or self._pp_bilateral_check.isChecked()
            or int(self._pp_median.value()) > 0
            or int(self._pp_box.value()) > 0
            or self._pp_nlm_check.isChecked()
            or self._pp_diffusion_check.isChecked()
        )
        self._smooth_popup.set_active(smooth_active)

        # Binarize: active if threshold or invert enabled
        binarize_active = (
            self._pp_threshold_check.isChecked()
            or self._pp_invert_check.isChecked()
        )
        self._binarize_popup.set_active(binarize_active)

        # Morphology: active if morph or fill holes enabled
        morph_active = (
            self._pp_morph_check.isChecked()
            or self._pp_fill_holes_check.isChecked()
        )
        self._morph_popup.set_active(morph_active)

    def _on_pp_changed(self, *args) -> None:
        """Debounce preprocessing preview: update indicators immediately,
        but delay the expensive preview emission by 500ms."""
        self._update_popup_active_states()
        self._pp_debounce_timer.start()  # restart the 500ms countdown

    def _emit_pp_preview(self) -> None:
        """Actually emit the preprocessing preview signal (called after debounce)."""
        config = self.get_preprocessing_config()
        self.preprocessing_preview_requested.emit(config)

    def _on_builtin_preset_changed(self, name: str) -> None:
        """Apply a built-in preprocessing preset."""
        from core.preprocessing import BUILTIN_PRESETS
        config = BUILTIN_PRESETS.get(name)
        if config is not None:
            self.set_preprocessing_config(config)

    def _on_save_preprocessed(self) -> None:
        """Emit current preprocessing config for standalone save."""
        config = self.get_preprocessing_config()
        self.save_preprocessed_requested.emit(config)

    def _on_save_preset(self) -> None:
        """Save current preprocessing settings to a JSON file."""
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Preprocessing Preset", "",
            "Preset Files (*.json);;All Files (*)",
        )
        if not path:
            return

        try:
            from core.preprocessing import save_preset

            config = self.get_preprocessing_config()
            save_preset(config, path)
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(
                self, "Save Error", f"Failed to save preset:\n{e}",
            )

    def _on_load_preset(self) -> None:
        """Load preprocessing settings from a JSON file."""
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Preprocessing Preset", "",
            "Preset Files (*.json);;All Files (*)",
        )
        if not path:
            return

        try:
            from core.preprocessing import load_preset

            config = load_preset(path)
            self.set_preprocessing_config(config)
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(
                self, "Load Error", f"Failed to load preset:\n{e}",
            )

    # ── Spatial / temporal preset mappings ──

    _SPATIAL_PRESETS = {
        "Light (5 iterations)": {"iterations": 5, "gaussian_sigma": 2.0},
        "Moderate (10 iterations)": {"iterations": 10, "gaussian_sigma": 2.0},
        "Standard (20 iterations)": {"iterations": 20, "gaussian_sigma": 2.0},
        "Strong (50 iterations)": {"iterations": 50, "gaussian_sigma": 2.0},
    }
    _TEMPORAL_PRESETS = {
        "Repair Bad Frames Only": {"sigma": 2.0, "temporal_sigma": 0},
        "Repair + Light Smoothing": {"sigma": 2.0, "temporal_sigma": 0.5},
        "Repair + Standard Smoothing": {"sigma": 2.0, "temporal_sigma": 1.0},
        "Repair + Strong Smoothing": {"sigma": 2.0, "temporal_sigma": 2.0},
    }

    def _get_spatial_params(self) -> dict:
        """Build spatial smoothing params from the selected preset."""
        preset = self._SPATIAL_PRESETS.get(
            self._spatial_strength.value(),
            self._SPATIAL_PRESETS["Standard (20 iterations)"],
        )
        return {
            "iterations": preset["iterations"],
            "dt": 0.1,
            "kappa": 30.0,
            "option": 1,
            "gaussian_sigma": preset["gaussian_sigma"],
            "replace_originals": self._spatial_replace_check.isChecked(),
        }

    def _on_spatial_smooth(self) -> None:
        self.spatial_smooth_requested.emit(self._get_spatial_params())

    def _on_spatial_preview(self) -> None:
        self.spatial_preview_requested.emit(self._get_spatial_params())

    def _on_temporal_smooth(self) -> None:
        preset = self._TEMPORAL_PRESETS.get(
            self._temporal_strength.value(),
            self._TEMPORAL_PRESETS["Repair Bad Frames Only"],
        )
        params = {
            "sigma": preset["sigma"],
            "temporal_sigma": preset.get("temporal_sigma", 0),
            "neighbors": 2,
            "variance_threshold": None,
            "replace_originals": self._temporal_replace_check.isChecked(),
        }
        self.temporal_smooth_requested.emit(params)

    # ── Step 0 manual edit controls ──

    def _on_manual_tool_selected(self, tool: str) -> None:
        """Handle Brush / Eraser toggle — enforce mutual exclusion."""
        is_brush = tool == "brush"
        self._manual_brush_btn.setChecked(is_brush)
        self._manual_eraser_btn.setChecked(not is_brush)
        self.manual_tool_changed.emit(tool)

    def _on_manual_brush_size_changed(self, value: float) -> None:
        """Re-emit the slider's float value as an integer radius."""
        self.manual_brush_size_changed.emit(int(round(value)))

    def set_manual_undo_state(self, can_undo: bool, can_redo: bool) -> None:
        """Enable/disable Undo and Redo buttons (wired to controller signal)."""
        self._manual_undo_btn.setEnabled(can_undo)
        self._manual_redo_btn.setEnabled(can_redo)

    def manual_current_tool(self) -> str:
        """Return the currently selected manual edit tool."""
        return "brush" if self._manual_brush_btn.isChecked() else "eraser"

    def manual_current_brush_size(self) -> int:
        """Return the current brush radius in pixels."""
        return int(round(self._manual_brush_size.value()))

    # ── Step navigation ──

    def _advance_step(self, completed_step: int) -> None:
        """Mark a step as done and unlock the next one."""
        if completed_step >= self._pp_current_step:
            self._pp_current_step = completed_step + 1
            self._refresh_step_visuals()
            self.pp_step_advanced.emit(completed_step)

    def _refresh_step_visuals(self) -> None:
        """Update all step sections to reflect current progress."""
        for i, section in enumerate(self._pp_step_sections):
            if i < self._pp_current_step:
                # Completed
                section.set_badge("DONE", Colors.SUCCESS, Colors.SUCCESS_BG)
                section.set_title_color(Colors.SUCCESS)
                section.set_content_enabled(True)
                section.set_open(False)
            elif i == self._pp_current_step:
                # Active
                section.set_badge("", "", "")
                section.set_title_color(Colors.TEXT_MUTED)
                section.set_content_enabled(True)
                section.set_open(True)
            else:
                # Locked
                section.set_badge("LOCKED", Colors.TEXT_DIM, Colors.BG_INPUT)
                section.set_title_color(Colors.TEXT_DIM)
                section.set_content_enabled(False)
                section.set_open(False)

    def reset_pp_steps(self) -> None:
        """Reset post-processing step progress (e.g. on new project)."""
        self._pp_current_step = 0
        self._refresh_step_visuals()

    def _on_mask_view_changed(self, label: str) -> None:
        """Emit the mask subdirectory corresponding to the selected view."""
        subdir = self._MASK_VIEW_MAP.get(label, "masks")
        self.mask_view_changed.emit(subdir)

    # Map display labels → directory names
    _MASK_VIEW_MAP = {
        "Original (masks/)": "masks",
        "Manually Edited": "manual_edited",
        "Spatial Smoothed": "mask_spatial_smoothing",
        "Temporal Smoothed": "mask_temporal_smoothing",
    }

    def add_mask_view_option(self, label: str) -> None:
        """Add a new option to the mask view selector if not already present."""
        current_items = self._mask_view_select.options()
        if label not in current_items:
            self._mask_view_select.add_option(label)

    def set_mask_view(self, label: str) -> None:
        """Programmatically set the active mask view (no signal emitted)."""
        self._mask_view_select.blockSignals(True)
        self._mask_view_select.set_value(label)
        self._mask_view_select.blockSignals(False)

    def update_temporal_source_label(
        self, source_dir: str, is_chained: bool = False,
    ) -> None:
        """Update the temporal smoothing input source indicator.

        Args:
            source_dir: Display text for the source directory.
            is_chained: True when temporal chains from spatial results.
        """
        self._temporal_source_label.setText(f"Input: {source_dir}")
        color = Colors.SUCCESS if is_chained else Colors.TEXT_DIM
        self._temporal_source_label.setStyleSheet(
            f"color: {color}; font-size: {Fonts.SIZE_SM}px; "
            f"background: transparent; padding: 2px 0; font-style: italic;"
        )
