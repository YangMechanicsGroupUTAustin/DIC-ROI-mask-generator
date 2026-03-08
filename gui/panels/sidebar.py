"""Left sidebar with configuration panels.

Contains collapsible sections for file paths, model configuration,
spatial smoothing, and temporal smoothing.
"""

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.icons import get_pixmap
from gui.theme import Colors, Fonts
from gui.widgets.collapsible_section import CollapsibleSection
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

    title = QLabel("SAM2 Studio")
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
    threshold_changed = pyqtSignal(float)
    preprocessing_preview_requested = pyqtSignal(object)  # PreprocessingConfig
    preprocessing_preview_reset = pyqtSignal()
    spatial_smooth_requested = pyqtSignal(dict)
    temporal_smooth_requested = pyqtSignal(dict)

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

        # Scrollable content
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

        # --- Model Configuration section ---
        model_section = CollapsibleSection(
            "Model Configuration", icon_name="settings", default_open=True
        )

        self._device_select = SelectField(
            "Device",
            options=["CUDA", "CPU", "MPS"],
            default="CUDA",
            icon_name="cpu",
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

        # --- Preprocessing section ---
        preprocess_section = CollapsibleSection(
            "Preprocessing", icon_name="sliders", default_open=False
        )

        # Gain + Brightness row
        pp_grid1 = QWidget()
        pp_grid1_layout = QGridLayout(pp_grid1)
        pp_grid1_layout.setContentsMargins(0, 0, 0, 0)
        pp_grid1_layout.setSpacing(8)

        self._pp_gain = NumberInput(
            "Gain", default=1.0, min_val=0.1, max_val=5.0,
            step=0.1, decimals=2, icon_name="zap",
        )
        self._pp_gain.setToolTip(
            "Linear multiplier for pixel intensity.\n1.0 = no change."
        )
        pp_grid1_layout.addWidget(self._pp_gain, 0, 0)

        self._pp_brightness = NumberInput(
            "Brightness", default=0, min_val=-255, max_val=255,
            step=1, decimals=0, icon_name="hash",
        )
        self._pp_brightness.setToolTip(
            "Additive brightness offset.\n0 = no change."
        )
        pp_grid1_layout.addWidget(self._pp_brightness, 0, 1)

        preprocess_section.add_widget(pp_grid1)

        # Contrast
        self._pp_contrast = NumberInput(
            "Contrast", default=1.0, min_val=0.0, max_val=5.0,
            step=0.1, decimals=2, icon_name="sigma",
        )
        self._pp_contrast.setToolTip(
            "Contrast factor around midpoint (128).\n1.0 = no change."
        )
        preprocess_section.add_widget(self._pp_contrast)

        # Min/Max Clip row
        pp_grid2 = QWidget()
        pp_grid2_layout = QGridLayout(pp_grid2)
        pp_grid2_layout.setContentsMargins(0, 0, 0, 0)
        pp_grid2_layout.setSpacing(8)

        self._pp_clip_min = NumberInput(
            "Clip Min", default=0, min_val=0, max_val=254,
            step=1, decimals=0, icon_name="hash",
        )
        pp_grid2_layout.addWidget(self._pp_clip_min, 0, 0)

        self._pp_clip_max = NumberInput(
            "Clip Max", default=255, min_val=1, max_val=255,
            step=1, decimals=0, icon_name="hash",
        )
        pp_grid2_layout.addWidget(self._pp_clip_max, 0, 1)

        preprocess_section.add_widget(pp_grid2)

        # CLAHE
        self._pp_clahe_check = QCheckBox("CLAHE (Adaptive Histogram Eq.)")
        self._pp_clahe_check.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent;"
        )
        preprocess_section.add_widget(self._pp_clahe_check)

        pp_grid3 = QWidget()
        pp_grid3_layout = QGridLayout(pp_grid3)
        pp_grid3_layout.setContentsMargins(0, 0, 0, 0)
        pp_grid3_layout.setSpacing(8)

        self._pp_clahe_clip = NumberInput(
            "CLAHE Clip", default=2.0, min_val=0.1, max_val=40.0,
            step=0.5, decimals=1, icon_name="gauge",
        )
        pp_grid3_layout.addWidget(self._pp_clahe_clip, 0, 0)

        self._pp_clahe_tile = NumberInput(
            "Tile Size", default=8, min_val=2, max_val=32,
            step=1, decimals=0, icon_name="hash",
        )
        pp_grid3_layout.addWidget(self._pp_clahe_tile, 0, 1)

        preprocess_section.add_widget(pp_grid3)

        # Gaussian Smooth
        self._pp_gaussian = NumberInput(
            "Gaussian Sigma", default=0.0, min_val=0.0, max_val=10.0,
            step=0.1, decimals=1, icon_name="waves",
        )
        self._pp_gaussian.setToolTip("Gaussian blur sigma. 0 = disabled.")
        preprocess_section.add_widget(self._pp_gaussian)

        # Bilateral Filter
        self._pp_bilateral_check = QCheckBox("Bilateral Filter (Edge-Preserving)")
        self._pp_bilateral_check.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent;"
        )
        preprocess_section.add_widget(self._pp_bilateral_check)

        # Binary Threshold
        self._pp_threshold_check = QCheckBox("Binary Threshold")
        self._pp_threshold_check.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_BASE}px; "
            f"background: transparent;"
        )
        preprocess_section.add_widget(self._pp_threshold_check)

        pp_grid4 = QWidget()
        pp_grid4_layout = QGridLayout(pp_grid4)
        pp_grid4_layout.setContentsMargins(0, 0, 0, 0)
        pp_grid4_layout.setSpacing(8)

        self._pp_threshold_val = NumberInput(
            "Threshold", default=127, min_val=0, max_val=255,
            step=1, decimals=0, icon_name="hash",
        )
        pp_grid4_layout.addWidget(self._pp_threshold_val, 0, 0)

        self._pp_threshold_method = SelectField(
            "Method",
            options=["Fixed", "Otsu", "Adaptive"],
            default="Fixed",
        )
        pp_grid4_layout.addWidget(self._pp_threshold_method, 0, 1)

        preprocess_section.add_widget(pp_grid4)

        # Preview + Reset buttons
        self._pp_preview_btn = _create_accent_button("Preview Preprocessing")
        self._pp_preview_btn.clicked.connect(self._on_preview_preprocessing)
        preprocess_section.add_widget(self._pp_preview_btn)

        self._pp_reset_btn = QPushButton("Reset to Original")
        self._pp_reset_btn.setMinimumHeight(30)
        self._pp_reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pp_reset_btn.clicked.connect(
            lambda: self.preprocessing_preview_reset.emit()
        )
        preprocess_section.add_widget(self._pp_reset_btn)

        scroll_layout.addWidget(preprocess_section)

        # --- Spatial Smoothing section ---
        spatial_section = CollapsibleSection(
            "Spatial Smoothing", icon_name="waves", default_open=False
        )

        # 2-column grid: Iterations + dt
        spatial_grid = QWidget()
        spatial_grid_layout = QGridLayout(spatial_grid)
        spatial_grid_layout.setContentsMargins(0, 0, 0, 0)
        spatial_grid_layout.setSpacing(8)

        self._spatial_iterations = NumberInput(
            "Iterations", default=50, min_val=1, max_val=500,
            step=1, decimals=0, icon_name="hash",
        )
        spatial_grid_layout.addWidget(self._spatial_iterations, 0, 0)

        self._spatial_dt = NumberInput(
            "dt", default=0.1, min_val=0.001, max_val=1.0,
            step=0.01, decimals=3, icon_name="timer",
        )
        spatial_grid_layout.addWidget(self._spatial_dt, 0, 1)

        spatial_section.add_widget(spatial_grid)

        self._spatial_kappa = NumberInput(
            "Lambda / Kappa", default=30.0, min_val=0.1, max_val=200.0,
            step=0.1, decimals=1, icon_name="sigma",
        )
        spatial_section.add_widget(self._spatial_kappa)

        self._spatial_option = SelectField(
            "Option",
            options=["Option 1 (exponential)", "Option 2 (rational)"],
            default="Option 1 (exponential)",
        )
        spatial_section.add_widget(self._spatial_option)

        self._spatial_btn = _create_accent_button("Apply Spatial Smooth")
        self._spatial_btn.clicked.connect(self._on_spatial_smooth)
        spatial_section.add_widget(self._spatial_btn)

        scroll_layout.addWidget(spatial_section)

        # --- Temporal Smoothing section ---
        temporal_section = CollapsibleSection(
            "Temporal Smoothing", icon_name="sliders", default_open=False
        )

        # 2-column grid: Var Threshold + Neighbors
        temporal_grid = QWidget()
        temporal_grid_layout = QGridLayout(temporal_grid)
        temporal_grid_layout.setContentsMargins(0, 0, 0, 0)
        temporal_grid_layout.setSpacing(8)

        self._temporal_var = NumberInput(
            "Var Threshold", default=50000, min_val=0, max_val=999999,
            step=1000, decimals=0, icon_name="gauge",
        )
        temporal_grid_layout.addWidget(self._temporal_var, 0, 0)

        self._temporal_neighbors = NumberInput(
            "Neighbors", default=2, min_val=1, max_val=10,
            step=1, decimals=0, icon_name="hash",
        )
        temporal_grid_layout.addWidget(self._temporal_neighbors, 0, 1)

        temporal_section.add_widget(temporal_grid)

        self._temporal_sigma = NumberInput(
            "Sigma", default=2.0, min_val=0.1, max_val=20.0,
            step=0.1, decimals=1, icon_name="sigma",
        )
        temporal_section.add_widget(self._temporal_sigma)

        self._temporal_btn = _create_accent_button("Apply Temporal Smooth")
        self._temporal_btn.clicked.connect(self._on_temporal_smooth)
        temporal_section.add_widget(self._temporal_btn)

        scroll_layout.addWidget(temporal_section)

        # Spacer to push footer down
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll, 1)

        # Version footer
        version_label = QLabel(
            f"SAM2 Studio v2.0.0 | PyTorch {_get_pytorch_version()}"
        )
        version_label.setStyleSheet(
            f"color: {Colors.TEXT_DIM}; font-size: {Fonts.SIZE_SM}px; "
            f"padding: 8px 12px; border-top: 1px solid {Colors.BORDER}; "
            f"background: transparent;"
        )
        main_layout.addWidget(version_label)

    # --- Public API ---

    def set_device_options(self, options: list[str], default: str = "") -> None:
        """Update available device options (called by DeviceManager)."""
        self._device_select.set_options(options)
        if default:
            self._device_select.set_value(default)

    def set_input_path(self, path: str) -> None:
        """Set input directory path."""
        self._input_path.set_path(path)

    def set_output_path(self, path: str) -> None:
        """Set output directory path."""
        self._output_path.set_path(path)

    def get_preprocessing_config(self):
        """Build PreprocessingConfig from current sidebar values."""
        from core.preprocessing import PreprocessingConfig
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
            threshold_enabled=self._pp_threshold_check.isChecked(),
            threshold_value=int(self._pp_threshold_val.value()),
            threshold_method=self._pp_threshold_method.value().lower(),
        )

    # --- Private slots ---

    def _on_preview_preprocessing(self) -> None:
        config = self.get_preprocessing_config()
        self.preprocessing_preview_requested.emit(config)

    def _on_spatial_smooth(self) -> None:
        params = {
            "iterations": int(self._spatial_iterations.value()),
            "dt": self._spatial_dt.value(),
            "kappa": self._spatial_kappa.value(),
            "option": 1 if "1" in self._spatial_option.value() else 2,
        }
        self.spatial_smooth_requested.emit(params)

    def _on_temporal_smooth(self) -> None:
        params = {
            "variance_threshold": int(self._temporal_var.value()),
            "neighbors": int(self._temporal_neighbors.value()),
            "sigma": self._temporal_sigma.value(),
        }
        self.temporal_smooth_requested.emit(params)
