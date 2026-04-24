# DIC Mask Generator v2.0 - Complete Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild the SAM2 Mask Generator from a Tkinter prototype into an industrial-grade PyQt6 desktop application with professional dark theme, MVC architecture, and improved algorithms.

**Architecture:** MVC pattern with PyQt6 Signals/Slots. Central AppState manages all shared state. Controllers mediate between GUI panels and core processing. QThread workers handle all heavy computation. Command pattern for undo/redo.

**Tech Stack:** Python 3.10+, PyQt6, OpenCV, NumPy, PyTorch, SAM2, SciPy

**Reference Design:** `GUI_reference/GUI_example.png` and `GUI_reference/Figma_files/`

---

## Target Directory Structure

```
Mask_generater/
├── main.py                          # Entry point (minimal ~30 lines)
├── setup.py                         # Package config (updated deps)
├── requirements.txt                 # Flat dependency list
│
├── gui/                             # PyQt6 GUI layer
│   ├── __init__.py
│   ├── main_window.py               # QMainWindow assembly
│   ├── theme.py                     # Dark theme QSS + color constants
│   ├── icons.py                     # SVG icon paths & resource helpers
│   │
│   ├── widgets/                     # Reusable custom widgets
│   │   ├── __init__.py
│   │   ├── collapsible_section.py   # Sidebar accordion section
│   │   ├── select_field.py          # Labeled combo box
│   │   ├── number_input.py          # Labeled number entry
│   │   ├── slider_input.py          # Labeled slider with value
│   │   ├── tool_button.py           # Toolbar icon button + tooltip
│   │   └── path_selector.py         # Directory path + Browse button
│   │
│   ├── panels/                      # Major UI sections
│   │   ├── __init__.py
│   │   ├── sidebar.py               # Left sidebar (config panels)
│   │   ├── toolbar.py               # Top toolbar (tools + actions)
│   │   ├── canvas_area.py           # 3-panel container + zoom bar
│   │   ├── canvas_panel.py          # Single image viewer (zoom/pan/annotate)
│   │   ├── frame_navigator.py       # Bottom frame nav bar
│   │   └── status_bar.py            # Bottom status bar
│   │
│   └── dialogs/                     # Modal dialogs
│       ├── __init__.py
│       └── error_dialog.py          # Styled error display
│
├── controllers/                     # Business logic coordination
│   ├── __init__.py
│   ├── app_state.py                 # Central state (QObject + signals)
│   ├── annotation_controller.py     # Point management + undo/redo
│   ├── processing_controller.py     # Mask generation orchestration
│   └── smoothing_controller.py      # Post-processing orchestration
│
├── core/                            # Processing algorithms (refactored)
│   ├── __init__.py
│   ├── image_processing.py          # Image I/O utilities (improved)
│   ├── mask_generator.py            # SAM2 predictor lifecycle wrapper
│   ├── spatial_smoothing.py         # Fixed Perona-Malik diffusion
│   ├── temporal_smoothing.py        # Improved temporal filtering
│   └── annotation_config.py         # Config persistence (v2 schema)
│
├── utils/                           # Shared utilities
│   ├── __init__.py
│   ├── device_manager.py            # GPU detection + VRAM monitoring
│   └── logging_config.py            # Structured logging
│
├── resources/                       # Static assets
│   └── icons/                       # SVG icons for toolbar/sidebar
│
├── sam2/                            # SAM2 model (UNCHANGED)
├── checkpoints/                     # Model weights
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── test_app_state.py
│   ├── test_annotation_controller.py
│   ├── test_image_processing.py
│   ├── test_spatial_smoothing.py
│   ├── test_temporal_smoothing.py
│   ├── test_mask_generator.py
│   └── test_device_manager.py
│
└── docs/
    └── plans/
```

## Color Theme (from Figma reference)

```python
COLORS = {
    "bg_darkest":    "#0c0d12",   # Canvas background
    "bg_dark":       "#0f1117",   # Main background
    "bg_medium":     "#13141b",   # Sidebar, toolbar, panels
    "bg_light":      "#16171f",   # Panel headers
    "bg_input":      "#1a1b23",   # Input fields, dropdowns

    "border":        "rgba(255,255,255,0.06)",
    "border_hover":  "rgba(255,255,255,0.12)",

    "primary":       "#6366f1",   # Indigo - main accent
    "primary_hover": "#818cf8",
    "primary_glow":  "rgba(99,102,241,0.20)",

    "success":       "#10b981",   # Emerald - foreground/apply
    "success_bg":    "rgba(16,185,129,0.20)",
    "danger":        "#ef4444",   # Red - clear/stop/background
    "danger_bg":     "rgba(239,68,68,0.10)",

    "text_primary":  "#e4e4e7",   # Zinc-200
    "text_secondary":"#a1a1aa",   # Zinc-400
    "text_muted":    "#71717a",   # Zinc-500
    "text_dim":      "#52525b",   # Zinc-600

    "fg_point":      "#10b981",   # Foreground annotation points
    "bg_point":      "#f43f5e",   # Background annotation points
}
```

## State Machine

```
                   ┌─────────────────────────────────────┐
                   │                                     │
    ┌──────┐   load images   ┌────────────┐   start   ┌────────────┐
    │ Init │ ──────────────> │ Annotating │ ────────> │ Processing │
    └──────┘                 └────────────┘           └────────────┘
                                  ^  │                     │
                                  │  │ add correction      │ done
                                  │  v                     v
                             ┌────────────┐         ┌──────────────┐
                             │ Correction │ <────── │  Reviewing   │
                             └────────────┘         └──────────────┘
                                                          │
                                                          │ smooth
                                                          v
                                                   ┌──────────────┐
                                                   │PostProcessing│
                                                   └──────────────┘
```

---

## Phase 1: Foundation

### Task 1: Project Setup & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `gui/__init__.py`
- Create: `gui/widgets/__init__.py`
- Create: `gui/panels/__init__.py`
- Create: `gui/dialogs/__init__.py`
- Create: `controllers/__init__.py`
- Create: `resources/icons/.gitkeep`
- Create: `tests/__init__.py`

**Step 1:** Create `requirements.txt` with PyQt6 added:
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.4
tqdm>=4.66.1
hydra-core>=1.3.2
iopath>=0.1.10
omegaconf>=2.3.0
pillow>=9.4.0
opencv-python>=4.7.0
scipy>=1.10.0
PyQt6>=6.6.0
```

**Step 2:** Create all `__init__.py` files (empty) for the new packages.

**Step 3:** Install PyQt6: `pip install PyQt6>=6.6.0`

**Step 4:** Commit: `feat: scaffold v2 project structure with PyQt6 dependency`

---

### Task 2: Theme & Style System

**Files:**
- Create: `gui/theme.py`

**Implementation:** Central QSS stylesheet generator + color constants. All colors from Figma reference.

Key components:
- `COLORS` dict (as defined above)
- `FONTS` dict with JetBrains Mono for numbers, Inter/Segoe UI for text
- `generate_stylesheet()` function returning complete QSS string
- QSS must cover: QMainWindow, QWidget, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QSlider, QProgressBar, QLabel, QScrollArea, QSplitter, QToolTip
- Special classes: `.toolbar-button`, `.toolbar-button-active`, `.primary-button`, `.danger-button`, `.success-button`
- Scrollbar styling (thin, dark theme)

**Step 5:** Commit: `feat: add dark theme stylesheet system`

---

### Task 3: SVG Icon System

**Files:**
- Create: `gui/icons.py`
- Create: `resources/icons/` (SVG files)

**Implementation:** We need icons for:
- Toolbar: select (cursor), draw (pencil), erase, foreground dot, background dot, clear (trash), undo, redo, save, load, add-correction, apply-correction, play, stop
- Sidebar: folder, settings, brain, cpu, zap, waves, sliders, hash, sigma, timer, gauge, chevron-down, chevron-right
- Canvas: eye, eye-off, maximize, zoom-in, zoom-out, reset, grid, layers, image-placeholder
- Status: circle (status dot), cpu, hard-drive, clock

Use inline SVG strings (no external file dependency). `icons.py` provides `get_icon(name, color, size) -> QIcon` helper.

**Step 6:** Commit: `feat: add SVG icon system`

---

### Task 4: Reusable Widget Library

**Files:**
- Create: `gui/widgets/collapsible_section.py`
- Create: `gui/widgets/select_field.py`
- Create: `gui/widgets/number_input.py`
- Create: `gui/widgets/slider_input.py`
- Create: `gui/widgets/tool_button.py`
- Create: `gui/widgets/path_selector.py`

#### 4a: CollapsibleSection
Accordion panel for sidebar. Click header to expand/collapse content area with animation.
```python
class CollapsibleSection(QWidget):
    def __init__(self, title: str, icon_name: str, default_open: bool = True):
        # Header: icon + title + chevron (rotate on toggle)
        # Content: QWidget container, show/hide with animation
        # Border-bottom separator
```

#### 4b: SelectField
Labeled dropdown matching Figma style.
```python
class SelectField(QWidget):
    value_changed = pyqtSignal(str)
    def __init__(self, label: str, options: list[str], icon_name: str = ""):
        # Layout: label (11px uppercase zinc-500) above combo box
        # Combo box: dark bg, border, icon prefix
```

#### 4c: NumberInput
Labeled number entry with optional unit suffix.
```python
class NumberInput(QWidget):
    value_changed = pyqtSignal(float)
    def __init__(self, label: str, default: float, icon_name: str = "",
                 unit: str = "", min_val: float = 0, max_val: float = 99999):
        # Layout: label above input field
        # Input: QDoubleSpinBox styled dark, monospace font
```

#### 4d: SliderInput
Labeled slider with gradient track and value display.
```python
class SliderInput(QWidget):
    value_changed = pyqtSignal(float)
    def __init__(self, label: str, default: float, min_val: float, max_val: float,
                 step: float = 0.01, decimals: int = 2):
        # Layout: label + value on same row, slider below
        # Custom painted gradient track (indigo gradient)
        # Value display in indigo monospace
```

#### 4e: ToolButton
Toolbar button with icon, optional label, optional shortcut badge, tooltip.
```python
class ToolButton(QPushButton):
    def __init__(self, icon_name: str, label: str, shortcut: str = "",
                 variant: str = "default"):
        # Variants: default, primary, danger, success
        # Active state: bg-white/8%, border-white/12%
        # Tooltip on hover
        # Shortcut badge (small text, bg-white/5%)
```

#### 4f: PathSelector
Directory path display + Browse button.
```python
class PathSelector(QWidget):
    path_changed = pyqtSignal(str)
    def __init__(self, label: str):
        # Layout: label above (path_display + browse_button)
        # path_display: truncated text, dark bg
        # browse_button: indigo accent style
        # Opens QFileDialog.getExistingDirectory on click
```

**Step 7:** Commit: `feat: add reusable widget library`

---

## Phase 2: GUI Panels

### Task 5: Sidebar Panel

**Files:**
- Create: `gui/panels/sidebar.py`

**Implementation:** 300px wide left sidebar with:
1. **Logo header** — "DIC Mask Generator" + "Mask Generator v2.0" + gradient icon
2. **File Paths section** (default open) — Input/Output PathSelectors
3. **Model Configuration section** (default open) — Device, Model, Intermediate Format (SelectFields) + Mask Threshold (SliderInput)
4. **Spatial Smoothing section** (default closed) — Iterations, dt (2-col grid), Lambda, "Apply Spatial Smooth" button
5. **Temporal Smoothing section** (default closed) — Var Threshold, Neighbors (2-col grid), Sigma, "Apply Temporal Smooth" button
6. **Version footer** — "DIC Mask Generator v2.0.0 | PyTorch {version}"

Sidebar scrolls vertically if content exceeds height.

Signals emitted:
- `input_dir_changed(str)`
- `output_dir_changed(str)`
- `device_changed(str)`
- `model_changed(str)`
- `format_changed(str)`
- `threshold_changed(float)`
- `spatial_smooth_requested(int iterations, float dt, float lam)`
- `temporal_smooth_requested(float var_thresh, int neighbors, float sigma)`

**Step 8:** Commit: `feat: add sidebar panel`

---

### Task 6: Toolbar Panel

**Files:**
- Create: `gui/panels/toolbar.py`

**Implementation:** 56px height horizontal toolbar with:
1. **Tool group** (rounded container) — Select | Draw Points | Erase — mutually exclusive toggle
2. **Divider**
3. **Mode toggle** (rounded container) — Foreground (emerald) | Background (rose) — mutually exclusive
4. **Divider**
5. **Actions** — Clear (danger), Undo (Ctrl+Z), Redo (Ctrl+Y)
6. **Divider**
7. **File ops** — Save (Ctrl+S), Load (Ctrl+O)
8. **Spacer** (flex)
9. **Processing controls** — Add Correction, Apply Correction (success), Force Re-process (checkbox), Start Processing (primary, prominent) / Stop (danger, replaces Start when processing)

Signals emitted:
- `tool_changed(str)` — "select", "draw", "erase"
- `mode_changed(str)` — "foreground", "background"
- `clear_requested()`
- `undo_requested()`
- `redo_requested()`
- `save_requested()`
- `load_requested()`
- `add_correction_requested()`
- `apply_correction_requested()`
- `start_processing_requested()`
- `stop_processing_requested()`
- `force_reprocess_changed(bool)`

**Step 9:** Commit: `feat: add toolbar panel`

---

### Task 7: Canvas Panel (Image Viewer)

**Files:**
- Create: `gui/panels/canvas_panel.py`

**Implementation:** Single image viewer panel based on QGraphicsView + QGraphicsScene.

Features:
- **Header bar**: title + badge + eye toggle (visibility) + maximize button
- **Image display**: QGraphicsPixmapItem, centered in view
- **Zoom**: Mouse wheel zoom (10% increments), fit-to-view
- **Pan**: Middle-mouse-drag OR hold Space + left-drag
- **Annotation overlay** (Original panel only):
  - Foreground points: emerald circles with white outline
  - Background points: rose circles with white outline
  - Draggable: click-and-drag existing points to reposition
  - Click to place new point (when draw tool active)
  - Hover: point highlights, crosshair cursor shows pixel coordinates
- **Grid overlay**: optional subtle grid (toggled from zoom bar)
- **Placeholder**: when no image loaded, show icon + text + "Load Images" button

Coordinate system:
- All point coordinates stored in image-space (not view-space)
- Transform between view↔image on zoom/pan changes
- Points rendered as QGraphicsEllipseItems in a separate layer

```python
class CanvasPanel(QWidget):
    point_added = pyqtSignal(float, float)      # image-space x, y
    point_moved = pyqtSignal(int, float, float)  # index, new_x, new_y
    point_removed = pyqtSignal(int)              # index (erase tool)
    load_images_requested = pyqtSignal()

    def set_image(self, image: np.ndarray): ...
    def set_points(self, points: list, labels: list): ...
    def set_crosshair_visible(self, visible: bool): ...
    def fit_to_view(self): ...
    def set_zoom(self, percent: int): ...
    def get_zoom(self) -> int: ...
```

**Step 10:** Commit: `feat: add interactive canvas panel with zoom/pan/annotate`

---

### Task 8: Canvas Area (3-Panel Container)

**Files:**
- Create: `gui/panels/canvas_area.py`

**Implementation:** Container holding 3 CanvasPanels + zoom bar.

Layout:
1. **Zoom bar** (top): "3 Panels" label | spacer | zoom-out | zoom% | zoom-in | divider | reset | grid-toggle
2. **3 panels** (center, horizontal splitter):
   - Original Frame (badge: "Input", flex: 2) — receives annotations
   - Mask Preview (badge: "Segmentation", flex: 1)
   - Overlay (badge: "Result", flex: 1)

Zoom controls apply to all 3 panels simultaneously (synchronized zoom level).
Pan is per-panel (independent).

```python
class CanvasArea(QWidget):
    def set_original_image(self, image: np.ndarray): ...
    def set_mask_image(self, mask: np.ndarray): ...
    def set_overlay_image(self, overlay: np.ndarray): ...
    def set_annotation_points(self, points: list, labels: list): ...
    def get_zoom(self) -> int: ...
    def set_zoom(self, percent: int): ...
```

**Step 11:** Commit: `feat: add 3-panel canvas area with synchronized zoom`

---

### Task 9: Frame Navigator

**Files:**
- Create: `gui/panels/frame_navigator.py`

**Implementation:** 48px height bottom bar with:
1. **Frame range** — "START" label + QSpinBox(1) + "END" label + QSpinBox(total)
2. **Divider**
3. **Navigation** — Prev button (<) + "Frame {N} / {total}" display + Next button (>)
4. **Divider**
5. **Preview slider** — "PREVIEW" label + full-width slider with indigo gradient track
   - Slider value synced with frame display
   - Left/Right arrow keys navigate frames

Signals:
- `frame_changed(int)` — 1-based frame index
- `start_frame_changed(int)`
- `end_frame_changed(int)`

```python
class FrameNavigator(QWidget):
    def set_total_frames(self, total: int): ...
    def set_current_frame(self, frame: int): ...
    def get_frame_range(self) -> tuple[int, int]: ...
```

**Step 12:** Commit: `feat: add frame navigator bar`

---

### Task 10: Status Bar

**Files:**
- Create: `gui/panels/status_bar.py`

**Implementation:** 28px height bottom bar with:
1. **Status indicator** — green/yellow/red circle + "Ready"/"Processing"/"Error" text
2. **Divider**
3. **Device info** — CPU icon + "CUDA: NVIDIA RTX 4090" (or detected device name)
4. **Divider**
5. **VRAM usage** — HDD icon + "VRAM: 2.4 / 24.0 GB" + mini progress bar (16px wide)
6. **Spacer**
7. **Elapsed time** — Clock icon + "Elapsed: 00:00:00"

VRAM updates via QTimer (every 2 seconds when processing, every 10 seconds idle).
Elapsed timer runs during processing.

```python
class StatusBar(QWidget):
    def set_status(self, status: str, level: str = "ready"): ...
    def set_device_info(self, name: str): ...
    def set_vram_usage(self, used_gb: float, total_gb: float): ...
    def start_timer(self): ...
    def stop_timer(self): ...
    def reset_timer(self): ...
```

**Step 13:** Commit: `feat: add status bar with VRAM monitoring`

---

### Task 11: Main Window Assembly

**Files:**
- Create: `gui/main_window.py`
- Create: new `main.py` (replace old Tkinter entry point)

**Implementation:**

`main_window.py`:
```python
class MainWindow(QMainWindow):
    def __init__(self):
        # Layout (matches App.tsx):
        # QVBoxLayout:
        #   QHBoxLayout:
        #     Sidebar (fixed 300px)
        #     QVBoxLayout:
        #       Toolbar
        #       CanvasArea (expand)
        #       FrameNavigator
        #   StatusBar

    def _setup_shortcuts(self):
        # Ctrl+Z: undo
        # Ctrl+Y: redo
        # Ctrl+S: save config
        # Ctrl+O: load config
        # Left/Right: prev/next frame
        # Space: toggle draw mode
        # Escape: stop processing
        # V: select tool
        # D: draw tool
        # E: erase tool
        # F: fit to view
        # +/-: zoom in/out
```

New `main.py`:
```python
import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow
from gui.theme import generate_stylesheet

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(generate_stylesheet())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
```

**Step 14:** Rename old `main.py` → `main_tkinter_legacy.py`

**Step 15:** Commit: `feat: assemble main window with all panels`

---

## Phase 3: Controllers & State

### Task 12: Central State Management

**Files:**
- Create: `controllers/app_state.py`

**Implementation:** Single source of truth for all application state. Uses PyQt6 signals for reactivity.

```python
class AppState(QObject):
    # --- Signals ---
    state_changed = pyqtSignal(str)           # state machine transitions
    input_dir_changed = pyqtSignal(str)
    output_dir_changed = pyqtSignal(str)
    image_files_changed = pyqtSignal(list)
    current_frame_changed = pyqtSignal(int)
    frame_range_changed = pyqtSignal(int, int)
    points_changed = pyqtSignal(list, list)   # points, labels
    device_changed = pyqtSignal(str)
    model_changed = pyqtSignal(str)
    threshold_changed = pyqtSignal(float)
    processing_progress = pyqtSignal(int, int, str)  # current, total, message
    processing_finished = pyqtSignal()
    processing_error = pyqtSignal(str)
    current_images_changed = pyqtSignal()     # original, mask, overlay updated
    vram_updated = pyqtSignal(float, float)   # used, total

    # --- State ---
    class State(Enum):
        INIT = "init"
        ANNOTATING = "annotating"
        PROCESSING = "processing"
        REVIEWING = "reviewing"
        CORRECTION = "correction"
        POST_PROCESSING = "post_processing"

    # Properties with signal emission on change:
    # input_dir, output_dir, image_files, current_frame,
    # start_frame, end_frame, points, labels,
    # device, model_name, model_cfg, checkpoint,
    # threshold, intermediate_format, force_reprocess,
    # current_original (ndarray), current_mask (ndarray), current_overlay (ndarray),
    # inference_state, predictor

    MODEL_REGISTRY = {
        "SAM2 Hiera Large":     ("sam2.1_hiera_l.yaml",   "sam2.1_hiera_large.pt"),
        "SAM2 Hiera Base Plus": ("sam2.1_hiera_b+.yaml",  "sam2.1_hiera_base_plus.pt"),
        "SAM2 Hiera Small":     ("sam2.1_hiera_s.yaml",   "sam2.1_hiera_small.pt"),
        "SAM2 Hiera Tiny":      ("sam2.1_hiera_t.yaml",   "sam2.1_hiera_tiny.pt"),
    }
```

**Step 16:** Commit: `feat: add central state management with signals`

---

### Task 13: Annotation Controller (Undo/Redo)

**Files:**
- Create: `controllers/annotation_controller.py`

**Implementation:** Command pattern for full undo/redo stack.

```python
class Command(ABC):
    @abstractmethod
    def execute(self, state: AppState): ...
    @abstractmethod
    def undo(self, state: AppState): ...

class AddPointCommand(Command):
    def __init__(self, x: float, y: float, label: int): ...

class MovePointCommand(Command):
    def __init__(self, index: int, old_x: float, old_y: float,
                 new_x: float, new_y: float): ...

class RemovePointCommand(Command):
    def __init__(self, index: int): ...

class ClearPointsCommand(Command):
    """Stores snapshot for undo."""

class AnnotationController(QObject):
    def __init__(self, state: AppState):
        self._undo_stack: list[Command] = []
        self._redo_stack: list[Command] = []
        self._max_history = 100

    def add_point(self, x: float, y: float, label: int): ...
    def move_point(self, index: int, new_x: float, new_y: float): ...
    def remove_point(self, index: int): ...
    def clear_points(self): ...
    def undo(self): ...
    def redo(self): ...
    def can_undo(self) -> bool: ...
    def can_redo(self) -> bool: ...

    def save_config(self, filepath: str): ...
    def load_config(self, filepath: str): ...
```

**Step 17:** Commit: `feat: add annotation controller with undo/redo`

---

### Task 14: Device Manager

**Files:**
- Create: `utils/device_manager.py`

**Implementation:**
```python
class DeviceManager:
    @staticmethod
    def detect_available_devices() -> list[str]:
        """Return list like ["CUDA", "CPU"] or ["MPS", "CPU"]."""

    @staticmethod
    def get_device_string(choice: str) -> str:
        """Convert "CUDA" -> "cuda", "CPU" -> "cpu", "MPS" -> "mps"."""

    @staticmethod
    def get_gpu_name() -> str:
        """Return e.g. 'NVIDIA RTX 4090' or 'Apple M2 Pro' or 'CPU'."""

    @staticmethod
    def get_vram_usage() -> tuple[float, float]:
        """Return (used_gb, total_gb). Returns (0, 0) for CPU."""

    @staticmethod
    def get_torch_version() -> str:
        """Return PyTorch version string."""
```

**Step 18:** Commit: `feat: add device manager with VRAM monitoring`

---

### Task 15: Logging Configuration

**Files:**
- Create: `utils/logging_config.py`

**Implementation:**
```python
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Configure structured logging to file + memory buffer.

    - File: logs/sam2studio_YYYYMMDD_HHMMSS.log
    - Memory buffer: last 500 entries for in-app log viewer
    - Format: [YYYY-MM-DD HH:MM:SS] [LEVEL] [module] message
    """
```

Replace all `print()` statements across core/ with proper `logger.info/warning/error`.

**Step 19:** Commit: `feat: add structured logging system`

---

## Phase 4: Core Refactoring

### Task 16: Image Processing Improvements

**Files:**
- Modify: `core/image_processing.py`
- Create: `tests/test_image_processing.py`

**Changes:**
1. Add logging (replace all `print()`)
2. DRY: extract shared normalization logic from `convert_to_jpeg` and `convert_to_png` into `_normalize_image(img) -> np.ndarray`
3. Add `convert_image(img_path, output_path, format="jpeg", quality=95)` — unified conversion
4. Add `load_image_for_display(path) -> QImage` — converts any format to QImage for Qt display
5. Add `numpy_to_qimage(arr: np.ndarray) -> QImage` — RGB ndarray to QImage
6. Make `create_overlay` configurable: color parameter (not just red), adjustable alpha

**Tests:**
- Test normalization of 8/16/32-bit and float images
- Test JPEG and PNG conversion roundtrip
- Test overlay with different colors
- Test natural sort ordering
- Test graceful failure on corrupt files

**Step 20:** Commit: `refactor: improve image processing with unified conversion`

---

### Task 17: Mask Generator Service

**Files:**
- Create: `core/mask_generator.py`
- Create: `tests/test_mask_generator.py`

**Implementation:** Clean wrapper around SAM2 predictor lifecycle.

```python
class MaskGenerator:
    """Manages SAM2 video predictor lifecycle and inference.

    Architecture note: designed for single-object now,
    but obj_id parameter allows future multi-object support.
    """

    def __init__(self):
        self._predictor = None
        self._inference_state = None
        self._config_key = None  # (checkpoint, cfg, device) tuple

    def initialize(self, model_name: str, device: str,
                   progress_callback=None) -> None:
        """Load model. Reuses if config unchanged."""

    def set_video(self, image_dir: str) -> None:
        """Initialize inference state from image directory."""

    def add_points(self, frame_idx: int, points: np.ndarray,
                   labels: np.ndarray, obj_id: int = 1) -> np.ndarray:
        """Add annotation points. Returns preview mask for frame."""

    def propagate(self, threshold: float = 0.0,
                  progress_callback=None) -> dict[int, np.ndarray]:
        """Run propagation. Returns {frame_idx: binary_mask}."""

    def add_correction(self, frame_idx: int, points: np.ndarray,
                       labels: np.ndarray, obj_id: int = 1) -> None:
        """Add correction points at specific frame."""

    def propagate_from(self, frame_idx: int, threshold: float = 0.0,
                       progress_callback=None) -> dict[int, np.ndarray]:
        """Re-propagate from correction frame onward."""

    def cleanup(self) -> None:
        """Release model, clear GPU cache."""

    @property
    def is_initialized(self) -> bool: ...
```

**Tests:** Mock SAM2 predictor, test lifecycle management, config reuse, cleanup.

**Step 21:** Commit: `feat: add mask generator service with clean API`

---

### Task 18: Fix Spatial Smoothing Algorithm

**Files:**
- Modify: `core/spatial_smoothing.py`
- Create: `tests/test_spatial_smoothing.py`

**Issues to fix:**
1. Current implementation applies Gaussian blur INSIDE the iteration loop — this fights against the anisotropic diffusion and destroys edges
2. Gradient computation uses `np.gradient()` (central difference) instead of forward differences
3. Diffusion equation `smoothed + dt * (laplacian_d * laplacian_s)` is incorrect — should be divergence of (D * gradient)

**Fixed implementation:**
```python
def perona_malik_smooth(
    image: np.ndarray,
    num_iterations: int = 50,
    dt: float = 0.1,
    kappa: float = 30.0,
    option: int = 1,
    post_gaussian_sigma: float = 0.0,
) -> np.ndarray:
    """Perona-Malik anisotropic diffusion (corrected implementation).

    Args:
        image: Binary mask (uint8, 0 or 255).
        num_iterations: Diffusion iterations.
        dt: Time step (must be <= 0.25 for stability).
        kappa: Conductance parameter (gradient magnitude threshold).
        option: 1 = exp(-|grad|^2/kappa^2), 2 = 1/(1+(|grad|/kappa)^2)
        post_gaussian_sigma: Optional Gaussian blur AFTER diffusion (0 = off).

    Returns:
        Smoothed binary mask (uint8, 0 or 255).
    """
    u = image.astype(np.float64)
    if u.max() > 1.0:
        u = u / 255.0

    for _ in range(num_iterations):
        # Forward differences (North, South, East, West)
        dN = np.roll(u, -1, axis=0) - u
        dS = np.roll(u, 1, axis=0) - u
        dE = np.roll(u, -1, axis=1) - u
        dW = np.roll(u, 1, axis=1) - u

        # Diffusion coefficients
        if option == 1:
            cN = np.exp(-(dN / kappa) ** 2)
            cS = np.exp(-(dS / kappa) ** 2)
            cE = np.exp(-(dE / kappa) ** 2)
            cW = np.exp(-(dW / kappa) ** 2)
        else:
            cN = 1.0 / (1.0 + (dN / kappa) ** 2)
            cS = 1.0 / (1.0 + (dS / kappa) ** 2)
            cE = 1.0 / (1.0 + (dE / kappa) ** 2)
            cW = 1.0 / (1.0 + (dW / kappa) ** 2)

        # Update (divergence of flux)
        u = u + dt * (cN * dN + cS * dS + cE * dE + cW * dW)

    # Optional post-smoothing
    if post_gaussian_sigma > 0:
        u = gaussian_filter(u, sigma=post_gaussian_sigma)

    return (u > 0.5).astype(np.uint8) * 255
```

**Parameter changes for sidebar:** rename `lambda` → `kappa`, add `option` dropdown (1 or 2), add `post_gaussian_sigma` (default 0.0, meaning off).

**Tests:**
- Test that edge pixels are preserved better than pure Gaussian
- Test stability (dt <= 0.25 enforced)
- Test roundtrip: binary mask in → binary mask out
- Test with blank/all-white/all-black masks (edge cases)
- Compare old vs new on sample mask (verify improvement)

**Step 22:** Commit: `fix: correct Perona-Malik diffusion implementation`

---

### Task 19: Improve Temporal Smoothing

**Files:**
- Modify: `core/temporal_smoothing.py`
- Create: `tests/test_temporal_smoothing.py`

**Improvements:**
1. **Use float32 instead of float64** — halves memory (4.7GB → 2.3GB for 1080p×300 frames)
2. **Sliding window 3D Gaussian** — don't load entire sequence at once:
   ```python
   def temporal_smooth_chunked(frames, chunk_size=50, overlap=10, sigma=2.0):
       """Process in overlapping chunks to limit memory."""
   ```
3. **Adaptive bad frame detection** — use median absolute deviation instead of fixed threshold:
   ```python
   def detect_bad_frames_adaptive(frames):
       variances = [np.var(f) for f in frames]
       median_var = np.median(variances)
       mad = np.median(np.abs(variances - median_var))
       threshold = median_var + 5 * mad  # adaptive
   ```
4. **Progress callback per-frame** (not just 4 coarse steps)
5. Add `progress_callback(step_name, current, total)` with fine-grained updates

**Tests:**
- Test memory usage with large sequence (assert float32)
- Test bad frame detection: inject known bad frames, verify detection
- Test chunked processing matches full processing (within tolerance)
- Test edge case: all frames identical, single frame, two frames

**Step 23:** Commit: `refactor: improve temporal smoothing memory and accuracy`

---

### Task 20: Annotation Config v2

**Files:**
- Modify: `core/annotation_config.py`
- Create: `tests/test_annotation_config.py`

**Changes:**
- Version bump to "2.0"
- Add smoothing parameters to saved config
- Add backward compatibility: load v1 configs with defaults for new fields
- Add `correction_points` field (separate from initial points)
- Validate schema on load

```python
CONFIG_SCHEMA_V2 = {
    "version": "2.0",
    "saved_at": str,
    "annotation": {
        "points": list,           # Initial annotation points
        "labels": list,
        "correction_points": [],  # Per-frame corrections
    },
    "parameters": {
        "model": str,
        "device": str,
        "threshold": float,
        "start_frame": int,
        "end_frame": int,
        "intermediate_format": str,
    },
    "smoothing": {
        "spatial": {"iterations": 50, "dt": 0.1, "kappa": 30.0, "option": 1},
        "temporal": {"sigma": 2.0, "neighbors": 2},
    }
}
```

**Step 24:** Commit: `refactor: upgrade annotation config to v2 schema`

---

## Phase 5: Processing Controllers

### Task 21: Processing Controller

**Files:**
- Create: `controllers/processing_controller.py`

**Implementation:** Orchestrates the full mask generation pipeline via QThread.

```python
class ProcessingWorker(QThread):
    """Background worker for mask generation."""
    progress = pyqtSignal(int, int, str)  # current, total, message
    frame_processed = pyqtSignal(int, object, object)  # idx, mask, overlay
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, mask_generator, state):
        # Receives all needed data at construction (thread-safe copy)

    def run(self):
        # Stage 1: Convert images (with per-file progress)
        # Stage 2: Initialize model (progress: "Loading model...")
        # Stage 3: Set video + add points
        # Stage 4: Propagate (per-frame progress + emit frame_processed)
        # Stage 5: Save masks

    def stop(self): ...


class ProcessingController(QObject):
    """Coordinates between GUI and processing worker."""

    def __init__(self, state: AppState, mask_generator: MaskGenerator):
        ...

    def start_processing(self): ...
    def stop_processing(self): ...
    def add_correction(self, frame_idx: int, points, labels): ...
    def apply_correction(self): ...
```

Key behaviors:
- Copy state data before starting thread (immutable snapshot)
- Worker emits signals → controller updates AppState → GUI auto-updates
- Stop flag checked between frames (graceful cancellation)
- Error handling: catch exceptions in worker, emit error signal
- Cleanup predictor on finish or error

**Step 25:** Commit: `feat: add processing controller with background worker`

---

### Task 22: Smoothing Controller

**Files:**
- Create: `controllers/smoothing_controller.py`

**Implementation:** Similar pattern to ProcessingController but for post-processing.

```python
class SpatialSmoothWorker(QThread):
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(str)       # output directory
    error = pyqtSignal(str)

class TemporalSmoothWorker(QThread):
    progress = pyqtSignal(int, int, str)  # current, total, step_name
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

class SmoothingController(QObject):
    def start_spatial(self, input_dir: str, output_dir: str,
                      iterations: int, dt: float, kappa: float, option: int): ...
    def start_temporal(self, input_dir: str, output_dir: str,
                       sigma: float, neighbors: int): ...
    def stop(self): ...
```

**Step 26:** Commit: `feat: add smoothing controller with background workers`

---

## Phase 6: Integration & Wiring

### Task 23: Wire Everything Together

**Files:**
- Modify: `gui/main_window.py`

**Implementation:** Connect all signals between panels, controllers, and state.

Signal wiring map:
```
Sidebar.input_dir_changed → AppState.set_input_dir → load images → CanvasArea.set_original
Sidebar.output_dir_changed → AppState.set_output_dir
Sidebar.device_changed → AppState.set_device → StatusBar.set_device_info
Sidebar.model_changed → AppState.set_model
Sidebar.threshold_changed → AppState.set_threshold
Sidebar.spatial_smooth_requested → SmoothingController.start_spatial
Sidebar.temporal_smooth_requested → SmoothingController.start_temporal

Toolbar.tool_changed → CanvasPanel.set_active_tool
Toolbar.mode_changed → AppState.set_point_mode
Toolbar.clear_requested → AnnotationController.clear
Toolbar.undo_requested → AnnotationController.undo
Toolbar.redo_requested → AnnotationController.redo
Toolbar.save_requested → AnnotationController.save_config
Toolbar.load_requested → AnnotationController.load_config
Toolbar.start_processing → ProcessingController.start
Toolbar.stop_processing → ProcessingController.stop
Toolbar.add_correction → enter correction state
Toolbar.apply_correction → ProcessingController.apply_correction

CanvasPanel.point_added → AnnotationController.add_point
CanvasPanel.point_moved → AnnotationController.move_point
CanvasPanel.point_removed → AnnotationController.remove_point

FrameNavigator.frame_changed → AppState.set_current_frame → load + display
FrameNavigator.start_changed → AppState.set_start_frame
FrameNavigator.end_changed → AppState.set_end_frame

AppState.points_changed → CanvasPanel.set_points
AppState.current_images_changed → CanvasArea.update_displays
AppState.processing_progress → StatusBar + FrameNavigator progress
AppState.vram_updated → StatusBar.set_vram

ProcessingController.frame_processed → update mask/overlay display
ProcessingController.finished → state → REVIEWING
ProcessingController.error → ErrorDialog

DeviceManager (QTimer 2s/10s) → AppState.vram_updated
```

State-dependent UI enablement:
```
INIT:           sidebar enabled, toolbar disabled (except file ops), canvas: placeholder
ANNOTATING:     all enabled except stop/apply-correction, canvas: interactive
PROCESSING:     sidebar read-only, toolbar: only stop, canvas: display-only, progress active
REVIEWING:      all enabled, add-correction enabled
CORRECTION:     canvas: interactive, apply-correction enabled
POST_PROCESSING: sidebar read-only, toolbar: only stop, progress active
```

**Step 27:** Commit: `feat: wire all components together`

---

### Task 24: Error Dialog

**Files:**
- Create: `gui/dialogs/error_dialog.py`

**Implementation:** Styled dark-theme error dialog with:
- Error icon + title
- Error message (scrollable for long tracebacks)
- "Copy to clipboard" button
- "OK" button

Used for: model loading failures, GPU OOM, file I/O errors, invalid configurations.

**Step 28:** Commit: `feat: add styled error dialog`

---

## Phase 7: Polish & Edge Cases

### Task 25: Keyboard Shortcuts

**Files:**
- Modify: `gui/main_window.py`

Register all shortcuts:
| Key | Action |
|-----|--------|
| `V` | Select tool |
| `D` | Draw tool |
| `E` | Erase tool |
| `Space` | Toggle draw mode |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` / `Ctrl+Shift+Z` | Redo |
| `Ctrl+S` | Save config |
| `Ctrl+O` | Load config |
| `Left` / `Right` | Prev/next frame |
| `Home` / `End` | First/last frame |
| `F` | Fit to view |
| `Ctrl++` / `Ctrl+-` | Zoom in/out |
| `Ctrl+0` | Reset zoom to 100% |
| `Escape` | Stop processing / exit mode |
| `Delete` | Remove selected point (erase mode) |

**Step 29:** Commit: `feat: add comprehensive keyboard shortcuts`

---

### Task 26: Edge Case Handling

**Files:** Various

Handle these scenarios:
1. **Empty input directory** — show message in canvas placeholder, disable processing
2. **Single frame** — disable temporal smoothing, allow spatial only
3. **Huge images (8K+)** — downsample for display, keep full-res for processing
4. **No GPU available** — auto-fallback to CPU, warn about speed
5. **Model checkpoint missing** — prompt download URL, disable processing
6. **Processing interrupted mid-way** — clean up partial output, reset state
7. **Output dir = Input dir** — warn and block
8. **Corrupt images in sequence** — skip with warning, create placeholder mask
9. **VRAM OOM during inference** — catch, suggest smaller model, cleanup
10. **Config file from v1** — auto-migrate to v2 schema on load
11. **Window resize** — all panels responsive, canvas maintains aspect ratio
12. **Multiple clicks on Start** — debounce, disable button during init
13. **Switching model during processing** — block, show warning

**Step 30:** Commit: `feat: add comprehensive edge case handling`

---

### Task 27: Performance Optimizations

**Files:** Various

1. **Image display**: Use `QPixmap.fromImage()` directly, not matplotlib
2. **Lazy loading**: Only load current frame + 2 neighbors into memory
3. **Image conversion**: Use `ThreadPoolExecutor(max_workers=4)` for parallel I/O
4. **Mask save**: Batch write with thread pool
5. **Canvas rendering**: Only update changed regions (dirty rect optimization)
6. **VRAM polling**: 10s interval when idle, 2s when processing
7. **Frame navigation**: Debounce slider to 50ms (don't load every pixel change)

**Step 31:** Commit: `perf: optimize image loading, display, and I/O`

---

### Task 28: Cleanup Legacy Code

**Files:**
- Rename: `main.py` → `main_tkinter_legacy.py` (keep for reference)
- Remove: `utils/dpi_scaling.py` (Tkinter-specific, no longer needed)
- Update: `setup.py` — add PyQt6, scipy to dependencies
- Update: `README.md` — new installation instructions, screenshots

**Step 32:** Commit: `chore: remove legacy Tkinter code, update dependencies`

---

## Phase 8: Testing

### Task 29: Core Tests

**Files:**
- Create: `tests/test_image_processing.py`
- Create: `tests/test_spatial_smoothing.py`
- Create: `tests/test_temporal_smoothing.py`
- Create: `tests/test_annotation_config.py`
- Create: `tests/test_device_manager.py`

Focus on:
- Algorithm correctness (spatial smoothing edge preservation)
- Memory efficiency (temporal smoothing float32)
- Config backward compatibility (v1 → v2)
- Error handling (corrupt files, missing dirs)
- Edge cases (empty inputs, single frames)

### Task 30: Controller Tests

**Files:**
- Create: `tests/test_app_state.py`
- Create: `tests/test_annotation_controller.py`

Focus on:
- State transitions (valid and invalid)
- Undo/redo stack correctness
- Signal emission on state changes
- Thread safety

**Step 33:** Commit: `test: add core and controller test suites`

---

## Use Case Scenarios Considered

| Scenario | How Handled |
|----------|-------------|
| **First-time user** | Placeholder UI guides to "Load Images", progressive disclosure of options |
| **Quick single-frame mask** | Select dir → draw 2 points → Start → done in seconds |
| **Large sequence (1000+ frames)** | Progress bar with ETA, cancelable, per-frame display updates |
| **High-precision annotation** | Zoom to pixel level, crosshair, drag points, undo/redo |
| **Iterative refinement** | Process → review → add correction → re-propagate → repeat |
| **Batch smoothing** | Process masks → spatial smooth → temporal smooth (pipeline) |
| **Resume after crash** | Config save/load preserves all annotation + parameters |
| **Different image formats** | 8/16/32-bit TIFF, PNG, JPEG, BMP all handled transparently |
| **Low VRAM GPU** | Use Tiny/Small model, VRAM monitoring warns before OOM |
| **CPU-only machine** | Auto-detect, works (slower), no VRAM display |
| **4K/HiDPI display** | PyQt6 handles DPI natively, no manual scaling needed |
| **Multiple monitors** | Standard Qt window management, remembers position |
| **Parameter tuning** | Slider with live value display, defaults for common cases |

---

## Execution Order & Dependencies

```
Task 1  (Setup)          ─┐
Task 2  (Theme)           ├─ Foundation (no dependencies)
Task 3  (Icons)           │
Task 15 (Logging)        ─┘

Task 4  (Widgets)        ─── depends on Task 2, 3

Task 5  (Sidebar)        ─┐
Task 6  (Toolbar)         │
Task 7  (CanvasPanel)     ├─ depends on Task 4
Task 9  (FrameNav)        │
Task 10 (StatusBar)      ─┘

Task 8  (CanvasArea)     ─── depends on Task 7

Task 11 (MainWindow)     ─── depends on Tasks 5,6,8,9,10

Task 12 (AppState)       ─┐
Task 13 (AnnotCtrl)       ├─ can parallel with GUI tasks
Task 14 (DeviceMgr)     ─┘

Task 16 (ImageProc)      ─┐
Task 17 (MaskGen)         │
Task 18 (SpatialFix)     ├─ can parallel with GUI tasks
Task 19 (TemporalFix)    │
Task 20 (ConfigV2)       ─┘

Task 21 (ProcCtrl)       ─── depends on Tasks 12, 17
Task 22 (SmoothCtrl)     ─── depends on Tasks 12, 18, 19

Task 23 (Wiring)         ─── depends on ALL above
Task 24 (ErrorDialog)    ─── depends on Task 2
Task 25 (Shortcuts)      ─── depends on Task 23
Task 26 (EdgeCases)      ─── depends on Task 23
Task 27 (Performance)    ─── depends on Task 23
Task 28 (Cleanup)        ─── depends on Task 23

Task 29 (CoreTests)      ─── depends on Tasks 16-20
Task 30 (CtrlTests)      ─── depends on Tasks 12-13
```

## Estimated Commit Count: ~32 commits across 30 tasks
