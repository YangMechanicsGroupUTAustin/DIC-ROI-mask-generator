# SAM2 Mask Generator Comprehensive Refactor Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all identified bugs, refactor architecture for maintainability, add missing features, and unify Python/MATLAB workflow into a single Python application.

**Architecture:** Extract monolithic `main.py` (1068 lines) into modular components: `ui/` for GUI, `core/` for processing logic, `utils/` for shared utilities. Move image processing to background thread via `threading.Thread`. Replace MATLAB smoothing with Python (scipy/numpy) implementation.

**Tech Stack:** Python 3.8+, PyTorch, SAM2, Tkinter, OpenCV, NumPy, SciPy, PIL/Pillow, Matplotlib

**Scope exclusions:** Multi-object segmentation (confirmed single-object only), mask hand-painting editor, box prompt support.

---

## Phase 1: Bug Fixes & Code Cleanup

### Task 1.1: Fix Bare Excepts and License Inconsistency

**Files:**
- Modify: `main.py:729`, `main.py:973`, `main.py:1031`
- Modify: `setup.py:17`

**Step 1: Fix bare except at main.py:729**

Replace:
```python
except:
    print(f"Failed to create placeholder for {img_path}")
```
With:
```python
except Exception as e:
    print(f"Failed to create placeholder for {img_path}: {e}")
```

**Step 2: Fix bare except at main.py:973**

Replace:
```python
except:
    return 1.25
```
With:
```python
except Exception:
    return 1.25
```

**Step 3: Fix bare except at main.py:1031**

Replace:
```python
except:
    pass
```
With:
```python
except Exception:
    pass
```

**Step 4: Fix LICENSE field in setup.py:17**

Replace:
```python
LICENSE = "Apache 2.0"
```
With:
```python
LICENSE = "MIT"
```

**Step 5: Commit**

```bash
git add main.py setup.py
git commit -m "fix: replace bare excepts and correct LICENSE to MIT"
```

---

### Task 1.2: Remove Dead Code

**Files:**
- Modify: `main.py` — remove `get_effective_points` method (lines 869-916)
- Modify: `main.py` — remove unused `prev_masks` initialization (line 71)

**Step 1: Remove `get_effective_points` method**

Delete the entire method at lines 869-916. Also delete `self.prev_masks = []` at line 71.

**Step 2: Remove duplicate global `get_dpi_scale_factor` function**

The standalone function `get_dpi_scale_factor(root)` at lines 950-974 duplicates the instance method at lines 79-110. Since the standalone version is still used in `configure_dpi_aware_fonts()`, keep the standalone for now — it will be refactored in Phase 2.

**Step 3: Commit**

```bash
git add main.py
git commit -m "refactor: remove dead code (get_effective_points, prev_masks)"
```

---

### Task 1.3: Fix Overlay Visualization

**Files:**
- Modify: `main.py` — `display_image()` overlay section (lines 365-390)

**Step 1: Fix overlay blend calculation**

Replace the overlay section:
```python
# Create colored mask overlay (red for mask)
mask_colored = np.zeros_like(overlay_image)
mask_colored[:, :, 0] = self.current_masks[0] / 255.0  # Red channel

# Blend images
alpha = 0.4  # Transparency
blended = overlay_image * (1 - alpha) + mask_colored * alpha * 255
blended = np.clip(blended, 0, 255).astype(np.uint8)
```
With:
```python
# Create colored mask overlay (red for mask region)
mask_bool = self.current_masks[0] > 127  # Binary mask
mask_overlay = overlay_image.copy()

# Apply semi-transparent red tint to masked region
alpha = 0.4
mask_overlay[mask_bool, 0] = np.clip(
    overlay_image[mask_bool, 0] * (1 - alpha) + 255 * alpha, 0, 255
).astype(np.uint8)
mask_overlay[mask_bool, 1] = (overlay_image[mask_bool, 1] * (1 - alpha)).astype(np.uint8)
mask_overlay[mask_bool, 2] = (overlay_image[mask_bool, 2] * (1 - alpha)).astype(np.uint8)

blended = mask_overlay
```

**Step 2: Verify visually**

Run: `python main.py`, load example images, process, and confirm overlay shows a clear red tint on masked regions.

**Step 3: Commit**

```bash
git add main.py
git commit -m "fix: correct overlay blend calculation for clearer mask visualization"
```

---

### Task 1.4: Fix current_masks Type Inconsistency

**Files:**
- Modify: `main.py` — initialization and assignment of `current_masks`

**Step 1: Standardize initialization**

Replace line 72:
```python
self.current_masks = [None, None, None]
```
With:
```python
self.current_masks = None  # Will hold numpy array (1, H, W) or None
```

**Step 2: Update all checks for current_masks**

Replace `all(item is None for item in self.current_masks)` (appears at lines 352 and 367) with:
```python
self.current_masks is None
```

Replace `not all(item is None for item in self.current_masks)` with:
```python
self.current_masks is not None
```

**Step 3: Commit**

```bash
git add main.py
git commit -m "fix: standardize current_masks type to numpy array or None"
```

---

### Task 1.5: Fix Double Rendering Per Frame

**Files:**
- Modify: `main.py` — `_process_images_internal()` lines 825-830

**Step 1: Merge the two display_image calls into one**

Replace:
```python
# Update current image display
self.current_image = image
self.display_image(keep_points=True)

# Update current mask display
self.current_masks = binary_masks
self.display_image()
```
With:
```python
# Update display with both image and mask in a single render
self.current_image = image
self.current_masks = binary_masks
self.display_image()
```

**Step 2: Commit**

```bash
git add main.py
git commit -m "perf: eliminate redundant double rendering per frame"
```

---

### Task 1.6: Fix Placeholder Image Size

**Files:**
- Modify: `main.py` — lines 724-730

**Step 1: Use a dynamically-sized placeholder**

Replace:
```python
try:
    blank_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(jpeg_path, blank_img)
    jpeg_files.append(jpeg_path)
    print(f"Created blank placeholder for {img_path}")
except:
    print(f"Failed to create placeholder for {img_path}")
```
With:
```python
try:
    # Try to get dimensions from a successfully loaded neighbor
    ref_shape = None
    for existing_jpeg in jpeg_files:
        ref_img = cv2.imread(existing_jpeg)
        if ref_img is not None:
            ref_shape = ref_img.shape
            break
    if ref_shape is None:
        ref_shape = (512, 512, 3)  # Reasonable default
    blank_img = np.zeros(ref_shape, dtype=np.uint8)
    cv2.imwrite(jpeg_path, blank_img)
    jpeg_files.append(jpeg_path)
    print(f"Created blank placeholder ({ref_shape[1]}x{ref_shape[0]}) for {img_path}")
except Exception as e:
    print(f"Failed to create placeholder for {img_path}: {e}")
```

**Step 2: Commit**

```bash
git add main.py
git commit -m "fix: use correct dimensions for placeholder images on load failure"
```

---

### Task 1.7: Fix MATLAB Temporal Smoothing Index Bounds

**Files:**
- Modify: `imageSmoothingGUI.m` — `removeBadFrames()` function (line 600)

**Step 1: Clamp bad frame neighbor indices to valid range**

Replace:
```matlab
badFrameIndices = unique([badFrameIndices1, badFrameIndices2, badFrameIndices2-1, badFrameIndices2+1]);
```
With:
```matlab
neighborIndices = [badFrameIndices2-1, badFrameIndices2+1];
neighborIndices = neighborIndices(neighborIndices >= 1 & neighborIndices <= numFrames);
badFrameIndices = unique([badFrameIndices1, badFrameIndices2, neighborIndices]);
```

**Step 2: Remove dead computation in smoothSingleImage**

In `smoothSingleImage()` (lines 264-308), the viscous and surface tension terms always evaluate to 0 since `params.viscousness = 0.0` and `params.surfaceTension = 0.0`. Remove the dead computation:

Delete lines 281-285:
```matlab
viscousTerm = params.viscousness * del2(smoothedImage);
[Gxx, Gyy] = gradient(Gx);
[~, Gyx] = gradient(Gy);
curvature = Gxx + Gyy;
surfaceTensionTerm = params.surfaceTension * curvature;
```

And simplify line 288-289:
```matlab
smoothedImage = smoothedImage + params.dt * (laplacianDiffusivity .* del2(smoothedImage)) ...
    - params.dt * surfaceTensionTerm - params.dt * viscousTerm;
```
To:
```matlab
smoothedImage = smoothedImage + params.dt * (laplacianDiffusivity .* del2(smoothedImage));
```

**Step 3: Commit**

```bash
git add imageSmoothingGUI.m
git commit -m "fix: clamp temporal smoothing indices and remove dead viscous computation"
```

---

## Phase 2: Architecture Refactoring

### Task 2.1: Extract DPI Scaling Utility

**Files:**
- Create: `utils/__init__.py`
- Create: `utils/dpi_scaling.py`
- Modify: `main.py` — replace all inline DPI logic

**Step 1: Create utils package**

Create `utils/__init__.py` (empty file).

**Step 2: Create DPI scaling module**

Create `utils/dpi_scaling.py`:

```python
"""DPI-aware scaling utilities for high-resolution displays."""

import tkinter as tk
from tkinter import ttk


class DPIScaler:
    """Centralized DPI scaling for 4K, 2K, and standard displays."""

    # Resolution thresholds
    THRESHOLD_4K = 3840
    THRESHOLD_2K = 2560

    # Scale factor limits per resolution tier
    SCALE_LIMITS = {
        "4k": (1.5, 3.0),
        "2k": (1.2, 2.0),
        "standard": (1.0, 1.5),
    }

    # Base font sizes per resolution tier
    FONT_SIZES = {
        "4k": {"font": 12, "title": 14, "padding": 12},
        "2k": {"font": 10, "title": 12, "padding": 11},
        "standard": {"font": 9, "title": 10, "padding": 10},
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.tier = self._detect_tier()
        self.scale = self._compute_scale()

    def _detect_tier(self) -> str:
        if self.screen_width >= self.THRESHOLD_4K or self.screen_height >= 2160:
            return "4k"
        elif self.screen_width >= self.THRESHOLD_2K or self.screen_height >= 1440:
            return "2k"
        return "standard"

    def _compute_scale(self) -> float:
        try:
            dpi = self.root.winfo_fpixels('1i')
            raw_scale = dpi / 96.0
        except Exception:
            return 1.25

        lo, hi = self.SCALE_LIMITS[self.tier]
        scale = max(lo, min(raw_scale, hi))
        print(f"Screen: {self.screen_width}x{self.screen_height} ({self.tier})")
        print(f"DPI: {dpi:.1f}, Scale: {scale:.2f}")
        return scale

    def font_size(self, role: str = "font") -> int:
        """Get scaled font size. role: 'font', 'title', or 'padding'."""
        base = self.FONT_SIZES[self.tier].get(role, 10)
        minimums = {"font": 10, "title": 12, "padding": 10}
        return max(minimums.get(role, 8), int(base * self.scale))

    def point_size(self) -> int:
        """Get scaled scatter point size for annotations."""
        bases = {"4k": 120, "2k": 100, "standard": 90}
        mins = {"4k": 100, "2k": 80, "standard": 70}
        return max(mins[self.tier], int(bases[self.tier] * self.scale))

    def matplotlib_sizes(self) -> dict:
        """Return dict of matplotlib font sizes for current resolution."""
        return {
            "title": self.font_size("title"),
            "label": self.font_size("font"),
            "tick": max(7, int(9 * self.scale)),
            "legend": max(8, int(10 * self.scale)),
            "text": max(10, int(12 * self.scale)),
            "point": self.point_size(),
        }

    def configure_ttk_style(self, style: ttk.Style):
        """Apply DPI-scaled fonts and sizes to ttk Style."""
        font_sz = self.font_size("font")
        title_sz = self.font_size("title")
        pad = self.font_size("padding")
        bar_thickness = max(20, int(25 * self.scale))

        style.configure('TLabel', font=('Segoe UI', font_sz))
        style.configure('TButton', font=('Segoe UI', font_sz), padding=(pad, pad // 2))
        style.configure('TEntry', font=('Segoe UI', font_sz), fieldbackground='white')
        style.configure('TCombobox', font=('Segoe UI', font_sz))
        style.configure('TCheckbutton', font=('Segoe UI', font_sz))
        style.configure('TLabelFrame', font=('Segoe UI', font_sz, 'bold'))
        style.configure('TLabelFrame.Label', font=('Segoe UI', title_sz, 'bold'))
        style.configure('TProgressbar', thickness=bar_thickness)

    def scaled_window_size(self, base_w=1400, base_h=900, min_w=1200, min_h=700):
        """Return (width, height, min_width, min_height) scaled by DPI."""
        return (
            int(base_w * self.scale),
            int(base_h * self.scale),
            int(min_w * self.scale),
            int(min_h * self.scale),
        )
```

**Step 3: Replace all inline DPI code in main.py**

Remove:
- Instance method `get_dpi_scale_factor()` (lines 79-110)
- Standalone function `get_dpi_scale_factor(root)` (lines 950-974)
- Standalone function `configure_dpi_aware_fonts(root, style)` (lines 976-1021)
- All repeated `screen_width >= 3840` blocks inside `display_image()` and `update_image_with_points()`

Replace with `DPIScaler` usage:
```python
from utils.dpi_scaling import DPIScaler

# In SAM2App.__init__:
self.dpi = DPIScaler(master)

# In display_image / update_image_with_points:
sizes = self.dpi.matplotlib_sizes()
ax.set_title('Original Image', fontsize=sizes['title'], fontweight='bold', pad=10)
ax.set_xlabel('X', fontsize=sizes['label'])
# ... etc
```

**Step 4: Commit**

```bash
git add utils/ main.py
git commit -m "refactor: extract DPI scaling into DPIScaler utility class"
```

---

### Task 2.2: Extract Image Processing Logic

**Files:**
- Create: `core/__init__.py`
- Create: `core/image_processing.py`
- Modify: `main.py` — move processing functions out

**Step 1: Create core package**

Create `core/__init__.py` (empty file).

**Step 2: Create image processing module**

Create `core/image_processing.py` with functions extracted from main.py:

```python
"""Image loading, conversion, and mask processing utilities."""

import os
import re
import glob

import cv2
import numpy as np
from PIL import Image


def extract_numbers(file_name: str) -> tuple:
    """Extract numeric parts from filename for natural sorting."""
    numbers = re.findall(r'\d+', file_name)
    return tuple(map(int, numbers))


def get_image_files(directory: str) -> list[str]:
    """Find and sort image files in directory using natural ordering."""
    image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.tif')
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    return sorted(image_files, key=lambda x: extract_numbers(os.path.basename(x)))


def load_image_as_rgb(image_path: str) -> np.ndarray | None:
    """Load image from any supported format and return as RGB numpy array.

    Handles 8-bit, 16-bit, 32-bit, and floating-point images.
    Falls back from OpenCV to PIL for unsupported formats.
    Returns None on failure.
    """
    # Try OpenCV first
    img = cv2.imread(image_path)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Fallback to PIL
    try:
        pil_image = Image.open(image_path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return np.array(pil_image)
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")
        return None


def convert_to_jpeg(img_path: str, jpeg_path: str, quality: int = 95) -> bool:
    """Convert any supported image format to JPEG for SAM2 inference.

    Handles 8-bit, 16-bit, 32-bit, and float images.
    Returns True on success, False on failure.
    """
    try:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            # Fallback to PIL for formats OpenCV can't handle
            pil_image = Image.open(img_path)
            if pil_image.mode in ('F', 'I', 'I;16', 'I;32'):
                img_array = np.array(pil_image)
                min_val, max_val = np.min(img_array), np.max(img_array)
                if max_val > min_val:
                    img = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img_array, dtype=np.uint8)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                pil_image = pil_image.convert('RGB')
                img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            # Normalize bit depth
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
            elif img.dtype in (np.float32, np.float64):
                min_val, max_val = np.min(img), np.max(img)
                if max_val > min_val:
                    img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)

            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(jpeg_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return True

    except Exception as e:
        print(f"Failed to convert {img_path} to JPEG: {e}")
        return False


def convert_to_png(img_path: str, png_path: str) -> bool:
    """Convert any supported image format to lossless PNG for SAM2 inference.

    Same normalization logic as convert_to_jpeg but uses PNG (lossless).
    Returns True on success, False on failure.
    """
    try:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            pil_image = Image.open(img_path)
            if pil_image.mode in ('F', 'I', 'I;16', 'I;32'):
                img_array = np.array(pil_image)
                min_val, max_val = np.min(img_array), np.max(img_array)
                if max_val > min_val:
                    img = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img_array, dtype=np.uint8)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                pil_image = pil_image.convert('RGB')
                img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
            elif img.dtype in (np.float32, np.float64):
                min_val, max_val = np.min(img), np.max(img)
                if max_val > min_val:
                    img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)

            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(png_path, img)
        return True

    except Exception as e:
        print(f"Failed to convert {img_path} to PNG: {e}")
        return False


def create_placeholder_image(
    jpeg_path: str, reference_files: list[str], fallback_size: tuple = (512, 512, 3)
) -> bool:
    """Create a black placeholder image matching the size of existing frames."""
    ref_shape = fallback_size
    for existing in reference_files:
        ref_img = cv2.imread(existing)
        if ref_img is not None:
            ref_shape = ref_img.shape
            break
    try:
        blank = np.zeros(ref_shape, dtype=np.uint8)
        cv2.imwrite(jpeg_path, blank)
        print(f"Created placeholder ({ref_shape[1]}x{ref_shape[0]}) at {jpeg_path}")
        return True
    except Exception as e:
        print(f"Failed to create placeholder: {e}")
        return False


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Create red-tinted overlay of mask on original image.

    Args:
        image: RGB image array (H, W, 3) uint8
        mask: Binary mask array (H, W) with values 0 or 255
        alpha: Transparency of overlay (0=invisible, 1=opaque red)

    Returns:
        Blended overlay image (H, W, 3) uint8
    """
    overlay = image.copy()
    if len(overlay.shape) == 2:
        overlay = np.stack([overlay] * 3, axis=-1)

    mask_bool = mask > 127
    overlay[mask_bool, 0] = np.clip(
        image[mask_bool, 0] * (1 - alpha) + 255 * alpha, 0, 255
    ).astype(np.uint8)
    overlay[mask_bool, 1] = (image[mask_bool, 1] * (1 - alpha)).astype(np.uint8)
    overlay[mask_bool, 2] = (image[mask_bool, 2] * (1 - alpha)).astype(np.uint8)

    return overlay
```

**Step 3: Update main.py imports**

Replace inline image processing code with calls to `core.image_processing` functions.

**Step 4: Commit**

```bash
git add core/ main.py
git commit -m "refactor: extract image processing logic into core/image_processing.py"
```

---

### Task 2.3: Move Processing to Background Thread

**Files:**
- Modify: `main.py` — `process_images()` and `_process_images_internal()`

**Step 1: Add threading import**

```python
import threading
```

**Step 2: Refactor process_images to launch background thread**

Replace `process_images()`:
```python
def process_images(self):
    """Launch image processing in background thread."""
    try:
        os.makedirs(self.config.output_dir, exist_ok=True)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to create output directory: {str(e)}")
        return

    self.processing = True
    self.start_btn.config(state=tk.DISABLED)
    self.stop_btn.config(state=tk.NORMAL)

    thread = threading.Thread(target=self._process_thread, daemon=True)
    thread.start()

def _process_thread(self):
    """Background thread for image processing."""
    try:
        self._process_images_internal()
    except Exception as e:
        self.master.after(0, lambda: messagebox.showerror(
            "Processing Error", f"Unexpected error: {str(e)}"
        ))
    finally:
        self.master.after(0, self._on_processing_complete)

def _on_processing_complete(self):
    """Called on main thread when processing finishes."""
    self.processing = False
    self.start_btn.config(state=tk.NORMAL)
    self.stop_btn.config(state=tk.DISABLED)
```

**Step 3: Replace all direct UI updates in _process_images_internal with master.after()**

All `self.master.update()`, `self.display_image()`, `self.update_progress()`, `messagebox.*()`, and `self.progress_label.config()` calls inside `_process_images_internal` must be wrapped:

```python
# Instead of:
self.update_progress(mask_progress)
self.progress_label.config(text=f"Mask Prediction: {i+1}/{total}")
self.display_image()
self.master.update()

# Use:
self.master.after(0, lambda p=mask_progress: self.update_progress(p))
self.master.after(0, lambda t=f"Mask Prediction: {i+1}/{total}": self.progress_label.config(text=t))
self.master.after(0, self.display_image)
# Remove self.master.update() — no longer needed with threading
```

All `messagebox` calls inside the thread must also use `master.after`:
```python
self.master.after(0, lambda msg=error_msg: messagebox.showerror("Error", msg))
```

**Step 4: Verify GUI responsiveness**

Run: `python main.py`, start processing, verify window remains responsive (can drag, resize, click stop).

**Step 5: Commit**

```bash
git add main.py
git commit -m "feat: move image processing to background thread for responsive GUI"
```

---

### Task 2.4: Temp File Cleanup on Exit

**Files:**
- Modify: `main.py` — `on_closing()` and `cleanup_temp_folders()`

**Step 1: Add exit cleanup**

Update `on_closing()`:
```python
def on_closing():
    app.cleanup_temp_folders()
    app._cleanup_predictor()
    root.destroy()
```

**Step 2: Remove __del__ method**

Delete `__del__` (line 946-948) since cleanup is handled in `on_closing()`.

**Step 3: Commit**

```bash
git add main.py
git commit -m "fix: clean up temp folders on app exit, remove fragile __del__"
```

---

## Phase 3: Core Feature Enhancements

### Task 3.1: Auto-Detect Best Device (GPU/MPS/CPU)

**Files:**
- Modify: `main.py` — device selection UI and logic

**Step 1: Update device dropdown to include MPS**

Replace device setup in `setup_analysis_controls()`:
```python
# Build device list based on availability
available_devices = ["CPU"]
if torch.cuda.is_available():
    available_devices.append("GPU (CUDA)")
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    available_devices.append("GPU (MPS)")

self.device_var = tk.StringVar(self.master)
# Auto-select best available device
if "GPU (CUDA)" in available_devices:
    self.device_var.set("GPU (CUDA)")
    self.device = 'cuda'
elif "GPU (MPS)" in available_devices:
    self.device_var.set("GPU (MPS)")
    self.device = 'mps'
else:
    self.device_var.set("CPU")
    self.device = 'cpu'

self.device_menu = ttk.Combobox(
    device_frame, textvariable=self.device_var,
    values=available_devices
)
```

**Step 2: Update on_device_change handler**

```python
def on_device_change(self, event):
    selected = self.device_var.get()
    if selected == "GPU (CUDA)":
        self.device = 'cuda'
    elif selected == "GPU (MPS)":
        self.device = 'mps'
    else:
        self.device = 'cpu'
    print(f"Device set to: {self.device}")
```

**Step 3: Remove the unused global `device` variable at line 35**

It was never used by the app (the app uses `self.device`).

**Step 4: Commit**

```bash
git add main.py
git commit -m "feat: auto-detect and select best device (CUDA > MPS > CPU)"
```

---

### Task 3.2: Frame Navigation Preview

**Files:**
- Modify: `main.py` — add frame slider and navigation buttons

**Step 1: Add navigation controls**

In `setup_gui()`, after the range frame, add a navigation frame:

```python
# Frame navigation controls
nav_frame = ttk.Frame(main_frame)
nav_frame.pack(fill=tk.X, pady=(0, 10))

ttk.Button(nav_frame, text="<< Prev", command=self.prev_frame).pack(side=tk.LEFT, padx=2)
ttk.Button(nav_frame, text="Next >>", command=self.next_frame).pack(side=tk.LEFT, padx=2)

ttk.Label(nav_frame, text="Preview frame:").pack(side=tk.LEFT, padx=(10, 5))
self.preview_index_var = tk.IntVar(value=1)
self.preview_slider = ttk.Scale(
    nav_frame, from_=1, to=100,
    variable=self.preview_index_var,
    orient=tk.HORIZONTAL,
    command=self._on_slider_change
)
self.preview_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
self.preview_label = ttk.Label(nav_frame, text="1 / ?")
self.preview_label.pack(side=tk.LEFT, padx=5)
```

**Step 2: Implement navigation methods**

```python
def prev_frame(self):
    current = self.preview_index_var.get()
    if current > 1:
        self.preview_index_var.set(current - 1)
        self._preview_frame(current - 1)

def next_frame(self):
    current = self.preview_index_var.get()
    image_files = self.get_image_files(self.config.input_dir)
    if current < len(image_files):
        self.preview_index_var.set(current + 1)
        self._preview_frame(current + 1)

def _on_slider_change(self, value):
    frame_idx = int(float(value))
    self._preview_frame(frame_idx)

def _preview_frame(self, frame_idx: int):
    """Load and display a specific frame for preview."""
    image_files = self.get_image_files(self.config.input_dir)
    if not image_files or frame_idx < 1 or frame_idx > len(image_files):
        return
    self.preview_label.config(text=f"{frame_idx} / {len(image_files)}")
    image = load_image_as_rgb(image_files[frame_idx - 1])
    if image is not None:
        self.current_image = image
        self.display_image()
```

**Step 3: Update slider range when input directory changes**

In `select_input()`, after loading images:
```python
image_files = self.get_image_files(self.config.input_dir)
if image_files:
    self.preview_slider.config(to=len(image_files))
    self.preview_label.config(text=f"1 / {len(image_files)}")
```

**Step 4: Commit**

```bash
git add main.py
git commit -m "feat: add frame navigation slider and prev/next buttons for preview"
```

---

### Task 3.3: Single Point Undo

**Files:**
- Modify: `main.py` — add undo button and logic

**Step 1: Add Undo button in setup_viz_controls**

After the Clear Points button:
```python
self.undo_point_btn = ttk.Button(parent, text="Undo Last Point", command=self.undo_last_point)
self.undo_point_btn.pack(side=tk.LEFT, padx=5, pady=5)
```

**Step 2: Implement undo_last_point**

```python
def undo_last_point(self):
    """Remove the last added annotation point."""
    if self.config.input_points:
        removed_point = self.config.input_points.pop()
        removed_label = self.config.input_labels.pop()
        label_name = "foreground" if removed_label == 1 else "background"
        print(f"Undone {label_name} point at {removed_point}")
        self.display_image()
    else:
        messagebox.showinfo("Info", "No points to undo.")
```

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add single-point undo for annotation corrections"
```

---

### Task 3.4: Configurable Binary Threshold

**Files:**
- Modify: `main.py` — add threshold slider to analysis controls

**Step 1: Add threshold control**

In `setup_analysis_controls()`, add:
```python
threshold_frame = ttk.Frame(parent)
threshold_frame.pack(fill=tk.X, pady=5)
ttk.Label(threshold_frame, text="Mask Threshold:").pack(side=tk.LEFT, padx=(0, 5))
self.threshold_var = tk.DoubleVar(value=0.0)
self.threshold_slider = ttk.Scale(
    threshold_frame, from_=-5.0, to=5.0,
    variable=self.threshold_var,
    orient=tk.HORIZONTAL,
)
self.threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
self.threshold_label = ttk.Label(threshold_frame, text="0.0")
self.threshold_label.pack(side=tk.LEFT, padx=5)

def update_threshold_label(val):
    self.threshold_label.config(text=f"{float(val):.1f}")
self.threshold_slider.config(command=update_threshold_label)
```

**Step 2: Use threshold in processing**

Replace hardcoded threshold in `_process_images_internal()`:
```python
# Old:
(out_mask_logits[i] > 0.0).cpu().numpy()

# New:
threshold = self.threshold_var.get()
(out_mask_logits[i] > threshold).cpu().numpy()
```

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add configurable mask binarization threshold slider"
```

---

### Task 3.5: PNG Lossless Intermediate Format Option

**Files:**
- Modify: `main.py` — add format toggle, update conversion logic

**Step 1: Add format selection in analysis controls**

```python
ttk.Label(device_frame, text="Intermediate:").pack(side=tk.LEFT, padx=(10, 5))
self.intermediate_format_var = tk.StringVar(value="JPEG (fast)")
self.format_menu = ttk.Combobox(
    device_frame, textvariable=self.intermediate_format_var,
    values=["JPEG (fast)", "PNG (lossless)"],
    width=14
)
self.format_menu.pack(side=tk.LEFT, padx=(0, 10))
```

**Step 2: Update temp file conversion logic**

In `_process_images_internal()`, replace the JPEG conversion with format-aware logic:
```python
use_png = self.intermediate_format_var.get() == "PNG (lossless)"
ext = ".png" if use_png else ".jpg"

for idx, img_path in enumerate(processing_images):
    temp_name = f"{idx + self.start_index:06d}{ext}"
    temp_path = os.path.join(temp_dir, temp_name)

    if os.path.exists(temp_path):
        temp_files.append(temp_path)
        continue

    if use_png:
        success = convert_to_png(img_path, temp_path)
    else:
        success = convert_to_jpeg(img_path, temp_path)

    if success:
        temp_files.append(temp_path)
    else:
        create_placeholder_image(temp_path, temp_files)
        temp_files.append(temp_path)
```

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add PNG lossless option for intermediate format conversion"
```

---

### Task 3.6: Annotation Points Save/Load

**Files:**
- Create: `core/annotation_config.py`
- Modify: `main.py` — add Save/Load buttons

**Step 1: Create annotation config module**

Create `core/annotation_config.py`:
```python
"""Save and load annotation configurations (points, labels, parameters)."""

import json
import os
from datetime import datetime


def save_annotation_config(
    filepath: str,
    input_points: list,
    input_labels: list,
    model_name: str = "",
    device: str = "",
    threshold: float = 0.0,
    start_frame: int = 1,
    end_frame: int = 0,
    intermediate_format: str = "JPEG",
) -> None:
    """Save annotation points and experiment config to JSON file."""
    config = {
        "version": "1.0",
        "saved_at": datetime.now().isoformat(),
        "annotation": {
            "points": input_points,
            "labels": input_labels,
        },
        "parameters": {
            "model": model_name,
            "device": device,
            "threshold": threshold,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "intermediate_format": intermediate_format,
        },
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def load_annotation_config(filepath: str) -> dict:
    """Load annotation config from JSON file. Returns parsed dict."""
    with open(filepath, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Validate required fields
    if "annotation" not in config:
        raise ValueError("Invalid config file: missing 'annotation' section")
    if "points" not in config["annotation"] or "labels" not in config["annotation"]:
        raise ValueError("Invalid config file: missing points or labels")

    return config
```

**Step 2: Add Save/Load buttons to GUI**

In `setup_viz_controls()`:
```python
self.save_config_btn = ttk.Button(parent, text="Save Config", command=self.save_config)
self.save_config_btn.pack(side=tk.LEFT, padx=5, pady=5)

self.load_config_btn = ttk.Button(parent, text="Load Config", command=self.load_config)
self.load_config_btn.pack(side=tk.LEFT, padx=5, pady=5)
```

**Step 3: Implement save/load methods**

```python
def save_config(self):
    """Save current annotation points and parameters to JSON file."""
    if not self.config.input_points:
        messagebox.showwarning("Warning", "No annotation points to save.")
        return

    filepath = filedialog.asksaveasfilename(
        title="Save Annotation Config",
        defaultextension=".json",
        filetypes=[("JSON files", "*.json")],
    )
    if not filepath:
        return

    try:
        save_annotation_config(
            filepath=filepath,
            input_points=self.config.input_points,
            input_labels=self.config.input_labels,
            model_name=self.model_var.get(),
            device=self.device,
            threshold=self.threshold_var.get(),
            start_frame=int(self.start_index_entry.get() or 1),
            end_frame=int(self.end_index_entry.get() or 0),
            intermediate_format=self.intermediate_format_var.get(),
        )
        messagebox.showinfo("Success", f"Config saved to {filepath}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save config: {e}")

def load_config(self):
    """Load annotation points and parameters from JSON file."""
    filepath = filedialog.askopenfilename(
        title="Load Annotation Config",
        filetypes=[("JSON files", "*.json")],
    )
    if not filepath:
        return

    try:
        config = load_annotation_config(filepath)

        # Restore annotation points
        self.config.input_points = config["annotation"]["points"]
        self.config.input_labels = config["annotation"]["labels"]

        # Restore parameters if present
        params = config.get("parameters", {})
        if params.get("model"):
            self.model_var.set(params["model"])
        if params.get("threshold") is not None:
            self.threshold_var.set(params["threshold"])
        if params.get("start_frame"):
            self.start_index_entry.delete(0, tk.END)
            self.start_index_entry.insert(0, str(params["start_frame"]))
        if params.get("end_frame"):
            self.end_index_entry.delete(0, tk.END)
            self.end_index_entry.insert(0, str(params["end_frame"]))

        self.display_image()
        messagebox.showinfo("Success", f"Loaded {len(self.config.input_points)} points from config")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load config: {e}")
```

**Step 4: Commit**

```bash
git add core/annotation_config.py main.py
git commit -m "feat: add annotation config save/load with experiment parameters"
```

---

### Task 3.7: Mid-Sequence Correction Points

**Files:**
- Modify: `main.py` — support adding correction points during processing

This is the most complex feature. The SAM2 Video Predictor supports adding new points to any frame via `add_new_points_or_box()` with arbitrary `frame_idx`.

**Step 1: Add correction mode UI**

Add a "Correction Mode" section that appears after initial processing:
```python
# In setup_analysis_controls:
self.correction_btn = ttk.Button(
    control_frame, text="Add Correction",
    command=self.start_correction_mode, state=tk.DISABLED
)
self.correction_btn.pack(side=tk.LEFT, padx=(0, 5))
```

**Step 2: Implement correction workflow**

```python
def start_correction_mode(self):
    """Enter correction mode: user navigates to a frame, adds points, re-propagates."""
    if self.inference_state is None:
        messagebox.showerror("Error", "Run initial processing first.")
        return

    self.correction_mode = True
    self.correction_points = []
    self.correction_labels = []

    messagebox.showinfo(
        "Correction Mode",
        "Navigate to the frame you want to correct using the slider.\n"
        "Draw correction points, then click 'Apply Correction'."
    )

    # Enable drawing mode
    self.plotting_mode = True
    self.plot_points_btn.config(text="Stop Drawing")
    self.canvas.mpl_connect('button_press_event', self.on_correction_click)

    # Show Apply button
    self.apply_correction_btn.config(state=tk.NORMAL)

def on_correction_click(self, event):
    """Handle clicks during correction mode."""
    if event.inaxes == self.ax1 and self.correction_mode:
        x, y = int(event.xdata), int(event.ydata)
        self.correction_points.append([x, y])
        self.correction_labels.append(1 if self.point_mode == 'foreground' else 0)
        self.display_image()

def apply_correction(self):
    """Apply correction points to current preview frame and re-propagate."""
    if not self.correction_points:
        messagebox.showwarning("Warning", "No correction points added.")
        return

    frame_idx = self.preview_index_var.get() - self.start_index
    if frame_idx < 0:
        messagebox.showerror("Error", "Invalid frame for correction.")
        return

    predictor = self.current_predictor
    if predictor is None or self.inference_state is None:
        messagebox.showerror("Error", "No active inference state.")
        return

    points_array = np.array(self.correction_points)
    labels_array = np.array(self.correction_labels)

    predictor.add_new_points_or_box(
        inference_state=self.inference_state,
        frame_idx=frame_idx,
        obj_id=1,
        points=points_array,
        labels=labels_array,
    )

    # Re-propagate from correction point onward
    self.correction_mode = False
    self.correction_points = []
    self.correction_labels = []
    self.apply_correction_btn.config(state=tk.DISABLED)

    # Re-run propagation (in background thread)
    self.processing = True
    self.start_btn.config(state=tk.DISABLED)
    self.stop_btn.config(state=tk.NORMAL)
    thread = threading.Thread(target=self._repropagate_thread, daemon=True)
    thread.start()

def _repropagate_thread(self):
    """Re-propagate masks after correction in background."""
    try:
        predictor = self.current_predictor
        image_files = self.get_image_files(self.config.input_dir)
        processing_images = image_files[self.start_index - 1:self.end_index]

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(self.inference_state):
            if not self.processing:
                break

            adjusted = out_frame_idx + self.start_index - 1
            if adjusted < 0 or adjusted >= len(image_files):
                continue

            threshold = self.threshold_var.get()
            video_segments = {
                out_obj_id: (out_mask_logits[i] > threshold).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            binary_masks = video_segments[1].astype(np.uint8) * 255

            # Save updated mask
            image_name = os.path.basename(image_files[adjusted])
            mask1_dir = os.path.join(self.config.output_dir, "mask1")
            os.makedirs(mask1_dir, exist_ok=True)
            cv2.imwrite(os.path.join(mask1_dir, image_name), binary_masks[0])

            # Update display
            image = cv2.imread(image_files[adjusted])
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.current_image = image
                self.current_masks = binary_masks

            progress = (out_frame_idx + 1) / len(processing_images) * 100
            self.master.after(0, lambda p=progress: self.update_progress(p))
            self.master.after(0, self.display_image)

    except Exception as e:
        self.master.after(0, lambda: messagebox.showerror("Error", f"Re-propagation failed: {e}"))
    finally:
        self.master.after(0, self._on_processing_complete)
```

**Step 3: Add Apply Correction button in setup_analysis_controls**

```python
self.apply_correction_btn = ttk.Button(
    control_frame, text="Apply Correction",
    command=self.apply_correction, state=tk.DISABLED
)
self.apply_correction_btn.pack(side=tk.LEFT, padx=(0, 5))
```

**Step 4: Initialize correction state in __init__**

```python
self.correction_mode = False
self.correction_points = []
self.correction_labels = []
```

**Step 5: Enable correction button after initial processing completes**

In `_on_processing_complete`:
```python
self.correction_btn.config(state=tk.NORMAL)
```

**Step 6: Commit**

```bash
git add main.py
git commit -m "feat: add mid-sequence correction points with re-propagation"
```

---

## Phase 4: Advanced Features

### Task 4.1: Resume Processing From Checkpoint

**Files:**
- Modify: `main.py` — detect existing masks and skip processed frames

**Step 1: Add checkpoint detection in _process_images_internal**

Before starting propagation, check which frames already have masks:
```python
mask1_dir = os.path.join(self.config.output_dir, "mask1")
existing_masks = set()
if os.path.exists(mask1_dir):
    for f in os.listdir(mask1_dir):
        existing_masks.add(f)

# During frame processing loop, skip frames with existing masks:
if image_name in existing_masks:
    frame_count += 1
    # Still need to run propagation for SAM2 state, but skip saving
    continue
```

Note: SAM2's propagate_in_video must process all frames sequentially for its memory mechanism. We can skip re-saving already saved masks but cannot skip inference. Add a "Force Re-process" checkbox to override this.

**Step 2: Add Force Re-process checkbox**

```python
self.force_reprocess_var = tk.BooleanVar(value=False)
ttk.Checkbutton(
    control_frame, text="Force Re-process",
    variable=self.force_reprocess_var
).pack(side=tk.LEFT, padx=(0, 5))
```

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat: skip saving already-processed frames unless force re-process enabled"
```

---

### Task 4.2: Python-Based Spatial Smoothing (Replace MATLAB)

**Files:**
- Create: `core/spatial_smoothing.py`
- Modify: `main.py` — add smoothing tab or post-processing section

**Step 1: Implement Perona-Malik diffusion in Python**

Create `core/spatial_smoothing.py`:
```python
"""Perona-Malik anisotropic diffusion smoothing for binary masks.

Port of imageSmoothingGUI.m spatial smoothing to Python.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, laplace


def perona_malik_smooth(
    image: np.ndarray,
    num_iterations: int = 50,
    dt: float = 0.1,
    lam: float = 0.5,
    gaussian_sigma: float = 4.0,
) -> np.ndarray:
    """Apply Perona-Malik anisotropic diffusion smoothing.

    Args:
        image: Input image as float64 array normalized to [0, 1].
        num_iterations: Number of diffusion iterations.
        dt: Time step per iteration (controls convergence speed).
        lam: Gradient sensitivity (lambda parameter).
        gaussian_sigma: Sigma for post-iteration Gaussian filter.

    Returns:
        Smoothed binary mask as uint8 array (0 or 255).
    """
    smoothed = image.astype(np.float64)
    if smoothed.max() > 1.0:
        smoothed = smoothed / 255.0

    for _ in range(num_iterations):
        # Calculate gradient magnitude
        gy, gx = np.gradient(smoothed)
        grad_mag_sq = gx ** 2 + gy ** 2

        # Perona-Malik diffusivity (exponential)
        diffusivity = np.exp(-grad_mag_sq / (2 * lam ** 2))

        # Diffusion update
        laplacian_d = laplace(diffusivity)
        laplacian_s = laplace(smoothed)
        smoothed = smoothed + dt * (laplacian_d * laplacian_s)

        # Post-iteration Gaussian smoothing
        smoothed = gaussian_filter(smoothed, sigma=gaussian_sigma)

    # Binarize
    binary = (smoothed > 0.5).astype(np.uint8) * 255
    return binary


def smooth_mask_sequence(
    masks: list[np.ndarray],
    num_iterations: int = 50,
    dt: float = 0.1,
    lam: float = 0.5,
    gaussian_sigma: float = 4.0,
    progress_callback=None,
) -> list[np.ndarray]:
    """Apply spatial smoothing to a sequence of mask images.

    Args:
        masks: List of mask images (uint8 or float).
        progress_callback: Optional callable(current, total) for progress.

    Returns:
        List of smoothed binary mask images.
    """
    results = []
    total = len(masks)
    for i, mask in enumerate(masks):
        smoothed = perona_malik_smooth(
            mask, num_iterations, dt, lam, gaussian_sigma
        )
        results.append(smoothed)
        if progress_callback:
            progress_callback(i + 1, total)
    return results
```

**Step 2: Commit**

```bash
git add core/spatial_smoothing.py
git commit -m "feat: port Perona-Malik spatial smoothing from MATLAB to Python"
```

---

### Task 4.3: Python-Based Temporal Smoothing (Replace MATLAB)

**Files:**
- Create: `core/temporal_smoothing.py`

**Step 1: Implement temporal smoothing**

Create `core/temporal_smoothing.py`:
```python
"""Temporal smoothing for mask sequences — bad frame detection and 3D Gaussian filtering.

Port of imageSmoothingGUI.m temporal smoothing to Python.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d


def detect_bad_frames(
    frames: np.ndarray,
    variance_threshold: float = 50000,
) -> list[int]:
    """Detect frames with abnormal variance (noise or blank frames).

    Args:
        frames: 3D array (H, W, N) of frame data.
        variance_threshold: Maximum variance before flagging as bad.

    Returns:
        Sorted list of bad frame indices (0-based).
    """
    num_frames = frames.shape[2]
    variances = np.array([np.var(frames[:, :, i]) for i in range(num_frames)])

    bad_high = set(np.where(variances > variance_threshold)[0])
    bad_zero = set(np.where(variances == 0)[0])

    # Include immediate neighbors of zero-variance frames
    bad_neighbors = set()
    for idx in bad_zero:
        if idx > 0:
            bad_neighbors.add(idx - 1)
        if idx < num_frames - 1:
            bad_neighbors.add(idx + 1)

    return sorted(bad_high | bad_zero | bad_neighbors)


def fill_nan_frames(
    sequence: np.ndarray,
    bad_indices: list[int],
    num_neighbors: int = 2,
    sigma: float = 2.0,
) -> np.ndarray:
    """Replace bad frames with Gaussian-weighted temporal average of neighbors.

    Args:
        sequence: 3D array (H, W, N).
        bad_indices: Indices of frames to replace.
        num_neighbors: Number of neighbor frames on each side.
        sigma: Gaussian sigma for neighbor weighting.

    Returns:
        Filled sequence with bad frames replaced.
    """
    filled = sequence.copy()
    num_frames = sequence.shape[2]

    # Set bad frames to NaN
    for idx in bad_indices:
        filled[:, :, idx] = np.nan

    # Fill each bad frame from neighbors
    for idx in bad_indices:
        lo = max(0, idx - num_neighbors)
        hi = min(num_frames, idx + num_neighbors + 1)
        neighbors = filled[:, :, lo:hi].copy()

        # Apply Gaussian smoothing to each neighbor
        for k in range(neighbors.shape[2]):
            frame = neighbors[:, :, k]
            if not np.any(np.isnan(frame)):
                neighbors[:, :, k] = gaussian_filter(frame, sigma=sigma)

        # Average ignoring NaN
        with np.errstate(all='ignore'):
            avg = np.nanmean(neighbors, axis=2)
        avg = np.nan_to_num(avg, nan=0.0)
        filled[:, :, idx] = avg

    return filled


def apply_3d_gaussian(
    sequence: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 2.0,
) -> np.ndarray:
    """Apply 3D Gaussian filter for spatio-temporal smoothing.

    Args:
        sequence: 3D array (H, W, N).
        kernel_size: Not directly used (scipy uses sigma only).
        sigma: Gaussian sigma for all 3 dimensions.

    Returns:
        Smoothed 3D sequence.
    """
    return gaussian_filter(sequence, sigma=[sigma, sigma, sigma])


def temporal_smooth_sequence(
    frames: list[np.ndarray],
    variance_threshold: float = 50000,
    num_neighbors: int = 2,
    sigma: float = 2.0,
    kernel_size: int = 5,
    progress_callback=None,
) -> list[np.ndarray]:
    """Full temporal smoothing pipeline.

    Args:
        frames: List of 2D mask arrays (grayscale).
        variance_threshold: Bad frame detection threshold.
        num_neighbors: Temporal window for NaN filling.
        sigma: Gaussian sigma for both filling and 3D filtering.
        kernel_size: 3D Gaussian kernel size (informational).
        progress_callback: Optional callable(step_name, current, total).

    Returns:
        List of temporally smoothed and binarized mask arrays.
    """
    # Stack into 3D array
    h, w = frames[0].shape[:2]
    sequence = np.zeros((h, w, len(frames)), dtype=np.float64)
    for i, f in enumerate(frames):
        gray = f if len(f.shape) == 2 else f[:, :, 0]
        sequence[:, :, i] = gray.astype(np.float64)

    if progress_callback:
        progress_callback("Detecting bad frames", 1, 4)

    bad_indices = detect_bad_frames(sequence, variance_threshold)

    if progress_callback:
        progress_callback("Filling bad frames", 2, 4)

    filled = fill_nan_frames(sequence, bad_indices, num_neighbors, sigma)

    if progress_callback:
        progress_callback("Applying 3D Gaussian filter", 3, 4)

    smoothed = apply_3d_gaussian(filled, kernel_size, sigma)

    if progress_callback:
        progress_callback("Binarizing results", 4, 4)

    # Binarize and return as list
    results = []
    for i in range(smoothed.shape[2]):
        binary = (smoothed[:, :, i] > 127).astype(np.uint8) * 255
        results.append(binary)

    return results
```

**Step 2: Commit**

```bash
git add core/temporal_smoothing.py
git commit -m "feat: port temporal smoothing (bad frame detection + 3D Gaussian) from MATLAB to Python"
```

---

### Task 4.4: Integrate Smoothing Into GUI

**Files:**
- Modify: `main.py` — add Smoothing tab/section to GUI

**Step 1: Add post-processing section to GUI**

After the analysis controls frame, add a post-processing LabelFrame:

```python
# Post-processing controls
postproc_frame = ttk.LabelFrame(middle_frame, text="Post-Processing")
postproc_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

# Spatial smoothing
spatial_frame = ttk.Frame(postproc_frame)
spatial_frame.pack(fill=tk.X, pady=2)
ttk.Label(spatial_frame, text="Iterations:").pack(side=tk.LEFT, padx=2)
self.smooth_iter_var = tk.IntVar(value=50)
ttk.Entry(spatial_frame, textvariable=self.smooth_iter_var, width=5).pack(side=tk.LEFT, padx=2)
ttk.Label(spatial_frame, text="dt:").pack(side=tk.LEFT, padx=2)
self.smooth_dt_var = tk.DoubleVar(value=0.1)
ttk.Entry(spatial_frame, textvariable=self.smooth_dt_var, width=5).pack(side=tk.LEFT, padx=2)
ttk.Label(spatial_frame, text="Lambda:").pack(side=tk.LEFT, padx=2)
self.smooth_lambda_var = tk.DoubleVar(value=0.5)
ttk.Entry(spatial_frame, textvariable=self.smooth_lambda_var, width=5).pack(side=tk.LEFT, padx=2)

ttk.Button(
    spatial_frame, text="Spatial Smooth",
    command=self.run_spatial_smoothing
).pack(side=tk.LEFT, padx=5)

# Temporal smoothing
temporal_frame = ttk.Frame(postproc_frame)
temporal_frame.pack(fill=tk.X, pady=2)
ttk.Label(temporal_frame, text="Var Thresh:").pack(side=tk.LEFT, padx=2)
self.temporal_var_thresh = tk.DoubleVar(value=50000)
ttk.Entry(temporal_frame, textvariable=self.temporal_var_thresh, width=7).pack(side=tk.LEFT, padx=2)
ttk.Label(temporal_frame, text="Neighbors:").pack(side=tk.LEFT, padx=2)
self.temporal_neighbors_var = tk.IntVar(value=2)
ttk.Entry(temporal_frame, textvariable=self.temporal_neighbors_var, width=3).pack(side=tk.LEFT, padx=2)

ttk.Button(
    temporal_frame, text="Temporal Smooth",
    command=self.run_temporal_smoothing
).pack(side=tk.LEFT, padx=5)
```

**Step 2: Implement smoothing methods with threading**

```python
def run_spatial_smoothing(self):
    """Run spatial smoothing on generated masks in background thread."""
    mask_dir = os.path.join(self.config.output_dir, "mask1")
    if not os.path.exists(mask_dir):
        messagebox.showerror("Error", "No masks found. Run mask generation first.")
        return

    output_dir = os.path.join(self.config.output_dir, "mask_spatial_smoothing")
    os.makedirs(output_dir, exist_ok=True)

    self.start_btn.config(state=tk.DISABLED)
    thread = threading.Thread(
        target=self._spatial_smooth_thread,
        args=(mask_dir, output_dir),
        daemon=True,
    )
    thread.start()

def _spatial_smooth_thread(self, mask_dir, output_dir):
    from core.spatial_smoothing import perona_malik_smooth
    try:
        mask_files = sorted(
            [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.tiff', '.tif', '.jpg', '.bmp'))],
            key=extract_numbers
        )
        total = len(mask_files)

        for i, fname in enumerate(mask_files):
            if not self.processing and i > 0:
                break

            mask = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            smoothed = perona_malik_smooth(
                mask,
                num_iterations=self.smooth_iter_var.get(),
                dt=self.smooth_dt_var.get(),
                lam=self.smooth_lambda_var.get(),
            )
            cv2.imwrite(os.path.join(output_dir, fname), smoothed)

            progress = (i + 1) / total * 100
            self.master.after(0, lambda p=progress: self.update_progress(p))
            self.master.after(0, lambda t=f"Spatial smoothing: {i+1}/{total}": self.progress_label.config(text=t))

        self.master.after(0, lambda: messagebox.showinfo("Done", f"Spatial smoothing saved to {output_dir}"))
    except Exception as e:
        self.master.after(0, lambda: messagebox.showerror("Error", str(e)))
    finally:
        self.master.after(0, lambda: self.start_btn.config(state=tk.NORMAL))

def run_temporal_smoothing(self):
    """Run temporal smoothing on spatially-smoothed masks."""
    spatial_dir = os.path.join(self.config.output_dir, "mask_spatial_smoothing")
    if not os.path.exists(spatial_dir):
        spatial_dir = os.path.join(self.config.output_dir, "mask1")
    if not os.path.exists(spatial_dir):
        messagebox.showerror("Error", "No masks found for temporal smoothing.")
        return

    output_dir = os.path.join(self.config.output_dir, "mask_temporal_smoothing")
    os.makedirs(output_dir, exist_ok=True)

    self.start_btn.config(state=tk.DISABLED)
    thread = threading.Thread(
        target=self._temporal_smooth_thread,
        args=(spatial_dir, output_dir),
        daemon=True,
    )
    thread.start()

def _temporal_smooth_thread(self, input_dir, output_dir):
    from core.temporal_smoothing import temporal_smooth_sequence
    try:
        mask_files = sorted(
            [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.tiff', '.tif', '.jpg', '.bmp'))],
            key=extract_numbers
        )

        frames = []
        for fname in mask_files:
            mask = cv2.imread(os.path.join(input_dir, fname), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                frames.append(mask)

        def progress_cb(step_name, current, total):
            self.master.after(0, lambda: self.progress_label.config(text=f"Temporal: {step_name} ({current}/{total})"))

        results = temporal_smooth_sequence(
            frames,
            variance_threshold=self.temporal_var_thresh.get(),
            num_neighbors=self.temporal_neighbors_var.get(),
            progress_callback=progress_cb,
        )

        for i, (result, fname) in enumerate(zip(results, mask_files)):
            cv2.imwrite(os.path.join(output_dir, fname), result)

        self.master.after(0, lambda: messagebox.showinfo("Done", f"Temporal smoothing saved to {output_dir}"))
    except Exception as e:
        self.master.after(0, lambda: messagebox.showerror("Error", str(e)))
    finally:
        self.master.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
```

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat: integrate spatial and temporal smoothing into GUI (Python-only workflow)"
```

---

### Task 4.5: Keyboard Shortcuts

**Files:**
- Modify: `main.py` — bind keyboard shortcuts

**Step 1: Add key bindings in setup_gui**

At the end of `setup_gui()`:
```python
# Keyboard shortcuts
self.master.bind('<Control-z>', lambda e: self.undo_last_point())
self.master.bind('<Control-s>', lambda e: self.save_config())
self.master.bind('<Control-o>', lambda e: self.load_config())
self.master.bind('<Left>', lambda e: self.prev_frame())
self.master.bind('<Right>', lambda e: self.next_frame())
self.master.bind('<space>', lambda e: self.toggle_plotting_mode())
self.master.bind('<Tab>', lambda e: self.toggle_point_mode())
self.master.bind('<Escape>', lambda e: self.stop_processing())
self.master.bind('<Return>', lambda e: self.start_processing() if not self.processing else None)
```

**Step 2: Add keyboard shortcut hints to button text or tooltips**

```python
self.plot_points_btn = ttk.Button(parent, text="Draw Points [Space]", ...)
self.undo_point_btn = ttk.Button(parent, text="Undo [Ctrl+Z]", ...)
self.save_config_btn = ttk.Button(parent, text="Save [Ctrl+S]", ...)
self.load_config_btn = ttk.Button(parent, text="Load [Ctrl+O]", ...)
```

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add keyboard shortcuts (Ctrl+Z undo, arrows navigate, space toggle draw)"
```

---

## Implementation Summary

| Phase | Tasks | Focus |
|-------|-------|-------|
| **Phase 1** | 1.1–1.7 | Bug fixes, dead code removal, correctness |
| **Phase 2** | 2.1–2.4 | Architecture: DPI extraction, threading, cleanup |
| **Phase 3** | 3.1–3.7 | Core features: device auto-detect, frame nav, undo, threshold, PNG, save/load, correction |
| **Phase 4** | 4.1–4.5 | Advanced: checkpoint resume, Python smoothing, GUI integration, shortcuts |

**Total tasks:** 18
**Estimated scope:** ~1500 lines of new/modified code across ~8 files

**Dependencies:**
- Phase 2 depends on Phase 1 (clean code base)
- Phase 3 depends on Phase 2 (threading for correction mode)
- Phase 4 depends on Phase 3 (annotation config for experiment export)
- Tasks within each phase are mostly independent

**New files created:**
- `utils/__init__.py`
- `utils/dpi_scaling.py`
- `core/__init__.py`
- `core/image_processing.py`
- `core/annotation_config.py`
- `core/spatial_smoothing.py`
- `core/temporal_smoothing.py`
