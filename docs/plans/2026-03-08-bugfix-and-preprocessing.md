# Bug Fixes + Image Preprocessing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all critical/high/medium bugs, then add an image preprocessing pipeline that enhances images in-memory before SAM2 inference.

**Architecture:** Preprocessing operates on already-downsampled 1024px images (in-memory only, no disk I/O). A new `core/preprocessing.py` module provides pure functions. A new Sidebar section exposes controls. `ProcessingWorker` applies preprocessing between conversion and SAM2 inference. Existing JPEG conversion skip logic is preserved.

**Tech Stack:** Python 3.10+, PyQt6, OpenCV, NumPy

---

## Part 1: Bug Fixes

### Task 1: Fix QImage Memory Ownership in CanvasPanel

**Files:**
- Modify: `gui/panels/canvas_panel.py:324-344`
- Test: `tests/test_canvas_panel.py` (create)

**Context:** `set_image()` creates QImage from numpy buffer via `image.data.tobytes()` but does NOT call `.copy()`. When the numpy array is garbage-collected, QImage holds a dangling pointer. The sibling function `numpy_to_qimage()` in `core/image_processing.py:307` already does this correctly with `.copy()`.

**Step 1: Write the failing test**

```python
# tests/test_canvas_panel.py
"""Tests for CanvasPanel QImage memory safety."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def test_set_image_uses_qimage_copy():
    """Verify set_image creates an owned QImage copy, not a dangling pointer."""
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication.instance() or QApplication(sys.argv)

    from gui.panels.canvas_panel import CanvasPanel

    panel = CanvasPanel("Test", is_interactive=False)

    # Create a temporary numpy array
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Set image - should not crash
    panel.set_image(image)

    # Delete the original array - if QImage holds a copy, this is safe
    del image

    # Force garbage collection
    import gc
    gc.collect()

    # Access the pixmap - should still be valid (not corrupted)
    pixmap = panel._pixmap_item.pixmap()
    assert not pixmap.isNull()
    assert pixmap.width() == 100
    assert pixmap.height() == 100


def test_set_image_grayscale():
    """Verify grayscale images are handled correctly."""
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication.instance() or QApplication(sys.argv)

    from gui.panels.canvas_panel import CanvasPanel

    panel = CanvasPanel("Test", is_interactive=False)
    gray = np.random.randint(0, 255, (100, 150), dtype=np.uint8)
    panel.set_image(gray)

    pixmap = panel._pixmap_item.pixmap()
    assert not pixmap.isNull()
    assert pixmap.width() == 150
    assert pixmap.height() == 100
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_canvas_panel.py -v`
Expected: May pass intermittently (memory corruption is non-deterministic), but the fix is still necessary.

**Step 3: Fix the implementation**

In `gui/panels/canvas_panel.py`, modify `set_image()` method (lines 324-344). Add `.copy()` to all QImage constructors:

```python
def set_image(self, image: np.ndarray) -> None:
    """Set displayed image (RGB uint8 ndarray or grayscale)."""
    if image.ndim == 2:
        h, w = image.shape
        qimg = QImage(
            image.data.tobytes(), w, h, w, QImage.Format.Format_Grayscale8
        ).copy()
    else:
        h, w, ch = image.shape
        bytes_per_line = ch * w
        if ch == 3:
            qimg = QImage(
                image.data.tobytes(), w, h, bytes_per_line,
                QImage.Format.Format_RGB888,
            ).copy()
        else:
            qimg = QImage(
                image.data.tobytes(), w, h, bytes_per_line,
                QImage.Format.Format_RGBA8888,
            ).copy()

    pixmap = QPixmap.fromImage(qimg)
    # ... rest unchanged
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_canvas_panel.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add gui/panels/canvas_panel.py tests/test_canvas_panel.py
git commit -m "fix: add .copy() to QImage constructors to prevent dangling pointer"
```

---

### Task 2: Fix Worker Signal Disconnect in ProcessingController

**Files:**
- Modify: `controllers/processing_controller.py:467-472`
- Modify: `controllers/smoothing_controller.py:262-274`
- Test: `tests/test_signal_disconnect.py` (create)

**Context:** `_connect_worker()` connects signals but never disconnects old ones. When start/stop/start cycles occur, the same slot gets called multiple times.

**Step 1: Write the failing test**

```python
# tests/test_signal_disconnect.py
"""Tests for worker signal connection/disconnection."""
import pytest
from unittest.mock import MagicMock, patch
from PyQt6.QtCore import QObject


def test_processing_controller_no_duplicate_signals():
    """Starting processing twice should not cause duplicate signal connections."""
    from controllers.app_state import AppState
    from controllers.processing_controller import ProcessingController
    from core.mask_generator import MaskGenerator

    state = AppState()
    state.set_output_dir("/tmp/test_output")
    state._image_files = ["/tmp/img1.jpg"]
    state._points = [[100, 100]]
    state._labels = [1]
    state._device = "cpu"

    gen = MagicMock(spec=MaskGenerator)
    gen.is_initialized = False
    controller = ProcessingController(state, gen)

    # Track how many times processing_finished is emitted
    finish_count = 0
    def on_finished():
        nonlocal finish_count
        finish_count += 1
    controller.processing_finished.connect(on_finished)

    # Simulate two workers being connected (the bug scenario)
    # After fix, old worker signals should be disconnected before new ones
    from controllers.processing_controller import ProcessingWorker
    worker1 = MagicMock(spec=ProcessingWorker)
    worker1.progress = MagicMock()
    worker1.frame_processed = MagicMock()
    worker1.finished = MagicMock()
    worker1.error = MagicMock()

    controller._connect_worker(worker1)
    # The _on_finished slot should be connected exactly once per worker
    worker1.finished.connect.assert_called_once()
```

**Step 2: Fix ProcessingController._connect_worker**

In `controllers/processing_controller.py`, replace `_connect_worker`:

```python
def _connect_worker(self, worker: QThread) -> None:
    """Connect worker signals to controller signals.

    Disconnects any previous worker first to prevent duplicate connections.
    """
    # Disconnect previous worker if it exists
    if self._worker is not None:
        try:
            self._worker.progress.disconnect(self.progress)
            self._worker.frame_processed.disconnect(self.frame_processed)
            self._worker.finished.disconnect(self._on_finished)
            self._worker.error.disconnect(self.processing_error)
        except (TypeError, RuntimeError):
            pass  # Already disconnected or destroyed

    worker.progress.connect(self.progress)
    worker.frame_processed.connect(self.frame_processed)
    worker.finished.connect(self._on_finished)
    worker.error.connect(self.processing_error)
```

**Step 3: Fix SmoothingController similarly**

In `controllers/smoothing_controller.py`, add disconnect logic before connect in both `_connect_worker_spatial` and `_connect_worker_temporal`:

```python
def _connect_worker_spatial(self, worker: SpatialSmoothWorker) -> None:
    self._disconnect_previous_worker()
    worker.progress.connect(
        lambda cur, tot: self.progress.emit(
            cur, tot, f"Spatial smoothing ({cur}/{tot})"
        )
    )
    worker.finished.connect(self._on_finished)
    worker.error.connect(self.smoothing_error)

def _connect_worker_temporal(self, worker: TemporalSmoothWorker) -> None:
    self._disconnect_previous_worker()
    worker.progress.connect(self.progress)
    worker.finished.connect(self._on_finished)
    worker.error.connect(self.smoothing_error)

def _disconnect_previous_worker(self) -> None:
    """Disconnect signals from previous worker to prevent duplicates."""
    if self._worker is not None:
        try:
            self._worker.finished.disconnect(self._on_finished)
            self._worker.error.disconnect(self.smoothing_error)
        except (TypeError, RuntimeError):
            pass
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_signal_disconnect.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add controllers/processing_controller.py controllers/smoothing_controller.py tests/test_signal_disconnect.py
git commit -m "fix: disconnect old worker signals before connecting new ones"
```

---

### Task 3: Fix ThreadPoolExecutor CancelledError

**Files:**
- Modify: `controllers/processing_controller.py:231-238`

**Context:** When `cancel_futures=True` is called, already-dequeued futures may raise `CancelledError` on `.result()`. This is not caught.

**Step 1: Fix the implementation**

In `controllers/processing_controller.py`, wrap `future.result()` in try/except:

```python
# In _convert_images_parallel, replace lines 231-244:
for future in as_completed(futures):
    if self._stop_event.is_set():
        pool.shutdown(wait=False, cancel_futures=True)
        return "cancelled"

    idx, src = futures[future]
    try:
        if not future.result():
            return f"Failed to convert: {os.path.basename(src)}"
    except Exception as exc:
        # CancelledError or any conversion exception
        if self._stop_event.is_set():
            return "cancelled"
        return f"Failed to convert: {os.path.basename(src)} ({exc})"

    completed += 1
    self.progress.emit(
        completed, len(tasks),
        f"Converting images ({completed}/{len(tasks)})",
    )
```

**Step 2: Commit**

```bash
git add controllers/processing_controller.py
git commit -m "fix: catch CancelledError in parallel image conversion"
```

---

### Task 4: Fix Smoothing Worker Error-then-Finished Signal

**Files:**
- Modify: `controllers/smoothing_controller.py:52-58` and `129-135`

**Context:** Both `SpatialSmoothWorker.run()` and `TemporalSmoothWorker.run()` emit `finished` even after emitting `error`. This causes the GUI to transition to REVIEWING state despite the error.

**Step 1: Fix both workers**

```python
# SpatialSmoothWorker.run() - line 52
def run(self):
    try:
        self._run_smoothing()
    except Exception as e:
        logger.exception("Spatial smoothing failed")
        self.error.emit(str(e))
        return  # Do NOT emit finished on error
    self.finished.emit(self._output_dir)

# TemporalSmoothWorker.run() - line 129
def run(self):
    try:
        self._run_smoothing()
    except Exception as e:
        logger.exception("Temporal smoothing failed")
        self.error.emit(str(e))
        return  # Do NOT emit finished on error
    self.finished.emit(self._output_dir)
```

Also update `SmoothingController._on_finished` to clean up worker reference:

```python
def _on_finished(self, output_dir: str) -> None:
    self._worker = None
    self.smoothing_finished.emit(output_dir)
    logger.info(f"Smoothing finished: {output_dir}")
```

And add error handler to also clean up:

```python
# Add to SmoothingController, wire in _connect_worker_* methods:
def _on_error(self, error_msg: str) -> None:
    """Clean up worker reference on error."""
    self._worker = None
    self.smoothing_error.emit(error_msg)
```

Then update `_connect_worker_spatial` and `_connect_worker_temporal` to use `self._on_error` instead of directly forwarding:
```python
worker.error.connect(self._on_error)
```

**Step 2: Commit**

```bash
git add controllers/smoothing_controller.py
git commit -m "fix: don't emit finished signal after error in smoothing workers"
```

---

### Task 5: Fix closeEvent Logic Bug

**Files:**
- Modify: `gui/main_window.py:120-145`

**Context:** When user clicks "No" on the close dialog, `event.ignore()` is called but execution continues to `event.accept()` at line 145.

**Step 1: Fix closeEvent**

```python
def closeEvent(self, event) -> None:
    """Warn user if processing is still running before closing."""
    try:
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
            else:
                event.ignore()
                return  # Critical: must return to prevent event.accept()
    except RuntimeError:
        pass
    event.accept()
```

**Step 2: Commit**

```bash
git add gui/main_window.py
git commit -m "fix: prevent window close when user clicks No on processing dialog"
```

---

### Task 6: Fix RemovePointCommand Direct Private Access

**Files:**
- Modify: `controllers/annotation_controller.py:74-78`

**Context:** `RemovePointCommand.undo()` directly accesses `state._points` and `state._labels` (private attributes), bypassing the public API. Use `AppState.set_points()` instead to maintain consistency.

**Step 1: Fix the undo method**

```python
class RemovePointCommand(Command):
    # ... __init__ and execute unchanged ...

    def undo(self, state) -> None:
        # Re-insert at original position via immutable list operations
        new_points = list(state.points)
        new_labels = list(state.labels)
        new_points.insert(self._index, self._point)
        new_labels.insert(self._index, self._label)
        state.set_points(new_points, new_labels)
```

**Step 2: Commit**

```bash
git add controllers/annotation_controller.py
git commit -m "fix: use public API in RemovePointCommand.undo instead of private access"
```

---

### Task 7: Fix GPU Resource Cleanup on Error

**Files:**
- Modify: `controllers/processing_controller.py:77-84`

**Context:** If `_run_pipeline()` throws, the MaskGenerator is not cleaned up, leaking GPU memory.

**Step 1: Add cleanup in ProcessingWorker.run()**

```python
def run(self):
    try:
        self._run_pipeline()
    except Exception as e:
        logger.exception("Processing failed")
        self.error.emit(str(e))
        # Attempt GPU cleanup on error
        try:
            self._mask_generator.cleanup()
        except Exception:
            logger.warning("Failed to cleanup mask generator after error")
    finally:
        self.finished.emit()
```

**Step 2: Commit**

```bash
git add controllers/processing_controller.py
git commit -m "fix: cleanup GPU resources when processing fails with exception"
```

---

## Part 2: Image Preprocessing Feature

### Task 8: Create Preprocessing Engine (core/preprocessing.py)

**Files:**
- Create: `core/preprocessing.py`
- Test: `tests/test_preprocessing.py` (create)

**Context:** Pure functions that take a uint8 BGR image and return a new uint8 BGR image. Each operation is independent and composable. All functions are stateless and create new arrays (immutable pattern).

**Step 1: Write the failing tests**

```python
# tests/test_preprocessing.py
"""Tests for image preprocessing pipeline."""
import numpy as np
import pytest


class TestBrightness:
    def test_increase_brightness(self):
        from core.preprocessing import adjust_brightness
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        result = adjust_brightness(img, offset=50)
        assert result.dtype == np.uint8
        assert np.all(result == 150)

    def test_brightness_clamp_upper(self):
        from core.preprocessing import adjust_brightness
        img = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = adjust_brightness(img, offset=100)
        assert np.all(result == 255)

    def test_brightness_clamp_lower(self):
        from core.preprocessing import adjust_brightness
        img = np.full((10, 10, 3), 50, dtype=np.uint8)
        result = adjust_brightness(img, offset=-100)
        assert np.all(result == 0)

    def test_zero_offset_no_change(self):
        from core.preprocessing import adjust_brightness
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = adjust_brightness(img, offset=0)
        np.testing.assert_array_equal(result, img)

    def test_does_not_mutate_input(self):
        from core.preprocessing import adjust_brightness
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        original = img.copy()
        adjust_brightness(img, offset=50)
        np.testing.assert_array_equal(img, original)


class TestContrast:
    def test_increase_contrast(self):
        from core.preprocessing import adjust_contrast
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = adjust_contrast(img, factor=2.0)
        # At midpoint 128, contrast scaling should keep it near 128
        assert result.dtype == np.uint8

    def test_zero_contrast(self):
        from core.preprocessing import adjust_contrast
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = adjust_contrast(img, factor=0.0)
        # All pixels should be 128 (mean)
        assert np.all(result == 128)

    def test_unity_factor_no_change(self):
        from core.preprocessing import adjust_contrast
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = adjust_contrast(img, factor=1.0)
        np.testing.assert_array_equal(result, img)


class TestGain:
    def test_gain_doubles(self):
        from core.preprocessing import adjust_gain
        img = np.full((10, 10, 3), 50, dtype=np.uint8)
        result = adjust_gain(img, factor=2.0)
        assert np.all(result == 100)

    def test_gain_clamp(self):
        from core.preprocessing import adjust_gain
        img = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = adjust_gain(img, factor=2.0)
        assert np.all(result == 255)

    def test_unity_gain_no_change(self):
        from core.preprocessing import adjust_gain
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = adjust_gain(img, factor=1.0)
        np.testing.assert_array_equal(result, img)


class TestMinMaxClip:
    def test_clip_range(self):
        from core.preprocessing import clip_min_max
        img = np.array([[[0, 50, 100], [150, 200, 255]]], dtype=np.uint8)
        result = clip_min_max(img, min_val=50, max_val=200)
        # Values outside [50, 200] should be rescaled to [0, 255]
        assert result.dtype == np.uint8
        assert result.min() == 0
        assert result.max() == 255

    def test_full_range_no_change(self):
        from core.preprocessing import clip_min_max
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = clip_min_max(img, min_val=0, max_val=255)
        np.testing.assert_array_equal(result, img)


class TestCLAHE:
    def test_clahe_runs(self):
        from core.preprocessing import apply_clahe
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = apply_clahe(img, clip_limit=2.0, tile_size=8)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_clahe_grayscale(self):
        from core.preprocessing import apply_clahe
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Convert to gray-like (all channels same)
        gray_rgb = np.stack([img[:, :, 0]] * 3, axis=-1)
        result = apply_clahe(gray_rgb, clip_limit=2.0, tile_size=8)
        assert result.shape == gray_rgb.shape


class TestGaussianSmooth:
    def test_smooth_reduces_noise(self):
        from core.preprocessing import gaussian_smooth
        rng = np.random.default_rng(42)
        img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        result = gaussian_smooth(img, sigma=2.0)
        # Smoothed image should have lower variance
        assert result.std() < img.std()

    def test_zero_sigma_no_change(self):
        from core.preprocessing import gaussian_smooth
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = gaussian_smooth(img, sigma=0.0)
        np.testing.assert_array_equal(result, img)


class TestBilateralFilter:
    def test_bilateral_runs(self):
        from core.preprocessing import bilateral_filter
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = bilateral_filter(img, d=9, sigma_color=75, sigma_space=75)
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestBinaryThreshold:
    def test_fixed_threshold(self):
        from core.preprocessing import binary_threshold
        img = np.array([[[50, 50, 50], [200, 200, 200]]], dtype=np.uint8)
        result = binary_threshold(img, threshold=127, method="fixed")
        # First pixel below threshold -> 0, second above -> 255
        assert result[0, 0, 0] == 0
        assert result[0, 1, 0] == 255

    def test_otsu_threshold(self):
        from core.preprocessing import binary_threshold
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = binary_threshold(img, method="otsu")
        unique = np.unique(result)
        # Binary: only 0 and 255
        assert len(unique) <= 2


class TestApplyPipeline:
    def test_empty_pipeline(self):
        from core.preprocessing import apply_pipeline, PreprocessingConfig
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        config = PreprocessingConfig()  # All defaults = no-op
        result = apply_pipeline(img, config)
        np.testing.assert_array_equal(result, img)

    def test_pipeline_with_brightness_and_contrast(self):
        from core.preprocessing import apply_pipeline, PreprocessingConfig
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        config = PreprocessingConfig(brightness=50, contrast=1.5)
        result = apply_pipeline(img, config)
        assert result.dtype == np.uint8
        # Should be different from input
        assert not np.array_equal(result, img)

    def test_pipeline_immutability(self):
        from core.preprocessing import apply_pipeline, PreprocessingConfig
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        original = img.copy()
        config = PreprocessingConfig(brightness=50, gain=2.0)
        apply_pipeline(img, config)
        np.testing.assert_array_equal(img, original)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_preprocessing.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'core.preprocessing'"

**Step 3: Implement core/preprocessing.py**

```python
# core/preprocessing.py
"""Image preprocessing pipeline for SAM2 input enhancement.

All functions are pure: they take an image and return a NEW image.
No mutation of input arrays. All operate on uint8 BGR images.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessingConfig:
    """Immutable configuration for the preprocessing pipeline.

    Default values produce a no-op pipeline (identity transform).
    """
    # Applied in this order:
    gain: float = 1.0              # Linear multiplier (1.0 = no change)
    brightness: int = 0            # Additive offset (-255 to 255, 0 = no change)
    contrast: float = 1.0          # Factor around midpoint (1.0 = no change)
    clip_min: int = 0              # Min value for clip+rescale (0 = no clip)
    clip_max: int = 255            # Max value for clip+rescale (255 = no clip)
    clahe_enabled: bool = False    # CLAHE histogram equalization
    clahe_clip_limit: float = 2.0  # CLAHE clip limit
    clahe_tile_size: int = 8       # CLAHE tile grid size
    gaussian_sigma: float = 0.0    # Gaussian blur sigma (0 = disabled)
    bilateral_enabled: bool = False  # Bilateral filter
    bilateral_d: int = 9            # Bilateral diameter
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    threshold_enabled: bool = False  # Binary thresholding
    threshold_value: int = 127       # Fixed threshold value
    threshold_method: str = "fixed"  # "fixed", "otsu", "adaptive"

    def is_identity(self) -> bool:
        """Return True if this config produces no changes."""
        return (
            self.gain == 1.0
            and self.brightness == 0
            and self.contrast == 1.0
            and self.clip_min == 0
            and self.clip_max == 255
            and not self.clahe_enabled
            and self.gaussian_sigma == 0.0
            and not self.bilateral_enabled
            and not self.threshold_enabled
        )


def adjust_gain(image: np.ndarray, factor: float) -> np.ndarray:
    """Apply linear gain (multiply pixel values)."""
    if factor == 1.0:
        return image
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def adjust_brightness(image: np.ndarray, offset: int) -> np.ndarray:
    """Add constant offset to pixel values."""
    if offset == 0:
        return image
    return np.clip(image.astype(np.int16) + offset, 0, 255).astype(np.uint8)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale pixel values around midpoint (128)."""
    if factor == 1.0:
        return image
    midpoint = 128.0
    result = midpoint + factor * (image.astype(np.float32) - midpoint)
    return np.clip(result, 0, 255).astype(np.uint8)


def clip_min_max(image: np.ndarray, min_val: int, max_val: int) -> np.ndarray:
    """Clip to [min_val, max_val] then rescale to [0, 255]."""
    if min_val == 0 and max_val == 255:
        return image
    clipped = np.clip(image.astype(np.float32), min_val, max_val)
    if max_val > min_val:
        rescaled = (clipped - min_val) / (max_val - min_val) * 255.0
    else:
        rescaled = np.zeros_like(clipped)
    return rescaled.astype(np.uint8)


def apply_clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8,
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    # Convert to LAB, apply CLAHE to L channel, convert back
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(tile_size, tile_size),
    )
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gaussian_smooth(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur."""
    if sigma <= 0:
        return image
    # Kernel size must be odd; use 0 to auto-compute from sigma
    return cv2.GaussianBlur(image, (0, 0), sigma)


def bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Apply bilateral filter (edge-preserving smoothing)."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def binary_threshold(
    image: np.ndarray,
    threshold: int = 127,
    method: str = "fixed",
) -> np.ndarray:
    """Apply binary thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2,
        )
    else:  # fixed
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Convert back to 3-channel for pipeline consistency
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def apply_pipeline(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Apply the full preprocessing pipeline in order.

    Pipeline order:
    1. Gain
    2. Brightness
    3. Contrast
    4. Min/Max clip + rescale
    5. CLAHE
    6. Gaussian smooth
    7. Bilateral filter
    8. Binary threshold

    Returns a NEW image; input is never mutated.
    """
    if config.is_identity():
        return image

    result = image  # No copy needed; each step creates new array

    result = adjust_gain(result, config.gain)
    result = adjust_brightness(result, config.brightness)
    result = adjust_contrast(result, config.contrast)
    result = clip_min_max(result, config.clip_min, config.clip_max)

    if config.clahe_enabled:
        result = apply_clahe(result, config.clahe_clip_limit, config.clahe_tile_size)

    result = gaussian_smooth(result, config.gaussian_sigma)

    if config.bilateral_enabled:
        result = bilateral_filter(
            result, config.bilateral_d,
            config.bilateral_sigma_color, config.bilateral_sigma_space,
        )

    if config.threshold_enabled:
        result = binary_threshold(
            result, config.threshold_value, config.threshold_method,
        )

    return result
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_preprocessing.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add core/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add image preprocessing engine with composable pipeline"
```

---

### Task 9: Add Preprocessing Config to AppState

**Files:**
- Modify: `controllers/app_state.py`
- Test: `tests/test_app_state_preprocessing.py` (create)

**Context:** AppState needs to store the preprocessing configuration so it can be passed to ProcessingWorker and used for live preview.

**Step 1: Write the test**

```python
# tests/test_app_state_preprocessing.py
"""Tests for preprocessing config in AppState."""
from core.preprocessing import PreprocessingConfig


def test_default_preprocessing_config():
    from controllers.app_state import AppState
    state = AppState()
    assert state.preprocessing_config.is_identity()


def test_set_preprocessing_config():
    from controllers.app_state import AppState
    state = AppState()

    config = PreprocessingConfig(brightness=50, contrast=1.5)
    state.set_preprocessing_config(config)

    assert state.preprocessing_config.brightness == 50
    assert state.preprocessing_config.contrast == 1.5


def test_preprocessing_changed_signal():
    from controllers.app_state import AppState
    state = AppState()

    received = []
    state.preprocessing_changed.connect(lambda: received.append(True))

    config = PreprocessingConfig(gain=2.0)
    state.set_preprocessing_config(config)

    assert len(received) == 1
```

**Step 2: Add to AppState**

In `controllers/app_state.py`, add:

1. Import at top:
```python
from core.preprocessing import PreprocessingConfig
```

2. Add signal:
```python
preprocessing_changed = pyqtSignal()  # preprocessing config updated
```

3. Add to `__init__`:
```python
self._preprocessing_config = PreprocessingConfig()
```

4. Add property + setter:
```python
@property
def preprocessing_config(self) -> PreprocessingConfig:
    return self._preprocessing_config

def set_preprocessing_config(self, config: PreprocessingConfig) -> None:
    self._preprocessing_config = config
    self.preprocessing_changed.emit()
```

**Step 3: Run tests**

Run: `python -m pytest tests/test_app_state_preprocessing.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add controllers/app_state.py tests/test_app_state_preprocessing.py
git commit -m "feat: add preprocessing config to AppState with signal"
```

---

### Task 10: Add Preprocessing Sidebar Panel

**Files:**
- Modify: `gui/panels/sidebar.py`

**Context:** Add a new "Preprocessing" collapsible section between "Model Configuration" and "Spatial Smoothing" in the sidebar. It includes controls for all preprocessing parameters and emits a signal whenever config changes.

**Step 1: Add the signal and UI**

In `gui/panels/sidebar.py`:

1. Add new signal:
```python
preprocessing_changed = pyqtSignal(object)  # PreprocessingConfig
```

2. After the model_section block (line ~216), before spatial_section, add:

```python
# --- Preprocessing section ---
preprocess_section = CollapsibleSection(
    "Preprocessing", icon_name="sliders", default_open=False
)

# Gain + Brightness row
preprocess_grid1 = QWidget()
preprocess_grid1_layout = QGridLayout(preprocess_grid1)
preprocess_grid1_layout.setContentsMargins(0, 0, 0, 0)
preprocess_grid1_layout.setSpacing(8)

self._pp_gain = NumberInput(
    "Gain", default=1.0, min_val=0.1, max_val=5.0,
    step=0.1, decimals=2, icon_name="zap",
)
self._pp_gain.setToolTip("Linear multiplier for pixel intensity.\n1.0 = no change.")
preprocess_grid1_layout.addWidget(self._pp_gain, 0, 0)

self._pp_brightness = NumberInput(
    "Brightness", default=0, min_val=-255, max_val=255,
    step=1, decimals=0, icon_name="sun",
)
self._pp_brightness.setToolTip("Additive brightness offset.\n0 = no change.")
preprocess_grid1_layout.addWidget(self._pp_brightness, 0, 1)

preprocess_section.add_widget(preprocess_grid1)

# Contrast
self._pp_contrast = NumberInput(
    "Contrast", default=1.0, min_val=0.0, max_val=5.0,
    step=0.1, decimals=2, icon_name="contrast",
)
self._pp_contrast.setToolTip("Contrast factor around midpoint (128).\n1.0 = no change.")
preprocess_section.add_widget(self._pp_contrast)

# Min/Max Clip row
preprocess_grid2 = QWidget()
preprocess_grid2_layout = QGridLayout(preprocess_grid2)
preprocess_grid2_layout.setContentsMargins(0, 0, 0, 0)
preprocess_grid2_layout.setSpacing(8)

self._pp_clip_min = NumberInput(
    "Clip Min", default=0, min_val=0, max_val=254,
    step=1, decimals=0, icon_name="arrow-down",
)
preprocess_grid2_layout.addWidget(self._pp_clip_min, 0, 0)

self._pp_clip_max = NumberInput(
    "Clip Max", default=255, min_val=1, max_val=255,
    step=1, decimals=0, icon_name="arrow-up",
)
preprocess_grid2_layout.addWidget(self._pp_clip_max, 0, 1)

preprocess_section.add_widget(preprocess_grid2)

# CLAHE
from PyQt6.QtWidgets import QCheckBox
self._pp_clahe_check = QCheckBox("CLAHE (Adaptive Histogram Eq.)")
self._pp_clahe_check.setStyleSheet(
    f"color: {Colors.TEXT_PRIMARY}; font-size: {Fonts.SIZE_BASE}px; "
    f"background: transparent;"
)
preprocess_section.add_widget(self._pp_clahe_check)

preprocess_grid3 = QWidget()
preprocess_grid3_layout = QGridLayout(preprocess_grid3)
preprocess_grid3_layout.setContentsMargins(0, 0, 0, 0)
preprocess_grid3_layout.setSpacing(8)

self._pp_clahe_clip = NumberInput(
    "CLAHE Clip", default=2.0, min_val=0.1, max_val=40.0,
    step=0.5, decimals=1, icon_name="gauge",
)
preprocess_grid3_layout.addWidget(self._pp_clahe_clip, 0, 0)

self._pp_clahe_tile = NumberInput(
    "Tile Size", default=8, min_val=2, max_val=32,
    step=1, decimals=0, icon_name="hash",
)
preprocess_grid3_layout.addWidget(self._pp_clahe_tile, 0, 1)

preprocess_section.add_widget(preprocess_grid3)

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
    f"color: {Colors.TEXT_PRIMARY}; font-size: {Fonts.SIZE_BASE}px; "
    f"background: transparent;"
)
preprocess_section.add_widget(self._pp_bilateral_check)

# Binary Threshold
self._pp_threshold_check = QCheckBox("Binary Threshold")
self._pp_threshold_check.setStyleSheet(
    f"color: {Colors.TEXT_PRIMARY}; font-size: {Fonts.SIZE_BASE}px; "
    f"background: transparent;"
)
preprocess_section.add_widget(self._pp_threshold_check)

preprocess_grid4 = QWidget()
preprocess_grid4_layout = QGridLayout(preprocess_grid4)
preprocess_grid4_layout.setContentsMargins(0, 0, 0, 0)
preprocess_grid4_layout.setSpacing(8)

self._pp_threshold_val = NumberInput(
    "Threshold", default=127, min_val=0, max_val=255,
    step=1, decimals=0, icon_name="hash",
)
preprocess_grid4_layout.addWidget(self._pp_threshold_val, 0, 0)

self._pp_threshold_method = SelectField(
    "Method",
    options=["Fixed", "Otsu", "Adaptive"],
    default="Fixed",
)
preprocess_grid4_layout.addWidget(self._pp_threshold_method, 0, 1)

preprocess_section.add_widget(preprocess_grid4)

scroll_layout.addWidget(preprocess_section)
```

3. Add a public method to build the config:

```python
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
```

**Step 2: Commit**

```bash
git add gui/panels/sidebar.py
git commit -m "feat: add preprocessing controls to sidebar panel"
```

---

### Task 11: Integrate Preprocessing into ProcessingWorker Pipeline

**Files:**
- Modify: `controllers/processing_controller.py`
- Modify: `gui/main_window.py`

**Context:** The preprocessing step happens AFTER image conversion (which produces 1024px images) and BEFORE SAM2 inference. It reads converted images, applies preprocessing in-memory, and writes them back (overwriting the converted files). This preserves the "skip if already converted" logic — the conversion skip works as before, and preprocessing is always re-applied in-memory.

**IMPORTANT DESIGN NOTE:** Preprocessing is applied to the converted images on disk (overwritten) so SAM2 reads the preprocessed versions. This is necessary because SAM2's `set_video()` reads from a directory. However, the originals are NOT modified.

**Step 1: Add preprocessing_config to ProcessingWorker constructor**

In `controllers/processing_controller.py`:

1. Add import:
```python
from core.preprocessing import PreprocessingConfig, apply_pipeline
```

2. Add parameter to `ProcessingWorker.__init__`:
```python
def __init__(
    self,
    mask_generator: MaskGenerator,
    image_files: list[str],
    output_dir: str,
    model_cfg: str,
    checkpoint: str,
    device: str,
    points: list[list[float]],
    labels: list[int],
    threshold: float,
    start_frame: int,
    end_frame: int,
    intermediate_format: str,
    force_reprocess: bool = False,
    preprocessing_config: PreprocessingConfig | None = None,
    parent=None,
):
    # ... existing code ...
    self._preprocessing_config = preprocessing_config or PreprocessingConfig()
```

3. Add preprocessing stage in `_run_pipeline()` after conversion and before model init:

```python
# --- Stage 1.5: Apply preprocessing to converted images (in-place) ---
if not self._preprocessing_config.is_identity():
    self._apply_preprocessing(converted_dir, ext, total_files)
```

4. Add the preprocessing method:

```python
def _apply_preprocessing(self, converted_dir: str, ext: str, total_files: int) -> None:
    """Apply preprocessing to converted images in-place."""
    self.progress.emit(0, total_files, "Applying preprocessing...")

    for i in range(total_files):
        if self._stop_event.is_set():
            return

        img_path = os.path.join(converted_dir, f"{i:06d}{ext}")
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        processed = apply_pipeline(img, self._preprocessing_config)
        cv2.imwrite(img_path, processed)

        self.progress.emit(
            i + 1, total_files,
            f"Preprocessing ({i + 1}/{total_files})",
        )
```

5. Pass config in `ProcessingController.start_processing()`:

```python
worker = ProcessingWorker(
    # ... existing args ...
    preprocessing_config=self._state.preprocessing_config,
)
```

**Step 2: Wire sidebar to AppState in MainWindow**

In `gui/main_window.py`, add to `_connect_signals()`:

```python
# Preprocessing config from sidebar
# Update state when processing starts (read at start time, not on every change)
```

And in `_on_start_processing()`, read config from sidebar before starting:

```python
def _on_start_processing(self) -> None:
    # ... existing validation ...

    # Capture current preprocessing config from sidebar
    preprocessing_config = self._sidebar.get_preprocessing_config()
    self._state.set_preprocessing_config(preprocessing_config)

    self._status_bar.reset_timer()
    self._state.set_state(self._state.State.PROCESSING)
    self._processing.start_processing()
```

**Step 3: Commit**

```bash
git add controllers/processing_controller.py gui/main_window.py
git commit -m "feat: integrate preprocessing pipeline into ProcessingWorker"
```

---

### Task 12: Add Preview Preprocessing on Current Frame

**Files:**
- Modify: `gui/main_window.py`
- Modify: `gui/panels/sidebar.py`

**Context:** When the user changes preprocessing settings, show a preview of the effect on the Original panel. This previews at display resolution only (fast, <1ms for 1024px image).

**Step 1: Add preview button to sidebar**

In `gui/panels/sidebar.py`, add a "Preview" button after the preprocessing controls:

```python
self._pp_preview_btn = _create_accent_button("Preview Preprocessing")
self._pp_preview_btn.clicked.connect(self._on_preview_preprocessing)
preprocess_section.add_widget(self._pp_preview_btn)

# Also add a "Reset" button
self._pp_reset_btn = QPushButton("Reset to Original")
self._pp_reset_btn.setMinimumHeight(30)
self._pp_reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
self._pp_reset_btn.clicked.connect(lambda: self.preprocessing_preview_reset.emit())
preprocess_section.add_widget(self._pp_reset_btn)
```

Add signals:
```python
preprocessing_preview_requested = pyqtSignal(object)  # PreprocessingConfig
preprocessing_preview_reset = pyqtSignal()
```

Add handler:
```python
def _on_preview_preprocessing(self) -> None:
    config = self.get_preprocessing_config()
    self.preprocessing_preview_requested.emit(config)
```

**Step 2: Handle preview in MainWindow**

In `gui/main_window.py`, add signal connections:

```python
self._sidebar.preprocessing_preview_requested.connect(
    self._on_preview_preprocessing
)
self._sidebar.preprocessing_preview_reset.connect(
    self._on_reset_preprocessing_preview
)
```

Add handlers:

```python
def _on_preview_preprocessing(self, config) -> None:
    """Preview preprocessing effect on current frame."""
    if self._state is None or self._state.current_original is None:
        return

    from core.preprocessing import apply_pipeline
    import cv2

    # Current original is RGB; preprocessing works on BGR
    original_bgr = cv2.cvtColor(self._state.current_original, cv2.COLOR_RGB2BGR)

    # Downsample for fast preview if large
    h, w = original_bgr.shape[:2]
    max_preview = 1024
    if max(h, w) > max_preview:
        scale = max_preview / max(h, w)
        preview_bgr = cv2.resize(
            original_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        preview_bgr = original_bgr

    processed = apply_pipeline(preview_bgr, config)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    # Scale back to original display size if needed
    if processed_rgb.shape[:2] != (h, w):
        processed_rgb = cv2.resize(
            processed_rgb, (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

    self._state.set_display_images(original=processed_rgb)

def _on_reset_preprocessing_preview(self) -> None:
    """Reset to the unprocessed original image."""
    self._load_current_frame()
```

**Step 3: Commit**

```bash
git add gui/panels/sidebar.py gui/main_window.py
git commit -m "feat: add preprocessing preview on current frame"
```

---

## Summary of All Tasks

| Task | Type | Severity | Description |
|------|------|----------|-------------|
| 1 | Bug Fix | CRITICAL | QImage memory ownership (.copy()) |
| 2 | Bug Fix | CRITICAL | Worker signal disconnect |
| 3 | Bug Fix | CRITICAL | ThreadPoolExecutor CancelledError |
| 4 | Bug Fix | HIGH | Smoothing error-then-finished signal |
| 5 | Bug Fix | MEDIUM | closeEvent logic error |
| 6 | Bug Fix | HIGH | RemovePointCommand private access |
| 7 | Bug Fix | MEDIUM | GPU cleanup on error |
| 8 | Feature | - | Preprocessing engine (core/preprocessing.py) |
| 9 | Feature | - | AppState preprocessing config |
| 10 | Feature | - | Sidebar preprocessing controls |
| 11 | Feature | - | ProcessingWorker integration |
| 12 | Feature | - | Live preview of preprocessing |

**Estimated total: ~12 tasks, each 2-10 minutes**

**Key design decisions:**
- Preprocessing runs on 1024px converted images (fast, <1ms/frame)
- `PreprocessingConfig` is a frozen dataclass (immutable)
- Pipeline is composable: each function is pure and creates new arrays
- Skip conversion logic preserved: if converted files exist, skip conversion, but still apply preprocessing
- Preview operates at display resolution only
