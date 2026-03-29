# DIC Mask Generator v2.0 — User Manual

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Interface Overview](#2-interface-overview)
3. [Complete Workflow](#3-complete-workflow)
4. [Preprocessing Pipeline](#4-preprocessing-pipeline)
5. [SAM2 Processing](#5-sam2-processing)
6. [Post-Processing (Smoothing)](#6-post-processing-smoothing)
7. [Project Management](#7-project-management)
8. [Batch Processing](#8-batch-processing)
9. [Contour Export](#9-contour-export)
10. [Keyboard Shortcuts Reference](#10-keyboard-shortcuts-reference)
11. [Menu Reference](#11-menu-reference)
12. [Parameter Reference](#12-parameter-reference)
13. [Tips & Best Practices](#13-tips--best-practices)
14. [FAQ](#14-faq)

---

## 1. Getting Started

### First Launch

When you launch DIC Mask Generator for the first time, a **Welcome Dialog** appears explaining the basic workflow. The application opens with an empty workspace in the `INIT` state.

### Minimum Setup

To start working, you need:

1. **Input directory** — A folder containing your image sequence (TIFF, PNG, JPEG, or BMP).
2. **Output directory** — A folder where masks and intermediate files will be saved.
3. **At least one SAM2 checkpoint** — A `.pt` file in the `checkpoints/` directory.

### Supported Image Formats

| Format | Input | Mask Output |
|--------|:-----:|:-----------:|
| TIFF   | Yes   | Yes (default) |
| PNG    | Yes   | Yes (optional) |
| JPEG   | Yes   | No (lossy) |
| BMP    | Yes   | No |

Images are sorted using **natural sorting** (e.g., `frame_2.png` before `frame_10.png`).

---

## 2. Interface Overview

The application window is divided into five regions:

```
┌─────────────────────────────────────────────────────────┐
│ Menu Bar (File | Edit | View | Tools | Help)            │
├──────────┬──────────────────────────────────────────────┤
│          │ Toolbar (Select | Draw | Erase | Process)    │
│          ├──────────────────────────────────────────────┤
│ Sidebar  │                                              │
│ (300px)  │  Canvas Area (Original | Mask | Overlay)     │
│          │                                              │
│          ├──────────────────────────────────────────────┤
│          │ Filmstrip (thumbnail preview)                │
│          ├──────────────────────────────────────────────┤
│          │ Frame Navigator (slider + controls)          │
├──────────┴──────────────────────────────────────────────┤
│ Status Bar (status | progress | VRAM | timer)           │
└─────────────────────────────────────────────────────────┘
```

### Sidebar

The sidebar has two switchable panels, toggled via the **Processing** / **Post-Processing** buttons at the top:

**Processing Panel** (default):
- **File Paths** — Input/output directory selectors.
- **Preprocessing** — 28-parameter pipeline with live preview (see [Section 4](#4-preprocessing-pipeline)).
- **Model Configuration** — Device, model, format, threshold settings.

**Post-Processing Panel**:
- **Spatial Smoothing** — Perona-Malik anisotropic diffusion on masks.
- **Temporal Smoothing** — 3D Gaussian filter for frame-to-frame coherence.
- **Mask Statistics** — Area, consistency, and anomaly analysis.

### Canvas Area

Three synchronized panels show the current frame:

| Panel | Content |
|-------|---------|
| **Original** | Source image (or preprocessed preview) |
| **Mask** | Generated binary mask (white = foreground) |
| **Overlay** | Source image with mask overlay (red, adjustable alpha) |

All three panels share zoom and pan state — scrolling or zooming one panel updates all three.

### Toolbar

| Tool | Key | Function |
|------|-----|----------|
| Select | `V` | Place foreground (positive) annotation points |
| Draw | `D` | Place background (negative) annotation points |
| Erase | `E` | Enter correction mode for re-propagation |

### Status Bar

Displays (left to right):
- Current status message and level (ready / processing / error / warning)
- Processing progress bar with ETA
- GPU device name and VRAM usage (updated every 5 seconds)
- Elapsed timer during processing

---

## 3. Complete Workflow

### Step 1: Set Directories

1. In the sidebar **File Paths** section, click the folder icon next to **Input Directory** and select your image folder.
2. Set the **Output Directory** the same way.

The app auto-discovers all supported images in the input directory and displays the count in the frame navigator. Frame range defaults to all frames.

### Step 2: Configure Model (Optional)

In the **Model Configuration** section:

- **Device** — `CUDA` for GPU (default if available), `CPU` as fallback.
- **Model** — `SAM2 Hiera Large` for best quality, `Tiny` for speed.
- **Mask Threshold** — Default `0.0` works well. Increase for stricter masks (smaller), decrease for more permissive (larger).

### Step 3: Preprocess (Optional)

If your images need enhancement (e.g., DIC microscopy), open the **Preprocessing** section:

1. Select a **built-in preset** (e.g., "DIC Microscopy") or adjust parameters manually.
2. The canvas shows a **live preview** of preprocessing as you adjust sliders (500ms debounce).
3. Click **Save Preprocessed** to write enhanced images to disk. SAM2 will automatically use the preprocessed versions.

### Step 4: Annotate

1. Press `V` (or click the Select tool) to enter foreground mode.
2. Click on the first frame to place **green foreground points** on your target region.
3. Press `Space` to toggle to background mode, then click to place **red background points** on areas to exclude.
4. Use `Ctrl+Z` / `Ctrl+Y` to undo/redo.

**Tips**:
- A single well-placed foreground point is often enough for simple objects.
- Background points help when the model includes unwanted areas.
- You only need to annotate the **first frame** — SAM2 propagates to all frames automatically.

### Step 5: Process

1. Click **Start Processing** in the toolbar or press `Ctrl+Enter`.
2. The app converts images, loads the SAM2 model, and propagates your annotations across all frames.
3. Progress is shown in the status bar with frame count and ETA.
4. Each frame's mask appears in real time as it is processed.

### Step 6: Review

After processing completes:
- Browse frames with `Left` / `Right` arrow keys.
- The mask and overlay panels show the generated results.
- Press `Space` to toggle overlay visibility for comparison.
- Use `Ctrl+0` to reset zoom, `Ctrl+1` to fit to window.

### Step 7: Correct (If Needed)

If masks are inaccurate on certain frames:

1. Navigate to the problematic frame.
2. Press `E` to enter correction mode.
3. Add new foreground/background points on that frame.
4. Click **Start Processing** — SAM2 re-propagates from the corrected frame.

### Step 8: Post-Process

After segmentation completes, the app asks: *"Would you like to enter post-processing mode?"*

- Click **Yes** to switch to the Post-Processing panel.
- Or switch manually by clicking the **Post-Processing** tab in the sidebar.

Apply **Spatial Smoothing** for edge refinement or **Temporal Smoothing** for frame-to-frame consistency (see [Section 6](#6-post-processing-smoothing)).

### Step 9: Export

Masks are automatically saved to `output/run_YYYYMMDD_HHMMSS/masks/`. Additional export options:

- **Overlay images**: Enable "Auto-export overlay images" in Model Configuration before processing.
- **Contour export**: `Tools > Export Contours as PNG/SVG...`

---

## 4. Preprocessing Pipeline

The preprocessing pipeline applies transformations in a **fixed order** to ensure reproducibility. Every configuration is immutable (`frozen dataclass`) — adjusting any parameter creates a new config object.

### Pipeline Execution Order

```
Input Image
  │
  ├─ 1. Gain (linear multiplier)
  ├─ 2. Brightness (additive offset)
  ├─ 3. Contrast (factor around midpoint)
  ├─ 4. Clip Min/Max (dynamic range rescaling)
  ├─ 5. CLAHE (adaptive histogram equalization)
  ├─ 6. Gaussian Blur
  ├─ 7. Bilateral Filter (edge-preserving)
  ├─ 8. Median Filter (salt & pepper noise)
  ├─ 9. Box/Mean Filter
  ├─ 10. Non-Local Means Denoise
  ├─ 11. Anisotropic Diffusion (Perona-Malik)
  ├─ 12. Binary Threshold (Fixed / Otsu / Adaptive)
  ├─ 13. Invert
  ├─ 14. Morphology (dilate/erode/open/close/gradient/tophat/blackhat)
  ├─ 15. Fill Holes
  └─ 16. Shape Overlays (add/cut rectangles, circles, polygons)
  │
Output Image
```

### Category: Tone

Access via the **Tone** hover button in the Preprocessing section.

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| Gain | 0.1 – 5.0 | 1.0 | Multiplies pixel values. >1.0 brightens, <1.0 darkens |
| Brightness | -255 – +255 | 0 | Adds a constant offset to all pixels |
| Contrast | 0.0 – 5.0 | 1.0 | Scales around the midpoint (127.5). >1.0 increases contrast |
| Clip Min | 0 – 254 | 0 | Minimum input value; pixels below are mapped to 0 |
| Clip Max | 1 – 255 | 255 | Maximum input value; pixels above are mapped to 255 |
| CLAHE | on/off | off | Contrast Limited Adaptive Histogram Equalization |
| CLAHE Clip Limit | 0.1 – 40.0 | 2.0 | Controls contrast amplification (higher = more enhancement) |
| CLAHE Tile Size | 2 – 32 | 8 | Grid tile size for local histogram equalization |

### Category: Smoothing

Access via the **Smoothing** hover button.

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| Gaussian Sigma | 0.0 – 10.0 | 0.0 | Standard deviation for Gaussian blur (0 = disabled) |
| Bilateral | on/off | off | Edge-preserving smoothing |
| Median Ksize | 0 – 31 | 0 | Kernel size for median filter (0 = disabled, must be odd) |
| Box Ksize | 0 – 31 | 0 | Kernel size for box/mean filter (0 = disabled) |
| NLM Denoise | on/off | off | Non-Local Means denoising (slow but effective) |
| NLM Strength | 1.0 – 40.0 | 10.0 | Higher = stronger denoising but more blurring |
| Anisotropic Diffusion | on/off | off | Perona-Malik edge-preserving smoothing |
| Diffusion Iterations | 1 – 100 | 10 | More iterations = smoother result |
| Diffusion Kappa | 1.0 – 200.0 | 30.0 | Conductance threshold — higher preserves fewer edges |
| Diffusion Option | 1 or 2 | 1 | 1 = exponential (sharp edges), 2 = reciprocal (broad regions) |

### Category: Binarize

Access via the **Binarize** hover button.

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| Binary Threshold | on/off | off | Enable binary thresholding |
| Threshold Value | 0 – 255 | 127 | Pixel value cutoff (used with Fixed method) |
| Threshold Method | Fixed / Otsu / Adaptive | Fixed | Otsu auto-selects optimal threshold |
| Invert | on/off | off | Flip black and white after thresholding |

### Category: Morphology

Access via the **Morphology** hover button.

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| Morphology | on/off | off | Enable morphological operation |
| Operation | Dilate / Erode / Open / Close / Gradient / Top-hat / Black-hat | Close | Type of morphological operation |
| Kernel Size | 3 – 51 | 3 | Structuring element size (odd numbers only) |
| Iterations | 1 – 20 | 1 | Number of times to apply the operation |
| Fill Holes | on/off | off | Flood-fill enclosed dark regions inside contours |

### Shape Overlays

Use **Add** / **Cut** buttons to draw shapes on the canvas:
- **Add** shapes fill an area with white (foreground).
- **Cut** shapes fill an area with black (background).
- Available shapes: Rectangle, Circle, Polygon.

Shapes are applied as the last preprocessing step, after all pixel transformations.

### Built-in Presets

| Preset | Description | Key Settings |
|--------|-------------|-------------|
| None (identity) | No preprocessing | All defaults |
| DIC Microscopy | Optimized for DIC images | Contrast 1.5, CLAHE on, Gaussian 0.5 |
| Fluorescence | Fluorescence imaging | Gain 2.0, Clip 10-240, CLAHE on, Gaussian 0.8 |
| Phase Contrast | Phase contrast microscopy | Contrast 1.8, Bilateral on |
| Brightfield | Standard brightfield | Contrast 1.3, CLAHE on |
| High Noise (Denoise) | Strong noise reduction | Gaussian 1.5, NLM on |
| Edge Enhancement | Emphasize boundaries | Contrast 2.0, Diffusion on (Option 2) |

### Custom Presets

- **Save Preset** — Exports current parameters to a JSON file.
- **Load Preset** — Imports parameters from a previously saved JSON file.

### Custom Frame Selection

Enable **Custom frames** checkbox and enter frame ranges like `1-10, 15, 20-30`. Frames outside this range will be skipped during preprocessing.

---

## 5. SAM2 Processing

### How It Works

1. **Image conversion** — Source images are converted to the format SAM2 expects (JPEG or PNG), automatically downsampled to max 1024px for inference.
2. **Model initialization** — The selected SAM2 model is loaded onto the chosen device (GPU/CPU).
3. **Prompt encoding** — Your annotation points (foreground/background) are encoded as prompts for frame 0.
4. **Video propagation** — SAM2 propagates the segmentation across all frames in the selected range.
5. **Mask upsampling** — Inferred masks are upsampled to the original image resolution.
6. **Mask saving** — Binary masks are saved to `output/run_YYYYMMDD_HHMMSS/masks/`.

### Model Comparison

| Model | VRAM Usage | Speed | Quality | Best For |
|-------|:----------:|:-----:|:-------:|----------|
| Hiera Large | ~2-3 GB | Slowest | Best | Final results, complex shapes |
| Hiera Base Plus | ~1.5 GB | Moderate | Good | Balanced workflows |
| Hiera Small | ~1 GB | Fast | Adequate | Quick iterations |
| Hiera Tiny | ~500 MB | Fastest | Basic | Preview, large batches |

### Mask Threshold

The **Mask Threshold** slider controls the logit cutoff for binarization:

- `0.0` (default) — Standard cutoff. Works well for most images.
- `> 0` — Stricter, produces smaller masks (removes low-confidence regions).
- `< 0` — More permissive, produces larger masks (includes uncertain regions).

### Correction Mode

After initial processing, you can refine results on any frame:

1. Navigate to a frame with inaccurate masks.
2. Press `E` to enter correction mode.
3. Add/adjust annotation points.
4. Press `Ctrl+Enter` — SAM2 re-propagates from this frame forward and backward.

This is particularly useful for long sequences where error accumulates.

---

## 6. Post-Processing (Smoothing)

After segmentation, switch to the **Post-Processing** panel for mask refinement. Both smoothing operations process **all mask files** in the `masks/` directory.

### How Smoothing Results Are Displayed

When you apply smoothing, the software automatically:

1. **Processes all masks** in the `masks/` directory.
2. **Saves results** to a new directory (`mask_spatial_smoothing/` or `mask_temporal_smoothing/`) by default, or overwrites `masks/` if "Replace original masks" is checked.
3. **Switches the display** to show the smoothed masks immediately after completion.
4. **Shows a status message** indicating which mask directory is currently being viewed (e.g., "Smoothing complete — viewing: mask_spatial_smoothing").

You can tell which masks you are viewing by checking the status bar. When you start a new processing run, the display automatically switches back to `masks/`.

### Spatial Smoothing

Applies **Perona-Malik anisotropic diffusion** to each mask independently. This is edge-preserving: it smooths mask interiors while maintaining sharp boundaries.

| Parameter | Range | Default | Guidance |
|-----------|-------|---------|----------|
| Iterations | 1 – 500 | 50 | More = smoother. 20-50 for light smoothing, 100+ for aggressive. |
| dt | 0.001 – 1.0 | 0.1 | Time step. Must be <= 0.25 for numerical stability. |
| Kappa | 0.1 – 200.0 | 30.0 | Gradient threshold. Lower = preserves more edges. |
| Option | 1 or 2 | 1 | 1 = exponential (favors sharp edges), 2 = reciprocal (favors broad regions). |
| Replace originals | on/off | off | If on, overwrites `masks/`. If off, writes to `mask_spatial_smoothing/`. |

### Temporal Smoothing

Applies **3D Gaussian filtering** across the frame sequence for spatio-temporal coherence. Automatically detects and interpolates anomalous ("bad") frames.

| Parameter | Range | Default | Guidance |
|-----------|-------|---------|----------|
| Var Threshold | 0 – 999999 | 50000 | Variance threshold for bad frame detection. 0 = adaptive (recommended). |
| Neighbors | 1 – 10 | 2 | Number of neighboring frames for Gaussian averaging. |
| Sigma | 0.1 – 20.0 | 2.0 | Gaussian sigma. Higher = more averaging across frames. |
| Replace originals | on/off | off | If on, overwrites `masks/`. If off, writes to `mask_temporal_smoothing/`. |

**Bad frame detection**: The algorithm computes per-frame variance. Frames with variance deviating more than 5x MAD (Median Absolute Deviation) from the median are flagged and replaced by Gaussian-weighted interpolation from neighbors.

### Progress and Display

- Both operations show a **progress bar** with frame count, percentage, and estimated time remaining (ETA).
- After completion, the canvas **automatically switches to the smoothed output directory** and reloads the current frame.
- The **status bar** shows which mask directory is currently being viewed (e.g., `mask_spatial_smoothing`).
- The **Mask Statistics** section in the Post-Processing panel also reads from the active mask directory, so you can compare statistics before and after smoothing.
- Running a new segmentation automatically switches the display back to `masks/`.

---

## 7. Project Management

### Save Project (Ctrl+Shift+S)

Saves the complete workspace state as a `.s2proj` file (JSON format):

| Saved Data | Details |
|------------|---------|
| Paths | Input and output directories |
| Model | Name, device, threshold, formats |
| Annotations | All point coordinates and labels |
| Frames | Start/end range, bookmarked frames |
| Preprocessing | All 28+ parameters |
| Overlay | Alpha transparency, color |
| Metadata | Timestamp, version |

### Open Project (Ctrl+Shift+P)

Restores the entire workspace from a `.s2proj` file. Image files in the saved input directory are re-discovered automatically.

### Recent Projects

`File > Recent Projects` shows recently opened/saved projects for quick access.

### Annotation Config (Lightweight)

For sharing just annotation points (without all settings):
- **Save Config** (`Ctrl+S`) — Saves points and labels to a `.json` file.
- **Load Config** (`Ctrl+O`) — Loads points and labels from a `.json` file.

---

## 8. Batch Processing

`Tools > Batch Processing...` opens a dialog for processing multiple image directories with the same settings.

### How to Use

1. Open the Batch Processing dialog.
2. Click **Add Directory** to add input/output directory pairs.
3. Click **Start** — each directory is processed sequentially.
4. The current model settings and annotation points are applied to all directories.

This is useful for applying the same segmentation to multiple experiments or time-lapse sequences.

---

## 9. Contour Export

`Tools > Export Contours as PNG...` or `Export Contours as SVG...`

Extracts mask boundaries from all generated masks and exports them:

- **PNG format** — White contour lines on black background. Good for visualization and further image processing.
- **SVG format** — Vector graphics, resolution-independent. Ideal for publications and reports.

The export processes all masks in the output `masks/` directory and creates a new directory (`contours_png/` or `contours_svg/`) with one file per mask.

---

## 10. Keyboard Shortcuts Reference

### Annotation

| Key | Action |
|-----|--------|
| `V` | Foreground point mode (green, positive) |
| `D` | Background point mode (red, negative) |
| `Space` | Toggle between foreground/background mode |
| `E` | Enter correction mode |
| `Ctrl+Z` | Undo last annotation action |
| `Ctrl+Y` | Redo annotation |
| `Ctrl+Shift+Z` | Redo annotation (alternate) |
| `Delete` | Delete selected points |

### Navigation

| Key | Action |
|-----|--------|
| `Left` | Previous frame |
| `Right` | Next frame |
| `Home` | Jump to first frame |
| `End` | Jump to last frame |
| `Ctrl+G` | Go to specific frame number |
| `M` | Toggle bookmark on current frame |

### File Operations

| Key | Action |
|-----|--------|
| `Ctrl+N` | New project (reset all state) |
| `Ctrl+Shift+O` | Open input directory |
| `Ctrl+Shift+S` | Save project (.s2proj) |
| `Ctrl+Shift+P` | Open project (.s2proj) |
| `Ctrl+S` | Save annotation config |
| `Ctrl+O` | Load annotation config |
| `Ctrl+Q` | Exit application |

### Processing & View

| Key | Action |
|-----|--------|
| `Ctrl+Enter` | Start SAM2 processing |
| `Escape` | Cancel current processing |
| `Ctrl+0` | Reset zoom |
| `Ctrl+1` | Fit image to window |
| `F1` | Show keyboard shortcuts dialog |

---

## 11. Menu Reference

### File Menu

| Item | Shortcut | Description |
|------|----------|-------------|
| New Project | `Ctrl+N` | Reset all state for a fresh project |
| Open Input Directory... | `Ctrl+Shift+O` | Browse for input image folder |
| Set Output Directory... | — | Browse for output folder |
| Recent Projects | — | Submenu of recently used projects |
| Save Project... | `Ctrl+Shift+S` | Save full workspace as `.s2proj` |
| Open Project... | `Ctrl+Shift+P` | Load workspace from `.s2proj` |
| Save Config... | `Ctrl+S` | Save annotation points to JSON |
| Load Config... | `Ctrl+O` | Load annotation points from JSON |
| Exit | `Ctrl+Q` | Close the application |

### Edit Menu

| Item | Shortcut | Description |
|------|----------|-------------|
| Undo | `Ctrl+Z` | Undo last annotation action |
| Redo | `Ctrl+Y` | Redo annotation action |
| Clear All Points | — | Remove all annotation points |
| Go to Frame... | `Ctrl+G` | Jump to a specific frame number |

### View Menu

| Item | Shortcut | Description |
|------|----------|-------------|
| Reset Zoom | `Ctrl+0` | Reset canvas zoom to 100% |
| Fit to Window | `Ctrl+1` | Scale image to fit the canvas |
| Overlay Settings... | — | Adjust overlay alpha and color |

### Tools Menu

| Item | Description |
|------|-------------|
| Export Contours as PNG... | Extract mask boundaries as raster images |
| Export Contours as SVG... | Extract mask boundaries as vector graphics |
| Batch Processing... | Process multiple directories sequentially |

### Help Menu

| Item | Shortcut | Description |
|------|----------|-------------|
| Keyboard Shortcuts | `F1` | Show shortcuts reference dialog |
| About DIC Mask Generator | — | Version and attribution info |

---

## 12. Parameter Reference

### PreprocessingConfig Fields

All fields with their types, ranges, and defaults:

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `gain` | float | 1.0 | 0.1 – 5.0 | Linear pixel multiplier |
| `brightness` | int | 0 | -255 – 255 | Additive pixel offset |
| `contrast` | float | 1.0 | 0.0 – 5.0 | Contrast factor around midpoint |
| `clip_min` | int | 0 | 0 – 254 | Minimum clip value |
| `clip_max` | int | 255 | 1 – 255 | Maximum clip value |
| `clahe_enabled` | bool | False | — | Enable CLAHE |
| `clahe_clip_limit` | float | 2.0 | 0.1 – 40.0 | CLAHE contrast limit |
| `clahe_tile_size` | int | 8 | 2 – 32 | CLAHE grid tile size |
| `gaussian_sigma` | float | 0.0 | 0.0 – 10.0 | Gaussian blur sigma |
| `bilateral_enabled` | bool | False | — | Enable bilateral filter |
| `bilateral_d` | int | 9 | — | Bilateral diameter |
| `bilateral_sigma_color` | float | 75.0 | — | Color sigma |
| `bilateral_sigma_space` | float | 75.0 | — | Space sigma |
| `median_ksize` | int | 0 | 0 – 31 | Median filter kernel (0=off) |
| `box_ksize` | int | 0 | 0 – 31 | Box filter kernel (0=off) |
| `nlm_enabled` | bool | False | — | Enable NLM denoise |
| `nlm_h` | float | 10.0 | 1.0 – 40.0 | NLM filter strength |
| `nlm_template_window` | int | 7 | — | NLM template window |
| `nlm_search_window` | int | 21 | — | NLM search window |
| `diffusion_enabled` | bool | False | — | Enable anisotropic diffusion |
| `diffusion_iterations` | int | 10 | 1 – 100 | Diffusion steps |
| `diffusion_kappa` | float | 30.0 | 1.0 – 200.0 | Conductance parameter |
| `diffusion_dt` | float | 0.1 | 0.01 – 0.25 | Time step |
| `diffusion_option` | int | 1 | 1 or 2 | Diffusivity function |
| `threshold_enabled` | bool | False | — | Enable binary threshold |
| `threshold_value` | int | 127 | 0 – 255 | Threshold cutoff |
| `threshold_method` | str | "fixed" | fixed/otsu/adaptive | Method |
| `invert` | bool | False | — | Invert after threshold |
| `morphology_op` | str | "none" | see list | Morphological operation |
| `morphology_kernel_size` | int | 3 | 3 – 51 | Kernel size (odd) |
| `morphology_iterations` | int | 1 | 1 – 20 | Iterations |
| `fill_holes` | bool | False | — | Fill enclosed dark regions |
| `custom_frames` | str | "" | e.g. "1-10,15" | Frame selection |

### Spatial Smoothing Parameters

| Parameter | Type | Default | Stable Range | Description |
|-----------|------|---------|-------------|-------------|
| `num_iterations` | int | 50 | 1 – 500 | Diffusion steps |
| `dt` | float | 0.1 | 0.001 – 0.25 | Time step (>0.25 causes instability) |
| `kappa` | float | 30.0 | 0.1 – 200.0 | Edge sensitivity threshold |
| `option` | int | 1 | 1 or 2 | 1=exponential, 2=reciprocal |

### Temporal Smoothing Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `sigma` | float | 2.0 | 0.1 – 20.0 | Gaussian sigma (spatial + temporal) |
| `num_neighbors` | int | 2 | 1 – 10 | Neighbor frames for averaging |
| `variance_threshold` | float | 50000 | 0 – 999999 | Bad frame variance cutoff (0=adaptive) |

---

## 13. Tips & Best Practices

### Annotation Tips

- **Start simple** — One foreground point on the center of your target is often sufficient.
- **Use background points sparingly** — Add them only when the model incorrectly includes unwanted regions.
- **Annotate on a clear frame** — Choose a frame where your target is most distinct from the background.
- **Bookmark problem frames** — Press `M` to mark frames that need correction, then use the bookmark navigation buttons to jump between them.

### Performance Tips

- **GPU is essential** — CUDA gives ~100x speedup. Always verify GPU detection in the status bar.
- **Start with Tiny model** — Use `hiera_tiny` for quick annotation previews, then switch to `hiera_large` for final results.
- **Reduce frame range** — Process a subset first (`start_frame` to `end_frame`) to verify annotation quality.
- **Close other GPU applications** — VRAM is limited. Close other apps using the GPU before processing.

### Preprocessing Tips

- **Preview first** — Adjust preprocessing parameters and watch the live preview before saving or processing.
- **Use built-in presets** — For common imaging modalities, presets provide good starting points.
- **DIC images** — Use the DIC Microscopy preset. Key: increase contrast and enable CLAHE.
- **Noisy images** — Enable NLM denoise or anisotropic diffusion. Bilateral filter is a fast alternative.

### Post-Processing Tips

- **Spatial smoothing** — Use for jagged mask edges. Start with 50 iterations, kappa=30, dt=0.1.
- **Temporal smoothing** — Use when masks flicker between frames. Sigma=2.0, neighbors=2 is a good baseline.
- **Apply spatial first, then temporal** — Spatial cleans edges, temporal smooths transitions.
- **Keep originals** — Leave "Replace original masks" unchecked until you're satisfied with the result.

### Project Management Tips

- **Save projects frequently** — Use `Ctrl+Shift+S` to save your workspace. Projects are small JSON files.
- **Version your configs** — Save annotation configs (`Ctrl+S`) as named JSON files for different experiments.
- **Use batch processing** — When applying the same segmentation to multiple image sets.

---

## 14. FAQ

### General

**Q: What image formats are supported?**
A: Input: TIFF, PNG, JPEG, BMP. Mask output: TIFF (default) or PNG. Images are sorted using natural sorting.

**Q: How many frames can I process?**
A: There is no hard limit. Memory usage scales linearly with frame count. For very long sequences (1000+), consider processing in batches.

**Q: Does it work without a GPU?**
A: Yes, but processing will be ~100x slower. Set device to "CPU" in Model Configuration.

### Annotation

**Q: How many annotation points do I need?**
A: Often just 1-3 foreground points are enough. Add background points only if the model includes unwanted regions.

**Q: Can I annotate on frames other than the first?**
A: For initial processing, annotate on frame 1. After processing, use correction mode (`E` key) to add points on any frame and re-propagate.

**Q: What's the difference between V and D modes?**
A: `V` (Select) places foreground points (green, label=1) marking regions to include. `D` (Draw) places background points (red, label=0) marking regions to exclude.

### Processing

**Q: Why does processing show "Converting images..."?**
A: SAM2 requires images in a specific format (JPEG or PNG in a numbered sequence). The app automatically converts your source images.

**Q: What does "auto-downsampling" mean?**
A: Images larger than 1024px are resized for SAM2 inference (which expects 1024px input). The generated masks are then upscaled back to your original resolution.

**Q: Can I stop processing midway?**
A: Yes, press `Escape`. Already-processed frames keep their masks.

### Post-Processing

**Q: Does smoothing modify my original masks?**
A: Only if "Replace original masks" is checked. By default, smoothed masks are saved to a separate directory (`mask_spatial_smoothing/` or `mask_temporal_smoothing/`).

**Q: What is "bad frame detection" in temporal smoothing?**
A: Frames with abnormal variance (e.g., completely black or extremely noisy masks) are automatically detected and replaced by interpolation from neighboring frames.

**Q: Should I apply spatial or temporal smoothing first?**
A: Typically spatial first (edge refinement), then temporal (frame-to-frame consistency).

### Troubleshooting

**Q: The app says "No supported images found in directory"**
A: Verify your input directory contains TIFF, PNG, JPEG, or BMP files. Subdirectories are not scanned.

**Q: Masks don't appear after processing**
A: Check the output directory for a `masks/` subfolder. Try clicking a different frame and back. If the issue persists, check the console for error messages.

**Q: CUDA out of memory**
A: Switch to a smaller model (Tiny or Small), reduce the frame range, or close other GPU-using applications.

**Q: Chinese/special characters in file paths cause errors**
A: The app uses Unicode-safe I/O. Ensure you are running the latest version with `imread_safe` / `imwrite_safe`.

**Q: How do I reset everything?**
A: `File > New Project` (`Ctrl+N`) resets all state including paths, annotations, bookmarks, and display images.
