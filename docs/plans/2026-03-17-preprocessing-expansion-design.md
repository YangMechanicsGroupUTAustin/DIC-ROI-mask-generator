# Preprocessing Expansion Design

**Date**: 2026-03-17
**Status**: Approved

## Summary

Expand the preprocessing pipeline with morphological operations, binary inversion,
fill holes, custom frame selection, and a standalone "Save Preprocessed" workflow
that decouples preprocessing from the SAM2 pipeline.

## Pipeline Order

```
Gain → Brightness → Contrast → Clip → CLAHE → Gaussian → Bilateral
→ Binary Threshold → Invert → Morphology → Fill Holes
```

## New PreprocessingConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `invert` | bool | False | Bitwise NOT after binarization |
| `morphology_op` | str | "none" | none/dilate/erode/open/close/gradient/tophat/blackhat |
| `morphology_kernel_size` | int | 3 | 3-51, odd only |
| `morphology_iterations` | int | 1 | 1-20 |
| `fill_holes` | bool | False | Fill enclosed contour interiors |
| `custom_frames` | str | "" | Empty = all frames, otherwise "1-10, 15, 20-30" |

## UI Layout (Sidebar Preprocessing Panel)

```
☐ Custom frames  [1-10, 15, 20-30]    ← top of panel
── Gain slider
── Brightness slider
── Contrast slider
── Clip Min/Max sliders
── ☐ CLAHE (expand: clip limit, tile size)
── Gaussian Sigma slider
── ☐ Bilateral Filter
── ☐ Binary Threshold (expand: value, method)
── ☐ Invert                             ← new
── ☐ Morphology (expand: op, kernel, iterations) ← new
── ☐ Fill Holes                          ← new
```

## Buttons

- **Save Preprocessed** — standalone button, applies preprocessing and saves to
  `output_dir/preprocessed/`. Uses original filenames for easy reference.
- **Start Processing** — SAM2 pipeline, detects `preprocessed/` directory and uses
  it if present.

## SAM2 Stage 1.5 Change

- Remove internal preprocessing execution from the SAM2 pipeline.
- New logic: check if `output_dir/preprocessed/` exists and is non-empty.
  - Yes → use preprocessed images as SAM2 input.
  - No → use `converted/` images (unchanged behavior).

## Save Preprocessed Workflow

```
User clicks "Save Preprocessed"
    ↓
Read preprocessing config from sidebar
    ↓
Check config.is_identity() → if true, warn user "no preprocessing enabled", return
    ↓
Determine frame range:
  - custom_frames == "" → all loaded frames
  - custom_frames != "" → parse range (e.g., "1-10, 15, 20-30")
    ↓
For each frame:
  1. imread_safe() original image
  2. apply_pipeline(img, config)
  3. imwrite_safe() to output_dir/preprocessed/
    ↓
Show progress bar
    ↓
Completion dialog with save path
```

## Implementation Details

### Morphology Operations

Use `cv2.morphologyEx()` with `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ...)`:
- dilate: `cv2.MORPH_DILATE`
- erode: `cv2.MORPH_ERODE`
- open: `cv2.MORPH_OPEN`
- close: `cv2.MORPH_CLOSE`
- gradient: `cv2.MORPH_GRADIENT`
- tophat: `cv2.MORPH_TOPHAT`
- blackhat: `cv2.MORPH_BLACKHAT`

### Fill Holes

1. `cv2.findContours()` on the binary image
2. Create a filled mask using `cv2.drawContours(..., cv2.FILLED)`
3. OR the filled mask with the original to preserve existing foreground

### Invert

`cv2.bitwise_not(img)` — simple pixel-level inversion.

### Custom Frame Parsing

Parse comma-separated ranges: "1-10, 15, 20-30" → set of frame indices.
Validate against loaded frame count. 1-indexed for user-facing UI, 0-indexed internally.

## Design Principles

- **Immutability**: all new functions return new arrays, never mutate input
- **Identity optimization**: update `is_identity()` to include new fields
- **Unicode-safe I/O**: use `imread_safe` / `imwrite_safe` throughout
- **Decoupled preprocessing**: preprocessing is an explicit user step, not embedded in SAM2
