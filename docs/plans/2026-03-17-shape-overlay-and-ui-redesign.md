# Shape Overlay & Preprocessing UI Redesign

**Date**: 2026-03-17
**Status**: Approved

## Summary

Add shape drawing tools (Add/Cut with Rect/Circle/Polygon) to the preprocessing
pipeline, and consolidate the preprocessing sidebar into hover-popup category
buttons for a cleaner layout.

## Part 1: Shape Overlay

### Data Model

```python
@dataclass(frozen=True)
class ShapeOverlay:
    mode: str          # "add" or "cut"
    shape_type: str    # "rect", "circle", "polygon"
    points: list       # rect: [x1,y1,x2,y2], circle: [cx,cy,r], polygon: [[x,y],...]

# Added to PreprocessingConfig:
shape_overlays: tuple[ShapeOverlay, ...] = ()
```

### Pipeline Position

Last step, after Fill Holes:
```
... → Fill Holes → Shape Overlays
```

- `mode == "add"` → fill region with white (255,255,255)
- `mode == "cut"` → fill region with black (0,0,0)

### Canvas Interaction

1. User hovers [Add] or [Cut] in sidebar → popup menu: Rect / Circle / Polygon
2. Canvas enters shape drawing mode (temporary tool override)
3. Drawing:
   - **Rect**: click-drag from corner to corner
   - **Circle**: click-drag from center
   - **Polygon**: click points, double-click to close
4. After drawing, adjustment handles appear (drag corners/edges to resize)
5. Enter = confirm, Escape = cancel
6. On confirm: shape added to config, preview updates, shape list updates
7. Canvas restores previous tool mode

### Canvas Rendering

- **Add shapes**: semi-transparent white fill + green dashed border (preview)
- **Cut shapes**: semi-transparent black fill + red dashed border (preview)
- Selected shape: solid border + resize handles
- All confirmed shapes rendered as overlay layer on the image

### Shape List (sidebar)

```
── [Add ▼]  [Cut ▼]
── Shape list:
   ├─ Add Rect #1      [x]
   ├─ Cut Circle #2    [x]
   └─ Add Polygon #3   [x]
```

- Click row → highlight shape on canvas
- Click [x] → remove shape, update preview
- Empty list is hidden

## Part 2: Sidebar UI Consolidation

### Before (15+ flat controls)

```
Custom frames, Gain, Brightness, Contrast, Clip Min, Clip Max,
CLAHE (+2 params), Gaussian, Bilateral, Threshold (+2 params),
Invert, Morphology (+3 params), Fill Holes, Add/Cut, Save
```

### After (7 rows)

```
── ☐ Custom frames  [1-10, 15, 20-30]
── [Tone ▼]            → Gain, Brightness, Contrast, Clip Min/Max
── [Enhancement ▼]     → CLAHE (+clip, tile), Gaussian Sigma, Bilateral
── [Binarize ▼]        → Binary Threshold (+value, method), Invert
── [Morphology ▼]      → Operation (+kernel, iterations), Fill Holes
── [Add ▼]  [Cut ▼]
── Shape list (dynamic)
── [Save Preprocessed]
```

### Category Button Behavior

- **Hover** → floating panel (QWidget popup) appears with all controls for that category
- **Mouse leaves** popup → popup hides
- **Button appearance** → shows active indicator when any parameter in the category
  is non-default (e.g., button text turns accent color)
- All control changes within popup trigger live preview via `_on_pp_changed`

### Popup Widget Design

Each popup is a QWidget with `Qt.WindowType.Popup` or `Qt.WindowType.ToolTip` flag:
- Dark background matching sidebar theme
- Contains the same SliderInput/NumberInput/CheckBox/SelectField widgets
- Positioned adjacent to the button (right or below)
- Auto-hides when mouse leaves the popup area

## New Signals

```python
# Sidebar
shape_draw_requested = pyqtSignal(str, str)  # mode ("add"/"cut"), shape_type
shape_removed = pyqtSignal(int)               # shape index
shape_selected = pyqtSignal(int)              # shape index
```

## Implementation Order

1. Create `HoverPopupButton` widget (reusable category button with popup)
2. Refactor sidebar: replace flat controls with HoverPopupButton groups
3. Add ShapeOverlay data model to preprocessing.py
4. Add shape rendering to apply_pipeline
5. Add shape drawing tools to CanvasArea
6. Wire shape signals in MainWindow
7. Add shape list to sidebar
