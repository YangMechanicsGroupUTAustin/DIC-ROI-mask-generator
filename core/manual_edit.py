"""Pure brush and stroke primitives for manual mask editing.

This module is deliberately Qt-free so it can be unit-tested without a
GUI. It provides two primitives used by `ManualEditController`:

- `paint_dot`    — stamp a filled disk at a single point
- `paint_stroke` — stamp a thick, round-capped polyline

Both functions mutate the mask in place and return the dirty bounding
box as `(x0, y0, x1, y1)` in numpy-slice style (x1/y1 exclusive),
already clipped to image bounds. Returns `None` if the operation
affects no pixels (e.g. empty stroke or dot fully outside the image).
"""

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]


def _clip_bbox(
    cx: int, cy: int, radius: int, width: int, height: int
) -> Optional[BBox]:
    """Clip a disk bbox (cx +/- radius) to image bounds, exclusive upper."""
    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    x1 = min(width, cx + radius + 1)
    y1 = min(height, cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    return (x0, y0, x1, y1)


def paint_dot(
    mask: np.ndarray,
    cx: float,
    cy: float,
    radius: int,
    value: int,
) -> Optional[BBox]:
    """Fill a disk of `radius` centered at (cx, cy) with `value`.

    Mutates `mask` in place. Returns the dirty bbox clipped to image
    bounds, or `None` if the disk is entirely outside the image.
    """
    h, w = mask.shape[:2]
    r = int(radius)
    cxi = int(round(cx))
    cyi = int(round(cy))
    v = int(value)

    bbox = _clip_bbox(cxi, cyi, r, w, h)
    if bbox is None:
        return None

    cv2.circle(mask, (cxi, cyi), r, v, thickness=-1, lineType=cv2.LINE_8)
    return bbox


def paint_stroke(
    mask: np.ndarray,
    points: Sequence[Tuple[float, float]],
    radius: int,
    value: int,
) -> Optional[BBox]:
    """Paint a thick polyline connecting consecutive points with `value`.

    Each segment is drawn with a round end cap so the stroke is a
    consistent thickness (matching `paint_dot`) everywhere. A single-
    point stroke is equivalent to `paint_dot`. An empty point list is a
    no-op and returns `None`.
    """
    if not points:
        return None

    h, w = mask.shape[:2]
    r = int(radius)
    v = int(value)

    # Round once; reuse rounded coords for drawing and bbox computation.
    pts = [(int(round(x)), int(round(y))) for (x, y) in points]

    # Cap the start of the stroke.
    cv2.circle(mask, pts[0], r, v, thickness=-1, lineType=cv2.LINE_8)

    # Draw each segment + a round cap at the far endpoint so joints are
    # round and the stroke width matches `paint_dot` along its length.
    line_thickness = max(1, 2 * r + 1)
    for i in range(1, len(pts)):
        cv2.line(
            mask, pts[i - 1], pts[i], v,
            thickness=line_thickness, lineType=cv2.LINE_8,
        )
        cv2.circle(mask, pts[i], r, v, thickness=-1, lineType=cv2.LINE_8)

    # Union bbox = convex hull of each point's disk footprint, clipped.
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0 = max(0, min(xs) - r)
    y0 = max(0, min(ys) - r)
    x1 = min(w, max(xs) + r + 1)
    y1 = min(h, max(ys) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    return (x0, y0, x1, y1)
