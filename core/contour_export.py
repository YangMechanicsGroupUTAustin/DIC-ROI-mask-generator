"""Contour/boundary extraction and export from binary masks.

Supports exporting mask boundaries as:
- PNG images (white contours on black background)
- SVG vector files (resolution-independent)
"""

import logging
import os
from xml.etree.ElementTree import Element, SubElement, tostring

import cv2
import numpy as np

from core.image_processing import imread_safe, imwrite_safe

logger = logging.getLogger("sam2studio.contour_export")


def extract_contours(
    mask: np.ndarray, thickness: int = 2,
) -> list[np.ndarray]:
    """Extract contours from a binary mask.

    Args:
        mask: 2D uint8 array (0 or 255).
        thickness: Not used here, but kept for API consistency.

    Returns:
        List of contour arrays, each shape (N, 1, 2).
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    return list(contours)


def export_contour_png(
    mask: np.ndarray,
    output_path: str,
    thickness: int = 2,
    color: tuple[int, int, int] = (255, 255, 255),
) -> bool:
    """Export mask contours as a PNG image.

    Args:
        mask: 2D uint8 mask array.
        output_path: Path to save the PNG file.
        thickness: Contour line thickness in pixels.
        color: BGR color for contour lines.

    Returns:
        True if successful, False otherwise.
    """
    contours = extract_contours(mask)
    if not contours:
        logger.warning(f"No contours found for {output_path}")
        return False

    h, w = mask.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.drawContours(canvas, contours, -1, color, thickness)
    return imwrite_safe(output_path, canvas)


def _contour_to_svg_path(contour: np.ndarray) -> str:
    """Convert an OpenCV contour array to an SVG path d-attribute string."""
    pts = contour.squeeze()
    if pts.ndim == 1:
        # Single point
        return f"M {pts[0]} {pts[1]} Z"
    parts = [f"M {pts[0][0]} {pts[0][1]}"]
    for pt in pts[1:]:
        parts.append(f"L {pt[0]} {pt[1]}")
    parts.append("Z")
    return " ".join(parts)


def export_contour_svg(
    mask: np.ndarray,
    output_path: str,
    stroke_width: float = 2.0,
    stroke_color: str = "#FFFFFF",
) -> bool:
    """Export mask contours as an SVG vector file.

    Args:
        mask: 2D uint8 mask array.
        output_path: Path to save the SVG file.
        stroke_width: SVG stroke width.
        stroke_color: SVG stroke color (hex string).

    Returns:
        True if successful, False otherwise.
    """
    contours = extract_contours(mask)
    if not contours:
        logger.warning(f"No contours found for {output_path}")
        return False

    h, w = mask.shape[:2]
    svg = Element("svg")
    svg.set("xmlns", "http://www.w3.org/2000/svg")
    svg.set("width", str(w))
    svg.set("height", str(h))
    svg.set("viewBox", f"0 0 {w} {h}")

    for contour in contours:
        path = SubElement(svg, "path")
        path.set("d", _contour_to_svg_path(contour))
        path.set("fill", "none")
        path.set("stroke", stroke_color)
        path.set("stroke-width", str(stroke_width))

    xml_bytes = tostring(svg, encoding="unicode")
    svg_content = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_bytes
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        return True
    except OSError as e:
        logger.error(f"Failed to write SVG: {e}")
        return False


def batch_export_contours(
    mask_dir: str,
    output_dir: str,
    fmt: str = "PNG",
    thickness: int = 2,
    progress_callback=None,
) -> int:
    """Export contours for all masks in a directory.

    Args:
        mask_dir: Directory containing mask files.
        output_dir: Directory to save contour files.
        fmt: "PNG" or "SVG".
        thickness: Contour line thickness.
        progress_callback: Optional (current, total, message) callback.

    Returns:
        Number of files successfully exported.
    """
    os.makedirs(output_dir, exist_ok=True)

    mask_exts = (".tiff", ".tif", ".png")
    mask_files = sorted(
        f for f in os.listdir(mask_dir)
        if f.lower().endswith(mask_exts)
    )

    if not mask_files:
        logger.warning(f"No mask files found in {mask_dir}")
        return 0

    total = len(mask_files)
    count = 0

    for i, fname in enumerate(mask_files):
        mask_path = os.path.join(mask_dir, fname)
        mask = imread_safe(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning(f"Could not read mask: {mask_path}")
            continue

        base = os.path.splitext(fname)[0]
        ext = ".png" if fmt.upper() == "PNG" else ".svg"
        out_path = os.path.join(output_dir, f"{base}_contour{ext}")

        if fmt.upper() == "PNG":
            ok = export_contour_png(mask, out_path, thickness=thickness)
        else:
            ok = export_contour_svg(mask, out_path, stroke_width=float(thickness))

        if ok:
            count += 1

        if progress_callback:
            progress_callback(i + 1, total, f"Exporting contours: {fname}")

    logger.info(f"Exported {count}/{total} contour files to {output_dir}")
    return count
