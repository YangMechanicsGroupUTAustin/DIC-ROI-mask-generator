"""Image loading, conversion, and mask processing utilities."""

import os
import re
import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("sam2studio.image_processing")


def extract_numbers(file_name: str) -> tuple:
    """Extract numeric parts from filename for natural sorting."""
    numbers = re.findall(r'\d+', file_name)
    return tuple(map(int, numbers))


def get_image_files(directory: str) -> list:
    """Find and sort image files in directory using natural ordering."""
    image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.tif')
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    return sorted(image_files, key=lambda x: extract_numbers(os.path.basename(x)))


def load_image_as_rgb(image_path: str):
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
        logger.error(f"Failed to load image {image_path}: {e}")
        return None


def _normalize_image(img_path: str) -> np.ndarray | None:
    """Load and normalize an image to uint8 BGR format.

    Shared normalization logic used by convert_to_jpeg and convert_to_png.
    Handles 8-bit, 16-bit, 32-bit, and float images via OpenCV and PIL fallback.

    Args:
        img_path: Path to the source image file.

    Returns:
        Normalized uint8 BGR numpy array, or None on failure.
    """
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

    return img


def convert_to_jpeg(img_path: str, jpeg_path: str, quality: int = 95) -> bool:
    """Convert any supported image format to JPEG for SAM2 inference.

    Handles 8-bit, 16-bit, 32-bit, and float images.
    Returns True on success, False on failure.
    """
    try:
        img = _normalize_image(img_path)
        if img is None:
            return False
        cv2.imwrite(jpeg_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return True
    except Exception as e:
        logger.error(f"Failed to convert {img_path} to JPEG: {e}")
        return False


def convert_to_png(img_path: str, png_path: str) -> bool:
    """Convert any supported image format to lossless PNG for SAM2 inference.

    Same normalization logic as convert_to_jpeg but uses PNG (lossless).
    Returns True on success, False on failure.
    """
    try:
        img = _normalize_image(img_path)
        if img is None:
            return False
        cv2.imwrite(png_path, img)
        return True
    except Exception as e:
        logger.error(f"Failed to convert {img_path} to PNG: {e}")
        return False


def convert_image(img_path: str, output_path: str, format: str = "jpeg", quality: int = 95) -> bool:
    """Convert image to specified format.

    Dispatches to convert_to_jpeg or convert_to_png based on format parameter.

    Args:
        img_path: Source image path.
        output_path: Destination path.
        format: Target format - "jpeg" or "png".
        quality: JPEG quality (ignored for PNG).

    Returns:
        True on success, False on failure.
    """
    fmt = format.lower()
    if fmt in ("jpeg", "jpg"):
        return convert_to_jpeg(img_path, output_path, quality)
    elif fmt == "png":
        return convert_to_png(img_path, output_path)
    else:
        logger.error(f"Unsupported format: {format}")
        return False


def create_placeholder_image(
    output_path: str, reference_files: list, fallback_size: tuple = (512, 512, 3)
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
        cv2.imwrite(output_path, blank)
        logger.info(f"Created placeholder ({ref_shape[1]}x{ref_shape[0]}) at {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create placeholder: {e}")
        return False


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4,
                   color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """Create colored overlay of mask on original image.

    Args:
        image: RGB image array (H, W, 3) uint8
        mask: Binary mask array (H, W) with values 0 or 255
        alpha: Transparency of overlay (0=invisible, 1=opaque)
        color: RGB color tuple for overlay (default: red)

    Returns:
        Blended overlay image (H, W, 3) uint8
    """
    overlay = image.copy()
    if len(overlay.shape) == 2:
        overlay = np.stack([overlay] * 3, axis=-1)

    mask_bool = mask > 127
    for i, c in enumerate(color):
        overlay[mask_bool, i] = np.clip(
            image[mask_bool, i] * (1 - alpha) + c * alpha, 0, 255
        ).astype(np.uint8)

    return overlay


def numpy_to_qimage(arr: np.ndarray):
    """Convert RGB numpy array to QImage for PyQt6 display.

    Args:
        arr: RGB image array (H, W, 3) uint8, or grayscale (H, W) uint8.

    Returns:
        QImage instance.
    """
    from PyQt6.QtGui import QImage

    if arr is None:
        return QImage()

    if len(arr.shape) == 2:
        # Grayscale
        h, w = arr.shape
        bytes_per_line = w
        return QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8).copy()

    h, w, ch = arr.shape
    if ch == 3:
        bytes_per_line = 3 * w
        return QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
    elif ch == 4:
        bytes_per_line = 4 * w
        return QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888).copy()

    return QImage()
