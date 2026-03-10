"""Image preprocessing pipeline for SAM2 input enhancement.

All functions are pure: they take an image and return a NEW image.
No mutation of input arrays. All operate on uint8 BGR images.
"""
from dataclasses import dataclass

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
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )
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
    1. Gain  2. Brightness  3. Contrast  4. Min/Max clip
    5. CLAHE  6. Gaussian smooth  7. Bilateral filter  8. Binary threshold

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
        result = apply_clahe(
            result, config.clahe_clip_limit, config.clahe_tile_size,
        )

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
