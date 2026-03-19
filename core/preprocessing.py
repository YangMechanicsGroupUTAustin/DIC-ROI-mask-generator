"""Image preprocessing pipeline for SAM2 input enhancement.

All functions are pure: they take an image and return a NEW image.
No mutation of input arrays. All operate on uint8 BGR images.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields
from datetime import datetime, timezone

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShapeOverlay:
    """A shape to draw on frames as part of preprocessing."""
    mode: str          # "add" or "cut"
    shape_type: str    # "rect", "circle", "polygon"
    points: tuple      # rect: (x1,y1,x2,y2), circle: (cx,cy,r), polygon: ((x1,y1),(x2,y2),...)


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
    median_ksize: int = 0           # Median filter kernel (0 = disabled, odd >=3)
    box_ksize: int = 0              # Box/mean filter kernel (0 = disabled)
    nlm_enabled: bool = False       # Non-local means denoising
    nlm_h: float = 10.0            # NLM filter strength
    nlm_template_window: int = 7   # NLM template patch size
    nlm_search_window: int = 21    # NLM search area size
    diffusion_enabled: bool = False  # Perona-Malik anisotropic diffusion
    diffusion_iterations: int = 10   # Number of diffusion steps
    diffusion_kappa: float = 30.0    # Conductance parameter
    diffusion_dt: float = 0.1       # Time step (<=0.25 for stability)
    diffusion_option: int = 1       # 1=high-contrast edges, 2=wide regions
    threshold_enabled: bool = False  # Binary thresholding
    threshold_value: int = 127       # Fixed threshold value
    threshold_method: str = "fixed"  # "fixed", "otsu", "adaptive"
    invert: bool = False             # Bitwise NOT after binarization
    morphology_op: str = "none"      # none/dilate/erode/open/close/gradient/tophat/blackhat
    morphology_kernel_size: int = 3  # 3-51, odd only
    morphology_iterations: int = 1   # 1-20
    fill_holes: bool = False         # Fill enclosed contour interiors
    custom_frames: str = ""          # "" = all frames, otherwise "1-10, 15, 20-30"
    shape_overlays: tuple[ShapeOverlay, ...] = ()

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
            and self.median_ksize == 0
            and self.box_ksize == 0
            and not self.nlm_enabled
            and not self.diffusion_enabled
            and not self.threshold_enabled
            and not self.invert
            and self.morphology_op == "none"
            and not self.fill_holes
            and len(self.shape_overlays) == 0
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict (excludes shape_overlays and custom_frames)."""
        skip = {"shape_overlays", "custom_frames"}
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in skip
        }

    @classmethod
    def from_dict(cls, data: dict) -> PreprocessingConfig:
        """Deserialize from a plain dict, ignoring unknown keys."""
        valid = {f.name for f in fields(cls)} - {"shape_overlays"}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


_PRESET_VERSION = "1.0"


# Built-in preset library for common imaging modalities
BUILTIN_PRESETS: dict[str, PreprocessingConfig] = {
    "None (identity)": PreprocessingConfig(),
    "DIC Microscopy": PreprocessingConfig(
        contrast=1.8,
        clahe_enabled=True,
        clahe_clip_limit=3.0,
        clahe_tile_size=8,
        gaussian_sigma=0.5,
    ),
    "Fluorescence": PreprocessingConfig(
        gain=2.0,
        clip_min=10,
        clip_max=250,
        clahe_enabled=True,
        clahe_clip_limit=2.0,
        clahe_tile_size=16,
        gaussian_sigma=1.0,
    ),
    "Phase Contrast": PreprocessingConfig(
        contrast=2.0,
        brightness=-20,
        bilateral_enabled=True,
    ),
    "Brightfield": PreprocessingConfig(
        contrast=1.5,
        clahe_enabled=True,
        clahe_clip_limit=2.5,
        clahe_tile_size=8,
    ),
    "High Noise (Denoise)": PreprocessingConfig(
        gaussian_sigma=1.5,
        nlm_enabled=True,
        nlm_h=15.0,
    ),
    "Edge Enhancement": PreprocessingConfig(
        contrast=2.0,
        diffusion_enabled=True,
        diffusion_iterations=15,
        diffusion_kappa=25.0,
        diffusion_option=1,
    ),
}


def save_preset(config: PreprocessingConfig, path: str) -> None:
    """Save preprocessing config to a JSON preset file."""
    payload = {
        "version": _PRESET_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "preprocessing": config.to_dict(),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Saved preprocessing preset to %s", path)


def load_preset(path: str) -> PreprocessingConfig:
    """Load preprocessing config from a JSON preset file.

    Raises ``ValueError`` on incompatible or corrupt files.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict) or "preprocessing" not in data:
        raise ValueError("Invalid preset file: missing 'preprocessing' key")

    return PreprocessingConfig.from_dict(data["preprocessing"])


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


def median_filter(image: np.ndarray, ksize: int) -> np.ndarray:
    """Apply median filter. Excellent for salt-and-pepper noise."""
    if ksize <= 0:
        return image
    ks = max(3, ksize | 1)  # Ensure odd and >= 3
    return cv2.medianBlur(image, ks)


def box_filter(image: np.ndarray, ksize: int) -> np.ndarray:
    """Apply box (mean) filter. Fast uniform smoothing."""
    if ksize <= 0:
        return image
    ks = max(1, ksize)
    return cv2.blur(image, (ks, ks))


def nlm_denoise(
    image: np.ndarray,
    h: float = 10.0,
    template_window: int = 7,
    search_window: int = 21,
) -> np.ndarray:
    """Apply Non-Local Means denoising. High quality but slower."""
    return cv2.fastNlMeansDenoisingColored(
        image, None, h, h, template_window, search_window,
    )


def anisotropic_diffusion(
    image: np.ndarray,
    iterations: int = 10,
    kappa: float = 30.0,
    dt: float = 0.1,
    option: int = 1,
) -> np.ndarray:
    """Apply Perona-Malik anisotropic diffusion (edge-preserving smoothing).

    Args:
        image: Input BGR uint8 image.
        iterations: Number of diffusion steps.
        kappa: Conductance parameter controlling sensitivity to edges.
        dt: Time step per iteration (<=0.25 for numerical stability).
        option: 1 = exp function (favors high-contrast edges),
                2 = reciprocal function (favors wide regions).
    """
    dt = min(dt, 0.25)  # Clamp for stability
    img = image.astype(np.float64)

    for _ in range(iterations):
        # Four-neighbor finite differences
        nabla_n = np.roll(img, -1, axis=0) - img
        nabla_s = np.roll(img, 1, axis=0) - img
        nabla_e = np.roll(img, -1, axis=1) - img
        nabla_w = np.roll(img, 1, axis=1) - img

        if option == 1:
            c_n = np.exp(-(nabla_n / kappa) ** 2)
            c_s = np.exp(-(nabla_s / kappa) ** 2)
            c_e = np.exp(-(nabla_e / kappa) ** 2)
            c_w = np.exp(-(nabla_w / kappa) ** 2)
        else:
            c_n = 1.0 / (1.0 + (nabla_n / kappa) ** 2)
            c_s = 1.0 / (1.0 + (nabla_s / kappa) ** 2)
            c_e = 1.0 / (1.0 + (nabla_e / kappa) ** 2)
            c_w = 1.0 / (1.0 + (nabla_w / kappa) ** 2)

        img += dt * (c_n * nabla_n + c_s * nabla_s
                     + c_e * nabla_e + c_w * nabla_w)

    return np.clip(img, 0, 255).astype(np.uint8)


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


def invert_image(image: np.ndarray) -> np.ndarray:
    """Apply bitwise NOT (invert pixel values)."""
    return cv2.bitwise_not(image)


# Mapping from config string to OpenCV morphology type
_MORPH_OPS = {
    "dilate": cv2.MORPH_DILATE,
    "erode": cv2.MORPH_ERODE,
    "open": cv2.MORPH_OPEN,
    "close": cv2.MORPH_CLOSE,
    "gradient": cv2.MORPH_GRADIENT,
    "tophat": cv2.MORPH_TOPHAT,
    "blackhat": cv2.MORPH_BLACKHAT,
}


def apply_morphology(
    image: np.ndarray,
    op: str = "none",
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """Apply a morphological operation with an elliptical kernel."""
    if op == "none" or op not in _MORPH_OPS:
        return image
    # Ensure odd kernel size
    ks = max(3, kernel_size | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    return cv2.morphologyEx(image, _MORPH_OPS[op], kernel, iterations=iterations)


def fill_holes_op(image: np.ndarray) -> np.ndarray:
    """Fill holes in a binary (or near-binary) image.

    Uses flood-fill from the border: everything NOT reached by the
    background flood is interior and gets filled as foreground.
    Works on 3-channel BGR images by converting to grayscale internally.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Threshold to ensure binary
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Flood-fill background from top-left corner
    h, w = binary.shape
    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)

    # Invert the flood-filled image: holes become white
    flood_inv = cv2.bitwise_not(flood)

    # Combine: original foreground OR filled holes
    filled = binary | flood_inv

    if len(image.shape) == 3:
        return cv2.cvtColor(filled, cv2.COLOR_GRAY2BGR)
    return filled


def parse_custom_frames(spec: str, total_frames: int) -> list[int]:
    """Parse a custom frame range string into a sorted list of 0-based indices.

    Accepts comma-separated values and ranges (1-indexed, inclusive).
    Examples: "1-10, 15, 20-30"  →  [0,1,...,9, 14, 19,...,29]

    Invalid tokens are silently skipped. Out-of-range values are clamped.
    """
    if not spec or not spec.strip():
        return list(range(total_frames))

    indices: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            try:
                start = max(1, int(parts[0].strip()))
                end = min(total_frames, int(parts[1].strip()))
                indices.update(range(start - 1, end))
            except (ValueError, IndexError):
                continue
        else:
            try:
                val = int(token)
                if 1 <= val <= total_frames:
                    indices.add(val - 1)
            except ValueError:
                continue

    return sorted(indices)


def apply_shape_overlays(image: np.ndarray, overlays: tuple) -> np.ndarray:
    """Draw shape overlays onto the image. Last step in the pipeline.

    - mode "add": fill with white (255,255,255)
    - mode "cut": fill with black (0,0,0)
    """
    result = image.copy()
    for overlay in overlays:
        color = (255, 255, 255) if overlay.mode == "add" else (0, 0, 0)
        if overlay.shape_type == "rect":
            x1, y1, x2, y2 = overlay.points
            cv2.rectangle(result, (x1, y1), (x2, y2), color, cv2.FILLED)
        elif overlay.shape_type == "circle":
            cx, cy, r = overlay.points
            cv2.circle(result, (cx, cy), r, color, cv2.FILLED)
        elif overlay.shape_type == "polygon":
            pts = np.array(overlay.points, dtype=np.int32)
            cv2.fillPoly(result, [pts], color)
    return result


def apply_pipeline(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Apply the full preprocessing pipeline in order.

    Pipeline order:
      Tone:       1. Gain  2. Brightness  3. Contrast  4. Clip  5. CLAHE
      Smoothing:  6. Gaussian  7. Bilateral  8. Median  9. Box
                  10. NLM Denoise  11. Anisotropic Diffusion
      Binarize:   12. Threshold  13. Invert
      Morphology: 14. Morphology ops  15. Fill holes
      Shapes:     16. Shape overlays

    Returns a NEW image; input is never mutated.
    """
    if config.is_identity():
        return image

    result = image  # No copy needed; each step creates new array

    # -- Tone --
    result = adjust_gain(result, config.gain)
    result = adjust_brightness(result, config.brightness)
    result = adjust_contrast(result, config.contrast)
    result = clip_min_max(result, config.clip_min, config.clip_max)

    if config.clahe_enabled:
        result = apply_clahe(
            result, config.clahe_clip_limit, config.clahe_tile_size,
        )

    # -- Smoothing --
    result = gaussian_smooth(result, config.gaussian_sigma)

    if config.bilateral_enabled:
        result = bilateral_filter(
            result, config.bilateral_d,
            config.bilateral_sigma_color, config.bilateral_sigma_space,
        )

    result = median_filter(result, config.median_ksize)
    result = box_filter(result, config.box_ksize)

    if config.nlm_enabled:
        result = nlm_denoise(
            result, config.nlm_h,
            config.nlm_template_window, config.nlm_search_window,
        )

    if config.diffusion_enabled:
        result = anisotropic_diffusion(
            result, config.diffusion_iterations,
            config.diffusion_kappa, config.diffusion_dt,
            config.diffusion_option,
        )

    # -- Binarize --
    if config.threshold_enabled:
        result = binary_threshold(
            result, config.threshold_value, config.threshold_method,
        )

    if config.invert:
        result = invert_image(result)

    # -- Morphology --
    if config.morphology_op != "none":
        result = apply_morphology(
            result, config.morphology_op,
            config.morphology_kernel_size, config.morphology_iterations,
        )

    if config.fill_holes:
        result = fill_holes_op(result)

    # -- Shapes --
    if config.shape_overlays:
        result = apply_shape_overlays(result, config.shape_overlays)

    return result
