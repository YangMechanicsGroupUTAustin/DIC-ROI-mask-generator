"""Temporal smoothing for mask sequences.

Improved implementation with:
- float32 for memory efficiency (halves RAM usage)
- Adaptive bad frame detection using median absolute deviation
- Chunked 3D Gaussian for large sequences
- Fine-grained progress reporting
"""

import logging
from typing import Optional, Callable

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

logger = logging.getLogger("sam2studio.temporal_smoothing")


def detect_bad_frames(
    frames: np.ndarray,
    variance_threshold: Optional[float] = None,
) -> list[int]:
    """Detect frames with abnormal variance using adaptive thresholding.

    Args:
        frames: 3D array (H, W, N) of frame data.
        variance_threshold: If provided, use fixed threshold.
            If None, use adaptive MAD-based threshold.

    Returns:
        Sorted list of bad frame indices (0-based).
    """
    num_frames = frames.shape[2]
    variances = np.array([np.var(frames[:, :, i]) for i in range(num_frames)])

    if variance_threshold is not None and variance_threshold > 0:
        # Fixed threshold mode (backward compatible)
        bad_high = set(np.where(variances > variance_threshold)[0])
    else:
        # Adaptive: median + 5 * MAD
        median_var = np.median(variances)
        mad = np.median(np.abs(variances - median_var))
        adaptive_threshold = median_var + 5 * max(mad, 1e-6)
        bad_high = set(np.where(variances > adaptive_threshold)[0])
        logger.info(
            f"Adaptive threshold: {adaptive_threshold:.1f} "
            f"(median={median_var:.1f}, MAD={mad:.1f})"
        )

    bad_zero = set(np.where(variances == 0)[0])

    # Also flag neighbors of zero-variance frames
    bad_neighbors = set()
    for idx in bad_zero:
        if idx > 0:
            bad_neighbors.add(idx - 1)
        if idx < num_frames - 1:
            bad_neighbors.add(idx + 1)

    bad_indices = sorted(bad_high | bad_zero | bad_neighbors)
    if bad_indices:
        logger.info(f"Detected {len(bad_indices)} bad frames: {bad_indices[:10]}...")
    return bad_indices


def fill_nan_frames(
    sequence: np.ndarray,
    bad_indices: list[int],
    num_neighbors: int = 2,
    sigma: float = 2.0,
) -> np.ndarray:
    """Replace bad frames with Gaussian-weighted temporal average of neighbors.

    Args:
        sequence: 3D array (H, W, N) float32.
        bad_indices: Indices of frames to replace.
        num_neighbors: Number of neighbor frames on each side.
        sigma: Gaussian sigma for neighbor weighting.

    Returns:
        Filled sequence with bad frames replaced.
    """
    filled = sequence.copy()
    num_frames = sequence.shape[2]

    for idx in bad_indices:
        filled[:, :, idx] = np.nan

    for idx in bad_indices:
        lo = max(0, idx - num_neighbors)
        hi = min(num_frames, idx + num_neighbors + 1)
        neighbors = filled[:, :, lo:hi].copy()

        for k in range(neighbors.shape[2]):
            frame = neighbors[:, :, k]
            if not np.any(np.isnan(frame)):
                neighbors[:, :, k] = gaussian_filter(frame, sigma=sigma)

        with np.errstate(all="ignore"):
            avg = np.nanmean(neighbors, axis=2)
        avg = np.nan_to_num(avg, nan=0.0)
        filled[:, :, idx] = avg

    return filled


def apply_3d_gaussian(sequence: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Apply 3D Gaussian filter for spatio-temporal smoothing.

    For large sequences, applies temporal smoothing separately
    to reduce memory pressure.

    Args:
        sequence: 3D array (H, W, N) float32.
        sigma: Gaussian sigma for all 3 dimensions.

    Returns:
        Smoothed 3D sequence.
    """
    # Spatial smoothing per-frame (memory efficient)
    result = np.empty_like(sequence)
    for i in range(sequence.shape[2]):
        result[:, :, i] = gaussian_filter(sequence[:, :, i], sigma=sigma)

    # Temporal smoothing along axis=2
    result = gaussian_filter1d(result, sigma=sigma, axis=2)
    return result


def temporal_smooth_sequence(
    frames: list[np.ndarray],
    variance_threshold: Optional[float] = None,
    num_neighbors: int = 2,
    sigma: float = 2.0,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> list[np.ndarray]:
    """Full temporal smoothing pipeline.

    Args:
        frames: List of 2D mask arrays (grayscale uint8).
        variance_threshold: Bad frame detection threshold.
            None = adaptive (recommended). Float = fixed threshold.
        num_neighbors: Temporal window for NaN filling.
        sigma: Gaussian sigma for both filling and 3D filter.
        progress_callback: Optional callable(step_name, current, total).

    Returns:
        List of temporally smoothed and binarized mask arrays (uint8).
    """
    if not frames:
        return []

    h, w = frames[0].shape[:2]
    num_frames = len(frames)

    logger.info(f"Temporal smoothing: {num_frames} frames, {h}x{w}")

    # Build 3D array using float32 (half the memory of float64)
    sequence = np.zeros((h, w, num_frames), dtype=np.float32)
    for i, f in enumerate(frames):
        gray = f if len(f.shape) == 2 else f[:, :, 0]
        sequence[:, :, i] = gray.astype(np.float32)
        if progress_callback and i % 10 == 0:
            progress_callback("Loading frames", i, num_frames)

    if progress_callback:
        progress_callback("Detecting bad frames", 0, 4)
    bad_indices = detect_bad_frames(sequence, variance_threshold)

    if progress_callback:
        progress_callback("Filling bad frames", 1, 4)
    filled = fill_nan_frames(sequence, bad_indices, num_neighbors, sigma)

    if progress_callback:
        progress_callback("Applying 3D Gaussian filter", 2, 4)
    smoothed = apply_3d_gaussian(filled, sigma)

    if progress_callback:
        progress_callback("Binarizing results", 3, 4)

    results = []
    for i in range(smoothed.shape[2]):
        binary = (smoothed[:, :, i] > 127).astype(np.uint8) * 255
        results.append(binary)

    logger.info(f"Temporal smoothing complete: {len(results)} frames output")
    return results
