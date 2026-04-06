"""Temporal smoothing for mask sequences.

Improved implementation with:
- float32 for memory efficiency (halves RAM usage)
- Adaptive bad frame detection using variance AND area analysis
- Adaptive neighbor window that expands to find valid frames
- Smart fill order: fills nearest-to-good frames first for propagation
- Optional temporal-only Gaussian (no spatial blurring — that is Step 1's job)
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
    """Detect abnormal frames using variance AND area outlier analysis.

    Detection methods:
    1. Variance-based: catches noisy or high-variance frames
    2. Zero-variance: catches blank (all-black or all-white) frames
    3. Area-based (MAD): catches inverted or dramatically altered frames
       whose variance is normal but pixel distribution is abnormal

    Args:
        frames: 3D array (H, W, N) of frame data.
        variance_threshold: If provided, use fixed threshold for variance.
            If None, use adaptive MAD-based threshold.

    Returns:
        Sorted list of bad frame indices (0-based).
    """
    num_frames = frames.shape[2]

    # === 1. Variance-based detection ===
    variances = np.array([np.var(frames[:, :, i]) for i in range(num_frames)])

    if variance_threshold is not None and variance_threshold > 0:
        bad_var = set(np.where(variances > variance_threshold)[0])
    else:
        median_var = np.median(variances)
        mad_var = np.median(np.abs(variances - median_var))
        # Floor at 10% of median to avoid flagging legitimate area changes.
        # Binary mask variance = p(1-p)*255^2 changes with area ratio;
        # a tiny MAD (common when most frames have similar area) would
        # otherwise flag normal frames that simply have a different area.
        mad_var = max(mad_var, median_var * 0.10)
        var_threshold = median_var + 5 * mad_var
        bad_var = set(np.where(variances > var_threshold)[0])
        logger.info(
            f"Variance threshold: {var_threshold:.1f} "
            f"(median={median_var:.1f}, MAD={mad_var:.1f})"
        )

    # === 2. Zero-variance detection ===
    bad_zero = set(np.where(variances == 0)[0])

    # === 3. Area-based detection (catches inverted/altered frames) ===
    areas = np.array(
        [np.sum(frames[:, :, i] > 127) for i in range(num_frames)]
    )
    median_area = np.median(areas)
    mad_area = np.median(np.abs(areas - median_area))
    # Ensure minimum sensitivity: at least 5% of median area
    mad_area = max(mad_area, median_area * 0.05)
    area_threshold = 5 * mad_area
    bad_area = set(
        np.where(np.abs(areas - median_area) > area_threshold)[0]
    )

    if bad_area:
        logger.info(
            f"Area outliers detected: {sorted(bad_area)} "
            f"(median_area={median_area:.0f}, threshold=+/-{area_threshold:.0f})"
        )

    # === 4. Neighbor flagging for zero-variance frames ===
    bad_neighbors = set()
    for idx in bad_zero:
        if idx > 0:
            bad_neighbors.add(idx - 1)
        if idx < num_frames - 1:
            bad_neighbors.add(idx + 1)

    bad_indices = sorted(bad_var | bad_zero | bad_neighbors | bad_area)
    if bad_indices:
        logger.info(f"Total bad frames: {len(bad_indices)}: {bad_indices[:20]}...")
    return bad_indices


def fill_nan_frames(
    sequence: np.ndarray,
    bad_indices: list[int],
    num_neighbors: int = 2,
    sigma: float = 2.0,
) -> np.ndarray:
    """Replace bad frames with Gaussian-weighted average of nearest valid neighbors.

    Improvements over basic approach:
    - Adaptive window: expands beyond num_neighbors until valid frames are found
    - Smart fill order: fills frames closest to good frames first, so that
      already-filled frames can serve as sources for more distant bad frames
    - Boundary handling: always finds valid frames even at sequence edges

    Args:
        sequence: 3D array (H, W, N) float32.
        bad_indices: Indices of frames to replace.
        num_neighbors: Minimum neighbor window on each side.
        sigma: Gaussian sigma for spatial smoothing of neighbor frames.

    Returns:
        Filled sequence with bad frames replaced.
    """
    filled = sequence.copy()
    num_frames = sequence.shape[2]
    bad_set = set(bad_indices)
    good_set = set(range(num_frames)) - bad_set

    # Mark all bad frames as NaN
    for idx in bad_indices:
        filled[:, :, idx] = np.nan

    if not good_set:
        logger.warning("All frames are marked bad -- cannot fill any frames")
        return sequence  # return original rather than all-NaN

    # Sort bad frames by distance to nearest good frame (fill nearest first).
    # This enables propagation: frames near good frames get filled first,
    # then become valid sources for frames further away.
    def _dist_to_good(idx: int) -> int:
        return min(abs(idx - g) for g in good_set)

    fill_order = sorted(bad_indices, key=_dist_to_good)

    for idx in fill_order:
        # Expand search radius until at least one valid neighbor is found
        search_radius = num_neighbors
        while search_radius < num_frames:
            lo = max(0, idx - search_radius)
            hi = min(num_frames, idx + search_radius + 1)

            has_valid = any(
                not np.all(np.isnan(filled[:, :, k]))
                for k in range(lo, hi)
                if k != idx
            )
            if has_valid:
                break
            search_radius += 1

        # Build neighbor stack and apply spatial Gaussian to valid frames
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


def apply_temporal_gaussian(
    sequence: np.ndarray, sigma: float = 1.0,
) -> np.ndarray:
    """Apply Gaussian smoothing along the temporal axis only.

    Unlike the old apply_3d_gaussian, this does NOT apply spatial
    smoothing — spatial smoothing is handled by Step 1 (Spatial Smoothing).
    This only blurs along the time dimension (axis=2).

    Args:
        sequence: 3D array (H, W, N) float32.
        sigma: Gaussian sigma for temporal smoothing.

    Returns:
        Temporally smoothed sequence (spatial dimensions untouched).
    """
    return gaussian_filter1d(sequence, sigma=sigma, axis=2)


def temporal_smooth_sequence(
    frames: list[np.ndarray],
    variance_threshold: Optional[float] = None,
    num_neighbors: int = 2,
    sigma: float = 2.0,
    temporal_sigma: Optional[float] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> list[np.ndarray]:
    """Full temporal smoothing pipeline.

    Args:
        frames: List of 2D mask arrays (grayscale uint8).
        variance_threshold: Bad frame detection threshold.
            None = adaptive (recommended). Float = fixed threshold.
        num_neighbors: Minimum temporal window for NaN filling.
        sigma: Gaussian sigma for spatial smoothing during NaN filling.
        temporal_sigma: Gaussian sigma for temporal-only smoothing after
            outlier correction. None or 0 = skip (fix outliers only).
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

    # Optional temporal-only Gaussian (no spatial blur)
    if temporal_sigma and temporal_sigma > 0:
        if progress_callback:
            progress_callback("Applying temporal Gaussian", 2, 4)
        smoothed = apply_temporal_gaussian(filled, sigma=temporal_sigma)
    else:
        logger.info("Skipping temporal Gaussian (outlier fix only mode)")
        smoothed = filled

    if progress_callback:
        progress_callback("Binarizing results", 3, 4)

    results = []
    for i in range(smoothed.shape[2]):
        binary = (smoothed[:, :, i] > 127).astype(np.uint8) * 255
        results.append(binary)

    logger.info(f"Temporal smoothing complete: {len(results)} frames output")
    return results
