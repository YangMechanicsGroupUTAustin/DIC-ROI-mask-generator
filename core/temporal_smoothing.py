"""Temporal smoothing for mask sequences.

Port of imageSmoothingGUI.m temporal smoothing to Python.
Bad frame detection, NaN filling, and 3D Gaussian filtering.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def detect_bad_frames(frames, variance_threshold=50000):
    """Detect frames with abnormal variance (noise or blank frames).

    Args:
        frames: 3D array (H, W, N) of frame data.
        variance_threshold: Maximum variance before flagging as bad.

    Returns:
        Sorted list of bad frame indices (0-based).
    """
    num_frames = frames.shape[2]
    variances = np.array([np.var(frames[:, :, i]) for i in range(num_frames)])

    bad_high = set(np.where(variances > variance_threshold)[0])
    bad_zero = set(np.where(variances == 0)[0])

    bad_neighbors = set()
    for idx in bad_zero:
        if idx > 0:
            bad_neighbors.add(idx - 1)
        if idx < num_frames - 1:
            bad_neighbors.add(idx + 1)

    return sorted(bad_high | bad_zero | bad_neighbors)


def fill_nan_frames(sequence, bad_indices, num_neighbors=2, sigma=2.0):
    """Replace bad frames with Gaussian-weighted temporal average of neighbors.

    Args:
        sequence: 3D array (H, W, N).
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

        with np.errstate(all='ignore'):
            avg = np.nanmean(neighbors, axis=2)
        avg = np.nan_to_num(avg, nan=0.0)
        filled[:, :, idx] = avg

    return filled


def apply_3d_gaussian(sequence, sigma=2.0):
    """Apply 3D Gaussian filter for spatio-temporal smoothing.

    Args:
        sequence: 3D array (H, W, N).
        sigma: Gaussian sigma for all 3 dimensions.

    Returns:
        Smoothed 3D sequence.
    """
    return gaussian_filter(sequence, sigma=[sigma, sigma, sigma])


def temporal_smooth_sequence(
    frames,
    variance_threshold=50000,
    num_neighbors=2,
    sigma=2.0,
    progress_callback=None,
):
    """Full temporal smoothing pipeline.

    Args:
        frames: List of 2D mask arrays (grayscale uint8).
        variance_threshold: Bad frame detection threshold.
        num_neighbors: Temporal window for NaN filling.
        sigma: Gaussian sigma for both filling and 3D filter.
        progress_callback: Optional callable(step_name, current, total).

    Returns:
        List of temporally smoothed and binarized mask arrays (uint8).
    """
    h, w = frames[0].shape[:2]
    sequence = np.zeros((h, w, len(frames)), dtype=np.float64)
    for i, f in enumerate(frames):
        gray = f if len(f.shape) == 2 else f[:, :, 0]
        sequence[:, :, i] = gray.astype(np.float64)

    if progress_callback:
        progress_callback("Detecting bad frames", 1, 4)
    bad_indices = detect_bad_frames(sequence, variance_threshold)

    if progress_callback:
        progress_callback("Filling bad frames", 2, 4)
    filled = fill_nan_frames(sequence, bad_indices, num_neighbors, sigma)

    if progress_callback:
        progress_callback("Applying 3D Gaussian filter", 3, 4)
    smoothed = apply_3d_gaussian(filled, sigma)

    if progress_callback:
        progress_callback("Binarizing results", 4, 4)

    results = []
    for i in range(smoothed.shape[2]):
        binary = (smoothed[:, :, i] > 127).astype(np.uint8) * 255
        results.append(binary)

    return results
