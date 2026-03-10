"""Perona-Malik anisotropic diffusion smoothing for binary masks.

Corrected implementation using forward differences and proper divergence formula.
Reference: P. Perona and J. Malik, "Scale-Space and Edge Detection Using
Anisotropic Diffusion," IEEE TPAMI, 1990.
"""

import logging

import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger("sam2studio.spatial_smoothing")


def perona_malik_smooth(
    image: np.ndarray,
    num_iterations: int = 50,
    dt: float = 0.1,
    kappa: float = 30.0,
    option: int = 1,
    post_gaussian_sigma: float = 0.0,
) -> np.ndarray:
    """Apply Perona-Malik anisotropic diffusion smoothing.

    Uses forward differences and proper divergence of (c * grad(u)) formula.
    Edge-preserving: smooths homogeneous regions while preserving boundaries.

    Args:
        image: Input image as numpy array (uint8 0-255 or float 0-1).
        num_iterations: Number of diffusion iterations.
        dt: Time step per iteration. Must be <= 0.25 for stability.
        kappa: Conductance parameter (gradient magnitude threshold).
            Higher kappa = more smoothing across edges.
        option: Diffusivity function:
            1 = exp(-(|grad|/kappa)^2) -- favors high-contrast edges
            2 = 1 / (1 + (|grad|/kappa)^2) -- favors wide regions
        post_gaussian_sigma: Optional Gaussian blur AFTER all iterations (0 = off).

    Returns:
        Smoothed binary mask as uint8 array (0 or 255).
    """
    # Clamp dt for numerical stability
    dt = min(dt, 0.25)

    u = image.astype(np.float64)
    if u.max() > 1.0:
        u = u / 255.0

    for iteration in range(num_iterations):
        # Forward differences in 4 directions with Neumann boundary (zero gradient)
        delta_n = np.roll(u, -1, axis=0) - u  # North
        delta_s = np.roll(u, 1, axis=0) - u   # South
        delta_e = np.roll(u, -1, axis=1) - u  # East
        delta_w = np.roll(u, 1, axis=1) - u   # West

        # Zero out wrapped boundaries to prevent edge artifacts
        delta_n[-1, :] = 0
        delta_s[0, :] = 0
        delta_e[:, -1] = 0
        delta_w[:, 0] = 0

        # Diffusion coefficients per direction
        if option == 1:
            c_n = np.exp(-(delta_n / kappa) ** 2)
            c_s = np.exp(-(delta_s / kappa) ** 2)
            c_e = np.exp(-(delta_e / kappa) ** 2)
            c_w = np.exp(-(delta_w / kappa) ** 2)
        else:
            c_n = 1.0 / (1.0 + (delta_n / kappa) ** 2)
            c_s = 1.0 / (1.0 + (delta_s / kappa) ** 2)
            c_e = 1.0 / (1.0 + (delta_e / kappa) ** 2)
            c_w = 1.0 / (1.0 + (delta_w / kappa) ** 2)

        # Divergence of diffusion flux
        u = u + dt * (c_n * delta_n + c_s * delta_s + c_e * delta_e + c_w * delta_w)

    # Optional post-smoothing (applied once AFTER all iterations)
    if post_gaussian_sigma > 0:
        u = gaussian_filter(u, sigma=post_gaussian_sigma)

    binary = (u > 0.5).astype(np.uint8) * 255
    return binary
