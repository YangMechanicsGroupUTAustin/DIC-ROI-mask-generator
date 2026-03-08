"""Perona-Malik anisotropic diffusion smoothing for binary masks.

Port of imageSmoothingGUI.m spatial smoothing to Python.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, laplace


def perona_malik_smooth(
    image,
    num_iterations=50,
    dt=0.1,
    lam=0.5,
    gaussian_sigma=4.0,
):
    """Apply Perona-Malik anisotropic diffusion smoothing.

    Args:
        image: Input image as numpy array (uint8 0-255 or float 0-1).
        num_iterations: Number of diffusion iterations.
        dt: Time step per iteration (controls convergence speed).
        lam: Gradient sensitivity (lambda parameter).
        gaussian_sigma: Sigma for post-iteration Gaussian filter.

    Returns:
        Smoothed binary mask as uint8 array (0 or 255).
    """
    smoothed = image.astype(np.float64)
    if smoothed.max() > 1.0:
        smoothed = smoothed / 255.0

    for _ in range(num_iterations):
        gy, gx = np.gradient(smoothed)
        grad_mag_sq = gx ** 2 + gy ** 2

        diffusivity = np.exp(-grad_mag_sq / (2 * lam ** 2))

        laplacian_d = laplace(diffusivity)
        laplacian_s = laplace(smoothed)
        smoothed = smoothed + dt * (laplacian_d * laplacian_s)

        smoothed = gaussian_filter(smoothed, sigma=gaussian_sigma)

    binary = (smoothed > 0.5).astype(np.uint8) * 255
    return binary
