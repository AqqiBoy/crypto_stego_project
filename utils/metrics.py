"""
Image quality metrics.
"""

from __future__ import annotations

import math
import numpy as np


def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean squared error between two images of the same shape."""
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape for MSE")
    diff = img1.astype(np.float64) - img2.astype(np.float64)
    return float(np.mean(diff ** 2))


def psnr(img1: np.ndarray, img2: np.ndarray, max_pixel: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    error = mse(img1, img2)
    if error == 0:
        return float("inf")
    return 20 * math.log10(max_pixel) - 10 * math.log10(error)
