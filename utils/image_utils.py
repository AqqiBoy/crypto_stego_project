"""
Image IO helpers to keep pillow and cv2 usage consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

PathLike = Union[str, Path]


def load_image_pillow(path: PathLike) -> np.ndarray:
    """Load an image using Pillow and return an RGB numpy array (uint8)."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return np.array(rgb, dtype=np.uint8)


def save_image_pillow(arr: np.ndarray, path: PathLike) -> None:
    """Save an RGB numpy array using Pillow."""
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def load_image_cv2(path: PathLike) -> np.ndarray:
    """Load an image using cv2 (BGR order)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    return img


def load_image_cv2_rgb(path: PathLike) -> np.ndarray:
    """Load an image as truecolor RGB and return BGR for cv2 workflows."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        rgb_arr = np.array(rgb, dtype=np.uint8)
    return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)


def save_image_cv2(arr: np.ndarray, path: PathLike) -> None:
    """Save an image using cv2 (expects BGR)."""
    success = cv2.imwrite(str(path), arr)
    if not success:
        raise IOError(f"Could not save image to {path}")
