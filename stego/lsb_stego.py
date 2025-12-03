"""
Simple LSB-based spatial steganography for RGB images.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from crypto_stego_project.utils.bit_utils import parse_payload
from crypto_stego_project.utils.image_utils import load_image_pillow, save_image_pillow


def get_lsb_capacity(image_array: np.ndarray) -> int:
    """Return capacity in bits for 1 LSB per channel (3 bits per pixel)."""
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3)")
    h, w, _ = image_array.shape
    return h * w * 3


def embed_lsb(cover_path: str, output_path: str, payload_bits: list[int]) -> None:
    """
    Embed payload bits into the least significant bits of an RGB image.
    """
    image = load_image_pillow(cover_path)
    capacity = get_lsb_capacity(image)
    if len(payload_bits) > capacity:
        raise ValueError(f"Payload too large for cover image. Capacity={capacity} bits")

    flat = image.flatten()
    modified = flat.copy()
    modified[: len(payload_bits)] = (modified[: len(payload_bits)] & 0xFE) | np.array(
        payload_bits, dtype=np.uint8
    )
    stego = modified.reshape(image.shape)
    save_image_pillow(stego, output_path)


def extract_lsb(stego_path: str, key: bytes) -> str:
    """
    Extract and decrypt payload bits from an LSB stego image.
    """
    image = load_image_pillow(stego_path)
    bits = (image.flatten() & 1).tolist()
    return parse_payload(bits, key)
