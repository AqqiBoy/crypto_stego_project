"""
DCT-based steganography on the luminance channel in YCrCb space.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from crypto_stego_project.utils.bit_utils import parse_payload
from crypto_stego_project.utils.image_utils import load_image_cv2, load_image_cv2_rgb, save_image_cv2

MID_FREQ_POSITIONS: List[Tuple[int, int]] = [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]


def _set_lsb(value: float, bit: int) -> float:
    """Set the least significant bit of an integer-valued float."""
    as_int = int(round(value))
    as_int = (as_int & ~1) | (bit & 1)
    return float(as_int)


def _get_lsb(value: float) -> int:
    """Get the least significant bit of an integer-valued float."""
    as_int = int(round(value))
    return as_int & 1


def _stabilize_block(
    base_block: np.ndarray,
    targets: List[Tuple[Tuple[int, int], int]],
    max_iters: int = 20,
) -> np.ndarray:
    """
    Re-embed bits until parity survives the DCT->IDCT->uint8 round-trip.
    """
    dct_block = base_block.copy()
    for _ in range(max_iters):
        for pos, bit in targets:
            dct_block[pos] = _set_lsb(dct_block[pos], bit)
        spatial = cv2.idct(dct_block)
        spatial_uint8 = np.clip(spatial, 0, 255).astype(np.uint8)
        round_trip_dct = cv2.dct(spatial_uint8.astype(np.float32))
        if all(_get_lsb(round_trip_dct[pos]) == bit for pos, bit in targets):
            return round_trip_dct
        # If mismatch, nudge the problematic coefficients and try again.
        for pos, bit in targets:
            if _get_lsb(round_trip_dct[pos]) != bit:
                sign = 1 if round_trip_dct[pos] >= 0 else -1
                round_trip_dct[pos] += sign  # increase magnitude to make parity stick
        dct_block = round_trip_dct
    return dct_block


def get_dct_capacity(image_array: np.ndarray, positions: List[Tuple[int, int]] = MID_FREQ_POSITIONS) -> int:
    """
    Compute capacity in bits based on number of 8x8 blocks and positions.
    """
    h, w = image_array.shape[:2]
    blocks = (h // 8) * (w // 8)
    return blocks * len(positions)


def embed_dct(cover_path: str, output_path: str, payload_bits: List[int]) -> None:
    """
    Embed payload bits into mid-frequency DCT coefficients of the Y channel.
    Converts cover to truecolor RGB first to avoid palette/alpha surprises.
    """
    bgr = load_image_cv2_rgb(cover_path)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0].astype(np.float32)

    h, w = y_channel.shape
    block_h = (h // 8) * 8
    block_w = (w // 8) * 8

    capacity = get_dct_capacity(y_channel, MID_FREQ_POSITIONS)
    if len(payload_bits) > capacity:
        raise ValueError(f"Payload too large for DCT method. Capacity={capacity} bits")

    bit_idx = 0
    for row in range(0, block_h, 8):
        for col in range(0, block_w, 8):
            block = y_channel[row : row + 8, col : col + 8]
            dct_block = cv2.dct(block)
            targets: List[Tuple[Tuple[int, int], int]] = []
            for pos in MID_FREQ_POSITIONS:
                if bit_idx >= len(payload_bits):
                    break
                bit = payload_bits[bit_idx]
                dct_block[pos] = _set_lsb(dct_block[pos], bit)
                targets.append((pos, bit))
                bit_idx += 1
            if targets:
                dct_block = _stabilize_block(dct_block, targets)
            y_channel[row : row + 8, col : col + 8] = cv2.idct(dct_block)
            if bit_idx >= len(payload_bits):
                break
        if bit_idx >= len(payload_bits):
            break

    y_channel = np.clip(y_channel, 0, 255).astype(np.uint8)
    ycrcb[:block_h, :block_w, 0] = y_channel[:block_h, :block_w]
    stego_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    save_image_cv2(stego_bgr, output_path)


def extract_dct(stego_path: str, key: bytes) -> str:
    """
    Extract and decrypt payload bits from a DCT stego image.
    """
    bgr = load_image_cv2(stego_path)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0].astype(np.float32)
    h, w = y_channel.shape
    block_h = (h // 8) * 8
    block_w = (w // 8) * 8

    bits: List[int] = []
    for row in range(0, block_h, 8):
        for col in range(0, block_w, 8):
            block = y_channel[row : row + 8, col : col + 8]
            dct_block = cv2.dct(block)
            for pos in MID_FREQ_POSITIONS:
                bits.append(_get_lsb(dct_block[pos]))
    return parse_payload(bits, key)
