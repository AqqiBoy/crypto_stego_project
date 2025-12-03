"""
Bit-level helpers and payload (en|de)coding utilities.
"""

from __future__ import annotations

import struct
from typing import List

from crypto_stego_project.cipher.custom_block_cipher import decrypt_message, encrypt_message


def bytes_to_bits(data: bytes) -> List[int]:
    """Convert a byte string to a list of bits (most significant bit first)."""
    bits: List[int] = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bits_to_bytes(bits: List[int]) -> bytes:
    """
    Convert a list of bits to bytes. Incomplete final byte is zero-padded.
    """
    if not bits:
        return b""
    # Pad to nearest byte boundary with zeros.
    padded_length = ((len(bits) + 7) // 8) * 8
    padded_bits = bits + [0] * (padded_length - len(bits))
    out = bytearray()
    for i in range(0, len(padded_bits), 8):
        byte = 0
        for b in padded_bits[i : i + 8]:
            byte = (byte << 1) | (b & 1)
        out.append(byte)
    return bytes(out)


def build_payload(plaintext: str, key: bytes) -> List[int]:
    """
    Encrypt plaintext and build a payload with a length prefix.

    Payload format:
        [4 bytes big-endian ciphertext length] + [ciphertext bytes]
    """
    plaintext_bytes = plaintext.encode("utf-8")
    ciphertext = encrypt_message(plaintext_bytes, key)
    header = struct.pack(">I", len(ciphertext))
    payload = header + ciphertext
    return bytes_to_bits(payload)


def parse_payload(bits: List[int], key: bytes) -> str:
    """
    Parse payload bits and decrypt to recover plaintext.

    Stops after reading the declared ciphertext length to avoid consuming
    unnecessary trailing bits.
    """
    if len(bits) < 32:
        raise ValueError("Not enough bits to read payload length")

    # Read header first (32 bits).
    header_bytes = bits_to_bytes(bits[:32])
    cipher_length = struct.unpack(">I", header_bytes)[0]
    total_bits_needed = (4 + cipher_length) * 8
    if len(bits) < total_bits_needed:
        # Try to consume only available bits but ensure we have enough.
        raise ValueError(
            f"Not enough bits for payload: need {total_bits_needed}, have {len(bits)}"
        )

    payload_bytes = bits_to_bytes(bits[:total_bits_needed])
    ciphertext = payload_bytes[4 : 4 + cipher_length]
    plaintext_bytes = decrypt_message(ciphertext, key)
    return plaintext_bytes.decode("utf-8")
