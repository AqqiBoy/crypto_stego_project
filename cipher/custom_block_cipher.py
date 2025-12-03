"""
Implementation of a toy 64-bit Feistel-based block cipher.

WARNING: This cipher is for educational purposes only and must not be used
for real-world security.
"""

from __future__ import annotations

from typing import List

BLOCK_SIZE = 8  # bytes
KEY_SIZE = 16  # bytes
ROUNDS = 8

# Simple nibble substitution box for non-linearity.
S_BOX = [
    0xE,
    0x4,
    0xD,
    0x1,
    0x2,
    0xF,
    0xB,
    0x8,
    0x3,
    0xA,
    0x6,
    0xC,
    0x5,
    0x9,
    0x0,
    0x7,
]

# Round constants to mix with key schedule.
ROUND_CONSTANTS = [
    0xA5A5A5A5,
    0x5A5A5A5A,
    0x3C3C3C3C,
    0xC3C3C3C3,
    0x0F0F0F0F,
    0xF0F0F0F0,
    0x96969696,
    0x69696969,
]


def _rotate_left(value: int, bits: int, width: int = 32) -> int:
    """Left rotate an integer within a fixed bit width."""
    bits %= width
    mask = (1 << width) - 1
    return ((value << bits) & mask) | ((value & mask) >> (width - bits))


def _substitute_nibbles(value: int) -> int:
    """Apply the S-box substitution to each 4-bit nibble of a 32-bit integer."""
    result = 0
    for i in range(8):
        nibble = (value >> (i * 4)) & 0xF
        substituted = S_BOX[nibble]
        result |= substituted << (i * 4)
    return result


def derive_round_keys(key: bytes) -> List[int]:
    """
    Derive round keys from a 128-bit (16-byte) user key.

    The key is split into four 32-bit chunks. For each round, a chunk is
    rotated and xored with a round constant to produce an individual round key.
    """
    if len(key) != KEY_SIZE:
        raise ValueError(f"Key must be {KEY_SIZE} bytes, got {len(key)}")

    chunks = [
        int.from_bytes(key[i : i + 4], byteorder="big") for i in range(0, KEY_SIZE, 4)
    ]
    round_keys: List[int] = []
    for i in range(ROUNDS):
        base = chunks[i % 4]
        rotated = _rotate_left(base, i + 1, width=32)
        round_key = (rotated ^ ROUND_CONSTANTS[i]) & 0xFFFFFFFF
        round_keys.append(round_key)
    return round_keys


def _round_function(right: int, round_key: int) -> int:
    """
    Feistel round function mixing the right half with a round key.

    Steps:
    1) XOR with round key.
    2) Apply nibble substitution.
    3) Rotate left by 3 bits.
    """
    mixed = (right ^ round_key) & 0xFFFFFFFF
    substituted = _substitute_nibbles(mixed)
    rotated = _rotate_left(substituted, 3, width=32)
    return rotated & 0xFFFFFFFF


def encrypt_block(block: bytes, round_keys: List[int]) -> bytes:
    """
    Encrypt a single 64-bit block using the provided round keys.
    """
    if len(block) != BLOCK_SIZE:
        raise ValueError(f"Block must be {BLOCK_SIZE} bytes, got {len(block)}")
    if len(round_keys) != ROUNDS:
        raise ValueError(f"Expected {ROUNDS} round keys, got {len(round_keys)}")

    left = int.from_bytes(block[:4], byteorder="big")
    right = int.from_bytes(block[4:], byteorder="big")

    for rk in round_keys:
        left, right = right, (left ^ _round_function(right, rk)) & 0xFFFFFFFF

    combined = (left << 32) | right
    return combined.to_bytes(BLOCK_SIZE, byteorder="big")


def decrypt_block(block: bytes, round_keys: List[int]) -> bytes:
    """
    Decrypt a single 64-bit block using the provided round keys.
    """
    if len(block) != BLOCK_SIZE:
        raise ValueError(f"Block must be {BLOCK_SIZE} bytes, got {len(block)}")
    if len(round_keys) != ROUNDS:
        raise ValueError(f"Expected {ROUNDS} round keys, got {len(round_keys)}")

    left = int.from_bytes(block[:4], byteorder="big")
    right = int.from_bytes(block[4:], byteorder="big")

    for rk in reversed(round_keys):
        left, right = (right ^ _round_function(left, rk)) & 0xFFFFFFFF, left

    combined = (left << 32) | right
    return combined.to_bytes(BLOCK_SIZE, byteorder="big")


def pad(data: bytes, block_size: int = BLOCK_SIZE) -> bytes:
    """PKCS#7 padding."""
    padding_len = block_size - (len(data) % block_size)
    return data + bytes([padding_len] * padding_len)


def unpad(data: bytes, block_size: int = BLOCK_SIZE) -> bytes:
    """Remove PKCS#7 padding."""
    if not data or len(data) % block_size != 0:
        raise ValueError("Invalid padded data length")
    padding_len = data[-1]
    if padding_len == 0 or padding_len > block_size:
        raise ValueError("Invalid padding byte")
    if data[-padding_len:] != bytes([padding_len] * padding_len):
        raise ValueError("Invalid PKCS#7 padding")
    return data[:-padding_len]


def encrypt_message(plaintext: bytes, key: bytes) -> bytes:
    """
    Encrypt a plaintext of arbitrary length with PKCS#7 padding.
    """
    if not key:
        raise ValueError("Key must not be empty")
    round_keys = derive_round_keys(key)
    padded = pad(plaintext, BLOCK_SIZE)
    ciphertext = bytearray()
    for i in range(0, len(padded), BLOCK_SIZE):
        ciphertext.extend(encrypt_block(padded[i : i + BLOCK_SIZE], round_keys))
    return bytes(ciphertext)


def decrypt_message(ciphertext: bytes, key: bytes) -> bytes:
    """
    Decrypt ciphertext produced by encrypt_message.
    """
    if not key:
        raise ValueError("Key must not be empty")
    if len(ciphertext) % BLOCK_SIZE != 0:
        raise ValueError("Ciphertext is not aligned to block size")
    round_keys = derive_round_keys(key)
    plaintext_padded = bytearray()
    for i in range(0, len(ciphertext), BLOCK_SIZE):
        plaintext_padded.extend(decrypt_block(ciphertext[i : i + BLOCK_SIZE], round_keys))
    return unpad(bytes(plaintext_padded), BLOCK_SIZE)


if __name__ == "__main__":
    # Basic sanity checks for encryption/decryption symmetry.
    test_key = b"Sixteen byte key"
    messages = [
        b"",
        b"hello",
        b"Feistel networks are neat!",
        b"A" * 100,
    ]
    for msg in messages:
        cipher = encrypt_message(msg, test_key)
        recovered = decrypt_message(cipher, test_key)
        assert recovered == msg, f"Round-trip failed for {msg!r}"
    print("All cipher self-tests passed.")
