"""
Cipher package exposing the custom block cipher.
"""

from .custom_block_cipher import (
    decrypt_block,
    decrypt_message,
    derive_round_keys,
    encrypt_block,
    encrypt_message,
    pad,
    unpad,
)

__all__ = [
    "derive_round_keys",
    "encrypt_block",
    "decrypt_block",
    "pad",
    "unpad",
    "encrypt_message",
    "decrypt_message",
]
