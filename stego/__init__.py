"""
Steganography methods (LSB and DCT).
"""

from .lsb_stego import embed_lsb, extract_lsb, get_lsb_capacity
from .dct_stego import embed_dct, extract_dct, get_dct_capacity

__all__ = [
    "get_lsb_capacity",
    "embed_lsb",
    "extract_lsb",
    "get_dct_capacity",
    "embed_dct",
    "extract_dct",
]
