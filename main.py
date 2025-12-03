"""
CLI for embedding and extracting messages using LSB or DCT steganography.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Callable, List

from crypto_stego_project.stego.dct_stego import embed_dct, extract_dct
from crypto_stego_project.stego.lsb_stego import embed_lsb, extract_lsb
from crypto_stego_project.utils.bit_utils import build_payload


def derive_key(key_str: str) -> bytes:
    """Derive a fixed-length 16-byte key from an arbitrary string."""
    return hashlib.sha256(key_str.encode("utf-8")).digest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Custom cipher steganography toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed_parser = subparsers.add_parser("embed", help="Embed a message into an image")
    embed_parser.add_argument("--method", choices=["lsb", "dct"], required=True)
    embed_parser.add_argument("--cover", required=True, help="Path to cover image")
    embed_parser.add_argument("--output", required=True, help="Path to save stego image")
    embed_parser.add_argument("--message", help="Secret message to embed")
    embed_parser.add_argument("--message-file", help="Path to text file containing secret message")
    embed_parser.add_argument("--key", required=True, help="Key string (will be hashed to 16 bytes)")

    extract_parser = subparsers.add_parser("extract", help="Extract a message from an image")
    extract_parser.add_argument("--method", choices=["lsb", "dct"], required=True)
    extract_parser.add_argument("--stego", required=True, help="Path to stego image")
    extract_parser.add_argument("--key", required=True, help="Key string used for embedding")

    return parser.parse_args()


def get_message(args: argparse.Namespace) -> str:
    if args.message_file:
        return Path(args.message_file).read_text(encoding="utf-8")
    if args.message is None:
        raise ValueError("Provide --message or --message-file")
    return args.message


def embed_command(args: argparse.Namespace) -> None:
    key_bytes = derive_key(args.key)
    message = get_message(args)
    payload_bits: List[int] = build_payload(message, key_bytes)

    if args.method == "lsb":
        embed_fn: Callable[[str, str, List[int]], None] = embed_lsb
    else:
        embed_fn = embed_dct

    embed_fn(args.cover, args.output, payload_bits)
    print(f"Embedded message using {args.method.upper()} into {args.output}")


def extract_command(args: argparse.Namespace) -> None:
    key_bytes = derive_key(args.key)
    if args.method == "lsb":
        extractor = extract_lsb
    else:
        extractor = extract_dct
    message = extractor(args.stego, key_bytes)
    print("Recovered message:")
    print(message)


def main() -> None:
    args = parse_args()
    if args.command == "embed":
        embed_command(args)
    else:
        extract_command(args)


if __name__ == "__main__":
    main()
