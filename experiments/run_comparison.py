"""
Run comparison experiments between LSB and DCT steganography.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from crypto_stego_project.stego.dct_stego import embed_dct, extract_dct
from crypto_stego_project.stego.lsb_stego import embed_lsb, extract_lsb
from crypto_stego_project.utils.bit_utils import build_payload
from crypto_stego_project.utils.image_utils import load_image_cv2, load_image_pillow
from crypto_stego_project.utils.metrics import psnr


def derive_key(key_str: str) -> bytes:
    """Derive a fixed 16-byte key from a string using SHA-256 truncation."""
    return hashlib.sha256(key_str.encode("utf-8")).digest()[:16]


def bit_error_rate(bits_a: List[int], bits_b: List[int]) -> float:
    """Compute BER between two equal-length bit lists."""
    if len(bits_a) != len(bits_b):
        raise ValueError("Bit strings must be the same length for BER")
    if not bits_a:
        return 0.0
    errors = sum(1 for a, b in zip(bits_a, bits_b) if a != b)
    return errors / len(bits_a)


def compare_methods(
    cover_path: Path,
    messages: Sequence[str],
    key_bytes: bytes,
    output_dir: Path,
    jpeg_qualities: Sequence[int],
) -> List[dict]:
    """Run embedding/extraction for LSB and DCT on a single cover image."""
    results: List[dict] = []
    cover_rgb = load_image_pillow(cover_path)
    cover_bgr = load_image_cv2(cover_path)

    for message in messages:
        payload_bits = build_payload(message, key_bytes)

        # LSB
        lsb_stego = output_dir / f"{cover_path.stem}_lsb.png"
        embed_lsb(str(cover_path), str(lsb_stego), payload_bits)
        lsb_extracted = extract_lsb(str(lsb_stego), key_bytes)
        lsb_psnr = psnr(cover_rgb, load_image_pillow(lsb_stego))
        recovered_bits = build_payload(lsb_extracted, key_bytes)
        lsb_ber = bit_error_rate(payload_bits, recovered_bits)
        results.append(
            {
                "image": cover_path.name,
                "method": "lsb",
                "message_len": len(message),
                "psnr": lsb_psnr,
                "ber": lsb_ber,
                "compressed_quality": None,
                "extracted": lsb_extracted,
            }
        )

        # LSB robustness to JPEG
        for q in jpeg_qualities:
            compressed_path = output_dir / f"{cover_path.stem}_lsb_q{q}.jpg"
            with Image.open(lsb_stego) as img:
                img.save(compressed_path, quality=q)
            extracted = extract_lsb(str(compressed_path), key_bytes)
            recovered_bits_q = build_payload(extracted, key_bytes)
            ber_q = bit_error_rate(payload_bits, recovered_bits_q)
            results.append(
                {
                    "image": cover_path.name,
                    "method": "lsb",
                    "message_len": len(message),
                    "psnr": psnr(cover_rgb, load_image_pillow(compressed_path)),
                    "ber": ber_q,
                    "compressed_quality": q,
                    "extracted": extracted,
                }
            )

        # DCT
        dct_stego = output_dir / f"{cover_path.stem}_dct.png"
        embed_dct(str(cover_path), str(dct_stego), payload_bits)
        dct_extracted = extract_dct(str(dct_stego), key_bytes)
        dct_psnr = psnr(cover_bgr, load_image_cv2(dct_stego))
        recovered_bits_dct = build_payload(dct_extracted, key_bytes)
        dct_ber = bit_error_rate(payload_bits, recovered_bits_dct)
        results.append(
            {
                "image": cover_path.name,
                "method": "dct",
                "message_len": len(message),
                "psnr": dct_psnr,
                "ber": dct_ber,
                "compressed_quality": None,
                "extracted": dct_extracted,
            }
        )

        # DCT robustness to JPEG
        for q in jpeg_qualities:
            compressed_path = output_dir / f"{cover_path.stem}_dct_q{q}.jpg"
            cv2.imwrite(str(compressed_path), load_image_cv2(dct_stego), [cv2.IMWRITE_JPEG_QUALITY, q])
            extracted = extract_dct(str(compressed_path), key_bytes)
            recovered_bits_q = build_payload(extracted, key_bytes)
            ber_q = bit_error_rate(payload_bits, recovered_bits_q)
            results.append(
                {
                    "image": cover_path.name,
                    "method": "dct",
                    "message_len": len(message),
                    "psnr": psnr(cover_bgr, load_image_cv2(compressed_path)),
                    "ber": ber_q,
                    "compressed_quality": q,
                    "extracted": extracted,
                }
            )

    return results


def run_experiments(
    cover_dir: Path,
    output_dir: Path,
    key: str,
    messages: Iterable[str],
    jpeg_qualities: Sequence[int],
) -> pd.DataFrame:
    """Run experiments for all cover images in directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    key_bytes = derive_key(key)
    all_results: List[dict] = []
    for cover_path in sorted(cover_dir.iterdir()):
        if not cover_path.is_file():
            continue
        if cover_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        all_results.extend(compare_methods(cover_path, list(messages), key_bytes, output_dir, jpeg_qualities))
    df = pd.DataFrame(all_results)
    csv_path = output_dir / "comparison_results.csv"
    df.to_csv(csv_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LSB and DCT steganography.")
    parser.add_argument("cover_dir", type=Path, help="Directory containing cover images")
    parser.add_argument("output_dir", type=Path, help="Directory to store stego images and results")
    parser.add_argument("--key", type=str, default="defaultkey", help="Key string for cipher")
    parser.add_argument(
        "--messages",
        nargs="+",
        default=["short", "medium message", "This is a longer secret message for testing purposes."],
        help="Messages to embed",
    )
    parser.add_argument(
        "--jpeg-qualities",
        nargs="+",
        type=int,
        default=[90, 70, 50],
        help="JPEG qualities to test for robustness",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_experiments(args.cover_dir, args.output_dir, args.key, args.messages, args.jpeg_qualities)
    print(df)


if __name__ == "__main__":
    main()
