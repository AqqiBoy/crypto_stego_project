# Crypto Stego Project

Educational project combining a simple custom block cipher with two steganography schemes (LSB and DCT) to embed and recover text messages from images. **Not for real-world security.**

## Requirements

- Python 3.10+
- Install dependencies:
  ```bash
  pip install -r crypto_stego_project/requirements.txt
  ```

## Usage

All commands are run from the directory that contains the `crypto_stego_project/` folder.

### Embed

```bash
python -m crypto_stego_project.main embed \
  --method lsb \
  --cover crypto_stego_project/Cover/1.png \
  --output crypto_stego_project/Output/stego.png \
  --message "Secret text" \
  --key "your password"
```

For DCT-based embedding, use `--method dct`. You can also supply `--message-file path/to/text.txt`.

### Extract

```bash
python -m crypto_stego_project.main extract \
  --method lsb \
  --stego crypto_stego_project/Output/stego.png \
  --key "your password"
```

Use the same key and method that were used for embedding.

### DCT Tips

- Covers are auto-converted to truecolor RGB; prefer lossless PNGs and dimensions that are multiples of 8 for full capacity.
- DCT embeds bits in mid-frequency coefficients to better survive mild JPEG compression; heavy recompression or resizing can still corrupt payloads.

## Experiments

Run comparisons for all images in a directory:

```bash
python -m crypto_stego_project.experiments.run_comparison crypto_stego_project/Cover crypto_stego_project/Output --key "password" --messages "short" "longer test message"
```

The script saves stego images, JPEG-compressed variants, and a `comparison_results.csv` summarizing PSNR and BER for LSB vs. DCT.

## Notes

- The block cipher is a tiny Feistel network intended only for learning.
- DCT embedding operates on the luminance channel and is more robust to JPEG compression than basic LSB, but both are fragile compared to production-grade stego systems.
