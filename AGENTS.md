# Repository Guidelines

## Project Structure & Modules
- `main.py`: CLI entrypoint for embed/extract operations.
- `cipher/`: custom Feistel-based block cipher used to encrypt payloads.
- `stego/`: LSB and DCT embedding/extraction implementations.
- `utils/`: bit packing, image I/O, and metric helpers.
- `experiments/`: comparison script that batches covers, saves stego variants, and writes `comparison_results.csv`.

## Setup, Build, and Run
- Install deps: `pip install -r requirements.txt` (Python 3.10+).
- Embed: `python -m crypto_stego_project.main embed --method lsb --cover input.png --output stego.png --message "Hi" --key "pass"`.
- Extract: `python -m crypto_stego_project.main extract --method lsb --stego stego.png --key "pass"`.
- DCT variant: swap `--method dct` in the above commands.
- Experiments: `python -m crypto_stego_project.experiments.run_comparison covers/ outputs/ --key "pass" --messages "short" "longer"`.
- DCT guidance: covers are auto-converted to truecolor RGB; prefer lossless PNGs with dimensions divisible by 8 for full capacity and better bit stability.

## Coding Style & Naming
- Follow PEP 8: 4-space indentation, snake_case for functions/variables, CapWords for classes (if added).
- Keep type hints (used across modules) and short docstrings describing inputs/outputs.
- Prefer pure functions; avoid global state and mutate arrays in place only when intentional.
- Use existing helpers (`bit_utils`, `image_utils`) rather than reimplementing conversions.

## Testing & Verification
- No automated test suite yet; smoke-test with a small PNG: run `embed` then `extract` and confirm the recovered text matches.
- For robustness checks, run the experiments script; review `comparison_results.csv` and the generated stego/JPEG images under the output directory.
- When adding features, include at least one reproducible command in notes/PRs showing the expected output.

## Commit & Pull Request Guidelines
- Current history uses concise subject lines (e.g., "Initial commit"); keep using imperative, present-tense subjects under ~72 characters.
- Scope prefixes are helpful: `lsb:`, `dct:`, `cipher:`, `utils:`, `cli:`.
- PRs should include: goal summary, sample commands/output, and any visual artifacts (before/after images if relevant).
- Link issues when available and call out risk areas (capacity limits, payload parsing, image format quirks).

## Security & Data Handling
- This toolkit is educational; do not use for real secrets or production deployments.
- Treat keys as short-lived; avoid hard-coding or committing them, and clear sample data that contains sensitive content.
- Use lossless covers (PNG) for LSB tests; note that JPEG recompression can distort payloads unless using the DCT flow.
