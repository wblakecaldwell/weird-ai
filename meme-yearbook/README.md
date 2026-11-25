# Meme Valley High School — Class of 2025

Local AI pipeline for generating an original **Gumball-style mixed-media Meme Yearbook** on Apple Silicon using **LM Studio** (world-building & roster) + **`mflux`** (native MLX image generation) + a print-ready **HTML Yearbook**.

<p align="center">
  <img src="./assets/yearbook-spread-a-h.png" alt="Meme Valley High School Class of 2025 — Seniors A–C & C–H Two-Page Spread" width="100%" />
  <br />
  <em>Sample two-page spread (Seniors A–C &amp; C–H): 10 original mixed-media meme characters generated locally with LM Studio + mflux.</em>
</p>

## Setup

1. Copy `.env.example` to `.env` and add your LM Studio API token (from **LM Studio -> Developer -> Server Settings**, or disable server auth for localhost):
   ```bash
   cp .env.example .env
   ```
2. Load a model in LM Studio (for example `qwen3-vl-30b-a3b-instruct-mlx`) and start the local server on port `1234`.

## Pipeline Commands

### 1. Generate a fresh 20-student roster via LM Studio
```bash
uv run python scripts/generate_roster.py --count 20
```
*(A starter 20-student `data/roster.json` is already included so you can test portrait generation immediately.)*

### 2. Generate portraits locally with MLX (`mflux`)
Fast start using your already-cached FLUX.1-schnell 4-bit + cached pixel-art LoRA:
```bash
uv run python scripts/generate_portraits.py --model schnell-4bit --variants 1
```

Or try newer higher-detail MLX models (auto-downloaded on first run):
```bash
# Great stylized / Gumball variety and fast on M5 Pro
uv run python scripts/generate_portraits.py --model z-image-turbo --overwrite

# Great mixed-media studio lighting
uv run python scripts/generate_portraits.py --model flux2-klein-9b --overwrite

# Highest prompt fidelity for weird character concepts
uv run python scripts/generate_portraits.py --model qwen-image-2.1-turbo --overwrite
```

Re-roll just one student:
```bash
uv run python scripts/generate_portraits.py --model schnell-4bit --only chester --variants 3 --overwrite
```

### 3. Normalize portraits (4:5 crop, crisp pixel-grid snap, studio paper grain)
```bash
uv run python scripts/normalize_portraits.py
```

### 4. Build the Class of 2025 HTML Yearbook
```bash
uv run python scripts/build_yearbook.py
open index.html
```
Use **Print / Save PDF** in the top toolbar to export a clean multi-page PDF.
