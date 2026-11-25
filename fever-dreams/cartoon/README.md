# Fever Dreams Cartoon Pipeline

Turns a finished Little Fever Dream Book (`books/<slug>/story.json` + `art/`) into a narrated, voiced, partially animated 1950s read-along cartoon. 100% local on Apple Silicon.

| Step | Script | What it does |
|---|---|---|
| 1 | `prepare_script.py` | Gemma 31B (LM Studio) turns `story.json` into a per-page voice + animation script (`books/<slug>/cartoon/script.json`) |
| 2 | `render_voices.py` | Casts consistent character voices with Qwen3-TTS VoiceDesign (reference clips in `voice_refs/`) + Qwen3-TTS Base cloning, Kokoro fallback, ffmpeg DSP + loudness normalization |
| 3 | `animate_book.py` | Animates each page illustration with LTX-2.5 image-to-video (DistilledPipeline) |
| 4 | `assemble_cartoon.py` | Assembles clips, Ken Burns extensions, voices, page-turn chimes, subtitles, and credits into a 24fps MP4 |

Steps 2–4 append a run to `books/<slug>/cartoon/render_log.json` (via `render_log.py`) recording models, settings, seeds, per-line / per-clip / per-shot results, and timings. It is saved after every item, so interrupted runs are recorded too.

> [!NOTE]
> Book 01's cartoon was rendered before logging existed, so its `render_log.json` only covers runs made after this change. Its `script.json`, `script_voiced.json`, and `.srt` are the record of that render.

## Setup

> [!NOTE]
> Reconstructed from the working environment; dependencies live in a separate `.venv` from the book engine.

```bash
cd fever-dreams/cartoon

# Vendored LTX-2 MLX code (git-ignored)
git clone https://github.com/xocialize/ltx-2-mlx.git vendor/ltx-2-mlx && git -C vendor/ltx-2-mlx checkout c952c8f
git clone https://github.com/Blaizzy/mlx-video.git vendor/mlx-video && git -C vendor/mlx-video checkout 87db56a

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mlx-audio==0.6.8 misaki
uv pip install --python .venv/bin/python -e vendor/ltx-2-mlx/packages/ltx-core-mlx -e vendor/ltx-2-mlx/packages/ltx-pipelines-mlx

# LTX-2.5 weights (git-ignored)
hf download mlx-community/ltx-2.5-mlx-q8 --local-dir models/ltx-2.5-q8

brew install ffmpeg
```

## Usage (from `fever-dreams/`)

```bash
uv run python cartoon/prepare_script.py books/01-the-very-big-world-problem/story.json
cartoon/.venv/bin/python cartoon/render_voices.py books/01-the-very-big-world-problem/cartoon/script.json
cartoon/.venv/bin/python cartoon/animate_book.py books/01-the-very-big-world-problem/cartoon/script.json --frames 121
cartoon/.venv/bin/python cartoon/assemble_cartoon.py books/01-the-very-big-world-problem/cartoon/script_voiced.json
```
