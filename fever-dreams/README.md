# Little Fever Dream Books

100% local AI storybook engine for Apple Silicon (**LM Studio** + **`mflux`** multi-reference editing) that casts seniors from `../meme-yearbook/data/roster.json` into deadpan 1950s **Little Golden Book**–style hardboard picture books with **Gumball-style mixed-media** illustrations.

## Setup

```bash
uv sync
lms get https://huggingface.co/mlx-community/gemma-4-31b-it-4bit   # writer model (~18 GB)
```

The story pipeline talks to [LM Studio](https://lmstudio.ai/) via `lms chat` (no API key) and auto-loads the writer model with a 32K context. Override with `FEVER_WRITER_MODEL` / `FEVER_CONTEXT_LENGTH`. `generate_illustrations.py` runs `lms unload --all` before loading `mflux` so unified memory stays free.

## How a story gets written: the local Writers' Room

`generate_story.py` runs several role-played agents on the same local model:

1. **Showrunner** pitches `--pitches` (default 3) competing beat sheets, each page linked by BUT/THEREFORE causality (cop-outs like magic or dreams are rejected).
2. **Network Exec** scores the pitches, picks a winner, and gives notes.
3. **Narrator** writes deadpan 1950s prose with real dialogue.
4. **Reviewer panel** (Comedy, Story, and Character editors) scores the draft out of 10 with page notes; the Narrator rewrites until the average hits `--target-score` (default 8) or `--review-rounds` (default 2) run out.
5. **Art Director** storyboards cast, camera angle, scene, and lighting per page (readable text is scrubbed from scenes).

Every pitch, score, and note is saved under `writers_room` in `story.json`.

Per-book tone or secret subtext goes in a text file under `tones/` and is passed with `--tone-file` (it reaches every agent, including reviewers).

## Tinkering First: Model Consistency Bake-Off

Before rendering a full 12-page book, run a 3-shot character-consistency bake-off (side profile, low-angle full-body, and 2-character diner scene) across local multi-reference edit models and open the side-by-side HTML comparison board:

```bash
uv run python scripts/bakeoff.py --cast chester marisol --models flux2-klein-4b qwen-image-2.1-turbo
open bakeoff_out/index.html
```

Try adding `--sheets` to compare conditioning directly on yearbook headshots vs. generated 3-view full-body turnaround sheets:

```bash
uv run python scripts/bakeoff.py --cast chester marisol --models flux2-klein-4b --sheets
```

## Making a Book (Step-by-Step for Tinkering)

### 1. Cast characters & write the story with the Writers' Room
```bash
# Random diverse-medium cast of 3 seniors, 12 pages:
uv run python scripts/generate_story.py

# Specific cast, longer book, custom premise and tone:
uv run python scripts/generate_story.py --cast chester marisol winston --pages 20 --words 40-80 \
  --premise "The gang breaks a classroom globe and decides to replace it before anyone notices." \
  --tone-file tones/frenemies-subtle-puns.txt

# Faster tinkering run (~15 min instead of ~30):
uv run python scripts/generate_story.py --pitches 2 --review-rounds 1

# Preview the flippable HTML book right away (with artwork placeholders):
uv run python scripts/build_book.py --all
open index.html
```

### 2. Illustrate locally with multi-reference `mflux`
```bash
# Fast tinkering model (Apache-2.0, 4 steps):
uv run python scripts/generate_illustrations.py books/01-*/story.json --model flux2-klein-4b --sheets

# Higher prompt fidelity for weird mixed-media anatomy (Apache-2.0, 8 steps):
uv run python scripts/generate_illustrations.py books/01-*/story.json --model qwen-image-2.1-turbo --sheets

# Re-roll just one page with a new seed:
uv run python scripts/generate_illustrations.py books/01-*/story.json --only-page 4 --seed-offset 1 --overwrite
```

Supported `--model` options:
- `qwen-image-2.1-turbo` (default, 8 steps, Apache-2.0)
- `flux2-klein-4b` (fast 4 steps, Apache-2.0)
- `flux2-klein-9b` / `flux2-klein-9b-kv` (4 steps, rich textures)
- `flux3-image-klein` (1–10 reference images)
- `qwen-image-edit-2511` / `qwen-image-2.1`

### 3. Build the browser-flippable Little Golden Book + Series Bookshelf
```bash
uv run python scripts/build_book.py --all
open index.html
```

Inside any book, press **← / →** (or click pages) to flip spreads, click the bookplate line to write your name, or press **`D`** (**Director Mode**) to inspect each page's model, seed, reference images, prompt, and copy-paste re-roll command.

### One-Command Full Run
```bash
uv run python scripts/make_book.py --cast chester marisol winston --pages 12 --model flux2-klein-4b --sheets
```
