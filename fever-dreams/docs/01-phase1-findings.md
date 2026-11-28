# Phase 1 Findings: Local VLM Critic & Auto-Repair Benchmark

Plain-English record of what we actually proved on the **449-shot, 100% human-verified benchmark** across all 121 Meme Valley High School characters, 14 sets, 14 recurring props, and both visual styles (`golden_book_mixed_media` and `stop_motion_miniature`).

> **Reproducing locally:** Ground-truth labels live in [`evals/mined_data/gold_manifest.json`](../evals/mined_data/gold_manifest.json) (232 Dev / 217 Holdout shots, split strictly by cycle so zero characters or scenes leak across splits). Run `uv run python evals/fetch_mined_data.py` to download the 676 lossless PNGs from GitHub Releases.

---

## 1. Local VLM Routing (Proven on 330 Unseen Holdout Cases)

Small Book-1 tests (10 cases) gave a false sense of confidence in tiny models (`qwen3-vl-4b` scored "100%" on 10 anchors, then collapsed to 72% Anchors / 26% Adjacent Cuts with 45 JSON parse errors on real unseen diversity).

| Task | Recommended Local Model | Blind Holdout Performance | Avg Speed | Notes |
|---|---|---|---|---|
| **Single-Frame Glitches** (clones, limbs, clipping, accessories) | `gemma-4-31b-it` | **75.0% catch rate**, 4.8% false alarms (84.9% acc) | 24.4s | Requires 2x2 quad-zoom + 3 decomposed probes |
| **Reference Matching** (`[REF, TEST]` char/prop/style) | `gemma-4-31b-it` | **87.7% catch rate**, 6.3% false alarms (89.5% acc) | 13.9s | Zero parse errors |
| **Shot-to-Shot Continuity** (lighting, prop/cast jumps) | `qwen3-vl-30b-a3b-instruct-mlx` | **97.5% catch rate**, **0.0% false alarms** (97.7% acc) | **3.7s** | Matches Gemma 31B at 3.5x speed |
| **2x2 Contact Sheets** | `qwen3-vl-8b-instruct` (or `gemma-4-31b-it`) | **96.3% tile precision**, 35.2% exact sheet | **4.8s** | Gemma hits 100% precision at 10.8s |
| **Do NOT use in production** | `qwen3-vl-4b`, `qwen3-vl-2b` | High false alarms + heavy JSON parse failures | — | Unreliable on unseen characters/styles |

---

## 2. How We Lifted Single-Frame Glitch Detection from 10% -> 75% (Zero Overfitting)

With all Book-1 hints stripped out, monolithic prompting (*"is anything wrong with this image?"*) only caught **10%** of broken frames on `gemma-4-31b-it`.

Switching to **3 general binary checks + a 2x2 zoomed quadrant grid + expected character count** lifted catch rate to **70% on Dev** and **75% on Blind Holdout** while keeping false alarms under 5%:
1. **Cast count & duplicate check** (using the deterministic `cast_ids` count from `story.json`)
2. **Limbs, hands, and spatial clipping check** (catches extra limbs, fused hands, and characters phasing through solid tables/furniture)
3. **Face & accessory check** (catches half-glasses, foreign chin/mouth blobs, and garbled text without penalizing intentional puppet/doodle/pixel mediums)

---

## 3. Critical Production Bugs Uncovered & Fixed

1. **3-Pose Turnaround Sheet Cloning Bug (Fixed in `visual_bible.py` & `critic_loop.py`):**
   - **Cause:** Feeding wide 3-view character/prop turnaround sheets directly into `qwen-image-2.1-turbo` caused the generator to literally copy-paste **2 or 3 clones** of the same character (front + walking + side views) into scenes and repairs.
   - **Fix:** Automatically crop the left single front pose (~34% width) before passing any wide sheet (`w/h >= 1.5`) to `mflux`, while still letting the Critic VLM see the full reference sheet.
2. **Multi-Character Blind Spot (Fixed in `critic_loop.py`):**
   - **Cause:** Checking only 1 character's reference in a 4-character scene let a cloned character replace a missing 4th character while keeping total headcount at 4.
   - **Fix:** Loop through **every** character and prop reference in the shot one by one (`ref_images` + `ref_labels`).
3. **Collateral Damage During Auto-Repairs (Fixed in `critic_loop.py`):**
   - **Cause:** Re-auditing only for the original bug allowed an edit to accidentally pixelate or erase an unrelated character and still score `10/10`.
   - **Fix:** Added a `BEFORE vs. AFTER` regression guard that rejects any repair candidate introducing unintended collateral damage.

---

## 4. Practical Production Takeaway

The local Critic + auto-fixer now catches and repairs ~75–98% of visual and continuity errors automatically with zero cloud tokens. For final book/video releases, pair this overnight local loop with a **30-second human skim of a single grid of the finished pages** to catch the final few subtle edge cases.
