# Phase 3: New Stop-Motion Book and Film Production

**Document Style:** ~70% ASD-STE100.  
**Scope:** Production specification for the next Fever Dream Story (Book 2)—a realistic full-length stop-motion cartoon and matching Little Golden Book. Execute only after Phase 1 and Phase 2 pass their eval bars.

---

## 1. Creative Brief

### 1.1 Format and Output
- **Primary Deliverable:** Full-length, realistic stop-motion cartoon (24 fps MP4 with animation delivered at 12 fps "on twos", full character voices, ambient SFX, subtitles, and zero Ken Burns still-image padding).
- **Companion Deliverable:** Browser-flippable Little Golden Book (`books/<slug>/index.html`) whose page illustrations are **screenshots extracted directly from the approved stop-motion video clips**.

### 1.2 Tone and Secret Subtext
Create a premise-neutral tone file under `tones/` based on `tones/frenemies-subtle-puns.txt` (removing Book 1's globe-specific examples):

- **User Direction:** *"I'd like some clever, but very subtle puns and sarcasm - again, slight. The secret is that none of these freaks really like each other that much, but they're too dumb to know it."*
- **Delivery Rules:**
  - Backhanded compliments delivered with total sincerity.
  - Self-serving "help" that nobody notices.
  - Polite avoidance and talking past each other while assuming the conversation went great.
  - Faintly sarcastic, deadpan 1950s narrator.
  - At most one quiet pun per page, never on consecutive pages, never stated outright, never mean-spirited.

### 1.3 Cast and Premise Rules
- **Source Roster:** Meme Valley High School Class of 2025 (`../meme-yearbook/data/roster.json`).
- **Exclusions:** Exclude Book 1 leads (`chester`, `marisol`, `winston`) so this story features a fresh group.
- **Cast Size:** 4–5 characters per pitch, selected by the Showrunners from distinct shuffled slices of 12–15 seniors.
- **Visual Contrast:** Curate slices so every candidate senior has a distinct silhouette and texture when translated into a physical stop-motion puppet.
- **Premise:** Open to Showrunner pitches (or optionally seeded with a user premise before launch, such as the wrestling-ring stop-motion concept).

### 1.4 Stop-Motion Aesthetic
- Physical miniature puppets (plasticine clay, felt, woven fabric, visible craftsmanship, subtle fingerprints).
- Practical tabletop diorama sets with shallow depth-of-field and warm practical studio lighting.
- Consistent 12 fps ("on twos") animation feel across every shot.

---

## 2. Production Workflow (Voice-First, Multi-Shot)

1. **Writers' Room (Phase 2 Pipeline):** Generate competing pitches, run adversarial dual-Exec selection, run Writer bake-off, pass Reviewer Panel + Continuity Expert gate, and lock the story text.
2. **Audio Render First (`Qwen3-TTS`):** Cast distinct voices for the 4–5 selected characters and narrator, render all audio lines, and record exact per-page and per-line durations.
3. **Shot Breakdown & Art Direction Loop:** Art Director and Art Direction Critic split each page into 2–4 shots (4–6 seconds each) matching exact audio durations, with locked entity metadata (`cast_ids`, `prop_ids`, `set_id`, `reuse_plate`).
4. **Visual Bible & Master Set Plates (`mflux`):**
   - Render 3-view stop-motion puppet turnarounds for the 4–5 chosen characters.
   - Render canonical sheets for every recurring prop (2+ shots).
   - Render master empty set plates once for every recurring `(set_id, camera_angle)` pair.
   - Visually verify the Visual Bible before rendering any scene shots.
5. **Batched Keyframe Render + Auto-Fix Loop (`mflux` + Calibrated VLM):**
   - Render shot keyframes using multi-ref conditioning on `[set_plate] + [character_sheets] + [prop_sheets]`.
   - Run Phase 1 VLM continuity audit (Single-Frame Defects, Hub-and-Spoke Anchors, Adjacent Cuts, Contact-Sheet Outliers).
   - Run up to 3 automated repair rounds; resolve any remaining edge cases in the HTML review picker.
6. **Full-Length Stop-Motion Animation + Video QC (`LTX-2.5`):**
   - Animate every shot to match its target duration (using cuts, last-frame chaining, or `keyframe_interpolation` where needed).
   - Deliver at 12 fps "on twos" inside a 24 fps stream.
   - Run batched start/mid/end VLM video QC and auto-reroll unstable clips.
7. **Book Art Extraction & Final Assembly:**
   - Extract candidate hero frames directly from the approved video clips per page and populate `books/<slug>/art/`.
   - Assemble the final MP4 cartoon (voices, ambient audio, subtitles, credits) and rebuild the HTML Little Golden Book + series bookshelf (`scripts/build_book.py --all`).

---

## 3. Mandatory Gate: 2-Page Micro-Pilot

Before launching a full 12–16 page overnight render (~45–65 clips), run an end-to-end **2-page micro-pilot** (~20–30 seconds of finished video):

1. Render 2 consecutive pages set in the same recurring location using the locked master set plate.
2. Run the full audio -> multi-shot keyframe -> VLM continuity auto-fix -> 12 fps `LTX-2.5` animation -> video QC -> book screenshot extraction chain.
3. Review the ~25-second pilot video and 2 extracted book pages to confirm puppet consistency, set stability, stop-motion feel, and pacing.
4. Adjust prompts, seeds, or model routing if needed, then greenlight the full book and film batch run.

---

## 4. Execution & Quota Policy

- **Local AI First:** All story generation, adversarial critique, TTS, Visual Bible sheets, keyframes, VLM continuity loops, `LTX-2.5` animation, frame extraction, and `ffmpeg` assembly run locally via Python scripts (zero cloud tokens during long batch runs).
- **Assistant Subagents:** Delegate routine/mechanical inspection or boilerplate tasks to lighter subagent tiers (`flash` / `flash_lite`) to conserve quota.
- **Resumability:** Every render and audit stage writes progress to `render_log.json` after each item so interrupted runs resume cleanly without repeating completed work.
