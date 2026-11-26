# Phase 1: Engine, Subsystems, and Eval Harness

**Document Style:** ~70% ASD-STE100.  
**Scope:** Build and verify the style-agnostic **Maker -> Critic -> Tooling -> Retry** engine, Visual Bible, single-render set reuse, and evaluation subsystems. Video and stop-motion cadence belong to [Phase 3](03-stop-motion-book-production.md).

---

## 1. Subsystem Goals

1. Prevent visual drift across characters, recurring props, and recurring sets with a canonical Visual Bible.
2. Render each recurring scene or location once as a master set plate and reuse it across shots.
3. Automate visual defect detection and continuity checking with `O(N)` VLM checks.
4. Close the **Maker -> Critic -> Tooling -> Retry** loop so the Critic diagnoses bad renders, invokes our local repair tooling, and verifies improvement automatically.
5. Test every subsystem immediately with labeled golden fixtures—including starting from known poorly generated images and verifying the loop detects and fixes them.
6. Walk model strength backwards (`30B -> 8B -> 4B -> 2B`) to determine the minimum horsepower needed per task.

---

## 2. Core Engine Subsystems

### 2.1 Shot Entity Metadata Contract (`story.json`)

Every shot or page declares explicit entity IDs and states so Python groups shared characters, props, and sets deterministically.

```json
{
  "shot_id": "p04_s02",
  "page_number": 4,
  "cast_ids": ["marisol", "winston"],
  "prop_ids": [
    {"id": "yellow_crane_truck", "state": "boom_extended_muddy"}
  ],
  "set_id": "mvhs_parking_lot",
  "camera_angle": "wide_establishing",
  "lighting_state": "overcast_afternoon",
  "reuse_plate": "refs/sets/mvhs_parking_lot__wide_establishing.png",
  "scene_description": "..."
}
```

### 2.2 Visual Bible Subsystem (Canonical Reference Sheets)

Before rendering any story shot, generate one canonical reference sheet for every entity that appears in **2 or more shots**:

- **Characters (`refs/cast/<id>.png`):** 3-view full-body turnaround sheet conditioned on the character portrait.
- **Recurring Props (`refs/props/<prop_id>.png`):** Multi-angle canonical sheet (for example, `yellow_crane_truck.png` locking cab color, boom shape, decals, and tires).
- **Recurring Sets (`refs/sets/<set_id>__<angle>.png`):** Clean, empty set plates.

### 2.3 Single-Render Scene and Set Reuse Subsystem

When a location appears in multiple frames or shots, do not re-invent the environment on every render:

1. **Master Set Plate Generation:** Render one clean empty set plate per recurring `(set_id, camera_angle)` pair (for example, `wide`, `reverse_left`, `reverse_right`, `close_bg`).
2. **Level 1 Reuse (Multi-Reference Conditioning):** Pass `[set_plate] + [character_sheets] + [prop_sheets]` into `mflux` (`flux3-image-klein` supports 1–10 refs; `qwen-image-2.1-turbo` supports multi-ref edit). This locks architecture, wall color, prop placement, and lighting across cuts.
3. **Level 2 Reuse (Locked Camera Plate Edit):** For consecutive frames on the same framing, edit directly on the master set plate (or prior shot frame) so background objects never shift or morph.

### 2.4 `O(N)` Visual Continuity Subsystem

Do not compare every frame containing Character X against every other frame containing Character X. All-pairs comparison (`N*(N-1)/2`) is slow and causes drift because neither frame is ground truth.

Run four deterministic checks instead:

1. **Single-Frame Defect Check:** Inspect one frame alone for anatomy errors, extra limbs, missing heads, merged hands, broken accessories (such as half a pair of glasses), melted props, or stray text. Run on both the full frame and zoomed character crops when needed.
2. **Hub-and-Spoke Anchor Check:** Compare each frame against the **single canonical reference sheet** for each tagged character, prop, and set (`O(N)` calls).
3. **Adjacent-Cut Check:** For consecutive shots in the same `set_id`, compare `shot N-1` against `shot N` to catch lighting flips, mirrored props, or dropped state changes.
4. **Contact-Sheet Outlier Check (1 VLM call per entity):** Stitch up to 4–9 instances of one entity onto a numbered 2x2 or 3x3 contact sheet with Tile 0 as the canonical reference. Ask the VLM to list any tile numbers that do not match Tile 0 in design, color, accessory, or shape.

### 2.5 Closed-Loop Maker -> Critic -> Tooling -> Retry Engine (`scripts/critic_loop.py`)

Execute the repair loop in **batched rounds** (maximum 3 rounds) to prevent swap thrash on 48 GB RAM:

1. **Render / Input Batch:** Start from newly rendered keyframes (or existing defective renders during recovery tests) -> unload `mflux`.
2. **Audit Batch (Critic VLM via LM Studio):** Run Defect, Anchor, Adjacent-Cut, and Contact-Sheet checks using structured JSON output (`response_format: json_schema`). Emit per item:
   - `passed`: `true | false`
   - `score`: `1–10`
   - `severity`: `none | minor | blocking`
   - `defect_type`: `anatomy | accessory | character_drift | prop_drift | set_drift | lighting_jump | none`
   - `location`: concise description of where the defect appears (plus optional normalized `[x1, y1, x2, y2]` defect bbox)
   - `fix_action`: `none | targeted_edit | masked_inpaint | reroll_seed`
   - `fix_instruction`: concrete edit or reroll prompt for our `mflux` tooling
   - Unload VLM (`lms unload --all`).
3. **Invoke Tooling (Auto-Fix Batch):** For every failed item, invoke our local `mflux` repair tool (`targeted_edit`, `masked_inpaint` with scrubbed mask, or `reroll_seed` with canonical refs) -> generate candidate(s) -> unload `mflux`.
4. **Re-Audit & Select Best:** Critic VLM scores the new candidate(s), confirms whether the original defect is resolved without introducing new defects, and either accepts the best passing candidate or continues to the next round.
5. **Human Escalation:** After round 3, export any remaining unresolved items to a side-by-side before/round-by-round/after HTML report for manual review.

---

## 3. Subsystem Eval Harness (`evals/`)

Build and run isolated evals immediately—in seconds to minutes—before any multi-hour run.

### 3.1 Directory Structure

```text
fever-dreams/
  evals/
    vlm_client.py         # LM Studio HTTP vision client, structured JSON, model load/unload, timing
    fixtures/
      defects/            # Real failed frames + clean controls + labels.json
      anchors/            # Canonical ref sheets + matching/mismatched test shots
      contact_sheets/     # Numbered 2x2 / 3x3 grids with planted outlier tiles
      adjacent_cuts/      # Consecutive shot pairs (consistent vs. broken)
    suites/
      eval_defects.py     # Suite 1: Single-frame defect & anatomy detection
      eval_anchors.py     # Suite 2: Hub-and-spoke reference matching
      eval_contacts.py    # Suite 3: Contact-sheet outlier detection
      eval_adjacent.py    # Suite 4: Adjacent-cut & set continuity
      eval_refgen.py      # Suite 5: Reference sheet + set-reuse generation (mflux)
      eval_recovery.py    # Suite 6: Closed-loop bad-to-good recovery test (Critic -> Tooling -> Re-audit)
    run_evals.py          # Runs suites across a model ladder -> matrix.json + report.html
    results/              # Timestamped benchmark runs & visual HTML reports
```

### 3.2 Seed Fixtures Available Right Now

Populate `evals/fixtures/` immediately from real failures already on disk:

- **Half-glasses defect (user-supplied image):** Green clay character with half a glasses frame floating on the left side of his face (`defect_type: "accessory"`, `expected: FAIL`).
- **Book 1 historical failures** in `books/01-the-very-big-world-problem/art/.history/` and `.retouch/`:
  - Page 1: rocky artifact on Chester's chin (`defect_type: "anatomy"`).
  - Page 4 / 11 / 12 / 17 / 18 / cover: Winston's 3 legs, mismatched crane truck (`page_12` vs `page_10_crane_ref`), hand merging into beam, missing principal head, cover Chester drift.
- **Clean control frames:** Approved pages from `books/01-the-very-big-world-problem/art/` (`expected: PASS`).

### 3.3 Eval Suites and Acceptance Bars

| Suite | Purpose | Ground Truth | Primary Metrics | Acceptance Bar |
|---|---|---|---|---|
| **1. Single-Frame Defects** | Catch broken accessories, extra limbs, missing heads, and merged hands without false alarms. | Real failed frames + clean controls. | Defect Recall (TPR), False Positive Rate (FPR), Localization accuracy, sec/image. | Recall `>= 90%`, FPR `<= 15%`. |
| **2. Hub-and-Spoke Anchor** | Detect mismatched character appearance or prop design against a single ref sheet. | Canonical refs + matching shots + real/perturbed mismatch shots (e.g., `page_12` original crane vs `page_10_crane_ref`, cover Chester drift). | Mismatch Recall, FPR, sec/pair. | Mismatch Recall `>= 90%`, FPR `<= 10%`. |
| **3. Contact-Sheet Outlier** | Name exact bad tile IDs on a numbered 2x2 or 3x3 grid against Tile 0 (canonical ref). | Clean entity tiles + 1–2 planted defect/imposter tiles (e.g., half-glasses tile, wrong crane tile). | Exact Tile Recall, Tile Precision, sec/sheet. | Exact Tile Recall `>= 90%`, Precision `>= 85%`. |
| **4. Adjacent-Cut & Set Reuse** | Catch lighting jumps, vanished props, or mutated backgrounds between `shot N-1` and `shot N`. | Consistent same-scene pairs vs. intentionally perturbed pairs. | Break Recall, FPR, sec/pair. | Break Recall `>= 85%`, FPR `<= 15%`. |
| **5. Reference & Set-Reuse Gen** | Measure how well each `mflux` model preserves character, prop, and reused set plate across multiple shots. | 1 character sheet + 1 prop sheet + 1 master set plate + 4 test shots; scored via calibrated Suites 1–3. | First-roll pass rate, sec/image, peak RAM. | `>= 75%` first roll (`>= 95%` after 1 auto-fix). |
| **6. Closed-Loop Recovery** | Start from known poorly generated images and verify Detect -> Diagnose -> Invoke Tooling -> Improve -> Pass. | Seeded bad outputs (half-glasses, rocky chin, 3-legged Winston, mismatched crane, merged hand). | Detection rate, Recovery rate (FAIL -> PASS within 3 rounds), Score delta, Visual before/after HTML. | Detection `100%`, Recovery `>= 80%`. |

---

## 4. Walking Model Strength Backwards (Right-Sizing Ladder)

Run `evals/run_evals.py` from strongest model down to smallest model using LM Studio. Keep the smallest, fastest model that clears each suite's acceptance bar.

### 4.1 VLM Downscaling Ladder (Suites 1–4 & 6 Critic)
1. `qwen3-vl-30b-a3b-instruct-mlx` (30B MoE ceiling, ~18 GB — installed)
2. `mlx-community/Qwen3-VL-8B-Instruct-4bit` (8B mid-tier, ~5 GB)
3. `mlx-community/Qwen3-VL-4B-Instruct-4bit` (4B fast tier, ~2.8 GB)
4. `mlx-community/Qwen3-VL-2B-Instruct-4bit` (2B floor probe, ~1.5 GB)

### 4.2 Image Model Ladder (Suites 5 & 6 Tooling)
1. `qwen-image-2.1-turbo` (8 steps, high prompt & mask fidelity)
2. `flux3-image-klein` (1–10 reference images)
3. `flux2-klein-9b` / `flux2-klein-4b` (4 steps, fast edit/draft tier)

### 4.3 Two-Tier Escalation Cascade
If a fast 4B–8B VLM hits `>= 95%` defect recall at 4–6x speed but has a higher false-positive rate (for example, 25% FPR), pair two models:

1. **Tier 1 (Fast 4B–8B VLM):** Screens every frame and contact sheet quickly.
2. **Tier 2 (Heavy 30B VLM):** Confirms only the frames flagged by Tier 1 and writes surgical repair instructions.

---

## 5. Risks and Mitigations

1. **Tiny defect blind spots:** Small artifacts (such as half-glasses) disappear at low resolution. Keep native or cropped character tiles in Suites 1 and 3.
2. **Crowded contact sheets:** High tile counts degrade VLM accuracy. Cap contact sheets at 4–9 tiles (2x2 or 3x3) with high-contrast tile numbers.
3. **Broken JSON on small models:** 2B–8B VLMs can break formatting. Enforce LM Studio structured JSON schema output (`response_format: json_schema`) on every call, with fallback JSON extraction.
