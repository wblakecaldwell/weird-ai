# Phase 2: Updated Agentic Creation Pipeline

**Document Style:** ~70% ASD-STE100.  
**Scope:** Upgrade `scripts/generate_story.py` into a multi-model, adversarial Writers' Room with strict cohesiveness, continuity enforcement, and Story Trap evals. Build after Phase 1 subsystems are tested.

---

## 1. Pipeline Goals

1. Increase story diversity by running multiple Showrunners on different local LLMs.
2. Enforce story cohesiveness through adversarial dual-Executive selection requiring unanimous agreement.
3. Improve prose and comedic voice by running a full-draft Writer bake-off across different models.
4. Add a dedicated Continuity Expert to track props, physical positions, character knowledge, and setup-to-payoff chains.
5. Add an adversarial Art Director vs. Art Direction Critic loop that validates stop-motion feasibility and entity metadata before rendering starts.
6. Validate all writer and critic roles with objective Story Trap evals (Suite 7) and right-size models per role.

---

## 2. Multi-Model Writers' Room Architecture

### 2.1 Stage 1 — Multi-Showrunner Pitching

- Run **2 or more Showrunners** on different local LLMs (for example, `gemma-4-31b-it` and `qwen3-vl-30b-a3b-instruct-mlx`, plus an optional 3rd model if installed).
- Each Showrunner generates **2–3 competing beat-sheet pitches** (4–6 total pitches; avoid exceeding 8 so execs retain sharp comparison focus).
- **Roster Slicing:** Do not feed the entire Class of 2025 roster in identical order to every Showrunner (LLMs bias toward the top and bottom of long lists). Give each Showrunner a **different shuffled slice of 12–15 seniors** (excluding Book 1's cast) and require each pitch to select **4 or 5 characters** whose flaws naturally clash.
- Every beat after page 1 must connect via strict `BUT` or `THEREFORE` causality.

### 2.2 Stage 2 — Adversarial Dual-Executive Selection

Replace the single Network Exec with **two adversarial Network Execs on different models**:

- **Exec A (Cohesiveness & Causality Hawk):** Audits plot logic, `BUT`/`THEREFORE` chains, character motivation, active cast usage (no dead-weight characters), prop/setup payoffs, and whether the ending resolves page 1.
- **Exec B (Comedy, Puns & Subtext Hawk):** Audits comedic escalation, deadpan 1950s tone, subtle puns, and whether the story's secret subtext lands naturally without turning mean or obvious.

**Selection Protocol:**
1. **Blind Scoring:** Each Exec independently scores and critiques all pitches before reading the other Exec's output.
2. **Adversarial Debate (1–2 rounds):** Each Exec reads the other's top pick and attacks its weak spots.
3. **Unanimous Sign-Off Gate:** A pitch advances **only if both Execs explicitly approve it** (`approved: true`, zero blocking defects). If they disagree or either vetoes the batch, return combined veto notes to the Showrunners for a targeted re-pitch (cap at 2 re-pitch cycles).

### 2.3 Stage 3 — Story Writer Bake-Off

- Run **2 Story Writers on different models**.
- Each Writer drafts the **complete book** from the locked winning beat sheet. Never split pages between writers (splitting destroys voice consistency and callbacks).
- The Reviewer Panel evaluates both complete drafts, picks the stronger draft as the base, and optionally harvests standout jokes or puns from the runner-up as specific revision notes.

### 2.4 Stage 4 — Reviewer Panel and Continuity Expert

Four specialized reviewers critique the draft (split across models, batched by model):

1. **Comedy Editor:** Flags flat pages, missing punchlines, weak escalation, and forced puns. Marks specific funny lines to **preserve** during rewrites.
2. **Story Editor:** Checks causal links, pacing, earned climax, and anticlimactic callback ending.
3. **Character Editor:** Checks distinct voices, bio-driven actions, and subtle delivery of any active tone/subtext rules.
4. **Continuity Expert (New):** Maintains an explicit state ledger across pages—tracking every prop, location, injury/wardrobe change, character presence, and who knows what—and flags any contradiction or vanished item.

**Approval Gate (Preventing Committee-itis):**
- Do not chase a raw `9.5/10` average across 4 editors over unlimited rounds; local 30B models are noisy above `8.5` and over-editing flattens deadpan comedy.
- Use a **two-part gate**:
  1. **Hard Pass/Fail:** Story Editor and Continuity Expert must report zero unresolved plot holes and zero continuity breaks.
  2. **Calibrated Quality Bar:** Unanimous `>= 8.5–9.0` approval across the panel, capped at **3–4 rewrite rounds**, with an explicit rule to keep lines tagged as funny.

### 2.5 Stage 5 — Art Director vs. Art Direction Critic Loop

Run an adversarial storyboard loop between the **Art Director** and an **Art Direction Critic** (on a different model) until both sign off (cap at 3 rounds):

- **Art Director** converts the locked book + audio timings into per-page stop-motion shots (2–4 shots per page, 4–6 seconds each) with full entity metadata (`cast_ids`, `prop_ids` + `state`, `set_id`, `camera_angle`, `lighting_state`, `reuse_plate`, `motion_prompt`).
- **Art Direction Critic** attacks the shot list on four checks:
  1. **Stop-Motion Feasibility:** Are puppet actions simple enough for `LTX-2.5` to animate cleanly in 4–6 seconds without limb morphing?
  2. **Camera Variety:** Does the sequence vary shot sizes and angles (wide, medium, reaction close-up, cutaway)?
  3. **Set Reuse Discipline:** Do shots in the same location reuse existing `(set_id, camera_angle)` master plates where possible?
  4. **Metadata Completeness (Metadata Auditor):** Is every character, prop, and location mentioned in `scene_description` explicitly tagged in `cast_ids`, `prop_ids`, and `set_id`?

---

## 3. Model Batching Schedule (48 GB RAM)

To avoid reloading 18 GB models on every turn, group calls by model inside each stage:

1. **Load Model A** -> Showrunner A pitches -> unload.
2. **Load Model B** -> Showrunner B pitches -> Exec B blind scores -> unload.
3. **Load Model A** -> Exec A blind scores -> Exec A debate rebuttal -> unload.
4. **Load Model B** -> Exec B debate rebuttal + sign-off -> Writer B full draft -> unload.
5. **Load Model A** -> Writer A full draft -> Model A reviewer roles -> unload.
6. Continue batching reviewer and storyboard passes by model until locked.

---

## 4. Phase 2 Eval Harness — Suite 7: Story Traps

Test the Writers' Room critics objectively before running full story generation.

### 4.1 Fixture Layout

```text
fever-dreams/
  evals/
    fixtures/
      story_traps/        # 6–10 short test stories + labels.json (planted bugs vs. clean)
    suites/
      eval_story_qc.py    # Suite 7: Tests Execs, Continuity Expert, and Metadata Auditor
```

### 4.2 Planted Trap Categories
- **Prop Continuity Trap:** A character drops or destroys a prop on page 3, then uses it intact on page 7.
- **Broken Causality Trap:** Page 5 introduces a sudden unearned event ("and then") with no `BUT`/`THEREFORE` link to page 4.
- **Dead-Weight Cast Trap:** One of the 5 cast members appears on page 1 and vanishes until page 12.
- **Heavy-Handed Subtext Trap:** Characters bluntly announce they hate each other instead of keeping it oblivious and subtle.
- **Metadata Omission Trap:** `scene_description` mentions a yellow crane truck, but `prop_ids` omits `yellow_crane_truck`.
- **Clean Control Stories:** Cohesive, bug-free drafts that critics must approve (prevents critics that blindly fail everything).

### 4.3 Acceptance Bar and Model Downscaling
- **Acceptance Bar:** Trap Catch Rate `>= 90%`, Clean Story Approval Rate `>= 80%`.
- **LLM Downscaling Ladder:** Benchmark `30B–31B` (`gemma-4-31b-it`, `qwen3-30b`) -> `14B–24B` -> `8B–9B` per role.
- **Expected Routing:** Keep 30B models for creative pitching, prose drafting, and subtle comedy/subtext review; route mechanical metadata auditing (and possibly basic prop-ledger tracking) to smaller 8B–14B models if they clear `>= 90%` on Suite 7.

---

## 5. Phase 2 Build Sequence

1. Create `evals/fixtures/story_traps/` (6–10 planted-bug and clean stories) and `eval_story_qc.py` (Suite 7).
2. Benchmark local LLMs on Suite 7 to verify the Continuity Expert, Adversarial Execs, and Metadata Auditor catch `>= 90%` of planted bugs.
3. Implement batched multi-model orchestration in `scripts/generate_story.py` (or `v2` module) with structured JSON schema outputs.
4. Wire the Art Director vs. Art Direction Critic loop to output the Phase 1 shot-metadata schema.
5. Run one dry-run story generation and inspect `writers_room` logs before moving to Phase 3.
