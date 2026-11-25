"""Generate a Class of 2025 roster of original Gumball-style meme seniors using LM Studio.

Usage:
    uv run python scripts/generate_roster.py --count 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, AuthenticationError, OpenAI

from styles import STYLES, build_image_prompt, get_style_plan

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "roster.json"

SYSTEM_PROMPT = """You are the creative director of an iconic, deadpan-hilarious high school yearbook:
"Meme Valley High School — Class of 2025".

Your job is to invent 100% ORIGINAL, Gumball-style mixed-media characters whose faces crystallize
specific, recognizable, funny everyday human micro-emotions.

CREATURE & VISUAL RULES:
- Invent completely original characters—never copy or remix existing memes or pop-culture IP.
- Follow each slot's assigned `style_id` AND `creature_type` strictly. Make every character in the
  batch a totally different species/object, silhouette, and color palette, with different outfits.
- `visual_description` MUST describe a HEAD-AND-SHOULDERS studio portrait ONLY: head/object/creature
  anatomy, facial features (eyes, eyebrows, mouth, expression), surface textures/colors, and
  upper-chest school clothes (collar, sweater, cardigan, varsity jacket, tie, glasses, braces, hat).
- Focus strictly on the upper torso and head; keep shirts plain/patterned with zero printed words.
- Never use the word "meme" inside `visual_description`.

CHARACTER, NAME & QUOTE RULES:
- Make each student's `archetype` and `micro_emotion` about real, funny, relatable everyday human
  awkwardness (hallways, cafeteria, group projects, driving, texting, hobbies, small talk)—NOT
  about their art style or becoming a school mascot.
- Invent funny, plausible American high school names with varied, quirky everyday surnames
  (like Turnip or Plunkett—always invent brand-new surnames instead of reusing examples). Use
  distinct first names not already in the roster.
- Follow each slot's assigned `quote_style` closely so every quote has a completely different tone
  and sentence structure (dry, punchy, specific, under 90 chars).

Output ONLY valid JSON matching the exact requested schema — no markdown fences, no commentary.
"""

CREATURE_TYPES = [
    "an unexpected sentient household or kitchen object (invent a fresh everyday item, not a repeat)",
    "an absurd animal wearing high school clothes (pick an uncommon bird, bug, fish, reptile, or mammal)",
    "a sentient food, pastry, fruit, or houseplant with an expressive face",
    "a bizarre office, classroom, music, or sports object brought to life",
    "an odd surreal folk-tale critter or weird geometric creature (not a generic human)",
    "a weird vintage appliance, tool, or hardware gadget with a face",
]

QUOTE_STYLES = [
    'start with "Do not" or "Warning:" — a cryptic one-sentence warning to incoming freshmen',
    'start with "Shoutout to" — thank a weirdly specific inanimate object or hallway fixture',
    'start with a rating like "3.5/5 stars:" — an oddly specific review of high school',
    'start with "Whoever" — a petty 4-year grudge over a tiny unresolved locker or cafeteria mystery',
    'start with "Dear" — a dramatic farewell to a favorite chair, vending machine, or parking spot',
    'start with "Pro tip:" — a cheerful piece of objectively terrible life advice',
    'start with "Effective immediately," — an overly formal corporate resignation from high school',
    'start with "Still thinking about" — a baffling oddly specific memory from 10th grade',
    'start with "Ask me about" — suspiciously proud of an absurd tiny achievement',
    'start with "To the person who" — a weirdly specific accusation with zero context',
    'start with "Never" — a fake inspirational proverb that falls apart halfway through',
    'start with "Officially" — a deadpan graduation status update about a minor habit',
]

VIBE_ARENAS = [
    "awkward hallway small talk or accidental eye contact",
    "unearned confidence while giving terrible advice or directions",
    "sleep-deprived 2 AM hyper-fixation on an unimportant task",
    "cafeteria lunch-table politics or defending a weird snack",
    "group-project delusion or last-minute presentation panic",
    "trying to act completely normal after tripping or mispronouncing a word",
    "stubbornly refusing to make two trips while carrying way too much stuff",
    "pretending to know a classmate's name after knowing them for four years",
    "unhinged pride in a bizarrely niche club or hobby",
    "quietly panicking while nodding along in a conversation you stopped following",
    "holding the door open for someone who is awkwardly too far away",
    "suspiciously intense devotion to an arbitrary school rule or routine",
]


def find_lms_bin() -> str | None:
    found = shutil.which("lms")
    if found:
        return found
    default_bin = Path.home() / ".lmstudio" / "bin" / "lms"
    return str(default_bin) if default_bin.exists() else None


def extract_json(raw: str) -> dict:
    """Robustly parse JSON even if a local LLM wraps it in markdown fences, ANSI codes, or thinking tags."""
    cleaned = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


FIRST_INITIALS = list("ABCDEFGHJKLMNPRSTUVWZ")


def generate_batch(
    client: OpenAI | None,
    lms_bin: str | None,
    model: str,
    style_slots: list[tuple[int, str]],
    existing_names: list[str],
    existing_archetypes: list[str],
    retry_note: str = "",
) -> list[dict]:
    slot_lines = []
    for pos, (idx, style_id) in enumerate(style_slots):
        spec = STYLES[style_id]
        creature_hint = CREATURE_TYPES[(idx - 1) % len(CREATURE_TYPES)]
        quote_style = QUOTE_STYLES[(idx - 1) % len(QUOTE_STYLES)]
        vibe_arena = VIBE_ARENAS[(idx - 1 + pos) % len(VIBE_ARENAS)]
        first_letter = FIRST_INITIALS[(idx * 5 + pos * 3) % len(FIRST_INITIALS)]
        last_letter = FIRST_INITIALS[(idx * 7 + pos * 11 + 4) % len(FIRST_INITIALS)]
        slot_lines.append(
            f"- Slot {idx}: style_id=\"{style_id}\" ({spec.label}) -> {spec.llm_guidance}\n"
            f"    • name_initials: First name starts with '{first_letter}', last name starts with '{last_letter}'\n"
            f"    • creature_type: {creature_hint}\n"
            f"    • vibe_arena: {vibe_arena}\n"
            f"    • quote_style: {quote_style}"
        )

    avoid_block = ""
    if existing_names:
        used_firsts = sorted({n.split()[0] for n in existing_names if n.strip()})
        avoid_block = (
            f"\nAlready used full names (DO NOT repeat): {', '.join(existing_names)}"
            f"\nAlready used first names (choose DIFFERENT first names): {', '.join(used_firsts)}"
            f"\nAlready used archetypes (DO NOT repeat): {', '.join(existing_archetypes)}"
        )
    if retry_note:
        avoid_block += f"\n\nRETRY INSTRUCTION: {retry_note}"

    user_prompt = f"""Invent {len(style_slots)} completely original Class of 2025 senior students,
one for each slot below, strictly matching each slot's assigned `style_id`, `creature_type`,
`vibe_arena`, and `quote_style`:

{chr(10).join(slot_lines)}
{avoid_block}

Return a single JSON object with this exact shape:
{{
  "students": [
    {{
      "slot": 1,
      "style_id": "pixel_16bit",
      "name": "First Last",
      "handle": "@short_handle",
      "archetype": "Short catchy 3-5 word label for their relatable vibe",
      "micro_emotion": "One funny sentence describing the exact relatable everyday moment this face represents",
      "visual_description": "Vivid 2-3 sentence HEAD-AND-SHOULDERS studio portrait description ONLY: exact creature/object head anatomy, weird proportions, facial expression, textures/colors, and collar/shoulders outfit (sweater, polo, cardigan, tie, glasses, braces, hat). Upper torso and head only, plain/patterned clothes with zero words, no backdrop/camera info, no word 'meme'.",
      "clubs": ["Club 1", "Club 2"],
      "superlative": "Most Likely To ...",
      "quote": "Short funny senior quote matching this slot's exact quote_style (under 90 chars)"
    }}
  ]
}}"""

    if lms_bin is not None:
        proc = subprocess.run(
            [
                lms_bin,
                "chat",
                model,
                "--dont-fetch-catalog",
                "-y",
                "-s",
                SYSTEM_PROMPT,
                "-p",
                user_prompt,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout).strip()
            raise RuntimeError(f"lms chat failed (exit {proc.returncode}): {err[:300]}")
        content = proc.stdout or ""
    else:
        assert client is not None
        resp = client.chat.completions.create(
            model=model,
            temperature=0.92,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = resp.choices[0].message.content or ""
    parsed = extract_json(content)
    students = parsed.get("students", [])
    if len(students) != len(style_slots):
        raise ValueError(f"Expected {len(style_slots)} students, got {len(students)}")
    cliche_re = re.compile(r"^\s*(i['’]?m not|i am not|it['’]?s not|it is not|i didn['’]?t)\b", re.I)
    existing_lower = {n.lower() for n in existing_names}
    batch_names: set[str] = set()
    for st in students:
        q = (st.get("quote") or "").strip()
        if cliche_re.search(q):
            raise ValueError(f"Rejected cliché quote format: {q!r}")
        nm = (st.get("name") or "").strip().lower()
        if not nm or nm in existing_lower or nm in batch_names:
            raise ValueError(f"Duplicate or empty student name: {st.get('name')!r}")
        batch_names.add(nm)
    return students


def main() -> None:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Generate Class of 2025 meme roster via LM Studio.")
    parser.add_argument("--count", type=int, default=20, help="Number of new students to generate (default: 20)")
    parser.add_argument("--batch-size", type=int, default=5, help="Students per LM Studio call (default: 5)")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Keep existing school, principal, and students in --output and append --count new students",
    )
    parser.add_argument(
        "--backend",
        choices=["cli", "http"],
        default="cli",
        help="How to talk to LM Studio: 'cli' uses `lms chat` with no token needed (default), 'http' uses localhost:1234",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
        help="LM Studio OpenAI-compatible base URL (used with --backend http)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LM_STUDIO_MODEL", "qwen3-vl-30b-a3b-instruct-mlx"),
        help="Model identifier in LM Studio",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Output JSON path")
    args = parser.parse_args()

    client: OpenAI | None = None
    lms_bin: str | None = None

    if args.backend == "cli":
        lms_bin = find_lms_bin()
        if not lms_bin:
            print(
                "\nError: Could not find LM Studio CLI (`lms`).\n"
                "Fix: Enable the `lms` CLI in LM Studio, or pass `--backend http`.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Using LM Studio CLI ({lms_bin}) with model '{args.model}' (no API token required)")
    else:
        api_key = os.getenv("LM_API_TOKEN") or os.getenv("LM_STUDIO_API_KEY") or "lm-studio"
        client = OpenAI(base_url=args.base_url, api_key=api_key)

        # Verify connection & loaded model up front with clear actionable errors
        try:
            models = client.models.list()
            available = [m.id for m in models.data]
            print(f"Connected to LM Studio at {args.base_url}")
            if available:
                print(f"Loaded models: {', '.join(available)}")
                if args.model not in available:
                    print(f"Note: requested '{args.model}' not in loaded list; using '{available[0]}'")
                    args.model = available[0]
        except AuthenticationError:
            print(
                "\nError: LM Studio rejected the API token.\n"
                "Fix: Run with the default `--backend cli` (no token needed), or pass LM_API_TOKEN.",
                file=sys.stderr,
            )
            sys.exit(1)
        except APIConnectionError:
            print(
                f"\nError: Could not connect to LM Studio at {args.base_url}.\n"
                "Fix: Start the local server in LM Studio (or run `lms server start` and `lms load`).",
                file=sys.stderr,
            )
            sys.exit(1)

    existing_roster: dict = {}
    all_students: list[dict] = []
    if args.append and args.output.exists():
        existing_roster = json.loads(args.output.read_text(encoding="utf-8"))
        all_students = list(existing_roster.get("students", []))
        print(f"Append mode: keeping {len(all_students)} existing student(s) from {args.output}")

    existing_names: list[str] = [s["name"] for s in all_students]
    existing_archetypes: list[str] = [s["archetype"] for s in all_students]
    existing_ids: set[str] = {s["id"] for s in all_students}
    start_slot = max((int(s.get("slot", 0)) for s in all_students), default=0)

    style_plan = get_style_plan(args.count)

    def save_checkpoint() -> None:
        sorted_students = sorted(
            all_students,
            key=lambda s: (s["name"].split()[-1].lower(), s["name"].lower()),
        )
        school = dict(
            existing_roster.get(
                "school",
                {
                    "name": "Meme Valley High School",
                    "mascot": "The Fighting Glitches",
                    "class_year": 2025,
                    "motto": "Ctrl + Alt + Graduate",
                },
            )
        )
        total_now = len(sorted_students)
        school["edition"] = (
            f"Vol. 42 — Full Class Edition ({total_now} Seniors)"
            if total_now > 20
            else f"Vol. 42 — Demo Edition ({total_now} Seniors)"
        )
        roster_out: dict = {"school": school}
        if existing_roster.get("principal"):
            roster_out["principal"] = existing_roster["principal"]
        roster_out["generated_by"] = args.model
        roster_out["students"] = sorted_students
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(roster_out, indent=2) + "\n", encoding="utf-8")

    for start in range(0, args.count, args.batch_size):
        end = min(start + args.batch_size, args.count)
        slots = [(start_slot + i + 1, style_plan[i]) for i in range(start, end)]
        print(
            f"Generating new students {start + 1}–{end} of {args.count} "
            f"(slots {slots[0][0]}–{slots[-1][0]}) via {args.model}..."
        )

        batch_result = None
        retry_note = ""
        for attempt in range(1, 6):
            try:
                batch_result = generate_batch(
                    client,
                    lms_bin,
                    args.model,
                    slots,
                    existing_names,
                    existing_archetypes,
                    retry_note=retry_note,
                )
                break
            except Exception as exc:
                print(f"  Attempt {attempt} failed ({exc}); retrying...")
                retry_note = (
                    f"Attempt {attempt} was rejected because: {exc}. "
                    f"Invent completely different brand-new names and quotes on attempt {attempt + 1}!"
                )
        if batch_result is None:
            print("Failed to generate batch after 5 attempts (prior batches saved).", file=sys.stderr)
            sys.exit(1)

        for (slot_num, expected_style), item in zip(slots, batch_result):
            style_id = expected_style
            slug = re.sub(r"[^a-z0-9]+", "-", item["name"].lower()).strip("-") or f"student-{slot_num}"
            student_id = f"{slot_num:02d}-{slug}"
            suffix = 2
            while student_id in existing_ids:
                student_id = f"{slot_num:02d}-{slug}-{suffix}"
                suffix += 1
            existing_ids.add(student_id)
            record = {
                "id": student_id,
                "slot": slot_num,
                "name": item["name"].strip(),
                "handle": item.get("handle", f"@{slug.replace('-', '_')}").strip(),
                "grade": "Senior",
                "style_id": style_id,
                "style_label": STYLES[style_id].label,
                "archetype": item["archetype"].strip(),
                "micro_emotion": item["micro_emotion"].strip(),
                "visual_description": item["visual_description"].strip(),
                "prompt": build_image_prompt(style_id, item["visual_description"]),
                "clubs": item.get("clubs", [])[:3],
                "superlative": item.get("superlative", "").strip(),
                "quote": item.get("quote", "").strip(),
                "seed": 202500 + slot_num * 17,
                "selected_variant": 0,
                "portraits": {
                    "raw": None,
                    "final": None,
                    "thumb": None,
                },
            }
            all_students.append(record)
            existing_names.append(record["name"])
            existing_archetypes.append(record["archetype"])

        save_checkpoint()
        out_disp = args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output
        print(f"  -> checkpoint saved ({len(all_students)} total students in {out_disp})")

    print(f"\nDone! Saved {len(all_students)} total students to {args.output}")


if __name__ == "__main__":
    main()
