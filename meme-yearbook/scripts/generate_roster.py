"""Generate a Class of 2025 roster of original Gumball-style meme seniors using LM Studio.

Usage:
    uv run python scripts/generate_roster.py --count 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, AuthenticationError, OpenAI

from styles import STYLES, build_image_prompt, get_style_plan

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "roster.json"

SYSTEM_PROMPT = """You are the creative director of an iconic, deeply funny high school yearbook:
"Meme Valley High School — Class of 2025".

Your job is to invent 100% ORIGINAL internet meme characters — think the wildly mixed-media,
bizarre cast of *The Amazing World of Gumball* combined with classic internet meme iconography
(like how Trollface, Wojak, or Stonks crystallize one hyper-specific relatable emotion).

STRICT ORIGINALITY RULES:
- NEVER copy, reference, or remix existing memes (NO Trollface, Pepe, Wojak, Doge, Shrek,
  Skibidi, Sigma, Gigachad, Rickroll, Smurf Cat, SpongeBob, Kermit, etc.).
- NEVER use the word "meme" inside `visual_description`. Describe pure visual geometry,
  proportions, materials, facial expression, clothing, and tiny iconic accessories.
- Each character MUST embody a specific, recognizable, funny micro-emotion or archetype
  (e.g., "quietly pretending you heard what someone said after asking 'what?' twice",
  "unearned confidence while giving wrong directions", "3 AM sudden urge to reorganize the garage").
- Make every student's silhouette, species/object type, color palette, and expression wildly
  distinct from the others. Mix weird sentient objects, oddball creatures, surreal humanoids,
  and chaotic critters.
- Names should sound hilariously plausible in a real American high school yearbook
  (e.g. "Braxton Buffering", "Chester JPEG", "Marisol Crisp", "Gavin Glorp", "Chloe Clipping").
- Senior quotes should feel genuinely 2025 high-school-funny (dry, chaotic, relatable, concise).

Output ONLY valid JSON matching the exact requested schema — no markdown fences, no commentary.
"""


def extract_json(raw: str) -> dict:
    """Robustly parse JSON even if a local LLM wraps it in markdown fences or thinking tags."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def generate_batch(
    client: OpenAI,
    model: str,
    style_slots: list[tuple[int, str]],
    existing_names: list[str],
    existing_archetypes: list[str],
) -> list[dict]:
    slot_lines = []
    for idx, style_id in style_slots:
        spec = STYLES[style_id]
        slot_lines.append(
            f"- Slot {idx}: style_id=\"{style_id}\" ({spec.label}) -> {spec.llm_guidance}"
        )

    avoid_block = ""
    if existing_names:
        avoid_block = (
            f"\nAlready used names (DO NOT repeat): {', '.join(existing_names)}"
            f"\nAlready used archetypes (DO NOT repeat): {', '.join(existing_archetypes)}"
        )

    user_prompt = f"""Invent {len(style_slots)} completely original Class of 2025 senior students,
one for each slot below, strictly matching each slot's assigned `style_id`:

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
      "archetype": "Short catchy title of the micro-emotion/meme vibe (3-6 words)",
      "micro_emotion": "One funny sentence describing the exact relatable moment this face represents",
      "visual_description": "Detailed, vivid 2-3 sentence visual description of the character's head, weird anatomy/object form, exact facial expression, color palette, and high school outfit (collar, varsity jacket, hoodie, lanyard, braces, glasses, etc.). Do NOT include backdrop or camera info and do NOT use the word 'meme'.",
      "clubs": ["Club 1", "Club 2"],
      "superlative": "Most Likely To ...",
      "quote": "Short funny authentic Class of 2025 senior quote (under 90 chars)"
    }}
  ]
}}"""

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
    return students


def main() -> None:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Generate Class of 2025 meme roster via LM Studio.")
    parser.add_argument("--count", type=int, default=20, help="Number of students to generate (default: 20)")
    parser.add_argument("--batch-size", type=int, default=5, help="Students per LM Studio call (default: 5)")
    parser.add_argument(
        "--base-url",
        default=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
        help="LM Studio OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LM_STUDIO_MODEL", "qwen3-vl-30b-a3b-instruct-mlx"),
        help="Loaded model identifier in LM Studio",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Output JSON path")
    args = parser.parse_args()

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
            "Fix: Copy your API token from LM Studio -> Developer -> Server Settings into .env:\n"
            "     LM_API_TOKEN=your_token_here\n"
            "   (or disable 'Require Authentication' in LM Studio for localhost).",
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

    style_plan = get_style_plan(args.count)
    all_students: list[dict] = []
    existing_names: list[str] = []
    existing_archetypes: list[str] = []

    for start in range(0, args.count, args.batch_size):
        end = min(start + args.batch_size, args.count)
        slots = [(i + 1, style_plan[i]) for i in range(start, end)]
        print(f"Generating students {start + 1}–{end} of {args.count} via {args.model}...")

        batch_result = None
        for attempt in range(1, 4):
            try:
                batch_result = generate_batch(
                    client, args.model, slots, existing_names, existing_archetypes
                )
                break
            except Exception as exc:
                print(f"  Attempt {attempt} failed ({exc}); retrying...")
        if batch_result is None:
            print("Failed to generate batch after 3 attempts.", file=sys.stderr)
            sys.exit(1)

        for (slot_num, expected_style), item in zip(slots, batch_result):
            style_id = expected_style
            slug = re.sub(r"[^a-z0-9]+", "-", item["name"].lower()).strip("-")
            student_id = f"{slot_num:02d}-{slug}"
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
            }
            all_students.append(record)
            existing_names.append(record["name"])
            existing_archetypes.append(record["archetype"])

    # Sort alphabetically by last name like a real high school yearbook!
    all_students.sort(key=lambda s: (s["name"].split()[-1].lower(), s["name"].lower()))

    roster = {
        "school": {
            "name": "Meme Valley High School",
            "mascot": "The Fighting Glitches",
            "class_year": 2025,
            "motto": "Ctrl + Alt + Graduate",
            "edition": "Vol. 42 — Demo Edition (20 Seniors)",
        },
        "generated_by": args.model,
        "students": all_students,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(roster, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {len(all_students)} students to {args.output}")


if __name__ == "__main__":
    main()
