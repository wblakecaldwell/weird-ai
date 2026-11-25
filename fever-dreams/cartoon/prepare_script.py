"""Step 1 of the cartoon pipeline: use local LM Studio (Gemma 31B) to turn story.json
into a per-page voice & animation script (cartoon/script.json), then unload LM Studio.

Usage:
    uv run python cartoon/prepare_script.py books/03-the-very-big-world-problem/story.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from llm import DEFAULT_WRITER_MODEL, chat_local_json, unload_lms_models  # noqa: E402

SYSTEM_PROMPT = """You are a deadpan animation director adapting a 1950s Little Golden Book parody
with mixed-media characters (like The Amazing World of Gumball) into a voiced cartoon script.

You will receive one page at a time: its prose text, cast info, camera angle, scene description,
and lighting mood.

Return a single JSON object with:
1. "segments": an ordered list of spoken audio lines that cover the page prose:
   - Each item: {"speaker": "<speaker_id>", "text": "<exact line to speak>", "delivery": "<2-6 word acting note>"}
   - Valid speaker_id values: "narrator", plus the exact character short names provided in the prompt
     (e.g. "chester", "marisol", "winston", "teacher", "principal").
   - Rules for splitting prose:
     * Keep EVERY joke, observation, and story beat from the page text. Do NOT summarize or rewrite prose.
     * Put quoted dialogue ("...") under the character who says it (strip the quotation marks).
     * When a character speaks a quoted line followed by a bare attribution tag like `said Marisol.` or
       `cried Winston.`, omit ONLY that bare attribution tag so the narrator doesn't awkwardly repeat
       "said Marisol" right after Marisol spoke. If the attribution adds extra description
       (e.g. `said Marisol, glaring at the ruler.`), keep the descriptive part for "narrator"
       (e.g. `Marisol glared at the ruler.`).
     * All non-dialogue sentences go to "narrator".
     * Never leave an empty "text".
2. "motion_prompt": 1-2 crisp sentences for an Image-to-Video model animating THIS exact illustration.
   - Start from what is already in the image; describe subtle, readable motion (head turn, blink,
     hand gesture, object wobbling/rolling, gentle camera drift).
   - Explicitly preserve the characters' physical media (16-bit chunky pixel sprite, lumpy stop-motion
     plasticine clay, fuzzy felt puppet) and the hand-painted 1950s gouache storybook background.
   - Avoid chaotic explosions or fast morphing that would smear pixel/clay/felt textures.
3. "sfx_cue": short 3-8 word ambient sound effect hint (e.g. "classroom hum, tiny plastic crack",
   "soft felt footsteps, quiet chime").
"""


def short_speaker_id(cast_member: dict) -> str:
    return cast_member["name"].split()[0].lower()


def build_cover_entry(story: dict) -> dict:
    title = story.get("title", "Untitled")
    subtitle = story.get("subtitle", "")
    series = story.get("series_name", "A Little Fever Dream Book")
    cast_desc = ", ".join(f"{c['name']} ({c['style_label']})" for c in story.get("cast", []))
    intro = f"{series}. {title}."
    if subtitle:
        intro += f" {subtitle}."
    return {
        "page_number": 0,
        "kind": "cover",
        "image": story["cover"]["image"],
        "segments": [
            {
                "speaker": "narrator",
                "text": intro,
                "delivery": "warm cheerful 1950s filmstrip announcer",
            }
        ],
        "motion_prompt": (
            f"Gentle slow zoom-in on a 1950s children's storybook cover featuring {cast_desc}. "
            "Chester blinks awkwardly, Marisol raises her tiny ruler slightly, Winston blinks his googly eyes, "
            "preserving crisp 16-bit pixel art, lumpy stop-motion clay, fuzzy felt textures, and painted gouache background."
        ),
        "sfx_cue": "warm vintage chime, soft classroom murmur",
    }


def adapt_page(page: dict, cast_by_id: dict[str, dict], model: str, backend: str) -> dict:
    cast_lines = []
    allowed = ["narrator", "teacher", "principal"]
    for cid in page.get("cast_ids", []):
        c = cast_by_id.get(cid)
        if c:
            sid = short_speaker_id(c)
            allowed.append(sid)
            cast_lines.append(f"- {sid}: {c['name']} ({c['style_label']}, {c['archetype']})")
    if not cast_lines:
        for c in cast_by_id.values():
            sid = short_speaker_id(c)
            allowed.append(sid)
            cast_lines.append(f"- {sid}: {c['name']} ({c['style_label']})")

    user_prompt = (
        f"Page {page['page_number']}\n"
        f"Allowed speaker_id values: {sorted(set(allowed))}\n"
        f"Cast in scene:\n" + "\n".join(cast_lines) + "\n\n"
        f"Page prose:\n{page['text']}\n\n"
        f"Visual context:\n"
        f"- Camera: {page.get('camera_angle', '')}\n"
        f"- Scene: {page.get('scene_description', '')}\n"
        f"- Lighting/mood: {page.get('lighting_mood', '')}\n\n"
        "Return JSON with keys: segments, motion_prompt, sfx_cue."
    )

    valid_set = set(allowed)
    for attempt in range(1, 4):
        res = chat_local_json(
            SYSTEM_PROMPT,
            user_prompt,
            model=model,
            backend=backend,
            temperature=0.35,
            max_tokens=2048,
        )
        segs = res.get("segments") or []
        cleaned = []
        for s in segs:
            txt = str(s.get("text") or "").strip()
            spk = str(s.get("speaker") or "narrator").strip().lower()
            if spk not in valid_set:
                spk = "narrator"
            if txt:
                cleaned.append(
                    {
                        "speaker": spk,
                        "text": txt,
                        "delivery": str(s.get("delivery") or "deadpan 1950s").strip(),
                    }
                )
        motion = str(res.get("motion_prompt") or "").strip()
        sfx = str(res.get("sfx_cue") or "").strip()
        # Sanity check: ensure we didn't drop >45% of the page's words
        orig_words = len(page["text"].split())
        new_words = sum(len(s["text"].split()) for s in cleaned)
        if cleaned and motion and new_words >= max(4, int(orig_words * 0.55)):
            return {
                "page_number": page["page_number"],
                "kind": "page",
                "image": page.get("image") or f"art/page_{page['page_number']:02d}.png",
                "original_text": page["text"],
                "segments": cleaned,
                "motion_prompt": motion,
                "sfx_cue": sfx,
            }
        print(f"  [retry {attempt}] page {page['page_number']} incomplete (orig={orig_words}w, got={new_words}w)")

    raise RuntimeError(f"Failed to adapt page {page['page_number']} cleanly")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare per-page cartoon script via local LM Studio.")
    parser.add_argument("story_json", type=Path, help="Path to books/<slug>/story.json")
    parser.add_argument("--model", default=DEFAULT_WRITER_MODEL)
    parser.add_argument("--backend", choices=["cli", "http"], default="cli")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    story_path = args.story_json.resolve()
    book_dir = story_path.parent
    out_dir = book_dir / "cartoon"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "script.json"

    if out_path.exists() and not args.overwrite:
        print(f"Script already exists: {out_path} (use --overwrite to regenerate)")
        return

    story = json.loads(story_path.read_text(encoding="utf-8"))
    cast_by_id = {c["id"]: c for c in story.get("cast", [])}

    entries = [build_cover_entry(story)]
    try:
        for page in story["pages"]:
            print(f"Adapting page {page['page_number']}/{len(story['pages'])} via {args.model}...")
            entry = adapt_page(page, cast_by_id, model=args.model, backend=args.backend)
            for s in entry["segments"]:
                print(f"  [{s['speaker']}] {s['text']}")
            entries.append(entry)
    finally:
        unload_lms_models()

    script_doc = {
        "title": story["title"],
        "subtitle": story.get("subtitle", ""),
        "slug": story["slug"],
        "llm_model": args.model,
        "cast": [
            {
                "id": c["id"],
                "speaker_id": short_speaker_id(c),
                "name": c["name"],
                "style_id": c["style_id"],
                "style_label": c["style_label"],
                "archetype": c["archetype"],
            }
            for c in story.get("cast", [])
        ],
        "shots": entries,
    }
    out_path.write_text(json.dumps(script_doc, indent=2) + "\n", encoding="utf-8")
    print(f"Saved cartoon script -> {out_path}")


if __name__ == "__main__":
    main()
