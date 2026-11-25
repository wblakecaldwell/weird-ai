"""Step 1: Cast Meme Valley seniors and write a deadpan 1950s Little Golden Book fever dream via LM Studio.

Uses a 3-pass local "Writers' Room":
  Pass 1  Showrunner   -> causal BUT/THEREFORE beat sheet (plot logic only, no prose, no art)
  Pass 2  Narrator     -> deadpan 1950s prose + real dialogue for each locked beat
  Pass 3  Art Director -> per-page cast, camera angle, visual scene, lighting (no readable text in art)

Usage:
    uv run python scripts/generate_story.py
    uv run python scripts/generate_story.py --cast chester marisol winston --pages 12
    uv run python scripts/generate_story.py --premise "Someone microwaves fish in the teachers' lounge"
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from llm import DEFAULT_LMS_BASE_URL, DEFAULT_WRITER_MODEL, chat_local_json
from prompts import build_scene_prompt
from roster import DEFAULT_ROSTER_PATH, FEVER_DREAMS_ROOT, cast_characters, load_roster

BOOKS_DIR = FEVER_DREAMS_ROOT / "books"

VINTAGE_COVER_PALETTES = [
    {"name": "midcentury-coral", "bg": "#c9533c", "accent": "#f3d37a", "ink": "#201712"},
    {"name": "atomic-teal", "bg": "#2b6d77", "accent": "#f4d06f", "ink": "#182124"},
    {"name": "harvest-mustard", "bg": "#c89027", "accent": "#fff4d6", "ink": "#251c10"},
    {"name": "avocado-pine", "bg": "#486b4f", "accent": "#f4d984", "ink": "#192218"},
    {"name": "vintage-plum", "bg": "#7b3950", "accent": "#f3d17c", "ink": "#24151a"},
    {"name": "cobalt-diner", "bg": "#325288", "accent": "#f6d573", "ink": "#141d2e"},
]

# Each premise starts from a SMALL, relatable, concrete mistake — the engine of Gumball-style escalation.
WILD_PREMISE_SEEDS = [
    "Someone accidentally takes the wrong lunch bag from the cafeteria fridge and is too polite to give it back.",
    "A student says 'you too' when the lunch lady says 'enjoy your meal' and becomes determined to make it true.",
    "The gang breaks a classroom globe and decides to replace it before anyone notices, instead of just saying sorry.",
    "Someone waves back at a person who was actually waving at someone behind them, and refuses to admit it.",
    "A student signs up for the wrong after-school club by mistake and is too embarrassed to leave.",
    "Someone borrows a pencil on the first day of school and, four years later, still has not returned it.",
    "The class pet hamster goes missing during the one week our heroes were trusted to look after it.",
    "A student accidentally hits 'Reply All' on the school newsletter and tries to fix it in person, door to door.",
    "Someone tells a small lie about knowing how to drive the school's ride-on lawnmower.",
    "The gang locks their keys inside the principal's car while trying to do him a favor.",
]

# -------------------------------------------------------------------------------------------------
# PASS 1 — SHOWRUNNER: plot logic only
# -------------------------------------------------------------------------------------------------
SHOWRUNNER_SYSTEM = """You are the showrunner of a cult cartoon in the spirit of *The Amazing World of Gumball*,
breaking a story for a 1950s-style "Little Golden Book" parody called "A Little Fever Dream Book".

Your ONLY job right now is PLOT LOGIC. No prose, no art notes.

HOW GUMBALL COMEDY WORKS (follow this strictly):
- It starts SMALL and RELATABLE: one tiny, specific, everyday mistake or awkward moment.
- It escalates by CAUSE AND EFFECT, never by randomness. Every beat happens BECAUSE of the
  previous beat. Connect beats with "BUT" or "THEREFORE" — never "and then".
- Characters make it worse by trying to FIX it in the way their specific personality demands.
  Each character's flaw (from their bio) must be the reason a beat goes wrong.
- The absurdity is in the SCALE, not in nonsense: bureaucracy, civic overreaction, a whole town
  taking something tiny extremely seriously. The world stays internally consistent.
- BANNED cop-outs: magic, time machines, aliens, portals, dreams, "it was all imaginary", objects
  randomly transforming, sudden unexplained events. The climax must be EARNED by earlier beats.
- The ending resolves the ORIGINAL tiny problem, ideally in the most anticlimactic way possible
  (the simple fix that was available on page 1), with a callback to an earlier detail.
- PG and never mean-spirited or vulgar. Weird, not gross.

EXAMPLE of good causal escalation (different characters, for style reference only):
  1. Gerald returns a library book one day late.
  2. BUT the librarian says the 5-cent fine must be paid in the currency of the year the book was printed: 1911.
  3. THEREFORE Gerald, who never argues with authority, sets out to find a 1911 nickel.
  4. BUT the only 1911 nickel in town is in the museum, guarded by Doris, who takes rules even more seriously.
  5. THEREFORE Gerald applies for a museum job to get close to the nickel.
  6. BUT Doris promotes him to Head of Coins on his first day because he is so polite.
  7. THEREFORE he is now in charge of the nickel, and must fine HIMSELF for removing it.
  8. ...and so on, until the climax collides everyone's flaws, and the ending is that the librarian
     simply waives the fine because it was Tuesday, which was always allowed.

Output ONLY valid JSON. No markdown fences, no commentary.
"""


def build_showrunner_prompt(bios_text: str, cast_names: list[str], page_count: int, premise: str) -> str:
    climax_page = max(page_count - 2, 3)
    return f"""Break a {page_count}-page story starring ONLY these Meme Valley High School seniors:

{bios_text}

PREMISE SPARK (keep the small relatable mistake; you may adjust details):
{premise}

STRUCTURE:
- Page 1: Drop straight into a concrete scene with the tiny mistake happening. No introductions or bio recaps.
- Pages 2–{climax_page - 1}: Escalation. Each page is caused by the previous one (BUT/THEREFORE).
  Each cast member's specific personality flaw must cause at least one escalation.
- Page {climax_page}: Climax where everyone's flaws collide at the biggest scale of the story.
- Pages {climax_page + 1}–{page_count}: Anticlimactic deadpan resolution that fixes the ORIGINAL tiny problem,
  with a callback to an earlier detail.
- Every cast member ({', '.join(cast_names)}) must appear meaningfully in at least 3 beats.

Return JSON with this exact shape:
{{
  "title": "Punchy vintage Golden Book style title (3-7 words)",
  "logline": "One sentence describing the whole escalation from tiny mistake to absurd climax",
  "inciting_mistake": "The tiny relatable mistake on page 1",
  "running_gag": "One recurring joke that pays off at the end (used at most 3 times)",
  "beats": [
    {{
      "page_number": 1,
      "connector": "START",
      "characters": ["Full Name"],
      "beat": "What happens on this page and WHY, in 1-2 plain sentences"
    }},
    {{
      "page_number": 2,
      "connector": "BUT",
      "characters": ["Full Name", "Full Name"],
      "beat": "..."
    }}
  ]
}}
Exactly {page_count} beats. Every connector after page 1 must be "BUT" or "THEREFORE"."""


def validate_beats(raw: dict, page_count: int, cast_names: list[str]) -> dict:
    beats = raw.get("beats") or []
    if len(beats) != page_count:
        raise ValueError(f"Beat sheet has {len(beats)} beats, expected {page_count}")
    banned = re.compile(r"\b(time machine|magic|alien|portal|dream(ed|t)?|wizard|teleport)\w*", re.I)
    for i, b in enumerate(beats, start=1):
        text = str(b.get("beat") or "").strip()
        if not text:
            raise ValueError(f"Beat {i} is empty")
        if banned.search(text):
            raise ValueError(f"Beat {i} uses a banned cop-out ({banned.search(text).group(0)!r}): {text}")
        b["page_number"] = i
        conn = str(b.get("connector") or "").upper().strip()
        b["connector"] = "START" if i == 1 else (conn if conn in {"BUT", "THEREFORE"} else "THEREFORE")
    counts = {n: 0 for n in cast_names}
    for b in beats:
        for n in b.get("characters") or []:
            for full in cast_names:
                if full.lower() in str(n).lower() or str(n).lower() in full.lower():
                    counts[full] += 1
    starved = [n for n, c in counts.items() if c < 2]
    if starved:
        raise ValueError(f"These cast members barely appear in the beats: {', '.join(starved)}")
    return raw


# -------------------------------------------------------------------------------------------------
# PASS 2 — NARRATOR: deadpan 1950s prose + dialogue
# -------------------------------------------------------------------------------------------------
NARRATOR_SYSTEM = """You are the narrator-writer of "A Little Fever Dream Book", a parody of 1950s Little
Golden Books. The plot is already locked. Your job is the VOICE.

VOICE RULES:
- Earnest, cheerful, matter-of-fact 1950s storybook narration describing absurd events as if they
  were perfectly normal and wholesome. Short declarative sentences. Gentle exclamations
  ("Goodness!", "Oh dear.", "What a day it was turning out to be.").
- Include REAL DIALOGUE on most pages, in quotation marks, with "said Name". Characters should talk
  to each other and sound like their bios (the nervous one is nervous, the overconfident one is wrong).
- Comedy comes from deadpan understatement: the narrator calmly reports insane things. Undercut big
  moments with small mundane details.
- Each page: 2-5 sentences, at the word count given in the request. Every page must clearly follow from the previous one.
- NEVER start pages with "Here is [Name]". Never recap yearbook bios. Never repeat the same verb gag
  (e.g. "nodded") more than twice in the whole book.
- PG, never mean or vulgar.

EXAMPLE of the voice (different characters):
  "I will simply explain that I am sorry," said Gerald. Gerald did not explain that he was sorry.
  Instead, he applied for a job at the museum. It had a very nice dental plan.

Output ONLY valid JSON. No markdown fences, no commentary.
"""


def format_beats(beat_sheet: dict) -> str:
    return "\n".join(
        f"  Page {b['page_number']} [{b['connector']}] ({', '.join(b.get('characters') or [])}): {b['beat']}"
        for b in beat_sheet["beats"]
    )


def build_narrator_prompt(bios_text: str, beat_sheet: dict, page_count: int, exec_notes: str = "") -> str:
    beats_text = format_beats(beat_sheet)
    notes_block = f"\nNOTES FROM THE NETWORK (apply these):\n{exec_notes}\n" if exec_notes else ""
    return f"""CAST:
{bios_text}

LOCKED STORY: "{beat_sheet.get('title', '')}"
Logline: {beat_sheet.get('logline', '')}
Running gag (pay it off on the final pages): {beat_sheet.get('running_gag', '')}
{notes_block}
BEAT SHEET:
{beats_text}

Write the final page text for each beat, plus the front/back matter.

Return JSON with this exact shape:
{{
  "title": "Final title (may polish the working title)",
  "subtitle": "A deadpan 1950s subtitle, e.g. 'A Story About Proper Lunch Etiquette'",
  "dedication": "Short funny 1950s-style dedication",
  "back_cover_blurb": "2-3 sentence deadpan nostalgic back-cover blurb",
  "moral": "One earnest, utterly unhelpful 1950s moral that relates to THIS story",
  "pages": [
    {{"page_number": 1, "text": "Page text with narration and dialogue."}}
  ]
}}
Exactly {page_count} pages."""


def validate_prose(raw: dict, page_count: int) -> dict:
    pages = raw.get("pages") or []
    if len(pages) != page_count:
        raise ValueError(f"Prose pass returned {len(pages)} pages, expected {page_count}")
    for i, p in enumerate(pages, start=1):
        text = str(p.get("text") or "").strip()
        if len(text.split()) < 12:
            raise ValueError(f"Page {i} text too short: {text!r}")
        if re.match(r"^\s*here is\b", text, re.I):
            raise ValueError(f"Page {i} starts with banned 'Here is' bio intro")
        p["page_number"] = i
        p["text"] = text
    quoted = sum(1 for p in pages if '"' in p["text"] or "“" in p["text"])
    if quoted < page_count // 3:
        raise ValueError(f"Only {quoted} pages contain dialogue; need at least {page_count // 3}")
    return raw


# -------------------------------------------------------------------------------------------------
# PASS 3 — ART DIRECTOR: storyboard
# -------------------------------------------------------------------------------------------------
ART_DIRECTOR_SYSTEM = """You are the art director storyboarding "A Little Fever Dream Book": 1950s Little
Golden Book painted gouache backgrounds with mixed-media characters (pixel sprites, clay, felt
puppets, CGI, etc.) composited in, Gumball-style.

RULES:
- For each page, choose which cast members are VISIBLE (1-3, using exact IDs) — only those who are
  physically present in that moment of the text.
- Vary camera angles dramatically between consecutive pages (wide establishing shot, low-angle
  worm's-eye, bird's-eye overhead, over-the-shoulder, tight reaction close-up, side-profile action,
  dutch angle, through-a-window, etc.).
- scene_description is PURELY VISUAL: poses, expressions, props, 1950s environment details. Refer to
  characters by full name. Depict the single funniest moment of that page's text.
- NEVER include readable text, words, signs with writing, labels, letters, speech bubbles, or
  captions in scene_description. Signs and papers must be described as blank or pictorial.
- Vary layout among: "top-art", "full-bleed-caption", "framed-classic", "vignette".

Output ONLY valid JSON. No markdown fences, no commentary.
"""


def build_art_director_prompt(cast_dicts: list[dict], prose: dict, page_count: int) -> str:
    cast_lines = "\n".join(f"  - {c['id']}: {c['name']} ({c['style_label']}) — {c['visual_description']}" for c in cast_dicts)
    pages_text = "\n".join(f"  Page {p['page_number']}: {p['text']}" for p in prose["pages"])
    valid_ids = [c["id"] for c in cast_dicts]
    return f"""CAST (use these exact IDs):
{cast_lines}

FINAL STORY: "{prose.get('title', '')}"
{pages_text}

Storyboard the cover and every page.

Return JSON with this exact shape:
{{
  "cover": {{
    "cast_ids": {json.dumps(valid_ids)},
    "camera_angle": "Heroic slightly low-angle three-quarter composition",
    "scene_description": "Iconic cover moment capturing the premise, no text in art",
    "lighting_mood": "Warm sunlit 1950s picture-book afternoon glow"
  }},
  "pages": [
    {{
      "page_number": 1,
      "layout": "framed-classic",
      "cast_ids": ["{valid_ids[0]}"],
      "camera_angle": "...",
      "scene_description": "...",
      "lighting_mood": "..."
    }}
  ]
}}
Exactly {page_count} pages. Only use IDs from: {json.dumps(valid_ids)}"""


TEXT_IN_ART_RE = re.compile(
    r"""(\b(reads?|labeled|labelled|says?|saying|titled|inscribed|written|spelling)\b\s*[:,]?\s*['"“‘][^'"”’]+['"”’])""",
    re.I,
)


def scrub_text_from_scene(scene: str) -> str:
    """Remove phrases like `a sign that reads 'NO PARKING'` so mflux doesn't paint gibberish letters."""
    cleaned = TEXT_IN_ART_RE.sub("", scene)
    cleaned = re.sub(r"['\"“”‘’][A-Z][A-Z0-9 \-!?.']{2,}['\"“”‘’]", "", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).replace(" ,", ",").replace(" .", ".").strip()


def validate_storyboard(raw: dict, page_count: int) -> dict:
    pages = raw.get("pages") or []
    if len(pages) != page_count:
        raise ValueError(f"Storyboard returned {len(pages)} pages, expected {page_count}")
    for i, p in enumerate(pages, start=1):
        if not str(p.get("scene_description") or "").strip():
            raise ValueError(f"Storyboard page {i} missing scene_description")
    return raw


# -------------------------------------------------------------------------------------------------
# PITCH COMPETITION — Network Exec picks the strongest of N beat sheets
# -------------------------------------------------------------------------------------------------
EXEC_SYSTEM = """You are a tough, experienced animation network executive who greenlit *The Amazing World
of Gumball*. Writers pitch you competing beat sheets for "A Little Fever Dream Book" (a deadpan 1950s
Little Golden Book parody). You pick the one that will actually be FUNNY.

JUDGE ON:
1. Escalation: does a tiny relatable mistake grow, step by step, to an absurd scale (whole school,
   town, civic institutions) by the climax?
2. Causality: does every beat genuinely cause the next (BUT/THEREFORE), with no random events?
3. Character-driven: does each character's SPECIFIC flaw cause a turn in a fresh way, rather than
   just restating their bio?
4. Earned ending: does the climax collide everyone's flaws, and does the ending anticlimactically
   fix the original tiny problem with a callback?
5. Visual comedy: are there concrete, specific, funny images that would look great illustrated?

Be harsh. Most pitches are a 5. Output ONLY valid JSON. No markdown fences, no commentary.
"""


def build_exec_prompt(bios_text: str, pitches: list[dict]) -> str:
    blocks = []
    for i, p in enumerate(pitches, start=1):
        blocks.append(
            f"PITCH {i}: \"{p.get('title', '')}\"\nLogline: {p.get('logline', '')}\n"
            f"Running gag: {p.get('running_gag', '')}\n{format_beats(p)}"
        )
    return f"""CAST:
{bios_text}

{chr(10).join(blocks)}

Score every pitch and pick a winner. Then give the writers concrete notes to make the winner funnier
(specific jokes to add, beats to sharpen, ways to escalate the scale).

Return JSON:
{{
  "scores": [{{"pitch": 1, "score": 6, "why": "one sentence"}}],
  "winner": 1,
  "notes_for_writers": "3-5 specific, actionable notes for the winning pitch"
}}"""


def validate_exec(raw: dict, n: int) -> dict:
    winner = int(raw.get("winner") or 0)
    if not 1 <= winner <= n:
        raise ValueError(f"winner must be between 1 and {n}, got {winner}")
    return raw


# -------------------------------------------------------------------------------------------------
# REVIEWER PANEL — three local editors critique the draft, then the narrator revises
# -------------------------------------------------------------------------------------------------
REVIEWERS: dict[str, str] = {
    "comedy-editor": """You are the head comedy writer on *The Amazing World of Gumball*, punching up a
draft of "A Little Fever Dream Book" (a deadpan 1950s Little Golden Book parody).

Read every page and ask: WHERE IS THE JOKE? Flag pages that merely report events. Good pages end on a
deadpan punchline, an absurd understatement, or a mundane detail that undercuts something huge
(e.g. "It had a very nice dental plan."). Flag weak or missing escalation, a running gag that never
pays off, a sincere (not absurd) moral, and generic phrasing. Suggest SPECIFIC replacement jokes.""",
    "story-editor": """You are a meticulous story editor checking the logic of a comedic picture book,
"A Little Fever Dream Book". Absurd is fine; INCONSISTENT is not.

Check: does each page follow causally from the last? Are there plot holes, dropped threads, characters
who appear or vanish without reason, or a climax that isn't earned? Does the ending resolve the
ORIGINAL problem with a callback? Is any cop-out used (magic, dreams, random transformations)?
Give specific fixes.""",
    "character-editor": """You are the character and voice editor for "A Little Fever Dream Book", a deadpan
1950s Little Golden Book parody starring weird high-school seniors.

Check: does each character act from their bio in FRESH, surprising ways (not just restating their
archetype, e.g. "Chester nodded" over and over)? Do they sound distinct in dialogue? Is the narration
consistently earnest, cheerful 1950s deadpan? Flag repeated words or gags, flat dialogue, and pages
where a character has nothing to do. Suggest specific lines.""",
}

REVIEW_SUFFIX = """

Be honest and harsh: a typical first draft is a 5/10. Only give 8+ if it would genuinely make an
adult laugh out loud. Output ONLY valid JSON. No markdown fences, no commentary."""


def build_review_prompt(bios_text: str, beat_sheet: dict, prose: dict) -> str:
    pages = "\n".join(f"  Page {p['page_number']}: {p['text']}" for p in prose["pages"])
    return f"""CAST:
{bios_text}

DRAFT: "{prose.get('title', '')}" — {prose.get('subtitle', '')}
Logline: {beat_sheet.get('logline', '')}
Running gag: {beat_sheet.get('running_gag', '')}

{pages}

MORAL: {prose.get('moral', '')}

Return JSON:
{{
  "score": 5,
  "biggest_problem": "The single most important thing to fix",
  "keep": ["What is working and must be preserved"],
  "page_notes": [{{"page_number": 1, "note": "Specific problem and a specific suggested fix or line"}}]
}}"""


def validate_review(raw: dict) -> dict:
    score = float(raw.get("score"))
    if not 1 <= score <= 10:
        raise ValueError(f"score out of range: {score}")
    raw["score"] = score
    raw.setdefault("page_notes", [])
    raw.setdefault("keep", [])
    return raw


REWRITE_SYSTEM = NARRATOR_SYSTEM + """
YOU ARE NOW REVISING. A panel of editors has given notes. Apply the notes aggressively: add real
punchlines, sharpen escalation, fix logic gaps, make characters act in fresher ways, pay off the
running gag, and make the moral absurd. Preserve everything the editors said is working. You may
adjust what happens on a page if the notes require it, but keep the same number of pages and keep
the story causally connected from page to page.
"""


def build_rewrite_prompt(bios_text: str, beat_sheet: dict, prose: dict, reviews: dict[str, dict], page_count: int) -> str:
    pages = "\n".join(f"  Page {p['page_number']}: {p['text']}" for p in prose["pages"])
    note_blocks = []
    for name, r in reviews.items():
        notes = "\n".join(f"    - p{n.get('page_number', '?')}: {n.get('note', '')}" for n in r.get("page_notes", []))
        keep = "; ".join(r.get("keep", []))
        note_blocks.append(
            f"[{name}] score {r['score']}/10\n  Biggest problem: {r.get('biggest_problem', '')}\n"
            f"  Keep: {keep}\n  Page notes:\n{notes}"
        )
    return f"""CAST:
{bios_text}

STORY SPINE (logline): {beat_sheet.get('logline', '')}
Running gag: {beat_sheet.get('running_gag', '')}

CURRENT DRAFT: "{prose.get('title', '')}" — {prose.get('subtitle', '')}
{pages}
MORAL: {prose.get('moral', '')}

EDITOR NOTES:
{chr(10).join(note_blocks)}

Rewrite the whole book applying the notes. Return JSON with this exact shape:
{{
  "title": "...",
  "subtitle": "...",
  "dedication": "...",
  "back_cover_blurb": "...",
  "moral": "...",
  "pages": [{{"page_number": 1, "text": "..."}}]
}}
Exactly {page_count} pages."""


def tone_block(tone: str | None) -> str:
    """Per-book special tone / secret subtext, appended to every writers' room prompt."""
    if not tone:
        return ""
    return f"""

SPECIAL TONE FOR THIS BOOK ONLY (applies to plot, prose, and reviews):
{tone.strip()}"""


def run_review_panel(bios_text: str, beat_sheet: dict, prose: dict, args) -> dict[str, dict]:
    reviews: dict[str, dict] = {}
    tb = tone_block(getattr(args, "tone", None))
    if tb:
        tb += "\nAlso judge how well (and how SUBTLY) the draft delivers this special tone, and suggest specific lines."
    for name, persona in REVIEWERS.items():
        try:
            reviews[name] = run_pass(
                name, persona + REVIEW_SUFFIX, build_review_prompt(bios_text, beat_sheet, prose) + tb, validate_review, args, attempts=2
            )
            r = reviews[name]
            print(f"    {name}: {r['score']:.1f}/10 — {r.get('biggest_problem', '')}")
        except RuntimeError as exc:
            print(f"    {name}: skipped ({exc})")
    return reviews


# -------------------------------------------------------------------------------------------------
# Assembly
# -------------------------------------------------------------------------------------------------
def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "untitled-fever-dream"


def next_series_number() -> int:
    if not BOOKS_DIR.exists():
        return 1
    highest = 0
    for story_file in BOOKS_DIR.glob("*/story.json"):
        try:
            data = json.loads(story_file.read_text(encoding="utf-8"))
            highest = max(highest, int(data.get("series_number", 0)))
        except Exception:
            continue
    return highest + 1


def run_pass(name: str, system: str, user: str, validator, args, attempts: int = 3, model: str | None = None) -> dict:
    last_err: Exception | None = None
    retry_note = ""
    for attempt in range(1, attempts + 1):
        print(f"  [{name}] attempt {attempt}...")
        try:
            raw = chat_local_json(
                system,
                user + retry_note,
                model=model or args.model,
                backend=args.backend,
                base_url=args.base_url,
            )
            return validator(raw)
        except Exception as exc:
            last_err = exc
            print(f"    rejected: {exc}")
            retry_note = f"\n\nIMPORTANT: Your previous attempt was rejected because: {exc}. Fix that this time."
    raise RuntimeError(f"{name} failed after {attempts} attempts: {last_err}")


def merge_story(beat_sheet: dict, prose: dict, board: dict) -> dict:
    """Merge the three passes into the raw story shape consumed by validate_and_enrich_story."""
    pages: list[dict] = []
    for beat, text_page, shot in zip(beat_sheet["beats"], prose["pages"], board["pages"]):
        pages.append(
            {
                "page_number": beat["page_number"],
                "layout": shot.get("layout"),
                "cast_ids": shot.get("cast_ids"),
                "text": text_page["text"],
                "camera_angle": shot.get("camera_angle"),
                "scene_description": scrub_text_from_scene(str(shot.get("scene_description") or "")),
                "lighting_mood": shot.get("lighting_mood"),
                "beat": beat["beat"],
                "connector": beat["connector"],
            }
        )
    cover = dict(board.get("cover") or {})
    if cover.get("scene_description"):
        cover["scene_description"] = scrub_text_from_scene(str(cover["scene_description"]))
    return {
        "title": prose.get("title") or beat_sheet.get("title"),
        "subtitle": prose.get("subtitle"),
        "dedication": prose.get("dedication"),
        "back_cover_blurb": prose.get("back_cover_blurb"),
        "moral": prose.get("moral"),
        "cover": cover,
        "pages": pages,
    }


def validate_and_enrich_story(
    raw_story: dict,
    cast_dicts: list[dict],
    page_count: int,
    base_seed: int,
    series_number: int,
    palette: dict,
    model_name: str,
    premise_hint: str,
) -> dict:
    cast_by_id = {c["id"]: c for c in cast_dicts}
    valid_ids = list(cast_by_id.keys())

    def clean_cast_ids(ids: list | None, fallback_count: int = 2) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for cid in ids or []:
            if cid in cast_by_id and cid not in seen:
                seen.add(cid)
                unique.append(cid)
        if not unique:
            unique = valid_ids[:fallback_count]
        return unique[:3]

    cover_raw = raw_story.get("cover") or {}
    cover_cast_ids = clean_cast_ids(cover_raw.get("cast_ids"), fallback_count=min(3, len(valid_ids)))
    cover_cast = [cast_by_id[cid] for cid in cover_cast_ids]
    cover_scene = str(cover_raw.get("scene_description") or f"{cover_cast[0]['name']} standing proudly in a sunny 1950s town square")
    cover_angle = str(cover_raw.get("camera_angle") or "Heroic slightly low-angle three-quarter view")
    cover_mood = str(cover_raw.get("lighting_mood") or "Warm golden-hour storybook glow")

    cover = {
        "cast_ids": cover_cast_ids,
        "camera_angle": cover_angle,
        "scene_description": cover_scene,
        "lighting_mood": cover_mood,
        "prompt": build_scene_prompt(cover_cast, cover_scene, cover_angle, cover_mood, is_cover=True),
        "seed": base_seed,
        "image": None,
        "render_meta": None,
    }

    valid_layouts = {"top-art", "full-bleed-caption", "framed-classic", "vignette"}
    fallback_layouts = ["framed-classic", "top-art", "full-bleed-caption", "vignette"]

    pages: list[dict] = []
    for idx, p in enumerate(raw_story.get("pages") or [], start=1):
        p_cast_ids = clean_cast_ids(p.get("cast_ids"), fallback_count=min(2, len(valid_ids)))
        p_cast = [cast_by_id[cid] for cid in p_cast_ids]
        p_scene = str(p.get("scene_description") or "").strip()
        p_angle = str(p.get("camera_angle") or "Medium three-quarter storybook shot").strip()
        p_mood = str(p.get("lighting_mood") or "Warm mid-century storybook lighting").strip()
        layout = str(p.get("layout") or "").strip()
        if layout not in valid_layouts:
            layout = fallback_layouts[(idx - 1) % len(fallback_layouts)]
        pages.append(
            {
                "page_number": idx,
                "layout": layout,
                "cast_ids": p_cast_ids,
                "text": str(p.get("text") or "").strip(),
                "beat": p.get("beat"),
                "connector": p.get("connector"),
                "camera_angle": p_angle,
                "scene_description": p_scene,
                "lighting_mood": p_mood,
                "prompt": build_scene_prompt(p_cast, p_scene, p_angle, p_mood, is_cover=False),
                "seed": base_seed + idx * 97,
                "image": None,
                "render_meta": None,
            }
        )

    title = str(raw_story.get("title") or "The Curious Incident at Meme Valley").strip()
    return {
        "schema_version": 2,
        "series_name": "A Little Fever Dream Book",
        "series_number": series_number,
        "slug": f"{series_number:02d}-{slugify(title)}",
        "price": "25¢",
        "title": title,
        "subtitle": str(raw_story.get("subtitle") or "A Meme Valley High School Storybook").strip(),
        "dedication": str(raw_story.get("dedication") or "For everyone who forgot their locker combination.").strip(),
        "back_cover_blurb": str(raw_story.get("back_cover_blurb") or "").strip(),
        "moral": str(raw_story.get("moral") or "Always double-check which lever you are pulling.").strip(),
        "premise_seed": premise_hint,
        "cover_palette": palette,
        "generated_by_llm": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cast": cast_dicts,
        "sheets": {},
        "cover": cover,
        "pages": pages,
    }


def print_story(story: dict, beat_sheet: dict) -> None:
    wrap = lambda s, ind="    ": textwrap.fill(s, width=96, initial_indent=ind, subsequent_indent=ind)  # noqa: E731
    bar = "=" * 100
    print(f"\n{bar}\n  {story['title'].upper()}\n  {story['subtitle']}\n{bar}")
    print(wrap(f"Logline: {beat_sheet.get('logline', '')}", "  "))
    print(wrap(f"Running gag: {beat_sheet.get('running_gag', '')}", "  "))
    print(wrap(f"Dedication: {story['dedication']}", "  "))
    print()
    for p in story["pages"]:
        print(f"  — Page {p['page_number']} [{p.get('connector', '')}] —")
        print(wrap(p["text"]))
        print()
    print(wrap(f"MORAL: {story['moral']}", "  "))
    print(wrap(f"BACK COVER: {story['back_cover_blurb']}", "  "))
    print(bar)


def main() -> None:
    load_dotenv(FEVER_DREAMS_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Generate a Little Fever Dream Book story.json via a local 3-pass writers' room.")
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER_PATH, help="Path to meme-yearbook/data/roster.json")
    parser.add_argument("--cast", nargs="+", default=None, help="Specific character names/ids to star (e.g. chester marisol winston)")
    parser.add_argument("--cast-size", type=int, default=3, help="Number of main cast characters (default: 3)")
    parser.add_argument("--include-principal", action="store_true", help="Include Principal Dr. Arthur Raster in the cast")
    parser.add_argument("--pages", type=int, default=12, help="Number of story pages (default: 12)")
    parser.add_argument("--premise", type=str, default=None, help="Optional custom premise spark (best: a tiny relatable mistake)")
    parser.add_argument("--number", type=int, default=None, help="Override series number (default: auto-increment)")
    parser.add_argument("--slug", type=str, default=None, help="Override book folder slug")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for casting and page seeds")
    parser.add_argument("--backend", choices=["cli", "http"], default="cli", help="LM Studio backend (default: cli)")
    parser.add_argument("--base-url", default=DEFAULT_LMS_BASE_URL, help="LM Studio HTTP URL (with --backend http)")
    parser.add_argument("--model", default=DEFAULT_WRITER_MODEL, help="Local LM Studio writer model")
    parser.add_argument("--reviewer-model", default=None, help="Model for the reviewer panel (default: same as --model)")
    parser.add_argument("--art-model", default=None, help="Model for the storyboard pass (default: same as --model)")
    parser.add_argument("--pitches", type=int, default=3, help="Number of competing beat sheets to pitch (default: 3)")
    parser.add_argument("--review-rounds", type=int, default=2, help="Max reviewer-panel revision rounds (default: 2)")
    parser.add_argument("--target-score", type=float, default=8.0, help="Panel average that locks the draft (default: 8.0)")
    parser.add_argument("--tone", type=str, default=None, help="Special tone / secret subtext for THIS book only")
    parser.add_argument("--words", type=str, default="30-60", help="Target words per page, e.g. '40-80' (default: 30-60)")
    parser.add_argument("--tone-file", type=Path, default=None, help="Read --tone text from a file")
    args = parser.parse_args()
    if args.tone_file:
        args.tone = args.tone_file.read_text(encoding="utf-8").strip()

    rng = random.Random(args.seed)
    base_seed = args.seed if args.seed is not None else rng.randint(100000, 899999)

    _, all_chars = load_roster(args.roster)
    cast = cast_characters(
        all_chars,
        requested=args.cast,
        count=args.cast_size,
        include_principal=args.include_principal,
        seed=base_seed,
        diverse_styles=True,
    )

    series_number = args.number if args.number is not None else next_series_number()
    palette = VINTAGE_COVER_PALETTES[(series_number - 1) % len(VINTAGE_COVER_PALETTES)]
    premise_hint = args.premise or rng.choice(WILD_PREMISE_SEEDS)

    print(f"Casting for Little Fever Dream Book #{series_number:02d}:")
    for ch in cast:
        print(f"  - {ch.name} [{ch.style_label}] — {ch.archetype}")
    print(f"Premise spark: {premise_hint}")
    print(f"Writers' room: local LM Studio ({args.model} via {args.backend})")

    cast_dicts = [ch.to_story_dict() for ch in cast]
    cast_names = [c["name"] for c in cast_dicts]
    bios_text = "\n\n".join(ch.bio_for_llm() for ch in cast)

    writers_room: dict = {"writer_model": args.model, "tone": args.tone, "pitches": [], "exec": None, "review_rounds": []}
    tb = tone_block(args.tone)
    lb = f"\n\nLENGTH: Each page's text should be {args.words} words."
    sb = ""
    if args.pages >= 16:
        sb = (
            f"\n\nLONGER BOOK ({args.pages} pages): Give it room to breathe and have fun. Include a small B-plot "
            "(a side errand, rivalry, or misunderstanding involving one character) that runs alongside the main "
            "escalation and crashes into it at the climax. Include at least two big visual set-pieces "
            "(e.g. an assembly, a parade, a town meeting, a public demonstration) where the scale jumps."
        )
    if args.tone:
        print(f"Special tone: {args.tone}")
    try:
        # --- Pitch competition -------------------------------------------------------------
        print(f"\nShowrunner: pitching {args.pitches} competing beat sheet(s)")
        pitches: list[dict] = []
        for k in range(1, args.pitches + 1):
            avoid = ""
            if pitches:
                prior = "\n".join(f"  - {p.get('logline', '')}" for p in pitches)
                avoid = f"\n\nThis is pitch #{k}. It must take a DIFFERENT comedic direction from these earlier pitches:\n{prior}"
            try:
                pitch = run_pass(
                    f"pitch-{k}",
                    SHOWRUNNER_SYSTEM,
                    build_showrunner_prompt(bios_text, cast_names, args.pages, premise_hint) + sb + tb + avoid,
                    lambda r: validate_beats(r, args.pages, cast_names),
                    args,
                )
                pitches.append(pitch)
                print(f"    Pitch {k}: \"{pitch.get('title', '')}\" — {pitch.get('logline', '')}")
            except RuntimeError as exc:
                print(f"    Pitch {k} failed: {exc}")
        if not pitches:
            raise RuntimeError("No valid pitches were produced.")
        writers_room["pitches"] = [{"title": p.get("title"), "logline": p.get("logline")} for p in pitches]

        exec_notes = ""
        beat_sheet = pitches[0]
        if len(pitches) > 1:
            print("\nNetwork Exec: picking the winning pitch")
            verdict = run_pass("exec", EXEC_SYSTEM, build_exec_prompt(bios_text, pitches) + tb, lambda r: validate_exec(r, len(pitches)), args)
            for s in verdict.get("scores", []):
                print(f"    Pitch {s.get('pitch')}: {s.get('score')}/10 — {s.get('why', '')}")
            beat_sheet = pitches[int(verdict["winner"]) - 1]
            exec_notes = str(verdict.get("notes_for_writers") or "")
            writers_room["exec"] = verdict
            print(f"    WINNER: Pitch {verdict['winner']} — \"{beat_sheet.get('title', '')}\"")
            print(textwrap.fill(f"Notes: {exec_notes}", width=96, initial_indent="    ", subsequent_indent="    "))

        print("\nWinning beat sheet:")
        for b in beat_sheet["beats"]:
            print(f"    {b['page_number']:>2}. [{b['connector']}] {b['beat']}")

        # --- First draft -------------------------------------------------------------------
        print("\nNarrator: deadpan 1950s prose + dialogue")
        prose = run_pass(
            "narrator",
            NARRATOR_SYSTEM,
            build_narrator_prompt(bios_text, beat_sheet, args.pages, exec_notes) + lb + tb,
            lambda r: validate_prose(r, args.pages),
            args,
        )

        # --- Reviewer panel + revision loop ------------------------------------------------
        reviewer_model = args.reviewer_model or args.model
        for round_num in range(1, args.review_rounds + 1):
            print(f"\nReviewer panel — round {round_num} ({reviewer_model})")
            rv_args = argparse.Namespace(**{**vars(args), "model": reviewer_model})
            reviews = run_review_panel(bios_text, beat_sheet, prose, rv_args)
            if not reviews:
                break
            avg = sum(r["score"] for r in reviews.values()) / len(reviews)
            writers_room["review_rounds"].append({"round": round_num, "avg_score": round(avg, 2), "reviews": reviews})
            print(f"    Panel average: {avg:.1f}/10 (target {args.target_score})")
            if avg >= args.target_score:
                print("    Target reached, locking the draft.")
                break
            print("  Narrator: revising with panel notes")
            try:
                prose = run_pass(
                    f"rewrite-{round_num}",
                    REWRITE_SYSTEM,
                    build_rewrite_prompt(bios_text, beat_sheet, prose, reviews, args.pages) + lb + tb,
                    lambda r: validate_prose(r, args.pages),
                    args,
                )
            except RuntimeError as exc:
                print(f"    Rewrite failed, keeping previous draft: {exc}")
                break

        # --- Storyboard --------------------------------------------------------------------
        print("\nArt Director: storyboard")
        board = run_pass(
            "art-director",
            ART_DIRECTOR_SYSTEM,
            build_art_director_prompt(cast_dicts, prose, args.pages),
            lambda r: validate_storyboard(r, args.pages),
            args,
            model=args.art_model or args.model,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    story = validate_and_enrich_story(
        merge_story(beat_sheet, prose, board),
        cast_dicts=cast_dicts,
        page_count=args.pages,
        base_seed=base_seed,
        series_number=series_number,
        palette=palette,
        model_name=args.model,
        premise_hint=premise_hint,
    )
    story["logline"] = beat_sheet.get("logline", "")
    story["running_gag"] = beat_sheet.get("running_gag", "")
    story["inciting_mistake"] = beat_sheet.get("inciting_mistake", "")
    story["writers_room"] = writers_room

    if args.slug:
        story["slug"] = args.slug

    book_dir = BOOKS_DIR / story["slug"]
    book_dir.mkdir(parents=True, exist_ok=True)
    out_path = book_dir / "story.json"
    out_path.write_text(json.dumps(story, indent=2) + "\n", encoding="utf-8")

    print_story(story, beat_sheet)
    print(f"\nSaved to: {out_path}")
    print("\nNext steps:")
    print(f"  uv run python scripts/generate_illustrations.py {out_path.relative_to(FEVER_DREAMS_ROOT)} --model flux2-klein-4b")
    print("  uv run python scripts/build_book.py --all")


if __name__ == "__main__":
    main()
