"""Continuous, balanced, crash-safe local test-data miner for Fever Dreams Critic evals.

Runs 100% locally via mflux (qwen-image-2.1-turbo) with zero cloud tokens.
Cycles through all 121 Meme Valley High School characters, 14 sets, 14 recurring props,
and both world styles (1950s Little Golden Book mixed-media + tactile stop-motion miniature),
generating an evenly balanced mix of all eval data types in every single cycle:

  - Canonical Visual Bible refs (character turnarounds, prop sheets, master empty set plates)
  - Clean set-plate reuse sequences (Suites 2, 4, 5 PASS controls)
  - Hard multi-character & hand/prop interaction stress shots (Suite 1 natural defect mining)
  - Tight face/accessory close-up stress shots (Suite 1 fine-grained defect mining)
  - Crisp subtle character drift via dropped character ref (Suites 2, 3)
  - Crisp subtle prop drift via dropped prop ref (Suites 2, 3, 4)
  - Realistic lighting/time-of-day cut breaks (Suite 4)
  - Medium/style drift stress shots without medium-lock (Suites 2, 3)
  - Auto-stitched 2x2 numbered contact sheets per cycle (Suite 3)

Every image gets a sidecar `.json` with full `generation`, `scene`, and `triage` metadata
for deterministic eval assembly and a future click-through human labeling webpage.

Usage:
    # Run indefinitely until stopped (default)
    uv run python evals/mine_test_data.py

    # Quick smoke test (1 mini-cycle with 3 shots)
    uv run python evals/mine_test_data.py --smoke --max-cycles 1
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

FEVER_DREAMS_ROOT = Path(__file__).resolve().parent.parent
MEME_YEARBOOK_ROOT = FEVER_DREAMS_ROOT.parent / "meme-yearbook"
ROSTER_PATH = MEME_YEARBOOK_ROOT / "data" / "roster.json"
SCRIPTS_DIR = FEVER_DREAMS_ROOT / "scripts"
EVALS_DIR = FEVER_DREAMS_ROOT / "evals"

for p in (SCRIPTS_DIR, EVALS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from generate_illustrations import load_edit_model, render_edit, save_clean  # noqa: E402
from llm import unload_lms_models  # noqa: E402
from prompts import GOLDEN_BOOK_WORLD_STYLE, build_turnaround_prompt, medium_lock_phrase  # noqa: E402
from seed_fixtures import make_contact_sheet  # noqa: E402
from visual_bible import (  # noqa: E402
    build_prop_sheet_prompt,
    build_set_plate_prompt,
    build_set_reuse_shot_prompt,
)

STOP_MOTION_WORLD_STYLE = (
    "Realistic handcrafted stop-motion animation still photograph on a miniature practical tabletop set, "
    "tangible physical textures (sculpted plasticine clay with subtle thumbprints, felt fabric fibers, "
    "balsa wood, painted cardboard, miniature brass hardware), shallow depth of field, warm practical "
    "miniature studio key lighting with soft shadows, Laika / Aardman / Wes Anderson tactile miniature "
    "cinematography, crisp 35mm macro lens look, zero text, zero speech bubbles, zero watermarks"
)

WORLD_STYLES: dict[str, str] = {
    "golden_book_mixed_media": GOLDEN_BOOK_WORLD_STYLE,
    "stop_motion_miniature": STOP_MOTION_WORLD_STYLE,
}

# 14 distinct Meme Valley High School / town sets
SETS_LIBRARY: list[dict[str, str]] = [
    {
        "id": "mvhs_cafeteria",
        "name": "Meme Valley High School Cafeteria",
        "description": "Retro 1950s high school lunchroom with checkered linoleum floor, long turquoise laminate lunch tables, chrome serving counter, wall clock, and tall sunlit windows",
        "lighting_day": "bright warm afternoon sunlight streaming through tall cafeteria windows",
        "lighting_break": "dim flickering blue-green fluorescent midnight lighting with dark night windows outside",
    },
    {
        "id": "mvhs_chemistry_lab",
        "name": "High School Chemistry Lab",
        "description": "Black slate lab benches, glass beakers and bubbling flasks, wooden stools, green chalkboard covered in blank geometric diagrams (no readable text), brass sink faucets, and safety shower",
        "lighting_day": "crisp warm morning daylight from side windows with bright overhead lamps",
        "lighting_break": "eerie emerald-green glowing darkroom lighting at night with deep dramatic shadows",
    },
    {
        "id": "mvhs_woodshop",
        "name": "High School Woodshop",
        "description": "Sawdust-dusted oak workbenches, pegboard wall hung with hand saws and wooden mallets, stacked pine planks, vintage green bandsaw, and hanging industrial enamel pendant lamps",
        "lighting_day": "warm golden late-afternoon sunbeams through dusty workshop windows",
        "lighting_break": "cold overcast twilight blue shadows with a single harsh red warning lamp",
    },
    {
        "id": "mvhs_gym_bleachers",
        "name": "High School Gymnasium & Bleachers",
        "description": "Polished honey-maple basketball court floor with painted court lines, folded-out wooden bleachers, vintage scoreboard shapes, hanging pennants (no text), and high clerestory windows",
        "lighting_day": "bright warm gymnasium daylight mixed with high-bay arena lamps",
        "lighting_break": "dramatic dark midnight spotlight beaming onto the center floor with deep indigo shadows",
    },
    {
        "id": "mvhs_locker_hallway",
        "name": "Main Locker Hallway",
        "description": "Long perspective corridor lined with sage-green metal lockers, terrazzo speckled floor, round hallway wall clock, trophy display alcove, and water fountain",
        "lighting_day": "even cheerful warm school-day sunlight through hallway transoms",
        "lighting_break": "stormy purple-blue thunderstorm twilight with stark lightning flashes from windows",
    },
    {
        "id": "mvhs_library_stacks",
        "name": "School Library Stacks",
        "description": "Tall walnut bookshelves packed with colorful clothbound books, rolling wooden library ladder, green banker's lamps on oak reading tables, and arched windows",
        "lighting_day": "cozy sunlit afternoon warmth with dusty golden light shafts",
        "lighting_break": "pitch-dark midnight library lit only by cold bluish moonlight through arched windows",
    },
    {
        "id": "mvhs_auditorium_stage",
        "name": "School Auditorium Stage",
        "description": "Heavy burgundy velvet stage curtains, polished dark-wood stage planks, brass footlights, cardboard painted cloud backdrops suspended on ropes, and empty theater seats beyond",
        "lighting_day": "warm theatrical golden stage wash and bright footlights",
        "lighting_break": "stark cold cyan backstage worklight in an otherwise dark theater",
    },
    {
        "id": "mvhs_parking_lot",
        "name": "High School Parking Lot",
        "description": "Sun-baked asphalt lot with painted yellow parking lines, brick school facade with flagpole in background, chain-link fence, and autumn maple trees",
        "lighting_day": "crisp clear warm afternoon sunshine with long soft shadows",
        "lighting_break": "heavy rainy midnight darkness with wet asphalt reflecting orange sodium streetlights",
    },
    {
        "id": "mvhs_rooftop_greenhouse",
        "name": "Rooftop Botany Greenhouse",
        "description": "White-painted iron glasshouse frame, terracotta pots overflowing with oversized ferns and venus flytraps, slatted cedar potting benches, and watering cans",
        "lighting_day": "lush bright dappled midday sunlight through glass roof panels",
        "lighting_break": "deep blue-violet moonlit night with glowing purple grow-lamps",
    },
    {
        "id": "mvhs_band_room",
        "name": "High School Band Room",
        "description": "Tiered semicircle risers, black metal music stands, snare drums, acoustic wall tiles, instrument storage cubbies, and conductor's wooden podium",
        "lighting_day": "warm bright afternoon rehearsal room lighting",
        "lighting_break": "moody dark amber-and-indigo late-night jazz club lighting",
    },
    {
        "id": "mvhs_auto_shop",
        "name": "High School Auto Shop Garage",
        "description": "Concrete garage bay with rolled-up steel door, hydraulic red car lift, rolling tool chests, stacked rubber tires, and hanging drop lights",
        "lighting_day": "bright open-door daytime sunlight flooding into the garage bay",
        "lighting_break": "dark midnight garage lit only by harsh cold white welding-arc glare",
    },
    {
        "id": "mvhs_detention_room",
        "name": "After-School Detention Classroom",
        "description": "Rows of vintage wood-and-steel school desks, creaky wall clock, pull-down world map (no readable text), teacher's oak desk with stack of folders, and venetian blinds",
        "lighting_day": "warm golden 4 PM late-afternoon sun stripes through venetian blinds",
        "lighting_break": "cold dreary dark-blue midnight rainstorm outside unlit windows",
    },
    {
        "id": "mvhs_trophy_foyer",
        "name": "School Entrance Trophy Foyer",
        "description": "Checkered marble lobby floor, tall glass-and-oak trophy cases filled with gleaming cups and ribbons, double glass entrance doors, and potted ficus trees",
        "lighting_day": "bright welcoming morning sunshine through glass entrance doors",
        "lighting_break": "spooky dim blue after-hours security light casting stark shadows",
    },
    {
        "id": "mvhs_football_sideline",
        "name": "High School Football Field Sideline",
        "description": "Lush green grass turf with white yard stripes, aluminum team bench, bright orange Gatorade cooler, metal stadium bleachers, and tall goalposts in the distance",
        "lighting_day": "golden autumn afternoon sunlight under puffy clouds",
        "lighting_break": "dark night stadium under blinding cold white floodlights and black sky",
    },
]

# 14 distinct tactile recurring props
PROPS_LIBRARY: list[dict[str, str]] = [
    {
        "id": "av_projector_cart",
        "name": "vintage teal rolling AV film projector cart",
        "description": "A tall teal-painted steel rolling school AV cart on four black rubber casters holding a bulky vintage 16mm metal film projector with two chrome reels and coiled black power cord",
    },
    {
        "id": "giant_brass_trophy",
        "name": "oversized ornate golden bowling trophy",
        "description": "A gleaming three-tiered golden brass championship trophy with two curved handles, marble base, and a tiny golden bowling pin figurine on top",
    },
    {
        "id": "science_fair_volcano",
        "name": "papier-mache science fair volcano model",
        "description": "A lumpy brown-and-terracotta painted papier-mache tabletop volcano mounted on a green plywood board with tiny plastic trees and bright orange foam lava bubbling at the crater",
    },
    {
        "id": "vintage_brass_tuba",
        "name": "giant gleaming brass marching tuba",
        "description": "An oversized polished gold-brass sousaphone tuba with a wide flared bell, coiled tubing, three pearl-tipped valves, and a tiny red tassel hanging from the mouthpiece",
    },
    {
        "id": "red_metal_toolbox",
        "name": "chunky cherry-red steel mechanic toolbox",
        "description": "A classic boxy cherry-red painted steel toolbox with two silver flip latches, black handle, and a hinged lid slightly ajar showing wrenches and a yellow screwdriver",
    },
    {
        "id": "cafeteria_tray_cart",
        "name": "chrome cafeteria rolling tray rack",
        "description": "A gleaming stainless-steel wheeled cafeteria cart stacked with bright mint-green plastic lunch trays, a wobbling tower of orange gelatin cubes, and cartons",
    },
    {
        "id": "retro_oscilloscope",
        "name": "boxy sage-green vintage oscilloscope",
        "description": "A chunky 1960s sage-green steel laboratory oscilloscope with a glowing round bright-green sine-wave CRT screen, four bakelite dials, and tangled red-and-black alligator clip wires",
    },
    {
        "id": "yellow_bullhorn_megaphone",
        "name": "chunky canary-yellow handheld bullhorn megaphone",
        "description": "A retro canary-yellow and white cone-shaped electric megaphone with a black pistol grip, coiled microphone cord, and red trigger switch",
    },
    {
        "id": "confetti_cannon_cart",
        "name": "wheeled striped carnival confetti cannon",
        "description": "A miniature red-and-cream striped wooden cannon mounted on two spoked brass wheels, packed with colorful paper confetti bursting slightly from the muzzle",
    },
    {
        "id": "squeaky_mop_bucket",
        "name": "bright yellow janitor mop bucket with wringer",
        "description": "A chunky bright-yellow wheeled janitor bucket with a red steel side-press wringer and a fluffy oversized cotton string mop sticking up at an angle",
    },
    {
        "id": "folding_ladder_easel",
        "name": "paint-splattered wooden A-frame step ladder",
        "description": "A sturdy vintage pine A-frame wooden step ladder covered in cheerful drips of teal, mustard, and coral paint, with a metal paint bucket hooked on the top tray",
    },
    {
        "id": "chrome_vending_machine",
        "name": "retro coral-red glass-front snack vending machine",
        "description": "A boxy 1950s coral-red and chrome snack vending machine with a glass window displaying colorful spiral coils of snack bags and a shiny metal push-flap dispenser",
    },
    {
        "id": "wooden_podium_mic",
        "name": "ornate mahogany speaker podium with vintage chrome microphone",
        "description": "A heavy carved mahogany school assembly lectern with a slanted wooden top, reading lamp, and a classic 1950s ribbed chrome Elvis-style microphone on a gooseneck stand",
    },
    {
        "id": "dented_yellow_wheelbarrow",
        "name": "mustard-yellow steel garden wheelbarrow",
        "description": "A sturdy mustard-yellow painted metal wheelbarrow with one chunky black rubber front wheel, wooden handles, and a pile of mismatched rubber balls inside",
    },
]

# Action templates to keep scenes varied, expressive, and physically engaging
CLEAN_ACTIONS_1: list[str] = [
    "{c1} proudly demonstrates the {prop} while {c2} leans in skeptically with arms crossed, both standing firmly on the floor",
    "{c1} carefully pushes the {prop} into the center of the room as {c2} points excitedly toward it with a deadpan expression",
    "{c1} and {c2} stand on opposite sides of the {prop}, trying to measure it with exaggerated seriousness",
    "{c1} holds up one finger lecturing about the {prop} while {c2} stares blankly past them",
]

CLEAN_ACTIONS_2: list[str] = [
    "Moments later, {c1} awkwardly tries to adjust the {prop} with both hands while {c2} steps back looking unimpressed",
    "Next beat: {c2} takes over inspecting the {prop} up close while {c1} gestures defensively with open palms",
    "Moments later, the {prop} tilts slightly as {c1} and {c2} both reach out trying to steady it at the same time",
    "Next beat: {c1} gives a triumphant thumbs-up beside the {prop} while {c2} rubs their forehead in quiet disbelief",
]

STRESS_CROWD_ACTIONS: list[str] = [
    "{c1}, {c2}, {c3}, and {c4} all huddle tightly around the {prop}, with multiple hands reaching in to grab levers, tools, and edges simultaneously in chaotic overlapping poses",
    "{c1} and {c2} try to lift the heavy {prop} together by its sides while {c3} balances on one foot directing traffic and {c4} hands over a wrench from behind",
    "All four characters ({c1}, {c2}, {c3}, {c4}) crowd side-by-side passing tangled cords, clipboards, and tools across the {prop}, hands overlapping and arms gesturing wildly",
    "{c1} sits crouched adjusting the bottom of the {prop} while {c2}, {c3}, and {c4} lean directly over them holding flashlights, rulers, and spare parts",
]

ACCESSORY_CLOSEUP_ACTIONS: list[str] = [
    "Tight medium close-up of {c1} and {c2} leaning face-to-face squinting intently at a tiny gear held right between their noses, showing clear facial features, eyewear, collars, and fingers",
    "Close-up reaction shot of {c1} nervously adjusting their collar/glasses while holding a tiny sandwich near their chin, as {c2} peers skeptically over their shoulder",
    "Tight two-shot from chest up: {c1} whispers a terrible plan into {c2}'s ear while gesturing with both hands close to their faces",
    "Close-up portrait-in-scene of {c1} beaming with absurd overconfidence while holding a small magnifying glass up near one eye",
]

# Graceful shutdown flag
_STOP_REQUESTED = False


def _handle_signal(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"\n[miner] Received signal {signum}; will finish current image and exit cleanly...", flush=True)


def _clear_mlx() -> None:
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def _rel_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(FEVER_DREAMS_ROOT.resolve()).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(MEME_YEARBOOK_ROOT.parent.resolve()).as_posix()
        except ValueError:
            return path.name


def load_full_roster() -> list[dict[str, Any]]:
    data = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))
    chars = [data["principal"]] + list(data["students"])
    out = []
    for c in chars:
        portrait = MEME_YEARBOOK_ROOT / c["portraits"]["final"]
        if portrait.exists():
            rec = dict(c)
            rec["portrait_path"] = str(portrait)
            out.append(rec)
    return out


def build_stratified_character_order(chars: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    """Order characters so Book 1 trio goes last and all 8 visual styles are evenly interleaved."""
    book1_ids = {"01-chester-antialiased", "03-marisol-brittle", "06-winston-crumb"}
    fresh = [c for c in chars if c["id"] not in book1_ids]
    book1 = [c for c in chars if c["id"] in book1_ids]

    by_style: dict[str, list[dict[str, Any]]] = {}
    for c in fresh:
        by_style.setdefault(c.get("style_id", "other"), []).append(c)
    for lst in by_style.values():
        rng.shuffle(lst)

    style_keys = sorted(by_style.keys())
    rng.shuffle(style_keys)

    interleaved: list[dict[str, Any]] = []
    while any(by_style.values()):
        for sk in style_keys:
            if by_style[sk]:
                interleaved.append(by_style[sk].pop())

    rng.shuffle(book1)
    return interleaved + book1


def _blank_canvas(path: Path, width: int, height: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        Image.new("RGB", (width, height), (238, 230, 214)).save(path, format="PNG")
    return path


def write_sidecar(img_path: Path, meta: dict[str, Any]) -> Path:
    sidecar = img_path.with_suffix(".json")
    sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return sidecar


def update_status(out_dir: Path, stats: dict[str, Any]) -> None:
    stats["updated_at"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "status.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def rebuild_index(out_dir: Path) -> dict[str, Any]:
    """Rebuild lightweight index.json from all sidecar JSON files on disk."""
    shots_dir = out_dir / "shots"
    sheets_dir = out_dir / "contact_sheets"
    shots = []
    by_category: dict[str, int] = {}
    by_style: dict[str, int] = {}
    unique_chars: set[str] = set()
    unique_props: set[str] = set()
    unique_sets: set[str] = set()

    if shots_dir.exists():
        for jf in sorted(shots_dir.glob("*.json")):
            try:
                rec = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            shots.append(
                {
                    "id": rec.get("id"),
                    "image": rec.get("image"),
                    "sidecar": jf.name,
                    "category": rec.get("scene", {}).get("category"),
                    "style": rec.get("scene", {}).get("style"),
                    "shot_group": rec.get("scene", {}).get("shot_group"),
                    "cast_ids": rec.get("scene", {}).get("cast_ids", []),
                    "prop_ids": rec.get("scene", {}).get("prop_ids", []),
                    "set_id": rec.get("scene", {}).get("set_id"),
                    "stress_tags": rec.get("scene", {}).get("stress_tags", []),
                    "human_label": rec.get("triage", {}).get("human_label"),
                }
            )
            cat = str(rec.get("scene", {}).get("category") or "unknown")
            by_category[cat] = by_category.get(cat, 0) + 1
            st = str(rec.get("scene", {}).get("style") or "unknown")
            by_style[st] = by_style.get(st, 0) + 1
            for cid in rec.get("scene", {}).get("cast_ids") or []:
                unique_chars.add(cid)
            for pid in rec.get("scene", {}).get("prop_ids") or []:
                unique_props.add(pid)
            if rec.get("scene", {}).get("set_id"):
                unique_sets.add(rec["scene"]["set_id"])

    sheets = []
    if sheets_dir.exists():
        for jf in sorted(sheets_dir.glob("*.json")):
            try:
                sheets.append(json.loads(jf.read_text(encoding="utf-8")))
            except Exception:
                continue

    index = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_shots": len(shots),
        "total_contact_sheets": len(sheets),
        "unique_characters": len(unique_chars),
        "unique_props": len(unique_props),
        "unique_sets": len(unique_sets),
        "by_category": by_category,
        "by_style": by_style,
        "shots": shots,
        "contact_sheets": sheets,
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def ensure_cast_turnaround(
    model: Any,
    mspec: Any,
    char: dict[str, Any],
    style_key: str,
    refs_dir: Path,
    seed: int,
) -> Path:
    out_p = refs_dir / "cast" / f"cast_{char['id']}__{style_key}.png"
    if out_p.exists() and out_p.with_suffix(".json").exists():
        return out_p

    out_p.parent.mkdir(parents=True, exist_ok=True)
    portrait = Path(char["portrait_path"])
    if style_key == "stop_motion_miniature":
        prompt = (
            f"Physical stop-motion puppet turnaround reference sheet of {char['name']} translated directly "
            f"from Image 1 into a handmade stop-motion animation puppet (sculpted plasticine clay, felt, "
            f"miniature stitched clothes, wire armature, visible handcrafted texture). Keep the exact face, "
            f"colors, silhouette, and outfit from Image 1. Show THREE full-body standing views side-by-side "
            f"on a neutral warm studio tabletop backdrop: (left) front view, (center) three-quarter walking view, "
            f"(right) side profile view. ({char['visual_description']}). Even studio lighting, zero text, zero labels."
        )
    else:
        prompt = build_turnaround_prompt(char)

    t0 = time.perf_counter()
    gen = render_edit(
        model,
        mspec,
        prompt=prompt,
        image_paths=[portrait],
        seed=seed,
        width=1024,
        height=576,
    )
    save_clean(gen, out_p)
    dt = round(time.perf_counter() - t0, 2)
    write_sidecar(
        out_p,
        {
            "id": out_p.stem,
            "kind": "canonical_cast_ref",
            "image": out_p.name,
            "char_id": char["id"],
            "char_name": char["name"],
            "char_style_id": char.get("style_id"),
            "world_style": style_key,
            "source_portrait": _rel_repo(portrait),
            "generation": {
                "model": mspec.key,
                "seed": seed,
                "steps": mspec.default_steps,
                "width": 1024,
                "height": 576,
                "elapsed_s": dt,
                "prompt": prompt,
            },
        },
    )
    print(f"  [ref:cast]  {out_p.name} ({dt}s)", flush=True)
    return out_p


def ensure_prop_sheet(
    model: Any,
    mspec: Any,
    prop: dict[str, str],
    style_key: str,
    refs_dir: Path,
    blank: Path,
    seed: int,
) -> Path:
    out_p = refs_dir / "props" / f"prop_{prop['id']}__{style_key}.png"
    if out_p.exists() and out_p.with_suffix(".json").exists():
        return out_p

    out_p.parent.mkdir(parents=True, exist_ok=True)
    prompt = build_prop_sheet_prompt(prop["name"], prop["description"], world_style=WORLD_STYLES[style_key])
    t0 = time.perf_counter()
    gen = render_edit(
        model,
        mspec,
        prompt=prompt,
        image_paths=[blank],
        seed=seed,
        width=1024,
        height=576,
    )
    save_clean(gen, out_p)
    dt = round(time.perf_counter() - t0, 2)
    write_sidecar(
        out_p,
        {
            "id": out_p.stem,
            "kind": "canonical_prop_ref",
            "image": out_p.name,
            "prop_id": prop["id"],
            "prop_name": prop["name"],
            "world_style": style_key,
            "generation": {
                "model": mspec.key,
                "seed": seed,
                "steps": mspec.default_steps,
                "width": 1024,
                "height": 576,
                "elapsed_s": dt,
                "prompt": prompt,
            },
        },
    )
    print(f"  [ref:prop]  {out_p.name} ({dt}s)", flush=True)
    return out_p


def ensure_set_plate(
    model: Any,
    mspec: Any,
    set_def: dict[str, str],
    style_key: str,
    refs_dir: Path,
    blank: Path,
    width: int,
    height: int,
    seed: int,
) -> Path:
    out_p = refs_dir / "sets" / f"set_{set_def['id']}__{style_key}.png"
    if out_p.exists() and out_p.with_suffix(".json").exists():
        return out_p

    out_p.parent.mkdir(parents=True, exist_ok=True)
    prompt = build_set_plate_prompt(
        set_def["name"],
        set_def["description"],
        camera_angle="wide eye-level establishing shot",
        lighting_state=set_def["lighting_day"],
        world_style=WORLD_STYLES[style_key],
    )
    t0 = time.perf_counter()
    gen = render_edit(
        model,
        mspec,
        prompt=prompt,
        image_paths=[blank],
        seed=seed,
        width=width,
        height=height,
    )
    save_clean(gen, out_p)
    dt = round(time.perf_counter() - t0, 2)
    write_sidecar(
        out_p,
        {
            "id": out_p.stem,
            "kind": "canonical_set_plate",
            "image": out_p.name,
            "set_id": set_def["id"],
            "set_name": set_def["name"],
            "world_style": style_key,
            "lighting_state": set_def["lighting_day"],
            "generation": {
                "model": mspec.key,
                "seed": seed,
                "steps": mspec.default_steps,
                "width": width,
                "height": height,
                "elapsed_s": dt,
                "prompt": prompt,
            },
        },
    )
    print(f"  [ref:set]   {out_p.name} ({dt}s)", flush=True)
    return out_p


def render_and_record_shot(
    model: Any,
    mspec: Any,
    *,
    shot_id: str,
    shots_dir: Path,
    prompt: str,
    image_paths: list[Path],
    seed: int,
    width: int,
    height: int,
    scene_meta: dict[str, Any],
) -> Path | None:
    out_p = shots_dir / f"{shot_id}.png"
    if out_p.exists() and out_p.with_suffix(".json").exists():
        print(f"  [skip]      {shot_id} already exists", flush=True)
        return out_p

    t0 = time.perf_counter()
    gen = render_edit(
        model,
        mspec,
        prompt=prompt,
        image_paths=image_paths,
        seed=seed,
        width=width,
        height=height,
    )
    save_clean(gen, out_p)
    dt = round(time.perf_counter() - t0, 2)

    meta = {
        "id": shot_id,
        "image": out_p.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation": {
            "model": mspec.key,
            "seed": seed,
            "steps": mspec.default_steps,
            "width": width,
            "height": height,
            "elapsed_s": dt,
            "prompt": prompt,
            "ref_images": [_rel_repo(p) for p in image_paths],
        },
        "scene": scene_meta,
        "triage": {
            "human_label": None,
            "defect_types": [],
            "severity": None,
            "notes": "",
            "split": None,
        },
    }
    write_sidecar(out_p, meta)
    print(f"  [shot]      {shot_id:38s} cat={scene_meta['category']:20s} ({dt:5.1f}s)", flush=True)
    return out_p


def run_miner(
    *,
    out_dir: Path,
    model_key: str,
    max_cycles: int,
    smoke: bool,
    width: int,
    height: int,
    base_seed: int,
) -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    out_dir.mkdir(parents=True, exist_ok=True)
    refs_dir = out_dir / "refs"
    shots_dir = out_dir / "shots"
    sheets_dir = out_dir / "contact_sheets"
    for d in (refs_dir, shots_dir, sheets_dir):
        d.mkdir(parents=True, exist_ok=True)

    roster = load_full_roster()
    rng = random.Random(base_seed)
    ordered_chars = build_stratified_character_order(roster, rng)
    blank = _blank_canvas(refs_dir / "_neutral_canvas.png", width, height)

    # Determine starting cycle index so runs are seamlessly resumable
    existing_cycles = set()
    for jf in shots_dir.glob("c*_*.json"):
        prefix = jf.name.split("_")[0]
        if prefix.startswith("c") and prefix[1:].isdigit():
            existing_cycles.add(int(prefix[1:]))
    start_cycle = (max(existing_cycles) + 1) if existing_cycles else 1

    print(
        f"[miner] Starting continuous test-data mining -> {out_dir}\n"
        f"[miner] Model: {model_key} | Roster: {len(ordered_chars)} chars | "
        f"Start cycle: {start_cycle} | Max cycles: {'INFINITE' if max_cycles <= 0 else max_cycles} | Smoke: {smoke}",
        flush=True,
    )

    unload_lms_models()
    model, mspec = load_edit_model(model_key)

    t_start = time.perf_counter()
    stats: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": model_key,
        "cycles_completed": 0,
        "shots_generated_this_run": 0,
        "errors_this_run": 0,
        "current_cycle": start_cycle,
        "running": True,
    }
    update_status(out_dir, stats)

    style_keys = list(WORLD_STYLES.keys())
    cycle = start_cycle
    cycles_done = 0

    try:
        while not _STOP_REQUESTED:
            if max_cycles > 0 and cycles_done >= max_cycles:
                break

            stats["current_cycle"] = cycle
            style_key = style_keys[(cycle - 1) % len(style_keys)]
            world_style = WORLD_STYLES[style_key]
            set_def = SETS_LIBRARY[(cycle - 1) % len(SETS_LIBRARY)]
            prop_def = PROPS_LIBRARY[(cycle - 1) % len(PROPS_LIBRARY)]

            # Pick 4 distinct characters per cycle rotating through the entire 121-char roster
            char_offset = ((cycle - 1) * 4) % len(ordered_chars)
            c_group = [ordered_chars[(char_offset + k) % len(ordered_chars)] for k in range(4)]
            c1, c2, c3, c4 = c_group

            cycle_seed = base_seed + cycle * 1009
            shot_group = f"c{cycle:04d}_{set_def['id']}"
            print(
                f"\n[miner] === Cycle {cycle:04d} | style={style_key} | set={set_def['id']} | prop={prop_def['id']} ===\n"
                f"        Cast: {c1['name']} ({c1['style_id']}), {c2['name']} ({c2['style_id']}), "
                f"{c3['name']} ({c3['style_id']}), {c4['name']} ({c4['style_id']})",
                flush=True,
            )

            try:
                # 1. Build/reuse canonical Visual Bible refs for this cycle
                set_plate = ensure_set_plate(
                    model, mspec, set_def, style_key, refs_dir, blank, width, height, cycle_seed + 1
                )
                if _STOP_REQUESTED:
                    break
                prop_ref = ensure_prop_sheet(
                    model, mspec, prop_def, style_key, refs_dir, blank, cycle_seed + 2
                )
                if _STOP_REQUESTED:
                    break
                c1_ref = ensure_cast_turnaround(model, mspec, c1, style_key, refs_dir, cycle_seed + 3)
                if _STOP_REQUESTED:
                    break
                c2_ref = ensure_cast_turnaround(model, mspec, c2, style_key, refs_dir, cycle_seed + 4)
                if _STOP_REQUESTED:
                    break

                # For c3 & c4 we can use their yearbook portraits directly in crowd shots (keeps ref budget fast)
                c3_portrait = Path(c3["portrait_path"])
                c4_portrait = Path(c4["portrait_path"])

                # Build the balanced shot schedule for this cycle
                act1 = CLEAN_ACTIONS_1[(cycle - 1) % len(CLEAN_ACTIONS_1)].format(
                    c1=c1["name"], c2=c2["name"], prop=prop_def["name"]
                )
                act2 = CLEAN_ACTIONS_2[(cycle - 1) % len(CLEAN_ACTIONS_2)].format(
                    c1=c1["name"], c2=c2["name"], prop=prop_def["name"]
                )
                act_crowd = STRESS_CROWD_ACTIONS[(cycle - 1) % len(STRESS_CROWD_ACTIONS)].format(
                    c1=c1["name"], c2=c2["name"], c3=c3["name"], c4=c4["name"], prop=prop_def["name"]
                )
                act_closeup = ACCESSORY_CLOSEUP_ACTIONS[(cycle - 1) % len(ACCESSORY_CLOSEUP_ACTIONS)].format(
                    c1=c1["name"], c2=c2["name"]
                )

                plan: list[dict[str, Any]] = [
                    # A: Clean Set-Plate Reuse Shot 1 (Suites 2, 4, 5 PASS anchor)
                    {
                        "shot_id": f"c{cycle:04d}_01_clean_set_a",
                        "category": "set_reuse_clean",
                        "image_paths": [set_plate, c1_ref, c2_ref, prop_ref],
                        "prompt": build_set_reuse_shot_prompt(
                            set_plate_index=1,
                            cast_refs=[(2, c1), (3, c2)],
                            prop_refs=[(4, prop_def["name"], "intact")],
                            action_description=act1,
                            camera_note="wide eye-level matching master set plate",
                            lighting_state=set_def["lighting_day"],
                            locked_camera=True,
                            world_style=world_style,
                        ),
                        "scene_meta": {
                            "category": "set_reuse_clean",
                            "target_suites": ["defects", "anchors", "contact_sheets", "adjacent_cuts", "set_reuse"],
                            "expected_continuity": "clean",
                            "style": style_key,
                            "cast_ids": [c1["id"], c2["id"]],
                            "prop_ids": [prop_def["id"]],
                            "set_id": set_def["id"],
                            "shot_group": shot_group,
                            "shot_index": 1,
                            "canonical_refs": {
                                "set": _rel_repo(set_plate),
                                "prop": _rel_repo(prop_ref),
                                "cast": [_rel_repo(c1_ref), _rel_repo(c2_ref)],
                            },
                            "stress_tags": ["set_plate_reuse", "two_char", "prop_interaction"],
                        },
                    },
                    # B: Clean Consecutive Beat on Same Set Plate (Suite 4 clean adjacent cut A -> B)
                    {
                        "shot_id": f"c{cycle:04d}_02_clean_set_b",
                        "category": "adjacent_clean",
                        "image_paths": [set_plate, c1_ref, c2_ref, prop_ref],
                        "prompt": build_set_reuse_shot_prompt(
                            set_plate_index=1,
                            cast_refs=[(2, c1), (3, c2)],
                            prop_refs=[(4, prop_def["name"], "intact")],
                            action_description=act2,
                            camera_note="slightly closer medium-wide matching master set plate",
                            lighting_state=set_def["lighting_day"],
                            locked_camera=True,
                            world_style=world_style,
                        ),
                        "scene_meta": {
                            "category": "adjacent_clean",
                            "target_suites": ["defects", "anchors", "contact_sheets", "adjacent_cuts", "set_reuse"],
                            "expected_continuity": "clean",
                            "paired_with": f"c{cycle:04d}_01_clean_set_a",
                            "style": style_key,
                            "cast_ids": [c1["id"], c2["id"]],
                            "prop_ids": [prop_def["id"]],
                            "set_id": set_def["id"],
                            "shot_group": shot_group,
                            "shot_index": 2,
                            "canonical_refs": {
                                "set": _rel_repo(set_plate),
                                "prop": _rel_repo(prop_ref),
                                "cast": [_rel_repo(c1_ref), _rel_repo(c2_ref)],
                            },
                            "stress_tags": ["set_plate_reuse", "adjacent_cut_clean", "hand_prop"],
                        },
                    },
                    # C: Hard 4-Character Crowd & Hands Stress Test (Suite 1 natural anatomy/merge defect miner)
                    {
                        "shot_id": f"c{cycle:04d}_03_stress_crowd",
                        "category": "stress_crowd_hands",
                        "image_paths": [set_plate, c1_ref, c2_ref, c3_portrait, c4_portrait],
                        "prompt": build_set_reuse_shot_prompt(
                            set_plate_index=1,
                            cast_refs=[(2, c1), (3, c2), (4, c3), (5, c4)],
                            prop_refs=[],
                            action_description=act_crowd,
                            camera_note="dynamic medium-wide action framing showing full bodies and hands",
                            lighting_state=set_def["lighting_day"],
                            locked_camera=False,
                            world_style=world_style,
                        ),
                        "scene_meta": {
                            "category": "stress_crowd_hands",
                            "target_suites": ["defects", "recovery"],
                            "expected_continuity": "stress_unknown",
                            "style": style_key,
                            "cast_ids": [c1["id"], c2["id"], c3["id"], c4["id"]],
                            "prop_ids": [prop_def["id"]],
                            "set_id": set_def["id"],
                            "shot_group": shot_group,
                            "shot_index": 3,
                            "canonical_refs": {
                                "set": _rel_repo(set_plate),
                                "cast": [_rel_repo(c1_ref), _rel_repo(c2_ref), _rel_repo(c3_portrait), _rel_repo(c4_portrait)],
                            },
                            "stress_tags": ["four_char_crowd", "overlapping_hands", "prop_interaction"],
                        },
                    },
                    # D: Subtle Character Drift (dropped c1 ref image -> crisp high-polish identity/wardrobe drift)
                    {
                        "shot_id": f"c{cycle:04d}_04_drift_char",
                        "category": "subtle_char_drift",
                        "image_paths": [set_plate, c2_ref, prop_ref],
                        "prompt": (
                            f"Image 1 is the MASTER SET PLATE for {set_def['name']}. Keep the background architecture "
                            f"and lighting from Image 1. Image 2 is {c2['name']}, keep their appearance from Image 2. "
                            f"Image 3 is the {prop_def['name']}. "
                            f"Next to {c2['name']} stands a quirky high school senior named {c1['name']} "
                            f"(loosely inspired by: {c1['archetype']}, wearing a casual vintage cardigan and sneakers) "
                            f"gesturing toward the {prop_def['name']}. "
                            f"Lighting: {set_def['lighting_day']}. {world_style}. Zero text, zero watermarks."
                        ),
                        "scene_meta": {
                            "category": "subtle_char_drift",
                            "target_suites": ["anchors", "contact_sheets", "adjacent_cuts", "recovery"],
                            "expected_continuity": "drift_character",
                            "drift_entity_id": c1["id"],
                            "paired_with": f"c{cycle:04d}_01_clean_set_a",
                            "style": style_key,
                            "cast_ids": [c1["id"], c2["id"]],
                            "prop_ids": [prop_def["id"]],
                            "set_id": set_def["id"],
                            "shot_group": shot_group,
                            "shot_index": 4,
                            "canonical_refs": {
                                "set": _rel_repo(set_plate),
                                "prop": _rel_repo(prop_ref),
                                "cast": [_rel_repo(c1_ref), _rel_repo(c2_ref)],
                            },
                            "stress_tags": ["dropped_char_ref", "subtle_identity_drift"],
                        },
                    },
                    # E: Subtle Prop Drift (dropped prop ref image -> crisp high-polish prop mutation)
                    {
                        "shot_id": f"c{cycle:04d}_05_drift_prop",
                        "category": "subtle_prop_drift",
                        "image_paths": [set_plate, c1_ref, c2_ref],
                        "prompt": build_set_reuse_shot_prompt(
                            set_plate_index=1,
                            cast_refs=[(2, c1), (3, c2)],
                            prop_refs=[],
                            action_description=(
                                f"{c1['name']} and {c2['name']} stand beside a quirky makeshift alternate version "
                                f"of a {prop_def['name']} painted in bright purple and chrome with different wheels and shape"
                            ),
                            camera_note="wide eye-level matching master set plate",
                            lighting_state=set_def["lighting_day"],
                            locked_camera=True,
                            world_style=world_style,
                        ),
                        "scene_meta": {
                            "category": "subtle_prop_drift",
                            "target_suites": ["anchors", "contact_sheets", "adjacent_cuts", "recovery"],
                            "expected_continuity": "drift_prop",
                            "drift_entity_id": prop_def["id"],
                            "paired_with": f"c{cycle:04d}_01_clean_set_a",
                            "style": style_key,
                            "cast_ids": [c1["id"], c2["id"]],
                            "prop_ids": [prop_def["id"]],
                            "set_id": set_def["id"],
                            "shot_group": shot_group,
                            "shot_index": 5,
                            "canonical_refs": {
                                "set": _rel_repo(set_plate),
                                "prop": _rel_repo(prop_ref),
                                "cast": [_rel_repo(c1_ref), _rel_repo(c2_ref)],
                            },
                            "stress_tags": ["dropped_prop_ref", "prop_mutation"],
                        },
                    },
                    # F: Adjacent-Cut Lighting Break (same set + cast + prop, clashing lighting/time-of-day)
                    {
                        "shot_id": f"c{cycle:04d}_06_lighting_break",
                        "category": "adjacent_lighting_break",
                        "image_paths": [set_plate, c1_ref, c2_ref, prop_ref],
                        "prompt": build_set_reuse_shot_prompt(
                            set_plate_index=1,
                            cast_refs=[(2, c1), (3, c2)],
                            prop_refs=[(4, prop_def["name"], "intact")],
                            action_description=act2,
                            camera_note="medium-wide shot in the same room",
                            lighting_state=set_def["lighting_break"],
                            locked_camera=False,
                            world_style=world_style,
                        ),
                        "scene_meta": {
                            "category": "adjacent_lighting_break",
                            "target_suites": ["adjacent_cuts"],
                            "expected_continuity": "drift_lighting",
                            "paired_with": f"c{cycle:04d}_01_clean_set_a",
                            "style": style_key,
                            "cast_ids": [c1["id"], c2["id"]],
                            "prop_ids": [prop_def["id"]],
                            "set_id": set_def["id"],
                            "shot_group": shot_group,
                            "shot_index": 6,
                            "canonical_refs": {
                                "set": _rel_repo(set_plate),
                                "prop": _rel_repo(prop_ref),
                                "cast": [_rel_repo(c1_ref), _rel_repo(c2_ref)],
                            },
                            "stress_tags": ["lighting_flip", "adjacent_cut_break"],
                        },
                    },
                    # G: Tight Face / Accessory Close-Up Stress Shot (Suite 1 fine-grained glitch miner)
                    {
                        "shot_id": f"c{cycle:04d}_07_closeup_stress",
                        "category": "stress_closeup_face",
                        "image_paths": [c1_ref, c2_ref, set_plate],
                        "prompt": (
                            f"Image 1 is {c1['name']} ({medium_lock_phrase(c1.get('style_id', ''))}). "
                            f"Image 2 is {c2['name']} ({medium_lock_phrase(c2.get('style_id', ''))}). "
                            f"Image 3 is the background location ({set_def['name']}). "
                            f"{act_closeup}. Lighting: {set_def['lighting_day']}. {world_style}. Zero text, zero watermarks."
                        ),
                        "scene_meta": {
                            "category": "stress_closeup_face",
                            "target_suites": ["defects", "anchors", "recovery"],
                            "expected_continuity": "stress_unknown",
                            "style": style_key,
                            "cast_ids": [c1["id"], c2["id"]],
                            "prop_ids": [],
                            "set_id": set_def["id"],
                            "shot_group": shot_group,
                            "shot_index": 7,
                            "canonical_refs": {
                                "set": _rel_repo(set_plate),
                                "cast": [_rel_repo(c1_ref), _rel_repo(c2_ref)],
                            },
                            "stress_tags": ["tight_closeup", "face_accessories", "hands_near_face"],
                        },
                    },
                    # H: Style / Medium Drift Stress Shot (renders c1 WITHOUT medium-lock -> realistic medium bleed)
                    {
                        "shot_id": f"c{cycle:04d}_08_style_drift",
                        "category": "style_medium_drift",
                        "image_paths": [set_plate, c1_ref],
                        "prompt": (
                            f"In the location from Image 1 ({set_def['name']}), re-imagine the character from Image 2 "
                            f"({c1['name']}) fully rendered as a smooth glossy 3D Pixar-like plastic figurine with "
                            f"soft airbrushed gradients and rounded features, standing cheerfully holding a clipboard. "
                            f"Lighting: {set_def['lighting_day']}. Zero text, zero watermarks."
                        ),
                        "scene_meta": {
                            "category": "style_medium_drift",
                            "target_suites": ["anchors", "contact_sheets", "recovery"],
                            "expected_continuity": "drift_style",
                            "drift_entity_id": c1["id"],
                            "paired_with": f"c{cycle:04d}_01_clean_set_a",
                            "style": style_key,
                            "cast_ids": [c1["id"]],
                            "prop_ids": [],
                            "set_id": set_def["id"],
                            "shot_group": shot_group,
                            "shot_index": 8,
                            "canonical_refs": {
                                "set": _rel_repo(set_plate),
                                "cast": [_rel_repo(c1_ref)],
                            },
                            "stress_tags": ["medium_bleed", "style_drift"],
                        },
                    },
                ]

                if smoke:
                    # Quick smoke test: just 3 representative shots (clean, crowd stress, subtle char drift)
                    plan = [plan[0], plan[2], plan[3]]

                rendered_map: dict[str, Path] = {}
                for idx_s, item in enumerate(plan):
                    if _STOP_REQUESTED:
                        break
                    p = render_and_record_shot(
                        model,
                        mspec,
                        shot_id=item["shot_id"],
                        shots_dir=shots_dir,
                        prompt=item["prompt"],
                        image_paths=item["image_paths"],
                        seed=cycle_seed + 100 + idx_s * 37,
                        width=width,
                        height=height,
                        scene_meta=item["scene_meta"],
                    )
                    if p is not None:
                        rendered_map[item["category"]] = p
                        stats["shots_generated_this_run"] += 1
                        stats["elapsed_s"] = round(time.perf_counter() - t_start, 1)
                        update_status(out_dir, stats)

                # 2. Auto-stitch a 2x2 Contact Sheet for this cycle if we have the tiles
                if not _STOP_REQUESTED:
                    clean_a = rendered_map.get("set_reuse_clean")
                    clean_b = rendered_map.get("adjacent_clean") or rendered_map.get("stress_crowd_hands")
                    drift_c = rendered_map.get("subtle_char_drift")
                    if clean_a and clean_b and drift_c:
                        sheet_id = f"sheet_c{cycle:04d}_char_{c1['id']}"
                        sheet_p = sheets_dir / f"{sheet_id}.png"
                        make_contact_sheet(
                            [
                                ("TILE 0 (REF)", c1_ref),
                                ("TILE 1", clean_a),
                                ("TILE 2", drift_c),
                                ("TILE 3", clean_b),
                            ],
                            sheet_p,
                            cols=2,
                        )
                        write_sidecar(
                            sheet_p,
                            {
                                "id": sheet_id,
                                "image": sheet_p.name,
                                "entity_type": "character",
                                "entity_id": c1["id"],
                                "entity_name": f"{c1['name']} ({c1.get('style_label', '')})",
                                "world_style": style_key,
                                "tile_count": 3,
                                "intended_outlier_tiles": [2],
                                "tiles": {
                                    "0": _rel_repo(c1_ref),
                                    "1": _rel_repo(clean_a),
                                    "2": _rel_repo(drift_c),
                                    "3": _rel_repo(clean_b),
                                },
                                "triage": {"human_verified_outliers": None, "notes": "", "split": None},
                            },
                        )
                        print(f"  [sheet]     {sheet_p.name}", flush=True)

                    drift_p = rendered_map.get("subtle_prop_drift")
                    if clean_a and rendered_map.get("adjacent_clean") and drift_p:
                        sheet_id = f"sheet_c{cycle:04d}_prop_{prop_def['id']}"
                        sheet_p = sheets_dir / f"{sheet_id}.png"
                        make_contact_sheet(
                            [
                                ("TILE 0 (REF)", prop_ref),
                                ("TILE 1", clean_a),
                                ("TILE 2", rendered_map["adjacent_clean"]),
                                ("TILE 3", drift_p),
                            ],
                            sheet_p,
                            cols=2,
                        )
                        write_sidecar(
                            sheet_p,
                            {
                                "id": sheet_id,
                                "image": sheet_p.name,
                                "entity_type": "prop",
                                "entity_id": prop_def["id"],
                                "entity_name": prop_def["name"],
                                "world_style": style_key,
                                "tile_count": 3,
                                "intended_outlier_tiles": [3],
                                "tiles": {
                                    "0": _rel_repo(prop_ref),
                                    "1": _rel_repo(clean_a),
                                    "2": _rel_repo(rendered_map["adjacent_clean"]),
                                    "3": _rel_repo(drift_p),
                                },
                                "triage": {"human_verified_outliers": None, "notes": "", "split": None},
                            },
                        )
                        print(f"  [sheet]     {sheet_p.name}", flush=True)

            except Exception as exc:
                stats["errors_this_run"] += 1
                print(f"  [ERROR] Cycle {cycle} encountered an error: {exc}", flush=True)
                traceback.print_exc()

            cycles_done += 1
            stats["cycles_completed"] = cycles_done
            stats["elapsed_s"] = round(time.perf_counter() - t_start, 1)
            idx_summary = rebuild_index(out_dir)
            stats["total_shots_on_disk"] = idx_summary["total_shots"]
            stats["total_contact_sheets_on_disk"] = idx_summary["total_contact_sheets"]
            stats["unique_characters_on_disk"] = idx_summary["unique_characters"]
            update_status(out_dir, stats)

            _clear_mlx()
            cycle += 1

    finally:
        del model
        _clear_mlx()
        stats["running"] = False
        stats["stopped_at"] = datetime.now(timezone.utc).isoformat()
        stats["elapsed_s"] = round(time.perf_counter() - t_start, 1)
        rebuild_index(out_dir)
        update_status(out_dir, stats)
        print(
            f"\n[miner] Clean exit. Total shots on disk: {stats.get('total_shots_on_disk', 0)} "
            f"(+{stats['shots_generated_this_run']} this run) in {stats['elapsed_s']}s.",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous balanced test-data miner for Fever Dreams evals.")
    parser.add_argument("--out-dir", type=Path, default=EVALS_DIR / "mined_data")
    parser.add_argument("--model", default="qwen-image-2.1-turbo")
    parser.add_argument("--max-cycles", type=int, default=0, help="0 = run indefinitely until stopped")
    parser.add_argument("--smoke", action="store_true", help="Mini-cycle (3 shots) to verify pipeline")
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--seed", type=int, default=90001)
    args = parser.parse_args()

    run_miner(
        out_dir=args.out_dir,
        model_key=args.model,
        max_cycles=args.max_cycles,
        smoke=args.smoke,
        width=args.width,
        height=args.height,
        base_seed=args.seed,
    )


if __name__ == "__main__":
    main()
