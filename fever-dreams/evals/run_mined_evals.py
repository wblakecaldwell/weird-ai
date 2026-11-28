"""General-Purpose v3 Mined Eval Harness (Zero Book-1 Hints, Strict Dev / Holdout Split).

Benchmarks local LM Studio VLMs on the 449-shot / 112-sheet Neon-gold mined dataset across:
  - M1 (single_frame):   Standalone anatomy, duplicate-character, spatial clipping, accessory, & face QA
  - M2 (anchors):        [CANONICAL_REF, TEST] character, style/medium, and prop fidelity
  - M3 (contact_sheets): 2x2 numbered contact sheets (Tile 0 = REF, Tiles 1..3 = instances)
  - M4 (adjacent_cuts):  [SHOT_A, SHOT_B] shot-to-shot lighting, prop, and character continuity

Supports two prompt strategies for A/B testing checked improvement on Dev:
  - `baseline`: Clean general observe-first schema (zero Book-1 hints)
  - `forensic`: Upgraded general forensic checklist (cast-count/duplicate check, spatial clipping check,
                explicit medium/style check, and optional 2x2 quad-zoom for M1 single-frame inspection)

Usage:
    uv run python evals/run_mined_evals.py --split dev --strategy baseline --out evals/results/mined_dev_baseline.json
    uv run python evals/run_mined_evals.py --split dev --strategy forensic --models gemma-4-31b-it qwen3-vl-8b-instruct qwen3-vl-4b-instruct
    uv run python evals/run_mined_evals.py --split holdout --strategy forensic --out evals/results/mined_holdout_final.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

EVALS_DIR = Path(__file__).resolve().parent
FEVER_DREAMS_ROOT = EVALS_DIR.parent
WORKSPACE_ROOT = FEVER_DREAMS_ROOT.parent
DATA_DIR = EVALS_DIR / "mined_data"
MANIFEST_PATH = DATA_DIR / "gold_manifest.json"
RESULTS_DIR = EVALS_DIR / "results"

if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from vlm_client import load_lms_model, query_vlm_json, unload_all_lms  # noqa: E402

DEFAULT_MODELS = [
    "gemma-4-31b-it",
    "qwen3-vl-30b-a3b-instruct-mlx",
    "qwen3-vl-8b-instruct",
    "qwen3-vl-4b-instruct",
    "qwen3-vl-2b-instruct",
]


def resolve_img(rel: str) -> Path:
    rel = rel.lstrip("/")
    for base in (DATA_DIR, FEVER_DREAMS_ROOT, WORKSPACE_ROOT):
        p = (base / rel).resolve()
        if p.is_file():
            return p
    raise FileNotFoundError(f"Could not resolve image: {rel}")


def make_quad_zoom_image(src: Path, cache_dir: Path) -> Path:
    """Create a 2x2 zoomed quadrant grid (with 12% overlap) so VLMs see faces, hands, and clipping at high res."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{src.stem}_quad.jpg"
    if out.exists():
        return out
    im = Image.open(src).convert("RGB")
    w, h = im.size
    mx, my = int(w * 0.56), int(h * 0.56)
    ox, oy = int(w * 0.44), int(h * 0.44)
    quads = [
        ("TOP-LEFT", im.crop((0, 0, mx, my))),
        ("TOP-RIGHT", im.crop((ox, 0, w, my))),
        ("BOTTOM-LEFT", im.crop((0, oy, mx, h))),
        ("BOTTOM-RIGHT", im.crop((ox, oy, w, h))),
    ]
    tw, th = 448, 384
    pad = 8
    canvas = Image.new("RGB", (tw * 2 + pad * 3, th * 2 + pad * 3), (18, 16, 14))
    draw = ImageDraw.Draw(canvas)
    for idx, (lbl, q) in enumerate(quads):
        r, c = divmod(idx, 2)
        x0 = pad + c * (tw + pad)
        y0 = pad + r * (th + pad)
        resized = q.resize((tw, th), Image.Resampling.LANCZOS)
        canvas.paste(resized, (x0, y0))
        draw.rectangle([x0, y0, x0 + tw - 1, y0 + th - 1], outline=(200, 180, 120), width=2)
        draw.rectangle([x0 + 4, y0 + 4, x0 + 115, y0 + 24], fill=(18, 16, 14))
        draw.text((x0 + 10, y0 + 8), lbl, fill=(244, 208, 111))
    canvas.save(out, format="JPEG", quality=88)
    return out


# ---------------------------------------------------------------------------
# General-purpose Schemas & Prompts (ZERO Book-1 hints!)
# ---------------------------------------------------------------------------

M1_BASELINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scene_observation": {"type": "string"},
        "defects_found": {"type": "array", "items": {"type": "string"}},
        "passed": {"type": "boolean"},
        "score": {"type": "integer"},
        "primary_defect": {
            "type": "string",
            "enum": ["none", "anatomy", "duplicate_char", "spatial_clipping", "accessory", "face_blob", "hand_merge", "other"],
        },
    },
    "required": ["scene_observation", "defects_found", "passed", "score", "primary_defect"],
    "additionalProperties": False,
}

M1_FORENSIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "character_count_and_duplicates_check": {"type": "string"},
        "limbs_hands_and_anatomy_check": {"type": "string"},
        "spatial_clipping_and_physics_check": {"type": "string"},
        "faces_and_accessories_check": {"type": "string"},
        "defects_found": {"type": "array", "items": {"type": "string"}},
        "passed": {"type": "boolean"},
        "score": {"type": "integer"},
        "primary_defect": {
            "type": "string",
            "enum": ["none", "anatomy", "duplicate_char", "spatial_clipping", "accessory", "face_blob", "hand_merge", "other"],
        },
    },
    "required": [
        "character_count_and_duplicates_check",
        "limbs_hands_and_anatomy_check",
        "spatial_clipping_and_physics_check",
        "faces_and_accessories_check",
        "defects_found",
        "passed",
        "score",
        "primary_defect",
    ],
    "additionalProperties": False,
}

M1_SYSTEM_BASELINE = (
    "You are a visual quality-control (QC) inspector for a stylized mixed-media and stop-motion storybook. "
    "Stylized characters (pixel art, felt puppets, clay figurines, paper cutouts, ink doodles, retro 3D CGI, "
    "anthropomorphic objects) are INTENTIONAL and must NOT be flagged as defects. "
    "FAIL (passed=false) ONLY if there is a genuine AI rendering glitch: duplicate/cloned identical character appearing twice, "
    "extra/missing/detached limbs or mangled hands, impossible spatial clipping (character phasing through a solid table/furniture), "
    "broken/half accessories, foreign blobs stuck on faces, or hands fused into solid objects. Return ONLY valid JSON."
)

M1_SYSTEM_FORENSIC = (
    "You are a strict forensic visual QC inspector for a mixed-media and stop-motion animation studio. "
    "IMPORTANT: Quirky intentional art mediums (pixel sprites, handmade felt puppets, lumpy clay, paper cutouts, "
    "hand-drawn ink doodles, low-poly 3D, sentient objects) are INTENTIONAL — do NOT fail an intentional style. "
    "However, you MUST FAIL (passed=false) ANY genuine AI rendering error:\n"
    "1. DUPLICATE / CLONED CHARACTER: The exact same character accidentally appearing twice in the same scene.\n"
    "2. ANATOMY & LIMBS: Extra legs/arms/heads, missing limbs, detached floating body parts, or severely melted/fused hands.\n"
    "3. SPATIAL CLIPPING / IMPOSSIBLE PHYSICS: A character's legs, waist, or body phasing straight through a solid table, bench, or wall, or floating impossibly.\n"
    "4. BROKEN ACCESSORIES / FACE ARTIFACTS: Half/partial floating glasses, foreign blobs stuck to a face/chin, or garbled gibberish text.\n"
    "Inspect each checklist field literally from the pixels BEFORE deciding `passed`. Only set passed=true when defects_found is []. Return ONLY valid JSON."
)

M2_ANCHOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ref_description": {"type": "string"},
        "test_description": {"type": "string"},
        "differences_found": {"type": "array", "items": {"type": "string"}},
        "passed": {"type": "boolean"},
        "score": {"type": "integer"},
        "mismatch_type": {
            "type": "string",
            "enum": ["none", "char_drift", "style_drift", "prop_drift", "anatomy_or_duplicate"],
        },
    },
    "required": ["ref_description", "test_description", "differences_found", "passed", "score", "mismatch_type"],
    "additionalProperties": False,
}

M2_SYSTEM = (
    "You are a visual continuity supervisor comparing a target entity in TEST_FRAME against CANONICAL_REFERENCE. "
    "Normal pose, expression, framing, and background changes are EXPECTED and should PASS. "
    "FAIL (passed=false) if:\n"
    "1. CHARACTER DRIFT: The character's species, face, hair, outfit, or main colors clearly do NOT match CANONICAL_REFERENCE (or the character is missing or duplicated twice).\n"
    "2. STYLE/MEDIUM DRIFT: The character's rendering medium changed (e.g., flat 2D pixel art, ink doodle, or paper cutout turned into smooth 3D clay/CGI, or vice versa).\n"
    "3. PROP DRIFT: The target prop's core shape, color palette, or object type clearly mutated away from CANONICAL_REFERENCE.\n"
    "Describe CANONICAL_REFERENCE and TEST_FRAME literally first, list only clear mismatches in differences_found, and set passed=(len(differences_found)==0). Return ONLY valid JSON."
)

M3_SHEET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ref_tile_0_summary": {"type": "string"},
        "tile_1_check": {"type": "string"},
        "tile_2_check": {"type": "string"},
        "tile_3_check": {"type": "string"},
        "outlier_tiles": {"type": "array", "items": {"type": "integer"}},
        "summary": {"type": "string"},
    },
    "required": ["ref_tile_0_summary", "tile_1_check", "tile_2_check", "tile_3_check", "outlier_tiles", "summary"],
    "additionalProperties": False,
}

M3_SYSTEM = (
    "You are a continuity supervisor auditing a numbered 2x2 contact sheet. "
    "TILE 0 (gold border, top-left) is the CANONICAL REFERENCE. TILES 1, 2, and 3 are scene instances of that entity. "
    "Inspect each tile 1..3 individually against TILE 0 BEFORE filling outlier_tiles. "
    "Ignore normal pose/framing/background differences. Flag a tile number in outlier_tiles ONLY if that tile shows "
    "a clear continuity break (wrong character/outfit/face, wrong rendering medium/style, wrong prop shape/color, or duplicate/glitched character). "
    "If all tiles match TILE 0, return outlier_tiles: []. Return ONLY valid JSON."
)

M4_ADJACENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lighting_comparison": {"type": "string"},
        "characters_and_props_comparison": {"type": "string"},
        "set_comparison": {"type": "string"},
        "breaks_found": {"type": "array", "items": {"type": "string"}},
        "passed": {"type": "boolean"},
        "score": {"type": "integer"},
        "break_type": {
            "type": "string",
            "enum": ["none", "lighting_break", "prop_drift", "char_drift", "style_drift", "set_break"],
        },
    },
    "required": [
        "lighting_comparison",
        "characters_and_props_comparison",
        "set_comparison",
        "breaks_found",
        "passed",
        "score",
        "break_type",
    ],
    "additionalProperties": False,
}

M4_SYSTEM = (
    "You are a film continuity supervisor comparing two consecutive shots (SHOT_A followed immediately by SHOT_B) "
    "in the SAME scene moments apart. Normal character movement, poses, and slight camera-angle shifts are EXPECTED and PASS. "
    "FAIL (passed=false) if there is a jarring continuity break between SHOT_A and SHOT_B:\n"
    "1. LIGHTING BREAK: Time-of-day or lighting drastically shifts (e.g., bright warm daytime -> dark blue/green night or storm).\n"
    "2. PROP DRIFT: A central prop abruptly changes color, shape, or object design between shots.\n"
    "3. CHARACTER / STYLE DRIFT: A character abruptly changes species, face, outfit, rendering medium, or gets cloned twice.\n"
    "Compare lighting, characters/props, and set first, list only clear breaks in breaks_found, and set passed=(len(breaks_found)==0). Return ONLY valid JSON."
)


# ---------------------------------------------------------------------------
# Case Builders (from gold_manifest.json, strictly isolated by split!)
# ---------------------------------------------------------------------------

def build_split_cases(manifest: dict[str, Any], split: str, max_per_suite: int = 0) -> dict[str, list[dict[str, Any]]]:
    shots = [s for s in manifest["shots"] if s["split"] == split]
    shots_by_id = {s["id"]: s for s in shots}
    sheets = [s for s in manifest["contact_sheets"] if s["split"] == split]

    def _cap_balanced(items: list[dict[str, Any]], key: str = "expected") -> list[dict[str, Any]]:
        if max_per_suite <= 0 or len(items) <= max_per_suite:
            return items
        pos = [x for x in items if x.get(key) == "FAIL"]
        neg = [x for x in items if x.get(key) == "PASS"]
        half = max_per_suite // 2
        chosen = pos[:half] + neg[: max_per_suite - min(half, len(pos))]
        if len(chosen) < max_per_suite:
            rem = [x for x in items if x not in chosen]
            chosen.extend(rem[: max_per_suite - len(chosen)])
        return chosen

    # M1: Single-frame natural defects & clean controls (categories with natural PASS/FAIL)
    m1_cats = {"set_reuse_clean", "adjacent_clean", "stress_crowd_hands", "stress_closeup_face"}
    m1_all = []
    for s in shots:
        cat = (s.get("scene") or {}).get("category")
        if cat not in m1_cats:
            continue
        # Only include cases where defects are single-frame observable (exclude pure ref-only char_drift if no ref is passed?
        # Keep anatomy, duplicate, spatial clipping, accessory, face_blob, hand_merge, set_warp, plus clean PASS!)
        defs = set(s.get("defect_types") or [])
        if s["gold_label"] == "PASS" or (defs & {"anatomy", "accessory", "face_blob", "hand_merge", "set_warp"}):
            m1_all.append(
                {
                    "id": s["id"],
                    "image": s["image"],
                    "expected": s["gold_label"],
                    "category": cat,
                    "style": (s.get("scene") or {}).get("style"),
                    "expected_cast_count": len((s.get("scene") or {}).get("cast_ids") or []),
                    "defect_types": s.get("defect_types") or [],
                    "reason": s.get("reason", ""),
                }
            )
    m1_cases = _cap_balanced(m1_all)

    # M2: Canonical Reference Anchors ([REF, TEST])
    m2_all = []
    for s in shots:
        sc = s.get("scene") or {}
        cat = sc.get("category")
        crefs = sc.get("canonical_refs") or {}
        cast_refs = crefs.get("cast") or []
        prop_ref = crefs.get("prop")
        defs = set(s.get("defect_types") or [])

        if cat in {"subtle_char_drift", "style_medium_drift"} and cast_refs:
            m2_all.append(
                {
                    "id": f"anc_char_{s['id']}",
                    "shot_id": s["id"],
                    "entity_type": "character",
                    "ref_image": cast_refs[0],
                    "test_image": s["image"],
                    "expected": "FAIL" if (s["gold_label"] == "FAIL" and (defs & {"char_drift", "style_drift", "anatomy"})) else "PASS",
                    "category": cat,
                    "style": sc.get("style"),
                    "defect_types": s.get("defect_types") or [],
                    "reason": s.get("reason", ""),
                }
            )
        elif cat == "subtle_prop_drift" and prop_ref:
            m2_all.append(
                {
                    "id": f"anc_prop_{s['id']}",
                    "shot_id": s["id"],
                    "entity_type": "prop",
                    "ref_image": prop_ref,
                    "test_image": s["image"],
                    "expected": "FAIL" if (s["gold_label"] == "FAIL" and ("prop_drift" in defs)) else "PASS",
                    "category": cat,
                    "style": sc.get("style"),
                    "defect_types": s.get("defect_types") or [],
                    "reason": s.get("reason", ""),
                }
            )
        elif cat == "set_reuse_clean" and s["gold_label"] == "PASS":
            if cast_refs:
                m2_all.append(
                    {
                        "id": f"anc_char_pass_{s['id']}",
                        "shot_id": s["id"],
                        "entity_type": "character",
                        "ref_image": cast_refs[0],
                        "test_image": s["image"],
                        "expected": "PASS",
                        "category": cat,
                        "style": sc.get("style"),
                        "defect_types": [],
                        "reason": s.get("reason", ""),
                    }
                )
            if prop_ref:
                m2_all.append(
                    {
                        "id": f"anc_prop_pass_{s['id']}",
                        "shot_id": s["id"],
                        "entity_type": "prop",
                        "ref_image": prop_ref,
                        "test_image": s["image"],
                        "expected": "PASS",
                        "category": cat,
                        "style": sc.get("style"),
                        "defect_types": [],
                        "reason": s.get("reason", ""),
                    }
                )
    m2_cases = _cap_balanced(m2_all)

    # M3: Contact sheets
    m3_cases = sheets[:max_per_suite] if max_per_suite > 0 else sheets

    # M4: Adjacent Cuts ([SHOT_A, SHOT_B])
    m4_all = []
    for s in shots:
        sc = s.get("scene") or {}
        cat = sc.get("category")
        paired_id = sc.get("paired_with")
        if not paired_id or paired_id not in shots_by_id:
            continue
        shot_a = shots_by_id[paired_id]
        if cat not in {"adjacent_clean", "adjacent_lighting_break", "subtle_prop_drift", "subtle_char_drift"}:
            continue
        # Only use clean shot_a as baseline if shot_a itself didn't fail completely, or for lighting/prop breaks
        if cat == "adjacent_clean":
            if s["gold_label"] != "PASS" or shot_a["gold_label"] != "PASS":
                continue
            exp = "PASS"
        else:
            if s["gold_label"] != "FAIL":
                continue
            exp = "FAIL"
        m4_all.append(
            {
                "id": f"adj_{s['id']}",
                "shot_a": shot_a["image"],
                "shot_b": s["image"],
                "expected": exp,
                "category": cat,
                "style": sc.get("style"),
                "defect_types": s.get("defect_types") or [],
                "reason": s.get("reason", ""),
            }
        )
    m4_cases = _cap_balanced(m4_all)

    return {
        "single_frame": m1_cases,
        "anchors": m2_cases,
        "contact_sheets": m3_cases,
        "adjacent_cuts": m4_cases,
    }


# ---------------------------------------------------------------------------
# Suite Runners
# ---------------------------------------------------------------------------

M1_PROBE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "literal_observation": {"type": "string"},
        "defect_detected": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "defect_summary": {"type": "string"},
    },
    "required": ["literal_observation", "defect_detected", "confidence", "defect_summary"],
    "additionalProperties": False,
}

M1_PROBE_SYSTEM = (
    "You are a literal forensic visual inspector answering ONE specific quality-control check on a stylized "
    "mixed-media or stop-motion animation still. Intentional quirky character mediums (pixel sprites, felt puppets, "
    "clay figurines, paper cutouts, ink doodles, retro 3D CGI, anthropomorphic objects) are NOT defects. "
    "Describe literally what you see in `literal_observation` first, then set `defect_detected` and `confidence`."
)


def run_m1(model: str, cases: list[dict[str, Any]], strategy: str, use_quad_zoom: bool) -> dict[str, Any]:
    tp = fp = tn = fn = parse_err = 0
    total_s = 0.0
    rows = []
    cache_dir = DATA_DIR / "_previews" / "quad_zoom"
    schema = M1_FORENSIC_SCHEMA if strategy == "forensic" else M1_BASELINE_SCHEMA
    sys_prompt = M1_SYSTEM_FORENSIC if strategy == "forensic" else M1_SYSTEM_BASELINE

    for c in cases:
        img_p = resolve_img(c["image"])
        imgs = [("IMAGE", img_p)]
        if use_quad_zoom:
            q_p = make_quad_zoom_image(img_p, cache_dir)
            imgs.append(("ZOOMED_QUADRANTS", q_p))

        if strategy == "probed_general":
            exp_n = c.get("expected_cast_count") or 0
            count_hint = f"This scene is supposed to contain {exp_n} distinct character(s). " if exp_n else ""
            probes = [
                (
                    "duplicate_or_count",
                    f"{count_hint}Count every character visible in IMAGE. "
                    f"Is ANY character accidentally DUPLICATED / CLONED (the exact same character appearing twice in the same shot), "
                    f"or is the character count clearly wrong compared to {exp_n or 'the expected cast'}? "
                    f"Set defect_detected=true with confidence='high' ONLY if you clearly see a cloned/duplicate character or missing/extra character.",
                ),
                (
                    "anatomy_and_clipping",
                    "Inspect every character's legs, feet, arms, hands, and body contact with tables/furniture/props. "
                    "Flag ONLY clear AI glitches: (a) extra or missing limbs (e.g. 3 legs, 3 arms, detached floating limbs), "
                    "(b) a hand/arm physically fused/melted into a solid object, OR "
                    "(c) impossible spatial clipping where a character's waist/legs phase straight through a solid tabletop or workbench. "
                    "Do NOT flag normal cartoon/puppet hands or stylized proportions.",
                ),
                (
                    "face_and_accessories",
                    "Inspect every character's face, eyes, eyewear, chin, and mouth closely. "
                    "Flag ONLY clear AI glitches: a broken/partial floating accessory (like half a pair of glasses on one side only), "
                    "a foreign rocky blob/artifact protruding from a face/chin/mouth, a severely garbled face, or garbled gibberish text. "
                    "Do NOT flag intact normal glasses or stylized cartoon/puppet faces.",
                ),
            ]
            hits = []
            case_s = 0.0
            for pname, q in probes:
                res = query_vlm_json(
                    model=model,
                    system_prompt=M1_PROBE_SYSTEM,
                    user_text=q,
                    images=imgs,
                    json_schema=M1_PROBE_SCHEMA,
                    schema_name=f"m1_probe_{pname}",
                    max_edge=1024,
                    max_tokens=360,
                )
                case_s += res.elapsed_s
                if res.parse_error:
                    parse_err += 1
                pp = res.parsed or {}
                if bool(pp.get("defect_detected")) and str(pp.get("confidence", "")).lower() == "high":
                    hits.append(f"{pname}: {pp.get('defect_summary') or pp.get('literal_observation','')}")
            total_s += case_s
            pred = "FAIL" if hits else "PASS"
            exp = c["expected"]
            if exp == "FAIL":
                tp += int(pred == "FAIL")
                fn += int(pred == "PASS")
            else:
                tn += int(pred == "PASS")
                fp += int(pred == "FAIL")
            rows.append(
                {
                    "id": c["id"],
                    "category": c["category"],
                    "style": c["style"],
                    "defect_types": c["defect_types"],
                    "expected": exp,
                    "predicted": pred,
                    "correct": pred == exp,
                    "score": 3 if hits else 9,
                    "elapsed_s": round(case_s, 2),
                    "defects_found": hits,
                }
            )
            continue

        if use_quad_zoom:
            user_txt = (
                "Inspect IMAGE (full scene) and ZOOMED_QUADRANTS (close-up crops of the 4 quadrants of IMAGE). "
                "Check for: (1) duplicate/cloned identical character, (2) extra/missing/mangled limbs or hands, "
                "(3) impossible spatial clipping (body phasing through a solid table/workbench), "
                "(4) broken half-accessories or foreign blobs on faces. Remember stylized mediums are intentional."
            )
        else:
            user_txt = (
                "Inspect IMAGE carefully for genuine AI rendering defects (duplicate character, anatomy/limb errors, "
                "spatial clipping through solid furniture, broken half-accessories, face blobs, or hand-to-prop fusion). "
                "Do NOT penalize intentional quirky art styles."
            )

        res = query_vlm_json(
            model=model,
            system_prompt=sys_prompt,
            user_text=user_txt,
            images=imgs,
            json_schema=schema,
            schema_name=f"m1_{strategy}",
            max_edge=1024,
            max_tokens=768,
        )
        total_s += res.elapsed_s
        if res.parse_error:
            parse_err += 1
        p = res.parsed or {}
        dlist = [str(x).strip() for x in (p.get("defects_found") or []) if str(x).strip() and str(x).strip().lower() not in {"none", "n/a"}]
        pred_pass = bool(p.get("passed", True)) and (len(dlist) == 0)
        pred = "PASS" if pred_pass else "FAIL"
        exp = c["expected"]
        if exp == "FAIL":
            tp += int(pred == "FAIL")
            fn += int(pred == "PASS")
        else:
            tn += int(pred == "PASS")
            fp += int(pred == "FAIL")
        rows.append(
            {
                "id": c["id"],
                "category": c["category"],
                "style": c["style"],
                "defect_types": c["defect_types"],
                "expected": exp,
                "predicted": pred,
                "correct": pred == exp,
                "score": p.get("score"),
                "elapsed_s": res.elapsed_s,
                "defects_found": dlist,
            }
        )

    rec = round(tp / max(1, tp + fn), 4)
    fpr = round(fp / max(1, fp + tn), 4)
    acc = round((tp + tn) / max(1, len(cases)), 4)
    prec = round(tp / max(1, tp + fp), 4)
    f1 = round(2 * prec * rec / max(1e-9, prec + rec), 4)
    return {
        "metrics": {
            "total": len(cases),
            "tp": tp,
            "fn": fn,
            "tn": tn,
            "fp": fp,
            "recall": rec,
            "precision": prec,
            "fpr": fpr,
            "f1": f1,
            "accuracy": acc,
            "parse_errors": parse_err,
            "avg_s": round(total_s / max(1, len(cases)), 2),
        },
        "cases": rows,
    }


def run_m2(model: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = tn = fn = parse_err = 0
    total_s = 0.0
    rows = []

    for c in cases:
        ref_p = resolve_img(c["ref_image"])
        test_p = resolve_img(c["test_image"])
        etype = c["entity_type"]
        user_txt = (
            f"Verify the target {etype} in TEST_FRAME against CANONICAL_REFERENCE. "
            f"Describe CANONICAL_REFERENCE, describe the corresponding {etype} in TEST_FRAME, "
            f"list any clear identity, outfit, rendering-medium/style, or prop design mismatches in differences_found, "
            f"and set passed=(len(differences_found)==0)."
        )
        res = query_vlm_json(
            model=model,
            system_prompt=M2_SYSTEM,
            user_text=user_txt,
            images=[("CANONICAL_REFERENCE", ref_p), ("TEST_FRAME", test_p)],
            json_schema=M2_ANCHOR_SCHEMA,
            schema_name="m2_anchor_v3",
            max_edge=1024,
            max_tokens=640,
        )
        total_s += res.elapsed_s
        if res.parse_error:
            parse_err += 1
        p = res.parsed or {}
        diffs = [str(x).strip() for x in (p.get("differences_found") or []) if str(x).strip() and str(x).strip().lower() not in {"none", "n/a"}]
        pred_pass = bool(p.get("passed", True)) and (len(diffs) == 0)
        pred = "PASS" if pred_pass else "FAIL"
        exp = c["expected"]
        if exp == "FAIL":
            tp += int(pred == "FAIL")
            fn += int(pred == "PASS")
        else:
            tn += int(pred == "PASS")
            fp += int(pred == "FAIL")
        rows.append(
            {
                "id": c["id"],
                "entity_type": etype,
                "category": c["category"],
                "style": c["style"],
                "expected": exp,
                "predicted": pred,
                "correct": pred == exp,
                "score": p.get("score"),
                "elapsed_s": res.elapsed_s,
                "differences_found": diffs,
            }
        )

    rec = round(tp / max(1, tp + fn), 4)
    fpr = round(fp / max(1, fp + tn), 4)
    acc = round((tp + tn) / max(1, len(cases)), 4)
    prec = round(tp / max(1, tp + fp), 4)
    f1 = round(2 * prec * rec / max(1e-9, prec + rec), 4)
    return {
        "metrics": {
            "total": len(cases),
            "tp": tp,
            "fn": fn,
            "tn": tn,
            "fp": fp,
            "recall": rec,
            "precision": prec,
            "fpr": fpr,
            "f1": f1,
            "accuracy": acc,
            "parse_errors": parse_err,
            "avg_s": round(total_s / max(1, len(cases)), 2),
        },
        "cases": rows,
    }


def run_m3(model: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    tile_tp = tile_fp = tile_fn = exact = parse_err = 0
    total_s = 0.0
    rows = []

    for c in cases:
        img_p = resolve_img(c["image"])
        user_txt = (
            f"Audit this 2x2 contact sheet for {c['entity_type']}: {c.get('entity_name', 'target entity')}. "
            f"TILE 0 (top-left, gold border) is the canonical reference. Inspect TILE 1, TILE 2, and TILE 3 "
            f"against TILE 0 and list any outlier tile numbers (from [1, 2, 3]) in outlier_tiles."
        )
        res = query_vlm_json(
            model=model,
            system_prompt=M3_SYSTEM,
            user_text=user_txt,
            images=[("CONTACT_SHEET", img_p)],
            json_schema=M3_SHEET_SCHEMA,
            schema_name="m3_sheet_v3",
            max_edge=1024,
            max_tokens=640,
        )
        total_s += res.elapsed_s
        if res.parse_error:
            parse_err += 1
        p = res.parsed or {}
        raw_out = p.get("outlier_tiles") or []
        pred_set = {int(x) for x in raw_out if isinstance(x, (int, float)) and int(x) in {1, 2, 3}}
        exp_set = set(int(x) for x in (c.get("expected_outliers") or []))

        tile_tp += len(pred_set & exp_set)
        tile_fp += len(pred_set - exp_set)
        tile_fn += len(exp_set - pred_set)
        is_exact = pred_set == exp_set
        exact += int(is_exact)
        rows.append(
            {
                "id": c["id"],
                "entity_type": c["entity_type"],
                "style": c.get("world_style"),
                "expected_outliers": sorted(exp_set),
                "predicted_outliers": sorted(pred_set),
                "exact": is_exact,
                "elapsed_s": res.elapsed_s,
            }
        )

    rec = round(tile_tp / max(1, tile_tp + tile_fn), 4)
    prec = round(tile_tp / max(1, tile_tp + tile_fp), 4)
    f1 = round(2 * prec * rec / max(1e-9, prec + rec), 4)
    return {
        "metrics": {
            "total_sheets": len(cases),
            "tile_tp": tile_tp,
            "tile_fp": tile_fp,
            "tile_fn": tile_fn,
            "tile_recall": rec,
            "tile_precision": prec,
            "tile_f1": f1,
            "exact_sheet_rate": round(exact / max(1, len(cases)), 4),
            "parse_errors": parse_err,
            "avg_s": round(total_s / max(1, len(cases)), 2),
        },
        "cases": rows,
    }


def run_m4(model: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = tn = fn = parse_err = 0
    total_s = 0.0
    rows = []

    for c in cases:
        pa = resolve_img(c["shot_a"])
        pb = resolve_img(c["shot_b"])
        user_txt = (
            "Compare SHOT_A and SHOT_B as two consecutive beats in the SAME scene moments apart. "
            "Check lighting/time-of-day, central prop consistency, and character identity/medium consistency. "
            "List any jarring continuity breaks in breaks_found and set passed=(len(breaks_found)==0)."
        )
        res = query_vlm_json(
            model=model,
            system_prompt=M4_SYSTEM,
            user_text=user_txt,
            images=[("SHOT_A", pa), ("SHOT_B", pb)],
            json_schema=M4_ADJACENT_SCHEMA,
            schema_name="m4_adjacent_v3",
            max_edge=1024,
            max_tokens=640,
        )
        total_s += res.elapsed_s
        if res.parse_error:
            parse_err += 1
        p = res.parsed or {}
        breaks = [str(x).strip() for x in (p.get("breaks_found") or []) if str(x).strip() and str(x).strip().lower() not in {"none", "n/a"}]
        pred_pass = bool(p.get("passed", True)) and (len(breaks) == 0)
        pred = "PASS" if pred_pass else "FAIL"
        exp = c["expected"]
        if exp == "FAIL":
            tp += int(pred == "FAIL")
            fn += int(pred == "PASS")
        else:
            tn += int(pred == "PASS")
            fp += int(pred == "FAIL")
        rows.append(
            {
                "id": c["id"],
                "category": c["category"],
                "style": c["style"],
                "expected": exp,
                "predicted": pred,
                "correct": pred == exp,
                "score": p.get("score"),
                "elapsed_s": res.elapsed_s,
                "breaks_found": breaks,
            }
        )

    rec = round(tp / max(1, tp + fn), 4)
    fpr = round(fp / max(1, fp + tn), 4)
    acc = round((tp + tn) / max(1, len(cases)), 4)
    prec = round(tp / max(1, tp + fp), 4)
    f1 = round(2 * prec * rec / max(1e-9, prec + rec), 4)
    return {
        "metrics": {
            "total": len(cases),
            "tp": tp,
            "fn": fn,
            "tn": tn,
            "fp": fp,
            "recall": rec,
            "precision": prec,
            "fpr": fpr,
            "f1": f1,
            "accuracy": acc,
            "parse_errors": parse_err,
            "avg_s": round(total_s / max(1, len(cases)), 2),
        },
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run general v3 mined evals on Dev or Holdout split.")
    parser.add_argument("--split", choices=["dev", "holdout"], required=True)
    parser.add_argument("--strategy", choices=["baseline", "forensic", "probed_general"], default="forensic")
    parser.add_argument("--quad-zoom", action="store_true", help="Attach 2x2 zoomed quadrants on M1 single-frame suite")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--suites", nargs="+", default=["single_frame", "anchors", "contact_sheets", "adjacent_cuts"])
    parser.add_argument("--max-per-suite", type=int, default=0, help="0 = full split")
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    split_cases = build_split_cases(manifest, args.split, max_per_suite=args.max_per_suite)
    print(
        f"\n[mined_eval] split={args.split} strategy={args.strategy} quad_zoom={args.quad_zoom}\n"
        f"  Case counts: " + ", ".join(f"{k}={len(v)}" for k, v in split_cases.items() if k in args.suites),
        flush=True,
    )

    t0 = time.perf_counter()
    model_results: list[dict[str, Any]] = []

    try:
        for m in args.models:
            print(f"\n=== Model: {m} ===", flush=True)
            load_lms_model(m, context_length=args.context_length)
            suites_out: dict[str, Any] = {}

            if "single_frame" in args.suites:
                r = run_m1(m, split_cases["single_frame"], args.strategy, args.quad_zoom)
                suites_out["single_frame"] = r
                print(f"  [M1 single_frame]  {r['metrics']}", flush=True)
            if "anchors" in args.suites:
                r = run_m2(m, split_cases["anchors"])
                suites_out["anchors"] = r
                print(f"  [M2 anchors]       {r['metrics']}", flush=True)
            if "contact_sheets" in args.suites:
                r = run_m3(m, split_cases["contact_sheets"])
                suites_out["contact_sheets"] = r
                print(f"  [M3 contact_sheet] {r['metrics']}", flush=True)
            if "adjacent_cuts" in args.suites:
                r = run_m4(m, split_cases["adjacent_cuts"])
                suites_out["adjacent_cuts"] = r
                print(f"  [M4 adjacent_cuts] {r['metrics']}", flush=True)

            model_results.append({"model": m, "suites": suites_out})
            # Save incremental progress after every model
            args.out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "split": args.split,
                "strategy": args.strategy,
                "quad_zoom": args.quad_zoom,
                "max_per_suite": args.max_per_suite,
                "elapsed_s": round(time.perf_counter() - t0, 1),
                "models": model_results,
            }
            args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    finally:
        unload_all_lms()

    print(f"\n[mined_eval] Done in {round(time.perf_counter() - t0, 1)}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
