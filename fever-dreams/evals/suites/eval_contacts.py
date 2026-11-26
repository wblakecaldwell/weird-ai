"""Eval Suite 3 (v2 Observe-First): Numbered contact-sheet outlier detection against Tile 0 reference.

Upgrades over v1:
  1. Forces per-tile observation (`tile_checks`) BEFORE deciding `outlier_tiles`.
  2. Explicitly instructs the VLM that TILE 0 is a plain-background reference cutout while Tiles 1..N
     are full storybook scenes — never flag a tile merely for having a background, pose change, or
     other characters in frame.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent.parent
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from vlm_client import query_vlm_json  # noqa: E402

FIXTURES_DIR = EVALS_DIR / "fixtures" / "contact_sheets"

CONTACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tile_checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tile": {"type": "integer"},
                    "matches_tile_0": {"type": "boolean"},
                    "note": {"type": "string"},
                },
                "required": ["tile", "matches_tile_0", "note"],
                "additionalProperties": False,
            },
        },
        "outlier_tiles": {
            "type": "array",
            "items": {"type": "integer"},
        },
        "reasons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tile": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["tile", "reason"],
                "additionalProperties": False,
            },
        },
        "all_consistent": {"type": "boolean"},
    },
    "required": ["tile_checks", "outlier_tiles", "reasons", "all_consistent"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a continuity supervisor inspecting a numbered contact sheet for ONE target character or prop.

CRITICAL CONTEXT — DO NOT MAKE THIS MISTAKE:
- TILE 0 (REF) is a cropped portrait or prop reference on a plain/simple background.
- TILE 1, TILE 2, ... are FULL STORYBOOK SCENES containing that entity in different poses, camera distances, backgrounds, and alongside OTHER characters.
- NEVER flag a tile as an outlier just because it has a storybook background, full-body framing, different pose, or other characters next to it!

WHAT ACTUALLY MAKES A TILE AN OUTLIER (matches_tile_0 = false):
1. Wrong rendering medium/style compared to TILE 0 (e.g., TILE 0 is flat 2D chunky 16-bit pixel art, while a tile shows a smooth 3D shaded clay character).
2. Wrong main color palette for the target entity (e.g., chartreuse yellow-green puppet turned blue/purple, or yellow crane turned green).
3. Wrong prop/vehicle type (e.g., TILE 0 is a yellow wheeled mobile construction crane truck, while a tile shows a green steam-train contraption).
4. Wrong/broken features or accessories on the target entity (e.g., horns or glasses or half a glasses frame added when TILE 0 has none).
5. Major anatomy glitch on the target entity (e.g., count the legs — an extra 3rd leg sticking out of the puppet's side).

Evaluate EVERY tile 1..N individually in `tile_checks` FIRST, comparing ONLY the target entity itself against TILE 0.
Then set `outlier_tiles` to the exact list of tile integers where `matches_tile_0` is false (or `[]` if all match). NEVER include 0."""


def run_suite(model: str, max_edge: int = 1280) -> dict[str, Any]:
    manifest = json.loads((FIXTURES_DIR / "labels.json").read_text(encoding="utf-8"))
    cases = manifest["cases"]

    results: list[dict[str, Any]] = []
    tile_tp = tile_fp = tile_fn = 0
    exact_matches = 0
    parse_errors = 0
    total_time = 0.0

    for case in cases:
        sheet_path = FIXTURES_DIR / case["image"]
        tc = int(case["tile_count"])
        user_prompt = (
            f"Target entity to check across tiles: {case['entity_name']}.\n"
            f"TILE 0 (REF) is the canonical reference. Inspect TILE 1 through TILE {tc} one by one in `tile_checks`.\n"
            f"Ignore background differences and other characters; check ONLY whether the target entity in each tile "
            f"matches TILE 0 in rendering medium/style, main colors, vehicle/prop design, accessories, and limb count.\n"
            f"Put any non-matching tile numbers (1..{tc}) into `outlier_tiles` (or [] if all tiles are consistent)."
        )
        res = query_vlm_json(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_text=user_prompt,
            images=[("CONTACT_SHEET", sheet_path)],
            json_schema=CONTACT_SCHEMA,
            schema_name="contact_sheet_qc_v2",
            max_edge=max_edge,
            max_tokens=1024,
        )
        total_time += res.elapsed_s
        if res.parse_error:
            parse_errors += 1

        parsed = res.parsed or {}
        raw_tiles = parsed.get("outlier_tiles") or []
        pred_set = {int(t) for t in raw_tiles if isinstance(t, (int, float)) and 1 <= int(t) <= tc}
        exp_set = {int(t) for t in case["expected_outliers"]}

        tile_tp += len(pred_set & exp_set)
        tile_fp += len(pred_set - exp_set)
        tile_fn += len(exp_set - pred_set)
        exact = pred_set == exp_set
        if exact:
            exact_matches += 1

        results.append(
            {
                "id": case["id"],
                "image": str(sheet_path.relative_to(EVALS_DIR.parent)),
                "entity_name": case["entity_name"],
                "expected_outliers": sorted(exp_set),
                "predicted_outliers": sorted(pred_set),
                "exact_match": exact,
                "elapsed_s": res.elapsed_s,
                "used_schema": res.used_schema,
                "parse_error": res.parse_error,
                "output": parsed,
                "description": case["description"],
            }
        )
        status = "EXACT" if exact else "DIFF"
        print(
            f"  [contacts:{model}] {case['id']:30s} exp={sorted(exp_set)} pred={sorted(pred_set)} "
            f"({status}) {res.elapsed_s:4.1f}s"
        )

    tile_recall = round(tile_tp / max(1, tile_tp + tile_fn), 4)
    tile_precision = round(tile_tp / max(1, tile_tp + tile_fp), 4) if (tile_tp + tile_fp) > 0 else 1.0
    exact_rate = round(exact_matches / max(1, len(cases)), 4)
    avg_s = round(total_time / max(1, len(cases)), 2)

    return {
        "suite": "contact_sheets",
        "model": model,
        "metrics": {
            "total_sheets": len(cases),
            "tile_tp": tile_tp,
            "tile_fp": tile_fp,
            "tile_fn": tile_fn,
            "tile_recall": tile_recall,
            "tile_precision": tile_precision,
            "exact_sheet_rate": exact_rate,
            "parse_errors": parse_errors,
            "avg_s": avg_s,
            "total_s": round(total_time, 2),
            "meets_bar": tile_recall >= 0.90 and tile_precision >= 0.85,
        },
        "cases": results,
    }
