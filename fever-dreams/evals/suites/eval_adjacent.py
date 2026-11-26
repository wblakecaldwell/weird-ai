"""Eval Suite 4 (v2 Observe-First): Adjacent-cut and set-reuse continuity verification.

Upgrades over v1:
  1. Forces explicit observation of location, lighting, shared prop, and shared character medium BEFORE `passed`.
  2. Explicitly permits normal storybook cut variation (camera angle, zoom, different corner of the same
     classroom/street, characters entering/leaving) so models stop false-failing valid consecutive pages.
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

FIXTURES_DIR = EVALS_DIR / "fixtures" / "adjacent_cuts"

ADJACENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "location_and_lighting_comparison": {"type": "string"},
        "shared_prop_and_character_style_comparison": {"type": "string"},
        "major_break_found": {"type": "boolean"},
        "passed": {"type": "boolean"},
        "score": {"type": "integer"},
        "break_type": {
            "type": "string",
            "enum": ["none", "lighting_jump", "prop_drift", "style_drift", "set_swap"],
        },
        "break_summary": {"type": "string"},
    },
    "required": [
        "location_and_lighting_comparison",
        "shared_prop_and_character_style_comparison",
        "major_break_found",
        "passed",
        "score",
        "break_type",
        "break_summary",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a continuity supervisor comparing two consecutive storybook pages: SHOT_PREV followed by SHOT_NEXT in the SAME scene.

NORMAL STORYBOOK CUT CHANGES THAT MUST PASS (major_break_found = false, passed = true):
- Camera moving from wide shot to close-up, looking at a different wall/corner of the same classroom or further down the same street.
- Different characters entering/leaving frame, changing poses/expressions, or slight natural painted-storybook variation in background details.
- Mixed-media characters (a 16-bit pixel goblin, felt puppet, knit owl) living inside a painted 1950s world is INTENTIONAL — do NOT call mixed-media characters a style break if the same character keeps the same medium across both shots!

ONLY FAIL (major_break_found = true, passed = false) FOR CLEAR, IMPOSSIBLE CONTINUITY BREAKS:
1. lighting_jump: Warm bright daylight abruptly flipping to dark cold blue night in the same scene.
2. prop_drift: The central vehicle/prop mutating into a completely different machine (e.g., yellow wheeled construction crane truck turning into a green steam-train contraption).
3. style_drift: The SAME recurring character abruptly switching rendering medium (e.g., flat 2D blocky 16-bit pixel goblin in SHOT_PREV turning into a smooth 3D shaded clay boy in SHOT_NEXT).
4. set_swap: Completely impossible location teleport (e.g., indoor classroom jumping to an outdoor city street when tagged as the same indoor room).

Compare location/lighting and shared props/character styles FIRST before deciding `passed`. Return ONLY valid JSON."""


def run_suite(model: str, max_edge: int = 1024) -> dict[str, Any]:
    manifest = json.loads((FIXTURES_DIR / "labels.json").read_text(encoding="utf-8"))
    cases = manifest["cases"]

    results: list[dict[str, Any]] = []
    tp = fp = tn = fn = 0
    parse_errors = 0
    total_time = 0.0

    for case in cases:
        prev_path = FIXTURES_DIR / case["shot_prev"]
        next_path = FIXTURES_DIR / case["shot_next"]
        user_prompt = (
            f"Scene context: {case['scene_context']}\n"
            f"Compare SHOT_PREV -> SHOT_NEXT:\n"
            f"1. location_and_lighting_comparison: Are both shots in the same general environment (both indoor classroom OR both outdoor street) with compatible daytime/nighttime lighting? Remember normal camera moves to another part of the same room/street PASS.\n"
            f"2. shared_prop_and_character_style_comparison: Do any shared vehicles/props keep the same core design/color, and do any shared characters keep the same rendering medium (pixel art stays pixel art, felt stays felt)?\n"
            f"3. Set major_break_found=true and passed=false ONLY if there is a clear lighting_jump, prop_drift, style_drift, or set_swap."
        )
        res = query_vlm_json(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_text=user_prompt,
            images=[("SHOT_PREV", prev_path), ("SHOT_NEXT", next_path)],
            json_schema=ADJACENT_SCHEMA,
            schema_name="adjacent_cut_qc_v2",
            max_edge=max_edge,
            max_tokens=1024,
        )
        total_time += res.elapsed_s
        if res.parse_error:
            parse_errors += 1

        parsed = res.parsed or {}
        pred_passed = bool(parsed.get("passed", True)) and not bool(parsed.get("major_break_found", False))
        pred_label = "PASS" if pred_passed else "FAIL"
        expected = case["expected"]

        if expected == "FAIL":
            if pred_label == "FAIL":
                tp += 1
            else:
                fn += 1
        else:
            if pred_label == "PASS":
                tn += 1
            else:
                fp += 1

        correct = pred_label == expected
        results.append(
            {
                "id": case["id"],
                "shot_prev": str(prev_path.relative_to(EVALS_DIR.parent)),
                "shot_next": str(next_path.relative_to(EVALS_DIR.parent)),
                "expected": expected,
                "predicted": pred_label,
                "correct": correct,
                "elapsed_s": res.elapsed_s,
                "used_schema": res.used_schema,
                "parse_error": res.parse_error,
                "output": parsed,
                "description": case["description"],
            }
        )
        status = "OK" if correct else "MISS"
        print(
            f"  [adjacent:{model}] {case['id']:30s} exp={expected:4s} pred={pred_label:4s} "
            f"({status}) {res.elapsed_s:4.1f}s | {str(parsed.get('break_summary',''))[:70]}"
        )

    recall = round(tp / max(1, tp + fn), 4)
    fpr = round(fp / max(1, fp + tn), 4)
    accuracy = round((tp + tn) / max(1, len(cases)), 4)
    avg_s = round(total_time / max(1, len(cases)), 2)

    return {
        "suite": "adjacent_cuts",
        "model": model,
        "metrics": {
            "total": len(cases),
            "tp": tp,
            "fn": fn,
            "tn": tn,
            "fp": fp,
            "break_recall": recall,
            "fpr": fpr,
            "accuracy": accuracy,
            "parse_errors": parse_errors,
            "avg_s": avg_s,
            "total_s": round(total_time, 2),
            "meets_bar": recall >= 0.85 and fpr <= 0.15,
        },
        "cases": results,
    }
