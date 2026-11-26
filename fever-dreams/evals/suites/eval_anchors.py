"""Eval Suite 2: Hub-and-spoke canonical reference matching (character & prop continuity)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent.parent
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from vlm_client import query_vlm_json  # noqa: E402

FIXTURES_DIR = EVALS_DIR / "fixtures" / "anchors"

ANCHOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "score": {"type": "integer"},
        "mismatch_type": {
            "type": "string",
            "enum": ["none", "style_drift", "color_drift", "prop_drift", "accessory_drift", "anatomy"],
        },
        "mismatch_summary": {"type": "string"},
        "fix_instruction": {"type": "string"},
    },
    "required": ["passed", "score", "mismatch_type", "mismatch_summary", "fix_instruction"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a visual continuity supervisor checking whether a target character or prop in TEST_FRAME matches CANONICAL_REFERENCE.

RULES:
- Camera angle, pose, facial expression, background, and other characters naturally change between shots — DO NOT fail for normal pose/camera/background differences.
- FAIL (passed = false) if the target entity itself drifts from CANONICAL_REFERENCE in any major way:
  1. Rendering medium / style drift (for example, a flat 2D chunky 16-bit pixel-art character turning into a smooth 3D shaded clay character).
  2. Major color drift (for example, a chartreuse yellow-green puppet turning blue/purple, or a yellow crane turning green).
  3. Wrong object/prop design (for example, a yellow wheeled construction crane truck replaced by a steam-train contraption).
  4. Added/missing permanent features or broken accessories (for example, horns or glasses added when the canonical character has none, or half a glasses frame).
  5. Severe anatomy error on the entity.

Return ONLY valid JSON matching the schema."""


def run_suite(model: str, max_edge: int = 1024) -> dict[str, Any]:
    manifest = json.loads((FIXTURES_DIR / "labels.json").read_text(encoding="utf-8"))
    cases = manifest["cases"]

    results: list[dict[str, Any]] = []
    tp = fp = tn = fn = 0
    parse_errors = 0
    total_time = 0.0

    for case in cases:
        ref_path = FIXTURES_DIR / case["ref_image"]
        test_path = FIXTURES_DIR / case["test_image"]
        user_prompt = (
            f"Target {case['entity_type']} to verify: {case['entity_name']}.\n"
            f"Compare this target entity in TEST_FRAME against CANONICAL_REFERENCE.\n"
            f"Does the target entity in TEST_FRAME faithfully match CANONICAL_REFERENCE in rendering medium/style, colors, design, and key features?\n"
            f"Set passed=true if it matches well enough for continuity, or passed=false if it drifted or has a mismatch/defect."
        )
        res = query_vlm_json(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_text=user_prompt,
            images=[("CANONICAL_REFERENCE", ref_path), ("TEST_FRAME", test_path)],
            json_schema=ANCHOR_SCHEMA,
            schema_name="anchor_qc",
            max_edge=max_edge,
        )
        total_time += res.elapsed_s
        if res.parse_error:
            parse_errors += 1

        parsed = res.parsed or {}
        pred_passed = bool(parsed.get("passed", True))
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
                "ref_image": str(ref_path.relative_to(EVALS_DIR.parent)),
                "test_image": str(test_path.relative_to(EVALS_DIR.parent)),
                "entity_name": case["entity_name"],
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
            f"  [anchors:{model}] {case['id']:28s} exp={expected:4s} pred={pred_label:4s} "
            f"({status}) {res.elapsed_s:4.1f}s | {str(parsed.get('mismatch_summary',''))[:70]}"
        )

    recall = round(tp / max(1, tp + fn), 4)
    fpr = round(fp / max(1, fp + tn), 4)
    accuracy = round((tp + tn) / max(1, len(cases)), 4)
    avg_s = round(total_time / max(1, len(cases)), 2)

    return {
        "suite": "anchors",
        "model": model,
        "metrics": {
            "total": len(cases),
            "tp": tp,
            "fn": fn,
            "tn": tn,
            "fp": fp,
            "mismatch_recall": recall,
            "fpr": fpr,
            "accuracy": accuracy,
            "parse_errors": parse_errors,
            "avg_s": avg_s,
            "total_s": round(total_time, 2),
            "meets_bar": recall >= 0.90 and fpr <= 0.10,
        },
        "cases": results,
    }
