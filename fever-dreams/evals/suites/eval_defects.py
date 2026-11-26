"""Eval Suite 1 (v2 Observe-First): Single-frame rendering, anatomy, and accessory defect detection.

Key design upgrade over v1:
  Schema forces the VLM to OBSERVE FIRST (face/glasses/chin check + limb/hand/merge check +
  explicit defects_found list) BEFORE outputting `passed`. Eliminates premature PASS bias in
  strict JSON schema decoding.
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

FIXTURES_DIR = EVALS_DIR / "fixtures" / "defects"

# IMPORTANT: Observation fields come FIRST so autoregressive JSON decoding inspects before judging!
DEFECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "face_and_accessories_check": {"type": "string"},
        "limbs_and_prop_contact_check": {"type": "string"},
        "defects_found": {
            "type": "array",
            "items": {"type": "string"},
        },
        "passed": {"type": "boolean"},
        "score": {"type": "integer"},
        "defect_type": {
            "type": "string",
            "enum": ["none", "anatomy", "accessory", "merge_artifact", "other"],
        },
        "defect_summary": {"type": "string"},
        "location": {"type": "string"},
        "fix_action": {
            "type": "string",
            "enum": ["none", "targeted_edit", "masked_inpaint", "reroll_seed"],
        },
        "fix_instruction": {"type": "string"},
    },
    "required": [
        "face_and_accessories_check",
        "limbs_and_prop_contact_check",
        "defects_found",
        "passed",
        "score",
        "defect_type",
        "defect_summary",
        "location",
        "fix_action",
        "fix_instruction",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a forensic visual quality-control (QC) inspector for a mixed-media children's storybook.

INTENTIONAL STYLE (NEVER FLAG THESE AS DEFECTS):
- Pixel-art / blocky 16-bit sprite characters (like a green pixel goblin boy) are INTENTIONAL.
- Fuzzy felt puppets with asymmetrical plastic googly eyes are INTENTIONAL.
- Knit/clay owl characters wearing complete round spectacles (both lenses connected by a bridge) are INTENTIONAL.
- Painted 1950s gouache backgrounds and stylized cartoon hands/fingers are INTENTIONAL.

GENUINE AI RENDERING DEFECTS (MUST RECORD IN defects_found AND SET passed = false):
1. BROKEN/HALF ACCESSORIES: e.g., a hollow single lens or half-glasses frame stuck to ONE side of a character's cheek/temple with no matching lens on the other eye.
2. FOREIGN FACE/CHIN ARTIFACTS: e.g., a weird rocky red/brown lump, blob, or jagged object protruding out of a character's chin or mouth against a prop.
3. EXTRA OR MISSING LIMBS: e.g., count every character's legs and shoes — a character with 3 legs or an extra shoe sticking out of their side is a BLOCKING FAIL.
4. LIMB-TO-PROP MERGING: e.g., a character's arm or hand physically fusing/melting into a wooden beam or solid pole so you cannot tell where the arm ends and the wood begins.
5. MISSING HEADS or severely garbled faces.

CRITICAL DISCIPLINE:
- Perform both checks carefully BEFORE deciding `passed`.
- If a character's glasses are intact (both lenses present), do NOT invent a broken-glasses defect.
- If a character clearly has only 2 legs and clean hands, do NOT invent an anatomy defect.
- If you find ANY real defect from 1–5 above, list it in `defects_found` and set `passed = false`.
- Only set `passed = true` when `defects_found` is empty `[]`.

Return ONLY valid JSON matching the schema."""

USER_PROMPT = """Inspect IMAGE forensically step-by-step:
1. face_and_accessories_check: Inspect every character's face, eyes, cheeks, chin, and glasses. Check specifically for any hollow half-glasses loop on one side of a face, or any weird red/rocky blob stuck to a chin/mouth.
2. limbs_and_prop_contact_check: Count the legs/shoes and arms/hands on every character. Check specifically for an extra 3rd leg/shoe sticking out of a body, or any hand/arm melting/fusing into a wooden beam or prop.
3. defects_found: List ONLY concrete AI glitches observed in steps 1–2 (or [] if clean).
4. Set passed = (len(defects_found) == 0), score (1-5 if defective, 8-10 if clean), defect_type, defect_summary, location, fix_action, and fix_instruction."""


def run_suite(model: str, max_edge: int = 1024) -> dict[str, Any]:
    manifest = json.loads((FIXTURES_DIR / "labels.json").read_text(encoding="utf-8"))
    cases = manifest["cases"]

    results: list[dict[str, Any]] = []
    tp = fp = tn = fn = 0
    keyword_hits = 0
    fail_total = 0
    parse_errors = 0
    total_time = 0.0

    for case in cases:
        img_path = FIXTURES_DIR / case["image"]
        res = query_vlm_json(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_text=USER_PROMPT,
            images=[("IMAGE", img_path)],
            json_schema=DEFECT_SCHEMA,
            schema_name="defect_qc_v2",
            max_edge=max_edge,
            max_tokens=1024,
        )
        total_time += res.elapsed_s
        if res.parse_error:
            parse_errors += 1

        parsed = res.parsed or {}
        defects_list = [str(x).strip() for x in (parsed.get("defects_found") or []) if str(x).strip() and str(x).strip().lower() != "none"]
        # Enforce consistency: if defects_found is non-empty, passed is False
        pred_passed = bool(parsed.get("passed", True)) and (len(defects_list) == 0)
        pred_label = "PASS" if pred_passed else "FAIL"
        expected = case["expected"]

        if expected == "FAIL":
            fail_total += 1
            if pred_label == "FAIL":
                tp += 1
            else:
                fn += 1
        else:
            if pred_label == "PASS":
                tn += 1
            else:
                fp += 1

        diag_text = " ".join(
            [
                str(parsed.get("defect_summary", "")),
                str(parsed.get("location", "")),
                str(parsed.get("fix_instruction", "")),
                " ".join(defects_list),
            ]
        ).lower()
        kw_hit = False
        if expected == "FAIL" and pred_label == "FAIL":
            kws = case.get("expected_keywords") or []
            kw_hit = any(k.lower() in diag_text for k in kws) if kws else True
            if kw_hit:
                keyword_hits += 1

        correct = pred_label == expected
        results.append(
            {
                "id": case["id"],
                "image": str(img_path.relative_to(EVALS_DIR.parent)),
                "expected": expected,
                "predicted": pred_label,
                "correct": correct,
                "keyword_hit": kw_hit if expected == "FAIL" else None,
                "elapsed_s": res.elapsed_s,
                "used_schema": res.used_schema,
                "parse_error": res.parse_error,
                "output": parsed,
                "description": case["description"],
            }
        )
        status = "OK" if correct else "MISS"
        print(
            f"  [defects:{model}] {case['id']:28s} exp={expected:4s} pred={pred_label:4s} "
            f"({status}) {res.elapsed_s:4.1f}s | {str(parsed.get('defect_summary',''))[:72]}"
        )

    recall = round(tp / max(1, tp + fn), 4)
    fpr = round(fp / max(1, fp + tn), 4)
    accuracy = round((tp + tn) / max(1, len(cases)), 4)
    loc_rate = round(keyword_hits / max(1, fail_total), 4)
    avg_s = round(total_time / max(1, len(cases)), 2)

    return {
        "suite": "defects",
        "model": model,
        "metrics": {
            "total": len(cases),
            "tp": tp,
            "fn": fn,
            "tn": tn,
            "fp": fp,
            "defect_recall": recall,
            "fpr": fpr,
            "accuracy": accuracy,
            "localization_rate": loc_rate,
            "parse_errors": parse_errors,
            "avg_s": avg_s,
            "total_s": round(total_time, 2),
            "meets_bar": recall >= 0.90 and fpr <= 0.15,
        },
        "cases": results,
    }
