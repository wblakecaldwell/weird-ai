"""Eval Suite 1B: Decomposed Targeted-Probe Defect Detection (testing checked improvement over v1/v2).

Hypothesis: Monolithic "find any defect" prompts suffer severe false negatives on stylized cartoons
because VLMs rationalize subtle artifacts. Splitting inspection into 4 focused, literal yes/no visual
probes (plus canonical reference comparison when available) should dramatically boost defect recall
while keeping FPR low.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent.parent
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from vlm_client import load_lms_model, query_vlm_json, unload_all_lms  # noqa: E402

FIXTURES_DIR = EVALS_DIR / "fixtures" / "defects"

PROBE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "observation": {"type": "string"},
        "defect_detected": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "location": {"type": "string"},
        "fix_instruction": {"type": "string"},
    },
    "required": ["observation", "defect_detected", "confidence", "location", "fix_instruction"],
    "additionalProperties": False,
}

PROBES = [
    (
        "half_glasses",
        "accessory",
        (
            "Look ONLY at the left and right edges of every character's face, cheeks, and temples. "
            "Ignore complete normal eyeglasses that cover BOTH eyes evenly. "
            "Is there a BROKEN or HALF pair of glasses — specifically a single hollow lens, triangular/rectangular "
            "frame loop, or glasses rim sticking out on ONLY ONE side of a character's face/cheek with NO matching lens on the other eye? "
            "Set defect_detected=true ONLY with high confidence if you literally see a partial/one-sided floating glasses loop."
        ),
    ),
    (
        "chin_blob",
        "anatomy",
        (
            "Look ONLY at every character's chin, lower jaw, and mouth. "
            "Is there a strange jagged red/brown rocky lump, blob, or foreign AI artifact protruding directly out of a character's chin or mouth "
            "where their face touches a globe or object? "
            "Do NOT flag normal cartoon chins, open mouths, or hands holding tools. "
            "Set defect_detected=true ONLY with high confidence if a weird foreign rocky object is literally attached to or sticking out of a chin."
        ),
    ),
    (
        "extra_leg",
        "anatomy",
        (
            "Carefully count the legs, feet, and shoes on EVERY standing character in the image. "
            "Look especially near the waist/hip/side of any fuzzy yellow-green puppet character. "
            "Does ANY character have THREE legs or an extra raised leg/green shoe sticking sideways out of their torso/hip while two other feet are already on the floor? "
            "Set defect_detected=true ONLY with high confidence if you literally count 3 legs/shoes on one character."
        ),
    ),
    (
        "hand_merge",
        "merge_artifact",
        (
            "Look ONLY at every character's hands and arms where they touch wooden beams, poles, or large props. "
            "Is any character's hand or arm physically FUSED, MELTED, or seamlessly morphed INTO a wooden beam or pole so the limb turns into the wood? "
            "Normal gripping or touching a prop cleanly is NOT a defect. "
            "Set defect_detected=true ONLY with high confidence if a hand/arm clearly merges into wood."
        ),
    ),
]

SYSTEM_PROMPT = (
    "You are a literal forensic pixel inspector checking ONE specific AI artifact question at a time. "
    "Answer strictly from what you actually see in the pixels. Do NOT guess or invent defects. "
    "Describe what you literally see in `observation` first, then set `defect_detected` and `confidence`."
)


def run_probed_suite(model: str, max_edge: int = 1024) -> dict[str, Any]:
    manifest = json.loads((FIXTURES_DIR / "labels.json").read_text(encoding="utf-8"))
    cases = manifest["cases"]

    results = []
    tp = fp = tn = fn = 0
    total_time = 0.0

    for case in cases:
        img_path = FIXTURES_DIR / case["image"]
        hits = []
        case_time = 0.0
        probe_outputs = {}
        for pname, dtype, question in PROBES:
            res = query_vlm_json(
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_text=question,
                images=[("IMAGE", img_path)],
                json_schema=PROBE_SCHEMA,
                schema_name=f"probe_{pname}",
                max_edge=max_edge,
                max_tokens=320,
            )
            case_time += res.elapsed_s
            p = res.parsed or {}
            probe_outputs[pname] = p
            if bool(p.get("defect_detected")) and str(p.get("confidence", "")).lower() == "high":
                hits.append((pname, dtype, p))
        total_time += case_time

        pred_label = "FAIL" if hits else "PASS"
        expected = case["expected"]
        if expected == "FAIL":
            tp += int(pred_label == "FAIL")
            fn += int(pred_label == "PASS")
        else:
            tn += int(pred_label == "PASS")
            fp += int(pred_label == "FAIL")

        correct = pred_label == expected
        hit_summary = "; ".join(f"{h[0]}: {str(h[2].get('observation',''))[:60]}" for h in hits) or "clean"
        print(
            f"  [probed:{model}] {case['id']:28s} exp={expected:4s} pred={pred_label:4s} "
            f"({'OK' if correct else 'MISS'}) {case_time:4.1f}s | {hit_summary[:85]}"
        )
        results.append(
            {
                "id": case["id"],
                "expected": expected,
                "predicted": pred_label,
                "correct": correct,
                "hits": [h[0] for h in hits],
                "elapsed_s": round(case_time, 2),
                "probes": probe_outputs,
            }
        )

    recall = round(tp / max(1, tp + fn), 4)
    fpr = round(fp / max(1, fp + tn), 4)
    acc = round((tp + tn) / max(1, len(cases)), 4)
    metrics = {
        "total": len(cases),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "defect_recall": recall,
        "fpr": fpr,
        "accuracy": acc,
        "avg_s": round(total_time / max(1, len(cases)), 2),
        "meets_bar": recall >= 0.90 and fpr <= 0.15,
    }
    return {"suite": "defects_probed", "model": model, "metrics": metrics, "cases": results}


def main() -> None:
    out_all = []
    for m in ["gemma-4-31b-it", "qwen3-vl-8b-instruct"]:
        print(f"\n=== Probed Defect Eval: {m} ===")
        load_lms_model(m, context_length=16384)
        res = run_probed_suite(m)
        print("  METRICS:", res["metrics"])
        out_all.append(res)
    unload_all_lms()
    out_p = EVALS_DIR / "results" / "suite1b_probed_defects.json"
    out_p.write_text(json.dumps(out_all, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_p}")


if __name__ == "__main__":
    main()
