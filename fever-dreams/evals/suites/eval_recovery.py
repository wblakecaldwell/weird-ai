"""Eval Suite 6: Closed-Loop Bad-to-Good Recovery Test.

Starts directly from known poorly generated images (half-glasses crop, 3-legged Winston,
mismatched green crane truck, rocky chin artifact, hand merged into beam) and verifies the
full automated loop:
  Critic detects defect -> Diagnoses & picks tool -> Invokes local mflux repair -> Re-audits -> Improves to PASS.

Usage:
    uv run python evals/suites/eval_recovery.py --critic qwen3-vl-30b-a3b-instruct-mlx --edit-model qwen-image-2.1-turbo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = EVALS_DIR.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
for p in (SCRIPTS_DIR, EVALS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from critic_loop import RepairItem, run_closed_loop  # noqa: E402

DEFECTS_DIR = EVALS_DIR / "fixtures" / "defects"
ANCHORS_DIR = EVALS_DIR / "fixtures" / "anchors"
RESULTS_DIR = EVALS_DIR / "results" / "suite6_recovery"


def build_recovery_items() -> list[RepairItem]:
    items: list[RepairItem] = []

    hg = DEFECTS_DIR / "fail_half_glasses_crop.jpg"
    if hg.exists():
        items.append(
            RepairItem(
                item_id="recover_half_glasses",
                image_path=hg,
                description="User-supplied half-glasses artifact: green clay boy with a hollow broken half-glasses loop floating on the left side of his face.",
            )
        )

    items.extend(
        [
            RepairItem(
                item_id="recover_p11_three_legs",
                image_path=DEFECTS_DIR / "fail_p11_three_legs.png",
                description="Book 1 p11 original: Winston Crumb has an extra raised third leg sticking out of his side.",
                entity_name="Winston Crumb (chartreuse felt puppet with two legs)",
                ref_images=[ANCHORS_DIR / "ref_winston.webp"],
            ),
            RepairItem(
                item_id="recover_p12_wrong_crane",
                image_path=ANCHORS_DIR / "test_p12_wrong_crane.png",
                description="Book 1 p12 original: green steam-train contraption instead of the canonical yellow wheeled construction crane truck.",
                entity_name="yellow wheeled mobile construction crane truck",
                ref_images=[ANCHORS_DIR / "ref_crane.png"],
            ),
            RepairItem(
                item_id="recover_cover_chester_drift",
                image_path=ANCHORS_DIR / "test_cover_wrong_chester.png",
                description="Book 1 cover original: smooth clay horned boy in glasses instead of canonical 16-bit pixel-art Chester (no horns, no glasses).",
                entity_name="Chester Antialiased (16-bit retro pixel-art green goblin boy, NO glasses, NO horns)",
                ref_images=[ANCHORS_DIR / "ref_chester.webp"],
            ),
            RepairItem(
                item_id="recover_p17_chester_style",
                image_path=ANCHORS_DIR / "test_p17_3d_chester.png",
                description="Book 1 p17 original: smooth 3D shaded green boy instead of flat 2D 16-bit pixel-art Chester.",
                entity_name="Chester Antialiased (flat 2D 16-bit retro pixel-art green goblin boy)",
                ref_images=[ANCHORS_DIR / "ref_chester.webp"],
            ),
        ]
    )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Eval Suite 6: Closed-Loop Bad-to-Good Recovery.")
    parser.add_argument("--critic", default="gemma-4-31b-it")
    parser.add_argument("--fast-screener", default=None)
    parser.add_argument("--edit-model", default="qwen-image-2.1-turbo")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--work-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    items = build_recovery_items()
    summary = run_closed_loop(
        items,
        work_dir=args.work_dir,
        critic_model=args.critic,
        fast_screener_model=args.fast_screener,
        edit_model_key=args.edit_model,
        max_rounds=args.max_rounds,
        candidates_per_round=args.candidates,
    )
    print("\n=== SUITE 6 RECOVERY SUMMARY ===")
    print(
        json.dumps(
            {
                "initial_detected_bad": summary["initial_detected_bad"],
                "recovered_to_pass": summary["recovered_to_pass"],
                "recovery_rate": summary["recovery_rate"],
                "avg_score_delta": summary["avg_score_delta"],
                "total_elapsed_s": summary["total_elapsed_s"],
                "report": str(args.work_dir / "report.html"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
