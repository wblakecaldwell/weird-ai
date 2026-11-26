"""Eval Suite 5: Visual Bible & Single-Render Set-Plate Reuse Bake-Off across local mflux models.

Generates 1 character turnaround sheet, 1 recurring prop sheet, and 1 empty master set plate ONCE,
renders 3 shots reusing that exact master set plate, stitches numbered contact sheets, and scores
visual quality + continuity automatically with a calibrated local LM Studio Critic VLM.

Usage:
    uv run python evals/suites/eval_refgen.py --models qwen-image-2.1-turbo flux2-klein-4b --critic qwen3-vl-30b-a3b-instruct-mlx
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = EVALS_DIR.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
for p in (SCRIPTS_DIR, EVALS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from critic_loop import audit_single_image  # noqa: E402
from roster import DEFAULT_ROSTER_PATH, load_roster, resolve_project_path  # noqa: E402
from seed_fixtures import make_contact_sheet  # noqa: E402
from visual_bible import generate_bible_and_shots  # noqa: E402
from vlm_client import load_lms_model, query_vlm_json, unload_all_lms  # noqa: E402

RESULTS_DIR = EVALS_DIR / "results" / "suite5_refgen"


def build_test_spec() -> dict[str, Any]:
    """Pick a fresh Meme Valley senior (excluding Book 1 leads) and build a 3-shot set-reuse test."""
    roster = load_roster(DEFAULT_ROSTER_PATH)
    excluded = {"chester", "marisol", "winston"}
    chosen = None
    for c in roster:
        cid = str(c.get("id", "")).lower()
        if not any(ex in cid for ex in excluded):
            p = resolve_project_path(c.get("portrait_raw") or c.get("portrait_final"))
            if p and p.exists():
                chosen = dict(c)
                chosen["portrait_path"] = str(p)
                break
    if not chosen:
        raise RuntimeError("Could not find a fresh character with portrait in roster.json.")

    return {
        "cast": [chosen],
        "props": [
            {
                "id": "yellow_crane_toy",
                "name": "bright yellow wheeled toy crane truck",
                "description": (
                    "A chunky bright yellow wheeled miniature construction crane truck with black rubber tires, "
                    "a square glass cab, red warning stripe on the bumper, and an angled yellow lattice boom with a small hook."
                ),
            }
        ],
        "sets": [
            {
                "id": "mvhs_hallway",
                "name": "Meme Valley High School Hallway",
                "camera_angle": "wide",
                "lighting_state": "warm sunlit afternoon streaming through high hallway windows",
                "description": (
                    "A nostalgic 1950s high school corridor with dusty-teal metal lockers lining the left wall, "
                    "a warm cream-painted right wall with a round wall clock and cork bulletin board, "
                    "checkered linoleum floor tiles, and double wooden doors with glass panes at the far end."
                ),
            }
        ],
        "shots": [
            {
                "shot_id": "s01_arrive",
                "set_id": "mvhs_hallway",
                "camera_angle": "wide",
                "locked_camera": True,
                "cast_ids": [chosen["id"]],
                "prop_ids": [{"id": "yellow_crane_toy", "state": "parked on hallway floor"}],
                "action": (
                    f"{chosen['name']} stands on the left side of the hallway looking proudly down at the bright yellow "
                    f"wheeled crane truck parked in the center of the checkered floor."
                ),
                "lighting_state": "warm sunlit afternoon",
            },
            {
                "shot_id": "s02_inspect",
                "set_id": "mvhs_hallway",
                "camera_angle": "wide",
                "locked_camera": True,
                "cast_ids": [chosen["id"]],
                "prop_ids": [{"id": "yellow_crane_toy", "state": "boom angled up toward the lockers"}],
                "action": (
                    f"{chosen['name']} kneels beside the bright yellow wheeled crane truck in the middle of the hallway, "
                    f"pointing excitedly up at the raised yellow crane boom."
                ),
                "lighting_state": "warm sunlit afternoon",
            },
            {
                "shot_id": "s03_retreat",
                "set_id": "mvhs_hallway",
                "camera_angle": "wide",
                "locked_camera": True,
                "cast_ids": [chosen["id"]],
                "prop_ids": [{"id": "yellow_crane_toy", "state": "tilted slightly on hallway floor"}],
                "action": (
                    f"{chosen['name']} steps back to the right side of the hallway with hands raised in comic surprise "
                    f"while the bright yellow wheeled crane truck sits in the center of the floor."
                ),
                "lighting_state": "warm sunlit afternoon",
            },
        ],
    }


def write_suite5_html(report: dict[str, Any], out_path: Path) -> None:
    sections = []
    for run in report["runs"]:
        base = out_path.parent
        rel = lambda p: os.path.relpath(Path(p), base)  # noqa: E731
        tiles = [
            f'<div class="tile"><div class="lbl">Master Set Plate (Rendered Once)</div><img src="{html.escape(rel(run["set_plate"]))}"></div>',
            f'<div class="tile"><div class="lbl">Character Turnaround Ref</div><img src="{html.escape(rel(run["cast_ref"]))}"></div>',
            f'<div class="tile"><div class="lbl">Prop Canonical Ref</div><img src="{html.escape(rel(run["prop_ref"]))}"></div>',
            f'<div class="tile"><div class="lbl">Set Reuse Contact Sheet</div><img src="{html.escape(rel(run["contact_sheet"]))}"></div>',
        ]
        shot_tiles = []
        for s in run["shot_evals"]:
            cls = "ok" if s["passed"] else "bad"
            shot_tiles.append(
                f'<div class="tile"><div class="lbl">{html.escape(s["shot_id"])} ({s["render_s"]}s)</div>'
                f'<img src="{html.escape(rel(s["path"]))}">'
                f'<div class="cap"><span class="pill {cls}">{"PASS" if s["passed"] else "FAIL"} {s["score"]}/10</span> '
                f'{html.escape(str(s["summary"])[:140])}</div></div>'
            )
        sections.append(
            f"""<section class="card">
              <h2>Image Model: <code>{html.escape(run['model'])}</code> &mdash; Pass Rate {run['pass_rate']*100:.0f}% (avg score {run['avg_score']}/10, avg {run['avg_render_s']}s/shot)</h2>
              <div class="grid">{"".join(tiles)}</div>
              <h3>Shots Reusing the Exact Master Set Plate</h3>
              <div class="grid">{"".join(shot_tiles)}</div>
            </section>"""
        )

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Suite 5 — Visual Bible &amp; Set-Plate Reuse Bake-Off</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #141210; color: #f2eadc; margin: 0; padding: 24px 28px 60px; }}
  h1, h2, h3 {{ color: #f4d378; }}
  .card {{ background: #1d1915; border: 1px solid #342c24; border-radius: 12px; padding: 18px; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; margin-bottom: 14px; }}
  .tile {{ background: #14110f; border: 1px solid #352d25; border-radius: 8px; overflow: hidden; }}
  .lbl {{ background: #26201a; padding: 6px 10px; font-size: 11.5px; font-weight: 700; color: #e8cfa0; text-transform: uppercase; }}
  .tile img {{ width: 100%; max-height: 280px; object-fit: contain; display: block; }}
  .cap {{ padding: 8px 10px; font-size: 12px; color: #cfc3ae; }}
  .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11.5px; font-weight: 700; }}
  .pill.ok {{ background: #1e3b28; color: #8ff0b0; }}
  .pill.bad {{ background: #3b1c1c; color: #f69696; }}
  code {{ color: #f4d378; }}
</style></head>
<body>
  <h1>Suite 5 — Visual Bible &amp; Single-Render Set Reuse Bake-Off</h1>
  <p>Critic VLM: <code>{html.escape(report['critic_model'])}</code> &bull; Generated {html.escape(report['created_at'])}</p>
  {"".join(sections)}
</body></html>"""
    out_path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Eval Suite 5: Visual Bible & Single-Render Set Reuse.")
    parser.add_argument("--models", nargs="+", default=["qwen-image-2.1-turbo", "flux2-klein-4b"])
    parser.add_argument("--critic", default="qwen3-vl-30b-a3b-instruct-mlx")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    spec_data = build_test_spec()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[tuple[str, dict[str, Any], Path]] = []

    # Batch 1: Render all mflux models first while LM Studio is unloaded
    unload_all_lms()
    for mkey in args.models:
        print(f"\n=== Suite 5 Render: {mkey} ===")
        mdir = args.out_dir / mkey
        man = generate_bible_and_shots(spec_data, out_dir=mdir, model_key=mkey)
        set_plate = Path(next(iter(man["refs"]["sets"].values()))["path"])
        sheet_tiles = [("TILE 0 (SET)", set_plate)] + [
            (f"TILE {idx}", Path(s["path"])) for idx, s in enumerate(man["shots"], start=1)
        ]
        sheet_path = make_contact_sheet(sheet_tiles, mdir / "set_reuse_contact_sheet.png", cols=2)
        manifests.append((mkey, man, sheet_path))

    # Batch 2: Load Critic VLM once and evaluate all generated shots
    print(f"\n=== Suite 5 Critic Audit: {args.critic} ===")
    load_lms_model(args.critic, context_length=16384)
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "critic_model": args.critic,
        "runs": [],
    }
    char_name = spec_data["cast"][0]["name"]

    for mkey, man, sheet_path in manifests:
        set_plate = Path(next(iter(man["refs"]["sets"].values()))["path"])
        cast_ref = Path(next(iter(man["refs"]["cast"].values()))["path"])
        prop_ref = Path(next(iter(man["refs"]["props"].values()))["path"])
        shot_evals = []
        for s in man["shots"]:
            v = audit_single_image(
                model=args.critic,
                image_path=Path(s["path"]),
                ref_images=[set_plate, cast_ref],
                entity_name=f"the hallway set from REF_1 and {char_name} from REF_2",
            )
            shot_evals.append(
                {
                    "shot_id": s["shot_id"],
                    "path": s["path"],
                    "render_s": s["elapsed_s"],
                    "passed": bool(v["passed"]),
                    "score": int(v["score"]),
                    "summary": v["defect_summary"],
                    "verdict": v,
                }
            )
        pass_rate = round(sum(1 for x in shot_evals if x["passed"]) / max(1, len(shot_evals)), 4)
        avg_score = round(sum(x["score"] for x in shot_evals) / max(1, len(shot_evals)), 2)
        avg_render = round(sum(x["render_s"] for x in shot_evals) / max(1, len(shot_evals)), 2)
        report["runs"].append(
            {
                "model": mkey,
                "set_plate": str(set_plate),
                "cast_ref": str(cast_ref),
                "prop_ref": str(prop_ref),
                "contact_sheet": str(sheet_path),
                "pass_rate": pass_rate,
                "avg_score": avg_score,
                "avg_render_s": avg_render,
                "shot_evals": shot_evals,
            }
        )

    unload_all_lms()
    (args.out_dir / "suite5_refgen.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_suite5_html(report, args.out_dir / "report.html")
    print(f"\nSaved Suite 5 report to {args.out_dir / 'report.html'}")


if __name__ == "__main__":
    main()
