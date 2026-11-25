"""Tinkering Lab: Compare multi-reference character consistency across local mflux edit models.

Generates 3 stress-test shots for a mixed-media cast:
  1. Solo three-quarter side-profile action (tests angle change + medium lock)
  2. Solo low-angle full-body action (tests full-body invention from a headshot)
  3. Two-character mixed-media interaction in a 1950s painted diner (tests multi-ref separation)

Outputs a clean side-by-side HTML comparison board in bakeoff_out/index.html.

Usage:
    uv run python scripts/bakeoff.py --models flux2-klein-4b qwen-image-2.1-turbo
    uv run python scripts/bakeoff.py --cast chester marisol --models flux2-klein-4b --sheets
"""

from __future__ import annotations

import argparse
import html
import shutil
import time
from pathlib import Path

from generate_illustrations import EDIT_MODELS, load_edit_model, render_edit, save_clean
from llm import unload_lms_models
from prompts import build_scene_prompt, build_turnaround_prompt
from roster import DEFAULT_ROSTER_PATH, FEVER_DREAMS_ROOT, cast_characters, load_roster, resolve_project_path

OUT_DIR = FEVER_DREAMS_ROOT / "bakeoff_out"


def build_board_html(cast_dicts: list[dict], results: list[dict], out_dir: Path) -> str:
    ref_cards = "".join(
        f"""
        <div class="ref-card">
          <img src="refs/{html.escape(c['id'])}{Path(c['portrait_display']).suffix}" alt="{html.escape(c['name'])}">
          <div><strong>{html.escape(c['name'])}</strong><br><span>{html.escape(c['style_label'])}</span></div>
        </div>"""
        for c in cast_dicts
    )

    sections = []
    for row in results:
        cards = "".join(
            f"""
            <figure class="shot">
              <img src="{html.escape(s['rel_path'])}" alt="{html.escape(s['label'])}">
              <figcaption>
                <strong>{html.escape(s['label'])}</strong>
                <span class="timing">{s['elapsed_s']:.1f}s</span>
                <p>{html.escape(s['camera'])}</p>
              </figcaption>
            </figure>"""
            for s in row["shots"]
        )
        sections.append(
            f"""
            <section class="model-row">
              <h2>{html.escape(row['model'])} <small>(total {row['total_s']:.1f}s, sheets={'on' if row['used_sheets'] else 'off'})</small></h2>
              <div class="shot-grid">{cards}</div>
            </section>"""
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Fever Dreams — Local Model Bake-Off</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #191613; color: #f6eedb; margin: 0; padding: 28px; }}
  h1 {{ margin: 0 0 6px; font-family: Georgia, serif; color: #f3d37a; }}
  .sub {{ opacity: 0.8; margin-bottom: 20px; }}
  .refs {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }}
  .ref-card {{ display: flex; gap: 12px; align-items: center; background: #26211c; padding: 10px 14px; border-radius: 10px; border: 1px solid #3c3329; }}
  .ref-card img {{ width: 64px; height: 80px; object-fit: cover; border-radius: 6px; }}
  .ref-card span {{ font-size: 12px; color: #d7c39a; }}
  .model-row {{ margin-bottom: 34px; background: #221d18; padding: 18px; border-radius: 12px; border: 1px solid #3a3127; }}
  .model-row h2 {{ margin: 0 0 14px; font-size: 18px; color: #f3d37a; }}
  .model-row h2 small {{ color: #bfb19a; font-weight: normal; font-size: 13px; }}
  .shot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
  .shot {{ margin: 0; background: #161310; border-radius: 8px; overflow: hidden; border: 1px solid #362e25; }}
  .shot img {{ width: 100%; display: block; }}
  .shot figcaption {{ padding: 10px 12px; font-size: 13px; }}
  .shot figcaption p {{ margin: 4px 0 0; opacity: 0.75; font-size: 12px; }}
  .timing {{ float: right; background: #3b3124; color: #f3d37a; padding: 2px 7px; border-radius: 999px; font-size: 11px; }}
</style>
</head>
<body>
  <h1>Fever Dreams — Character Consistency Bake-Off</h1>
  <p class="sub">Original yearbook portraits (top) vs. multi-angle &amp; multi-character 1950s Little Golden Book scenes.</p>
  <div class="refs">{ref_cards}</div>
  {''.join(sections)}
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local mflux multi-reference models on Meme Yearbook characters.")
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER_PATH)
    parser.add_argument("--cast", nargs="+", default=["chester", "marisol"], help="2 characters to test (default: chester marisol)")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["flux2-klein-4b", "qwen-image-2.1-turbo"],
        choices=list(EDIT_MODELS.keys()),
        help="Models to compare side-by-side",
    )
    parser.add_argument("--sheets", action="store_true", help="Also generate turnaround sheets and condition scenes on them")
    parser.add_argument("--seed", type=int, default=195501)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    _, all_chars = load_roster(args.roster)
    cast = cast_characters(all_chars, requested=args.cast, count=2, seed=args.seed)
    c1, c2 = cast[0].to_story_dict(), cast[1].to_story_dict()
    c1["portrait_display"] = str(cast[0].best_display_image)
    c2["portrait_display"] = str(cast[1].best_display_image)

    args.out.mkdir(parents=True, exist_ok=True)
    refs_dir = args.out / "refs"
    refs_dir.mkdir(exist_ok=True)
    for ch, d in zip(cast[:2], (c1, c2)):
        src = ch.best_display_image
        if src:
            shutil.copy2(src, refs_dir / f"{ch.id}{src.suffix}")

    unload_lms_models()

    import mlx.core as mx

    mx.set_cache_limit(8 * (1024**3))

    tests = [
        (
            "01_side_angle",
            f"Side Profile: {c1['name']}",
            [c1],
            "Three-quarter side-profile medium shot looking down a gleaming 1950s high school hallway",
            f"{c1['name']} is leaning cautiously around the edge of an open green metal locker, staring suspiciously at a giant wobbling lime gelatin mold sitting on a rolling library cart.",
            "Bright nostalgic mid-century afternoon sunlight slanting through tall school windows",
        ),
        (
            "02_low_fullbody",
            f"Low-Angle Full Body: {c2['name']}",
            [c2],
            "Dramatic low-angle full-body shot from floor level looking up",
            f"{c2['name']} stands triumphantly on top of a checkered 1950s cafeteria table holding a gleaming brass megaphone and a giant clipboard while neatly stacked lunch trays tower around them.",
            "Warm theatrical 1950s storybook spotlight with cheerful pastel gouache shadows",
        ),
        (
            "03_duo_diner",
            f"Two-Character Scene: {c1['name']} + {c2['name']}",
            [c1, c2],
            "Wide over-the-counter three-quarter storybook view inside a cozy 1950s chrome-and-turquoise malt shop diner",
            f"On the left, {c1['name']} sits on a red vinyl barstool nervously pointing at a towering stack of seventeen pancakes; on the right, {c2['name']} stands beside them gesturing earnestly with an oversized blueprint rolled open across the Formica counter.",
            "Cozy warm vintage diner lamp glow and painted gouache shadows",
        ),
    ]

    all_results: list[dict] = []
    for model_key in args.models:
        print(f"\n=== Bake-Off Model: {model_key} ===")
        model_dir = args.out / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        model, spec = load_edit_model(model_key)
        model_t0 = time.time()
        shot_records: list[dict] = []

        char_refs: dict[str, Path] = {
            c1["id"]: resolve_project_path(c1["portrait_raw"]) or resolve_project_path(c1["portrait_final"]),
            c2["id"]: resolve_project_path(c2["portrait_raw"]) or resolve_project_path(c2["portrait_final"]),
        }

        if args.sheets:
            for idx, cd in enumerate((c1, c2), start=1):
                t0 = time.time()
                print(f"  Rendering turnaround sheet for {cd['name']}...")
                sheet_img = render_edit(
                    model,
                    spec,
                    prompt=build_turnaround_prompt(cd),
                    image_paths=[char_refs[cd["id"]]],
                    seed=args.seed + idx * 11,
                    width=960,
                    height=576,
                )
                sheet_out = model_dir / f"sheet_{cd['id']}.png"
                save_clean(sheet_img, sheet_out)
                elapsed = time.time() - t0
                char_refs[cd["id"]] = sheet_out
                shot_records.append(
                    {
                        "label": f"Turnaround Sheet: {cd['name']}",
                        "camera": "3-view full-body character sheet",
                        "rel_path": sheet_out.relative_to(args.out).as_posix(),
                        "elapsed_s": elapsed,
                    }
                )
                mx.clear_cache()

        for s_idx, (slug, label, chars_in_shot, camera, scene, mood) in enumerate(tests, start=1):
            t0 = time.time()
            prompt = build_scene_prompt(chars_in_shot, scene, camera, mood)
            refs = [char_refs[c["id"]] for c in chars_in_shot]
            print(f"  Rendering {label} ({len(refs)} ref)...")
            img = render_edit(
                model,
                spec,
                prompt=prompt,
                image_paths=refs,
                seed=args.seed + s_idx * 101,
                width=args.width,
                height=args.height,
            )
            out_file = model_dir / f"{slug}.png"
            save_clean(img, out_file)
            elapsed = time.time() - t0
            print(f"    -> {out_file.name} ({elapsed:.1f}s)")
            shot_records.append(
                {
                    "label": label,
                    "camera": camera,
                    "rel_path": out_file.relative_to(args.out).as_posix(),
                    "elapsed_s": elapsed,
                }
            )
            mx.clear_cache()

        all_results.append(
            {
                "model": model_key,
                "used_sheets": args.sheets,
                "total_s": time.time() - model_t0,
                "shots": shot_records,
            }
        )
        del model
        mx.clear_cache()

    board_path = args.out / "index.html"
    board_path.write_text(build_board_html([c1, c2], all_results, args.out), encoding="utf-8")
    print(f"\nBake-off complete! Open comparison board:\n  open {board_path}")


if __name__ == "__main__":
    main()
