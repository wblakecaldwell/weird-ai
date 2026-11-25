"""Step 2: Render turnaround sheets, cover, and story pages using local multi-reference mflux edit models.

Supports multiple local MLX image-conditioning models so you can tinker and compare:
  - qwen-image-2.1-turbo  (default; Apache-2.0 ungated, 8 steps, top weird-anatomy & multi-ref adherence)
  - flux2-klein-4b        (fast 4-step, Apache-2.0 ungated, lightweight for rapid tinkering)
  - flux2-klein-9b        (4-step 9B Q8, great physical textures like clay/felt/cardboard)
  - flux2-klein-9b-kv     (4-step 9B KV-cached edit variant)
  - flux3-image-klein     (FLUX 3 Image Klein Edit, supports 1-10 reference images)
  - qwen-image-edit-2511  (Qwen Image Edit 2511)

Usage:
    uv run python scripts/generate_illustrations.py books/01-my-book/story.json --sheets
    uv run python scripts/generate_illustrations.py books/01-my-book/story.json --only-page 3 --seed-offset 1 --overwrite
    uv run python scripts/generate_illustrations.py books/01-my-book/story.json --model flux2-klein-4b --only-cover
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from llm import unload_lms_models
from prompts import build_scene_prompt, build_turnaround_prompt
from roster import resolve_project_path


@dataclass(frozen=True)
class EditModelSpec:
    key: str
    family: str  # "qwen21", "qwen", "flux2", "flux3"
    mflux_name: str
    default_quantize: int
    default_steps: int
    default_guidance: float


EDIT_MODELS: dict[str, EditModelSpec] = {
    "qwen-image-2.1-turbo": EditModelSpec(
        key="qwen-image-2.1-turbo",
        family="qwen21",
        mflux_name="qwen-image-2.1-turbo",
        default_quantize=6,
        default_steps=8,
        default_guidance=1.0,
    ),
    "qwen-image-2.1": EditModelSpec(
        key="qwen-image-2.1",
        family="qwen21",
        mflux_name="qwen-image-2.1",
        default_quantize=6,
        default_steps=20,
        default_guidance=2.5,
    ),
    "flux2-klein-4b": EditModelSpec(
        key="flux2-klein-4b",
        family="flux2",
        mflux_name="flux2-klein-4b",
        default_quantize=8,
        default_steps=4,
        default_guidance=1.0,
    ),
    "flux2-klein-9b": EditModelSpec(
        key="flux2-klein-9b",
        family="flux2",
        mflux_name="flux2-klein-9b",
        default_quantize=8,
        default_steps=4,
        default_guidance=1.0,
    ),
    "flux2-klein-9b-kv": EditModelSpec(
        key="flux2-klein-9b-kv",
        family="flux2",
        mflux_name="flux2-klein-9b-kv",
        default_quantize=8,
        default_steps=4,
        default_guidance=1.0,
    ),
    "flux3-image-klein": EditModelSpec(
        key="flux3-image-klein",
        family="flux3",
        mflux_name="flux3-image-klein",
        default_quantize=8,
        default_steps=4,
        default_guidance=1.0,
    ),
    "qwen-image-edit-2511": EditModelSpec(
        key="qwen-image-edit-2511",
        family="qwen",
        mflux_name="qwen-image-edit-2511",
        default_quantize=6,
        default_steps=20,
        default_guidance=4.0,
    ),
}


def load_edit_model(model_key: str, quantize: int | None = None):
    """Instantiate a local MLX multi-reference edit model once so all pages share loaded weights."""
    from mflux.models.common.config import ModelConfig

    spec = EDIT_MODELS[model_key]
    q = quantize if quantize is not None else spec.default_quantize
    config = ModelConfig.from_name(model_name=spec.mflux_name)

    if spec.family == "qwen21":
        from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit

        return QwenImage21Edit(model_config=config, quantize=q), spec

    if spec.family == "flux2":
        from mflux.models.flux2.variants import Flux2KleinEdit

        return Flux2KleinEdit(model_config=config, quantize=q), spec

    if spec.family == "flux3":
        from mflux.models.flux3.variants import Flux3KleinEdit

        return Flux3KleinEdit(model_config=config, quantize=q), spec

    if spec.family == "qwen":
        from mflux.models.qwen.variants.edit import QwenImageEdit

        return QwenImageEdit(model_config=config, quantize=q), spec

    raise ValueError(f"Unsupported model family: {spec.family}")


def render_edit(
    model,
    spec: EditModelSpec,
    *,
    prompt: str,
    image_paths: list[Path],
    seed: int,
    width: int,
    height: int,
    steps: int | None = None,
    guidance: float | None = None,
):
    """Invoke the loaded edit model with normalized arguments across mflux families."""
    str_paths = [str(p) for p in image_paths]
    n_steps = steps if steps is not None else spec.default_steps
    g = guidance if guidance is not None else spec.default_guidance

    if spec.family == "qwen21":
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            image_paths=str_paths,
            num_inference_steps=n_steps,
            width=width,
            height=height,
            guidance=g,
        )
    if spec.family == "flux2":
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            image_paths=str_paths,
            num_inference_steps=n_steps,
            width=width,
            height=height,
            guidance=g,
        )
    if spec.family == "flux3":
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            image_paths=str_paths,
            num_inference_steps=n_steps,
            width=width,
            height=height,
            guidance=g,
        )
    if spec.family == "qwen":
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            image_paths=str_paths,
            num_inference_steps=n_steps,
            width=width,
            height=height,
            guidance=g,
        )
    raise ValueError(f"Unsupported model family: {spec.family}")


def save_clean(generated, path: Path | str) -> None:
    """Save the PNG without mflux's embedded EXIF/XMP metadata (which includes absolute reference-image
    paths). Seed, model, steps, and refs are already recorded in story.json's render_meta."""
    generated.image.save(str(path), format="PNG")


def resolve_story_path(path_arg: Path) -> Path:
    if path_arg.is_dir():
        return path_arg / "story.json"
    return path_arg


def portrait_for_char(char: dict) -> Path:
    for key in ("portrait_raw", "portrait_final"):
        resolved = resolve_project_path(char.get(key))
        if resolved:
            return resolved
    raise FileNotFoundError(
        f"Could not find portrait image on disk for {char['name']} ({char['id']}). "
        f"Checked portrait_raw={char.get('portrait_raw')} and portrait_final={char.get('portrait_final')}."
    )


def reference_for_char(char: dict, book_dir: Path, story: dict, use_sheets: bool) -> Path:
    if use_sheets:
        sheet_rel = (story.get("sheets") or {}).get(char["id"])
        if sheet_rel:
            sheet_path = book_dir / sheet_rel
            if sheet_path.exists():
                return sheet_path
    return portrait_for_char(char)


def save_story(story_path: Path, story: dict) -> None:
    story_path.write_text(json.dumps(story, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Illustrate a Little Fever Dream Book locally with mflux.")
    parser.add_argument("story", type=Path, help="Path to books/<slug>/story.json (or book directory)")
    parser.add_argument(
        "--model",
        default="qwen-image-2.1-turbo",
        choices=list(EDIT_MODELS.keys()),
        help="Local multi-reference mflux edit model (default: qwen-image-2.1-turbo)",
    )
    parser.add_argument("--quantize", type=int, default=None, choices=[3, 4, 5, 6, 8], help="Override quantization bits")
    parser.add_argument("--steps", type=int, default=None, help="Override inference steps")
    parser.add_argument("--guidance", type=float, default=None, help="Override guidance scale")
    parser.add_argument("--width", type=int, default=896, help="Page illustration width (default: 896)")
    parser.add_argument("--height", type=int, default=768, help="Page illustration height (default: 768)")
    parser.add_argument("--cover-width", type=int, default=768, help="Cover illustration width (default: 768)")
    parser.add_argument("--cover-height", type=int, default=960, help="Cover illustration height (default: 960)")
    parser.add_argument(
        "--sheets",
        action="store_true",
        help="Generate 3-view full-body character turnaround sheets first and condition pages on them",
    )
    parser.add_argument(
        "--no-use-sheets",
        action="store_true",
        help="Even if turnaround sheets exist, condition pages directly on raw yearbook portraits",
    )
    parser.add_argument("--only-sheets", action="store_true", help="Generate turnaround sheets only")
    parser.add_argument("--only-cover", action="store_true", help="Generate cover illustration only")
    parser.add_argument("--only-page", type=int, nargs="+", default=None, help="Generate only specific page number(s)")
    parser.add_argument("--limit", type=int, default=None, help="Stop after rendering N pages")
    parser.add_argument("--seed-offset", type=int, default=0, help="Add offset to stored seeds when re-rolling")
    parser.add_argument("--overwrite", action="store_true", help="Re-render images that already exist")
    parser.add_argument("--refresh-prompts", action="store_true", help="Rebuild prompts from prompts.py before rendering")
    parser.add_argument("--keep-lms", action="store_true", help="Do not run `lms unload --all` before loading mflux")
    parser.add_argument("--cache-limit-gb", type=float, default=8.0, help="MLX Metal cache limit in GB (default: 8.0)")
    args = parser.parse_args()

    story_path = resolve_story_path(args.story).resolve()
    if not story_path.exists():
        print(f"Error: story.json not found at {story_path}", file=sys.stderr)
        sys.exit(1)

    book_dir = story_path.parent
    sheets_dir = book_dir / "sheets"
    art_dir = book_dir / "art"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    story = json.loads(story_path.read_text(encoding="utf-8"))
    cast_by_id = {c["id"]: c for c in story["cast"]}
    story.setdefault("sheets", {})

    if args.refresh_prompts:
        cov = story["cover"]
        cov["prompt"] = build_scene_prompt(
            [cast_by_id[cid] for cid in cov["cast_ids"]],
            cov["scene_description"],
            cov["camera_angle"],
            cov.get("lighting_mood", ""),
            is_cover=True,
        )
        for p in story["pages"]:
            p["prompt"] = build_scene_prompt(
                [cast_by_id[cid] for cid in p["cast_ids"]],
                p["scene_description"],
                p["camera_angle"],
                p.get("lighting_mood", ""),
                is_cover=False,
            )
        save_story(story_path, story)

    # Plan jobs before loading heavy model weights
    sheet_jobs: list[tuple[dict, Path]] = []
    if args.sheets or args.only_sheets:
        for ch in story["cast"]:
            out_path = sheets_dir / f"{ch['id']}_sheet.png"
            if args.overwrite or not out_path.exists():
                sheet_jobs.append((ch, out_path))
            else:
                story["sheets"][ch["id"]] = out_path.relative_to(book_dir).as_posix()

    do_cover = (not args.only_sheets and not args.only_page) or args.only_cover
    cover_out = art_dir / "cover.png"
    need_cover = do_cover and (args.overwrite or not cover_out.exists())
    if do_cover and cover_out.exists() and not story["cover"].get("image"):
        story["cover"]["image"] = cover_out.relative_to(book_dir).as_posix()

    page_jobs: list[tuple[dict, Path]] = []
    if not args.only_sheets and not args.only_cover:
        selected_pages = story["pages"]
        if args.only_page:
            wanted = set(args.only_page)
            selected_pages = [p for p in selected_pages if int(p["page_number"]) in wanted]
        if args.limit is not None:
            selected_pages = selected_pages[: args.limit]
        for p in selected_pages:
            out_path = art_dir / f"page_{int(p['page_number']):02d}.png"
            if args.overwrite or not out_path.exists():
                page_jobs.append((p, out_path))
            elif not p.get("image"):
                p["image"] = out_path.relative_to(book_dir).as_posix()

    save_story(story_path, story)

    total_jobs = len(sheet_jobs) + (1 if need_cover else 0) + len(page_jobs)
    if total_jobs == 0:
        print("Nothing to render (all target illustrations already exist; pass --overwrite to re-roll).")
        return

    if not args.keep_lms:
        print("Unloading LM Studio models to free Apple Silicon unified memory...")
        unload_lms_models()

    import mlx.core as mx

    if args.cache_limit_gb > 0:
        mx.set_cache_limit(int(args.cache_limit_gb * (1024**3)))

    print(f"Loading {args.model} for {total_jobs} render(s)...")
    model, spec = load_edit_model(args.model, quantize=args.quantize)
    t_all = time.time()
    done = 0

    # 1. Character turnaround sheets
    for ch, out_path in sheet_jobs:
        done += 1
        ref_portrait = portrait_for_char(ch)
        prompt = build_turnaround_prompt(ch)
        seed = int(story["cover"].get("seed", 42000)) + done * 31 + args.seed_offset
        t0 = time.time()
        print(f"[{done}/{total_jobs}] Turnaround sheet: {ch['name']} ({ch['style_label']}) seed={seed}...")
        img = render_edit(
            model,
            spec,
            prompt=prompt,
            image_paths=[ref_portrait],
            seed=seed,
            width=1024,
            height=640,
            steps=args.steps,
            guidance=args.guidance,
        )
        save_clean(img, out_path)
        story["sheets"][ch["id"]] = out_path.relative_to(book_dir).as_posix()
        save_story(story_path, story)
        print(f"    -> saved {story['sheets'][ch['id']]} ({time.time() - t0:.1f}s)")
        mx.clear_cache()

    use_sheets = not args.no_use_sheets

    # 2. Cover art
    if need_cover:
        done += 1
        cov = story["cover"]
        refs = [reference_for_char(cast_by_id[cid], book_dir, story, use_sheets) for cid in cov["cast_ids"]]
        seed = int(cov.get("seed", 42000)) + args.seed_offset
        t0 = time.time()
        print(f"[{done}/{total_jobs}] Cover illustration ({len(refs)} ref(s)) seed={seed}...")
        img = render_edit(
            model,
            spec,
            prompt=cov["prompt"],
            image_paths=refs,
            seed=seed,
            width=args.cover_width,
            height=args.cover_height,
            steps=args.steps,
            guidance=args.guidance,
        )
        save_clean(img, cover_out)
        elapsed = round(time.time() - t0, 1)
        cov["image"] = cover_out.relative_to(book_dir).as_posix()
        cov["render_meta"] = {
            "model": spec.key,
            "seed": seed,
            "steps": args.steps or spec.default_steps,
            "refs": [p.name for p in refs],
            "elapsed_s": elapsed,
        }
        save_story(story_path, story)
        print(f"    -> saved {cov['image']} ({elapsed}s)")
        mx.clear_cache()

    # 3. Story pages
    for page, out_path in page_jobs:
        done += 1
        refs = [reference_for_char(cast_by_id[cid], book_dir, story, use_sheets) for cid in page["cast_ids"]]
        seed = int(page.get("seed", 42000 + page["page_number"] * 97)) + args.seed_offset
        t0 = time.time()
        names = ", ".join(cast_by_id[cid]["name"] for cid in page["cast_ids"])
        print(f"[{done}/{total_jobs}] Page {page['page_number']:02d} [{names}] seed={seed}...")
        img = render_edit(
            model,
            spec,
            prompt=page["prompt"],
            image_paths=refs,
            seed=seed,
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance=args.guidance,
        )
        save_clean(img, out_path)
        elapsed = round(time.time() - t0, 1)
        page["image"] = out_path.relative_to(book_dir).as_posix()
        page["render_meta"] = {
            "model": spec.key,
            "seed": seed,
            "steps": args.steps or spec.default_steps,
            "refs": [p.name for p in refs],
            "elapsed_s": elapsed,
        }
        save_story(story_path, story)
        print(f"    -> saved {page['image']} ({elapsed}s)")
        mx.clear_cache()

    print(f"\nAll {total_jobs} render(s) complete in {time.time() - t_all:.1f}s.")


if __name__ == "__main__":
    main()
