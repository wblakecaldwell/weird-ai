"""Step 3 of the cartoon pipeline: animate each page illustration via local LTX-2.5
(DistilledPipeline) using Image-to-Video conditioning.

Usage:
    cartoon/.venv/bin/python cartoon/animate_book.py books/01-the-very-big-world-problem/cartoon/script.json
    cartoon/.venv/bin/python cartoon/animate_book.py books/01-the-very-big-world-problem/cartoon/script.json --only-pages 1 2 4 --frames 121
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_log import RenderLog  # noqa: E402

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "ltx-2.5-q8"


def prepare_conditioning_image(src_path: Path, dst_path: Path, width: int, height: int) -> Path:
    """Fit src_path onto width x height cleanly (exact copy if already matching, or warm book-mat letterbox)."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src_path).convert("RGB")
    if im.size == (width, height):
        im.save(dst_path)
        return dst_path

    # Warm Little Golden Book gold/cream background for mismatched aspect (e.g. portrait cover)
    bg = Image.new("RGB", (width, height), (235, 222, 192))
    scale = min((width - 48) / im.width, (height - 32) / im.height)
    nw = max(1, int(round(im.width * scale)))
    nh = max(1, int(round(im.height * scale)))
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)

    x = (width - nw) // 2
    y = (height - nh) // 2
    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    sdraw.rectangle([x + 6, y + 8, x + nw + 6, y + nh + 8], fill=(40, 28, 16, 110))
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))

    canvas = Image.alpha_composite(bg.convert("RGBA"), shadow).convert("RGB")
    canvas.paste(resized, (x, y))
    canvas.save(dst_path)
    return dst_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate script.json shots with local LTX-2.5 MLX.")
    parser.add_argument("script_json", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--frames", type=int, default=121, help="8k+1 frames (97=4s, 121=5s, 145=6s @24fps)")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=1955)
    parser.add_argument("--low-ram", action="store_true", help="Enable DiT block streaming")
    parser.add_argument("--only-pages", type=int, nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    script_path = args.script_json.resolve()
    cartoon_dir = script_path.parent
    book_dir = cartoon_dir.parent
    clips_dir = cartoon_dir / "clips"
    cond_dir = cartoon_dir / "cond_frames"
    clips_dir.mkdir(parents=True, exist_ok=True)
    cond_dir.mkdir(parents=True, exist_ok=True)

    script = json.loads(script_path.read_text(encoding="utf-8"))
    shots = script["shots"]
    if args.only_pages is not None:
        wanted = set(args.only_pages)
        shots = [s for s in shots if int(s["page_number"]) in wanted]

    from ltx_pipelines_mlx import DistilledPipeline

    model_str = str(args.model.resolve()) if args.model.exists() else str(args.model)
    print(f"Initializing DistilledPipeline from {model_str} ({args.width}x{args.height}, {args.frames}f @{args.fps}fps, low_ram={args.low_ram})")
    pipe = DistilledPipeline(
        model_dir=model_str,
        low_memory=True,
        low_ram_streaming=args.low_ram,
    )
    log = RenderLog(
        cartoon_dir,
        "animate",
        {
            "pipeline": "ltx_pipelines_mlx.DistilledPipeline",
            "model": args.model.name,
            "width": args.width,
            "height": args.height,
            "frames": args.frames,
            "fps": args.fps,
            "base_seed": args.seed,
            "seed_rule": "base_seed + page_number * 37",
            "low_ram": args.low_ram,
            "only_pages": args.only_pages,
            "overwrite": args.overwrite,
        },
    )
    rendered = 0

    for shot in shots:
        pnum = int(shot["page_number"])
        out_mp4 = clips_dir / f"clip_{pnum:02d}.mp4"
        if out_mp4.exists() and not args.overwrite:
            print(f"[skip] {out_mp4.name} already exists")
            continue

        src_img = book_dir / shot["image"]
        cond_img = prepare_conditioning_image(src_img, cond_dir / f"cond_{pnum:02d}.png", args.width, args.height)
        prompt = shot["motion_prompt"]
        if shot.get("sfx_cue"):
            prompt = f"{prompt} Ambient audio: {shot['sfx_cue']}."

        t0 = time.time()
        seed = args.seed + pnum * 37
        print(f"\n=== Animating shot p{pnum:02d} ({shot['kind']}) ===")
        print(f"  Prompt: {prompt}")
        pipe.generate_and_save(
            prompt=prompt,
            output_path=str(out_mp4),
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            frame_rate=args.fps,
            seed=seed,
            image=str(cond_img),
        )
        dt = time.time() - t0
        print(f"  Done {out_mp4.name} in {dt:.1f}s")
        log.add(
            page_number=pnum,
            kind=shot["kind"],
            clip=out_mp4.relative_to(cartoon_dir).as_posix(),
            source_image=shot["image"],
            seed=seed,
            prompt=prompt,
            elapsed_s=round(dt, 1),
        )
        rendered += 1
        gc.collect()
        mx.clear_cache()

    log.finish(clips_rendered=rendered)
    print("\nAnimation pass complete.")


if __name__ == "__main__":
    main()
