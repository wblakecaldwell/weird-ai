"""Generate local MLX portraits for the Meme Yearbook roster using mflux.

Supports:
  - schnell-4bit      (default; uses already-cached FLUX.1-schnell 4-bit + cached pixel LoRA!)
  - z-image-turbo     (fast 2026 MLX model with strong stylized/Gumball aesthetics)
  - flux2-klein-9b    (FLUX.2 Klein 9B Q8 — great studio realism & mixed media)
  - flux2-klein-4b    (FLUX.2 Klein 4B Q8 — lighter/faster)
  - qwen-image-2.1-turbo (Qwen-Image 2.1 Turbo Q6 — top prompt adherence for weird anatomy/styles)

Usage:
    uv run python scripts/generate_portraits.py --model schnell-4bit --variants 1
    uv run python scripts/generate_portraits.py --model z-image-turbo --only 01-
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

from styles import STYLES, build_image_prompt

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROSTER = ROOT / "data" / "roster.json"
RAW_DIR = ROOT / "portraits" / "raw"

# Auto-detect cached FLUX.1-schnell pixel art LoRA if present
PIXEL_LORA_GLOB = str(
    Path.home()
    / ".cache/huggingface/hub/models--hliu143--dr-diffusion-flux-1-s-pixel-art-lora/snapshots/*/pixelart_schnell_v1.safetensors"
)


def find_cached_pixel_lora() -> str | None:
    matches = sorted(glob.glob(PIXEL_LORA_GLOB))
    return matches[-1] if matches else None


def load_generator(model_key: str, lora_path: str | None = None, lora_scale: float = 0.85):
    """Instantiate the selected mflux model once so all portraits share loaded weights."""
    from mflux.models.common.config import ModelConfig

    lora_kwargs = {}
    if lora_path:
        lora_kwargs = {"lora_paths": [lora_path], "lora_scales": [lora_scale]}

    if model_key == "schnell-4bit":
        from mflux.models.flux.variants.txt2img.flux import Flux1

        config = ModelConfig.from_name(
            model_name="dhairyashil/FLUX.1-schnell-mflux-4bit",
            base_model="schnell",
        )
        return Flux1(model_config=config, quantize=None, **lora_kwargs), {"steps": 4, "guidance": 0.0}

    if model_key == "z-image-turbo":
        from mflux.models.z_image.variants.z_image import ZImage

        config = ModelConfig.from_name(model_name="z-image-turbo")
        return ZImage(model_config=config, quantize=8, **lora_kwargs), {"steps": 8, "guidance": 0.0}

    if model_key in ("flux2-klein-4b", "flux2-klein-9b"):
        from mflux.models.flux2.variants import Flux2Klein

        config = ModelConfig.from_name(model_name=model_key)
        return Flux2Klein(model_config=config, quantize=8, **lora_kwargs), {"steps": 4, "guidance": 1.0}

    if model_key == "qwen-image-2.1-turbo":
        from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

        config = ModelConfig.from_name(model_name="qwen-image-2.1-turbo")
        return QwenImage21(model_config=config, quantize=6, **lora_kwargs), {"steps": 8, "guidance": 1.0}

    raise ValueError(f"Unsupported model key: {model_key}")


def generate_one(model, model_key: str, prompt: str, seed: int, width: int, height: int, steps: int, guidance: float):
    kwargs = {
        "seed": seed,
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
    }
    if model_key in ("flux2-klein-4b", "flux2-klein-9b", "qwen-image-2.1-turbo"):
        kwargs["guidance"] = guidance
    return model.generate_image(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Meme Yearbook portraits locally with mflux.")
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER, help="Path to roster.json")
    parser.add_argument(
        "--model",
        default="schnell-4bit",
        choices=["schnell-4bit", "z-image-turbo", "flux2-klein-4b", "flux2-klein-9b", "qwen-image-2.1-turbo"],
        help="Local MLX model to use (default: schnell-4bit)",
    )
    parser.add_argument("--variants", type=int, default=1, help="Number of seed variants per student (default: 1)")
    parser.add_argument("--width", type=int, default=768, help="Portrait width (default: 768)")
    parser.add_argument("--height", type=int, default=960, help="Portrait height (default: 960, 4:5 ratio)")
    parser.add_argument("--steps", type=int, default=None, help="Override inference steps")
    parser.add_argument("--only", type=str, default=None, help="Filter students whose id/name contains this substring")
    parser.add_argument("--limit", type=int, default=None, help="Only generate first N matching students")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if output PNG already exists")
    parser.add_argument("--no-pixel-lora", action="store_true", help="Disable cached pixel LoRA for schnell-4bit")
    parser.add_argument("--cache-limit-gb", type=float, default=6.0, help="MLX Metal cache limit in GB (default: 6.0)")
    args = parser.parse_args()

    import mlx.core as mx

    if args.cache_limit_gb > 0:
        mx.set_cache_limit(int(args.cache_limit_gb * (1024**3)))

    if not args.roster.exists():
        print(f"Error: roster not found at {args.roster}", file=sys.stderr)
        sys.exit(1)

    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    students = roster["students"]

    if args.only:
        needle = args.only.lower()
        students = [s for s in students if needle in s["id"].lower() or needle in s["name"].lower()]
    if args.limit is not None:
        students = students[: args.limit]

    if not students:
        print("No matching students found.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Group students by whether they use the cached pixel LoRA (only applies to schnell-4bit)
    pixel_lora = None if (args.no_pixel_lora or args.model != "schnell-4bit") else find_cached_pixel_lora()
    groups: list[tuple[str | None, list[dict]]] = []
    if pixel_lora:
        pixel_students = [s for s in students if STYLES.get(s["style_id"]) and STYLES[s["style_id"]].use_pixel_lora]
        other_students = [s for s in students if s not in pixel_students]
        if other_students:
            groups.append((None, other_students))
        if pixel_students:
            groups.append((pixel_lora, pixel_students))
    else:
        groups.append((None, students))

    total_jobs = len(students) * args.variants
    done = 0
    t0 = time.time()

    for lora_path, group_students in groups:
        # Skip loading the model if every image in this group already exists
        pending = []
        for student in group_students:
            for v in range(args.variants):
                out_path = RAW_DIR / f"{student['id']}_v{v}.png"
                if args.overwrite or not out_path.exists():
                    pending.append((student, v, out_path))
                else:
                    done += 1

        if not pending:
            continue

        lora_note = f" + pixel LoRA ({Path(lora_path).name})" if lora_path else ""
        print(f"\nLoading {args.model}{lora_note} for {len(pending)} image(s)...")
        model, defaults = load_generator(args.model, lora_path=lora_path)
        steps = args.steps or defaults["steps"]
        guidance = defaults["guidance"]

        for student, v, out_path in pending:
            done += 1
            seed = int(student.get("seed", 202500)) + v * 101
            prompt = student.get("prompt") or build_image_prompt(
                student["style_id"], student["visual_description"]
            )
            if lora_path:
                prompt = f"pixelart, {prompt}"

            start_t = time.time()
            print(f"[{done}/{total_jobs}] {student['name']} ({student['style_label']}) v{v} seed={seed}...")
            img = generate_one(
                model=model,
                model_key=args.model,
                prompt=prompt,
                seed=seed,
                width=args.width,
                height=args.height,
                steps=steps,
                guidance=guidance,
            )
            img.save(path=str(out_path))
            print(f"    -> saved {out_path.relative_to(ROOT)} ({time.time() - start_t:.1f}s)")
            mx.clear_cache()

        # Explicitly release model reference before loading a different LoRA group
        del model
        mx.clear_cache()

    print(f"\nDone in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
