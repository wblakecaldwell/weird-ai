"""Interactive local AI page retoucher & re-roller for Little Fever Dream Books.

Generates N candidate variations of a page (or cover) using local multi-reference mflux
edit models, opens a side-by-side HTML comparison board (Original + Mask Overlay + Candidates),
and lets you pick a winner, roll more seeds, refine the prompt, or chain-edit off a candidate —
all without reloading the model weights.

When you accept a winner, it fixes the page EVERYWHERE:
  1. Backs up the previous art to books/<slug>/art/.history/
  2. Saves the winning PNG to art/page_XX.png (or art/cover.png)
  3. Updates story.json (render_meta + edit_history)
  4. Rebuilds the HTML flipbook & bookshelf (scripts/build_book.py --all)
  5. Optionally re-animates that single page & re-assembles the cartoon MP4

Usage:
    # Fast cached instruction edit (flux2-klein-4b):
    uv run python scripts/edit_page.py books/01-the-very-big-world-problem --page 1 \\
      "Remove the weird red rocky object sticking out of Chester's chin; show his clean green blocky pixelated chin pressed directly against the cracked globe" \\
      --count 4

    # Surgical Qwen3-VL auto-masked inpainting (qwen-image-2.1-turbo):
    uv run python scripts/edit_page.py books/01-the-very-big-world-problem --page 1 \\
      "Chester's clean green blocky pixelated chin pressed directly against the cracked blue globe" \\
      --model qwen-image-2.1-turbo --auto-mask "the red rocky object sticking out of the goblin boy's chin" --count 4

    # Explicit bounding-box mask x1,y1,x2,y2 (normalized 0..1 or pixel coords, works with ANY model):
    uv run python scripts/edit_page.py books/01-the-very-big-world-problem --page 1 \\
      "Chester's clean green blocky pixelated chin pressed directly against the cracked blue globe" \\
      --bbox 380,330,535,475 --count 4
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from generate_illustrations import (
    EDIT_MODELS,
    EditModelSpec,
    load_edit_model,
    reference_for_char,
    resolve_story_path,
    save_clean,
    save_story,
)
from llm import unload_lms_models
from prompts import build_scene_prompt, medium_lock_phrase
from roster import FEVER_DREAMS_ROOT


def build_edit_prompt(
    instruction: str,
    cast_in_scene: list[dict],
    *,
    include_cast_refs: bool,
) -> str:
    """Wrap a plain-English retouch instruction for multi-reference image editing."""
    ref_notes: list[str] = []
    if include_cast_refs:
        for idx, char in enumerate(cast_in_scene, start=2):
            lock = medium_lock_phrase(char["style_id"])
            ref_notes.append(
                f"Image {idx} is {char['name']} ({char['style_label']}) for character reference; "
                f"keep their exact face, silhouette, colors, and outfit from Image {idx} so they {lock}."
            )
    ref_block = (" " + " ".join(ref_notes)) if ref_notes else ""
    clean_instr = instruction.strip().rstrip(".")
    return (
        f"Edit Image 1 to apply this specific fix: {clean_instr}. "
        f"Keep everything else in Image 1 identical — the same composition, camera angle, lighting, "
        f"hand-painted 1950s Little Golden Book gouache classroom/background, props, and character poses — "
        f"changing ONLY what the fix requests.{ref_block} "
        f"Maintain the mixed-media Gumball-style contrast with natural shadows. "
        f"Zero text, zero speech bubbles, zero watermarks."
    )


def build_reroll_prompt(
    instruction: str,
    target: dict,
    cast_in_scene: list[dict],
    *,
    is_cover: bool,
    append_scene: bool = False,
) -> str:
    """Build a fresh from-scratch scene prompt incorporating the user's direction."""
    base_scene = target.get("scene_description", "").strip()
    clean_instr = instruction.strip()
    if append_scene and clean_instr and base_scene:
        scene = f"{base_scene.rstrip('.')}. Important art direction: {clean_instr}"
    else:
        scene = clean_instr or base_scene
    return build_scene_prompt(
        cast_in_scene,
        scene,
        target.get("camera_angle", "Medium storybook shot"),
        target.get("lighting_mood", ""),
        is_cover=is_cover,
    )


def build_bbox_mask(bbox_str: str, width: int, height: int, feather_px: int = 12) -> Image.Image:
    """Create a feathered grayscale ('L') mask from 'x1,y1,x2,y2' (0..1 normalized or pixel coords)."""
    parts = [float(x.strip()) for x in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--bbox must have 4 comma-separated numbers (x1,y1,x2,y2), got: {bbox_str!r}")
    x1, y1, x2, y2 = parts
    if max(parts) <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([x1, y1, x2, y2], radius=max(6, feather_px), fill=255)
    if feather_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_px))
    return mask


def scrub_masked_region(source_path: Path, mask_l: Image.Image, out_path: Path) -> Path:
    """Diffuse clean surrounding colors across the masked region so conditioning latents
    do not keep seeing the artifact being removed, while unmasked pixels stay untouched."""
    src_im = Image.open(source_path).convert("RGB")
    arr = np.asarray(src_im, dtype=np.float32).copy()
    m = np.asarray(mask_l.convert("L").resize(src_im.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    hole = m > 0.08
    if not np.any(hole):
        src_im.save(out_path, format="PNG")
        return out_path

    # Seed hole with coarse blurred neighbor average, then relax smoothly from boundary inward
    seed_im = src_im.resize((max(1, src_im.width // 16), max(1, src_im.height // 16)), Image.Resampling.BILINEAR)
    seed_im = seed_im.filter(ImageFilter.GaussianBlur(radius=2)).resize(src_im.size, Image.Resampling.BILINEAR)
    filled = arr.copy()
    filled[hole] = np.asarray(seed_im, dtype=np.float32)[hole]
    for _ in range(120):
        avg = 0.25 * (
            np.roll(filled, 1, axis=0)
            + np.roll(filled, -1, axis=0)
            + np.roll(filled, 1, axis=1)
            + np.roll(filled, -1, axis=1)
        )
        filled[hole] = avg[hole]

    blended = arr * (1.0 - m[..., None]) + filled * m[..., None]
    Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB").save(out_path, format="PNG")
    return out_path


def save_mask_overlay(source_path: Path, mask_l: Image.Image, out_path: Path) -> None:
    """Save a visual diagnostic image showing the repainted mask region tinted bright coral/red."""
    base = Image.open(source_path).convert("RGBA")
    m = mask_l.convert("L").resize(base.size, Image.Resampling.BILINEAR)
    # Dim unmasked area slightly and highlight masked area in translucent coral-red
    overlay = Image.new("RGBA", base.size, (255, 68, 68, 0))
    alpha = m.point(lambda v: int(v * 0.52))
    overlay.putalpha(alpha)
    comp = Image.alpha_composite(base, overlay)
    # Draw crisp contour hint around ~50% mask threshold
    comp.convert("RGB").save(out_path, format="PNG")


def resolve_preview_mask(
    model,
    spec: EditModelSpec,
    *,
    source_path: Path,
    width: int,
    height: int,
    mask_path: Path | None,
    bbox_str: str | None,
    auto_mask: str | None,
    feather_px: int = 10,
) -> Image.Image | None:
    """Resolve explicit mask, bbox mask, or Qwen3-VL auto_mask into a PIL 'L' mask for preview/blending."""
    if mask_path is not None:
        return Image.open(mask_path).convert("L").resize((width, height), Image.Resampling.BILINEAR)
    if bbox_str is not None:
        return build_bbox_mask(bbox_str, width, height, feather_px=feather_px)
    if auto_mask and spec.family == "qwen21":
        src_rgba = Image.open(source_path).convert("RGBA")
        return model._resolve_mask(None, auto_mask, src_rgba, width, height)
    return None


def render_candidate(
    model,
    spec: EditModelSpec,
    *,
    prompt: str,
    image_paths: list[Path],
    seed: int,
    width: int,
    height: int,
    steps: int | None,
    guidance: float | None,
    strength: float | None,
    resolved_mask: Image.Image | None,
    source_for_blend: Path | None,
) -> Image.Image:
    str_paths = [str(p) for p in image_paths]
    n_steps = steps if steps is not None else spec.default_steps
    g = guidance if guidance is not None else spec.default_guidance

    kwargs: dict = {
        "seed": seed,
        "prompt": prompt,
        "image_paths": str_paths,
        "num_inference_steps": n_steps,
        "width": width,
        "height": height,
        "guidance": g,
    }
    if spec.family == "qwen21":
        if strength is not None:
            kwargs["strength"] = strength
        if resolved_mask is not None:
            kwargs["mask_image"] = resolved_mask
        generated = model.generate_image(**kwargs)
        out_qwen = generated.image.convert("RGB")
        # If image_paths[0] was pre-scrubbed for cleaner conditioning, re-composite outside
        # the feathered mask against the true untouched source_for_blend!
        if resolved_mask is not None and source_for_blend is not None and source_for_blend.exists():
            orig_rgb = Image.open(source_for_blend).convert("RGB").resize(out_qwen.size, Image.Resampling.LANCZOS)
            mask_resized = resolved_mask.convert("L").resize(out_qwen.size, Image.Resampling.BILINEAR)
            out_qwen = Image.composite(out_qwen, orig_rgb, mask_resized)
        return out_qwen

    if spec.family == "flux2" and strength is not None:
        kwargs["image_strength"] = strength

    generated = model.generate_image(**kwargs)
    out_img = generated.image.convert("RGB")

    # For non-qwen21 models (e.g. flux2-klein-4b), if an explicit --mask or --bbox was provided,
    # composite the edited region smoothly onto the untouched original image!
    if resolved_mask is not None and source_for_blend is not None and source_for_blend.exists():
        orig_rgb = Image.open(source_for_blend).convert("RGB").resize(out_img.size, Image.Resampling.LANCZOS)
        mask_resized = resolved_mask.convert("L").resize(out_img.size, Image.Resampling.BILINEAR)
        out_img = Image.composite(out_img, orig_rgb, mask_resized)

    return out_img


def write_picker_html(
    board_path: Path,
    *,
    book_title: str,
    target_label: str,
    mode: str,
    model_key: str,
    instruction: str,
    source_rel: str,
    source_label: str,
    mask_overlay_rel: str | None,
    mask_desc: str | None,
    candidates: list[dict],
    round_num: int,
) -> None:
    bust = int(time.time() * 1000)
    mask_card = ""
    if mask_overlay_rel:
        m_caption = html.escape(mask_desc or "Repaint Mask")
        mask_card = f"""
    <figure class="card original" onclick="openLightbox(this.querySelector('img').src, 'Mask Overlay — {m_caption}')">
      <div class="badge">Mask Region</div>
      <img src="{html.escape(mask_overlay_rel)}?v={bust}" alt="Mask Overlay">
      <figcaption>
        <strong>Repaint Zone (Red)</strong>
        <span class="meta">{m_caption}</span>
      </figcaption>
    </figure>"""

    cards = []
    for c in candidates:
        cards.append(
            f"""
            <figure class="card" onclick="openLightbox(this.querySelector('img').src, '{html.escape(c['title'])}')">
              <div class="badge">#{c['idx']}</div>
              <img src="{html.escape(c['rel_path'])}?v={bust}" alt="Candidate #{c['idx']}">
              <figcaption>
                <strong>Candidate #{c['idx']}</strong>
                <span class="meta">seed {c['seed']} • {c['elapsed_s']:.1f}s</span>
              </figcaption>
            </figure>"""
        )

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Retouch Picker — {html.escape(target_label)} ({html.escape(book_title)})</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #171411;
    color: #f6eedb;
    margin: 0;
    padding: 24px 28px 40px;
  }}
  header {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 14px; }}
  h1 {{ margin: 0; font-family: Georgia, serif; color: #f3d37a; font-size: 24px; }}
  .pill {{ background: #2c241b; border: 1px solid #4d3e2d; color: #e8cfa0; padding: 4px 10px; border-radius: 999px; font-size: 12px; }}
  .prompt-box {{
    background: #231e19;
    border: 1px solid #3c3227;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 20px;
    font-size: 14px;
    line-height: 1.45;
  }}
  .prompt-box strong {{ color: #f3d37a; }}
  .hint {{ color: #b9ab94; font-size: 13px; margin: 0 0 18px; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 18px;
  }}
  .card {{
    position: relative;
    margin: 0;
    background: #211c17;
    border-radius: 10px;
    overflow: hidden;
    border: 2px solid #392f25;
    cursor: zoom-in;
    transition: transform 0.12s ease, border-color 0.12s ease;
  }}
  .card:hover {{ transform: translateY(-2px); border-color: #f3d37a; }}
  .card.original {{ border-color: #6e5a3b; }}
  .card img {{ width: 100%; display: block; }}
  .badge {{
    position: absolute;
    top: 10px;
    left: 10px;
    background: rgba(20, 16, 12, 0.88);
    color: #f3d37a;
    border: 1px solid #f3d37a;
    font-weight: 800;
    font-size: 15px;
    padding: 4px 10px;
    border-radius: 999px;
  }}
  .card.original .badge {{ color: #d5c7b0; border-color: #9e8a6c; }}
  figcaption {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    font-size: 13px;
  }}
  .meta {{ color: #bfb19a; font-variant-numeric: tabular-nums; font-size: 12px; text-align: right; }}
  #lightbox {{
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(10, 8, 6, 0.92);
    z-index: 99;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 12px;
    padding: 20px;
    cursor: zoom-out;
  }}
  #lightbox.open {{ display: flex; }}
  #lightbox img {{ max-width: 94vw; max-height: 86vh; border-radius: 8px; box-shadow: 0 20px 60px rgba(0,0,0,0.8); }}
  #lb-cap {{ color: #f3d37a; font-weight: 700; font-size: 15px; }}
</style>
</head>
<body>
  <header>
    <h1>{html.escape(book_title)} — {html.escape(target_label)}</h1>
    <span class="pill">Round {round_num} • {html.escape(model_key)} • mode={html.escape(mode)}</span>
  </header>
  <div class="prompt-box"><strong>Instruction:</strong> {html.escape(instruction)}</div>
  <p class="hint">Click any image to zoom full-size. Choose <strong>1–{len(candidates)}</strong> (accept), <strong>m</strong> (more seeds), <strong>p</strong> (edit prompt), <strong>e &lt;n&gt;</strong> (chain-edit off #n), or <strong>q</strong> (quit).</p>
  <div class="grid">
    <figure class="card original" onclick="openLightbox(this.querySelector('img').src, '{html.escape(source_label)}')">
      <div class="badge">{html.escape(source_label)}</div>
      <img src="{html.escape(source_rel)}?v={bust}" alt="{html.escape(source_label)}">
      <figcaption>
        <strong>{html.escape(source_label)}</strong>
        <span class="meta">Before</span>
      </figcaption>
    </figure>
    {mask_card}
    {''.join(cards)}
  </div>
  <div id="lightbox" onclick="this.classList.remove('open')">
    <img id="lb-img" src="" alt="">
    <div id="lb-cap"></div>
  </div>
  <script>
    function openLightbox(src, cap) {{
      document.getElementById('lb-img').src = src;
      document.getElementById('lb-cap').textContent = cap;
      document.getElementById('lightbox').classList.add('open');
    }}
    document.addEventListener('keydown', e => {{
      if (e.key === 'Escape') document.getElementById('lightbox').classList.remove('open');
    }});
  </script>
</body>
</html>
"""
    board_path.write_text(doc, encoding="utf-8")


def open_in_browser(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)


def run_book_rebuild() -> None:
    print("\nRebuilding HTML flipbook & bookshelf...")
    cmd = [sys.executable, str(FEVER_DREAMS_ROOT / "scripts" / "build_book.py"), "--all"]
    subprocess.run(cmd, check=True)


def run_cartoon_refresh(book_dir: Path, page_number: int, *, frames: int = 121) -> None:
    cartoon_dir = book_dir / "cartoon"
    script_json = cartoon_dir / "script.json"
    voiced_json = cartoon_dir / "script_voiced.json"
    cartoon_py = FEVER_DREAMS_ROOT / "cartoon" / ".venv" / "bin" / "python"

    if not script_json.exists():
        print(f"No cartoon script found at {script_json}; skipping cartoon refresh.")
        return
    if not cartoon_py.exists():
        print(f"Cartoon venv not found at {cartoon_py}; skipping cartoon refresh.", file=sys.stderr)
        return

    script_arg = (
        script_json.relative_to(FEVER_DREAMS_ROOT).as_posix()
        if script_json.is_relative_to(FEVER_DREAMS_ROOT)
        else str(script_json)
    )
    print(f"\nRe-animating cartoon shot for page {page_number} (LTX-2.5, {frames} frames)...")
    subprocess.run(
        [
            str(cartoon_py),
            "cartoon/animate_book.py",
            script_arg,
            "--only-pages",
            str(page_number),
            "--frames",
            str(frames),
            "--overwrite",
        ],
        cwd=str(FEVER_DREAMS_ROOT),
        check=True,
    )

    if voiced_json.exists():
        voiced_arg = (
            voiced_json.relative_to(FEVER_DREAMS_ROOT).as_posix()
            if voiced_json.is_relative_to(FEVER_DREAMS_ROOT)
            else str(voiced_json)
        )
        print("\nRe-assembling final cartoon MP4 & subtitles...")
        subprocess.run(
            [
                str(cartoon_py),
                "cartoon/assemble_cartoon.py",
                voiced_arg,
            ],
            cwd=str(FEVER_DREAMS_ROOT),
            check=True,
        )
    else:
        print(f"Note: {voiced_json} not found; skipped assemble_cartoon.py.")


def apply_winner(
    *,
    winner: dict,
    art_path: Path,
    book_dir: Path,
    target_slug: str,
    target_label: str,
    target: dict,
    story: dict,
    story_path: Path,
    page_num_for_cartoon: int,
    no_rebuild: bool,
    cartoon_flag: bool,
    ask_cartoon: bool,
    frames: int,
) -> None:
    # 1. Backup existing art
    art_path.parent.mkdir(parents=True, exist_ok=True)
    if art_path.exists():
        hist_dir = art_path.parent / ".history"
        hist_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = hist_dir / f"{target_slug}_{stamp}.png"
        shutil.copy2(art_path, backup_path)
        print(f"\nBacked up previous art to {backup_path.relative_to(book_dir)}")

    # 2. Install winning PNG
    shutil.copy2(winner["path"], art_path)
    print(f"Saved winner (#{winner['idx']}, seed={winner['seed']}) -> {art_path.relative_to(book_dir)}")

    # 3. Record metadata in story.json
    target["image"] = art_path.relative_to(book_dir).as_posix()
    meta = {
        "model": winner["model"],
        "seed": winner["seed"],
        "steps": winner["steps"],
        "refs": winner["refs"],
        "elapsed_s": winner["elapsed_s"],
        "mode": winner["mode"],
        "edit_instruction": winner["instruction"],
    }
    if winner.get("mask_desc"):
        meta["mask"] = winner["mask_desc"]
    target["render_meta"] = meta
    target.setdefault("edit_history", []).append(
        {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": winner["mode"],
            "model": winner["model"],
            "seed": winner["seed"],
            "mask": winner.get("mask_desc"),
            "instruction": winner["instruction"],
        }
    )
    save_story(story_path, story)
    print(f"Updated {story_path.relative_to(FEVER_DREAMS_ROOT)}")

    # 4. Rebuild HTML flipbook & bookshelf
    if not no_rebuild:
        run_book_rebuild()

    # 5. Refresh cartoon clip + final MP4 if requested or confirmed
    cartoon_script = book_dir / "cartoon" / "script.json"
    if cartoon_script.exists():
        do_cartoon = cartoon_flag
        if not do_cartoon and ask_cartoon:
            try:
                ans = input(
                    f"\nAlso update the cartoon (re-animate {target_label} + re-assemble MP4)? [y/N] "
                ).strip().lower()
                do_cartoon = ans in ("y", "yes")
            except (EOFError, KeyboardInterrupt):
                do_cartoon = False
        if do_cartoon:
            run_cartoon_refresh(book_dir, page_num_for_cartoon, frames=frames)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively retouch or re-roll a book page using local mflux models."
    )
    parser.add_argument("book", type=Path, help="Path to books/<slug> directory or books/<slug>/story.json")
    parser.add_argument("instruction", nargs="?", default=None, help="What to fix or change in the illustration")
    target_grp = parser.add_mutually_exclusive_group(required=True)
    target_grp.add_argument("--page", type=int, help="Page number to edit (1-based)")
    target_grp.add_argument("--cover", action="store_true", help="Edit the front cover instead of a story page")
    parser.add_argument(
        "--mode",
        choices=["edit", "reroll"],
        default="edit",
        help="'edit' modifies the existing illustration in place (default); 'reroll' generates fresh from scratch",
    )
    parser.add_argument(
        "--model",
        default="flux2-klein-4b",
        choices=list(EDIT_MODELS.keys()),
        help="Local mflux edit model (default: flux2-klein-4b, fast & cached)",
    )
    parser.add_argument(
        "--append-scene",
        action="store_true",
        help="In reroll mode, append your instruction to story.json's existing scene_description instead of replacing it",
    )
    parser.add_argument(
        "--replace-scene",
        action="store_true",
        help="In reroll mode, replace story.json's scene_description with your instruction (now the default when an instruction is given)",
    )
    parser.add_argument("--count", type=int, default=4, help="Number of candidate examples per round (default: 4)")
    parser.add_argument("--seed", type=int, default=None, help="Starting seed (default: random)")
    parser.add_argument("--quantize", type=int, default=None, choices=[3, 4, 5, 6, 8], help="Override quantization bits")
    parser.add_argument("--steps", type=int, default=None, help="Override inference steps")
    parser.add_argument("--guidance", type=float, default=None, help="Override guidance scale")
    parser.add_argument("--strength", type=float, default=None, help="Optional edit strength override (0..1)")
    parser.add_argument(
        "--auto-mask",
        type=str,
        default=None,
        help="Text description of region to auto-mask via Qwen3-VL (supported by qwen-image-2.1-turbo / qwen-image-2.1)",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="Explicit feathered bounding-box mask 'x1,y1,x2,y2' in pixels or 0..1 coords (works with ANY model)",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Optional grayscale mask PNG path, white=repaint black=keep (works with ANY model)",
    )
    parser.add_argument(
        "--feather",
        type=int,
        default=10,
        help="Feather blur radius in pixels for --bbox masks (default: 10)",
    )
    parser.add_argument(
        "--scrub-mask",
        action="store_true",
        help="Pre-scrub the masked artifact from the conditioning input so edit models don't copy remnants of it",
    )
    parser.add_argument(
        "--no-cast-refs",
        action="store_true",
        help="In edit mode, pass only the current illustration without character reference portraits",
    )
    parser.add_argument(
        "--no-use-sheets",
        action="store_true",
        help="Use raw yearbook portraits even if turnaround sheets exist",
    )
    parser.add_argument("--keep-lms", action="store_true", help="Do not run `lms unload --all` before loading mflux")
    parser.add_argument("--cache-limit-gb", type=float, default=8.0, help="MLX Metal cache limit in GB (default: 8.0)")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open the HTML comparison board")
    parser.add_argument("--no-rebuild", action="store_true", help="Do not auto-run build_book.py after accepting")
    parser.add_argument(
        "--cartoon",
        action="store_true",
        help="After accepting, automatically re-animate this page and re-assemble the cartoon MP4 without asking",
    )
    parser.add_argument("--frames", type=int, default=121, help="LTX-2.5 frames when refreshing cartoon (default: 121)")
    parser.add_argument(
        "--batch-only",
        action="store_true",
        help="Render one batch of candidates + HTML board and exit without prompting (pair with --accept N later)",
    )
    parser.add_argument(
        "--accept",
        type=int,
        default=None,
        help="Accept candidate #N from the latest batch in .retouch/ without re-rendering",
    )
    parser.add_argument(
        "--from-candidate",
        type=int,
        default=None,
        help="Start an edit pass using candidate #N from the latest batch as Image 1 (instead of art/page_XX.png)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Start an edit pass using an explicit image file path as Image 1",
    )
    args = parser.parse_args()

    story_path = resolve_story_path(args.book).resolve()
    if not story_path.exists():
        print(f"Error: story.json not found at {story_path}", file=sys.stderr)
        sys.exit(1)

    book_dir = story_path.parent
    story = json.loads(story_path.read_text(encoding="utf-8"))
    cast_by_id = {c["id"]: c for c in story["cast"]}

    if args.cover:
        target = story["cover"]
        target_slug = "cover"
        target_label = "Front Cover"
        page_num_for_cartoon = 0
        art_path = book_dir / (target.get("image") or "art/cover.png")
        default_w, default_h = 768, 960
    else:
        matches = [p for p in story["pages"] if int(p["page_number"]) == args.page]
        if not matches:
            print(f"Error: page {args.page} not found in {story_path}", file=sys.stderr)
            sys.exit(1)
        target = matches[0]
        target_slug = f"page_{args.page:02d}"
        target_label = f"Page {args.page:02d}"
        page_num_for_cartoon = args.page
        art_path = book_dir / (target.get("image") or f"art/page_{args.page:02d}.png")
        default_w, default_h = 896, 768

    session_dir = book_dir / ".retouch" / target_slug
    manifest_path = session_dir / "latest_batch.json"

    # Fast path: --accept N applies an already-rendered candidate from .retouch/
    if args.accept is not None:
        if not manifest_path.exists():
            print(f"Error: no saved candidate batch found at {manifest_path}", file=sys.stderr)
            sys.exit(1)
        saved_batch = json.loads(manifest_path.read_text(encoding="utf-8"))
        matches_c = [c for c in saved_batch["candidates"] if int(c["idx"]) == args.accept]
        if not matches_c:
            print(f"Error: candidate #{args.accept} not in {manifest_path}", file=sys.stderr)
            sys.exit(1)
        w = matches_c[0]
        w["path"] = Path(w["path"])
        apply_winner(
            winner=w,
            art_path=art_path,
            book_dir=book_dir,
            target_slug=target_slug,
            target_label=target_label,
            target=target,
            story=story,
            story_path=story_path,
            page_num_for_cartoon=page_num_for_cartoon,
            no_rebuild=args.no_rebuild,
            cartoon_flag=args.cartoon,
            ask_cartoon=not args.batch_only,
            frames=args.frames,
        )
        print("\nAll done!")
        return

    # Resolve optional custom base image (--from-candidate N or --source path.png)
    custom_base_src: Path | None = None
    custom_base_label = "Original"
    if args.source is not None:
        custom_base_src = args.source.resolve()
        if not custom_base_src.exists():
            print(f"Error: --source file not found at {custom_base_src}", file=sys.stderr)
            sys.exit(1)
        custom_base_label = f"Base: {custom_base_src.name}"
    elif args.from_candidate is not None:
        if not manifest_path.exists():
            print(f"Error: no saved candidate batch found at {manifest_path}", file=sys.stderr)
            sys.exit(1)
        saved_batch = json.loads(manifest_path.read_text(encoding="utf-8"))
        matches_c = [c for c in saved_batch["candidates"] if int(c["idx"]) == args.from_candidate]
        if not matches_c:
            print(f"Error: candidate #{args.from_candidate} not in {manifest_path}", file=sys.stderr)
            sys.exit(1)
        custom_base_src = Path(matches_c[0]["path"]).resolve()
        custom_base_label = f"Base: Candidate #{args.from_candidate} (seed {matches_c[0]['seed']})"

    effective_mode = "edit" if custom_base_src is not None else args.mode

    if effective_mode == "edit" and custom_base_src is None and not art_path.exists():
        print(f"Error: {art_path} does not exist yet. Use --mode reroll to generate from scratch.", file=sys.stderr)
        sys.exit(1)

    if args.auto_mask and EDIT_MODELS[args.model].family != "qwen21":
        print(
            f"Error: --auto-mask requires Qwen3-VL (--model qwen-image-2.1-turbo or qwen-image-2.1). "
            f"For {args.model}, use --bbox x1,y1,x2,y2 or --mask mask.png instead.",
            file=sys.stderr,
        )
        sys.exit(1)

    instruction = args.instruction or ""
    if not instruction and effective_mode == "edit":
        try:
            instruction = input(f"Describe what to change on {target_label}: ").strip()
        except EOFError:
            instruction = ""
        if not instruction:
            print("No instruction provided; exiting.", file=sys.stderr)
            sys.exit(1)

    width, height = default_w, default_h
    if art_path.exists():
        with Image.open(art_path) as im:
            width, height = im.size

    use_sheets = not args.no_use_sheets
    cast_in_scene = [cast_by_id[cid] for cid in target.get("cast_ids", []) if cid in cast_by_id]
    cast_refs = [reference_for_char(c, book_dir, story, use_sheets) for c in cast_in_scene]

    if not args.keep_lms:
        print("Unloading LM Studio models to free Apple Silicon unified memory...")
        unload_lms_models()

    import mlx.core as mx

    if args.cache_limit_gb > 0:
        mx.set_cache_limit(int(args.cache_limit_gb * (1024**3)))

    print(f"Loading {args.model} ({width}x{height})...")
    model, spec = load_edit_model(args.model, quantize=args.quantize)

    session_dir.mkdir(parents=True, exist_ok=True)
    board_path = session_dir / "index.html"

    orig_copy = session_dir / "original.png"
    if art_path.exists():
        shutil.copy2(art_path, orig_copy)

    if custom_base_src is not None:
        base_copy = session_dir / f"base_ref_{int(time.time())}.png"
        shutil.copy2(custom_base_src, base_copy)
        current_source = base_copy
        source_label = custom_base_label
    else:
        current_source = orig_copy if orig_copy.exists() else art_path
        source_label = "Original"
    mode = effective_mode
    next_seed = args.seed if args.seed is not None else random.randint(100000, 899999)
    round_num = 0
    winner: dict | None = None

    try:
        while True:
            round_num += 1
            resolved_mask: Image.Image | None = None
            mask_overlay_rel: str | None = None
            mask_desc: str | None = None

            if mode == "edit":
                full_prompt = build_edit_prompt(
                    instruction,
                    cast_in_scene,
                    include_cast_refs=not args.no_cast_refs,
                )
                resolved_mask = resolve_preview_mask(
                    model,
                    spec,
                    source_path=current_source,
                    width=width,
                    height=height,
                    mask_path=args.mask,
                    bbox_str=args.bbox,
                    auto_mask=args.auto_mask,
                    feather_px=args.feather,
                )
                cond_first = current_source
                if resolved_mask is not None:
                    mask_png = session_dir / f"r{round_num:02d}_mask.png"
                    overlay_png = session_dir / f"r{round_num:02d}_mask_overlay.png"
                    resolved_mask.save(mask_png, format="PNG")
                    save_mask_overlay(current_source, resolved_mask, overlay_png)
                    mask_overlay_rel = overlay_png.name
                    mask_desc = (
                        f"auto_mask: {args.auto_mask}"
                        if args.auto_mask
                        else (f"bbox: {args.bbox}" if args.bbox else f"mask: {args.mask.name if args.mask else ''}")
                    )
                    if args.scrub_mask:
                        scrub_png = session_dir / f"r{round_num:02d}_scrubbed.png"
                        cond_first = scrub_masked_region(current_source, resolved_mask, scrub_png)
                        mask_desc += " + scrub"
                    print(f"  Mask resolved ({mask_desc}) -> {overlay_png.name}")
                image_paths = [cond_first] + ([] if args.no_cast_refs else cast_refs)
            else:
                full_prompt = build_reroll_prompt(
                    instruction,
                    target,
                    cast_in_scene,
                    is_cover=args.cover,
                    append_scene=args.append_scene and not args.replace_scene,
                )
                image_paths = list(cast_refs)

            print(
                f"\n=== Round {round_num} ({target_label} | mode={mode} | model={spec.key} | "
                f"{len(image_paths)} ref(s)) ==="
            )
            print(f"Instruction: {instruction}")

            candidates: list[dict] = []
            for idx in range(1, args.count + 1):
                seed = next_seed
                next_seed += 97
                t0 = time.time()
                print(f"  [{idx}/{args.count}] Rendering candidate #{idx} (seed={seed})...", flush=True)
                out_pil = render_candidate(
                    model,
                    spec,
                    prompt=full_prompt,
                    image_paths=image_paths,
                    seed=seed,
                    width=width,
                    height=height,
                    steps=args.steps,
                    guidance=args.guidance,
                    strength=args.strength,
                    resolved_mask=resolved_mask,
                    source_for_blend=current_source if mode == "edit" else None,
                )
                out_file = session_dir / f"r{round_num:02d}_cand_{idx:02d}_s{seed}.png"
                out_pil.save(str(out_file), format="PNG")
                elapsed = time.time() - t0
                print(f"      -> {out_file.name} ({elapsed:.1f}s)")
                candidates.append(
                    {
                        "idx": idx,
                        "title": f"Candidate #{idx} (seed {seed})",
                        "path": str(out_file),
                        "rel_path": out_file.name,
                        "seed": seed,
                        "elapsed_s": round(elapsed, 1),
                        "prompt": full_prompt,
                        "refs": [p.name for p in image_paths],
                        "mode": mode,
                        "model": spec.key,
                        "steps": args.steps or spec.default_steps,
                        "mask_desc": mask_desc,
                        "instruction": instruction,
                    }
                )
                mx.clear_cache()

            manifest_path.write_text(
                json.dumps({"round": round_num, "board": str(board_path), "candidates": candidates}, indent=2) + "\n",
                encoding="utf-8",
            )
            write_picker_html(
                board_path,
                book_title=story.get("title", book_dir.name),
                target_label=target_label,
                mode=mode,
                model_key=spec.key,
                instruction=instruction,
                source_rel=current_source.name,
                source_label=source_label,
                mask_overlay_rel=mask_overlay_rel,
                mask_desc=mask_desc,
                candidates=candidates,
                round_num=round_num,
            )
            print(f"\nComparison board ready: {board_path}")
            if not args.no_open:
                open_in_browser(board_path)

            if args.batch_only:
                print(
                    f"\nBatch complete! To accept a candidate and update book/cartoon:\n"
                    f"  uv run python scripts/edit_page.py {book_dir.relative_to(FEVER_DREAMS_ROOT)} "
                    f"{'--cover' if args.cover else f'--page {args.page}'} --accept <1-{len(candidates)}> --cartoon"
                )
                return

            while True:
                print(
                    f"\nPick a winner [1-{len(candidates)}], "
                    f"'m' more seeds, 'p' edit prompt, 'e <n>' chain-edit off #n, "
                    f"'mode' switch edit/reroll, or 'q' quit:"
                )
                try:
                    choice = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    choice = "q"

                low = choice.lower()
                if low in ("q", "quit", "exit"):
                    print("Exited without modifying the book.")
                    return

                if low in ("m", "more"):
                    break

                if low in ("p", "prompt"):
                    new_instr = input("New instruction: ").strip()
                    if new_instr:
                        instruction = new_instr
                        break
                    continue

                if low == "mode":
                    mode = "reroll" if mode == "edit" else "edit"
                    print(f"Switched mode to: {mode}")
                    break

                if low.startswith("e "):
                    parts = low.split()
                    if len(parts) == 2 and parts[1].isdigit():
                        c_idx = int(parts[1])
                        if 1 <= c_idx <= len(candidates):
                            chosen = candidates[c_idx - 1]
                            current_source = Path(chosen["path"])
                            source_label = f"Base: Round {round_num} #{c_idx}"
                            mode = "edit"
                            new_instr = input(
                                f"Next edit instruction starting from #{c_idx} (Enter to reuse current): "
                            ).strip()
                            if new_instr:
                                instruction = new_instr
                            break
                    print(f"Usage: e <1-{len(candidates)}>")
                    continue

                if choice.isdigit():
                    c_idx = int(choice)
                    if 1 <= c_idx <= len(candidates):
                        winner = dict(candidates[c_idx - 1])
                        winner["path"] = Path(winner["path"])
                        break
                    print(f"Please enter a number between 1 and {len(candidates)}.")
                    continue

                print("Unrecognized input. Enter a candidate number, 'm', 'p', 'e <n>', 'mode', or 'q'.")

            if winner is not None:
                break
    finally:
        del model
        gc.collect()
        mx.clear_cache()

    assert winner is not None
    apply_winner(
        winner=winner,
        art_path=art_path,
        book_dir=book_dir,
        target_slug=target_slug,
        target_label=target_label,
        target=target,
        story=story,
        story_path=story_path,
        page_num_for_cartoon=page_num_for_cartoon,
        no_rebuild=args.no_rebuild,
        cartoon_flag=args.cartoon,
        ask_cartoon=True,
        frames=args.frames,
    )
    print("\nAll done!")


if __name__ == "__main__":
    main()
