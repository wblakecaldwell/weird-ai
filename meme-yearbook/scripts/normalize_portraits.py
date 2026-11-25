"""Normalize raw generated portraits into cohesive Class of 2025 yearbook photos.

Applies:
  - Exact 4:5 portrait framing (800x1000)
  - Crisp nearest-neighbor pixel quantization for `pixel_16bit` students (eliminates mushy "mixels")
  - Consistent subtle yearbook studio vignette, print tonal curve, and light matte paper grain

Usage:
    uv run python scripts/normalize_portraits.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from styles import STYLES

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROSTER = ROOT / "data" / "roster.json"
RAW_DIR = ROOT / "portraits" / "raw"
FINAL_DIR = ROOT / "portraits" / "final"
THUMB_DIR = ROOT / "portraits" / "thumb"


def apply_pixel_snap(img: Image.Image, grid_width: int) -> Image.Image:
    """Downscale cleanly then nearest-neighbor upscale so pixel characters look razor-sharp."""
    w, h = img.size
    grid_h = max(1, round(grid_width * (h / w)))
    # Use BOX downsampling to average softly into clean sprite pixels, then quantize palette lightly
    small = img.resize((grid_width, grid_h), resample=Image.Resampling.BOX)
    small = small.quantize(colors=48, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).convert("RGB")
    return small.resize((w, h), resample=Image.Resampling.NEAREST)


def apply_yearbook_finish(img: Image.Image, seed: int, is_pixel: bool = False) -> Image.Image:
    """Give every portrait the subtle unified feel of a real 2025 printed school yearbook."""
    # Gentle tonal standardization (slight contrast & warmth cohesion)
    if not is_pixel:
        img = ImageEnhance.Contrast(img).enhance(1.04)
        img = ImageEnhance.Color(img).enhance(0.96)

    arr = np.asarray(img, dtype=np.float32)
    h, w, _ = arr.shape

    # Soft studio vignette (subtle edge falloff like real school portrait strobe lighting)
    y = np.linspace(-1.0, 1.0, h)[:, None]
    x = np.linspace(-1.0, 1.0, w)[None, :]
    r2 = x * x + y * y
    vignette_strength = 0.08 if is_pixel else 0.14
    vignette = 1.0 - vignette_strength * np.clip(r2 - 0.35, 0.0, 1.5)
    arr *= vignette[:, :, None]

    # Very fine deterministic paper/print grain so mixed mediums sit on the same page stock
    rng = np.random.default_rng(seed)
    grain_sigma = 1.8 if is_pixel else 3.2
    grain = rng.normal(0.0, grain_sigma, size=(h, w, 1)).astype(np.float32)
    arr = np.clip(arr + grain, 0.0, 255.0).astype(np.uint8)

    return Image.fromarray(arr, mode="RGB")


def process_portrait(
    raw_path: Path,
    out_path: Path,
    thumb_path: Path,
    style_id: str,
    seed: int,
    width: int,
    height: int,
    thumb_width: int = 480,
    thumb_height: int = 600,
) -> None:
    spec = STYLES.get(style_id)
    img = Image.open(raw_path).convert("RGB")
    img = ImageOps.fit(img, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.42))

    is_pixel = bool(spec and spec.pixelate_grid)
    if is_pixel and spec and spec.pixelate_grid:
        img = apply_pixel_snap(img, spec.pixelate_grid)

    img = apply_yearbook_finish(img, seed=seed, is_pixel=is_pixel)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if is_pixel:
        img.save(out_path, format="WEBP", lossless=True, quality=100)
    else:
        img.save(out_path, format="WEBP", quality=90, method=6)

    # Fast 2x-Retina WebP thumbnail for yearbook page grids (full 800x1000 WebP opens in the modal)
    thumb_resample = Image.Resampling.NEAREST if is_pixel else Image.Resampling.LANCZOS
    thumb = img.resize((thumb_width, thumb_height), resample=thumb_resample)
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    if is_pixel:
        thumb.save(thumb_path, format="WEBP", lossless=True, quality=100)
    else:
        thumb.save(thumb_path, format="WEBP", quality=88, method=6)


def ensure_portraits(character: dict) -> dict[str, str | None]:
    portraits = character.get("portraits")
    if not isinstance(portraits, dict):
        portraits = {}
        character["portraits"] = portraits
    for key in ("raw", "final", "thumb"):
        portraits.setdefault(key, None)
    return portraits


def resolve_raw_path(student: dict, portraits: dict[str, str | None]) -> Path | None:
    variant = int(student.get("selected_variant", 0))
    variant_path = RAW_DIR / f"{student['id']}_v{variant}.png"
    existing_raw = portraits.get("raw")

    # Respect custom raw paths outside the default variant naming convention
    if existing_raw:
        custom_path = ROOT / existing_raw
        is_default_variant = custom_path.parent == RAW_DIR and custom_path.name.startswith(f"{student['id']}_v")
        if custom_path.exists() and not is_default_variant:
            return custom_path

    if variant_path.exists():
        return variant_path

    if existing_raw and (ROOT / existing_raw).exists():
        return ROOT / existing_raw

    fallback = RAW_DIR / f"{student['id']}_v0.png"
    return fallback if fallback.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize raw portraits for the Class of 2025 yearbook.")
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--thumb-width", type=int, default=480)
    parser.add_argument("--thumb-height", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true", help="Re-normalize even if outputs are already up to date")
    args = parser.parse_args()

    if not args.roster.exists():
        print(f"Error: roster not found at {args.roster}", file=sys.stderr)
        sys.exit(1)

    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    subjects = list(roster["students"])
    if roster.get("principal"):
        subjects.insert(0, roster["principal"])
    count = 0
    skipped = 0
    roster_dirty = False

    for student in subjects:
        had_portraits = isinstance(student.get("portraits"), dict)
        portraits = ensure_portraits(student)
        if not had_portraits:
            roster_dirty = True

        raw_path = resolve_raw_path(student, portraits)
        if raw_path is None:
            continue

        out_path = FINAL_DIR / f"{student['id']}.webp"
        thumb_path = THUMB_DIR / f"{student['id']}.webp"
        raw_mtime = raw_path.stat().st_mtime
        needs_render = (
            args.overwrite
            or not out_path.exists()
            or not thumb_path.exists()
            or out_path.stat().st_mtime < raw_mtime
            or thumb_path.stat().st_mtime < raw_mtime
        )

        if needs_render:
            process_portrait(
                raw_path=raw_path,
                out_path=out_path,
                thumb_path=thumb_path,
                style_id=student["style_id"],
                seed=int(student.get("seed", 2025)),
                width=args.width,
                height=args.height,
                thumb_width=args.thumb_width,
                thumb_height=args.thumb_height,
            )
            count += 1
            print(f"Normalized {student['name']} -> {out_path.relative_to(ROOT)} + {thumb_path.relative_to(ROOT)}")
        else:
            skipped += 1

        desired = {
            "raw": raw_path.relative_to(ROOT).as_posix(),
            "final": out_path.relative_to(ROOT).as_posix(),
            "thumb": thumb_path.relative_to(ROOT).as_posix(),
        }
        for key, rel in desired.items():
            if portraits.get(key) != rel:
                portraits[key] = rel
                roster_dirty = True

    if roster_dirty:
        args.roster.write_text(json.dumps(roster, indent=2) + "\n", encoding="utf-8")
        print(f"Updated portrait paths in {args.roster.relative_to(ROOT) if args.roster.is_relative_to(ROOT) else args.roster}")

    print(f"\nNormalized {count} portrait(s) ({skipped} already up to date).")


if __name__ == "__main__":
    main()
