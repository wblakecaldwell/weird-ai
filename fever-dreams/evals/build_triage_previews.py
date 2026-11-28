"""Stitch compact 1-image side-by-side review sheets for AI pre-rating batches.

Selects N balanced unlabeled shots (round-robin across categories and both styles),
renders one annotated JPEG per shot showing:
  - Top: CANDIDATE (large) + PAIRED BASELINE (if any)
  - Bottom: Canonical Refs (Cast Ref 1..N, Prop Ref, Master Set Plate)
and writes batch manifest JSON files for parallel subagent inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

FEVER_DREAMS_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = FEVER_DREAMS_ROOT.parent
DATA_DIR = FEVER_DREAMS_ROOT / "evals" / "mined_data"


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def resolve_img(rel: str) -> Path | None:
    rel = rel.lstrip("/")
    for base in (FEVER_DREAMS_ROOT, WORKSPACE_ROOT):
        p = (base / rel).resolve()
        if p.is_file():
            return p
    return None


def fit_tile(im: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / im.width, max_h / im.height)
    nw, nh = max(1, int(im.width * scale)), max(1, int(im.height * scale))
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def stitch_shot_preview(
    shot: dict[str, Any],
    shots_by_id: dict[str, dict[str, Any]],
    out_path: Path,
) -> Path:
    sc = shot.get("scene") or {}
    cand_p = DATA_DIR / "shots" / shot["image"]
    cand_im = Image.open(cand_p).convert("RGB")

    paired_id = sc.get("paired_with")
    paired_im = None
    if paired_id and paired_id in shots_by_id:
        pp = DATA_DIR / "shots" / shots_by_id[paired_id]["image"]
        if pp.exists():
            paired_im = Image.open(pp).convert("RGB")

    refs: list[tuple[str, Image.Image]] = []
    crefs = sc.get("canonical_refs") or {}
    for i, r in enumerate(crefs.get("cast") or [], start=1):
        rp = resolve_img(r)
        if rp:
            refs.append((f"CAST REF {i}", Image.open(rp).convert("RGB")))
    if crefs.get("prop"):
        rp = resolve_img(crefs["prop"])
        if rp:
            refs.append(("PROP REF", Image.open(rp).convert("RGB")))
    if crefs.get("set"):
        rp = resolve_img(crefs["set"])
        if rp:
            refs.append(("SET PLATE", Image.open(rp).convert("RGB")))

    pad = 12
    hdr_h = 34
    top_h = 460
    ref_h = 190 if refs else 0
    canvas_w = 1120
    canvas_h = hdr_h + pad + top_h + (pad + ref_h if refs else 0) + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), (20, 18, 15))
    draw = ImageDraw.Draw(canvas)
    font_hdr = load_font(18)
    font_lbl = load_font(14)

    header_txt = (
        f"{shot['id']}  |  cat: {sc.get('category', '?')}  |  style: {sc.get('style', '?')}  |  "
        f"intent: {sc.get('expected_continuity', '?')}  |  cast_count: {len(sc.get('cast_ids') or [])}"
    )
    draw.rectangle([0, 0, canvas_w, hdr_h], fill=(32, 28, 22))
    draw.text((12, 7), header_txt, fill=(244, 208, 111), font=font_hdr)

    # Top row: Candidate (+ Paired Baseline if present)
    y_top = hdr_h + pad
    if paired_im is not None:
        w1 = int((canvas_w - 3 * pad) * 0.54)
        w2 = (canvas_w - 3 * pad) - w1
        c_fit = fit_tile(cand_im, w1, top_h - 24)
        p_fit = fit_tile(paired_im, w2, top_h - 24)

        draw.rectangle([pad, y_top, pad + w1, y_top + top_h], outline=(244, 208, 111), width=2)
        draw.text((pad + 6, y_top + 4), "CANDIDATE SHOT (EVALUATE THIS)", fill=(244, 208, 111), font=font_lbl)
        canvas.paste(c_fit, (pad + (w1 - c_fit.width) // 2, y_top + 22 + (top_h - 24 - c_fit.height) // 2))

        x2 = pad * 2 + w1
        draw.rectangle([x2, y_top, x2 + w2, y_top + top_h], outline=(120, 160, 210), width=2)
        draw.text((x2 + 6, y_top + 4), f"PAIRED BASELINE ({paired_id})", fill=(158, 203, 255), font=font_lbl)
        canvas.paste(p_fit, (x2 + (w2 - p_fit.width) // 2, y_top + 22 + (top_h - 24 - p_fit.height) // 2))
    else:
        w1 = canvas_w - 2 * pad
        c_fit = fit_tile(cand_im, w1, top_h - 24)
        draw.rectangle([pad, y_top, pad + w1, y_top + top_h], outline=(244, 208, 111), width=2)
        draw.text((pad + 6, y_top + 4), "CANDIDATE SHOT (EVALUATE THIS)", fill=(244, 208, 111), font=font_lbl)
        canvas.paste(c_fit, (pad + (w1 - c_fit.width) // 2, y_top + 22 + (top_h - 24 - c_fit.height) // 2))

    # Bottom row: Canonical Refs
    if refs:
        y_ref = y_top + top_h + pad
        n = len(refs)
        rw = (canvas_w - (n + 1) * pad) // n
        for i, (rlbl, rim) in enumerate(refs):
            rx = pad + i * (rw + pad)
            r_fit = fit_tile(rim, rw - 6, ref_h - 24)
            draw.rectangle([rx, y_ref, rx + rw, y_ref + ref_h], outline=(90, 82, 70), width=1)
            draw.text((rx + 6, y_ref + 4), rlbl, fill=(210, 195, 165), font=font_lbl)
            canvas.paste(r_fit, (rx + (rw - r_fit.width) // 2, y_ref + 22 + (ref_h - 24 - r_fit.height) // 2))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="JPEG", quality=84, optimize=True)
    return out_path


def select_balanced_unlabeled(shots: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Select unlabeled, un-suggested shots evenly interleaved across all categories."""
    unlabeled = [
        s
        for s in shots
        if (s.get("triage") or {}).get("human_label") is None
        and s.get("ai_suggestion") is None
    ]
    buckets: dict[str, list[dict[str, Any]]] = {}
    for s in unlabeled:
        cat = str((s.get("scene") or {}).get("category") or "other")
        buckets.setdefault(cat, []).append(s)

    keys = sorted(buckets.keys())
    selected: list[dict[str, Any]] = []
    while len(selected) < count and any(buckets.values()):
        for k in keys:
            if buckets[k] and len(selected) < count:
                selected.append(buckets[k].pop(0))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build side-by-side preview JPEGs for AI pre-rating.")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--batches", type=int, default=5)
    args = parser.parse_args()

    shots_dir = DATA_DIR / "shots"
    shots = [json.loads(jf.read_text(encoding="utf-8")) for jf in sorted(shots_dir.glob("*.json"))]
    shots_by_id = {s["id"]: s for s in shots}

    selected = select_balanced_unlabeled(shots, args.count)
    previews_dir = DATA_DIR / "_previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    items_meta: list[dict[str, Any]] = []
    for s in selected:
        p = stitch_shot_preview(s, shots_by_id, previews_dir / f"{s['id']}.jpg")
        sc = s.get("scene") or {}
        items_meta.append(
            {
                "id": s["id"],
                "preview_path": str(p),
                "category": sc.get("category"),
                "style": sc.get("style"),
                "expected_continuity": sc.get("expected_continuity"),
                "cast_ids": sc.get("cast_ids", []),
                "prop_ids": sc.get("prop_ids", []),
                "set_id": sc.get("set_id"),
                "paired_with": sc.get("paired_with"),
            }
        )

    # Split into N batches
    batches_dir = previews_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    chunk = (len(items_meta) + args.batches - 1) // args.batches
    for b in range(args.batches):
        sub = items_meta[b * chunk : (b + 1) * chunk]
        if not sub:
            continue
        bp = batches_dir / f"batch_{b + 1:02d}.json"
        bp.write_text(json.dumps({"batch": b + 1, "items": sub}, indent=2), encoding="utf-8")
        print(f"Wrote {bp.name}: {len(sub)} shots")

    cats: dict[str, int] = {}
    for it in items_meta:
        c = str(it["category"])
        cats[c] = cats.get(c, 0) + 1
    print(f"Total previews stitched: {len(items_meta)} | By category: {cats}")


if __name__ == "__main__":
    main()
