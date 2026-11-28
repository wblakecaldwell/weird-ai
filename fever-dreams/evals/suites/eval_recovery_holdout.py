"""Eval Suite 6 (Holdout): Closed-Loop Bad-to-Good Recovery on 10 Diverse Unseen Holdout Failures.

Selects 10 genuine failed shots from the locked Holdout split across both Golden Book mixed-media
and Stop-Motion miniature styles covering:
  - 2x duplicate / anatomy failures
  - 2x subtle character drift failures (with canonical CAST REF)
  - 2x style/medium drift failures (with canonical CAST REF)
  - 2x prop drift failures (with canonical PROP REF)
  - 2x crowd / closeup anatomy failures

Runs `scripts/critic_loop.py` end-to-end (`gemma-4-31b-it` + `qwen-image-2.1-turbo`), then stitches
compact Before/After comparison JPEGs so an independent cloud model (Neon Medium) can verify true
recovery rate rather than relying solely on Gemma self-grading.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

EVALS_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = EVALS_DIR.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
for p in (SCRIPTS_DIR, EVALS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from critic_loop import RepairItem, run_closed_loop  # noqa: E402
from run_mined_evals import DATA_DIR, MANIFEST_PATH, resolve_img  # noqa: E402

RESULTS_DIR = EVALS_DIR / "results" / "suite6_holdout_recovery"


def select_holdout_recovery_items() -> list[RepairItem]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    holdout_fails = [s for s in manifest["shots"] if s["split"] == "holdout" and s["gold_label"] == "FAIL"]

    chosen: list[dict[str, Any]] = []
    used_cycles: set[str] = set()

    def pick(predicate, target_n: int) -> None:
        picked = 0
        # Prefer alternating styles (golden_book_mixed_media & stop_motion_miniature)
        for want_style in ("golden_book_mixed_media", "stop_motion_miniature", None):
            for s in holdout_fails:
                if picked >= target_n:
                    return
                if s["cycle"] in used_cycles:
                    continue
                if want_style and (s.get("scene") or {}).get("style") != want_style:
                    continue
                if predicate(s):
                    chosen.append(s)
                    used_cycles.add(s["cycle"])
                    picked += 1
                    break

    # 2x character drift
    pick(lambda s: (s.get("scene") or {}).get("category") == "subtle_char_drift", 2)
    # 2x style/medium drift
    pick(lambda s: (s.get("scene") or {}).get("category") == "style_medium_drift", 2)
    # 2x prop drift
    pick(lambda s: (s.get("scene") or {}).get("category") == "subtle_prop_drift", 2)
    # 2x duplicate / anatomy from clean/adjacent
    pick(lambda s: (s.get("scene") or {}).get("category") in {"set_reuse_clean", "adjacent_clean"} and "anatomy" in (s.get("defect_types") or []), 2)
    # 2x crowd / closeup anatomy/face failures
    pick(lambda s: (s.get("scene") or {}).get("category") in {"stress_crowd_hands", "stress_closeup_face"} and bool(set(s.get("defect_types") or []) & {"anatomy", "face_blob", "accessory"}), 2)

    items: list[RepairItem] = []
    for s in chosen[:10]:
        sc = s.get("scene") or {}
        cat = sc.get("category", "")
        crefs = sc.get("canonical_refs") or {}
        refs: list[Path] = []
        rlabels: list[str] = []
        cast_ids = sc.get("cast_ids") or []
        prop_ids = sc.get("prop_ids") or []
        if cat == "subtle_prop_drift" and crefs.get("prop"):
            refs.append(resolve_img(crefs["prop"]))
            rlabels.append(f"prop ({prop_ids[0] if prop_ids else 'target'})")
        for ci, cp in enumerate(crefs.get("cast") or []):
            refs.append(resolve_img(cp))
            cid = cast_ids[ci] if ci < len(cast_ids) else f"cast_{ci + 1}"
            rlabels.append(f"character ({cid})")
        if cat != "subtle_prop_drift" and crefs.get("prop"):
            refs.append(resolve_img(crefs["prop"]))
            rlabels.append(f"prop ({prop_ids[0] if prop_ids else 'target'})")
        ename = rlabels[0] if rlabels else ""

        items.append(
            RepairItem(
                item_id=f"rec_{s['id']}",
                image_path=resolve_img(s["image"]),
                description=f"[{cat} | {sc.get('style')}] Gold failure: {s.get('reason', '')}",
                entity_name=ename,
                ref_images=refs,
                ref_labels=rlabels,
                scene_prompt=(s.get("generation") or {}).get("prompt") or "",
                expected_cast_count=len(cast_ids),
                extra_meta={
                    "gold_defect_types": s.get("defect_types") or [],
                    "gold_reason": s.get("reason", ""),
                    "style": sc.get("style"),
                    "category": cat,
                },
            )
        )
    return items


def stitch_before_after_cards(summary: dict[str, Any], work_dir: Path) -> list[Path]:
    """Create compact Ref | BEFORE | AFTER comparison JPEGs for independent Neon Medium verification."""
    out_dir = work_dir / "verify_cards"
    out_dir.mkdir(parents=True, exist_ok=True)
    cards: list[Path] = []

    for it in summary["items"]:
        imgs: list[tuple[str, Path]] = []
        if it.get("ref_images"):
            rp = Path(it["ref_images"][0])
            if not rp.is_absolute():
                rp = ROOT_DIR / rp
            if rp.exists():
                imgs.append(("CANONICAL REF", rp))
        imgs.append(("BEFORE (ORIGINAL FAIL)", work_dir / it["original_image"]))
        imgs.append((f"AFTER (CRITIC: {'PASS' if it['final_passed'] else 'FAIL'})", work_dir / it["final_image"]))

        tw, th = 420, 360
        pad = 12
        hdr_h = 34
        canvas = Image.new("RGB", (len(imgs) * tw + (len(imgs) + 1) * pad, th + hdr_h + pad * 2), (20, 18, 15))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, canvas.width, hdr_h], fill=(32, 28, 22))
        draw.text((12, 10), f"{it['item_id']} | {it['description'][:110]}", fill=(244, 208, 111))

        for i, (lbl, p) in enumerate(imgs):
            im = Image.open(p).convert("RGB")
            scale = min(tw / im.width, (th - 22) / im.height)
            nw, nh = max(1, int(im.width * scale)), max(1, int(im.height * scale))
            res = im.resize((nw, nh), Image.Resampling.LANCZOS)
            x0 = pad + i * (tw + pad)
            y0 = hdr_h + pad
            draw.rectangle([x0, y0, x0 + tw, y0 + th], outline=(100, 90, 75), width=2)
            draw.text((x0 + 6, y0 + 4), lbl, fill=(230, 220, 200))
            canvas.paste(res, (x0 + (tw - nw) // 2, y0 + 22 + ((th - 22) - nh) // 2))

        cp = out_dir / f"{it['item_id']}_verify.jpg"
        canvas.save(cp, format="JPEG", quality=86)
        cards.append(cp)
    return cards


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Suite 6 Holdout Closed-Loop Recovery.")
    parser.add_argument("--work-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--only", default="", help="Optional substring filter to run a single item")
    args = parser.parse_args()

    items = select_holdout_recovery_items()
    if args.only:
        items = [it for it in items if args.only in it.item_id]
    print(f"[suite6_holdout] Selected {len(items)} diverse unseen Holdout failures -> {args.work_dir}:", flush=True)
    for it in items:
        print(f"  - {it.item_id:34s} (cast={it.expected_cast_count}) | {it.description[:80]}", flush=True)

    summary = run_closed_loop(
        items,
        work_dir=args.work_dir,
        critic_model="gemma-4-31b-it",
        edit_model_key="qwen-image-2.1-turbo",
        max_rounds=2,
        candidates_per_round=2,
        base_seed=88001,
    )
    cards = stitch_before_after_cards(summary, args.work_dir)
    print(f"\n[suite6_holdout] Done! Caught: {summary['initial_detected_bad']}/{summary['total_items']} | Self-graded recovery: {summary['recovered_to_pass']}/{summary['initial_detected_bad']} ({summary['recovery_rate']*100:.1f}%)")
    print(f"[suite6_holdout] Stitched {len(cards)} independent verification cards in {args.work_dir / 'verify_cards'}")


if __name__ == "__main__":
    main()
