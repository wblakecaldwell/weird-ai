"""Merge cycle-isolated Neon Medium ratings into sidecar JSONs and lock a 50/50 Dev/Holdout manifest.

Rules:
- Human labels (`triage.human_label`) ALWAYS take precedence over AI ratings.
- Neon Medium ratings are written to `ai_suggestion` (`model: "neon-medium"`) so the triage UI
  shows them immediately for optional spot-checking without overwriting human labels.
- Dev / Holdout split is strictly 50/50 grouped by cycle (28 cycles Dev / 28 cycles Holdout),
  stratified evenly across `golden_book_mixed_media` and `stop_motion_miniature` styles so zero
  characters, props, or scenes from a cycle ever leak across Dev and Holdout.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FEVER_DREAMS_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = FEVER_DREAMS_ROOT / "evals" / "mined_data"
VALID_TAGS = {
    "anatomy",
    "accessory",
    "face_blob",
    "hand_merge",
    "char_drift",
    "prop_drift",
    "style_drift",
    "lighting_break",
    "set_warp",
}


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    shots_dir = DATA_DIR / "shots"
    batches_dir = DATA_DIR / "_previews" / "neon_batches"

    # 1. Merge all 18 Neon Medium rating files into sidecars
    neon_merged = 0
    for bf in sorted(batches_dir.glob("neon_batch_*_ratings.json")):
        data = json.loads(bf.read_text(encoding="utf-8"))
        for r in data.get("ratings", []):
            sid = r["id"]
            jf = shots_dir / f"{sid}.json"
            if not jf.exists():
                raise FileNotFoundError(f"Missing shot sidecar for {sid}")
            rec = json.loads(jf.read_text(encoding="utf-8"))
            lbl = str(r.get("suggested_label", "PASS")).upper()
            if lbl not in {"PASS", "FAIL"}:
                lbl = "FAIL"
            defs = [d for d in (r.get("suggested_defects") or []) if d in VALID_TAGS] if lbl == "FAIL" else []
            sev = r.get("suggested_severity") if lbl == "FAIL" else None
            if lbl == "FAIL" and sev not in {"subtle", "obvious"}:
                sev = "obvious"
            rec["ai_suggestion"] = {
                "model": "neon-medium",
                "suggested_label": lbl,
                "suggested_defects": defs,
                "suggested_severity": sev,
                "reason": str(r.get("reason") or "").strip(),
                "rated_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write(jf, rec)
            neon_merged += 1

    # 2. Load all 449 shots and assign stratified 50/50 cycle-grouped Dev/Holdout split
    all_shots: list[dict[str, Any]] = []
    cycles_by_style: dict[str, set[str]] = {}
    for jf in sorted(shots_dir.glob("*.json")):
        rec = json.loads(jf.read_text(encoding="utf-8"))
        cyc = rec["id"].split("_")[0]
        st = str((rec.get("scene") or {}).get("style") or "unknown")
        cycles_by_style.setdefault(st, set()).add(cyc)
        all_shots.append(rec)

    rng = random.Random(20261127)
    holdout_cycles: set[str] = set()
    dev_cycles: set[str] = set()
    for st in sorted(cycles_by_style.keys()):
        cycs = sorted(cycles_by_style[st])
        rng.shuffle(cycs)
        half = len(cycs) // 2
        holdout_cycles.update(cycs[:half])
        dev_cycles.update(cycs[half:])

    assert dev_cycles.isdisjoint(holdout_cycles), "Dev/Holdout cycle overlap!"

    manifest_shots: list[dict[str, Any]] = []
    shots_by_id: dict[str, dict[str, Any]] = {}
    counts = {
        "dev": {"PASS": 0, "FAIL": 0, "human": 0, "neon-medium": 0},
        "holdout": {"PASS": 0, "FAIL": 0, "human": 0, "neon-medium": 0},
    }
    by_cat: dict[str, dict[str, int]] = {}
    tag_counts: dict[str, int] = {}

    for rec in all_shots:
        cyc = rec["id"].split("_")[0]
        split = "holdout" if cyc in holdout_cycles else "dev"
        tr = rec.get("triage") or {}
        ai = rec.get("ai_suggestion") or {}

        if tr.get("human_label") in {"PASS", "FAIL"}:
            gold_label = tr["human_label"]
            gold_defects = tr.get("defect_types") or []
            gold_sev = tr.get("severity")
            gold_reason = tr.get("notes") or ""
            source = "human"
        else:
            assert ai.get("model") == "neon-medium", f"Shot {rec['id']} missing Neon Medium rating!"
            gold_label = ai["suggested_label"]
            gold_defects = ai.get("suggested_defects") or []
            gold_sev = ai.get("suggested_severity")
            gold_reason = ai.get("reason") or ""
            source = "neon-medium"

        # Record split in sidecar
        rec.setdefault("triage", {})["split"] = split
        atomic_write(shots_dir / f"{rec['id']}.json", rec)

        counts[split][gold_label] += 1
        counts[split][source] += 1
        cat = str((rec.get("scene") or {}).get("category") or "unknown")
        by_cat.setdefault(cat, {"PASS": 0, "FAIL": 0})[gold_label] += 1
        for d in gold_defects:
            tag_counts[d] = tag_counts.get(d, 0) + 1

        entry = {
            "id": rec["id"],
            "cycle": cyc,
            "image": f"shots/{rec['image']}",
            "split": split,
            "gold_label": gold_label,
            "label_source": source,
            "defect_types": gold_defects,
            "severity": gold_sev,
            "reason": gold_reason,
            "scene": rec.get("scene") or {},
            "generation": {
                "model": (rec.get("generation") or {}).get("model"),
                "seed": (rec.get("generation") or {}).get("seed"),
                "width": (rec.get("generation") or {}).get("width"),
                "height": (rec.get("generation") or {}).get("height"),
                "prompt": (rec.get("generation") or {}).get("prompt"),
            },
        }
        manifest_shots.append(entry)
        shots_by_id[rec["id"]] = entry

    # 3. Build verified Contact Sheets manifest (derive exact outlier tiles from constituent tile gold labels!)
    sheets_dir = DATA_DIR / "contact_sheets"
    manifest_sheets: list[dict[str, Any]] = []
    for jf in sorted(sheets_dir.glob("*.json")):
        sh = json.loads(jf.read_text(encoding="utf-8"))
        parts = sh["id"].split("_")
        cyc = parts[1] if len(parts) > 1 else ""
        split = "holdout" if cyc in holdout_cycles else "dev"
        tiles = sh.get("tiles") or {}
        etype = sh.get("entity_type")  # "character" or "prop"

        # Derive true outlier tiles from each tile shot's gold defects!
        outlier_tiles: list[int] = []
        for tnum in (1, 2, 3):
            tpath = str(tiles.get(str(tnum), ""))
            tid = Path(tpath).stem
            tshot = shots_by_id.get(tid)
            if not tshot:
                continue
            tdefs = set(tshot.get("defect_types") or [])
            if etype == "character":
                if tshot["gold_label"] == "FAIL" and (
                    tdefs & {"char_drift", "style_drift", "anatomy", "accessory", "face_blob"}
                ):
                    outlier_tiles.append(tnum)
            else:
                if tshot["gold_label"] == "FAIL" and (tdefs & {"prop_drift", "style_drift"}):
                    outlier_tiles.append(tnum)

        sh.setdefault("triage", {})["split"] = split
        atomic_write(jf, sh)
        manifest_sheets.append(
            {
                "id": sh["id"],
                "cycle": cyc,
                "image": f"contact_sheets/{sh['image']}",
                "split": split,
                "entity_type": etype,
                "entity_id": sh.get("entity_id"),
                "entity_name": sh.get("entity_name"),
                "world_style": sh.get("world_style"),
                "tile_count": sh.get("tile_count", 3),
                "expected_outliers": outlier_tiles,
                "tiles": tiles,
            }
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_shots": len(manifest_shots),
        "total_contact_sheets": len(manifest_sheets),
        "dev_cycles": sorted(dev_cycles),
        "holdout_cycles": sorted(holdout_cycles),
        "counts": counts,
        "by_category": by_cat,
        "defect_tags": dict(sorted(tag_counts.items(), key=lambda x: -x[1])),
        "shots": manifest_shots,
        "contact_sheets": manifest_sheets,
    }
    out_p = DATA_DIR / "gold_manifest.json"
    atomic_write(out_p, manifest)

    print(f"Neon Medium merged: {neon_merged}")
    print(f"Total gold shots: {len(manifest_shots)} | Dev cycles: {len(dev_cycles)} | Holdout cycles: {len(holdout_cycles)}")
    print("Split counts:", json.dumps(counts, indent=2))
    print("By category:", json.dumps(by_cat, indent=2))
    print("Defect tags:", manifest["defect_tags"])
    print(f"Contact sheets: {len(manifest_sheets)} (Dev: {sum(1 for s in manifest_sheets if s['split']=='dev')}, Holdout: {sum(1 for s in manifest_sheets if s['split']=='holdout')})")
    print(f"Wrote {out_p}")


if __name__ == "__main__":
    main()
