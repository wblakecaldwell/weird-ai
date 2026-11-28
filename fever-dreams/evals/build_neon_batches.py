"""Build strictly cycle-isolated batches for Gemini 4 Neon (Medium) visual QA rating.

Guarantees mathematically (and asserts) that NO batch contains two shots from the
same cycle (c0001..c0056), eliminating any possibility of cross-shot memory bleed
(e.g., remembering a prop or lighting state from another shot in the same cycle).
Preserves any shot already labeled by the human (`triage.human_label != null`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

from build_triage_previews import DATA_DIR, stitch_shot_preview  # noqa: E402


def cycle_of(shot_id: str) -> str:
    return shot_id.split("_")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cycle-isolated batches for Neon Medium visual QA.")
    parser.add_argument("--batches", type=int, default=18)
    args = parser.parse_args()

    shots_dir = DATA_DIR / "shots"
    shots = [json.loads(jf.read_text(encoding="utf-8")) for jf in sorted(shots_dir.glob("*.json"))]
    shots_by_id = {s["id"]: s for s in shots}

    # Skip shots already verified by human
    pending = [s for s in shots if (s.get("triage") or {}).get("human_label") is None]
    human_cnt = len(shots) - len(pending)

    previews_dir = DATA_DIR / "_previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    rebuilt = 0
    for s in pending:
        p = previews_dir / f"{s['id']}.jpg"
        if not p.exists():
            stitch_shot_preview(s, shots_by_id, p)
            rebuilt += 1

    # Group by cycle
    by_cycle: dict[str, list[dict[str, Any]]] = {}
    for s in pending:
        by_cycle.setdefault(cycle_of(s["id"]), []).append(s)

    n_batches = max(args.batches, max((len(v) for v in by_cycle.values()), default=1))
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(n_batches)]
    bucket_cycles: list[set[str]] = [set() for _ in range(n_batches)]

    # Place shots round-robin into least-loaded valid batch that doesn't already contain this cycle
    for cyc in sorted(by_cycle.keys()):
        cyc_shots = by_cycle[cyc]
        for s in cyc_shots:
            # Pick least-full bucket that does not have `cyc`
            valid_indices = [i for i in range(n_batches) if cyc not in bucket_cycles[i]]
            assert valid_indices, f"No valid batch for {s['id']} (cycle {cyc})"
            best_i = min(valid_indices, key=lambda i: len(buckets[i]))
            sc = s.get("scene") or {}
            crefs = sc.get("canonical_refs") or {}
            buckets[best_i].append(
                {
                    "id": s["id"],
                    "cycle": cyc,
                    "preview_path": str(previews_dir / f"{s['id']}.jpg"),
                    "category": sc.get("category"),
                    "style": sc.get("style"),
                    "has_paired_baseline": bool(sc.get("paired_with")),
                    "has_prop_ref": bool(crefs.get("prop")),
                    "has_set_ref": bool(crefs.get("set")),
                    "cast_ref_count": len(crefs.get("cast") or []),
                }
            )
            bucket_cycles[best_i].add(cyc)

    out_dir = previews_dir / "neon_batches"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("neon_batch_*.json"):
        old.unlink()

    total_written = 0
    for idx, items in enumerate(buckets, start=1):
        # Hard mathematical guarantee: zero duplicate cycles in any batch
        cycles_here = [it["cycle"] for it in items]
        assert len(cycles_here) == len(set(cycles_here)), f"Cycle collision in batch {idx}!"
        bp = out_dir / f"neon_batch_{idx:02d}.json"
        bp.write_text(json.dumps({"batch": idx, "items": items}, indent=2), encoding="utf-8")
        total_written += len(items)

    print(
        f"Prepared {n_batches} cycle-isolated Neon batches in {out_dir}\n"
        f"  Total shots: {len(shots)} | Human-preserved: {human_cnt} | Pending Neon QA: {total_written} | Newly stitched: {rebuilt}\n"
        f"  Batch sizes: {[len(b) for b in buckets]} (ZERO same-cycle collisions verified!)"
    )


if __name__ == "__main__":
    main()
