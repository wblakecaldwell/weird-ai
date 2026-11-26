"""Run Critic Eval Suites 1-4 across a local LM Studio VLM ladder and generate matrix.json + report.html.

Usage:
    uv run python evals/run_evals.py
    uv run python evals/run_evals.py --models qwen3-vl-30b-a3b-instruct-mlx qwen3-vl-8b-instruct-4bit
    uv run python evals/run_evals.py --suites defects contact_sheets
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
ROOT_DIR = EVALS_DIR.parent
RESULTS_DIR = EVALS_DIR / "results"
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))
if str(EVALS_DIR / "suites") not in sys.path:
    sys.path.insert(0, str(EVALS_DIR / "suites"))

import eval_adjacent  # noqa: E402
import eval_anchors  # noqa: E402
import eval_contacts  # noqa: E402
import eval_defects  # noqa: E402
from vlm_client import list_lms_models, load_lms_model, model_size_gb, unload_all_lms  # noqa: E402

SUITES = {
    "defects": eval_defects.run_suite,
    "anchors": eval_anchors.run_suite,
    "contact_sheets": eval_contacts.run_suite,
    "adjacent_cuts": eval_adjacent.run_suite,
}


def discover_vision_models() -> list[str]:
    """Return all locally installed LM Studio vision models, sorted largest to smallest."""
    models = [m for m in list_lms_models() if m.get("vision") is True and m.get("type") == "llm"]
    models.sort(key=lambda m: float(m.get("sizeBytes") or 0), reverse=True)
    return [str(m["modelKey"]) for m in models]


def pct(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val * 100:.0f}%"


def recommend_routing(matrix: dict[str, Any]) -> list[dict[str, str]]:
    """Walk down the model ladder (smallest to largest) to pick the smallest passing model per suite."""
    models_sorted = sorted(matrix["models"], key=lambda m: m["size_gb"])
    suite_defs = [
        ("defects", "Single-Frame Defects", "defect_recall", "fpr", 0.90, 0.15),
        ("anchors", "Hub-and-Spoke Anchors", "mismatch_recall", "fpr", 0.90, 0.10),
        ("contact_sheets", "Contact-Sheet Outliers", "tile_recall", "tile_precision", 0.90, 0.85),
        ("adjacent_cuts", "Adjacent-Cut Continuity", "break_recall", "fpr", 0.85, 0.15),
    ]
    recs: list[dict[str, str]] = []
    for skey, sname, primary_k, sec_k, p_bar, s_bar in suite_defs:
        passing = []
        high_recall_fast = []
        best_overall = None
        best_score = -1.0
        for m in models_sorted:
            sdata = m["suites"].get(skey)
            if not sdata:
                continue
            met = sdata["metrics"]
            p_val = float(met.get(primary_k, 0.0))
            s_val = float(met.get(sec_k, 0.0))
            # Composite score for fallback
            comp = p_val + (s_val if sec_k == "tile_precision" else (1.0 - s_val))
            if comp > best_score:
                best_score = comp
                best_overall = (m, met)
            if met.get("meets_bar"):
                passing.append((m, met))
            elif p_val >= p_bar and m["size_gb"] <= 8.0:
                high_recall_fast.append((m, met))

        if passing:
            winner, met = passing[0]  # smallest passing model!
            recs.append(
                {
                    "suite": sname,
                    "recommended_model": winner["model"],
                    "strategy": "Single-tier (smallest passing model)",
                    "detail": f"{winner['size_gb']} GB, {met['avg_s']}s/item — {primary_k}={pct(met[primary_k])}, {sec_k}={pct(met[sec_k])}",
                }
            )
        elif high_recall_fast and best_overall:
            fast_m, fast_met = high_recall_fast[0]
            heavy_m, heavy_met = best_overall
            recs.append(
                {
                    "suite": sname,
                    "recommended_model": f"{fast_m['model']} -> {heavy_m['model']}",
                    "strategy": "Two-tier cascade (fast screen + heavy confirm)",
                    "detail": f"Fast {fast_m['size_gb']} GB hits {primary_k}={pct(fast_met[primary_k])}; confirm flagged items on {heavy_m['model']}.",
                }
            )
        elif best_overall:
            bm, bmet = best_overall
            recs.append(
                {
                    "suite": sname,
                    "recommended_model": bm["model"],
                    "strategy": "Best available (below strict bar — tune prompt/crops)",
                    "detail": f"{bm['size_gb']} GB, {bmet['avg_s']}s/item — {primary_k}={pct(bmet[primary_k])}, {sec_k}={pct(bmet[sec_k])}",
                }
            )
    return recs


def rel_from_results(rel_from_root: str) -> str:
    return os.path.relpath(ROOT_DIR / rel_from_root, RESULTS_DIR)


def write_html_report(matrix: dict[str, Any], out_path: Path) -> None:
    recs = recommend_routing(matrix)
    models = matrix["models"]

    # Summary Matrix Rows
    matrix_rows = []
    for m in models:
        s1 = m["suites"].get("defects", {}).get("metrics", {})
        s2 = m["suites"].get("anchors", {}).get("metrics", {})
        s3 = m["suites"].get("contact_sheets", {}).get("metrics", {})
        s4 = m["suites"].get("adjacent_cuts", {}).get("metrics", {})

        def badge(met: dict, text: str) -> str:
            if not met:
                return '<span class="pill muted">—</span>'
            cls = "ok" if met.get("meets_bar") else "warn"
            return f'<span class="pill {cls}">{html.escape(text)}</span>'

        s1_t = f"Rec {pct(s1.get('defect_recall'))} | FPR {pct(s1.get('fpr'))} | Loc {pct(s1.get('localization_rate'))} | {s1.get('avg_s','?')}s" if s1 else "—"
        s2_t = f"Rec {pct(s2.get('mismatch_recall'))} | FPR {pct(s2.get('fpr'))} | {s2.get('avg_s','?')}s" if s2 else "—"
        s3_t = f"Rec {pct(s3.get('tile_recall'))} | Prec {pct(s3.get('tile_precision'))} | Exact {pct(s3.get('exact_sheet_rate'))} | {s3.get('avg_s','?')}s" if s3 else "—"
        s4_t = f"Rec {pct(s4.get('break_recall'))} | FPR {pct(s4.get('fpr'))} | {s4.get('avg_s','?')}s" if s4 else "—"

        matrix_rows.append(
            f"""<tr>
              <td><strong>{html.escape(m['model'])}</strong></td>
              <td>{m['size_gb']} GB</td>
              <td>{m['load_s']}s</td>
              <td>{badge(s1, s1_t)}</td>
              <td>{badge(s2, s2_t)}</td>
              <td>{badge(s3, s3_t)}</td>
              <td>{badge(s4, s4_t)}</td>
            </tr>"""
        )

    rec_rows = []
    for r in recs:
        rec_rows.append(
            f"""<tr>
              <td><strong>{html.escape(r['suite'])}</strong></td>
              <td><code>{html.escape(r['recommended_model'])}</code></td>
              <td>{html.escape(r['strategy'])}</td>
              <td>{html.escape(r['detail'])}</td>
            </tr>"""
        )

    # Detailed per-suite visual grids
    detail_sections = []
    for skey, stitle in [
        ("defects", "Suite 1 — Single-Frame Defects (Real Half-Glasses & Book 1 Failures)"),
        ("anchors", "Suite 2 — Hub-and-Spoke Canonical Anchors"),
        ("contact_sheets", "Suite 3 — Contact-Sheet Outlier Detection"),
        ("adjacent_cuts", "Suite 4 — Adjacent-Cut & Set Continuity"),
    ]:
        # Gather case IDs from first model that ran this suite
        base_cases = None
        for m in models:
            if skey in m["suites"]:
                base_cases = m["suites"][skey]["cases"]
                break
        if not base_cases:
            continue

        cards = []
        for idx, bcase in enumerate(base_cases):
            imgs_html = ""
            if skey in {"defects", "contact_sheets"}:
                src = rel_from_results(bcase["image"])
                imgs_html = f'<img class="thumb" src="{html.escape(src)}" alt="{html.escape(bcase["id"])}">'
            elif skey == "anchors":
                s_ref = rel_from_results(bcase["ref_image"])
                s_tst = rel_from_results(bcase["test_image"])
                imgs_html = (
                    f'<div class="pair">'
                    f'<div><div class="sublbl">Canonical Ref</div><img class="thumb" src="{html.escape(s_ref)}"></div>'
                    f'<div><div class="sublbl">Test Frame</div><img class="thumb" src="{html.escape(s_tst)}"></div>'
                    f"</div>"
                )
            elif skey == "adjacent_cuts":
                s_p = rel_from_results(bcase["shot_prev"])
                s_n = rel_from_results(bcase["shot_next"])
                imgs_html = (
                    f'<div class="pair">'
                    f'<div><div class="sublbl">Shot Prev</div><img class="thumb" src="{html.escape(s_p)}"></div>'
                    f'<div><div class="sublbl">Shot Next</div><img class="thumb" src="{html.escape(s_n)}"></div>'
                    f"</div>"
                )

            verdicts = []
            for m in models:
                if skey not in m["suites"]:
                    continue
                mc = m["suites"][skey]["cases"][idx]
                if skey == "contact_sheets":
                    ok = mc["exact_match"]
                    vtxt = f"pred={mc['predicted_outliers']}"
                    note = "; ".join(f"T{r.get('tile')}: {r.get('reason','')}" for r in (mc["output"].get("reasons") or []))[:140]
                else:
                    ok = mc["correct"]
                    vtxt = f"{mc['predicted']}"
                    note = str(
                        mc["output"].get("defect_summary")
                        or mc["output"].get("mismatch_summary")
                        or mc["output"].get("break_summary")
                        or ""
                    )[:140]
                cls = "ok" if ok else "bad"
                verdicts.append(
                    f'<div class="verdict {cls}"><code>{html.escape(m["model"])}</code> '
                    f'<strong>{html.escape(vtxt)}</strong> ({mc["elapsed_s"]}s) — {html.escape(note)}</div>'
                )

            exp_txt = (
                f"Expected outliers: {bcase['expected_outliers']}"
                if skey == "contact_sheets"
                else f"Expected: {bcase['expected']}"
            )
            cards.append(
                f"""<div class="card">
                  <div class="card-head">
                    <strong>{html.escape(bcase['id'])}</strong>
                    <span class="pill muted">{html.escape(exp_txt)}</span>
                  </div>
                  <p class="desc">{html.escape(bcase.get('description',''))}</p>
                  {imgs_html}
                  <div class="verdicts">{"".join(verdicts)}</div>
                </div>"""
            )
        detail_sections.append(f'<section><h2>{html.escape(stitle)}</h2><div class="grid">{"".join(cards)}</div></section>')

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Fever Dreams — VLM Critic Eval Matrix</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #141210; color: #f2eadc; margin: 0; padding: 24px 28px 60px; }}
  h1 {{ margin: 0 0 6px; font-family: Georgia, serif; color: #f4d378; }}
  h2 {{ margin: 32px 0 14px; color: #f4d378; border-bottom: 1px solid #352d25; padding-bottom: 6px; }}
  .meta {{ color: #b7aa95; font-size: 13px; margin-bottom: 20px; }}
  table {{ width: 100%; border-collapse: collapse; background: #1d1915; border-radius: 10px; overflow: hidden; margin-bottom: 20px; font-size: 13px; }}
  th, td {{ padding: 10px 12px; border-bottom: 1px solid #2f2820; text-align: left; vertical-align: middle; }}
  th {{ background: #26201a; color: #e8cfa0; font-weight: 600; }}
  .pill {{ display: inline-block; padding: 3px 9px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
  .pill.ok {{ background: #1e3b28; color: #8ff0b0; border: 1px solid #2e6942; }}
  .pill.warn {{ background: #3f281a; color: #f6b27c; border: 1px solid #7a4522; }}
  .pill.muted {{ background: #2b251f; color: #cfc3ae; border: 1px solid #443a30; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }}
  .card {{ background: #1d1915; border: 1px solid #342c24; border-radius: 10px; padding: 14px; display: flex; flex-direction: column; gap: 10px; }}
  .card-head {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; }}
  .desc {{ margin: 0; font-size: 12.5px; color: #c7bba6; }}
  .thumb {{ width: 100%; max-height: 290px; object-fit: contain; background: #110f0d; border-radius: 6px; }}
  .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  .sublbl {{ font-size: 11px; color: #a99b84; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.04em; }}
  .verdicts {{ display: flex; flex-direction: column; gap: 5px; font-size: 12px; }}
  .verdict {{ padding: 6px 8px; border-radius: 6px; line-height: 1.35; }}
  .verdict.ok {{ background: #182b1f; border-left: 3px solid #59c983; }}
  .verdict.bad {{ background: #321a1a; border-left: 3px solid #e86b6b; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; color: #f4d378; }}
</style>
</head>
<body>
  <h1>Fever Dreams Phase 1 — VLM Critic Downscaling Matrix</h1>
  <div class="meta">Generated {html.escape(matrix['created_at'])} • 100% local LM Studio evals • Suites: {html.escape(', '.join(matrix['suites_run']))}</div>

  <h2>Model Ladder Benchmark</h2>
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Size</th>
        <th>Load</th>
        <th>Suite 1: Defects (bar: Rec &gt;=90%, FPR &lt;=15%)</th>
        <th>Suite 2: Anchors (bar: Rec &gt;=90%, FPR &lt;=10%)</th>
        <th>Suite 3: Contact Sheets (bar: Rec &gt;=90%, Prec &gt;=85%)</th>
        <th>Suite 4: Adjacent Cuts (bar: Rec &gt;=85%, FPR &lt;=15%)</th>
      </tr>
    </thead>
    <tbody>
      {"".join(matrix_rows)}
    </tbody>
  </table>

  <h2>Recommended Model Routing (Smallest Passing Model per Task)</h2>
  <table>
    <thead>
      <tr><th>Subsystem Task</th><th>Recommended Model(s)</th><th>Strategy</th><th>Measured Performance</th></tr>
    </thead>
    <tbody>
      {"".join(rec_rows)}
    </tbody>
  </table>

  {"".join(detail_sections)}
</body>
</html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local LM Studio VLM Critic Eval Suites 1-4.")
    parser.add_argument("--models", nargs="+", default=None, help="Explicit list of LM Studio model keys.")
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["defects", "anchors", "contact_sheets", "adjacent_cuts"],
        choices=list(SUITES.keys()),
    )
    parser.add_argument("--context-length", type=int, default=16384)
    args = parser.parse_args()

    models = args.models or discover_vision_models()
    if not models:
        raise SystemExit("No vision models found in LM Studio (`lms ls --json`).")

    print(f"Running suites {args.suites} across {len(models)} local vision models:")
    for m in models:
        print(f"  - {m} ({model_size_gb(m)} GB)")

    matrix: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suites_run": args.suites,
        "models": [],
    }

    for model in models:
        print(f"\n==================== MODEL: {model} ====================")
        load_s = load_lms_model(model, context_length=args.context_length)
        print(f"  Loaded in {load_s:.2f}s (disk size {model_size_gb(model)} GB)")
        m_entry: dict[str, Any] = {
            "model": model,
            "size_gb": model_size_gb(model),
            "load_s": load_s,
            "suites": {},
        }
        for sname in args.suites:
            print(f"\n--- Running suite: {sname} ---")
            m_entry["suites"][sname] = SUITES[sname](model)
        matrix["models"].append(m_entry)

        # Save incremental progress after every model so nothing is lost
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "latest_matrix.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
        write_html_report(matrix, RESULTS_DIR / "report.html")

    unload_all_lms()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (RESULTS_DIR / f"matrix_{ts}.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    print(f"\nSaved matrix to {RESULTS_DIR / 'latest_matrix.json'}")
    print(f"Saved visual report to {RESULTS_DIR / 'report.html'}")


if __name__ == "__main__":
    main()
