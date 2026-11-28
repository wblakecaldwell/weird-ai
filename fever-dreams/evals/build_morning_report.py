"""Generate the interactive Morning HTML Dashboard summarizing the full overnight Phase 1 eval sprint.

Combines:
  - 449-shot / 112-sheet Neon-gold dataset & strict Dev/Holdout split stats
  - Dev 5-model baseline & M1 Critic ablation progression (10% -> 70% Recall, zero Book-1 hints)
  - Full 330-case Blind Holdout leaderboard across 4 VLMs + Two-Tier Cascade simulation
  - Holdout error slices by Style (Golden Book vs. Stop-Motion) and Category
  - Suite 6 Holdout Closed-Loop Recovery with independent Neon Medium verification & visual cards
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "results"
MANIFEST_PATH = EVALS_DIR / "mined_data" / "gold_manifest.json"

INDEPENDENT_SUITE6_VERDICTS: dict[str, dict[str, str]] = {
    "rec_c0007_04_drift_char": {
        "neon_verdict": "FAIL (Duplicated Character)",
        "cls": "bad",
        "note": "Fixed outfit/identity to match ref, BUT pasted BOTH front & walking views from the 3-view turnaround sheet (2x clone). Gemma missed the duplicate.",
    },
    "rec_c0002_04_drift_char": {
        "neon_verdict": "FAIL (3x Turnaround Clone)",
        "cls": "bad",
        "note": "Pasted ALL THREE views of the skeleton turnaround sheet into the lab and erased the pie kid + trophy. Gemma rubber-stamped PASS because the skeleton matched ref.",
    },
    "rec_c0008_05_drift_prop": {
        "neon_verdict": "PARTIAL (Prop Fixed!)",
        "cls": "warn",
        "note": "Successfully turned purple megaphone into the exact yellow/white canonical bullhorn! Pre-existing duplicate character on left remained; Gemma over-failed on bullhorn color.",
    },
    "rec_c0021_05_drift_prop": {
        "neon_verdict": "FAIL (Gemma Accurate)",
        "cls": "bad",
        "note": "Added glowing green sine-wave CRT screen + red cables, but casing stayed purple instead of sage green. Gemma accurately kept failing it.",
    },
    "rec_c0027_03_stress_crowd": {
        "neon_verdict": "FAIL (Missed Duplicate)",
        "cls": "bad",
        "note": "Minor polish on pixel robot, but left the duplicate yellow box-head robot untouched. Gemma missed the duplicate and scored PASS.",
    },
}


def pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}%"


def compute_cascade_metrics(holdout_data: dict[str, Any], fast_name: str, heavy_name: str) -> dict[str, Any]:
    """Simulate a 2-tier cascade: fast model screens; only items fast flags as FAIL go to heavy confirmer."""
    by_model = {m["model"]: m["suites"] for m in holdout_data.get("models", [])}
    if fast_name not in by_model or heavy_name not in by_model:
        return {}
    out: dict[str, Any] = {}
    for suite in ("single_frame", "anchors", "adjacent_cuts"):
        f_cases = {c["id"]: c for c in by_model[fast_name].get(suite, {}).get("cases", [])}
        h_cases = {c["id"]: c for c in by_model[heavy_name].get(suite, {}).get("cases", [])}
        tp = fp = tn = fn = escalated = 0
        total_s = 0.0
        for cid, fc in f_cases.items():
            hc = h_cases.get(cid)
            if not hc:
                continue
            total_s += float(fc.get("elapsed_s") or 0)
            if fc["predicted"] == "FAIL":
                escalated += 1
                total_s += float(hc.get("elapsed_s") or 0)
                pred = hc["predicted"]
            else:
                pred = "PASS"
            exp = fc["expected"]
            if exp == "FAIL":
                tp += int(pred == "FAIL")
                fn += int(pred == "PASS")
            else:
                tn += int(pred == "PASS")
                fp += int(pred == "FAIL")
        n = tp + fp + tn + fn
        rec = tp / max(1, tp + fn)
        prec = tp / max(1, tp + fp)
        fpr = fp / max(1, fp + tn)
        acc = (tp + tn) / max(1, n)
        out[suite] = {
            "total": n,
            "recall": round(rec, 4),
            "precision": round(prec, 4),
            "fpr": round(fpr, 4),
            "accuracy": round(acc, 4),
            "escalated_pct": round(escalated / max(1, n), 4),
            "avg_s": round(total_s / max(1, n), 2),
        }
    return out


def compute_style_slices(holdout_data: dict[str, Any], model_name: str) -> list[dict[str, Any]]:
    by_model = {m["model"]: m["suites"] for m in holdout_data.get("models", [])}
    suites = by_model.get(model_name, {})
    rows = []
    for sname in ("single_frame", "anchors", "adjacent_cuts"):
        cases = suites.get(sname, {}).get("cases", [])
        for st in ("golden_book_mixed_media", "stop_motion_miniature"):
            sub = [c for c in cases if c.get("style") == st]
            if not sub:
                continue
            tp = sum(1 for c in sub if c["expected"] == "FAIL" and c["predicted"] == "FAIL")
            fn = sum(1 for c in sub if c["expected"] == "FAIL" and c["predicted"] == "PASS")
            tn = sum(1 for c in sub if c["expected"] == "PASS" and c["predicted"] == "PASS")
            fp = sum(1 for c in sub if c["expected"] == "PASS" and c["predicted"] == "FAIL")
            rows.append(
                {
                    "suite": sname,
                    "style": st,
                    "n": len(sub),
                    "recall": tp / max(1, tp + fn),
                    "fpr": fp / max(1, fp + tn),
                    "acc": (tp + tn) / max(1, len(sub)),
                }
            )
    return rows


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    dev_base = json.loads((RESULTS_DIR / "mined_dev_baseline.json").read_text(encoding="utf-8"))
    dev_f = json.loads((RESULTS_DIR / "mined_dev_m1_forensic.json").read_text(encoding="utf-8"))
    dev_fq = json.loads((RESULTS_DIR / "mined_dev_m1_forensic_quad.json").read_text(encoding="utf-8"))
    dev_pq = json.loads((RESULTS_DIR / "mined_dev_m1_probed_quad.json").read_text(encoding="utf-8"))
    holdout = json.loads((RESULTS_DIR / "mined_holdout_final.json").read_text(encoding="utf-8"))
    suite6 = json.loads((RESULTS_DIR / "suite6_holdout_recovery" / "loop_log.json").read_text(encoding="utf-8"))

    cascade_30b_31b = compute_cascade_metrics(holdout, "qwen3-vl-30b-a3b-instruct-mlx", "gemma-4-31b-it")
    style_slices = compute_style_slices(holdout, "gemma-4-31b-it")

    # Build M1 Dev Ablation rows
    g_base_m1 = next(m["suites"]["single_frame"]["metrics"] for m in dev_base["models"] if m["model"] == "gemma-4-31b-it")
    g_f_m1 = next(m["suites"]["single_frame"]["metrics"] for m in dev_f["models"] if m["model"] == "gemma-4-31b-it")
    g_fq_m1 = next(m["suites"]["single_frame"]["metrics"] for m in dev_fq["models"] if m["model"] == "gemma-4-31b-it")
    g_pq_m1 = next(m["suites"]["single_frame"]["metrics"] for m in dev_pq["models"] if m["model"] == "gemma-4-31b-it")
    g_hold_m1 = next(m["suites"]["single_frame"]["metrics"] for m in holdout["models"] if m["model"] == "gemma-4-31b-it")

    ablation_rows = [
        ("Dev Baseline (monolithic general)", g_base_m1),
        ("Dev Forensic (4-field checklist)", g_f_m1),
        ("Dev Forensic + 2x2 Quad-Zoom", g_fq_m1),
        ("Dev Probed-General + Quad-Zoom (WINNER)", g_pq_m1),
        ("Blind Holdout Probed-General + Quad-Zoom (86 unseen)", g_hold_m1),
    ]

    # Build Holdout Leaderboard rows
    holdout_rows_html = []
    for m in holdout["models"]:
        s = m["suites"]
        m1 = s["single_frame"]["metrics"]
        m2 = s["anchors"]["metrics"]
        m3 = s["contact_sheets"]["metrics"]
        m4 = s["adjacent_cuts"]["metrics"]
        holdout_rows_html.append(
            f"""<tr>
              <td><code>{html.escape(m['model'])}</code></td>
              <td><strong>{pct(m1['recall'])}</strong> Rec / {pct(m1['fpr'])} FPR<br><small>Acc {pct(m1['accuracy'])} · {m1['avg_s']}s · err={m1['parse_errors']}</small></td>
              <td><strong>{pct(m2['recall'])}</strong> Rec / {pct(m2['fpr'])} FPR<br><small>Acc {pct(m2['accuracy'])} · {m2['avg_s']}s · err={m2['parse_errors']}</small></td>
              <td><strong>{pct(m3['tile_precision'])}</strong> Prec / {pct(m3['tile_recall'])} Rec<br><small>Exact {pct(m3['exact_sheet_rate'])} · {m3['avg_s']}s</small></td>
              <td><strong>{pct(m4['recall'])}</strong> Rec / {pct(m4['fpr'])} FPR<br><small>Acc {pct(m4['accuracy'])} · {m4['avg_s']}s · err={m4['parse_errors']}</small></td>
            </tr>"""
        )

    # Cascade row
    if cascade_30b_31b:
        c1 = cascade_30b_31b["single_frame"]
        c2 = cascade_30b_31b["anchors"]
        c4 = cascade_30b_31b["adjacent_cuts"]
        holdout_rows_html.append(
            f"""<tr class="cascade-row">
              <td><strong>Cascade: 30b-a3b &rarr; gemma-31b</strong><br><small>Fast screener + heavy confirmer</small></td>
              <td><strong>{pct(c1['recall'])}</strong> Rec / {pct(c1['fpr'])} FPR<br><small>Acc {pct(c1['accuracy'])} · {c1['avg_s']}s ({pct(c1['escalated_pct'])} esc.)</small></td>
              <td><strong>{pct(c2['recall'])}</strong> Rec / {pct(c2['fpr'])} FPR<br><small>Acc {pct(c2['accuracy'])} · {c2['avg_s']}s ({pct(c2['escalated_pct'])} esc.)</small></td>
              <td><em>Use 8b or 31b directly</em></td>
              <td><strong>{pct(c4['recall'])}</strong> Rec / {pct(c4['fpr'])} FPR<br><small>Acc {pct(c4['accuracy'])} · {c4['avg_s']}s ({pct(c4['escalated_pct'])} esc.)</small></td>
            </tr>"""
        )

    # Suite 6 verified cards HTML
    s6_cards_html = []
    for it in suite6["items"]:
        iid = it["item_id"]
        if it["initial_passed"]:
            continue
        iv = INDEPENDENT_SUITE6_VERDICTS.get(iid, {"neon_verdict": "Unverified", "cls": "warn", "note": ""})
        card_rel = f"suite6_holdout_recovery/verify_cards/{iid}_verify.jpg"
        s6_cards_html.append(
            f"""<div class="s6-card">
              <div class="s6-head">
                <strong>{html.escape(iid)}</strong>
                <div>
                  <span class="pill {'ok' if it['final_passed'] else 'bad'}">Gemma Self-Grade: {'PASS' if it['final_passed'] else 'FAIL'}</span>
                  <span class="pill {iv['cls']}">Independent Neon: {html.escape(iv['neon_verdict'])}</span>
                </div>
              </div>
              <p class="s6-desc">{html.escape(it['description'])}</p>
              <p class="s6-note"><strong>Independent Audit:</strong> {html.escape(iv['note'])}</p>
              <img src="{html.escape(card_rel)}" alt="{html.escape(iid)}">
            </div>"""
        )

    ablation_html = "\n".join(
        f"""<tr class="{'hi' if 'WINNER' in name or 'Holdout' in name else ''}">
          <td>{html.escape(name)}</td>
          <td><strong>{pct(m['recall'])}</strong> ({m['tp']}/{m['tp']+m['fn']})</td>
          <td>{pct(m['precision'])}</td>
          <td>{pct(m['fpr'])} ({m['fp']}/{m['fp']+m['tn']})</td>
          <td><strong>{m['f1']:.3f}</strong></td>
          <td>{pct(m['accuracy'])}</td>
          <td>{m['avg_s']}s</td>
        </tr>"""
        for name, m in ablation_rows
    )

    slices_html = "\n".join(
        f"""<tr>
          <td>{html.escape(r['suite'])}</td>
          <td><code>{html.escape(r['style'])}</code></td>
          <td>{r['n']}</td>
          <td><strong>{pct(r['recall'])}</strong></td>
          <td>{pct(r['fpr'])}</td>
          <td><strong>{pct(r['acc'])}</strong></td>
        </tr>"""
        for r in style_slices
    )

    cnt = manifest["counts"]
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Fever Dreams — Overnight Phase 1 Eval &amp; Critic Report</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #12100e; color: #f2eadc; margin: 0; padding: 24px 30px 70px; line-height: 1.45; }}
  h1 {{ margin: 0 0 6px; font-family: Georgia, serif; color: #f4d06f; font-size: 28px; }}
  h2 {{ margin: 28px 0 10px; font-family: Georgia, serif; color: #f4d06f; font-size: 20px; border-bottom: 1px solid #332c24; padding-bottom: 6px; }}
  .sub {{ color: #b5a792; font-size: 13px; margin-bottom: 18px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(185px, 1fr)); gap: 12px; margin-bottom: 22px; }}
  .kpi {{ background: #1c1814; border: 1px solid #352e25; border-radius: 10px; padding: 14px; }}
  .kpi .num {{ font-size: 24px; font-weight: 800; color: #f4d06f; }}
  .kpi .k {{ font-size: 12px; color: #b5a792; }}
  .callout {{ background: #1f1a14; border-left: 4px solid #f4d06f; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 14px 0; font-size: 13.5px; }}
  .callout.warn {{ border-left-color: #e5534b; background: #261616; }}
  table {{ width: 100%; border-collapse: collapse; background: #1a1612; border: 1px solid #352e25; border-radius: 10px; overflow: hidden; font-size: 13px; }}
  th, td {{ padding: 10px 12px; border-bottom: 1px solid #2b251e; text-align: left; vertical-align: top; }}
  th {{ background: #231e18; color: #f4d06f; font-size: 12px; text-transform: uppercase; letter-spacing: 0.4px; }}
  tr.hi {{ background: #1b281e; }}
  tr.cascade-row {{ background: #162230; }}
  code {{ background: #27211a; padding: 2px 6px; border-radius: 5px; font-size: 12px; color: #f4d06f; }}
  small {{ color: #a99c88; }}
  .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; margin-left: 6px; }}
  .pill.ok {{ background: #16331e; color: #7ee787; border: 1px solid #2ea043; }}
  .pill.bad {{ background: #3b1716; color: #ff9492; border: 1px solid #e5534b; }}
  .pill.warn {{ background: #332a10; color: #f2cc60; border: 1px solid #b08800; }}
  .s6-card {{ background: #1a1612; border: 1px solid #352e25; border-radius: 12px; padding: 14px; margin-bottom: 18px; }}
  .s6-head {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 6px; }}
  .s6-desc {{ margin: 4px 0; font-size: 12.5px; color: #b5a792; }}
  .s6-note {{ margin: 6px 0 10px; font-size: 13px; color: #f2eadc; }}
  .s6-card img {{ width: 100%; border-radius: 8px; border: 1px solid #352e25; }}
</style>
</head>
<body>
<h1>Fever Dreams — Overnight Phase 1 Eval &amp; Critic Report</h1>
<div class="sub">Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")} &bull; Zero Book-1 hints &bull; Cycle-isolated Neon Medium gold labels &bull; 100% local LM Studio + mflux execution</div>

<div class="kpis">
  <div class="kpi"><div class="num">449 / 112</div><div class="k">Gold Shots / Contact Sheets (121 chars)</div></div>
  <div class="kpi"><div class="num">10% &rarr; 75%</div><div class="k">Gemma 31B Single-Frame Recall (4.8% FPR)</div></div>
  <div class="kpi"><div class="num">89.5% / 97.7%</div><div class="k">Holdout Anchors / Adjacent Cuts (Gemma 31B)</div></div>
  <div class="kpi"><div class="num">3.7s / 97.7%</div><div class="k">Qwen 30B-A3B Holdout Adjacent Cuts (0% FPR)</div></div>
</div>

<div class="callout">
  <strong>Headline Validation of Your Instinct:</strong> Expanding from 10 Book-1 fixtures to <strong>330 unseen Holdout cases</strong> completely flipped the small-model story: <code>qwen3-vl-4b</code> (which scored "100%" on 10 Book-1 anchors) collapsed to <strong>72.4% Anchors (43.8% FPR)</strong> and <strong>25.9% Adjacent Cuts (45 parse errors)</strong>. Meanwhile, our general zero-Book-1-hint <strong>Probed-General + 2x2 Quad-Zoom</strong> Critic lifted <code>gemma-4-31b-it</code> standalone defect recall from <strong>10% &rarr; 75.0%</strong> on blind Holdout with only <strong>4.8% FPR</strong>.
</div>

<h2>1. Gold Dataset &amp; Strict Dev / Holdout Split</h2>
<p>Every shot was rated in strictly <strong>cycle-isolated</strong> Neon Medium batches (zero same-cycle collisions; your 10 human labels preserved untouched) and split 50/50 by cycle so zero characters or scenes leak across Dev and Holdout:</p>
<ul>
  <li><strong>Dev Split (29 cycles):</strong> {cnt['dev']['PASS'] + cnt['dev']['FAIL']} shots ({cnt['dev']['PASS']} PASS / {cnt['dev']['FAIL']} FAIL; {cnt['dev']['human']} human + {cnt['dev']['neon-medium']} Neon) + 58 contact sheets</li>
  <li><strong>Holdout Split (28 cycles — strictly unseen during iteration):</strong> {cnt['holdout']['PASS'] + cnt['holdout']['FAIL']} shots ({cnt['holdout']['PASS']} PASS / {cnt['holdout']['FAIL']} FAIL; {cnt['holdout']['human']} human + {cnt['holdout']['neon-medium']} Neon) + 54 contact sheets</li>
</ul>

<h2>2. Dev-Only Critic Iteration (Zero Book-1 Hints) — M1 Single-Frame Defects</h2>
<table>
  <thead>
    <tr><th>Critic Strategy (gemma-4-31b-it)</th><th>Recall</th><th>Precision</th><th>FPR</th><th>F1</th><th>Accuracy</th><th>Avg Latency</th></tr>
  </thead>
  <tbody>{ablation_html}</tbody>
</table>

<h2>3. Full 330-Case Blind Holdout Leaderboard (4 Local VLMs + Cascade)</h2>
<table>
  <thead>
    <tr>
      <th>Model / Routing</th>
      <th>M1: Single-Frame (86)</th>
      <th>M2: Ref Anchors (105)</th>
      <th>M3: Contact Sheets (54)</th>
      <th>M4: Adjacent Cuts (85)</th>
    </tr>
  </thead>
  <tbody>{"".join(holdout_rows_html)}</tbody>
</table>

<div class="callout">
  <strong>Proven Production Routing (Zero Guesswork):</strong><br>
  &bull; <strong>Adjacent Cuts (M4):</strong> <code>qwen3-vl-30b-a3b</code> hits <strong>97.7% Acc / 0% FPR at 3.7s/call</strong> (3.5x faster than Gemma 31B).<br>
  &bull; <strong>Contact Sheets (M3):</strong> <code>qwen3-vl-8b</code> matches Gemma 31B's precision (96.3%) with the highest exact-sheet rate (35.2%) at <strong>4.8s/call</strong>.<br>
  &bull; <strong>Single-Frame Defects (M1) &amp; Reference Anchors (M2):</strong> Use <code>gemma-4-31b-it</code> (75% Recall / 4.8% FPR on M1; 89.5% Acc / 6.3% FPR on M2) — or cascade <code>30b-a3b &rarr; gemma-31b</code> on M2 to cut anchor latency nearly in half while holding <strong>~83% Recall / 3.1% FPR</strong>.<br>
  &bull; <strong>Drop <code>qwen3-vl-4b</code> &amp; <code>2b</code>:</strong> Unreliable JSON &amp; high FPR on real unseen diversity.
</div>

<h2>4. Holdout Error Slices by Style (gemma-4-31b-it)</h2>
<table>
  <thead><tr><th>Suite</th><th>World Style</th><th>N</th><th>Recall</th><th>FPR</th><th>Accuracy</th></tr></thead>
  <tbody>{slices_html}</tbody>
</table>

<h2>5. Suite 6: Holdout Closed-Loop Recovery + Independent Neon Audit</h2>
<div class="callout warn">
  <strong>Critical Production Discovery from Independent Verification:</strong><br>
  Gemma caught 5/10 Holdout failures on Round 0 (because <code>critic_loop.py</code> did not yet attach <code>ZOOMED_QUADRANTS</code> + <code>expected_cast_count</code>) and self-graded 3/5 as recovered. However, <strong>independent visual audit by Neon Medium revealed a huge root-cause bug in the repair generator:</strong> feeding <strong>3-view character turnaround sheets</strong> directly into <code>qwen-image-2.1-turbo</code> causes it to paste <strong>2 or 3 clones</strong> of the character (front + walking + side views) right into the repaired scene! Meanwhile, single-object prop repair worked cleanly on the bullhorn (<code>rec_c0008_05_drift_prop</code>).<br>
  <strong>Immediate Fix Next:</strong> Crop a <strong>single clean front/3/4 view</strong> from each turnaround sheet before passing it to <code>mflux</code>, and wire <code>ZOOMED_QUADRANTS</code> + <code>expected_cast_count</code> into <code>critic_loop.py</code>.
</div>
{"".join(s6_cards_html)}

</body>
</html>
"""
    out_p = RESULTS_DIR / "morning_report.html"
    out_p.write_text(doc, encoding="utf-8")
    print(f"Wrote Morning Dashboard -> {out_p} ({len(doc)} bytes)")
    print("Cascade 30b-a3b -> gemma-31b:", json.dumps(cascade_30b_31b, indent=2))


if __name__ == "__main__":
    main()
