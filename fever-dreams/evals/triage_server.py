"""Fast local click-through labeling server for evals/mined_data/.

Zero external dependencies (Python stdlib HTTP server + embedded dark-mode SPA).
Reads per-image sidecar JSONs from evals/mined_data/{shots,contact_sheets}/,
serves images + canonical references + paired shots side-by-side, writes triage
decisions immediately back to each sidecar JSON, and supports stratified
group-safe Dev/Holdout splitting + manifest export.

Usage:
    uv run python evals/triage_server.py
    # Then open http://127.0.0.1:8765/
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import random
import sys
import urllib.parse
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

FEVER_DREAMS_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = FEVER_DREAMS_ROOT.parent  # contains fever-dreams/ and meme-yearbook/
DEFAULT_DATA_DIR = FEVER_DREAMS_ROOT / "evals" / "mined_data"

ALLOWED_IMG_ROOTS = [
    FEVER_DREAMS_ROOT.resolve(),
    (WORKSPACE_ROOT / "meme-yearbook").resolve(),
]


def _resolve_safe_image(rel_path: str) -> Path | None:
    rel = rel_path.lstrip("/")
    candidates = [
        (FEVER_DREAMS_ROOT / rel).resolve(),
        (WORKSPACE_ROOT / rel).resolve(),
    ]
    for cand in candidates:
        if not cand.is_file():
            continue
        for allowed in ALLOWED_IMG_ROOTS:
            try:
                cand.relative_to(allowed)
                return cand
            except ValueError:
                continue
    return None


def _load_shots(data_dir: Path) -> list[dict[str, Any]]:
    shots_dir = data_dir / "shots"
    items: list[dict[str, Any]] = []
    if not shots_dir.exists():
        return items
    for jf in sorted(shots_dir.glob("*.json")):
        try:
            rec = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        rec["_file"] = jf.name
        items.append(rec)
    return items


def _load_sheets(data_dir: Path) -> list[dict[str, Any]]:
    sheets_dir = data_dir / "contact_sheets"
    items: list[dict[str, Any]] = []
    if not sheets_dir.exists():
        return items
    for jf in sorted(sheets_dir.glob("*.json")):
        try:
            rec = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        rec["_file"] = jf.name
        items.append(rec)
    return items


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _crop_single_pose_bytes(img_path: Path) -> bytes:
    """Return left single-pose view (~34% width) if image is a wide 3-view sheet, else original JPEG."""
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    if w / max(1, h) >= 1.5:
        im = im.crop((0, 0, int(w * 0.34), h))
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _auto_focus_crop_bytes(data_dir: Path, shot_id: str) -> bytes | None:
    """Fast foreground crop isolating characters/props by diffing against the Master Set Plate."""
    jf = data_dir / "shots" / f"{shot_id}.json"
    if not jf.exists():
        return None
    rec = json.loads(jf.read_text(encoding="utf-8"))
    shot_p = data_dir / "shots" / rec["image"]
    if not shot_p.exists():
        return None

    im = Image.open(shot_p).convert("RGB")
    w, h = im.size
    set_rel = ((rec.get("scene") or {}).get("canonical_refs") or {}).get("set")
    set_p = _resolve_safe_image(set_rel) if set_rel else None

    box: tuple[int, int, int, int] | None = None
    if set_p is not None:
        try:
            set_im = Image.open(set_p).convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
            arr_s = np.asarray(im, dtype=np.float32)
            arr_b = np.asarray(set_im, dtype=np.float32)
            diff = np.mean(np.abs(arr_s - arr_b), axis=2)
            # Ignore outer 2% border noise
            margin_y, margin_x = max(4, int(h * 0.02)), max(4, int(w * 0.02))
            diff[:margin_y, :] = 0
            diff[-margin_y:, :] = 0
            diff[:, :margin_x] = 0
            diff[:, -margin_x:] = 0

            # Adaptive threshold: pixels strongly differing from empty set plate
            thresh = max(24.0, float(np.percentile(diff, 84)))
            mask = diff >= thresh
            # Filter sparse rows/cols (require >= 3% of axis pixels active to suppress stray noise)
            row_active = np.where(mask.sum(axis=1) >= max(8, int(w * 0.03)))[0]
            col_active = np.where(mask.sum(axis=0) >= max(8, int(h * 0.03)))[0]
            if len(row_active) > 10 and len(col_active) > 10:
                y1, y2 = int(row_active[0]), int(row_active[-1])
                x1, x2 = int(col_active[0]), int(col_active[-1])
                bw, bh = x2 - x1, y2 - y1
                # Only use diff crop if it meaningfully zooms in (between 18% and 88% of frame area)
                area_ratio = (bw * bh) / float(w * h)
                if 0.08 <= area_ratio <= 0.85:
                    pad_x, pad_y = int(w * 0.06), int(h * 0.06)
                    box = (
                        max(0, x1 - pad_x),
                        max(0, y1 - pad_y),
                        min(w, x2 + pad_x),
                        min(h, y2 + pad_y),
                    )
        except Exception:
            box = None

    if box is None:
        # Clean center-subject fallback crop (middle 68% width, lower-middle 76% height)
        box = (int(w * 0.16), int(h * 0.14), int(w * 0.84), int(h * 0.90))

    cropped = im.crop(box)
    buf = BytesIO()
    cropped.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _assign_dev_holdout_split(data_dir: Path, holdout_ratio: float = 0.5, seed: int = 2026) -> dict[str, Any]:
    """Stratified group-safe split by shot_group (cycle) so characters/scenes never leak across Dev/Holdout."""
    shots_dir = data_dir / "shots"
    shots = _load_shots(data_dir)
    labeled = [s for s in shots if s.get("triage", {}).get("human_label") in {"PASS", "FAIL"}]

    # Group labeled shots by cycle prefix (e.g. c0001) so all shots from a cycle stay together
    groups: dict[str, list[dict[str, Any]]] = {}
    for s in labeled:
        gkey = str(s.get("scene", {}).get("shot_group") or s["id"].split("_")[0])
        groups.setdefault(gkey, []).append(s)

    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n_holdout = max(1, int(round(len(group_keys) * holdout_ratio))) if len(group_keys) > 1 else 0
    holdout_groups = set(group_keys[:n_holdout])

    counts = {"dev": {"PASS": 0, "FAIL": 0}, "holdout": {"PASS": 0, "FAIL": 0}}
    export_shots: list[dict[str, Any]] = []

    for s in labeled:
        gkey = str(s.get("scene", {}).get("shot_group") or s["id"].split("_")[0])
        split = "holdout" if gkey in holdout_groups else "dev"
        lbl = s["triage"]["human_label"]
        s["triage"]["split"] = split
        jf = shots_dir / s["_file"]
        clean_rec = {k: v for k, v in s.items() if k != "_file"}
        _atomic_write_json(jf, clean_rec)
        counts[split][lbl] += 1
        export_shots.append(
            {
                "id": s["id"],
                "image": f"shots/{s['image']}",
                "split": split,
                "human_label": lbl,
                "defect_types": s["triage"].get("defect_types", []),
                "severity": s["triage"].get("severity"),
                "notes": s["triage"].get("notes", ""),
                "scene": s.get("scene", {}),
            }
        )

    # Also split any human-verified contact sheets using the same cycle groups
    sheets_dir = data_dir / "contact_sheets"
    sheets = _load_sheets(data_dir)
    export_sheets: list[dict[str, Any]] = []
    for sh in sheets:
        if sh.get("triage", {}).get("human_verified_outliers") is None:
            continue
        # sheet_c0001_... -> c0001
        parts = sh["id"].split("_")
        cycle_prefix = parts[1] if len(parts) > 1 else sh["id"]
        in_holdout = any(gk.startswith(cycle_prefix + "_") or gk == cycle_prefix for gk in holdout_groups)
        split = "holdout" if in_holdout else "dev"
        sh["triage"]["split"] = split
        jf = sheets_dir / sh["_file"]
        clean_rec = {k: v for k, v in sh.items() if k != "_file"}
        _atomic_write_json(jf, clean_rec)
        export_sheets.append(
            {
                "id": sh["id"],
                "image": f"contact_sheets/{sh['image']}",
                "split": split,
                "entity_type": sh.get("entity_type"),
                "entity_id": sh.get("entity_id"),
                "entity_name": sh.get("entity_name"),
                "world_style": sh.get("world_style"),
                "tile_count": sh.get("tile_count", 3),
                "expected_outliers": sh["triage"]["human_verified_outliers"],
                "notes": sh["triage"].get("notes", ""),
            }
        )

    manifest = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_labeled_shots": len(export_shots),
        "total_labeled_sheets": len(export_sheets),
        "counts": counts,
        "shots": export_shots,
        "contact_sheets": export_sheets,
    }
    _atomic_write_json(data_dir / "labeled_manifest.json", manifest)
    return manifest


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fever Dreams — Eval Data Triage</title>
<style>
  :root {
    --bg: #11100e;
    --panel: #1b1814;
    --panel2: #24201a;
    --border: #362f26;
    --text: #f3ecdf;
    --muted: #a99c88;
    --gold: #f4d06f;
    --pass: #2ea043;
    --pass-bg: #16331e;
    --fail: #e5534b;
    --fail-bg: #3b1716;
    --skip: #b08800;
    --skip-bg: #332a10;
    --accent: #58a6ff;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  body { display: flex; flex-direction: column; overflow: hidden; }

  header {
    display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;
    padding: 10px 16px; background: var(--panel); border-bottom: 1px solid var(--border);
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .brand h1 { margin: 0; font-size: 16px; font-family: Georgia, serif; color: var(--gold); letter-spacing: 0.2px; }
  .tabs { display: flex; gap: 6px; }
  .tab {
    background: var(--panel2); color: var(--muted); border: 1px solid var(--border);
    border-radius: 7px; padding: 5px 11px; font-size: 12px; font-weight: 600; cursor: pointer;
  }
  .tab.active { background: var(--gold); color: #16130f; border-color: var(--gold); }

  .filters { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  select, button.hdr-btn {
    background: var(--panel2); color: var(--text); border: 1px solid var(--border);
    border-radius: 7px; padding: 5px 9px; font-size: 12px; cursor: pointer;
  }
  button.hdr-btn:hover { border-color: var(--gold); }
  label.chk { display: flex; align-items: center; gap: 5px; font-size: 12px; color: var(--muted); cursor: pointer; user-select: none; }

  .stats-bar {
    display: flex; align-items: center; justify-content: space-between; gap: 14px;
    padding: 6px 16px; background: #161310; border-bottom: 1px solid var(--border); font-size: 12px; color: var(--muted);
  }
  .stats-pills { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  .pill { padding: 2px 8px; border-radius: 999px; font-weight: 600; font-size: 11px; border: 1px solid var(--border); background: var(--panel2); color: var(--text); }
  .pill.pass { background: var(--pass-bg); border-color: var(--pass); color: #7ee787; }
  .pill.fail { background: var(--fail-bg); border-color: var(--fail); color: #ff9492; }
  .pill.skip { background: var(--skip-bg); border-color: var(--skip); color: #f2cc60; }
  .pill.info { border-color: #3a4f66; color: #9ecbff; background: #152232; }

  main { flex: 1; display: grid; grid-template-columns: 1fr 370px; min-height: 0; }
  @media (max-width: 1024px) { main { grid-template-columns: 1fr 320px; } }

  .stage {
    display: flex; flex-direction: column; min-height: 0; padding: 12px 16px; gap: 10px; overflow-y: auto;
  }
  .stage-header {
    display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap;
  }
  .item-title { font-size: 15px; font-weight: 700; color: var(--gold); }
  .item-meta { font-size: 12px; color: var(--muted); display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }

  .primary-row {
    display: grid; grid-template-columns: 1fr; gap: 12px; align-items: start;
  }
  .primary-row.has-pair { grid-template-columns: 1.15fr 0.85fr; }
  .img-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 8px;
    display: flex; flex-direction: column; gap: 6px;
  }
  .img-card.candidate { border-color: #5c4f3a; }
  .img-card .cap {
    display: flex; justify-content: space-between; align-items: center; font-size: 11px;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.4px; font-weight: 700;
  }
  .img-card img {
    width: 100%; max-height: 56vh; object-fit: contain; border-radius: 6px; background: #0a0908; cursor: zoom-in;
  }

  .refs-strip {
    display: flex; gap: 10px; overflow-x: auto; padding-bottom: 4px;
  }
  .ref-card {
    flex: 0 0 auto; width: 220px; background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 6px; display: flex; flex-direction: column; gap: 4px;
  }
  .ref-card .rlbl { font-size: 10px; color: var(--gold); font-weight: 700; text-transform: uppercase; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .ref-card img { width: 100%; height: 122px; object-fit: contain; background: #0a0908; border-radius: 5px; cursor: zoom-in; }

  details.prompt-box {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 12px; color: var(--muted);
  }
  details.prompt-box summary { cursor: pointer; font-weight: 600; color: var(--text); }
  details.prompt-box p { margin: 6px 0 0; line-height: 1.45; white-space: pre-wrap; }

  aside {
    background: var(--panel); border-left: 1px solid var(--border); padding: 14px;
    display: flex; flex-direction: column; gap: 12px; overflow-y: auto;
  }
  .nav-row { display: flex; gap: 8px; align-items: center; }
  .nav-row button {
    flex: 1; background: var(--panel2); color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 7px 10px; font-size: 12px; font-weight: 600; cursor: pointer;
  }
  .nav-row button:hover { border-color: var(--gold); }
  .pos-badge { font-size: 12px; color: var(--muted); text-align: center; min-width: 74px; font-variant-numeric: tabular-nums; }

  .verdict-row { display: grid; grid-template-columns: 1fr 1fr 0.7fr; gap: 8px; }
  .vbtn {
    border-radius: 10px; padding: 13px 8px; font-size: 14px; font-weight: 800; cursor: pointer;
    border: 2px solid var(--border); background: var(--panel2); color: var(--text);
    display: flex; flex-direction: column; align-items: center; gap: 2px; transition: transform 0.05s;
  }
  .vbtn:active { transform: scale(0.97); }
  .vbtn small { font-size: 10px; font-weight: 600; opacity: 0.75; }
  .vbtn.pass { border-color: #265933; }
  .vbtn.pass:hover, .vbtn.pass.active { background: var(--pass); border-color: #56d364; color: #08150b; }
  .vbtn.fail { border-color: #6b2624; }
  .vbtn.fail:hover, .vbtn.fail.active { background: var(--fail); border-color: #ff807c; color: #1a0605; }
  .vbtn.skip { border-color: #5d4a15; }
  .vbtn.skip:hover, .vbtn.skip.active { background: var(--skip); border-color: #f2cc60; color: #161204; }

  .section-lbl { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); margin-bottom: 5px; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip {
    background: var(--panel2); border: 1px solid var(--border); color: var(--text);
    border-radius: 7px; padding: 6px 8px; font-size: 12px; cursor: pointer; user-select: none;
    display: flex; align-items: center; gap: 5px;
  }
  .chip kbd { font-size: 10px; background: #12100d; border: 1px solid var(--border); border-radius: 4px; padding: 0 4px; color: var(--muted); }
  .chip.active { background: var(--fail-bg); border-color: var(--fail); color: #ffb3b1; font-weight: 700; }

  .sev-row { display: flex; gap: 8px; }
  .sev-btn {
    flex: 1; background: var(--panel2); border: 1px solid var(--border); color: var(--muted);
    border-radius: 7px; padding: 7px; font-size: 12px; font-weight: 600; cursor: pointer;
  }
  .sev-btn.active { background: #2c2419; border-color: var(--gold); color: var(--gold); }

  textarea {
    width: 100%; min-height: 54px; resize: vertical; background: #13110e; color: var(--text);
    border: 1px solid var(--border); border-radius: 8px; padding: 8px; font-size: 12px; font-family: inherit;
  }
  .save-next {
    width: 100%; padding: 11px; border-radius: 9px; border: 1px solid var(--gold);
    background: var(--gold); color: #15120d; font-weight: 800; font-size: 13px; cursor: pointer;
  }
  .save-next:hover { filter: brightness(1.06); }

  .tile-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .tile-btn {
    padding: 12px; border-radius: 9px; border: 2px solid var(--border); background: var(--panel2);
    color: var(--text); font-weight: 700; font-size: 13px; cursor: pointer;
  }
  .tile-btn.active { background: var(--fail-bg); border-color: var(--fail); color: #ffb3b1; }
  .tile-btn.clean.active { background: var(--pass-bg); border-color: var(--pass); color: #7ee787; }

  .ai-box {
    background: #17212e; border: 1px solid #355980; border-radius: 10px; padding: 10px 11px;
    display: flex; flex-direction: column; gap: 7px;
  }
  .ai-box.fail { background: #291819; border-color: #7d3635; }
  .ai-box.pass { background: #15261b; border-color: #2f6b40; }
  .ai-head { display: flex; justify-content: space-between; align-items: center; gap: 6px; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.4px; }
  .ai-reason { font-size: 12px; color: var(--text); line-height: 1.4; }
  .approve-btn {
    width: 100%; padding: 9px; border-radius: 8px; border: 1px solid #58a6ff;
    background: #1f6feb; color: #ffffff; font-weight: 800; font-size: 13px; cursor: pointer;
  }
  .approve-btn:hover { filter: brightness(1.1); }

  .shortcuts {
    margin-top: auto; padding-top: 8px; border-top: 1px solid var(--border);
    font-size: 11px; color: var(--muted); line-height: 1.6;
  }
  .shortcuts kbd {
    background: #12100d; border: 1px solid var(--border); border-radius: 4px; padding: 1px 5px; color: var(--text); font-size: 10px;
  }

  /* Fast Audit Mode */
  .audit-wrap {
    flex: 1; display: flex; flex-direction: column; min-height: 0; padding: 10px 16px 14px; gap: 10px; overflow-y: auto;
  }
  .group-bar {
    display: flex; gap: 6px; flex-wrap: wrap; align-items: center;
  }
  .gbtn {
    background: var(--panel2); color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 6px 10px; font-size: 12px; font-weight: 600; cursor: pointer;
    display: flex; align-items: center; gap: 6px;
  }
  .gbtn.active { background: var(--gold); color: #15120d; border-color: var(--gold); }
  .gbtn .rcnt { background: var(--fail); color: #fff; border-radius: 999px; padding: 1px 6px; font-size: 10px; font-weight: 800; }
  .big-status-bar {
    width: 100%; border-radius: 12px; padding: 14px 20px; cursor: pointer; user-select: none;
    display: flex; align-items: center; justify-content: space-between; gap: 14px; flex-wrap: wrap;
    font-size: 18px; font-weight: 900; letter-spacing: 0.3px; transition: background 0.1s, border-color 0.1s;
  }
  .big-status-bar.green {
    background: #196c2e; border: 3px solid #56d364; color: #f0fff4;
    box-shadow: 0 0 24px rgba(46, 160, 67, 0.25);
  }
  .big-status-bar.red {
    background: #a42220; border: 3px solid #ff7b72; color: #fff5f5;
    box-shadow: 0 0 24px rgba(229, 83, 75, 0.35);
  }
  .big-status-bar .hint { font-size: 13px; font-weight: 600; opacity: 0.92; }
  .audit-reason-bar {
    background: var(--panel); border: 1px solid var(--border); border-radius: 9px; padding: 9px 14px;
    display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; font-size: 13px;
  }
  .audit-trio {
    display: grid; grid-template-columns: 0.72fr 1.15fr 1.13fr; gap: 10px; align-items: stretch;
  }
  @media (max-width: 1100px) { .audit-trio { grid-template-columns: 1fr 1fr; } }
  .audit-trio .img-card img { max-height: 52vh; }
  .audit-footer {
    display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap;
    padding-top: 4px;
  }
  .audit-footer button {
    background: var(--panel2); color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 14px; font-size: 13px; font-weight: 700; cursor: pointer;
  }
  .audit-footer button.primary { background: var(--gold); color: #16130f; border-color: var(--gold); }
  .audit-footer button.rerate { background: var(--fail-bg); color: #ff9492; border-color: var(--fail); }

  .toast {
    position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%);
    background: #23201b; border: 1px solid var(--gold); color: var(--gold);
    padding: 8px 16px; border-radius: 999px; font-size: 12px; font-weight: 700;
    opacity: 0; pointer-events: none; transition: opacity 0.18s; z-index: 50;
  }
  .toast.show { opacity: 1; }

  .lightbox {
    position: fixed; inset: 0; background: rgba(0,0,0,0.93); display: none;
    align-items: center; justify-content: center; z-index: 100; overflow: hidden;
  }
  .lightbox.open { display: flex; }
  .lightbox img {
    max-width: 94vw; max-height: 90vh; object-fit: contain; border-radius: 8px;
    box-shadow: 0 12px 48px rgba(0,0,0,0.8); transform-origin: center center;
    cursor: grab; user-select: none; -webkit-user-drag: none;
  }
  .lightbox img.dragging { cursor: grabbing; }
  .lb-bar {
    position: fixed; top: 14px; right: 18px; z-index: 110; display: flex; gap: 8px; align-items: center;
    background: rgba(24,20,16,0.9); border: 1px solid var(--border); padding: 6px 10px; border-radius: 10px;
  }
  .lb-bar button {
    background: var(--panel2); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 10px; font-size: 13px; font-weight: 700; cursor: pointer;
  }
  .lb-bar button:hover { border-color: var(--gold); color: var(--gold); }
  .lb-bar span { font-size: 12px; color: var(--gold); min-width: 44px; text-align: center; font-weight: 700; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <h1>Fever Dreams Triage</h1>
    <div class="tabs">
      <button class="tab active" id="tabAudit" onclick="switchMode('audit')">Fast Audit (Spacebar)</button>
      <button class="tab" id="tabShots" onclick="switchMode('shots')">Full Rating</button>
      <button class="tab" id="tabSheets" onclick="switchMode('sheets')">Contact Sheets</button>
    </div>
  </div>
  <div class="filters" id="topFilters">
    <select id="fStatus" onchange="applyFilters()">
      <option value="needs_rerate">Needs Re-Rating (Red)</option>
      <option value="unlabeled">Unlabeled only</option>
      <option value="ai_prerated">AI Pre-rated (needs verify)</option>
      <option value="labeled_recent">Labeled (recent first)</option>
      <option value="all">All items</option>
      <option value="PASS">PASS only</option>
      <option value="FAIL">FAIL only</option>
      <option value="SKIP">SKIP only</option>
    </select>
    <select id="fCategory" onchange="applyFilters()">
      <option value="all">All categories</option>
    </select>
    <select id="fStyle" onchange="applyFilters()">
      <option value="all">All styles</option>
      <option value="golden_book_mixed_media">Golden Book</option>
      <option value="stop_motion_miniature">Stop-Motion</option>
    </select>
    <label class="chk" title="Interleave categories round-robin so stopping early still gives an even spread">
      <input type="checkbox" id="fBalanced" checked onchange="applyFilters()"> Balanced queue
    </label>
    <button class="hdr-btn" onclick="runSplit()" title="Assign 50/50 Dev/Holdout split grouped by cycle and export labeled_manifest.json">
      Export Dev/Holdout Split
    </button>
  </div>
</header>

<div class="stats-bar">
  <div class="stats-pills" id="statsPills">Loading...</div>
  <div id="saveState">Ready</div>
</div>

<main id="mainGrid" style="display:none">
  <section class="stage" id="stage"></section>
  <aside id="sidebar"></aside>
</main>
<section class="audit-wrap" id="auditView"></section>

<div class="toast" id="toast"></div>
<div class="lightbox" id="lightbox" onclick="if (event.target === this) closeLightbox()">
  <div class="lb-bar" onclick="event.stopPropagation()">
    <button onclick="lbZoom(-0.5)">&minus;</button>
    <span id="lbScale">100%</span>
    <button onclick="lbZoom(0.5)">+</button>
    <button onclick="lbReset()">Reset</button>
    <button onclick="closeLightbox()">Close [Esc]</button>
  </div>
  <img id="lightboxImg" src="" draggable="false">
</div>

<script>
const DEFECT_TYPES = [
  { id: "anatomy", label: "Anatomy / Limbs", key: "1" },
  { id: "accessory", label: "Broken Accessory", key: "2" },
  { id: "face_blob", label: "Face / Chin Blob", key: "3" },
  { id: "hand_merge", label: "Hand / Prop Merge", key: "4" },
  { id: "char_drift", label: "Character Drift", key: "5" },
  { id: "prop_drift", label: "Prop Drift", key: "6" },
  { id: "style_drift", label: "Style / Medium Drift", key: "7" },
  { id: "lighting_break", label: "Lighting Break", key: "8" },
  { id: "set_warp", label: "Set Warp / Text", key: "9" },
];

const AUDIT_GROUPS = [
  { id: "duplicate_character", label: "Duplicate Characters" },
  { id: "style_drift", label: "Style / Medium Drift" },
  { id: "char_drift", label: "Character Drift" },
  { id: "lighting_break", label: "Lighting Break" },
  { id: "prop_drift", label: "Prop Drift" },
  { id: "anatomy_limbs", label: "Anatomy / Limbs" },
  { id: "face_accessory_other", label: "Face / Accessory / Other" },
  { id: "presumed_pass", label: "Presumed Clean (PASS)" },
];

const DUP_REGEX = /duplicat|clone|twice|two copies|appear.*twice|double/i;

function classifyAuditGroup(s) {
  const ai = s.ai_suggestion || {};
  const tr = s.triage || {};
  const lbl = tr.human_label || ai.suggested_label || "PASS";
  const defs = new Set(tr.defect_types && tr.defect_types.length ? tr.defect_types : (ai.suggested_defects || []));
  const reason = ((tr.notes || "") + " " + (ai.reason || "")).trim();
  if (lbl === "PASS") return "presumed_pass";
  if (DUP_REGEX.test(reason)) return "duplicate_character";
  if (defs.has("lighting_break")) return "lighting_break";
  if (defs.has("prop_drift")) return "prop_drift";
  if (defs.has("style_drift")) return "style_drift";
  if (defs.has("char_drift")) return "char_drift";
  if (defs.has("anatomy")) return "anatomy_limbs";
  return "face_accessory_other";
}

let STATE = {
  mode: "audit",
  auditGroup: "duplicate_character",
  auditItems: [],
  auditIdx: 0,
  shots: [],
  shotsById: {},
  sheets: [],
  filtered: [],
  idx: 0,
  draftDefects: new Set(),
  draftSeverity: "obvious",
  draftNotes: "",
  draftLabel: null,
  draftOutliers: new Set(),
  undoStack: [],
};

function imgUrl(rel) {
  return "/img/" + encodeURI(rel.replace(/^\/+/, ""));
}

function showToast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove("show"), 2000);
}

let LB = { scale: 1, x: 0, y: 0, dragging: false, sx: 0, sy: 0 };

function lbApply() {
  const img = document.getElementById("lightboxImg");
  img.style.transform = `translate(${LB.x}px, ${LB.y}px) scale(${LB.scale})`;
  document.getElementById("lbScale").textContent = Math.round(LB.scale * 100) + "%";
}
function lbReset() {
  LB.scale = 1; LB.x = 0; LB.y = 0;
  lbApply();
}
function lbZoom(delta) {
  LB.scale = Math.max(0.5, Math.min(5, LB.scale + delta));
  if (LB.scale <= 1) { LB.x = 0; LB.y = 0; }
  lbApply();
}
function openLightbox(src) {
  document.getElementById("lightboxImg").src = src;
  lbReset();
  document.getElementById("lightbox").classList.add("open");
}
function closeLightbox() {
  document.getElementById("lightbox").classList.remove("open");
}
(function initLightboxPanZoom() {
  const lb = document.getElementById("lightbox");
  const img = document.getElementById("lightboxImg");
  lb.addEventListener("wheel", (e) => {
    if (!lb.classList.contains("open")) return;
    e.preventDefault();
    lbZoom(e.deltaY < 0 ? 0.35 : -0.35);
  }, { passive: false });
  img.addEventListener("mousedown", (e) => {
    LB.dragging = true; LB.sx = e.clientX - LB.x; LB.sy = e.clientY - LB.y;
    img.classList.add("dragging");
  });
  window.addEventListener("mousemove", (e) => {
    if (!LB.dragging) return;
    LB.x = e.clientX - LB.sx; LB.y = e.clientY - LB.sy;
    lbApply();
  });
  window.addEventListener("mouseup", () => {
    LB.dragging = false; img.classList.remove("dragging");
  });
})();

async function loadData() {
  const res = await fetch("/api/data");
  const data = await res.json();
  STATE.shots = data.shots || [];
  STATE.sheets = data.sheets || [];
  STATE.shotsById = {};
  for (const s of STATE.shots) STATE.shotsById[s.id] = s;

  // Populate category dropdown
  const cats = [...new Set(STATE.shots.map(s => (s.scene || {}).category).filter(Boolean))].sort();
  const sel = document.getElementById("fCategory");
  const cur = sel.value;
  sel.innerHTML = '<option value="all">All categories</option>' + cats.map(c => `<option value="${c}">${c}</option>`).join("");
  if (cats.includes(cur)) sel.value = cur;

  if (STATE.mode === "audit") {
    switchMode("audit");
  } else {
    applyFilters(false);
  }
  renderStats();
}

function switchMode(mode) {
  STATE.mode = mode;
  document.getElementById("tabAudit").classList.toggle("active", mode === "audit");
  document.getElementById("tabShots").classList.toggle("active", mode === "shots");
  document.getElementById("tabSheets").classList.toggle("active", mode === "sheets");

  const isAudit = mode === "audit";
  document.getElementById("auditView").style.display = isAudit ? "flex" : "none";
  document.getElementById("mainGrid").style.display = isAudit ? "none" : "grid";
  document.getElementById("fStatus").style.display = isAudit ? "none" : "";
  document.getElementById("fCategory").style.display = mode === "shots" ? "" : "none";
  document.getElementById("fBalanced").parentElement.style.display = mode === "shots" ? "" : "none";

  if (isAudit) {
    rebuildAuditQueue(false);
    renderAudit();
  } else {
    applyFilters(false);
  }
}

function selectAuditGroup(gid) {
  STATE.auditGroup = gid;
  rebuildAuditQueue(false);
  renderAudit();
}

function rebuildAuditQueue(keepId = true) {
  const prevId = keepId && STATE.auditItems[STATE.auditIdx] ? STATE.auditItems[STATE.auditIdx].id : null;
  const fStyle = document.getElementById("fStyle").value;
  STATE.auditItems = STATE.shots.filter(s => {
    if (classifyAuditGroup(s) !== STATE.auditGroup) return false;
    if (fStyle !== "all" && (s.scene || {}).style !== fStyle) return false;
    return true;
  });
  if (prevId) {
    const ni = STATE.auditItems.findIndex(x => x.id === prevId);
    STATE.auditIdx = ni >= 0 ? ni : Math.min(STATE.auditIdx, Math.max(0, STATE.auditItems.length - 1));
  } else {
    STATE.auditIdx = 0;
  }
}

async function toggleAuditCorrect() {
  const it = STATE.auditItems[STATE.auditIdx];
  if (!it) return;
  const tr = it.triage || {};
  const currentlyWrong = tr.audit_status === "wrong" || tr.needs_rerate === true;
  const nextStatus = currentlyWrong ? "correct" : "wrong";
  document.getElementById("saveState").textContent = "Saving...";
  const res = await fetch("/api/audit_toggle", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: it.id, audit_status: nextStatus }),
  });
  if (!res.ok) { showToast("Save failed!"); return; }
  it.triage = it.triage || {};
  it.triage.audit_status = nextStatus;
  it.triage.needs_rerate = (nextStatus === "wrong");
  if (nextStatus === "wrong") it.triage.human_label = null;
  document.getElementById("saveState").textContent = `${it.id}: ${nextStatus.toUpperCase()}`;
  renderAudit();
}

async function commitAuditGroupAndNext() {
  const ids = STATE.auditItems.map(x => x.id);
  if (!ids.length) return;
  const res = await fetch("/api/audit_commit_group", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids }),
  });
  const data = await res.json();
  await loadData();
  showToast(`Locked ${data.approved || 0} GREEN items; ${data.flagged_red || 0} RED queued for re-rating!`);
  const curIdx = AUDIT_GROUPS.findIndex(g => g.id === STATE.auditGroup);
  if (curIdx >= 0 && curIdx < AUDIT_GROUPS.length - 1) {
    selectAuditGroup(AUDIT_GROUPS[curIdx + 1].id);
  } else {
    goToRerateRed();
  }
}

function goToRerateRed() {
  document.getElementById("fStatus").value = "needs_rerate";
  switchMode("shots");
}

function stepAudit(delta) {
  if (!STATE.auditItems.length) return;
  const ni = STATE.auditIdx + delta;
  if (ni < 0) { showToast("Start of group"); return; }
  if (ni >= STATE.auditItems.length) { showToast("End of group — click 'Lock In Greens & Next Group' or switch group!"); return; }
  STATE.auditIdx = ni;
  renderAudit();
}

function renderAudit() {
  renderStats();
  const root = document.getElementById("auditView");
  const fStyle = document.getElementById("fStyle").value;
  const totalRed = STATE.shots.filter(s => (s.triage || {}).needs_rerate === true).length;

  const groupBtns = AUDIT_GROUPS.map(g => {
    const items = STATE.shots.filter(s => classifyAuditGroup(s) === g.id && (fStyle === "all" || (s.scene || {}).style === fStyle));
    const reds = items.filter(s => (s.triage || {}).needs_rerate === true).length;
    const act = g.id === STATE.auditGroup ? "active" : "";
    return `<button class="gbtn ${act}" onclick="selectAuditGroup('${g.id}')">
      ${g.label} (${items.length})${reds ? `<span class="rcnt">${reds}</span>` : ""}
    </button>`;
  }).join("");

  const it = STATE.auditItems[STATE.auditIdx];
  if (!it) {
    root.innerHTML = `<div class="group-bar">${groupBtns}</div>
      <div style="padding:40px;text-align:center;color:var(--muted)">No items in this group for the current style filter.</div>`;
    return;
  }

  const tr = it.triage || {};
  const ai = it.ai_suggestion || {};
  const sc = it.scene || {};
  const isWrong = tr.audit_status === "wrong" || tr.needs_rerate === true;
  const candSrc = imgUrl("evals/mined_data/shots/" + it.image);
  const focusSrc = "/img_focus/" + encodeURIComponent(it.id);
  const paired = sc.paired_with ? STATE.shotsById[sc.paired_with] : null;
  const pairedSrc = paired ? imgUrl("evals/mined_data/shots/" + paired.image) : null;

  const refs = [];
  const crefs = sc.canonical_refs || {};
  if (Array.isArray(crefs.cast)) crefs.cast.forEach((p, i) => refs.push({ label: `Cast Ref ${i + 1}`, rel: p, src: imgUrl(p) }));
  if (crefs.prop) refs.push({ label: "Prop Ref", rel: crefs.prop, src: imgUrl(crefs.prop) });
  if (crefs.set) refs.push({ label: "Master Set Plate", rel: crefs.set, src: imgUrl(crefs.set) });

  // Pick the primary single-pose reference for the left column (Prop Ref for prop_drift, else Cast Ref 1)
  let primaryRefRel = null;
  let primaryRefLbl = "Canonical Reference";
  if (STATE.auditGroup === "prop_drift" && crefs.prop) {
    primaryRefRel = crefs.prop;
    primaryRefLbl = "Prop Ref (Single View)";
  } else if (STATE.auditGroup === "lighting_break" && pairedSrc) {
    primaryRefRel = null; // Lighting break compares against paired baseline directly
  } else if (Array.isArray(crefs.cast) && crefs.cast.length > 0) {
    primaryRefRel = crefs.cast[0];
    primaryRefLbl = "Cast Ref 1 (Single Pose)";
  } else if (crefs.prop) {
    primaryRefRel = crefs.prop;
    primaryRefLbl = "Prop Ref (Single View)";
  }
  const singleRefSrc = primaryRefRel ? ("/img_single/" + encodeURI(primaryRefRel.replace(/^\/+/, ""))) : null;

  const aiLbl = tr.human_label || ai.suggested_label || "PASS";
  const aiTags = (tr.defect_types && tr.defect_types.length ? tr.defect_types : (ai.suggested_defects || [])).join(", ") || "none";
  const aiReason = (tr.notes || ai.reason || "").replace(/</g, "&lt;");

  const mainVisualHtml = singleRefSrc ? `
    <div class="audit-trio">
      <div class="img-card">
        <div class="cap"><span>${primaryRefLbl}</span><span>Compare &rarr;</span></div>
        <img src="${singleRefSrc}" onclick="openLightbox('${imgUrl(primaryRefRel)}')" title="Click for zoomable full ref sheet">
      </div>
      <div class="img-card candidate">
        <div class="cap"><span>Auto-Zoomed Subject Crop</span><span>Click to zoom/pan</span></div>
        <img src="${focusSrc}" onclick="openLightbox('${focusSrc}')" title="Click to zoom/pan crop">
      </div>
      <div class="img-card">
        <div class="cap"><span>Full Scene (Click to Zoom/Pan)</span><span>${STATE.auditIdx + 1} / ${STATE.auditItems.length}</span></div>
        <img src="${candSrc}" onclick="openLightbox('${candSrc}')" title="Click to open interactive zoomable full scene">
      </div>
    </div>
  ` : `
    <div class="primary-row ${pairedSrc ? "has-pair" : ""}">
      <div class="img-card candidate">
        <div class="cap"><span>Candidate Shot (Click to Zoom/Pan)</span><span>${STATE.auditIdx + 1} / ${STATE.auditItems.length}</span></div>
        <img src="${candSrc}" style="max-height:52vh" onclick="openLightbox('${candSrc}')">
      </div>
      ${pairedSrc ? `
      <div class="img-card">
        <div class="cap"><span>Paired Baseline (${sc.paired_with})</span><span>Compare lighting/continuity</span></div>
        <img src="${pairedSrc}" style="max-height:52vh" onclick="openLightbox('${pairedSrc}')">
      </div>` : ""}
    </div>
  `;

  root.innerHTML = `
    <div class="group-bar">${groupBtns}</div>

    <div class="big-status-bar ${isWrong ? "red" : "green"}" onclick="toggleAuditCorrect()">
      <span>${isWrong ? "&#10005; GEMINI WRONG — FLAGGED FOR RE-RATING" : "&#10003; GEMINI CORRECT"}</span>
      <span class="hint">[SPACEBAR] to toggle &bull; [&larr; / &rarr;] to browse (${STATE.auditIdx + 1} / ${STATE.auditItems.length})</span>
    </div>

    <div class="audit-reason-bar">
      <div>
        <strong style="color:var(--gold)">${it.id}</strong> &nbsp;
        <span class="pill ${aiLbl.toLowerCase()}">Gemini: ${aiLbl}</span>
        <span class="pill info">${aiTags}</span>
        <span style="margin-left:8px">${aiReason}</span>
      </div>
      <div style="color:var(--muted);font-size:12px">${sc.style || ""} &bull; ${sc.category || ""}</div>
    </div>

    ${mainVisualHtml}

    ${refs.length ? `
    <div class="refs-strip">
      ${pairedSrc && singleRefSrc ? `
        <div class="ref-card">
          <div class="rlbl">Paired Baseline</div>
          <img src="${pairedSrc}" onclick="openLightbox('${pairedSrc}')">
        </div>` : ""}
      ${refs.map(r => `
        <div class="ref-card">
          <div class="rlbl">${r.label}</div>
          <img src="${r.src}" onclick="openLightbox('${r.src}')">
        </div>
      `).join("")}
    </div>` : ""}

    <div class="audit-footer">
      <div style="display:flex;gap:8px">
        <button onclick="stepAudit(-1)">&larr; Prev</button>
        <button onclick="toggleAuditCorrect()">Toggle Correct/Wrong [Space]</button>
        <button onclick="stepAudit(1)">Next &rarr;</button>
      </div>
      <div style="display:flex;gap:8px">
        <button class="primary" onclick="commitAuditGroupAndNext()">Lock In Greens &amp; Next Group &rarr;</button>
        <button class="rerate" onclick="goToRerateRed()">Re-Rate All Red Items (${totalRed})</button>
      </div>
    </div>
  `;
}

function interleaveByCategory(items) {
  const buckets = new Map();
  for (const it of items) {
    const k = (it.scene || {}).category || "other";
    if (!buckets.has(k)) buckets.set(k, []);
    buckets.get(k).push(it);
  }
  const keys = [...buckets.keys()];
  const out = [];
  let more = true;
  while (more) {
    more = false;
    for (const k of keys) {
      const b = buckets.get(k);
      if (b.length) {
        out.push(b.shift());
        more = true;
      }
    }
  }
  return out;
}

function applyFilters(keepCurrentId = false) {
  if (STATE.mode === "audit") {
    rebuildAuditQueue(keepCurrentId);
    renderAudit();
    return;
  }
  const prevId = keepCurrentId && STATE.filtered[STATE.idx] ? STATE.filtered[STATE.idx].id : null;
  const fStatus = document.getElementById("fStatus").value;
  const fCat = document.getElementById("fCategory").value;
  const fStyle = document.getElementById("fStyle").value;
  const balanced = document.getElementById("fBalanced").checked;

  let list = STATE.mode === "shots" ? [...STATE.shots] : [...STATE.sheets];
  list = list.filter(it => {
    const tr = it.triage || {};
    if (STATE.mode === "shots") {
      const lbl = tr.human_label;
      if (fStatus === "needs_rerate" && tr.needs_rerate !== true) return false;
      if (fStatus === "unlabeled" && lbl !== null && lbl !== undefined) return false;
      if (fStatus === "ai_prerated" && !((lbl === null || lbl === undefined) && it.ai_suggestion)) return false;
      if (fStatus === "labeled_recent" && (lbl === null || lbl === undefined)) return false;
      if ((fStatus === "PASS" || fStatus === "FAIL" || fStatus === "SKIP") && lbl !== fStatus) return false;
      if (fCat !== "all" && (it.scene || {}).category !== fCat) return false;
      if (fStyle !== "all" && (it.scene || {}).style !== fStyle) return false;
    } else {
      const out = tr.human_verified_outliers;
      if (fStatus === "unlabeled" && out !== null && out !== undefined) return false;
      if (fStatus === "ai_prerated" && !((out === null || out === undefined) && it.ai_suggestion)) return false;
      if (fStatus === "labeled_recent" && (out === null || out === undefined)) return false;
      if (fStatus === "PASS" && !(Array.isArray(out) && out.length === 0)) return false;
      if (fStatus === "FAIL" && !(Array.isArray(out) && out.length > 0)) return false;
      if (fStyle !== "all" && it.world_style !== fStyle) return false;
    }
    return true;
  });

  if (fStatus === "labeled_recent") {
    list.sort((a, b) => String((b.triage || {}).labeled_at || "").localeCompare(String((a.triage || {}).labeled_at || "")));
  } else if (STATE.mode === "shots" && balanced) {
    list = interleaveByCategory(list);
  }

  STATE.filtered = list;
  if (prevId) {
    const ni = list.findIndex(x => x.id === prevId);
    STATE.idx = ni >= 0 ? ni : Math.min(STATE.idx, Math.max(0, list.length - 1));
  } else {
    STATE.idx = 0;
  }
  loadCurrentDraft();
  render();
}

function loadCurrentDraft() {
  const it = STATE.filtered[STATE.idx];
  if (!it) return;
  const tr = it.triage || {};
  const ai = it.ai_suggestion || null;
  if (STATE.mode === "shots") {
    if (tr.human_label) {
      STATE.draftLabel = tr.human_label;
      STATE.draftDefects = new Set(tr.defect_types || []);
      STATE.draftSeverity = tr.severity || "obvious";
      STATE.draftNotes = tr.notes || "";
    } else if (ai) {
      STATE.draftLabel = ai.suggested_label || null;
      STATE.draftDefects = new Set(ai.suggested_defects || []);
      STATE.draftSeverity = ai.suggested_severity || "obvious";
      STATE.draftNotes = ai.reason || "";
    } else {
      STATE.draftLabel = null;
      STATE.draftDefects = new Set();
      STATE.draftSeverity = "obvious";
      STATE.draftNotes = "";
    }
  } else {
    const existing = tr.human_verified_outliers;
    const initial = Array.isArray(existing) ? existing : (it.intended_outlier_tiles || []);
    STATE.draftOutliers = new Set(initial);
    STATE.draftNotes = tr.notes || "";
  }
}

function renderStats() {
  let pass = 0, fail = 0, skip = 0, dev = 0, hold = 0, aiReady = 0;
  for (const s of STATE.shots) {
    const tr = s.triage || {};
    if (tr.human_label === "PASS") pass++;
    else if (tr.human_label === "FAIL") fail++;
    else if (tr.human_label === "SKIP") skip++;
    else if (s.ai_suggestion) aiReady++;
    if (tr.split === "dev") dev++;
    else if (tr.split === "holdout") hold++;
  }
  const shDone = STATE.sheets.filter(x => x.triage && x.triage.human_verified_outliers !== null && x.triage.human_verified_outliers !== undefined).length;
  document.getElementById("statsPills").innerHTML = `
    <span class="pill">Shots: ${pass + fail + skip} / ${STATE.shots.length}</span>
    <span class="pill pass">PASS: ${pass}</span>
    <span class="pill fail">FAIL: ${fail}</span>
    <span class="pill skip">SKIP: ${skip}</span>
    ${aiReady ? `<span class="pill info">AI Ready: ${aiReady}</span>` : ""}
    <span class="pill info">Dev: ${dev} | Holdout: ${hold}</span>
    <span class="pill">Sheets: ${shDone} / ${STATE.sheets.length}</span>
  `;
}

function render() {
  renderStats();
  const stage = document.getElementById("stage");
  const side = document.getElementById("sidebar");
  const it = STATE.filtered[STATE.idx];

  if (!it) {
    stage.innerHTML = `<div style="padding:40px;text-align:center;color:var(--muted)">
      <h2 style="color:var(--gold)">Queue empty!</h2>
      <p>No items match the current filter. Switch filter to "All items" or pick another category.</p>
    </div>`;
    side.innerHTML = "";
    return;
  }

  if (STATE.mode === "shots") {
    renderShotView(it, stage, side);
  } else {
    renderSheetView(it, stage, side);
  }
}

function renderShotView(it, stage, side) {
  const sc = it.scene || {};
  const gen = it.generation || {};
  const tr = it.triage || {};
  const candSrc = imgUrl("evals/mined_data/shots/" + it.image);
  const paired = sc.paired_with ? STATE.shotsById[sc.paired_with] : null;
  const pairedSrc = paired ? imgUrl("evals/mined_data/shots/" + paired.image) : null;

  const refs = [];
  const crefs = sc.canonical_refs || {};
  if (Array.isArray(crefs.cast)) {
    crefs.cast.forEach((p, i) => refs.push({ label: `Cast Ref ${i + 1}`, src: imgUrl(p) }));
  }
  if (crefs.prop) refs.push({ label: "Prop Ref", src: imgUrl(crefs.prop) });
  if (crefs.set) refs.push({ label: "Master Set Plate", src: imgUrl(crefs.set) });

  const curBadge = tr.human_label
    ? `<span class="pill ${tr.human_label.toLowerCase()}">${tr.human_label}</span>`
    : `<span class="pill">UNLABELED</span>`;

  stage.innerHTML = `
    <div class="stage-header">
      <div>
        <div class="item-title">${it.id}</div>
        <div class="item-meta">
          ${curBadge}
          <span class="pill info">${sc.category || "?"}</span>
          <span class="pill">${sc.style || "?"}</span>
          <span>Intent: <strong>${sc.expected_continuity || "?"}</strong></span>
          <span>Set: ${sc.set_id || "?"}</span>
          ${tr.split ? `<span class="pill info">split: ${tr.split}</span>` : ""}
        </div>
      </div>
    </div>

    <div class="primary-row ${pairedSrc ? "has-pair" : ""}">
      <div class="img-card candidate">
        <div class="cap"><span>Candidate Shot (click to zoom)</span><span>${gen.width || ""}x${gen.height || ""}</span></div>
        <img src="${candSrc}" onclick="openLightbox('${candSrc}')">
      </div>
      ${pairedSrc ? `
      <div class="img-card">
        <div class="cap"><span>Paired Baseline (${sc.paired_with})</span><span>Compare continuity</span></div>
        <img src="${pairedSrc}" onclick="openLightbox('${pairedSrc}')">
      </div>` : ""}
    </div>

    ${refs.length ? `
    <div>
      <div class="section-lbl">Canonical References (click any to zoom)</div>
      <div class="refs-strip">
        ${refs.map(r => `
          <div class="ref-card">
            <div class="rlbl">${r.label}</div>
            <img src="${r.src}" onclick="openLightbox('${r.src}')">
          </div>
        `).join("")}
      </div>
    </div>` : ""}

    <details class="prompt-box">
      <summary>Generation Prompt &amp; Metadata</summary>
      <p>${(gen.prompt || "").replace(/</g, "&lt;")}</p>
    </details>
  `;

  const ai = it.ai_suggestion || null;
  const aiBoxHtml = ai ? `
    <div class="ai-box ${(ai.suggested_label || "").toLowerCase()}">
      <div class="ai-head">
        <span>AI Suggestion (${ai.model || "flash"})</span>
        <span class="pill ${(ai.suggested_label || "").toLowerCase()}">${ai.suggested_label || "?"}${ai.suggested_severity ? " · " + ai.suggested_severity : ""}</span>
      </div>
      <div class="ai-reason">${String(ai.reason || "").replace(/</g, "&lt;")}</div>
      <button class="approve-btn" onclick="approveAI()">Approve AI [A]</button>
    </div>
  ` : "";

  side.innerHTML = `
    <div class="nav-row">
      <button onclick="step(-1)">&larr; Prev</button>
      <div class="pos-badge">${STATE.idx + 1} / ${STATE.filtered.length}</div>
      <button onclick="step(1)">Next &rarr;</button>
    </div>

    ${aiBoxHtml}

    <div>
      <div class="section-lbl">Verdict</div>
      <div class="verdict-row">
        <button class="vbtn pass ${STATE.draftLabel === "PASS" ? "active" : ""}" onclick="quickSave('PASS')">
          PASS <small>[P]</small>
        </button>
        <button class="vbtn fail ${STATE.draftLabel === "FAIL" ? "active" : ""}" onclick="selectFail()">
          FAIL <small>[F]</small>
        </button>
        <button class="vbtn skip ${STATE.draftLabel === "SKIP" ? "active" : ""}" onclick="quickSave('SKIP')">
          SKIP <small>[S]</small>
        </button>
      </div>
    </div>

    <div>
      <div class="section-lbl">Defect Tags (if FAIL — keys 1-9)</div>
      <div class="chips">
        ${DEFECT_TYPES.map(d => `
          <div class="chip ${STATE.draftDefects.has(d.id) ? "active" : ""}" onclick="toggleDefect('${d.id}')">
            <kbd>${d.key}</kbd> ${d.label}
          </div>
        `).join("")}
      </div>
    </div>

    <div>
      <div class="section-lbl">Severity</div>
      <div class="sev-row">
        <button class="sev-btn ${STATE.draftSeverity === "subtle" ? "active" : ""}" onclick="setSeverity('subtle')">Subtle / Hard</button>
        <button class="sev-btn ${STATE.draftSeverity === "obvious" ? "active" : ""}" onclick="setSeverity('obvious')">Obvious</button>
      </div>
    </div>

    <div>
      <div class="section-lbl">Optional Note</div>
      <textarea id="noteInput" placeholder="e.g. half glasses on left cheek, wrong hat color..." oninput="STATE.draftNotes = this.value">${STATE.draftNotes.replace(/</g, "&lt;")}</textarea>
    </div>

    <button class="save-next" onclick="saveCurrentAndNext()">Save &amp; Next [Enter]</button>

    <div class="shortcuts">
      <strong>Fast flow:</strong> <kbd>A</kbd> approve AI &bull; <kbd>P</kbd> PASS &bull; <kbd>1</kbd>-<kbd>9</kbd> tags &bull; <kbd>F</kbd>/<kbd>Enter</kbd> save &bull; <kbd>S</kbd> skip &bull; <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> nav &bull; <kbd>Z</kbd> undo
    </div>
  `;
}

function renderSheetView(it, stage, side) {
  const tr = it.triage || {};
  const src = imgUrl("evals/mined_data/contact_sheets/" + it.image);
  const verified = tr.human_verified_outliers;
  const badge = Array.isArray(verified)
    ? `<span class="pill ${verified.length ? "fail" : "pass"}">VERIFIED: ${verified.length ? "Tiles " + verified.join(",") : "ALL CLEAN"}</span>`
    : `<span class="pill">UNLABELED</span>`;

  stage.innerHTML = `
    <div class="stage-header">
      <div>
        <div class="item-title">${it.id}</div>
        <div class="item-meta">
          ${badge}
          <span class="pill info">${it.entity_type}: ${it.entity_name}</span>
          <span class="pill">${it.world_style || ""}</span>
          <span>Intended outlier hint: <strong>${(it.intended_outlier_tiles || []).join(", ") || "none"}</strong></span>
        </div>
      </div>
    </div>
    <div class="img-card candidate">
      <div class="cap"><span>2x2 Contact Sheet (Tile 0 = Canonical Ref)</span><span>Click to zoom</span></div>
      <img src="${src}" style="max-height:72vh" onclick="openLightbox('${src}')">
    </div>
  `;

  const isClean = STATE.draftOutliers.size === 0;
  const ai = it.ai_suggestion || null;
  const aiBoxHtml = ai ? `
    <div class="ai-box ${(ai.suggested_label || "").toLowerCase()}">
      <div class="ai-head">
        <span>AI Suggestion (${ai.model || "flash"})</span>
        <span class="pill ${(ai.suggested_label || "").toLowerCase()}">${ai.suggested_label || "?"}${ai.suggested_severity ? " · " + ai.suggested_severity : ""}</span>
      </div>
      <div class="ai-reason">${String(ai.reason || "").replace(/</g, "&lt;")}</div>
      <button class="approve-btn" onclick="approveAI()">Approve AI [A]</button>
    </div>
  ` : "";

  side.innerHTML = `
    <div class="nav-row">
      <button onclick="step(-1)">&larr; Prev</button>
      <div class="pos-badge">${STATE.idx + 1} / ${STATE.filtered.length}</div>
      <button onclick="step(1)">Next &rarr;</button>
    </div>

    ${aiBoxHtml}

    <div>
      <div class="section-lbl">Which tiles do NOT match Tile 0 (REF)?</div>
      <div class="tile-row">
        <button class="tile-btn clean ${isClean ? "active" : ""}" onclick="clearOutliers()">0: All Clean [0]</button>
        <button class="tile-btn ${STATE.draftOutliers.has(1) ? "active" : ""}" onclick="toggleOutlier(1)">Tile 1 Outlier [1]</button>
        <button class="tile-btn ${STATE.draftOutliers.has(2) ? "active" : ""}" onclick="toggleOutlier(2)">Tile 2 Outlier [2]</button>
        <button class="tile-btn ${STATE.draftOutliers.has(3) ? "active" : ""}" onclick="toggleOutlier(3)">Tile 3 Outlier [3]</button>
      </div>
    </div>

    <div>
      <div class="section-lbl">Optional Note</div>
      <textarea id="noteInput" placeholder="Optional note..." oninput="STATE.draftNotes = this.value">${STATE.draftNotes.replace(/</g, "&lt;")}</textarea>
    </div>

    <button class="save-next" onclick="saveSheetAndNext()">Verify &amp; Next [Enter]</button>

    <div class="shortcuts">
      <strong>Fast flow:</strong> <kbd>0</kbd> all clean &bull; <kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd> toggle outlier tile &bull; <kbd>Enter</kbd> save &amp; next &bull; <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> navigate
    </div>
  `;
}

function step(delta) {
  if (!STATE.filtered.length) return;
  const nextIdx = STATE.idx + delta;
  if (nextIdx < 0) {
    showToast("At start of queue");
    return;
  }
  if (nextIdx >= STATE.filtered.length) {
    showToast("At end of queue");
    return;
  }
  STATE.idx = nextIdx;
  loadCurrentDraft();
  render();
}

function toggleDefect(id) {
  if (STATE.draftDefects.has(id)) STATE.draftDefects.delete(id);
  else {
    STATE.draftDefects.add(id);
    STATE.draftLabel = "FAIL";
  }
  render();
}

function setSeverity(s) {
  STATE.draftSeverity = s;
  render();
}

function selectFail() {
  STATE.draftLabel = "FAIL";
  if (STATE.draftDefects.size > 0) {
    saveCurrentAndNext();
  } else {
    render();
    showToast("Marked FAIL — pick defect tag(s) 1-9 or press Enter to save");
  }
}

async function approveAI() {
  const it = STATE.filtered[STATE.idx];
  if (!it || !it.ai_suggestion) {
    showToast("No AI suggestion on this item");
    return;
  }
  const ai = it.ai_suggestion;
  STATE.draftLabel = ai.suggested_label || "PASS";
  STATE.draftDefects = new Set(ai.suggested_defects || []);
  STATE.draftSeverity = ai.suggested_severity || "obvious";
  if (!STATE.draftNotes) STATE.draftNotes = ai.reason || "";
  await saveCurrentAndNext();
}

async function quickSave(label) {
  STATE.draftLabel = label;
  if (label === "PASS") {
    STATE.draftDefects.clear();
  }
  await saveCurrentAndNext();
}

async function saveCurrentAndNext() {
  const it = STATE.filtered[STATE.idx];
  if (!it) return;
  const label = STATE.draftLabel || (STATE.draftDefects.size > 0 ? "FAIL" : "PASS");
  const prevTriage = JSON.parse(JSON.stringify(it.triage || {}));
  const payload = {
    mode: "shots",
    id: it.id,
    triage: {
      human_label: label,
      defect_types: label === "FAIL" ? [...STATE.draftDefects] : [],
      severity: label === "FAIL" ? STATE.draftSeverity : null,
      notes: STATE.draftNotes.trim(),
      split: (it.triage || {}).split || null,
      audit_status: "correct",
      needs_rerate: false,
    },
  };
  await persistTriage(it, payload, prevTriage);
}

function clearOutliers() {
  STATE.draftOutliers.clear();
  render();
}
function toggleOutlier(n) {
  if (STATE.draftOutliers.has(n)) STATE.draftOutliers.delete(n);
  else STATE.draftOutliers.add(n);
  render();
}

async function saveSheetAndNext() {
  const it = STATE.filtered[STATE.idx];
  if (!it) return;
  const prevTriage = JSON.parse(JSON.stringify(it.triage || {}));
  const payload = {
    mode: "sheets",
    id: it.id,
    triage: {
      human_verified_outliers: [...STATE.draftOutliers].sort((a, b) => a - b),
      notes: STATE.draftNotes.trim(),
      split: (it.triage || {}).split || null,
    },
  };
  await persistTriage(it, payload, prevTriage);
}

async function persistTriage(it, payload, prevTriage) {
  document.getElementById("saveState").textContent = "Saving...";
  const res = await fetch("/api/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    showToast("Save failed!");
    return;
  }
  it.triage = payload.triage;
  STATE.undoStack.push({ mode: payload.mode, item: it, prevTriage });
  document.getElementById("saveState").textContent = "Saved " + it.id;

  // Keep saved items in the current queue so Prev (<-) always lets you step back
  // and edit your rating, tags, and notes. Advance to the next item if available.
  if (STATE.idx < STATE.filtered.length - 1) {
    STATE.idx += 1;
  } else {
    showToast("Saved — reached end of queue!");
  }
  loadCurrentDraft();
  render();
}

async function undoLast() {
  const last = STATE.undoStack.pop();
  if (!last) { showToast("Nothing to undo"); return; }
  await fetch("/api/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: last.mode, id: last.item.id, triage: last.prevTriage }),
  });
  last.item.triage = last.prevTriage;
  applyFilters(false);
  const idx = STATE.filtered.findIndex(x => x.id === last.item.id);
  if (idx >= 0) STATE.idx = idx;
  loadCurrentDraft();
  render();
  showToast("Undid " + last.item.id);
}

async function runSplit() {
  const res = await fetch("/api/split", { method: "POST" });
  const data = await res.json();
  await loadData();
  const c = data.counts || { dev: {}, holdout: {} };
  showToast(`Exported! Dev: ${c.dev.PASS||0}P/${c.dev.FAIL||0}F | Holdout: ${c.holdout.PASS||0}P/${c.holdout.FAIL||0}F`);
}

document.addEventListener("keydown", (e) => {
  if (document.getElementById("lightbox").classList.contains("open")) {
    if (e.key === "Escape") closeLightbox();
    return;
  }
  if (e.target && (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT")) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      STATE.mode === "shots" ? saveCurrentAndNext() : saveSheetAndNext();
    }
    return;
  }
  const k = e.key.toLowerCase();
  if (STATE.mode === "audit") {
    if (k === "arrowleft") { e.preventDefault(); stepAudit(-1); }
    else if (k === "arrowright") { e.preventDefault(); stepAudit(1); }
    else if (e.code === "Space" || k === " ") { e.preventDefault(); toggleAuditCorrect(); }
    return;
  }
  if (k === "arrowleft") { e.preventDefault(); step(-1); }
  else if (k === "arrowright") { e.preventDefault(); step(1); }
  else if (k === "z") { e.preventDefault(); undoLast(); }
  else if (STATE.mode === "shots") {
    if (k === "a") { e.preventDefault(); approveAI(); }
    else if (k === "p") { e.preventDefault(); quickSave("PASS"); }
    else if (k === "f") { e.preventDefault(); selectFail(); }
    else if (k === "s") { e.preventDefault(); quickSave("SKIP"); }
    else if (k === "enter") { e.preventDefault(); saveCurrentAndNext(); }
    else {
      const d = DEFECT_TYPES.find(x => x.key === k);
      if (d) { e.preventDefault(); toggleDefect(d.id); }
    }
  } else {
    if (k === "0") { e.preventDefault(); clearOutliers(); }
    else if (k === "1" || k === "2" || k === "3") { e.preventDefault(); toggleOutlier(Number(k)); }
    else if (k === "enter") { e.preventDefault(); saveSheetAndNext(); }
  }
});

loadData();
</script>
</body>
</html>
"""


def make_handler(data_dir: Path) -> type[BaseHTTPRequestHandler]:
    class TriageHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            # Keep terminal quiet unless error
            return

        def _send_json(self, obj: Any, status: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path in ("/", "/index.html"):
                body = HTML_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/api/data":
                self._send_json(
                    {
                        "shots": _load_shots(data_dir),
                        "sheets": _load_sheets(data_dir),
                    }
                )
                return

            if path.startswith("/img_single/"):
                rel = urllib.parse.unquote(path[len("/img_single/") :])
                resolved = _resolve_safe_image(rel)
                if resolved is None:
                    self.send_error(HTTPStatus.NOT_FOUND, f"Image not found: {rel}")
                    return
                raw = _crop_single_pose_bytes(resolved)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(raw)
                return

            if path.startswith("/img_focus/"):
                sid = urllib.parse.unquote(path[len("/img_focus/") :]).strip("/")
                raw = _auto_focus_crop_bytes(data_dir, sid)
                if raw is None:
                    self.send_error(HTTPStatus.NOT_FOUND, f"Shot not found: {sid}")
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(raw)
                return

            if path.startswith("/img/"):
                rel = urllib.parse.unquote(path[len("/img/") :])
                resolved = _resolve_safe_image(rel)
                if resolved is None:
                    self.send_error(HTTPStatus.NOT_FOUND, f"Image not found: {rel}")
                    return
                ctype = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
                raw = resolved.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(raw)
                return

            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"

            if parsed.path == "/api/save":
                try:
                    body = json.loads(raw.decode("utf-8"))
                    mode = body.get("mode", "shots")
                    item_id = str(body["id"])
                    triage = dict(body["triage"])
                    subdir = "shots" if mode == "shots" else "contact_sheets"
                    jf = data_dir / subdir / f"{item_id}.json"
                    if not jf.exists():
                        self._send_json({"error": f"Missing sidecar: {jf.name}"}, status=404)
                        return
                    rec = json.loads(jf.read_text(encoding="utf-8"))
                    rec["triage"] = triage
                    rec["triage"]["labeled_at"] = datetime.now(timezone.utc).isoformat()
                    _atomic_write_json(jf, rec)
                    self._send_json({"ok": True, "id": item_id})
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=400)
                return

            if parsed.path == "/api/audit_toggle":
                try:
                    body = json.loads(raw.decode("utf-8"))
                    item_id = str(body["id"])
                    status = str(body.get("audit_status", "correct"))
                    jf = data_dir / "shots" / f"{item_id}.json"
                    if not jf.exists():
                        self._send_json({"error": f"Missing sidecar: {jf.name}"}, status=404)
                        return
                    rec = json.loads(jf.read_text(encoding="utf-8"))
                    tr = rec.setdefault("triage", {})
                    tr["audit_status"] = status
                    tr["needs_rerate"] = status == "wrong"
                    if status == "wrong":
                        tr["human_label"] = None
                    tr["audited_at"] = datetime.now(timezone.utc).isoformat()
                    _atomic_write_json(jf, rec)
                    self._send_json({"ok": True, "id": item_id, "audit_status": status})
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=400)
                return

            if parsed.path == "/api/audit_commit_group":
                try:
                    body = json.loads(raw.decode("utf-8"))
                    ids = [str(x) for x in (body.get("ids") or [])]
                    approved = 0
                    flagged_red = 0
                    now = datetime.now(timezone.utc).isoformat()
                    for item_id in ids:
                        jf = data_dir / "shots" / f"{item_id}.json"
                        if not jf.exists():
                            continue
                        rec = json.loads(jf.read_text(encoding="utf-8"))
                        tr = rec.setdefault("triage", {})
                        ai = rec.get("ai_suggestion") or {}
                        if tr.get("audit_status") == "wrong" or tr.get("needs_rerate") is True:
                            tr["needs_rerate"] = True
                            tr["human_label"] = None
                            flagged_red += 1
                        else:
                            if not tr.get("human_label") and ai.get("suggested_label"):
                                tr["human_label"] = ai["suggested_label"]
                                tr["defect_types"] = ai.get("suggested_defects") or []
                                tr["severity"] = ai.get("suggested_severity")
                                if not tr.get("notes"):
                                    tr["notes"] = ai.get("reason") or ""
                            tr["audit_status"] = "correct"
                            tr["needs_rerate"] = False
                            tr["labeled_at"] = now
                            approved += 1
                        _atomic_write_json(jf, rec)
                    self._send_json({"ok": True, "approved": approved, "flagged_red": flagged_red})
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=400)
                return

            if parsed.path == "/api/split":
                try:
                    manifest = _assign_dev_holdout_split(data_dir)
                    self._send_json(manifest)
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=500)
                return

            self.send_error(HTTPStatus.NOT_FOUND)

    return TriageHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Local click-through triage server for mined eval data.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.data_dir))
    url = f"http://{args.host}:{args.port}/"
    print(f"[triage] Serving {args.data_dir} at {url} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[triage] Stopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
