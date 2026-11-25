"""Build the Class of 2025 Meme Yearbook HTML demo from data/roster.json + portraits/final/.

Usage:
    uv run python scripts/build_yearbook.py
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

from styles import build_image_prompt

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROSTER = ROOT / "data" / "roster.json"
FINAL_DIR = ROOT / "portraits" / "final"
RAW_DIR = ROOT / "portraits" / "raw"
SITE_OUT = ROOT / "index.html"


def esc(val: str) -> str:
    return html.escape(str(val or ""), quote=True)


def portrait_rel_src(student_id: str) -> str | None:
    final_path = FINAL_DIR / f"{student_id}.png"
    if final_path.exists():
        return f"portraits/final/{student_id}.png"
    raw_path = RAW_DIR / f"{student_id}_v0.png"
    if raw_path.exists():
        return f"portraits/raw/{student_id}_v0.png"
    return None


def render_portrait_card(student: dict, index: int) -> str:
    src = portrait_rel_src(student["id"])
    clubs_str = " • ".join(student.get("clubs", []))
    if src:
        media_html = f'<img src="{esc(src)}" alt="{esc(student["name"])}" loading="lazy">'
    else:
        media_html = f"""
        <div class="makeup-placeholder">
          <span class="makeup-badge">MAKE-UP DAY</span>
          <span class="makeup-style">{esc(student["style_label"])}</span>
          <span class="makeup-hint">Run generate_portraits.py</span>
        </div>"""

    return f"""
    <article class="senior-card" data-index="{index}" data-style="{esc(student['style_id'])}" onclick="openModal({index})">
      <div class="photo-frame">
        {media_html}
        <span class="debug-badge">{esc(student['style_label'])}</span>
      </div>
      <div class="senior-meta">
        <h3 class="senior-name">{esc(student['name'])}</h3>
        <p class="senior-quote">“{esc(student['quote'])}”</p>
        <p class="senior-clubs">{esc(clubs_str)}</p>
      </div>
    </article>"""


def build_html(roster: dict) -> str:
    school = roster["school"]
    students = roster["students"]

    # Chunk into 6 per page (12 per 2-page spread -> clean authentic yearbook density; 20 seniors = 4 pages / 2 spreads)
    per_page = 5
    pages: list[list[tuple[int, dict]]] = []
    indexed = list(enumerate(students))
    for i in range(0, len(indexed), per_page):
        pages.append(indexed[i : i + per_page])
    # Ensure even number of senior pages for 2-page spreads
    if len(pages) % 2 == 1:
        pages.append([])

    # Build senior 2-page spreads
    spreads_html: list[str] = []
    page_num = 4
    for s_idx in range(0, len(pages), 2):
        left_items = pages[s_idx]
        right_items = pages[s_idx + 1]

        def render_page(items: list[tuple[int, dict]], pnum: int, side: str) -> str:
            cards = "".join(render_portrait_card(st, idx) for idx, st in items)
            if items:
                first_last = items[0][1]["name"].split()[-1][0].upper()
                end_last = items[-1][1]["name"].split()[-1][0].upper()
                alpha_range = f"{first_last}–{end_last}" if first_last != end_last else first_last
            else:
                alpha_range = "NOTES"
            body = (
                f'<div class="senior-grid">{cards}</div>'
                if items
                else """
                <div class="autographs-box">
                  <h3>Autographs &amp; Doodles</h3>
                  <p>Have a great summer! Stay weird. Don't change.</p>
                </div>"""
            )
            return f"""
            <section class="book-page page-{side}">
              <header class="page-header">
                <span class="running-section">SENIORS • CLASS OF {esc(school['class_year'])}</span>
                <span class="running-alpha">{esc(alpha_range)}</span>
              </header>
              {body}
              <footer class="page-footer">
                <span class="school-foot">{esc(school['name'].upper())}</span>
                <span class="page-number">{pnum:02d}</span>
              </footer>
            </section>"""

        spreads_html.append(
            f"""
            <div class="spread" data-spread-label="Seniors ({page_num:02d}–{page_num + 1:02d})">
              {render_page(left_items, page_num, 'left')}
              <div class="gutter"></div>
              {render_page(right_items, page_num + 1, 'right')}
            </div>"""
        )
        page_num += 2

    # Build Superlatives showcase (pick up to 6 students with portraits first)
    sup_candidates = sorted(
        indexed,
        key=lambda pair: (portrait_rel_src(pair[1]["id"]) is None, pair[0]),
    )[:6]
    sup_cards = []
    for idx, st in sup_candidates:
        src = portrait_rel_src(st["id"])
        img_tag = (
            f'<img src="{esc(src)}" alt="{esc(st["name"])}">'
            if src
            else '<div class="makeup-placeholder small"><span class="makeup-badge">TBA</span></div>'
        )
        sup_cards.append(
            f"""
            <div class="sup-card" onclick="openModal({idx})">
              <div class="sup-ribbon">{esc(st.get('superlative') or 'Senior Standout')}</div>
              <div class="sup-photo">{img_tag}</div>
              <div class="sup-name">{esc(st['name'])}</div>
              <div class="sup-archetype">{esc(st['archetype'])}</div>
            </div>"""
        )

    # Index list sorted alphabetically
    index_rows = "".join(
        f'<li><span class="idx-name">{esc(st["name"])}</span><span class="idx-dots"></span><span class="idx-pg">{4 + (i // per_page):02d}</span></li>'
        for i, st in enumerate(students)
    )

    generated_count = sum(1 for st in students if portrait_rel_src(st["id"]) is not None)
    students_json = json.dumps(
        [
            {
                **st,
                "prompt": st.get("prompt") or build_image_prompt(st["style_id"], st["visual_description"]),
                "portrait_src": portrait_rel_src(st["id"]),
            }
            for st in students
        ]
    ).replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{esc(school['name'])} — Class of {esc(school['class_year'])} Yearbook</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,600;0,700;1,500&family=Inter:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,600;0,800;1,600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --navy: #14213d;
      --navy-deep: #0b1326;
      --gold: #c9a227;
      --gold-soft: #e6cf84;
      --paper: #f8f5ee;
      --paper-edge: #e9e2d2;
      --ink: #1e222b;
      --muted: #5f6675;
      --desk: #272320;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
      background:
        radial-gradient(circle at 20% 10%, #3a332d 0%, #221e1b 55%, #171412 100%);
      color: var(--ink);
      min-height: 100vh;
      padding: 76px 24px 64px;
    }}

    /* Floating Studio Toolbar (hidden in print) */
    .toolbar {{
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      height: 56px;
      background: rgba(15, 21, 36, 0.92);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid rgba(201, 162, 39, 0.25);
      color: #f3f1eb;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      z-index: 100;
      gap: 12px;
    }}

    .toolbar-brand {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 600;
      font-size: 13px;
      letter-spacing: 0.04em;
    }}

    .toolbar-brand .pill {{
      background: rgba(201, 162, 39, 0.18);
      color: var(--gold-soft);
      border: 1px solid rgba(201, 162, 39, 0.4);
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
    }}

    .toolbar-actions {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}

    .tb-btn {{
      background: rgba(255,255,255,0.08);
      color: #f4f2ec;
      border: 1px solid rgba(255,255,255,0.16);
      border-radius: 6px;
      padding: 6px 11px;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .tb-btn:hover {{ background: rgba(255,255,255,0.16); }}
    .tb-btn.active {{
      background: var(--gold);
      color: var(--navy-deep);
      border-color: var(--gold);
      font-weight: 600;
    }}

    /* Book Container */
    .yearbook-stack {{
      max-width: 1220px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 44px;
    }}

    /* Hardcover Cover */
    .cover-wrap {{
      display: flex;
      justify-content: center;
    }}

    .hardcover {{
      width: 560px;
      max-width: 100%;
      aspect-ratio: 8.5 / 11;
      background:
        repeating-linear-gradient(45deg, rgba(255,255,255,0.015) 0 2px, transparent 2px 4px),
        linear-gradient(135deg, #1a2b50 0%, #101b33 60%, #091020 100%);
      border-radius: 4px 12px 12px 4px;
      box-shadow:
        -10px 0 0 #0a1020,
        0 28px 65px rgba(0, 0, 0, 0.65),
        inset 14px 0 24px rgba(0, 0, 0, 0.45),
        inset 0 0 0 1px rgba(230, 207, 132, 0.25);
      padding: 36px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      align-items: center;
      text-align: center;
      color: var(--gold-soft);
      position: relative;
    }}

    .hardcover::before {{
      content: "";
      position: absolute;
      inset: 18px 18px 18px 26px;
      border: 2px solid rgba(201, 162, 39, 0.55);
      pointer-events: none;
    }}
    .hardcover::after {{
      content: "";
      position: absolute;
      inset: 24px 24px 24px 32px;
      border: 1px solid rgba(201, 162, 39, 0.28);
      pointer-events: none;
    }}

    .cover-kicker {{
      font-size: 11px;
      letter-spacing: 0.28em;
      text-transform: uppercase;
      color: rgba(230, 207, 132, 0.8);
      margin-top: 16px;
    }}

    .crest {{
      width: 96px;
      height: 110px;
      border: 2px solid var(--gold);
      border-radius: 8px 8px 48px 48px;
      display: grid;
      place-items: center;
      background: rgba(201, 162, 39, 0.08);
      box-shadow: inset 0 0 18px rgba(201, 162, 39, 0.2);
      margin: 12px auto;
    }}
    .crest span {{
      font-family: "Playfair Display", Georgia, serif;
      font-size: 32px;
      font-weight: 800;
      letter-spacing: 0.04em;
      color: var(--gold);
    }}

    .cover-title {{
      font-family: "Playfair Display", Georgia, serif;
      font-size: 40px;
      line-height: 1.08;
      margin: 8px 0 6px;
      color: #f3dfb0;
      text-shadow: 0 2px 12px rgba(0, 0, 0, 0.45);
    }}

    .cover-year {{
      font-family: "Cormorant Garamond", Georgia, serif;
      font-size: 62px;
      font-weight: 700;
      letter-spacing: 0.14em;
      color: var(--gold);
      margin: 10px 0;
      line-height: 1;
    }}

    .cover-motto {{
      font-family: "Cormorant Garamond", Georgia, serif;
      font-style: italic;
      font-size: 20px;
      color: rgba(243, 223, 176, 0.88);
    }}

    .cover-edition {{
      font-size: 11px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      opacity: 0.72;
      margin-bottom: 12px;
    }}

    /* Open 2-Page Spread */
    .spread {{
      display: grid;
      grid-template-columns: 1fr 2px 1fr;
      background: var(--paper);
      border-radius: 6px;
      box-shadow:
        0 26px 60px rgba(0, 0, 0, 0.55),
        0 2px 0 #d6cebc,
        0 5px 0 #c6bda8;
      overflow: hidden;
      position: relative;
    }}

    .gutter {{
      background: linear-gradient(
        to right,
        rgba(0,0,0,0.16),
        rgba(0,0,0,0.28),
        rgba(0,0,0,0.12)
      );
    }}

    .book-page {{
      padding: 28px 30px 22px;
      min-height: 760px;
      display: flex;
      flex-direction: column;
      background: var(--paper);
      position: relative;
    }}

    .page-left {{
      box-shadow: inset -22px 0 26px -22px rgba(25, 20, 12, 0.32);
    }}
    .page-right {{
      box-shadow: inset 22px 0 26px -22px rgba(25, 20, 12, 0.32);
    }}

    .page-header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      border-bottom: 2px solid var(--navy);
      padding-bottom: 8px;
      margin-bottom: 18px;
    }}

    .running-section {{
      font-size: 10.5px;
      font-weight: 700;
      letter-spacing: 0.2em;
      color: var(--navy);
    }}

    .running-alpha {{
      font-family: "Playfair Display", Georgia, serif;
      font-size: 18px;
      font-weight: 800;
      color: var(--gold);
    }}

    .page-footer {{
      margin-top: auto;
      padding-top: 14px;
      border-top: 1px solid var(--paper-edge);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 10px;
      letter-spacing: 0.16em;
      color: var(--muted);
    }}

    .page-left .page-footer {{
      flex-direction: row-reverse;
    }}

    .page-number {{
      font-family: "Playfair Display", Georgia, serif;
      font-size: 13px;
      font-weight: 700;
      color: var(--navy);
    }}

    /* Senior Portrait Grid (5 per page: 2 featured on top, 3 below — classic senior layout!) */
    .senior-grid {{
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 18px 14px;
      align-content: start;
    }}

    .senior-card {{
      grid-column: span 2;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      transition: transform 0.14s ease;
    }}

    /* Top two portraits on each senior page sit slightly wider like classic senior panels */
    .senior-card:nth-child(1),
    .senior-card:nth-child(2) {{
      grid-column: span 3;
    }}

    .senior-card:hover {{
      transform: translateY(-2px);
    }}

    .photo-frame {{
      width: 100%;
      aspect-ratio: 4 / 5;
      background: linear-gradient(160deg, #5d7491 0%, #3d5169 100%);
      border: 3px solid #ffffff;
      box-shadow:
        0 0 0 1px rgba(20, 33, 61, 0.22),
        0 4px 10px rgba(20, 24, 34, 0.16);
      overflow: hidden;
      position: relative;
    }}

    .photo-frame img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}

    .senior-card[data-style="pixel_16bit"] .photo-frame img {{
      image-rendering: pixelated;
    }}

    .debug-badge {{
      display: none;
      position: absolute;
      bottom: 6px;
      left: 6px;
      background: rgba(11, 19, 38, 0.85);
      color: var(--gold-soft);
      font-size: 9.5px;
      font-weight: 600;
      padding: 2px 6px;
      border-radius: 3px;
      letter-spacing: 0.04em;
    }}
    body.show-debug .debug-badge {{
      display: inline-block;
    }}

    .makeup-placeholder {{
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 6px;
      color: rgba(255,255,255,0.88);
      text-align: center;
      padding: 12px;
      background:
        radial-gradient(circle at 50% 35%, rgba(255,255,255,0.14), transparent 60%),
        linear-gradient(160deg, #5c738f, #394b63);
    }}
    .makeup-badge {{
      border: 1px solid rgba(255,255,255,0.5);
      padding: 2px 8px;
      font-size: 9px;
      letter-spacing: 0.15em;
      font-weight: 700;
    }}
    .makeup-style {{
      font-family: "Playfair Display", Georgia, serif;
      font-size: 13px;
    }}
    .makeup-hint {{
      font-size: 10px;
      opacity: 0.7;
    }}

    .senior-meta {{
      padding-top: 7px;
      text-align: center;
    }}

    .senior-name {{
      font-family: "Playfair Display", Georgia, serif;
      font-size: 14.5px;
      font-weight: 700;
      color: var(--navy);
      margin: 0 0 3px;
      line-height: 1.15;
    }}

    .senior-quote {{
      font-family: "Cormorant Garamond", Georgia, serif;
      font-style: italic;
      font-size: 13px;
      line-height: 1.22;
      color: #2b313e;
      margin: 0 0 4px;
    }}

    .senior-clubs {{
      font-size: 9.5px;
      letter-spacing: 0.03em;
      color: var(--muted);
      margin: 0;
      line-height: 1.3;
    }}

    /* Opening Spread (Principal Letter + Class Overview) */
    .editorial-title {{
      font-family: "Playfair Display", Georgia, serif;
      font-size: 30px;
      color: var(--navy);
      margin: 4px 0 12px;
      line-height: 1.1;
    }}
    .gold-rule {{
      width: 56px;
      height: 3px;
      background: var(--gold);
      margin-bottom: 18px;
    }}
    .letter-body {{
      font-family: "Cormorant Garamond", Georgia, serif;
      font-size: 18px;
      line-height: 1.55;
      color: #262b36;
    }}
    .letter-body p {{ margin: 0 0 12px; }}
    .signature {{
      margin-top: 22px;
      font-family: "Playfair Display", Georgia, serif;
      font-style: italic;
      font-size: 20px;
      color: var(--navy);
    }}
    .signature small {{
      display: block;
      font-family: "Inter", sans-serif;
      font-style: normal;
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-top: 4px;
    }}

    .stat-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 18px;
    }}
    .stat-card {{
      border: 1px solid var(--paper-edge);
      background: rgba(255,255,255,0.6);
      padding: 14px;
      border-radius: 4px;
    }}
    .stat-num {{
      font-family: "Playfair Display", Georgia, serif;
      font-size: 28px;
      font-weight: 800;
      color: var(--navy);
      line-height: 1;
    }}
    .stat-label {{
      font-size: 10.5px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
      margin-top: 4px;
    }}

    /* Superlatives & Index Back Spread */
    .sup-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 14px;
    }}
    .sup-card {{
      background: #fff;
      border: 1px solid var(--paper-edge);
      padding: 10px;
      text-align: center;
      cursor: pointer;
      box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }}
    .sup-ribbon {{
      background: var(--navy);
      color: var(--gold-soft);
      font-size: 9.5px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      padding: 4px 6px;
      margin-bottom: 8px;
    }}
    .sup-photo {{
      width: 100%;
      aspect-ratio: 4 / 5;
      max-height: 150px;
      overflow: hidden;
      margin-bottom: 6px;
    }}
    .sup-photo img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
    }}
    .sup-name {{
      font-family: "Playfair Display", Georgia, serif;
      font-size: 13.5px;
      font-weight: 700;
      color: var(--navy);
    }}
    .sup-archetype {{
      font-size: 10.5px;
      color: var(--muted);
      margin-top: 2px;
    }}

    .index-list {{
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      grid-template-columns: 1fr;
      gap: 6px;
      font-size: 12.5px;
    }}
    .index-list li {{
      display: flex;
      align-items: baseline;
      gap: 6px;
    }}
    .idx-name {{ font-weight: 600; color: var(--navy); }}
    .idx-dots {{ flex: 1; border-bottom: 1px dotted #bdb4a2; }}
    .idx-pg {{ font-family: "Playfair Display", Georgia, serif; font-weight: 700; color: var(--muted); }}

    .sponsor-ad {{
      margin-top: 24px;
      border: 2px dashed var(--navy);
      padding: 14px 16px;
      background: rgba(201, 162, 39, 0.08);
      text-align: center;
    }}
    .sponsor-ad h4 {{
      font-family: "Playfair Display", Georgia, serif;
      margin: 0 0 4px;
      font-size: 16px;
      color: var(--navy);
    }}
    .sponsor-ad p {{
      margin: 0;
      font-size: 11.5px;
      color: var(--muted);
    }}

    /* Flat Gallery View Mode */
    body.view-gallery .cover-wrap,
    body.view-gallery .spread-opening,
    body.view-gallery .spread-back {{
      display: none;
    }}
    body.view-gallery .spread {{
      display: block;
      background: transparent;
      box-shadow: none;
    }}
    body.view-gallery .gutter,
    body.view-gallery .page-header,
    body.view-gallery .page-footer {{
      display: none;
    }}
    body.view-gallery .book-page {{
      background: transparent;
      box-shadow: none;
      min-height: 0;
      padding: 0;
    }}
    body.view-gallery .senior-grid {{
      grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
      gap: 20px;
      margin-bottom: 20px;
    }}
    body.view-gallery .senior-card,
    body.view-gallery .senior-card:nth-child(1),
    body.view-gallery .senior-card:nth-child(2) {{
      grid-column: span 1;
      background: var(--paper);
      padding: 12px;
      border-radius: 6px;
      box-shadow: 0 12px 26px rgba(0,0,0,0.35);
    }}

    /* Modal Lightbox */
    .modal-backdrop {{
      position: fixed;
      inset: 0;
      background: rgba(8, 12, 22, 0.8);
      backdrop-filter: blur(6px);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
      z-index: 200;
    }}
    .modal-backdrop.open {{ display: flex; }}
    .modal-card {{
      background: var(--paper);
      max-width: 860px;
      width: 100%;
      border-radius: 8px;
      overflow: hidden;
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr);
      box-shadow: 0 30px 80px rgba(0,0,0,0.65);
      border: 1px solid rgba(201,162,39,0.4);
    }}
    .modal-photo {{
      background: #2c3c52;
      width: 100%;
      height: 100%;
      min-height: 425px;
      overflow: hidden;
    }}
    .modal-photo img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      object-position: center top;
      display: block;
    }}
    .modal-body {{
      padding: 24px 28px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      position: relative;
      min-width: 0;
      background: var(--paper);
    }}
    .modal-close {{
      position: absolute;
      top: 14px;
      right: 16px;
      background: none;
      border: none;
      font-size: 24px;
      cursor: pointer;
      color: var(--muted);
    }}
    .modal-style-tag {{
      align-self: flex-start;
      background: var(--navy);
      color: var(--gold-soft);
      font-size: 10.5px;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      padding: 3px 9px;
      border-radius: 3px;
    }}
    .modal-name {{
      font-family: "Playfair Display", Georgia, serif;
      font-size: 28px;
      margin: 0;
      color: var(--navy);
    }}
    .modal-handle {{ font-size: 13px; color: var(--muted); margin-top: -6px; }}
    .modal-quote {{
      font-family: "Cormorant Garamond", Georgia, serif;
      font-style: italic;
      font-size: 20px;
      color: var(--ink);
      border-left: 3px solid var(--gold);
      padding-left: 12px;
      margin: 4px 0;
    }}
    .modal-kv {{ font-size: 13px; line-height: 1.45; }}
    .modal-kv strong {{ color: var(--navy); }}
    .modal-prompt {{
      margin-top: auto;
      background: #ece7db;
      border: 1px solid #d7cfbe;
      padding: 10px;
      border-radius: 4px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px;
      line-height: 1.45;
      color: #363c48;
      max-height: 132px;
      overflow-y: auto;
    }}

    @media (max-width: 900px) {{
      .spread {{ grid-template-columns: 1fr; }}
      .gutter {{ display: none; }}
      .modal-card {{ grid-template-columns: 1fr; max-height: 90vh; overflow-y: auto; }}
      .modal-photo {{ aspect-ratio: 4 / 5; min-height: 0; height: auto; }}
    }}

    /* Print-Ready PDF Export */
    @media print {{
      @page {{ size: letter portrait; margin: 0.35in; }}
      body {{ background: #fff; padding: 0; }}
      .toolbar, .modal-backdrop, .debug-badge {{ display: none !important; }}
      .yearbook-stack {{ gap: 0; max-width: none; }}
      .cover-wrap {{ page-break-after: always; }}
      .hardcover {{ width: 100%; box-shadow: none; }}
      .spread {{ display: block; box-shadow: none; background: #fff; }}
      .gutter {{ display: none; }}
      .book-page {{
        page-break-after: always;
        box-shadow: none;
        min-height: 9.8in;
        border: 1px solid #e5dfd2;
      }}
    }}
  </style>
</head>
<body>
  <nav class="toolbar">
    <div class="toolbar-brand">
      <span>{esc(school['name'])}</span>
      <span class="pill">Class of {esc(school['class_year'])}</span>
      <span class="pill">{generated_count}/{len(students)} Portraits Rendered</span>
    </div>
    <div class="toolbar-actions">
      <button class="tb-btn active" id="btnBook" onclick="setView('book')">Open Book</button>
      <button class="tb-btn" id="btnGallery" onclick="setView('gallery')">Gallery Grid</button>
      <button class="tb-btn" id="btnDebug" onclick="toggleDebug()">Style Tags</button>
      <button class="tb-btn" onclick="window.print()">Print / Save PDF</button>
    </div>
  </nav>

  <main class="yearbook-stack">
    <!-- HARDCOVER -->
    <div class="cover-wrap">
      <section class="hardcover">
        <div class="cover-kicker">{esc(school['mascot'])}</div>
        <div>
          <div class="crest"><span>MV</span></div>
          <h1 class="cover-title">{esc(school['name'])}</h1>
          <div class="cover-year">{esc(school['class_year'])}</div>
          <div class="cover-motto">“{esc(school['motto'])}”</div>
        </div>
        <div class="cover-edition">{esc(school['edition'])}</div>
      </section>
    </div>

    <!-- OPENING SPREAD -->
    <div class="spread spread-opening">
      <section class="book-page page-left">
        <header class="page-header">
          <span class="running-section">OPENING • FOREWORD</span>
          <span class="running-alpha">2025</span>
        </header>
        <h2 class="editorial-title">A Note From the Principal</h2>
        <div class="gold-rule"></div>
        <div class="letter-body">
          <p>Dear Class of {esc(school['class_year'])},</p>
          <p>
            When you first walked—or clipped through the drywall—into Meme Valley High four years ago,
            nobody knew how a student body made of 16-bit sprites, shaky MS Paint doodles, early-2000s
            CGI renders, and at least two sentient household objects could ever share one cafeteria.
          </p>
          <p>
            Yet here you are: weird, unbothered, and somehow still passing AP Physics. Whether you are
            heading to college, going straight into a group chat reaction folder, or remaining permanently
            at 96×120 resolution, never forget who you were on Picture Day.
          </p>
          <p>Congratulations, Seniors. Remember to save your work.</p>
          <div class="signature">
            Dr. Arthur Raster
            <small>Principal, {esc(school['name'])}</small>
          </div>
        </div>
        <footer class="page-footer">
          <span class="school-foot">{esc(school['name'].upper())}</span>
          <span class="page-number">02</span>
        </footer>
      </section>
      <div class="gutter"></div>
      <section class="book-page page-right">
        <header class="page-header">
          <span class="running-section">YEAR IN NUMBERS</span>
          <span class="running-alpha">STATS</span>
        </header>
        <h2 class="editorial-title">Class of {esc(school['class_year'])} At A Glance</h2>
        <div class="gold-rule"></div>
        <div class="letter-body">
          <p>
            Every student on the following pages is a 100% original character invented locally and
            photographed on Picture Day under the exact same mottled slate-blue studio canvas.
          </p>
        </div>
        <div class="stat-grid">
          <div class="stat-card">
            <div class="stat-num">{len(students)}</div>
            <div class="stat-label">Original Seniors</div>
          </div>
          <div class="stat-card">
            <div class="stat-num">8</div>
            <div class="stat-label">Mixed-Media Styles</div>
          </div>
          <div class="stat-card">
            <div class="stat-num">4</div>
            <div class="stat-label">16-Bit Pixel Students</div>
          </div>
          <div class="stat-card">
            <div class="stat-num">0</div>
            <div class="stat-label">Recycled Memes</div>
          </div>
        </div>
        <div class="sponsor-ad">
          <h4>Click Any Senior Portrait</h4>
          <p>Inspect their full meme archetype, relatable micro-emotion, clubs, superlative, and generation prompt.</p>
        </div>
        <footer class="page-footer">
          <span class="school-foot">{esc(school['name'].upper())}</span>
          <span class="page-number">03</span>
        </footer>
      </section>
    </div>

    <!-- SENIOR PORTRAIT SPREADS -->
    {''.join(spreads_html)}

    <!-- SUPERLATIVES & INDEX SPREAD -->
    <div class="spread spread-back">
      <section class="book-page page-left">
        <header class="page-header">
          <span class="running-section">SENIOR SUPERLATIVES</span>
          <span class="running-alpha">BEST OF '25</span>
        </header>
        <div class="sup-grid">
          {''.join(sup_cards)}
        </div>
        <footer class="page-footer">
          <span class="school-foot">{esc(school['name'].upper())}</span>
          <span class="page-number">{page_num:02d}</span>
        </footer>
      </section>
      <div class="gutter"></div>
      <section class="book-page page-right">
        <header class="page-header">
          <span class="running-section">SENIOR DIRECTORY</span>
          <span class="running-alpha">INDEX</span>
        </header>
        <ul class="index-list">
          {index_rows}
        </ul>
        <div class="sponsor-ad">
          <h4>Congratulations Class of 2025!</h4>
          <p>Sponsored by Dave’s Low-Poly Auto Body &amp; The Copy Room Toner Fund — Proud Supporters of Meme Valley High.</p>
        </div>
        <footer class="page-footer">
          <span class="school-foot">{esc(school['name'].upper())}</span>
          <span class="page-number">{page_num + 1:02d}</span>
        </footer>
      </section>
    </div>
  </main>

  <!-- STUDENT INSPECTOR MODAL -->
  <div class="modal-backdrop" id="modal" onclick="if(event.target===this) closeModal()">
    <div class="modal-card">
      <div class="modal-photo" id="modalPhoto"></div>
      <div class="modal-body">
        <button class="modal-close" onclick="closeModal()" aria-label="Close">&times;</button>
        <span class="modal-style-tag" id="modalStyle"></span>
        <h2 class="modal-name" id="modalName"></h2>
        <div class="modal-handle" id="modalHandle"></div>
        <blockquote class="modal-quote" id="modalQuote"></blockquote>
        <div class="modal-kv"><strong>Archetype:</strong> <span id="modalArchetype"></span></div>
        <div class="modal-kv"><strong>The Vibe:</strong> <span id="modalEmotion"></span></div>
        <div class="modal-kv"><strong>Superlative:</strong> <span id="modalSuperlative"></span></div>
        <div class="modal-kv"><strong>Activities:</strong> <span id="modalClubs"></span></div>
        <div class="modal-prompt" id="modalPrompt"></div>
      </div>
    </div>
  </div>

  <script>
    const STUDENTS = {students_json};

    function setView(mode) {{
      document.body.classList.toggle('view-gallery', mode === 'gallery');
      document.getElementById('btnBook').classList.toggle('active', mode === 'book');
      document.getElementById('btnGallery').classList.toggle('active', mode === 'gallery');
    }}

    function toggleDebug() {{
      const on = document.body.classList.toggle('show-debug');
      document.getElementById('btnDebug').classList.toggle('active', on);
    }}

    function openModal(idx) {{
      const s = STUDENTS[idx];
      if (!s) return;
      const photoEl = document.getElementById('modalPhoto');
      if (s.portrait_src) {{
        const pxStyle = s.style_id === 'pixel_16bit' ? 'style="image-rendering: pixelated;"' : '';
        photoEl.innerHTML = `<img src="${{s.portrait_src}}" alt="${{s.name}}" ${{pxStyle}}>`;
      }} else {{
        photoEl.innerHTML = `<div class="makeup-placeholder"><span class="makeup-badge">MAKE-UP DAY</span><span class="makeup-style">${{s.style_label}}</span></div>`;
      }}
      document.getElementById('modalStyle').textContent = s.style_label;
      document.getElementById('modalName').textContent = s.name;
      document.getElementById('modalHandle').textContent = `${{s.handle}} • Seed ${{s.seed}}`;
      document.getElementById('modalQuote').textContent = `“${{s.quote}}”`;
      document.getElementById('modalArchetype').textContent = s.archetype;
      document.getElementById('modalEmotion').textContent = s.micro_emotion;
      document.getElementById('modalSuperlative').textContent = s.superlative || '—';
      document.getElementById('modalClubs').textContent = (s.clubs || []).join(' • ') || '—';
      document.getElementById('modalPrompt').textContent = s.prompt;
      document.getElementById('modal').classList.add('open');
    }}

    function closeModal() {{
      document.getElementById('modal').classList.remove('open');
    }}

    document.addEventListener('keydown', (e) => {{
      if (e.key === 'Escape') closeModal();
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Class of 2025 HTML Yearbook.")
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--output", type=Path, default=SITE_OUT)
    args = parser.parse_args()

    if not args.roster.exists():
        print(f"Error: roster not found at {args.roster}", file=sys.stderr)
        sys.exit(1)

    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(roster), encoding="utf-8")
    print(f"Built yearbook demo -> {args.output}")


if __name__ == "__main__":
    main()
