"""Step 3: Reusable Little Golden Book HTML template renderer + Series Bookshelf generator.

Builds:
  - books/<slug>/index.html  (browser-flippable hardboard book with gold-foil spine,
                              bookplate, cast yearbook portraits, 3D page-flip, Director Mode, Print CSS)
  - index.html               (vintage bookshelf index listing every book in books/)

Usage:
    uv run python scripts/build_book.py books/01-my-book
    uv run python scripts/build_book.py --all
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path

from roster import FEVER_DREAMS_ROOT, resolve_project_path

BOOKS_DIR = FEVER_DREAMS_ROOT / "books"


def esc(val: object) -> str:
    return html.escape(str(val or ""), quote=True)


def copy_cast_portraits(story: dict, book_dir: Path) -> dict[str, str]:
    """Copy cast yearbook portraits into books/<slug>/cast/ so each book folder is portable."""
    cast_dir = book_dir / "cast"
    cast_dir.mkdir(parents=True, exist_ok=True)
    rel_map: dict[str, str] = {}
    for ch in story.get("cast", []):
        src: Path | None = None
        for key in ("portrait_final", "portrait_thumb", "portrait_raw"):
            resolved = resolve_project_path(ch.get(key))
            if resolved:
                src = resolved
                break
        if src:
            dest = cast_dir / f"{ch['id']}{src.suffix}"
            if not dest.exists() or dest.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dest)
            rel_map[ch["id"]] = f"cast/{dest.name}"
    return rel_map


def collect_series_catalog() -> list[dict]:
    """Read metadata for all books in books/*/story.json sorted by series_number."""
    catalog: list[dict] = []
    if not BOOKS_DIR.exists():
        return catalog
    for story_file in sorted(BOOKS_DIR.glob("*/story.json")):
        try:
            data = json.loads(story_file.read_text(encoding="utf-8"))
            book_dir = story_file.parent
            cover_rel = (data.get("cover") or {}).get("image")
            has_cover = bool(cover_rel and (book_dir / cover_rel).exists())
            catalog.append(
                {
                    "slug": book_dir.name,
                    "series_number": int(data.get("series_number", 0)),
                    "title": data.get("title", book_dir.name),
                    "subtitle": data.get("subtitle", ""),
                    "price": data.get("price", "25¢"),
                    "palette": data.get("cover_palette")
                    or {"bg": "#c9533c", "accent": "#f3d37a", "ink": "#201712"},
                    "cover_image": f"books/{book_dir.name}/{cover_rel}" if has_cover else None,
                    "page_count": len(data.get("pages", [])),
                    "cast_names": [c.get("name", "") for c in data.get("cast", [])],
                }
            )
        except Exception:
            continue
    catalog.sort(key=lambda b: (b["series_number"], b["slug"]))
    return catalog


def render_art_block(
    img_rel: str | None,
    book_dir: Path,
    alt_text: str,
    placeholder_title: str,
    placeholder_hint: str,
    extra_class: str = "",
) -> str:
    if img_rel and (book_dir / img_rel).exists():
        return f'<img class="illus-img {extra_class}" src="{esc(img_rel)}" alt="{esc(alt_text)}" loading="lazy">'
    return f"""
    <div class="art-placeholder {extra_class}">
      <div class="art-ph-badge">ILLUSTRATION PENDING</div>
      <div class="art-ph-title">{esc(placeholder_title)}</div>
      <div class="art-ph-hint">{esc(placeholder_hint)}</div>
    </div>"""


def render_director_drawer(
    label: str,
    story_rel_cmd: str,
    meta: dict | None,
    camera: str,
    prompt: str,
    cast_names: list[str],
    reroll_flags: str,
) -> str:
    meta = meta or {}
    model = meta.get("model", "not rendered yet")
    seed = meta.get("seed", "—")
    elapsed = f"{meta['elapsed_s']}s" if "elapsed_s" in meta else "—"
    refs = ", ".join(meta.get("refs", [])) or "—"
    cmd = f"uv run python scripts/generate_illustrations.py {story_rel_cmd} {reroll_flags} --seed-offset 1 --overwrite"
    return f"""
    <aside class="director-panel" aria-label="Director metadata">
      <div class="dir-head">
        <strong>{esc(label)}</strong>
        <span>{esc(model)} • seed {esc(seed)} • {esc(elapsed)}</span>
      </div>
      <div class="dir-row"><b>Cast:</b> {esc(', '.join(cast_names))}</div>
      <div class="dir-row"><b>Angle:</b> {esc(camera)}</div>
      <div class="dir-row"><b>Refs:</b> {esc(refs)}</div>
      <details class="dir-prompt">
        <summary>Prompt &amp; Re-roll Command</summary>
        <p>{esc(prompt)}</p>
        <code>{esc(cmd)}</code>
      </details>
    </aside>"""


def render_story_page(
    page: dict,
    book_dir: Path,
    cast_by_id: dict[str, dict],
    is_last_page: bool,
    moral: str,
    story_rel_cmd: str,
) -> str:
    pnum = int(page["page_number"])
    layout = page.get("layout", "framed-classic")
    cast_names = [cast_by_id[cid]["name"] for cid in page.get("cast_ids", []) if cid in cast_by_id]
    art_html = render_art_block(
        page.get("image"),
        book_dir,
        alt_text=page.get("scene_description", f"Page {pnum}"),
        placeholder_title=f"Page {pnum:02d} • {page.get('camera_angle', '')}",
        placeholder_hint=page.get("scene_description", "")[:170],
    )
    director_html = render_director_drawer(
        label=f"Page {pnum:02d}",
        story_rel_cmd=story_rel_cmd,
        meta=page.get("render_meta"),
        camera=page.get("camera_angle", ""),
        prompt=page.get("prompt", ""),
        cast_names=cast_names,
        reroll_flags=f"--only-page {pnum}",
    )

    ending_html = ""
    if is_last_page:
        ending_html = f"""
        <div class="the-end-block">
          <div class="the-end-ornament">❦ &nbsp; THE END &nbsp; ❦</div>
          <p class="moral-line"><strong>A Little Lesson:</strong> {esc(moral)}</p>
        </div>"""

    # First letter drop-cap styling on opening page
    text_html = esc(page.get("text", ""))
    drop_cap_class = " has-dropcap" if pnum == 1 else ""

    return f"""
    <div class="page-inner story-layout layout-{esc(layout)}">
      <div class="art-slot">
        <div class="art-mat">
          {art_html}
        </div>
      </div>
      <div class="story-copy">
        <p class="story-text{drop_cap_class}">{text_html}</p>
        {ending_html}
      </div>
      <div class="page-folio">{pnum}</div>
      {director_html}
    </div>"""


def build_book_html(story: dict, book_dir: Path, catalog: list[dict]) -> str:
    cast_rel = copy_cast_portraits(story, book_dir)
    cast_by_id = {c["id"]: c for c in story.get("cast", [])}
    palette = story.get("cover_palette") or {"bg": "#c9533c", "accent": "#f3d37a", "ink": "#201712"}
    series_num = int(story.get("series_number", 1))
    slug = story.get("slug", book_dir.name)
    story_rel_cmd = f"books/{book_dir.name}/story.json"

    # Cover art & director metadata
    cov = story.get("cover") or {}
    cover_cast_names = [cast_by_id[cid]["name"] for cid in cov.get("cast_ids", []) if cid in cast_by_id]
    cover_art = render_art_block(
        cov.get("image"),
        book_dir,
        alt_text=cov.get("scene_description", story["title"]),
        placeholder_title="Front Cover Illustration",
        placeholder_hint=cov.get("scene_description", "")[:180],
        extra_class="cover-art-img",
    )
    cover_director = render_director_drawer(
        label="Front Cover",
        story_rel_cmd=story_rel_cmd,
        meta=cov.get("render_meta"),
        camera=cov.get("camera_angle", ""),
        prompt=cov.get("prompt", ""),
        cast_names=cover_cast_names,
        reroll_flags="--only-cover",
    )

    # Cast strip for Title Page
    cast_cards_html = []
    for ch in story.get("cast", []):
        p_src = cast_rel.get(ch["id"])
        img_tag = f'<img src="{esc(p_src)}" alt="{esc(ch["name"])}">' if p_src else '<div class="cast-ph">?</div>'
        meta_label = f"{ch['title']} • {ch['style_label']}" if ch.get("title") else ch["style_label"]
        cast_cards_html.append(
            f"""
            <div class="cast-chip">
              <div class="cast-avatar">{img_tag}</div>
              <div class="cast-info">
                <div class="cast-name">{esc(ch['name'])}</div>
                <div class="cast-meta">{esc(meta_label)}</div>
                <div class="cast-quote">“{esc(ch.get('quote', ''))}”</div>
              </div>
            </div>"""
        )

    # Build spreads list:
    #   spread 0: closed front cover (single hardboard cover)
    #   spread 1: inside front cover (bookplate) + title/cast page
    #   spread 2..N: story page pairs (2 pages per spread)
    #   last spread: closed back cover (single hardboard back cover)
    spreads: list[dict] = []

    # Spread 0: Front Cover
    front_cover_html = f"""
    <section class="spread spread-cover active" data-spread="0" data-label="Front Cover">
      <div class="hardboard-cover front-cover" onclick="nextSpread(event)">
        <div class="gold-spine spine-left" aria-hidden="true"></div>
        <div class="cover-face">
          <header class="cover-header">
            <div class="golden-emblem">
              <span class="emblem-top">A LITTLE</span>
              <span class="emblem-main">FEVER DREAM</span>
              <span class="emblem-sub">BOOK &bull; No. {series_num:02d}</span>
            </div>
            <div class="price-circle">{esc(story.get('price', '25¢'))}</div>
          </header>

          <div class="cover-title-block">
            <h1 class="cover-title">{esc(story['title'])}</h1>
            <p class="cover-subtitle">{esc(story.get('subtitle', ''))}</p>
          </div>

          <div class="cover-art-frame">
            {cover_art}
          </div>

          <footer class="cover-footer">
            <span>STARRING THE CLASS OF 2025 &bull; MEME VALLEY HIGH SCHOOL</span>
          </footer>
          {cover_director}
        </div>
        <div class="open-hint">Click cover or press → to open</div>
      </div>
    </section>"""
    spreads.append({"label": "Cover", "html": front_cover_html})

    # Spread 1: Inside Endpaper Bookplate + Title/Cast Page
    front_matter_html = f"""
    <section class="spread spread-open" data-spread="1" data-label="Title &amp; Cast">
      <div class="open-book-shell">
        <div class="book-page page-left endpaper-page" onclick="prevSpread(event)">
          <div class="endpaper-pattern">
            <div class="bookplate">
              <div class="bookplate-ribbon">A LITTLE FEVER DREAM BOOK</div>
              <div class="bookplate-heading">This Fever Dream Belongs To</div>
              <div class="bookplate-line" contenteditable="true" spellcheck="false" data-bookplate="{esc(slug)}" onclick="event.stopPropagation()"></div>
              <div class="bookplate-note">(Click line above to write your name)</div>
              <div class="bookplate-seal">★ MEME VALLEY PRESS &bull; EST. 1955 ★</div>
            </div>
          </div>
        </div>
        <div class="book-page page-right title-page" onclick="nextSpread(event)">
          <div class="page-inner title-page-inner">
            <div class="tp-series">A LITTLE FEVER DREAM BOOK &bull; No. {series_num:02d}</div>
            <h2 class="tp-title">{esc(story['title'])}</h2>
            <p class="tp-sub">{esc(story.get('subtitle', ''))}</p>
            <div class="tp-rule">✦ ✦ ✦</div>
            <p class="tp-dedication"><em>{esc(story.get('dedication', ''))}</em></p>
            <div class="cast-section">
              <div class="cast-heading">Starring From The Class of 2025</div>
              <div class="cast-list">
                {''.join(cast_cards_html)}
              </div>
            </div>
            <div class="tp-imprint">
              Written &amp; Illustrated 100% Locally on Apple Silicon<br>
              Meme Valley High School Publishing Co.
            </div>
          </div>
        </div>
      </div>
    </section>"""
    spreads.append({"label": "Title & Cast", "html": front_matter_html})

    # Story page pairs
    pages = list(story.get("pages", []))
    total_pages = len(pages)
    for i in range(0, total_pages, 2):
        left_p = pages[i]
        right_p = pages[i + 1] if i + 1 < total_pages else None
        s_idx = len(spreads)
        left_html = render_story_page(
            left_p,
            book_dir,
            cast_by_id,
            is_last_page=(i == total_pages - 1),
            moral=story.get("moral", ""),
            story_rel_cmd=story_rel_cmd,
        )
        if right_p is not None:
            right_html = render_story_page(
                right_p,
                book_dir,
                cast_by_id,
                is_last_page=(i + 1 == total_pages - 1),
                moral=story.get("moral", ""),
                story_rel_cmd=story_rel_cmd,
            )
            spread_label = f"Pages {left_p['page_number']}–{right_p['page_number']}"
        else:
            right_html = f"""
            <div class="page-inner colophon-page">
              <div class="colophon-card">
                <div class="colophon-star">★</div>
                <h3>Remember, Boys &amp; Girls!</h3>
                <p class="colophon-moral">“{esc(story.get('moral', ''))}”</p>
                <div class="tp-rule">✦</div>
                <p class="colophon-small">
                  Story by local LLM <code>{esc(story.get('generated_by_llm', 'local'))}</code>.<br>
                  Characters cast from <em>Meme Valley High School Class of 2025</em>.
                </p>
              </div>
            </div>"""
            spread_label = f"Page {left_p['page_number']}"

        spread_html = f"""
        <section class="spread spread-open" data-spread="{s_idx}" data-label="{esc(spread_label)}">
          <div class="open-book-shell">
            <div class="book-page page-left" onclick="prevSpread(event)">
              {left_html}
            </div>
            <div class="book-page page-right" onclick="nextSpread(event)">
              {right_html}
            </div>
          </div>
        </section>"""
        spreads.append({"label": spread_label, "html": spread_html})

    # Back Cover spread
    other_books_items = []
    for item in catalog:
        marker = "★" if item["slug"] == book_dir.name else "□"
        link = f"../{esc(item['slug'])}/index.html"
        other_books_items.append(
            f'<li><a href="{link}" onclick="event.stopPropagation()">'
            f'<span class="cb-mark">{marker}</span> '
            f'<strong>No. {item["series_number"]:02d}:</strong> {esc(item["title"])}'
            f'</a></li>'
        )
    if len(other_books_items) < 4:
        placeholders = [
            "Coming Soon: The Great Cafeteria Tray Accord",
            "Coming Soon: Who Touched The Thermostat?",
            "Coming Soon: A Very Reasonable Fire Drill",
        ]
        for ph in placeholders[: max(0, 4 - len(other_books_items))]:
            other_books_items.append(f'<li class="upcoming"><span class="cb-mark">□</span> {esc(ph)}</li>')

    back_idx = len(spreads)
    back_cover_html = f"""
    <section class="spread spread-cover" data-spread="{back_idx}" data-label="Back Cover">
      <div class="hardboard-cover back-cover" onclick="prevSpread(event)">
        <div class="cover-face back-face">
          <div class="back-banner">A LITTLE FEVER DREAM BOOK</div>
          <h2 class="back-title">{esc(story['title'])}</h2>
          <p class="back-blurb">{esc(story.get('back_cover_blurb') or story.get('subtitle', ''))}</p>

          <div class="collect-box">
            <div class="collect-head">COLLECT THE WHOLE SERIES!</div>
            <ul class="collect-list">
              {''.join(other_books_items)}
            </ul>
          </div>

          <div class="back-bottom">
            <div class="back-seal">
              <span>PRINTED LOCALLY</span>
              <strong>100% MLX &amp; LM STUDIO</strong>
            </div>
            <a class="shelf-btn" href="../../index.html" onclick="event.stopPropagation()">← Back to Bookshelf</a>
          </div>
        </div>
        <div class="gold-spine spine-right" aria-hidden="true"></div>
      </div>
    </section>"""
    spreads.append({"label": "Back Cover", "html": back_cover_html})

    # Dots for scrubber
    dots_html = "".join(
        f'<button class="dot{" active" if idx == 0 else ""}" data-goto="{idx}" '
        f'title="{esc(sp["label"])}" onclick="goToSpread({idx})"></button>'
        for idx, sp in enumerate(spreads)
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(story['title'])} — A Little Fever Dream Book No. {series_num:02d}</title>
<style>
  :root {{
    --cover-bg: {esc(palette['bg'])};
    --cover-accent: {esc(palette['accent'])};
    --cover-ink: {esc(palette['ink'])};
    --paper: #f6eedb;
    --paper-edge: #e4d6b7;
    --ink: #241c15;
    --muted: #6c5b48;
    --gold-1: #7d5b10;
    --gold-2: #e7c667;
    --gold-3: #fff2b2;
    --gold-4: #b98b24;
    --desk: #271f19;
  }}

  * {{ box-sizing: border-box; }}

  body {{
    margin: 0;
    min-height: 100vh;
    background:
      radial-gradient(circle at 50% 25%, #3c3026 0%, #241c16 62%, #18120e 100%);
    color: var(--ink);
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, "Times New Roman", serif;
    display: flex;
    flex-direction: column;
    overflow-x: hidden;
  }}

  /* Top Vintage Reader Toolbar */
  .toolbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 10px 20px;
    background: rgba(18, 14, 11, 0.82);
    border-bottom: 1px solid rgba(243, 211, 122, 0.22);
    color: #f3e8d2;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 13px;
    backdrop-filter: blur(8px);
    position: sticky;
    top: 0;
    z-index: 40;
  }}
  .tb-left, .tb-center, .tb-right {{ display: flex; align-items: center; gap: 10px; }}
  .tb-link {{
    color: #f3d37a;
    text-decoration: none;
    font-weight: 600;
    padding: 5px 10px;
    border-radius: 6px;
    border: 1px solid rgba(243, 211, 122, 0.3);
  }}
  .tb-link:hover {{ background: rgba(243, 211, 122, 0.14); }}
  .tb-title {{ font-family: Georgia, serif; font-size: 14px; color: #f6eedb; opacity: 0.9; }}
  .tb-btn {{
    background: #2f261e;
    color: #f6eedb;
    border: 1px solid #574636;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }}
  .tb-btn:hover:not(:disabled) {{ background: #433528; border-color: #f3d37a; }}
  .tb-btn:disabled {{ opacity: 0.35; cursor: default; }}
  .tb-btn.active {{ background: #b98b24; color: #1a1207; border-color: #f3d37a; }}
  .tb-status {{ min-width: 112px; text-align: center; font-variant-numeric: tabular-nums; color: #e2d1b3; }}

  /* Stage */
  .stage {{
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 24px 18px 18px;
    perspective: 2200px;
  }}

  .book-viewport {{
    position: relative;
    width: min(1140px, 96vw);
    display: flex;
    align-items: center;
    justify-content: center;
  }}

  .spread {{
    display: none;
    width: 100%;
    justify-content: center;
    transform-origin: center center;
  }}
  .spread.active {{
    display: flex;
    animation: spreadFade 0.28s ease-out;
  }}
  .spread.flip-next {{ animation: flipInNext 0.42s cubic-bezier(0.22, 0.82, 0.32, 1); }}
  .spread.flip-prev {{ animation: flipInPrev 0.42s cubic-bezier(0.22, 0.82, 0.32, 1); }}

  @keyframes spreadFade {{
    from {{ opacity: 0; transform: scale(0.985); }}
    to {{ opacity: 1; transform: scale(1); }}
  }}
  @keyframes flipInNext {{
    0% {{ opacity: 0.25; transform: rotateY(14deg) scale(0.98); }}
    100% {{ opacity: 1; transform: rotateY(0deg) scale(1); }}
  }}
  @keyframes flipInPrev {{
    0% {{ opacity: 0.25; transform: rotateY(-14deg) scale(0.98); }}
    100% {{ opacity: 1; transform: rotateY(0deg) scale(1); }}
  }}

  /* Iconic Gold-Foil Little Golden Book Tape Spine */
  .gold-spine {{
    width: 56px;
    flex-shrink: 0;
    position: relative;
    background-color: #c89b2c;
    background-image:
      linear-gradient(90deg,
        rgba(85, 56, 6, 0.55) 0%,
        rgba(255, 243, 180, 0.42) 18%,
        rgba(176, 127, 22, 0.25) 45%,
        rgba(255, 248, 200, 0.52) 72%,
        rgba(92, 62, 10, 0.55) 100%
      ),
      url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24'%3E%3Cpath d='M12 1 L23 12 L12 23 L1 12 Z' fill='none' stroke='%235f4108' stroke-opacity='0.34' stroke-width='1.3'/%3E%3Ccircle cx='12' cy='12' r='2.3' fill='%23fff2b2' fill-opacity='0.45'/%3E%3Cpath d='M0 0 L6 6 M24 0 L18 6 M0 24 L6 18 M24 24 L18 18' stroke='%235f4108' stroke-opacity='0.28' stroke-width='1'/%3E%3C/svg%3E");
    background-size: 100% 100%, 24px 24px;
  }}
  .spine-left {{
    border-right: 3px solid rgba(55, 36, 4, 0.55);
    box-shadow: inset -6px 0 10px rgba(45, 28, 3, 0.35), 2px 0 6px rgba(0,0,0,0.22);
  }}
  .spine-right {{
    border-left: 3px solid rgba(55, 36, 4, 0.55);
    box-shadow: inset 6px 0 10px rgba(45, 28, 3, 0.35), -2px 0 6px rgba(0,0,0,0.22);
  }}

  /* Hard Cardboard Front & Back Covers */
  .hardboard-cover {{
    position: relative;
    width: min(560px, 92vw);
    min-height: min(720px, 80vh);
    display: flex;
    border-radius: 6px 14px 14px 6px;
    overflow: hidden;
    background: var(--cover-bg);
    cursor: pointer;
    box-shadow:
      0 26px 55px rgba(0, 0, 0, 0.62),
      0 4px 0 #d9cbb0,
      0 7px 0 #b8a586,
      inset 0 0 0 2px rgba(255, 245, 215, 0.16);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }}
  .back-cover {{ border-radius: 14px 6px 6px 14px; }}
  .hardboard-cover:hover {{ transform: translateY(-2px); }}

  /* Subtle vintage cardboard paper patina */
  .cover-face::before {{
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background:
      radial-gradient(circle at 20% 15%, rgba(255, 248, 224, 0.14), transparent 55%),
      radial-gradient(circle at 85% 90%, rgba(20, 12, 5, 0.22), transparent 60%);
    mix-blend-mode: soft-light;
  }}

  .cover-face {{
    position: relative;
    flex: 1;
    padding: 20px 24px 18px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    color: #fff8e7;
  }}

  .cover-header {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
  }}
  .golden-emblem {{
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    background: #f6eedb;
    color: var(--cover-ink);
    border: 2px solid #2b2016;
    outline: 2px solid var(--cover-accent);
    padding: 5px 11px 4px;
    border-radius: 4px;
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.25);
    line-height: 1.05;
  }}
  .emblem-top {{ font-family: -apple-system, sans-serif; font-size: 9px; letter-spacing: 0.18em; font-weight: 800; }}
  .emblem-main {{ font-size: 13px; font-weight: 900; letter-spacing: 0.06em; color: #8c2c1c; }}
  .emblem-sub {{ font-family: -apple-system, sans-serif; font-size: 9px; letter-spacing: 0.14em; font-weight: 700; opacity: 0.85; }}

  .price-circle {{
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: var(--cover-accent);
    color: var(--cover-ink);
    border: 2px dashed rgba(32, 23, 18, 0.5);
    display: grid;
    place-items: center;
    font-family: -apple-system, sans-serif;
    font-weight: 900;
    font-size: 14px;
    transform: rotate(7deg);
    box-shadow: 0 3px 6px rgba(0, 0, 0, 0.25);
  }}

  .cover-title-block {{ text-align: center; padding: 2px 6px; }}
  .cover-title {{
    margin: 0;
    font-size: clamp(26px, 3.6vw, 36px);
    line-height: 1.06;
    color: #fffaf0;
    text-shadow: 0 2px 0 rgba(26, 17, 10, 0.55);
    letter-spacing: -0.01em;
  }}
  .cover-subtitle {{
    margin: 6px 0 0;
    font-size: 14px;
    font-style: italic;
    color: var(--cover-accent);
    text-shadow: 0 1px 2px rgba(0,0,0,0.45);
  }}

  .cover-art-frame {{
    flex: 1;
    background: #f6eedb;
    padding: 8px;
    border-radius: 8px;
    border: 2px solid rgba(36, 25, 16, 0.45);
    box-shadow: inset 0 1px 4px rgba(0,0,0,0.18), 0 8px 18px rgba(0,0,0,0.28);
    display: flex;
    overflow: hidden;
    min-height: 320px;
  }}
  .cover-art-img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 4px;
    display: block;
  }}

  .cover-footer {{
    text-align: center;
    font-family: -apple-system, sans-serif;
    font-size: 10px;
    letter-spacing: 0.16em;
    font-weight: 700;
    color: rgba(255, 247, 226, 0.88);
    padding-top: 2px;
  }}

  .open-hint {{
    position: absolute;
    right: 16px;
    bottom: 12px;
    background: rgba(20, 14, 10, 0.72);
    color: #f3d37a;
    font-family: -apple-system, sans-serif;
    font-size: 11px;
    padding: 4px 9px;
    border-radius: 999px;
    pointer-events: none;
    opacity: 0.85;
  }}

  /* Open Two-Page Storybook Spread */
  .open-book-shell {{
    width: 100%;
    min-height: min(700px, 78vh);
    display: grid;
    grid-template-columns: 1fr 1fr;
    background: var(--cover-bg);
    padding: 10px 12px;
    border-radius: 10px;
    box-shadow:
      0 28px 60px rgba(0, 0, 0, 0.65),
      inset 0 0 0 2px rgba(243, 211, 122, 0.22);
    position: relative;
  }}

  .book-page {{
    position: relative;
    background: var(--paper);
    color: var(--ink);
    cursor: pointer;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }}
  .page-left {{
    border-radius: 4px 0 0 4px;
    box-shadow:
      inset -24px 0 26px -18px rgba(40, 26, 12, 0.38),
      inset 2px 0 0 rgba(255,255,255,0.5);
    border-right: 1px solid rgba(90, 66, 40, 0.25);
  }}
  .page-right {{
    border-radius: 0 4px 4px 0;
    box-shadow:
      inset 24px 0 26px -18px rgba(40, 26, 12, 0.34),
      inset -2px 0 0 rgba(255,255,255,0.45);
  }}

  /* Subtle antique paper grain */
  .book-page::after {{
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background-image: radial-gradient(rgba(110, 82, 46, 0.06) 1px, transparent 0);
    background-size: 14px 14px;
    opacity: 0.55;
  }}

  .page-inner {{
    position: relative;
    z-index: 1;
    flex: 1;
    padding: 26px 28px 34px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 16px;
  }}

  /* Story Page Layouts */
  .art-slot {{ flex: 1; display: flex; align-items: center; justify-content: center; min-height: 290px; }}
  .art-mat {{
    width: 100%;
    height: 100%;
    max-height: 410px;
    background: #fffdf7;
    padding: 7px;
    border: 1px solid #d2c19f;
    border-radius: 4px;
    box-shadow: 0 4px 12px rgba(44, 30, 16, 0.14);
    display: flex;
    overflow: hidden;
  }}
  .illus-img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    border-radius: 2px;
  }}

  .layout-vignette .art-mat {{
    border-radius: 18px;
    border: 2px solid #c5b08a;
    max-width: 92%;
    margin: 0 auto;
  }}
  .layout-full-bleed-caption .art-mat {{
    padding: 0;
    border: none;
    border-radius: 6px;
    max-height: 440px;
  }}
  .layout-top-art .story-copy {{ padding-top: 4px; }}

  .story-copy {{
    padding: 4px 8px 6px;
  }}
  .story-text {{
    margin: 0;
    font-size: clamp(16px, 1.55vw, 19px);
    line-height: 1.54;
    color: #261d14;
    letter-spacing: 0.005em;
  }}
  .story-text.has-dropcap::first-letter {{
    float: left;
    font-size: 3.1em;
    line-height: 0.84;
    padding-right: 8px;
    padding-top: 4px;
    font-weight: 700;
    color: var(--cover-bg);
  }}

  .the-end-block {{
    margin-top: 14px;
    padding-top: 10px;
    border-top: 1px dashed #c8b490;
    text-align: center;
  }}
  .the-end-ornament {{
    font-size: 13px;
    letter-spacing: 0.22em;
    font-weight: 700;
    color: var(--cover-bg);
  }}
  .moral-line {{
    margin: 6px 0 0;
    font-size: 13.5px;
    font-style: italic;
    color: #4d3e2e;
  }}

  .page-folio {{
    position: absolute;
    bottom: 11px;
    font-size: 12px;
    color: #847056;
    font-variant-numeric: tabular-nums;
  }}
  .page-left .page-folio {{ left: 22px; }}
  .page-right .page-folio {{ right: 22px; }}

  /* Placeholder before mflux renders */
  .art-placeholder {{
    width: 100%;
    min-height: 280px;
    background:
      repeating-linear-gradient(45deg, #f0e4ca, #f0e4ca 12px, #e8d9b8 12px, #e8d9b8 24px);
    border: 2px dashed #b99e70;
    border-radius: 4px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: 8px;
    color: #5b4832;
  }}
  .art-ph-badge {{
    font-family: -apple-system, sans-serif;
    font-size: 10px;
    letter-spacing: 0.16em;
    font-weight: 800;
    background: #5b4832;
    color: #f6eedb;
    padding: 3px 9px;
    border-radius: 999px;
  }}
  .art-ph-title {{ font-weight: 700; font-size: 15px; }}
  .art-ph-hint {{ font-size: 12px; opacity: 0.85; max-width: 340px; line-height: 1.4; }}

  /* Inside Front Cover Endpaper + Bookplate */
  .endpaper-page {{
    background: #efe3c6;
  }}
  .endpaper-pattern {{
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 28px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='56' height='56' viewBox='0 0 56 56'%3E%3Ccircle cx='28' cy='28' r='2' fill='%23b29258' fill-opacity='0.34'/%3E%3Cpath d='M28 10 L31 18 L39 21 L31 24 L28 32 L25 24 L17 21 L25 18 Z' fill='none' stroke='%23b29258' stroke-opacity='0.26' stroke-width='1'/%3E%3C/svg%3E");
  }}
  .bookplate {{
    width: 100%;
    max-width: 360px;
    background: #faf3e3;
    border: 3px double #8a6c37;
    outline: 5px solid #faf3e3;
    box-shadow: 0 10px 24px rgba(48, 32, 12, 0.18);
    padding: 28px 24px;
    text-align: center;
  }}
  .bookplate-ribbon {{
    font-family: -apple-system, sans-serif;
    font-size: 10px;
    letter-spacing: 0.18em;
    font-weight: 800;
    color: var(--cover-bg);
    margin-bottom: 10px;
  }}
  .bookplate-heading {{
    font-size: 21px;
    font-weight: 700;
    color: #2b2016;
    margin-bottom: 18px;
  }}
  .bookplate-line {{
    min-height: 34px;
    border-bottom: 2px solid #5c4932;
    font-family: "Bradley Hand", "Segoe Script", Georgia, cursive;
    font-size: 22px;
    color: #2d4c66;
    padding: 2px 6px;
    outline: none;
  }}
  .bookplate-line:focus {{ background: rgba(243, 211, 122, 0.25); }}
  .bookplate-note {{ font-family: -apple-system, sans-serif; font-size: 11px; color: #8c775c; margin-top: 8px; }}
  .bookplate-seal {{
    margin-top: 22px;
    font-family: -apple-system, sans-serif;
    font-size: 10px;
    letter-spacing: 0.16em;
    color: #7d6439;
    font-weight: 700;
  }}

  /* Title & Cast Page */
  .title-page-inner {{ text-align: center; justify-content: flex-start; gap: 10px; }}
  .tp-series {{
    font-family: -apple-system, sans-serif;
    font-size: 10px;
    letter-spacing: 0.2em;
    font-weight: 800;
    color: var(--cover-bg);
  }}
  .tp-title {{ margin: 4px 0 0; font-size: 28px; line-height: 1.12; color: #241911; }}
  .tp-sub {{ margin: 4px 0 0; font-size: 15px; font-style: italic; color: #5b4935; }}
  .tp-rule {{ color: #b98b24; font-size: 13px; letter-spacing: 0.3em; margin: 4px 0; }}
  .tp-dedication {{ margin: 0 0 8px; font-size: 13.5px; color: #5f4d3b; }}

  .cast-section {{
    text-align: left;
    background: rgba(226, 209, 173, 0.38);
    border: 1px solid #d8c7a3;
    border-radius: 6px;
    padding: 12px 14px;
    flex: 1;
  }}
  .cast-heading {{
    text-align: center;
    font-family: -apple-system, sans-serif;
    font-size: 10px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    font-weight: 800;
    color: #6b553b;
    margin-bottom: 10px;
  }}
  .cast-list {{ display: flex; flex-direction: column; gap: 9px; }}
  .cast-chip {{ display: flex; gap: 10px; align-items: center; }}
  .cast-avatar {{
    width: 48px;
    height: 60px;
    flex-shrink: 0;
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid #b8a37c;
    background: #e9dec4;
  }}
  .cast-avatar img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
  .cast-name {{ font-weight: 700; font-size: 14px; color: #241911; }}
  .cast-meta {{ font-family: -apple-system, sans-serif; font-size: 10.5px; color: var(--cover-bg); font-weight: 700; }}
  .cast-quote {{ font-size: 11.5px; font-style: italic; color: #5b4935; line-height: 1.25; }}
  .tp-imprint {{
    font-family: -apple-system, sans-serif;
    font-size: 10.5px;
    color: #7d6a52;
    line-height: 1.4;
    padding-top: 4px;
  }}

  /* Colophon / Odd Final Page */
  .colophon-page {{ align-items: center; justify-content: center; text-align: center; }}
  .colophon-card {{
    max-width: 340px;
    padding: 26px 22px;
    border: 2px solid #c9b48d;
    border-radius: 8px;
    background: #fbf5e7;
  }}
  .colophon-star {{ font-size: 26px; color: var(--cover-bg); }}
  .colophon-card h3 {{ margin: 6px 0 10px; font-size: 20px; }}
  .colophon-moral {{ font-size: 16px; font-style: italic; line-height: 1.45; }}
  .colophon-small {{ font-family: -apple-system, sans-serif; font-size: 11.5px; color: #746048; line-height: 1.5; }}

  /* Back Cover */
  .back-face {{ justify-content: space-between; }}
  .back-banner {{
    font-family: -apple-system, sans-serif;
    font-size: 11px;
    letter-spacing: 0.2em;
    font-weight: 800;
    color: var(--cover-accent);
    text-align: center;
  }}
  .back-title {{ margin: 4px 0 0; text-align: center; font-size: 24px; color: #fffaf0; }}
  .back-blurb {{
    margin: 4px auto 0;
    max-width: 380px;
    text-align: center;
    font-size: 15px;
    line-height: 1.5;
    color: #fff6df;
  }}
  .collect-box {{
    background: #f6eedb;
    color: #241c15;
    border-radius: 8px;
    padding: 16px 18px;
    border: 2px solid rgba(36, 25, 16, 0.45);
    box-shadow: 0 6px 16px rgba(0,0,0,0.25);
  }}
  .collect-head {{
    font-family: -apple-system, sans-serif;
    font-size: 11px;
    letter-spacing: 0.16em;
    font-weight: 800;
    color: var(--cover-bg);
    text-align: center;
    margin-bottom: 10px;
  }}
  .collect-list {{ list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; font-size: 13.5px; }}
  .collect-list a {{ color: #241c15; text-decoration: none; }}
  .collect-list a:hover {{ text-decoration: underline; color: var(--cover-bg); }}
  .collect-list li.upcoming {{ opacity: 0.6; font-style: italic; }}
  .cb-mark {{ color: var(--cover-bg); font-weight: 700; margin-right: 4px; }}
  .back-bottom {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
  .back-seal {{
    display: flex;
    flex-direction: column;
    font-family: -apple-system, sans-serif;
    font-size: 9.5px;
    letter-spacing: 0.12em;
    color: rgba(255, 246, 223, 0.85);
  }}
  .shelf-btn {{
    background: var(--cover-accent);
    color: var(--cover-ink);
    text-decoration: none;
    font-family: -apple-system, sans-serif;
    font-size: 12px;
    font-weight: 800;
    padding: 8px 14px;
    border-radius: 6px;
    box-shadow: 0 3px 8px rgba(0,0,0,0.25);
  }}

  /* Scrubber Dots */
  .scrubber {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin-top: 16px;
  }}
  .dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    border: 1px solid rgba(243, 211, 122, 0.55);
    background: rgba(243, 211, 122, 0.18);
    cursor: pointer;
    padding: 0;
    transition: transform 0.15s, background 0.15s;
  }}
  .dot.active {{ background: #f3d37a; transform: scale(1.28); }}

  /* Director / Tinkering Mode Overlay */
  .director-panel {{
    display: none;
    margin-top: 8px;
    background: #1d1814;
    color: #efe3c8;
    border: 1px solid #6b5535;
    border-radius: 6px;
    padding: 8px 10px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 11px;
    line-height: 1.4;
    text-align: left;
  }}
  body.director-mode .director-panel {{ display: block; }}
  .dir-head {{ display: flex; justify-content: space-between; gap: 8px; color: #f3d37a; margin-bottom: 4px; }}
  .dir-row {{ opacity: 0.9; }}
  .dir-prompt {{ margin-top: 5px; }}
  .dir-prompt summary {{ cursor: pointer; color: #f3d37a; font-weight: 600; }}
  .dir-prompt p {{ margin: 6px 0; font-size: 10.5px; opacity: 0.85; }}
  .dir-prompt code {{
    display: block;
    background: #110e0b;
    padding: 6px 8px;
    border-radius: 4px;
    font-size: 10px;
    word-break: break-all;
    color: #9be0a8;
  }}

  @media (max-width: 860px) {{
    .open-book-shell {{ grid-template-columns: 1fr; gap: 10px; }}
    .page-left, .page-right {{ border-radius: 4px; box-shadow: none; border-right: none; }}
    .tb-title {{ display: none; }}
  }}

  @media print {{
    body {{ background: #fff; }}
    .toolbar, .scrubber, .open-hint, .director-panel {{ display: none !important; }}
    .stage {{ padding: 0; display: block; }}
    .book-viewport {{ width: 100%; display: block; }}
    .spread {{ display: flex !important; page-break-after: always; break-after: page; margin-bottom: 12mm; }}
    .hardboard-cover, .open-book-shell {{ box-shadow: none; }}
  }}
</style>
</head>
<body>
  <nav class="toolbar">
    <div class="tb-left">
      <a class="tb-link" href="../../index.html">Bookshelf</a>
      <span class="tb-title">No. {series_num:02d} &bull; {esc(story['title'])}</span>
    </div>
    <div class="tb-center">
      <button class="tb-btn" id="btnPrev" onclick="prevSpread()">← Prev</button>
      <span class="tb-status" id="spreadStatus">Front Cover</span>
      <button class="tb-btn" id="btnNext" onclick="nextSpread()">Next →</button>
    </div>
    <div class="tb-right">
      <button class="tb-btn" id="btnDirector" onclick="toggleDirector()" title="Show seeds, refs, prompts & re-roll commands (D)">Director Mode</button>
      <button class="tb-btn" onclick="window.print()">Print / PDF</button>
    </div>
  </nav>

  <main class="stage">
    <div class="book-viewport" id="bookViewport">
      {''.join(sp['html'] for sp in spreads)}
    </div>
    <div class="scrubber" id="scrubber">{dots_html}</div>
  </main>

<script>
  const spreads = Array.from(document.querySelectorAll('.spread'));
  const dots = Array.from(document.querySelectorAll('.dot'));
  const statusEl = document.getElementById('spreadStatus');
  const btnPrev = document.getElementById('btnPrev');
  const btnNext = document.getElementById('btnNext');
  const btnDirector = document.getElementById('btnDirector');
  let current = 0;

  function updateUI(direction) {{
    spreads.forEach((el, idx) => {{
      el.classList.remove('active', 'flip-next', 'flip-prev');
      if (idx === current) {{
        el.classList.add('active');
        if (direction === 1) el.classList.add('flip-next');
        if (direction === -1) el.classList.add('flip-prev');
      }}
    }});
    dots.forEach((d, idx) => d.classList.toggle('active', idx === current));
    statusEl.textContent = spreads[current]?.dataset.label || '';
    btnPrev.disabled = current === 0;
    btnNext.disabled = current === spreads.length - 1;
  }}

  function goToSpread(idx) {{
    const clamped = Math.max(0, Math.min(spreads.length - 1, idx));
    if (clamped === current) return;
    const dir = clamped > current ? 1 : -1;
    current = clamped;
    updateUI(dir);
  }}

  function nextSpread(ev) {{
    if (ev && ev.target.closest('a, button, details, summary, code, [contenteditable]')) return;
    goToSpread(current + 1);
  }}

  function prevSpread(ev) {{
    if (ev && ev.target.closest('a, button, details, summary, code, [contenteditable]')) return;
    goToSpread(current - 1);
  }}

  function toggleDirector() {{
    const on = document.body.classList.toggle('director-mode');
    btnDirector.classList.toggle('active', on);
  }}

  document.addEventListener('keydown', (e) => {{
    if (document.activeElement && document.activeElement.isContentEditable) return;
    if (e.key === 'ArrowRight' || e.key === ' ') {{ e.preventDefault(); nextSpread(); }}
    if (e.key === 'ArrowLeft') {{ e.preventDefault(); prevSpread(); }}
    if (e.key === 'd' || e.key === 'D') {{ toggleDirector(); }}
  }});

  // Persist "This Fever Dream Belongs To" bookplate locally
  const bp = document.querySelector('[data-bookplate]');
  if (bp) {{
    const key = 'fever-dreams-owner-' + bp.dataset.bookplate;
    bp.textContent = localStorage.getItem(key) || '';
    bp.addEventListener('input', () => localStorage.setItem(key, bp.textContent));
  }}

  // Basic touch swipe
  let touchX = null;
  document.addEventListener('touchstart', (e) => {{ touchX = e.touches[0].clientX; }}, {{ passive: true }});
  document.addEventListener('touchend', (e) => {{
    if (touchX === null) return;
    const dx = e.changedTouches[0].clientX - touchX;
    if (dx < -45) nextSpread();
    if (dx > 45) prevSpread();
    touchX = null;
  }});

  updateUI(0);
</script>
</body>
</html>
"""


def build_bookshelf_html(catalog: list[dict]) -> str:
    cards: list[str] = []
    for item in catalog:
        pal = item["palette"]
        art = (
            f'<img src="{esc(item["cover_image"])}" alt="{esc(item["title"])}">'
            if item["cover_image"]
            else '<div class="shelf-ph">Artwork Pending</div>'
        )
        cast_str = " • ".join(item["cast_names"][:3])
        cards.append(
            f"""
            <a class="shelf-book" href="books/{esc(item['slug'])}/index.html"
               style="--book-bg: {esc(pal['bg'])}; --book-accent: {esc(pal['accent'])};">
              <div class="mini-spine"></div>
              <div class="mini-cover">
                <div class="mini-top">
                  <span>No. {item['series_number']:02d}</span>
                  <span>{esc(item['price'])}</span>
                </div>
                <div class="mini-title">{esc(item['title'])}</div>
                <div class="mini-art">{art}</div>
                <div class="mini-cast">{esc(cast_str)}</div>
              </div>
            </a>"""
        )

    empty_state = (
        '<div class="empty-shelf">No books on the shelf yet! Run '
        '<code>uv run python scripts/generate_story.py</code> to create Book No. 01.</div>'
        if not cards
        else ""
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Little Fever Dream Books — Series Bookshelf</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    min-height: 100vh;
    background: radial-gradient(circle at 50% 10%, #3d2f24 0%, #231a13 65%, #16100b 100%);
    color: #f6eedb;
    font-family: "Iowan Old Style", Palatino, Georgia, serif;
    padding: 36px 24px 60px;
  }}
  header {{ text-align: center; max-width: 780px; margin: 0 auto 38px; }}
  .badge {{
    display: inline-block;
    background: linear-gradient(90deg, #b98b24, #f5db82, #c89b2c);
    color: #211505;
    font-family: -apple-system, sans-serif;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 0.2em;
    padding: 5px 14px;
    border-radius: 999px;
    margin-bottom: 12px;
    text-decoration: none;
    transition: transform 0.15s ease, filter 0.15s ease;
  }}
  .badge:hover {{
    transform: translateY(-1px);
    filter: brightness(1.08);
  }}
  h1 {{ margin: 0; font-size: clamp(32px, 5vw, 48px); color: #fff8e7; }}
  header p {{ margin: 10px 0 0; font-size: 16px; color: #d8c6a6; font-style: italic; }}

  .shelf-grid {{
    max-width: 1180px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 28px;
  }}
  .shelf-book {{
    display: flex;
    text-decoration: none;
    color: #fff8e7;
    background: var(--book-bg);
    border-radius: 5px 12px 12px 5px;
    overflow: hidden;
    min-height: 340px;
    box-shadow: 0 18px 34px rgba(0, 0, 0, 0.55), 0 4px 0 #d5c3a1;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }}
  .shelf-book:hover {{
    transform: translateY(-6px) rotate(-0.6deg);
    box-shadow: 0 26px 44px rgba(0, 0, 0, 0.65), 0 4px 0 #f3d37a;
  }}
  .mini-spine {{
    width: 28px;
    flex-shrink: 0;
    background: linear-gradient(90deg, #7d5b10, #f5db82 40%, #b98b24 72%, #7d5b10);
    border-right: 2px solid rgba(50, 32, 4, 0.5);
  }}
  .mini-cover {{
    flex: 1;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .mini-top {{
    display: flex;
    justify-content: space-between;
    font-family: -apple-system, sans-serif;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.14em;
    color: var(--book-accent);
  }}
  .mini-title {{ font-size: 18px; font-weight: 700; line-height: 1.15; text-align: center; }}
  .mini-art {{
    flex: 1;
    background: #f6eedb;
    padding: 5px;
    border-radius: 5px;
    overflow: hidden;
    display: flex;
  }}
  .mini-art img {{ width: 100%; height: 100%; object-fit: cover; border-radius: 3px; }}
  .shelf-ph {{
    flex: 1;
    display: grid;
    place-items: center;
    color: #7c684d;
    font-family: -apple-system, sans-serif;
    font-size: 11px;
    border: 1px dashed #b99e70;
  }}
  .mini-cast {{
    font-family: -apple-system, sans-serif;
    font-size: 10px;
    text-align: center;
    color: rgba(255, 248, 231, 0.85);
  }}
  .empty-shelf {{
    grid-column: 1 / -1;
    text-align: center;
    padding: 48px 24px;
    border: 2px dashed rgba(243, 211, 122, 0.35);
    border-radius: 12px;
    color: #d8c6a6;
  }}
</style>
</head>
<body>
  <header>
    <a class="badge" href="../meme-yearbook/index.html">MEME VALLEY HIGH SCHOOL &bull; CLASS OF 2025</a>
    <h1>Little Fever Dream Books</h1>
    <p>Thin hardboard classics with gold-foil spines, deadpan 1950s narration, and deeply questionable decisions.</p>
  </header>
  <main class="shelf-grid">
    {''.join(cards)}{empty_state}
  </main>
</body>
</html>
"""


def build_one_book(book_dir: Path, catalog: list[dict]) -> Path:
    story_path = book_dir / "story.json"
    if not story_path.exists():
        raise FileNotFoundError(f"story.json not found in {book_dir}")
    story = json.loads(story_path.read_text(encoding="utf-8"))
    out_html = book_dir / "index.html"
    out_html.write_text(build_book_html(story, book_dir, catalog), encoding="utf-8")
    return out_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Build browser-flippable Little Golden Book HTML files.")
    parser.add_argument("target", nargs="?", type=Path, default=None, help="Book directory or story.json path")
    parser.add_argument("--all", action="store_true", help="Rebuild all books in books/ plus the bookshelf index")
    args = parser.parse_args()

    catalog = collect_series_catalog()

    if args.all or args.target is None:
        built = 0
        if BOOKS_DIR.exists():
            for story_file in sorted(BOOKS_DIR.glob("*/story.json")):
                out = build_one_book(story_file.parent, catalog)
                print(f"Built {out.relative_to(FEVER_DREAMS_ROOT)}")
                built += 1
        shelf_out = FEVER_DREAMS_ROOT / "index.html"
        shelf_out.write_text(build_bookshelf_html(catalog), encoding="utf-8")
        print(f"Updated bookshelf: {shelf_out.relative_to(FEVER_DREAMS_ROOT)} ({built} book(s))")
        return

    target = args.target.resolve()
    book_dir = target.parent if target.is_file() else target
    if not (book_dir / "story.json").exists():
        print(f"Error: no story.json found at {book_dir}", file=sys.stderr)
        sys.exit(1)

    out = build_one_book(book_dir, catalog)
    shelf_out = FEVER_DREAMS_ROOT / "index.html"
    shelf_out.write_text(build_bookshelf_html(catalog), encoding="utf-8")
    print(f"Built book: {out}")
    print(f"Updated bookshelf: {shelf_out}")


if __name__ == "__main__":
    main()
