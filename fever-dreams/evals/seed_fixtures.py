"""Seed labeled golden fixtures for Eval Suites 1-4 from real Book 1 assets and user defect samples.

Zero AI calls: pure deterministic file copies, PIL crops, contact-sheet grids, and color perturbations.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

FEVER_DREAMS_ROOT = Path(__file__).resolve().parent.parent
BOOK1 = FEVER_DREAMS_ROOT / "books" / "01-the-very-big-world-problem"
ART_DIR = BOOK1 / "art"
HIST_DIR = ART_DIR / ".history"
RETOUCH_DIR = BOOK1 / ".retouch"
CAST_DIR = BOOK1 / "cast"
FIXTURES_DIR = FEVER_DREAMS_ROOT / "evals" / "fixtures"

USER_HALF_GLASSES_SRC = FIXTURES_DIR / "defects" / "fail_half_glasses_crop.jpg"


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


def copy_if_exists(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(f"Missing source fixture asset: {src}")
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst


def save_crop(src: Path, dst: Path, box: tuple[int, int, int, int]) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    x1 = max(0, min(im.width - 1, box[0]))
    y1 = max(0, min(im.height - 1, box[1]))
    x2 = max(x1 + 1, min(im.width, box[2]))
    y2 = max(y1 + 1, min(im.height, box[3]))
    im.crop((x1, y1, x2, y2)).save(dst, format="PNG")
    return dst


def hue_shift_image(src: Path, dst: Path, shift_deg: int = 140) -> Path:
    """Shift hue dramatically to create a deterministic color-continuity mismatch fixture."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("HSV")
    arr = np.array(im, dtype=np.uint16)
    arr[..., 0] = (arr[..., 0] + int(shift_deg * 255 / 360)) % 256
    Image.fromarray(arr.astype(np.uint8), mode="HSV").convert("RGB").save(dst, format="PNG")
    return dst


def night_grade_image(src: Path, dst: Path) -> Path:
    """Apply a strong cold dark-blue night lighting shift for adjacent-cut lighting break tests."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    im = ImageEnhance.Brightness(im).enhance(0.45)
    arr = np.array(im, dtype=np.float32)
    arr[..., 0] *= 0.55  # reduce red
    arr[..., 1] *= 0.75  # slight green reduction
    arr[..., 2] = np.clip(arr[..., 2] * 1.45 + 24.0, 0, 255)  # boost cold blue
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB").save(dst, format="PNG")
    return dst


def make_contact_sheet(
    tiles: list[tuple[str, Path]],
    out_path: Path,
    cols: int = 2,
    tile_size: int = 448,
) -> Path:
    """Stitch numbered tiles (Tile 0 = canonical reference, Tiles 1..N = instances) with crisp badges."""
    rows = (len(tiles) + cols - 1) // cols
    pad = 16
    w = cols * tile_size + (cols + 1) * pad
    h = rows * tile_size + (rows + 1) * pad
    canvas = Image.new("RGB", (w, h), (22, 20, 18))
    draw = ImageDraw.Draw(canvas)
    font = load_font(26)

    for idx, (label, path) in enumerate(tiles):
        r, c = divmod(idx, cols)
        x0 = pad + c * (tile_size + pad)
        y0 = pad + r * (tile_size + pad)

        im = Image.open(path).convert("RGB")
        # Fit with centered letterboxing so aspect ratios stay undistorted
        scale = min(tile_size / im.width, tile_size / im.height)
        nw, nh = max(1, int(im.width * scale)), max(1, int(im.height * scale))
        resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
        tile_bg = Image.new("RGB", (tile_size, tile_size), (38, 34, 30))
        tile_bg.paste(resized, ((tile_size - nw) // 2, (tile_size - nh) // 2))
        canvas.paste(tile_bg, (x0, y0))

        # High-contrast border and top-left tile number badge
        border_col = (245, 208, 96) if idx == 0 else (180, 180, 180)
        draw.rectangle([x0, y0, x0 + tile_size - 1, y0 + tile_size - 1], outline=border_col, width=4)
        badge_bg = (245, 208, 96) if idx == 0 else (18, 18, 18)
        badge_fg = (18, 18, 18) if idx == 0 else (255, 255, 255)
        tw = int(draw.textlength(label, font=font)) + 20
        draw.rectangle([x0 + 6, y0 + 6, x0 + 6 + tw, y0 + 44], fill=badge_bg, outline=border_col, width=2)
        draw.text((x0 + 16, y0 + 11), label, fill=badge_fg, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")
    return out_path


def seed_defects() -> dict:
    d = FIXTURES_DIR / "defects"
    d.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []

    # --- FAIL CASES (real rendering/anatomy/accessory defects) ---
    if USER_HALF_GLASSES_SRC.exists():
        copy_if_exists(USER_HALF_GLASSES_SRC, d / "fail_half_glasses_crop.jpg")
        cases.append(
            {
                "id": "fail_half_glasses_crop",
                "image": "fail_half_glasses_crop.jpg",
                "expected": "FAIL",
                "defect_type": "accessory",
                "expected_keywords": ["glass", "lens", "frame", "half", "cheek", "eye", "face", "spectacle"],
                "description": "Green character has a hollow half-glasses frame floating on the left side of his face.",
            }
        )

    p17_hg = HIST_DIR / "page_17_20261126_052010.png"
    copy_if_exists(p17_hg, d / "fail_p17_half_glasses_full.png")
    cases.append(
        {
            "id": "fail_p17_half_glasses_full",
            "image": "fail_p17_half_glasses_full.png",
            "expected": "FAIL",
            "defect_type": "accessory",
            "expected_keywords": ["glass", "lens", "frame", "half", "cheek", "loop", "face", "ear", "green"],
            "description": "Full storybook frame where the green boy on the left has a hollow half-glasses loop stuck to his left cheek.",
        }
    )

    p01_chin = HIST_DIR / "page_01_20261126_023952.png"
    copy_if_exists(p01_chin, d / "fail_p01_chin_rock.png")
    cases.append(
        {
            "id": "fail_p01_chin_rock",
            "image": "fail_p01_chin_rock.png",
            "expected": "FAIL",
            "defect_type": "anatomy",
            "expected_keywords": ["chin", "mouth", "rock", "red", "blob", "object", "sticking", "artifact", "face", "globe"],
            "description": "Weird red rocky object/artifact sticking out of the green boy's chin against the globe.",
        }
    )
    # Zoomed crop of p01 chin region (896x768)
    save_crop(p01_chin, d / "fail_p01_chin_rock_crop.png", (240, 180, 680, 620))
    cases.append(
        {
            "id": "fail_p01_chin_rock_crop",
            "image": "fail_p01_chin_rock_crop.png",
            "expected": "FAIL",
            "defect_type": "anatomy",
            "expected_keywords": ["chin", "mouth", "rock", "red", "blob", "object", "artifact", "globe"],
            "description": "Zoomed crop of red rocky artifact protruding from the green boy's chin/mouth.",
        }
    )

    p11_legs = HIST_DIR / "page_11_20261126_030739.png"
    copy_if_exists(p11_legs, d / "fail_p11_three_legs.png")
    cases.append(
        {
            "id": "fail_p11_three_legs",
            "image": "fail_p11_three_legs.png",
            "expected": "FAIL",
            "defect_type": "anatomy",
            "expected_keywords": ["leg", "foot", "shoe", "extra", "third", "three", "limb", "side", "puppet"],
            "description": "Yellow-green felt puppet has an extra third leg and green shoe sticking out of his side.",
        }
    )
    save_crop(p11_legs, d / "fail_p11_three_legs_crop.png", (280, 260, 760, 740))
    cases.append(
        {
            "id": "fail_p11_three_legs_crop",
            "image": "fail_p11_three_legs_crop.png",
            "expected": "FAIL",
            "defect_type": "anatomy",
            "expected_keywords": ["leg", "foot", "shoe", "extra", "third", "three", "limb"],
            "description": "Zoomed crop of extra raised third leg and shoe sticking out of the puppet's side.",
        }
    )

    p18_beam = HIST_DIR / "page_18_20261126_033154.png"
    copy_if_exists(p18_beam, d / "fail_p18_hand_beam_merge.png")
    cases.append(
        {
            "id": "fail_p18_hand_beam_merge",
            "image": "fail_p18_hand_beam_merge.png",
            "expected": "FAIL",
            "defect_type": "anatomy",
            "expected_keywords": ["hand", "arm", "beam", "wood", "merg", "fus", "stick", "distort", "limb"],
            "description": "Green pixel boy's arm/hand merges unnaturally into the wooden support beam.",
        }
    )

    # --- PASS CASES (clean approved Book 1 illustrations) ---
    clean_pages = [
        ("pass_p02_clean", ART_DIR / "page_02.png", "Clean approved page 2"),
        ("pass_p03_clean", ART_DIR / "page_03.png", "Clean approved page 3"),
        ("pass_p05_clean", ART_DIR / "page_05.png", "Clean approved page 5"),
        ("pass_p08_clean", ART_DIR / "page_08.png", "Clean approved page 8"),
        ("pass_p10_clean", ART_DIR / "page_10.png", "Clean approved page 10"),
        ("pass_p11_fixed", ART_DIR / "page_11.png", "Approved fixed page 11 (2 legs only)"),
        ("pass_p17_fixed", ART_DIR / "page_17.png", "Approved fixed page 17 (clean pixel Chester, no half-glasses)"),
        ("pass_p18_fixed", ART_DIR / "page_18.png", "Approved fixed page 18 (Chester separated from beam)"),
    ]
    for cid, src, desc in clean_pages:
        fname = f"{cid}.png"
        copy_if_exists(src, d / fname)
        cases.append(
            {
                "id": cid,
                "image": fname,
                "expected": "PASS",
                "defect_type": "none",
                "expected_keywords": [],
                "description": desc,
            }
        )

    manifest = {"suite": "defects", "cases": cases}
    (d / "labels.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def seed_anchors() -> dict:
    d = FIXTURES_DIR / "anchors"
    d.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []

    # Copy canonical refs
    ref_chester = copy_if_exists(CAST_DIR / "01-chester-antialiased.webp", d / "ref_chester.webp")
    ref_marisol = copy_if_exists(CAST_DIR / "03-marisol-brittle.webp", d / "ref_marisol.webp")
    ref_winston = copy_if_exists(CAST_DIR / "06-winston-crumb.webp", d / "ref_winston.webp")
    ref_crane = copy_if_exists(RETOUCH_DIR / "page_12" / "page_10_crane_ref.png", d / "ref_crane.png")

    # Copy test images
    copy_if_exists(HIST_DIR / "page_12_20261126_050126.png", d / "test_p12_wrong_crane.png")
    copy_if_exists(ART_DIR / "page_12.png", d / "test_p12_good_crane.png")
    copy_if_exists(HIST_DIR / "cover_20261126_055622.png", d / "test_cover_wrong_chester.png")
    copy_if_exists(ART_DIR / "cover.png", d / "test_cover_good_chester.png")
    copy_if_exists(HIST_DIR / "page_17_20261126_062644.png", d / "test_p17_3d_chester.png")
    copy_if_exists(ART_DIR / "page_17.png", d / "test_p17_good_chester.png")
    copy_if_exists(ART_DIR / "page_04.png", d / "test_p04_good_winston.png")
    hue_shift_image(ART_DIR / "page_04.png", d / "test_p04_recolored_winston.png", shift_deg=155)
    copy_if_exists(ART_DIR / "page_05.png", d / "test_p05_good_marisol.png")
    if USER_HALF_GLASSES_SRC.exists():
        copy_if_exists(USER_HALF_GLASSES_SRC, d / "test_half_glasses_chester.jpg")

    cases = [
        {
            "id": "fail_crane_mismatch",
            "entity_type": "prop",
            "entity_name": "yellow wheeled mobile construction crane truck",
            "ref_image": ref_crane.name,
            "test_image": "test_p12_wrong_crane.png",
            "expected": "FAIL",
            "description": "Test image shows a green steam-train contraption instead of the yellow wheeled construction crane truck.",
        },
        {
            "id": "pass_crane_match",
            "entity_type": "prop",
            "entity_name": "yellow wheeled mobile construction crane truck",
            "ref_image": ref_crane.name,
            "test_image": "test_p12_good_crane.png",
            "expected": "PASS",
            "description": "Test image shows the matching yellow wheeled construction crane truck.",
        },
        {
            "id": "fail_cover_chester_drift",
            "entity_type": "character",
            "entity_name": "Chester Antialiased (16-bit pixel-art green goblin boy, no glasses, no horns)",
            "ref_image": ref_chester.name,
            "test_image": "test_cover_wrong_chester.png",
            "expected": "FAIL",
            "description": "Cover shows a smooth clay horned boy with glasses instead of pixel-art Chester without glasses or horns.",
        },
        {
            "id": "pass_cover_chester_match",
            "entity_type": "character",
            "entity_name": "Chester Antialiased (16-bit pixel-art green goblin boy, no glasses, no horns)",
            "ref_image": ref_chester.name,
            "test_image": "test_cover_good_chester.png",
            "expected": "PASS",
            "description": "Cover shows matching 16-bit pixel-art Chester.",
        },
        {
            "id": "fail_p17_chester_3d_drift",
            "entity_type": "character",
            "entity_name": "Chester Antialiased (flat 2D 16-bit pixel-art green goblin boy)",
            "ref_image": ref_chester.name,
            "test_image": "test_p17_3d_chester.png",
            "expected": "FAIL",
            "description": "Page 17 shows a smooth 3D clay/shaded boy instead of flat 16-bit pixel-art Chester.",
        },
        {
            "id": "pass_p17_chester_match",
            "entity_type": "character",
            "entity_name": "Chester Antialiased (flat 2D 16-bit pixel-art green goblin boy)",
            "ref_image": ref_chester.name,
            "test_image": "test_p17_good_chester.png",
            "expected": "PASS",
            "description": "Page 17 shows matching 16-bit pixel-art Chester.",
        },
        {
            "id": "fail_winston_color_drift",
            "entity_type": "character",
            "entity_name": "Winston Crumb (chartreuse yellow-green felt puppet with googly eyes)",
            "ref_image": ref_winston.name,
            "test_image": "test_p04_recolored_winston.png",
            "expected": "FAIL",
            "description": "Winston's fur and colors are drastically hue-shifted away from chartreuse yellow-green.",
        },
        {
            "id": "pass_winston_match",
            "entity_type": "character",
            "entity_name": "Winston Crumb (chartreuse yellow-green felt puppet with googly eyes)",
            "ref_image": ref_winston.name,
            "test_image": "test_p04_good_winston.png",
            "expected": "PASS",
            "description": "Page 4 shows matching chartreuse felt puppet Winston.",
        },
        {
            "id": "pass_marisol_match",
            "entity_type": "character",
            "entity_name": "Marisol Brittle (knit/clay owl girl)",
            "ref_image": ref_marisol.name,
            "test_image": "test_p05_good_marisol.png",
            "expected": "PASS",
            "description": "Page 5 shows matching Marisol Brittle.",
        },
    ]
    if USER_HALF_GLASSES_SRC.exists():
        cases.append(
            {
                "id": "fail_half_glasses_vs_ref",
                "entity_type": "character",
                "entity_name": "Chester Antialiased (16-bit pixel-art green goblin boy, no glasses)",
                "ref_image": ref_chester.name,
                "test_image": "test_half_glasses_chester.jpg",
                "expected": "FAIL",
                "description": "Smooth clay boy with a broken half-glasses frame does not match pixel-art Chester reference.",
            }
        )

    manifest = {"suite": "anchors", "cases": cases}
    (d / "labels.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def seed_contact_sheets() -> dict:
    d = FIXTURES_DIR / "contact_sheets"
    d.mkdir(parents=True, exist_ok=True)
    ref_chester = CAST_DIR / "01-chester-antialiased.webp"
    ref_winston = CAST_DIR / "06-winston-crumb.webp"
    ref_crane = RETOUCH_DIR / "page_12" / "page_10_crane_ref.png"

    # Helper perturbed image for Winston color drift
    winston_bad = d / "_tmp_winston_hue.png"
    hue_shift_image(ART_DIR / "page_04.png", winston_bad, shift_deg=155)

    cases: list[dict] = []

    # 1. Chester 2x2 with half-glasses outlier at Tile 2
    if USER_HALF_GLASSES_SRC.exists():
        p = make_contact_sheet(
            [
                ("TILE 0 (REF)", ref_chester),
                ("TILE 1", ART_DIR / "page_17.png"),
                ("TILE 2", USER_HALF_GLASSES_SRC),
                ("TILE 3", ART_DIR / "cover.png"),
            ],
            d / "sheet_chester_2x2_halfglasses.png",
            cols=2,
        )
        cases.append(
            {
                "id": "sheet_chester_2x2_halfglasses",
                "image": p.name,
                "entity_name": "Chester Antialiased (16-bit pixel-art green goblin boy, no glasses)",
                "tile_count": 3,
                "expected_outliers": [2],
                "description": "Tile 2 is a smooth 3D boy with a broken half-glasses loop instead of pixel-art Chester.",
            }
        )

    # 2. Crane 2x2 with wrong green steam-train at Tile 3
    p = make_contact_sheet(
        [
            ("TILE 0 (REF)", ref_crane),
            ("TILE 1", ART_DIR / "page_10.png"),
            ("TILE 2", ART_DIR / "page_12.png"),
            ("TILE 3", HIST_DIR / "page_12_20261126_050126.png"),
        ],
        d / "sheet_crane_2x2_wrong_truck.png",
        cols=2,
    )
    cases.append(
        {
            "id": "sheet_crane_2x2_wrong_truck",
            "image": p.name,
            "entity_name": "yellow wheeled mobile construction crane truck",
            "tile_count": 3,
            "expected_outliers": [3],
            "description": "Tile 3 has a green steam-train contraption instead of the yellow wheeled construction crane truck.",
        }
    )

    # 3. Winston 2x2 with 3-legged anatomy defect at Tile 3
    p = make_contact_sheet(
        [
            ("TILE 0 (REF)", ref_winston),
            ("TILE 1", ART_DIR / "page_04.png"),
            ("TILE 2", ART_DIR / "page_11.png"),
            ("TILE 3", HIST_DIR / "page_11_20261126_030739.png"),
        ],
        d / "sheet_winston_2x2_three_legs.png",
        cols=2,
    )
    cases.append(
        {
            "id": "sheet_winston_2x2_three_legs",
            "image": p.name,
            "entity_name": "Winston Crumb (chartreuse felt puppet with two legs)",
            "tile_count": 3,
            "expected_outliers": [3],
            "description": "Tile 3 shows Winston with an extra raised third leg sticking out of his side.",
        }
    )

    # 4. Winston 2x2 with hue-shifted color drift at Tile 2
    p = make_contact_sheet(
        [
            ("TILE 0 (REF)", ref_winston),
            ("TILE 1", ART_DIR / "page_04.png"),
            ("TILE 2", winston_bad),
            ("TILE 3", ART_DIR / "page_11.png"),
        ],
        d / "sheet_winston_2x2_color_drift.png",
        cols=2,
    )
    cases.append(
        {
            "id": "sheet_winston_2x2_color_drift",
            "image": p.name,
            "entity_name": "Winston Crumb (chartreuse yellow-green felt puppet)",
            "tile_count": 3,
            "expected_outliers": [2],
            "description": "Tile 2 has drastically wrong hue-shifted colors for Winston.",
        }
    )

    # 5. Chester 3x3 (6 tiles: Tile 0 + 5 instances) with TWO planted outliers (Tile 2 and Tile 5)
    p = make_contact_sheet(
        [
            ("TILE 0 (REF)", ref_chester),
            ("TILE 1", ART_DIR / "cover.png"),
            ("TILE 2", HIST_DIR / "cover_20261126_055622.png"),
            ("TILE 3", ART_DIR / "page_17.png"),
            ("TILE 4", ART_DIR / "page_18.png"),
            ("TILE 5", HIST_DIR / "page_17_20261126_062644.png"),
        ],
        d / "sheet_chester_3x2_two_outliers.png",
        cols=3,
        tile_size=384,
    )
    cases.append(
        {
            "id": "sheet_chester_3x2_two_outliers",
            "image": p.name,
            "entity_name": "Chester Antialiased (16-bit pixel-art green goblin boy, no glasses, no horns)",
            "tile_count": 5,
            "expected_outliers": [2, 5],
            "description": "Tile 2 (horned clay boy with glasses) and Tile 5 (smooth 3D shaded boy) drift from pixel-art Chester.",
        }
    )

    # 6. All-clean control sheet (zero outliers)
    p = make_contact_sheet(
        [
            ("TILE 0 (REF)", ref_chester),
            ("TILE 1", ART_DIR / "cover.png"),
            ("TILE 2", ART_DIR / "page_17.png"),
            ("TILE 3", ART_DIR / "page_18.png"),
        ],
        d / "sheet_chester_2x2_all_clean.png",
        cols=2,
    )
    cases.append(
        {
            "id": "sheet_chester_2x2_all_clean",
            "image": p.name,
            "entity_name": "Chester Antialiased (16-bit pixel-art green goblin boy, no glasses, no horns)",
            "tile_count": 3,
            "expected_outliers": [],
            "description": "Clean control: all three tiles show consistent 16-bit pixel-art Chester.",
        }
    )

    winston_bad.unlink(missing_ok=True)
    manifest = {"suite": "contact_sheets", "cases": cases}
    (d / "labels.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def seed_adjacent_cuts() -> dict:
    d = FIXTURES_DIR / "adjacent_cuts"
    d.mkdir(parents=True, exist_ok=True)

    # Copy base images
    copy_if_exists(ART_DIR / "page_01.png", d / "p01.png")
    copy_if_exists(ART_DIR / "page_02.png", d / "p02.png")
    night_grade_image(ART_DIR / "page_02.png", d / "p02_night_flip.png")
    copy_if_exists(ART_DIR / "page_10.png", d / "p10.png")
    copy_if_exists(ART_DIR / "page_12.png", d / "p12_good.png")
    copy_if_exists(HIST_DIR / "page_12_20261126_050126.png", d / "p12_bad_crane.png")
    copy_if_exists(ART_DIR / "page_16.png", d / "p16.png")
    copy_if_exists(ART_DIR / "page_17.png", d / "p17_good.png")
    copy_if_exists(HIST_DIR / "page_17_20261126_062644.png", d / "p17_bad_3d.png")

    cases = [
        {
            "id": "pass_classroom_p01_p02",
            "shot_prev": "p01.png",
            "shot_next": "p02.png",
            "same_set_expected": True,
            "scene_context": "Consecutive shots inside the sunlit 1950s classroom around the giant globe.",
            "expected": "PASS",
            "description": "Consistent sunlit classroom lighting and character styles across p01 -> p02.",
        },
        {
            "id": "fail_classroom_lighting_flip",
            "shot_prev": "p01.png",
            "shot_next": "p02_night_flip.png",
            "same_set_expected": True,
            "scene_context": "Consecutive shots inside the sunlit 1950s classroom moments apart.",
            "expected": "FAIL",
            "description": "Sudden shift from warm sunlit daylight to dark cold blue night lighting in the same scene.",
        },
        {
            "id": "pass_crane_p10_p12",
            "shot_prev": "p10.png",
            "shot_next": "p12_good.png",
            "same_set_expected": True,
            "scene_context": "Consecutive street shots featuring the yellow wheeled construction crane truck.",
            "expected": "PASS",
            "description": "Consistent yellow wheeled crane truck and daytime street lighting.",
        },
        {
            "id": "fail_crane_mutated_p10_p12",
            "shot_prev": "p10.png",
            "shot_next": "p12_bad_crane.png",
            "same_set_expected": True,
            "scene_context": "Consecutive street shots featuring the same construction crane truck.",
            "expected": "FAIL",
            "description": "Yellow wheeled construction crane truck mutates into a green steam-train contraption.",
        },
        {
            "id": "pass_street_p16_p17",
            "shot_prev": "p16.png",
            "shot_next": "p17_good.png",
            "same_set_expected": True,
            "scene_context": "Consecutive daytime street shots following Chester Antialiased.",
            "expected": "PASS",
            "description": "Consistent daytime street scene and 16-bit pixel-art Chester.",
        },
        {
            "id": "fail_chester_style_jump_p16_p17",
            "shot_prev": "p16.png",
            "shot_next": "p17_bad_3d.png",
            "same_set_expected": True,
            "scene_context": "Consecutive daytime street shots following pixel-art Chester Antialiased.",
            "expected": "FAIL",
            "description": "Chester jumps from flat 16-bit pixel art to a smooth 3D clay/shaded character.",
        },
        {
            "id": "fail_set_swap_classroom_to_street",
            "shot_prev": "p01.png",
            "shot_next": "p17_good.png",
            "same_set_expected": True,
            "scene_context": "Tagged as consecutive shots inside the same indoor classroom set.",
            "expected": "FAIL",
            "description": "Location abruptly changes from indoor classroom to outdoor city street despite same_set_expected=True.",
        },
    ]

    manifest = {"suite": "adjacent_cuts", "cases": cases}
    (d / "labels.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    m1 = seed_defects()
    m2 = seed_anchors()
    m3 = seed_contact_sheets()
    m4 = seed_adjacent_cuts()
    print(f"Seeded defects:        {len(m1['cases'])} cases")
    print(f"Seeded anchors:        {len(m2['cases'])} cases")
    print(f"Seeded contact_sheets: {len(m3['cases'])} cases")
    print(f"Seeded adjacent_cuts:  {len(m4['cases'])} cases")


if __name__ == "__main__":
    main()
