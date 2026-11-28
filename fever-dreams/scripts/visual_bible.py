"""Visual Bible & Single-Render Set-Plate Reuse Subsystem (Phase 1).

Deterministically maps entity metadata (`cast_ids`, `prop_ids`, `set_id`, `camera_angle`)
across shots, generates canonical reference sheets for recurring characters, props, and
master empty set plates, and renders shots conditioned on reused set plates + refs.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

FEVER_DREAMS_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = FEVER_DREAMS_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_illustrations import load_edit_model, render_edit, save_clean  # noqa: E402
from llm import unload_lms_models  # noqa: E402
from prompts import GOLDEN_BOOK_WORLD_STYLE, build_turnaround_prompt, medium_lock_phrase  # noqa: E402


@dataclass
class EntityRegistry:
    """Deterministic lookup of every recurring character, prop, and (set, angle) plate across shots."""

    cast_shots: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    prop_shots: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    set_angle_shots: dict[tuple[str, str], list[str]] = field(default_factory=lambda: defaultdict(list))

    @classmethod
    def from_shots(cls, shots: list[dict[str, Any]]) -> "EntityRegistry":
        reg = cls()
        for shot in shots:
            sid = str(shot.get("shot_id") or f"p{int(shot.get('page_number', 0)):02d}")
            for cid in shot.get("cast_ids") or []:
                reg.cast_shots[str(cid)].append(sid)
            for prop in shot.get("prop_ids") or []:
                pid = str(prop["id"] if isinstance(prop, dict) else prop)
                reg.prop_shots[pid].append(sid)
            set_id = shot.get("set_id")
            if set_id:
                angle = str(shot.get("camera_angle") or "wide").strip().lower().replace(" ", "_")
                reg.set_angle_shots[(str(set_id), angle)].append(sid)
        return reg

    def recurring_props(self, min_count: int = 2) -> list[str]:
        return sorted(pid for pid, sids in self.prop_shots.items() if len(sids) >= min_count)

    def recurring_set_plates(self, min_count: int = 2) -> list[tuple[str, str]]:
        return sorted(k for k, sids in self.set_angle_shots.items() if len(sids) >= min_count)


def build_prop_sheet_prompt(prop_name: str, prop_description: str, world_style: str = GOLDEN_BOOK_WORLD_STYLE) -> str:
    return (
        f"Canonical prop design reference sheet of {prop_name}: {prop_description.strip().rstrip('.')}. "
        f"Show THREE clean views side-by-side on a solid warm-cream studio backdrop: "
        f"(left) three-quarter front view, (center) full side profile, (right) three-quarter rear view. "
        f"Consistent colors, exact shape, proportions, wheels/hardware, and materials across all three views. "
        f"Style context: {world_style}. Clean even lighting, zero characters, zero text, zero labels."
    )


def build_set_plate_prompt(
    set_name: str,
    set_description: str,
    camera_angle: str,
    lighting_state: str = "warm afternoon daylight",
    world_style: str = GOLDEN_BOOK_WORLD_STYLE,
) -> str:
    return (
        f"EMPTY master background set plate (NO characters, NO people, NO animals anywhere in frame). "
        f"Location: {set_name} — {set_description.strip().rstrip('.')}. "
        f"Camera framing: {camera_angle.strip().rstrip('.')}. "
        f"Lighting: {lighting_state.strip().rstrip('.')}. "
        f"{world_style}. "
        f"Clear, stable architectural landmarks and floor space left open for characters to stand in later shots. "
        f"Zero characters, zero text, zero watermarks."
    )


def build_set_reuse_shot_prompt(
    *,
    set_plate_index: int,
    cast_refs: list[tuple[int, dict[str, Any]]],
    prop_refs: list[tuple[int, str, str]],
    action_description: str,
    camera_note: str = "",
    lighting_state: str = "",
    locked_camera: bool = False,
    world_style: str = GOLDEN_BOOK_WORLD_STYLE,
) -> str:
    """Build a multi-reference shot prompt anchored to a single-render master set plate."""
    if locked_camera:
        set_clause = (
            f"Image {set_plate_index} is the LOCKED MASTER SET PLATE. "
            f"Keep the EXACT camera framing, background walls, furniture, floor, architecture, and lighting of "
            f"Image {set_plate_index} completely unchanged — do NOT alter or move background objects."
        )
    else:
        set_clause = (
            f"Image {set_plate_index} is the MASTER SET REFERENCE for this location. "
            f"Keep the exact same room architecture, wall colors, fixtures, props, and lighting palette from "
            f"Image {set_plate_index} while matching the requested camera framing."
        )

    cast_clauses = []
    for idx, char in cast_refs:
        lock = medium_lock_phrase(char.get("style_id", ""))
        cast_clauses.append(
            f"Image {idx} is {char['name']} ({char.get('style_label', 'character')}), who MUST keep their exact "
            f"face, silhouette, colors, and outfit from Image {idx} and {lock}."
        )

    prop_clauses = []
    for idx, pname, pstate in prop_refs:
        state_txt = f" (current state: {pstate})" if pstate else ""
        prop_clauses.append(
            f"Image {idx} is the canonical prop '{pname}'{state_txt}; keep its exact shape, colors, and design from Image {idx}."
        )

    cam_txt = f" Camera framing: {camera_note.strip().rstrip('.')}." if camera_note else ""
    light_txt = f" Lighting: {lighting_state.strip().rstrip('.')}." if lighting_state else ""

    return (
        f"{set_clause} "
        f"{' '.join(cast_clauses)} "
        f"{' '.join(prop_clauses)} "
        f"Scene action: {action_description.strip().rstrip('.')}.{cam_txt}{light_txt} "
        f"Characters and props cast natural grounded shadows onto the set from Image {set_plate_index}. "
        f"{world_style}. Zero text, zero speech bubbles, zero watermarks."
    )


def _crop_single_view(sheet_path: Path, cache_dir: Path) -> Path:
    """Crop the left single-pose view (~34% width) from a wide 3-view sheet before mflux conditioning.

    Prevents qwen-image-2.1-turbo from copy-pasting 2 or 3 clones of the same character/prop into a scene.
    """
    im = Image.open(sheet_path).convert("RGB")
    w, h = im.size
    if w / max(1, h) < 1.5:
        return sheet_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{sheet_path.stem}_single.png"
    if not out.exists():
        im.crop((0, 0, int(w * 0.34), h)).save(out, format="PNG")
    return out


def _blank_canvas(path: Path, width: int, height: int, color: tuple[int, int, int] = (238, 230, 214)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        Image.new("RGB", (width, height), color).save(path, format="PNG")
    return path


def generate_bible_and_shots(
    spec_data: dict[str, Any],
    *,
    out_dir: Path,
    model_key: str = "qwen-image-2.1-turbo",
    width: int = 896,
    height: int = 768,
    seed: int = 2026,
) -> dict[str, Any]:
    """Generate canonical sheets + master empty set plates once, then render all shots reusing them."""
    unload_lms_models()
    out_dir.mkdir(parents=True, exist_ok=True)
    refs_dir = out_dir / "refs"
    shots_dir = out_dir / "shots"
    refs_dir.mkdir(parents=True, exist_ok=True)
    shots_dir.mkdir(parents=True, exist_ok=True)

    world_style = spec_data.get("world_style") or GOLDEN_BOOK_WORLD_STYLE
    blank = _blank_canvas(refs_dir / "_neutral_canvas.png", width, height)

    model, mspec = load_edit_model(model_key)
    manifest: dict[str, Any] = {
        "model": model_key,
        "width": width,
        "height": height,
        "refs": {"cast": {}, "props": {}, "sets": {}},
        "shots": [],
    }

    try:
        # 1. Character turnaround sheets
        for i, char in enumerate(spec_data.get("cast") or []):
            cid = char["id"]
            portrait = Path(char["portrait_path"])
            out_p = refs_dir / f"cast_{cid}.png"
            t0 = time.perf_counter()
            prompt = build_turnaround_prompt(char)
            gen = render_edit(
                model,
                mspec,
                prompt=prompt,
                image_paths=[portrait],
                seed=seed + 100 + i * 17,
                width=1024,
                height=576,
            )
            save_clean(gen, out_p)
            dt = round(time.perf_counter() - t0, 2)
            manifest["refs"]["cast"][cid] = {"path": str(out_p), "elapsed_s": dt}
            print(f"  [bible] cast turnaround {cid} -> {out_p.name} ({dt}s)")

        # 2. Recurring prop canonical sheets
        for i, prop in enumerate(spec_data.get("props") or []):
            pid = prop["id"]
            out_p = refs_dir / f"prop_{pid}.png"
            t0 = time.perf_counter()
            prompt = build_prop_sheet_prompt(prop["name"], prop["description"], world_style=world_style)
            init_ref = [Path(prop["seed_image"])] if prop.get("seed_image") else [blank]
            gen = render_edit(
                model,
                mspec,
                prompt=prompt,
                image_paths=init_ref,
                seed=seed + 200 + i * 17,
                width=1024,
                height=576,
            )
            save_clean(gen, out_p)
            dt = round(time.perf_counter() - t0, 2)
            manifest["refs"]["props"][pid] = {"path": str(out_p), "elapsed_s": dt}
            print(f"  [bible] prop sheet {pid} -> {out_p.name} ({dt}s)")

        # 3. Master empty set plates (rendered ONCE per recurring set+angle)
        for i, st in enumerate(spec_data.get("sets") or []):
            sid = st["id"]
            angle = st.get("camera_angle", "wide")
            key = f"{sid}__{angle}"
            out_p = refs_dir / f"set_{key}.png"
            t0 = time.perf_counter()
            prompt = build_set_plate_prompt(
                st["name"],
                st["description"],
                camera_angle=angle,
                lighting_state=st.get("lighting_state", "warm afternoon daylight"),
                world_style=world_style,
            )
            gen = render_edit(
                model,
                mspec,
                prompt=prompt,
                image_paths=[blank],
                seed=seed + 300 + i * 17,
                width=width,
                height=height,
            )
            save_clean(gen, out_p)
            dt = round(time.perf_counter() - t0, 2)
            manifest["refs"]["sets"][key] = {"path": str(out_p), "elapsed_s": dt}
            print(f"  [bible] master set plate {key} -> {out_p.name} ({dt}s)")

        # 4. Render shots reusing master set plates + canonical refs
        cast_by_id = {c["id"]: c for c in (spec_data.get("cast") or [])}
        props_by_id = {p["id"]: p for p in (spec_data.get("props") or [])}

        for i, shot in enumerate(spec_data.get("shots") or []):
            shot_id = shot["shot_id"]
            set_key = f"{shot['set_id']}__{shot.get('camera_angle', 'wide')}"
            set_plate = Path(manifest["refs"]["sets"][set_key]["path"])

            image_paths: list[Path] = [set_plate]
            cast_refs: list[tuple[int, dict[str, Any]]] = []
            prop_refs: list[tuple[int, str, str]] = []

            single_cache = refs_dir / "_single_views"
            for cid in shot.get("cast_ids") or []:
                cpath = _crop_single_view(Path(manifest["refs"]["cast"][cid]["path"]), single_cache)
                image_paths.append(cpath)
                cast_refs.append((len(image_paths), cast_by_id[cid]))

            for pitem in shot.get("prop_ids") or []:
                pid = pitem["id"] if isinstance(pitem, dict) else pitem
                pstate = pitem.get("state", "") if isinstance(pitem, dict) else ""
                ppath = _crop_single_view(Path(manifest["refs"]["props"][pid]["path"]), single_cache)
                image_paths.append(ppath)
                prop_refs.append((len(image_paths), props_by_id[pid]["name"], pstate))

            prompt = build_set_reuse_shot_prompt(
                set_plate_index=1,
                cast_refs=cast_refs,
                prop_refs=prop_refs,
                action_description=shot["action"],
                camera_note=shot.get("camera_note", ""),
                lighting_state=shot.get("lighting_state", ""),
                locked_camera=bool(shot.get("locked_camera", True)),
                world_style=world_style,
            )
            out_p = shots_dir / f"{shot_id}.png"
            t0 = time.perf_counter()
            gen = render_edit(
                model,
                mspec,
                prompt=prompt,
                image_paths=image_paths,
                seed=seed + 500 + i * 31,
                width=width,
                height=height,
            )
            save_clean(gen, out_p)
            dt = round(time.perf_counter() - t0, 2)
            manifest["shots"].append(
                {
                    "shot_id": shot_id,
                    "path": str(out_p),
                    "set_plate": str(set_plate),
                    "locked_camera": bool(shot.get("locked_camera", True)),
                    "elapsed_s": dt,
                    "prompt": prompt,
                }
            )
            print(f"  [bible] shot {shot_id} (reusing {set_plate.name}) -> {out_p.name} ({dt}s)")
    finally:
        del model
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass

    (out_dir / "bible_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
