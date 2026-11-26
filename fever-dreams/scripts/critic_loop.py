"""Closed-Loop Maker -> Critic -> Tooling -> Retry Engine (Phase 1 Subsystem).

Orchestrates batched local VLM critique and local mflux repair tooling on Apple Silicon (48 GB RAM):
  1. Audit batch with Critic VLM (single-frame defect check + optional canonical reference anchor check).
     Optionally uses a two-tier cascade (fast screener VLM -> heavy confirmer VLM).
  2. Unload LM Studio models completely before loading mflux.
  3. Invoke our local repair tooling (masked_inpaint with scrubbed mask, targeted_edit, or reroll_seed)
     on every failed item, producing N candidates per round.
  4. Unload mflux, reload Critic VLM, score candidates, select best passing candidate (or best improvement),
     and repeat up to `max_rounds`.
  5. Save structured JSON trace and a visual before / round-by-round / after HTML dashboard.
"""

from __future__ import annotations

import gc
import html
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

FEVER_DREAMS_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = FEVER_DREAMS_ROOT / "scripts"
EVALS_DIR = FEVER_DREAMS_ROOT / "evals"
for p in (SCRIPTS_DIR, EVALS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from edit_page import (  # noqa: E402
    build_bbox_mask,
    render_candidate,
    save_mask_overlay,
    scrub_masked_region,
)
from generate_illustrations import load_edit_model  # noqa: E402
from vlm_client import load_lms_model, query_vlm_json, unload_all_lms  # noqa: E402

# Observe-First schema: VLM inspects face/accessories, limbs/contacts, and reference fidelity BEFORE judging `passed`.
CRITIC_REPAIR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "face_and_accessories_check": {"type": "string"},
        "limbs_and_prop_contact_check": {"type": "string"},
        "reference_match_check": {"type": "string"},
        "defects_found": {
            "type": "array",
            "items": {"type": "string"},
        },
        "passed": {"type": "boolean"},
        "score": {"type": "integer"},
        "defect_type": {
            "type": "string",
            "enum": [
                "none",
                "anatomy",
                "accessory",
                "merge_artifact",
                "character_drift",
                "prop_drift",
                "set_drift",
                "other",
            ],
        },
        "defect_summary": {"type": "string"},
        "location": {"type": "string"},
        "bbox_norm_1000": {
            "type": "array",
            "items": {"type": "integer"},
        },
        "fix_action": {
            "type": "string",
            "enum": ["none", "masked_inpaint", "targeted_edit", "reroll_seed"],
        },
        "fix_instruction": {"type": "string"},
    },
    "required": [
        "face_and_accessories_check",
        "limbs_and_prop_contact_check",
        "reference_match_check",
        "defects_found",
        "passed",
        "score",
        "defect_type",
        "defect_summary",
        "location",
        "bbox_norm_1000",
        "fix_action",
        "fix_instruction",
    ],
    "additionalProperties": False,
}

CRITIC_SYSTEM_PROMPT = """You are a strict visual quality-control (QC) and continuity critic connected directly to an automated local image repair pipeline.

IMPORTANT STYLE CONTEXT:
- Stylized cartoon, clay, felt puppet, knit, and 16-bit pixel-art characters inside painted backgrounds are INTENTIONAL. Do NOT flag an intentional art style as a defect unless it mismatches the provided CANONICAL_REFERENCE.

WHAT YOU MUST FAIL (passed = false):
1. Broken/partial floating accessories (for example, half a pair of glasses stuck to one side of a face).
2. Anatomy errors: extra limbs (3 legs), missing heads, warped/melted hands, or limbs fused into props/beams.
3. Foreign blobs, rocks, or glitches protruding from faces, chins, or mouths.
4. Reference mismatch (when CANONICAL_REFERENCE is provided): wrong prop/vehicle design (e.g. green steam train instead of yellow wheeled crane), wrong character style (e.g. 3D clay instead of 16-bit pixel art), wrong colors, or added horns/glasses.

HOW TO PICK fix_action WHEN passed = false:
- "masked_inpaint": Best for localized glitches (half-glasses loop, extra third leg, chin artifact, or replacing one specific wrong prop) while preserving the rest of the image. Provide a tight-with-margin bounding box in bbox_norm_1000 as [x1, y1, x2, y2] in 0..1000 normalized coordinates!
- "targeted_edit": Best when the fix spans a larger interacting area (e.g. separating a hand merged into a wooden beam and changing a character's pose/expression, or editing the whole character) while keeping the overall scene composition.
- "reroll_seed": Best only when the whole composition is unsalvageable.

HOW TO WRITE fix_instruction:
- Be concrete and visual (describe what SHOULD appear in the repaired area, not just "fix it").
- When CANONICAL_REFERENCE is provided, explicitly reference "Image 2" as the canonical appearance to match.

Return ONLY valid JSON matching the schema."""


@dataclass
class RepairItem:
    item_id: str
    image_path: Path
    description: str = ""
    entity_name: str = ""
    ref_images: list[Path] = field(default_factory=list)
    scene_prompt: str = ""
    # Optional hint bbox (x1,y1,x2,y2 in 0..1 or pixels) ONLY if user/metadata provides one; normally None so Critic finds it!
    known_bbox: str | None = None
    extra_meta: dict[str, Any] = field(default_factory=dict)


def _clear_mlx() -> None:
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def _valid_bbox_1000(raw: Any) -> list[int] | None:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in raw]
    except (TypeError, ValueError):
        return None
    x1, x2 = max(0, min(1000, x1)), max(0, min(1000, x2))
    y1, y2 = max(0, min(1000, y1)), max(0, min(1000, y2))
    if x2 - x1 < 25 or y2 - y1 < 25:
        return None
    if (x2 - x1) > 950 and (y2 - y1) > 950:
        return None
    return [x1, y1, x2, y2]


def _expand_bbox_1000(bbox: list[int], pad: int = 35) -> str:
    x1, y1, x2, y2 = bbox
    nx1 = max(0.0, (x1 - pad) / 1000.0)
    ny1 = max(0.0, (y1 - pad) / 1000.0)
    nx2 = min(1.0, (x2 + pad) / 1000.0)
    ny2 = min(1.0, (y2 + pad) / 1000.0)
    return f"{nx1:.4f},{ny1:.4f},{nx2:.4f},{ny2:.4f}"


def audit_single_image(
    *,
    model: str,
    image_path: Path,
    ref_images: list[Path] | None = None,
    entity_name: str = "",
    prior_defect: str = "",
    max_edge: int = 1024,
) -> dict[str, Any]:
    """Run proven Critic VLM inspection: targeted forensic probes + canonical reference anchor check."""
    t_all = time.perf_counter()
    imgs: list[tuple[str, Path]] = []
    ref_note = ""
    if ref_images:
        for idx, rp in enumerate(ref_images, start=1):
            imgs.append((f"CANONICAL_REFERENCE_{idx}", rp))
        ref_note = (
            f"\nTarget entity to verify against CANONICAL_REFERENCE_1: {entity_name or 'the referenced character/prop'}. "
            f"Check BOTH rendering/anatomy defects AND strict fidelity to CANONICAL_REFERENCE_1 "
            f"(rendering medium/style like flat 2D 16-bit pixel art vs smooth 3D clay, main colors, vehicle/prop design, "
            f"and forbidden added accessories like horns or glasses)."
        )
    imgs.append(("CANDIDATE_IMAGE", image_path))

    prior_note = ""
    if prior_defect:
        prior_note = (
            f"\nNOTE: This image is a repair attempt for a previous defect: '{prior_defect}'. "
            f"Check carefully whether that exact defect is COMPLETELY GONE and NO new defect was introduced."
        )

    user_text = (
        f"Inspect CANDIDATE_IMAGE forensically as a strict QC critic.{ref_note}{prior_note}\n"
        f"Check these 4 specific failure modes directly from the pixels:\n"
        f"1. face_and_accessories_check: Look closely at the left and right edges of every character's face/cheek/temple "
        f"(is there a broken/hollow HALF pair of glasses sticking out on one side only?), and check chins/mouths for weird foreign rocky blobs.\n"
        f"2. limbs_and_prop_contact_check: Count the legs and shoes on every standing character (does anyone have 3 legs or an extra raised shoe on their side?), "
        f"and check whether any hand/arm is unnaturally fused/melted into a wooden beam or solid prop.\n"
        f"3. reference_match_check: If CANONICAL_REFERENCE_1 is shown, does the target entity match its rendering medium/style, colors, silhouette, and vehicle/prop type? (Write 'n/a' if no reference.)\n"
        f"4. defects_found: List ONLY genuine, clearly visible defects from 1-3 (or [] if clean). Do NOT rationalize a broken half-glasses loop, extra 3rd leg, fused hand, or wrong vehicle as intentional.\n"
        f"5. Set passed=(len(defects_found)==0), score (1-5 if defective, 8-10 if clean), defect_type, defect_summary, location, "
        f"bbox_norm_1000 ([x1,y1,x2,y2] in 0..1000 tightly around the defect, or [] if passed), fix_action, and fix_instruction."
    )
    res = query_vlm_json(
        model=model,
        system_prompt=CRITIC_SYSTEM_PROMPT,
        user_text=user_text,
        images=imgs,
        json_schema=CRITIC_REPAIR_SCHEMA,
        schema_name="critic_repair_verdict_v3",
        max_edge=max_edge,
        max_tokens=1024,
    )
    parsed = dict(res.parsed or {})
    defects_list = [
        str(x).strip()
        for x in (parsed.get("defects_found") or [])
        if str(x).strip() and str(x).strip().lower() not in {"none", "n/a", "no defects"}
    ]
    parsed["passed"] = bool(parsed.get("passed", False)) and (len(defects_list) == 0)

    # Secondary check A: If canonical reference exists and primary pass said PASS, run the
    # dedicated Suite-2 Anchor check (proven 100% accuracy / 0% FPR on gemma-4-31b-it & qwen3-vl-4b).
    if parsed["passed"] and ref_images:
        anchor_schema = {
            "type": "object",
            "properties": {
                "passed": {"type": "boolean"},
                "score": {"type": "integer"},
                "mismatch_type": {
                    "type": "string",
                    "enum": ["none", "style_drift", "color_drift", "prop_drift", "accessory_drift", "anatomy"],
                },
                "mismatch_summary": {"type": "string"},
                "fix_instruction": {"type": "string"},
            },
            "required": ["passed", "score", "mismatch_type", "mismatch_summary", "fix_instruction"],
            "additionalProperties": False,
        }
        ares = query_vlm_json(
            model=model,
            system_prompt=(
                "You are a visual continuity supervisor checking whether a target character or prop in TEST_FRAME matches CANONICAL_REFERENCE. "
                "Ignore normal pose/camera/background differences. FAIL (passed=false) if rendering medium/style drifted "
                "(e.g. flat 2D 16-bit pixel art turned into smooth 3D clay), main colors drifted, wrong vehicle/prop design "
                "(e.g. yellow wheeled crane replaced by green steam train), or forbidden horns/glasses were added."
            ),
            user_text=(
                f"Target entity to verify: {entity_name or 'target entity'}.\n"
                f"Compare this entity in TEST_FRAME against CANONICAL_REFERENCE.{prior_note}\n"
                f"Set passed=true ONLY if it faithfully matches CANONICAL_REFERENCE in rendering style, colors, and design."
            ),
            images=[("CANONICAL_REFERENCE", ref_images[0]), ("TEST_FRAME", image_path)],
            json_schema=anchor_schema,
            schema_name="loop_anchor_verify",
            max_edge=max_edge,
            max_tokens=480,
        )
        ap = ares.parsed or {}
        if not bool(ap.get("passed", True)):
            mtype = str(ap.get("mismatch_type") or "character_drift")
            dtype_map = {
                "style_drift": "character_drift",
                "color_drift": "character_drift",
                "prop_drift": "prop_drift",
                "accessory_drift": "accessory",
                "anatomy": "anatomy",
            }
            parsed["passed"] = False
            parsed["score"] = min(int(ap.get("score") or 3), 4)
            parsed["defect_type"] = dtype_map.get(mtype, "character_drift")
            parsed["defect_summary"] = str(ap.get("mismatch_summary") or "Reference mismatch")
            parsed["location"] = entity_name or "target entity"
            parsed["bbox_norm_1000"] = []
            parsed["fix_action"] = "targeted_edit"
            parsed["fix_instruction"] = str(ap.get("fix_instruction") or f"Match CANONICAL_REFERENCE for {entity_name}.")

    # Secondary check B: Forensic probe pass (Suite 1B proven pattern): if still PASS,
    # run targeted high-confidence half-glasses & extra-leg probes so subtle artifacts cannot slip through!
    if parsed["passed"]:
        probe_schema = {
            "type": "object",
            "properties": {
                "observation": {"type": "string"},
                "defect_detected": {"type": "boolean"},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "location": {"type": "string"},
                "bbox_norm_1000": {"type": "array", "items": {"type": "integer"}},
                "fix_instruction": {"type": "string"},
            },
            "required": ["observation", "defect_detected", "confidence", "location", "bbox_norm_1000", "fix_instruction"],
            "additionalProperties": False,
        }
        focused_probes = [
            (
                "accessory",
                "Look ONLY at the left and right edges of every character's face, cheeks, and temples in CANDIDATE_IMAGE. "
                "Ignore complete normal eyeglasses covering BOTH eyes. Is there a BROKEN/HALF glasses frame — a single hollow lens or rim loop "
                "sticking out on ONLY ONE side of a face/cheek with NO matching lens on the other eye? "
                "Set defect_detected=true ONLY with high confidence if you literally see a partial one-sided glasses loop, and return its bbox_norm_1000.",
            ),
            (
                "anatomy",
                "Carefully count the legs, feet, and shoes on EVERY standing character in CANDIDATE_IMAGE. "
                "Does ANY character have THREE legs or an extra raised leg/shoe sticking sideways out of their torso/hip while two other feet stand on the floor? "
                "Set defect_detected=true ONLY with high confidence if you literally count 3 legs/shoes on one character, and return its bbox_norm_1000.",
            ),
            (
                "merge_artifact",
                "Look ONLY at every character's hands and arms where they touch wooden beams or poles in CANDIDATE_IMAGE. "
                "Is any hand or arm physically fused or melted INTO a wooden beam so the limb turns into the wood? "
                "Set defect_detected=true ONLY with high confidence if clearly fused, and return its bbox_norm_1000.",
            ),
        ]
        for dtype, q in focused_probes:
            pres = query_vlm_json(
                model=model,
                system_prompt="You are a literal forensic pixel inspector answering ONE specific artifact check. Never invent defects; only flag high-confidence visible glitches.",
                user_text=q,
                images=[("CANDIDATE_IMAGE", image_path)],
                json_schema=probe_schema,
                schema_name=f"loop_probe_{dtype}",
                max_edge=max_edge,
                max_tokens=360,
            )
            pp = pres.parsed or {}
            if bool(pp.get("defect_detected")) and str(pp.get("confidence", "")).lower() == "high":
                parsed["passed"] = False
                parsed["score"] = min(int(parsed.get("score") or 4), 4)
                parsed["defect_type"] = dtype
                parsed["defect_summary"] = str(pp.get("observation") or f"Detected {dtype} defect")
                parsed["location"] = str(pp.get("location") or "detected region")
                parsed["bbox_norm_1000"] = pp.get("bbox_norm_1000") or []
                parsed["fix_action"] = "masked_inpaint" if dtype in {"accessory", "anatomy"} else "targeted_edit"
                parsed["fix_instruction"] = str(pp.get("fix_instruction") or f"Cleanly remove the {dtype} defect and restore natural appearance.")
                break

    parsed.setdefault("score", 5)
    parsed.setdefault("defect_type", "other" if not parsed["passed"] else "none")
    parsed.setdefault("defect_summary", "parse_error" if res.parse_error else "none")
    parsed.setdefault("location", "none")
    parsed.setdefault("bbox_norm_1000", [])
    parsed.setdefault("fix_action", "none" if parsed["passed"] else "targeted_edit")
    parsed.setdefault("fix_instruction", "none")
    parsed["_meta"] = {
        "model": model,
        "elapsed_s": round(time.perf_counter() - t_all, 2),
        "used_schema": res.used_schema,
        "parse_error": res.parse_error,
    }
    return parsed


def build_repair_prompt(
    instruction: str,
    *,
    has_refs: bool,
    entity_name: str = "",
    is_masked: bool = False,
) -> str:
    clean = instruction.strip().rstrip(".")
    ref_clause = (
        f" Match the exact appearance, rendering medium, colors, and design of {entity_name or 'the target entity'} shown in Image 2."
        if has_refs
        else ""
    )
    if is_masked:
        return (
            f"Seamlessly repair ONLY the masked area of Image 1: {clean}.{ref_clause} "
            f"Match the exact art style, brushwork, lighting, and surrounding background textures of Image 1. "
            f"Zero text, zero artifacts, clean natural edges."
        )
    return (
        f"Edit Image 1 to apply this exact repair: {clean}.{ref_clause} "
        f"Keep everything else in Image 1 identical — same composition, camera angle, background, lighting, and other characters — "
        f"fixing ONLY the described defect cleanly. Zero text, zero speech bubbles, zero watermarks."
    )


def run_closed_loop(
    items: list[RepairItem],
    *,
    work_dir: Path,
    critic_model: str,
    fast_screener_model: str | None = None,
    edit_model_key: str = "qwen-image-2.1-turbo",
    max_rounds: int = 3,
    candidates_per_round: int = 2,
    base_seed: int = 77001,
    context_length: int = 16384,
) -> dict[str, Any]:
    """Execute the batched Maker -> Critic -> Tooling -> Retry loop across all items."""
    work_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # Initialize per-item state
    states: dict[str, dict[str, Any]] = {}
    for item in items:
        idir = work_dir / item.item_id
        idir.mkdir(parents=True, exist_ok=True)
        orig_copy = idir / f"round_00_original{item.image_path.suffix}"
        shutil.copy2(item.image_path, orig_copy)
        states[item.item_id] = {
            "item": item,
            "dir": idir,
            "original_image": orig_copy,
            "current_image": orig_copy,
            "initial_verdict": None,
            "latest_verdict": None,
            "passed": False,
            "rounds": [],
        }

    # --- ROUND 0: Initial Critic Audit (batched by model) ---
    print(f"\n[critic_loop] Round 0 — Initial audit of {len(items)} items with {critic_model}")
    if fast_screener_model and fast_screener_model != critic_model:
        load_lms_model(fast_screener_model, context_length=context_length)
        flagged_ids: list[str] = []
        for item in items:
            st = states[item.item_id]
            v_fast = audit_single_image(
                model=fast_screener_model,
                image_path=st["current_image"],
                ref_images=item.ref_images,
                entity_name=item.entity_name,
            )
            st["fast_screen_verdict"] = v_fast
            if not v_fast["passed"]:
                flagged_ids.append(item.item_id)
            else:
                st["initial_verdict"] = v_fast
                st["latest_verdict"] = v_fast
                st["passed"] = True
        print(f"  Fast screener ({fast_screener_model}) flagged {len(flagged_ids)}/{len(items)} items.")
        confirm_items = [states[iid]["item"] for iid in flagged_ids]
    else:
        confirm_items = list(items)

    if confirm_items:
        load_lms_model(critic_model, context_length=context_length)
        for item in confirm_items:
            st = states[item.item_id]
            v = audit_single_image(
                model=critic_model,
                image_path=st["current_image"],
                ref_images=item.ref_images,
                entity_name=item.entity_name,
            )
            st["initial_verdict"] = v
            st["latest_verdict"] = v
            st["passed"] = bool(v["passed"])
            print(
                f"  [R0] {item.item_id:26s} passed={v['passed']} score={v['score']} "
                f"action={v['fix_action']} | {v['defect_summary'][:80]}"
            )

    unload_all_lms()

    # --- REPAIR ROUNDS 1..max_rounds ---
    for rnum in range(1, max_rounds + 1):
        pending = [st for st in states.values() if not st["passed"]]
        if not pending:
            print(f"\n[critic_loop] All {len(items)} items PASSED before round {rnum}!")
            break

        print(f"\n[critic_loop] === Round {rnum}/{max_rounds} — Repairing {len(pending)} items with {edit_model_key} ===")
        # 1. Load mflux once for the whole batch
        model, spec = load_edit_model(edit_model_key)
        round_candidates: dict[str, dict[str, Any]] = {}

        for idx_i, st in enumerate(pending):
            item: RepairItem = st["item"]
            verdict = st["latest_verdict"]
            src_img: Path = st["current_image"]
            im_size = Image.open(src_img).size
            # Align dimensions to multiples of 32 (required by Qwen-Image-2.1 and compatible with Flux2/Flux3)
            width = max(256, (im_size[0] // 32) * 32)
            height = max(256, (im_size[1] // 32) * 32)

            action = str(verdict.get("fix_action") or "targeted_edit")
            if action == "none":
                action = "targeted_edit"
            instruction = str(verdict.get("fix_instruction") or "").strip()
            if not instruction or instruction.lower() == "none":
                instruction = f"Fix this visual defect cleanly: {verdict.get('defect_summary', 'repair glitch')}"

            # Resolve mask if masked_inpaint and valid bbox exists
            resolved_mask = None
            mask_overlay_path: Path | None = None
            cond_img = src_img
            bbox_norm = _valid_bbox_1000(verdict.get("bbox_norm_1000"))

            if action == "masked_inpaint" and (bbox_norm is not None or item.known_bbox is not None):
                bbox_str = _expand_bbox_1000(bbox_norm) if bbox_norm is not None else str(item.known_bbox)
                resolved_mask = build_bbox_mask(bbox_str, width, height, feather_px=12)
                mask_overlay_path = st["dir"] / f"round_{rnum:02d}_mask_overlay.png"
                save_mask_overlay(src_img, resolved_mask, mask_overlay_path)
                # Pre-scrub artifact pixels so conditioning latents do not reproduce the glitch
                scrubbed = st["dir"] / f"round_{rnum:02d}_scrubbed.png"
                cond_img = scrub_masked_region(src_img, resolved_mask, scrubbed)
            elif action == "masked_inpaint":
                # Fallback to targeted_edit if Critic didn't return a usable bbox
                action = "targeted_edit"

            # Prepare image_paths for mflux
            if action == "reroll_seed" and item.ref_images:
                img_paths = list(item.ref_images)
                prompt = (
                    item.scene_prompt
                    + f" IMPORTANT FIX: {instruction}."
                    if item.scene_prompt
                    else build_repair_prompt(instruction, has_refs=True, entity_name=item.entity_name)
                )
                source_for_blend = None
                resolved_mask = None
            else:
                img_paths = [cond_img] + list(item.ref_images)
                prompt = build_repair_prompt(
                    instruction,
                    has_refs=bool(item.ref_images),
                    entity_name=item.entity_name,
                    is_masked=(resolved_mask is not None),
                )
                source_for_blend = src_img if resolved_mask is not None else None

            cands: list[dict[str, Any]] = []
            for c_idx in range(1, candidates_per_round + 1):
                seed = base_seed + rnum * 1000 + idx_i * 97 + c_idx * 13
                t0 = time.perf_counter()
                out_im = render_candidate(
                    model,
                    spec,
                    prompt=prompt,
                    image_paths=img_paths,
                    seed=seed,
                    width=width,
                    height=height,
                    steps=None,
                    guidance=None,
                    strength=None,
                    resolved_mask=resolved_mask,
                    source_for_blend=source_for_blend,
                )
                dt = round(time.perf_counter() - t0, 2)
                out_path = st["dir"] / f"round_{rnum:02d}_cand_{c_idx:02d}_s{seed}.png"
                out_im.save(out_path, format="PNG")
                cands.append({"idx": c_idx, "seed": seed, "path": out_path, "elapsed_s": dt})
                print(f"  [R{rnum} tool:{action}] {item.item_id} cand #{c_idx} ({width}x{height}) in {dt}s")

            round_candidates[item.item_id] = {
                "round": rnum,
                "action_used": action,
                "instruction": instruction,
                "prompt": prompt,
                "mask_overlay": mask_overlay_path,
                "candidates": cands,
            }

        # Unload mflux completely before reloading Critic VLM
        del model
        _clear_mlx()

        # 2. Re-audit all generated candidates with Critic VLM
        print(f"\n[critic_loop] Round {rnum} — Re-auditing candidates with {critic_model}")
        load_lms_model(critic_model, context_length=context_length)

        for st in pending:
            item = st["item"]
            rc = round_candidates[item.item_id]
            prior_def = str(st["latest_verdict"].get("defect_summary") or "")
            best_cand = None
            best_key = (-1, -1)

            for c in rc["candidates"]:
                v = audit_single_image(
                    model=critic_model,
                    image_path=c["path"],
                    ref_images=item.ref_images,
                    entity_name=item.entity_name,
                    prior_defect=prior_def,
                )
                c["verdict"] = v
                rank_key = (1 if v["passed"] else 0, int(v.get("score") or 0))
                if rank_key > best_key:
                    best_key = rank_key
                    best_cand = c
                print(
                    f"  [R{rnum} audit] {item.item_id} cand #{c['idx']} passed={v['passed']} "
                    f"score={v['score']} | {v['defect_summary'][:75]}"
                )

            assert best_cand is not None
            rc["selected_idx"] = best_cand["idx"]
            rc["selected_path"] = best_cand["path"]
            rc["selected_verdict"] = best_cand["verdict"]
            st["rounds"].append(rc)

            # Advance current_image if candidate passed or improved/matched score
            prev_score = int(st["latest_verdict"].get("score") or 0)
            new_score = int(best_cand["verdict"].get("score") or 0)
            if best_cand["verdict"]["passed"] or new_score >= prev_score:
                st["current_image"] = best_cand["path"]
                st["latest_verdict"] = best_cand["verdict"]
                st["passed"] = bool(best_cand["verdict"]["passed"])

        unload_all_lms()

    # Build summary & serializable report
    total_s = round(time.perf_counter() - t_start, 2)
    serialized_items: list[dict[str, Any]] = []
    initial_failed = 0
    recovered = 0
    score_deltas: list[int] = []

    for item in items:
        st = states[item.item_id]
        init_v = st["initial_verdict"] or {}
        final_v = st["latest_verdict"] or {}
        init_passed = bool(init_v.get("passed", False))
        final_passed = bool(st["passed"])
        if not init_passed:
            initial_failed += 1
            if final_passed:
                recovered += 1
        delta = int(final_v.get("score") or 0) - int(init_v.get("score") or 0)
        score_deltas.append(delta)

        rounds_ser = []
        for rc in st["rounds"]:
            rounds_ser.append(
                {
                    "round": rc["round"],
                    "action_used": rc["action_used"],
                    "instruction": rc["instruction"],
                    "mask_overlay": str(rc["mask_overlay"].relative_to(work_dir)) if rc["mask_overlay"] else None,
                    "selected_idx": rc["selected_idx"],
                    "selected_path": str(rc["selected_path"].relative_to(work_dir)),
                    "selected_verdict": rc["selected_verdict"],
                    "candidates": [
                        {
                            "idx": c["idx"],
                            "seed": c["seed"],
                            "path": str(c["path"].relative_to(work_dir)),
                            "elapsed_s": c["elapsed_s"],
                            "verdict": c["verdict"],
                        }
                        for c in rc["candidates"]
                    ],
                }
            )

        def _rel_repo(p: Path) -> str:
            try:
                return p.resolve().relative_to(FEVER_DREAMS_ROOT).as_posix()
            except ValueError:
                return p.name

        serialized_items.append(
            {
                "item_id": item.item_id,
                "description": item.description,
                "entity_name": item.entity_name,
                "ref_images": [_rel_repo(p) for p in item.ref_images],
                "original_image": str(st["original_image"].relative_to(work_dir)),
                "final_image": str(st["current_image"].relative_to(work_dir)),
                "initial_passed": init_passed,
                "final_passed": final_passed,
                "recovered": (not init_passed) and final_passed,
                "score_delta": delta,
                "rounds_run": len(st["rounds"]),
                "initial_verdict": init_v,
                "final_verdict": final_v,
                "rounds": rounds_ser,
            }
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "critic_model": critic_model,
        "fast_screener_model": fast_screener_model,
        "edit_model": edit_model_key,
        "max_rounds": max_rounds,
        "candidates_per_round": candidates_per_round,
        "total_items": len(items),
        "initial_detected_bad": initial_failed,
        "recovered_to_pass": recovered,
        "recovery_rate": round(recovered / max(1, initial_failed), 4),
        "avg_score_delta": round(sum(score_deltas) / max(1, len(score_deltas)), 2),
        "total_elapsed_s": total_s,
        "items": serialized_items,
    }
    (work_dir / "loop_log.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_recovery_html(summary, work_dir, work_dir / "report.html")
    return summary


def write_recovery_html(summary: dict[str, Any], work_dir: Path, out_path: Path) -> None:
    blocks = []
    for it in summary["items"]:
        status_cls = "ok" if it["final_passed"] else "warn"
        status_txt = "RECOVERED -> PASS" if it["recovered"] else ("PASSED" if it["final_passed"] else "NEEDS HUMAN REVIEW")
        iv = it["initial_verdict"]
        fv = it["final_verdict"]

        ref_html = ""
        if it["ref_images"]:
            ref_p = Path(it["ref_images"][0])
            if not ref_p.is_absolute():
                ref_p = FEVER_DREAMS_ROOT / ref_p
            rel_ref = os.path.relpath(ref_p, out_path.parent)
            ref_html = (
                f'<div class="tile"><div class="lbl">Canonical Reference</div>'
                f'<img src="{html.escape(rel_ref)}"><div class="cap">{html.escape(it["entity_name"])}</div></div>'
            )

        round_tiles = []
        for rc in it["rounds"]:
            if rc["mask_overlay"]:
                round_tiles.append(
                    f'<div class="tile"><div class="lbl">Round {rc["round"]} Auto-Mask</div>'
                    f'<img src="{html.escape(rc["mask_overlay"])}">'
                    f'<div class="cap">Tool: <code>{html.escape(rc["action_used"])}</code></div></div>'
                )
            for c in rc["candidates"]:
                cv = c["verdict"]
                sel = " selected" if c["idx"] == rc["selected_idx"] else ""
                pcls = "ok" if cv.get("passed") else "bad"
                round_tiles.append(
                    f'<div class="tile{sel}"><div class="lbl">R{rc["round"]} Cand #{c["idx"]} ({c["elapsed_s"]}s)</div>'
                    f'<img src="{html.escape(c["path"])}">'
                    f'<div class="cap"><span class="pill {pcls}">{"PASS" if cv.get("passed") else "FAIL"} {cv.get("score")}/10</span> '
                    f'{html.escape(str(cv.get("defect_summary",""))[:120])}</div></div>'
                )

        blocks.append(
            f"""<section class="item-card">
              <div class="head">
                <div>
                  <h2>{html.escape(it['item_id'])}</h2>
                  <p class="desc">{html.escape(it['description'])}</p>
                </div>
                <span class="pill {status_cls}">{status_txt} (score {iv.get('score','?')} &rarr; {fv.get('score','?')}, delta {it['score_delta']:+d})</span>
              </div>
              <div class="diag">
                <strong>Round 0 Critic Diagnosis:</strong> [{html.escape(str(iv.get('defect_type','')))}] {html.escape(str(iv.get('defect_summary','')))}<br>
                <strong>Critic Tool Call:</strong> <code>{html.escape(str(iv.get('fix_action','')))}</code> &mdash; {html.escape(str(iv.get('fix_instruction','')))}
              </div>
              <div class="strip">
                {ref_html}
                <div class="tile orig">
                  <div class="lbl">BEFORE (Original Bad Output)</div>
                  <img src="{html.escape(it['original_image'])}">
                  <div class="cap"><span class="pill bad">FAIL {iv.get('score','?')}/10</span></div>
                </div>
                <div class="tile final">
                  <div class="lbl">AFTER (Final Selected — {it['rounds_run']} round(s))</div>
                  <img src="{html.escape(it['final_image'])}">
                  <div class="cap"><span class="pill {status_cls}">{"PASS" if it["final_passed"] else "FAIL"} {fv.get('score','?')}/10</span> {html.escape(str(fv.get('defect_summary',''))[:120])}</div>
                </div>
              </div>
              <details>
                <summary>Show all round-by-round tool calls &amp; candidates ({len(it['rounds'])} rounds)</summary>
                <div class="strip small">{"".join(round_tiles)}</div>
              </details>
            </section>"""
        )

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Fever Dreams — Closed-Loop Recovery Report</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #141210; color: #f2eadc; margin: 0; padding: 24px 28px 60px; }}
  h1 {{ margin: 0 0 6px; font-family: Georgia, serif; color: #f4d378; }}
  h2 {{ margin: 0 0 4px; font-size: 19px; color: #f4d378; }}
  .meta {{ color: #b7aa95; font-size: 13px; margin-bottom: 18px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .kpi {{ background: #1d1915; border: 1px solid #342c24; border-radius: 10px; padding: 14px; }}
  .kpi .num {{ font-size: 26px; font-weight: 700; color: #f4d378; }}
  .kpi .k {{ font-size: 12px; color: #b7aa95; }}
  .item-card {{ background: #1d1915; border: 1px solid #342c24; border-radius: 12px; padding: 18px; margin-bottom: 22px; }}
  .head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; flex-wrap: wrap; }}
  .desc {{ margin: 0 0 10px; font-size: 13px; color: #c7bba6; }}
  .diag {{ background: #26201a; border-radius: 8px; padding: 10px 12px; font-size: 13px; line-height: 1.45; margin-bottom: 14px; }}
  .strip {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
  .strip.small {{ grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); margin-top: 12px; }}
  .tile {{ background: #15120f; border: 2px solid #332b23; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; }}
  .tile.orig {{ border-color: #8b3b3b; }}
  .tile.final {{ border-color: #3e8e5c; }}
  .tile.selected {{ border-color: #f4d378; }}
  .lbl {{ background: #241e18; padding: 6px 10px; font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: #e8cfa0; }}
  .tile img {{ width: 100%; max-height: 360px; object-fit: contain; background: #0d0b0a; }}
  .cap {{ padding: 8px 10px; font-size: 12px; color: #cfc3ae; line-height: 1.35; }}
  .pill {{ display: inline-block; padding: 3px 9px; border-radius: 999px; font-size: 12px; font-weight: 700; }}
  .pill.ok {{ background: #1e3b28; color: #8ff0b0; border: 1px solid #2e6942; }}
  .pill.warn {{ background: #3f281a; color: #f6b27c; border: 1px solid #7a4522; }}
  .pill.bad {{ background: #3b1c1c; color: #f69696; border: 1px solid #7a2e2e; }}
  details {{ margin-top: 12px; }}
  summary {{ cursor: pointer; color: #e8cfa0; font-size: 13px; font-weight: 600; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: #f4d378; }}
</style>
</head>
<body>
  <h1>Fever Dreams Phase 1 — Closed-Loop Recovery Eval (Suite 6)</h1>
  <div class="meta">Critic: <code>{html.escape(summary['critic_model'])}</code> &bull; Repair Tool: <code>{html.escape(summary['edit_model'])}</code> &bull; Max Rounds: {summary['max_rounds']} &bull; Total Time: {summary['total_elapsed_s']}s</div>
  <div class="kpis">
    <div class="kpi"><div class="num">{summary['initial_detected_bad']} / {summary['total_items']}</div><div class="k">Bad Outputs Detected at Round 0</div></div>
    <div class="kpi"><div class="num">{summary['recovered_to_pass']} / {summary['initial_detected_bad']} ({summary['recovery_rate']*100:.0f}%)</div><div class="k">Recovered to PASS</div></div>
    <div class="kpi"><div class="num">{summary['avg_score_delta']:+.2f}</div><div class="k">Mean Critic Score Delta</div></div>
    <div class="kpi"><div class="num">{summary['total_elapsed_s']}s</div><div class="k">Total Loop Wall Time</div></div>
  </div>
  {"".join(blocks)}
</body>
</html>"""
    out_path.write_text(doc, encoding="utf-8")
