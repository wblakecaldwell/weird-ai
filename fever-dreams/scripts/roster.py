"""Load characters and portrait paths from meme-yearbook/data/roster.json."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

FEVER_DREAMS_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROSTER_PATH = Path(
    os.getenv(
        "FEVER_DREAMS_ROSTER",
        str(FEVER_DREAMS_ROOT.parent / "meme-yearbook" / "data" / "roster.json"),
    )
)


def to_project_relative(path: Path | str | None) -> str | None:
    """Store paths in story.json relative to fever-dreams/ so books stay portable (and free of home paths)."""
    if not path:
        return None
    return Path(os.path.relpath(Path(path).resolve(), FEVER_DREAMS_ROOT)).as_posix()


def resolve_project_path(value: str | None) -> Path | None:
    """Resolve a story.json path (relative to fever-dreams/, or legacy absolute) to an existing file."""
    if not value:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = FEVER_DREAMS_ROOT / p
    p = p.resolve()
    return p if p.exists() else None


@dataclass(frozen=True)
class Character:
    id: str
    name: str
    handle: str
    grade: str
    style_id: str
    style_label: str
    archetype: str
    micro_emotion: str
    visual_description: str
    clubs: list[str]
    superlative: str
    quote: str
    portrait_final: Path | None
    portrait_raw: Path | None
    portrait_thumb: Path | None
    title: str | None = None

    @property
    def best_reference_image(self) -> Path | None:
        """Prefer raw PNG for full uncropped head/shoulders, fall back to final WebP."""
        for p in (self.portrait_raw, self.portrait_final):
            if p and p.exists():
                return p
        return None

    @property
    def best_display_image(self) -> Path | None:
        """Prefer normalized final WebP for display on the Cast page."""
        for p in (self.portrait_final, self.portrait_thumb, self.portrait_raw):
            if p and p.exists():
                return p
        return None

    def to_story_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "handle": self.handle,
            "grade": self.grade,
            "title": self.title,
            "style_id": self.style_id,
            "style_label": self.style_label,
            "archetype": self.archetype,
            "micro_emotion": self.micro_emotion,
            "visual_description": self.visual_description,
            "clubs": list(self.clubs),
            "superlative": self.superlative,
            "quote": self.quote,
            "portrait_raw": to_project_relative(self.portrait_raw),
            "portrait_final": to_project_relative(self.portrait_final),
            "portrait_thumb": to_project_relative(self.portrait_thumb),
        }

    def bio_for_llm(self) -> str:
        role = self.title or f"{self.grade}, Meme Valley High School"
        clubs = ", ".join(self.clubs) if self.clubs else "None"
        return (
            f"- ID: {self.id}\n"
            f"  Name: {self.name} ({role})\n"
            f"  Visual Medium: {self.style_label} ({self.style_id})\n"
            f"  Appearance: {self.visual_description}\n"
            f"  Personality / Archetype: {self.archetype}\n"
            f"  Signature Micro-Emotion: {self.micro_emotion}\n"
            f"  Clubs: {clubs}\n"
            f"  Superlative: {self.superlative}\n"
            f"  Yearbook Quote: \"{self.quote}\""
        )


def _resolve_portrait(yearbook_root: Path, rel: str | None) -> Path | None:
    if not rel:
        return None
    candidate = Path(rel)
    if not candidate.is_absolute():
        candidate = yearbook_root / rel
    return candidate.resolve() if candidate.exists() else None


def _parse_character(raw: dict, yearbook_root: Path) -> Character:
    portraits = raw.get("portraits") or {}
    return Character(
        id=str(raw["id"]),
        name=str(raw["name"]),
        handle=str(raw.get("handle", "")),
        grade=str(raw.get("grade", "Senior")),
        title=raw.get("title"),
        style_id=str(raw.get("style_id", "thick_cartoon_2d")),
        style_label=str(raw.get("style_label", "Mixed Media")),
        archetype=str(raw.get("archetype", "")),
        micro_emotion=str(raw.get("micro_emotion", "")),
        visual_description=str(raw.get("visual_description", "")),
        clubs=list(raw.get("clubs", [])),
        superlative=str(raw.get("superlative", "")),
        quote=str(raw.get("quote", "")),
        portrait_final=_resolve_portrait(yearbook_root, portraits.get("final")),
        portrait_raw=_resolve_portrait(yearbook_root, portraits.get("raw")),
        portrait_thumb=_resolve_portrait(yearbook_root, portraits.get("thumb")),
    )


def load_roster(roster_path: Path = DEFAULT_ROSTER_PATH) -> tuple[dict, list[Character]]:
    """Return (school_meta, all_characters_including_principal)."""
    roster_path = Path(roster_path).resolve()
    if not roster_path.exists():
        raise FileNotFoundError(f"Roster not found at {roster_path}")
    data = json.loads(roster_path.read_text(encoding="utf-8"))
    # Portraits in roster.json are relative to meme-yearbook/ (parent of data/)
    yearbook_root = roster_path.parent.parent

    chars: list[Character] = []
    if data.get("principal"):
        chars.append(_parse_character(data["principal"], yearbook_root))
    for st in data.get("students", []):
        chars.append(_parse_character(st, yearbook_root))
    return data.get("school", {}), chars


def find_character(characters: list[Character], query: str) -> Character:
    """Match a character by exact id, substring of id, or substring of name."""
    q = query.strip().lower()
    for ch in characters:
        if ch.id.lower() == q or ch.name.lower() == q:
            return ch
    matches = [ch for ch in characters if q in ch.id.lower() or q in ch.name.lower()]
    if not matches:
        raise ValueError(f"No character matching {query!r} found in roster.")
    if len(matches) > 1:
        names = ", ".join(f"{m.name} ({m.id})" for m in matches[:6])
        raise ValueError(f"Ambiguous character query {query!r}; matches: {names}")
    return matches[0]


def cast_characters(
    characters: list[Character],
    requested: list[str] | None = None,
    count: int = 3,
    include_principal: bool = False,
    seed: int | None = None,
    diverse_styles: bool = True,
) -> list[Character]:
    """Select a cast for a book, prioritizing characters with portraits and varied mediums."""
    with_art = [c for c in characters if c.best_reference_image is not None]
    if not with_art:
        raise RuntimeError("No characters in roster have portrait images on disk.")

    chosen: list[Character] = []
    chosen_ids: set[str] = set()

    if requested:
        for q in requested:
            ch = find_character(with_art, q)
            if ch.id not in chosen_ids:
                chosen.append(ch)
                chosen_ids.add(ch.id)

    if include_principal:
        for ch in with_art:
            if ch.grade.lower() == "faculty" and ch.id not in chosen_ids:
                chosen.append(ch)
                chosen_ids.add(ch.id)
                break

    target_count = max(count, len(chosen))
    if len(chosen) >= target_count:
        return chosen

    rng = random.Random(seed)
    pool = [c for c in with_art if c.id not in chosen_ids]
    rng.shuffle(pool)

    used_styles = {c.style_id for c in chosen}
    if diverse_styles:
        # First pass: pick characters whose medium isn't already represented
        for ch in pool:
            if len(chosen) >= target_count:
                break
            if ch.style_id not in used_styles:
                chosen.append(ch)
                chosen_ids.add(ch.id)
                used_styles.add(ch.style_id)

    # Second pass if we need more than the number of distinct mediums
    for ch in pool:
        if len(chosen) >= target_count:
            break
        if ch.id not in chosen_ids:
            chosen.append(ch)
            chosen_ids.add(ch.id)

    return chosen
