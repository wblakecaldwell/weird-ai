"""One-command pipeline: Cast & write story -> unload LM Studio -> render art -> build HTML book.

Usage:
    uv run python scripts/make_book.py
    uv run python scripts/make_book.py --cast chester marisol winston --pages 12 --model flux2-klein-4b
    uv run python scripts/make_book.py --cast gavin beverly --sheets --model qwen-image-2.1-turbo
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from roster import FEVER_DREAMS_ROOT

SCRIPTS_DIR = Path(__file__).resolve().parent
BOOKS_DIR = FEVER_DREAMS_ROOT / "books"


def run_step(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=FEVER_DREAMS_ROOT)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def newest_book_dir(before: set[Path]) -> Path:
    after = set(BOOKS_DIR.glob("*")) if BOOKS_DIR.exists() else set()
    created = [p for p in after - before if (p / "story.json").exists()]
    if created:
        return max(created, key=lambda p: p.stat().st_mtime)
    candidates = [p for p in after if (p / "story.json").exists()]
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end local Little Fever Dream Book generator.")
    parser.add_argument("--cast", nargs="+", default=None, help="Character names/ids to star")
    parser.add_argument("--cast-size", type=int, default=3)
    parser.add_argument("--include-principal", action="store_true")
    parser.add_argument("--pages", type=int, default=12)
    parser.add_argument("--premise", type=str, default=None)
    parser.add_argument("--llm-model", type=str, default=None, help="Override LM Studio model")
    parser.add_argument("--model", type=str, default="qwen-image-2.1-turbo", help="mflux multi-ref edit model")
    parser.add_argument("--sheets", action="store_true", help="Generate full-body turnaround sheets first")
    parser.add_argument("--story-only", action="store_true", help="Write story + build draft HTML without rendering art")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    before = set(BOOKS_DIR.glob("*")) if BOOKS_DIR.exists() else set()

    story_cmd = [sys.executable, str(SCRIPTS_DIR / "generate_story.py"), "--cast-size", str(args.cast_size), "--pages", str(args.pages)]
    if args.cast:
        story_cmd += ["--cast", *args.cast]
    if args.include_principal:
        story_cmd.append("--include-principal")
    if args.premise:
        story_cmd += ["--premise", args.premise]
    if args.llm_model:
        story_cmd += ["--model", args.llm_model]
    if args.seed is not None:
        story_cmd += ["--seed", str(args.seed)]

    run_step(story_cmd)
    book_dir = newest_book_dir(before)

    if not args.story_only:
        art_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "generate_illustrations.py"),
            str(book_dir / "story.json"),
            "--model",
            args.model,
        ]
        if args.sheets:
            art_cmd.append("--sheets")
        run_step(art_cmd)

    run_step([sys.executable, str(SCRIPTS_DIR / "build_book.py"), str(book_dir)])
    print(f"\nBook ready! Open it with:\n  open {book_dir / 'index.html'}")


if __name__ == "__main__":
    main()
