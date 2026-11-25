"""Append-only per-book render log for the cartoon pipeline.

Each step (voices, animate, assemble) appends a run to books/<slug>/cartoon/render_log.json with its
settings and one item per rendered unit (line, clip, shot), saved after every item so a crashed or
interrupted run still leaves a record. Paths are stored relative to the cartoon dir, never absolute.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RenderLog:
    def __init__(self, cartoon_dir: Path, step: str, settings: dict):
        self.path = cartoon_dir / "render_log.json"
        self.data = {"runs": []}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.data = {"runs": []}
        self.run = {
            "step": step,
            "started_at": _now(),
            "finished_at": None,
            "argv": [Path(sys.argv[0]).name, *sys.argv[1:]],
            "python": platform.python_version(),
            "settings": settings,
            "items": [],
        }
        self.data.setdefault("runs", []).append(self.run)
        self.save()

    def add(self, **item) -> None:
        item.setdefault("at", _now())
        self.run["items"].append(item)
        self.save()

    def finish(self, **summary) -> None:
        self.run["finished_at"] = _now()
        if summary:
            self.run["summary"] = summary
        self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
