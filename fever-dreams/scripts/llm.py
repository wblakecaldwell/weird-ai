"""Local LM Studio helpers (`lms chat` CLI default, optional OpenAI-compatible HTTP)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from openai import OpenAI

DEFAULT_LMS_MODEL = os.getenv("LM_STUDIO_MODEL", "qwen3-vl-30b-a3b-instruct-mlx")
# Dense writer model for the writers' room (plotting, prose, reviews)
DEFAULT_WRITER_MODEL = os.getenv("FEVER_WRITER_MODEL", "gemma-4-31b-it")
DEFAULT_CONTEXT_LENGTH = int(os.getenv("FEVER_CONTEXT_LENGTH", "32768"))
DEFAULT_LMS_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")

_ENSURED: set[str] = set()


def ensure_model_loaded(model: str, context_length: int = DEFAULT_CONTEXT_LENGTH) -> None:
    """Load `model` with a large context window if it isn't already loaded (unloading others to save RAM)."""
    if model in _ENSURED:
        return
    lms_bin = find_lms_bin()
    if not lms_bin:
        return
    proc = subprocess.run([lms_bin, "ps", "--json"], capture_output=True, text=True, check=False)
    loaded: list[dict] = []
    try:
        loaded = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        pass
    for m in loaded:
        ident = str(m.get("identifier") or m.get("modelKey") or "")
        ctx = int(m.get("contextLength") or 0)
        if ident == model or str(m.get("modelKey") or "") == model:
            if ctx >= context_length:
                _ENSURED.add(model)
                return
    if loaded:
        subprocess.run([lms_bin, "unload", "--all"], capture_output=True, text=True, check=False)
    print(f"  Loading {model} into LM Studio (context {context_length})...")
    res = subprocess.run(
        [lms_bin, "load", model, "--context-length", str(context_length), "--identifier", model, "-y"],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"lms load {model} failed: {(res.stderr or res.stdout).strip()[:400]}")
    _ENSURED.add(model)


def find_lms_bin() -> str | None:
    found = shutil.which("lms")
    if found:
        return found
    default_bin = Path.home() / ".lmstudio" / "bin" / "lms"
    return str(default_bin) if default_bin.exists() else None


def unload_lms_models() -> None:
    """Free Apple Silicon unified memory from LM Studio before running mflux."""
    lms_bin = find_lms_bin()
    if not lms_bin:
        return
    subprocess.run([lms_bin, "unload", "--all"], capture_output=True, text=True, check=False)


def extract_json(raw: str) -> dict:
    """Parse JSON robustly even if a local LLM wraps it in thinking tags, ANSI codes, or fences."""
    cleaned = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def chat_local_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = DEFAULT_LMS_MODEL,
    backend: str = "cli",
    base_url: str = DEFAULT_LMS_BASE_URL,
    temperature: float = 0.9,
    max_tokens: int = 8192,
    timeout: int = 420,
) -> dict:
    """Send a system + user prompt to local LM Studio and parse the returned JSON object."""
    ensure_model_loaded(model)
    if backend == "cli":
        lms_bin = find_lms_bin()
        if not lms_bin:
            raise RuntimeError(
                "Could not find LM Studio CLI (`lms`). Enable it in LM Studio or pass `--backend http`."
            )
        proc = subprocess.run(
            [
                lms_bin,
                "chat",
                model,
                "--dont-fetch-catalog",
                "-y",
                "-s",
                system_prompt,
                "-p",
                user_prompt,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout).strip()
            raise RuntimeError(f"lms chat failed (exit {proc.returncode}): {err[:400]}")
        raw = proc.stdout or ""
    elif backend == "http":
        api_key = os.getenv("LM_API_TOKEN") or os.getenv("LM_STUDIO_API_KEY") or "lm-studio"
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = resp.choices[0].message.content or ""
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return extract_json(raw)
