"""Local LM Studio vision-language client with structured JSON output, model batching, and timing."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AuthenticationError, OpenAI
from dotenv import load_dotenv
from PIL import Image

FEVER_DREAMS_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(FEVER_DREAMS_ROOT / ".env")
SCRIPTS_DIR = FEVER_DREAMS_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from llm import DEFAULT_LMS_BASE_URL, extract_json, find_lms_bin  # noqa: E402


@dataclass
class VLMCallResult:
    parsed: dict[str, Any]
    raw_text: str
    elapsed_s: float
    used_schema: bool
    parse_error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


def list_lms_models() -> list[dict[str, Any]]:
    """Return metadata for all locally downloaded LM Studio models via `lms ls --json`."""
    lms_bin = find_lms_bin()
    if not lms_bin:
        return []
    proc = subprocess.run([lms_bin, "ls", "--json"], capture_output=True, text=True, check=False)
    try:
        return json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return []


def model_size_gb(model_key: str) -> float:
    for m in list_lms_models():
        if model_key in {m.get("modelKey"), m.get("indexedModelIdentifier"), m.get("path")}:
            return round(float(m.get("sizeBytes") or 0) / (1024**3), 2)
    return 0.0


def unload_all_lms() -> None:
    lms_bin = find_lms_bin()
    if not lms_bin:
        return
    subprocess.run([lms_bin, "unload", "--all"], capture_output=True, text=True, check=False)


def load_lms_model(model_key: str, context_length: int = 16384) -> float:
    """Ensure `model_key` is exclusively loaded in LM Studio and return load time in seconds."""
    lms_bin = find_lms_bin()
    if not lms_bin:
        raise RuntimeError("LM Studio CLI (`lms`) not found.")

    proc = subprocess.run([lms_bin, "ps", "--json"], capture_output=True, text=True, check=False)
    loaded: list[dict] = []
    try:
        loaded = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        pass

    if len(loaded) == 1:
        ident = str(loaded[0].get("identifier") or "")
        mkey = str(loaded[0].get("modelKey") or "")
        ctx = int(loaded[0].get("contextLength") or 0)
        if model_key in {ident, mkey} and ctx >= context_length:
            return 0.0

    t0 = time.perf_counter()
    if loaded:
        subprocess.run([lms_bin, "unload", "--all"], capture_output=True, text=True, check=False)

    res = subprocess.run(
        [
            lms_bin,
            "load",
            model_key,
            "--context-length",
            str(context_length),
            "--identifier",
            model_key,
            "-y",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"lms load {model_key} failed: {(res.stderr or res.stdout).strip()[:500]}")
    return round(time.perf_counter() - t0, 2)


def encode_image_data_url(image_path: Path, max_edge: int = 1024, quality: int = 90) -> str:
    """Encode an image as a base64 JPEG data URL, downscaling only if larger than max_edge."""
    im = Image.open(image_path).convert("RGB")
    if max(im.size) > max_edge:
        scale = max_edge / float(max(im.size))
        im = im.resize(
            (max(1, int(round(im.width * scale))), max(1, int(round(im.height * scale)))),
            Image.Resampling.LANCZOS,
        )
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _clean_json_text(raw: str) -> dict[str, Any]:
    try:
        return extract_json(raw)
    except Exception:
        # Handle trailing commas or stray text around JSON objects
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            candidate = re.sub(r",\s*([}\]])", r"\1", cleaned[start : end + 1])
            return json.loads(candidate)
        raise


def query_vlm_json(
    *,
    model: str,
    system_prompt: str,
    user_text: str,
    images: list[tuple[str, Path]],
    json_schema: dict[str, Any] | None = None,
    schema_name: str = "vlm_response",
    max_edge: int = 1024,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    base_url: str = DEFAULT_LMS_BASE_URL,
    timeout: int = 180,
) -> VLMCallResult:
    """Send text + labeled images to local LM Studio and parse structured JSON."""
    api_key = os.getenv("LM_API_TOKEN") or os.getenv("LM_STUDIO_API_KEY") or "lm-studio"
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    content: list[dict[str, Any]] = []
    for label, path in images:
        content.append({"type": "text", "text": f"[{label}]"})
        content.append({"type": "image_url", "image_url": {"url": encode_image_data_url(path, max_edge=max_edge)}})
    content.append({"type": "text", "text": user_text})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    used_schema = False
    if json_schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": json_schema,
            },
        }
        used_schema = True

    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(**kwargs)
    except AuthenticationError as exc:
        raise RuntimeError(
            "LM Studio rejected the request (HTTP 401 — API token required).\n"
            "Fix: Add your LM Studio token to fever-dreams/.env as LM_API_TOKEN=<your-token> "
            "(LM Studio -> Developer -> Manage Tokens), or turn off 'Require Authentication' "
            "in LM Studio -> Developer -> Server Settings."
        ) from exc
    except Exception:
        if not used_schema:
            raise
        # Fallback if a specific model/backend rejects json_schema
        kwargs.pop("response_format", None)
        used_schema = False
        resp = client.chat.completions.create(**kwargs)

    elapsed_s = round(time.perf_counter() - t0, 2)
    raw_text = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    p_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
    c_tok = int(getattr(usage, "completion_tokens", 0) or 0)

    try:
        parsed = _clean_json_text(raw_text)
        return VLMCallResult(
            parsed=parsed,
            raw_text=raw_text,
            elapsed_s=elapsed_s,
            used_schema=used_schema,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
        )
    except Exception as exc:
        return VLMCallResult(
            parsed={},
            raw_text=raw_text,
            elapsed_s=elapsed_s,
            used_schema=used_schema,
            parse_error=str(exc),
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
        )
