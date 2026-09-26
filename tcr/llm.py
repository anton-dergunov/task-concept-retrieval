"""One entry point for dataset-building LLM calls: `complete_json`.

Never used on the matching hot path (CLAUDE.md hard rule 3). Three backends:

  gemini      AI Studio key GEMINI_API_KEY (free tier; falls back across models on
              quota errors). Free-tier prompts may be used by Google to improve its
              products, so never send private text here.
  vertex      Vertex AI via gcloud ADC; VERTEX_PROJECT / VERTEX_LOCATION in .env.
  claude-cli  Claude Code headless mode (`claude -p`) on the user's subscription,
              tools disabled, structured output via --json-schema.

Every response is cached on disk (.cache/llm/<sha1>.json), keyed by backend, system
prompt, prompt and schema — not by model — so reruns and resumes cost nothing and
a model fallback does not invalidate earlier work.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from typing import Callable, List, Optional, Sequence

from . import config

CACHE_DIR = config.ROOT / ".cache" / "llm"

# Free-tier Gemini models, in fallback order, with a per-request delay derived
# from each model's RPM limit (same list as scripts/describe_icons.py).
GEMINI_MODELS = [
    ("gemini-3.1-flash-lite", 5),
    ("gemini-2.5-flash-lite", 7),
    ("gemini-2.5-flash", 13),
    ("gemini-3-flash", 13),
    ("gemini-3.5-flash", 13),
]
GEMINI_STRONG = [("gemini-3.5-flash", 13), ("gemini-3-flash", 13), ("gemini-2.5-flash", 13)]

PRIVATE_OK = {"vertex", "claude-cli"}   # backends allowed to see private text


class QuotaExceeded(Exception):
    pass


class LLMError(Exception):
    pass


def _cache_path(backend: str, system: str, prompt: str, schema) -> "os.PathLike":
    key = json.dumps([backend, system, prompt, schema], ensure_ascii=False, sort_keys=True)
    return CACHE_DIR / (hashlib.sha1(key.encode("utf-8")).hexdigest() + ".json")


def _is_quota(exc: Exception) -> bool:
    msg = str(exc)
    return getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in msg


def _is_not_found(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 404 or "NOT_FOUND" in str(exc)


def _genai_client(backend: str):
    from dotenv import load_dotenv
    from google import genai
    load_dotenv(config.ROOT / ".env")
    if backend == "gemini":
        return genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return genai.Client(vertexai=True, project=os.environ["VERTEX_PROJECT"],
                        location=os.environ.get("VERTEX_LOCATION", "global"))


_CLIENTS = {}


def _call_genai(backend: str, model: str, system: str, prompt: str, schema, temperature: float) -> str:
    from google.genai import types
    if backend not in _CLIENTS:
        _CLIENTS[backend] = _genai_client(backend)
    cfg = types.GenerateContentConfig(
        temperature=temperature, response_mime_type="application/json",
        system_instruction=system or None,
        response_json_schema=schema if schema else None)
    resp = _CLIENTS[backend].models.generate_content(model=model, contents=prompt, config=cfg)
    return resp.text


def _call_claude(model: str, system: str, prompt: str, schema) -> dict:
    cmd = ["claude", "-p", "--model", model, "--output-format", "json", "--tools", "",
           "--no-session-persistence", "--disable-slash-commands", "--strict-mcp-config",
           "--setting-sources", ""]
    if system:
        cmd += ["--system-prompt", system]
    if schema:
        cmd += ["--json-schema", json.dumps(schema)]
    # Run outside the repo so no project CLAUDE.md or memory is pulled in.
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=tmp,
                              timeout=900)
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise LLMError(f"claude -p failed ({proc.returncode}): {proc.stderr[-500:] or proc.stdout[-500:]}")
    if out.get("is_error"):
        msg = str(out.get("result", ""))
        if "limit" in msg.lower():
            raise QuotaExceeded(msg)
        raise LLMError(msg[:500])
    if out.get("structured_output") is not None:
        return out["structured_output"]
    return _parse_json(out.get("result", ""))


def _parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def complete_json(prompt: str, *, validate: Callable[[object], bool] = lambda d: True,
                  system: str = "", schema: Optional[dict] = None, backend: str = "gemini",
                  models: Optional[Sequence] = None, temperature: float = 0.4,
                  private: bool = False, retries: int = 3, tag: str = "") -> dict:
    """Return {"data": parsed JSON, "model": model id}. Cached on disk.

    `private=True` asserts the prompt contains private text; it is refused on
    backends that may train on prompts. `tag` separates otherwise-identical
    prompts that must be sampled independently (e.g. repeated attacker runs).
    """
    if private and backend not in PRIVATE_OK:
        raise ValueError(f"backend {backend!r} must not receive private text")
    path = _cache_path(backend, system, prompt + ("\n#" + tag if tag else ""), schema)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    if backend == "gemini":
        chain = list(models or GEMINI_MODELS)
    elif backend == "vertex":
        chain = [(m, 1) for m in (models or [os.environ.get("VERTEX_MODEL", "gemini-3.1-pro")])]
    elif backend == "claude-cli":
        chain = [(m, 2) for m in (models or ["sonnet"])]
    else:
        raise ValueError(backend)

    last_err: Optional[Exception] = None
    for model, delay in chain:
        for attempt in range(retries):
            try:
                if backend == "claude-cli":
                    data = _call_claude(model, system, prompt, schema)
                else:
                    data = _parse_json(_call_genai(backend, model, system, prompt, schema, temperature))
                time.sleep(delay)
                if not validate(data):
                    raise LLMError("response failed validation")
                result = {"data": data, "model": model}
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                return result
            except QuotaExceeded as e:
                last_err = e
                break
            except Exception as e:  # noqa: BLE001 — classify, then retry or fall back
                last_err = e
                if _is_quota(e) or _is_not_found(e):
                    print(f"  {model}: {'quota' if _is_quota(e) else 'not found'}; next model")
                    break
                print(f"  {model}: attempt {attempt + 1} failed: {str(e)[:160]}")
                time.sleep(2 * (attempt + 1))
    raise LLMError(f"all models failed; last error: {last_err}")


def batched(items: List, n: int) -> List[List]:
    return [items[i:i + n] for i in range(0, len(items), n)]
