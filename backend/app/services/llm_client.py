import contextlib
import contextvars
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any
from google import genai
from google.genai import errors, types
from pydantic import BaseModel
from backend.app.config import BASE_DIR, is_production_env, settings

CACHE_ROOT_DIR = BASE_DIR / "samples" / ".cache"
LLM_CACHE_DIR = CACHE_ROOT_DIR / "llm"

_llm_cache_override: bool | None = None
_llm_cache_dir_var: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "llm_cache_dir", default=None
)
_llm_cache_stats = {"hits": 0, "misses": 0, "writes": 0, "skipped_invalid": 0}


@contextlib.contextmanager
def scoped_llm_cache_dir(cache_dir: Path | None):
    token = _llm_cache_dir_var.set(cache_dir)
    try:
        yield
    finally:
        _llm_cache_dir_var.reset(token)


def _active_cache_dir() -> Path:
    scoped = _llm_cache_dir_var.get()
    return scoped if scoped is not None else LLM_CACHE_DIR


def set_llm_cache_enabled(enabled: bool) -> None:
    global _llm_cache_override
    _llm_cache_override = enabled


def is_llm_cache_enabled() -> bool:
    if _llm_cache_override is False:
        return False
    if is_production_env():
        return False
    if _llm_cache_override is not None:
        return _llm_cache_override
    return settings.llm_cache_enabled


def reset_llm_cache_stats() -> None:
    for k in _llm_cache_stats:
        _llm_cache_stats[k] = 0


def get_llm_cache_stats() -> dict[str, int]:
    return dict(_llm_cache_stats)


def get_cache_dir_stats(cache_dir: Path | None = None) -> tuple[int, int]:
    target = cache_dir or CACHE_ROOT_DIR
    if not target.exists():
        return 0, 0
    file_count = 0
    total_bytes = 0
    for p in target.rglob("*"):
        if p.is_file():
            file_count += 1
            try:
                total_bytes += p.stat().st_size
            except OSError:
                pass
    return file_count, total_bytes


def format_cache_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def format_cache_summary(cache_dir: Path | None = None) -> str:
    target = cache_dir or CACHE_ROOT_DIR
    files, total_bytes = get_cache_dir_stats(target)
    stats = get_llm_cache_stats()
    mode = "enabled" if is_llm_cache_enabled() else "disabled"
    return (
        f"Cache dir ({target.relative_to(BASE_DIR) if target.is_relative_to(BASE_DIR) else target}): "
        f"{files} files, {format_cache_size(total_bytes)} | "
        f"llm_cache={mode} (hits={stats['hits']}, misses={stats['misses']}, skipped_unparseable={stats['skipped_invalid']})"
    )


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _serialize_schema(response_schema: Any) -> Any:
    if response_schema is None:
        return None
    if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
        return response_schema.model_json_schema()
    if hasattr(response_schema, "model_dump"):
        return response_schema.model_dump()
    if isinstance(response_schema, (dict, list, str, int, float, bool)):
        return response_schema
    return repr(response_schema)


def _build_cache_key(
    prompt: str | list[dict[str, str]],
    system: str | None,
    max_tokens: int,
    temperature: float | None,
    model_name: str,
    response_mime_type: str | None = None,
    response_schema: Any = None,
) -> str:
    payload = {
        "model": model_name,
        "temperature": float(temperature) if temperature is not None else None,
        "max_tokens": max_tokens,
        "system": system,
        "prompt": prompt,
        "response_mime_type": response_mime_type,
        "response_schema": _serialize_schema(response_schema),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expects_json(
    prompt: str | list[dict[str, str]],
    system: str | None,
    response_mime_type: str | None,
    response_schema: Any,
) -> bool:
    if response_schema is not None:
        return True
    if response_mime_type and "json" in response_mime_type.lower():
        return True
    if system and "json" in system.lower():
        return True
    prompt_text = (
        prompt
        if isinstance(prompt, str)
        else "\n".join(msg.get("content", "") for msg in prompt)
    )
    return bool(
        re.search(
            r"(?:only|valid|matching)\s+.*?\bjson\b|\bjson\b.*?(?:schema|format|only)",
            prompt_text,
            re.IGNORECASE,
        )
    )


def _is_cacheable_response(
    text: str,
    prompt: str | list[dict[str, str]],
    system: str | None,
    response_mime_type: str | None,
    response_schema: Any,
) -> bool:
    if not text or not text.strip():
        return False
    expects_json = _expects_json(prompt, system, response_mime_type, response_schema)
    if expects_json:
        cleaned = _strip_code_fences(text)
        try:
            json.loads(cleaned)
        except Exception:
            return False
        if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
            try:
                response_schema.model_validate_json(cleaned)
            except Exception:
                return False
    return True


def invalidate_llm_cache(
    prompt: str | list[dict[str, str]],
    system: str | None = None,
    max_tokens: int = 16384,
    temperature: float | None = None,
    model: str | None = None,
    response_mime_type: str | None = None,
    response_schema: Any = None,
) -> None:
    target_model = model or settings.gemini_model
    if not target_model:
        return
    key = _build_cache_key(
        prompt=prompt,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        model_name=target_model,
        response_mime_type=response_mime_type,
        response_schema=response_schema,
    )
    cache_path = _active_cache_dir() / f"{key}.json"
    if cache_path.exists():
        try:
            cache_path.unlink()
        except OSError:
            pass


def _get_client(model_name: str | None = None) -> tuple[genai.Client, str]:
    if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
        raise RuntimeError("GEMINI_API_KEY is not configured in .env")
    target_model = model_name or settings.gemini_model
    if not target_model:
        raise RuntimeError("GEMINI_MODEL is not configured in .env")
    return genai.Client(
        api_key=settings.gemini_api_key,
        http_options=types.HttpOptions(timeout=30000),
    ), target_model


def call_llm(
    prompt: str | list[dict[str, str]],
    system: str | None = None,
    max_tokens: int = 16384,
    temperature: float | None = None,
    model: str | None = None,
    response_mime_type: str | None = None,
    response_schema: Any = None,
) -> str:
    target_model = model or settings.gemini_model
    if not target_model:
        raise RuntimeError("GEMINI_MODEL is not configured in .env")

    cache_enabled = is_llm_cache_enabled()
    cache_key = _build_cache_key(
        prompt=prompt,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        model_name=target_model,
        response_mime_type=response_mime_type,
        response_schema=response_schema,
    )
    active_dir = _active_cache_dir()
    cache_path = active_dir / f"{cache_key}.json"
    read_paths = [cache_path]
    if active_dir != LLM_CACHE_DIR and not is_production_env():
        read_paths.append(LLM_CACHE_DIR / f"{cache_key}.json")

    if cache_enabled:
        for candidate_path in read_paths:
            if not candidate_path.exists():
                continue
            try:
                entry = json.loads(candidate_path.read_text(encoding="utf-8"))
                cached_text = entry.get("response") if isinstance(entry, dict) else None
                if isinstance(cached_text, str) and _is_cacheable_response(
                    cached_text, prompt, system, response_mime_type, response_schema
                ):
                    _llm_cache_stats["hits"] += 1
                    return cached_text
            except (OSError, ValueError):
                pass

    client, target_model = _get_client(target_model)
    if isinstance(prompt, str):
        contents = prompt
    else:
        contents = [
            types.Content(
                role="model" if msg["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=msg["content"])],
            )
            for msg in prompt
        ]

    config_kwargs: dict = {
        "system_instruction": system,
        "max_output_tokens": max_tokens,
    }
    if temperature is not None:
        config_kwargs["temperature"] = temperature
    if response_mime_type is not None:
        config_kwargs["response_mime_type"] = response_mime_type
    if response_schema is not None:
        config_kwargs["response_schema"] = response_schema
    config = types.GenerateContentConfig(**config_kwargs)

    text = ""
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=target_model,
                contents=contents,
                config=config,
            )
            text = (response.text or "").strip()
            break
        except errors.APIError as err:
            if err.code in (503, 504) or (err.code == 429 and "PerDay" not in str(err)):
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
            raise

    _llm_cache_stats["misses"] += 1
    if cache_enabled:
        if _is_cacheable_response(text, prompt, system, response_mime_type, response_schema):
            try:
                active_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = active_dir / f"{cache_key}.tmp"
                tmp_path.write_text(
                    json.dumps(
                        {
                            "model": target_model,
                            "temperature": float(temperature) if temperature is not None else None,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "response": text,
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                tmp_path.replace(cache_path)
                _llm_cache_stats["writes"] += 1
            except OSError:
                pass
        else:
            _llm_cache_stats["skipped_invalid"] += 1

    return text
