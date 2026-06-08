"""Context-aware retrieval query generation for already segmented scripts.

This module changes only the text query sent to retrieval models. Segment ids,
timing, and ground-truth evaluation remain tied to the original script segments.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1"
QUERY_MODES = {"original", "context_window", "llm_expanded", "hybrid_llm"}
LLM_QUERY_MODES = {"llm_expanded", "hybrid_llm"}


@dataclass
class QueryExpansionConfig:
    query_mode: str = "original"
    context_window_size: int = 1
    llm_model: Optional[str] = None
    use_query_cache: bool = True
    force_refresh_expansions: bool = False
    disable_llm_calls: bool = False
    prompt_version: str = PROMPT_VERSION
    provider: Optional[str] = None


class QueryExpansionProvider(Protocol):
    model_name: str

    def expand(self, previous_context: str, current_segment: str, next_context: str) -> str:
        """Return one concise visual sentence for the current segment."""


class MockQueryExpansionProvider:
    """Deterministic provider for tests and offline smoke checks."""

    def __init__(self, model_name: str = "mock-query-expander") -> None:
        self.model_name = model_name
        self.calls: List[Dict[str, str]] = []

    def expand(self, previous_context: str, current_segment: str, next_context: str) -> str:
        self.calls.append({
            "previous_context": previous_context,
            "current_segment": current_segment,
            "next_context": next_context,
        })
        pieces = [current_segment]
        if previous_context:
            pieces.append(f"after {previous_context}")
        if next_context:
            pieces.append(f"before {next_context}")
        return _clean_llm_sentence("; ".join(pieces))


class OllamaQueryExpansionProvider:
    def __init__(self, model_name: str = "llama3.2:3b", ollama_host: str = "http://localhost:11434") -> None:
        self.model_name = model_name
        self.ollama_host = ollama_host.rstrip("/")

    def expand(self, previous_context: str, current_segment: str, next_context: str) -> str:
        import requests

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt(previous_context, current_segment, next_context)},
            ],
            "stream": False,
            "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": 96},
        }
        response = requests.post(f"{self.ollama_host}/api/chat", json=payload, timeout=45)
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        return _clean_llm_sentence(content)


class OpenAICompatibleQueryExpansionProvider:
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI-compatible query expansion provider")

    def expand(self, previous_context: str, current_segment: str, next_context: str) -> str:
        import requests

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt(previous_context, current_segment, next_context)},
            ],
            "temperature": 0.2,
            "max_tokens": 96,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=45,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"].get("content", "")
        return _clean_llm_sentence(content)


def _system_prompt() -> str:
    return (
        "You rewrite one already segmented script line into a short concrete visual retrieval query. "
        "Describe only visible objects, actions, setting, and state for the current segment. "
        "Use neighboring context only to disambiguate vague wording. Do not add unsupported details. "
        "Do not mention narration, explanations, labels, quotes, bullet points, or camera directions unless present. "
        "Return exactly one concise sentence."
    )


def _user_prompt(previous_context: str, current_segment: str, next_context: str) -> str:
    return (
        f"Previous context: {previous_context or '(none)'}\n"
        f"Current segment: {current_segment}\n"
        f"Next context: {next_context or '(none)'}\n\n"
        "Visual retrieval description for the current segment:"
    )


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_llm_sentence(value: str) -> str:
    text = _clean_text(value)
    text = re.sub(r"^(```|json|text)\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^[\-\*\d\.\)\s\"']+", "", text).strip()
    text = text.strip("` \"'")
    text = re.sub(r"\s+", " ", text)
    if "\n" in text:
        text = text.split("\n", 1)[0].strip()
    if len(text) > 280:
        text = text[:280].rsplit(" ", 1)[0].strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _segment_texts(segments: Iterable[Dict[str, Any]]) -> List[str]:
    return [_clean_text(seg.get("original_text", seg.get("text", ""))) for seg in segments]


def script_hash(segments: Iterable[Dict[str, Any]]) -> str:
    payload = json.dumps(_segment_texts(segments), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _context_for(texts: List[str], index: int, window_size: int) -> Tuple[str, str]:
    window = max(0, int(window_size))
    prev = texts[max(0, index - window):index]
    nxt = texts[index + 1:index + 1 + window]
    return " ".join(item for item in prev if item), " ".join(item for item in nxt if item)


def build_context_window_query(current: str, previous_context: str, next_context: str) -> str:
    parts = [f"Current visual target: {current}"]
    if previous_context:
        parts.append(f"Previous context: {previous_context}")
    if next_context:
        parts.append(f"Next context: {next_context}")
    return "\n".join(parts)


def default_provider(config: QueryExpansionConfig) -> QueryExpansionProvider:
    provider_name = (config.provider or os.environ.get("QUERY_EXPANSION_PROVIDER") or "").lower().strip()
    model = config.llm_model or os.environ.get("LLM_MODEL") or "llama3.2:3b"
    if provider_name == "mock":
        return MockQueryExpansionProvider(model)
    if provider_name in {"openai", "openai_compatible"} or os.environ.get("OPENAI_API_KEY"):
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return OpenAICompatibleQueryExpansionProvider(model_name=model, base_url=base_url)
    return OllamaQueryExpansionProvider(model_name=model, ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))


def cache_path_for(cache_root: Path, config: QueryExpansionConfig, segments: List[Dict[str, Any]]) -> Path:
    cache_root = Path(cache_root)
    model = config.llm_model or os.environ.get("LLM_MODEL") or "none"
    key_payload = {
        "query_mode": config.query_mode,
        "context_window_size": int(config.context_window_size),
        "llm_model": model,
        "prompt_version": config.prompt_version,
        "script_hash": script_hash(segments),
    }
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return cache_root / "query_expansions" / f"{config.query_mode}_{key}.json"


def build_query_expansions(
    segments: List[Dict[str, Any]],
    config: Optional[QueryExpansionConfig] = None,
    cache_root: Optional[Path] = None,
    provider: Optional[QueryExpansionProvider] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    config = config or QueryExpansionConfig()
    mode = (config.query_mode or "original").strip().lower()
    if mode not in QUERY_MODES:
        raise ValueError(f"Unsupported query_mode '{config.query_mode}'. Expected one of {sorted(QUERY_MODES)}")
    config.query_mode = mode
    config.context_window_size = max(0, min(3, int(config.context_window_size)))

    original_segments = [dict(seg) for seg in segments]
    texts = _segment_texts(original_segments)
    cache_file = cache_path_for(Path(cache_root or "./cache"), config, original_segments)
    llm_model = config.llm_model or os.environ.get("LLM_MODEL") or ("none" if mode not in LLM_QUERY_MODES else "llama3.2:3b")

    if mode in LLM_QUERY_MODES and config.use_query_cache and not config.force_refresh_expansions and cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as handle:
            cached = json.load(handle)
        return _segments_from_cache(original_segments, cached), _metadata_from_payload(cached, cache_file, cache_used=True)

    if mode in LLM_QUERY_MODES and config.disable_llm_calls:
        raise RuntimeError(f"query_mode={mode} requires LLM calls, but LLM calls are disabled and no usable cache was found: {cache_file}")

    if mode in LLM_QUERY_MODES and provider is None:
        provider = default_provider(config)
        llm_model = getattr(provider, "model_name", llm_model)

    entries: List[Dict[str, Any]] = []
    for idx, current in enumerate(texts):
        previous_context, next_context = _context_for(texts, idx, config.context_window_size)
        expanded = ""
        if mode == "original":
            final_query = current
        elif mode == "context_window":
            final_query = build_context_window_query(current, previous_context, next_context)
        else:
            assert provider is not None
            expanded = provider.expand(previous_context, current, next_context)
            final_query = expanded if mode == "llm_expanded" else f"Original segment: {current}\nVisual retrieval description: {expanded}"
        entries.append({
            "index": idx,
            "original": current,
            "previous_context": previous_context,
            "next_context": next_context,
            "expanded": expanded,
            "final_query": final_query,
        })

    payload = {
        "query_mode": mode,
        "context_window_size": config.context_window_size,
        "llm_model": llm_model,
        "prompt_version": config.prompt_version,
        "script_hash": script_hash(original_segments),
        "cache_used": False,
        "segments": entries,
    }

    if mode in LLM_QUERY_MODES and config.use_query_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    return _segments_from_cache(original_segments, payload), _metadata_from_payload(payload, cache_file, cache_used=False)


def _segments_from_cache(original_segments: List[Dict[str, Any]], payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    by_index = {int(item["index"]): item for item in payload.get("segments", [])}
    result: List[Dict[str, Any]] = []
    for idx, segment in enumerate(original_segments):
        entry = by_index.get(idx, {})
        original = _clean_text(entry.get("original", segment.get("original_text", segment.get("text", ""))))
        final_query = _clean_text(entry.get("final_query", original))
        updated = dict(segment)
        updated["original_text"] = original
        updated["text"] = final_query
        updated["query_expansion"] = entry
        result.append(updated)
    return result


def _metadata_from_payload(payload: Dict[str, Any], cache_file: Path, cache_used: bool) -> Dict[str, Any]:
    return {
        "query_mode": payload.get("query_mode", "original"),
        "context_window_size": payload.get("context_window_size", 1),
        "llm_model": payload.get("llm_model"),
        "prompt_version": payload.get("prompt_version", PROMPT_VERSION),
        "script_hash": payload.get("script_hash"),
        "cache_used": bool(cache_used),
        "cache_path": str(cache_file),
        "final_queries": payload.get("segments", []),
    }


def config_from_values(
    query_mode: str = "original",
    context_window_size: int = 1,
    llm_model: Optional[str] = None,
    use_query_cache: bool = True,
    force_refresh_expansions: bool = False,
    disable_llm_calls: bool = False,
    provider: Optional[str] = None,
) -> QueryExpansionConfig:
    return QueryExpansionConfig(
        query_mode=query_mode or "original",
        context_window_size=context_window_size,
        llm_model=llm_model,
        use_query_cache=use_query_cache,
        force_refresh_expansions=force_refresh_expansions,
        disable_llm_calls=disable_llm_calls,
        provider=provider,
    )


def metadata_summary(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {k: metadata.get(k) for k in [
        "query_mode", "context_window_size", "llm_model", "prompt_version", "cache_used", "cache_path"
    ]}
