import tempfile
from pathlib import Path

from src.query_expansion import (
    MockQueryExpansionProvider,
    QueryExpansionConfig,
    build_query_expansions,
)


def sample_segments():
    return [
        {"text": "Cut the cooked chicken into pieces.", "duration": 1.0},
        {"text": "Perfect for meal prep.", "duration": 1.0},
        {"text": "Store the portions in containers.", "duration": 1.0},
    ]


def test_original_returns_unchanged_segment_text():
    segments, metadata = build_query_expansions(
        sample_segments(),
        QueryExpansionConfig(query_mode="original"),
        cache_root=Path(tempfile.mkdtemp()),
    )
    assert [seg["text"] for seg in segments] == [seg["text"] for seg in sample_segments()]
    assert metadata["query_mode"] == "original"


def test_context_window_includes_previous_current_next_and_boundaries():
    segments, metadata = build_query_expansions(
        sample_segments(),
        QueryExpansionConfig(query_mode="context_window", context_window_size=1),
        cache_root=Path(tempfile.mkdtemp()),
    )
    middle = segments[1]["text"]
    assert "Current visual target: Perfect for meal prep." in middle
    assert "Previous context: Cut the cooked chicken into pieces." in middle
    assert "Next context: Store the portions in containers." in middle
    assert "Previous context" not in segments[0]["text"]
    assert "Next context" not in segments[-1]["text"]
    assert metadata["final_queries"][1]["final_query"] == middle


def test_hybrid_llm_combines_original_and_expansion():
    provider = MockQueryExpansionProvider()
    segments, metadata = build_query_expansions(
        sample_segments(),
        QueryExpansionConfig(query_mode="hybrid_llm", context_window_size=1, llm_model="mock"),
        cache_root=Path(tempfile.mkdtemp()),
        provider=provider,
    )
    assert len(provider.calls) == 3
    assert segments[1]["text"].startswith("Original segment: Perfect for meal prep.")
    assert "Visual retrieval description:" in segments[1]["text"]
    assert metadata["query_mode"] == "hybrid_llm"


def test_cache_prevents_second_llm_call():
    cache_root = Path(tempfile.mkdtemp())
    config = QueryExpansionConfig(query_mode="hybrid_llm", context_window_size=1, llm_model="mock")
    first_provider = MockQueryExpansionProvider()
    build_query_expansions(sample_segments(), config, cache_root=cache_root, provider=first_provider)
    assert len(first_provider.calls) == 3

    second_provider = MockQueryExpansionProvider()
    segments, metadata = build_query_expansions(sample_segments(), config, cache_root=cache_root, provider=second_provider)
    assert len(second_provider.calls) == 0
    assert metadata["cache_used"] is True
    assert segments[1]["text"].startswith("Original segment: Perfect for meal prep.")
