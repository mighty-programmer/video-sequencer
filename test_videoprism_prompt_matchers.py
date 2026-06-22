import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from matching import (
    EnsembleVideoTextMatcher,
    LLMEnsembleVideoTextMatcher,
    PromptedVideoTextMatcher,
    VideoTextMatcher,
)


def _fake_base_batch(self, texts):
    self.recorded_batches.append(list(texts))
    return np.array(
        [[float(index + 1), 1.0] for index, _ in enumerate(texts)],
        dtype=np.float32,
    )


def test_prompted_videoprism_matcher_applies_template_to_batch_queries():
    matcher = object.__new__(PromptedVideoTextMatcher)
    matcher.prompt_template = "a video of {}"
    matcher.recorded_batches = []

    with patch.object(VideoTextMatcher, "get_text_embeddings_batch", _fake_base_batch):
        embeddings = matcher.get_text_embeddings_batch(["cats", "dogs"])

    assert matcher.recorded_batches == [["a video of cats", "a video of dogs"]]
    assert embeddings.shape == (2, 2)


def test_prompted_videoprism_single_query_does_not_double_apply_template():
    matcher = object.__new__(PromptedVideoTextMatcher)
    matcher.prompt_template = "a video of {}"
    matcher.recorded_batches = []

    with patch.object(VideoTextMatcher, "get_text_embeddings_batch", _fake_base_batch):
        matcher.get_text_embedding("cats")

    assert matcher.recorded_batches == [["a video of cats"]]


def test_ensemble_videoprism_matcher_batches_flattened_prompt_templates():
    matcher = object.__new__(EnsembleVideoTextMatcher)
    matcher.ensemble_templates = ["{}", "a video of {}"]
    matcher.recorded_batches = []

    with patch.object(VideoTextMatcher, "get_text_embeddings_batch", _fake_base_batch):
        embeddings = matcher.get_text_embeddings_batch(["cats", "dogs"])

    assert matcher.recorded_batches == [["cats", "a video of cats", "dogs", "a video of dogs"]]
    assert embeddings.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), np.ones(2), atol=1e-6)


def test_llm_ensemble_videoprism_matcher_batches_segment_specific_prompts():
    matcher = object.__new__(LLMEnsembleVideoTextMatcher)
    matcher.llm_prompts = {
        "cats": ["cats", "close-up video of cats"],
        "dogs": ["dogs"],
    }
    matcher.recorded_batches = []

    with patch.object(VideoTextMatcher, "get_text_embeddings_batch", _fake_base_batch):
        embeddings = matcher.get_text_embeddings_batch(["cats", "dogs"])

    assert matcher.recorded_batches == [["cats", "close-up video of cats"], ["dogs"]]
    assert embeddings.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(embeddings[0]), 1.0, atol=1e-6)
