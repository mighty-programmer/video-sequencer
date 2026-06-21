import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from matching import VideoTextMatcher, _compute_assignment_similarity_matrix


class MatcherWithNormalization:
    def __init__(self):
        self.kwargs = None

    def compute_similarity_matrix(
        self,
        script_segments,
        match_only=False,
        use_dual_softmax=False,
        temperature=0.05,
        score_normalization="none",
        csls_k=5,
    ):
        self.kwargs = {
            "script_segments": script_segments,
            "match_only": match_only,
            "use_dual_softmax": use_dual_softmax,
            "temperature": temperature,
            "score_normalization": score_normalization,
            "csls_k": csls_k,
        }
        return np.array([[1.0]], dtype=np.float32), ["clip_a"]


class MatcherWithoutNormalization:
    def __init__(self):
        self.kwargs = None

    def compute_similarity_matrix(self, script_segments, match_only=False):
        self.kwargs = {
            "script_segments": script_segments,
            "match_only": match_only,
        }
        return np.array([[1.0]], dtype=np.float32), ["clip_a"]


def test_assignment_matrix_forwards_normalization_kwargs_when_supported():
    matcher = MatcherWithNormalization()
    matrix, metadata = _compute_assignment_similarity_matrix(
        matcher,
        [{"text": "a segment"}],
        match_only=True,
        use_dual_softmax=True,
        dual_softmax_temp=0.07,
        score_normalization="csls",
        csls_k=3,
    )

    assert matrix.shape == (1, 1)
    assert metadata == ["clip_a"]
    assert matcher.kwargs["match_only"] is True
    assert matcher.kwargs["use_dual_softmax"] is True
    assert matcher.kwargs["temperature"] == 0.07
    assert matcher.kwargs["score_normalization"] == "csls"
    assert matcher.kwargs["csls_k"] == 3


def test_assignment_matrix_ignores_normalization_kwargs_when_unsupported():
    matcher = MatcherWithoutNormalization()
    matrix, metadata = _compute_assignment_similarity_matrix(
        matcher,
        [{"text": "a segment"}],
        match_only=True,
        use_dual_softmax=True,
        score_normalization="csls",
        csls_k=3,
    )

    assert matrix.shape == (1, 1)
    assert metadata == ["clip_a"]
    assert matcher.kwargs == {
        "script_segments": [{"text": "a segment"}],
        "match_only": True,
    }


def test_csls_normalization_penalizes_broad_hubs_without_labels():
    matrix = np.array(
        [
            [0.90, 0.80, 0.10],
            [0.85, 0.20, 0.10],
        ],
        dtype=np.float32,
    )

    normalized = VideoTextMatcher._apply_csls_normalization(matrix, k=1)
    expected = np.array(
        [
            [0.00, -0.10, -0.80],
            [-0.05, -1.25, -0.75],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(normalized, expected, atol=1e-6)
