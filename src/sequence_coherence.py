"""
Sequence-coherence reranking for VideoPrism retrieval results.

This module adds an optional beam-search assignment layer on top of an
already-computed segment-to-clip similarity matrix. It never encodes videos or
text itself; callers pass the similarity matrix and indexed video embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class BeamItem:
    assignments: List[int]
    used: set[int]
    semantic_score: float
    coherence_score: float
    combined_score: float
    transitions: List[Dict[str, Any]]


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    values = matrix.astype(np.float32, copy=True)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    if max_value - min_value < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_value) / (max_value - min_value)


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    vectors = embeddings.astype(np.float32, copy=True)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def _top_candidates(row: np.ndarray, top_k: int) -> List[int]:
    if row.size == 0:
        return []
    count = min(max(1, int(top_k)), row.size)
    return np.argsort(row)[::-1][:count].astype(int).tolist()


def coherence_beam_search_assignment(
    similarity_matrix: np.ndarray,
    clip_ids: Sequence[str],
    video_embeddings: np.ndarray,
    top_k: int = 5,
    beam_size: int = 10,
    lambda_coherence: float = 0.1,
    allow_reuse: bool = False,
    normalize_scores: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Assign one clip to each segment using semantic score plus transition coherence.

    Returns:
      assignments: selected clip ids, one per segment
      diagnostics: scores, transition details, selected indices, and parameters
    """
    if similarity_matrix is None or similarity_matrix.size == 0:
        return [], {
            "assignment_method": "coherence_beam",
            "error": "empty similarity matrix",
        }

    matrix = np.asarray(similarity_matrix, dtype=np.float32)
    embeddings = np.asarray(video_embeddings, dtype=np.float32)
    num_segments, num_clips = matrix.shape
    if len(clip_ids) != num_clips:
        raise ValueError(f"clip_ids length {len(clip_ids)} does not match matrix width {num_clips}")
    if embeddings.shape[0] != num_clips:
        raise ValueError(f"video_embeddings rows {embeddings.shape[0]} do not match matrix width {num_clips}")

    scoring_matrix = _normalize_matrix(matrix) if normalize_scores else matrix.copy()
    normalized_embeddings = _normalize_embeddings(embeddings)
    top_k = max(1, int(top_k))
    beam_size = max(1, int(beam_size))
    lambda_coherence = float(lambda_coherence)

    candidate_lists = [_top_candidates(scoring_matrix[i], top_k) for i in range(num_segments)]
    beams: List[BeamItem] = []

    for clip_idx in candidate_lists[0]:
        semantic = float(scoring_matrix[0, clip_idx])
        beams.append(
            BeamItem(
                assignments=[clip_idx],
                used={clip_idx},
                semantic_score=semantic,
                coherence_score=0.0,
                combined_score=semantic,
                transitions=[],
            )
        )
    beams.sort(key=lambda item: item.combined_score, reverse=True)
    beams = beams[:beam_size]

    for seg_idx in range(1, num_segments):
        expanded: List[BeamItem] = []
        candidate_pool = candidate_lists[seg_idx]
        for beam in beams:
            valid_candidates = [idx for idx in candidate_pool if allow_reuse or idx not in beam.used]
            if not valid_candidates and not allow_reuse:
                valid_candidates = [idx for idx in np.argsort(scoring_matrix[seg_idx])[::-1].astype(int).tolist() if idx not in beam.used]
            for clip_idx in valid_candidates:
                prev_idx = beam.assignments[-1]
                cosine = float(np.dot(normalized_embeddings[prev_idx], normalized_embeddings[clip_idx]))
                transition = (cosine + 1.0) / 2.0 if normalize_scores else cosine
                semantic = beam.semantic_score + float(scoring_matrix[seg_idx, clip_idx])
                coherence = beam.coherence_score + transition
                combined = semantic + lambda_coherence * coherence
                expanded.append(
                    BeamItem(
                        assignments=beam.assignments + [clip_idx],
                        used=beam.used | {clip_idx},
                        semantic_score=semantic,
                        coherence_score=coherence,
                        combined_score=combined,
                        transitions=beam.transitions + [
                            {
                                "from_clip": clip_ids[prev_idx],
                                "to_clip": clip_ids[clip_idx],
                                "from_index": prev_idx,
                                "to_index": clip_idx,
                                "cosine": cosine,
                                "transition_coherence": transition,
                            }
                        ],
                    )
                )
        if not expanded:
            break
        expanded.sort(key=lambda item: item.combined_score, reverse=True)
        beams = expanded[:beam_size]

    if not beams:
        return [], {
            "assignment_method": "coherence_beam",
            "error": "beam search produced no valid assignments",
        }

    best = beams[0]
    selected_ids = [clip_ids[idx] for idx in best.assignments]
    diagnostics = {
        "assignment_method": "coherence_beam",
        "semantic_score": float(best.semantic_score),
        "coherence_score": float(best.coherence_score),
        "combined_score": float(best.combined_score),
        "selected_clip_ids": selected_ids,
        "selected_indices": best.assignments,
        "transition_diagnostics": best.transitions,
        "params": {
            "top_k": top_k,
            "beam_size": beam_size,
            "lambda_coherence": lambda_coherence,
            "allow_reuse": bool(allow_reuse),
            "normalize_scores": bool(normalize_scores),
        },
        "num_segments": int(num_segments),
        "num_clips": int(num_clips),
        "beam_count_final": len(beams),
    }
    return selected_ids, diagnostics
