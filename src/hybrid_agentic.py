"""
Hybrid agentic VideoPrism pipeline.

This module adds an optional tool-augmented pipeline on top of the existing
VideoPrism/FAISS backbone.  It is intentionally conservative: the retrieval
and final assignment remain deterministic and benchmarkable, while the agent
layer can inspect candidates, issue extra FAISS searches, and return advisory
or hard-lock constraints.

When Codex CLI is not available on the server the same tool layer runs in a
safe dry-run/debug mode.  The dry-run mode is useful for validating indexing,
search, contact-sheet generation, logging, and UI/CLI integration without
exposing ground truth to an agent.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import logging
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import BenchmarkEvaluator
from hybrid_options import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
    HYBRID_PROMPT_MODES,
    load_codex_model_catalog,
)
from indexing import VideoIndexer
from matching import (
    ClipSelection,
    EnsembleVideoTextMatcher,
    PromptedVideoTextMatcher,
    VideoTextMatcher,
)

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - dependency exists in the server env
    linear_sum_assignment = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}

VIDEOPRISM_PROMPT_TEMPLATES = {
    "none": "{}",
    "video": "a video of {}",
    "photo": "a photo of {}",
    "scene": "a scene showing {}",
    "cooking": "a cooking video showing {}",
    "clip": "a short clip of {}",
}

VIDEOPRISM_ENSEMBLE_TEMPLATES = [
    "{}",
    "a video of {}",
    "a photo of {}",
    "a scene showing {}",
    "a short clip of {}",
]


@dataclass
class HybridVideoPrismConfig:
    model_name: str = "videoprism_lvt_public_v1_large"
    num_frames: int = 8
    resolution: int = 288
    use_dual_softmax: bool = False
    prompt_mode: str = "none"
    query_mode: str = "original"
    assignment_method: str = "hungarian"
    score_normalization: str = "none"
    csls_k: int = 5
    no_windowing: bool = True
    window_size: float = 5.0
    window_overlap: float = 0.5


@dataclass
class HybridAgentConfig:
    enabled: bool = True
    use_codex: bool = False
    review_scope: str = "ambiguous"  # ambiguous or all
    decision_mode: str = "advisory"  # advisory or hard_lock
    shortlist_size: int = 5
    ambiguity_margin_threshold: float = 0.05
    max_agent_segments: int = 10
    search_budget_per_segment: int = 2
    allow_exclusions: bool = True
    allow_candidate_expansion: bool = True
    allow_contact_sheet: bool = True
    advisory_boost: float = 0.03
    codex_model: str = DEFAULT_CODEX_MODEL
    codex_reasoning_effort: str = DEFAULT_CODEX_REASONING_EFFORT
    codex_timeout_seconds: int = 600
    hard_lock_min_confidence: float = 0.9
    require_direct_inspection_for_hard_constraints: bool = True
    verify_assignment_cycles: bool = True
    cycle_critic_min_confidence: float = 0.75


@dataclass
class HybridRunConfig:
    benchmark: Optional[str] = None
    video_dir: Optional[str] = None
    segments_file: Optional[str] = None
    ground_truth_file: Optional[str] = None
    output_dir: str = "./output/hybrid_agentic"
    cache_dir: str = "./cache"
    device: str = "cuda:0"
    video: HybridVideoPrismConfig = field(default_factory=HybridVideoPrismConfig)
    agent: HybridAgentConfig = field(default_factory=HybridAgentConfig)


@dataclass
class HybridResult:
    config: Dict[str, Any]
    exact_match_accuracy: Optional[float]
    top_3_accuracy: Optional[float]
    top_5_accuracy: Optional[float]
    mrr: Optional[float]
    avg_similarity: Optional[float]
    baseline_exact_match_accuracy: Optional[float]
    baseline_top_3_accuracy: Optional[float]
    baseline_top_5_accuracy: Optional[float]
    baseline_mrr: Optional[float]
    exact_match_delta: Optional[float]
    indexing_time: float
    matching_time: float
    total_time: float
    reviewed_segments: int
    codex_available: bool
    agent_mode: str
    log_file: str


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(data), handle, indent=2)


def _resolve_project_path(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return str(path)


def _normalize_benchmark_id(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    match = re.search(r"(\d+)", str(value))
    return match.group(1) if match else str(value)


def resolve_benchmark_paths(benchmark: str, benchmarks_dir: str = "./data/benchmarks") -> Dict[str, str]:
    benchmark_id = _normalize_benchmark_id(benchmark)
    base = Path(_resolve_project_path(benchmarks_dir) or benchmarks_dir)
    paths = {
        "video_dir": base / "videos" / f"video_{benchmark_id}",
        "segments_file": base / "segments" / f"benchmark_{benchmark_id}_segments.json",
        "ground_truth_file": base / "gdtruth" / f"benchmark_{benchmark_id}_ground_truth.json",
    }
    missing = [f"{key}: {path}" for key, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Benchmark is incomplete:\n  " + "\n  ".join(missing))
    return {key: str(path) for key, path in paths.items()}


def load_segments(path: str) -> List[Dict[str, Any]]:
    payload = _read_json(Path(path), {})
    if isinstance(payload, list):
        segments = payload
    elif isinstance(payload, dict):
        segments = payload.get("segments", payload.get("mappings", []))
    else:
        segments = []
    cleaned: List[Dict[str, Any]] = []
    for idx, segment in enumerate(segments):
        if "text" in segment:
            text = segment.get("text")
        elif "segment_text" in segment:
            text = segment.get("segment_text")
        elif "description" in segment:
            text = segment.get("description")
        else:
            text = ""
        start = float(segment.get("start_time", segment.get("start", 0.0)) or 0.0)
        end = float(segment.get("end_time", segment.get("end", 0.0)) or 0.0)
        duration = float(segment.get("duration", 0.0) or 0.0)
        if duration <= 0 and end > start:
            duration = end - start
        cleaned.append({
            "segment_id": int(segment.get("segment_id", segment.get("index", idx))),
            # Empty narration is meaningful. An ordinal placeholder would be
            # embedded as real language and can perturb every Dual Softmax row.
            "text": "" if text is None else str(text),
            "duration": max(0.2, duration),
            "start_time": start,
            "end_time": end,
        })
    return cleaned


def _find_codex_cli() -> Optional[str]:
    path = shutil.which("codex")
    if path:
        return path
    home = Path.home()
    for candidate in (
        home / ".local" / "bin" / "codex",
        home / ".npm-global" / "bin" / "codex",
        home / ".npm" / "bin" / "codex",
        home / ".codex" / "packages" / "standalone" / "current" / "bin" / "codex",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def codex_status() -> Dict[str, Any]:
    models = load_codex_model_catalog()
    path = _find_codex_cli()
    if not path:
        return {
            "available": False,
            "authenticated": False,
            "ready": False,
            "path": None,
            "message": "Codex CLI was not found on PATH or in a supported user-local installation.",
            "models": models,
            "default_model": DEFAULT_CODEX_MODEL,
            "default_reasoning_effort": DEFAULT_CODEX_REASONING_EFFORT,
        }
    try:
        result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10, check=False)
    except Exception as exc:
        return {
            "available": False,
            "authenticated": False,
            "ready": False,
            "path": path,
            "message": f"Codex CLI probe failed: {exc}",
            "models": models,
            "default_model": DEFAULT_CODEX_MODEL,
            "default_reasoning_effort": DEFAULT_CODEX_REASONING_EFFORT,
        }
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0 or "Please ask your administrator" in output:
        return {
            "available": False,
            "authenticated": False,
            "ready": False,
            "path": path,
            "returncode": result.returncode,
            "message": output or "Codex CLI is not usable.",
            "models": models,
            "default_model": DEFAULT_CODEX_MODEL,
            "default_reasoning_effort": DEFAULT_CODEX_REASONING_EFFORT,
        }
    try:
        login = subprocess.run([path, "login", "status"], capture_output=True, text=True, timeout=10, check=False)
        login_output = (login.stdout + login.stderr).strip()
    except Exception as exc:
        login = None
        login_output = f"Codex authentication probe failed: {exc}"
    authenticated = bool(login and login.returncode == 0 and "Logged in" in login_output)
    return {
        "available": True,
        "authenticated": authenticated,
        "ready": authenticated,
        "path": path,
        "returncode": result.returncode,
        "message": output,
        "auth_message": login_output,
        "models": models,
        "default_model": DEFAULT_CODEX_MODEL,
        "default_reasoning_effort": DEFAULT_CODEX_REASONING_EFFORT,
    }


def _prompt_settings(prompt_mode: str) -> Tuple[Optional[str], Optional[List[str]]]:
    mode = prompt_mode or "none"
    if mode.startswith("template:"):
        key = mode.split(":", 1)[1]
        return VIDEOPRISM_PROMPT_TEMPLATES.get(key), None
    if mode == "ensemble:template":
        return None, list(VIDEOPRISM_ENSEMBLE_TEMPLATES)
    if mode in VIDEOPRISM_PROMPT_TEMPLATES and mode != "none":
        return VIDEOPRISM_PROMPT_TEMPLATES.get(mode), None
    return None, None


def validate_run_config(run_config: HybridRunConfig) -> None:
    video = run_config.video
    agent = run_config.agent
    if video.model_name not in {"videoprism_lvt_public_v1_base", "videoprism_lvt_public_v1_large"}:
        raise ValueError(f"Unsupported VideoPrism model: {video.model_name}")
    if video.prompt_mode not in HYBRID_PROMPT_MODES:
        raise ValueError(f"Unsupported baseline prompt mode: {video.prompt_mode}")
    if video.query_mode != "original":
        raise ValueError("Hybrid Agentic mode always uses the original script segment for its baseline query")
    if int(video.num_frames) <= 0:
        raise ValueError("num_frames must be greater than zero")
    if int(video.resolution) <= 0:
        raise ValueError("resolution must be greater than zero")
    if video.score_normalization not in {"none", "csls", "rank_fusion", "zscore"}:
        raise ValueError(f"Unsupported score normalization: {video.score_normalization}")
    if int(video.csls_k) <= 0:
        raise ValueError("csls_k must be greater than zero")
    if not 0.0 <= float(video.window_overlap) < 1.0:
        raise ValueError("window_overlap must be in the range [0, 1)")
    if float(video.window_size) <= 0:
        raise ValueError("window_size must be greater than zero")
    if not video.no_windowing:
        raise ValueError("Hybrid Agentic mode requires one whole-clip candidate per source; temporal windowing is not supported")
    if agent.decision_mode not in {"advisory", "hard_lock"}:
        raise ValueError(f"Unsupported agent decision mode: {agent.decision_mode}")
    if agent.review_scope not in {"ambiguous", "all"}:
        raise ValueError(f"Unsupported agent review scope: {agent.review_scope}")
    if agent.decision_mode == "hard_lock" and not agent.use_codex:
        raise ValueError("hard_lock mode requires the authenticated Codex MCP agent; deterministic heuristic runs are advisory-only")
    if int(agent.shortlist_size) <= 0:
        raise ValueError("shortlist_size must be greater than zero")
    if float(agent.ambiguity_margin_threshold) < 0:
        raise ValueError("ambiguity_margin_threshold cannot be negative")
    if int(agent.max_agent_segments) < 0:
        raise ValueError("max_agent_segments cannot be negative")
    if int(agent.search_budget_per_segment) < 0:
        raise ValueError("search_budget_per_segment cannot be negative")
    if not 0.0 <= float(agent.hard_lock_min_confidence) <= 1.0:
        raise ValueError("hard_lock_min_confidence must be in the range [0, 1]")
    if not 0.0 <= float(agent.cycle_critic_min_confidence) <= 1.0:
        raise ValueError("cycle_critic_min_confidence must be in the range [0, 1]")
    if (
        agent.decision_mode == "hard_lock"
        and agent.require_direct_inspection_for_hard_constraints
        and not agent.allow_contact_sheet
    ):
        raise ValueError("hard_lock mode requires contact-sheet inspection; use advisory mode when inspection is disabled")
    if (
        agent.use_codex
        and agent.review_scope == "all"
        and agent.verify_assignment_cycles
        and not agent.allow_contact_sheet
    ):
        raise ValueError("the independent assignment-cycle critic requires contact-sheet inspection")
    if agent.use_codex:
        catalog = load_codex_model_catalog()
        selected = next((item for item in catalog if item.get("value") == agent.codex_model), None)
        if not selected:
            available = ", ".join(str(item.get("value")) for item in catalog)
            raise ValueError(f"Unsupported Codex model: {agent.codex_model}. Available models: {available}")
        efforts = [str(value) for value in selected.get("supported_reasoning_efforts", [])]
        if agent.codex_reasoning_effort not in efforts:
            raise ValueError(
                f"Codex model {agent.codex_model} does not support reasoning effort "
                f"{agent.codex_reasoning_effort}. Supported efforts: {', '.join(efforts)}"
            )


def build_matcher(indexer: VideoIndexer, config: HybridVideoPrismConfig, device: str) -> VideoTextMatcher:
    prompt_template, ensemble_prompts = _prompt_settings(config.prompt_mode)
    kwargs = {"video_indexer": indexer, "model_name": config.model_name, "device": device, "min_similarity_threshold": 0.0}
    if ensemble_prompts:
        return EnsembleVideoTextMatcher(**kwargs, ensemble_templates=ensemble_prompts)
    if prompt_template:
        return PromptedVideoTextMatcher(**kwargs, prompt_template=prompt_template)
    return VideoTextMatcher(**kwargs)


def index_cache_dir(cache_dir: str, video_dir: str, config: HybridVideoPrismConfig) -> Path:
    video_token = Path(video_dir).name.replace(" ", "_")
    model_token = "large" if "large" in config.model_name else "base"
    window_token = "nowin" if config.no_windowing else f"win{config.window_size:g}_{config.window_overlap:g}"
    return Path(_resolve_project_path(cache_dir) or cache_dir) / "hybrid_agentic" / f"vp_{video_token}_{model_token}_{config.num_frames}f_{config.resolution}p_{window_token}"


def jax_runtime_status(requested_device: str) -> Dict[str, Any]:
    import jax

    devices = jax.devices()
    status = {
        "requested_device": requested_device,
        "backend": jax.default_backend(),
        "devices": [
            {
                "id": int(getattr(device, "id", index)),
                "platform": str(getattr(device, "platform", "unknown")),
                "device_kind": str(getattr(device, "device_kind", type(device).__name__)),
            }
            for index, device in enumerate(devices)
        ],
    }
    if str(requested_device).lower().startswith("cuda") and status["backend"] != "gpu":
        raise RuntimeError(
            f"CUDA was requested ({requested_device}) but JAX initialized the {status['backend']} backend. "
            "Install a CUDA-enabled JAX plugin before running VideoPrism."
        )
    return status


def build_indexer(run_config: HybridRunConfig) -> Tuple[VideoIndexer, float, bool, Dict[str, Any]]:
    assert run_config.video_dir
    start = time.time()
    idx_dir = index_cache_dir(run_config.cache_dir, run_config.video_dir, run_config.video)
    indexer = VideoIndexer(
        model_name=run_config.video.model_name,
        index_dir=str(idx_dir),
        device=run_config.device,
        num_frames=int(run_config.video.num_frames),
        resolution=int(run_config.video.resolution),
    )
    accelerator = jax_runtime_status(run_config.device)
    loaded = indexer.load_index()
    if not loaded:
        indexer.index_videos(
            run_config.video_dir,
            use_windowing=not run_config.video.no_windowing,
            window_size=float(run_config.video.window_size),
            window_overlap=float(run_config.video.window_overlap),
        )
    return indexer, round(time.time() - start, 2), loaded, accelerator


def solve_assignment(
    matrix: np.ndarray,
    metadata: Sequence[Any],
    segments: Sequence[Dict[str, Any]],
    locks: Optional[Dict[int, int]] = None,
    exclusions: Optional[Dict[int, Iterable[int]]] = None,
    log: Optional[Dict[str, Any]] = None,
) -> Tuple[List[int], List[str]]:
    if linear_sum_assignment is None:
        raise RuntimeError("scipy is required for Hungarian assignment.")
    scores = np.array(matrix, dtype=np.float64, copy=True)
    num_segments, num_videos = scores.shape
    locks = {int(k): int(v) for k, v in (locks or {}).items()}
    exclusions = {int(k): {int(v) for v in vals} for k, vals in (exclusions or {}).items()}
    rejected: List[str] = []

    seen_rows = set()
    seen_cols = set()
    valid_locks: Dict[int, int] = {}
    for row, col in locks.items():
        reason = None
        if row < 0 or row >= num_segments:
            reason = f"lock row {row} is outside segment range"
        elif col < 0 or col >= num_videos:
            reason = f"lock col {col} is outside candidate range"
        elif row in seen_rows:
            reason = f"segment {row} has multiple locks"
        elif col in seen_cols:
            reason = f"clip {col} is locked more than once"
        elif col in exclusions.get(row, set()):
            reason = f"lock ({row}, {col}) is also excluded"
        if reason:
            rejected.append(reason)
            continue
        seen_rows.add(row)
        seen_cols.add(col)
        valid_locks[row] = col

    for row, cols in exclusions.items():
        if 0 <= row < num_segments:
            for col in cols:
                if 0 <= col < num_videos and valid_locks.get(row) != col:
                    scores[row, col] = -1e9

    assignment = [-1 for _ in range(num_segments)]
    for row, col in valid_locks.items():
        assignment[row] = col

    remaining_rows = [row for row in range(num_segments) if row not in valid_locks]
    remaining_cols = [col for col in range(num_videos) if col not in set(valid_locks.values())]
    if remaining_rows and remaining_cols:
        sub_scores = scores[np.ix_(remaining_rows, remaining_cols)]
        size = max(sub_scores.shape)
        cost = np.full((size, size), 1e6, dtype=np.float64)
        cost[: sub_scores.shape[0], : sub_scores.shape[1]] = -sub_scores
        row_idx, col_idx = linear_sum_assignment(cost)
        for r, c in zip(row_idx, col_idx):
            if r < len(remaining_rows) and c < len(remaining_cols):
                row = remaining_rows[r]
                col = remaining_cols[c]
                if scores[row, col] > -1e8:
                    assignment[row] = col

    if log is not None:
        log["validated_locks"] = {str(k): v for k, v in valid_locks.items()}
        log["rejected_decisions"] = rejected
    return assignment, rejected


def build_clip_selections(
    assignment: Sequence[int],
    matrix: np.ndarray,
    metadata: Sequence[Any],
    segments: Sequence[Dict[str, Any]],
) -> List[ClipSelection]:
    selections: List[ClipSelection] = []
    for seg_idx, col in enumerate(assignment):
        if col < 0 or col >= len(metadata):
            continue
        meta = metadata[col]
        segment = segments[seg_idx]
        window_start = float(getattr(meta, "window_start", 0.0) or 0.0)
        duration = float(getattr(meta, "duration", segment.get("duration", 0.2)) or 0.2)
        trim_start = window_start
        trim_end = window_start + duration
        selections.append(ClipSelection(
            segment_id=seg_idx,
            video_id=meta.video_id,
            video_file_path=meta.file_path,
            start_time=0.0,
            end_time=0.0,
            duration=float(segment.get("duration", duration)),
            trim_start=trim_start,
            trim_end=trim_end,
            trim_duration=max(0.0, trim_end - trim_start),
            similarity_score=float(matrix[seg_idx, col]),
            motion_score=0.0,
            context_score=0.0,
            combined_score=float(matrix[seg_idx, col]),
            is_reused=False,
        ))
    return selections


def assignment_to_log(assignment: Sequence[int], matrix: np.ndarray, metadata: Sequence[Any], segments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for seg_idx, col in enumerate(assignment):
        meta = metadata[col] if 0 <= col < len(metadata) else None
        rows.append({
            "segment_id": int(segments[seg_idx].get("segment_id", seg_idx)),
            "segment_index": seg_idx,
            "segment_text": segments[seg_idx].get("text", ""),
            "candidate_index": int(col) if col >= 0 else None,
            "anonymous_clip_id": f"clip_{col:03d}" if col >= 0 else None,
            "score": float(matrix[seg_idx, col]) if col >= 0 else None,
        })
    return rows


def find_ambiguous_segments(
    matrix: np.ndarray,
    assignment: Sequence[int],
    segments: Sequence[Dict[str, Any]],
    threshold: float,
    max_segments: int,
    review_scope: str = "ambiguous",
) -> List[Dict[str, Any]]:
    """Rank uncertainty and keep assignment-linked alternatives together.

    A no-reuse Hungarian assignment can displace a segment from its local best
    clip.  For each assigned pair we therefore forbid that pair, solve again,
    and measure the loss in the total assignment score.  A small global margin
    means another complete assignment is almost as good.  The rows changed by
    that alternative are included in the same review set so an agent can reason
    about a coordinated swap instead of seeing only one side of it.
    """
    baseline_total = _assignment_total_score(matrix, assignment)
    records: List[Dict[str, Any]] = []
    ambiguity_seeds: List[Dict[str, Any]] = []
    for seg_idx, row in enumerate(matrix):
        if row.size == 0:
            continue
        top = np.sort(row)[::-1]
        best = float(top[0])
        second = float(top[1]) if len(top) > 1 else -1.0
        margin = best - second
        assigned = assignment[seg_idx] if seg_idx < len(assignment) else -1
        assigned_score = float(row[assigned]) if 0 <= assigned < len(row) else -1.0
        assigned_rank = int(np.sum(row > assigned_score) + 1) if assigned >= 0 else None
        assignment_gap = best - assigned_score if assigned >= 0 else math.inf
        global_margin = math.inf
        alternative_candidate = None
        linked_segments: List[int] = []
        if assigned >= 0:
            alternative, _ = solve_assignment(
                matrix,
                (),
                segments,
                exclusions={seg_idx: [assigned]},
            )
            if alternative[seg_idx] >= 0:
                global_margin = max(0.0, baseline_total - _assignment_total_score(matrix, alternative))
                alternative_candidate = int(alternative[seg_idx])
                linked_segments = [
                    int(row_idx)
                    for row_idx, (current, alternate) in enumerate(zip(assignment, alternative))
                    if current != alternate
                ]
        record = {
            "segment_index": seg_idx,
            "segment_id": int(segments[seg_idx].get("segment_id", seg_idx)),
            "text": segments[seg_idx].get("text", ""),
            "best_score": best,
            "second_score": second,
            "margin": margin,
            "assigned_index": int(assigned),
            "assigned_score": assigned_score,
            "assigned_rank": assigned_rank,
            "assignment_gap": assignment_gap,
            "global_assignment_margin": global_margin,
            "alternative_candidate_index": alternative_candidate,
            "linked_segment_indices": linked_segments,
        }
        records.append(record)
        if global_margin <= threshold or margin <= threshold or assignment_gap > threshold:
            ambiguity_seeds.append(record)

    ambiguity_seeds.sort(key=lambda item: (
        item["margin"],
        item["global_assignment_margin"],
        -item["assignment_gap"],
        -item["best_score"],
    ))
    record_by_index = {int(item["segment_index"]): item for item in records}
    selected: List[Dict[str, Any]] = []
    selected_indices = set()
    if review_scope not in {"ambiguous", "all"}:
        raise ValueError(f"Unsupported agent review scope: {review_scope}")
    limit = len(records) if review_scope == "all" else max(0, int(max_segments))

    for seed in ambiguity_seeds:
        if len(selected) >= limit:
            break
        seed_index = int(seed["segment_index"])
        group = [seed_index] + [
            idx for idx in seed["linked_segment_indices"] if idx != seed_index
        ]
        for index in group:
            if len(selected) >= limit:
                break
            if index in selected_indices or index not in record_by_index:
                continue
            item = dict(record_by_index[index])
            item["selection_role"] = "ambiguity_seed" if index == seed_index else "assignment_link"
            item["review_group_seed"] = seed_index
            selected.append(item)
            selected_indices.add(index)
    if review_scope == "all":
        for record in records:
            index = int(record["segment_index"])
            if index in selected_indices:
                continue
            item = dict(record)
            item["selection_role"] = "global_audit"
            item["review_group_seed"] = index
            selected.append(item)
            selected_indices.add(index)
    return selected


def _assignment_total_score(matrix: np.ndarray, assignment: Sequence[int]) -> float:
    return float(sum(
        matrix[row, col]
        for row, col in enumerate(assignment)
        if 0 <= col < matrix.shape[1]
    ))


def _keywords(text: str, limit: int = 8) -> List[str]:
    stop = {"the", "and", "that", "this", "with", "from", "your", "you", "are", "was", "for", "into", "then", "they", "them", "have", "will", "not", "but", "now", "again", "only", "just", "more", "less"}
    words = re.findall(r"\b[a-z][a-z0-9]{2,}\b", text.lower())
    seen: List[str] = []
    for word in words:
        if word not in stop and word not in seen:
            seen.append(word)
        if len(seen) >= limit:
            break
    return seen


class HybridAgentTools:
    """Benchmark-safe tools over the loaded index.

    The baseline matcher may shape script queries with a prompt template. Agent
    searches intentionally bypass that wrapper so Codex's complete query is
    embedded verbatim by the same VideoPrism text encoder.
    """

    def __init__(
        self,
        matcher: VideoTextMatcher,
        indexer: VideoIndexer,
        segments: List[Dict[str, Any]],
        matrix: np.ndarray,
        metadata: Sequence[Any],
        initial_assignment: Sequence[int],
        run_dir: Path,
    ) -> None:
        self.matcher = matcher
        self.indexer = indexer
        self.segments = segments
        self.matrix = matrix
        self.metadata = list(metadata)
        self.initial_assignment = list(initial_assignment)
        self.run_dir = run_dir
        self.contact_dir = run_dir / "contact_sheets"
        self.contact_dir.mkdir(parents=True, exist_ok=True)

    def get_segment_context(self, segment_index: int, radius: int = 2) -> Dict[str, Any]:
        start = max(0, segment_index - radius)
        end = min(len(self.segments), segment_index + radius + 1)
        return {
            "segment_index": segment_index,
            "segment": self._safe_segment(segment_index),
            "neighbors": [self._safe_segment(i) for i in range(start, end)],
        }

    def get_global_script_context(self) -> Dict[str, Any]:
        return {
            "segment_count": len(self.segments),
            "segments": [self._safe_segment(i) for i in range(len(self.segments))],
        }

    def get_candidate_catalog(self) -> Dict[str, Any]:
        return {
            "candidate_count": len(self.metadata),
            "candidates": [
                {"candidate_index": idx, "anonymous_clip_id": f"clip_{idx:03d}"}
                for idx in range(len(self.metadata))
            ],
        }

    def get_candidate_scores(self, segment_index: int, top_k: int = 10) -> List[Dict[str, Any]]:
        row = self.matrix[segment_index]
        order = np.argsort(row)[::-1][: int(top_k)]
        return [self._candidate_payload(segment_index, int(col), float(row[col]), rank + 1) for rank, col in enumerate(order)]

    def search_video_index(self, query: str, top_k: int = 10, exclude_assigned: bool = False) -> List[Dict[str, Any]]:
        embedding = VideoTextMatcher.get_text_embeddings_batch(
            self.matcher,
            [query],
        )[0]
        results = self.indexer.search_by_embedding(embedding, k=min(max(int(top_k), 1), len(self.metadata)))
        assigned = set(self.initial_assignment) if exclude_assigned else set()
        payload = []
        rank = 1
        for video_id, score, meta in results:
            idx = self._metadata_index(video_id, meta)
            if idx in assigned:
                continue
            payload.append({
                "rank": rank,
                "candidate_index": idx,
                "anonymous_clip_id": f"clip_{idx:03d}" if idx is not None else None,
                "score": float(score),
                "score_space": "raw_videoprism_cosine_similarity",
                "is_windowed": bool(getattr(meta, "is_windowed", False)),
            })
            rank += 1
            if len(payload) >= top_k:
                break
        return payload

    def inspect_clip(
        self,
        candidate_index: int,
        frames: Sequence[float] = (0.12, 0.5, 0.88),
    ) -> Dict[str, Any]:
        if candidate_index < 0 or candidate_index >= len(self.metadata):
            raise ValueError(f"candidate_index out of range: {candidate_index}")
        meta = self.metadata[candidate_index]
        path = Path(meta.file_path)
        output = self.contact_dir / f"clip_{candidate_index:03d}_sheet.jpg"
        ok = make_contact_sheet(path, output, frames=frames)
        return {
            "candidate_index": candidate_index,
            "anonymous_clip_id": f"clip_{candidate_index:03d}",
            "contact_sheet": str(output) if ok else None,
            "sampled_frame_fractions": [float(value) for value in frames],
            "frame_resolution": [512, 288],
        }

    def _metadata_index(self, video_id: str, result_meta: Optional[Any] = None) -> Optional[int]:
        for idx, meta in enumerate(self.metadata):
            if meta is result_meta:
                return idx
            if (
                result_meta is not None
                and meta.video_id == video_id
                and getattr(meta, "file_path", None) == getattr(result_meta, "file_path", None)
                and getattr(meta, "window_start", None) == getattr(result_meta, "window_start", None)
            ):
                return idx
        for idx, meta in enumerate(self.metadata):
            if meta.video_id == video_id:
                return idx
        return None

    def _safe_segment(self, idx: int) -> Dict[str, Any]:
        seg = self.segments[idx]
        text = "" if seg.get("text") is None else str(seg.get("text", ""))
        return {
            "segment_index": idx,
            "segment_id": seg.get("segment_id", idx),
            "text": text,
            "has_narration": bool(text.strip()),
        }

    def _candidate_payload(self, segment_index: int, candidate_index: int, score: float, rank: int) -> Dict[str, Any]:
        return {
            "rank": rank,
            "segment_index": segment_index,
            "candidate_index": candidate_index,
            "anonymous_clip_id": f"clip_{candidate_index:03d}",
            "score": score,
            "score_space": "configured_baseline_matrix",
            "is_initial_assignment": self.initial_assignment[segment_index] == candidate_index,
        }


class HybridAgentToolBridge:
    """Token-authenticated loopback bridge between Codex MCP and live FAISS state."""

    def __init__(
        self,
        tools: HybridAgentTools,
        ambiguous: Sequence[Dict[str, Any]],
        config: HybridAgentConfig,
    ) -> None:
        self.tools = tools
        self.config = config
        self.allowed_segments = {int(item["segment_index"]) for item in ambiguous}
        self.search_limit = max(0, int(config.search_budget_per_segment))
        self.search_counts: Dict[int, int] = collections.defaultdict(int)
        self.inspect_counts: Dict[int, int] = collections.defaultdict(int)
        self.globally_inspected_candidates: set[int] = set()
        self.known_candidates: Dict[int, set[int]] = collections.defaultdict(set)
        self.audit_log: List[Dict[str, Any]] = []
        self.token = secrets.token_urlsafe(32)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> str:
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - HTTP handler API
                response: Dict[str, Any]
                try:
                    if self.path != "/tool":
                        raise ValueError("Unknown bridge endpoint")
                    if self.headers.get("Authorization") != f"Bearer {bridge.token}":
                        raise PermissionError("Invalid bridge token")
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 1_000_000:
                        raise ValueError("Invalid request size")
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    result = bridge.execute(str(payload.get("name") or ""), payload.get("arguments") or {})
                    response = {"ok": True, "result": result}
                except Exception as exc:
                    response = {"ok": False, "error": str(exc)}
                body = json.dumps(_json_safe(response), separators=(",", ":")).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, name="hybrid-agent-tools", daemon=True)
        self._thread.start()
        host, port = self._server.server_address
        return f"http://{host}:{port}/tool"

    def close(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        with self._lock:
            result = self._execute(name, arguments)
            artifact_path = result.get("_contact_sheet_path") if isinstance(result, dict) else None
            public_result = dict(result) if isinstance(result, dict) else result
            if isinstance(public_result, dict):
                public_result.pop("_contact_sheet_path", None)
            audit_entry = {
                "tool": name,
                "arguments": _json_safe(arguments),
                "result": _json_safe(public_result),
                "timestamp": datetime.now().isoformat(),
            }
            if artifact_path:
                audit_entry["artifact_path"] = str(artifact_path)
            self.audit_log.append(audit_entry)
            return result

    def _segment_index(self, arguments: Dict[str, Any]) -> int:
        idx = int(arguments.get("segment_index", -1))
        if idx not in self.allowed_segments:
            raise ValueError(f"segment_index {idx} is not approved for agent review")
        return idx

    def _top_k(self, arguments: Dict[str, Any]) -> int:
        return max(1, min(int(arguments.get("top_k", self.config.shortlist_size)), int(self.config.shortlist_size)))

    def _execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name == "get_global_script_context":
            return self.tools.get_global_script_context()
        if name == "get_initial_assignment":
            for idx in self.allowed_segments:
                if 0 <= idx < len(self.tools.initial_assignment) and self.tools.initial_assignment[idx] >= 0:
                    self.known_candidates[idx].add(int(self.tools.initial_assignment[idx]))
            rows = []
            for idx, candidate in enumerate(self.tools.initial_assignment):
                rows.append({
                    "segment_index": idx,
                    "anonymous_clip_id": f"clip_{candidate:03d}" if candidate >= 0 else None,
                    "candidate_index": int(candidate) if candidate >= 0 else None,
                    "score": float(self.tools.matrix[idx, candidate]) if candidate >= 0 else None,
                })
            return {"assignment": rows, "clip_reuse_allowed": False}
        if name == "get_candidate_catalog":
            if self.config.review_scope != "all":
                raise ValueError("The full candidate catalog is available only in full-sequence audit mode")
            return self.tools.get_candidate_catalog()
        if name == "get_segment_context":
            idx = self._segment_index(arguments)
            radius = max(0, min(int(arguments.get("radius", 2)), 4))
            return self.tools.get_segment_context(idx, radius=radius)
        if name == "get_candidate_scores":
            idx = self._segment_index(arguments)
            results = self.tools.get_candidate_scores(idx, top_k=self._top_k(arguments))
            self.known_candidates[idx].update(int(item["candidate_index"]) for item in results)
            return {"segment_index": idx, "score_space": "configured_baseline_matrix", "candidates": results}
        if name == "search_video_index":
            idx = self._segment_index(arguments)
            if not self.config.allow_candidate_expansion:
                raise ValueError("Custom candidate expansion is disabled for this run")
            if self.search_counts[idx] >= self.search_limit:
                raise ValueError(f"Search budget exhausted for segment {idx}")
            query = str(arguments.get("query") or "").strip()
            if not query or len(query) > 500:
                raise ValueError("query must contain 1-500 characters")
            self.search_counts[idx] += 1
            results = self.tools.search_video_index(
                query,
                top_k=self._top_k(arguments),
                exclude_assigned=bool(arguments.get("exclude_assigned", False)),
            )
            self.known_candidates[idx].update(int(item["candidate_index"]) for item in results if item.get("candidate_index") is not None)
            return {
                "segment_index": idx,
                "query": query,
                "search_number": self.search_counts[idx],
                "search_budget": self.search_limit,
                "score_space": "raw_videoprism_cosine_similarity",
                "candidates": results,
            }
        if name == "inspect_clip":
            idx = self._segment_index(arguments)
            if not self.config.allow_contact_sheet:
                raise ValueError("Contact-sheet inspection is disabled for this run")
            candidate = int(arguments.get("candidate_index", -1))
            if candidate not in self.known_candidates[idx]:
                raise ValueError("Inspect candidates returned by get_candidate_scores or search_video_index first")
            if self.inspect_counts[idx] >= int(self.config.shortlist_size):
                raise ValueError(f"Inspection budget exhausted for segment {idx}")
            self.inspect_counts[idx] += 1
            inspected = self.tools.inspect_clip(candidate)
            path = inspected.pop("contact_sheet", None)
            inspected["contact_sheet_available"] = bool(path)
            if path:
                inspected["_contact_sheet_path"] = path
            return inspected
        if name == "inspect_candidate":
            if self.config.review_scope != "all":
                raise ValueError("Candidate-level inspection is available only in full-sequence audit mode")
            if not self.config.allow_contact_sheet:
                raise ValueError("Contact-sheet inspection is disabled for this run")
            candidate = int(arguments.get("candidate_index", -1))
            if candidate < 0 or candidate >= len(self.tools.metadata):
                raise ValueError(f"candidate_index out of range: {candidate}")
            inspected = self.tools.inspect_clip(candidate)
            path = inspected.pop("contact_sheet", None)
            inspected["contact_sheet_available"] = bool(path)
            inspected["already_inspected"] = candidate in self.globally_inspected_candidates
            self.globally_inspected_candidates.add(candidate)
            if path:
                inspected["_contact_sheet_path"] = path
            return inspected
        raise ValueError(f"Unknown hybrid agent tool: {name}")


def make_contact_sheet(
    video_path: Path,
    output_path: Path,
    frames: Sequence[float] = (0.12, 0.5, 0.88),
    frame_size: Tuple[int, int] = (512, 288),
) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return False
    images = []
    for fraction in frames:
        idx = int(max(0, min(1, float(fraction))) * max(total - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, frame_size)
        images.append(frame)
    cap.release()
    if not images:
        return False
    rows = []
    for start in range(0, len(images), 3):
        row = list(images[start:start + 3])
        while len(row) < 3:
            row.append(np.zeros_like(images[0]))
        rows.append(np.concatenate(row, axis=1))
    sheet = np.concatenate(rows, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), sheet))


def dry_run_agent_review(
    tools: HybridAgentTools,
    ambiguous: List[Dict[str, Any]],
    config: HybridAgentConfig,
) -> Dict[str, Any]:
    decisions = {
        "mode": "deterministic_heuristic",
        "decision_mode": config.decision_mode,
        "reviewed_segments": [],
        "locks": {},
        "exclusions": {},
        "advisory_scores": {},
        "advisory_penalties": {},
        "expanded_candidates": {},
    }
    review_items = ambiguous if config.review_scope == "all" else ambiguous[: config.max_agent_segments]
    for item in review_items:
        idx = int(item["segment_index"])
        context = tools.get_segment_context(idx)
        current = context["segment"]["text"]
        neighbor_text = " ".join(seg["text"] for seg in context["neighbors"])
        key_query = " ".join(_keywords(neighbor_text, limit=10)) or current
        queries = [current]
        if key_query and key_query != current:
            queries.append(key_query)
        queries = queries[: max(0, int(config.search_budget_per_segment))]

        review = {
            "segment_index": idx,
            "segment_id": item["segment_id"],
            "segment_text": current,
            "ambiguity": item,
            "context": context,
            "initial_candidates": tools.get_candidate_scores(idx, top_k=config.shortlist_size),
            "custom_queries": [],
            "inspections": [],
            "suggestion": None,
        }
        custom_counts: Dict[int, float] = {}
        for query in queries:
            results = tools.search_video_index(query, top_k=config.shortlist_size, exclude_assigned=False)
            review["custom_queries"].append({"query": query, "results": results})
            for rank, result in enumerate(results):
                col = result.get("candidate_index")
                if col is None:
                    continue
                custom_counts[int(col)] = custom_counts.get(int(col), 0.0) + max(0.0, 1.0 - (rank * 0.15))
        if config.allow_contact_sheet:
            candidate_ids = []
            for candidate in review["initial_candidates"][: min(3, config.shortlist_size)]:
                candidate_ids.append(int(candidate["candidate_index"]))
            for col in sorted(set(candidate_ids)):
                try:
                    review["inspections"].append(tools.inspect_clip(col))
                except Exception as exc:
                    review["inspections"].append({"candidate_index": col, "error": str(exc)})
        if custom_counts:
            best_col = max(custom_counts, key=custom_counts.get)
            review["suggestion"] = {"candidate_index": best_col, "confidence": float(custom_counts[best_col])}
            decisions["expanded_candidates"][str(idx)] = sorted(custom_counts)
            decisions["advisory_scores"][str(idx)] = {str(k): float(v) for k, v in custom_counts.items()}
        decisions["reviewed_segments"].append(review)
    return decisions


def _codex_decision_schema() -> Dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "reviewed_segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "segment_index": {"type": "integer", "minimum": 0},
                        "rationale": {"type": "string"},
                    },
                    "required": ["segment_index", "rationale"],
                    "additionalProperties": False,
                },
            },
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "segment_index": {"type": "integer", "minimum": 0},
                        "candidate_index": {"type": "integer", "minimum": 0},
                        "action": {"type": "string", "enum": ["recommend", "lock", "exclude"]},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "reason": {"type": "string"},
                    },
                    "required": ["segment_index", "candidate_index", "action", "confidence", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["summary", "reviewed_segments", "decisions"],
        "additionalProperties": False,
    }


def _codex_agent_prompt(ambiguous: Sequence[Dict[str, Any]], config: HybridAgentConfig) -> str:
    review_items = list(ambiguous) if config.review_scope == "all" else list(ambiguous[: config.max_agent_segments])
    allowed = [int(item["segment_index"]) for item in review_items]
    review_groups = [
        {
            "segment_index": int(item["segment_index"]),
            "role": item.get("selection_role", "ambiguity_seed"),
            "group_seed": int(item.get("review_group_seed", item["segment_index"])),
            "linked_segments": [int(value) for value in item.get("linked_segment_indices", [])],
        }
        for item in review_items
    ]
    scope_instructions = """
This is a full-sequence audit. Treat the initial Hungarian assignment as an untrusted proposal, not as a
default or a reason to preserve a pair. Call get_candidate_catalog, then use inspect_candidate to inspect every
anonymous candidate in the library exactly once, including candidates absent from every segment shortlist.
Construct a complete one-to-one storyboard independently from script semantics, sampled frames,
character/action continuity, and narrative order. Compare adjacent and semantically similar beats explicitly.
Before returning, verify the complete proposed permutation has no clip reuse. If it differs from the initial
proposal, emit a recommendation for every changed row in each coordinated cycle; never emit only a partial
cycle. Similarity scores are retrieval hints, not votes that override stronger visual evidence.
""" if config.review_scope == "all" else """
This is an uncertainty-focused audit. Concentrate on the supplied linked assignment groups and emit every
side of a coordinated no-reuse cycle when visual and narrative evidence supports a change.
"""
    tool_order_instructions = """
Begin with get_global_script_context and get_candidate_catalog. Inspect every catalog candidate with
inspect_candidate and draft an independent complete storyboard before calling get_initial_assignment,
get_candidate_scores, or search_video_index. Do not revise the independent draft merely to agree with
VideoPrism; use retrieval only to investigate genuine uncertainty. Finish with an adversarial second pass:
challenge each adjacent placement and each changed cycle against the sampled frames before returning JSON.
""" if config.review_scope == "all" else """
Begin with get_global_script_context and get_initial_assignment, then inspect candidate scores for every
allowed segment.
"""
    return f"""You are the reasoning layer of a benchmark-safe video clip matcher.

Review only segment indices {allowed}. Their non-oracle review groups are
{json.dumps(review_groups, sort_keys=True)}. A group contains rows changed by the best alternative Hungarian
assignment when one uncertain pair is forbidden. Evaluate linked rows together and prefer an explicit,
coordinated multi-segment recommendation when a swap is visually and narratively better than the initial
assignment. Rows labeled global_audit are included for full-sequence consistency checking even when their
retrieval margin is not small. Do not recommend only one side of a no-reuse swap.
{scope_instructions}

Use only the hybrid_agentic MCP tools. {tool_order_instructions}
You may make at most {config.search_budget_per_segment} custom VideoPrism/FAISS searches per
segment and inspect only anonymous candidates returned by the tools. Custom search text is embedded exactly
as you write it; the baseline prompt template is never added to your query.

Reason from script meaning, global narrative order, anonymous similarity scores, and sampled visual frames.
Segments with has_narration=false are intentional silent beats. Infer their role from adjacent narration and
visual continuity; never invent a retrieval query from their numeric segment index.
Use score ranks within each result set; do not compare absolute values across the baseline-matrix and raw-cosine
score spaces.
Do not use shell commands, web search, memory, filenames, paths, durations, timestamps, audio, captions,
benchmark metadata, or ground truth. Clip reuse is forbidden. Candidate indices are opaque IDs, not ordinal clues.

Decision mode is {config.decision_mode}. In advisory mode, recommend and exclude are soft score adjustments;
locks are also treated as recommendations. In hard_lock mode, confident locks and exclusions are hard
constraints while ordinary recommendations remain soft. Every hard constraint must have confidence at least
{config.hard_lock_min_confidence:.2f} and must target the exact segment/candidate pair you inspected with
inspect_clip; otherwise it is safely downgraded to a soft adjustment. Use exclude only for a clear visual
contradiction seen in the inspected frames.
If the initial assignment is best, record the segment as reviewed and emit no decision for it.
Return the requested structured JSON only after finishing all tool calls."""


def _build_codex_exec_command(
    codex_path: str,
    workspace: Path,
    schema_path: Path,
    response_path: Path,
    proxy_path: Path,
    enabled_tools: Sequence[str],
    config: HybridAgentConfig,
) -> List[str]:
    """Build a reproducible Codex command with explicit model and reasoning."""
    return [
        codex_path,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--cd",
        str(workspace),
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(response_path),
        "--json",
        "--color",
        "never",
        "--model",
        config.codex_model,
        "-c",
        f"model_reasoning_effort={json.dumps(config.codex_reasoning_effort)}",
        "-c",
        "features.shell_tool=false",
        "-c",
        'web_search="disabled"',
        "-c",
        "memories.generate_memories=false",
        "-c",
        f"mcp_servers.hybrid_agentic.command={json.dumps(sys.executable)}",
        "-c",
        f"mcp_servers.hybrid_agentic.args={json.dumps([str(proxy_path)])}",
        "-c",
        'mcp_servers.hybrid_agentic.env_vars=["HYBRID_AGENT_BRIDGE_URL","HYBRID_AGENT_BRIDGE_TOKEN"]',
        "-c",
        "mcp_servers.hybrid_agentic.required=true",
        "-c",
        'mcp_servers.hybrid_agentic.default_tools_approval_mode="approve"',
        "-c",
        f"mcp_servers.hybrid_agentic.enabled_tools={json.dumps(list(enabled_tools))}",
        "-c",
        "mcp_servers.hybrid_agentic.startup_timeout_sec=20",
        "-c",
        "mcp_servers.hybrid_agentic.tool_timeout_sec=120",
    ]


def _hard_constraint_evidence(
    row: int,
    col: int,
    confidence: float,
    audit_log: Sequence[Dict[str, Any]],
    config: HybridAgentConfig,
) -> Tuple[bool, Optional[str]]:
    if confidence < float(config.hard_lock_min_confidence):
        return False, (
            f"confidence {confidence:.3f} is below the hard-constraint threshold "
            f"{config.hard_lock_min_confidence:.3f}"
        )
    if not config.require_direct_inspection_for_hard_constraints:
        return True, None
    inspected_pairs = {
        (
            int(call.get("arguments", {}).get("segment_index", -1)),
            int(call.get("arguments", {}).get("candidate_index", -1)),
        )
        for call in audit_log
        if call.get("tool") == "inspect_clip"
        and call.get("result", {}).get("contact_sheet_available") is True
    }
    if (row, col) not in inspected_pairs:
        return False, "the exact segment/candidate pair has no successful contact-sheet inspection"
    return True, None


def _codex_failure_message(stdout: str, stderr: str) -> str:
    """Extract the actionable error emitted by `codex exec --json`."""
    messages: List[str] = []
    for line in (stdout or "").splitlines():
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(event, dict):
            continue
        message = event.get("message")
        if isinstance(message, str) and message.strip():
            messages.append(message.strip())
        error = event.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            messages.append(error["message"].strip())
    if messages:
        return messages[-1]
    return (stderr or stdout or "Codex exited without an error message")[-2000:].strip()


def _is_transient_codex_failure(message: str) -> bool:
    normalized = (message or "").lower()
    return any(marker in normalized for marker in (
        "at capacity",
        "temporarily unavailable",
        "service unavailable",
        "overloaded",
        "rate limit",
        "try again",
        "http 429",
        "http 503",
    ))


def _known_candidates_for_decisions(
    bridge: HybridAgentToolBridge,
    allowed_segments: Iterable[int],
    config: HybridAgentConfig,
) -> Dict[int, set[int]]:
    known = {idx: set(values) for idx, values in bridge.known_candidates.items()}
    if config.review_scope == "all":
        globally_inspected = set(bridge.globally_inspected_candidates)
        for row in allowed_segments:
            known.setdefault(int(row), set()).update(globally_inspected)
    return known


def codex_agent_review(
    tools: HybridAgentTools,
    ambiguous: List[Dict[str, Any]],
    config: HybridAgentConfig,
    run_dir: Path,
    status: Dict[str, Any],
) -> Dict[str, Any]:
    if not status.get("ready"):
        raise RuntimeError(status.get("auth_message") or status.get("message") or "Codex CLI is not ready")
    codex_path = str(status["path"])
    proxy_path = PROJECT_ROOT / "src" / "hybrid_agentic_mcp.py"
    if not proxy_path.exists():
        raise FileNotFoundError(f"Hybrid MCP proxy is missing: {proxy_path}")

    schema_path = run_dir / "codex_decision_schema.json"
    response_path = run_dir / "codex_decision_response.json"
    events_path = run_dir / "codex_events.jsonl"
    stderr_path = run_dir / "codex_stderr.log"
    workspace = run_dir / "agent_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    _write_json(schema_path, _codex_decision_schema())

    base_codex_env = os.environ.copy()
    enabled_tools = [
        "get_global_script_context",
        "get_candidate_catalog",
        "get_segment_context",
        "get_candidate_scores",
        "get_initial_assignment",
        "search_video_index",
        "inspect_clip",
        "inspect_candidate",
    ]
    command = _build_codex_exec_command(
        codex_path,
        workspace,
        schema_path,
        response_path,
        proxy_path,
        enabled_tools,
        config,
    )
    command.append(_codex_agent_prompt(ambiguous, config))

    started = time.time()
    completed: Optional[subprocess.CompletedProcess[str]] = None
    attempt_artifacts: List[Dict[str, Any]] = []
    successful_bridge: Optional[HybridAgentToolBridge] = None
    for attempt in range(1, 3):
        response_path.unlink(missing_ok=True)
        bridge = HybridAgentToolBridge(tools, ambiguous, config)
        bridge_url = bridge.start()
        codex_env = base_codex_env.copy()
        codex_env["HYBRID_AGENT_BRIDGE_URL"] = bridge_url
        codex_env["HYBRID_AGENT_BRIDGE_TOKEN"] = bridge.token
        try:
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=max(30, int(config.codex_timeout_seconds)),
                    check=False,
                    env=codex_env,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"Codex agent timed out after {config.codex_timeout_seconds}s") from exc
        finally:
            bridge.close()

        attempt_events = run_dir / f"codex_events_attempt_{attempt}.jsonl"
        attempt_stderr = run_dir / f"codex_stderr_attempt_{attempt}.log"
        attempt_events.write_text(completed.stdout or "", encoding="utf-8")
        attempt_stderr.write_text(completed.stderr or "", encoding="utf-8")
        failure_message = "" if completed.returncode == 0 else _codex_failure_message(
            completed.stdout or "",
            completed.stderr or "",
        )
        attempt_artifacts.append({
            "attempt": attempt,
            "returncode": completed.returncode,
            "events": str(attempt_events),
            "stderr": str(attempt_stderr),
            "error": failure_message or None,
        })
        if completed.returncode == 0:
            successful_bridge = bridge
            break
        if attempt == 1 and _is_transient_codex_failure(failure_message):
            logger.warning("Transient Codex failure; retrying once: %s", failure_message)
            time.sleep(2.0)
            continue
        raise RuntimeError(
            f"Codex agent failed with exit code {completed.returncode}: {failure_message}"
        )

    if completed is None or successful_bridge is None:
        raise RuntimeError("Codex agent did not start")
    bridge = successful_bridge
    events_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Codex agent failed with exit code {completed.returncode}: "
            f"{_codex_failure_message(completed.stdout or '', completed.stderr or '')}"
        )
    if not response_path.exists():
        raise RuntimeError("Codex completed without writing a structured decision response")
    payload = _read_json(response_path, {})
    if not isinstance(payload, dict):
        raise RuntimeError("Codex decision response is not a JSON object")

    allowed_segments = {int(item["segment_index"]) for item in ambiguous}
    candidate_count = len(tools.metadata)
    decisions: Dict[str, Any] = {
        "mode": "codex_mcp",
        "decision_mode": config.decision_mode,
        "reviewed_segments": [],
        "locks": {},
        "exclusions": {},
        "advisory_scores": {},
        "advisory_penalties": {},
        "expanded_candidates": {},
        "rejected_agent_decisions": [],
        "hard_constraint_downgrades": [],
        "tool_calls": bridge.audit_log,
        "codex_summary": str(payload.get("summary") or ""),
        "codex_artifacts": {
            "response": str(response_path),
            "events": str(events_path),
            "stderr": str(stderr_path),
            "duration_seconds": round(time.time() - started, 2),
            "model": config.codex_model,
            "reasoning_effort": config.codex_reasoning_effort,
            "attempts": attempt_artifacts,
        },
    }
    for reviewed in payload.get("reviewed_segments", []):
        try:
            idx = int(reviewed["segment_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if idx in allowed_segments:
            decisions["reviewed_segments"].append({
                "segment_index": idx,
                "rationale": str(reviewed.get("rationale") or ""),
            })

    for call in bridge.audit_log:
        if call.get("tool") == "search_video_index":
            idx = str(call.get("arguments", {}).get("segment_index"))
            candidates = call.get("result", {}).get("candidates", [])
            decisions["expanded_candidates"].setdefault(idx, [])
            decisions["expanded_candidates"][idx].extend(
                int(item["candidate_index"]) for item in candidates if item.get("candidate_index") is not None
            )
    for idx, values in list(decisions["expanded_candidates"].items()):
        decisions["expanded_candidates"][idx] = sorted(set(values))

    pending_locks: List[Tuple[float, int, int, Dict[str, Any]]] = []
    known_candidates = _known_candidates_for_decisions(bridge, allowed_segments, config)
    for proposed in payload.get("decisions", []):
        try:
            row = int(proposed["segment_index"])
            col = int(proposed["candidate_index"])
            action = str(proposed["action"])
            confidence = max(0.0, min(1.0, float(proposed["confidence"])))
        except (KeyError, TypeError, ValueError) as exc:
            decisions["rejected_agent_decisions"].append({"decision": proposed, "reason": str(exc)})
            continue
        if row not in allowed_segments or not 0 <= col < candidate_count:
            decisions["rejected_agent_decisions"].append({"decision": proposed, "reason": "segment or candidate is outside the approved range"})
            continue
        if col not in known_candidates.get(row, set()):
            decisions["rejected_agent_decisions"].append({
                "decision": proposed,
                "reason": "candidate was not exposed by an approved retrieval or full-audit inspection tool",
            })
            continue
        if action == "exclude":
            if not config.allow_exclusions:
                decisions["rejected_agent_decisions"].append({"decision": proposed, "reason": "exclusions are disabled"})
            elif config.decision_mode == "hard_lock":
                supported, reason = _hard_constraint_evidence(row, col, confidence, bridge.audit_log, config)
                if supported:
                    decisions["exclusions"].setdefault(str(row), []).append(col)
                else:
                    penalties = decisions["advisory_penalties"].setdefault(str(row), {})
                    penalties[str(col)] = max(float(penalties.get(str(col), 0.0)), confidence)
                    decisions["hard_constraint_downgrades"].append({
                        "decision": proposed,
                        "reason": reason,
                        "applied_as": "advisory_penalty",
                    })
            else:
                penalties = decisions["advisory_penalties"].setdefault(str(row), {})
                penalties[str(col)] = max(float(penalties.get(str(col), 0.0)), confidence)
        elif action == "lock" and config.decision_mode == "hard_lock":
            supported, reason = _hard_constraint_evidence(row, col, confidence, bridge.audit_log, config)
            if supported:
                pending_locks.append((confidence, row, col, proposed))
            else:
                scores = decisions["advisory_scores"].setdefault(str(row), {})
                scores[str(col)] = max(float(scores.get(str(col), 0.0)), confidence)
                decisions["hard_constraint_downgrades"].append({
                    "decision": proposed,
                    "reason": reason,
                    "applied_as": "advisory_recommendation",
                })
        else:
            scores = decisions["advisory_scores"].setdefault(str(row), {})
            scores[str(col)] = max(float(scores.get(str(col), 0.0)), confidence)

    locked_rows = set()
    locked_cols = set()
    for _confidence, row, col, proposed in sorted(pending_locks, key=lambda item: item[0], reverse=True):
        if row in locked_rows:
            decisions["rejected_agent_decisions"].append({
                "decision": proposed,
                "reason": "a higher-confidence lock already exists for this segment",
            })
            continue
        if col in locked_cols:
            decisions["rejected_agent_decisions"].append({
                "decision": proposed,
                "reason": "a higher-confidence lock already uses this no-reuse candidate",
            })
            continue
        decisions["locks"][str(row)] = col
        locked_rows.add(row)
        locked_cols.add(col)
    return decisions


def apply_agent_decisions(matrix: np.ndarray, decisions: Dict[str, Any], config: HybridAgentConfig) -> Tuple[np.ndarray, Dict[int, int], Dict[int, List[int]]]:
    refined = np.array(matrix, dtype=np.float64, copy=True)
    locks = {int(k): int(v) for k, v in decisions.get("locks", {}).items()}
    exclusions = {int(k): [int(v) for v in vals] for k, vals in decisions.get("exclusions", {}).items()}
    if config.decision_mode != "hard_lock":
        locks = {}
        exclusions = {}
    for row_key, scores in decisions.get("advisory_scores", {}).items():
        row = int(row_key)
        if row < 0 or row >= refined.shape[0]:
            continue
        for col_key, value in scores.items():
            col = int(col_key)
            if 0 <= col < refined.shape[1]:
                refined[row, col] += float(config.advisory_boost) * float(value)
    for row_key, scores in decisions.get("advisory_penalties", {}).items():
        row = int(row_key)
        if row < 0 or row >= refined.shape[0]:
            continue
        for col_key, value in scores.items():
            col = int(col_key)
            if 0 <= col < refined.shape[1]:
                refined[row, col] -= float(config.advisory_boost) * float(value)
    return refined, locks, exclusions


def analyze_full_audit_cycles(
    decisions: Dict[str, Any],
    initial_assignment: Sequence[int],
) -> Dict[str, Any]:
    """Describe complete no-reuse assignment components without labels."""
    recommendations: Dict[int, int] = {}
    for row_key, scores in decisions.get("advisory_scores", {}).items():
        if not scores:
            continue
        row = int(row_key)
        candidate = int(max(scores.items(), key=lambda item: float(item[1]))[0])
        if 0 <= row < len(initial_assignment) and candidate != int(initial_assignment[row]):
            recommendations[row] = candidate
    for row_key, candidate_value in decisions.get("locks", {}).items():
        row = int(row_key)
        candidate = int(candidate_value)
        if 0 <= row < len(initial_assignment) and candidate != int(initial_assignment[row]):
            recommendations[row] = candidate

    auxiliary_rows = {
        int(row)
        for key in ("advisory_penalties", "exclusions")
        for row, values in decisions.get(key, {}).items()
        if values
    }
    analysis: Dict[str, Any] = {
        "changed_rows": sorted(recommendations),
        "recommendations": {str(row): candidate for row, candidate in sorted(recommendations.items())},
        "component_count": 0,
        "components": [],
        "structurally_valid": True,
        "reason": "No changed recommendation rows were emitted.",
    }
    if not recommendations:
        if auxiliary_rows:
            analysis.update({
                "structurally_valid": False,
                "reason": "Penalty or exclusion adjustments were emitted without a coordinated target assignment.",
            })
        return analysis
    if auxiliary_rows - set(recommendations):
        analysis.update({
            "structurally_valid": False,
            "reason": "Penalty or exclusion rows are disconnected from the coordinated target assignment.",
        })
        return analysis

    targets = list(recommendations.values())
    if len(set(targets)) != len(targets):
        analysis.update({
            "structurally_valid": False,
            "reason": "Multiple rows target the same no-reuse candidate.",
        })
        return analysis

    owner = {int(candidate): row for row, candidate in enumerate(initial_assignment) if int(candidate) >= 0}
    graph = {row: set() for row in recommendations}
    for row, candidate in recommendations.items():
        linked_owner = owner.get(candidate)
        if linked_owner is not None and linked_owner not in graph:
            analysis.update({
                "structurally_valid": False,
                "reason": (
                    f"Segment {row} targets candidate {candidate}, which is still owned by unchanged "
                    f"segment {linked_owner}."
                ),
            })
            return analysis
        if linked_owner in graph:
            graph[row].add(linked_owner)
            graph[linked_owner].add(row)

    unseen = set(graph)
    components: List[Dict[str, Any]] = []
    while unseen:
        stack = [unseen.pop()]
        rows: List[int] = []
        while stack:
            row = stack.pop()
            rows.append(row)
            for neighbor in graph[row]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        rows.sort()
        baseline_candidates = {int(initial_assignment[row]) for row in rows}
        proposed_candidates = {int(recommendations[row]) for row in rows}
        has_unassigned_target = any(owner.get(int(recommendations[row])) is None for row in rows)
        component_type = "path_to_unassigned" if has_unassigned_target else "cycle"
        if component_type == "cycle" and baseline_candidates != proposed_candidates:
            analysis.update({
                "structurally_valid": False,
                "reason": f"Rows {rows} do not form a closed assignment cycle.",
            })
            return analysis
        component_id = "component_" + "_".join(str(row) for row in rows)
        components.append({
            "component_id": component_id,
            "component_type": component_type,
            "rows": [
                {
                    "segment_index": row,
                    "baseline_candidate_index": int(initial_assignment[row]),
                    "proposed_candidate_index": int(recommendations[row]),
                    "previous_context": (
                        {
                            "segment_index": row - 1,
                            "baseline_candidate_index": int(initial_assignment[row - 1]),
                            "proposed_candidate_index": int(
                                recommendations.get(row - 1, initial_assignment[row - 1])
                            ),
                        }
                        if row > 0 else None
                    ),
                    "next_context": (
                        {
                            "segment_index": row + 1,
                            "baseline_candidate_index": int(initial_assignment[row + 1]),
                            "proposed_candidate_index": int(
                                recommendations.get(row + 1, initial_assignment[row + 1])
                            ),
                        }
                        if row + 1 < len(initial_assignment) else None
                    ),
                }
                for row in rows
            ],
        })

    components.sort(key=lambda component: tuple(row["segment_index"] for row in component["rows"]))
    analysis.update({
        "component_count": len(components),
        "components": components,
        "reason": f"Found {len(components)} complete no-reuse assignment component(s).",
    })
    return analysis


def _filter_full_audit_adjustments(
    decisions: Dict[str, Any],
    accepted_rows: Iterable[int],
) -> Dict[str, Any]:
    accepted = {int(row) for row in accepted_rows}
    removed: Dict[str, Dict[str, Any]] = {}
    for key in ("advisory_scores", "advisory_penalties", "locks", "exclusions"):
        values = decisions.get(key, {})
        kept = {str(row): value for row, value in values.items() if int(row) in accepted}
        rejected = {str(row): value for row, value in values.items() if int(row) not in accepted}
        decisions[key] = kept
        if rejected:
            removed[key] = rejected
    if removed:
        decisions["gated_out_adjustments"] = removed
    return decisions


def gate_full_audit_decisions(
    decisions: Dict[str, Any],
    initial_assignment: Sequence[int],
) -> Dict[str, Any]:
    """Legacy structural gate used when no independent Codex critic runs."""
    analysis = analyze_full_audit_cycles(decisions, initial_assignment)
    components = analysis.get("components", [])
    accepted = bool(analysis.get("structurally_valid")) and len(components) <= 1
    accepted_rows = {
        int(row["segment_index"])
        for component in components
        for row in component.get("rows", [])
    } if accepted else set()
    gate = {
        **analysis,
        "policy": "single_connected_assignment_component",
        "accepted": accepted,
        "accepted_component_ids": [component["component_id"] for component in components] if accepted else [],
        "reason": (
            analysis.get("reason")
            if not analysis.get("structurally_valid") or len(components) <= 1
            else "Disconnected rewrite components require independent verification and were not applied."
        ),
    }
    _filter_full_audit_adjustments(decisions, accepted_rows)
    decisions["full_audit_cycle_gate"] = gate
    return decisions


def _cycle_critic_schema() -> Dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "components": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "component_id": {"type": "string"},
                        "verdict": {"type": "string", "enum": ["accept", "reject"]},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "reason": {"type": "string"},
                        "row_checks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "segment_index": {"type": "integer", "minimum": 0},
                                    "baseline_candidate_index": {"type": "integer", "minimum": 0},
                                    "proposed_candidate_index": {"type": "integer", "minimum": 0},
                                    "verdict": {
                                        "type": "string",
                                        "enum": ["improves", "neutral", "contradicts"],
                                    },
                                    "reason": {"type": "string"},
                                },
                                "required": [
                                    "segment_index", "baseline_candidate_index", "proposed_candidate_index",
                                    "verdict", "reason",
                                ],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["component_id", "verdict", "confidence", "reason", "row_checks"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["summary", "components"],
        "additionalProperties": False,
    }


def _cycle_critic_prompt(components: Sequence[Dict[str, Any]], config: HybridAgentConfig) -> str:
    manifest = json.dumps(list(components), sort_keys=True)
    candidate_ids = set()
    for component in components:
        for row in component.get("rows", []):
            for key in ("baseline_candidate_index", "proposed_candidate_index"):
                candidate_ids.add(int(row[key]))
            for context_key in ("previous_context", "next_context"):
                context = row.get(context_key)
                if not context:
                    continue
                candidate_ids.add(int(context["baseline_candidate_index"]))
                candidate_ids.add(int(context["proposed_candidate_index"]))
    candidate_ids = sorted(candidate_ids)
    return f"""You are an independent, skeptical verifier for a benchmark-safe video assignment system.

The first agent proposed these atomic no-reuse assignment components:
{manifest}

Do not assume the proposal is correct and do not infer the first agent's rationale. Use only the
hybrid_agentic MCP tools. First call get_global_script_context. Then inspect each anonymous candidate in
{candidate_ids} exactly once with inspect_candidate. Compare the baseline and proposed candidate for every
row using visible action, objects, setting, state, character continuity, recipe/procedure stage, and adjacent
narrative order. The manifest includes previous_context and next_context assignment anchors; compare the visual
transition into and out of each row under the complete baseline and proposed mappings. Segments with
has_narration=false are intentional silent beats and must be judged from their neighbors and visual continuity.

Each component is atomic: compare the complete proposed storyboard against the complete baseline storyboard.
Mark each row improves, neutral, or contradicts. Neutral includes an individually ambiguous trade-off that is
needed to unlock several stronger no-reuse corrections; it is not an automatic rejection. Use contradicts only
for a clear visual or narrative conflict. Accept a component when it has a clear net storyboard advantage, at
least one improved row, no contradicting row, and confidence at least
{config.cycle_critic_min_confidence:.2f}. Reject it when the net benefit is unclear or any proposed placement
clearly contradicts its beat. Return exactly one row_check for every supplied row using the exact indices.

Do not use retrieval scores, custom searches, shell commands, web search, memory, filenames, paths, durations,
timestamps, audio, captions, benchmark metadata, or ground truth. Candidate indices are opaque IDs, not order
clues. Return the requested structured JSON only after all required inspections."""


def codex_cycle_critic(
    tools: HybridAgentTools,
    decisions: Dict[str, Any],
    initial_assignment: Sequence[int],
    config: HybridAgentConfig,
    run_dir: Path,
    status: Dict[str, Any],
) -> Dict[str, Any]:
    """Independently accept or reject each complete assignment component."""
    analysis = analyze_full_audit_cycles(decisions, initial_assignment)
    components = analysis.get("components", [])
    if not analysis.get("structurally_valid") or not components:
        return gate_full_audit_decisions(decisions, initial_assignment)
    if not config.verify_assignment_cycles or not config.use_codex:
        gated = gate_full_audit_decisions(decisions, initial_assignment)
        gated["full_audit_cycle_gate"]["critic_status"] = "disabled"
        return gated
    if not status.get("ready"):
        raise RuntimeError(status.get("auth_message") or status.get("message") or "Codex CLI is not ready")

    codex_path = str(status["path"])
    proxy_path = PROJECT_ROOT / "src" / "hybrid_agentic_mcp.py"
    schema_path = run_dir / "cycle_critic_schema.json"
    response_path = run_dir / "cycle_critic_response.json"
    events_path = run_dir / "cycle_critic_events.jsonl"
    stderr_path = run_dir / "cycle_critic_stderr.log"
    workspace = run_dir / "cycle_critic_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    _write_json(schema_path, _cycle_critic_schema())

    review_rows = sorted({
        int(row["segment_index"])
        for component in components
        for row in component.get("rows", [])
    })
    bridge_items = [{"segment_index": row} for row in review_rows]
    command = _build_codex_exec_command(
        codex_path,
        workspace,
        schema_path,
        response_path,
        proxy_path,
        ["get_global_script_context", "inspect_candidate"],
        config,
    )
    command.append(_cycle_critic_prompt(components, config))

    started = time.time()
    completed: Optional[subprocess.CompletedProcess[str]] = None
    successful_bridge: Optional[HybridAgentToolBridge] = None
    attempt_artifacts: List[Dict[str, Any]] = []
    for attempt in range(1, 3):
        response_path.unlink(missing_ok=True)
        bridge = HybridAgentToolBridge(tools, bridge_items, config)
        bridge_url = bridge.start()
        codex_env = os.environ.copy()
        codex_env["HYBRID_AGENT_BRIDGE_URL"] = bridge_url
        codex_env["HYBRID_AGENT_BRIDGE_TOKEN"] = bridge.token
        try:
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=max(30, int(config.codex_timeout_seconds)),
                    check=False,
                    env=codex_env,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"Codex cycle critic timed out after {config.codex_timeout_seconds}s") from exc
        finally:
            bridge.close()

        attempt_events = run_dir / f"cycle_critic_events_attempt_{attempt}.jsonl"
        attempt_stderr = run_dir / f"cycle_critic_stderr_attempt_{attempt}.log"
        attempt_events.write_text(completed.stdout or "", encoding="utf-8")
        attempt_stderr.write_text(completed.stderr or "", encoding="utf-8")
        failure_message = "" if completed.returncode == 0 else _codex_failure_message(
            completed.stdout or "", completed.stderr or ""
        )
        attempt_artifacts.append({
            "attempt": attempt,
            "returncode": completed.returncode,
            "events": str(attempt_events),
            "stderr": str(attempt_stderr),
            "error": failure_message or None,
        })
        if completed.returncode == 0:
            successful_bridge = bridge
            break
        if attempt == 1 and _is_transient_codex_failure(failure_message):
            logger.warning("Transient Codex cycle-critic failure; retrying once: %s", failure_message)
            time.sleep(2.0)
            continue
        raise RuntimeError(f"Codex cycle critic failed with exit code {completed.returncode}: {failure_message}")

    if completed is None or successful_bridge is None:
        raise RuntimeError("Codex cycle critic did not start")
    events_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    if not response_path.exists():
        raise RuntimeError("Codex cycle critic completed without a structured response")
    payload = _read_json(response_path, {})
    if not isinstance(payload, dict):
        raise RuntimeError("Codex cycle critic response is not a JSON object")

    expected = {component["component_id"]: component for component in components}
    returned: Dict[str, Dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for verdict in payload.get("components", []):
        component_id = str(verdict.get("component_id") or "")
        if component_id in returned:
            duplicate_ids.add(component_id)
        returned[component_id] = verdict

    inspection_counts = collections.Counter(
        int(call.get("arguments", {}).get("candidate_index", -1))
        for call in successful_bridge.audit_log
        if call.get("tool") == "inspect_candidate"
        and call.get("result", {}).get("contact_sheet_available") is True
    )

    accepted_ids: set[str] = set()
    verdict_log: List[Dict[str, Any]] = []
    for component_id, component in expected.items():
        verdict = returned.get(component_id)
        validation_errors: List[str] = []
        if verdict is None:
            validation_errors.append("The critic omitted this component.")
            verdict = {}
        if component_id in duplicate_ids:
            validation_errors.append("The critic returned this component more than once.")
        required_candidates = set()
        for row in component.get("rows", []):
            for key in ("baseline_candidate_index", "proposed_candidate_index"):
                required_candidates.add(int(row[key]))
            for context_key in ("previous_context", "next_context"):
                context = row.get(context_key)
                if not context:
                    continue
                required_candidates.add(int(context["baseline_candidate_index"]))
                required_candidates.add(int(context["proposed_candidate_index"]))
        missing_or_duplicate = sorted(
            candidate for candidate in required_candidates if inspection_counts[candidate] != 1
        )
        if missing_or_duplicate:
            validation_errors.append(
                "Every baseline/proposed candidate must be successfully inspected exactly once; "
                f"invalid candidates: {missing_or_duplicate}."
            )
        expected_rows = {
            (
                int(row["segment_index"]),
                int(row["baseline_candidate_index"]),
                int(row["proposed_candidate_index"]),
            )
            for row in component.get("rows", [])
        }
        returned_rows = set()
        row_verdicts = []
        for row in verdict.get("row_checks", []):
            try:
                key = (
                    int(row["segment_index"]),
                    int(row["baseline_candidate_index"]),
                    int(row["proposed_candidate_index"]),
                )
            except (KeyError, TypeError, ValueError):
                validation_errors.append("A row check contains invalid indices.")
                continue
            returned_rows.add(key)
            row_verdicts.append(str(row.get("verdict") or "reject"))
        if returned_rows != expected_rows:
            validation_errors.append("Row checks do not exactly match the supplied assignment component.")
        confidence = max(0.0, min(1.0, float(verdict.get("confidence", 0.0) or 0.0)))
        accepted = (
            not validation_errors
            and str(verdict.get("verdict") or "reject") == "accept"
            and confidence >= float(config.cycle_critic_min_confidence)
            and len(row_verdicts) == len(expected_rows)
            and "improves" in row_verdicts
            and "contradicts" not in row_verdicts
        )
        if accepted:
            accepted_ids.add(component_id)
        verdict_log.append({
            "component_id": component_id,
            "accepted": accepted,
            "confidence": confidence,
            "reason": str(verdict.get("reason") or ""),
            "validation_errors": validation_errors,
            "row_checks": verdict.get("row_checks", []),
        })

    unexpected_ids = sorted(set(returned) - set(expected))
    accepted_rows = {
        int(row["segment_index"])
        for component in components
        if component["component_id"] in accepted_ids
        for row in component.get("rows", [])
    }
    _filter_full_audit_adjustments(decisions, accepted_rows)
    decisions["cycle_critic"] = {
        "summary": str(payload.get("summary") or ""),
        "minimum_confidence": float(config.cycle_critic_min_confidence),
        "verdicts": verdict_log,
        "unexpected_component_ids": unexpected_ids,
        "tool_calls": successful_bridge.audit_log,
        "artifacts": {
            "response": str(response_path),
            "events": str(events_path),
            "stderr": str(stderr_path),
            "duration_seconds": round(time.time() - started, 2),
            "model": config.codex_model,
            "reasoning_effort": config.codex_reasoning_effort,
            "attempts": attempt_artifacts,
        },
    }
    decisions["full_audit_cycle_gate"] = {
        **analysis,
        "policy": "independent_codex_component_critic",
        "accepted": bool(accepted_ids),
        "accepted_component_ids": sorted(accepted_ids),
        "rejected_component_ids": sorted(set(expected) - accepted_ids),
        "reason": (
            f"Independent critic accepted {len(accepted_ids)} of {len(components)} component(s)."
        ),
    }
    return decisions


def run_hybrid_benchmark(run_config: HybridRunConfig, write_result: bool = True) -> Tuple[HybridResult, Dict[str, Any]]:
    validate_run_config(run_config)
    if run_config.benchmark and (not run_config.video_dir or not run_config.segments_file or not run_config.ground_truth_file):
        resolved = resolve_benchmark_paths(run_config.benchmark)
        run_config.video_dir = run_config.video_dir or resolved["video_dir"]
        run_config.segments_file = run_config.segments_file or resolved["segments_file"]
        run_config.ground_truth_file = run_config.ground_truth_file or resolved["ground_truth_file"]
    run_config.video_dir = _resolve_project_path(run_config.video_dir)
    run_config.segments_file = _resolve_project_path(run_config.segments_file)
    run_config.ground_truth_file = _resolve_project_path(run_config.ground_truth_file)
    if not run_config.video_dir or not run_config.segments_file:
        raise ValueError("video_dir and segments_file are required")

    output_root = Path(_resolve_project_path(run_config.output_dir) or run_config.output_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = output_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    codex = codex_status()
    log: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "pipeline": "hybrid_agentic",
        "benchmark": _normalize_benchmark_id(run_config.benchmark),
        "video_dir": run_config.video_dir,
        "segments_file": run_config.segments_file,
        "ground_truth_file": run_config.ground_truth_file,
        "videoprism_config": asdict(run_config.video),
        "agent_config": asdict(run_config.agent),
        "codex_status": codex,
        "notes": [],
    }
    if run_config.agent.use_codex and not codex.get("ready"):
        log["agent_error"] = "Codex CLI is not ready"
        _write_json(run_dir / "hybrid_agentic_run_log.json", log)
        raise RuntimeError(
            "Codex execution was requested, but Codex CLI is not installed and authenticated. "
            "Run --mode check-codex or disable --use-codex for the explicit deterministic heuristic."
        )
    if run_config.agent.use_codex:
        print(
            "[hybrid-agentic] Codex execution: "
            f"model={run_config.agent.codex_model} "
            f"reasoning_effort={run_config.agent.codex_reasoning_effort}",
            flush=True,
        )

    segments = load_segments(run_config.segments_file)
    query_segments = [dict(segment) for segment in segments]
    log["query_generation"] = {
        "query_mode": "original",
        "agent_custom_queries": "raw_unprompted_videoprism",
    }

    indexer, indexing_time, loaded_index, accelerator = build_indexer(run_config)
    log["index"] = {"loaded_existing": loaded_index, "entries": len(indexer.metadata_list), "cache_dir": str(indexer.index_dir)}
    log["accelerator"] = accelerator
    matcher = build_matcher(indexer, run_config.video, run_config.device)

    match_start = time.time()
    matrix, metadata = matcher.compute_similarity_matrix(
        query_segments,
        match_only=True,
        use_dual_softmax=bool(run_config.video.use_dual_softmax),
        score_normalization=run_config.video.score_normalization,
        csls_k=int(run_config.video.csls_k),
    )
    initial_assignment, _ = solve_assignment(matrix, metadata, query_segments)
    log["initial_assignment"] = assignment_to_log(initial_assignment, matrix, metadata, segments)
    log["query_contract"] = {
        "baseline_prompt_mode": run_config.video.prompt_mode,
        "baseline_query_mode": run_config.video.query_mode,
        "agent_search_prompt_mode": "none",
    }

    ambiguous = find_ambiguous_segments(
        matrix,
        initial_assignment,
        segments,
        threshold=float(run_config.agent.ambiguity_margin_threshold),
        max_segments=int(run_config.agent.max_agent_segments) if run_config.agent.enabled else 0,
        review_scope=run_config.agent.review_scope,
    )
    log["ambiguous_segments"] = ambiguous

    decisions = {
        "mode": "disabled",
        "reviewed_segments": [],
        "locks": {},
        "exclusions": {},
        "advisory_scores": {},
        "advisory_penalties": {},
    }
    tools: Optional[HybridAgentTools] = None
    if run_config.agent.enabled and ambiguous:
        tools = HybridAgentTools(matcher, indexer, segments, matrix, metadata, initial_assignment, run_dir)
        if run_config.agent.use_codex and codex.get("ready"):
            try:
                print(
                    f"[hybrid-agentic] Codex MCP reasoning started for {len(ambiguous)} ambiguous segment(s).",
                    flush=True,
                )
                decisions = codex_agent_review(tools, ambiguous, run_config.agent, run_dir, codex)
                print(
                    f"[hybrid-agentic] Codex MCP reasoning completed with {len(decisions.get('tool_calls', []))} tool call(s).",
                    flush=True,
                )
            except Exception as exc:
                logger.exception("Codex MCP review failed: %s", exc)
                log["agent_error"] = str(exc)
                _write_json(run_dir / "hybrid_agentic_run_log.json", log)
                raise RuntimeError(f"Codex MCP review failed; no fallback result was recorded: {exc}") from exc
        else:
            decisions = dry_run_agent_review(tools, ambiguous, run_config.agent)
    if run_config.agent.review_scope == "all":
        if (
            tools is not None
            and run_config.agent.use_codex
            and run_config.agent.verify_assignment_cycles
        ):
            print("[hybrid-agentic] Independent assignment-component critic started.", flush=True)
            decisions = codex_cycle_critic(
                tools,
                decisions,
                initial_assignment,
                run_config.agent,
                run_dir,
                codex,
            )
            print(
                "[hybrid-agentic] Independent assignment-component critic completed: "
                f"{decisions.get('full_audit_cycle_gate', {}).get('reason', 'no verdict')}",
                flush=True,
            )
        else:
            decisions = gate_full_audit_decisions(decisions, initial_assignment)
    log["agent_decisions"] = decisions

    refined_matrix, locks, exclusions = apply_agent_decisions(matrix, decisions, run_config.agent)
    final_assignment, rejected = solve_assignment(refined_matrix, metadata, segments, locks=locks, exclusions=exclusions, log=log)
    baseline_clip_selections = build_clip_selections(initial_assignment, matrix, metadata, segments)
    clip_selections = build_clip_selections(final_assignment, refined_matrix, metadata, segments)
    log["final_assignment"] = assignment_to_log(final_assignment, refined_matrix, metadata, segments)
    log["rejected_decisions"] = rejected
    log["prediction_contract"] = {
        "predictions_frozen_before_ground_truth_load": True,
        "ground_truth_visible_to_agent": False,
        "oracle_fallback_enabled": False,
        "output_selection": "requested_pipeline_output_only",
        "baseline_comparison": "post_hoc_evaluation_only",
    }
    matching_time = round(time.time() - match_start, 2)

    baseline_metrics = {
        "exact_match_accuracy": None,
        "top_3_accuracy": None,
        "top_5_accuracy": None,
        "mrr": None,
        "avg_similarity": None,
    }
    metrics = {
        "exact_match_accuracy": None,
        "top_3_accuracy": None,
        "top_5_accuracy": None,
        "mrr": None,
        "avg_similarity": None,
    }
    if run_config.ground_truth_file and Path(run_config.ground_truth_file).exists():
        # Ground truth is intentionally loaded only after both assignments and
        # their clip selections have been frozen above. These metrics report
        # performance; they never choose or replace a prediction in this run.
        evaluator = BenchmarkEvaluator(run_config.ground_truth_file)
        baseline_results = evaluator.evaluate(
            clip_selections=baseline_clip_selections,
            segment_dicts=segments,
            similarity_matrix=matrix,
            all_metadata=metadata,
            matching_mode="hybrid_agentic_baseline",
            allow_reuse=False,
            match_only=True,
            assignment_metadata={
                "assignment_method": "hungarian",
                "agentic": False,
                "post_hoc_role": "same_config_baseline",
            },
        )
        benchmark_results = evaluator.evaluate(
            clip_selections=clip_selections,
            segment_dicts=segments,
            similarity_matrix=refined_matrix,
            all_metadata=metadata,
            matching_mode="hybrid_agentic",
            allow_reuse=False,
            match_only=True,
            assignment_metadata={
                "assignment_method": "hungarian",
                "agentic": True,
                "agent_decision_mode": run_config.agent.decision_mode,
                "agent_review_scope": run_config.agent.review_scope,
                "reviewed_segments": len(decisions.get("reviewed_segments", [])),
                "codex_available": bool(codex.get("ready")),
            },
        )
        metrics = {
            "exact_match_accuracy": benchmark_results.exact_match_accuracy,
            "top_3_accuracy": benchmark_results.top_3_accuracy,
            "top_5_accuracy": benchmark_results.top_5_accuracy,
            "mrr": benchmark_results.mean_reciprocal_rank,
            "avg_similarity": benchmark_results.avg_predicted_similarity,
        }
        baseline_metrics = {
            "exact_match_accuracy": baseline_results.exact_match_accuracy,
            "top_3_accuracy": baseline_results.top_3_accuracy,
            "top_5_accuracy": baseline_results.top_5_accuracy,
            "mrr": baseline_results.mean_reciprocal_rank,
            "avg_similarity": baseline_results.avg_predicted_similarity,
        }
        log["baseline_benchmark_results"] = asdict(baseline_results)
        log["benchmark_results"] = asdict(benchmark_results)
        log["posthoc_comparison"] = {
            "exact_match_delta": float(metrics["exact_match_accuracy"] - baseline_metrics["exact_match_accuracy"]),
            "top_3_delta": float(metrics["top_3_accuracy"] - baseline_metrics["top_3_accuracy"]),
            "top_5_delta": float(metrics["top_5_accuracy"] - baseline_metrics["top_5_accuracy"]),
            "mrr_delta": float(metrics["mrr"] - baseline_metrics["mrr"]),
            "used_for_prediction_selection": False,
        }

    total_time = round(time.time() - total_start, 2)
    log_path = run_dir / "hybrid_agentic_run_log.json"
    _write_json(log_path, log)

    result_config = {
        "video": asdict(run_config.video),
        "agent": asdict(run_config.agent),
        "model_name": run_config.video.model_name,
        "num_frames": run_config.video.num_frames,
        "resolution": run_config.video.resolution,
        "use_dual_softmax": run_config.video.use_dual_softmax,
        "prompt_mode": run_config.video.prompt_mode,
        "baseline_prompt_mode": run_config.video.prompt_mode,
        "agent_search_prompt_mode": "none",
        "query_mode": run_config.video.query_mode,
        "score_normalization": run_config.video.score_normalization,
        "csls_k": run_config.video.csls_k,
        "assignment_method": "hybrid_agentic",
        "agent_enabled": run_config.agent.enabled,
        "agent_decision_mode": run_config.agent.decision_mode,
        "agent_review_scope": run_config.agent.review_scope,
        "agent_shortlist_size": run_config.agent.shortlist_size,
        "agent_ambiguity_margin_threshold": run_config.agent.ambiguity_margin_threshold,
        "agent_max_segments": run_config.agent.max_agent_segments,
        "agent_search_budget_per_segment": run_config.agent.search_budget_per_segment,
        "agent_allow_exclusions": run_config.agent.allow_exclusions,
        "agent_allow_candidate_expansion": run_config.agent.allow_candidate_expansion,
        "agent_allow_contact_sheet": run_config.agent.allow_contact_sheet,
        "agent_advisory_boost": run_config.agent.advisory_boost,
        "agent_codex_model": run_config.agent.codex_model,
        "agent_codex_reasoning_effort": run_config.agent.codex_reasoning_effort,
        "agent_verify_assignment_cycles": run_config.agent.verify_assignment_cycles,
        "agent_cycle_critic_min_confidence": run_config.agent.cycle_critic_min_confidence,
        "agent_runtime": decisions.get("mode", "disabled"),
        "codex_authenticated": bool(codex.get("authenticated")),
    }

    result = HybridResult(
        config=result_config,
        exact_match_accuracy=metrics["exact_match_accuracy"],
        top_3_accuracy=metrics["top_3_accuracy"],
        top_5_accuracy=metrics["top_5_accuracy"],
        mrr=metrics["mrr"],
        avg_similarity=metrics["avg_similarity"],
        baseline_exact_match_accuracy=baseline_metrics["exact_match_accuracy"],
        baseline_top_3_accuracy=baseline_metrics["top_3_accuracy"],
        baseline_top_5_accuracy=baseline_metrics["top_5_accuracy"],
        baseline_mrr=baseline_metrics["mrr"],
        exact_match_delta=(
            float(metrics["exact_match_accuracy"] - baseline_metrics["exact_match_accuracy"])
            if metrics["exact_match_accuracy"] is not None and baseline_metrics["exact_match_accuracy"] is not None
            else None
        ),
        indexing_time=indexing_time,
        matching_time=matching_time,
        total_time=total_time,
        reviewed_segments=len(decisions.get("reviewed_segments", [])),
        codex_available=bool(codex.get("ready")),
        agent_mode=decisions.get("mode", "unknown"),
        log_file=str(log_path),
    )

    if write_result:
        payload = {
            "timestamp": datetime.now().isoformat(),
            "encoder": "hybrid_agentic",
            "retrieval_mode": "hybrid_agentic",
            "total_configs_tested": 1,
            "total_time_seconds": total_time,
            "video_dir": run_config.video_dir,
            "segments_file": run_config.segments_file,
            "ground_truth_file": run_config.ground_truth_file,
            "evaluation_protocol": {
                "ground_truth_use": "post_hoc_scoring_only",
                "predictions_fixed_before_evaluation": True,
                "oracle_output_selection": False,
            },
            "results": [asdict(result)],
        }
        _write_json(output_root / f"hybrid_agentic_result_{run_id}.json", payload)
        _write_json(output_root / "hybrid_agentic_result.json", payload)
    return result, log


def generate_grid_configs(args: argparse.Namespace) -> List[HybridRunConfig]:
    base = HybridRunConfig(
        benchmark=args.benchmark,
        video_dir=args.video_dir,
        segments_file=args.segments,
        ground_truth_file=args.ground_truth,
        output_dir=args.output,
        cache_dir=args.cache_dir,
        device=args.device,
    )
    models = args.models or [args.model]
    frames = args.frames or [args.num_frames]
    resolutions = args.resolutions or [args.resolution]
    dual_softmax = args.dual_softmax if args.dual_softmax is not None else [args.use_dual_softmax]
    prompt_modes = args.prompt_modes or [args.prompt_mode]
    shortlist_sizes = args.shortlist_sizes or [args.shortlist_size]
    margins = args.ambiguity_margins or [args.ambiguity_margin_threshold]
    max_segments = args.max_agent_segments_list or [args.max_agent_segments]
    budgets = args.agent_search_budgets or [args.agent_search_budget_per_segment]
    decision_modes = args.agent_decision_modes or [args.agent_decision_mode]
    review_scopes = args.agent_review_scopes or [args.agent_review_scope]
    score_normalization = "none"
    csls_k = 5

    if args.fast_grid:
        best_backbone = load_best_videoprism_backbone(args)
        if best_backbone:
            logger.info(
                "Fast grid is using the best saved VideoPrism backbone (%s%%) from %s",
                best_backbone.get("exact_match_accuracy"),
                best_backbone.get("source_file"),
            )
        else:
            logger.warning("No saved VideoPrism grid result was found; fast grid is using the explicit CLI defaults.")
        models = args.models or [str((best_backbone or {}).get("model_name", args.model))]
        frames = args.frames or [int((best_backbone or {}).get("num_frames", args.num_frames))]
        resolutions = args.resolutions or [int((best_backbone or {}).get("resolution", args.resolution))]
        dual_softmax = args.dual_softmax if args.dual_softmax is not None else [bool((best_backbone or {}).get("use_dual_softmax", args.use_dual_softmax))]
        prompt_modes = args.prompt_modes or [str((best_backbone or {}).get("prompt_mode", args.prompt_mode))]
        score_normalization = str((best_backbone or {}).get("score_normalization", "none"))
        csls_k = int((best_backbone or {}).get("csls_k", 5))
        shortlist_sizes = args.shortlist_sizes or [3, 5]
        margins = args.ambiguity_margins or [0.02, 0.05]
        max_segments = args.max_agent_segments_list or [5, 10]
        budgets = args.agent_search_budgets or [2]
        decision_modes = args.agent_decision_modes or ["advisory"]
        review_scopes = args.agent_review_scopes or ["all"]

    configs: List[HybridRunConfig] = []
    for model, frame, resolution, dual, prompt, shortlist, margin, max_seg, budget, decision_mode, review_scope in itertools.product(
        models, frames, resolutions, dual_softmax, prompt_modes, shortlist_sizes, margins, max_segments, budgets, decision_modes, review_scopes
    ):
        cfg = HybridRunConfig(**{k: getattr(base, k) for k in ["benchmark", "video_dir", "segments_file", "ground_truth_file", "output_dir", "cache_dir", "device"]})
        cfg.video = HybridVideoPrismConfig(
            model_name=model,
            num_frames=int(frame),
            resolution=int(resolution),
            use_dual_softmax=bool(dual),
            prompt_mode=prompt,
            query_mode="original",
            assignment_method="hungarian",
            score_normalization=score_normalization,
            csls_k=csls_k,
            no_windowing=True,
        )
        cfg.agent = HybridAgentConfig(
            enabled=not args.disable_agent,
            use_codex=args.use_codex,
            review_scope=review_scope,
            decision_mode=decision_mode,
            shortlist_size=int(shortlist),
            ambiguity_margin_threshold=float(margin),
            max_agent_segments=int(max_seg),
            search_budget_per_segment=int(budget),
            allow_exclusions=not args.no_agent_exclusions,
            allow_candidate_expansion=not args.no_agent_candidate_expansion,
            allow_contact_sheet=not args.no_contact_sheets,
            advisory_boost=float(args.advisory_boost),
            codex_model=args.codex_model,
            codex_reasoning_effort=args.codex_reasoning_effort,
            codex_timeout_seconds=int(args.codex_timeout_seconds),
            hard_lock_min_confidence=float(args.hard_lock_min_confidence),
            require_direct_inspection_for_hard_constraints=not args.allow_uninspected_hard_constraints,
            verify_assignment_cycles=not args.no_cycle_critic,
            cycle_critic_min_confidence=float(args.cycle_critic_min_confidence),
        )
        configs.append(cfg)
    return configs


def run_grid(args: argparse.Namespace) -> List[HybridResult]:
    configs = generate_grid_configs(args)
    grid_run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    print(f"STARTING HYBRID AGENTIC GRID SEARCH: {len(configs)} configurations", flush=True)
    if args.estimate_only:
        print(json.dumps({"estimated_configurations": len(configs), "fast_grid": bool(args.fast_grid)}, indent=2))
        return []
    output_root = Path(_resolve_project_path(args.output) or args.output)
    results: List[HybridResult] = []
    failures: List[Dict[str, Any]] = []
    start = time.time()
    for idx, cfg in enumerate(configs, 1):
        print(f"[{idx}/{len(configs)}] Running configuration", flush=True)
        cfg.output_dir = str(output_root / f"config_{idx:04d}")
        try:
            result, _ = run_hybrid_benchmark(cfg, write_result=False)
            results.append(result)
            print(f"  Result: Exact={result.exact_match_accuracy} Top5={result.top_5_accuracy} reviewed={result.reviewed_segments}", flush=True)
        except Exception as exc:
            logger.exception("Hybrid config failed: %s", exc)
            failures.append({"config_index": idx, "config": _json_safe(asdict(cfg)), "error": str(exc)})
    ranked_results = sorted(
        enumerate(results),
        key=lambda pair: (
            float(pair[1].exact_match_accuracy) if pair[1].exact_match_accuracy is not None else -1.0,
            float(pair[1].mrr) if pair[1].mrr is not None else -1.0,
        ),
        reverse=True,
    )
    payload = {
        "timestamp": datetime.now().isoformat(),
        "encoder": "hybrid_agentic",
        "retrieval_mode": "hybrid_agentic",
        "total_configs_tested": len(results),
        "total_configurations_requested": len(configs),
        "total_time_seconds": round(time.time() - start, 2),
        "video_dir": configs[0].video_dir if configs else args.video_dir,
        "segments_file": configs[0].segments_file if configs else args.segments,
        "ground_truth_file": configs[0].ground_truth_file if configs else args.ground_truth,
        "evaluation_protocol": {
            "scientific_role": "validation_grid_search",
            "ground_truth_use": "post_hoc_scoring_and_configuration_ranking_only",
            "predictions_fixed_before_evaluation": True,
            "oracle_output_selection": False,
        },
        "results": [asdict(result) for result in results],
        "posthoc_tuning_ranking": [
            {
                "rank": rank,
                "result_index": result_index,
                "exact_match_accuracy": result.exact_match_accuracy,
                "mrr": result.mrr,
                "config": result.config,
            }
            for rank, (result_index, result) in enumerate(ranked_results, 1)
        ],
        "failures": failures,
    }
    versioned_path = output_root / f"hybrid_agentic_grid_search_results_{grid_run_id}.json"
    canonical_path = output_root / "hybrid_agentic_grid_search_results.json"
    canonical_source = "current_run"
    canonical_payload = payload
    canonical_score = _payload_validation_score(payload)
    saved_candidates = [canonical_path]
    if output_root.exists():
        saved_candidates.extend(sorted(output_root.glob("hybrid_agentic_grid_search_results_*.json")))
    for saved_path in saved_candidates:
        saved_payload = _read_json(saved_path, {})
        saved_score = _payload_validation_score(saved_payload)
        if saved_score > canonical_score:
            canonical_source = str(saved_path)
            canonical_payload = saved_payload
            canonical_score = saved_score
    payload["canonical_selection"] = {
        "source": canonical_source,
        "validation_score": list(canonical_score),
        "current_run_promoted": canonical_source == "current_run",
        "selection_role": "post_hoc_validation_tuning_only",
    }
    _write_json(versioned_path, payload)
    if results:
        _write_json(canonical_path, canonical_payload)
        validation_leader = ranked_results[0][1]
        print("Post-hoc validation leader (ground truth did not generate or replace predictions):")
        print(json.dumps(_json_safe(asdict(validation_leader)), indent=2))
        if canonical_source != "current_run":
            print(
                "Best-known canonical validation result retained from "
                f"{canonical_source} (Exact={canonical_score[0]}, MRR={canonical_score[1]})."
            )
    return results


def _payload_validation_score(payload: Any) -> Tuple[float, float]:
    if not isinstance(payload, dict):
        return (-1.0, -1.0)
    scores = []
    for result in payload.get("results", []):
        if not isinstance(result, dict):
            continue
        exact = result.get("exact_match_accuracy")
        mrr = result.get("mrr")
        scores.append((
            float(exact) if exact is not None else -1.0,
            float(mrr) if mrr is not None else -1.0,
        ))
    return max(scores, default=(-1.0, -1.0))


def build_run_config_from_args(args: argparse.Namespace) -> HybridRunConfig:
    return HybridRunConfig(
        benchmark=args.benchmark,
        video_dir=args.video_dir,
        segments_file=args.segments,
        ground_truth_file=args.ground_truth,
        output_dir=args.output,
        cache_dir=args.cache_dir,
        device=args.device,
        video=HybridVideoPrismConfig(
            model_name=args.model,
            num_frames=args.num_frames,
            resolution=args.resolution,
            use_dual_softmax=args.use_dual_softmax,
            prompt_mode=args.prompt_mode,
            query_mode="original",
            assignment_method="hungarian",
            score_normalization="none",
            csls_k=5,
            no_windowing=True,
        ),
        agent=HybridAgentConfig(
            enabled=not args.disable_agent,
            use_codex=args.use_codex,
            review_scope=args.agent_review_scope,
            decision_mode=args.agent_decision_mode,
            shortlist_size=args.shortlist_size,
            ambiguity_margin_threshold=args.ambiguity_margin_threshold,
            max_agent_segments=args.max_agent_segments,
            search_budget_per_segment=args.agent_search_budget_per_segment,
            allow_exclusions=not args.no_agent_exclusions,
            allow_candidate_expansion=not args.no_agent_candidate_expansion,
            allow_contact_sheet=not args.no_contact_sheets,
            advisory_boost=args.advisory_boost,
            codex_model=args.codex_model,
            codex_reasoning_effort=args.codex_reasoning_effort,
            codex_timeout_seconds=args.codex_timeout_seconds,
            hard_lock_min_confidence=args.hard_lock_min_confidence,
            require_direct_inspection_for_hard_constraints=not args.allow_uninspected_hard_constraints,
            verify_assignment_cycles=not args.no_cycle_critic,
            cycle_critic_min_confidence=args.cycle_critic_min_confidence,
        ),
    )


def load_best_videoprism_backbone(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    benchmark_id = _normalize_benchmark_id(args.benchmark)
    if not benchmark_id:
        return None
    output_root = Path(_resolve_project_path(args.output) or args.output)
    candidates = [
        output_root.parent / "videoprism" / "videoprism_grid_search_results.json",
        PROJECT_ROOT / "output" / f"benchmark_{benchmark_id}" / "videoprism" / "videoprism_grid_search_results.json",
    ]
    saved_output = PROJECT_ROOT / "output"
    if saved_output.exists():
        for path in saved_output.rglob("videoprism_grid_search_results.json"):
            lowered = str(path).lower()
            if f"benchmark_{benchmark_id}" in lowered or f"video_{benchmark_id}" in lowered:
                candidates.append(path)
    candidates = list(dict.fromkeys(candidates))
    best: Optional[Dict[str, Any]] = None
    best_score = -1.0
    for path in candidates:
        payload = _read_json(path, {})
        if not isinstance(payload, dict):
            continue
        for result in payload.get("results", []):
            if not isinstance(result, dict):
                continue
            try:
                score = float(result.get("exact_match_accuracy", -1.0))
            except (TypeError, ValueError):
                continue
            if score <= best_score:
                continue
            config = result.get("config") if isinstance(result.get("config"), dict) else {}
            video = config.get("video") if isinstance(config.get("video"), dict) else {}
            merged = dict(video)
            merged.update(config)
            merged["source_file"] = str(path)
            merged["exact_match_accuracy"] = score
            best = merged
            best_score = score
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid agentic VideoPrism pipeline")
    parser.add_argument("--mode", choices=["benchmark", "grid", "estimate", "check-codex"], default="benchmark")
    parser.add_argument("--benchmark", "-b", default=None)
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--segments", default=None)
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--output", default="./output/hybrid_agentic")
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--model", default="videoprism_lvt_public_v1_large", choices=["videoprism_lvt_public_v1_base", "videoprism_lvt_public_v1_large"])
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=288)
    parser.add_argument("--use-dual-softmax", action="store_true")
    parser.add_argument(
        "--prompt-mode",
        default="none",
        choices=HYBRID_PROMPT_MODES,
        help="Prompt shaping for the initial VideoPrism baseline only; agent searches remain raw.",
    )
    parser.add_argument("--disable-agent", action="store_true")
    parser.add_argument("--use-codex", action="store_true")
    parser.add_argument("--agent-review-scope", default="ambiguous", choices=["ambiguous", "all"])
    parser.add_argument("--agent-decision-mode", default="advisory", choices=["advisory", "hard_lock"])
    parser.add_argument("--shortlist-size", type=int, default=5)
    parser.add_argument("--ambiguity-margin-threshold", type=float, default=0.05)
    parser.add_argument("--max-agent-segments", type=int, default=10)
    parser.add_argument("--agent-search-budget-per-segment", type=int, default=2)
    parser.add_argument("--no-agent-exclusions", action="store_true")
    parser.add_argument("--no-agent-candidate-expansion", action="store_true")
    parser.add_argument("--no-contact-sheets", action="store_true")
    parser.add_argument("--advisory-boost", type=float, default=0.03)
    parser.add_argument("--codex-model", default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--codex-reasoning-effort", default=DEFAULT_CODEX_REASONING_EFFORT)
    parser.add_argument("--codex-timeout-seconds", type=int, default=600)
    parser.add_argument("--hard-lock-min-confidence", type=float, default=0.9)
    parser.add_argument(
        "--no-cycle-critic",
        action="store_true",
        help="Disable the independent Codex verification pass for full-audit assignment components.",
    )
    parser.add_argument("--cycle-critic-min-confidence", type=float, default=0.75)
    parser.add_argument(
        "--allow-uninspected-hard-constraints",
        action="store_true",
        help="Unsafe research override: permit hard constraints without direct contact-sheet inspection.",
    )

    parser.add_argument("--fast-grid", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--frames", nargs="+", type=int, default=None)
    parser.add_argument("--resolutions", nargs="+", type=int, default=None)
    parser.add_argument("--dual-softmax", nargs="+", type=lambda x: str(x).lower() in {"true", "1", "yes", "on"}, default=None)
    parser.add_argument("--prompt-modes", nargs="+", choices=HYBRID_PROMPT_MODES, default=None)
    parser.add_argument("--shortlist-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--ambiguity-margins", nargs="+", type=float, default=None)
    parser.add_argument("--max-agent-segments-list", nargs="+", type=int, default=None)
    parser.add_argument("--agent-search-budgets", nargs="+", type=int, default=None)
    parser.add_argument("--agent-decision-modes", nargs="+", choices=["advisory", "hard_lock"], default=None)
    parser.add_argument("--agent-review-scopes", nargs="+", choices=["ambiguous", "all"], default=None)

    args = parser.parse_args()
    if args.mode == "check-codex":
        print(json.dumps(codex_status(), indent=2))
        return
    if args.mode in {"grid", "estimate"}:
        if args.mode == "estimate":
            args.estimate_only = True
        run_grid(args)
        return
    result, _ = run_hybrid_benchmark(build_run_config_from_args(args), write_result=True)
    print(json.dumps(_json_safe(asdict(result)), indent=2))


if __name__ == "__main__":
    main()
