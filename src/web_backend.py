"""
Backend services for the Video Sequencer web application.

This module keeps the existing CLI pipeline intact and adds a reusable
service layer that the web app can call for:
1. Benchmark discovery/import/export/delete
2. Cache inspection and clearing
3. Persistent settings
4. Background job execution for existing CLI commands
5. An interactive Write-A-Video-inspired editing session
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import cv2
import numpy as np

from assembly import VideoAssembler, VideoSequenceBuilder
from main import load_manual_segments
from matching import ClipSelection, VideoTextMatcher
from menu import discover_benchmark_numbers
from models import VideoMetadata
from openclip_indexing import OpenCLIPTextMatcher, OpenCLIPVideoIndexer
from segmentation import ScriptSegment, ScriptSegmenter
from transcription import TranscriptionResult, VoiceTranscriber, WordSegment
from wav_indexing import MultiModalKeywordIndexer, WriteAVideoMatcher

try:
    from indexing import VideoIndexer
    VIDEOPRISM_AVAILABLE = True
except ImportError:
    VideoIndexer = None
    VIDEOPRISM_AVAILABLE = False


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBAPP_ROOT = PROJECT_ROOT / "webapp"
STATE_ROOT = WEBAPP_ROOT / "state"
SESSIONS_ROOT = STATE_ROOT / "sessions"
SETTINGS_FILE = STATE_ROOT / "settings.json"

STATE_ROOT.mkdir(parents=True, exist_ok=True)
SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
SAFE_MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS | {".png", ".jpg", ".jpeg", ".json", ".txt", ".log", ".zip"}

IDIOM_OPTIONS = [
    "Extend Shot Duration",
    "Reduce Shot Duration",
    "More Movement",
    "Less Movement",
]

DEFAULT_SETTINGS = {
    "gpu_device": "cuda:0",
    "output": "./output",
    "cache_dir": "./cache",
    "server_hostname": "neghvar.ced.tuc.gr",
    "benchmarks_dir": "./data/benchmarks",
    "editor_mode": "writeavideo",
    "openclip_model": "ViT-B-32",
    "whisper_model": "base",
    "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _bytes_to_mb(size: int) -> float:
    return round(size / 1024 / 1024, 2)


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return copy.deepcopy(default)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: Any) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _extract_text_keywords(text: str) -> List[str]:
    stop_words = {
        "the", "and", "this", "that", "with", "from", "they", "have", "been",
        "were", "will", "would", "could", "should", "your", "you", "for",
        "are", "was", "not", "but", "what", "all", "can", "had", "her",
        "his", "him", "has", "its", "just", "into", "over", "such", "than",
        "then", "them", "these", "some", "very", "when", "where", "which",
        "while", "who", "why", "how", "each", "she", "does", "doing", "being",
        "now", "going", "want", "make", "take", "get", "let", "put", "use",
        "way", "look", "like", "come", "see", "our", "their", "there", "here",
    }
    words = re.findall(r"\b[a-z][a-z0-9]{2,}\b", text.lower())
    seen = []
    for word in words:
        if word not in stop_words and word not in seen:
            seen.append(word)
    return seen[:12]


def _safe_relpath(path: Path, start: Path) -> str:
    try:
        return str(path.relative_to(start))
    except ValueError:
        return str(path)


@dataclass
class WebJob:
    job_id: str
    name: str
    command: List[str]
    cwd: str
    status: str = "queued"
    created_at: str = field(default_factory=_now_iso)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    returncode: Optional[int] = None
    log_lines: List[str] = field(default_factory=list)

    def snapshot(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["log"] = "".join(self.log_lines[-600:])
        return payload


@dataclass
class EditorCandidate:
    candidate_id: str
    video_id: str
    file_path: str
    file_name: str
    duration: float
    similarity_score: float
    combined_score: float
    motion_score: float
    context_score: float
    keyword_score: float
    trim_start: float
    trim_end: float
    trim_duration: float
    matched_keywords: List[str] = field(default_factory=list)
    thumbnail_path: Optional[str] = None
    rank: int = 0


@dataclass
class EditorSegment:
    segment_id: int
    text: str
    start_time: float
    end_time: float
    duration: float
    description: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    extra_keywords: List[str] = field(default_factory=list)
    movement_preference: str = "auto"
    duration_multiplier: float = 1.0
    timing_bias: float = 0.0
    selected_candidate_id: Optional[str] = None
    candidates: List[EditorCandidate] = field(default_factory=list)


@dataclass
class EditorSession:
    session_id: str
    name: str
    created_at: str
    updated_at: str
    benchmark: Optional[str]
    video_dir: str
    audio_file: Optional[str]
    segments_file: Optional[str]
    output_dir: str
    cache_dir: str
    retrieval_mode: str
    config: Dict[str, Any]
    global_keywords: List[str] = field(default_factory=list)
    assembled_video_path: Optional[str] = None
    segments: List[EditorSegment] = field(default_factory=list)


@dataclass
class EditorRuntime:
    session: EditorSession
    session_dir: Path
    matcher: Any = None
    indexer: Any = None
    keyword_indexer: Optional[MultiModalKeywordIndexer] = None
    motion_cache: Dict[str, float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


class SettingsStore:
    def __init__(self, settings_file: Path = SETTINGS_FILE):
        self.settings_file = settings_file
        self._lock = threading.Lock()

    def load(self) -> Dict[str, Any]:
        with self._lock:
            settings = copy.deepcopy(DEFAULT_SETTINGS)
            settings.update(_read_json(self.settings_file, {}))
            return settings

    def save(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            current = copy.deepcopy(DEFAULT_SETTINGS)
            current.update(_read_json(self.settings_file, {}))
            current.update({k: v for k, v in updates.items() if v is not None})
            _write_json(self.settings_file, current)
            return current


class CacheManager:
    def inspect(self, cache_dir: str) -> Dict[str, Any]:
        cache_path = (PROJECT_ROOT / cache_dir).resolve() if not Path(cache_dir).is_absolute() else Path(cache_dir)
        entries = []
        total_size = 0
        if cache_path.exists():
            for item in sorted(cache_path.iterdir()):
                size = _dir_size(item)
                total_size += size
                entries.append(
                    {
                        "name": item.name,
                        "path": str(item),
                        "type": "dir" if item.is_dir() else "file",
                        "size_bytes": size,
                        "size_mb": _bytes_to_mb(size),
                    }
                )
        return {
            "cache_dir": str(cache_path),
            "exists": cache_path.exists(),
            "total_size_bytes": total_size,
            "total_size_mb": _bytes_to_mb(total_size),
            "entries": entries,
        }

    def clear(self, cache_dir: str, action: str) -> Dict[str, Any]:
        cache_path = (PROJECT_ROOT / cache_dir).resolve() if not Path(cache_dir).is_absolute() else Path(cache_dir)
        cleared: List[str] = []
        prefix_map = {
            "openclip-grid": "gs_",
            "videoprism-grid": "vp_",
            "write-a-video-grid": "wav_kw_",
        }

        if action == "all":
            if cache_path.exists():
                shutil.rmtree(cache_path)
                cleared.append(str(cache_path))
        elif action == "videoprism-index":
            target = cache_path / "video_index"
            if target.exists():
                shutil.rmtree(target)
                cleared.append(str(target))
        elif action == "openclip-index":
            target = cache_path / "video_index_openclip"
            if target.exists():
                shutil.rmtree(target)
                cleared.append(str(target))
        elif action in prefix_map:
            prefix = prefix_map[action]
            if cache_path.exists():
                for item in cache_path.iterdir():
                    if item.is_dir() and item.name.startswith(prefix):
                        shutil.rmtree(item)
                        cleared.append(str(item))
        return {"cleared": cleared, "cache": self.inspect(cache_dir)}


class BenchmarkManager:
    def list(self, base_dir: str) -> List[Dict[str, Any]]:
        base_path = (PROJECT_ROOT / base_dir).resolve() if not Path(base_dir).is_absolute() else Path(base_dir)
        benchmarks = discover_benchmark_numbers(str(base_path))
        metadata_dir = base_path / "metadata"
        enriched = []
        for benchmark in benchmarks:
            meta_file = metadata_dir / f"benchmark_{benchmark['number']}_meta.json"
            meta = _read_json(meta_file, {})
            benchmark["title"] = meta.get("title", f"Benchmark {benchmark['number']}")
            enriched.append(benchmark)
        return enriched

    def delete(self, benchmark_number: str, base_dir: str) -> Dict[str, Any]:
        base_path = (PROJECT_ROOT / base_dir).resolve() if not Path(base_dir).is_absolute() else Path(base_dir)
        removed = []
        targets = [
            base_path / "videos" / f"video_{benchmark_number}",
            base_path / "segments" / f"benchmark_{benchmark_number}_segments.json",
            base_path / "gdtruth" / f"benchmark_{benchmark_number}_ground_truth.json",
            base_path / "metadata" / f"benchmark_{benchmark_number}_meta.json",
        ]
        for target in targets:
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
                removed.append(str(target))
        return {"removed": removed}

    def export_zip(self, benchmark_number: str, base_dir: str) -> Path:
        base_path = (PROJECT_ROOT / base_dir).resolve() if not Path(base_dir).is_absolute() else Path(base_dir)
        export_dir = STATE_ROOT / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        zip_path = export_dir / f"benchmark_{benchmark_number}.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            video_dir = base_path / "videos" / f"video_{benchmark_number}"
            if video_dir.exists():
                for video in sorted(video_dir.iterdir()):
                    if video.suffix.lower() in VIDEO_EXTENSIONS:
                        archive.write(video, arcname=video.name)

            segments = base_path / "segments" / f"benchmark_{benchmark_number}_segments.json"
            if segments.exists():
                archive.write(segments, arcname="segments.json")

            ground_truth = base_path / "gdtruth" / f"benchmark_{benchmark_number}_ground_truth.json"
            if ground_truth.exists():
                archive.write(ground_truth, arcname="ground_truth.json")

            meta = base_path / "metadata" / f"benchmark_{benchmark_number}_meta.json"
            if meta.exists():
                archive.write(meta, arcname="metadata.json")

        return zip_path

    def import_uploaded_files(
        self,
        benchmark_number: str,
        benchmark_title: str,
        base_dir: str,
        uploaded_paths: List[Tuple[str, bytes]],
    ) -> Dict[str, Any]:
        base_path = (PROJECT_ROOT / base_dir).resolve() if not Path(base_dir).is_absolute() else Path(base_dir)
        staging_dir = STATE_ROOT / "uploads" / f"benchmark_{benchmark_number}_{uuid.uuid4().hex[:8]}"
        staging_dir.mkdir(parents=True, exist_ok=True)

        try:
            for relative_path, payload in uploaded_paths:
                clean_name = Path(relative_path).name
                target = staging_dir / clean_name
                with open(target, "wb") as handle:
                    handle.write(payload)

            files = list(staging_dir.iterdir())
            videos = [path for path in files if path.suffix.lower() in VIDEO_EXTENSIONS]
            json_files = [path for path in files if path.suffix.lower() == ".json"]

            if not videos:
                raise ValueError("No video files were uploaded.")

            segments_file = None
            ground_truth_file = None
            for item in json_files:
                lower = item.name.lower()
                if "mapping" in lower or "ground" in lower or "truth" in lower or lower == "gt.json":
                    ground_truth_file = item
                elif "segment" in lower:
                    segments_file = item

            video_dir = base_path / "videos" / f"video_{benchmark_number}"
            segments_dir = base_path / "segments"
            ground_truth_dir = base_path / "gdtruth"
            meta_dir = base_path / "metadata"

            video_dir.mkdir(parents=True, exist_ok=True)
            segments_dir.mkdir(parents=True, exist_ok=True)
            ground_truth_dir.mkdir(parents=True, exist_ok=True)
            meta_dir.mkdir(parents=True, exist_ok=True)

            for video in videos:
                shutil.move(str(video), str(video_dir / video.name))
            if segments_file:
                shutil.move(str(segments_file), str(segments_dir / f"benchmark_{benchmark_number}_segments.json"))
            if ground_truth_file:
                shutil.move(str(ground_truth_file), str(ground_truth_dir / f"benchmark_{benchmark_number}_ground_truth.json"))

            meta = {
                "number": benchmark_number,
                "title": benchmark_title or f"Benchmark {benchmark_number}",
                "video_count": len(videos),
                "has_segments": bool(segments_file),
                "has_ground_truth": bool(ground_truth_file),
                "created_at": _now_iso(),
            }
            _write_json(meta_dir / f"benchmark_{benchmark_number}_meta.json", meta)

            return {
                "benchmark_number": benchmark_number,
                "title": meta["title"],
                "video_count": len(videos),
                "segments": bool(segments_file),
                "ground_truth": bool(ground_truth_file),
            }
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, WebJob] = {}
        self._lock = threading.Lock()

    def submit(self, name: str, command: List[str], cwd: Path = PROJECT_ROOT) -> Dict[str, Any]:
        job = WebJob(
            job_id=uuid.uuid4().hex,
            name=name,
            command=[str(item) for item in command],
            cwd=str(cwd),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        thread = threading.Thread(target=self._run_job, args=(job.job_id,), daemon=True)
        thread.start()
        return job.snapshot()

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = [job.snapshot() for job in self._jobs.values()]
        return sorted(jobs, key=lambda item: item["created_at"], reverse=True)

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs[job_id]
            return job.snapshot()

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = _now_iso()

        process = subprocess.Popen(
            job.command,
            cwd=job.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        try:
            assert process.stdout is not None
            for line in process.stdout:
                with self._lock:
                    self._jobs[job_id].log_lines.append(line)
                    if len(self._jobs[job_id].log_lines) > 4000:
                        self._jobs[job_id].log_lines = self._jobs[job_id].log_lines[-4000:]
            process.wait()
            with self._lock:
                self._jobs[job_id].returncode = process.returncode
                self._jobs[job_id].status = "succeeded" if process.returncode == 0 else "failed"
                self._jobs[job_id].finished_at = _now_iso()
        except Exception as exc:
            with self._lock:
                self._jobs[job_id].status = "failed"
                self._jobs[job_id].finished_at = _now_iso()
                self._jobs[job_id].log_lines.append(f"\n[webapp] Job runner failed: {exc}\n")


def _resolve_path(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return str(path)


def resolve_benchmark_paths(benchmark_number: str, benchmarks_dir: str) -> Dict[str, Optional[str]]:
    base = Path(_resolve_path(benchmarks_dir) or benchmarks_dir)
    video_dir = base / "videos" / f"video_{benchmark_number}"
    segments = base / "segments" / f"benchmark_{benchmark_number}_segments.json"
    ground_truth = base / "gdtruth" / f"benchmark_{benchmark_number}_ground_truth.json"
    audio = None
    audio_dir = base / "audio"
    if audio_dir.exists():
        for pattern in (f"voiceover_{benchmark_number}.mp3", f"benchmark_{benchmark_number}.mp3"):
            candidate = audio_dir / pattern
            if candidate.exists():
                audio = str(candidate)
                break
    return {
        "video_dir": str(video_dir) if video_dir.exists() else None,
        "segments": str(segments) if segments.exists() else None,
        "ground_truth": str(ground_truth) if ground_truth.exists() else None,
        "audio": audio,
    }


def build_job_command(action: str, payload: Dict[str, Any], settings: Dict[str, Any]) -> Tuple[str, List[str]]:
    payload = copy.deepcopy(payload)
    benchmarks_dir = payload.get("benchmarks_dir") or settings.get("benchmarks_dir", "./data/benchmarks")
    output_dir = payload.get("output") or settings.get("output", "./output")
    gpu_device = payload.get("gpu_device") or settings.get("gpu_device", "cuda:0")

    if action == "quick-benchmark":
        resolved = resolve_benchmark_paths(str(payload["benchmark"]), benchmarks_dir)
        command = [
            "python", "src/main.py",
            "--video-dir", resolved["video_dir"],
            "--segments", resolved["segments"],
            "--output", output_dir,
            "--match-only",
            "--no-reuse",
            "--no-windowing",
            "--gpu-device", gpu_device,
        ]
        if resolved["ground_truth"]:
            command.extend(["--ground-truth", resolved["ground_truth"]])
        encoder = payload.get("encoder", "videoprism")
        command.extend(["--encoder", encoder])
        if encoder == "openclip":
            command.extend(["--openclip-model", payload.get("openclip_model", settings.get("openclip_model", "ViT-B-32"))])
        if resolved["audio"] and payload.get("include_audio"):
            command.extend(["--audio", resolved["audio"]])
        if payload.get("verbose", True):
            command.append("--verbose")
        return ("Quick Benchmark", command)

    if action == "full-pipeline":
        benchmark = payload.get("benchmark")
        resolved = resolve_benchmark_paths(str(benchmark), benchmarks_dir) if benchmark else {}
        video_dir = payload.get("video_dir") or resolved.get("video_dir")
        audio = payload.get("audio") or resolved.get("audio")
        command = [
            "python", "src/main.py",
            "--video-dir", video_dir,
            "--output", output_dir,
            "--gpu-device", gpu_device,
        ]
        if audio:
            command.extend(["--audio", audio])
        if payload.get("segments") or resolved.get("segments"):
            command.extend(["--segments", payload.get("segments") or resolved.get("segments")])
        if payload.get("ground_truth") or resolved.get("ground_truth"):
            command.extend(["--ground-truth", payload.get("ground_truth") or resolved.get("ground_truth")])
        if payload.get("no_reuse"):
            command.append("--no-reuse")
        if payload.get("greedy"):
            command.append("--greedy")
        encoder = payload.get("encoder", "videoprism")
        command.extend(["--encoder", encoder])
        if encoder == "openclip":
            command.extend(["--openclip-model", payload.get("openclip_model", settings.get("openclip_model", "ViT-B-32"))])
        if payload.get("no_windowing", False):
            command.append("--no-windowing")
        else:
            command.extend(["--window-size", str(payload.get("window_size", 5.0))])
            command.extend(["--window-overlap", str(payload.get("window_overlap", 0.5))])
        if payload.get("verbose"):
            command.append("--verbose")
        return ("Full Pipeline", command)

    if action == "openclip-grid-search":
        resolved = resolve_benchmark_paths(str(payload["benchmark"]), benchmarks_dir)
        command = [
            "python", "src/grid_search.py",
            "--video-dir", resolved["video_dir"],
            "--segments", resolved["segments"],
            "--ground-truth", resolved["ground_truth"],
            "--output", output_dir,
            "--device", gpu_device,
        ]
        command.extend(["--models"] + payload.get("models", ["ViT-B-32", "ViT-B-16", "ViT-L-14"]))
        command.extend(["--frames"] + [str(item) for item in payload.get("frames", [4, 8, 16, 32])])
        command.extend(["--aggregations"] + payload.get("aggregations", ["mean", "max", "best_frame"]))
        prompt_modes = payload.get("prompt_modes", ["none", "template:video", "template:photo", "template:cooking", "template:scene"])
        command.extend(["--prompt-modes"] + prompt_modes)
        if payload.get("llm_model"):
            command.extend(["--llm-model", payload["llm_model"]])
        if payload.get("reset_llm_cache"):
            command.append("--reset-llm-cache")
        if payload.get("no_windowing", True):
            command.append("--no-windowing")
        return ("OpenCLIP Grid Search", command)

    if action == "videoprism-grid-search":
        resolved = resolve_benchmark_paths(str(payload["benchmark"]), benchmarks_dir)
        command = [
            "python", "src/videoprism_grid_search.py",
            "--video-dir", resolved["video_dir"],
            "--segments", resolved["segments"],
            "--ground-truth", resolved["ground_truth"],
            "--output", output_dir,
            "--device", gpu_device,
        ]
        command.extend(["--models"] + payload.get("models", ["videoprism_lvt_public_v1_base", "videoprism_lvt_public_v1_large"]))
        command.extend(["--frames"] + [str(item) for item in payload.get("frames", [8, 16, 32])])
        command.extend(["--prompt-modes"] + payload.get("prompt_modes", ["none", "template:video", "template:photo", "template:scene", "template:cooking"]))
        if payload.get("llm_model"):
            command.extend(["--llm-model", payload["llm_model"]])
        if payload.get("reset_llm_cache"):
            command.append("--reset-llm-cache")
        if payload.get("no_windowing", True):
            command.append("--no-windowing")
        return ("VideoPrism Grid Search", command)

    if action == "write-a-video-grid-search":
        resolved = resolve_benchmark_paths(str(payload["benchmark"]), benchmarks_dir)
        command = [
            "python", "src/wav_grid_search.py",
            "--video-dir", resolved["video_dir"],
            "--segments", resolved["segments"],
            "--ground-truth", resolved["ground_truth"],
            "--output", output_dir,
            "--device", gpu_device,
            "--yolo-model", payload.get("yolo_model", "yolov8n"),
            "--fps", str(payload.get("fps", 1.0)),
        ]
        command.extend(["--models"] + payload.get("models", ["ViT-B-32", "ViT-B-16", "ViT-L-14"]))
        command.extend(["--frames"] + [str(item) for item in payload.get("frames", [4, 8, 16, 32])])
        command.extend(["--aggregations"] + payload.get("aggregations", ["mean", "max", "best_frame"]))
        command.extend(["--prompt-modes"] + payload.get("prompt_modes", ["none", "template:video", "template:photo", "template:cooking", "template:scene", "ensemble:template"]))
        if payload.get("pool_sizes"):
            command.extend(["--pool-sizes"] + [str(item) for item in payload["pool_sizes"]])
        if payload.get("keyword_weights"):
            command.extend(["--keyword-weights"] + [str(item) for item in payload["keyword_weights"]])
        if payload.get("llm_model"):
            command.extend(["--llm-model", payload["llm_model"]])
        if payload.get("reset_llm_cache"):
            command.append("--reset-llm-cache")
        if payload.get("no_windowing", True):
            command.append("--no-windowing")
        if payload.get("disable_object_detection"):
            command.append("--no-object-detection")
        if payload.get("disable_face_detection"):
            command.append("--no-face-detection")
        return ("Write-A-Video Grid Search", command)

    if action == "compare-all-models":
        command = [
            "python", "src/compare_all_models.py",
            "--benchmark", str(payload.get("benchmark", "all")),
            "--output", output_dir,
            "--llm-model", payload.get("llm_model", settings.get("llm_model", "llama3.2:3b")),
            "--device", gpu_device,
        ]
        if payload.get("no_windowing", True):
            command.append("--no-windowing")
        return ("Compare All Models", command)

    raise ValueError(f"Unsupported job action: {action}")


class EditorSessionManager:
    def __init__(self) -> None:
        self._runtimes: Dict[str, EditorRuntime] = {}
        self._lock = threading.Lock()

    def list_sessions(self) -> List[Dict[str, Any]]:
        sessions = []
        for path in sorted(SESSIONS_ROOT.glob("*/session.json")):
            data = _read_json(path, {})
            if data:
                sessions.append(self._serialize_session(self._session_from_dict(data)))
        return sorted(sessions, key=lambda item: item["updated_at"], reverse=True)

    def create_session(self, payload: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
        benchmark = payload.get("benchmark")
        benchmarks_dir = payload.get("benchmarks_dir") or settings.get("benchmarks_dir", "./data/benchmarks")
        resolved = resolve_benchmark_paths(str(benchmark), benchmarks_dir) if benchmark else {}

        session_id = uuid.uuid4().hex
        session_dir = SESSIONS_ROOT / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = Path(_resolve_path(payload.get("cache_dir") or str(session_dir / "cache")))
        output_dir = Path(_resolve_path(payload.get("output_dir") or str(session_dir / "output")))
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_dir = _resolve_path(payload.get("video_dir") or resolved.get("video_dir"))
        audio_file = _resolve_path(payload.get("audio") or resolved.get("audio"))
        segments_file = _resolve_path(payload.get("segments") or resolved.get("segments"))

        if not video_dir:
            raise ValueError("A video directory or benchmark is required to create an editor session.")

        segments = self._load_segments(
            audio_file=audio_file,
            segments_file=segments_file,
            cache_dir=cache_dir,
            whisper_model=payload.get("whisper_model") or settings.get("whisper_model", "base"),
            llm_model=payload.get("llm_model") or settings.get("llm_model", "meta-llama/Llama-3.2-3B-Instruct"),
            gpu_device=payload.get("gpu_device") or settings.get("gpu_device", "cuda:0"),
            use_simple_segmentation=payload.get("simple_segmentation", False),
        )

        session = EditorSession(
            session_id=session_id,
            name=payload.get("name") or f"edit-{session_id[:8]}",
            created_at=_now_iso(),
            updated_at=_now_iso(),
            benchmark=str(benchmark) if benchmark else None,
            video_dir=video_dir,
            audio_file=audio_file,
            segments_file=segments_file,
            output_dir=str(output_dir),
            cache_dir=str(cache_dir),
            retrieval_mode=payload.get("retrieval_mode") or settings.get("editor_mode", "writeavideo"),
            config={
                "gpu_device": payload.get("gpu_device") or settings.get("gpu_device", "cuda:0"),
                "llm_model": payload.get("llm_model") or settings.get("llm_model", "meta-llama/Llama-3.2-3B-Instruct"),
                "whisper_model": payload.get("whisper_model") or settings.get("whisper_model", "base"),
                "openclip_model": payload.get("openclip_model") or settings.get("openclip_model", "ViT-B-32"),
                "videoprism_model": payload.get("videoprism_model", "videoprism_lvt_public_v1_base"),
                "windowing": not payload.get("no_windowing", True),
                "window_size": payload.get("window_size", 5.0),
                "window_overlap": payload.get("window_overlap", 0.5),
                "candidate_pool_size": payload.get("candidate_pool_size", 10),
                "keyword_weight": payload.get("keyword_weight", 0.2),
                "enable_object_detection": payload.get("enable_object_detection", True),
                "enable_face_detection": payload.get("enable_face_detection", False),
                "yolo_model": payload.get("yolo_model", "yolov8n"),
                "frames_per_second": payload.get("frames_per_second", 1.0),
            },
            segments=segments,
        )

        runtime = EditorRuntime(session=session, session_dir=session_dir)
        self._initialize_runtime(runtime)
        self._regenerate_all_candidates(runtime)
        self._update_global_keywords(runtime)
        self._save_runtime(runtime)

        with self._lock:
            self._runtimes[session_id] = runtime
        return self._serialize_session(runtime.session)

    def get_session(self, session_id: str) -> Dict[str, Any]:
        runtime = self._get_runtime(session_id)
        return self._serialize_session(runtime.session)

    def update_segment(self, session_id: str, segment_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
        runtime = self._get_runtime(session_id)
        with runtime.lock:
            segment = runtime.session.segments[segment_id]
            if "text" in updates:
                segment.text = updates["text"].strip()
            if "description" in updates:
                segment.description = updates["description"]
            if "keywords" in updates:
                segment.extra_keywords = [item.strip() for item in updates["keywords"] if item and item.strip()]
            if "movement_preference" in updates:
                segment.movement_preference = updates["movement_preference"]
            if "duration_multiplier" in updates:
                segment.duration_multiplier = _clamp(float(updates["duration_multiplier"]), 0.4, 2.2)
            if "timing_bias" in updates:
                segment.timing_bias = _clamp(float(updates["timing_bias"]), -1.0, 1.0)
            segment.duration = max(0.2, segment.end_time - segment.start_time)
            self._regenerate_segment_candidates(runtime, segment.segment_id)
            self._update_global_keywords(runtime)
            self._touch(runtime.session)
            self._save_runtime(runtime)
        return self._serialize_session(runtime.session)

    def split_segment(self, session_id: str, segment_id: int) -> Dict[str, Any]:
        runtime = self._get_runtime(session_id)
        with runtime.lock:
            segment = runtime.session.segments[segment_id]
            words = segment.text.split()
            if len(words) < 2:
                raise ValueError("Need at least two words to split a segment.")
            split_index = max(1, len(words) // 2)
            midpoint = segment.start_time + (segment.duration / 2.0)

            first = EditorSegment(
                segment_id=segment.segment_id,
                text=" ".join(words[:split_index]),
                start_time=segment.start_time,
                end_time=midpoint,
                duration=max(0.2, midpoint - segment.start_time),
                description=segment.description,
                keywords=list(segment.keywords),
                extra_keywords=list(segment.extra_keywords),
                movement_preference=segment.movement_preference,
                duration_multiplier=segment.duration_multiplier,
                timing_bias=segment.timing_bias,
            )
            second = EditorSegment(
                segment_id=segment.segment_id + 1,
                text=" ".join(words[split_index:]),
                start_time=midpoint,
                end_time=segment.end_time,
                duration=max(0.2, segment.end_time - midpoint),
                description=segment.description,
                keywords=list(segment.keywords),
                extra_keywords=list(segment.extra_keywords),
                movement_preference=segment.movement_preference,
                duration_multiplier=segment.duration_multiplier,
                timing_bias=segment.timing_bias,
            )
            runtime.session.segments = runtime.session.segments[:segment_id] + [first, second] + runtime.session.segments[segment_id + 1:]
            self._reindex_segments(runtime.session)
            self._regenerate_all_candidates(runtime)
            self._update_global_keywords(runtime)
            self._touch(runtime.session)
            self._save_runtime(runtime)
        return self._serialize_session(runtime.session)

    def merge_with_next(self, session_id: str, segment_id: int) -> Dict[str, Any]:
        runtime = self._get_runtime(session_id)
        with runtime.lock:
            if segment_id >= len(runtime.session.segments) - 1:
                raise ValueError("There is no following segment to merge with.")
            current = runtime.session.segments[segment_id]
            following = runtime.session.segments[segment_id + 1]
            merged = EditorSegment(
                segment_id=current.segment_id,
                text=f"{current.text} {following.text}".strip(),
                start_time=current.start_time,
                end_time=following.end_time,
                duration=max(0.2, following.end_time - current.start_time),
                description=current.description or following.description,
                keywords=sorted(set(current.keywords + following.keywords)),
                extra_keywords=sorted(set(current.extra_keywords + following.extra_keywords)),
                movement_preference=current.movement_preference,
                duration_multiplier=(current.duration_multiplier + following.duration_multiplier) / 2.0,
                timing_bias=(current.timing_bias + following.timing_bias) / 2.0,
            )
            runtime.session.segments = runtime.session.segments[:segment_id] + [merged] + runtime.session.segments[segment_id + 2:]
            self._reindex_segments(runtime.session)
            self._regenerate_all_candidates(runtime)
            self._update_global_keywords(runtime)
            self._touch(runtime.session)
            self._save_runtime(runtime)
        return self._serialize_session(runtime.session)

    def move_segment(self, session_id: str, segment_id: int, direction: str) -> Dict[str, Any]:
        runtime = self._get_runtime(session_id)
        if runtime.session.audio_file:
            raise ValueError("Segment reordering is disabled when a voice-over track is attached.")
        with runtime.lock:
            delta = -1 if direction == "up" else 1
            target = segment_id + delta
            if target < 0 or target >= len(runtime.session.segments):
                return self._serialize_session(runtime.session)
            runtime.session.segments[segment_id], runtime.session.segments[target] = runtime.session.segments[target], runtime.session.segments[segment_id]
            self._reindex_segments(runtime.session)
            self._touch(runtime.session)
            self._save_runtime(runtime)
        return self._serialize_session(runtime.session)

    def select_candidate(self, session_id: str, segment_id: int, candidate_id: str) -> Dict[str, Any]:
        runtime = self._get_runtime(session_id)
        with runtime.lock:
            segment = runtime.session.segments[segment_id]
            if not any(candidate.candidate_id == candidate_id for candidate in segment.candidates):
                raise ValueError("Candidate not found for this segment.")
            segment.selected_candidate_id = candidate_id
            self._touch(runtime.session)
            self._save_runtime(runtime)
        return self._serialize_session(runtime.session)

    def regenerate_segment(self, session_id: str, segment_id: int) -> Dict[str, Any]:
        runtime = self._get_runtime(session_id)
        with runtime.lock:
            self._regenerate_segment_candidates(runtime, segment_id)
            self._touch(runtime.session)
            self._save_runtime(runtime)
        return self._serialize_session(runtime.session)

    def regenerate_all(self, session_id: str) -> Dict[str, Any]:
        runtime = self._get_runtime(session_id)
        with runtime.lock:
            self._regenerate_all_candidates(runtime)
            self._touch(runtime.session)
            self._save_runtime(runtime)
        return self._serialize_session(runtime.session)

    def assemble(self, session_id: str) -> Dict[str, Any]:
        runtime = self._get_runtime(session_id)
        with runtime.lock:
            clip_selections = self._build_clip_selections(runtime)
            assembler = VideoAssembler(use_ffmpeg=True)
            builder = VideoSequenceBuilder(assembler, temp_dir=str(runtime.session_dir / "temp"))
            output_path = Path(runtime.session.output_dir) / f"{runtime.session.name}_assembled.mp4"
            success = builder.build_sequence(
                clip_selections,
                runtime.session.audio_file,
                str(output_path),
            )
            if not success:
                raise RuntimeError("Video assembly failed.")
            runtime.session.assembled_video_path = str(output_path)
            self._touch(runtime.session)
            self._save_runtime(runtime)
        return self._serialize_session(runtime.session)

    def export_segments(self, session_id: str) -> Path:
        runtime = self._get_runtime(session_id)
        export_path = runtime.session_dir / "edited_segments.json"
        payload = {
            "segments": [
                {
                    "text": segment.text,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "duration": segment.duration,
                    "description": segment.description,
                    "keywords": sorted(set(segment.keywords + segment.extra_keywords)),
                }
                for segment in runtime.session.segments
            ]
        }
        _write_json(export_path, payload)
        return export_path

    def _get_runtime(self, session_id: str) -> EditorRuntime:
        with self._lock:
            runtime = self._runtimes.get(session_id)
        if runtime is not None:
            return runtime

        session_path = SESSIONS_ROOT / session_id / "session.json"
        if not session_path.exists():
            raise KeyError(f"Unknown session: {session_id}")
        session = self._session_from_dict(_read_json(session_path, {}))
        runtime = EditorRuntime(session=session, session_dir=session_path.parent)
        self._initialize_runtime(runtime)
        with self._lock:
            self._runtimes[session_id] = runtime
        return runtime

    def _initialize_runtime(self, runtime: EditorRuntime) -> None:
        session = runtime.session
        mode = session.retrieval_mode
        cache_dir = Path(session.cache_dir)
        gpu_device = session.config["gpu_device"]
        use_windowing = session.config.get("windowing", False)
        window_size = float(session.config.get("window_size", 5.0))
        window_overlap = float(session.config.get("window_overlap", 0.5))

        if mode == "videoprism":
            if not VIDEOPRISM_AVAILABLE or VideoIndexer is None:
                raise RuntimeError("VideoPrism is not available in this environment.")
            index_dir = cache_dir / "editor_videoprism"
            indexer = VideoIndexer(
                model_name=session.config.get("videoprism_model", "videoprism_lvt_public_v1_base"),
                index_dir=str(index_dir),
                device=gpu_device,
            )
            if not indexer.load_index():
                indexer.index_videos(session.video_dir, use_windowing=use_windowing, window_size=window_size, window_overlap=window_overlap)
            runtime.indexer = indexer
            runtime.matcher = VideoTextMatcher(indexer, model_name=session.config.get("videoprism_model", "videoprism_lvt_public_v1_base"), device=gpu_device)
            runtime.keyword_indexer = None
            return

        if mode == "openclip":
            index_dir = cache_dir / "editor_openclip"
            indexer = OpenCLIPVideoIndexer(
                model_name=session.config.get("openclip_model", "ViT-B-32"),
                index_dir=str(index_dir),
                device=gpu_device,
            )
            if not indexer.load_index():
                indexer.index_videos(session.video_dir, use_windowing=use_windowing, window_size=window_size, window_overlap=window_overlap)
            runtime.indexer = indexer
            runtime.matcher = OpenCLIPTextMatcher(indexer)
            runtime.keyword_indexer = None
            return

        if mode != "writeavideo":
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        keyword_index = MultiModalKeywordIndexer(
            index_dir=str(cache_dir / "editor_wav_keywords"),
            yolo_model=session.config.get("yolo_model", "yolov8n"),
            frames_per_second=float(session.config.get("frames_per_second", 1.0)),
            enable_object_detection=bool(session.config.get("enable_object_detection", True)),
            enable_face_detection=bool(session.config.get("enable_face_detection", False)),
        )
        if not keyword_index.load_index():
            keyword_index.index_videos(session.video_dir)
            keyword_index.save_index()

        openclip_index = OpenCLIPVideoIndexer(
            model_name=session.config.get("openclip_model", "ViT-B-32"),
            index_dir=str(cache_dir / "editor_openclip"),
            device=gpu_device,
        )
        if not openclip_index.load_index():
            openclip_index.index_videos(session.video_dir, use_windowing=use_windowing, window_size=window_size, window_overlap=window_overlap)

        runtime.indexer = openclip_index
        runtime.keyword_indexer = keyword_index
        runtime.matcher = WriteAVideoMatcher(
            keyword_indexer=keyword_index,
            openclip_indexer=openclip_index,
            candidate_pool_size=int(session.config.get("candidate_pool_size", 10)),
            keyword_weight=float(session.config.get("keyword_weight", 0.2)),
        )

    def _load_segments(
        self,
        audio_file: Optional[str],
        segments_file: Optional[str],
        cache_dir: Path,
        whisper_model: str,
        llm_model: str,
        gpu_device: str,
        use_simple_segmentation: bool,
    ) -> List[EditorSegment]:
        if segments_file:
            raw_segments = load_manual_segments(segments_file)
            return [self._segment_from_script_segment(segment) for segment in raw_segments]

        if not audio_file:
            raise ValueError("Either a segments file or an audio file is required.")

        transcription_cache = cache_dir / "transcription.json"
        segments_cache = cache_dir / "segments.json"

        if segments_cache.exists():
            loaded = ScriptSegmenter.load_segments(str(segments_cache))
            return [self._segment_from_script_segment(segment) for segment in loaded]

        if transcription_cache.exists():
            with open(transcription_cache, "r", encoding="utf-8") as handle:
                raw_transcription = json.load(handle)
            transcription = TranscriptionResult(
                full_text=raw_transcription["full_text"],
                language=raw_transcription["language"],
                duration=raw_transcription["duration"],
                segments=raw_transcription["segments"],
                words=[WordSegment(**word) for word in raw_transcription["words"]],
            )
        else:
            transcriber = VoiceTranscriber(model_size=whisper_model, device=gpu_device)
            transcription = transcriber.transcribe(audio_file)
            transcriber.save_transcription(transcription, str(transcription_cache))

        segmenter = ScriptSegmenter(
            model_name=llm_model,
            device=gpu_device,
            use_simple_segmentation=use_simple_segmentation,
        )
        words_with_timing = [
            {
                "word": word.word,
                "start_time": word.start_time,
                "end_time": word.end_time,
            }
            for word in transcription.words
        ]
        scripted = segmenter.segment_script(transcription.full_text, words_with_timing)
        segmenter.save_segments(scripted, str(segments_cache))
        return [self._segment_from_script_segment(segment) for segment in scripted]

    def _segment_from_script_segment(self, segment: ScriptSegment) -> EditorSegment:
        return EditorSegment(
            segment_id=segment.segment_id,
            text=segment.text,
            start_time=segment.start_time,
            end_time=segment.end_time,
            duration=max(0.2, segment.duration),
            description=segment.description,
            keywords=list(segment.keywords or []),
            extra_keywords=list(segment.keywords or []),
        )

    def _regenerate_all_candidates(self, runtime: EditorRuntime) -> None:
        for segment in runtime.session.segments:
            self._regenerate_segment_candidates(runtime, segment.segment_id)

    def _regenerate_segment_candidates(self, runtime: EditorRuntime, segment_id: int) -> None:
        segment = runtime.session.segments[segment_id]
        effective_duration = max(0.2, segment.duration * segment.duration_multiplier)
        query_text = self._build_query_text(segment)
        matcher = runtime.matcher

        candidates = matcher.match_segment_to_videos(
            query_text,
            effective_duration,
            k=8,
            allow_reuse=True,
            used_videos=set(),
            match_only=False,
        )

        built: List[EditorCandidate] = []
        for rank, candidate in enumerate(candidates, start=1):
            trim_start, trim_end = self._candidate_trim_times(matcher, candidate, effective_duration)
            motion_score = self._measure_motion(runtime, candidate["file_path"], trim_start, trim_end)
            adjusted_score = float(candidate.get("combined_score", candidate.get("similarity", 0.0)))
            if segment.movement_preference == "more":
                adjusted_score += 0.18 * motion_score
            elif segment.movement_preference == "less":
                adjusted_score += 0.18 * (1.0 - motion_score)

            matched_keywords = self._matched_keywords(runtime, segment, candidate)
            thumbnail_path = self._ensure_thumbnail(runtime, candidate["file_path"], trim_start, trim_end, rank)

            built.append(
                EditorCandidate(
                    candidate_id=f"{segment.segment_id}:{rank}:{candidate['video_id']}",
                    video_id=candidate["video_id"],
                    file_path=candidate["file_path"],
                    file_name=Path(candidate["file_path"]).name,
                    duration=float(candidate["duration"]),
                    similarity_score=float(candidate.get("similarity_score", candidate.get("similarity", 0.0))),
                    combined_score=float(adjusted_score),
                    motion_score=float(motion_score),
                    context_score=float(candidate.get("context_score", 0.0)),
                    keyword_score=float(candidate.get("keyword_score", 0.0)),
                    trim_start=float(trim_start),
                    trim_end=float(trim_end),
                    trim_duration=float(trim_end - trim_start),
                    matched_keywords=matched_keywords,
                    thumbnail_path=thumbnail_path,
                    rank=rank,
                )
            )

        built.sort(key=lambda item: item.combined_score, reverse=True)
        for rank, candidate in enumerate(built, start=1):
            candidate.rank = rank

        segment.candidates = built
        candidate_ids = {item.candidate_id for item in built}
        if segment.selected_candidate_id not in candidate_ids:
            segment.selected_candidate_id = built[0].candidate_id if built else None

    def _candidate_trim_times(self, matcher: Any, candidate: Dict[str, Any], target_duration: float) -> Tuple[float, float]:
        if "trim_start" in candidate and "trim_end" in candidate:
            return float(candidate["trim_start"]), float(candidate["trim_end"])
        if hasattr(matcher, "_calculate_trim_times"):
            return matcher._calculate_trim_times(float(candidate["duration"]), target_duration)
        if hasattr(matcher, "text_matcher") and hasattr(matcher.text_matcher, "_calculate_trim_times"):
            return matcher.text_matcher._calculate_trim_times(float(candidate["duration"]), target_duration)
        clip_duration = float(candidate["duration"])
        if clip_duration <= target_duration:
            return 0.0, clip_duration
        trim_start = (clip_duration - target_duration) / 2.0
        return trim_start, trim_start + target_duration

    def _matched_keywords(self, runtime: EditorRuntime, segment: EditorSegment, candidate: Dict[str, Any]) -> List[str]:
        query_keywords = set(_extract_text_keywords(self._build_query_text(segment)))
        if runtime.keyword_indexer is None:
            filename_keywords = set(_extract_text_keywords(Path(candidate["file_path"]).stem.replace("_", " ")))
            return sorted(query_keywords & filename_keywords)
        keyword_entry = next(
            (entry for entry in runtime.keyword_indexer.video_entries if entry.video_id == candidate["video_id"]),
            None,
        )
        if keyword_entry is None:
            return []
        return sorted(query_keywords & set(keyword_entry.all_keywords))

    def _build_query_text(self, segment: EditorSegment) -> str:
        extras = [keyword.strip() for keyword in segment.extra_keywords if keyword.strip()]
        if not extras:
            return segment.text
        return f"{segment.text}. Visual focus: {', '.join(extras)}."

    def _measure_motion(self, runtime: EditorRuntime, file_path: str, start_time: float, end_time: float) -> float:
        cache_key = f"{file_path}:{start_time:.2f}:{end_time:.2f}"
        if cache_key in runtime.motion_cache:
            return runtime.motion_cache[cache_key]

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return 0.5
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        start_frame = max(0, int(start_time * fps))
        end_frame = max(start_frame + 2, int(end_time * fps))
        sample_frames = np.linspace(start_frame, end_frame - 1, num=min(10, max(2, end_frame - start_frame)), dtype=int)

        previous = None
        diffs = []
        for frame_index in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90))
            if previous is not None:
                diff = cv2.absdiff(previous, gray)
                diffs.append(float(np.mean(diff)))
            previous = gray
        cap.release()

        if not diffs:
            score = 0.5
        else:
            score = _clamp(float(np.mean(diffs)) / 30.0, 0.0, 1.0)
        runtime.motion_cache[cache_key] = score
        return score

    def _ensure_thumbnail(self, runtime: EditorRuntime, file_path: str, start_time: float, end_time: float, rank: int) -> str:
        thumb_dir = runtime.session_dir / "thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        thumb_name = f"{Path(file_path).stem}_{int(start_time * 100)}_{int(end_time * 100)}_{rank}.jpg"
        thumb_path = thumb_dir / thumb_name
        if thumb_path.exists():
            return str(thumb_path)

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return ""
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        midpoint = max(start_time, (start_time + end_time) / 2.0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(midpoint * fps))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return ""
        cv2.imwrite(str(thumb_path), frame)
        return str(thumb_path)

    def _build_clip_selections(self, runtime: EditorRuntime) -> List[ClipSelection]:
        clip_selections: List[ClipSelection] = []
        current_time = 0.0
        for segment in runtime.session.segments:
            candidate = next((item for item in segment.candidates if item.candidate_id == segment.selected_candidate_id), None)
            if candidate is None:
                raise RuntimeError(f"Segment {segment.segment_id} does not have a selected candidate.")

            desired_duration = max(0.2, segment.duration * segment.duration_multiplier)
            trim_start, trim_end = self._apply_timing_bias(candidate, desired_duration, segment.timing_bias)
            selection = ClipSelection(
                segment_id=segment.segment_id,
                video_id=candidate.video_id,
                video_file_path=candidate.file_path,
                start_time=current_time,
                end_time=current_time + desired_duration,
                duration=desired_duration,
                trim_start=trim_start,
                trim_end=trim_end,
                trim_duration=trim_end - trim_start,
                similarity_score=candidate.similarity_score,
                motion_score=candidate.motion_score,
                context_score=candidate.context_score,
                combined_score=candidate.combined_score,
                is_reused=False,
            )
            current_time = selection.end_time
            clip_selections.append(selection)
        return clip_selections

    def _apply_timing_bias(self, candidate: EditorCandidate, desired_duration: float, timing_bias: float) -> Tuple[float, float]:
        available = max(candidate.duration, candidate.trim_duration)
        clip_duration = min(desired_duration, available)
        slack = max(0.0, available - clip_duration)
        bias_factor = (timing_bias + 1.0) / 2.0
        start_offset = slack * bias_factor
        trim_start = start_offset
        trim_end = trim_start + clip_duration
        return trim_start, trim_end

    def _update_global_keywords(self, runtime: EditorRuntime) -> None:
        keywords = set()
        for segment in runtime.session.segments:
            keywords.update(segment.keywords)
            keywords.update(segment.extra_keywords)
            keywords.update(_extract_text_keywords(segment.text))
        if runtime.keyword_indexer is not None:
            for keyword in sorted(runtime.keyword_indexer.inverted_index.keys())[:80]:
                keywords.add(keyword)
        runtime.session.global_keywords = sorted(keyword for keyword in keywords if keyword)[:80]

    def _reindex_segments(self, session: EditorSession) -> None:
        for index, segment in enumerate(session.segments):
            segment.segment_id = index
        session.segments.sort(key=lambda item: item.segment_id)

    def _touch(self, session: EditorSession) -> None:
        session.updated_at = _now_iso()

    def _save_runtime(self, runtime: EditorRuntime) -> None:
        runtime.session_dir.mkdir(parents=True, exist_ok=True)
        _write_json(runtime.session_dir / "session.json", asdict(runtime.session))

    def _serialize_session(self, session: EditorSession) -> Dict[str, Any]:
        payload = asdict(session)
        payload["assembled_video_url"] = self.media_url(session.assembled_video_path) if session.assembled_video_path else None
        for segment in payload["segments"]:
            segment["locked_by_audio"] = bool(session.audio_file)
            for candidate in segment["candidates"]:
                candidate["video_url"] = self.media_url(candidate["file_path"])
                candidate["thumbnail_url"] = self.media_url(candidate["thumbnail_path"]) if candidate.get("thumbnail_path") else None
        return payload

    def _session_from_dict(self, data: Dict[str, Any]) -> EditorSession:
        segments = []
        for segment_data in data.get("segments", []):
            cleaned_candidates = []
            for candidate in segment_data.get("candidates", []):
                candidate_payload = dict(candidate)
                candidate_payload.pop("video_url", None)
                candidate_payload.pop("thumbnail_url", None)
                cleaned_candidates.append(EditorCandidate(**candidate_payload))
            candidates = cleaned_candidates
            segment_payload = dict(segment_data)
            segment_payload.pop("locked_by_audio", None)
            segment_payload["candidates"] = candidates
            segments.append(EditorSegment(**segment_payload))
        session_payload = dict(data)
        session_payload.pop("assembled_video_url", None)
        session_payload["segments"] = segments
        return EditorSession(**session_payload)

    def media_url(self, file_path: Optional[str]) -> Optional[str]:
        if not file_path:
            return None
        return f"/api/media?path={quote(file_path)}"


settings_store = SettingsStore()
cache_manager = CacheManager()
benchmark_manager = BenchmarkManager()
job_manager = JobManager()
editor_manager = EditorSessionManager()
