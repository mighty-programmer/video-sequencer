#!/usr/bin/env python3
"""
Run a small, reproducible VideoPrism validation sweep for the feature branch.

This script intentionally targets the two fast-sweep hard cases where VideoPrism
lagged the best pipeline (Benchmarks 1 and 6). It does not inspect ground-truth
contents directly; it delegates metric computation to the existing benchmark
runner and compares aggregate result JSON files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_CASES: Dict[str, Dict[str, Any]] = {
    # Fast-sweep best VideoPrism settings for the gap cases.
    "1": {
        "model": "videoprism_lvt_public_v1_base",
        "frames": 8,
        "resolution": 288,
    },
    "6": {
        "model": "videoprism_lvt_public_v1_large",
        "frames": 16,
        "resolution": 396,
    },
}


def run_command(command: List[str], cwd: Path) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def load_best_result(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    results = data.get("results") or []
    if not results:
        raise RuntimeError(f"No results found in {path}")
    best = results[0]
    return {
        "path": str(path),
        "exact_match_accuracy": best.get("exact_match_accuracy"),
        "top_3_accuracy": best.get("top_3_accuracy"),
        "top_5_accuracy": best.get("top_5_accuracy"),
        "mrr": best.get("mrr"),
        "config": best.get("config", {}),
        "total_configs_tested": data.get("total_configs_tested"),
        "total_time_seconds": data.get("total_time_seconds"),
    }


def write_summary(run_dir: Path, summary: Dict[str, Any]) -> None:
    json_path = run_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    lines = [
        "# VideoPrism Improvement Validation",
        "",
        f"Run directory: `{run_dir}`",
        f"Timestamp: `{summary['timestamp']}`",
        "",
        "| Benchmark | Best Exact | Top-3 | MRR | Config |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for bm, result in summary["benchmarks"].items():
        cfg = result["config"]
        config_label = (
            f"{cfg.get('model_name')} | {cfg.get('num_frames')}f | "
            f"{cfg.get('resolution')}p | dual={cfg.get('use_dual_softmax')} | "
            f"norm={cfg.get('score_normalization', 'none')} | "
            f"csls_k={cfg.get('csls_k', 5)}"
        )
        lines.append(
            f"| {bm} | {result['exact_match_accuracy']} | "
            f"{result['top_3_accuracy']} | {result['mrr']} | `{config_label}` |"
        )
    lines.append("")
    lines.append("Each benchmark tests: raw Hungarian, Dual Softmax Hungarian, and CSLS Hungarian.")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate VideoPrism assignment improvements on hard benchmarks.")
    parser.add_argument("--benchmarks", nargs="+", default=sorted(DEFAULT_CASES), help="Benchmark numbers to run")
    parser.add_argument("--output-root", default="output/videoprism_improvement_validation")
    parser.add_argument("--cache-dir", default="./cache/videoprism_improvement_validation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--csls-k", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--keep-windowing", action="store_true", help="Keep temporal windowing enabled")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = repo_root / args.output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "timestamp": timestamp,
        "run_dir": str(run_dir),
        "benchmarks": {},
    }

    for benchmark in args.benchmarks:
        if benchmark not in DEFAULT_CASES:
            raise ValueError(f"No default hard-case config for benchmark {benchmark}")
        case = DEFAULT_CASES[benchmark]
        output_dir = run_dir / f"benchmark_{benchmark}" / "videoprism"
        command = [
            args.python,
            "-u",
            "src/videoprism_grid_search.py",
            "--benchmark",
            str(benchmark),
            "--output",
            str(output_dir),
            "--cache-dir",
            args.cache_dir,
            "--device",
            args.device,
            "--models",
            case["model"],
            "--frames",
            str(case["frames"]),
            "--resolutions",
            str(case["resolution"]),
            "--dual-softmax",
            "false",
            "true",
            "--score-normalizations",
            "none",
            "csls",
            "--csls-k",
            *[str(item) for item in args.csls_k],
            "--prompt-modes",
            "none",
            "--query-modes",
            "original",
            "--assignment-methods",
            "hungarian",
        ]
        if not args.keep_windowing:
            command.append("--no-windowing")

        run_command(command, cwd=repo_root)
        result_path = output_dir / "videoprism_grid_search_results.json"
        summary["benchmarks"][benchmark] = load_best_result(result_path)

    write_summary(run_dir, summary)
    print(f"\nValidation summary written to {run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
