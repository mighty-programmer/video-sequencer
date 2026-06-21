import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from validate_videoprism_improvements import (
    attach_baseline_comparison,
    load_baseline_result,
    write_summary,
)


def write_result(path, exact):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "exact_match_accuracy": exact,
                        "top_3_accuracy": exact,
                        "top_5_accuracy": exact,
                        "mrr": exact / 100.0,
                        "config": {
                            "model_name": "videoprism_lvt_public_v1_large",
                            "num_frames": 8,
                            "resolution": 288,
                            "use_dual_softmax": False,
                            "score_normalization": "csls",
                            "csls_k": 3,
                        },
                    }
                ],
                "total_configs_tested": 1,
                "total_time_seconds": 1.0,
            }
        )
    )


def test_validation_summary_reports_baseline_delta(tmp_path):
    repo_root = tmp_path / "repo"
    baseline_path = repo_root / "output" / "benchmark_1" / "videoprism" / "videoprism_grid_search_results.json"
    write_result(baseline_path, exact=71.4)

    baseline = load_baseline_result(repo_root, "output", "1")
    summary = {
        "timestamp": "test",
        "run_dir": str(tmp_path / "run"),
        "benchmarks": {
            "1": {
                "exact_match_accuracy": 78.6,
                "top_3_accuracy": 80.0,
                "mrr": 0.8,
                "config": baseline["config"],
                "baseline": baseline,
            }
        },
    }

    attach_baseline_comparison(summary)
    assert summary["benchmarks"]["1"]["exact_match_delta"] == 7.2
    assert summary["benchmarks"]["1"]["improved_over_baseline"] is True
    assert summary["comparison"]["all_comparable_improved"] is True

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_summary(run_dir, summary)
    text = (run_dir / "summary.md").read_text()
    assert "Comparable benchmarks improved: 1/1." in text
