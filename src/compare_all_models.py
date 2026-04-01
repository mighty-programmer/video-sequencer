#!/usr/bin/env python3
"""
Parallel Grid Search Orchestrator

This script coordinates running OpenCLIP, VideoPrism, and Write-A-Video grid
searches in parallel on the same benchmark. It ensures perfect cache isolation,
shares pre-generated LLM prompts, forces maximum configurations, and generates
a summary script to compare their results.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import re
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from prompt_generator import create_ensemble_templates

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')


def resolve_benchmark(benchmark_num: str, base_dir: str = './data/benchmarks') -> dict:
    base = Path(base_dir)
    video_dir = base / 'videos' / f'video_{benchmark_num}'
    segments_file = base / 'segments' / f'benchmark_{benchmark_num}_segments.json'
    gt_file = base / 'gdtruth' / f'benchmark_{benchmark_num}_ground_truth.json'
    
    if not video_dir.exists() or not segments_file.exists() or not gt_file.exists():
        raise FileNotFoundError(f"Benchmark {benchmark_num} files not found in {base_dir}")
        
    return {
        'video_dir': str(video_dir),
        'segments': str(segments_file),
        'ground_truth': str(gt_file),
    }

def discover_all_benchmarks(base_dir: str = './data/benchmarks') -> list:
    base = Path(base_dir)
    videos_dir = base / 'videos'
    bms = []
    if videos_dir.exists():
        for d in videos_dir.iterdir():
            if d.is_dir():
                m = re.match(r'^video_(\d+)$', d.name)
                if m:
                    bms.append(m.group(1))
    return sorted(bms, key=int)


def setup_isolation(base_output: Path, cache_dir: Path):
    """Create isolated output and cache directories."""
    dirs = {
        'openclip': {'out': base_output / 'openclip', 'cache': cache_dir / 'compare_openclip'},
        'videoprism': {'out': base_output / 'videoprism', 'cache': cache_dir / 'compare_videoprism'},
        'wav': {'out': base_output / 'wav', 'cache': cache_dir / 'compare_wav'}
    }
    
    for v in dirs.values():
        v['out'].mkdir(parents=True, exist_ok=True)
        v['cache'].mkdir(parents=True, exist_ok=True)
        
    return dirs


def create_summary_script(base_output: Path, benchmarks: list):
    """Generate a bash script to summarize the results across all benchmarks."""
    script_path = base_output / "generate_comparison_report.sh"
    
    script_content = [
        "#!/bin/bash",
        "# Automatically generated comparison summary script",
        "",
        'BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"',
        'OUTPUT_FILE="$BASE_DIR/comparison_summary.txt"',
        "",
        'echo "=====================================================================" > "$OUTPUT_FILE"',
        'echo "                PARALLEL GRID SEARCH COMPARISON REPORT                 " >> "$OUTPUT_FILE"',
        'echo "=====================================================================" >> "$OUTPUT_FILE"',
        'echo "" >> "$OUTPUT_FILE"',
        "",
        "function extract_best {",
        "    local name=$1",
        "    local json_file=$2",
        "    ",
        '    if [ ! -f "$json_file" ]; then',
        '        echo "[$name] Error: Results file not found at $json_file" >> "$OUTPUT_FILE"',
        '        echo "" >> "$OUTPUT_FILE"',
        "        return",
        "    fi",
        "    ",
        "    local exact=$(jq -r '.results[0].exact_match_accuracy' \"$json_file\")",
        "    local top3=$(jq -r '.results[0].top_3_accuracy' \"$json_file\")",
        "    local top5=$(jq -r '.results[0].top_5_accuracy' \"$json_file\")",
        "    local mrr=$(jq -r '.results[0].mrr' \"$json_file\")",
        "    local total_time=$(jq -r '.total_time_seconds' \"$json_file\")",
        "    local configs=$(jq -r '.total_configs_tested' \"$json_file\")",
        "    ",
        '    echo "[$name]" >> "$OUTPUT_FILE"',
        '    echo "  Tested Configurations: $configs" >> "$OUTPUT_FILE"',
        '    echo "  Total Execution Time:  ${total_time}s" >> "$OUTPUT_FILE"',
        '    echo "  -------------------------------------------------" >> "$OUTPUT_FILE"',
        '    echo "  Best Exact Match Accuracy: ${exact}%" >> "$OUTPUT_FILE"',
        '    echo "  Best Top-3 Accuracy:       ${top3}%" >> "$OUTPUT_FILE"',
        '    echo "  Best Top-5 Accuracy:       ${top5}%" >> "$OUTPUT_FILE"',
        '    echo "  Best Mean Reciprocal Rank: ${mrr}" >> "$OUTPUT_FILE"',
        '    echo "" >> "$OUTPUT_FILE"',
        "}"
    ]
    
    for bm in benchmarks:
        script_content.extend([
            f'echo "=== BENCHMARK {bm} ===" >> "$OUTPUT_FILE"',
            f'extract_best "OpenCLIP" "$BASE_DIR/benchmark_{bm}/openclip/grid_search_results.json"',
            f'extract_best "Write-A-Video" "$BASE_DIR/benchmark_{bm}/wav/wav_grid_search_results.json"',
            f'extract_best "VideoPrism" "$BASE_DIR/benchmark_{bm}/videoprism/videoprism_grid_search_results.json"',
            'echo "" >> "$OUTPUT_FILE"'
        ])
        
    script_content.extend([
        'echo "Summary saved to $OUTPUT_FILE"',
        'cat "$OUTPUT_FILE"'
    ])
    
    with open(script_path, 'w') as f:
        f.write("\\n".join(script_content))
    
    # Make executable
    script_path.chmod(0o755)
    return script_path


def main():
    parser = argparse.ArgumentParser(description='Parallel Grid Search Orchestrator')
    parser.add_argument('--benchmark', '-b', required=True, help='Benchmark number(s) comma-separated (e.g. 6,7) or "all"')
    parser.add_argument('--output', required=True, help='Base output directory (e.g., output/comparison_multi)')
    parser.add_argument('--cache-dir', default='./cache', help='Base cache directory')
    parser.add_argument('--llm-model', default='llama3.2:3b', help='LLM model to use for shared prompts')
    parser.add_argument('--device', default='cuda:0', help='GPU Device')
    parser.add_argument('--no-windowing', action='store_true', help='Disable temporal windowing')
    
    args = parser.parse_args()
    
    if args.benchmark.lower() == 'all':
        benchmarks = discover_all_benchmarks()
        if not benchmarks:
            logger.error("No benchmarks found to process.")
            sys.exit(1)
    else:
        benchmarks = [b.strip() for b in args.benchmark.split(',') if b.strip()]
        
    base_output = Path(args.output)
    base_cache = Path(args.cache_dir)
    
    # Parse comma-separated list of GPUs
    devices = [d.strip() for d in args.device.split(',') if d.strip()]
    if not devices:
        devices = ['cuda:0']
        
    command_lists = {'openclip_compare': [], 'videoprism_compare': [], 'wav_compare': []}
    
    for bm in benchmarks:
        logger.info(f"\\n--- Setup for Benchmark {bm} ---")
        try:
            bm_paths = resolve_benchmark(bm)
        except Exception as e:
            logger.error(str(e))
            continue
            
        bm_output = base_output / f"benchmark_{bm}"
        dirs = setup_isolation(bm_output, base_cache)
        
        # Generate common LLM prompts securely to the first output directory, then copy
        logger.info(f"Pre-generating LLM prompts using {args.llm_model}...")
        
        with open(bm_paths['segments'], 'r') as f:
            segments_data = json.load(f)
            segments = segments_data if isinstance(segments_data, list) else segments_data.get('segments', segments_data.get('mappings', []))
            for i, seg in enumerate(segments):
                if 'text' not in seg:
                    seg['text'] = seg.get('segment_text', f'segment_{i}')
                    
        master_cache_path = dirs['openclip']['out'] / 'llm_prompts_cache.json'
        
        create_ensemble_templates(
            segments=segments,
            llm_model=args.llm_model,
            num_variations=5,
            cache_file=str(master_cache_path)
        )
        
        # Distribute the cache file to the isolated outputs so they each find it naturally
        for target in ['videoprism', 'wav']:
            target_cache = dirs[target]['out'] / 'llm_prompts_cache.json'
            with open(master_cache_path, 'r') as src, open(target_cache, 'w') as dst:
                dst.write(src.read())

        common_args = [
            f"--video-dir {bm_paths['video_dir']}",
            f"--segments {bm_paths['segments']}",
            f"--ground-truth {bm_paths['ground_truth']}",
            f"--llm-model {args.llm_model}"
        ]
        if args.no_windowing:
            common_args.append("--no-windowing")
            
        # OpenCLIP Max Combinations (Gets GPU 0)
        oc_args = common_args + [
            f"--device {devices[0 % len(devices)]}",
            f"--output {dirs['openclip']['out']}",
            f"--cache-dir {dirs['openclip']['cache']}",
            "--models ViT-B-32 ViT-B-16 ViT-L-14",
            "--frames 4 8 16 32",
            "--aggregations mean max best_frame",
            "--prompt-modes none template:video template:photo template:cooking ensemble:template ensemble:llm"
        ]
        command_lists['openclip_compare'].append(f"python src/grid_search.py {' '.join(oc_args)}")
        
        # VideoPrism Max Combinations (Gets GPU 1)
        vp_args = common_args + [
            f"--device {devices[1 % len(devices)]}",
            f"--output {dirs['videoprism']['out']}",
            f"--cache-dir {dirs['videoprism']['cache']}",
            "--models videoprism_lvt_public_v1_base videoprism_lvt_public_v1_large",
            "--frames 8 16 32",
            "--prompt-modes none template:video template:photo template:scene template:cooking ensemble:template ensemble:llm"
        ]
        command_lists['videoprism_compare'].append(f"python src/videoprism_grid_search.py {' '.join(vp_args)}")
        
        # WAV Max Combinations (Gets GPU 2)
        wav_args = common_args + [
            f"--device {devices[2 % len(devices)]}",
            f"--output {dirs['wav']['out']}",
            f"--cache-dir {dirs['wav']['cache']}",
            "--models ViT-B-32 ViT-B-16 ViT-L-14",
            "--frames 4 8 16 32",
            "--aggregations mean max best_frame",
            "--prompt-modes none template:video template:photo template:cooking template:scene ensemble:template ensemble:llm",
            "--pool-sizes 5 10 20",
            "--keyword-weights 0.0 0.1 0.2 0.3"
        ]
        command_lists['wav_compare'].append(f"python src/wav_grid_search.py {' '.join(wav_args)}")

    create_summary_script(base_output, benchmarks)

    # Spawn tmux sessions, chaining the benchmark commands with ;
    for session_name, cmds in command_lists.items():
        if not cmds:
            continue
        logger.info(f"Spawning {session_name} with {len(cmds)} benchmark runs...")
        chained_cmd = " ; ".join(cmds)
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name, f"{chained_cmd} ; read -p 'Press Enter to exit'"])
        
    print(f"\\nAll 3 grid search pipelines explicitly maxed out across {len(benchmarks)} benchmarks and launched in parallel!")
    print(f"Monitor them via:")
    print(f"  tmux attach -t openclip_compare")
    print(f"  tmux attach -t videoprism_compare")
    print(f"  tmux attach -t wav_compare")
    print(f"When all finish, run:")
    print(f"  bash {base_output / 'generate_comparison_report.sh'}")

if __name__ == '__main__':
    main()
