#!/usr/bin/env python3
"""
Interactive Terminal Menu for Video Sequencer

Provides a user-friendly TUI for selecting pipeline options without
needing to remember CLI flags. Supports all pipeline modes including
the OpenCLIP grid search optimizer.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple


# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')


def print_header():
    """Print the application header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          VIDEO SEQUENCER - Interactive Menu                 ║")
    print("║          Text-Video Matching & Assembly Pipeline            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")


def print_menu(title: str, options: List[Tuple[str, str]], show_back: bool = True):
    """Print a numbered menu."""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}  {title}{Colors.END}")
    print(f"  {'─' * 56}")
    for i, (label, desc) in enumerate(options, 1):
        print(f"  {Colors.GREEN}{i}.{Colors.END} {Colors.BOLD}{label}{Colors.END}")
        if desc:
            print(f"     {Colors.DIM}{desc}{Colors.END}")
    if show_back:
        print(f"  {Colors.RED}0.{Colors.END} {Colors.BOLD}Back / Exit{Colors.END}")
    print()


def get_choice(max_val: int, prompt: str = "  Select option: ") -> int:
    """Get a numeric choice from the user."""
    while True:
        try:
            choice = input(f"{Colors.CYAN}{prompt}{Colors.END}").strip()
            if choice == '':
                continue
            val = int(choice)
            if 0 <= val <= max_val:
                return val
            print(f"  {Colors.RED}Please enter a number between 0 and {max_val}{Colors.END}")
        except ValueError:
            print(f"  {Colors.RED}Please enter a valid number{Colors.END}")
        except (KeyboardInterrupt, EOFError):
            print()
            return 0


def get_input(prompt: str, default: str = None) -> str:
    """Get text input with optional default."""
    if default:
        display = f"  {prompt} [{Colors.DIM}{default}{Colors.END}]: "
    else:
        display = f"  {prompt}: "
    try:
        val = input(display).strip()
        return val if val else (default or '')
    except (KeyboardInterrupt, EOFError):
        print()
        return default or ''


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get a yes/no answer."""
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        val = input(f"  {prompt} {suffix}: ").strip().lower()
        if val == '':
            return default
        return val in ('y', 'yes')
    except (KeyboardInterrupt, EOFError):
        print()
        return default


def browse_directory(prompt: str, default: str = None) -> str:
    """Browse for a directory path."""
    path = get_input(prompt, default)
    if path and Path(path).exists():
        print(f"  {Colors.GREEN}✓ Found: {path}{Colors.END}")
    elif path:
        print(f"  {Colors.YELLOW}⚠ Path not found: {path}{Colors.END}")
    return path


def browse_file(prompt: str, default: str = None, extensions: List[str] = None) -> str:
    """Browse for a file path."""
    path = get_input(prompt, default)
    if path and Path(path).exists():
        print(f"  {Colors.GREEN}✓ Found: {path}{Colors.END}")
    elif path:
        print(f"  {Colors.YELLOW}⚠ File not found: {path}{Colors.END}")
    return path


def discover_benchmarks(base_dir: str = './data/benchmarks') -> List[Dict]:
    """Auto-discover available benchmarks."""
    benchmarks = []
    base = Path(base_dir)
    
    # Look for segment files
    segments_dir = base / 'segments'
    if segments_dir.exists():
        for seg_file in sorted(segments_dir.glob('*.json')):
            name = seg_file.stem.replace('_segments', '')
            
            # Try to find matching ground truth and video dir
            gt_file = base / 'gdtruth' / f'{name}_ground_truth.json'
            
            # Try common video directory patterns
            video_dir = None
            for pattern in [base / 'videos' / name, base / 'videos' / name.replace('benchmark_', 'video_')]:
                if pattern.exists():
                    video_dir = str(pattern)
                    break
            
            # Try to find audio
            audio_file = None
            for pattern in [base / 'audio' / f'voiceover_{name.split("_")[-1]}.mp3',
                          base / 'audio' / f'{name}.mp3']:
                if pattern.exists():
                    audio_file = str(pattern)
                    break
            
            benchmarks.append({
                'name': name,
                'segments': str(seg_file),
                'ground_truth': str(gt_file) if gt_file.exists() else None,
                'video_dir': video_dir,
                'audio': audio_file
            })
    
    return benchmarks


def build_command(config: Dict) -> List[str]:
    """Build the command line from a config dict."""
    cmd = ['python', 'src/main.py']
    
    cmd.extend(['--video-dir', config['video_dir']])
    cmd.extend(['--output', config.get('output', './output')])
    
    if config.get('audio'):
        cmd.extend(['--audio', config['audio']])
    
    if config.get('segments'):
        cmd.extend(['--segments', config['segments']])
    
    if config.get('match_only'):
        cmd.append('--match-only')
    
    if not config.get('allow_reuse', True):
        cmd.append('--no-reuse')
    
    if not config.get('use_optimal', True):
        cmd.append('--greedy')
    
    if config.get('ground_truth'):
        cmd.extend(['--ground-truth', config['ground_truth']])
    
    if config.get('encoder', 'videoprism') == 'openclip':
        cmd.extend(['--encoder', 'openclip'])
        if config.get('openclip_model'):
            cmd.extend(['--openclip-model', config['openclip_model']])
    
    if config.get('no_windowing'):
        cmd.append('--no-windowing')
    elif config.get('window_size'):
        cmd.extend(['--window-size', str(config['window_size'])])
    if config.get('window_overlap'):
        cmd.extend(['--window-overlap', str(config['window_overlap'])])
    
    if config.get('gpu_device'):
        cmd.extend(['--gpu-device', config['gpu_device']])
    
    if config.get('verbose'):
        cmd.append('--verbose')
    
    return cmd


def run_command(cmd: List[str], dry_run: bool = False):
    """Run a command and display it."""
    cmd_str = ' \\\n    '.join(cmd)
    print(f"\n{Colors.BOLD}{Colors.BLUE}  Command:{Colors.END}")
    print(f"  {Colors.DIM}{cmd_str}{Colors.END}\n")
    
    if dry_run:
        print(f"  {Colors.YELLOW}(Dry run - command not executed){Colors.END}")
        return
    
    if not get_yes_no("Run this command?", default=True):
        return
    
    print(f"\n{Colors.BOLD}{'═' * 60}{Colors.END}")
    print()
    
    try:
        process = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
        print(f"\n{Colors.BOLD}{'═' * 60}{Colors.END}")
        if process.returncode == 0:
            print(f"  {Colors.GREEN}✓ Pipeline completed successfully{Colors.END}")
        else:
            print(f"  {Colors.RED}✗ Pipeline exited with code {process.returncode}{Colors.END}")
    except KeyboardInterrupt:
        print(f"\n  {Colors.YELLOW}⚠ Interrupted by user{Colors.END}")
    except Exception as e:
        print(f"  {Colors.RED}Error: {e}{Colors.END}")


# ─── Main Menu Screens ─────────────────────────────────────────────

def screen_quick_benchmark(config: Dict):
    """Quick benchmark run with auto-discovered settings."""
    clear_screen()
    print_header()
    
    benchmarks = discover_benchmarks()
    
    if not benchmarks:
        print(f"  {Colors.YELLOW}No benchmarks found in ./data/benchmarks/{Colors.END}")
        print(f"  {Colors.DIM}Make sure you have segments files in data/benchmarks/segments/{Colors.END}")
        input("\n  Press Enter to continue...")
        return
    
    options = []
    for bm in benchmarks:
        status = []
        if bm['video_dir']:
            status.append("videos ✓")
        else:
            status.append("videos ✗")
        if bm['ground_truth']:
            status.append("ground truth ✓")
        else:
            status.append("ground truth ✗")
        if bm['audio']:
            status.append("audio ✓")
        else:
            status.append("audio ✗")
        options.append((bm['name'], ', '.join(status)))
    
    print_menu("Select Benchmark", options)
    choice = get_choice(len(options))
    if choice == 0:
        return
    
    bm = benchmarks[choice - 1]
    
    if not bm['video_dir']:
        bm['video_dir'] = browse_directory("Video directory")
    
    # Select encoder
    print_menu("Select Encoder", [
        ("VideoPrism", "Temporal video understanding (requires JAX)"),
        ("OpenCLIP ViT-B-32", "Frame-level baseline (fast)"),
        ("OpenCLIP ViT-B-16", "Frame-level baseline (better)"),
        ("OpenCLIP ViT-L-14", "Frame-level baseline (best quality)"),
    ], show_back=False)
    enc_choice = get_choice(4, "  Select encoder: ")
    
    encoder_map = {
        1: ('videoprism', None),
        2: ('openclip', 'ViT-B-32'),
        3: ('openclip', 'ViT-B-16'),
        4: ('openclip', 'ViT-L-14'),
    }
    encoder, openclip_model = encoder_map.get(enc_choice, ('videoprism', None))
    
    config.update({
        'video_dir': bm['video_dir'],
        'segments': bm['segments'],
        'ground_truth': bm['ground_truth'],
        'audio': bm.get('audio'),
        'match_only': True,
        'allow_reuse': False,
        'encoder': encoder,
        'openclip_model': openclip_model,
        'no_windowing': get_yes_no("Disable windowing?", default=True),
        'verbose': get_yes_no("Verbose output?", default=True),
    })
    
    cmd = build_command(config)
    run_command(cmd)
    input("\n  Press Enter to continue...")


def screen_full_pipeline(config: Dict):
    """Full pipeline with assembly."""
    clear_screen()
    print_header()
    print(f"  {Colors.BOLD}Full Pipeline (Indexing + Matching + Assembly){Colors.END}\n")
    
    config['video_dir'] = browse_directory("Video directory", config.get('video_dir', './data/benchmarks/videos/video_2'))
    config['audio'] = browse_file("Audio file (voiceover)", config.get('audio'))
    config['output'] = get_input("Output directory", config.get('output', './output'))
    
    segments = browse_file("Segments file (optional, skip for auto-segmentation)")
    if segments:
        config['segments'] = segments
    
    config['match_only'] = False
    config['allow_reuse'] = not get_yes_no("Prevent clip reuse?", default=True)
    
    # Encoder selection
    print_menu("Select Encoder", [
        ("VideoPrism", "Temporal video understanding"),
        ("OpenCLIP", "Frame-level baseline"),
    ], show_back=False)
    enc = get_choice(2, "  Select encoder: ")
    config['encoder'] = 'openclip' if enc == 2 else 'videoprism'
    
    if config['encoder'] == 'openclip':
        print_menu("OpenCLIP Model", [
            ("ViT-B-32", "Fast, good baseline"),
            ("ViT-B-16", "Better spatial resolution"),
            ("ViT-L-14", "Best quality, slower"),
        ], show_back=False)
        model_choice = get_choice(3, "  Select model: ")
        config['openclip_model'] = ['ViT-B-32', 'ViT-B-16', 'ViT-L-14'][model_choice - 1]
    
    config['no_windowing'] = get_yes_no("Disable windowing?", default=False)
    if not config['no_windowing']:
        config['window_size'] = float(get_input("Window size (seconds)", "5.0"))
        config['window_overlap'] = float(get_input("Window overlap (0-1)", "0.5"))
    
    config['verbose'] = get_yes_no("Verbose output?", default=False)
    
    cmd = build_command(config)
    run_command(cmd)
    input("\n  Press Enter to continue...")


def screen_grid_search(config: Dict):
    """OpenCLIP grid search optimizer."""
    clear_screen()
    print_header()
    print(f"  {Colors.BOLD}OpenCLIP Grid Search Optimizer{Colors.END}")
    print(f"  {Colors.DIM}Sweep parameters to find optimal OpenCLIP configuration{Colors.END}\n")
    
    # Get required paths
    config['video_dir'] = browse_directory("Video directory", config.get('video_dir', './data/benchmarks/videos/video_2'))
    config['segments'] = browse_file("Segments file", config.get('segments'))
    config['ground_truth'] = browse_file("Ground truth file", config.get('ground_truth'))
    config['output'] = get_input("Output directory", config.get('output', './output'))
    
    if not config.get('video_dir') or not config.get('segments') or not config.get('ground_truth'):
        print(f"  {Colors.RED}Video dir, segments, and ground truth are required for grid search{Colors.END}")
        input("\n  Press Enter to continue...")
        return
    
    # Parameter selection
    print(f"\n  {Colors.BOLD}Select parameters to sweep:{Colors.END}")
    
    sweep_models = get_yes_no("Sweep OpenCLIP models (ViT-B-32, ViT-B-16, ViT-L-14)?", default=True)
    sweep_frames = get_yes_no("Sweep number of frames (4, 8, 16, 32)?", default=True)
    sweep_aggregation = get_yes_no("Sweep aggregation methods (mean, max, best_frame)?", default=True)
    sweep_prompts = get_yes_no("Sweep prompt templates (none, 'a video of...', 'a photo of...')?", default=True)
    use_ensemble = get_yes_no("Include ensemble of prompts (LLM-generated)?", default=True)
    
    if use_ensemble:
        print_menu("LLM for Prompt Generation", [
            ("Llama 3.2 3B Instruct", "Fast, good quality (via Ollama)"),
            ("Mistral 7B Instruct", "Higher quality paraphrasing (via Ollama)"),
            ("Phi-3 Mini 3.8B", "Compact, efficient (via Ollama)"),
            ("Custom Ollama model", "Specify your own model name"),
        ], show_back=False)
        llm_choice = get_choice(4, "  Select LLM: ")
        llm_models = {
            1: 'llama3.2:3b',
            2: 'mistral:7b-instruct',
            3: 'phi3:mini',
        }
        if llm_choice == 4:
            llm_model = get_input("Ollama model name")
        else:
            llm_model = llm_models.get(llm_choice, 'llama3.2:3b')
    else:
        llm_model = None
    
    gpu_device = get_input("GPU device", config.get('gpu_device', 'cuda:0'))
    
    # Build grid search command
    cmd = ['python', 'src/grid_search.py']
    cmd.extend(['--video-dir', config['video_dir']])
    cmd.extend(['--segments', config['segments']])
    cmd.extend(['--ground-truth', config['ground_truth']])
    cmd.extend(['--output', config['output']])
    cmd.extend(['--device', gpu_device])
    
    # Build custom parameter lists based on user selections
    if sweep_models:
        cmd.extend(['--models', 'ViT-B-32', 'ViT-B-16', 'ViT-L-14'])
    else:
        cmd.extend(['--models', 'ViT-B-32'])
    
    if sweep_frames:
        cmd.extend(['--frames', '4', '8', '16', '32'])
    else:
        cmd.extend(['--frames', '16'])
    
    if sweep_aggregation:
        cmd.extend(['--aggregations', 'mean', 'max', 'best_frame'])
    else:
        cmd.extend(['--aggregations', 'mean'])
    
    # Build prompt modes list
    prompt_modes = []
    if sweep_prompts:
        prompt_modes.extend(['none', 'template:video', 'template:photo', 'template:cooking', 'template:scene'])
    else:
        prompt_modes.append('none')
    
    if use_ensemble:
        prompt_modes.append('ensemble:template')
        if llm_model:
            prompt_modes.append('ensemble:llm')
            cmd.extend(['--llm-model', llm_model])
    
    cmd.extend(['--prompt-modes'] + prompt_modes)
    
    cmd_str = ' \\\n    '.join(cmd)
    print(f"\n{Colors.BOLD}{Colors.BLUE}  Command:{Colors.END}")
    print(f"  {Colors.DIM}{cmd_str}{Colors.END}\n")
    
    if not get_yes_no("Run grid search?", default=True):
        return
    
    print(f"\n{Colors.BOLD}{'═' * 60}{Colors.END}")
    print()
    
    try:
        process = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
        print(f"\n{Colors.BOLD}{'═' * 60}{Colors.END}")
        if process.returncode == 0:
            print(f"  {Colors.GREEN}✓ Grid search completed successfully{Colors.END}")
        else:
            print(f"  {Colors.RED}✗ Grid search exited with code {process.returncode}{Colors.END}")
    except KeyboardInterrupt:
        print(f"\n  {Colors.YELLOW}⚠ Interrupted by user{Colors.END}")
    except Exception as e:
        print(f"  {Colors.RED}Error: {e}{Colors.END}")
    
    input("\n  Press Enter to continue...")


def screen_settings(config: Dict):
    """Global settings."""
    clear_screen()
    print_header()
    print(f"  {Colors.BOLD}Settings{Colors.END}\n")
    
    config['gpu_device'] = get_input("GPU device", config.get('gpu_device', 'cuda:0'))
    config['output'] = get_input("Default output directory", config.get('output', './output'))
    config['cache_dir'] = get_input("Cache directory", config.get('cache_dir', './cache'))
    
    print(f"\n  {Colors.GREEN}✓ Settings updated{Colors.END}")
    input("\n  Press Enter to continue...")


def screen_cache_management(config: Dict):
    """Cache management."""
    clear_screen()
    print_header()
    print(f"  {Colors.BOLD}Cache Management{Colors.END}\n")
    
    cache_dir = Path(config.get('cache_dir', './cache'))
    
    # List cache contents
    if cache_dir.exists():
        items = list(cache_dir.iterdir())
        if items:
            print(f"  Cache directory: {cache_dir}")
            for item in sorted(items):
                if item.is_dir():
                    size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    print(f"    📁 {item.name} ({size / 1024:.1f} KB)")
                else:
                    print(f"    📄 {item.name} ({item.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  {Colors.DIM}Cache is empty{Colors.END}")
    else:
        print(f"  {Colors.DIM}Cache directory does not exist{Colors.END}")
    
    print_menu("Cache Actions", [
        ("Clear VideoPrism index cache", "Remove cached video_index/"),
        ("Clear OpenCLIP index cache", "Remove cached video_index_openclip/"),
        ("Clear ALL caches", "Remove entire cache directory"),
    ])
    
    choice = get_choice(3)
    if choice == 0:
        return
    
    if choice == 1:
        target = cache_dir / 'video_index'
    elif choice == 2:
        target = cache_dir / 'video_index_openclip'
    else:
        target = cache_dir
    
    if target.exists():
        if get_yes_no(f"Delete {target}?", default=False):
            import shutil
            shutil.rmtree(target)
            print(f"  {Colors.GREEN}✓ Deleted {target}{Colors.END}")
    else:
        print(f"  {Colors.DIM}Nothing to delete{Colors.END}")
    
    input("\n  Press Enter to continue...")


def main_menu():
    """Main interactive menu loop."""
    config = {
        'gpu_device': 'cuda:0',
        'output': './output',
        'cache_dir': './cache',
    }
    
    while True:
        clear_screen()
        print_header()
        
        print_menu("Main Menu", [
            ("Quick Benchmark", "Run a benchmark test with auto-discovered settings"),
            ("Full Pipeline", "Index → Match → Assemble with custom settings"),
            ("OpenCLIP Grid Search", "Find optimal OpenCLIP parameters via automated sweep"),
            ("Cache Management", "View and clear cached indexes"),
            ("Settings", "Configure GPU, output directory, etc."),
        ])
        
        choice = get_choice(5)
        
        if choice == 0:
            print(f"\n  {Colors.DIM}Goodbye!{Colors.END}\n")
            break
        elif choice == 1:
            screen_quick_benchmark(config)
        elif choice == 2:
            screen_full_pipeline(config)
        elif choice == 3:
            screen_grid_search(config)
        elif choice == 4:
            screen_cache_management(config)
        elif choice == 5:
            screen_settings(config)


if __name__ == '__main__':
    try:
        main_menu()
    except (KeyboardInterrupt, EOFError):
        print(f"\n\n  {Colors.DIM}Goodbye!{Colors.END}\n")
