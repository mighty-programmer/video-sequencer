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
import re
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


def discover_benchmark_numbers(base_dir: str = './data/benchmarks') -> List[Dict]:
    """
    Auto-discover available benchmarks by scanning the videos directory
    for folders matching the pattern 'video_*'.
    
    Returns a list of benchmark dicts with number, paths, and status info.
    The benchmark number is extracted from the video folder name.
    Segments and ground truth paths are derived from the number.
    """
    benchmarks = []
    base = Path(base_dir)
    videos_dir = base / 'videos'
    
    if not videos_dir.exists():
        return benchmarks
    
    # Scan video directories for pattern video_*
    for video_folder in sorted(videos_dir.iterdir()):
        if not video_folder.is_dir():
            continue
        
        # Extract benchmark number from folder name (e.g., video_2 -> 2, video_10 -> 10)
        match = re.match(r'^video_(\d+)$', video_folder.name)
        if not match:
            continue
        
        bm_number = match.group(1)
        
        # Derive paths from the benchmark number
        segments_file = base / 'segments' / f'benchmark_{bm_number}_segments.json'
        gt_file = base / 'gdtruth' / f'benchmark_{bm_number}_ground_truth.json'
        
        # Count videos in the folder
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        video_count = sum(
            1 for f in video_folder.iterdir()
            if f.suffix.lower() in video_extensions
        )
        
        # Count segments if file exists
        segment_count = 0
        if segments_file.exists():
            try:
                with open(segments_file, 'r') as f:
                    seg_data = json.load(f)
                if isinstance(seg_data, list):
                    segment_count = len(seg_data)
                elif isinstance(seg_data, dict):
                    segment_count = len(seg_data.get('segments', seg_data.get('mappings', [])))
            except Exception:
                pass
        
        # Try to find audio
        audio_file = None
        audio_dir = base / 'audio'
        if audio_dir.exists():
            for pattern in [f'voiceover_{bm_number}.mp3', f'benchmark_{bm_number}.mp3']:
                candidate = audio_dir / pattern
                if candidate.exists():
                    audio_file = str(candidate)
                    break
        
        benchmarks.append({
            'number': bm_number,
            'name': f'benchmark_{bm_number}',
            'video_dir': str(video_folder),
            'segments': str(segments_file) if segments_file.exists() else None,
            'ground_truth': str(gt_file) if gt_file.exists() else None,
            'audio': audio_file,
            'video_count': video_count,
            'segment_count': segment_count,
            'has_segments': segments_file.exists(),
            'has_ground_truth': gt_file.exists(),
        })
    
    return benchmarks


def select_benchmark(base_dir: str = './data/benchmarks') -> Optional[Dict]:
    """
    Display available benchmarks and let the user select one by number.
    
    Returns the selected benchmark dict or None if cancelled.
    """
    benchmarks = discover_benchmark_numbers(base_dir)
    
    if not benchmarks:
        print(f"  {Colors.YELLOW}No benchmarks found in {base_dir}/videos/{Colors.END}")
        print(f"  {Colors.DIM}Expected folder pattern: video_* (e.g., video_1, video_2, ...){Colors.END}")
        return None
    
    print(f"\n{Colors.BOLD}{Colors.YELLOW}  Available Benchmarks{Colors.END}")
    print(f"  {'─' * 56}")
    
    for bm in benchmarks:
        # Build status indicators
        vid_status = f"{Colors.GREEN}{bm['video_count']} clips ✓{Colors.END}"
        seg_status = (f"{Colors.GREEN}{bm['segment_count']} segments ✓{Colors.END}" 
                      if bm['has_segments'] 
                      else f"{Colors.RED}segments ✗{Colors.END}")
        gt_status = (f"{Colors.GREEN}ground truth ✓{Colors.END}" 
                     if bm['has_ground_truth'] 
                     else f"{Colors.RED}ground truth ✗{Colors.END}")
        audio_status = (f"{Colors.GREEN}audio ✓{Colors.END}" 
                        if bm['audio'] 
                        else f"{Colors.DIM}audio ✗{Colors.END}")
        
        print(f"  {Colors.GREEN}{bm['number']:>3}{Colors.END}. {Colors.BOLD}Benchmark {bm['number']}{Colors.END}")
        print(f"       {vid_status} | {seg_status} | {gt_status} | {audio_status}")
    
    print(f"  {Colors.RED}  0{Colors.END}. {Colors.BOLD}Back / Cancel{Colors.END}")
    print()
    
    # Get user selection
    valid_numbers = [bm['number'] for bm in benchmarks]
    while True:
        try:
            choice = input(f"{Colors.CYAN}  Select benchmark number: {Colors.END}").strip()
            if choice == '':
                continue
            if choice == '0':
                return None
            if choice in valid_numbers:
                return next(bm for bm in benchmarks if bm['number'] == choice)
            print(f"  {Colors.RED}Invalid selection. Available: {', '.join(valid_numbers)} (or 0 to cancel){Colors.END}")
        except (KeyboardInterrupt, EOFError):
            print()
            return None


# Keep old function for backward compatibility
def discover_benchmarks(base_dir: str = './data/benchmarks') -> List[Dict]:
    """Auto-discover available benchmarks (legacy compatibility)."""
    return discover_benchmark_numbers(base_dir)


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
    """Quick benchmark run with simplified benchmark selection."""
    clear_screen()
    print_header()
    print(f"  {Colors.BOLD}Quick Benchmark{Colors.END}")
    print(f"  {Colors.DIM}Run a benchmark test with auto-discovered settings{Colors.END}")
    
    # Select benchmark by number
    bm = select_benchmark()
    if bm is None:
        return
    
    if not bm['has_segments']:
        print(f"  {Colors.RED}Benchmark {bm['number']} has no segments file!{Colors.END}")
        input("\n  Press Enter to continue...")
        return
    
    print(f"\n  {Colors.GREEN}✓ Selected Benchmark {bm['number']}{Colors.END}")
    print(f"    Videos:       {bm['video_dir']} ({bm['video_count']} clips)")
    print(f"    Segments:     {bm['segments']} ({bm['segment_count']} segments)")
    if bm['ground_truth']:
        print(f"    Ground truth: {bm['ground_truth']}")
    if bm['audio']:
        print(f"    Audio:        {bm['audio']}")
    
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
        'no_windowing': True,
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
    
    # Use benchmark selector or manual path
    print_menu("Video Source", [
        ("Select from benchmarks", "Auto-discover benchmark videos"),
        ("Enter path manually", "Specify a custom video directory"),
    ], show_back=False)
    source_choice = get_choice(2, "  Select: ")
    
    if source_choice == 1:
        bm = select_benchmark()
        if bm is None:
            return
        config['video_dir'] = bm['video_dir']
        if bm['has_segments']:
            config['segments'] = bm['segments']
        if bm['ground_truth']:
            config['ground_truth'] = bm['ground_truth']
        if bm['audio']:
            config['audio'] = bm['audio']
    else:
        config['video_dir'] = browse_directory("Video directory", config.get('video_dir'))
    
    if not config.get('audio'):
        config['audio'] = browse_file("Audio file (voiceover)", config.get('audio'))
    config['output'] = get_input("Output directory", config.get('output', './output'))
    
    if not config.get('segments'):
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
    """OpenCLIP grid search optimizer with simplified benchmark selection."""
    clear_screen()
    print_header()
    print(f"  {Colors.BOLD}OpenCLIP Grid Search Optimizer{Colors.END}")
    print(f"  {Colors.DIM}Sweep parameters to find optimal OpenCLIP configuration{Colors.END}")
    
    # Select benchmark by number
    bm = select_benchmark()
    if bm is None:
        return
    
    if not bm['has_segments']:
        print(f"  {Colors.RED}Benchmark {bm['number']} has no segments file!{Colors.END}")
        input("\n  Press Enter to continue...")
        return
    
    if not bm['has_ground_truth']:
        print(f"  {Colors.RED}Benchmark {bm['number']} has no ground truth file!{Colors.END}")
        print(f"  {Colors.DIM}Ground truth is required for grid search evaluation.{Colors.END}")
        input("\n  Press Enter to continue...")
        return
    
    config['video_dir'] = bm['video_dir']
    config['segments'] = bm['segments']
    config['ground_truth'] = bm['ground_truth']
    
    print(f"\n  {Colors.GREEN}✓ Selected Benchmark {bm['number']}{Colors.END}")
    print(f"    Videos:       {bm['video_dir']} ({bm['video_count']} clips)")
    print(f"    Segments:     {bm['segments']} ({bm['segment_count']} segments)")
    print(f"    Ground truth: {bm['ground_truth']}")
    
    config['output'] = get_input("Output directory", config.get('output', './output'))
    
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


def screen_videoprism_grid_search(config: Dict):
    """VideoPrism grid search optimizer with simplified benchmark selection."""
    clear_screen()
    print_header()
    print(f"  {Colors.BOLD}VideoPrism Grid Search Optimizer{Colors.END}")
    print(f"  {Colors.DIM}Sweep parameters to find optimal VideoPrism configuration{Colors.END}")
    
    # Select benchmark by number
    bm = select_benchmark()
    if bm is None:
        return
    
    if not bm['has_segments']:
        print(f"  {Colors.RED}Benchmark {bm['number']} has no segments file!{Colors.END}")
        input("\n  Press Enter to continue...")
        return
    
    if not bm['has_ground_truth']:
        print(f"  {Colors.RED}Benchmark {bm['number']} has no ground truth file!{Colors.END}")
        print(f"  {Colors.DIM}Ground truth is required for grid search evaluation.{Colors.END}")
        input("\n  Press Enter to continue...")
        return
    
    config['video_dir'] = bm['video_dir']
    config['segments'] = bm['segments']
    config['ground_truth'] = bm['ground_truth']
    
    print(f"\n  {Colors.GREEN}✓ Selected Benchmark {bm['number']}{Colors.END}")
    print(f"    Videos:       {bm['video_dir']} ({bm['video_count']} clips)")
    print(f"    Segments:     {bm['segments']} ({bm['segment_count']} segments)")
    print(f"    Ground truth: {bm['ground_truth']}")
    
    config['output'] = get_input("Output directory", config.get('output', './output'))
    
    # Parameter selection
    print(f"\n  {Colors.BOLD}Select parameters to sweep:{Colors.END}")
    
    sweep_models = get_yes_no("Sweep both models (base + large)?", default=True)
    sweep_frames = get_yes_no("Sweep number of frames (8, 16, 32)?", default=True)
    sweep_prompts = get_yes_no("Sweep prompt templates (none, video, photo, scene, cooking)?", default=True)
    use_ensemble = get_yes_no("Include ensemble prompts (template-based + LLM-generated)?", default=True)
    
    llm_model = None
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
    
    gpu_device = get_input("GPU device", config.get('gpu_device', 'cuda:0'))
    
    # Build grid search command
    cmd = ['python', 'src/videoprism_grid_search.py']
    cmd.extend(['--video-dir', config['video_dir']])
    cmd.extend(['--segments', config['segments']])
    cmd.extend(['--ground-truth', config['ground_truth']])
    cmd.extend(['--output', config['output']])
    cmd.extend(['--device', gpu_device])
    cmd.append('--no-windowing')
    
    if sweep_models:
        cmd.extend(['--models', 'videoprism_lvt_public_v1_base', 'videoprism_lvt_public_v1_large'])
    else:
        cmd.extend(['--models', 'videoprism_lvt_public_v1_base'])
    
    if sweep_frames:
        cmd.extend(['--frames', '8', '16', '32'])
    else:
        cmd.extend(['--frames', '16'])
    
    # Build prompt modes list
    prompt_modes = []
    if sweep_prompts:
        prompt_modes.extend(['none', 'template:video', 'template:photo', 'template:scene', 'template:cooking'])
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
    
    # Calculate total configs
    n_models = 2 if sweep_models else 1
    n_frames = 3 if sweep_frames else 1
    n_prompts = len(prompt_modes)
    total = n_models * n_frames * n_prompts
    print(f"  {Colors.CYAN}Total configurations to test: {total}{Colors.END}\n")
    
    if not get_yes_no("Run VideoPrism grid search?", default=True):
        return
    
    print(f"\n{Colors.BOLD}{'═' * 60}{Colors.END}")
    print()
    
    try:
        process = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
        print(f"\n{Colors.BOLD}{'═' * 60}{Colors.END}")
        if process.returncode == 0:
            print(f"  {Colors.GREEN}✓ VideoPrism grid search completed successfully{Colors.END}")
        else:
            print(f"  {Colors.RED}✗ Grid search exited with code {process.returncode}{Colors.END}")
    except KeyboardInterrupt:
        print(f"\n  {Colors.YELLOW}⚠ Interrupted by user{Colors.END}")
    except Exception as e:
        print(f"  {Colors.RED}Error: {e}{Colors.END}")
    
    input("\n  Press Enter to continue...")


def screen_wav_grid_search(config: Dict):
    """Write-A-Video two-stage grid search optimizer."""
    clear_screen()
    print_header()
    print(f"  {Colors.BOLD}Write-A-Video (Two-Stage) Grid Search Optimizer{Colors.END}")
    print(f"  {Colors.DIM}Multi-modal keyword indexing + OpenCLIP reranking{Colors.END}")
    print(f"  {Colors.DIM}Based on Wang et al., 'Write-A-Video', TOG 2019{Colors.END}")
    
    # Select benchmark by number
    bm = select_benchmark()
    if bm is None:
        return
    
    if not bm['has_segments']:
        print(f"  {Colors.RED}Benchmark {bm['number']} has no segments file!{Colors.END}")
        input("\n  Press Enter to continue...")
        return
    
    if not bm['has_ground_truth']:
        print(f"  {Colors.RED}Benchmark {bm['number']} has no ground truth file!{Colors.END}")
        print(f"  {Colors.DIM}Ground truth is required for grid search evaluation.{Colors.END}")
        input("\n  Press Enter to continue...")
        return
    
    config['video_dir'] = bm['video_dir']
    config['segments'] = bm['segments']
    config['ground_truth'] = bm['ground_truth']
    
    print(f"\n  {Colors.GREEN}✓ Selected Benchmark {bm['number']}{Colors.END}")
    print(f"    Videos:       {bm['video_dir']} ({bm['video_count']} clips)")
    print(f"    Segments:     {bm['segments']} ({bm['segment_count']} segments)")
    print(f"    Ground truth: {bm['ground_truth']}")
    
    config['output'] = get_input("Output directory", config.get('output', './output'))
    
    # Stage 1: Keyword indexing parameters
    print(f"\n  {Colors.BOLD}Stage 1: Multi-Modal Keyword Indexing{Colors.END}")
    print(f"  {Colors.DIM}Object detection (YOLOv8) + Face recognition (DeepFace){Colors.END}")
    
    enable_objects = get_yes_no("Enable object detection (YOLOv8)?", default=True)
    enable_faces = get_yes_no("Enable face detection & clustering (DeepFace)?", default=True)
    
    print_menu("YOLO Model Size", [
        ("YOLOv8n (Nano)", "Fastest, good for benchmarking"),
        ("YOLOv8s (Small)", "Better accuracy, still fast"),
        ("YOLOv8m (Medium)", "Balanced speed/accuracy"),
        ("YOLOv8l (Large)", "Best accuracy, slowest"),
    ], show_back=False)
    yolo_choice = get_choice(4, "  Select YOLO model: ")
    yolo_models = {1: 'yolov8n', 2: 'yolov8s', 3: 'yolov8m', 4: 'yolov8l'}
    yolo_model = yolo_models.get(yolo_choice, 'yolov8n')
    
    fps = float(get_input("Frames per second to analyze", "1.0"))
    
    # Stage 2: OpenCLIP reranking parameters
    print(f"\n  {Colors.BOLD}Stage 2: OpenCLIP Reranking Parameters{Colors.END}")
    
    sweep_models = get_yes_no("Sweep OpenCLIP models (ViT-B-32, ViT-B-16, ViT-L-14)?", default=True)
    sweep_frames = get_yes_no("Sweep number of frames (4, 8, 16, 32)?", default=True)
    sweep_aggregation = get_yes_no("Sweep aggregation methods (mean, max, best_frame)?", default=True)
    sweep_prompts = get_yes_no("Sweep prompt templates?", default=True)
    use_ensemble = get_yes_no("Include ensemble prompts (LLM-generated)?", default=True)
    
    llm_model = None
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
    
    # WAV-specific: candidate pool size and keyword weight
    print(f"\n  {Colors.BOLD}Two-Stage Specific Parameters{Colors.END}")
    sweep_pool = get_yes_no("Sweep candidate pool sizes (5, 10, 20)?", default=False)
    sweep_kw_weight = get_yes_no("Sweep keyword weights (0.0, 0.1, 0.2, 0.3)?", default=False)
    
    gpu_device = get_input("GPU device", config.get('gpu_device', 'cuda:0'))
    
    # Build grid search command
    cmd = ['python', 'src/wav_grid_search.py']
    cmd.extend(['--video-dir', config['video_dir']])
    cmd.extend(['--segments', config['segments']])
    cmd.extend(['--ground-truth', config['ground_truth']])
    cmd.extend(['--output', config['output']])
    cmd.extend(['--device', gpu_device])
    cmd.append('--no-windowing')
    cmd.extend(['--yolo-model', yolo_model])
    cmd.extend(['--fps', str(fps)])
    
    if not enable_objects:
        cmd.append('--no-object-detection')
    if not enable_faces:
        cmd.append('--no-face-detection')
    
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
    
    if sweep_pool:
        cmd.extend(['--pool-sizes', '5', '10', '20'])
    
    if sweep_kw_weight:
        cmd.extend(['--keyword-weights', '0.0', '0.1', '0.2', '0.3'])
    
    cmd_str = ' \\\n    '.join(cmd)
    print(f"\n{Colors.BOLD}{Colors.BLUE}  Command:{Colors.END}")
    print(f"  {Colors.DIM}{cmd_str}{Colors.END}\n")
    
    if not get_yes_no("Run Write-A-Video grid search?", default=True):
        return
    
    print(f"\n{Colors.BOLD}{'═' * 60}{Colors.END}")
    print()
    
    try:
        process = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
        print(f"\n{Colors.BOLD}{'═' * 60}{Colors.END}")
        if process.returncode == 0:
            print(f"  {Colors.GREEN}✓ Write-A-Video grid search completed successfully{Colors.END}")
        else:
            print(f"  {Colors.RED}✗ Grid search exited with code {process.returncode}{Colors.END}")
    except KeyboardInterrupt:
        print(f"\n  {Colors.YELLOW}⚠ Interrupted by user{Colors.END}")
    except Exception as e:
        print(f"  {Colors.RED}Error: {e}{Colors.END}")
    
    input("\n  Press Enter to continue...")


def screen_benchmark_upload(config: Dict):
    """Upload a benchmark from a local folder."""
    import shutil
    import socket

    clear_screen()
    print_header()
    print(f"  {Colors.BOLD}Upload Benchmark{Colors.END}")
    print(f"  {Colors.DIM}Import a local folder containing videos + JSON files{Colors.END}")
    print(f"  {Colors.DIM}Expected folder structure (flat):{Colors.END}")
    print(f"  {Colors.DIM}  my_benchmark/{Colors.END}")
    print(f"  {Colors.DIM}    ├── segments.json{Colors.END}")
    print(f"  {Colors.DIM}    ├── ground_truth.json{Colors.END}")
    print(f"  {Colors.DIM}    ├── clip_001.mp4{Colors.END}")
    print(f"  {Colors.DIM}    ├── clip_002.mp4{Colors.END}")
    print(f"  {Colors.DIM}    └── ...{Colors.END}")

    base_dir = Path(config.get('benchmarks_dir', './data/benchmarks'))

    # Determine next available benchmark number
    existing = discover_benchmark_numbers(str(base_dir))
    existing_nums = [int(bm['number']) for bm in existing] if existing else []
    next_num = max(existing_nums) + 1 if existing_nums else 1

    if existing:
        print(f"\n  {Colors.DIM}Existing benchmarks: {', '.join(str(n) for n in sorted(existing_nums))}{Colors.END}")
    print(f"  {Colors.GREEN}Next available benchmark number: {next_num}{Colors.END}")

    # Let user override the number
    num_input = get_input("Benchmark number", str(next_num))
    try:
        bm_num = int(num_input)
    except ValueError:
        print(f"  {Colors.RED}Invalid number{Colors.END}")
        input("\n  Press Enter to continue...")
        return

    if bm_num in existing_nums:
        if not get_yes_no(f"Benchmark {bm_num} already exists. Overwrite?", default=False):
            input("\n  Press Enter to continue...")
            return

    # Get benchmark title
    bm_title = get_input("Benchmark title/description", f"Benchmark {bm_num}")

    print(f"\n  {Colors.BOLD}How to upload:{Colors.END}")
    print(f"  {Colors.DIM}Drag and drop your local folder into the terminal below.{Colors.END}")
    print(f"  {Colors.DIM}(This pastes the full path to the folder on your Mac){Colors.END}")

    local_path = get_input("Local folder path (drag & drop here)").strip()

    # Clean up path: remove trailing spaces, quotes, and escaped spaces
    local_path = local_path.strip("'\"")
    # macOS terminal sometimes escapes spaces with backslashes
    local_path = local_path.replace('\\ ', ' ')

    if not local_path:
        print(f"  {Colors.RED}No path provided{Colors.END}")
        input("\n  Press Enter to continue...")
        return

    # Create staging directory on the server
    staging_dir = Path(f'/tmp/benchmark_upload_{bm_num}')
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Get server hostname and username for scp command
    hostname = socket.gethostname()
    username = os.environ.get('USER', os.environ.get('LOGNAME', 'user'))
    home_dir = Path.home()

    # Generate scp command for the user
    print(f"\n  {Colors.BOLD}{Colors.BLUE}Run this command in a NEW local terminal on your Mac:{Colors.END}")
    print(f"  {Colors.DIM}{'─' * 56}{Colors.END}")
    scp_target = staging_dir
    print(f"\n  {Colors.CYAN}scp -r \"{local_path}/\"* {username}@{hostname}:{scp_target}/{Colors.END}")
    print(f"\n  {Colors.DIM}{'─' * 56}{Colors.END}")
    print(f"  {Colors.DIM}This will upload all files from your local folder to the server.{Colors.END}")
    print(f"  {Colors.DIM}After the upload finishes, come back here and press Enter.{Colors.END}")

    input(f"\n  {Colors.YELLOW}Press Enter AFTER the scp upload is complete...{Colors.END}")

    # Check what was uploaded
    uploaded_files = list(staging_dir.iterdir()) if staging_dir.exists() else []
    if not uploaded_files:
        print(f"  {Colors.RED}No files found in staging directory: {staging_dir}{Colors.END}")
        print(f"  {Colors.DIM}Make sure the scp command completed successfully.{Colors.END}")
        input("\n  Press Enter to continue...")
        return

    # Classify uploaded files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    json_files = [f for f in uploaded_files if f.suffix.lower() == '.json']
    video_files = [f for f in uploaded_files if f.suffix.lower() in video_extensions]

    print(f"\n  {Colors.GREEN}Found {len(video_files)} video files and {len(json_files)} JSON files{Colors.END}")

    if not video_files:
        print(f"  {Colors.RED}No video files found! Aborting.{Colors.END}")
        input("\n  Press Enter to continue...")
        return

    # Identify segments and ground truth JSON files
    segments_file = None
    gt_file = None

    for jf in json_files:
        name_lower = jf.name.lower()
        if 'segment' in name_lower:
            segments_file = jf
        elif 'ground' in name_lower or 'truth' in name_lower or 'gdtruth' in name_lower or 'gt' in name_lower:
            gt_file = jf

    # If we couldn't auto-detect, ask the user
    if len(json_files) >= 2 and (segments_file is None or gt_file is None):
        print(f"\n  {Colors.YELLOW}Could not auto-detect JSON file roles. Please identify them:{Colors.END}")
        for i, jf in enumerate(json_files, 1):
            print(f"    {i}. {jf.name}")
        if segments_file is None:
            seg_choice = get_choice(len(json_files), "  Which file is the SEGMENTS file? ")
            if seg_choice > 0:
                segments_file = json_files[seg_choice - 1]
        if gt_file is None:
            gt_choice = get_choice(len(json_files), "  Which file is the GROUND TRUTH file? ")
            if gt_choice > 0:
                gt_file = json_files[gt_choice - 1]
    elif len(json_files) == 1:
        # Only one JSON — ask what it is
        print(f"\n  {Colors.YELLOW}Found 1 JSON file: {json_files[0].name}{Colors.END}")
        print(f"    1. Segments file")
        print(f"    2. Ground truth file")
        jtype = get_choice(2, "  What is this file? ")
        if jtype == 1:
            segments_file = json_files[0]
        elif jtype == 2:
            gt_file = json_files[0]

    # Show summary before organizing
    print(f"\n  {Colors.BOLD}Benchmark {bm_num} Summary:{Colors.END}")
    print(f"    Title:        {bm_title}")
    print(f"    Videos:       {len(video_files)} clips")
    print(f"    Segments:     {segments_file.name if segments_file else Colors.RED + 'None' + Colors.END}")
    print(f"    Ground truth: {gt_file.name if gt_file else Colors.RED + 'None' + Colors.END}")

    if not get_yes_no("Proceed with organizing files?", default=True):
        # Cleanup staging
        shutil.rmtree(staging_dir, ignore_errors=True)
        input("\n  Press Enter to continue...")
        return

    # Create benchmark directory structure
    video_dir = base_dir / 'videos' / f'video_{bm_num}'
    segments_dir = base_dir / 'segments'
    gt_dir = base_dir / 'gdtruth'

    video_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Move video files
    print(f"\n  Moving video files...")
    for vf in video_files:
        dest = video_dir / vf.name
        shutil.move(str(vf), str(dest))
    print(f"  {Colors.GREEN}✓ Moved {len(video_files)} videos to {video_dir}{Colors.END}")

    # Move segments file
    if segments_file:
        dest = segments_dir / f'benchmark_{bm_num}_segments.json'
        shutil.move(str(segments_file), str(dest))
        print(f"  {Colors.GREEN}✓ Segments saved as {dest.name}{Colors.END}")

    # Move ground truth file
    if gt_file:
        dest = gt_dir / f'benchmark_{bm_num}_ground_truth.json'
        shutil.move(str(gt_file), str(dest))
        print(f"  {Colors.GREEN}✓ Ground truth saved as {dest.name}{Colors.END}")

    # Save benchmark metadata
    meta_dir = base_dir / 'metadata'
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_file = meta_dir / f'benchmark_{bm_num}_meta.json'
    meta = {
        'number': bm_num,
        'title': bm_title,
        'video_count': len(video_files),
        'has_segments': segments_file is not None,
        'has_ground_truth': gt_file is not None,
        'created': str(Path(sys.argv[0]).parent),
    }
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    # Cleanup staging directory
    shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"\n  {Colors.GREEN}{Colors.BOLD}✓ Benchmark {bm_num} '{bm_title}' created successfully!{Colors.END}")
    print(f"  {Colors.DIM}You can now run it with: python src/grid_search.py -b {bm_num}{Colors.END}")

    input("\n  Press Enter to continue...")


def screen_benchmark_download(config: Dict):
    """Download/export a benchmark to a single folder."""
    import shutil
    import socket

    clear_screen()
    print_header()
    print(f"  {Colors.BOLD}Download Benchmark{Colors.END}")
    print(f"  {Colors.DIM}Package a benchmark into a single folder for download{Colors.END}")

    base_dir = Path(config.get('benchmarks_dir', './data/benchmarks'))

    # Select benchmark
    bm = select_benchmark(str(base_dir))
    if bm is None:
        return

    bm_num = bm['number']

    # Create export directory
    export_dir = Path(f'/tmp/benchmark_export_{bm_num}')
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  {Colors.DIM}Packaging benchmark {bm_num}...{Colors.END}")

    # Copy video files
    video_dir = Path(bm['video_dir'])
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    video_count = 0
    if video_dir.exists():
        for vf in video_dir.iterdir():
            if vf.suffix.lower() in video_extensions:
                shutil.copy2(str(vf), str(export_dir / vf.name))
                video_count += 1
    print(f"  {Colors.GREEN}✓ Copied {video_count} video files{Colors.END}")

    # Copy segments file
    if bm['has_segments'] and bm['segments']:
        seg_src = Path(bm['segments'])
        shutil.copy2(str(seg_src), str(export_dir / 'segments.json'))
        print(f"  {Colors.GREEN}✓ Copied segments.json{Colors.END}")

    # Copy ground truth file
    if bm['has_ground_truth'] and bm['ground_truth']:
        gt_src = Path(bm['ground_truth'])
        shutil.copy2(str(gt_src), str(export_dir / 'ground_truth.json'))
        print(f"  {Colors.GREEN}✓ Copied ground_truth.json{Colors.END}")

    # Copy metadata if exists
    meta_file = base_dir / 'metadata' / f'benchmark_{bm_num}_meta.json'
    if meta_file.exists():
        shutil.copy2(str(meta_file), str(export_dir / 'metadata.json'))
        print(f"  {Colors.GREEN}✓ Copied metadata.json{Colors.END}")

    # Count total files and size
    total_files = list(export_dir.iterdir())
    total_size = sum(f.stat().st_size for f in total_files if f.is_file())

    print(f"\n  {Colors.BOLD}Export ready:{Colors.END}")
    print(f"    Location: {export_dir}")
    print(f"    Files:    {len(total_files)}")
    print(f"    Size:     {total_size / 1024 / 1024:.1f} MB")

    # Generate scp command for download
    hostname = socket.gethostname()
    username = os.environ.get('USER', os.environ.get('LOGNAME', 'user'))

    print(f"\n  {Colors.BOLD}{Colors.BLUE}Run this command in a local terminal on your Mac to download:{Colors.END}")
    print(f"  {Colors.DIM}{'─' * 56}{Colors.END}")
    print(f"\n  {Colors.CYAN}scp -r {username}@{hostname}:{export_dir}/ ~/Downloads/benchmark_{bm_num}/{Colors.END}")
    print(f"\n  {Colors.DIM}{'─' * 56}{Colors.END}")
    print(f"  {Colors.DIM}This will download the benchmark to ~/Downloads/benchmark_{bm_num}/ on your Mac.{Colors.END}")
    print(f"  {Colors.DIM}Change the destination path as needed.{Colors.END}")

    print(f"\n  {Colors.DIM}The export folder will remain at {export_dir} until you exit.{Colors.END}")

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
            total_size = 0
            for item in sorted(items):
                if item.is_dir():
                    size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    total_size += size
                    print(f"    📁 {item.name} ({size / 1024:.1f} KB)")
                else:
                    size = item.stat().st_size
                    total_size += size
                    print(f"    📄 {item.name} ({size / 1024:.1f} KB)")
            print(f"\n  {Colors.DIM}Total cache size: {total_size / 1024 / 1024:.1f} MB{Colors.END}")
        else:
            print(f"  {Colors.DIM}Cache is empty{Colors.END}")
    else:
        print(f"  {Colors.DIM}Cache directory does not exist{Colors.END}")
    
    print_menu("Cache Actions", [
        ("Clear VideoPrism index cache", "Remove cached video_index/"),
        ("Clear OpenCLIP index cache", "Remove cached video_index_openclip/"),
        ("Clear OpenCLIP grid search cache", "Remove cached gs_* indices"),
        ("Clear VideoPrism grid search cache", "Remove cached vp_* indices"),
        ("Clear Write-A-Video keyword cache", "Remove cached wav_kw_* indices"),
        ("Clear ALL caches", "Remove entire cache directory"),
    ])
    
    choice = get_choice(6)
    if choice == 0:
        return
    
    if choice == 1:
        target = cache_dir / 'video_index'
    elif choice == 2:
        target = cache_dir / 'video_index_openclip'
    elif choice in (3, 4, 5):
        # Clear grid search directories by prefix
        import shutil
        prefix_map = {3: ('gs_', 'OpenCLIP'), 4: ('vp_', 'VideoPrism'), 5: ('wav_kw_', 'Write-A-Video')}
        prefix, label = prefix_map[choice]
        cleared = 0
        if cache_dir.exists():
            for item in cache_dir.iterdir():
                if item.is_dir() and item.name.startswith(prefix):
                    shutil.rmtree(item)
                    cleared += 1
        if cleared:
            print(f"  {Colors.GREEN}✓ Cleared {cleared} {label} grid search cache directories{Colors.END}")
        else:
            print(f"  {Colors.DIM}No {label} grid search caches found{Colors.END}")
        input("\n  Press Enter to continue...")
        return
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
            ("VideoPrism Grid Search", "Find optimal VideoPrism parameters via automated sweep"),
            ("Write-A-Video Grid Search", "Two-stage retrieval: keyword filtering + OpenCLIP reranking"),
            ("Upload Benchmark", "Import a local folder as a new benchmark (via scp)"),
            ("Download Benchmark", "Export a benchmark to a single folder for download"),
            ("Cache Management", "View and clear cached indexes"),
            ("Settings", "Configure GPU, output directory, etc."),
        ])
        
        choice = get_choice(9)
        
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
            screen_videoprism_grid_search(config)
        elif choice == 5:
            screen_wav_grid_search(config)
        elif choice == 6:
            screen_benchmark_upload(config)
        elif choice == 7:
            screen_benchmark_download(config)
        elif choice == 8:
            screen_cache_management(config)
        elif choice == 9:
            screen_settings(config)


if __name__ == '__main__':
    try:
        main_menu()
    except (KeyboardInterrupt, EOFError):
        print(f"\n\n  {Colors.DIM}Goodbye!{Colors.END}\n")
