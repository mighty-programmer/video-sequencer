"""
OpenCLIP Grid Search Optimizer

Systematically tests all parameter combinations for the OpenCLIP pipeline
against a benchmark with ground truth, and reports the best configuration.

Parameters searched:
- model_name: ViT-B-32, ViT-B-16, ViT-L-14
- num_frames: 4, 8, 16, 32
- aggregation: mean, max, best_frame
- prompt_template: none, video, photo, scene, cooking
- ensemble: off, template-based, LLM-generated

Usage:
    python src/grid_search.py \
        --video-dir data/benchmarks/videos/video_2/ \
        --segments data/benchmarks/segments/benchmark_2_segments.json \
        --ground-truth data/benchmarks/gdtruth/benchmark_2_ground_truth.json \
        --output output/grid_search/
"""

import os
import sys
import json
import time
import logging
import itertools
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from openclip_indexing import (
    OpenCLIPVideoIndexer, OpenCLIPTextMatcher,
    PROMPT_TEMPLATES, DEFAULT_ENSEMBLE_TEMPLATES
)
from matching import create_sequence, ClipSelection
from benchmark import BenchmarkEvaluator, BenchmarkResults
from prompt_generator import create_ensemble_templates

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')


@dataclass
class GridSearchConfig:
    """A single parameter configuration to test."""
    model_name: str
    num_frames: int
    aggregation: str
    prompt_mode: str  # 'none', 'template:<name>', 'ensemble:template', 'ensemble:llm'
    prompt_template: Optional[str] = None
    ensemble_prompts: Optional[List[str]] = None


@dataclass
class GridSearchResult:
    """Result of a single grid search configuration."""
    config: Dict
    exact_match_accuracy: float
    top_3_accuracy: float
    top_5_accuracy: float
    mrr: float
    avg_similarity: float
    indexing_time: float
    matching_time: float
    total_time: float


class OpenCLIPGridSearch:
    """
    Grid search optimizer for OpenCLIP pipeline parameters.
    
    Tests all combinations of tunable parameters against a benchmark
    and reports the best configuration.
    """
    
    def __init__(
        self,
        video_dir: str,
        segments_file: str,
        ground_truth_file: str,
        output_dir: str = './output/grid_search',
        cache_base_dir: str = './cache',
        device: str = 'cuda:0',
        use_windowing: bool = False,
        window_size: float = 5.0,
        window_overlap: float = 0.5
    ):
        """
        Initialize the grid search optimizer.
        
        Args:
            video_dir: Directory containing video files
            segments_file: Path to segments JSON file
            ground_truth_file: Path to ground truth JSON file
            output_dir: Directory for grid search results
            cache_base_dir: Base directory for caching indices
            device: GPU device to use
            use_windowing: Whether to use temporal windowing
            window_size: Window size in seconds
            window_overlap: Window overlap fraction
        """
        self.video_dir = video_dir
        self.segments_file = segments_file
        self.ground_truth_file = ground_truth_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_base_dir = Path(cache_base_dir)
        self.device = device
        self.use_windowing = use_windowing
        self.window_size = window_size
        self.window_overlap = window_overlap
        
        # Load segments
        with open(segments_file, 'r') as f:
            segments_data = json.load(f)
        
        if isinstance(segments_data, list):
            self.segments = segments_data
        elif isinstance(segments_data, dict):
            self.segments = segments_data.get('segments', segments_data.get('mappings', []))
        
        # Ensure segments have required fields
        for i, seg in enumerate(self.segments):
            if 'text' not in seg:
                seg['text'] = seg.get('segment_text', f'segment_{i}')
            if 'duration' not in seg:
                seg['duration'] = seg.get('end_time', 0) - seg.get('start_time', 0)
        
        # Initialize benchmark evaluator
        self.evaluator = BenchmarkEvaluator(ground_truth_file)
        
        # Results storage
        self.results: List[GridSearchResult] = []
        
        logger.info(f"Grid Search initialized:")
        logger.info(f"  Video dir: {video_dir}")
        logger.info(f"  Segments: {len(self.segments)}")
        logger.info(f"  Ground truth: {ground_truth_file}")
        logger.info(f"  Device: {device}")
    
    def _generate_configs(
        self,
        models: List[str] = None,
        num_frames_list: List[int] = None,
        aggregations: List[str] = None,
        prompt_modes: List[str] = None,
        llm_model: str = None
    ) -> List[GridSearchConfig]:
        """
        Generate all parameter configurations to test.
        
        Args:
            models: List of model names to test
            num_frames_list: List of frame counts to test
            aggregations: List of aggregation methods to test
            prompt_modes: List of prompt modes to test
            llm_model: Ollama model name for LLM ensemble (None to skip)
            
        Returns:
            List of GridSearchConfig objects
        """
        if models is None:
            models = ['ViT-B-32', 'ViT-B-16', 'ViT-L-14']
        if num_frames_list is None:
            num_frames_list = [4, 8, 16, 32]
        if aggregations is None:
            aggregations = ['mean', 'max', 'best_frame']
        if prompt_modes is None:
            prompt_modes = ['none', 'template:video', 'template:photo',
                          'template:cooking', 'ensemble:template']
            if llm_model:
                prompt_modes.append('ensemble:llm')
        
        configs = []
        
        for model, n_frames, agg, prompt_mode in itertools.product(
            models, num_frames_list, aggregations, prompt_modes
        ):
            config = GridSearchConfig(
                model_name=model,
                num_frames=n_frames,
                aggregation=agg,
                prompt_mode=prompt_mode
            )
            
            # Set prompt template or ensemble prompts based on mode
            if prompt_mode.startswith('template:'):
                template_name = prompt_mode.split(':')[1]
                config.prompt_template = PROMPT_TEMPLATES.get(template_name, '{text}')
            elif prompt_mode == 'ensemble:template':
                config.ensemble_prompts = DEFAULT_ENSEMBLE_TEMPLATES
            elif prompt_mode == 'ensemble:llm':
                # Will be populated during run
                config.prompt_template = None
                config.ensemble_prompts = None  # Populated per-segment
            
            configs.append(config)
        
        logger.info(f"Generated {len(configs)} configurations to test")
        return configs
    
    def _get_cache_dir(self, config: GridSearchConfig) -> str:
        """Get a unique cache directory for a configuration."""
        cache_name = (
            f"gs_{config.model_name}_{config.num_frames}f_{config.aggregation}"
        )
        return str(self.cache_base_dir / cache_name)
    
    def _run_single_config(
        self,
        config: GridSearchConfig,
        llm_prompts: Optional[Dict[str, List[str]]] = None
    ) -> Optional[GridSearchResult]:
        """
        Run the pipeline with a single configuration and evaluate.
        
        Args:
            config: GridSearchConfig to test
            llm_prompts: Pre-generated LLM prompts (for ensemble:llm mode)
            
        Returns:
            GridSearchResult or None if failed
        """
        config_desc = (
            f"{config.model_name} | {config.num_frames}f | {config.aggregation} | {config.prompt_mode}"
        )
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {config_desc}")
        logger.info(f"{'='*60}")
        
        try:
            t_start = time.time()
            
            # 1. Create indexer and index videos
            cache_dir = self._get_cache_dir(config)
            
            indexer = OpenCLIPVideoIndexer(
                model_name=config.model_name,
                index_dir=cache_dir,
                device=self.device,
                num_frames=config.num_frames,
                aggregation=config.aggregation
            )
            
            # Try to load cached index
            t_index_start = time.time()
            if not indexer.load_index():
                indexer.index_videos(
                    self.video_dir,
                    use_windowing=self.use_windowing,
                    window_size=self.window_size,
                    window_overlap=self.window_overlap
                )
            indexing_time = time.time() - t_index_start
            
            # 2. Create matcher with prompt configuration
            ensemble_prompts = config.ensemble_prompts
            
            # Handle LLM ensemble mode
            if config.prompt_mode == 'ensemble:llm' and llm_prompts:
                # Convert per-segment prompts to per-query ensemble
                # The matcher will use these as ensemble prompts for each query
                ensemble_prompts = None  # We'll handle this differently
                # Create a custom matcher that uses per-segment prompts
                matcher = _LLMEnsembleMatcher(
                    video_indexer=indexer,
                    llm_prompts=llm_prompts,
                    min_similarity_threshold=0.0
                )
            else:
                matcher = OpenCLIPTextMatcher(
                    video_indexer=indexer,
                    min_similarity_threshold=0.0,
                    prompt_template=config.prompt_template,
                    ensemble_prompts=ensemble_prompts
                )
            
            # 3. Run matching
            t_match_start = time.time()
            clip_selections = create_sequence(
                self.segments, matcher,
                match_only=True, allow_reuse=False, use_optimal=True
            )
            matching_time = time.time() - t_match_start
            
            if not clip_selections:
                logger.error(f"No clip selections produced for {config_desc}")
                return None
            
            # 4. Compute similarity matrix for evaluation
            similarity_matrix, all_metadata = matcher.compute_similarity_matrix(
                self.segments, match_only=True
            )
            
            # 5. Evaluate against ground truth
            benchmark_results = self.evaluator.evaluate(
                clip_selections=clip_selections,
                segment_dicts=self.segments,
                similarity_matrix=similarity_matrix,
                all_metadata=all_metadata,
                matching_mode='optimal',
                allow_reuse=False,
                match_only=True
            )
            
            total_time = time.time() - t_start
            
            result = GridSearchResult(
                config=asdict(config),
                exact_match_accuracy=benchmark_results.exact_match_accuracy,
                top_3_accuracy=benchmark_results.top_3_accuracy,
                top_5_accuracy=benchmark_results.top_5_accuracy,
                mrr=benchmark_results.mean_reciprocal_rank,
                avg_similarity=benchmark_results.avg_predicted_similarity,
                indexing_time=round(indexing_time, 2),
                matching_time=round(matching_time, 2),
                total_time=round(total_time, 2)
            )
            
            logger.info(f"  Result: Exact={result.exact_match_accuracy:.1f}% | "
                        f"Top-3={result.top_3_accuracy:.1f}% | "
                        f"Top-5={result.top_5_accuracy:.1f}% | "
                        f"MRR={result.mrr:.4f} | "
                        f"Time={result.total_time:.1f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing {config_desc}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run(
        self,
        models: List[str] = None,
        num_frames_list: List[int] = None,
        aggregations: List[str] = None,
        prompt_modes: List[str] = None,
        llm_model: str = None,
        quick: bool = False
    ) -> List[GridSearchResult]:
        """
        Run the full grid search.
        
        Args:
            models: Models to test (default: all)
            num_frames_list: Frame counts to test (default: [4, 8, 16, 32])
            aggregations: Aggregation methods to test (default: all)
            prompt_modes: Prompt modes to test (default: all)
            llm_model: Ollama model for LLM ensemble (None to skip)
            quick: If True, use a reduced parameter grid for faster testing
            
        Returns:
            List of GridSearchResult sorted by exact match accuracy
        """
        if quick:
            if models is None:
                models = ['ViT-B-32']
            if num_frames_list is None:
                num_frames_list = [8, 16]
            if aggregations is None:
                aggregations = ['mean', 'best_frame']
            if prompt_modes is None:
                prompt_modes = ['none', 'template:video', 'ensemble:template']
        
        # Generate all configurations
        configs = self._generate_configs(
            models=models,
            num_frames_list=num_frames_list,
            aggregations=aggregations,
            prompt_modes=prompt_modes,
            llm_model=llm_model
        )
        
        # Pre-generate LLM prompts if needed (do this once, not per config)
        llm_prompts = None
        if llm_model and any(c.prompt_mode == 'ensemble:llm' for c in configs):
            logger.info(f"\nPre-generating LLM prompts using {llm_model}...")
            cache_file = str(self.output_dir / 'llm_prompts_cache.json')
            llm_prompts = create_ensemble_templates(
                self.segments,
                llm_model=llm_model,
                num_variations=5,
                cache_file=cache_file
            )
            logger.info(f"Generated prompts for {len(llm_prompts)} segments")
        
        # Run each configuration
        total = len(configs)
        logger.info(f"\n{'#'*60}")
        logger.info(f"STARTING GRID SEARCH: {total} configurations")
        logger.info(f"{'#'*60}\n")
        
        start_time = time.time()
        
        for i, config in enumerate(configs):
            logger.info(f"\n[{i+1}/{total}] Running configuration...")
            result = self._run_single_config(config, llm_prompts=llm_prompts)
            if result:
                self.results.append(result)
        
        elapsed = time.time() - start_time
        
        # Sort results by exact match accuracy (primary), then MRR (secondary)
        self.results.sort(
            key=lambda r: (r.exact_match_accuracy, r.mrr),
            reverse=True
        )
        
        # Save results
        self._save_results(elapsed)
        
        # Print summary
        self._print_summary(elapsed)
        
        return self.results
    
    def _save_results(self, elapsed: float):
        """Save grid search results to JSON."""
        output_path = self.output_dir / 'grid_search_results.json'
        
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'total_configs_tested': len(self.results),
            'total_time_seconds': round(elapsed, 2),
            'video_dir': self.video_dir,
            'segments_file': self.segments_file,
            'ground_truth_file': self.ground_truth_file,
            'results': [asdict(r) for r in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Saved grid search results to {output_path}")
    
    def _print_summary(self, elapsed: float):
        """Print a beautiful summary of grid search results."""
        print("\n")
        print("╔" + "═" * 98 + "╗")
        print("║" + " " * 30 + "GRID SEARCH RESULTS SUMMARY" + " " * 41 + "║")
        print("╠" + "═" * 98 + "╣")
        print(f"║  Configurations tested: {len(self.results):<10}  "
              f"Total time: {elapsed:.1f}s" + " " * (98 - 50 - len(f"{elapsed:.1f}")) + "║")
        print("╠" + "═" * 98 + "╣")
        
        # Header
        print("║  RANK │ MODEL      │ FRAMES │ AGG        │ PROMPT MODE        │ EXACT% │ TOP3%  │ MRR    ║")
        print("╟" + "─" * 98 + "╢")
        
        # Top results (show all, max 30)
        for i, result in enumerate(self.results[:30]):
            cfg = result.config
            model = cfg['model_name'][:10]
            frames = str(cfg['num_frames'])
            agg = cfg['aggregation'][:10]
            prompt = cfg['prompt_mode'][:18]
            
            marker = " ★" if i == 0 else "  "
            
            print(f"║{marker}{i+1:3d}  │ {model:<10} │ {frames:>6} │ {agg:<10} │ {prompt:<18} │ "
                  f"{result.exact_match_accuracy:5.1f}% │ {result.top_3_accuracy:5.1f}% │ {result.mrr:.4f} ║")
        
        if len(self.results) > 30:
            print(f"║  ... and {len(self.results) - 30} more configurations" + " " * (98 - 40) + "║")
        
        print("╠" + "═" * 98 + "╣")
        
        # Best configuration
        if self.results:
            best = self.results[0]
            cfg = best.config
            print("║" + " " * 35 + "★ BEST CONFIGURATION ★" + " " * 41 + "║")
            print("╟" + "─" * 98 + "╢")
            print(f"║  Model:       {cfg['model_name']:<82} ║")
            print(f"║  Frames:      {cfg['num_frames']:<82} ║")
            print(f"║  Aggregation: {cfg['aggregation']:<82} ║")
            print(f"║  Prompt Mode: {cfg['prompt_mode']:<82} ║")
            print("╟" + "─" * 98 + "╢")
            print(f"║  Exact Match: {best.exact_match_accuracy:.1f}%   │   "
                  f"Top-3: {best.top_3_accuracy:.1f}%   │   "
                  f"Top-5: {best.top_5_accuracy:.1f}%   │   "
                  f"MRR: {best.mrr:.4f}" + " " * 20 + "║")
            
            # Generate CLI command for best config
            print("╟" + "─" * 98 + "╢")
            print("║  Run with best config:" + " " * 75 + "║")
            
            cmd_parts = [
                "python src/main.py",
                f"--encoder openclip",
                f"--openclip-model {cfg['model_name']}",
            ]
            cmd = "  ".join(cmd_parts)
            print(f"║    {cmd:<94} ║")
        
        print("╚" + "═" * 98 + "╝")
        print()


class _LLMEnsembleMatcher(OpenCLIPTextMatcher):
    """
    Special matcher that uses per-segment LLM-generated prompts.
    
    Instead of using the same ensemble templates for all segments,
    this uses unique LLM-generated variations per segment text.
    """
    
    def __init__(
        self,
        video_indexer: OpenCLIPVideoIndexer,
        llm_prompts: Dict[str, List[str]],
        min_similarity_threshold: float = 0.0
    ):
        super().__init__(
            video_indexer=video_indexer,
            min_similarity_threshold=min_similarity_threshold,
            prompt_template=None,
            ensemble_prompts=None
        )
        self.llm_prompts = llm_prompts
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding using LLM-generated prompts specific to this text.
        """
        prompts = self.llm_prompts.get(text, [text])
        
        if len(prompts) <= 1:
            return self._encode_single_text(text)
        
        all_embeddings = []
        for prompt in prompts:
            emb = self._encode_single_text(prompt)
            all_embeddings.append(emb)
        
        avg_embedding = np.mean(all_embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        return avg_embedding.astype(np.float32)


def main():
    """CLI entry point for grid search."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='OpenCLIP Grid Search Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full grid search (all combinations)
  python src/grid_search.py \\
    --video-dir data/benchmarks/videos/video_2/ \\
    --segments data/benchmarks/segments/benchmark_2_segments.json \\
    --ground-truth data/benchmarks/gdtruth/benchmark_2_ground_truth.json

  # Quick search (reduced grid)
  python src/grid_search.py \\
    --video-dir data/benchmarks/videos/video_2/ \\
    --segments data/benchmarks/segments/benchmark_2_segments.json \\
    --ground-truth data/benchmarks/gdtruth/benchmark_2_ground_truth.json \\
    --quick

  # With LLM ensemble prompts
  python src/grid_search.py \\
    --video-dir data/benchmarks/videos/video_2/ \\
    --segments data/benchmarks/segments/benchmark_2_segments.json \\
    --ground-truth data/benchmarks/gdtruth/benchmark_2_ground_truth.json \\
    --llm-model llama3.2:3b

  # Custom parameter grid
  python src/grid_search.py \\
    --video-dir data/benchmarks/videos/video_2/ \\
    --segments data/benchmarks/segments/benchmark_2_segments.json \\
    --ground-truth data/benchmarks/gdtruth/benchmark_2_ground_truth.json \\
    --models ViT-B-32 ViT-L-14 \\
    --frames 8 16 \\
    --aggregations mean best_frame
        """
    )
    
    parser.add_argument('--video-dir', required=True, help='Directory containing video files')
    parser.add_argument('--segments', required=True, help='Path to segments JSON file')
    parser.add_argument('--ground-truth', required=True, help='Path to ground truth JSON file')
    parser.add_argument('--output', default='./output/grid_search', help='Output directory')
    parser.add_argument('--cache-dir', default='./cache', help='Cache directory')
    parser.add_argument('--device', default='cuda:0', help='GPU device')
    parser.add_argument('--quick', action='store_true', help='Use reduced parameter grid')
    parser.add_argument('--llm-model', default=None, help='Ollama model for LLM ensemble prompts')
    
    # Custom grid parameters
    parser.add_argument('--models', nargs='+', default=None,
                       choices=['ViT-B-32', 'ViT-B-16', 'ViT-L-14'],
                       help='Models to test')
    parser.add_argument('--frames', nargs='+', type=int, default=None,
                       help='Frame counts to test')
    parser.add_argument('--aggregations', nargs='+', default=None,
                       choices=['mean', 'max', 'best_frame'],
                       help='Aggregation methods to test')
    parser.add_argument('--prompt-modes', nargs='+', default=None,
                       help='Prompt modes to test')
    
    # Windowing
    parser.add_argument('--no-windowing', action='store_true', help='Disable windowing')
    parser.add_argument('--window-size', type=float, default=5.0, help='Window size')
    parser.add_argument('--window-overlap', type=float, default=0.5, help='Window overlap')
    
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run grid search
    searcher = OpenCLIPGridSearch(
        video_dir=args.video_dir,
        segments_file=args.segments,
        ground_truth_file=args.ground_truth,
        output_dir=args.output,
        cache_base_dir=args.cache_dir,
        device=args.device,
        use_windowing=not args.no_windowing,
        window_size=args.window_size,
        window_overlap=args.window_overlap
    )
    
    results = searcher.run(
        models=args.models,
        num_frames_list=args.frames,
        aggregations=args.aggregations,
        prompt_modes=args.prompt_modes,
        llm_model=args.llm_model,
        quick=args.quick
    )
    
    if results:
        print(f"\nBest configuration saved to: {args.output}/grid_search_results.json")
    else:
        print("\nNo results produced. Check logs for errors.")
        sys.exit(1)


if __name__ == '__main__':
    main()
