"""
VideoPrism Grid Search Optimizer

Systematically tests all parameter combinations for the VideoPrism LVT pipeline
against a benchmark with ground truth, and reports the best configuration.

Parameters searched:
- model_name: videoprism_lvt_public_v1_base, videoprism_lvt_public_v1_large
- num_frames: 8, 16, 32
- prompt_mode: none, template:video, template:photo, template:scene, template:cooking,
               ensemble:template, ensemble:llm

Note: Unlike OpenCLIP, VideoPrism produces a single global temporal embedding
from all frames together, so there is no per-frame aggregation parameter.

The ensemble modes work by generating multiple text descriptions for each segment,
encoding each one separately, and averaging the resulting embeddings. This tests
whether richer text representations improve matching accuracy.

Usage:
    # Simplest: use benchmark number
    python src/videoprism_grid_search.py --benchmark 2

    # Full grid search with all prompt modes including LLM ensemble
    python src/videoprism_grid_search.py --benchmark 3 --no-windowing \\
        --llm-model llama3.2:3b

    # Quick search
    python src/videoprism_grid_search.py --benchmark 2 --quick
"""

import os
import sys
import json
import time
import logging
import hashlib
import itertools
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from indexing import VideoIndexer
from matching import (
    VideoTextMatcher, create_sequence, ClipSelection,
    PromptedVideoTextMatcher, EnsembleVideoTextMatcher, LLMEnsembleVideoTextMatcher,
    DEFAULT_VIDEOPRISM_ENSEMBLE_TEMPLATES
)
from benchmark import BenchmarkEvaluator, BenchmarkResults
from prompt_generator import create_ensemble_templates
from query_expansion import build_query_expansions, config_from_values

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')


# Prompt templates for VideoPrism text encoding
# These wrap the raw segment text before tokenization
VIDEOPRISM_PROMPT_TEMPLATES = {
    'video': 'a video of {}',
    'photo': 'a photo of {}',
    'scene': 'a scene showing {}',
    'cooking': 'a cooking video showing {}',
}

# Default ensemble templates (used for ensemble:template mode)
# These are applied to each segment text and the embeddings are averaged
DEFAULT_VIDEOPRISM_ENSEMBLE_TEMPLATES = [
    '{}',                          # raw text
    'a video of {}',
    'a photo of {}',
    'a scene showing {}',
    'a short clip of {}',
]


@dataclass
class VideoPrismGridSearchConfig:
    """A single parameter configuration to test."""
    model_name: str
    num_frames: int
    prompt_mode: str  # 'none', 'template:*', 'ensemble:template', 'ensemble:llm'
    resolution: int
    use_dual_softmax: bool
    assignment_method: str = 'hungarian'
    top_k: int = 5
    beam_size: int = 10
    lambda_coherence: float = 0.1
    normalize_scores: bool = True
    query_mode: str = 'original'
    context_window_size: int = 1
    query_llm_model: Optional[str] = None
    use_query_cache: bool = True
    force_refresh_expansions: bool = False
    disable_llm_expansion: bool = False
    prompt_template: Optional[str] = None
    ensemble_prompts: Optional[List[str]] = None


@dataclass
class VideoPrismGridSearchResult:
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


# Matcher subclasses are defined in matching.py and imported above.
# _PromptedVideoTextMatcher  -> PromptedVideoTextMatcher
# _EnsembleVideoTextMatcher  -> EnsembleVideoTextMatcher
# _LLMEnsembleVideoTextMatcher -> LLMEnsembleVideoTextMatcher


class VideoPrismGridSearch:
    """
    Grid search optimizer for VideoPrism pipeline parameters.
    
    Tests all combinations of tunable parameters against a benchmark
    and reports the best configuration.
    """
    
    def __init__(
        self,
        video_dir: str,
        segments_file: str,
        ground_truth_file: str,
        output_dir: str = './output/videoprism_grid_search',
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
        
        # Cache for VideoIndexer instances to avoid OOM by re-loading weights
        self._indexer_cache = {}
        
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
        self.results: List[VideoPrismGridSearchResult] = []
        
        logger.info(f"VideoPrism Grid Search initialized:")
        logger.info(f"  Video dir: {video_dir}")
        logger.info(f"  Segments: {len(self.segments)}")
        logger.info(f"  Ground truth: {ground_truth_file}")
        logger.info(f"  Device: {device}")
    
    def _generate_configs(
        self,
        models: List[str] = None,
        num_frames_list: List[int] = None,
        prompt_modes: List[str] = None,
        resolutions_list: List[int] = None,
        use_dual_softmax_list: List[bool] = None,
        assignment_methods: List[str] = None,
        coherence_top_k_list: List[int] = None,
        coherence_beam_size_list: List[int] = None,
        lambda_coherence_list: List[float] = None,
        normalize_scores: bool = True,
        query_modes: List[str] = None,
        context_window_sizes: List[int] = None,
        query_llm_model: str = None,
        use_query_cache: bool = True,
        force_refresh_expansions: bool = False,
        disable_llm_expansion: bool = False,
        llm_model: str = None
    ) -> List[VideoPrismGridSearchConfig]:
        """
        Generate all parameter configurations to test.
        
        Args:
            models: List of model names to test
            num_frames_list: List of frame counts to test
            prompt_modes: List of prompt modes to test
            llm_model: Ollama model name for LLM ensemble (None to skip)
            
        Returns:
            List of VideoPrismGridSearchConfig objects
        """
        if models is None:
            models = ['videoprism_lvt_public_v1_large']
        if num_frames_list is None:
            num_frames_list = [8, 16, 32]
        if resolutions_list is None:
            resolutions_list = [288, 396]
        if use_dual_softmax_list is None:
            use_dual_softmax_list = [False, True]
        if prompt_modes is None:
            prompt_modes = ['none', 'template:video', 'template:photo',
                          'template:scene', 'template:cooking',
                          'ensemble:template']
            if llm_model:
                prompt_modes.append('ensemble:llm')
        if assignment_methods is None:
            assignment_methods = ['hungarian']
        if coherence_top_k_list is None:
            coherence_top_k_list = [5]
        if coherence_beam_size_list is None:
            coherence_beam_size_list = [10]
        if lambda_coherence_list is None:
            lambda_coherence_list = [0.1]
        if query_modes is None:
            query_modes = ['original']
        if context_window_sizes is None:
            context_window_sizes = [1]
        
        configs = []
        base_product = itertools.product(
            models, num_frames_list, resolutions_list, use_dual_softmax_list, prompt_modes, assignment_methods, query_modes, context_window_sizes
        )
        for model, num_frames, resolution, use_dual_softmax, prompt_mode, assignment_method, query_mode, context_window_size in base_product:
            beam_param_product = [(5, 10, 0.0)]
            if assignment_method == 'coherence_beam':
                beam_param_product = itertools.product(coherence_top_k_list, coherence_beam_size_list, lambda_coherence_list)
            for top_k, beam_size, lambda_coherence in beam_param_product:
                config = VideoPrismGridSearchConfig(
                    model_name=model,
                    num_frames=num_frames,
                    resolution=resolution,
                    use_dual_softmax=use_dual_softmax,
                    prompt_mode=prompt_mode,
                    assignment_method=assignment_method,
                    top_k=int(top_k),
                    beam_size=int(beam_size),
                    lambda_coherence=float(lambda_coherence),
                    normalize_scores=normalize_scores,
                    query_mode=query_mode,
                    context_window_size=int(context_window_size),
                    query_llm_model=query_llm_model or llm_model,
                    use_query_cache=use_query_cache,
                    force_refresh_expansions=force_refresh_expansions,
                    disable_llm_expansion=disable_llm_expansion,
                )
            
                # Set prompt template or ensemble prompts based on mode
                if prompt_mode.startswith('template:'):
                    template_name = prompt_mode.split(':', 1)[1]
                    template = VIDEOPRISM_PROMPT_TEMPLATES.get(template_name)
                    if template is None:
                        logger.warning(f"Unknown template: {template_name}, skipping")
                        continue
                    config.prompt_template = template
                elif prompt_mode == 'ensemble:template':
                    config.ensemble_prompts = DEFAULT_VIDEOPRISM_ENSEMBLE_TEMPLATES
                elif prompt_mode == 'ensemble:llm':
                    # Will be populated during run with pre-generated LLM prompts
                    config.prompt_template = None
                    config.ensemble_prompts = None
                
                configs.append(config)
        
        logger.info(f"Generated {len(configs)} configurations to test")
        return configs
    
    def _clear_sweep_cache(self):
        """Clear cached VideoPrism grid search indices specific to this benchmark to ensure fresh results."""
        import shutil
        import hashlib
        from pathlib import Path
        
        # Calculate the hash for the current video_dir being tested
        # So we don't accidentally wipe a parallel grid search on another benchmark
        video_dir_str = str(self.video_dir)
        video_hash = hashlib.md5(video_dir_str.encode()).hexdigest()[:8]
        prefix = f"vp_{video_hash}_"
        
        if self.cache_base_dir.exists():
            cleared = 0
            for item in self.cache_base_dir.iterdir():
                if item.is_dir() and item.name.startswith(prefix):
                    shutil.rmtree(item)
                    cleared += 1
            if cleared:
                video_name = Path(video_dir_str).name
                logger.info(f"Cleared {cleared} cached VideoPrism indices for benchmark {video_name}")
    
    def _get_cache_dir(self, config: VideoPrismGridSearchConfig) -> str:
        """Get a unique cache directory for a configuration.
        
        Includes a hash of the video directory path so that different
        benchmarks get separate caches and stale indices are never loaded.
        """
        video_hash = hashlib.md5(self.video_dir.encode()).hexdigest()[:8]
        # Cache key: video_hash + model + num_frames + resolution
        # Prompt mode and dual softmax don't affect video indexing, only text encoding/matching
        cache_name = (
            f"vp_{video_hash}_{config.model_name}_{config.num_frames}f_{config.resolution}p"
        )
        return str(self.cache_base_dir / cache_name)
    
    def _run_single_config(
        self,
        config: VideoPrismGridSearchConfig,
        llm_prompts: Optional[Dict[str, List[str]]] = None
    ) -> Optional[VideoPrismGridSearchResult]:
        """
        Run the pipeline with a single configuration and evaluate.
        
        Args:
            config: VideoPrismGridSearchConfig to test
            llm_prompts: Pre-generated LLM prompts (for ensemble:llm mode)
            
        Returns:
            VideoPrismGridSearchResult or None if failed
        """
        # Short model name for display
        model_short = 'base' if 'base' in config.model_name else 'large'
        config_desc = (
            f"VP-{model_short} | {config.num_frames}f | {config.resolution}p | DualSM:{config.use_dual_softmax} | "
            f"{config.prompt_mode} | query:{config.query_mode}/w{config.context_window_size} | assign:{config.assignment_method} | k:{config.top_k} | beam:{config.beam_size} | λ:{config.lambda_coherence}"
        )
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {config_desc}")
        logger.info(f"{'='*60}")
        
        try:
            t_start = time.time()
            
            # 1. Create indexer and index videos
            cache_dir = self._get_cache_dir(config)
            
            # Cache key based on model and resolution/frames (axes that affect the index)
            cache_key = (config.model_name, config.num_frames, config.resolution)
            
            if cache_key in self._indexer_cache:
                indexer = self._indexer_cache[cache_key]
                # Update index_dir just in case, though it should be consistent
                indexer.index_dir = Path(cache_dir)
                logger.info(f"Reusing cached VideoIndexer for {config.model_name} at {config.resolution}p")
            else:
                indexer = VideoIndexer(
                    model_name=config.model_name,
                    index_dir=cache_dir,
                    device=self.device,
                    num_frames=config.num_frames,
                    resolution=config.resolution,
                )
                self._indexer_cache[cache_key] = indexer
            
            # Try to load cached index
            t_index_start = time.time()
            if not indexer.load_index():
                num_indexed = indexer.index_videos(
                    self.video_dir,
                    use_windowing=self.use_windowing,
                    window_size=self.window_size,
                    window_overlap=self.window_overlap
                )
                if num_indexed == 0:
                    raise RuntimeError(f"Failed to index any videos! Check JAX shape constraints (must be multiple of 18) or OOM limits for {config.resolution}p.")
            else:
                logger.info(f"Loaded cached index: {len(indexer.metadata_list)} entries")
            indexing_time = time.time() - t_index_start
            
            # 2. Create matcher based on prompt mode
            if config.prompt_mode == 'ensemble:llm' and llm_prompts:
                # LLM ensemble: per-segment unique prompts
                matcher = LLMEnsembleVideoTextMatcher(
                    video_indexer=indexer,
                    model_name=config.model_name,
                    device=self.device,
                    min_similarity_threshold=0.0,
                    llm_prompts=llm_prompts
                )
            elif config.prompt_mode == 'ensemble:template':
                # Template ensemble: same set of templates for all segments
                matcher = EnsembleVideoTextMatcher(
                    video_indexer=indexer,
                    model_name=config.model_name,
                    device=self.device,
                    min_similarity_threshold=0.0,
                    ensemble_templates=config.ensemble_prompts
                )
            elif config.prompt_template:
                # Single template: wrap text before encoding
                matcher = PromptedVideoTextMatcher(
                    video_indexer=indexer,
                    model_name=config.model_name,
                    device=self.device,
                    min_similarity_threshold=0.0,
                    prompt_template=config.prompt_template
                )
            else:
                # No prompt: raw text
                matcher = VideoTextMatcher(
                    video_indexer=indexer,
                    model_name=config.model_name,
                    device=self.device,
                    min_similarity_threshold=0.0
                )
            
            query_config = config_from_values(
                query_mode=config.query_mode,
                context_window_size=config.context_window_size,
                llm_model=config.query_llm_model,
                use_query_cache=config.use_query_cache,
                force_refresh_expansions=config.force_refresh_expansions,
                disable_llm_calls=config.disable_llm_expansion,
            )
            query_segments, query_metadata = build_query_expansions(
                self.segments,
                config=query_config,
                cache_root=self.cache_base_dir,
            )

            # 3. Run matching
            t_match_start = time.time()
            clip_selections = create_sequence(
                query_segments, matcher,
                match_only=True,
                allow_reuse=False,
                use_optimal=True,
                assignment_method=config.assignment_method,
                top_k=config.top_k,
                beam_size=config.beam_size,
                lambda_coherence=config.lambda_coherence,
                normalize_scores=config.normalize_scores,
            )
            assignment_metadata = getattr(matcher, 'last_assignment_diagnostics', None) or {
                'assignment_method': config.assignment_method,
                'params': {
                    'top_k': config.top_k,
                    'beam_size': config.beam_size,
                    'lambda_coherence': config.lambda_coherence,
                    'normalize_scores': config.normalize_scores,
                    'allow_reuse': False,
                },
            }
            assignment_metadata['query_generation'] = query_metadata
            matching_time = time.time() - t_match_start
            
            if not clip_selections:
                logger.error(f"No clip selections produced for {config_desc}")
                return None
            
            # 4. Compute similarity matrix for evaluation
            similarity_matrix, all_metadata = matcher.compute_similarity_matrix(
                query_segments, 
                match_only=True,
                use_dual_softmax=config.use_dual_softmax
            )
            
            # 5. Evaluate against ground truth
            benchmark_results = self.evaluator.evaluate(
                clip_selections=clip_selections,
                segment_dicts=query_segments,
                similarity_matrix=similarity_matrix,
                all_metadata=all_metadata,
                matching_mode=config.assignment_method,
                allow_reuse=False,
                match_only=True,
                assignment_metadata=assignment_metadata,
            )
            
            total_time = time.time() - t_start
            
            result = VideoPrismGridSearchResult(
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
        prompt_modes: List[str] = None,
        resolutions_list: List[int] = None,
        use_dual_softmax_list: List[bool] = None,
        assignment_methods: List[str] = None,
        coherence_top_k_list: List[int] = None,
        coherence_beam_size_list: List[int] = None,
        lambda_coherence_list: List[float] = None,
        normalize_scores: bool = True,
        query_modes: List[str] = None,
        context_window_sizes: List[int] = None,
        query_llm_model: str = None,
        use_query_cache: bool = True,
        force_refresh_expansions: bool = False,
        disable_llm_expansion: bool = False,
        llm_model: str = None,
        quick: bool = False,
        reset_llm_cache: bool = False
    ) -> List[VideoPrismGridSearchResult]:
        """
        Run the full grid search.
        
        Args:
            models: Models to test (default: [base, large])
            num_frames_list: Frame counts to test (default: [8, 16, 32])
            prompt_modes: Prompt modes to test (default: all)
            llm_model: Ollama model for LLM ensemble (None to skip)
            quick: If True, use a reduced parameter grid for faster testing
            
        Returns:
            List of VideoPrismGridSearchResult sorted by exact match accuracy
        """
        if quick:
            if models is None:
                models = ['videoprism_lvt_public_v1_large']
            if num_frames_list is None:
                num_frames_list = [16]
            if resolutions_list is None:
                resolutions_list = [288]
            if use_dual_softmax_list is None:
                use_dual_softmax_list = [False, True]
            if prompt_modes is None:
                prompt_modes = ['none', 'template:video', 'template:scene']
        
        # Clear any cached indices from previous runs to ensure fresh results
        self._clear_sweep_cache()
        
        # Generate all configurations
        configs = self._generate_configs(
            models=models,
            num_frames_list=num_frames_list,
            prompt_modes=prompt_modes,
            resolutions_list=resolutions_list,
            use_dual_softmax_list=use_dual_softmax_list,
            assignment_methods=assignment_methods,
            coherence_top_k_list=coherence_top_k_list,
            coherence_beam_size_list=coherence_beam_size_list,
            lambda_coherence_list=lambda_coherence_list,
            normalize_scores=normalize_scores,
            query_modes=query_modes,
            context_window_sizes=context_window_sizes,
            query_llm_model=query_llm_model or llm_model,
            use_query_cache=use_query_cache,
            force_refresh_expansions=force_refresh_expansions,
            disable_llm_expansion=disable_llm_expansion,
            llm_model=llm_model
        )
        
        llm_prompts = None
        if llm_model and any(c.prompt_mode == 'ensemble:llm' for c in configs):
            logger.info(f"\nPre-generating LLM prompts using {llm_model}...")
            cache_file = self.output_dir / 'llm_prompts_cache.json'
            if reset_llm_cache and cache_file.exists():
                logger.info(f"Deleting existing LLM prompt cache at {cache_file}")
                cache_file.unlink()
            
            llm_prompts = create_ensemble_templates(
                self.segments,
                llm_model=llm_model,
                num_variations=5,
                cache_file=str(cache_file)
            )
            logger.info(f"Generated prompts for {len(llm_prompts)} segments")
        
        # Run each configuration
        total = len(configs)
        logger.info(f"\n{'#'*60}")
        logger.info(f"STARTING VIDEOPRISM GRID SEARCH: {total} configurations")
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
        
        # Print summary (top 30)
        self._print_summary(elapsed)
        
        return self.results
    
    def _save_results(self, elapsed: float):
        """Save grid search results to JSON and TXT."""
        output_path = self.output_dir / 'videoprism_grid_search_results.json'
        txt_path = self.output_dir / 'videoprism_grid_search_summary.txt'
        
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'encoder': 'videoprism',
            'total_configs_tested': len(self.results),
            'total_time_seconds': round(elapsed, 2),
            'video_dir': self.video_dir,
            'segments_file': self.segments_file,
            'ground_truth_file': self.ground_truth_file,
            'results': [asdict(r) for r in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
            
        summary_text = self._generate_summary_text(elapsed, limit=30)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        if any(r.get('config', {}).get('assignment_method') == 'coherence_beam' for r in results_data['results']):
            benchmark_token = Path(self.video_dir).name.replace('video_', 'benchmark_')
            coherence_path = self.output_dir / f'{benchmark_token}_videoprism_coherence_beam_results.json'
            with open(coherence_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            logger.info(f"Saved coherence beam comparison results to {coherence_path}")

        logger.info(f"Saved grid search results to {output_path} and {txt_path}")

    def _print_summary(self, elapsed: float):
        """Print a formatted summary of grid search results (top 30)."""
        print(self._generate_summary_text(elapsed, limit=30))

    def _generate_summary_text(self, elapsed: float, limit: Optional[int] = None) -> str:
        """Generate a formatted summary of grid search results as a string."""
        results_to_show = self.results[:limit] if limit else self.results
        
        lines = []
        lines.append("\n")
        lines.append("╔" + "═" * 90 + "╗")
        lines.append("║" + " " * 22 + "VIDEOPRISM GRID SEARCH RESULTS SUMMARY" + " " * 30 + "║")
        lines.append("╠" + "═" * 90 + "╣")
        lines.append(f"║  Configurations tested: {len(self.results):<10}  "
              f"Total time: {elapsed:.1f}s" + " " * (90 - 51 - len(f"{elapsed:.1f}")) + "║")
        lines.append("╠" + "═" * 90 + "╣")
        
        # Header
        lines.append("║  RANK │ MODEL   │ FRAMES │ RES  │ DUALSM │ ASSIGNMENT       │ PROMPT MODE        │ EXACT% │ TOP3%  │ TOP5%  │ MRR    ║")
        lines.append("╟" + "─" * 99 + "╢")
        
        # Results
        for i, result in enumerate(results_to_show):
            cfg = result.config
            model = 'base' if 'base' in cfg['model_name'] else 'large'
            frames = str(cfg['num_frames'])
            res = str(cfg['resolution'])
            dualsm = "Yes" if cfg['use_dual_softmax'] else "No"
            prompt = f"{cfg['prompt_mode']}|{cfg.get('query_mode', 'original')}"[:18]
            assignment = cfg.get('assignment_method', 'hungarian')[:16]
            
            marker = " ★" if i == 0 else "  "
            
            lines.append(f"║{marker}{i+1:3d}  │ {model:<7} │ {frames:>6} │ {res:>4} │ {dualsm:<6} │ {assignment:<16} │ {prompt:<18} │ "
                  f"{result.exact_match_accuracy:5.1f}% │ {result.top_3_accuracy:5.1f}% │ "
                  f"{result.top_5_accuracy:5.1f}% │ {result.mrr:.4f} ║")
        
        if limit is not None and len(self.results) > limit:
            lines.append(f"║  ... and {len(self.results) - limit} more configurations" + " " * (99 - 40) + "║")
        
        lines.append("╠" + "═" * 99 + "╣")
        
        # Best configuration
        if self.results:
            best = self.results[0]
            cfg = best.config
            model_display = 'base' if 'base' in cfg['model_name'] else 'large'
            lines.append("║" + " " * 35 + "★ BEST CONFIGURATION ★" + " " * 42 + "║")
            lines.append("╟" + "─" * 99 + "╢")
            lines.append(f"║  Model:        VideoPrism LVT {model_display:<65} ║")
            lines.append(f"║  Full name:    {cfg['model_name']:<81} ║")
            lines.append(f"║  Frames:       {cfg['num_frames']:<81} ║")
            lines.append(f"║  Resolution:   {cfg['resolution']:<81} ║")
            lines.append(f"║  Dual Softmax: {cfg['use_dual_softmax']:<81} ║")
            lines.append(f"║  Prompt Mode:  {cfg['prompt_mode']:<81} ║")
            lines.append(f"║  Query Mode:   {cfg.get('query_mode', 'original'):<81} ║")
            lines.append(f"║  Context Win:  {cfg.get('context_window_size', 1):<81} ║")
            lines.append("╟" + "─" * 99 + "╢")
            lines.append(f"║  Exact Match: {best.exact_match_accuracy:.1f}%   │   "
                  f"Top-3: {best.top_3_accuracy:.1f}%   │   "
                  f"Top-5: {best.top_5_accuracy:.1f}%   │   "
                  f"MRR: {best.mrr:.4f}" + " " * 21 + "║")
            
            # Generate CLI command for best config
            lines.append("╟" + "─" * 99 + "╢")
            lines.append("║  Run with best config:" + " " * 76 + "║")
            
            cmd = f"python src/main.py --encoder videoprism --videoprism-model {cfg['model_name']}"
            cmd += f" --videoprism-frames {cfg['num_frames']} --videoprism-resolution {cfg['resolution']}"
            if cfg.get('use_dual_softmax'):
                cmd += " --use-dual-softmax"
            if cfg.get('query_mode', 'original') != 'original':
                cmd += f" --query-mode {cfg.get('query_mode')} --context-window-size {cfg.get('context_window_size', 1)}"
                if cfg.get('query_llm_model'):
                    cmd += f" --query-llm-model {cfg.get('query_llm_model')}"
            assignment_method = cfg.get('assignment_method', 'hungarian')
            if assignment_method != 'hungarian':
                cmd += f" --assignment-method {assignment_method}"
                cmd += f" --coherence-top-k {cfg.get('top_k', 5)}"
                cmd += f" --coherence-beam-size {cfg.get('beam_size', 10)}"
                cmd += f" --lambda-coherence {cfg.get('lambda_coherence', 0.1)}"
                if not cfg.get('normalize_scores', True):
                    cmd += " --no-normalize-coherence-scores"
            # Wrap long command
            if len(cmd) > 95:
                lines.append(f"║    {cmd[:95]} ║")
                lines.append(f"║    {cmd[95:]:<95} ║")
            else:
                lines.append(f"║    {cmd:<95} ║")
        
        lines.append("╚" + "═" * 99 + "╝")
        return "\n".join(lines)


def _resolve_benchmark_paths(benchmark_num: str, base_dir: str = './data/benchmarks') -> dict:
    """
    Resolve all benchmark paths from a benchmark number.
    
    Given a benchmark number (e.g., '2'), resolves:
      - video_dir:    data/benchmarks/videos/video_2/
      - segments:     data/benchmarks/segments/benchmark_2_segments.json
      - ground_truth: data/benchmarks/gdtruth/benchmark_2_ground_truth.json
    """
    base = Path(base_dir)
    
    video_dir = base / 'videos' / f'video_{benchmark_num}'
    segments_file = base / 'segments' / f'benchmark_{benchmark_num}_segments.json'
    gt_file = base / 'gdtruth' / f'benchmark_{benchmark_num}_ground_truth.json'
    
    errors = []
    if not video_dir.exists():
        errors.append(f"Video directory not found: {video_dir}")
    if not segments_file.exists():
        errors.append(f"Segments file not found: {segments_file}")
    if not gt_file.exists():
        errors.append(f"Ground truth file not found: {gt_file}")
    
    if errors:
        raise FileNotFoundError(
            f"Benchmark {benchmark_num} is incomplete:\n  " + "\n  ".join(errors)
        )
    
    return {
        'video_dir': str(video_dir),
        'segments': str(segments_file),
        'ground_truth': str(gt_file),
    }


def main():
    """CLI entry point for VideoPrism grid search."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='VideoPrism Grid Search Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Grid search using benchmark number (simplest)
  python src/videoprism_grid_search.py --benchmark 2

  # Full sweep with LLM ensemble prompts
  python src/videoprism_grid_search.py --benchmark 3 --no-windowing \\
    --llm-model llama3.2:3b

  # Quick search on benchmark 3
  python src/videoprism_grid_search.py --benchmark 3 --quick

  # Full grid search with explicit paths
  python src/videoprism_grid_search.py \\
    --video-dir data/benchmarks/videos/video_2/ \\
    --segments data/benchmarks/segments/benchmark_2_segments.json \\
    --ground-truth data/benchmarks/gdtruth/benchmark_2_ground_truth.json

  # Custom parameter grid
  python src/videoprism_grid_search.py --benchmark 2 \\
    --models videoprism_lvt_public_v1_base \\
    --frames 8 16 32 \\
    --prompt-modes none ensemble:template ensemble:llm \\
    --llm-model llama3.2:3b
        """
    )
    
    # Benchmark selection: either --benchmark or explicit paths
    parser.add_argument('--benchmark', '-b', type=str, default=None,
                       help='Benchmark number (e.g., 2). Auto-resolves video-dir, segments, and ground-truth paths.')
    parser.add_argument('--benchmarks-dir', default='./data/benchmarks',
                       help='Base directory for benchmarks (default: ./data/benchmarks)')
    parser.add_argument('--video-dir', default=None, help='Directory containing video files (overrides --benchmark)')
    parser.add_argument('--segments', default=None, help='Path to segments JSON file (overrides --benchmark)')
    parser.add_argument('--ground-truth', default=None, help='Path to ground truth JSON file (overrides --benchmark)')
    parser.add_argument('--output', default='./output/videoprism_grid_search', help='Output directory')
    parser.add_argument('--cache-dir', default='./cache', help='Cache directory')
    parser.add_argument('--device', default='cuda:0', help='GPU device')
    parser.add_argument('--quick', action='store_true', help='Use reduced parameter grid')
    
    # Custom grid parameters
    parser.add_argument('--models', nargs='+', default=None,
                       choices=['videoprism_lvt_public_v1_base', 'videoprism_lvt_public_v1_large'],
                       help='Models to test')
    parser.add_argument('--frames', nargs='+', type=int, default=None,
                       help='Frame counts to test')
    parser.add_argument('--resolutions', nargs='+', type=int, default=None,
                       help='Resolutions to test (e.g., 288 384)')
    parser.add_argument('--dual-softmax', nargs='+', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=None,
                       help='Whether to use dual softmax matching (e.g., true false)')
    parser.add_argument('--prompt-modes', nargs='+', default=None,
                       help='Prompt modes to test (e.g., none template:video ensemble:template ensemble:llm)')
    parser.add_argument('--assignment-methods', nargs='+', default=None,
                       choices=['hungarian', 'coherence_beam'],
                       help='Assignment methods to test. Defaults to hungarian only.')
    parser.add_argument('--coherence-top-k', nargs='+', type=int, default=None,
                       help='Top-K candidate counts for coherence beam search')
    parser.add_argument('--coherence-beam-sizes', nargs='+', type=int, default=None,
                       help='Beam sizes for coherence beam search')
    parser.add_argument('--lambda-coherence-values', nargs='+', type=float, default=None,
                       help='Lambda coherence weights for coherence beam search')
    parser.add_argument('--no-normalize-coherence-scores', action='store_false', dest='normalize_coherence_scores', default=True,
                       help='Disable score normalization for coherence beam search')
    parser.add_argument('--query-modes', nargs='+', default=None,
                       choices=['original', 'context_window', 'llm_expanded', 'hybrid_llm'],
                       help='Retrieval query generation modes to test. Defaults to original only.')
    parser.add_argument('--context-window-sizes', nargs='+', type=int, default=None,
                       help='Context window sizes for query generation, e.g. 1 2')
    parser.add_argument('--query-llm-model', default=None,
                       help='LLM model for llm_expanded/hybrid_llm query generation')
    parser.add_argument('--no-query-cache', action='store_false', dest='use_query_cache', default=True,
                       help='Disable cached query expansions')
    parser.add_argument('--force-refresh-expansions', action='store_true',
                       help='Regenerate query expansions even if cache exists')
    parser.add_argument('--disable-llm-expansion', action='store_true',
                       help='Do not call an LLM for query expansion; requires existing cache for LLM modes')
    
    # LLM ensemble
    parser.add_argument('--llm-model', default=None,
                       help='Ollama model for LLM ensemble prompts (e.g., llama3.2:3b). '
                            'Required for ensemble:llm prompt mode.')
    parser.add_argument('--reset-llm-cache', action='store_true',
                       help='Force delete and regenerate existing LLM prompt cache')
    
    # Windowing
    parser.add_argument('--no-windowing', action='store_true', help='Disable windowing')
    parser.add_argument('--window-size', type=float, default=5.0, help='Window size')
    parser.add_argument('--window-overlap', type=float, default=0.5, help='Window overlap')
    
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Resolve benchmark paths if --benchmark is provided
    if args.benchmark:
        try:
            bm_paths = _resolve_benchmark_paths(args.benchmark, args.benchmarks_dir)
            # Use resolved paths, but allow explicit overrides
            if args.video_dir is None:
                args.video_dir = bm_paths['video_dir']
            if args.segments is None:
                args.segments = bm_paths['segments']
            if args.ground_truth is None:
                args.ground_truth = bm_paths['ground_truth']
            logger.info(f"Benchmark {args.benchmark} resolved:")
            logger.info(f"  Video dir:    {args.video_dir}")
            logger.info(f"  Segments:     {args.segments}")
            logger.info(f"  Ground truth: {args.ground_truth}")
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)
    
    # Validate that required paths are set
    if not args.video_dir or not args.segments or not args.ground_truth:
        parser.error(
            "Either --benchmark <number> or all of --video-dir, --segments, "
            "and --ground-truth are required."
        )
    
    # Run grid search
    searcher = VideoPrismGridSearch(
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
        resolutions_list=args.resolutions,
        use_dual_softmax_list=args.dual_softmax,
        prompt_modes=args.prompt_modes,
        assignment_methods=args.assignment_methods,
        coherence_top_k_list=args.coherence_top_k,
        coherence_beam_size_list=args.coherence_beam_sizes,
        lambda_coherence_list=args.lambda_coherence_values,
        normalize_scores=args.normalize_coherence_scores,
        query_modes=args.query_modes,
        context_window_sizes=args.context_window_sizes,
        query_llm_model=args.query_llm_model,
        use_query_cache=args.use_query_cache,
        force_refresh_expansions=args.force_refresh_expansions,
        disable_llm_expansion=args.disable_llm_expansion,
        llm_model=args.llm_model,
        quick=args.quick,
        reset_llm_cache=args.reset_llm_cache
    )
    
    if results:
        print(f"\nBest configuration saved to: {args.output}/videoprism_grid_search_results.json")
    else:
        print("\nNo results produced. Check logs for errors.")
        sys.exit(1)


if __name__ == '__main__':
    main()
