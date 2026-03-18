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
from matching import VideoTextMatcher, create_sequence, ClipSelection
from benchmark import BenchmarkEvaluator, BenchmarkResults
from prompt_generator import create_ensemble_templates

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


class _PromptedVideoTextMatcher(VideoTextMatcher):
    """
    Extended VideoTextMatcher that applies a prompt template to text before encoding.
    
    VideoPrism's default behavior passes raw text directly to the tokenizer.
    This wrapper applies a template (e.g., "a video of {text}") before tokenization
    to test whether prompt engineering improves matching accuracy.
    """
    
    def __init__(
        self,
        video_indexer,
        model_name: str = 'videoprism_lvt_public_v1_base',
        device: str = 'gpu',
        min_similarity_threshold: float = 0.0,
        prompt_template: Optional[str] = None
    ):
        super().__init__(
            video_indexer=video_indexer,
            model_name=model_name,
            device=device,
            min_similarity_threshold=min_similarity_threshold
        )
        self.prompt_template = prompt_template
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text, applying a prompt template first.
        """
        if self.prompt_template:
            prompted_text = self.prompt_template.format(text)
        else:
            prompted_text = text
        return super().get_text_embedding(prompted_text)


class _EnsembleVideoTextMatcher(VideoTextMatcher):
    """
    VideoTextMatcher that encodes multiple prompt variations and averages embeddings.
    
    For ensemble:template mode, applies a fixed set of templates to each segment text.
    The resulting embeddings are averaged and normalized to produce a single query vector.
    """
    
    def __init__(
        self,
        video_indexer,
        model_name: str = 'videoprism_lvt_public_v1_base',
        device: str = 'gpu',
        min_similarity_threshold: float = 0.0,
        ensemble_templates: Optional[List[str]] = None
    ):
        super().__init__(
            video_indexer=video_indexer,
            model_name=model_name,
            device=device,
            min_similarity_threshold=min_similarity_threshold
        )
        self.ensemble_templates = ensemble_templates or DEFAULT_VIDEOPRISM_ENSEMBLE_TEMPLATES
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get averaged embedding from multiple prompt variations.
        """
        # Use batch encoding for all templates at once
        prompts = [template.format(text) for template in self.ensemble_templates]
        all_embeddings = super().get_text_embeddings_batch(prompts)
        
        avg_embedding = np.mean(all_embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        return avg_embedding.astype(np.float32)


class _LLMEnsembleVideoTextMatcher(VideoTextMatcher):
    """
    VideoTextMatcher that uses per-segment LLM-generated prompt variations.
    
    Instead of using the same templates for all segments, this uses unique
    LLM-generated descriptions per segment text. The LLM produces diverse
    paraphrases and visual descriptions that are encoded and averaged.
    """
    
    def __init__(
        self,
        video_indexer,
        model_name: str = 'videoprism_lvt_public_v1_base',
        device: str = 'gpu',
        min_similarity_threshold: float = 0.0,
        llm_prompts: Optional[Dict[str, List[str]]] = None
    ):
        super().__init__(
            video_indexer=video_indexer,
            model_name=model_name,
            device=device,
            min_similarity_threshold=min_similarity_threshold
        )
        self.llm_prompts = llm_prompts or {}
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get averaged embedding using LLM-generated prompts specific to this text.
        Falls back to raw text if no LLM prompts are available.
        """
        prompts = self.llm_prompts.get(text, [text])
        
        if len(prompts) <= 1:
            return super().get_text_embedding(text)
        
        # Use batch encoding for all LLM prompts at once
        all_embeddings = super().get_text_embeddings_batch(prompts)
        
        avg_embedding = np.mean(all_embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        return avg_embedding.astype(np.float32)


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
        
        configs = []
        for model, num_frames, resolution, use_dual_softmax, prompt_mode in itertools.product(
            models, num_frames_list, resolutions_list, use_dual_softmax_list, prompt_modes
        ):
            config = VideoPrismGridSearchConfig(
                model_name=model,
                num_frames=num_frames,
                resolution=resolution,
                use_dual_softmax=use_dual_softmax,
                prompt_mode=prompt_mode
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
            f"VP-{model_short} | {config.num_frames}f | {config.resolution}p | DualSM:{config.use_dual_softmax} | {config.prompt_mode}"
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
                    device=self.device
                )
                self._indexer_cache[cache_key] = indexer
            
            # Override the default num_frames and resolution for frame extraction
            original_extract_frames = indexer.extract_frames
            
            def patched_extract_frames(video_path, num_frames=config.num_frames,
                                       target_size=(config.resolution, config.resolution), start_time=0.0, end_time=-1.0):
                return original_extract_frames(
                    video_path, num_frames=num_frames,
                    target_size=target_size, start_time=start_time, end_time=end_time
                )
            
            indexer.extract_frames = patched_extract_frames
            
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
                matcher = _LLMEnsembleVideoTextMatcher(
                    video_indexer=indexer,
                    model_name=config.model_name,
                    device=self.device,
                    min_similarity_threshold=0.0,
                    llm_prompts=llm_prompts
                )
            elif config.prompt_mode == 'ensemble:template':
                # Template ensemble: same set of templates for all segments
                matcher = _EnsembleVideoTextMatcher(
                    video_indexer=indexer,
                    model_name=config.model_name,
                    device=self.device,
                    min_similarity_threshold=0.0,
                    ensemble_templates=config.ensemble_prompts
                )
            elif config.prompt_template:
                # Single template: wrap text before encoding
                matcher = _PromptedVideoTextMatcher(
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
                self.segments, 
                match_only=True,
                use_dual_softmax=config.use_dual_softmax
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
        lines.append("║  RANK │ MODEL   │ FRAMES │ RES  │ DUALSM │ PROMPT MODE        │ EXACT% │ TOP3%  │ TOP5%  │ MRR    ║")
        lines.append("╟" + "─" * 99 + "╢")
        
        # Results
        for i, result in enumerate(results_to_show):
            cfg = result.config
            model = 'base' if 'base' in cfg['model_name'] else 'large'
            frames = str(cfg['num_frames'])
            res = str(cfg['resolution'])
            dualsm = "Yes" if cfg['use_dual_softmax'] else "No"
            prompt = cfg['prompt_mode'][:18]
            
            marker = " ★" if i == 0 else "  "
            
            lines.append(f"║{marker}{i+1:3d}  │ {model:<7} │ {frames:>6} │ {res:>4} │ {dualsm:<6} │ {prompt:<18} │ "
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
            lines.append("╟" + "─" * 99 + "╢")
            lines.append(f"║  Exact Match: {best.exact_match_accuracy:.1f}%   │   "
                  f"Top-3: {best.top_3_accuracy:.1f}%   │   "
                  f"Top-5: {best.top_5_accuracy:.1f}%   │   "
                  f"MRR: {best.mrr:.4f}" + " " * 21 + "║")
            
            # Generate CLI command for best config
            lines.append("╟" + "─" * 99 + "╢")
            lines.append("║  Run with best config:" + " " * 76 + "║")
            
            cmd = f"python src/main.py --encoder videoprism --videoprism-model {cfg['model_name']}"
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
