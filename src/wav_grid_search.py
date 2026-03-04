"""
Write-A-Video (WAV) Two-Stage Grid Search Optimizer

Systematically tests all parameter combinations for the Write-A-Video
inspired two-stage retrieval pipeline against a benchmark with ground truth.

Architecture (from Wang et al., "Write-A-Video", TOG 2019):
  Stage 1: Multi-modal keyword-based candidate filtering
            (Object Detection via YOLOv8, Face Recognition via DeepFace,
             Filename keyword extraction, TF-IDF scoring)
  Stage 2: Visual-semantic reranking using OpenCLIP embeddings

Parameters searched:
- OpenCLIP model: ViT-B-32, ViT-B-16, ViT-L-14
- Frame count (for OpenCLIP): 4, 8, 16, 32
- Aggregation: mean, max, best_frame
- Prompt mode: none, template:video, template:photo, template:scene,
               template:cooking, ensemble:template, ensemble:llm
- Candidate pool size: how many Stage 1 candidates to rerank
- Keyword weight: weight of keyword score in final ranking
- Keyword sources: which modalities to use for keyword extraction

Usage:
    # Simplest: use benchmark number
    python src/wav_grid_search.py --benchmark 3

    # Full sweep with LLM ensemble
    python src/wav_grid_search.py --benchmark 3 --no-windowing --llm-model llama3.2:3b

    # Quick search
    python src/wav_grid_search.py --benchmark 3 --quick
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

from openclip_indexing import (
    OpenCLIPVideoIndexer, OpenCLIPTextMatcher,
    PROMPT_TEMPLATES, DEFAULT_ENSEMBLE_TEMPLATES
)
from wav_indexing import MultiModalKeywordIndexer, WriteAVideoMatcher
from matching import create_sequence, ClipSelection
from benchmark import BenchmarkEvaluator, BenchmarkResults
from prompt_generator import create_ensemble_templates

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')


@dataclass
class WAVGridSearchConfig:
    """A single parameter configuration to test."""
    # OpenCLIP parameters (Stage 2)
    model_name: str
    num_frames: int
    aggregation: str
    prompt_mode: str  # 'none', 'template:<name>', 'ensemble:template', 'ensemble:llm'
    # WAV-specific parameters
    candidate_pool_size: int = 10
    keyword_weight: float = 0.0  # 0 = pure rerank, >0 = blend keyword + semantic
    enable_face_detection: bool = True
    enable_object_detection: bool = True
    # Derived
    prompt_template: Optional[str] = None
    ensemble_prompts: Optional[List[str]] = None


@dataclass
class WAVGridSearchResult:
    """Result of a single grid search configuration."""
    config: Dict
    exact_match_accuracy: float
    top_3_accuracy: float
    top_5_accuracy: float
    mrr: float
    avg_similarity: float
    keyword_indexing_time: float
    openclip_indexing_time: float
    matching_time: float
    total_time: float


class _LLMEnsembleWAVMatcher(WriteAVideoMatcher):
    """
    Write-A-Video matcher that uses per-segment LLM-generated prompts
    for the Stage 2 OpenCLIP reranking.
    """

    def __init__(
        self,
        keyword_indexer: MultiModalKeywordIndexer,
        openclip_indexer: OpenCLIPVideoIndexer,
        llm_prompts: Dict[str, List[str]],
        candidate_pool_size: int = 10,
        keyword_weight: float = 0.0,
    ):
        super().__init__(
            keyword_indexer=keyword_indexer,
            openclip_indexer=openclip_indexer,
            candidate_pool_size=candidate_pool_size,
            keyword_weight=keyword_weight,
            prompt_template=None,
            ensemble_prompts=None,
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


class WAVGridSearch:
    """
    Grid search optimizer for the Write-A-Video two-stage pipeline.

    Tests all combinations of tunable parameters against a benchmark
    and reports the best configuration.
    """

    def __init__(
        self,
        video_dir: str,
        segments_file: str,
        ground_truth_file: str,
        output_dir: str = './output/wav_grid_search',
        cache_base_dir: str = './cache',
        device: str = 'cuda:0',
        use_windowing: bool = False,
        window_size: float = 5.0,
        window_overlap: float = 0.5,
        yolo_model: str = 'yolov8n',
        yolo_confidence: float = 0.3,
        frames_per_second: float = 1.0,
    ):
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
        self.yolo_model = yolo_model
        self.yolo_confidence = yolo_confidence
        self.frames_per_second = frames_per_second

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
        self.results: List[WAVGridSearchResult] = []

        # Cached keyword indexer (shared across all configs)
        self._keyword_indexer_cache: Dict[str, MultiModalKeywordIndexer] = {}

        logger.info(f"WAV Grid Search initialized:")
        logger.info(f"  Video dir: {video_dir}")
        logger.info(f"  Segments: {len(self.segments)}")
        logger.info(f"  Ground truth: {ground_truth_file}")
        logger.info(f"  YOLO model: {yolo_model} (conf={yolo_confidence})")
        logger.info(f"  Device: {device}")

    def _generate_configs(
        self,
        models: List[str] = None,
        num_frames_list: List[int] = None,
        aggregations: List[str] = None,
        prompt_modes: List[str] = None,
        candidate_pool_sizes: List[int] = None,
        keyword_weights: List[float] = None,
        llm_model: str = None
    ) -> List[WAVGridSearchConfig]:
        """Generate all parameter configurations to test."""
        if models is None:
            models = ['ViT-B-32', 'ViT-B-16', 'ViT-L-14']
        if num_frames_list is None:
            num_frames_list = [4, 8, 16, 32]
        if aggregations is None:
            aggregations = ['mean', 'max', 'best_frame']
        if prompt_modes is None:
            prompt_modes = ['none', 'template:video', 'template:photo',
                            'template:cooking', 'template:scene',
                            'ensemble:template']
            if llm_model:
                prompt_modes.append('ensemble:llm')
        if candidate_pool_sizes is None:
            candidate_pool_sizes = [10]
        if keyword_weights is None:
            keyword_weights = [0.0]

        configs = []
        for model, n_frames, agg, prompt_mode, pool_size, kw_weight in itertools.product(
            models, num_frames_list, aggregations, prompt_modes,
            candidate_pool_sizes, keyword_weights
        ):
            config = WAVGridSearchConfig(
                model_name=model,
                num_frames=n_frames,
                aggregation=agg,
                prompt_mode=prompt_mode,
                candidate_pool_size=pool_size,
                keyword_weight=kw_weight,
                enable_face_detection=True,
                enable_object_detection=True,
            )

            # Set prompt template or ensemble prompts based on mode
            if prompt_mode.startswith('template:'):
                template_name = prompt_mode.split(':')[1]
                config.prompt_template = PROMPT_TEMPLATES.get(template_name, '{text}')
            elif prompt_mode == 'ensemble:template':
                config.ensemble_prompts = DEFAULT_ENSEMBLE_TEMPLATES
            elif prompt_mode == 'ensemble:llm':
                config.prompt_template = None
                config.ensemble_prompts = None

            configs.append(config)

        logger.info(f"Generated {len(configs)} configurations to test")
        return configs

    def _clear_sweep_cache(self):
        """Clear cached WAV grid search indices specific to this benchmark to ensure fresh results."""
        import shutil
        import hashlib
        from pathlib import Path
        
        video_dir_str = str(self.video_dir)
        video_hash = hashlib.md5(video_dir_str.encode()).hexdigest()[:8]
        prefix_gs = f"gs_{video_hash}_"
        prefix_wav = f"wav_kw_{video_hash}_"
        
        if self.cache_base_dir.exists():
            cleared = 0
            for item in self.cache_base_dir.iterdir():
                if item.is_dir() and (item.name.startswith(prefix_gs) or item.name.startswith(prefix_wav)):
                    shutil.rmtree(item)
                    cleared += 1
            if cleared:
                video_name = Path(video_dir_str).name
                logger.info(f"Cleared {cleared} cached WAV grid search indices for benchmark {video_name}")
    
    def _get_keyword_cache_dir(self, config: WAVGridSearchConfig) -> str:
        """Get cache directory for keyword index."""
        video_hash = hashlib.md5(self.video_dir.encode()).hexdigest()[:8]
        face_flag = 'f1' if config.enable_face_detection else 'f0'
        obj_flag = 'o1' if config.enable_object_detection else 'o0'
        cache_name = f"wav_kw_{video_hash}_{face_flag}_{obj_flag}"
        return str(self.cache_base_dir / cache_name)

    def _get_openclip_cache_dir(self, config: WAVGridSearchConfig) -> str:
        """Get cache directory for OpenCLIP index."""
        video_hash = hashlib.md5(self.video_dir.encode()).hexdigest()[:8]
        cache_name = (
            f"gs_{video_hash}_{config.model_name}_{config.num_frames}f_{config.aggregation}"
        )
        return str(self.cache_base_dir / cache_name)

    def _get_or_create_keyword_indexer(
        self, config: WAVGridSearchConfig
    ) -> Tuple[MultiModalKeywordIndexer, float]:
        """
        Get or create a keyword indexer for the given config.
        Caches the indexer to avoid re-indexing for different OpenCLIP configs.
        """
        cache_dir = self._get_keyword_cache_dir(config)
        cache_key = cache_dir

        if cache_key in self._keyword_indexer_cache:
            logger.info("Using cached keyword indexer from memory")
            return self._keyword_indexer_cache[cache_key], 0.0

        t_start = time.time()

        indexer = MultiModalKeywordIndexer(
            index_dir=cache_dir,
            yolo_model=self.yolo_model,
            yolo_confidence=self.yolo_confidence,
            frames_per_second=self.frames_per_second,
            enable_face_detection=config.enable_face_detection,
            enable_object_detection=config.enable_object_detection,
        )

        if not indexer.load_index():
            logger.info("Building keyword index (this only happens once)...")
            indexer.index_videos(self.video_dir)
            indexer.save_index()
        else:
            logger.info(f"Loaded cached keyword index: {len(indexer.video_entries)} videos")

        elapsed = time.time() - t_start
        self._keyword_indexer_cache[cache_key] = indexer
        return indexer, elapsed

    def _run_single_config(
        self,
        config: WAVGridSearchConfig,
        llm_prompts: Optional[Dict[str, List[str]]] = None
    ) -> Optional[WAVGridSearchResult]:
        """Run the pipeline with a single configuration and evaluate."""
        config_desc = (
            f"{config.model_name} | {config.num_frames}f | {config.aggregation} | "
            f"{config.prompt_mode} | pool={config.candidate_pool_size} | kw={config.keyword_weight}"
        )
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing: {config_desc}")
        logger.info(f"{'='*70}")

        try:
            t_start = time.time()

            # 1. Get or create keyword indexer (Stage 1)
            keyword_indexer, kw_index_time = self._get_or_create_keyword_indexer(config)

            # 2. Create OpenCLIP indexer (Stage 2)
            openclip_cache_dir = self._get_openclip_cache_dir(config)

            openclip_indexer = OpenCLIPVideoIndexer(
                model_name=config.model_name,
                index_dir=openclip_cache_dir,
                device=self.device,
                num_frames=config.num_frames,
                aggregation=config.aggregation
            )

            t_clip_start = time.time()
            if not openclip_indexer.load_index():
                openclip_indexer.index_videos(
                    self.video_dir,
                    use_windowing=self.use_windowing,
                    window_size=self.window_size,
                    window_overlap=self.window_overlap
                )
            clip_index_time = time.time() - t_clip_start

            # 3. Create two-stage matcher
            if config.prompt_mode == 'ensemble:llm' and llm_prompts:
                matcher = _LLMEnsembleWAVMatcher(
                    keyword_indexer=keyword_indexer,
                    openclip_indexer=openclip_indexer,
                    llm_prompts=llm_prompts,
                    candidate_pool_size=config.candidate_pool_size,
                    keyword_weight=config.keyword_weight,
                )
            else:
                matcher = WriteAVideoMatcher(
                    keyword_indexer=keyword_indexer,
                    openclip_indexer=openclip_indexer,
                    candidate_pool_size=config.candidate_pool_size,
                    keyword_weight=config.keyword_weight,
                    prompt_template=config.prompt_template,
                    ensemble_prompts=config.ensemble_prompts,
                )

            # 4. Run matching
            t_match_start = time.time()
            clip_selections = create_sequence(
                self.segments, matcher,
                match_only=True, allow_reuse=False, use_optimal=True
            )
            matching_time = time.time() - t_match_start

            if not clip_selections:
                logger.error(f"No clip selections produced for {config_desc}")
                return None

            # 5. Compute similarity matrix for evaluation
            similarity_matrix, all_metadata = matcher.compute_similarity_matrix(
                self.segments, match_only=True
            )

            # 6. Evaluate against ground truth
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

            result = WAVGridSearchResult(
                config=asdict(config),
                exact_match_accuracy=benchmark_results.exact_match_accuracy,
                top_3_accuracy=benchmark_results.top_3_accuracy,
                top_5_accuracy=benchmark_results.top_5_accuracy,
                mrr=benchmark_results.mean_reciprocal_rank,
                avg_similarity=benchmark_results.avg_predicted_similarity,
                keyword_indexing_time=round(kw_index_time, 2),
                openclip_indexing_time=round(clip_index_time, 2),
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
        candidate_pool_sizes: List[int] = None,
        keyword_weights: List[float] = None,
        llm_model: str = None,
        quick: bool = False
    ) -> List[WAVGridSearchResult]:
        """
        Run the full grid search.

        Args:
            models: Models to test (default: all)
            num_frames_list: Frame counts to test (default: [4, 8, 16, 32])
            aggregations: Aggregation methods to test (default: all)
            prompt_modes: Prompt modes to test (default: all)
            candidate_pool_sizes: Candidate pool sizes to test
            keyword_weights: Keyword weights to test
            llm_model: Ollama model for LLM ensemble (None to skip)
            quick: If True, use a reduced parameter grid

        Returns:
            List of WAVGridSearchResult sorted by exact match accuracy
        """
        if quick:
            if models is None:
                models = ['ViT-B-32']
            if num_frames_list is None:
                num_frames_list = [8, 16]
            if aggregations is None:
                aggregations = ['mean', 'best_frame']
            if prompt_modes is None:
                prompt_modes = ['none', 'template:scene', 'ensemble:template']
            if candidate_pool_sizes is None:
                candidate_pool_sizes = [10]
            if keyword_weights is None:
                keyword_weights = [0.0]

        # Clear any cached indices from previous runs to ensure fresh results
        self._clear_sweep_cache()
        
        # Generate all configurations
        configs = self._generate_configs(
            models=models,
            num_frames_list=num_frames_list,
            aggregations=aggregations,
            prompt_modes=prompt_modes,
            candidate_pool_sizes=candidate_pool_sizes,
            keyword_weights=keyword_weights,
            llm_model=llm_model
        )

        # Pre-generate LLM prompts if needed
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
        logger.info(f"\n{'#'*70}")
        logger.info(f"STARTING WRITE-A-VIDEO GRID SEARCH: {total} configurations")
        logger.info(f"{'#'*70}\n")

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
        output_path = self.output_dir / 'wav_grid_search_results.json'

        results_data = {
            'timestamp': datetime.now().isoformat(),
            'encoder': 'write-a-video (two-stage)',
            'total_configs_tested': len(self.results),
            'total_time_seconds': round(elapsed, 2),
            'video_dir': self.video_dir,
            'segments_file': self.segments_file,
            'ground_truth_file': self.ground_truth_file,
            'yolo_model': self.yolo_model,
            'results': [asdict(r) for r in self.results]
        }

        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Saved grid search results to {output_path}")

    def _print_summary(self, elapsed: float):
        """Print a formatted summary of grid search results."""
        print("\n")
        print("╔" + "═" * 105 + "╗")
        print("║" + " " * 25 + "WRITE-A-VIDEO GRID SEARCH RESULTS SUMMARY" + " " * 39 + "║")
        print("╠" + "═" * 105 + "╣")
        print(f"║  Configurations tested: {len(self.results):<10}  "
              f"Total time: {elapsed:.1f}s" + " " * (105 - 50 - len(f"{elapsed:.1f}")) + "║")
        print("╠" + "═" * 105 + "╣")

        # Header
        print("║  RANK │ MODEL      │ FRAMES │ AGG        │ PROMPT MODE        │ POOL │ EXACT% │ TOP3%  │ MRR    ║")
        print("╟" + "─" * 105 + "╢")

        # All results
        for i, result in enumerate(self.results[:30]):
            cfg = result.config
            model = cfg['model_name'][:10]
            frames = str(cfg['num_frames'])
            agg = cfg['aggregation'][:10]
            prompt = cfg['prompt_mode'][:18]
            pool = str(cfg['candidate_pool_size'])

            marker = " ★" if i == 0 else "  "

            print(f"║{marker}{i+1:3d}  │ {model:<10} │ {frames:>6} │ {agg:<10} │ {prompt:<18} │ {pool:>4} │ "
                  f"{result.exact_match_accuracy:5.1f}% │ {result.top_3_accuracy:5.1f}% │ {result.mrr:.4f} ║")

        if len(self.results) > 30:
            remaining = len(self.results) - 30
            print(f"║  ... and {remaining} more configurations" + " " * (105 - 35 - len(str(remaining))) + "║")

        print("╠" + "═" * 105 + "╣")

        # Best configuration
        if self.results:
            best = self.results[0]
            cfg = best.config
            print("║" + " " * 35 + "★ BEST CONFIGURATION ★" + " " * 48 + "║")
            print("╟" + "─" * 105 + "╢")
            print(f"║  Model:            {cfg['model_name']:<84} ║")
            print(f"║  Frames:           {cfg['num_frames']:<84} ║")
            print(f"║  Aggregation:      {cfg['aggregation']:<84} ║")
            print(f"║  Prompt Mode:      {cfg['prompt_mode']:<84} ║")
            print(f"║  Candidate Pool:   {cfg['candidate_pool_size']:<84} ║")
            print(f"║  Keyword Weight:   {cfg['keyword_weight']:<84} ║")
            print("╟" + "─" * 105 + "╢")
            print(f"║  Exact Match: {best.exact_match_accuracy:.1f}%   │   "
                  f"Top-3: {best.top_3_accuracy:.1f}%   │   "
                  f"Top-5: {best.top_5_accuracy:.1f}%   │   "
                  f"MRR: {best.mrr:.4f}" + " " * 27 + "║")
            print("╟" + "─" * 105 + "╢")
            print("║  Keyword indexing time: "
                  f"{best.keyword_indexing_time:.1f}s   │   "
                  f"OpenCLIP indexing: {best.openclip_indexing_time:.1f}s   │   "
                  f"Matching: {best.matching_time:.1f}s" + " " * 20 + "║")

        print("╚" + "═" * 105 + "╝")
        print()


def _resolve_benchmark_paths(benchmark_num: str, base_dir: str = './data/benchmarks') -> dict:
    """Resolve all benchmark paths from a benchmark number."""
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
    """CLI entry point for Write-A-Video grid search."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Write-A-Video Two-Stage Grid Search Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Grid search using benchmark number (simplest)
  python src/wav_grid_search.py --benchmark 3

  # Full sweep with LLM ensemble prompts
  python src/wav_grid_search.py --benchmark 3 --no-windowing --llm-model llama3.2:3b

  # Quick search
  python src/wav_grid_search.py --benchmark 3 --quick

  # Custom parameter grid
  python src/wav_grid_search.py --benchmark 3 \\
    --models ViT-B-16 ViT-L-14 \\
    --frames 8 16 \\
    --aggregations mean best_frame \\
    --pool-sizes 5 10 20

  # Test different keyword weights
  python src/wav_grid_search.py --benchmark 3 \\
    --keyword-weights 0.0 0.1 0.2 0.3
        """
    )

    # Benchmark selection
    parser.add_argument('--benchmark', '-b', type=str, default=None,
                        help='Benchmark number (e.g., 3). Auto-resolves all paths.')
    parser.add_argument('--benchmarks-dir', default='./data/benchmarks',
                        help='Base directory for benchmarks')
    parser.add_argument('--video-dir', default=None, help='Directory containing video files')
    parser.add_argument('--segments', default=None, help='Path to segments JSON file')
    parser.add_argument('--ground-truth', default=None, help='Path to ground truth JSON file')
    parser.add_argument('--output', default='./output/wav_grid_search', help='Output directory')
    parser.add_argument('--cache-dir', default='./cache', help='Cache directory')
    parser.add_argument('--device', default='cuda:0', help='GPU device')
    parser.add_argument('--quick', action='store_true', help='Use reduced parameter grid')
    parser.add_argument('--llm-model', default=None, help='Ollama model for LLM ensemble')

    # OpenCLIP grid parameters (Stage 2)
    parser.add_argument('--models', nargs='+', default=None,
                        choices=['ViT-B-32', 'ViT-B-16', 'ViT-L-14'],
                        help='OpenCLIP models to test')
    parser.add_argument('--frames', nargs='+', type=int, default=None,
                        help='Frame counts to test')
    parser.add_argument('--aggregations', nargs='+', default=None,
                        choices=['mean', 'max', 'best_frame'],
                        help='Aggregation methods to test')
    parser.add_argument('--prompt-modes', nargs='+', default=None,
                        help='Prompt modes to test')

    # WAV-specific parameters
    parser.add_argument('--pool-sizes', nargs='+', type=int, default=None,
                        help='Candidate pool sizes to test (default: [10])')
    parser.add_argument('--keyword-weights', nargs='+', type=float, default=None,
                        help='Keyword weights to test (default: [0.0])')
    parser.add_argument('--yolo-model', default='yolov8n',
                        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l'],
                        help='YOLOv8 model variant for object detection')
    parser.add_argument('--yolo-confidence', type=float, default=0.3,
                        help='YOLO confidence threshold')
    parser.add_argument('--fps', type=float, default=1.0,
                        help='Frames per second to analyze for keyword extraction')
    parser.add_argument('--no-face-detection', action='store_true',
                        help='Disable face detection')
    parser.add_argument('--no-object-detection', action='store_true',
                        help='Disable object detection')

    # Windowing
    parser.add_argument('--no-windowing', action='store_true', help='Disable windowing')
    parser.add_argument('--window-size', type=float, default=5.0, help='Window size')
    parser.add_argument('--window-overlap', type=float, default=0.5, help='Window overlap')

    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve benchmark paths
    if args.benchmark:
        try:
            bm_paths = _resolve_benchmark_paths(args.benchmark, args.benchmarks_dir)
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

    # Validate required paths
    if not args.video_dir or not args.segments or not args.ground_truth:
        parser.error(
            "Either --benchmark <number> or all of --video-dir, --segments, "
            "and --ground-truth are required."
        )

    # Run grid search
    searcher = WAVGridSearch(
        video_dir=args.video_dir,
        segments_file=args.segments,
        ground_truth_file=args.ground_truth,
        output_dir=args.output,
        cache_base_dir=args.cache_dir,
        device=args.device,
        use_windowing=not args.no_windowing,
        window_size=args.window_size,
        window_overlap=args.window_overlap,
        yolo_model=args.yolo_model,
        yolo_confidence=args.yolo_confidence,
        frames_per_second=args.fps,
    )

    results = searcher.run(
        models=args.models,
        num_frames_list=args.frames,
        aggregations=args.aggregations,
        prompt_modes=args.prompt_modes,
        candidate_pool_sizes=args.pool_sizes,
        keyword_weights=args.keyword_weights,
        llm_model=args.llm_model,
        quick=args.quick
    )

    if results:
        print(f"\nBest configuration saved to: {args.output}/wav_grid_search_results.json")
    else:
        print("\nNo results produced. Check logs for errors.")
        sys.exit(1)


if __name__ == '__main__':
    main()
