"""
Benchmark Evaluation Module

This module is responsible for:
1. Loading ground truth data for benchmark evaluation
2. Comparing predicted clip selections against ground truth
3. Computing accuracy metrics and statistics
4. Saving benchmark results to files

IMPORTANT: This module is EVALUATION-ONLY. It is called AFTER clip selection
is complete and NEVER influences the matching process. The ground truth file
is only used for scoring and comparison purposes.

Metrics computed:
- Exact Match Accuracy: % of segments where the predicted clip exactly matches ground truth
- Top-K Accuracy: % of segments where ground truth is in top K predictions
- Mean Reciprocal Rank (MRR): Average of 1/rank of ground truth in predictions
- Similarity Score Analysis: Statistics on predicted vs ground truth similarity scores
"""

import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SegmentBenchmarkResult:
    """Result for a single segment comparison"""
    segment_id: int
    segment_text: str
    ground_truth_clip: str
    predicted_clip: str
    is_exact_match: bool
    predicted_similarity: float
    ground_truth_rank: int  # Rank of ground truth in predictions (1 = best, -1 = not found)
    reciprocal_rank: float  # 1/rank or 0 if not found


@dataclass
class BenchmarkResults:
    """Complete benchmark evaluation results"""
    # Metadata
    benchmark_name: str
    timestamp: str
    num_segments: int
    num_videos: int
    matching_mode: str  # 'optimal' or 'greedy'
    allow_reuse: bool
    match_only: bool
    
    # Accuracy Metrics
    exact_match_accuracy: float  # % of exact matches (0-100)
    top_3_accuracy: float  # % where ground truth is in top 3
    top_5_accuracy: float  # % where ground truth is in top 5
    mean_reciprocal_rank: float  # MRR score (0-1)
    
    # Similarity Score Statistics
    avg_predicted_similarity: float
    avg_ground_truth_similarity: float
    similarity_correlation: float  # Correlation between predicted and GT similarities
    
    # Per-segment results
    segment_results: List[Dict]
    
    # Summary
    exact_matches: int
    mismatches: int
    mismatch_details: List[Dict]


class BenchmarkEvaluator:
    """
    Evaluates video-text matching predictions against ground truth.
    
    IMPORTANT: This class is for EVALUATION ONLY. It receives already-completed
    clip selections and compares them to ground truth. It NEVER influences
    the matching algorithm.
    """
    
    def __init__(self, ground_truth_file: str):
        """
        Initialize the evaluator with ground truth data.
        
        Args:
            ground_truth_file: Path to JSON file with ground truth mappings
        """
        self.ground_truth_file = Path(ground_truth_file)
        self.ground_truth = self._load_ground_truth()
        
    def _load_ground_truth(self) -> Dict:
        """Load ground truth from JSON file."""
        if not self.ground_truth_file.exists():
            raise FileNotFoundError(f"Ground truth file not found: {self.ground_truth_file}")
        
        with open(self.ground_truth_file, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded ground truth with {len(data.get('mappings', []))} segment mappings")
        return data
    
    def evaluate(
        self,
        clip_selections: List,
        segment_dicts: List[Dict],
        similarity_matrix: Optional[np.ndarray] = None,
        all_metadata: Optional[List] = None,
        matching_mode: str = 'optimal',
        allow_reuse: bool = True,
        match_only: bool = False
    ) -> BenchmarkResults:
        """
        Evaluate predictions against ground truth.
        
        Args:
            clip_selections: List of ClipSelection objects from the matcher
            segment_dicts: List of segment dictionaries with 'text'
            similarity_matrix: Optional full similarity matrix for ranking analysis
            all_metadata: Optional list of all video metadata
            matching_mode: 'optimal' or 'greedy'
            allow_reuse: Whether reuse was allowed
            match_only: Whether match-only mode was used
            
        Returns:
            BenchmarkResults with all metrics and statistics
        """
        gt_mappings = self.ground_truth.get('mappings', [])
        benchmark_name = self.ground_truth.get('name', 'unnamed_benchmark')
        
        if len(gt_mappings) != len(clip_selections):
            logger.warning(f"Ground truth has {len(gt_mappings)} mappings but got {len(clip_selections)} predictions")
        
        # Build ground truth lookup by segment_id
        gt_by_segment = {}
        for mapping in gt_mappings:
            seg_id = mapping.get('segment_id', mapping.get('index', -1))
            gt_by_segment[seg_id] = mapping
        
        # Build video name to index mapping if we have metadata
        video_name_to_idx = {}
        if all_metadata:
            for idx, meta in enumerate(all_metadata):
                video_name = Path(meta.file_path).stem  # filename without extension
                video_name_to_idx[video_name] = idx
                # Also add with extension
                video_name_to_idx[Path(meta.file_path).name] = idx
        
        # Evaluate each segment
        segment_results = []
        exact_matches = 0
        top_3_hits = 0
        top_5_hits = 0
        reciprocal_ranks = []
        predicted_similarities = []
        gt_similarities = []
        mismatch_details = []
        
        for i, selection in enumerate(clip_selections):
            gt_mapping = gt_by_segment.get(i, {})
            gt_clip = gt_mapping.get('clip', gt_mapping.get('video', ''))
            
            # Normalize clip names for comparison
            predicted_clip = Path(selection.video_file_path).stem
            gt_clip_normalized = Path(gt_clip).stem if gt_clip else ''
            
            # Check exact match
            is_exact = self._clips_match(predicted_clip, gt_clip_normalized)
            if is_exact:
                exact_matches += 1
            
            # Get segment text
            seg_text = segment_dicts[i]['text'] if i < len(segment_dicts) else ''
            
            # Calculate rank of ground truth in predictions
            gt_rank = -1
            reciprocal_rank = 0.0
            
            if similarity_matrix is not None and all_metadata and gt_clip_normalized:
                # Find the index of ground truth video
                gt_idx = video_name_to_idx.get(gt_clip_normalized, -1)
                if gt_idx == -1:
                    # Try with common variations
                    for name, idx in video_name_to_idx.items():
                        if gt_clip_normalized.lower() in name.lower() or name.lower() in gt_clip_normalized.lower():
                            gt_idx = idx
                            break
                
                if gt_idx >= 0 and i < similarity_matrix.shape[0]:
                    # Get similarity scores for this segment
                    seg_similarities = similarity_matrix[i]
                    # Rank videos by similarity (descending)
                    ranked_indices = np.argsort(seg_similarities)[::-1]
                    
                    # Find rank of ground truth
                    rank_positions = np.where(ranked_indices == gt_idx)[0]
                    if len(rank_positions) > 0:
                        gt_rank = int(rank_positions[0]) + 1  # 1-indexed
                        reciprocal_rank = 1.0 / gt_rank
                        
                        if gt_rank <= 3:
                            top_3_hits += 1
                        if gt_rank <= 5:
                            top_5_hits += 1
                    
                    # Get ground truth similarity score
                    gt_sim = seg_similarities[gt_idx]
                    gt_similarities.append(gt_sim)
            
            reciprocal_ranks.append(reciprocal_rank)
            predicted_similarities.append(selection.similarity_score)
            
            # Record segment result
            seg_result = SegmentBenchmarkResult(
                segment_id=i,
                segment_text=seg_text[:100] + '...' if len(seg_text) > 100 else seg_text,
                ground_truth_clip=gt_clip_normalized,
                predicted_clip=predicted_clip,
                is_exact_match=is_exact,
                predicted_similarity=selection.similarity_score,
                ground_truth_rank=gt_rank,
                reciprocal_rank=reciprocal_rank
            )
            segment_results.append(asdict(seg_result))
            
            # Track mismatches
            if not is_exact:
                mismatch_details.append({
                    'segment_id': i,
                    'segment_text': seg_text[:60] + '...' if len(seg_text) > 60 else seg_text,
                    'expected': gt_clip_normalized,
                    'predicted': predicted_clip,
                    'predicted_similarity': selection.similarity_score,
                    'ground_truth_rank': gt_rank
                })
        
        # Calculate aggregate metrics
        num_segments = len(clip_selections)
        num_videos = len(all_metadata) if all_metadata else 0
        
        exact_match_accuracy = (exact_matches / num_segments * 100) if num_segments > 0 else 0
        top_3_accuracy = (top_3_hits / num_segments * 100) if num_segments > 0 else 0
        top_5_accuracy = (top_5_hits / num_segments * 100) if num_segments > 0 else 0
        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
        
        avg_pred_sim = np.mean(predicted_similarities) if predicted_similarities else 0
        avg_gt_sim = np.mean(gt_similarities) if gt_similarities else 0
        
        # Calculate correlation if we have both
        sim_correlation = 0.0
        if len(predicted_similarities) == len(gt_similarities) and len(gt_similarities) > 1:
            try:
                sim_correlation = float(np.corrcoef(predicted_similarities, gt_similarities)[0, 1])
                if np.isnan(sim_correlation):
                    sim_correlation = 0.0
            except:
                sim_correlation = 0.0
        
        results = BenchmarkResults(
            benchmark_name=benchmark_name,
            timestamp=datetime.now().isoformat(),
            num_segments=num_segments,
            num_videos=num_videos,
            matching_mode=matching_mode,
            allow_reuse=allow_reuse,
            match_only=match_only,
            exact_match_accuracy=round(exact_match_accuracy, 2),
            top_3_accuracy=round(top_3_accuracy, 2),
            top_5_accuracy=round(top_5_accuracy, 2),
            mean_reciprocal_rank=round(mrr, 4),
            avg_predicted_similarity=round(avg_pred_sim, 4),
            avg_ground_truth_similarity=round(avg_gt_sim, 4),
            similarity_correlation=round(sim_correlation, 4),
            segment_results=segment_results,
            exact_matches=exact_matches,
            mismatches=num_segments - exact_matches,
            mismatch_details=mismatch_details
        )
        
        return results
    
    def _clips_match(self, clip1: str, clip2: str) -> bool:
        """Check if two clip names match (case-insensitive, ignoring extensions)."""
        if not clip1 or not clip2:
            return False
        
        # Normalize: lowercase, remove extension, remove common prefixes/suffixes
        c1 = Path(clip1).stem.lower().strip()
        c2 = Path(clip2).stem.lower().strip()
        
        return c1 == c2
    
    def save_results(self, results: BenchmarkResults, output_dir: Path) -> Path:
        """
        Save benchmark results to files.
        
        Args:
            results: BenchmarkResults object
            output_dir: Directory to save results
            
        Returns:
            Path to the saved JSON file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"benchmark_{results.benchmark_name}_{timestamp}"
        
        # Save full results as JSON
        json_path = output_dir / f"{base_name}.json"
        with open(json_path, 'w') as f:
            json.dump(asdict(results), f, indent=2)
        
        logger.info(f"Saved benchmark results to: {json_path}")
        
        # Save summary as text
        summary_path = output_dir / f"{base_name}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(self._generate_summary_text(results))
        
        logger.info(f"Saved benchmark summary to: {summary_path}")
        
        return json_path
    
    def _generate_summary_text(self, results: BenchmarkResults) -> str:
        """Generate human-readable summary text for file output."""
        lines = [
            "=" * 80,
            "BENCHMARK EVALUATION RESULTS",
            "=" * 80,
            "",
            f"Benchmark Name: {results.benchmark_name}",
            f"Timestamp: {results.timestamp}",
            f"Segments: {results.num_segments}",
            f"Videos: {results.num_videos}",
            f"Matching Mode: {results.matching_mode}",
            f"Allow Reuse: {results.allow_reuse}",
            f"Match-Only Mode: {results.match_only}",
            "",
            "-" * 80,
            "ACCURACY METRICS",
            "-" * 80,
            "",
            f"Exact Match Accuracy:    {results.exact_match_accuracy:6.2f}%  ({results.exact_matches}/{results.num_segments})",
            f"Top-3 Accuracy:          {results.top_3_accuracy:6.2f}%",
            f"Top-5 Accuracy:          {results.top_5_accuracy:6.2f}%",
            f"Mean Reciprocal Rank:    {results.mean_reciprocal_rank:.4f}",
            "",
            "-" * 80,
            "SIMILARITY SCORE ANALYSIS",
            "-" * 80,
            "",
            f"Avg Predicted Similarity:     {results.avg_predicted_similarity:.4f}",
            f"Avg Ground Truth Similarity:  {results.avg_ground_truth_similarity:.4f}",
            f"Similarity Correlation:       {results.similarity_correlation:.4f}",
            "",
        ]
        
        # Add comparison map
        lines.extend([
            "-" * 80,
            "SEGMENT COMPARISON MAP",
            "-" * 80,
            "",
        ])
        
        for seg in results.segment_results:
            match_symbol = "✓" if seg['is_exact_match'] else "✗"
            lines.append(f"[{seg['segment_id']:2d}] {match_symbol} {seg['segment_text'][:50]}")
            lines.append(f"     Predicted:    {seg['predicted_clip']}")
            lines.append(f"     Ground Truth: {seg['ground_truth_clip']}")
            lines.append("")
        
        if results.mismatch_details:
            lines.extend([
                "-" * 80,
                f"MISMATCH DETAILS ({results.mismatches} total)",
                "-" * 80,
                "",
            ])
            for m in results.mismatch_details[:10]:  # Show first 10
                lines.append(f"Segment {m['segment_id']}: {m['segment_text']}")
                lines.append(f"  Expected:  {m['expected']}")
                lines.append(f"  Predicted: {m['predicted']} (sim={m['predicted_similarity']:.4f}, GT rank={m['ground_truth_rank']})")
                lines.append("")
            
            if len(results.mismatch_details) > 10:
                lines.append(f"... and {len(results.mismatch_details) - 10} more mismatches")
        
        lines.extend([
            "",
            "=" * 80,
        ])
        
        return "\n".join(lines)
    
    def print_results(self, results: BenchmarkResults):
        """Print beautiful benchmark results to console."""
        
        # Header
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "BENCHMARK EVALUATION RESULTS" + " " * 30 + "║")
        print("╠" + "═" * 78 + "╣")
        
        # Metadata
        print(f"║  Benchmark: {results.benchmark_name:<64} ║")
        print(f"║  Mode: {results.matching_mode:<10}  │  Reuse: {'Yes' if results.allow_reuse else 'No':<5}  │  Match-Only: {'Yes' if results.match_only else 'No':<5}           ║")
        print(f"║  Segments: {results.num_segments:<5}  │  Videos: {results.num_videos:<5}                                       ║")
        print("╠" + "═" * 78 + "╣")
        
        # Accuracy Metrics with visual bars
        print("║" + " " * 30 + "ACCURACY METRICS" + " " * 32 + "║")
        print("╟" + "─" * 78 + "╢")
        
        # Exact Match with bar
        bar_len = int(results.exact_match_accuracy / 100 * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"║  Exact Match:     {results.exact_match_accuracy:6.2f}%  [{bar}]  ║")
        
        # Top-3 with bar
        bar_len = int(results.top_3_accuracy / 100 * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"║  Top-3 Accuracy:  {results.top_3_accuracy:6.2f}%  [{bar}]  ║")
        
        # Top-5 with bar
        bar_len = int(results.top_5_accuracy / 100 * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"║  Top-5 Accuracy:  {results.top_5_accuracy:6.2f}%  [{bar}]  ║")
        
        # MRR with bar
        bar_len = int(results.mean_reciprocal_rank * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"║  MRR Score:       {results.mean_reciprocal_rank:6.4f}   [{bar}]  ║")
        
        print("╠" + "═" * 78 + "╣")
        
        # Similarity Statistics
        print("║" + " " * 28 + "SIMILARITY STATISTICS" + " " * 29 + "║")
        print("╟" + "─" * 78 + "╢")
        print(f"║  Avg Predicted Similarity:     {results.avg_predicted_similarity:8.4f}                                ║")
        print(f"║  Avg Ground Truth Similarity:  {results.avg_ground_truth_similarity:8.4f}                                ║")
        print(f"║  Correlation:                  {results.similarity_correlation:8.4f}                                ║")
        
        print("╠" + "═" * 78 + "╣")
        
        # Comparison Map Header
        print("║" + " " * 25 + "SEGMENT COMPARISON MAP" + " " * 31 + "║")
        print("╟" + "─" * 78 + "╢")
        print("║  SEG │ MATCH │ SEGMENT TEXT                    │ PREDICTED vs GROUND TRUTH    ║")
        print("╟" + "─" * 78 + "╢")
        
        # Comparison rows
        for seg in results.segment_results:
            match_symbol = " ✓ " if seg['is_exact_match'] else " ✗ "
            seg_text = seg['segment_text'][:30] + "..." if len(seg['segment_text']) > 30 else seg['segment_text']
            pred_clip = seg['predicted_clip'][:12] + ".." if len(seg['predicted_clip']) > 14 else seg['predicted_clip']
            gt_clip = seg['ground_truth_clip'][:12] + ".." if len(seg['ground_truth_clip']) > 14 else seg['ground_truth_clip']
            
            # Pad strings for alignment
            seg_text = f"{seg_text:<33}"
            comparison = f"{pred_clip:<14} │ {gt_clip:<14}"
            
            print(f"║  {seg['segment_id']:3d} │{match_symbol}│ {seg_text}│ {comparison}║")
        
        print("╠" + "═" * 78 + "╣")
        
        # Final Score Summary
        print("║" + " " * 32 + "FINAL SCORE" + " " * 35 + "║")
        print("╟" + "─" * 78 + "╢")
        
        # Big percentage display
        score = results.exact_match_accuracy
        score_str = f"{score:.1f}%"
        padding = (78 - len(score_str) - 20) // 2
        print(f"║{' ' * padding}EXACT MATCH ACCURACY: {score_str}{' ' * (78 - padding - len(score_str) - 22)}║")
        
        # Score interpretation
        if score >= 80:
            interpretation = "EXCELLENT - VideoPrism matching is highly accurate!"
        elif score >= 60:
            interpretation = "GOOD - Matching is working reasonably well."
        elif score >= 40:
            interpretation = "MODERATE - Some improvement needed."
        elif score >= 20:
            interpretation = "POOR - Significant mismatch between predictions and ground truth."
        else:
            interpretation = "VERY POOR - Matching is essentially random."
        
        interp_padding = (78 - len(interpretation)) // 2
        print(f"║{' ' * interp_padding}{interpretation}{' ' * (78 - interp_padding - len(interpretation))}║")
        
        print("╚" + "═" * 78 + "╝")
        print("\n")


def load_ground_truth(file_path: str) -> Dict:
    """
    Load ground truth from a JSON file.
    
    Expected format:
    {
        "name": "benchmark_name",
        "description": "Optional description",
        "mappings": [
            {"segment_id": 0, "clip": "video_filename.mp4"},
            {"segment_id": 1, "clip": "another_video.mp4"},
            ...
        ]
    }
    """
    with open(file_path, 'r') as f:
        return json.load(f)
