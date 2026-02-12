"""
Video-Text Matching Module using VideoPrism LVT with Sub-clip Precision

This module handles:
1. Matching script segments to video sub-clips
2. Calculating combined scores (similarity + motion + context)
3. Determining precise trim times centered around the best-matching window
4. Global optimization using the Hungarian Algorithm
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

logger = logging.getLogger(__name__)

@dataclass
class ClipSelection:
    """Data class for a selected video clip for a segment"""
    segment_id: int
    video_id: str
    video_file_path: str
    start_time: float  # Start time in the final video
    end_time: float    # End time in the final video
    duration: float    # Duration in the final video
    trim_start: float  # Start time in the source video
    trim_end: float    # End time in the source video
    trim_duration: float
    similarity_score: float
    motion_score: float
    context_score: float
    combined_score: float
    is_reused: bool = False
    window_start: float = 0.0 # The start of the matching window
    window_end: float = 0.0   # The end of the matching window


class VideoTextMatcher:
    """Matches script segments to video sub-clips."""
    
    def __init__(self, indexer):
        self.indexer = indexer
        self.used_videos_history = []

    def get_all_video_metadata(self):
        return self.indexer.metadata_list

    def compute_similarity_matrix(self, script_segments: List[Dict], match_only: bool = False) -> Tuple[np.ndarray, List]:
        """Compute similarity matrix between segments and all indexed sub-clips."""
        all_metadata = self.get_all_video_metadata()
        num_segments = len(script_segments)
        num_videos = len(all_metadata)
        
        if num_videos == 0:
            return np.zeros((num_segments, 0)), []
            
        matrix = np.zeros((num_segments, num_videos))
        
        for i, segment in enumerate(script_segments):
            try:
                results = self.indexer.search_by_text(segment['text'], k=num_videos)
                # Map results back to matrix
                # Use a dictionary for fast lookup by video_id
                meta_to_score = {res[0].video_id: res[1] for res in results}
                for j, meta in enumerate(all_metadata):
                    matrix[i, j] = meta_to_score.get(meta.video_id, -1.0)
            except Exception as e:
                logger.error(f"Error computing similarity for segment {i}: {e}")
                # Fill with -1.0 for this segment if search fails
                matrix[i, :] = -1.0
                
        return matrix, all_metadata

    def match_segment_to_videos(self, segment_text: str, segment_duration: float, used_videos=None, match_only: bool = False, k: int = 10) -> List[Dict]:
        """Find top candidates for a segment."""
        if used_videos is None: used_videos = set()
        
        results = self.indexer.search_by_text(segment_text, k=k*2)
        candidates = []
        
        for metadata, similarity in results:
            similarity_score = (similarity + 1.0) / 2.0
            
            if match_only:
                combined_score = similarity_score
                motion_score = 0.0
                context_score = 0.0
            else:
                diversity_multiplier = 1.2 if metadata.video_id not in used_videos else 0.8
                motion_score = self._calculate_motion_score(metadata, segment_duration)
                context_score = self._calculate_context_score(segment_text, metadata)
                
                combined_score = (
                    0.5 * similarity_score * diversity_multiplier +
                    0.3 * motion_score +
                    0.2 * context_score
                )
            
            candidates.append({
                'video_id': metadata.video_id,
                'file_path': metadata.file_path,
                'duration': metadata.duration,
                'start_time': metadata.start_time,
                'end_time': metadata.end_time,
                'similarity': similarity,
                'similarity_score': similarity_score,
                'motion_score': motion_score,
                'context_score': context_score,
                'combined_score': combined_score,
                'is_reused': metadata.video_id in used_videos
            })
            
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        return candidates[:k]

    def _calculate_motion_score(self, metadata, segment_duration: float) -> float:
        video_duration = metadata.duration
        if video_duration == 0 or segment_duration == 0: return 0.5
        ratio = video_duration / segment_duration
        if ratio >= 1.0: return 1.0 if ratio <= 3.0 else 0.7
        return max(0.3, ratio)

    def _calculate_context_score(self, segment_text: str, metadata) -> float:
        import re
        words = re.findall(r'\b[a-z]{3,}\b', segment_text.lower())
        stop_words = {'the', 'and', 'this', 'that', 'with', 'from', 'your', 'you', 'for', 'are', 'was'}
        keywords = [w for w in words if w not in stop_words]
        video_name = Path(metadata.file_path).name.lower()
        matches = sum(1 for kw in keywords if kw in video_name)
        return min(1.0, 0.5 + matches * 0.25) if matches > 0 else 0.5

    def select_best_clip(self, candidates: List[Dict], segment_duration: float, match_only: bool = False) -> Optional[Dict]:
        if not candidates: return None
        best = candidates[0]
        
        if match_only:
            trim_start, trim_end = best['start_time'], best['end_time']
        else:
            trim_start, trim_end = self._calculate_trim_times(best, segment_duration)
            
        best['trim_start'] = trim_start
        best['trim_end'] = trim_end
        best['trim_duration'] = trim_end - trim_start
        return best

    def _calculate_trim_times(self, metadata_dict: Dict, segment_duration: float) -> Tuple[float, float]:
        """Calculate trim times centered around the matching window."""
        video_duration = metadata_dict['duration']
        win_start = metadata_dict['start_time']
        win_end = metadata_dict['end_time']
        win_center = (win_start + win_end) / 2
        
        if video_duration <= segment_duration:
            return 0.0, video_duration
            
        half_dur = segment_duration / 2
        trim_start = win_center - half_dur
        trim_end = win_center + half_dur
        
        # Adjust if out of bounds
        if trim_start < 0:
            trim_start = 0.0
            trim_end = segment_duration
        elif trim_end > video_duration:
            trim_end = video_duration
            trim_start = video_duration - segment_duration
            
        return trim_start, trim_end


def create_sequence_optimal(script_segments, video_matcher, match_only=False, allow_reuse=True):
    if linear_sum_assignment is None:
        return create_sequence_greedy(script_segments, video_matcher, match_only, allow_reuse)
        
    similarity_matrix, all_metadata = video_matcher.compute_similarity_matrix(script_segments, match_only)
    num_segments, num_videos = similarity_matrix.shape
    
    cost_matrix = -similarity_matrix
    if num_segments > num_videos:
        cost_matrix = np.hstack([cost_matrix, np.full((num_segments, num_segments - num_videos), 1e6)])
    if num_videos > num_segments:
        cost_matrix = np.vstack([cost_matrix, np.full((num_videos - num_segments, cost_matrix.shape[1]), 1e6)])
        
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    sequence = []
    used_videos = set()
    for i in range(num_segments):
        vid_idx = col_ind[i]
        if vid_idx >= num_videos: # Fallback
            meta = all_metadata[i % num_videos]
            score = 0.0
        else:
            meta = all_metadata[vid_idx]
            score = similarity_matrix[i, vid_idx]
            
        if match_only:
            trim_start, trim_end = meta.start_time, meta.end_time
        else:
            trim_start, trim_end = video_matcher._calculate_trim_times(
                {'duration': meta.duration, 'start_time': meta.start_time, 'end_time': meta.end_time},
                script_segments[i]['duration']
            )
            
        selection = ClipSelection(
            segment_id=i, video_id=meta.video_id, video_file_path=meta.file_path,
            start_time=0.0, end_time=0.0, duration=script_segments[i]['duration'],
            trim_start=trim_start, trim_end=trim_end, trim_duration=trim_end - trim_start,
            similarity_score=score, motion_score=0.0, context_score=0.0, combined_score=score,
            is_reused=meta.video_id in used_videos, window_start=meta.start_time, window_end=meta.end_time
        )
        sequence.append(selection)
        used_videos.add(meta.video_id)
        
    return sequence

def create_sequence_greedy(script_segments, video_matcher, match_only=False, allow_reuse=True):
    sequence = []
    used_videos = set()
    for i, segment in enumerate(script_segments):
        candidates = video_matcher.match_segment_to_videos(
            segment['text'], segment['duration'], 
            used_videos=used_videos if not allow_reuse else None,
            match_only=match_only
        )
        best = video_matcher.select_best_clip(candidates, segment['duration'], match_only)
        if not best: # Fallback
            all_meta = video_matcher.get_all_video_metadata()
            meta = all_meta[i % len(all_meta)]
            best = {'video_id': meta.video_id, 'file_path': meta.file_path, 'duration': meta.duration,
                    'start_time': meta.start_time, 'end_time': meta.end_time, 'similarity': 0.0}
            best['trim_start'], best['trim_end'] = 0.0, meta.duration
            best['trim_duration'] = meta.duration
            
        selection = ClipSelection(
            segment_id=i, video_id=best['video_id'], video_file_path=best['file_path'],
            start_time=0.0, end_time=0.0, duration=segment['duration'],
            trim_start=best['trim_start'], trim_end=best['trim_end'], trim_duration=best['trim_duration'],
            similarity_score=best.get('similarity', 0.0), motion_score=0.0, context_score=0.0, combined_score=0.0,
            is_reused=best['video_id'] in used_videos, window_start=best.get('start_time', 0.0), window_end=best.get('end_time', 0.0)
        )
        sequence.append(selection)
        used_videos.add(selection.video_id)
    return sequence

def create_sequence(script_segments, video_matcher, match_only=False, allow_reuse=True, use_optimal=True):
    if use_optimal:
        return create_sequence_optimal(script_segments, video_matcher, match_only, allow_reuse)
    return create_sequence_greedy(script_segments, video_matcher, match_only, allow_reuse)
