"""
Video-Text Matching and Sequencing Engine

This module is responsible for:
1. Matching script segments to video clips using semantic similarity
2. Considering motion, context, and timing constraints
3. Creating an optimal sequence of video clips
4. Handling clip reuse and trimming
"""

import logging
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path

try:
    from videoprism import models as vp
except ImportError:
    vp = None

try:
    import faiss
except ImportError:
    faiss = None


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class ClipSelection:
    """Represents a selected video clip for a script segment"""
    segment_id: int
    video_id: str
    video_file_path: str
    start_time: float
    end_time: float
    duration: float
    trim_start: float
    trim_end: float
    trim_duration: float
    similarity_score: float
    motion_score: float
    context_score: float
    combined_score: float
    is_reused: bool = False
    original_use_segment_id: Optional[int] = None


class VideoTextMatcher:
    """
    Matches script segments to video clips using semantic similarity.
    
    This class uses VideoPrism embeddings to find the best video clips
    for each script segment, considering motion, context, and timing.
    """
    
    def __init__(
        self,
        video_indexer,
        model_name: str = 'videoprism_lvt_public_v1_base',
        device: str = 'gpu',
        min_similarity_threshold: float = 0.01
    ):
        """
        Initialize the VideoTextMatcher.
        
        Args:
            video_indexer: VideoIndexer instance with indexed videos
            model_name: VideoPrism video-text model to use
            device: Device to use ('gpu' or 'cpu')
            min_similarity_threshold: Minimum similarity score to consider a match
        """
        self.video_indexer = video_indexer
        self.model_name = model_name
        self.device = device
        self.min_similarity_threshold = min_similarity_threshold
        self.used_videos_history = [] # Track order of used videos
        
        # Load VideoPrism video-text model
        logger.info(f"Loading VideoPrism video-text model: {model_name}")
        if vp is None:
            raise ImportError("VideoPrism not installed. Run: pip install videoprism")
        
        self.flax_model = vp.get_model(model_name)
        self.loaded_state = vp.load_pretrained_weights(model_name)
        self.text_tokenizer = vp.load_text_tokenizer('c4_en')
        
        logger.info("VideoTextMatcher initialized successfully")
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text query.
        """
        import jax
        
        text_ids, text_paddings = vp.tokenize_texts(self.text_tokenizer, [text])
        
        @jax.jit
        def forward_fn(text_ids, text_paddings):
            _, text_embeddings, _ = self.flax_model.apply(
                self.loaded_state,
                None,
                text_ids,
                text_paddings,
                train=False
            )
            return text_embeddings
        
        text_embeddings = forward_fn(text_ids, text_paddings)
        return np.array(text_embeddings[0])
    
    def get_num_indexed_videos(self) -> int:
        """Get the number of indexed videos."""
        if hasattr(self.video_indexer, 'metadata_list'):
            return len(self.video_indexer.metadata_list)
        return 0
    
    def get_all_video_metadata(self) -> List:
        """Get all video metadata from the indexer."""
        if hasattr(self.video_indexer, 'metadata_list'):
            return self.video_indexer.metadata_list
        return []
    
    def match_segment_to_videos(
        self,
        segment_text: str,
        segment_duration: float,
        k: int = 10,
        allow_reuse: bool = True,
        used_videos: Optional[Set[str]] = None
    ) -> List[Dict]:
        """
        Find the best matching video clips for a script segment.
        """
        if used_videos is None:
            used_videos = set()
        
        logger.info(f"Matching segment: '{segment_text[:50]}...' (duration: {segment_duration:.2f}s)")
        
        # Get text embedding
        text_embedding = self.get_text_embedding(segment_text)
        
        # Get number of indexed videos
        num_videos = self.get_num_indexed_videos()
        if num_videos == 0:
            logger.error("No videos indexed! Cannot match segments.")
            return []
        
        # Search for more candidates to allow for diversity filtering
        search_k = min(num_videos, k * 5)
        results = self.video_indexer.search_by_embedding(text_embedding, k=search_k)
        
        candidates = []
        for video_id, distance, metadata in results:
            # Calculate similarity score using exponential decay
            similarity_score = float(np.exp(-distance / 10.0))
            
            # Boost score if video hasn't been used yet (Diversity Boost)
            diversity_multiplier = 1.2 if video_id not in used_videos else 0.8
            
            # Penalize if it was the VERY LAST video used (Sequential Diversity)
            if self.used_videos_history and self.used_videos_history[-1] == video_id:
                diversity_multiplier *= 0.5
            
            # Calculate motion score
            motion_score = self._calculate_motion_score(metadata, segment_duration)
            
            # Calculate context score
            context_score = self._calculate_context_score(segment_text, metadata)
            
            # Combined score
            combined_score = (
                0.5 * similarity_score * diversity_multiplier +
                0.3 * motion_score +
                0.2 * context_score
            )
            
            candidates.append({
                'video_id': video_id,
                'file_path': metadata.file_path,
                'duration': metadata.duration,
                'distance': distance,
                'similarity_score': similarity_score,
                'motion_score': motion_score,
                'context_score': context_score,
                'combined_score': combined_score,
                'is_reused': video_id in used_videos
            })
        
        # Sort by combined score
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        return candidates[:k]
    
    def _calculate_motion_score(self, metadata, segment_duration: float) -> float:
        video_duration = metadata.duration
        if video_duration == 0 or segment_duration == 0:
            return 0.5
        ratio = video_duration / segment_duration
        if ratio >= 1.0:
            return 1.0 if ratio <= 3.0 else 0.7
        return max(0.3, ratio)
    
    def _calculate_context_score(self, segment_text: str, metadata) -> float:
        keywords = self._extract_keywords(segment_text)
        video_name = Path(metadata.file_path).name.lower()
        matches = sum(1 for kw in keywords if kw in video_name)
        return min(1.0, 0.5 + matches * 0.25) if matches > 0 else 0.5
    
    def _extract_keywords(self, text: str) -> List[str]:
        import re
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        stop_words = {'the', 'and', 'this', 'that', 'with', 'from', 'they', 'have'}
        return [w for w in words if w not in stop_words]
    
    def select_best_clip(
        self,
        candidates: List[Dict],
        segment_duration: float
    ) -> Optional[Dict]:
        if not candidates:
            return None
        
        best = candidates[0]
        trim_start, trim_end = self._calculate_trim_times(best['duration'], segment_duration)
        
        best['trim_start'] = trim_start
        best['trim_end'] = trim_end
        best['trim_duration'] = trim_end - trim_start
        
        # Update history
        self.used_videos_history.append(best['video_id'])
        
        return best
    
    def _calculate_trim_times(self, video_duration: float, segment_duration: float) -> Tuple[float, float]:
        if video_duration <= segment_duration:
            return 0.0, video_duration
        # Center trim
        trim_start = (video_duration - segment_duration) / 2
        return trim_start, trim_start + segment_duration


def create_sequence(
    script_segments: List[Dict],
    video_matcher: VideoTextMatcher
) -> List[ClipSelection]:
    sequence = []
    used_videos = set()
    all_metadata = video_matcher.get_all_video_metadata()
    
    for i, segment in enumerate(script_segments):
        logger.info(f"Processing segment {i+1}/{len(script_segments)}")
        
        # 1. Try to find a good match
        candidates = video_matcher.match_segment_to_videos(
            segment['text'], 
            segment['duration'], 
            used_videos=used_videos
        )
        
        best_clip_data = video_matcher.select_best_clip(candidates, segment['duration'])
        
        # 2. If no candidates, use Best-Effort Fallback (Diversity-aware)
        if not best_clip_data:
            logger.warning(f"  -> No candidates for segment {i+1}, using best-effort fallback")
            # Pick a video that hasn't been used recently
            unused = [m for m in all_metadata if m.video_id not in used_videos]
            fallback_meta = unused[0] if unused else all_metadata[i % len(all_metadata)]
            
            trim_start, trim_end = video_matcher._calculate_trim_times(fallback_meta.duration, segment['duration'])
            
            selection = ClipSelection(
                segment_id=i,
                video_id=fallback_meta.video_id,
                video_file_path=fallback_meta.file_path,
                start_time=0.0, end_time=0.0, # Set later
                duration=segment['duration'],
                trim_start=trim_start, trim_end=trim_end,
                trim_duration=trim_end - trim_start,
                similarity_score=0.1, motion_score=0.5, context_score=0.5, combined_score=0.3
            )
        else:
            selection = ClipSelection(
                segment_id=i,
                video_id=best_clip_data['video_id'],
                video_file_path=best_clip_data['file_path'],
                start_time=0.0, end_time=0.0,
                duration=segment['duration'],
                trim_start=best_clip_data['trim_start'],
                trim_end=best_clip_data['trim_end'],
                trim_duration=best_clip_data['trim_duration'],
                similarity_score=best_clip_data['similarity_score'],
                motion_score=best_clip_data['motion_score'],
                context_score=best_clip_data['context_score'],
                combined_score=best_clip_data['combined_score'],
                is_reused=best_clip_data['is_reused']
            )
            
        sequence.append(selection)
        used_videos.add(selection.video_id)
        logger.info(f"  -> Selected: {selection.video_id} (Score: {selection.combined_score:.3f})")
    
    # Set timeline
    current_time = 0.0
    for s in sequence:
        s.start_time = current_time
        s.end_time = current_time + s.duration
        current_time += s.duration
        
    return sequence
