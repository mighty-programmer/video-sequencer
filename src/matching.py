"""
Video-Text Matching and Sequencing Engine

This module is responsible for:
1. Matching script segments to video clips using semantic similarity
2. Using VideoPrism LVT for text embeddings (compatible with video embeddings)
3. Considering motion, context, and timing constraints
4. Creating an optimal sequence of video clips with diversity
5. Handling clip reuse and trimming

IMPORTANT: This module uses the same LVT model as indexing.py to ensure
video and text embeddings are in the same embedding space.
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
    
    This class uses VideoPrism LVT embeddings to find the best video clips
    for each script segment. The LVT model produces global embeddings for
    both video and text that can be compared using cosine similarity.
    """
    
    def __init__(
        self,
        video_indexer,
        model_name: str = 'videoprism_lvt_public_v1_base',
        device: str = 'gpu',
        min_similarity_threshold: float = 0.0  # Lowered since we use cosine similarity
    ):
        """
        Initialize the VideoTextMatcher.
        
        Args:
            video_indexer: VideoIndexer instance with indexed videos
            model_name: VideoPrism LVT model to use (must match indexer model)
            device: Device to use ('gpu' or 'cpu')
            min_similarity_threshold: Minimum similarity score to consider a match
        """
        self.video_indexer = video_indexer
        self.model_name = model_name
        self.device = device
        self.min_similarity_threshold = min_similarity_threshold
        self.used_videos_history = []  # Track order of used videos
        
        # Load VideoPrism LVT model for text embeddings
        logger.info(f"Loading VideoPrism video-text model: {model_name}")
        if vp is None:
            raise ImportError("VideoPrism not installed. Run: pip install videoprism")
        
        self.flax_model = vp.get_model(model_name)
        self.loaded_state = vp.load_pretrained_weights(model_name)
        self.text_tokenizer = vp.load_text_tokenizer('c4_en')
        
        logger.info("VideoTextMatcher initialized successfully")
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get GLOBAL embedding for a text query using LVT model.
        
        Args:
            text: Text query string
            
        Returns:
            Embedding array with shape (feature_channels,) - typically (768,)
        """
        import jax
        
        text_ids, text_paddings = vp.tokenize_texts(self.text_tokenizer, [text])
        
        @jax.jit
        def forward_fn(text_ids, text_paddings):
            # For LVT model: pass None for video to get only text embeddings
            _, text_embeddings, _ = self.flax_model.apply(
                self.loaded_state,
                None,  # video_inputs = None
                text_ids,
                text_paddings,
                train=False
            )
            return text_embeddings
        
        text_embeddings = forward_fn(text_ids, text_paddings)
        return np.array(text_embeddings[0])  # Shape: (768,)
    
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
        used_videos: Optional[Set[str]] = None,
        match_only: bool = False
    ) -> List[Dict]:
        """
        Find the best matching video clips for a script segment.
        
        Args:
            segment_text: The text of the script segment
            segment_duration: Duration of the segment in seconds
            k: Number of top candidates to return
            allow_reuse: Whether to allow reusing videos
            used_videos: Set of already used video IDs
            match_only: If True, ignore duration compatibility and diversity penalties
            
        Returns:
            List of candidate dictionaries sorted by combined score
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
        
        if not results:
            logger.warning("No results from FAISS search")
            return []
        
        candidates = []
        for video_id, similarity, metadata in results:
            # similarity is now cosine similarity (higher is better, range [-1, 1])
            # Normalize to [0, 1] for scoring
            similarity_score = (similarity + 1.0) / 2.0
            
            if match_only:
                # In match-only mode, we only care about semantic similarity and context
                motion_score = 1.0
                diversity_multiplier = 1.0
                context_score = self._calculate_context_score(segment_text, metadata)
                combined_score = (0.8 * similarity_score) + (0.2 * context_score)
            else:
                # Boost score if video hasn't been used yet (Diversity Boost)
                diversity_multiplier = 1.2 if video_id not in used_videos else 0.8
                
                # Penalize if it was the VERY LAST video used (Sequential Diversity)
                if self.used_videos_history and self.used_videos_history[-1] == video_id:
                    diversity_multiplier *= 0.5
                
                # Calculate motion score (duration compatibility)
                motion_score = self._calculate_motion_score(metadata, segment_duration)
                
                # Calculate context score (filename keyword matching)
                context_score = self._calculate_context_score(segment_text, metadata)
                
                # Combined score with weighted factors
                combined_score = (
                    0.5 * similarity_score * diversity_multiplier +
                    0.3 * motion_score +
                    0.2 * context_score
                )
            
            candidates.append({
                'video_id': video_id,
                'file_path': metadata.file_path,
                'duration': metadata.duration,
                'similarity': similarity,  # Raw cosine similarity
                'similarity_score': similarity_score,  # Normalized [0, 1]
                'motion_score': motion_score,
                'context_score': context_score,
                'combined_score': combined_score,
                'is_reused': video_id in used_videos
            })
        
        # Sort by combined score (higher is better)
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Log top candidates for debugging
        if candidates:
            logger.debug(f"Top 3 candidates:")
            for i, c in enumerate(candidates[:3]):
                logger.debug(f"  {i+1}. {c['video_id']}: sim={c['similarity']:.3f}, "
                           f"combined={c['combined_score']:.3f}")
        
        return candidates[:k]
    
    def _calculate_motion_score(self, metadata, segment_duration: float) -> float:
        """Calculate score based on video duration vs segment duration."""
        video_duration = metadata.duration
        if video_duration == 0 or segment_duration == 0:
            return 0.5
        
        ratio = video_duration / segment_duration
        
        # Prefer videos that are at least as long as the segment
        if ratio >= 1.0:
            # Video is long enough, slight penalty for very long videos
            return 1.0 if ratio <= 3.0 else 0.7
        else:
            # Video is shorter than segment, penalize proportionally
            return max(0.3, ratio)
    
    def _calculate_context_score(self, segment_text: str, metadata) -> float:
        """Calculate score based on keyword matches in filename."""
        keywords = self._extract_keywords(segment_text)
        video_name = Path(metadata.file_path).name.lower()
        
        # Count keyword matches
        matches = sum(1 for kw in keywords if kw in video_name)
        
        # Return score based on matches
        if matches > 0:
            return min(1.0, 0.5 + matches * 0.25)
        return 0.5
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        import re
        
        # Extract words with 3+ characters
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'this', 'that', 'with', 'from', 'they', 'have',
            'been', 'were', 'will', 'would', 'could', 'should', 'your',
            'you', 'for', 'are', 'was', 'not', 'but', 'what', 'all',
            'can', 'had', 'her', 'his', 'him', 'has', 'its', 'just',
            'into', 'over', 'such', 'than', 'then', 'them', 'these',
            'some', 'very', 'when', 'where', 'which', 'while', 'who',
            'why', 'how', 'each', 'she', 'does', 'doing', 'being'
        }
        
        return [w for w in words if w not in stop_words]
    
    def select_best_clip(
        self,
        candidates: List[Dict],
        segment_duration: float,
        match_only: bool = False
    ) -> Optional[Dict]:
        """
        Select the best clip from candidates and calculate trim times.
        
        Args:
            candidates: List of candidate dictionaries
            segment_duration: Duration needed for the segment
            match_only: If True, use full video duration instead of segment duration
            
        Returns:
            Best candidate with trim times added, or None if no candidates
        """
        if not candidates:
            return None
        
        best = candidates[0]
        
        if match_only:
            # In match-only mode, we use the full video
            trim_start = 0.0
            trim_end = best['duration']
        else:
            trim_start, trim_end = self._calculate_trim_times(best['duration'], segment_duration)
        
        best['trim_start'] = trim_start
        best['trim_end'] = trim_end
        best['trim_duration'] = trim_end - trim_start
        
        # Update history for diversity tracking
        self.used_videos_history.append(best['video_id'])
        
        return best
    
    def _calculate_trim_times(
        self, 
        video_duration: float, 
        segment_duration: float
    ) -> Tuple[float, float]:
        """Calculate trim start and end times for a video clip."""
        if video_duration <= segment_duration:
            # Video is shorter than or equal to segment, use full video
            return 0.0, video_duration
        
        # Video is longer, center-crop to segment duration
        trim_start = (video_duration - segment_duration) / 2
        return trim_start, trim_start + segment_duration


def create_sequence(
    script_segments: List[Dict],
    video_matcher: VideoTextMatcher,
    match_only: bool = False
) -> List[ClipSelection]:
    """
    Create a sequence of video clips matched to script segments.
    
    Args:
        script_segments: List of segment dictionaries with 'text' and 'duration'
        video_matcher: VideoTextMatcher instance
        match_only: If True, bypass duration constraints and diversity penalties
        
    Returns:
        List of ClipSelection objects representing the video sequence
    """
    sequence = []
    used_videos = set()
    all_metadata = video_matcher.get_all_video_metadata()
    
    if not all_metadata:
        logger.error("No video metadata available!")
        return []
    
    for i, segment in enumerate(script_segments):
        logger.info(f"Processing segment {i+1}/{len(script_segments)}")
        
        # 1. Try to find a good match using semantic similarity
        candidates = video_matcher.match_segment_to_videos(
            segment['text'], 
            segment['duration'], 
            used_videos=used_videos,
            match_only=match_only
        )
        
        best_clip_data = video_matcher.select_best_clip(
            candidates, 
            segment['duration'],
            match_only=match_only
        )
        
        # 2. If no candidates found, use diversity-aware fallback
        if not best_clip_data:
            logger.warning(f"  -> No candidates for segment {i+1}, using best-effort fallback")
            
            # Pick a video that hasn't been used recently
            unused = [m for m in all_metadata if m.video_id not in used_videos]
            fallback_meta = unused[0] if unused else all_metadata[i % len(all_metadata)]
            
            if match_only:
                trim_start, trim_end = 0.0, fallback_meta.duration
            else:
                trim_start, trim_end = video_matcher._calculate_trim_times(
                    fallback_meta.duration, segment['duration']
                )
            
            selection = ClipSelection(
                segment_id=i,
                video_id=fallback_meta.video_id,
                video_file_path=fallback_meta.file_path,
                start_time=0.0, end_time=0.0,  # Set later
                duration=segment['duration'] if not match_only else fallback_meta.duration,
                trim_start=trim_start, 
                trim_end=trim_end,
                trim_duration=trim_end - trim_start,
                similarity_score=0.1, 
                motion_score=0.5, 
                context_score=0.5, 
                combined_score=0.3
            )
        else:
            selection = ClipSelection(
                segment_id=i,
                video_id=best_clip_data['video_id'],
                video_file_path=best_clip_data['file_path'],
                start_time=0.0, end_time=0.0,
                duration=segment['duration'] if not match_only else best_clip_data['duration'],
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
    
    # Calculate start/end times in the final sequence
    current_time = 0.0
    for selection in sequence:
        selection.start_time = current_time
        selection.end_time = current_time + selection.trim_duration
        current_time = selection.end_time
        
    return sequence
