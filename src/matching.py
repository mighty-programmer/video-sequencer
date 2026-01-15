"""
Video-Text Matching and Sequencing Engine

This module is responsible for:
1. Matching script segments to video clips using semantic similarity
2. Considering motion, context, and timing constraints
3. Creating an optimal sequence of video clips
4. Handling clip reuse and trimming
"""

import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict
import numpy as np

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


@dataclass\nclass ClipSelection:
    \"\"\"Represents a selected video clip for a script segment\"\"\"
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
    \"\"\"
    Matches script segments to video clips using semantic similarity.
    
    This class uses VideoPrism embeddings to find the best video clips
    for each script segment, considering motion, context, and timing.
    \"\"\"
    
    def __init__(
        self,
        video_indexer,
        model_name: str = 'videoprism_lvt_public_v1_base',
        device: str = 'gpu'
    ):
        \"\"\"
        Initialize the VideoTextMatcher.
        
        Args:
            video_indexer: VideoIndexer instance with indexed videos
            model_name: VideoPrism video-text model to use
            device: Device to use ('gpu' or 'cpu')
        \"\"\"
        self.video_indexer = video_indexer
        self.model_name = model_name
        self.device = device
        
        # Load VideoPrism video-text model
        logger.info(f\"Loading VideoPrism video-text model: {model_name}\")
        if vp is None:
            raise ImportError(\"VideoPrism not installed. Run: pip install videoprism\")
        
        self.flax_model = vp.get_model(model_name)
        self.loaded_state = vp.load_pretrained_weights(model_name)
        self.text_tokenizer = vp.load_text_tokenizer('c4_en')
        
        logger.info(\"VideoTextMatcher initialized successfully\")
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        \"\"\"
        Get embedding for a text query.
        
        Args:
            text: Text query
        
        Returns:
            Text embedding array
        \"\"\"
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
    
    def match_segment_to_videos(
        self,
        segment_text: str,
        segment_duration: float,
        k: int = 5,
        allow_reuse: bool = True,
        used_videos: Optional[Set[str]] = None
    ) -> List[Dict]:
        \"\"\"
        Find the best matching video clips for a script segment.
        
        Args:
            segment_text: Text of the script segment
            segment_duration: Duration of the segment in seconds
            k: Number of candidates to return
            allow_reuse: Whether to allow reusing videos
            used_videos: Set of video IDs already used
        
        Returns:
            List of candidate matches with scores
        \"\"\"
        if used_videos is None:
            used_videos = set()
        
        logger.info(f\"Matching segment: '{segment_text[:50]}...' (duration: {segment_duration:.2f}s)\")
        
        # Get text embedding
        text_embedding = self.get_text_embedding(segment_text)
        
        # Search for similar videos
        results = self.video_indexer.search_by_embedding(text_embedding, k=k*2)
        
        candidates = []
        for video_id, distance, metadata in results:
            # Skip if video already used and reuse not allowed
            if not allow_reuse and video_id in used_videos:
                continue
            
            # Calculate similarity score (lower distance = higher similarity)
            similarity_score = 1.0 / (1.0 + distance)
            
            # Calculate motion score (based on video duration and segment duration)
            motion_score = self._calculate_motion_score(metadata, segment_duration)
            
            # Calculate context score (based on video content relevance)
            context_score = self._calculate_context_score(segment_text, metadata)
            
            # Combined score
            combined_score = (
                0.5 * similarity_score +
                0.3 * motion_score +
                0.2 * context_score
            )
            
            candidates.append({
                'video_id': video_id,
                'file_path': metadata.file_path,
                'duration': metadata.duration,
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
        \"\"\"
        Calculate motion score based on video duration match.
        
        A video is scored higher if its duration is close to the segment duration,
        allowing for trimming without losing important context.
        
        Args:
            metadata: VideoMetadata object
            segment_duration: Duration of the script segment
        
        Returns:
            Motion score (0-1)
        \"\"\"
        video_duration = metadata.duration
        
        # Ideal video is 1.5x to 3x the segment duration
        # (allows for trimming while keeping context)
        ideal_ratio_min = 1.5
        ideal_ratio_max = 3.0
        
        if video_duration == 0:
            return 0.0
        
        ratio = video_duration / segment_duration
        
        if ratio < ideal_ratio_min:
            # Too short, penalize
            return ratio / ideal_ratio_min * 0.5
        elif ratio > ideal_ratio_max:
            # Too long, penalize
            return ideal_ratio_max / ratio * 0.5 + 0.5
        else:
            # In ideal range
            return 1.0
    
    def _calculate_context_score(self, segment_text: str, metadata) -> float:
        \"\"\"
        Calculate context score based on segment keywords.
        
        Args:
            segment_text: Text of the segment
            metadata: VideoMetadata object
        
        Returns:
            Context score (0-1)
        \"\"\"
        # Extract keywords from segment text
        keywords = self._extract_keywords(segment_text)
        
        # For now, return a neutral score
        # In a more advanced implementation, this could use video captions
        # or other metadata to match keywords
        return 0.5
    
    def _extract_keywords(self, text: str) -> List[str]:
        \"\"\"
        Extract keywords from text.
        
        Args:
            text: Input text
        
        Returns:
            List of keywords
        \"\"\"
        # Simple keyword extraction (can be improved with NLP)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}
        
        words = text.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        return keywords
    
    def select_best_clip(
        self,
        candidates: List[Dict],
        segment_duration: float,
        prefer_non_reused: bool = True
    ) -> Optional[Dict]:
        \"\"\"
        Select the best clip from candidates.
        
        Args:
            candidates: List of candidate matches
            segment_duration: Duration of the script segment
            prefer_non_reused: Prefer non-reused clips if possible
        
        Returns:
            Best candidate or None
        \"\"\"
        if not candidates:
            return None
        
        # If prefer_non_reused, filter out reused clips first
        if prefer_non_reused:
            non_reused = [c for c in candidates if not c['is_reused']]
            if non_reused:
                candidates = non_reused
        
        # Select top candidate
        best = candidates[0]
        
        # Calculate trim times
        video_duration = best['duration']
        trim_start, trim_end = self._calculate_trim_times(
            video_duration,
            segment_duration
        )
        
        best['trim_start'] = trim_start
        best['trim_end'] = trim_end
        best['trim_duration'] = trim_end - trim_start
        
        return best
    
    def _calculate_trim_times(
        self,
        video_duration: float,
        segment_duration: float
    ) -> Tuple[float, float]:
        \"\"\"
        Calculate trim start and end times for a video clip.
        
        Args:
            video_duration: Duration of the video
            segment_duration: Duration of the script segment
        
        Returns:
            Tuple of (trim_start, trim_end)
        \"\"\"
        if video_duration <= segment_duration:
            # Video is shorter or equal to segment, use entire video
            return 0.0, video_duration
        
        # Video is longer than segment
        # Try to center the important part (middle of video)
        excess_duration = video_duration - segment_duration
        trim_start = excess_duration / 2
        trim_end = trim_start + segment_duration
        
        return trim_start, trim_end


class SequenceOptimizer:
    \"\"\"
    Optimizes the sequence of video clips.
    
    Considers factors like:
    - Minimizing clip reuse
    - Maintaining visual continuity
    - Ensuring timing accuracy
    \"\"\"
    
    def __init__(self, matcher: VideoTextMatcher):
        \"\"\"
        Initialize the SequenceOptimizer.
        
        Args:
            matcher: VideoTextMatcher instance
        \"\"\"
        self.matcher = matcher
    
    def optimize_sequence(
        self,
        segments: List,
        clip_selections: List[ClipSelection]
    ) -> List[ClipSelection]:
        \"\"\"
        Optimize the sequence of selected clips.
        
        Args:
            segments: List of ScriptSegment objects
            clip_selections: Initial clip selections
        
        Returns:
            Optimized clip selections
        \"\"\"
        logger.info(f\"Optimizing sequence of {len(clip_selections)} clips\")
        
        # Count reuse
        video_usage = {}
        for clip in clip_selections:
            video_usage[clip.video_id] = video_usage.get(clip.video_id, 0) + 1
        
        reused_count = sum(1 for count in video_usage.values() if count > 1)
        logger.info(f\"Clip reuse: {reused_count} videos used multiple times\")
        
        return clip_selections


def create_sequence(
    segments: List,
    matcher: VideoTextMatcher,
    allow_reuse: bool = True,
    prefer_non_reused: bool = True
) -> List[ClipSelection]:
    \"\"\"
    Create a sequence of video clips matching script segments.
    
    Args:
        segments: List of ScriptSegment objects
        matcher: VideoTextMatcher instance
        allow_reuse: Whether to allow reusing videos
        prefer_non_reused: Prefer non-reused clips if possible
    
    Returns:
        List of ClipSelection objects
    \"\"\"
    logger.info(f\"Creating sequence for {len(segments)} segments\")
    
    clip_selections = []
    used_videos: Set[str] = set()
    
    for segment in segments:
        logger.info(f\"Processing segment {segment.segment_id}: {segment.description}\")
        
        # Find matching videos
        candidates = matcher.match_segment_to_videos(
            segment_text=segment.text,
            segment_duration=segment.duration,
            k=5,
            allow_reuse=allow_reuse,
            used_videos=used_videos
        )
        
        if not candidates:
            logger.warning(f\"No candidates found for segment {segment.segment_id}\")
            continue
        
        # Select best clip
        best_clip = matcher.select_best_clip(
            candidates,
            segment.duration,
            prefer_non_reused=prefer_non_reused
        )
        
        if best_clip:
            is_reused = best_clip['video_id'] in used_videos
            
            clip_selection = ClipSelection(
                segment_id=segment.segment_id,
                video_id=best_clip['video_id'],
                video_file_path=best_clip['file_path'],
                start_time=segment.start_time,
                end_time=segment.end_time,
                duration=segment.duration,
                trim_start=best_clip['trim_start'],
                trim_end=best_clip['trim_end'],
                trim_duration=best_clip['trim_duration'],
                similarity_score=best_clip['similarity_score'],
                motion_score=best_clip['motion_score'],
                context_score=best_clip['context_score'],
                combined_score=best_clip['combined_score'],
                is_reused=is_reused
            )
            
            clip_selections.append(clip_selection)
            used_videos.add(best_clip['video_id'])
            
            logger.info(f\"Selected {best_clip['video_id']} (score: {best_clip['combined_score']:.3f})\")
    
    logger.info(f\"Created sequence with {len(clip_selections)} clips\")
    return clip_selections


if __name__ == '__main__':
    # Example usage would require initialized indexer and matcher
    pass
