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
        min_similarity_threshold: float = 0.0  # No minimum threshold by default
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
        
        Args:
            text: Text query
        
        Returns:
            Text embedding array
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
        
        Args:
            segment_text: Text of the script segment
            segment_duration: Duration of the segment in seconds
            k: Number of candidates to return
            allow_reuse: Whether to allow reusing videos
            used_videos: Set of video IDs already used
        
        Returns:
            List of candidate matches with scores
        """
        if used_videos is None:
            used_videos = set()
        
        logger.info(f"Matching segment: '{segment_text[:50]}...' (duration: {segment_duration:.2f}s)")
        
        # Get text embedding
        text_embedding = self.get_text_embedding(segment_text)
        
        # Search for similar videos - get more candidates to ensure we find matches
        num_videos = len(self.video_indexer.metadata) if hasattr(self.video_indexer, 'metadata') else 100
        search_k = min(num_videos, k * 5)  # Get more candidates
        results = self.video_indexer.search_by_embedding(text_embedding, k=search_k)
        
        candidates = []
        for video_id, distance, metadata in results:
            # Skip if video already used and reuse not allowed
            if not allow_reuse and video_id in used_videos:
                continue
            
            # Calculate similarity score using exponential decay
            # This gives better scores for closer matches
            similarity_score = np.exp(-distance / 10.0)  # Softer decay
            
            # Ensure minimum score for any video
            similarity_score = max(similarity_score, 0.1)
            
            # Calculate motion score (based on video duration and segment duration)
            motion_score = self._calculate_motion_score(metadata, segment_duration)
            
            # Calculate context score (based on video content relevance)
            context_score = self._calculate_context_score(segment_text, metadata)
            
            # Combined score with adjusted weights
            combined_score = (
                0.4 * similarity_score +
                0.35 * motion_score +
                0.25 * context_score
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
        
        # Log top candidates for debugging
        if candidates:
            logger.debug(f"Top 3 candidates:")
            for i, c in enumerate(candidates[:3]):
                logger.debug(f"  {i+1}. {c['video_id']}: sim={c['similarity_score']:.3f}, motion={c['motion_score']:.3f}, combined={c['combined_score']:.3f}")
        
        return candidates[:k]
    
    def _calculate_motion_score(self, metadata, segment_duration: float) -> float:
        """
        Calculate motion score based on video duration match.
        
        A video is scored higher if its duration is close to or longer than
        the segment duration, allowing for trimming without losing important context.
        
        Args:
            metadata: VideoMetadata object
            segment_duration: Duration of the script segment
        
        Returns:
            Motion score (0-1)
        """
        video_duration = metadata.duration
        
        if video_duration == 0 or segment_duration == 0:
            return 0.5  # Neutral score for edge cases
        
        ratio = video_duration / segment_duration
        
        if ratio >= 1.0:
            # Video is longer than or equal to segment - good!
            # Score decreases slightly for very long videos
            if ratio <= 2.0:
                return 1.0
            elif ratio <= 5.0:
                return 0.9
            else:
                return 0.7
        else:
            # Video is shorter than segment - can still use it
            # Score based on how much of the segment it covers
            return max(0.3, ratio * 0.8)
    
    def _calculate_context_score(self, segment_text: str, metadata) -> float:
        """
        Calculate context score based on segment keywords.
        
        Args:
            segment_text: Text of the segment
            metadata: VideoMetadata object
        
        Returns:
            Context score (0-1)
        """
        # Extract keywords from segment text
        keywords = self._extract_keywords(segment_text)
        
        # Check if video filename contains any keywords
        video_name = metadata.file_path.lower() if hasattr(metadata, 'file_path') else ''
        
        matches = sum(1 for kw in keywords if kw in video_name)
        
        if matches > 0:
            return min(1.0, 0.5 + matches * 0.2)
        
        # Return neutral score if no keyword matches
        return 0.5
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
        
        Returns:
            List of keywords
        """
        # Simple keyword extraction (can be improved with NLP)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'this', 'that', 'these', 'those', 'it', 'its', 'you', 'your', 'we', 'our',
            'they', 'their', 'i', 'my', 'me', 'he', 'she', 'him', 'her', 'with', 'from',
            'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
            'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'can', 'now', 'also', 'like', 'even', 'because',
            'dont', "don't", 'got', 'get', 'getting', 'going', 'go', 'goes'
        }
        
        # Clean and split text
        import re
        words = re.findall(r'\b[a-z]+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def select_best_clip(
        self,
        candidates: List[Dict],
        segment_duration: float,
        prefer_non_reused: bool = True
    ) -> Optional[Dict]:
        """
        Select the best clip from candidates.
        
        Args:
            candidates: List of candidate matches
            segment_duration: Duration of the script segment
            prefer_non_reused: Prefer non-reused clips if possible
        
        Returns:
            Best candidate or None
        """
        if not candidates:
            logger.warning("No candidates provided to select_best_clip")
            return None
        
        # If prefer_non_reused, filter out reused clips first
        working_candidates = candidates.copy()
        if prefer_non_reused:
            non_reused = [c for c in working_candidates if not c['is_reused']]
            if non_reused:
                working_candidates = non_reused
        
        # Select top candidate - ALWAYS return a match if we have candidates
        best = working_candidates[0]
        
        # Calculate trim times
        video_duration = best['duration']
        trim_start, trim_end = self._calculate_trim_times(
            video_duration,
            segment_duration
        )
        
        best['trim_start'] = trim_start
        best['trim_end'] = trim_end
        best['trim_duration'] = trim_end - trim_start
        
        logger.info(f"  Selected: {best['video_id']} (score: {best['combined_score']:.3f}, trim: {trim_start:.2f}-{trim_end:.2f}s)")
        
        return best
    
    def _calculate_trim_times(
        self,
        video_duration: float,
        segment_duration: float
    ) -> Tuple[float, float]:
        """
        Calculate trim start and end times for a video clip.
        
        Args:
            video_duration: Duration of the video
            segment_duration: Duration of the script segment
        
        Returns:
            Tuple of (trim_start, trim_end)
        """
        if video_duration <= segment_duration:
            # Video is shorter or equal to segment, use entire video
            return 0.0, video_duration
        
        # Video is longer than segment
        # Try to center the important part (middle of video)
        excess_duration = video_duration - segment_duration
        trim_start = excess_duration / 2
        trim_end = trim_start + segment_duration
        
        return trim_start, trim_end


def create_sequence(
    script_segments: List[Dict],
    video_matcher: VideoTextMatcher,
    allow_reuse: bool = True,
    prefer_non_reused: bool = True
) -> List[ClipSelection]:
    """
    Create the final video sequence.
    
    Args:
        script_segments: List of script segments with text and duration
        video_matcher: Initialized VideoTextMatcher instance
        allow_reuse: Whether to allow reusing video clips
        prefer_non_reused: Prefer non-reused clips if possible
    
    Returns:
        List of ClipSelection objects representing the final sequence
    """
    sequence = []
    used_videos = set()
    
    for i, segment in enumerate(script_segments):
        logger.info(f"Processing segment {i+1}/{len(script_segments)}")
        
        candidates = video_matcher.match_segment_to_videos(
            segment_text=segment['text'],
            segment_duration=segment['duration'],
            k=20,  # Get more candidates
            allow_reuse=allow_reuse,
            used_videos=used_videos
        )
        
        if not candidates:
            logger.warning(f"  -> No candidates found for segment {i+1}, trying with reuse allowed")
            # Try again with reuse allowed
            candidates = video_matcher.match_segment_to_videos(
                segment_text=segment['text'],
                segment_duration=segment['duration'],
                k=20,
                allow_reuse=True,
                used_videos=set()  # Empty set to get all videos
            )
        
        best_clip_data = video_matcher.select_best_clip(
            candidates,
            segment['duration'],
            prefer_non_reused=prefer_non_reused
        )
        
        if best_clip_data:
            selection = ClipSelection(
                segment_id=i,
                video_id=best_clip_data['video_id'],
                video_file_path=best_clip_data['file_path'],
                start_time=0.0,  # Placeholder, will be set during assembly
                end_time=0.0,  # Placeholder
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
            logger.info(f"  -> Matched with video: {selection.video_id} (Score: {selection.combined_score:.3f})")
        else:
            logger.warning(f"  -> No suitable clip found for segment {i+1}")
            # Create a placeholder selection using the first available video
            if video_matcher.video_indexer.metadata:
                first_video_id = list(video_matcher.video_indexer.metadata.keys())[0]
                first_metadata = video_matcher.video_indexer.metadata[first_video_id]
                logger.info(f"  -> Using fallback video: {first_video_id}")
                
                selection = ClipSelection(
                    segment_id=i,
                    video_id=first_video_id,
                    video_file_path=first_metadata.file_path,
                    start_time=0.0,
                    end_time=0.0,
                    duration=segment['duration'],
                    trim_start=0.0,
                    trim_end=min(segment['duration'], first_metadata.duration),
                    trim_duration=min(segment['duration'], first_metadata.duration),
                    similarity_score=0.1,
                    motion_score=0.5,
                    context_score=0.5,
                    combined_score=0.3,
                    is_reused=first_video_id in used_videos
                )
                sequence.append(selection)
                used_videos.add(selection.video_id)
    
    return sequence
