"""
Video-Text Matching and Sequencing Engine

This module is responsible for:
1. Matching script segments to video clips using semantic similarity
2. Using VideoPrism LVT for text embeddings (compatible with video embeddings)
3. Optimal global matching using the Hungarian Algorithm
4. Fallback greedy matching for comparison
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

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


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
        min_similarity_threshold: float = 0.0
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
    
    def compute_similarity_matrix(
        self,
        script_segments: List[Dict],
        match_only: bool = False
    ) -> Tuple[np.ndarray, List]:
        """
        Compute the full similarity matrix between all segments and all videos.
        
        Args:
            script_segments: List of segment dictionaries with 'text' and 'duration'
            match_only: If True, use only raw cosine similarity
            
        Returns:
            Tuple of (similarity_matrix, all_metadata)
            similarity_matrix shape: (num_segments, num_videos)
        """
        all_metadata = self.get_all_video_metadata()
        num_segments = len(script_segments)
        num_videos = len(all_metadata)
        
        if num_videos == 0:
            logger.error("No videos indexed!")
            return np.array([]), []
        
        logger.info(f"Computing similarity matrix: {num_segments} segments x {num_videos} videos")
        
        # Get all text embeddings
        text_embeddings = []
        for segment in script_segments:
            emb = self.get_text_embedding(segment['text'])
            text_embeddings.append(emb)
        text_embeddings = np.array(text_embeddings)
        
        # Normalize text embeddings for cosine similarity
        text_norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        text_embeddings_normalized = text_embeddings / np.maximum(text_norms, 1e-8)
        
        # Get all video embeddings from FAISS index
        # We'll search for all videos for each segment
        similarity_matrix = np.zeros((num_segments, num_videos))
        
        for seg_idx, segment in enumerate(script_segments):
            # Search all videos
            results = self.video_indexer.search_by_embedding(
                text_embeddings[seg_idx], 
                k=num_videos
            )
            
            # Build a mapping from video_id to similarity
            video_sim_map = {video_id: sim for video_id, sim, _ in results}
            
            # Fill in the similarity matrix
            for vid_idx, metadata in enumerate(all_metadata):
                raw_similarity = video_sim_map.get(metadata.video_id, -1.0)
                
                if match_only:
                    # Pure semantic matching - use raw cosine similarity
                    similarity_matrix[seg_idx, vid_idx] = raw_similarity
                else:
                    # Production mode - include motion and context scores
                    similarity_score = (raw_similarity + 1.0) / 2.0  # Normalize to [0, 1]
                    motion_score = self._calculate_motion_score(metadata, segment['duration'])
                    context_score = self._calculate_context_score(segment['text'], metadata)
                    
                    combined_score = (
                        0.5 * similarity_score +
                        0.3 * motion_score +
                        0.2 * context_score
                    )
                    similarity_matrix[seg_idx, vid_idx] = combined_score
        
        return similarity_matrix, all_metadata
    
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
        Find the best matching video clips for a script segment (greedy mode).
        
        Args:
            segment_text: The text of the script segment
            segment_duration: Duration of the segment in seconds
            k: Number of top candidates to return
            allow_reuse: Whether to allow reusing videos
            used_videos: Set of already used video IDs
            match_only: If True, use ONLY raw semantic similarity
            
        Returns:
            List of candidate dictionaries sorted by score
        """
        if used_videos is None:
            used_videos = set()
        
        logger.info(f"Matching segment: '{segment_text[:60]}...' (duration: {segment_duration:.2f}s)")
        
        # Get text embedding
        text_embedding = self.get_text_embedding(segment_text)
        
        # Get number of indexed videos
        num_videos = self.get_num_indexed_videos()
        if num_videos == 0:
            logger.error("No videos indexed! Cannot match segments.")
            return []
        
        # In match-only mode, get ALL videos to show full ranking
        if match_only:
            search_k = num_videos
        else:
            search_k = min(num_videos, k * 5)
        
        results = self.video_indexer.search_by_embedding(text_embedding, k=search_k)
        
        if not results:
            logger.warning("No results from FAISS search")
            return []
        
        candidates = []
        for video_id, similarity, metadata in results:
            # Skip used videos if reuse is not allowed
            if not allow_reuse and video_id in used_videos:
                continue
            
            # similarity is cosine similarity (higher is better, range [-1, 1])
            similarity_score = (similarity + 1.0) / 2.0  # Normalize to [0, 1]
            
            if match_only:
                # PURE SEMANTIC MATCHING MODE
                motion_score = 0.0
                context_score = 0.0
                combined_score = similarity  # Use raw cosine similarity for ranking
            else:
                # PRODUCTION MODE - Use all factors
                diversity_multiplier = 1.2 if video_id not in used_videos else 0.8
                
                if self.used_videos_history and self.used_videos_history[-1] == video_id:
                    diversity_multiplier *= 0.5
                
                motion_score = self._calculate_motion_score(metadata, segment_duration)
                context_score = self._calculate_context_score(segment_text, metadata)
                
                combined_score = (
                    0.5 * similarity_score * diversity_multiplier +
                    0.3 * motion_score +
                    0.2 * context_score
                )
            
            candidates.append({
                'video_id': video_id,
                'file_path': metadata.file_path,
                'duration': metadata.duration,
                'similarity': similarity,
                'similarity_score': similarity_score,
                'motion_score': motion_score,
                'context_score': context_score,
                'combined_score': combined_score,
                'is_reused': video_id in used_videos
            })
        
        # Sort by combined score (higher is better)
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # In match-only mode, log detailed results for debugging
        if match_only and candidates:
            logger.info(f"  Top 5 matches by raw cosine similarity:")
            for i, c in enumerate(candidates[:5]):
                video_name = Path(c['file_path']).name
                logger.info(f"    {i+1}. {video_name}: cosine_sim={c['similarity']:.4f}")
        
        return candidates[:k]
    
    def _calculate_motion_score(self, metadata, segment_duration: float) -> float:
        """Calculate score based on video duration vs segment duration."""
        video_duration = metadata.duration
        if video_duration == 0 or segment_duration == 0:
            return 0.5
        
        ratio = video_duration / segment_duration
        
        if ratio >= 1.0:
            return 1.0 if ratio <= 3.0 else 0.7
        else:
            return max(0.3, ratio)
    
    def _calculate_context_score(self, segment_text: str, metadata) -> float:
        """Calculate score based on keyword matches in filename."""
        keywords = self._extract_keywords(segment_text)
        video_name = Path(metadata.file_path).name.lower()
        
        matches = sum(1 for kw in keywords if kw in video_name)
        
        if matches > 0:
            return min(1.0, 0.5 + matches * 0.25)
        return 0.5
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        import re
        
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        
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
        """
        if not candidates:
            return None
        
        best = candidates[0]
        
        if match_only:
            trim_start = 0.0
            trim_end = best['duration']
        else:
            trim_start, trim_end = self._calculate_trim_times(best['duration'], segment_duration)
        
        best['trim_start'] = trim_start
        best['trim_end'] = trim_end
        best['trim_duration'] = trim_end - trim_start
        
        if not match_only:
            self.used_videos_history.append(best['video_id'])
        
        return best
    
    def _calculate_trim_times(
        self, 
        video_duration: float, 
        segment_duration: float,
        window_start: float = 0.0
    ) -> Tuple[float, float]:
        """Calculate trim start and end times for a video clip.
        
        For windowed clips, window_start offsets the trim times so they
        are relative to the full source video file.
        
        If segment_duration is 0 or negative, uses the full video/window duration.
        """
        # If segment duration is missing or zero, use the full clip/window
        if segment_duration <= 0:
            return window_start, window_start + video_duration
        
        if video_duration <= segment_duration:
            return window_start, window_start + video_duration
        
        trim_start = (video_duration - segment_duration) / 2
        return window_start + trim_start, window_start + trim_start + segment_duration


def create_sequence_optimal(
    script_segments: List[Dict],
    video_matcher: VideoTextMatcher,
    match_only: bool = False,
    allow_reuse: bool = True
) -> List[ClipSelection]:
    """
    Create a sequence using the Hungarian Algorithm for optimal global matching.
    
    This finds the assignment of clips to segments that maximizes the total
    similarity score across all segments.
    
    Args:
        script_segments: List of segment dictionaries with 'text' and 'duration'
        video_matcher: VideoTextMatcher instance
        match_only: If True, use only raw cosine similarity
        allow_reuse: Whether to allow reusing videos
        
    Returns:
        List of ClipSelection objects representing the video sequence
    """
    if linear_sum_assignment is None:
        logger.error("scipy not installed! Run: pip install scipy")
        logger.warning("Falling back to greedy matching...")
        return create_sequence_greedy(script_segments, video_matcher, match_only, allow_reuse)
    
    all_metadata = video_matcher.get_all_video_metadata()
    num_segments = len(script_segments)
    num_videos = len(all_metadata)
    
    if not all_metadata:
        logger.error("No video metadata available!")
        return []
    
    logger.info("=" * 60)
    logger.info("OPTIMAL MATCHING MODE (Hungarian Algorithm)")
    logger.info("=" * 60)
    
    # Compute full similarity matrix
    similarity_matrix, all_metadata = video_matcher.compute_similarity_matrix(
        script_segments, match_only=match_only
    )
    
    if similarity_matrix.size == 0:
        return []
    
    if allow_reuse:
        # When reuse is allowed, we can simply pick the best match for each segment
        # (This is equivalent to greedy, but we compute it from the matrix)
        logger.info("Reuse allowed: Selecting best match per segment from similarity matrix")
        
        sequence = []
        used_videos = set()
        
        for seg_idx, segment in enumerate(script_segments):
            best_vid_idx = np.argmax(similarity_matrix[seg_idx])
            best_similarity = similarity_matrix[seg_idx, best_vid_idx]
            metadata = all_metadata[best_vid_idx]
            
            is_reused = metadata.video_id in used_videos
            used_videos.add(metadata.video_id)
            
            # Get window offset for windowed clips
            window_start = getattr(metadata, 'window_start', 0.0)
            
            if match_only:
                trim_start = window_start
                trim_end = window_start + metadata.duration
            else:
                trim_start, trim_end = video_matcher._calculate_trim_times(
                    metadata.duration, segment['duration'], window_start=window_start
                )
            
            selection = ClipSelection(
                segment_id=seg_idx,
                video_id=metadata.video_id,
                video_file_path=metadata.file_path,
                start_time=0.0, end_time=0.0,
                duration=segment['duration'] if not match_only else metadata.duration,
                trim_start=trim_start,
                trim_end=trim_end,
                trim_duration=trim_end - trim_start,
                similarity_score=best_similarity,
                motion_score=0.0,
                context_score=0.0,
                combined_score=best_similarity,
                is_reused=is_reused
            )
            sequence.append(selection)
    else:
        # No reuse allowed: Use Hungarian Algorithm for optimal assignment
        logger.info(f"No reuse: Using Hungarian Algorithm for optimal {num_segments}x{num_videos} assignment")
        
        if num_segments > num_videos:
            logger.warning(f"More segments ({num_segments}) than videos ({num_videos}). "
                          f"Some segments will not get a match.")
        
        # Hungarian algorithm minimizes cost, so we negate the similarity matrix
        # to maximize similarity
        cost_matrix = -similarity_matrix
        
        # If we have more segments than videos, we need to pad the cost matrix
        if num_segments > num_videos:
            # Pad with very high cost (low similarity) columns
            padding = np.full((num_segments, num_segments - num_videos), 1e6)
            cost_matrix = np.hstack([cost_matrix, padding])
        
        # If we have more videos than segments, we need to pad with rows
        if num_videos > num_segments:
            # Pad with very high cost rows (these won't be assigned)
            padding = np.full((num_videos - num_segments, cost_matrix.shape[1]), 1e6)
            cost_matrix = np.vstack([cost_matrix, padding])
        
        # Run Hungarian Algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Build the sequence from the optimal assignment
        sequence = []
        used_videos = set()
        
        for seg_idx in range(num_segments):
            # Find the assigned video for this segment
            assignment_idx = np.where(row_indices == seg_idx)[0]
            
            if len(assignment_idx) == 0 or col_indices[assignment_idx[0]] >= num_videos:
                # No valid assignment (more segments than videos)
                logger.warning(f"Segment {seg_idx} has no valid assignment")
                # Use fallback
                fallback_meta = all_metadata[seg_idx % num_videos]
                selection = ClipSelection(
                    segment_id=seg_idx,
                    video_id=fallback_meta.video_id,
                    video_file_path=fallback_meta.file_path,
                    start_time=0.0, end_time=0.0,
                    duration=script_segments[seg_idx]['duration'],
                    trim_start=0.0,
                    trim_end=fallback_meta.duration,
                    trim_duration=fallback_meta.duration,
                    similarity_score=0.0,
                    motion_score=0.0,
                    context_score=0.0,
                    combined_score=0.0,
                    is_reused=True
                )
            else:
                vid_idx = col_indices[assignment_idx[0]]
                metadata = all_metadata[vid_idx]
                similarity = similarity_matrix[seg_idx, vid_idx]
                
                segment = script_segments[seg_idx]
                
                # Get window offset for windowed clips
                window_start = getattr(metadata, 'window_start', 0.0)
                
                if match_only:
                    trim_start = window_start
                    trim_end = window_start + metadata.duration
                else:
                    trim_start, trim_end = video_matcher._calculate_trim_times(
                        metadata.duration, segment['duration'], window_start=window_start
                    )
                
                is_reused = metadata.video_id in used_videos
                used_videos.add(metadata.video_id)
                
                selection = ClipSelection(
                    segment_id=seg_idx,
                    video_id=metadata.video_id,
                    video_file_path=metadata.file_path,
                    start_time=0.0, end_time=0.0,
                    duration=segment['duration'] if not match_only else metadata.duration,
                    trim_start=trim_start,
                    trim_end=trim_end,
                    trim_duration=trim_end - trim_start,
                    similarity_score=similarity,
                    motion_score=0.0,
                    context_score=0.0,
                    combined_score=similarity,
                    is_reused=is_reused
                )
            
            sequence.append(selection)
        
        # Log the optimal total score
        total_score = sum(s.similarity_score for s in sequence)
        logger.info(f"Optimal assignment total score: {total_score:.4f}")
    
    # Calculate start/end times in the final sequence
    current_time = 0.0
    for selection in sequence:
        selection.start_time = current_time
        selection.end_time = current_time + selection.trim_duration
        current_time = selection.end_time
    
    return sequence


def create_sequence_greedy(
    script_segments: List[Dict],
    video_matcher: VideoTextMatcher,
    match_only: bool = False,
    allow_reuse: bool = True
) -> List[ClipSelection]:
    """
    Create a sequence using greedy sequential matching.
    
    This is the original algorithm that picks the best available match
    for each segment in order.
    
    Args:
        script_segments: List of segment dictionaries with 'text' and 'duration'
        video_matcher: VideoTextMatcher instance
        match_only: If True, bypass duration constraints and diversity penalties
        allow_reuse: Whether to allow reusing videos
        
    Returns:
        List of ClipSelection objects representing the video sequence
    """
    sequence = []
    used_videos = set()
    all_metadata = video_matcher.get_all_video_metadata()
    
    if not all_metadata:
        logger.error("No video metadata available!")
        return []
    
    logger.info("=" * 60)
    logger.info("GREEDY MATCHING MODE (Sequential)")
    logger.info("=" * 60)
    
    # Check if we have enough unique clips for no-reuse mode
    if not allow_reuse and len(all_metadata) < len(script_segments):
        logger.warning(f"Not enough unique clips ({len(all_metadata)}) for all segments ({len(script_segments)}). Some reuse will be unavoidable.")

    for i, segment in enumerate(script_segments):
        logger.info(f"Processing segment {i+1}/{len(script_segments)}")
        
        # Determine which videos to exclude based on allow_reuse
        videos_to_exclude = set() if allow_reuse else used_videos
        
        # 1. Try to find a good match using semantic similarity
        candidates = video_matcher.match_segment_to_videos(
            segment['text'], 
            segment['duration'], 
            used_videos=videos_to_exclude,
            match_only=match_only
        )
        
        # If no candidates found and we are forbidding reuse, try to find ANY unused video
        if not candidates and not allow_reuse:
            unused_metadata = [m for m in all_metadata if m.video_id not in used_videos]
            if unused_metadata:
                logger.info(f"  -> No semantic candidates for segment {i+1}, picking first available unused clip for strict no-reuse")
                fallback_meta = unused_metadata[0]
                best_clip_data = {
                    'video_id': fallback_meta.video_id,
                    'file_path': fallback_meta.file_path,
                    'duration': fallback_meta.duration,
                    'similarity': 0.0,
                    'similarity_score': 0.0,
                    'motion_score': 0.0,
                    'context_score': 0.0,
                    'combined_score': 0.0,
                    'is_reused': False
                }
            else:
                best_clip_data = None
        else:
            best_clip_data = video_matcher.select_best_clip(
                candidates, 
                segment['duration'],
                match_only=match_only
            )
        
        # 2. If still no clip found
        if not best_clip_data:
            logger.warning(f"  -> Absolutely no clips available for segment {i+1}")
            
            # Ultimate fallback (will reuse if necessary)
            fallback_meta = all_metadata[i % len(all_metadata)]
            
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
                start_time=0.0, end_time=0.0,
                duration=segment['duration'] if not match_only else fallback_meta.duration,
                trim_start=trim_start, 
                trim_end=trim_end,
                trim_duration=trim_end - trim_start,
                similarity_score=0.0,
                motion_score=0.0, 
                context_score=0.0, 
                combined_score=0.0
            )
        else:
            # Check if this clip was already used (for reporting purposes)
            is_reused = best_clip_data['video_id'] in used_videos
            
            selection = ClipSelection(
                segment_id=i,
                video_id=best_clip_data['video_id'],
                video_file_path=best_clip_data['file_path'],
                start_time=0.0, end_time=0.0,
                duration=segment['duration'] if not match_only else best_clip_data['duration'],
                trim_start=best_clip_data['trim_start'],
                trim_end=best_clip_data['trim_end'],
                trim_duration=best_clip_data['trim_duration'],
                similarity_score=best_clip_data['similarity'],
                motion_score=best_clip_data['motion_score'],
                context_score=best_clip_data['context_score'],
                combined_score=best_clip_data['combined_score'],
                is_reused=is_reused
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


def create_sequence(
    script_segments: List[Dict],
    video_matcher: VideoTextMatcher,
    match_only: bool = False,
    allow_reuse: bool = True,
    use_optimal: bool = True
) -> List[ClipSelection]:
    """
    Create a sequence of video clips matched to script segments.
    
    Args:
        script_segments: List of segment dictionaries with 'text' and 'duration'
        video_matcher: VideoTextMatcher instance
        match_only: If True, use pure semantic similarity
        allow_reuse: Whether to allow reusing videos
        use_optimal: If True, use Hungarian Algorithm; if False, use greedy matching
        
    Returns:
        List of ClipSelection objects representing the video sequence
    """
    if use_optimal:
        return create_sequence_optimal(
            script_segments, video_matcher, match_only, allow_reuse
        )
    else:
        return create_sequence_greedy(
            script_segments, video_matcher, match_only, allow_reuse
        )
