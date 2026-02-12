"""
Video Indexing Module using VideoPrism LVT

This module is responsible for:
1. Loading B-roll video files from a directory
2. Extracting GLOBAL embeddings using VideoPrism LVT (Language-Vision-Text model)
3. Optionally using Temporal Sliding Windows for sub-clip indexing
4. Storing embeddings in a searchable vector database (FAISS)
5. Providing methods to retrieve similar videos based on text queries

IMPORTANT: This module uses the LVT model which produces global embeddings
with shape (feature_channels,) that are compatible with text embeddings
for cosine similarity matching.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass, asdict, field
import cv2
from tqdm import tqdm

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
class VideoMetadata:
    """Metadata for indexed videos"""
    video_id: str
    file_path: str
    duration: float
    num_frames: int
    fps: float
    width: int
    height: int
    embedding_dim: int
    # Windowing fields (optional, for sub-clip indexing)
    window_start: float = 0.0
    window_end: float = 0.0
    is_windowed: bool = False
    source_video_id: str = ""  # Original video ID before windowing


class VideoIndexer:
    """
    VideoPrism LVT-based video indexing system.
    
    This class handles loading videos, extracting GLOBAL embeddings using 
    VideoPrism LVT, and storing them in a FAISS index for efficient 
    similarity search with text queries.
    
    Supports optional Temporal Sliding Window indexing for longer videos,
    which creates multiple sub-clip entries per video for finer-grained matching.
    """
    
    def __init__(
        self,
        model_name: str = 'videoprism_lvt_public_v1_base',
        index_dir: str = './video_index',
        device: str = 'cuda:0'
    ):
        """
        Initialize the VideoIndexer with LVT model.
        
        Args:
            model_name: VideoPrism LVT model to use (must be an LVT model for text matching)
            index_dir: Directory to store the FAISS index and metadata
            device: Device to use ('cuda:0', 'cuda:1', etc., or 'cpu')
        """
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Ensure we're using an LVT model
        if 'lvt' not in model_name.lower():
            logger.warning(f"Model {model_name} is not an LVT model. "
                          f"For text-video matching, use 'videoprism_lvt_public_v1_base' or 'videoprism_lvt_public_v1_large'")
        
        # Set JAX to use specific GPU
        if 'cuda' in device:
            gpu_id = int(device.split(':')[1]) if ':' in device else 0
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            logger.info(f"Setting JAX to use GPU {gpu_id}")
        
        # Load VideoPrism LVT model
        logger.info(f"Loading VideoPrism LVT model: {model_name}")
        if vp is None:
            raise ImportError("VideoPrism not installed. Run: pip install videoprism")
        
        self.flax_model = vp.get_model(model_name)
        self.loaded_state = vp.load_pretrained_weights(model_name)
        self.text_tokenizer = vp.load_text_tokenizer('c4_en')
        
        # Define the forward function for video-only embedding (text=None)
        @jax.jit
        def forward_video_fn(video_inputs):
            # For LVT model: pass None for text to get only video embeddings
            video_embeddings, _, _ = self.flax_model.apply(
                self.loaded_state,
                video_inputs,
                None,  # text_token_ids = None
                None,  # text_token_paddings = None
                train=False
            )
            return video_embeddings
        
        self.forward_video_fn = forward_video_fn
        
        # Initialize FAISS index
        if faiss is None:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu or faiss-gpu")
        
        self.index = None
        self.metadata_list: List[VideoMetadata] = []
        self.video_id_to_idx: Dict[str, int] = {}
        
        logger.info("VideoIndexer initialized successfully with LVT model")
    
    def extract_frames(
        self,
        video_path: str,
        num_frames: int = 16,
        target_size: Tuple[int, int] = (288, 288),
        start_time: float = 0.0,
        end_time: float = -1.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Extract frames from a video file (or a sub-clip window).
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to extract (default 16 for VideoPrism-B)
            target_size: Target frame size (height, width) - must be 288x288 for VideoPrism
            start_time: Start time in seconds for windowed extraction (default: 0.0)
            end_time: End time in seconds for windowed extraction (default: -1.0 = full video)
        
        Returns:
            Tuple of (frames array, video info dict)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        full_duration = total_frames / fps if fps > 0 else 0
        
        # Calculate frame range for windowed extraction
        if end_time > 0 and fps > 0:
            start_frame = max(0, int(start_time * fps))
            end_frame = min(total_frames, int(end_time * fps))
            window_frames = end_frame - start_frame
            duration = end_time - start_time
        else:
            start_frame = 0
            end_frame = total_frames
            window_frames = total_frames
            duration = full_duration
        
        # Calculate frame indices to extract (uniformly sampled within window)
        if window_frames <= num_frames:
            frame_indices = [start_frame + i for i in range(window_frames)]
        else:
            frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize frame to 288x288
                frame = cv2.resize(frame, target_size)
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Normalize to [0, 1] as required by VideoPrism
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames extracted from video: {video_path}")
        
        # Pad or trim to exact num_frames
        frames = np.array(frames)
        if len(frames) < num_frames:
            # Pad with last frame (for very short videos)
            padding = np.tile(frames[-1:], (num_frames - len(frames), 1, 1, 1))
            frames = np.concatenate([frames, padding], axis=0)
        elif len(frames) > num_frames:
            frames = frames[:num_frames]
        
        video_info = {
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'duration': duration,
            'full_duration': full_duration,
            'extracted_frames': len(frames)
        }
        
        return frames, video_info
    
    def get_video_embedding(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: float = -1.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Extract GLOBAL embedding for a single video (or sub-clip) using LVT model.
        
        Args:
            video_path: Path to the video file
            start_time: Start time for windowed extraction
            end_time: End time for windowed extraction (-1 = full video)
        
        Returns:
            Tuple of (embedding array with shape (feature_channels,), video info dict)
        """
        # Extract frames
        frames, video_info = self.extract_frames(
            video_path, start_time=start_time, end_time=end_time
        )
        
        # Add batch dimension: [1, num_frames, height, width, 3]
        frames_batch = np.expand_dims(frames, axis=0)
        
        # Get GLOBAL embedding from VideoPrism LVT
        # Shape: [batch_size, feature_channels] -> [1, 768] for base model
        video_embeddings = self.forward_video_fn(frames_batch)
        
        # Remove batch dimension: [768]
        embedding = np.array(video_embeddings[0])
        
        return embedding, video_info
    
    def _compute_windows(
        self,
        duration: float,
        window_size: float = 5.0,
        window_overlap: float = 0.5
    ) -> List[Tuple[float, float]]:
        """
        Compute temporal sliding windows for a video.
        
        Args:
            duration: Video duration in seconds
            window_size: Size of each window in seconds
            window_overlap: Overlap fraction between consecutive windows (0.0 to 0.9)
            
        Returns:
            List of (start_time, end_time) tuples for each window
        """
        if duration <= window_size:
            # Video is shorter than window size, use the whole video
            return [(0.0, duration)]
        
        step_size = window_size * (1.0 - window_overlap)
        windows = []
        start = 0.0
        
        while start < duration:
            end = min(start + window_size, duration)
            # Only add window if it's at least half the window size
            if end - start >= window_size * 0.5:
                windows.append((start, end))
            start += step_size
        
        # Ensure the last window covers the end of the video
        if windows and windows[-1][1] < duration:
            last_start = max(0.0, duration - window_size)
            if last_start != windows[-1][0]:
                windows.append((last_start, duration))
        
        return windows
    
    def index_videos(
        self,
        video_dir: str,
        use_windowing: bool = True,
        window_size: float = 5.0,
        window_overlap: float = 0.5
    ) -> int:
        """
        Index all videos in a directory using LVT global embeddings.
        
        Args:
            video_dir: Directory containing video files
            use_windowing: If True, use temporal sliding windows for longer videos
            window_size: Window size in seconds (only used if use_windowing=True)
            window_overlap: Window overlap fraction (only used if use_windowing=True)
        
        Returns:
            Number of entries indexed (may be > number of video files if windowing is used)
        """
        video_dir = Path(video_dir)
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        
        video_files = [
            f for f in video_dir.rglob('*')
            if f.suffix.lower() in video_extensions
        ]
        
        if not video_files:
            logger.warning(f"No video files found in {video_dir}")
            return 0
        
        logger.info(f"Found {len(video_files)} video files to index")
        if use_windowing:
            logger.info(f"Temporal windowing enabled: window_size={window_size}s, overlap={window_overlap}")
        else:
            logger.info("Temporal windowing disabled: indexing full videos only")
        
        embeddings_list = []
        feature_dim = None
        
        for video_file in tqdm(video_files, desc="Indexing videos"):
            try:
                base_video_id = video_file.stem
                
                if use_windowing:
                    # Get video duration first
                    cap = cv2.VideoCapture(str(video_file))
                    if not cap.isOpened():
                        logger.error(f"Cannot open video: {video_file}")
                        continue
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    duration = total_frames / fps if fps > 0 else 0
                    cap.release()
                    
                    # Compute windows
                    windows = self._compute_windows(duration, window_size, window_overlap)
                    
                    for w_idx, (w_start, w_end) in enumerate(windows):
                        if len(windows) == 1:
                            # Only one window (short video) - use original ID
                            video_id = base_video_id
                        else:
                            video_id = f"{base_video_id}_w{w_idx}"
                        
                        # Skip if already indexed
                        if video_id in self.video_id_to_idx:
                            logger.debug(f"Video {video_id} already indexed, skipping")
                            continue
                        
                        # Extract embedding for this window
                        embedding, video_info = self.get_video_embedding(
                            str(video_file), start_time=w_start, end_time=w_end
                        )
                        
                        metadata = VideoMetadata(
                            video_id=video_id,
                            file_path=str(video_file),
                            duration=video_info['duration'],
                            num_frames=video_info['extracted_frames'],
                            fps=video_info['fps'],
                            width=video_info['width'],
                            height=video_info['height'],
                            embedding_dim=embedding.shape[0],
                            window_start=w_start,
                            window_end=w_end,
                            is_windowed=(len(windows) > 1),
                            source_video_id=base_video_id
                        )
                        
                        self.metadata_list.append(metadata)
                        self.video_id_to_idx[video_id] = len(self.metadata_list) - 1
                        embeddings_list.append(embedding)
                        
                        if feature_dim is None:
                            feature_dim = embedding.shape[0]
                        
                        logger.debug(f"Indexed {video_id}: window [{w_start:.1f}s-{w_end:.1f}s]")
                    
                    logger.info(f"Indexed {base_video_id}: {len(windows)} window(s)")
                
                else:
                    # No windowing - index full video
                    video_id = base_video_id
                    
                    # Skip if already indexed
                    if video_id in self.video_id_to_idx:
                        logger.info(f"Video {video_id} already indexed, skipping")
                        continue
                    
                    # Extract GLOBAL embedding
                    embedding, video_info = self.get_video_embedding(str(video_file))
                    
                    metadata = VideoMetadata(
                        video_id=video_id,
                        file_path=str(video_file),
                        duration=video_info['duration'],
                        num_frames=video_info['extracted_frames'],
                        fps=video_info['fps'],
                        width=video_info['width'],
                        height=video_info['height'],
                        embedding_dim=embedding.shape[0],
                        window_start=0.0,
                        window_end=video_info['duration'],
                        is_windowed=False,
                        source_video_id=video_id
                    )
                    
                    self.metadata_list.append(metadata)
                    self.video_id_to_idx[video_id] = len(self.metadata_list) - 1
                    embeddings_list.append(embedding)
                    
                    if feature_dim is None:
                        feature_dim = embedding.shape[0]
                    
                    logger.info(f"Indexed {video_id}: embedding dim {embedding.shape[0]}")
            
            except Exception as e:
                logger.error(f"Error indexing {video_file}: {e}")
                continue
        
        if not embeddings_list:
            logger.error("No videos were successfully indexed")
            return 0
        
        # Create FAISS index with cosine similarity (normalize + L2 = cosine)
        logger.info(f"Creating FAISS index with {len(embeddings_list)} entries")
        embeddings_array = np.vstack(embeddings_list).astype(np.float32)
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / (norms + 1e-8)
        
        # Use IndexFlatIP for inner product (cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(embeddings_array.shape[1])
        self.index.add(embeddings_array)
        
        # Save index and metadata
        self.save_index()
        
        logger.info(f"Successfully indexed {len(embeddings_list)} entries with LVT embeddings")
        return len(embeddings_list)
    
    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[str, float, 'VideoMetadata']]:
        """
        Search for similar videos using a text embedding.
        
        Args:
            query_embedding: Query embedding (shape: feature_channels,)
            k: Number of results to return
        
        Returns:
            List of tuples (video_id, similarity_score, metadata)
            Higher similarity scores are better (cosine similarity)
        """
        if self.index is None:
            raise RuntimeError("Index not initialized. Call index_videos() first.")
        
        # Normalize query for cosine similarity
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # Search returns similarity scores (higher is better for IndexFlatIP)
        similarities, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if idx < len(self.metadata_list) and idx >= 0:
                metadata = self.metadata_list[idx]
                results.append((metadata.video_id, float(similarity), metadata))
        
        return results
    
    def save_index(self):
        """Save the FAISS index and metadata to disk"""
        if self.index is None:
            logger.warning("No index to save")
            return
        
        index_path = self.index_dir / 'faiss_index.bin'
        metadata_path = self.index_dir / 'metadata.json'
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save metadata
        metadata_dicts = [asdict(m) for m in self.metadata_list]
        with open(metadata_path, 'w') as f:
            json.dump({
                'metadata': metadata_dicts,
                'video_id_to_idx': self.video_id_to_idx,
                'model_name': self.model_name
            }, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load_index(self):
        """Load the FAISS index and metadata from disk"""
        index_path = self.index_dir / 'faiss_index.bin'
        metadata_path = self.index_dir / 'metadata.json'
        
        if not index_path.exists() or not metadata_path.exists():
            logger.warning("Index files not found")
            return False
        
        # Load metadata first to check model compatibility
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        # Check if index was created with LVT model
        saved_model = data.get('model_name', '')
        if 'lvt' not in saved_model.lower():
            logger.warning(f"Existing index was created with non-LVT model ({saved_model}). "
                          f"Re-indexing required for text-video matching.")
            return False
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index from {index_path}")
        
        self.metadata_list = [
            VideoMetadata(**m) for m in data['metadata']
        ]
        self.video_id_to_idx = data['video_id_to_idx']
        
        logger.info(f"Loaded metadata for {len(self.metadata_list)} entries")
        return True


if __name__ == '__main__':
    # Example usage
    indexer = VideoIndexer(model_name='videoprism_lvt_public_v1_base')
    num_indexed = indexer.index_videos('./data/input/videos')
    print(f"Indexed {num_indexed} entries")
