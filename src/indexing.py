"""
Video Indexing Module using VideoPrism

This module is responsible for:
1. Loading B-roll video files from a directory
2. Extracting features using VideoPrism
3. Storing embeddings in a searchable vector database (FAISS)
4. Providing methods to retrieve similar videos based on text queries
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass, asdict
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
    embedding_shape: Tuple[int, int]


class VideoIndexer:
    """
    VideoPrism-based video indexing system.
    
    This class handles loading videos, extracting embeddings using VideoPrism,
    and storing them in a FAISS index for efficient similarity search.
    """
    
    def __init__(
        self,
        model_name: str = 'videoprism_public_v1_base',
        index_dir: str = './video_index',
        device: str = 'gpu'
    ):
        """
        Initialize the VideoIndexer.
        
        Args:
            model_name: VideoPrism model to use ('videoprism_public_v1_base' or 'videoprism_public_v1_large')
            index_dir: Directory to store the FAISS index and metadata
            device: Device to use ('gpu' or 'cpu')
        """
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Load VideoPrism model
        logger.info(f"Loading VideoPrism model: {model_name}")
        if vp is None:
            raise ImportError("VideoPrism not installed. Run: pip install videoprism")
        
        self.flax_model = vp.get_model(model_name)
        self.loaded_state = vp.load_pretrained_weights(model_name)
        
        # Define the forward function with JIT compilation
        @jax.jit
        def forward_fn(inputs):
            return self.flax_model.apply(self.loaded_state, inputs, train=False)
        
        self.forward_fn = forward_fn
        
        # Initialize FAISS index
        if faiss is None:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu or faiss-gpu")
        
        self.index = None
        self.metadata_list: List[VideoMetadata] = []
        self.video_id_to_idx: Dict[str, int] = {}
        
        logger.info("VideoIndexer initialized successfully")
    
    def extract_frames(
        self,
        video_path: str,
        num_frames: int = 16,
        target_size: Tuple[int, int] = (288, 288)
    ) -> Tuple[np.ndarray, Dict]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to extract
            target_size: Target frame size (height, width)
        
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
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate frame indices to extract
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize frame
                frame = cv2.resize(frame, target_size)
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames extracted from video: {video_path}")
        
        # Pad or trim to exact num_frames
        frames = np.array(frames)
        if len(frames) < num_frames:
            # Pad with last frame
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
            'extracted_frames': len(frames)
        }
        
        return frames, video_info
    
    def get_video_embedding(self, video_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Extract embedding for a single video.
        
        Args:
            video_path: Path to the video file
        
        Returns:
            Tuple of (embedding array, video info dict)
        """
        # Extract frames
        frames, video_info = self.extract_frames(video_path)
        
        # Add batch dimension: [1, num_frames, height, width, 3]
        frames_batch = np.expand_dims(frames, axis=0)
        
        # Get embedding from VideoPrism
        outputs, _ = self.forward_fn(frames_batch)
        
        # outputs shape: [batch_size, num_tokens, feature_channels]
        # For batch_size=1: [1, num_frames * 16 * 16, feature_channels]
        embedding = np.array(outputs[0])  # Remove batch dimension
        
        return embedding, video_info
    
    def index_videos(self, video_dir: str) -> int:
        """
        Index all videos in a directory.
        
        Args:
            video_dir: Directory containing video files
        
        Returns:
            Number of videos indexed
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
        
        embeddings_list = []
        feature_dim = None
        
        for video_file in tqdm(video_files, desc="Indexing videos"):
            try:
                video_id = video_file.stem
                
                # Skip if already indexed
                if video_id in self.video_id_to_idx:
                    logger.info(f"Video {video_id} already indexed, skipping")
                    continue
                
                # Extract embedding
                embedding, video_info = self.get_video_embedding(str(video_file))
                
                # Store metadata
                metadata = VideoMetadata(
                    video_id=video_id,
                    file_path=str(video_file),
                    duration=video_info['duration'],
                    num_frames=video_info['extracted_frames'],
                    fps=video_info['fps'],
                    width=video_info['width'],
                    height=video_info['height'],
                    embedding_shape=embedding.shape
                )
                
                self.metadata_list.append(metadata)
                self.video_id_to_idx[video_id] = len(self.metadata_list) - 1
                
                # Flatten embedding for FAISS
                embedding_flat = embedding.reshape(embedding.shape[0], -1)
                embeddings_list.append(embedding_flat)
                
                if feature_dim is None:
                    feature_dim = embedding_flat.shape[1]
                
                logger.info(f"Indexed {video_id}: embedding shape {embedding.shape}")
            
            except Exception as e:
                logger.error(f"Error indexing {video_file}: {e}")
                continue
        
        if not embeddings_list:
            logger.error("No videos were successfully indexed")
            return 0
        
        # Create FAISS index
        logger.info(f"Creating FAISS index with {len(embeddings_list)} videos")
        embeddings_array = np.vstack(embeddings_list).astype(np.float32)
        
        # Use IndexFlatL2 for L2 distance
        self.index = faiss.IndexFlatL2(embeddings_array.shape[1])
        self.index.add(embeddings_array)
        
        # Save index and metadata
        self.save_index()
        
        logger.info(f"Successfully indexed {len(embeddings_list)} videos")
        return len(embeddings_list)
    
    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[str, float, VideoMetadata]]:
        """
        Search for similar videos using an embedding.
        
        Args:
            query_embedding: Query embedding (flattened)
            k: Number of results to return
        
        Returns:
            List of tuples (video_id, distance, metadata)
        """
        if self.index is None:
            raise RuntimeError("Index not initialized. Call index_videos() first.")
        
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata_list):
                metadata = self.metadata_list[idx]
                results.append((metadata.video_id, float(distance), metadata))
        
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
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index from {index_path}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        self.metadata_list = [
            VideoMetadata(**m) for m in data['metadata']
        ]
        self.video_id_to_idx = data['video_id_to_idx']
        
        logger.info(f"Loaded metadata for {len(self.metadata_list)} videos")
        return True


if __name__ == '__main__':
    # Example usage
    indexer = VideoIndexer()
    num_indexed = indexer.index_videos('./data/input/videos')
    print(f"Indexed {num_indexed} videos")
