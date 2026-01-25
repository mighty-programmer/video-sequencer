"""
Video Indexing Module using VideoPrism LVT

This module is responsible for:
1. Loading B-roll video files from a directory
2. Extracting GLOBAL embeddings using VideoPrism LVT (Language-Vision-Text model)
3. Storing embeddings in a searchable vector database (FAISS)
4. Providing methods to retrieve similar videos based on text queries

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
    embedding_dim: int  # Changed from embedding_shape to single dimension


class VideoIndexer:
    """
    VideoPrism LVT-based video indexing system.
    
    This class handles loading videos, extracting GLOBAL embeddings using 
    VideoPrism LVT, and storing them in a FAISS index for efficient 
    similarity search with text queries.
    
    The LVT model produces embeddings of shape (batch_size, feature_channels)
    which can be directly compared with text embeddings using cosine similarity.
    """
    
    def __init__(
        self,
        model_name: str = 'videoprism_lvt_public_v1_base',  # Use LVT model!
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
        target_size: Tuple[int, int] = (288, 288)
    ) -> Tuple[np.ndarray, Dict]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to extract (default 16 for VideoPrism-B)
            target_size: Target frame size (height, width) - must be 288x288 for VideoPrism
        
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
        
        # Calculate frame indices to extract (uniformly sampled)
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
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
            'extracted_frames': len(frames)
        }
        
        return frames, video_info
    
    def get_video_embedding(self, video_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Extract GLOBAL embedding for a single video using LVT model.
        
        Args:
            video_path: Path to the video file
        
        Returns:
            Tuple of (embedding array with shape (feature_channels,), video info dict)
        """
        # Extract frames
        frames, video_info = self.extract_frames(video_path)
        
        # Add batch dimension: [1, num_frames, height, width, 3]
        frames_batch = np.expand_dims(frames, axis=0)
        
        # Get GLOBAL embedding from VideoPrism LVT
        # Shape: [batch_size, feature_channels] -> [1, 768] for base model
        video_embeddings = self.forward_video_fn(frames_batch)
        
        # Remove batch dimension: [768]
        embedding = np.array(video_embeddings[0])
        
        return embedding, video_info
    
    def index_videos(self, video_dir: str) -> int:
        """
        Index all videos in a directory using LVT global embeddings.
        
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
                
                # Extract GLOBAL embedding
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
                    embedding_dim=embedding.shape[0]
                )
                
                self.metadata_list.append(metadata)
                self.video_id_to_idx[video_id] = len(self.metadata_list) - 1
                
                # Embedding is already flat: (768,)
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
        logger.info(f"Creating FAISS index with {len(embeddings_list)} videos")
        embeddings_array = np.vstack(embeddings_list).astype(np.float32)
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / (norms + 1e-8)
        
        # Use IndexFlatIP for inner product (cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(embeddings_array.shape[1])
        self.index.add(embeddings_array)
        
        # Save index and metadata
        self.save_index()
        
        logger.info(f"Successfully indexed {len(embeddings_list)} videos with LVT embeddings")
        return len(embeddings_list)
    
    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[str, float, VideoMetadata]]:
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
        
        logger.info(f"Loaded metadata for {len(self.metadata_list)} videos")
        return True


if __name__ == '__main__':
    # Example usage
    indexer = VideoIndexer(model_name='videoprism_lvt_public_v1_base')
    num_indexed = indexer.index_videos('./data/input/videos')
    print(f"Indexed {num_indexed} videos")
