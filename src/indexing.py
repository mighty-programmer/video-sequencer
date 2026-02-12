"""
Video Indexing Module using VideoPrism LVT with Temporal Windowing

This module is responsible for:
1. Loading B-roll video files from a directory
2. Extracting GLOBAL embeddings using VideoPrism LVT
3. Supporting TEMPORAL WINDOWING for long videos (sliding window)
4. Storing embeddings in a searchable vector database (FAISS)
5. Providing methods to retrieve similar videos/sub-clips based on text queries
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
    """Metadata for indexed videos and sub-clips"""
    video_id: str
    file_path: str
    duration: float
    num_frames: int
    fps: float
    width: int
    height: int
    embedding_dim: int
    start_time: float = 0.0  # Start time of the sub-clip
    end_time: float = 0.0    # End time of the sub-clip
    is_subclip: bool = False # Whether this is a windowed sub-clip


class VideoIndexer:
    """
    VideoPrism LVT-based video indexing system with temporal windowing.
    """
    
    def __init__(
        self,
        model_name: str = 'videoprism_lvt_public_v1_base',
        index_dir: str = './video_index',
        device: str = 'cuda:0',
        window_size: float = 10.0,  # 10-second windows
        window_stride: float = 5.0   # 5-second overlap
    ):
        """
        Initialize the VideoIndexer.
        
        Args:
            model_name: VideoPrism LVT model to use
            index_dir: Directory to store the FAISS index and metadata
            device: Device to use
            window_size: Duration of each temporal window in seconds
            window_stride: Stride between windows in seconds
        """
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.window_size = window_size
        self.window_stride = window_stride
        
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
        
        @jax.jit
        def forward_video_fn(video_inputs):
            video_embeddings, _, _ = self.flax_model.apply(
                self.loaded_state,
                video_inputs,
                None,
                None,
                train=False
            )
            return video_embeddings
        
        self.forward_video_fn = forward_video_fn
        
        if faiss is None:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu or faiss-gpu")
        
        self.index = None
        self.metadata_list: List[VideoMetadata] = []
        self.video_id_to_idx: Dict[str, int] = {}
        
        logger.info(f"VideoIndexer initialized (window_size={window_size}s, stride={window_stride}s)")
    
    def extract_frames(
        self,
        video_path: str,
        start_time: float = 0.0,
        duration: Optional[float] = None,
        num_frames: int = 16,
        target_size: Tuple[int, int] = (288, 288)
    ) -> Tuple[np.ndarray, Dict]:
        """Extract frames from a specific time range in a video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_duration = total_frames / fps if fps > 0 else 0
        
        if duration is None:
            duration = video_duration - start_time
        
        end_time = min(start_time + duration, video_duration)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        num_frames_in_range = end_frame - start_frame
        
        if num_frames_in_range <= 0:
            cap.release()
            raise ValueError(f"Invalid time range: {start_time} to {end_time}")
            
        if num_frames_in_range <= num_frames:
            frame_indices = np.arange(start_frame, end_frame)
        else:
            frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, target_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames extracted from {video_path} at {start_time}s")
            
        frames = np.array(frames)
        if len(frames) < num_frames:
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
            'start_time': start_time,
            'end_time': end_time
        }
        return frames, video_info

    def get_video_embedding(self, video_path: str, start_time: float = 0.0, duration: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
        """Extract GLOBAL embedding for a video segment."""
        frames, video_info = self.extract_frames(video_path, start_time, duration)
        frames_batch = np.expand_dims(frames, axis=0)
        video_embeddings = self.forward_video_fn(frames_batch)
        embedding = np.array(video_embeddings[0])
        return embedding, video_info

    def index_videos(self, video_dir: str) -> int:
        """Index videos with temporal windowing."""
        video_dir = Path(video_dir)
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        video_files = [f for f in video_dir.rglob('*') if f.suffix.lower() in video_extensions]
        
        if not video_files:
            logger.warning(f"No video files found in {video_dir}")
            return 0
            
        embeddings_list = []
        
        for video_file in tqdm(video_files, desc="Indexing videos"):
            try:
                # Get basic info first
                cap = cv2.VideoCapture(str(video_file))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                video_duration = total_frames / fps if fps > 0 else 0
                cap.release()
                
                # Determine windows
                windows = []
                if video_duration <= self.window_size:
                    windows.append((0.0, video_duration))
                else:
                    start = 0.0
                    while start < video_duration:
                        end = min(start + self.window_size, video_duration)
                        windows.append((start, end))
                        if end == video_duration: break
                        start += self.window_stride
                
                for i, (start, end) in enumerate(windows):
                    subclip_id = f"{video_file.stem}_win{i:03d}"
                    embedding, info = self.get_video_embedding(str(video_file), start, end - start)
                    
                    metadata = VideoMetadata(
                        video_id=subclip_id,
                        file_path=str(video_file),
                        duration=video_duration,
                        num_frames=total_frames,
                        fps=fps,
                        width=info['width'],
                        height=info['height'],
                        embedding_dim=embedding.shape[0],
                        start_time=start,
                        end_time=end,
                        is_subclip=(len(windows) > 1)
                    )
                    
                    self.metadata_list.append(metadata)
                    embeddings_list.append(embedding)
                    
            except Exception as e:
                logger.error(f"Error indexing {video_file}: {e}")
                continue
                
        if not embeddings_list:
            return 0
            
        embeddings_array = np.vstack(embeddings_list).astype(np.float32)
        faiss.normalize_L2(embeddings_array)
        
        self.index = faiss.IndexFlatIP(embeddings_array.shape[1])
        self.index.add(embeddings_array)
        
        self.save_index()
        return len(self.metadata_list)

    def save_index(self):
        """Save FAISS index and metadata."""
        faiss.write_index(self.index, str(self.index_dir / 'index.faiss'))
        metadata_dicts = [asdict(m) for m in self.metadata_list]
        with open(self.index_dir / 'metadata.json', 'w') as f:
            json.dump(metadata_dicts, f)
        with open(self.index_dir / 'config.json', 'w') as f:
            json.dump({'model_name': self.model_name, 'window_size': self.window_size, 'window_stride': self.window_stride}, f)

    def load_index(self) -> bool:
        """Load FAISS index and metadata."""
        try:
            if not (self.index_dir / 'index.faiss').exists(): return False
            self.index = faiss.read_index(str(self.index_dir / 'index.faiss'))
            with open(self.index_dir / 'metadata.json', 'r') as f:
                data = json.load(f)
                self.metadata_list = [VideoMetadata(**d) for d in data]
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False

    def search_by_text(self, query_text: str, k: int = 5) -> List[Tuple[VideoMetadata, float]]:
        """Search for sub-clips matching the text query."""
        # Fix for 'SentencePieceTokenizer' object is not callable
        if hasattr(self.text_tokenizer, 'tokenize'):
            text_inputs = self.text_tokenizer.tokenize([query_text])
        else:
            text_inputs = self.text_tokenizer([query_text])
            
        text_embeddings, _, _ = self.flax_model.apply(
            self.loaded_state,
            None,
            text_inputs['token_ids'],
            text_inputs['token_paddings'],
            train=False
        )
        query_embedding = np.array(text_embeddings[0]).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append((self.metadata_list[idx], float(score)))
        return results
