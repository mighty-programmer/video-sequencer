"""
OpenCLIP Video Indexing and Text Matching Module (Baseline)

This module provides a CLIP-based baseline for comparison against VideoPrism.
It uses OpenCLIP (open-source CLIP) to encode video frames as images and
match them against text queries using cosine similarity.

Key differences from VideoPrism:
- Frame-level encoding: Each frame is encoded independently as an image
- No temporal understanding: CLIP does not model motion or temporal transitions
- Configurable aggregation: mean, max, or best-frame pooling
- Prompt engineering: Configurable prompt templates and ensemble averaging
- Shared text-image space: Uses CLIP's contrastive text-image embedding space

Tunable parameters for grid search optimization:
- model_name: ViT-B-32, ViT-B-16, ViT-L-14
- num_frames: Number of frames to sample per video (4, 8, 16, 32)
- aggregation: How to combine frame embeddings (mean, max, best_frame)
- prompt_template: Text template for queries (None, "a video of {}", etc.)
- ensemble_prompts: List of multiple prompts to average for each query
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from dataclasses import dataclass, asdict
from tqdm import tqdm

try:
    import torch
    import open_clip
    from PIL import Image
except ImportError:
    torch = None
    open_clip = None
    Image = None

try:
    import faiss
except ImportError:
    faiss = None


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Import shared VideoMetadata (avoids pulling in JAX from indexing.py)
from models import VideoMetadata


# ─── Built-in Prompt Templates ──────────────────────────────────────

PROMPT_TEMPLATES = {
    'none': '{text}',
    'video': 'a video of {text}',
    'photo': 'a photo of {text}',
    'scene': 'a scene showing {text}',
    'action': 'a person {text}',
    'cooking': 'a cooking video showing {text}',
    'clip': 'a short video clip of {text}',
}

# Default ensemble templates (used when ensemble is enabled without LLM)
DEFAULT_ENSEMBLE_TEMPLATES = [
    '{text}',
    'a video of {text}',
    'a photo of {text}',
    'a scene showing {text}',
    'a short clip of {text}',
]


class OpenCLIPVideoIndexer:
    """
    OpenCLIP-based video indexing system (baseline) with tunable parameters.
    
    Encodes videos by sampling frames, encoding each with CLIP's image encoder,
    and combining frame embeddings using a configurable aggregation strategy.
    
    Tunable parameters:
    - num_frames: Number of frames to sample (default: 16)
    - aggregation: 'mean', 'max', or 'best_frame' (default: 'mean')
    """
    
    # Available models (name -> pretrained weights)
    MODELS = {
        'ViT-B-32': 'laion2b_s34b_b79k',
        'ViT-B-16': 'datacomp_xl_s13b_b90k',
        'ViT-L-14': 'laion2b_s32b_b82k',
    }
    
    def __init__(
        self,
        model_name: str = 'ViT-B-32',
        index_dir: str = './video_index_openclip',
        device: str = 'cuda:0',
        num_frames: int = 16,
        aggregation: str = 'mean'
    ):
        """
        Initialize the OpenCLIP Video Indexer.
        
        Args:
            model_name: OpenCLIP model name (ViT-B-32, ViT-B-16, or ViT-L-14)
            index_dir: Directory to store the FAISS index and metadata
            device: Device to use ('cuda:0', 'cpu', etc.)
            num_frames: Number of frames to sample per video/window
            aggregation: Frame aggregation method ('mean', 'max', or 'best_frame')
        """
        if open_clip is None:
            raise ImportError(
                "OpenCLIP not installed. Run: pip install open_clip_torch"
            )
        if faiss is None:
            raise ImportError(
                "FAISS not installed. Run: pip install faiss-cpu or faiss-gpu"
            )
        
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.num_frames = num_frames
        self.aggregation = aggregation
        
        # Determine torch device
        if 'cuda' in device and torch.cuda.is_available():
            self.device = torch.device(device.replace('cuda:', 'cuda:'))
        else:
            self.device = torch.device('cpu')
            if 'cuda' in device:
                logger.warning("CUDA not available, falling back to CPU")
        
        # Load OpenCLIP model
        pretrained = self.MODELS.get(model_name, 'laion2b_s34b_b79k')
        logger.info(f"Loading OpenCLIP model: {model_name} (pretrained: {pretrained})")
        logger.info(f"  num_frames={num_frames}, aggregation={aggregation}")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # FAISS index and metadata
        self.index = None
        self.metadata_list: List[VideoMetadata] = []
        self.video_id_to_idx: Dict[str, int] = {}
        
        # Store per-frame embeddings for best_frame aggregation at query time
        self._frame_embeddings: Dict[str, np.ndarray] = {}
        
        logger.info(f"OpenCLIPVideoIndexer initialized on {self.device}")
    
    def extract_frames(
        self,
        video_path: str,
        num_frames: int = None,
        start_time: float = 0.0,
        end_time: float = -1.0
    ) -> Tuple[List, Dict]:
        """
        Extract frames from a video and preprocess them for CLIP.
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to sample (uses self.num_frames if None)
            start_time: Start time for windowed extraction
            end_time: End time for windowed extraction (-1 = full video)
            
        Returns:
            Tuple of (list of PIL Images, video info dict)
        """
        if num_frames is None:
            num_frames = self.num_frames
        
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
        
        # Calculate frame indices (uniformly sampled)
        if window_frames <= num_frames:
            frame_indices = [start_frame + i for i in range(window_frames)]
        else:
            frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames extracted from video: {video_path}")
        
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
    
    def get_frame_embeddings(
        self,
        video_path: str,
        num_frames: int = None,
        start_time: float = 0.0,
        end_time: float = -1.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Extract per-frame CLIP embeddings (normalized).
        
        Returns:
            Tuple of (frame_embeddings [N x D], video_info dict)
        """
        frames, video_info = self.extract_frames(
            video_path, num_frames=num_frames,
            start_time=start_time, end_time=end_time
        )
        
        preprocessed = torch.stack([self.preprocess(f) for f in frames]).to(self.device)
        
        with torch.no_grad():
            frame_features = self.model.encode_image(preprocessed)
            frame_features = frame_features / frame_features.norm(dim=-1, keepdim=True)
        
        return frame_features.float().cpu().numpy().astype(np.float32), video_info
    
    def get_video_embedding(
        self,
        video_path: str,
        num_frames: int = None,
        start_time: float = 0.0,
        end_time: float = -1.0,
        aggregation: str = None,
        store_frames: bool = False,
        video_id: str = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Extract embedding for a video using configurable aggregation.
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to sample (uses self.num_frames if None)
            start_time: Start time for windowed extraction
            end_time: End time for windowed extraction (-1 = full video)
            aggregation: Override aggregation method (uses self.aggregation if None)
            store_frames: If True, store per-frame embeddings for later best_frame matching
            video_id: Video ID for storing frame embeddings
            
        Returns:
            Tuple of (embedding array [D], video info dict)
        """
        if aggregation is None:
            aggregation = self.aggregation
        
        frame_embeddings, video_info = self.get_frame_embeddings(
            video_path, num_frames=num_frames,
            start_time=start_time, end_time=end_time
        )
        
        # Store per-frame embeddings if needed for best_frame matching
        if store_frames and video_id:
            self._frame_embeddings[video_id] = frame_embeddings
        
        if aggregation == 'mean':
            # Average pool all frame embeddings
            video_embedding = frame_embeddings.mean(axis=0)
        elif aggregation == 'max':
            # Max pool across frames (take max per dimension)
            video_embedding = frame_embeddings.max(axis=0)
        elif aggregation == 'best_frame':
            # For indexing, use mean as the representative embedding
            # The actual best-frame selection happens at query time
            video_embedding = frame_embeddings.mean(axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        # Normalize
        norm = np.linalg.norm(video_embedding)
        if norm > 0:
            video_embedding = video_embedding / norm
        
        return video_embedding.astype(np.float32), video_info
    
    def _compute_windows(
        self,
        duration: float,
        window_size: float = 5.0,
        window_overlap: float = 0.5
    ) -> List[Tuple[float, float]]:
        """Compute temporal sliding windows."""
        if duration <= window_size:
            return [(0.0, duration)]
        
        step_size = window_size * (1.0 - window_overlap)
        windows = []
        start = 0.0
        
        while start < duration:
            end = min(start + window_size, duration)
            if end - start >= window_size * 0.5:
                windows.append((start, end))
            start += step_size
        
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
        Index all videos in a directory using OpenCLIP frame embeddings.
        
        Args:
            video_dir: Directory containing video files
            use_windowing: If True, use temporal sliding windows
            window_size: Window size in seconds
            window_overlap: Window overlap fraction
            
        Returns:
            Number of entries indexed
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
        
        logger.info(f"Found {len(video_files)} video files to index with OpenCLIP")
        logger.info(f"  Model: {self.model_name}, Frames: {self.num_frames}, Aggregation: {self.aggregation}")
        if use_windowing:
            logger.info(f"  Temporal windowing: window_size={window_size}s, overlap={window_overlap}")
        else:
            logger.info("  Temporal windowing: disabled")
        
        embeddings_list = []
        feature_dim = None
        store_frames = (self.aggregation == 'best_frame')
        
        for video_file in tqdm(video_files, desc="Indexing videos (OpenCLIP)"):
            try:
                base_video_id = video_file.stem
                
                if use_windowing:
                    cap = cv2.VideoCapture(str(video_file))
                    if not cap.isOpened():
                        logger.error(f"Cannot open video: {video_file}")
                        continue
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    duration = total_frames / fps if fps > 0 else 0
                    cap.release()
                    
                    windows = self._compute_windows(duration, window_size, window_overlap)
                    
                    for w_idx, (w_start, w_end) in enumerate(windows):
                        if len(windows) == 1:
                            video_id = base_video_id
                        else:
                            video_id = f"{base_video_id}_w{w_idx}"
                        
                        if video_id in self.video_id_to_idx:
                            continue
                        
                        embedding, video_info = self.get_video_embedding(
                            str(video_file), start_time=w_start, end_time=w_end,
                            store_frames=store_frames, video_id=video_id
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
                    
                    logger.info(f"Indexed {base_video_id}: {len(windows)} window(s)")
                
                else:
                    video_id = base_video_id
                    
                    if video_id in self.video_id_to_idx:
                        continue
                    
                    embedding, video_info = self.get_video_embedding(
                        str(video_file),
                        store_frames=store_frames, video_id=video_id
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
                    
                    logger.info(f"Indexed {video_id}: dim={embedding.shape[0]}")
            
            except Exception as e:
                logger.error(f"Error indexing {video_file}: {e}")
                continue
        
        if not embeddings_list:
            logger.error("No videos were successfully indexed")
            return 0
        
        # Create FAISS index (embeddings already normalized)
        logger.info(f"Creating FAISS index with {len(embeddings_list)} entries")
        embeddings_array = np.vstack(embeddings_list).astype(np.float32)
        
        self.index = faiss.IndexFlatIP(embeddings_array.shape[1])
        self.index.add(embeddings_array)
        
        self.save_index()
        
        logger.info(f"Successfully indexed {len(embeddings_list)} entries with OpenCLIP")
        return len(embeddings_list)
    
    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[str, float, 'VideoMetadata']]:
        """Search for similar videos using a query embedding."""
        if self.index is None:
            raise RuntimeError("Index not initialized. Call index_videos() first.")
        
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        similarities, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if 0 <= idx < len(self.metadata_list):
                metadata = self.metadata_list[idx]
                results.append((metadata.video_id, float(similarity), metadata))
        
        return results
    
    def search_best_frame(
        self,
        query_embedding: np.ndarray,
        video_id: str
    ) -> float:
        """
        Find the best-matching frame for a query within a specific video.
        
        Used when aggregation='best_frame'. Instead of comparing against the
        mean embedding, compares the query against each individual frame and
        returns the maximum similarity.
        
        Args:
            query_embedding: Text query embedding [D]
            video_id: Video ID to search within
            
        Returns:
            Maximum frame-level similarity score
        """
        if video_id not in self._frame_embeddings:
            # Fall back to FAISS search
            results = self.search_by_embedding(query_embedding, k=len(self.metadata_list))
            for vid_id, sim, _ in results:
                if vid_id == video_id:
                    return sim
            return -1.0
        
        frame_embs = self._frame_embeddings[video_id]  # [N, D]
        query = query_embedding.reshape(1, -1)
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm
        
        # Cosine similarity with each frame
        similarities = (frame_embs @ query.T).flatten()
        return float(similarities.max())
    
    def save_index(self):
        """Save the FAISS index and metadata to disk."""
        if self.index is None:
            return
        
        index_path = self.index_dir / 'faiss_index.bin'
        metadata_path = self.index_dir / 'metadata.json'
        
        faiss.write_index(self.index, str(index_path))
        
        metadata_dicts = [asdict(m) for m in self.metadata_list]
        with open(metadata_path, 'w') as f:
            json.dump({
                'metadata': metadata_dicts,
                'video_id_to_idx': self.video_id_to_idx,
                'model_name': f'openclip_{self.model_name}',
                'num_frames': self.num_frames,
                'aggregation': self.aggregation
            }, f, indent=2)
        
        # Save frame embeddings if using best_frame
        if self._frame_embeddings:
            frame_emb_path = self.index_dir / 'frame_embeddings.npz'
            np.savez_compressed(
                str(frame_emb_path),
                **{k: v for k, v in self._frame_embeddings.items()}
            )
        
        logger.info(f"Saved OpenCLIP index to {self.index_dir}")
    
    def load_index(self):
        """Load the FAISS index and metadata from disk."""
        index_path = self.index_dir / 'faiss_index.bin'
        metadata_path = self.index_dir / 'metadata.json'
        
        if not index_path.exists() or not metadata_path.exists():
            logger.warning("OpenCLIP index files not found")
            return False
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        saved_model = data.get('model_name', '')
        if 'openclip' not in saved_model.lower():
            logger.warning(f"Existing index was not created with OpenCLIP ({saved_model}). "
                          f"Re-indexing required.")
            return False
        
        self.index = faiss.read_index(str(index_path))
        self.metadata_list = [VideoMetadata(**m) for m in data['metadata']]
        self.video_id_to_idx = data['video_id_to_idx']
        
        # Load frame embeddings if available
        frame_emb_path = self.index_dir / 'frame_embeddings.npz'
        if frame_emb_path.exists():
            loaded = np.load(str(frame_emb_path))
            self._frame_embeddings = {k: loaded[k] for k in loaded.files}
        
        logger.info(f"Loaded OpenCLIP index: {len(self.metadata_list)} entries")
        return True


class OpenCLIPTextMatcher:
    """
    Matches script segments to video clips using OpenCLIP text-image similarity.
    
    Supports tunable parameters:
    - prompt_template: Template for text queries (e.g., "a video of {text}")
    - ensemble_prompts: Multiple prompts to average for each query
    - aggregation: Inherited from indexer, affects best_frame matching
    """
    
    def __init__(
        self,
        video_indexer: OpenCLIPVideoIndexer,
        min_similarity_threshold: float = 0.0,
        prompt_template: str = None,
        ensemble_prompts: List[str] = None
    ):
        """
        Initialize the OpenCLIP text matcher.
        
        Args:
            video_indexer: OpenCLIPVideoIndexer instance with indexed videos
            min_similarity_threshold: Minimum similarity to consider a match
            prompt_template: Template string with {text} placeholder
            ensemble_prompts: List of template strings to average
        """
        self.video_indexer = video_indexer
        self.min_similarity_threshold = min_similarity_threshold
        self.used_videos_history = []
        
        # Reuse the model and tokenizer from the indexer
        self.model = video_indexer.model
        self.tokenizer = video_indexer.tokenizer
        self.device = video_indexer.device
        self.aggregation = video_indexer.aggregation
        
        # Prompt configuration
        self.prompt_template = prompt_template
        self.ensemble_prompts = ensemble_prompts
        
        if ensemble_prompts:
            logger.info(f"OpenCLIPTextMatcher: ensemble mode with {len(ensemble_prompts)} prompts")
        elif prompt_template:
            logger.info(f"OpenCLIPTextMatcher: prompt template = '{prompt_template}'")
        else:
            logger.info("OpenCLIPTextMatcher: no prompt template (raw text)")
    
    def _apply_prompt_template(self, text: str, template: str = None) -> str:
        """Apply a prompt template to the text."""
        if template is None:
            template = self.prompt_template
        if template is None:
            return text
        return template.replace('{text}', text)
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text query using CLIP's text encoder.
        
        If ensemble_prompts is set, generates embeddings for all prompt
        variations and returns their average.
        
        Args:
            text: Text query string
            
        Returns:
            Normalized embedding array
        """
        if self.ensemble_prompts:
            # Ensemble: encode all prompt variations and average
            all_embeddings = []
            for template in self.ensemble_prompts:
                prompted_text = self._apply_prompt_template(text, template)
                emb = self._encode_single_text(prompted_text)
                all_embeddings.append(emb)
            
            avg_embedding = np.mean(all_embeddings, axis=0)
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm
            return avg_embedding.astype(np.float32)
        else:
            prompted_text = self._apply_prompt_template(text)
            return self._encode_single_text(prompted_text)
    
    def _encode_single_text(self, text: str) -> np.ndarray:
        """Encode a single text string to a normalized embedding."""
        tokens = self.tokenizer([text]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features[0].float().cpu().numpy().astype(np.float32)
    
    def get_num_indexed_videos(self) -> int:
        """Get the number of indexed videos."""
        return len(self.video_indexer.metadata_list)
    
    def get_all_video_metadata(self) -> List:
        """Get all video metadata from the indexer."""
        return self.video_indexer.metadata_list
    
    def compute_similarity_matrix(
        self,
        script_segments: List[Dict],
        match_only: bool = False
    ) -> Tuple[np.ndarray, List]:
        """
        Compute the full similarity matrix between all segments and all videos.
        
        For best_frame aggregation, uses per-frame similarity instead of
        mean-embedding similarity.
        """
        all_metadata = self.get_all_video_metadata()
        num_segments = len(script_segments)
        num_videos = len(all_metadata)
        
        if num_videos == 0:
            logger.error("No videos indexed!")
            return np.array([]), []
        
        logger.info(f"Computing similarity matrix (OpenCLIP): {num_segments} segments x {num_videos} videos")
        
        # Get all text embeddings
        text_embeddings = []
        for segment in script_segments:
            emb = self.get_text_embedding(segment['text'])
            text_embeddings.append(emb)
        text_embeddings = np.array(text_embeddings)
        
        # Build similarity matrix
        similarity_matrix = np.zeros((num_segments, num_videos))
        
        for seg_idx, segment in enumerate(script_segments):
            if self.aggregation == 'best_frame':
                # Best-frame: compare query against each frame individually
                for vid_idx, metadata in enumerate(all_metadata):
                    sim = self.video_indexer.search_best_frame(
                        text_embeddings[seg_idx], metadata.video_id
                    )
                    if match_only:
                        similarity_matrix[seg_idx, vid_idx] = sim
                    else:
                        similarity_score = (sim + 1.0) / 2.0
                        motion_score = self._calculate_motion_score(metadata, segment['duration'])
                        context_score = self._calculate_context_score(segment['text'], metadata)
                        similarity_matrix[seg_idx, vid_idx] = (
                            0.5 * similarity_score +
                            0.3 * motion_score +
                            0.2 * context_score
                        )
            else:
                # Mean or Max aggregation: use FAISS search
                results = self.video_indexer.search_by_embedding(
                    text_embeddings[seg_idx], k=num_videos
                )
                
                video_sim_map = {video_id: sim for video_id, sim, _ in results}
                
                for vid_idx, metadata in enumerate(all_metadata):
                    raw_similarity = video_sim_map.get(metadata.video_id, -1.0)
                    
                    if match_only:
                        similarity_matrix[seg_idx, vid_idx] = raw_similarity
                    else:
                        similarity_score = (raw_similarity + 1.0) / 2.0
                        motion_score = self._calculate_motion_score(metadata, segment['duration'])
                        context_score = self._calculate_context_score(segment['text'], metadata)
                        similarity_matrix[seg_idx, vid_idx] = (
                            0.5 * similarity_score +
                            0.3 * motion_score +
                            0.2 * context_score
                        )
        
        return similarity_matrix, all_metadata
    
    def _calculate_trim_times(
        self,
        video_duration: float,
        segment_duration: float,
        window_start: float = 0.0
    ) -> Tuple[float, float]:
        """Calculate trim start and end times."""
        if segment_duration <= 0:
            return window_start, window_start + video_duration
        if video_duration <= segment_duration:
            return window_start, window_start + video_duration
        trim_start = (video_duration - segment_duration) / 2
        return window_start + trim_start, window_start + trim_start + segment_duration
    
    def _calculate_motion_score(self, metadata, segment_duration: float) -> float:
        """Calculate score based on duration match."""
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
        import re
        keywords = re.findall(r'\b[a-z]{3,}\b', segment_text.lower())
        stop_words = {
            'the', 'and', 'this', 'that', 'with', 'from', 'they', 'have',
            'been', 'were', 'will', 'would', 'could', 'should', 'your',
            'you', 'for', 'are', 'was', 'not', 'but', 'what', 'all',
            'can', 'had', 'her', 'his', 'him', 'has', 'its', 'just',
            'into', 'over', 'such', 'than', 'then', 'them', 'these',
            'some', 'very', 'when', 'where', 'which', 'while', 'who',
            'why', 'how', 'each', 'she', 'does', 'doing', 'being'
        }
        keywords = [w for w in keywords if w not in stop_words]
        video_name = Path(metadata.file_path).name.lower()
        matches = sum(1 for kw in keywords if kw in video_name)
        if matches > 0:
            return min(1.0, 0.5 + matches * 0.25)
        return 0.5
    
    def match_segment_to_videos(
        self,
        segment_text: str,
        segment_duration: float,
        k: int = 10,
        allow_reuse: bool = True,
        used_videos=None,
        match_only: bool = False
    ) -> List[Dict]:
        """Find best matching video clips for a segment (greedy mode)."""
        if used_videos is None:
            used_videos = set()
        
        text_embedding = self.get_text_embedding(segment_text)
        num_videos = self.get_num_indexed_videos()
        if num_videos == 0:
            return []
        
        search_k = num_videos if match_only else min(num_videos, k * 5)
        
        if self.aggregation == 'best_frame':
            # For best_frame, compute similarity against all videos
            candidates = []
            all_metadata = self.get_all_video_metadata()
            for metadata in all_metadata:
                if not allow_reuse and metadata.video_id in used_videos:
                    continue
                similarity = self.video_indexer.search_best_frame(
                    text_embedding, metadata.video_id
                )
                candidates.append(self._build_candidate(
                    metadata, similarity, segment_text, segment_duration,
                    used_videos, match_only
                ))
        else:
            results = self.video_indexer.search_by_embedding(text_embedding, k=search_k)
            candidates = []
            for video_id, similarity, metadata in results:
                if not allow_reuse and video_id in used_videos:
                    continue
                candidates.append(self._build_candidate(
                    metadata, similarity, segment_text, segment_duration,
                    used_videos, match_only
                ))
        
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        return candidates[:k]
    
    def _build_candidate(
        self, metadata, similarity, segment_text, segment_duration,
        used_videos, match_only
    ) -> Dict:
        """Build a candidate dict from metadata and similarity."""
        similarity_score = (similarity + 1.0) / 2.0
        
        if match_only:
            motion_score = 0.0
            context_score = 0.0
            combined_score = similarity
        else:
            diversity_multiplier = 1.2 if metadata.video_id not in used_videos else 0.8
            motion_score = self._calculate_motion_score(metadata, segment_duration)
            context_score = self._calculate_context_score(segment_text, metadata)
            combined_score = (
                0.5 * similarity_score * diversity_multiplier +
                0.3 * motion_score +
                0.2 * context_score
            )
        
        return {
            'video_id': metadata.video_id,
            'file_path': metadata.file_path,
            'duration': metadata.duration,
            'similarity': similarity,
            'similarity_score': similarity_score,
            'motion_score': motion_score,
            'context_score': context_score,
            'combined_score': combined_score,
            'is_reused': metadata.video_id in used_videos
        }
    
    def select_best_clip(
        self,
        candidates: List[Dict],
        segment_duration: float,
        match_only: bool = False
    ) -> Optional[Dict]:
        """Select the best clip from candidates."""
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
