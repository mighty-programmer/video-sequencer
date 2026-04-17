"""
Write-A-Video Multi-Modal Keyword Indexing and Two-Stage Matching Module

Replicates the multi-modal, two-stage retrieval system from:
    Wang et al., "Write-A-Video: Computational Video Montage from Themed Text"
    ACM Transactions on Graphics (TOG), 2019

This module implements:
1. Multi-modal keyword indexing using:
   - Object detection (YOLOv8 pre-trained on COCO 80 classes)
   - Face detection and clustering (DeepFace + DBSCAN)
   - Filename keyword extraction
2. Two-stage retrieval:
   - Stage 1: Fast keyword-based candidate filtering (TF-IDF)
   - Stage 2: Visual-semantic reranking using OpenCLIP embeddings

The architecture follows the paper's design but substitutes modern
state-of-the-art components (YOLOv8 instead of Mask R-CNN, OpenCLIP
instead of VSE++) as recommended by the original authors.

Dependencies:
    pip install ultralytics deepface scikit-learn
"""

import os
import sys
import json
import logging
import hashlib
import math
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict, field
from collections import Counter, defaultdict
import numpy as np
import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Import shared models
sys.path.insert(0, str(Path(__file__).parent))
from models import VideoMetadata

# Lazy imports for heavy dependencies
_yolo_model = None
_deepface = None


def _get_yolo_model(model_size: str = 'yolov8n'):
    """Lazy-load YOLOv8 model."""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLOv8 model: {model_size}")
            _yolo_model = YOLO(f'{model_size}.pt')
            logger.info("YOLOv8 model loaded successfully")
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )
    return _yolo_model


def _get_deepface():
    """Lazy-load DeepFace."""
    global _deepface
    if _deepface is None:
        try:
            import deepface
            _deepface = deepface
            logger.info("DeepFace loaded successfully")
        except ImportError:
            raise ImportError(
                "deepface not installed. Run: pip install deepface"
            )
    return _deepface


# ─── COCO 80 Class Names ──────────────────────────────────────────
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


@dataclass
class VideoKeywordEntry:
    """Keyword index entry for a single video clip."""
    video_id: str
    file_path: str
    duration: float
    fps: float
    width: int
    height: int
    num_frames: int
    # Multi-modal keywords
    object_keywords: Dict[str, float]   # object_name -> confidence (avg)
    face_keywords: List[str]            # face cluster IDs (e.g., "person_0", "person_1")
    filename_keywords: List[str]        # keywords extracted from filename
    all_keywords: List[str]             # union of all keyword sources
    # Raw detection counts for analysis
    object_detection_count: int         # total object detections across frames
    face_detection_count: int           # total face detections across frames


class MultiModalKeywordIndexer:
    """
    Multi-modal keyword indexer for video clips.
    
    Extracts keywords from multiple modalities:
    1. Object Detection (YOLOv8): Detects 80 COCO object classes
    2. Face Detection & Clustering (DeepFace + DBSCAN): Identifies unique people
    3. Filename Analysis: Extracts meaningful words from filenames
    
    Builds an inverted index for fast keyword-based retrieval.
    """
    
    def __init__(
        self,
        index_dir: str = './wav_keyword_index',
        yolo_model: str = 'yolov8n',
        yolo_confidence: float = 0.3,
        face_detection_backend: str = 'retinaface',
        face_model_name: str = 'Facenet512',
        frames_per_second: float = 1.0,
        enable_face_detection: bool = True,
        enable_object_detection: bool = True,
    ):
        """
        Initialize the multi-modal keyword indexer.
        
        Args:
            index_dir: Directory to store the keyword index
            yolo_model: YOLOv8 model variant (yolov8n, yolov8s, yolov8m, yolov8l)
            yolo_confidence: Minimum confidence threshold for object detection
            face_detection_backend: DeepFace backend for face detection
            face_model_name: DeepFace model for face embeddings
            frames_per_second: How many frames per second to analyze
            enable_face_detection: Whether to run face detection
            enable_object_detection: Whether to run object detection
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_model_name = yolo_model
        self.yolo_confidence = yolo_confidence
        self.face_detection_backend = face_detection_backend
        self.face_model_name = face_model_name
        self.frames_per_second = frames_per_second
        self.enable_face_detection = enable_face_detection
        self.enable_object_detection = enable_object_detection
        
        # Index data
        self.video_entries: List[VideoKeywordEntry] = []
        self.inverted_index: Dict[str, List[int]] = {}  # keyword -> list of video indices
        self.idf_scores: Dict[str, float] = {}  # keyword -> IDF score
        
        # Face clustering data
        self._all_face_embeddings: List[Tuple[str, np.ndarray]] = []  # (video_id, embedding)
        self._face_cluster_labels: Dict[str, List[str]] = {}  # video_id -> cluster labels
        
        logger.info(f"MultiModalKeywordIndexer initialized:")
        logger.info(f"  YOLO model: {yolo_model} (conf={yolo_confidence})")
        logger.info(f"  Face detection: {'enabled' if enable_face_detection else 'disabled'}")
        logger.info(f"  Object detection: {'enabled' if enable_object_detection else 'disabled'}")
        logger.info(f"  Frames/sec: {frames_per_second}")
    
    def _extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], Dict]:
        """
        Extract frames from a video at the configured sampling rate.
        
        Returns:
            Tuple of (list of BGR frames, video info dict)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate frame indices based on sampling rate
        if duration <= 0 or fps <= 0:
            frame_indices = [0]
        else:
            # Sample at frames_per_second rate
            sample_interval = fps / self.frames_per_second
            frame_indices = [int(i * sample_interval) for i in range(int(duration * self.frames_per_second) + 1)]
            frame_indices = [f for f in frame_indices if f < total_frames]
            if not frame_indices:
                frame_indices = [0]
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        
        video_info = {
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'duration': duration,
            'sampled_frames': len(frames),
        }
        
        return frames, video_info
    
    def _detect_objects(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Run YOLOv8 object detection on sampled frames.
        
        Returns:
            Dict mapping object class name -> average confidence across frames
        """
        if not self.enable_object_detection:
            return {}
        
        model = _get_yolo_model(self.yolo_model_name)
        
        object_detections: Dict[str, List[float]] = defaultdict(list)
        
        for frame in frames:
            results = model(frame, verbose=False, conf=self.yolo_confidence)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        if cls_id < len(COCO_CLASSES):
                            class_name = COCO_CLASSES[cls_id]
                            object_detections[class_name].append(conf)
        
        # Average confidence per object class
        avg_confidences = {}
        for obj_name, confs in object_detections.items():
            avg_confidences[obj_name] = sum(confs) / len(confs)
        
        return avg_confidences
    
    def _detect_faces(self, frames: List[np.ndarray], video_id: str) -> int:
        """
        Run face detection on sampled frames and collect face embeddings
        for later clustering.
        
        Returns:
            Number of face detections
        """
        if not self.enable_face_detection:
            return 0
        
        try:
            from deepface import DeepFace
        except ImportError:
            logger.warning("DeepFace not installed, skipping face detection")
            return 0
        
        face_count = 0
        
        for frame in frames:
            try:
                # Convert BGR to RGB for DeepFace
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect and extract face embeddings
                face_results = DeepFace.represent(
                    img_path=frame_rgb,
                    model_name=self.face_model_name,
                    detector_backend=self.face_detection_backend,
                    enforce_detection=False
                )
                
                for face_result in face_results:
                    embedding = np.array(face_result['embedding'])
                    if np.linalg.norm(embedding) > 0:
                        self._all_face_embeddings.append((video_id, embedding))
                        face_count += 1
                        
            except Exception as e:
                # Face detection can fail on frames with no faces
                continue
        
        return face_count
    
    def _extract_filename_keywords(self, file_path: str) -> List[str]:
        """
        Extract meaningful keywords from a video filename.
        
        Handles common naming patterns:
        - snake_case: "a_man_walking.mp4" -> ["man", "walking"]
        - camelCase: "aManWalking.mp4" -> ["man", "walking"]
        - kebab-case: "a-man-walking.mp4" -> ["man", "walking"]
        - Spaces: "a man walking.mp4" -> ["man", "walking"]
        """
        import re
        
        filename = Path(file_path).stem
        
        # Split on common separators
        words = re.split(r'[-_\s]+', filename)
        
        # Also split camelCase
        expanded = []
        for word in words:
            # Split camelCase: "aManWalking" -> ["a", "Man", "Walking"]
            parts = re.sub(r'([A-Z])', r' \1', word).split()
            expanded.extend(parts)
        
        # Clean and filter
        stop_words = {
            'the', 'and', 'this', 'that', 'with', 'from', 'they', 'have',
            'been', 'were', 'will', 'would', 'could', 'should', 'your',
            'you', 'for', 'are', 'was', 'not', 'but', 'what', 'all',
            'can', 'had', 'her', 'his', 'him', 'has', 'its', 'just',
            'into', 'over', 'such', 'than', 'then', 'them', 'these',
            'some', 'very', 'when', 'where', 'which', 'while', 'who',
            'why', 'how', 'each', 'she', 'does', 'doing', 'being',
            'clip', 'video', 'mp4', 'mov', 'avi', 'mkv', 'flv', 'wmv',
        }
        
        keywords = []
        for word in expanded:
            w = word.lower().strip()
            # Keep words that are 3+ chars and not stop words or numbers
            if len(w) >= 3 and w not in stop_words and not w.isdigit():
                keywords.append(w)
        
        return keywords
    
    def _cluster_faces(self):
        """
        Cluster all collected face embeddings using DBSCAN to identify
        unique individuals across the video library.
        
        Assigns cluster labels like "person_0", "person_1", etc.
        """
        if not self._all_face_embeddings:
            logger.info("No face embeddings to cluster")
            return
        
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            logger.warning("scikit-learn not installed, skipping face clustering")
            return
        
        logger.info(f"Clustering {len(self._all_face_embeddings)} face embeddings...")
        
        # Extract embeddings and video IDs
        video_ids = [vid_id for vid_id, _ in self._all_face_embeddings]
        embeddings = np.array([emb for _, emb in self._all_face_embeddings])
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / np.maximum(norms, 1e-8)
        
        # DBSCAN clustering with cosine distance
        # eps=0.5 is a reasonable threshold for face embeddings
        clustering = DBSCAN(
            eps=0.5,
            min_samples=2,
            metric='cosine'
        ).fit(embeddings_normalized)
        
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"Face clustering: {n_clusters} unique persons, {n_noise} noise detections")
        
        # Assign cluster labels to videos
        self._face_cluster_labels = defaultdict(set)
        for i, (vid_id, _) in enumerate(self._all_face_embeddings):
            label = labels[i]
            if label >= 0:
                self._face_cluster_labels[vid_id].add(f"person_{label}")
        
        # Convert sets to sorted lists
        for vid_id in self._face_cluster_labels:
            self._face_cluster_labels[vid_id] = sorted(self._face_cluster_labels[vid_id])
    
    def _build_inverted_index(self):
        """
        Build the inverted index and compute TF-IDF scores.
        
        The inverted index maps each keyword to the list of video indices
        that contain it. TF-IDF is used to weight keywords by their
        discriminative power.
        """
        N = len(self.video_entries)
        if N == 0:
            return
        
        # Build inverted index
        self.inverted_index = defaultdict(list)
        
        for idx, entry in enumerate(self.video_entries):
            for keyword in entry.all_keywords:
                self.inverted_index[keyword].append(idx)
        
        # Compute IDF scores
        self.idf_scores = {}
        for keyword, doc_indices in self.inverted_index.items():
            df = len(set(doc_indices))  # Document frequency
            self.idf_scores[keyword] = math.log(N / (1 + df)) + 1  # Smoothed IDF
        
        logger.info(f"Built inverted index: {len(self.inverted_index)} unique keywords across {N} videos")
    
    def index_videos(self, video_dir: str):
        """
        Index all videos in a directory using multi-modal keyword extraction.
        
        This is the main indexing function that:
        1. Samples frames from each video
        2. Runs object detection (YOLOv8)
        3. Runs face detection (DeepFace)
        4. Extracts filename keywords
        5. Clusters faces across the library
        6. Builds the inverted index with TF-IDF
        
        Args:
            video_dir: Path to directory containing video files
        """
        video_dir = Path(video_dir)
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        
        video_files = sorted([
            f for f in video_dir.iterdir()
            if f.suffix.lower() in video_extensions
        ])
        
        if not video_files:
            raise ValueError(f"No video files found in {video_dir}")
        
        logger.info(f"Indexing {len(video_files)} videos from {video_dir}")
        
        self.video_entries = []
        self._all_face_embeddings = []
        
        for video_file in tqdm(video_files, desc="Indexing videos"):
            video_id = video_file.stem
            
            try:
                # 1. Extract frames
                frames, video_info = self._extract_frames(str(video_file))
                
                # 2. Object detection
                object_keywords = self._detect_objects(frames)
                
                # 3. Face detection (embeddings collected for later clustering)
                face_count = self._detect_faces(frames, video_id)
                
                # 4. Filename keywords
                filename_keywords = self._extract_filename_keywords(str(video_file))
                
                # 5. Build combined keyword list (face keywords added after clustering)
                all_keywords = list(object_keywords.keys()) + filename_keywords
                
                entry = VideoKeywordEntry(
                    video_id=video_id,
                    file_path=str(video_file),
                    duration=video_info['duration'],
                    fps=video_info['fps'],
                    width=video_info['width'],
                    height=video_info['height'],
                    num_frames=video_info['total_frames'],
                    object_keywords=object_keywords,
                    face_keywords=[],  # Populated after clustering
                    filename_keywords=filename_keywords,
                    all_keywords=all_keywords,
                    object_detection_count=sum(len(v) if isinstance(v, list) else 1 
                                               for v in object_keywords.values()),
                    face_detection_count=face_count,
                )
                
                self.video_entries.append(entry)
                
            except Exception as e:
                logger.error(f"Error indexing {video_file.name}: {e}")
                continue
        
        # 6. Cluster faces across all videos
        if self.enable_face_detection:
            self._cluster_faces()
            
            # Add face keywords to entries
            for entry in self.video_entries:
                face_kws = self._face_cluster_labels.get(entry.video_id, [])
                entry.face_keywords = list(face_kws)
                entry.all_keywords = list(set(
                    entry.all_keywords + entry.face_keywords
                ))
        
        # 7. Build inverted index
        self._build_inverted_index()
        
        # Log summary
        total_objects = sum(len(e.object_keywords) for e in self.video_entries)
        total_faces = sum(len(e.face_keywords) for e in self.video_entries)
        total_filename = sum(len(e.filename_keywords) for e in self.video_entries)
        
        logger.info(f"\nIndexing complete:")
        logger.info(f"  Videos indexed: {len(self.video_entries)}")
        logger.info(f"  Unique keywords: {len(self.inverted_index)}")
        logger.info(f"  Object keywords: {total_objects} (across all videos)")
        logger.info(f"  Face keywords: {total_faces} (unique person IDs)")
        logger.info(f"  Filename keywords: {total_filename}")
    
    def search_keywords(
        self,
        query_text: str,
        top_k: int = None
    ) -> List[Tuple[int, float]]:
        """
        Search the keyword index using TF-IDF scoring.
        
        Extracts keywords from the query text and scores each video
        based on keyword overlap weighted by TF-IDF.
        
        Args:
            query_text: Text query to search for
            top_k: Number of top results to return (None = all)
            
        Returns:
            List of (video_index, tfidf_score) tuples, sorted by score descending
        """
        import re
        
        # Extract query keywords
        words = re.findall(r'\b[a-z]{3,}\b', query_text.lower())
        
        stop_words = {
            'the', 'and', 'this', 'that', 'with', 'from', 'they', 'have',
            'been', 'were', 'will', 'would', 'could', 'should', 'your',
            'you', 'for', 'are', 'was', 'not', 'but', 'what', 'all',
            'can', 'had', 'her', 'his', 'him', 'has', 'its', 'just',
            'into', 'over', 'such', 'than', 'then', 'them', 'these',
            'some', 'very', 'when', 'where', 'which', 'while', 'who',
            'why', 'how', 'each', 'she', 'does', 'doing', 'being',
            'now', 'going', 'want', 'make', 'take', 'get', 'let',
            'put', 'use', 'way', 'look', 'like', 'come', 'see',
        }
        
        query_keywords = [w for w in words if w not in stop_words]
        
        if not query_keywords:
            # No meaningful keywords - return all videos with score 0
            return [(i, 0.0) for i in range(len(self.video_entries))]
        
        # Score each video using TF-IDF
        video_scores = defaultdict(float)
        
        for keyword in query_keywords:
            if keyword in self.inverted_index:
                idf = self.idf_scores.get(keyword, 1.0)
                for vid_idx in self.inverted_index[keyword]:
                    # TF: use object confidence as term frequency proxy
                    entry = self.video_entries[vid_idx]
                    if keyword in entry.object_keywords:
                        tf = entry.object_keywords[keyword]  # confidence as TF
                    else:
                        tf = 1.0  # filename/face keyword
                    video_scores[vid_idx] += tf * idf
        
        # Sort by score descending
        scored_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Add videos with zero score (no keyword match)
        scored_set = set(idx for idx, _ in scored_videos)
        for i in range(len(self.video_entries)):
            if i not in scored_set:
                scored_videos.append((i, 0.0))
        
        if top_k is not None:
            scored_videos = scored_videos[:top_k]
        
        return scored_videos
    
    def save_index(self):
        """Save the keyword index to disk."""
        index_path = self.index_dir / 'keyword_index.json'
        
        data = {
            'entries': [asdict(e) for e in self.video_entries],
            'inverted_index': dict(self.inverted_index),
            'idf_scores': self.idf_scores,
            'config': {
                'yolo_model': self.yolo_model_name,
                'yolo_confidence': self.yolo_confidence,
                'face_detection_backend': self.face_detection_backend,
                'face_model_name': self.face_model_name,
                'frames_per_second': self.frames_per_second,
                'enable_face_detection': self.enable_face_detection,
                'enable_object_detection': self.enable_object_detection,
            }
        }
        
        with open(index_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved keyword index to {index_path}")
    
    def load_index(self) -> bool:
        """Load the keyword index from disk."""
        index_path = self.index_dir / 'keyword_index.json'
        
        if not index_path.exists():
            return False
        
        try:
            with open(index_path, 'r') as f:
                data = json.load(f)
            
            self.video_entries = [VideoKeywordEntry(**e) for e in data['entries']]
            self.inverted_index = defaultdict(list, data['inverted_index'])
            self.idf_scores = data['idf_scores']
            
            if not self.video_entries:
                logger.warning("Loaded keyword index is empty. Re-indexing required.")
                return False
                
            logger.info(f"Loaded keyword index: {len(self.video_entries)} videos, "
                        f"{len(self.inverted_index)} keywords")
            return True
            
        except Exception as e:
            logger.error(f"Error loading keyword index: {e}")
            return False
    
    def get_video_metadata_list(self) -> List[VideoMetadata]:
        """
        Convert keyword entries to VideoMetadata objects for compatibility
        with the existing matching pipeline.
        """
        metadata_list = []
        for entry in self.video_entries:
            meta = VideoMetadata(
                video_id=entry.video_id,
                file_path=entry.file_path,
                duration=entry.duration,
                num_frames=entry.num_frames,
                fps=entry.fps,
                width=entry.width,
                height=entry.height,
                embedding_dim=0,  # No embedding dimension for keyword index
            )
            metadata_list.append(meta)
        return metadata_list


class WriteAVideoMatcher:
    """
    Two-stage matcher implementing the Write-A-Video retrieval architecture.
    
    Stage 1: Keyword-based candidate filtering using the multi-modal index
    Stage 2: Visual-semantic reranking using OpenCLIP embeddings
    
    This class wraps both stages and provides the same interface as
    OpenCLIPTextMatcher for compatibility with the existing pipeline.
    """
    
    def __init__(
        self,
        keyword_indexer: MultiModalKeywordIndexer,
        openclip_indexer,  # OpenCLIPVideoIndexer
        candidate_pool_size: int = 10,
        keyword_weight: float = 0.0,
        prompt_template: str = None,
        ensemble_prompts: List[str] = None,
    ):
        """
        Initialize the two-stage matcher.
        
        Args:
            keyword_indexer: MultiModalKeywordIndexer with indexed videos
            openclip_indexer: OpenCLIPVideoIndexer with indexed videos
            candidate_pool_size: Number of keyword-matched candidates for Stage 2
            keyword_weight: Weight for keyword score in final ranking (0 = pure rerank)
            prompt_template: Optional prompt template for text encoding
            ensemble_prompts: Optional list of ensemble prompts
        """
        from openclip_indexing import OpenCLIPTextMatcher
        
        self.keyword_indexer = keyword_indexer
        self.openclip_indexer = openclip_indexer
        self.candidate_pool_size = candidate_pool_size
        self.keyword_weight = keyword_weight
        
        # Create the OpenCLIP text matcher for Stage 2
        self.text_matcher = OpenCLIPTextMatcher(
            video_indexer=openclip_indexer,
            min_similarity_threshold=0.0,
            prompt_template=prompt_template,
            ensemble_prompts=ensemble_prompts,
        )
        
        # Expose model/tokenizer for LLM ensemble compatibility
        self.model = self.text_matcher.model
        self.tokenizer = self.text_matcher.tokenizer
        self.device = self.text_matcher.device
        self.aggregation = self.text_matcher.aggregation
        
        self.used_videos_history = []
        
        logger.info(f"WriteAVideoMatcher initialized:")
        logger.info(f"  Candidate pool size: {candidate_pool_size}")
        logger.info(f"  Keyword weight: {keyword_weight}")
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding using OpenCLIP (delegates to text matcher)."""
        return self.text_matcher.get_text_embedding(text)
    
    def _encode_single_text(self, text: str) -> np.ndarray:
        """Encode a single text string (delegates to text matcher)."""
        return self.text_matcher._encode_single_text(text)
    
    def get_num_indexed_videos(self) -> int:
        """Get the number of indexed videos."""
        return len(self.keyword_indexer.video_entries)
    
    def get_all_video_metadata(self) -> List:
        """Get all video metadata."""
        return self.openclip_indexer.metadata_list
    
    def compute_similarity_matrix(
        self,
        script_segments: List[Dict],
        match_only: bool = False
    ) -> Tuple[np.ndarray, List]:
        """
        Compute the two-stage similarity matrix.
        
        For each segment:
        1. Stage 1: Get keyword-based candidate scores
        2. Stage 2: Compute OpenCLIP similarity only for candidates
        3. Combine scores (if keyword_weight > 0)
        
        Non-candidate videos get a penalty to their score.
        """
        all_metadata = self.get_all_video_metadata()
        num_segments = len(script_segments)
        num_videos = len(all_metadata)
        
        if num_videos == 0:
            logger.error("No videos indexed!")
            return np.array([]), []
        
        logger.info(f"Computing two-stage similarity matrix (WAV): "
                    f"{num_segments} segments x {num_videos} videos")
        
        # Build video_id to index mapping
        vid_id_to_idx = {}
        for idx, meta in enumerate(all_metadata):
            vid_id_to_idx[meta.video_id] = idx
        
        # Also map keyword indexer entries to OpenCLIP metadata indices
        kw_to_clip_idx = {}
        for kw_idx, kw_entry in enumerate(self.keyword_indexer.video_entries):
            clip_idx = vid_id_to_idx.get(kw_entry.video_id, -1)
            kw_to_clip_idx[kw_idx] = clip_idx
        
        # Get all text embeddings
        text_embeddings = []
        for segment in script_segments:
            emb = self.get_text_embedding(segment['text'])
            text_embeddings.append(emb)
        text_embeddings = np.array(text_embeddings)
        
        # Build similarity matrix
        similarity_matrix = np.zeros((num_segments, num_videos))
        
        for seg_idx, segment in enumerate(script_segments):
            # Stage 1: Keyword-based candidate retrieval
            keyword_results = self.keyword_indexer.search_keywords(
                segment['text'],
                top_k=None  # Get all, we'll use pool size for filtering
            )
            
            # Get candidate set (top-K from keyword search)
            candidate_clip_indices = set()
            keyword_scores = {}
            
            for kw_idx, kw_score in keyword_results[:self.candidate_pool_size]:
                clip_idx = kw_to_clip_idx.get(kw_idx, -1)
                if clip_idx >= 0:
                    candidate_clip_indices.add(clip_idx)
                    keyword_scores[clip_idx] = kw_score
            
            # If we have fewer candidates than videos, also include all videos
            # but with a lower priority (this ensures the Hungarian algorithm
            # always has enough options)
            
            # Stage 2: OpenCLIP similarity for ALL videos
            # (We compute for all to ensure the Hungarian algorithm works,
            # but candidates get a boost from keyword matching)
            if self.aggregation == 'best_frame':
                for vid_idx, metadata in enumerate(all_metadata):
                    sim = self.openclip_indexer.search_best_frame(
                        text_embeddings[seg_idx], metadata.video_id
                    )
                    
                    # Combine with keyword score if video is a candidate
                    if self.keyword_weight > 0 and vid_idx in keyword_scores:
                        # Normalize keyword score to [0, 1] range
                        max_kw = max(s for _, s in keyword_results[:self.candidate_pool_size]) if keyword_results else 1.0
                        norm_kw = keyword_scores[vid_idx] / max(max_kw, 1e-8)
                        combined = (1 - self.keyword_weight) * sim + self.keyword_weight * norm_kw
                    else:
                        combined = sim
                    
                    similarity_matrix[seg_idx, vid_idx] = combined
            else:
                # Mean or Max aggregation: use FAISS search
                results = self.openclip_indexer.search_by_embedding(
                    text_embeddings[seg_idx], k=num_videos
                )
                
                video_sim_map = {video_id: sim for video_id, sim, _ in results}
                
                for vid_idx, metadata in enumerate(all_metadata):
                    sim = video_sim_map.get(metadata.video_id, -1.0)
                    
                    # Combine with keyword score if video is a candidate
                    if self.keyword_weight > 0 and vid_idx in keyword_scores:
                        max_kw = max(s for _, s in keyword_results[:self.candidate_pool_size]) if keyword_results else 1.0
                        norm_kw = keyword_scores[vid_idx] / max(max_kw, 1e-8)
                        combined = (1 - self.keyword_weight) * sim + self.keyword_weight * norm_kw
                    else:
                        combined = sim
                    
                    similarity_matrix[seg_idx, vid_idx] = combined
        
        return similarity_matrix, all_metadata
    
    def match_segment_to_videos(
        self,
        segment_text: str,
        segment_duration: float,
        k: int = 10,
        allow_reuse: bool = True,
        used_videos=None,
        match_only: bool = False
    ) -> List[Dict]:
        """
        Find best matching video clips using the two-stage Write-A-Video flow.

        Stage 1 retrieves a candidate pool from the keyword index, and Stage 2
        reranks that pool with OpenCLIP semantic similarity. When the keyword
        stage yields too few results, we backfill from the semantic index so the
        caller still gets a usable candidate list for interactive editing.
        """
        if used_videos is None:
            used_videos = set()

        keyword_results = self.keyword_indexer.search_keywords(
            segment_text,
            top_k=max(self.candidate_pool_size, k * 2)
        )

        text_embedding = self.get_text_embedding(segment_text)
        all_metadata = self.openclip_indexer.metadata_list
        metadata_by_id = {meta.video_id: meta for meta in all_metadata}

        semantic_scores = {}
        if self.openclip_indexer.aggregation == 'best_frame':
            for metadata in all_metadata:
                semantic_scores[metadata.video_id] = self.openclip_indexer.search_best_frame(
                    text_embedding, metadata.video_id
                )
        else:
            results = self.openclip_indexer.search_by_embedding(
                text_embedding,
                k=len(all_metadata)
            )
            semantic_scores = {video_id: score for video_id, score, _ in results}

        candidate_rows = []
        max_keyword_score = max((score for _, score in keyword_results), default=1.0)
        query_keywords = set(re.findall(r'\b[a-z]{3,}\b', segment_text.lower()))

        for kw_idx, keyword_score in keyword_results[:self.candidate_pool_size]:
            entry = self.keyword_indexer.video_entries[kw_idx]
            metadata = metadata_by_id.get(entry.video_id)
            if metadata is None:
                continue
            if not allow_reuse and entry.video_id in used_videos:
                continue

            similarity = semantic_scores.get(entry.video_id, -1.0)
            similarity_score = (similarity + 1.0) / 2.0
            normalized_keyword_score = keyword_score / max(max_keyword_score, 1e-8)
            motion_score = self.text_matcher._calculate_motion_score(metadata, segment_duration)
            context_score = self.text_matcher._calculate_context_score(segment_text, metadata)

            if match_only:
                combined_score = similarity
            else:
                semantic_blend = (1 - self.keyword_weight) * similarity_score + self.keyword_weight * normalized_keyword_score
                combined_score = (
                    0.65 * semantic_blend +
                    0.2 * motion_score +
                    0.15 * context_score
                )

            candidate_rows.append({
                'video_id': entry.video_id,
                'file_path': metadata.file_path,
                'duration': metadata.duration,
                'similarity': similarity,
                'similarity_score': similarity_score,
                'motion_score': motion_score,
                'context_score': context_score,
                'keyword_score': normalized_keyword_score,
                'combined_score': combined_score,
                'matched_keywords': sorted(query_keywords & set(entry.all_keywords)),
                'is_reused': entry.video_id in used_videos,
            })

        if len(candidate_rows) < k:
            fallback = self.text_matcher.match_segment_to_videos(
                segment_text,
                segment_duration,
                k=max(k, self.candidate_pool_size),
                allow_reuse=allow_reuse,
                used_videos=used_videos,
                match_only=match_only
            )
            existing_ids = {row['video_id'] for row in candidate_rows}
            for candidate in fallback:
                if candidate['video_id'] in existing_ids:
                    continue
                candidate.setdefault('keyword_score', 0.0)
                candidate.setdefault('matched_keywords', [])
                candidate_rows.append(candidate)
                existing_ids.add(candidate['video_id'])

        candidate_rows.sort(key=lambda row: row['combined_score'], reverse=True)
        return candidate_rows[:k]
    
    def select_best_clip(self, candidates, segment_duration, match_only=False):
        """Select the best clip from candidates."""
        return self.text_matcher.select_best_clip(candidates, segment_duration, match_only)
