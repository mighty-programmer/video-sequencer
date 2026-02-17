"""
Shared data models for the Video Sequencing Pipeline.

This module contains dataclasses and shared types used across
multiple modules to avoid circular imports and unnecessary dependencies.
"""

from dataclasses import dataclass, asdict, field


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
