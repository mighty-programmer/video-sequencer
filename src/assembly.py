"""
Video Assembly and Export Module

This module is responsible for:
1. Trimming video clips to specified durations
2. Adjusting video speed to match target durations (Smart Speed Control)
3. Concatenating clips in sequence
4. Adding audio (voice-over) to the final video
5. Exporting to various formats
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import json
import os

try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, AudioFileClip
except ImportError:
    VideoFileClip = None
    concatenate_videoclips = None
    CompositeVideoClip = None
    AudioFileClip = None

try:
    import cv2
except ImportError:
    cv2 = None


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VideoAssembler:
    """
    Assembles video clips into a final sequence with smart speed control.
    """
    
    def __init__(self, use_ffmpeg: bool = True):
        """
        Initialize the VideoAssembler.
        
        Args:
            use_ffmpeg: Whether to use FFmpeg for video operations (recommended)
        """
        self.use_ffmpeg = use_ffmpeg
        
        if use_ffmpeg:
            self._check_ffmpeg()
        else:
            if VideoFileClip is None:
                raise ImportError("MoviePy not installed. Run: pip install moviepy")
        
        logger.info(f"VideoAssembler initialized (use_ffmpeg={use_ffmpeg})")
    
    def _check_ffmpeg(self):
        """Check if FFmpeg is installed"""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            logger.info("FFmpeg is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("FFmpeg not found. Install with: sudo apt-get install ffmpeg")
    
    def process_clip(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        target_duration: Optional[float] = None
    ) -> bool:
        """
        Trim and optionally adjust speed of a video clip.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            start_time: Start time in seconds
            end_time: End time in seconds
            target_duration: Desired duration in seconds (if different from end-start)
        
        Returns:
            True if successful, False otherwise
        """
        source_duration = end_time - start_time
        
        # Safety: ensure we have valid durations
        if source_duration <= 0:
            logger.warning(f"Clip has zero/negative source duration ({start_time:.2f}s to {end_time:.2f}s). Using full file.")
            start_time = 0.0
            end_time = 1.0  # Use first 1 second as fallback
            source_duration = 1.0
            
        if target_duration is None or target_duration <= 0:
            target_duration = source_duration
            
        speed_factor = source_duration / target_duration
        
        logger.info(f"Processing {input_path}: trim {start_time:.2f}s-{end_time:.2f}s, target {target_duration:.2f}s (speed x{1/speed_factor:.2f})")
        
        try:
            if self.use_ffmpeg:
                return self._process_clip_ffmpeg(input_path, output_path, start_time, end_time, target_duration)
            else:
                # Fallback to simple trim if speed control not implemented in moviepy path
                return self._trim_video_moviepy(input_path, output_path, start_time, end_time)
        except Exception as e:
            logger.error(f"Error processing clip: {e}")
            return False

    def _process_clip_ffmpeg(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        target_duration: float
    ) -> bool:
        """Trim and adjust speed using FFmpeg setpts filter"""
        source_duration = end_time - start_time
        
        # Safety: prevent division by zero
        if source_duration <= 0:
            source_duration = 0.1
        if target_duration <= 0:
            target_duration = source_duration
            
        # pts_factor = target_duration / source_duration
        # To speed up (target < source), pts_factor < 1
        # To slow down (target > source), pts_factor > 1
        pts_factor = target_duration / source_duration
        
        # Limit speed factor to avoid extreme distortion (0.5x to 2.0x)
        # pts_factor = 0.5 means 2x speed
        # pts_factor = 2.0 means 0.5x speed
        pts_factor = max(0.5, min(2.0, pts_factor))
        
        filter_complex = f"setpts={pts_factor}*PTS"
        
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),
            '-t', str(source_duration),
            '-i', input_path,
            '-vf', filter_complex,
            '-c:v', 'libx264',
            '-an', # Remove audio from individual clips to avoid sync issues
            '-y',
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            return False

    def _trim_video_moviepy(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float
    ) -> bool:
        """Trim video using MoviePy"""
        try:
            clip = VideoFileClip(input_path)
            trimmed = clip.subclip(start_time, end_time)
            trimmed.write_videofile(output_path, verbose=False, logger=None)
            clip.close()
            return True
        except Exception as e:
            logger.error(f"MoviePy error: {e}")
            return False
    
    def concatenate_videos(
        self,
        video_paths: List[str],
        output_path: str
    ) -> bool:
        """Concatenate multiple video clips."""
        logger.info(f"Concatenating {len(video_paths)} videos")
        
        # Create a concat demuxer file
        concat_file = Path(output_path).parent / 'concat_list.txt'
        with open(concat_file, 'w') as f:
            for video_path in video_paths:
                abs_path = str(Path(video_path).absolute())
                f.write(f"file '{abs_path}'\n")
        
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',
            '-y',
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=600)
            concat_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Concatenation error: {e}")
            return False
    
    def add_audio(
        self,
        video_path: str,
        audio_path: str,
        output_path: str
    ) -> bool:
        """Add audio track to a video."""
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            '-y',
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
            return True
        except Exception as e:
            logger.error(f"Audio addition error: {e}")
            return False

class VideoSequenceBuilder:
    """
    Handles the high-level workflow of building a video sequence from clip selections.
    """
    
    def __init__(self, assembler: VideoAssembler, temp_dir: str = './temp'):
        self.assembler = assembler
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def build_sequence(
        self,
        clip_selections: List,
        audio_path: str,
        output_path: str,
        use_speed_control: bool = True
    ) -> bool:
        """
        Build the final video sequence.
        
        Args:
            clip_selections: List of ClipSelection objects
            audio_path: Path to the voiceover audio file
            output_path: Path to save the final video
            use_speed_control: Whether to adjust clip speed to match segment duration
            
        Returns:
            True if successful, False otherwise
        """
        processed_clips = []
        
        try:
            # 1. Process each clip (trim + speed)
            for i, selection in enumerate(clip_selections):
                temp_clip_path = self.temp_dir / f"clip_{i:03d}.mp4"
                
                target_duration = selection.duration if use_speed_control else None
                
                success = self.assembler.process_clip(
                    selection.video_file_path,
                    str(temp_clip_path),
                    selection.trim_start,
                    selection.trim_end,
                    target_duration=target_duration
                )
                
                if not success:
                    logger.error(f"Failed to process clip {i}")
                    return False
                    
                processed_clips.append(str(temp_clip_path))
                
            # 2. Concatenate processed clips
            concat_video_path = self.temp_dir / "concatenated_video.mp4"
            success = self.assembler.concatenate_videos(processed_clips, str(concat_video_path))
            if not success:
                logger.error("Failed to concatenate videos")
                return False
                
            # 3. Add audio
            success = self.assembler.add_audio(str(concat_video_path), audio_path, output_path)
            if not success:
                logger.error("Failed to add audio")
                return False
                
            # 4. Cleanup temp files
            for clip_path in processed_clips:
                os.remove(clip_path)
            if concat_video_path.exists():
                os.remove(concat_video_path)
                
            return True
            
        except Exception as e:
            logger.error(f"Error in build_sequence: {e}")
            return False
