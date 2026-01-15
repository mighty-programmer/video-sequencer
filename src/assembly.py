"""
Video Assembly and Export Module

This module is responsible for:
1. Trimming video clips to specified durations
2. Concatenating clips in sequence
3. Adding audio (voice-over) to the final video
4. Exporting to various formats
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
    Assembles video clips into a final sequence.
    
    This class handles trimming, concatenating, and exporting video clips.
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
    
    def trim_video(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float
    ) -> bool:
        """
        Trim a video clip to a specified time range.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            start_time: Start time in seconds
            end_time: End time in seconds
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Trimming {input_path} from {start_time:.2f}s to {end_time:.2f}s")
        
        try:
            if self.use_ffmpeg:
                return self._trim_video_ffmpeg(input_path, output_path, start_time, end_time)
            else:
                return self._trim_video_moviepy(input_path, output_path, start_time, end_time)
        except Exception as e:
            logger.error(f"Error trimming video: {e}")
            return False
    
    def _trim_video_ffmpeg(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float
    ) -> bool:
        """Trim video using FFmpeg"""
        duration = end_time - start_time
        
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-y',  # Overwrite output file
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
            logger.info(f"Successfully trimmed to {output_path}")
            return True
        except subprocess.TimeoutExpired:
            logger.error("Trimming operation timed out")
            return False
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
        \"\"\"Trim video using MoviePy\"\"\"
        try:
            clip = VideoFileClip(input_path)
            trimmed = clip.subclip(start_time, end_time)
            trimmed.write_videofile(output_path, verbose=False, logger=None)
            clip.close()
            logger.info(f\"Successfully trimmed to {output_path}\")
            return True
        except Exception as e:
            logger.error(f\"MoviePy error: {e}\")
            return False
    
    def concatenate_videos(
        self,
        video_paths: List[str],
        output_path: str
    ) -> bool:
        \"\"\"
        Concatenate multiple video clips.
        
        Args:
            video_paths: List of input video paths
            output_path: Path to output video
        
        Returns:
            True if successful, False otherwise
        \"\"\"
        logger.info(f\"Concatenating {len(video_paths)} videos\")
        
        try:
            if self.use_ffmpeg:
                return self._concatenate_videos_ffmpeg(video_paths, output_path)
            else:
                return self._concatenate_videos_moviepy(video_paths, output_path)
        except Exception as e:
            logger.error(f\"Error concatenating videos: {e}\")
            return False
    
    def _concatenate_videos_ffmpeg(
        self,
        video_paths: List[str],
        output_path: str
    ) -> bool:
        \"\"\"Concatenate videos using FFmpeg\"\"\"
        # Create a concat demuxer file
        concat_file = Path(output_path).parent / 'concat_list.txt'
        
        with open(concat_file, 'w') as f:
            for video_path in video_paths:
                f.write(f\"file '{Path(video_path).absolute()}'\\n\")
        
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
            logger.info(f\"Successfully concatenated to {output_path}\")
            concat_file.unlink()  # Remove concat file
            return True
        except subprocess.TimeoutExpired:
            logger.error(\"Concatenation operation timed out\")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f\"FFmpeg error: {e.stderr.decode()}\")
            return False
    
    def _concatenate_videos_moviepy(
        self,
        video_paths: List[str],
        output_path: str
    ) -> bool:
        \"\"\"Concatenate videos using MoviePy\"\"\"
        try:
            clips = [VideoFileClip(path) for path in video_paths]
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(output_path, verbose=False, logger=None)
            
            for clip in clips:
                clip.close()
            
            logger.info(f\"Successfully concatenated to {output_path}\")
            return True
        except Exception as e:
            logger.error(f\"MoviePy error: {e}\")
            return False
    
    def add_audio(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        audio_start: float = 0.0
    ) -> bool:
        \"\"\"
        Add audio track to a video.
        
        Args:
            video_path: Path to input video
            audio_path: Path to audio file
            output_path: Path to output video
            audio_start: Start time of audio in seconds
        
        Returns:
            True if successful, False otherwise
        \"\"\"
        logger.info(f\"Adding audio from {audio_path} to {video_path}\")
        
        try:
            if self.use_ffmpeg:
                return self._add_audio_ffmpeg(video_path, audio_path, output_path, audio_start)
            else:
                return self._add_audio_moviepy(video_path, audio_path, output_path, audio_start)
        except Exception as e:
            logger.error(f\"Error adding audio: {e}\")
            return False
    
    def _add_audio_ffmpeg(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        audio_start: float = 0.0
    ) -> bool:
        \"\"\"Add audio using FFmpeg\"\"\"
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
            logger.info(f\"Successfully added audio to {output_path}\")
            return True
        except subprocess.TimeoutExpired:
            logger.error(\"Audio addition operation timed out\")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f\"FFmpeg error: {e.stderr.decode()}\")
            return False
    
    def _add_audio_moviepy(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        audio_start: float = 0.0
    ) -> bool:
        \"\"\"Add audio using MoviePy\"\"\"
        try:
            video = VideoFileClip(video_path)
            audio = AudioFileClip(audio_path)
            
            # Set audio start time
            if audio_start > 0:
                audio = audio.set_start(audio_start)
            
            final_video = video.set_audio(audio)
            final_video.write_videofile(output_path, verbose=False, logger=None)
            
            video.close()
            audio.close()
            
            logger.info(f\"Successfully added audio to {output_path}\")
            return True
        except Exception as e:
            logger.error(f\"MoviePy error: {e}\")
            return False
    
    def get_video_info(self, video_path: str) -> Optional[Dict]:
        \"\"\"
        Get information about a video file.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with video info or None
        \"\"\"
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,duration',
                '-of', 'json',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, check=True)
            data = json.loads(result.stdout)
            
            if data['streams']:
                stream = data['streams'][0]
                return {
                    'width': stream.get('width'),
                    'height': stream.get('height'),
                    'fps': eval(stream.get('r_frame_rate', '30/1')),
                    'duration': float(stream.get('duration', 0))
                }
        except Exception as e:
            logger.error(f\"Error getting video info: {e}\")
        
        return None


class VideoSequenceBuilder:
    \"\"\"
    Builds the final video sequence from clip selections.
    \"\"\"
    
    def __init__(self, assembler: VideoAssembler, temp_dir: str = './temp'):
        \"\"\"
        Initialize the VideoSequenceBuilder.
        
        Args:
            assembler: VideoAssembler instance
            temp_dir: Directory for temporary files
        \"\"\"
        self.assembler = assembler
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def build_sequence(
        self,
        clip_selections: List,
        audio_path: str,
        output_path: str
    ) -> bool:
        \"\"\"
        Build the final video sequence.
        
        Args:
            clip_selections: List of ClipSelection objects
            audio_path: Path to voice-over audio
            output_path: Path to output video
        
        Returns:
            True if successful, False otherwise
        \"\"\"
        logger.info(f\"Building video sequence with {len(clip_selections)} clips\")
        
        try:
            # Step 1: Trim all clips
            trimmed_clips = []
            for i, clip_sel in enumerate(clip_selections):
                trimmed_path = self.temp_dir / f\"clip_{i:03d}.mp4\"
                
                success = self.assembler.trim_video(
                    clip_sel.video_file_path,
                    str(trimmed_path),
                    clip_sel.trim_start,
                    clip_sel.trim_end
                )
                
                if success:
                    trimmed_clips.append(str(trimmed_path))
                else:
                    logger.error(f\"Failed to trim clip {i}\")
                    return False
            
            # Step 2: Concatenate clips
            concatenated_path = self.temp_dir / 'concatenated.mp4'
            success = self.assembler.concatenate_videos(trimmed_clips, str(concatenated_path))
            
            if not success:
                logger.error(\"Failed to concatenate clips\")
                return False
            
            # Step 3: Add audio
            success = self.assembler.add_audio(
                str(concatenated_path),
                audio_path,
                output_path
            )
            
            if success:
                logger.info(f\"Successfully built video sequence: {output_path}\")
                self._cleanup_temp_files()
                return True
            else:
                logger.error(\"Failed to add audio\")
                return False
        
        except Exception as e:
            logger.error(f\"Error building sequence: {e}\")
            return False
    
    def _cleanup_temp_files(self):
        \"\"\"Clean up temporary files\"\"\"
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info(\"Cleaned up temporary files\")
        except Exception as e:
            logger.warning(f\"Could not clean up temp files: {e}\")


if __name__ == '__main__':
    # Example usage
    assembler = VideoAssembler(use_ffmpeg=True)
    
    # Get video info
    info = assembler.get_video_info('./data/input/videos/sample.mp4')
    print(f\"Video info: {info}\")
