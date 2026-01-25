#!/usr/bin/env python3
"""
Video Clip Selection and Sequencing via Language and Vision Models

Main CLI application entry point.

This application automatically generates edited videos by:
1. Indexing B-roll video footage using VideoPrism
2. Transcribing voice-over audio using Whisper
3. Segmenting the script using an LLM
4. Matching script segments to video clips
5. Assembling and exporting the final video
"""

import argparse
import logging
import json
import sys
import shutil
from pathlib import Path
from typing import Optional

# Import modules
from indexing import VideoIndexer
from transcription import VoiceTranscriber, TranscriptionAnalyzer
from segmentation import ScriptSegmenter
from matching import VideoTextMatcher, create_sequence
from assembly import VideoAssembler, VideoSequenceBuilder


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_ffmpeg():
    """Check if FFmpeg is available"""
    if shutil.which('ffmpeg') is None:
        return False
    return True


class VideoSequencingPipeline:
    """Main pipeline orchestrating the entire video sequencing process"""
    
    def __init__(
        self,
        video_dir: str,
        audio_file: str,
        output_dir: str,
        cache_dir: str = './cache',
        gpu_device: str = 'cuda:0'
    ):
        """
        Initialize the pipeline.
        
        Args:
            video_dir: Directory containing B-roll videos
            audio_file: Path to voice-over audio file
            output_dir: Directory for output files
            cache_dir: Directory for caching intermediate results
            gpu_device: GPU device to use (e.g., 'cuda:0', 'cuda:1', or 'cpu')
        """
        self.video_dir = Path(video_dir)
        self.audio_file = Path(audio_file)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.gpu_device = gpu_device
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.indexer = None
        self.transcriber = None
        self.segmenter = None
        self.matcher = None
        self.assembler = None
        
        logger.info("VideoSequencingPipeline initialized")
    
    def run(
        self,
        llm_model: str = 'meta-llama/Llama-3.2-3B-Instruct',
        whisper_model: str = 'base',
        videoprism_model: str = 'videoprism_public_v1_base',
        use_simple_segmentation: bool = False
    ) -> Optional[Path]:
        """
        Run the complete pipeline.
        
        Args:
            llm_model: Local LLM model name (default: Llama-3.2-3B-Instruct)
            whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            videoprism_model: VideoPrism model to use
            use_simple_segmentation: Use simple rule-based segmentation instead of LLM
        
        Returns:
            Path to output video or None if failed
        """
        try:
            logger.info("=" * 80)
            logger.info("Starting Video Sequencing Pipeline")
            logger.info("=" * 80)
            
            # Check FFmpeg availability
            if not check_ffmpeg():
                logger.error("FFmpeg is not installed or not in PATH.")
                logger.error("Please install FFmpeg:")
                logger.error("  Ubuntu/Debian: sudo apt-get install ffmpeg")
                logger.error("  macOS: brew install ffmpeg")
                logger.error("  Or download from: https://ffmpeg.org/download.html")
                return None
            
            # Step 1: Index videos
            logger.info("\n[STEP 1] Indexing B-roll videos...")
            if not self._index_videos(videoprism_model):
                return None
            
            # Step 2: Transcribe audio
            logger.info("\n[STEP 2] Transcribing voice-over audio...")
            transcription = self._transcribe_audio(whisper_model)
            if transcription is None:
                return None
            
            # Step 3: Segment script
            logger.info("\n[STEP 3] Segmenting script into semantic chunks...")
            segments = self._segment_script(transcription, llm_model, use_simple_segmentation)
            if segments is None:
                return None
            
            # Step 4: Match segments to videos
            logger.info("\n[STEP 4] Matching script segments to video clips...")
            clip_selections = self._match_and_sequence(segments)
            if clip_selections is None:
                return None
            
            # Step 5: Assemble and export
            logger.info("\n[STEP 5] Assembling and exporting final video...")
            output_video = self._assemble_video(clip_selections, transcription)
            
            if output_video:
                logger.info("=" * 80)
                logger.info(f"SUCCESS! Output video: {output_video}")
                logger.info("=" * 80)
            
            return output_video
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return None
    
    def _index_videos(self, model_name: str) -> bool:
        """Index all videos in the video directory"""
        try:
            cache_file = self.cache_dir / 'video_index'
            
            self.indexer = VideoIndexer(
                model_name=model_name,
                index_dir=str(cache_file),
                device=self.gpu_device
            )
            
            # Try to load existing index
            if self.indexer.load_index():
                logger.info(f"Loaded existing video index with {len(self.indexer.metadata_list)} videos")
                
                # Check if there are new videos to index
                video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
                video_files = [
                    f for f in self.video_dir.rglob('*')
                    if f.suffix.lower() in video_extensions
                ]
                
                existing_ids = set(self.indexer.video_id_to_idx.keys())
                new_videos = [f for f in video_files if f.stem not in existing_ids]
                
                if new_videos:
                    logger.info(f"Found {len(new_videos)} new videos to index")
                    num_indexed = self.indexer.index_videos(str(self.video_dir))
                    logger.info(f"Indexed {num_indexed} new videos")
                
                return True
            
            # Create new index
            num_indexed = self.indexer.index_videos(str(self.video_dir))
            
            if num_indexed == 0:
                logger.error("No videos were indexed")
                return False
            
            logger.info(f"Successfully indexed {num_indexed} videos")
            return True
        
        except Exception as e:
            logger.error(f"Error indexing videos: {e}")
            return False
    
    def _transcribe_audio(self, model_size: str):
        """Transcribe voice-over audio"""
        try:
            cache_file = self.cache_dir / 'transcription.json'
            
            # Check if transcription already exists
            if cache_file.exists():
                logger.info("Loading cached transcription")
                self.transcriber = VoiceTranscriber(model_size=model_size)
                return self.transcriber.load_transcription(str(cache_file))
            
            # Transcribe audio
            self.transcriber = VoiceTranscriber(
                model_size=model_size,
                device=self.gpu_device
            )
            transcription = self.transcriber.transcribe(str(self.audio_file))
            
            # Save transcription
            self.transcriber.save_transcription(transcription, str(cache_file))
            
            logger.info(f"Transcription complete:")
            logger.info(f"  Language: {transcription.language}")
            logger.info(f"  Duration: {transcription.duration:.2f}s")
            logger.info(f"  Words: {len(transcription.words)}")
            
            return transcription
        
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None
    
    def _segment_script(self, transcription, llm_model: str, use_simple_segmentation: bool = False):
        """Segment script into semantic chunks"""
        try:
            cache_file = self.cache_dir / 'segments.json'
            
            # Check if segments already exist
            if cache_file.exists():
                logger.info("Loading cached segments")
                return ScriptSegmenter.load_segments(str(cache_file))
            
            # Segment script
            self.segmenter = ScriptSegmenter(
                model_name=llm_model,
                device=self.gpu_device,
                use_simple_segmentation=use_simple_segmentation
            )
            
            words_with_timing = [
                {
                    'word': w.word,
                    'start_time': w.start_time,
                    'end_time': w.end_time
                }
                for w in transcription.words
            ]
            
            segments = self.segmenter.segment_script(
                transcription.full_text,
                words_with_timing
            )
            
            # Save segments
            self.segmenter.save_segments(segments, str(cache_file))
            
            logger.info(f"Script segmentation complete:")
            logger.info(f"  Segments: {len(segments)}")
            for seg in segments:
                desc = seg.description[:50] if seg.description else "No description"
                logger.info(f"    [{seg.segment_id}] {desc}... ({seg.duration:.2f}s)")
            
            return segments
        
        except Exception as e:
            logger.error(f"Error segmenting script: {e}")
            return None
    
    def _match_and_sequence(
        self,
        segments
    ):
        """Match script segments to video clips"""
        try:
            cache_file = self.cache_dir / 'clip_selections.json'
            
            # Initialize matcher
            self.matcher = VideoTextMatcher(self.indexer)
            
            # Convert segments to dict format expected by create_sequence
            segment_dicts = [
                {
                    'text': seg.text,
                    'duration': seg.duration,
                    'start_time': seg.start_time,
                    'end_time': seg.end_time
                }
                for seg in segments
            ]
            
            # Create sequence
            clip_selections = create_sequence(
                segment_dicts,
                self.matcher
            )
            
            if not clip_selections:
                logger.error("No clips were selected")
                return None
            
            # Save selections
            selections_data = {
                'clips': [
                    {
                        'segment_id': c.segment_id,
                        'video_id': c.video_id,
                        'video_file_path': c.video_file_path,
                        'trim_start': c.trim_start,
                        'trim_end': c.trim_end,
                        'similarity_score': c.similarity_score,
                        'is_reused': c.is_reused
                    }
                    for c in clip_selections
                ]
            }
            
            with open(cache_file, 'w') as f:
                json.dump(selections_data, f, indent=2)
            
            logger.info(f"Clip selection complete:")
            logger.info(f"  Selected clips: {len(clip_selections)}")
            reused_count = sum(1 for c in clip_selections if c.is_reused)
            logger.info(f"  Reused clips: {reused_count}")
            
            return clip_selections
        
        except Exception as e:
            logger.error(f"Error matching and sequencing: {e}")
            return None
    
    def _assemble_video(self, clip_selections, transcription) -> Optional[Path]:
        """Assemble and export final video"""
        try:
            self.assembler = VideoAssembler(use_ffmpeg=True)
            builder = VideoSequenceBuilder(self.assembler, temp_dir=str(self.cache_dir / 'temp'))
            
            output_path = self.output_dir / 'output_video.mp4'
            
            success = builder.build_sequence(
                clip_selections,
                str(self.audio_file),
                str(output_path)
            )
            
            if success:
                logger.info(f"Video assembly complete: {output_path}")
                return output_path
            else:
                logger.error("Video assembly failed")
                return None
        
        except Exception as e:
            logger.error(f"Error assembling video: {e}")
            return None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Video Clip Selection and Sequencing via Language and Vision Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py --video-dir ./videos --audio ./voiceover.mp3 --output ./output
  
  # With custom models
  python main.py --video-dir ./videos --audio ./voiceover.mp3 --output ./output \\
    --whisper-model medium --llm-model meta-llama/Llama-3.2-3B-Instruct
  
  # Use simple segmentation (no LLM required)
  python main.py --video-dir ./videos --audio ./voiceover.mp3 --output ./output \\
    --simple-segmentation
  
        """
    )
    
    parser.add_argument(
        '--video-dir',
        required=True,
        help='Directory containing B-roll video clips'
    )
    parser.add_argument(
        '--audio',
        required=True,
        help='Path to voice-over audio file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for the final video'
    )
    parser.add_argument(
        '--cache-dir',
        default='./cache',
        help='Directory for caching intermediate results (default: ./cache)'
    )
    parser.add_argument(
        '--whisper-model',
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size (default: base)'
    )
    parser.add_argument(
        '--videoprism-model',
        default='videoprism_public_v1_base',
        choices=['videoprism_public_v1_base', 'videoprism_public_v1_large'],
        help='VideoPrism model to use (default: videoprism_public_v1_base)'
    )
    parser.add_argument(
        '--llm-model',
        default='meta-llama/Llama-3.2-3B-Instruct',
        help='Local LLM model for script segmentation (default: meta-llama/Llama-3.2-3B-Instruct)'
    )
    parser.add_argument(
        '--simple-segmentation',
        action='store_true',
        help='Use simple rule-based segmentation instead of LLM (faster, no model download required)'
    )
    parser.add_argument(
        '--gpu-device',
        default='cuda:0',
        help='GPU device to use (e.g., cuda:0, cuda:1, cuda:2, cuda:3, or cpu) (default: cuda:0)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate inputs
    if not Path(args.video_dir).exists():
        logger.error(f"Video directory not found: {args.video_dir}")
        sys.exit(1)
    
    if not Path(args.audio).exists():
        logger.error(f"Audio file not found: {args.audio}")
        sys.exit(1)
    
    # Run pipeline
    pipeline = VideoSequencingPipeline(
        video_dir=args.video_dir,
        audio_file=args.audio,
        output_dir=args.output,
        cache_dir=args.cache_dir,
        gpu_device=args.gpu_device
    )
    
    output_video = pipeline.run(
        llm_model=args.llm_model,
        whisper_model=args.whisper_model,
        videoprism_model=args.videoprism_model,
        use_simple_segmentation=args.simple_segmentation
    )
    
    if output_video:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
