"""
Voice-Over Transcription Module using OpenAI Whisper

This module is responsible for:
1. Loading audio files (voice-over)
2. Transcribing audio to text using Whisper
3. Extracting word-level timestamps
4. Providing structured output with timing information
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np

try:
    import whisper
except ImportError:
    whisper = None

try:
    import librosa
except ImportError:
    librosa = None


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class WordSegment:
    """Represents a single word with timing information"""
    word: str
    start_time: float
    end_time: float
    confidence: Optional[float] = None


@dataclass
class TranscriptionResult:
    """Complete transcription result"""
    full_text: str
    language: str
    duration: float
    segments: List[Dict]  # Segment-level information
    words: List[WordSegment]  # Word-level information


class VoiceTranscriber:
    """
    Whisper-based voice-over transcription system.
    
    This class handles loading audio files, transcribing them using Whisper,
    and extracting word-level timing information.
    """
    
    def __init__(
        self,
        model_size: str = 'base',
        device: str = 'cuda',
        language: Optional[str] = None
    ):
        """
        Initialize the VoiceTranscriber.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to use ('cuda' or 'cpu')
            language: Language code (e.g., 'en' for English). If None, auto-detect.
        """
        self.model_size = model_size
        self.device = device
        self.language = language
        
        # Load Whisper model
        logger.info(f"Loading Whisper model: {model_size}")
        if whisper is None:
            raise ImportError("Whisper not installed. Run: pip install openai-whisper")
        
        self.model = whisper.load_model(model_size, device=device)
        logger.info("Whisper model loaded successfully")
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file using librosa.
        
        Args:
            audio_path: Path to the audio file
        
        Returns:
            Tuple of (audio array, sample rate)
        """
        if librosa is None:
            raise ImportError("librosa not installed. Run: pip install librosa")
        
        audio, sr = librosa.load(audio_path, sr=16000)
        logger.info(f"Loaded audio: {len(audio)} samples at {sr} Hz")
        return audio, sr
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get the duration of an audio file in seconds.
        
        Args:
            audio_path: Path to the audio file
        
        Returns:
            Duration in seconds
        """
        if librosa is None:
            raise ImportError("librosa not installed. Run: pip install librosa")
        
        audio, sr = librosa.load(audio_path, sr=16000)
        duration = len(audio) / sr
        return duration
    
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribe audio file to text with timing information.
        
        Args:
            audio_path: Path to the audio file
        
        Returns:
            TranscriptionResult object with full text and timing info
        """
        logger.info(f"Transcribing audio: {audio_path}")
        
        # Get audio duration
        duration = self.get_audio_duration(audio_path)
        
        # Transcribe with Whisper
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            verbose=False,
            word_level_timestamps=True
        )
        
        full_text = result['text']
        language = result.get('language', 'unknown')
        
        logger.info(f"Transcription complete. Language: {language}")
        logger.info(f"Full text: {full_text[:100]}...")
        
        # Extract word-level timing information
        words = self._extract_word_timings(result)
        
        return TranscriptionResult(
            full_text=full_text,
            language=language,
            duration=duration,
            segments=result.get('segments', []),
            words=words
        )
    
    def _extract_word_timings(self, whisper_result: Dict) -> List[WordSegment]:
        """
        Extract word-level timing from Whisper result.
        
        Args:
            whisper_result: Raw result from Whisper transcribe
        
        Returns:
            List of WordSegment objects
        """
        words = []
        
        for segment in whisper_result.get('segments', []):
            segment_start = segment['start']
            segment_end = segment['end']
            segment_text = segment['text'].strip()
            
            # Split segment into words
            word_list = segment_text.split()
            
            if not word_list:
                continue
            
            # Estimate word timings by dividing segment duration equally
            segment_duration = segment_end - segment_start
            word_duration = segment_duration / len(word_list)
            
            for i, word in enumerate(word_list):
                start_time = segment_start + (i * word_duration)
                end_time = segment_start + ((i + 1) * word_duration)
                
                words.append(WordSegment(
                    word=word,
                    start_time=start_time,
                    end_time=end_time
                ))
        
        return words
    
    def get_text_for_time_range(
        self,
        words: List[WordSegment],
        start_time: float,
        end_time: float
    ) -> str:
        """
        Get text for a specific time range.
        
        Args:
            words: List of WordSegment objects
            start_time: Start time in seconds
            end_time: End time in seconds
        
        Returns:
            Text corresponding to the time range
        """
        matching_words = [
            w.word for w in words
            if w.start_time >= start_time and w.end_time <= end_time
        ]
        return ' '.join(matching_words)
    
    def save_transcription(
        self,
        result: TranscriptionResult,
        output_path: str
    ):
        """
        Save transcription result to a JSON file.
        
        Args:
            result: TranscriptionResult object
            output_path: Path to save the JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        data = {
            'full_text': result.full_text,
            'language': result.language,
            'duration': result.duration,
            'segments': result.segments,
            'words': [asdict(w) for w in result.words]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved transcription to {output_path}")
    
    def load_transcription(self, json_path: str) -> TranscriptionResult:
        """
        Load transcription result from a JSON file.
        
        Args:
            json_path: Path to the JSON file
        
        Returns:
            TranscriptionResult object
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        words = [WordSegment(**w) for w in data['words']]
        
        return TranscriptionResult(
            full_text=data['full_text'],
            language=data['language'],
            duration=data['duration'],
            segments=data['segments'],
            words=words
        )


class TranscriptionAnalyzer:
    """
    Utility class for analyzing transcription results.
    """
    
    @staticmethod
    def get_segment_duration(start_time: float, end_time: float) -> float:
        """Calculate duration of a time segment"""
        return end_time - start_time
    
    @staticmethod
    def get_words_in_range(
        words: List[WordSegment],
        start_time: float,
        end_time: float
    ) -> List[WordSegment]:
        """Get all words within a time range"""
        return [
            w for w in words
            if w.start_time >= start_time and w.end_time <= end_time
        ]
    
    @staticmethod
    def estimate_speaking_rate(
        words: List[WordSegment],
        start_time: float,
        end_time: float
    ) -> float:
        """
        Estimate speaking rate (words per second) for a time range.
        
        Args:
            words: List of WordSegment objects
            start_time: Start time in seconds
            end_time: End time in seconds
        
        Returns:
            Words per second
        """
        matching_words = TranscriptionAnalyzer.get_words_in_range(
            words, start_time, end_time
        )
        
        if not matching_words:
            return 0.0
        
        duration = end_time - start_time
        if duration == 0:
            return 0.0
        
        return len(matching_words) / duration


if __name__ == '__main__':
    # Example usage
    transcriber = VoiceTranscriber(model_size='base')
    result = transcriber.transcribe('./data/input/audio/voiceover.mp3')
    
    print(f"Language: {result.language}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Full text: {result.full_text}")
    print(f"Word count: {len(result.words)}")
    
    # Save transcription
    transcriber.save_transcription(result, './data/output/transcription.json')
