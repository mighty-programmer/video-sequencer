"""
Script Segmentation Module using Large Language Models

This module is responsible for:
1. Taking a transcribed script with timing information
2. Using a local LLM to split the script into semantic segments
3. Each segment represents one conceptual action or scene
4. Providing timing information for each segment
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class ScriptSegment:
    """Represents a semantic segment of the script"""
    segment_id: int
    text: str
    start_time: float
    end_time: float
    duration: float
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    action_type: Optional[str] = None


class ScriptSegmenter:
    """
    Local LLM-based script segmentation system.
    
    This class takes a transcribed script and uses a local LLM to identify
    semantic boundaries and create meaningful segments for video matching.
    
    Default model: Llama-3.2-3B-Instruct (efficient and accurate for text tasks)
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: str = "cuda:0"
    ):
        """
        Initialize the ScriptSegmenter with a local LLM.
        
        Args:
            model_name: Hugging Face model name (default: Llama-3.2-3B-Instruct)
            device: Device to run on ('cuda:0', 'cuda:1', etc., or 'cpu')
        """
        if torch is None or AutoTokenizer is None:
            raise ImportError(
                "Transformers and PyTorch not installed. "
                "Run: pip install transformers torch"
            )
        
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading local LLM: {self.model_name}")
        self._init_model()
        logger.info(f"ScriptSegmenter initialized with {self.model_name}")
    
    def _init_model(self):
        """Initialize the local Hugging Face model"""
        logger.info(f"Loading tokenizer and model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use specific GPU device instead of device_map="auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
        # Move model to specified device
        if torch.cuda.is_available() and 'cuda' in self.device:
            self.model = self.model.to(self.device)
        elif self.device == 'cpu':
            self.model = self.model.to('cpu')
        
        logger.info(f"Model loaded successfully on device: {self.model.device}")
    
    def segment_script(
        self,
        full_text: str,
        words_with_timing: List[Dict],
        max_segment_words: int = 50
    ) -> List[ScriptSegment]:
        """
        Segment a script into semantic chunks.
        
        Args:
            full_text: Complete transcribed text
            words_with_timing: List of dicts with 'word', 'start_time', 'end_time'
            max_segment_words: Maximum words per segment (for chunking)
        
        Returns:
            List of ScriptSegment objects
        """
        logger.info("Starting script segmentation")
        
        # First, chunk the text into reasonable segments
        text_chunks = self._chunk_text(full_text, words_with_timing, max_segment_words)
        
        logger.info(f"Created {len(text_chunks)} initial chunks")
        
        # Use LLM to identify semantic boundaries
        semantic_segments = self._identify_semantic_boundaries(text_chunks)
        
        # Convert to ScriptSegment objects with timing
        segments = self._create_segments_with_timing(
            semantic_segments,
            words_with_timing
        )
        
        logger.info(f"Created {len(segments)} semantic segments")
        return segments
    
    def _chunk_text(
        self,
        full_text: str,
        words_with_timing: List[Dict],
        max_words: int
    ) -> List[Dict]:
        """
        Chunk text into manageable pieces based on word count.
        
        Args:
            full_text: Complete text
            words_with_timing: List of word timing dicts
            max_words: Maximum words per chunk
        
        Returns:
            List of chunk dicts with text, start_time, end_time
        """
        chunks = []
        current_chunk_words = []
        current_chunk_start_idx = 0
        
        for i, word_info in enumerate(words_with_timing):
            current_chunk_words.append(word_info)
            
            if len(current_chunk_words) >= max_words or i == len(words_with_timing) - 1:
                # Create chunk
                chunk_text = ' '.join([w['word'] for w in current_chunk_words])
                chunk_start = current_chunk_words[0]['start_time']
                chunk_end = current_chunk_words[-1]['end_time']
                
                chunks.append({
                    'text': chunk_text,
                    'start_time': chunk_start,
                    'end_time': chunk_end,
                    'word_indices': (current_chunk_start_idx, i + 1)
                })
                
                current_chunk_words = []
                current_chunk_start_idx = i + 1
        
        return chunks
    
    def _identify_semantic_boundaries(
        self,
        text_chunks: List[Dict]
    ) -> List[Dict]:
        """
        Use local LLM to identify semantic boundaries and segment descriptions.
        
        Args:
            text_chunks: List of text chunks
        
        Returns:
            List of semantic segments with descriptions
        """
        # Combine chunks into a single text for LLM processing
        combined_text = '\n'.join([f"[{i}] {c['text']}" for i, c in enumerate(text_chunks)])
        
        prompt = self._create_segmentation_prompt(combined_text)
        
        response = self._query_model(prompt)
        
        # Parse LLM response to extract segments
        segments = self._parse_segmentation_response(response, text_chunks)
        
        return segments
    
    def _create_segmentation_prompt(self, text: str) -> str:
        """Create a prompt for the LLM to segment the script"""
        prompt = f"""You are an expert video editor and scriptwriter. Your task is to segment the following script into meaningful semantic chunks. Each chunk should represent a single conceptual action, scene, or idea.

For each segment, provide:
1. The chunk indices it covers (e.g., [0, 1, 2])
2. A brief description of what happens in this segment
3. Key action words or verbs
4. The type of action (e.g., "demonstration", "explanation", "transition")

Script:
{text}

Please analyze this script and identify the semantic segments. Format your response as JSON with the following structure:
{{
  "segments": [
    {{
      "chunk_indices": [0, 1, 2],
      "description": "Brief description of this segment",
      "keywords": ["keyword1", "keyword2"],
      "action_type": "demonstration"
    }},
    ...
  ]
}}

Ensure that:
- Each segment is coherent and represents a complete thought or action
- Segments flow naturally from one to the next
- Action types are descriptive (e.g., "demonstration", "explanation", "transition", "instruction")
- Keywords capture the main visual elements or actions

Respond only with valid JSON."""
        
        return prompt
    
    def _query_model(self, prompt: str) -> str:
        """Query the local Hugging Face model"""
        logger.info(f"Querying local LLM: {self.model_name}")
        
        # Format prompt for chat models
        messages = [
            {"role": "system", "content": "You are a helpful assistant for video editing and script analysis."},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = f"System: You are a helpful assistant for video editing and script analysis.\n\nUser: {prompt}\n\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1500,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        elif formatted_prompt in response:
            response = response.replace(formatted_prompt, "").strip()
        
        return response
    
    def _parse_segmentation_response(
        self,
        response: str,
        text_chunks: List[Dict]
    ) -> List[Dict]:
        """
        Parse LLM response to extract segment information.
        
        Args:
            response: LLM response text
            text_chunks: Original text chunks for reference
        
        Returns:
            List of segment dicts
        """
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("No JSON found in LLM response, using fallback segmentation")
                return self._fallback_segmentation(text_chunks)
            
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)
            
            segments = []
            for seg in parsed.get('segments', []):
                chunk_indices = seg.get('chunk_indices', [])
                
                # Get the text chunks for this segment
                segment_chunks = [text_chunks[i] for i in chunk_indices if i < len(text_chunks)]
                
                if not segment_chunks:
                    continue
                
                # Combine text and timing
                segment_text = ' '.join([c['text'] for c in segment_chunks])
                start_time = segment_chunks[0]['start_time']
                end_time = segment_chunks[-1]['end_time']
                
                segments.append({
                    'text': segment_text,
                    'start_time': start_time,
                    'end_time': end_time,
                    'description': seg.get('description', ''),
                    'keywords': seg.get('keywords', []),
                    'action_type': seg.get('action_type', 'general')
                })
            
            if not segments:
                logger.warning("No valid segments parsed, using fallback")
                return self._fallback_segmentation(text_chunks)
            
            return segments
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.warning("Using fallback segmentation")
            return self._fallback_segmentation(text_chunks)
    
    def _fallback_segmentation(self, text_chunks: List[Dict]) -> List[Dict]:
        """
        Fallback segmentation when LLM fails.
        Simply treats each chunk as a segment.
        
        Args:
            text_chunks: Original text chunks
        
        Returns:
            List of segment dicts
        """
        logger.info("Using fallback segmentation (one segment per chunk)")
        
        segments = []
        for chunk in text_chunks:
            segments.append({
                'text': chunk['text'],
                'start_time': chunk['start_time'],
                'end_time': chunk['end_time'],
                'description': 'Auto-generated segment',
                'keywords': [],
                'action_type': 'general'
            })
        
        return segments
    
    def _create_segments_with_timing(
        self,
        semantic_segments: List[Dict],
        words_with_timing: List[Dict]
    ) -> List[ScriptSegment]:
        """
        Convert semantic segments to ScriptSegment objects with timing.
        
        Args:
            semantic_segments: List of segment dicts from LLM
            words_with_timing: Original word timing information
        
        Returns:
            List of ScriptSegment objects
        """
        segments = []
        
        for i, seg in enumerate(semantic_segments):
            segment = ScriptSegment(
                segment_id=i,
                text=seg['text'],
                start_time=seg['start_time'],
                end_time=seg['end_time'],
                duration=seg['end_time'] - seg['start_time'],
                description=seg.get('description'),
                keywords=seg.get('keywords'),
                action_type=seg.get('action_type')
            )
            segments.append(segment)
        
        return segments
    
    def save_segments(self, segments: List[ScriptSegment], output_path: Path):
        """
        Save segments to a JSON file.
        
        Args:
            segments: List of ScriptSegment objects
            output_path: Path to save the JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        segments_data = [asdict(seg) for seg in segments]
        
        with open(output_path, 'w') as f:
            json.dump(segments_data, f, indent=2)
        
        logger.info(f"Saved {len(segments)} segments to {output_path}")
    
    @staticmethod
    def load_segments(input_path: Path) -> List[ScriptSegment]:
        """
        Load segments from a JSON file.
        
        Args:
            input_path: Path to the JSON file
        
        Returns:
            List of ScriptSegment objects
        """
        with open(input_path, 'r') as f:
            segments_data = json.load(f)
        
        segments = [ScriptSegment(**seg) for seg in segments_data]
        
        logger.info(f"Loaded {len(segments)} segments from {input_path}")
        return segments


def main():
    """Example usage"""
    # Example transcription data
    words_with_timing = [
        {'word': 'Hello', 'start_time': 0.0, 'end_time': 0.5},
        {'word': 'world', 'start_time': 0.5, 'end_time': 1.0},
        {'word': 'this', 'start_time': 1.0, 'end_time': 1.3},
        {'word': 'is', 'start_time': 1.3, 'end_time': 1.5},
        {'word': 'a', 'start_time': 1.5, 'end_time': 1.6},
        {'word': 'test', 'start_time': 1.6, 'end_time': 2.0},
    ]
    
    full_text = "Hello world this is a test"
    
    # Initialize segmenter with local LLM
    segmenter = ScriptSegmenter()
    
    # Segment the script
    segments = segmenter.segment_script(full_text, words_with_timing, max_segment_words=3)
    
    # Print results
    for seg in segments:
        print(f"\nSegment {seg.segment_id}:")
        print(f"  Text: {seg.text}")
        print(f"  Time: {seg.start_time:.2f}s - {seg.end_time:.2f}s ({seg.duration:.2f}s)")
        print(f"  Description: {seg.description}")
        print(f"  Keywords: {seg.keywords}")
        print(f"  Action Type: {seg.action_type}")
    
    # Save segments
    segmenter.save_segments(segments, Path("segments.json"))


if __name__ == "__main__":
    main()
