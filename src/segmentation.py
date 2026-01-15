"""
Script Segmentation Module using Large Language Models

This module is responsible for:
1. Taking a transcribed script with timing information
2. Using an LLM to split the script into semantic segments
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
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LLMProvider(Enum):
    """Available LLM providers"""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


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
    LLM-based script segmentation system.
    
    This class takes a transcribed script and uses an LLM to identify
    semantic boundaries and create meaningful segments for video matching.
    """
    
    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OPENAI,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the ScriptSegmenter.
        
        Args:
            provider: LLM provider to use
            model_name: Model name (e.g., 'gpt-4', 'mistral-7b')
            api_key: API key for the provider (if needed)
        """
        self.provider = provider
        self.model_name = model_name or self._get_default_model(provider)
        self.api_key = api_key
        
        if provider == LLMProvider.OPENAI:
            self._init_openai()
        elif provider == LLMProvider.HUGGINGFACE:
            self._init_huggingface()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        logger.info(f"ScriptSegmenter initialized with {provider.value} ({self.model_name})")
    
    def _get_default_model(self, provider: LLMProvider) -> str:
        """Get default model for a provider"""
        if provider == LLMProvider.OPENAI:
            return "gpt-4-mini"
        elif provider == LLMProvider.HUGGINGFACE:
            return "mistralai/Mistral-7B-Instruct-v0.1"
        return "unknown"
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        if OpenAI is None:
            raise ImportError("OpenAI not installed. Run: pip install openai")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def _init_huggingface(self):
        """Initialize Hugging Face model"""
        if torch is None or AutoTokenizer is None:
            raise ImportError("Transformers not installed. Run: pip install transformers torch")
        
        logger.info(f"Loading Hugging Face model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
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
        Use LLM to identify semantic boundaries and segment descriptions.
        
        Args:
            text_chunks: List of text chunks
        
        Returns:
            List of semantic segments with descriptions
        """
        # Combine chunks into a single text for LLM processing
        combined_text = '\n'.join([f"[{i}] {c['text']}" for i, c in enumerate(text_chunks)])
        
        prompt = self._create_segmentation_prompt(combined_text)
        
        if self.provider == LLMProvider.OPENAI:
            response = self._query_openai(prompt)
        else:
            response = self._query_huggingface(prompt)
        
        # Parse LLM response to extract segments
        segments = self._parse_segmentation_response(response, text_chunks)
        
        return segments
    
    def _create_segmentation_prompt(self, text: str) -> str:
        """Create a prompt for the LLM to segment the script"""
        prompt = f"""You are an expert video editor and scriptwriter. Your task is to segment the following script into meaningful semantic chunks. Each chunk should represent a single conceptual action, scene, or idea.

For each segment, provide:
1. The chunk indices it covers (e.g., [0-2])
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
    
    def _query_openai(self, prompt: str) -> str:
        """Query OpenAI API"""
        logger.info(f"Querying OpenAI with model {self.model_name}")
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for video editing and script analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    def _query_huggingface(self, prompt: str) -> str:
        """Query Hugging Face model"""
        logger.info(f"Querying Hugging Face model {self.model_name}")
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=2000,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
                logger.warning("Could not find JSON in LLM response, using default segmentation")
                return self._create_default_segments(text_chunks)
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            segments = data.get('segments', [])
            return segments
        
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._create_default_segments(text_chunks)
    
    def _create_default_segments(self, text_chunks: List[Dict]) -> List[Dict]:
        """Create default segments if LLM parsing fails"""
        segments = []
        for i, chunk in enumerate(text_chunks):
            segments.append({
                'chunk_indices': [i],
                'description': f"Segment {i+1}",
                'keywords': [],
                'action_type': 'unknown'
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
            semantic_segments: Segments from LLM
            words_with_timing: Word timing information
        
        Returns:
            List of ScriptSegment objects
        """
        script_segments = []
        
        for seg_id, seg_data in enumerate(semantic_segments):
            chunk_indices = seg_data.get('chunk_indices', [])
            
            # Find start and end times
            if isinstance(chunk_indices, list) and len(chunk_indices) > 0:
                # Assuming chunk_indices are indices into words
                start_idx = chunk_indices[0] if isinstance(chunk_indices[0], int) else 0
                end_idx = chunk_indices[-1] if isinstance(chunk_indices[-1], int) else len(words_with_timing) - 1
                
                # Clamp indices
                start_idx = max(0, min(start_idx, len(words_with_timing) - 1))
                end_idx = max(0, min(end_idx, len(words_with_timing) - 1))
                
                start_time = words_with_timing[start_idx].get('start_time', 0.0)
                end_time = words_with_timing[end_idx].get('end_time', 0.0)
            else:
                start_time = 0.0
                end_time = 0.0
            
            duration = end_time - start_time
            
            # Extract text for this segment
            segment_words = [w.get('word', '') for w in words_with_timing[start_idx:end_idx+1]]
            text = ' '.join(segment_words)
            
            script_segment = ScriptSegment(
                segment_id=seg_id,
                text=text,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                description=seg_data.get('description'),
                keywords=seg_data.get('keywords', []),
                action_type=seg_data.get('action_type')
            )
            
            script_segments.append(script_segment)
        
        return script_segments
    
    def save_segments(
        self,
        segments: List[ScriptSegment],
        output_path: str
    ):
        """
        Save segments to a JSON file.
        
        Args:
            segments: List of ScriptSegment objects
            output_path: Path to save the JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'segments': [asdict(s) for s in segments],
            'total_segments': len(segments),
            'total_duration': sum(s.duration for s in segments)
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(segments)} segments to {output_path}")
    
    def load_segments(self, json_path: str) -> List[ScriptSegment]:
        """
        Load segments from a JSON file.
        
        Args:
            json_path: Path to the JSON file
        
        Returns:
            List of ScriptSegment objects
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        segments = [ScriptSegment(**s) for s in data['segments']]
        return segments


if __name__ == '__main__':
    # Example usage
    segmenter = ScriptSegmenter(provider=LLMProvider.OPENAI)
    
    # Example words with timing
    example_words = [
        {'word': 'Hello', 'start_time': 0.0, 'end_time': 0.5},
        {'word': 'today', 'start_time': 0.5, 'end_time': 1.0},
        {'word': 'I', 'start_time': 1.0, 'end_time': 1.3},
        {'word': 'will', 'start_time': 1.3, 'end_time': 1.6},
        {'word': 'show', 'start_time': 1.6, 'end_time': 2.0},
        {'word': 'you', 'start_time': 2.0, 'end_time': 2.3},
        {'word': 'how', 'start_time': 2.3, 'end_time': 2.6},
        {'word': 'to', 'start_time': 2.6, 'end_time': 2.9},
        {'word': 'plug', 'start_time': 2.9, 'end_time': 3.2},
        {'word': 'in', 'start_time': 3.2, 'end_time': 3.5},
        {'word': 'a', 'start_time': 3.5, 'end_time': 3.7},
        {'word': 'USB', 'start_time': 3.7, 'end_time': 4.2},
        {'word': 'cable', 'start_time': 4.2, 'end_time': 4.7},
    ]
    
    full_text = ' '.join([w['word'] for w in example_words])
    
    segments = segmenter.segment_script(full_text, example_words)
    
    for seg in segments:
        print(f"Segment {seg.segment_id}: {seg.description}")
        print(f"  Text: {seg.text}")
        print(f"  Time: {seg.start_time:.2f}s - {seg.end_time:.2f}s ({seg.duration:.2f}s)")
        print()
