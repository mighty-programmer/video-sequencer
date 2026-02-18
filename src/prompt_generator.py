"""
LLM-based Prompt Generator for OpenCLIP Ensemble Matching

Generates diverse prompt variations for each segment text using a local LLM
(via Ollama). These variations are then encoded with CLIP and averaged to
create a more robust text embedding.

Supported LLM backends:
- Ollama (default): Runs locally, supports Llama 3.2, Mistral, Phi-3, etc.

The generator creates multiple paraphrases and prompt-engineered variations
of each segment text, optimized for CLIP's text-image matching.
"""

import json
import logging
import subprocess
import re
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class OllamaPromptGenerator:
    """
    Generate diverse prompt variations using a local LLM via Ollama.
    
    For each segment text, generates N variations that:
    1. Paraphrase the original text
    2. Add visual descriptors ("a video showing...", "a close-up of...")
    3. Focus on different aspects (action, object, setting)
    
    These variations are designed to improve CLIP matching by covering
    more of CLIP's training distribution.
    """
    
    SYSTEM_PROMPT = """You are a prompt engineering assistant for CLIP (Contrastive Language-Image Pre-training).
Your job is to generate diverse text descriptions for video-to-text matching.

Given a short text describing a video scene, generate exactly {num_variations} different descriptions of the same scene.
Each description should:
1. Describe the same visual content but from different angles
2. Use different phrasing and vocabulary
3. Include visual details that CLIP would recognize
4. Be concise (under 77 tokens, which is CLIP's max)

Output ONLY a JSON array of strings. No explanations, no numbering, just the JSON array.

Example input: "dice and rinse your potatoes"
Example output: ["a person dicing potatoes on a cutting board", "rinsing diced potatoes under running water", "a cooking scene showing potato preparation with dicing and rinsing", "hands cutting potatoes into small cubes and washing them", "close-up of potatoes being diced and rinsed in a kitchen"]"""

    def __init__(
        self,
        model_name: str = 'llama3.2:3b',
        num_variations: int = 5,
        ollama_host: str = 'http://localhost:11434'
    ):
        """
        Initialize the prompt generator.
        
        Args:
            model_name: Ollama model name (e.g., 'llama3.2:3b', 'mistral:7b-instruct')
            num_variations: Number of prompt variations to generate per segment
            ollama_host: Ollama API endpoint
        """
        self.model_name = model_name
        self.num_variations = num_variations
        self.ollama_host = ollama_host
        self._cache: Dict[str, List[str]] = {}
        
        logger.info(f"PromptGenerator initialized: model={model_name}, variations={num_variations}")
    
    def check_ollama_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            import requests
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                if any(self.model_name in name for name in model_names):
                    return True
                else:
                    logger.warning(
                        f"Model '{self.model_name}' not found in Ollama. "
                        f"Available models: {model_names}. "
                        f"Run: ollama pull {self.model_name}"
                    )
                    return False
            return False
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False
    
    def generate_prompts(self, text: str) -> List[str]:
        """
        Generate diverse prompt variations for a text segment.
        
        Args:
            text: Original segment text
            
        Returns:
            List of prompt variations (includes original text)
        """
        # Check cache
        if text in self._cache:
            return self._cache[text]
        
        # Always include the original text and basic templates
        prompts = [text]
        
        try:
            llm_prompts = self._generate_via_ollama(text)
            if llm_prompts:
                prompts.extend(llm_prompts)
            else:
                # Fallback to template-based generation
                prompts.extend(self._generate_template_prompts(text))
        except Exception as e:
            logger.warning(f"LLM generation failed for '{text[:50]}...': {e}")
            prompts.extend(self._generate_template_prompts(text))
        
        # Deduplicate while preserving order
        seen = set()
        unique_prompts = []
        for p in prompts:
            p_lower = p.strip().lower()
            if p_lower not in seen:
                seen.add(p_lower)
                unique_prompts.append(p.strip())
        
        self._cache[text] = unique_prompts
        return unique_prompts
    
    def _generate_via_ollama(self, text: str) -> List[str]:
        """Generate prompts using Ollama API."""
        import requests
        
        system_prompt = self.SYSTEM_PROMPT.format(num_variations=self.num_variations)
        
        payload = {
            'model': self.model_name,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': text}
            ],
            'stream': False,
            'options': {
                'temperature': 0.8,
                'top_p': 0.9,
                'num_predict': 512
            }
        }
        
        response = requests.post(
            f"{self.ollama_host}/api/chat",
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            logger.warning(f"Ollama API error: {response.status_code}")
            return []
        
        result = response.json()
        content = result.get('message', {}).get('content', '')
        
        return self._parse_llm_response(content)
    
    def _parse_llm_response(self, content: str) -> List[str]:
        """Parse the LLM response to extract prompt variations."""
        # Try to parse as JSON array
        try:
            # Find JSON array in the response
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                prompts = json.loads(match.group())
                if isinstance(prompts, list) and all(isinstance(p, str) for p in prompts):
                    return [p.strip() for p in prompts if p.strip()]
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: try to extract numbered or bullet-pointed items
        lines = content.strip().split('\n')
        prompts = []
        for line in lines:
            # Remove numbering, bullets, quotes
            cleaned = re.sub(r'^[\d\.\)\-\*\s"\']+', '', line).strip()
            cleaned = cleaned.strip('"\'')
            if cleaned and len(cleaned) > 5:
                prompts.append(cleaned)
        
        return prompts[:self.num_variations]
    
    def _generate_template_prompts(self, text: str) -> List[str]:
        """
        Generate prompt variations using templates (fallback when LLM unavailable).
        
        These templates are based on CLIP prompt engineering best practices
        from the original CLIP paper.
        """
        templates = [
            f"a video of {text}",
            f"a photo of {text}",
            f"a scene showing {text}",
            f"a short clip of {text}",
            f"a close-up video of {text}",
        ]
        return templates[:self.num_variations]
    
    def generate_all_prompts(
        self,
        segments: List[Dict],
        show_progress: bool = True
    ) -> Dict[str, List[str]]:
        """
        Generate prompt variations for all segments.
        
        Args:
            segments: List of segment dicts with 'text' key
            show_progress: Whether to show progress
            
        Returns:
            Dict mapping segment text -> list of prompt variations
        """
        all_prompts = {}
        
        for i, segment in enumerate(segments):
            text = segment['text']
            if show_progress:
                logger.info(f"Generating prompts for segment {i}/{len(segments)}: {text[:50]}...")
            
            prompts = self.generate_prompts(text)
            all_prompts[text] = prompts
            
            if show_progress:
                logger.info(f"  Generated {len(prompts)} variations")
        
        return all_prompts
    
    def save_prompts(self, prompts: Dict[str, List[str]], output_path: str):
        """Save generated prompts to a JSON file."""
        with open(output_path, 'w') as f:
            json.dump(prompts, f, indent=2)
        logger.info(f"Saved prompts to {output_path}")
    
    def load_prompts(self, input_path: str) -> Dict[str, List[str]]:
        """Load previously generated prompts from a JSON file."""
        with open(input_path, 'r') as f:
            prompts = json.load(f)
        self._cache.update(prompts)
        logger.info(f"Loaded prompts from {input_path}")
        return prompts


def create_ensemble_templates(
    segments: List[Dict],
    llm_model: str = None,
    num_variations: int = 5,
    cache_file: str = None
) -> Dict[str, List[str]]:
    """
    Convenience function to generate ensemble prompts for all segments.
    
    If an LLM model is specified and Ollama is available, uses LLM generation.
    Otherwise, falls back to template-based generation.
    
    Args:
        segments: List of segment dicts with 'text' key
        llm_model: Ollama model name (None for template-only)
        num_variations: Number of variations per segment
        cache_file: Optional path to cache/load prompts
        
    Returns:
        Dict mapping segment text -> list of prompt variations
    """
    # Check cache first
    if cache_file and Path(cache_file).exists():
        with open(cache_file, 'r') as f:
            cached = json.load(f)
        # Verify all segments are cached
        if all(seg['text'] in cached for seg in segments):
            logger.info(f"Loaded cached prompts from {cache_file}")
            return cached
    
    if llm_model:
        generator = OllamaPromptGenerator(
            model_name=llm_model,
            num_variations=num_variations
        )
        
        if generator.check_ollama_available():
            logger.info(f"Using LLM ({llm_model}) for prompt generation")
            prompts = generator.generate_all_prompts(segments)
        else:
            logger.warning(f"Ollama not available, falling back to template-based prompts")
            prompts = {}
            for seg in segments:
                text = seg['text']
                prompts[text] = generator._generate_template_prompts(text)
                prompts[text].insert(0, text)  # Include original
    else:
        # Template-only mode
        prompts = {}
        for seg in segments:
            text = seg['text']
            prompts[text] = [
                text,
                f"a video of {text}",
                f"a photo of {text}",
                f"a scene showing {text}",
                f"a short clip of {text}",
            ]
    
    # Save to cache
    if cache_file:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(prompts, f, indent=2)
        logger.info(f"Cached prompts to {cache_file}")
    
    return prompts
