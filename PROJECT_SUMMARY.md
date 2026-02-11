# Video Clip Selection and Sequencing - Project Summary

## Overview

This project implements an automated video editing system that uses state-of-the-art language and vision models to intelligently select and sequence B-roll video clips to match a voice-over script. It was developed as a diploma thesis project titled "Video Clip Selection and Sequencing via Language and Vision Models."

## Project Structure

```
video-sequencer/
├── src/                      # Source code
│   ├── __init__.py
│   ├── main.py              # Main CLI application
│   ├── indexing.py          # Video indexing with VideoPrism
│   ├── transcription.py     # Audio transcription with Whisper
│   ├── segmentation.py      # Script segmentation with LLM
│   ├── matching.py          # Video-text matching engine
│   └── assembly.py          # Video assembly and export
├── data/                    # Data directories
│   ├── input/
│   │   ├── videos/         # B-roll video clips
│   │   └── audio/          # Voice-over audio files
│   └── output/             # Generated videos
├── docs/                    # Documentation
│   ├── deployment.md       # Deployment guide
│   └── system_design.md    # System architecture
├── cache/                   # Cached intermediate results
├── tests/                   # Unit tests
├── README.md               # Project overview
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
├── LICENSE                # Apache 2.0 License
├── CONTRIBUTING.md        # Contributing guidelines
└── .gitignore            # Git ignore rules
```

## Core Components

### 1. Video Indexing (indexing.py)
- Uses VideoPrism to extract semantic embeddings from B-roll videos
- Stores embeddings in a FAISS vector database for efficient similarity search
- Supports both base and large VideoPrism models
- Caches indexed videos for faster subsequent runs

### 2. Voice Transcription (transcription.py)
- Leverages OpenAI Whisper for accurate speech-to-text conversion
- Extracts word-level timestamps for precise synchronization
- Supports multiple Whisper model sizes (tiny, base, small, medium, large)
- Saves transcription results in JSON format

### 3. Script Segmentation (segmentation.py)
- Uses Large Language Models to segment scripts into semantic chunks
- Supports OpenAI and Hugging Face LLM providers
- Each segment represents a single conceptual action or scene
- Extracts keywords and action types for better matching

### 4. Video-Text Matching (matching.py)
- Matches script segments to video clips using semantic similarity
- Considers motion, context, and timing constraints
- Implements scoring system with multiple factors:
  - Semantic similarity (50%)
  - Motion score (30%)
  - Context score (20%)
- Handles clip reuse and trimming intelligently

### 5. Video Assembly (assembly.py)
- Trims video clips to match segment durations
- Concatenates clips in sequence
- Adds voice-over audio to final video
- Supports both FFmpeg and MoviePy backends
- Cleans up temporary files automatically

### 6. Main CLI (main.py)
- Orchestrates the entire pipeline
- Provides command-line interface with extensive options
- Implements caching for intermediate results
- Comprehensive logging and error handling

## Technical Stack

- **Python**: 3.9+
- **VideoPrism**: Video understanding and embeddings
- **OpenAI Whisper**: Speech-to-text transcription
- **JAX/Flax**: Deep learning framework for VideoPrism
- **PyTorch**: Deep learning framework for Whisper and LLMs
- **FAISS**: Efficient similarity search
- **FFmpeg**: Video processing
- **OpenAI API**: LLM for script segmentation

## Hardware Requirements

The system is designed to run on high-performance workstations with:
- **CPU**: AMD Ryzen Threadripper PRO 5965WX (24 cores)
- **GPU**: 4x NVIDIA RTX 6000 Ada Generation
- **OS**: Ubuntu 24.04 LTS
- **RAM**: Sufficient for large model loading (32GB+ recommended)

## Usage

### Basic Command

```bash
python src/main.py --video-dir ./data/input/videos --audio ./data/input/audio/voiceover.mp3 --output ./data/output
```

### Advanced Options

```bash
python src/main.py \
  --video-dir ./data/input/videos \
  --audio ./data/input/audio/voiceover.mp3 \
  --output ./data/output \
  --whisper-model medium \
  --videoprism-model videoprism_public_v1_large \
  --llm-provider openai \
  --no-reuse \
  --verbose
```

## Pipeline Flow

1. **Indexing**: Process all B-roll videos and create embeddings
2. **Transcription**: Convert voice-over to text with timestamps
3. **Segmentation**: Split script into semantic chunks
4. **Matching**: Find best video clips for each segment
5. **Assembly**: Trim, concatenate, and add audio

## Key Features

- **Semantic Understanding**: Uses VideoPrism's powerful video understanding capabilities
- **Precise Timing**: Word-level timestamps ensure accurate synchronization
- **Intelligent Matching**: Multi-factor scoring for optimal clip selection
- **Efficient Caching**: Saves intermediate results to speed up iterations
- **Flexible Configuration**: Multiple model options and parameters
- **Production Ready**: Robust error handling and logging

## Future Enhancements

- PyTorch support for VideoPrism
- Web interface for easier interaction
- Real-time preview during processing
- Advanced motion analysis features
- Support for multiple audio tracks
- Batch processing capabilities
- Fine-tuning options for specific domains

## GitHub Repository

- **URL**: https://github.com/mighty-programmer/video-sequencer
- **License**: Apache 2.0
- **Version**: 0.1.0
- **Release Date**: January 2026

## Contact

For questions, issues, or contributions, please visit the GitHub repository and open an issue or pull request.
