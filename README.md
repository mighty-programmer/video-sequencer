# Video Clip Selection and Sequencing via Language and Vision Models

This project is a comprehensive system that automates the process of video editing by intelligently selecting and sequencing B-roll video clips to match a voice-over script. It leverages state-of-the-art language and vision models to understand both the narrative of the script and the content of the video footage, creating a seamless and contextually relevant final video.

## Features

- **Automated Video Indexing**: Utilizes the VideoPrism model to create a searchable index of your video library, extracting rich semantic and temporal features.
- **Accurate Voice Transcription**: Employs OpenAI's Whisper to transcribe voice-over audio with word-level timestamps.
- **Semantic Script Segmentation**: Uses a local Large Language Model (LLM) to intelligently segment the script into meaningful narrative chunks. No API keys required.
- **Intelligent Video-Text Matching**: The core of the system, this engine matches each script segment to the most relevant video clips, considering semantic similarity, motion, and context.
- **Optimal Clip Sequencing**: Creates a timeline of video clips, handling trimming and reuse to best fit the narrative.
- **Automated Video Assembly**: Assembles the final video by cutting and concatenating the selected clips and adding the voice-over audio.
- **Command-Line Interface**: A user-friendly CLI for easy control and integration into workflows.
- **Web Editing Studio**: A browser-based, Write-A-Video-inspired editor for segment editing, candidate review, preview, render, benchmark management, cache management, and pipeline job launching.

## System Architecture

The system is designed as a modular pipeline:

1.  **Video Indexing**: B-roll videos are processed by VideoPrism to create a vector database of video embeddings.
2.  **Transcription**: The voice-over audio is transcribed by Whisper to produce a time-stamped script.
3. - **Segmentation**: A local LLM breaks the script into semantic segments.4.  **Matching & Sequencing**: The matching engine finds the best video clips for each segment and arranges them in a timeline.
5.  **Assembly**: The final video is assembled using FFmpeg or MoviePy.

## Getting Started

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA support
- FFmpeg

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd video-sequencer
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Command-Line Interface

The main application is controlled via the `main.py` script. Here are the available options:

```
usage: main.py [-h] --video-dir VIDEO_DIR --audio AUDIO --output OUTPUT [--cache-dir CACHE_DIR] [--whisper-model {tiny,base,small,medium,large}] [--videoprism-model {videoprism_public_v1_base,videoprism_public_v1_large}] [--llm-model LLM_MODEL] [--gpu-device GPU_DEVICE] [--no-reuse] [--prefer-non-reused] [--verbose]

Video Clip Selection and Sequencing via Language and Vision Models

optional arguments:
  -h, --help            show this help message and exit
  --video-dir VIDEO_DIR
                        Directory containing B-roll video clips
  --audio AUDIO         Path to voice-over audio file
  --output OUTPUT       Output directory for the final video
  --cache-dir CACHE_DIR
                        Directory for caching intermediate results (default: ./cache)
  --whisper-model {tiny,base,small,medium,large}
                        Whisper model size (default: base)
  --videoprism-model {videoprism_public_v1_base,videoprism_public_v1_large}
                        VideoPrism model to use (default: videoprism_public_v1_base)
  --llm-model LLM_MODEL
                        Local LLM model for script segmentation (default: meta-llama/Llama-3.2-3B-Instruct)
  --gpu-device GPU_DEVICE
                        GPU device to use (e.g., cuda:0, cuda:1, or cpu) (default: cuda:0)
  --no-reuse            Prevent reusing video clips
  --prefer-non-reused   Prefer non-reused clips if available (default: True)
  --verbose             Enable verbose logging
```

### Examples

**Basic usage:**

```bash
python main.py --video-dir ./data/videos --audio ./data/voiceover.mp3 --output ./output
```

**Using a larger Whisper model and a different local LLM:**

```bash
python main.py --video-dir ./data/videos --audio ./data/voiceover.mp3 --output ./output --whisper-model medium --llm-model google/gemma-2b-it --gpu-device cuda:1
```

### Web Application

The repository now includes a browser-based web app that mirrors the paper-inspired editing workflow and also exposes the existing menu-driven project operations.

Start the server from the repository root:

```bash
python3 src/webapp.py
```

Then open:

```text
http://<server-host>:8000
```

The web app includes four major areas:

- **Editor**: A Write-A-Video-style workspace with a text editor, candidate shot panel, preview panel, keyword suggestions, and segment idiom controls.
- **Pipeline**: Web forms for Quick Benchmark, Full Pipeline, OpenCLIP Grid Search, VideoPrism Grid Search, Write-A-Video Grid Search, and Compare All Models.
- **Benchmarks**: Benchmark upload, download, deletion, and direct handoff into the editor.
- **Operations**: Settings, cache management, background job monitoring, and logs.

#### Editor Notes

- `writeavideo` mode uses the repo's two-stage retrieval stack: keyword filtering plus OpenCLIP reranking.
- Segment controls support keyword overrides, movement preference, shot-duration bias, split-timing bias, split/merge, candidate refresh, and render.
- Rendering now works with or without a voice-over file, which makes silent previews possible during editing.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project is built upon the excellent work of the [VideoPrism](https://github.com/google-deepmind/videoprism) and [Whisper](https://github.com/openai/whisper) teams.
