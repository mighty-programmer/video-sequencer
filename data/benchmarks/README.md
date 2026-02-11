# Benchmarks Directory

This directory contains test data for evaluating the video sequencer performance.

## Structure

```
benchmarks/
├── videos/          # B-roll video clips for testing
├── audio/           # Voice-over audio files for testing
└── README.md        # This file
```

## Usage

### Upload Test Files from Your Mac

```bash
# Upload videos
scp ~/path/to/test_videos/*.mp4 giannis_pantrakis@neghvar:~/video-sequencer/data/benchmarks/videos/

# Upload audio
scp ~/path/to/voiceover.mp3 giannis_pantrakis@neghvar:~/video-sequencer/data/benchmarks/audio/
```

### Run Benchmark Test

```bash
cd ~/video-sequencer
source john_thesis/bin/activate

python src/main.py \
  --video-dir ./data/benchmarks/videos \
  --audio ./data/benchmarks/audio/voiceover.mp3 \
  --output ./data/output/benchmark_test \
  --gpu-device cuda:0 \
  --verbose
```

## Notes

- Video files should be in common formats (MP4, MOV, AVI)
- Audio files should be in MP3, WAV, or M4A format
- Keep test files reasonably sized for quick iteration
- These directories are in `.gitignore` to avoid bloating the repository
