# Installation Guide

Complete installation instructions for the Video Sequencer project.

## System Requirements

- **Operating System**: Ubuntu 22.04+ or similar Linux distribution
- **Python**: 3.9 - 3.12 (tested on 3.12)
- **GPU**: NVIDIA GPU with CUDA 12+ support (recommended)
- **RAM**: 16GB+ recommended
- **Storage**: 20GB+ free space (for models and cache)

## Installation Steps

### 1. Clone the Repository

```bash
cd ~
git clone https://github.com/mighty-programmer/video-sequencer.git
cd video-sequencer
```

### 2. Create Virtual Environment

```bash
python3 -m venv john_thesis
source john_thesis/bin/activate
pip install --upgrade pip
```

### 3. Install Core Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install JAX with CUDA Support

For NVIDIA GPUs with CUDA 12:

```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For CPU-only (not recommended):

```bash
pip install --upgrade jax jaxlib
```

### 5. Install VideoPrism

VideoPrism must be installed from source:

```bash
cd ~
git clone https://github.com/google-deepmind/videoprism.git
cd videoprism
pip install --no-deps .
cd ~/video-sequencer
```

**Note**: We use `--no-deps` to prevent VideoPrism from downgrading your JAX/CUDA installation.

### 6. Verify Installation

```bash
python3 -c "import torch; import jax; import videoprism; import whisper; print('✅ All systems go! GPU available:', torch.cuda.is_available())"
```

You should see: `✅ All systems go! GPU available: True`

### 7. Set Up Hugging Face Authentication

The project uses Llama 3.2 which requires Hugging Face authentication:

```bash
python3 -c "from huggingface_hub import login; login('YOUR_HF_TOKEN_HERE')"
```

Get your token from: https://huggingface.co/settings/tokens

## Directory Structure

After installation, your project should look like this:

```
video-sequencer/
├── src/                    # Source code
├── data/
│   ├── benchmarks/        # Test data (not in git)
│   ├── input/             # Your video projects (not in git)
│   └── output/            # Generated videos (not in git)
├── cache/                 # Model cache (not in git)
├── john_thesis/           # Virtual environment (not in git)
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Troubleshooting

### Python Version Issues

If you see errors about Python version compatibility, ensure you're using Python 3.9-3.12:

```bash
python3 --version
```

### CUDA Not Available

If `torch.cuda.is_available()` returns `False`:

1. Check NVIDIA drivers: `nvidia-smi`
2. Verify CUDA installation: `nvcc --version`
3. Reinstall PyTorch with CUDA support

### JAX CUDA Issues

If JAX doesn't detect your GPU:

```bash
pip uninstall jax jaxlib
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### VideoPrism Import Error

If you get `ModuleNotFoundError: No module named 'videoprism'`:

1. Ensure you cloned the repository: `ls ~/videoprism`
2. Reinstall: `cd ~/videoprism && pip install --no-deps .`

### Permission Denied (sudo)

If you don't have sudo access, most Python packages can still be installed in your virtual environment. System packages like FFmpeg must be installed by your system administrator.

## Next Steps

Once installation is complete, see the main [README.md](README.md) for usage instructions.
