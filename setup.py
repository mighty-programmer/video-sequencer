"""Setup script for the Video Sequencer project"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="video-sequencer",
    version="0.1.0",
    author="Manus AI",
    description="Video Clip Selection and Sequencing via Language and Vision Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/video-sequencer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "opencv-python>=4.5.0",
        "librosa>=0.9.0",
        "moviepy>=1.0.0",
        "imageio>=2.9.0",
        "jax>=0.3.0",
        "jaxlib>=0.3.0",
        "flax>=0.4.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "transformers>=4.20.0",
        "videoprism>=0.1.0",
        "openai-whisper>=20230314",
        "openai>=1.0.0",
        "faiss-cpu>=1.7.0",
        "typer>=0.4.0",
        "click>=8.0.0",
        "tqdm>=4.62.0",
        "pydantic>=1.9.0",
        "pyyaml>=5.4.0",
    ],
    entry_points={
        "console_scripts": [
            "video-sequencer=main:main",
        ],
    },
)
