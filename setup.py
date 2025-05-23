"""
Setup script for the CritterDetector/owl_highlighter package.

This script handles the installation of the package and its dependencies.
It defines all required libraries and their minimum versions needed for
the various detection models and utilities.

To install the package:
    pip install .      # Standard installation
    pip install -e .   # Development installation (editable mode)

Note for Windows CUDA users:
    For Windows with CUDA, you might need to install PyTorch separately using:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    (Adjust cu118 to match your CUDA version)
"""

from setuptools import setup, find_packages
import sys
import os

# Platform-specific dependencies
if sys.platform.startswith('win'):
    # Windows-specific notes
    print("\n" + "="*80)
    print("NOTE FOR WINDOWS USERS:")
    print("This package requires PyTorch with CUDA support.")
    print("After installation, please verify your PyTorch CUDA support by running:")
    print("python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\"")
    print("If CUDA is not available, install PyTorch with CUDA using:")
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("(Adjust cu118 to match your CUDA version)")
    print("="*80 + "\n")

setup(
    name="owl_highlighter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Core deep learning packages
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.39.0",  # For OWLv2, DETR, CLIP, Grounding DINO
        "safetensors",           # For efficient model loading
        "timm",                  # Required by some transformers models
        "ultralytics>=8.0.0",    # For YOLOv8 and YOLO-World
        
        # Image processing
        "Pillow>=9.0.0",         # PIL for image handling
        "opencv-python==4.9.0.80", # OpenCV for video processing
        
        # Data handling and utilities
        "numpy<2.0",             # Array operations (capped at <2.0 for compatibility)
        "pandas",                # Data manipulation
        "scipy",                 # Scientific computing
        "scikit-learn",          # For evaluation metrics
        
        # Visualization and output
        "matplotlib",            # For plotting
        "colorama>=0.4.6",       # Terminal colors
        "tqdm",                  # Progress bars
        
        # Configuration and cloud
        "toml",                  # Config file parsing
        "boto3",                 # AWS integration
        "wandb",                 # Weights & Biases logging
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ]
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for detecting and highlighting objects in videos using OWL-ViT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/owl_highlighter",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    package_data={
        "owl_highlighter": ["*.ttf"],
    },
    entry_points={
        "console_scripts": [
            "owl-evaluate=owl_highlighter.evaluate_detections:main",
            "owl-highlight=owl_highlighter.run_highlighter:main",
            "owl-setup=owl_highlighter.setup_env:main",
        ],
    },
)