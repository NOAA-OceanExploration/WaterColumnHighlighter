"""
Environment setup utility for CritterDetector.

This script provides utilities for setting up the CritterDetector environment,
including creating the directory structure, checking CUDA compatibility,
and initializing the configuration.
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path
import toml

from .utils import (
    normalize_path, 
    create_directory_structure, 
    get_cuda_info, 
    set_environment_variables_for_cuda,
    is_windows,
    is_macos,
    is_linux
)

def check_environment():
    """
    Check the environment for required dependencies and CUDA setup.
    
    Returns:
        dict: Environment information
    """
    env_info = {
        "platform": platform.system(),
        "python_version": sys.version,
        "executable": sys.executable,
    }
    
    # Check for required packages
    required_packages = [
        "torch", "transformers", "opencv-python", 
        "ultralytics", "PIL", "numpy"
    ]
    
    packages_status = {}
    for package in required_packages:
        try:
            if package == "PIL":
                import PIL
                packages_status[package] = PIL.__version__
            elif package == "torch":
                import torch
                packages_status[package] = torch.__version__
            elif package == "transformers":
                import transformers
                packages_status[package] = transformers.__version__
            elif package == "ultralytics":
                import ultralytics
                packages_status[package] = ultralytics.__version__
            elif package == "opencv-python":
                import cv2
                packages_status[package] = cv2.__version__
            elif package == "numpy":
                import numpy
                packages_status[package] = numpy.__version__
            else:
                module = __import__(package)
                packages_status[package] = getattr(module, "__version__", "Unknown")
        except ImportError:
            packages_status[package] = "Not installed"
    
    env_info["packages"] = packages_status
    
    # Add CUDA information
    env_info["cuda"] = get_cuda_info()
    
    return env_info

def print_environment_info(env_info):
    """Print environment information in a formatted way."""
    print("\n==== CritterDetector Environment Information ====")
    print(f"Platform: {env_info['platform']}")
    print(f"Python: {env_info['python_version']}")
    print("\n---- Required Packages ----")
    
    for package, version in env_info["packages"].items():
        status = "✓" if version != "Not installed" else "✗"
        print(f"{status} {package}: {version}")
    
    print("\n---- CUDA Information ----")
    cuda_info = env_info["cuda"]
    if cuda_info["cuda_available"]:
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {cuda_info.get('cuda_version', 'Unknown')}")
        print(f"Device Count: {cuda_info.get('device_count', 0)}")
        print(f"Device Name: {cuda_info.get('device_name', 'Unknown')}")
        if "driver_version" in cuda_info:
            print(f"Driver Version: {cuda_info['driver_version']}")
        if "total_memory" in cuda_info:
            print(f"Total Memory: {cuda_info['total_memory']}")
    else:
        print("CUDA Available: No")
        if "error" in cuda_info:
            print(f"Error: {cuda_info['error']}")

def generate_default_config():
    """
    Generate a default config.toml file.
    
    Returns:
        str: Path to the created config file
    """
    # Default configuration based on the platform
    config = {
        "paths": {
            "model_save_path": "models",
            "dataset_cache_dir": "models",
            "checkpoint_dir": "checkpoints",
            "video_dir": "videos",
            "csv_dir": "annotations",
            "highlight_output_dir": "highlights",
            "timeline_output_dir": "timelines",
            "evaluation_output_dir": "evaluation_results",
            "annotation_csv": os.path.join("annotations", "example_annotations.csv"),
            "verification_output_dir": "verification_frames",
        },
        "training": {
            "window_size": 20,
            "stride": 1,
            "batch_size": 4,
            "num_epochs": 1,
            "learning_rate": 0.001,
            "k_folds": 5,
            "early_stopping_patience": 10,
            "gradient_clip": 1.0,
            "checkpoint_steps": 1000,
            "scheduler_factor": 0.5,
            "scheduler_patience": 5,
            "optimizer": "Adam",
            "loss_function": "FocalLoss",
            "mixed_precision": True,
        },
        "data": {
            "frame_rate": 29,
        },
        "augmentation": {
            "random_crop_size": 200,
            "color_jitter_brightness": 0.1,
            "color_jitter_contrast": 0.1,
            "color_jitter_saturation": 0.1,
            "color_jitter_hue": 0.1,
        },
        "logging": {
            "wandb_project": "critter_detector",
            "wandb_entity": "your_wandb_username",
            "log_interval": 10,
        },
        "model": {
            "model_type": "lstm",
            "feature_extractor": "resnet",
            "fine_tune": False,
            "hidden_dim": 32,
            "num_layers": 5,
        },
        "detection": {
            "model": "yoloworld",
            "model_variant": "l",
            "score_threshold": 0.25,
            "use_ensemble": False,
            "ensemble_weights": {"owl": 0.7, "yoloworld": 0.3},
            "labels_csv_path": "marine_labels.csv",
        },
        "clip": {
            "base_detector": "yolo",
            "base_detector_variant": "v8n",
            "base_detector_threshold": 0.05,
        },
        "aws": {
            "use_aws": False,
            "s3_bucket_name": "your-s3-bucket-name",
            "s3_data_prefix": "data/",
            "aws_region": "us-west-2",
        },
        "evaluation": {
            "temporal_tolerance": 300.0,
            "simplified_mode": False,
            "skip_organism_filter": True,
        },
    }
    
    # Add CUDA-specific settings
    cuda_info = get_cuda_info()
    cuda_section = {
        "enable_memory_efficient_attention": True,
        "device": 0,
    }
    
    # Add platform-specific optimizations
    if is_windows():
        cuda_section["batch_size_limit"] = 4
    else:
        cuda_section["batch_size_limit"] = 8
    
    config["cuda"] = cuda_section
    
    # Write config to file
    config_path = os.path.join(os.getcwd(), "config.toml")
    with open(config_path, "w") as f:
        toml.dump(config, f)
    
    return config_path

def copy_sample_media():
    """
    Copy sample media files to the videos directory if available.
    
    Returns:
        bool: True if sample files were copied, False otherwise
    """
    # Look for sample media in the package directory
    package_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(package_dir, "samples")
    
    if not os.path.exists(sample_dir):
        return False
    
    # Create videos directory if it doesn't exist
    videos_dir = os.path.join(os.getcwd(), "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    # Copy sample files
    count = 0
    for filename in os.listdir(sample_dir):
        if filename.endswith((".mp4", ".avi", ".mov")):
            src_path = os.path.join(sample_dir, filename)
            dst_path = os.path.join(videos_dir, filename)
            shutil.copy2(src_path, dst_path)
            count += 1
    
    return count > 0

def download_models():
    """
    Download model weights using the appropriate script.
    
    Returns:
        bool: True if models were downloaded successfully, False otherwise
    """
    try:
        # Determine which script to run based on platform
        if is_windows():
            script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "download_models.bat")
            if os.path.exists(script_path):
                print("Downloading models using Windows batch script...")
                result = subprocess.run([script_path], shell=True, check=True)
                return result.returncode == 0
            else:
                print(f"Warning: download_models.bat not found at {script_path}")
                return False
        else:
            script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "download_models.sh")
            if os.path.exists(script_path):
                print("Downloading models using shell script...")
                result = subprocess.run(["bash", script_path], check=True)
                return result.returncode == 0
            else:
                print(f"Warning: download_models.sh not found at {script_path}")
                return False
    except subprocess.SubprocessError as e:
        print(f"Error downloading models: {e}")
        return False

def main():
    """Main entry point for the setup utility."""
    print("=== CritterDetector Setup Utility ===")
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Set up the CritterDetector environment")
    parser.add_argument("--check", action="store_true", help="Check environment without making changes")
    parser.add_argument("--download-models", action="store_true", help="Download model weights")
    parser.add_argument("--generate-config", action="store_true", help="Generate a default config.toml file")
    parser.add_argument("--copy-samples", action="store_true", help="Copy sample media files if available")
    parser.add_argument("--full", action="store_true", help="Perform full setup (all operations)")
    
    args = parser.parse_args()
    
    # Default to full setup if no specific options are provided
    if not any([args.check, args.download_models, args.generate_config, args.copy_samples]):
        args.full = True
    
    # Check environment
    env_info = check_environment()
    print_environment_info(env_info)
    
    if args.check:
        # Just check the environment and exit
        return
    
    # Create directory structure
    if args.full:
        print("\nCreating directory structure...")
        create_directory_structure()
    
    # Generate config
    if args.generate_config or args.full:
        print("\nGenerating default configuration...")
        config_path = generate_default_config()
        print(f"Configuration saved to {config_path}")
    
    # Copy sample media
    if args.copy_samples or args.full:
        print("\nCopying sample media files...")
        if copy_sample_media():
            print("Sample media files copied to videos directory")
        else:
            print("No sample media files available")
    
    # Download models
    if args.download_models or args.full:
        print("\nDownloading model weights...")
        if download_models():
            print("Model weights downloaded successfully")
        else:
            print("Failed to download all model weights. You can try again later or they will be downloaded on first use.")
    
    print("\nSetup complete!")
    print("You can now use CritterDetector to process videos.")
    print("Example usage:")
    print("  python -m owl_highlighter.run_highlighter --video_path videos/your_video.mp4 --verbose")

if __name__ == "__main__":
    main() 