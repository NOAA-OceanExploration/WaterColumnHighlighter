"""
Utility functions for CritterDetector.

This module contains platform-specific utilities and helper functions that are used across
the CritterDetector application.
"""

import os
import sys
import platform
from pathlib import Path
from typing import Union, List, Optional

def normalize_path(path: Union[str, Path]) -> str:
    """
    Normalize file paths for cross-platform compatibility.
    
    Args:
        path: Input path as string or Path object
        
    Returns:
        Normalized path string appropriate for the current platform
    """
    if path is None:
        return None
        
    # Convert to Path object for consistent handling
    if not isinstance(path, Path):
        path = Path(path)
    
    # Resolve to absolute path and normalize
    try:
        normalized = path.resolve()
        return str(normalized)
    except (OSError, RuntimeError):
        # If resolution fails, return the original but normalized
        return str(path)
        
def create_directory_structure(base_dir: Optional[str] = None) -> None:
    """
    Create the standard directory structure needed for CritterDetector.
    
    Args:
        base_dir: Base directory to create structure in (default: current directory)
    """
    if base_dir is None:
        base_dir = os.getcwd()
        
    directories = [
        "models",
        "videos",
        "checkpoints",
        "highlights",
        "timelines",
        "evaluation_results",
        "annotations",
        "verification_frames"
    ]
    
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
        else:
            print(f"Directory already exists: {dir_path}")
            
def is_windows() -> bool:
    """Check if the current platform is Windows."""
    return platform.system() == "Windows"
    
def is_macos() -> bool:
    """Check if the current platform is macOS."""
    return platform.system() == "Darwin"
    
def is_linux() -> bool:
    """Check if the current platform is Linux."""
    return platform.system() == "Linux"
    
def get_cuda_info() -> dict:
    """
    Get information about CUDA availability and configuration.
    
    Returns:
        Dictionary with CUDA information
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        info = {
            "cuda_available": cuda_available,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "current_device": torch.cuda.current_device() if cuda_available else None,
            "device_name": torch.cuda.get_device_name(0) if cuda_available and torch.cuda.device_count() > 0 else None,
            "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        }
        
        # Try to get more detailed driver info on Windows
        if is_windows() and cuda_available:
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,memory.total', '--format=csv,noheader'], 
                                       capture_output=True, text=True, check=True)
                driver_info = result.stdout.strip().split(',')
                if len(driver_info) >= 2:
                    info["driver_version"] = driver_info[0].strip()
                    info["total_memory"] = driver_info[1].strip()
            except (subprocess.SubprocessError, FileNotFoundError, IndexError):
                # nvidia-smi not available or error running it
                pass
                
        return info
    except ImportError:
        return {"cuda_available": False, "error": "PyTorch not installed"}
        
def get_safe_temp_dir() -> str:
    """
    Get a safe temporary directory path that works across platforms.
    
    Returns:
        Path to a temporary directory
    """
    import tempfile
    return tempfile.gettempdir()
    
def set_environment_variables_for_cuda():
    """Set environment variables to optimize CUDA performance."""
    if is_windows():
        # Windows-specific optimizations
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    else:
        # Linux/macOS optimizations
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
        
    # Common optimizations
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"  # Enable cuDNN v8 API 