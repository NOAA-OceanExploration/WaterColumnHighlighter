"""
Run highlighter command-line utility for CritterDetector.

This module provides a command-line interface for processing videos using the CritterDetector.
It allows for configuring various detection parameters and output options.

Example:
    ```
    python -m owl_highlighter.run_highlighter --video_path /path/to/video.mp4 --output_dir /path/to/output
    ```
"""

import os
import argparse
import sys
import toml
from pathlib import Path
import torch
from .highlighter import CritterDetector
from .utils import normalize_path, create_directory_structure, get_cuda_info, set_environment_variables_for_cuda, is_windows

def find_config_file():
    """
    Search for config.toml in common locations.
    """
    search_paths = [
        # Current directory
        os.path.join(os.getcwd(), "config.toml"),
        # Repository root (assuming running as module)
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.toml"),
        # User's home directory
        os.path.join(os.path.expanduser("~"), "config.toml"),
        # Package directory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.toml"),
    ]
    
    for path in search_paths:
        if os.path.isfile(path):
            return path
    
    return None

def main():
    """
    Main entry point for the run_highlighter command-line utility.
    """
    # Set environment variables for optimal CUDA performance
    set_environment_variables_for_cuda()
    
    parser = argparse.ArgumentParser(description="Run the marine organism highlighter on a video")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Directory to save results. Defaults to video directory")
    parser.add_argument("--config_path", type=str, default=None, 
                        help="Path to config.toml file. If not provided, will search in common locations")
    parser.add_argument("--frame_interval", type=int, default=5, 
                        help="Analyze every Nth frame (default: 5)")
    parser.add_argument("--no-timeline", action="store_true",
                        help="Don't save timeline visualization")
    parser.add_argument("--create_highlights", action="store_true",
                        help="Create clip files of highlight segments")
    parser.add_argument("--no-show-labels", dest="show_labels", action="store_false",
                        help="Don't show labels on timeline visualization")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--setup", action="store_true",
                        help="Create the necessary directory structure before processing")
    parser.add_argument("--cuda_info", action="store_true",
                        help="Display detailed CUDA information and exit")
    
    # Set default for show_labels
    parser.set_defaults(show_labels=True)
    
    args = parser.parse_args()
    
    # Display CUDA info if requested
    if args.cuda_info:
        cuda_info = get_cuda_info()
        print("\nCUDA Information:")
        print("=" * 40)
        for key, value in cuda_info.items():
            print(f"{key}: {value}")
        print("=" * 40)
        print("\nUse --setup to create the necessary directory structure")
        sys.exit(0)
    
    # Create directory structure if requested
    if args.setup:
        print("Creating directory structure...")
        create_directory_structure()
        print("Directory structure created. You can now place your videos in the 'videos' directory.")
        print("Run the script again without --setup to process videos.")
        sys.exit(0)
    
    # Normalize video path
    video_path = normalize_path(args.video_path)
    
    # Verify video path exists
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Find config file
    if args.config_path:
        config_path = normalize_path(args.config_path)
    else:
        config_path = find_config_file()
        
    if not config_path or not os.path.isfile(config_path):
        print(f"Warning: Config file not found. Using default settings.")
        config = None
    else:
        try:
            with open(config_path, 'r') as f:
                config = toml.load(f)
            print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default settings.")
            config = None
    
    # Set output directory if not provided
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(video_path))
    else:
        args.output_dir = normalize_path(args.output_dir)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print CUDA availability
    if args.verbose:
        cuda_info = get_cuda_info()
        print(f"CUDA available: {cuda_info['cuda_available']}")
        if cuda_info['cuda_available']:
            print(f"CUDA device count: {cuda_info['device_count']}")
            print(f"CUDA device name: {cuda_info['device_name']}")
    
    # Initialize detector
    detector = CritterDetector(config_path=config_path, config=config, show_labels=args.show_labels)
    
    # Process video
    print(f"Processing video: {video_path}")
    result = detector.process_video(
        video_path=video_path,
        output_dir=args.output_dir,
        create_highlight_clips=args.create_highlights,
        frame_interval=args.frame_interval,
        save_timeline=not args.no_timeline,
        verbose=args.verbose
    )
    
    if not args.no_timeline:
        # Create timeline visualization
        timeline_path = os.path.join(args.output_dir, Path(video_path).stem + "_timeline.jpg")
        detector.create_timeline(result, timeline_path)
        print(f"Timeline saved to: {timeline_path}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main() 