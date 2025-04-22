"""
Data models for the detection system.

This module defines the core data structures used throughout the CritterDetector system.
It includes classes for individual detections and the overall result of processing a video.

These classes are used to standardize and organize the detection results,
making them easier to work with and visualize.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from PIL import Image

@dataclass
class Detection:
    """
    Represents a single detected object in a video frame.
    
    This class stores information about a detected marine organism or object,
    including its location in the frame, timestamp, label, confidence score,
    and an image patch showing the detected object.
    
    Attributes:
        frame_number: The frame number in the video where the detection occurred.
        timestamp: The time in seconds from the start of the video.
        label: The class name or description of the detected object.
        confidence: Confidence score between 0.0 and 1.0.
        bbox: Bounding box coordinates (x1, y1, x2, y2) in pixels.
        image_patch: A PIL Image containing just the detected object.
    """
    frame_number: int
    timestamp: float
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    image_patch: Image.Image

@dataclass
class VideoProcessingResult:
    """
    Contains the results of processing a video for organism detection.
    
    This class aggregates all detections found in a video along with
    video metadata needed for visualization and further processing.
    
    Attributes:
        video_name: Name of the processed video file.
        fps: Frames per second of the video.
        frame_count: Total number of frames in the video.
        duration: Duration of the video in seconds.
        detections: List of Detection objects found in the video.
    """
    video_name: str
    fps: float
    frame_count: int
    duration: float
    detections: List[Detection]