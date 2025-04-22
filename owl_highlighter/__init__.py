"""
CritterDetector: Marine Organism Detection for Underwater Video.

This package provides tools for detecting and highlighting marine organisms
in underwater video footage using various state-of-the-art deep learning models.
It supports multiple detection approaches:

1. OWLv2 (Google's Open-vocabulary detector)
2. YOLOv8 (Ultralytics' object detection) 
3. DETR (Facebook's Detection Transformer)
4. CLIP (OpenAI's model used as a patch classifier)
5. Grounding DINO (Open-vocabulary detector)
6. YOLO-World (Fast zero-shot detector)

These models can be used individually or combined in an ensemble.

The main entry point is the CritterDetector class, which handles model initialization,
video processing, and visualization. OWLHighlighter is maintained as an alias
for backward compatibility.

Example:
    ```python
    from owl_highlighter import CritterDetector
    
    # Initialize detector with default settings from config.toml
    detector = CritterDetector()
    
    # Process a video
    result = detector.process_video("path/to/video.mp4")
    
    # Create a timeline visualization
    detector.create_timeline(result, "timeline_output.jpg")
    ```
"""

from .highlighter import CritterDetector, OWLHighlighter, OwlDetector, YoloDetector, DetrDetector, EnsembleDetector, ClipDetector, GroundingDinoDetector, YoloWorldDetector
from .models import Detection, VideoProcessingResult

__version__ = "0.1.0"
__all__ = [
    # Main classes for end users
    "CritterDetector",        # Primary entry point
    "OWLHighlighter",         # Alias for backward compatibility
    "Detection",              # Data model for single detections
    "VideoProcessingResult",  # Data model for processing results
    
    # Individual detector implementations (for advanced usage)
    "OwlDetector",            # Google's OWLv2
    "YoloDetector",           # Ultralytics YOLOv8
    "DetrDetector",           # Facebook's DETR
    "ClipDetector",           # OpenAI's CLIP as a detector
    "GroundingDinoDetector",  # Grounding DINO
    "YoloWorldDetector",      # YOLO-World
    "EnsembleDetector"        # Combination of multiple detectors
]