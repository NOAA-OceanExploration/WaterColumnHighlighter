from .highlighter import CritterDetector, OWLHighlighter, OwlDetector, YoloDetector, DetrDetector, EnsembleDetector
from .models import Detection, VideoProcessingResult

__version__ = "0.1.0"
__all__ = [
    "CritterDetector",
    "OWLHighlighter",  # For backward compatibility 
    "OwlDetector",
    "YoloDetector",
    "DetrDetector",
    "EnsembleDetector",
    "Detection", 
    "VideoProcessingResult"
]