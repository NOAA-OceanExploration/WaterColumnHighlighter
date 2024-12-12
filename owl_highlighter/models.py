from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from PIL import Image

@dataclass
class Detection:
    frame_number: int
    timestamp: float
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    image_patch: Image.Image

@dataclass
class VideoProcessingResult:
    video_name: str
    fps: float
    frame_count: int
    duration: float
    detections: List[Detection]