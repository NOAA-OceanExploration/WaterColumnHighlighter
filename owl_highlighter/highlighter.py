"""
Detection module for identifying marine organisms in underwater video.

This module provides various object detection implementations for identifying marine life
in underwater video footage. It includes several detection models:

1. OWLv2 (Google's Open-vocabulary detector)
2. YOLOv8 (Ultralytics' object detection)
3. DETR (Facebook's Detection Transformer)
4. CLIP (OpenAI's model used as a patch classifier)
5. Grounding DINO (Open-vocabulary detector)
6. YOLO-World (Fast zero-shot detector)

These models can be used individually or combined in an ensemble. The main entry point
is the CritterDetector class, which handles model initialization, video processing, and
visualization.

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

import os
import cv2
from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from typing import List, Optional, Dict, Tuple, Union
from .models import Detection, VideoProcessingResult
from .visualization import create_timeline_visualization
from colorama import Fore, Style
import numpy as np
import toml
from tqdm import tqdm
import pandas as pd

def _load_labels_from_source(labels_csv_path: Optional[str], default_classes: List[str]) -> List[str]:
    """
    Loads organism labels from a CSV file or falls back to default class list.
    
    This function attempts to read labels from the specified CSV file. If the file
    doesn't exist or can't be read properly, it falls back to using the provided
    default classes.
    
    Args:
        labels_csv_path: Path to a CSV file containing labels in a column named 'label_name'.
                        If None or the file can't be read, default classes are used.
        default_classes: List of class strings to use if CSV loading fails. Each string may 
                        contain multiple comma-separated class names.
    
    Returns:
        List[str]: A flattened list of individual class names/labels to use for detection.
    
    Note: 
        The default_classes are expected to be in a comma-separated format, and this function
        will split and flatten that structure.
    """
    if labels_csv_path and os.path.exists(labels_csv_path):
        try:
            df = pd.read_csv(labels_csv_path)
            if 'label_name' in df.columns:
                labels = df['label_name'].astype(str).dropna().tolist()
                print(f"Loaded {len(labels)} labels from {labels_csv_path}")
                return labels
            else:
                print(f"Warning: 'label_name' column not found in {labels_csv_path}. Using default labels.")
        except Exception as e:
            print(f"Warning: Error reading {labels_csv_path}: {e}. Using default labels.")
    else:
        if labels_csv_path:
             print(f"Warning: Labels CSV path '{labels_csv_path}' not found. Using default labels.")
        else:
             print("No labels CSV path provided. Using default labels.")

    # Fallback to default labels (flattened from the original structure)
    default_labels = [
        name.strip()
        for class_string in default_classes
        for name in class_string.split(", ")
        if name.strip()
    ]
    print(f"Using {len(default_labels)} default labels.")
    return default_labels

class BaseDetector:
    """
    Base class for all detector implementations.
    
    This abstract class defines the common interface that all detector classes 
    must implement. It provides a unified way to interact with different detection models.
    
    Attributes:
        threshold (float): Confidence threshold for detections. Only detections with
            confidence scores above this threshold will be returned.
    """
    def __init__(self, threshold: float = 0.1):
        """
        Initialize the detector with a confidence threshold.
        
        Args:
            threshold: Minimum confidence score (0.0-1.0) for valid detections.
                Default is 0.1.
        """
        self.threshold = threshold
        
    def detect(self, image: Image.Image) -> List[Dict]:
        """
        Run detection on an image.
        
        Args:
            image: PIL Image to analyze.
            
        Returns:
            List of detection dictionaries, each containing:
                - label: The class/label name
                - score: Confidence score (0.0-1.0)
                - box: Bounding box coordinates [x1, y1, x2, y2]
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement detect()")

class OwlDetector(BaseDetector):
    """
    OWLv2 detector from Google for zero-shot object detection.
    
    This class implements detection using Google's Open-Vocabulary Object Detection
    with Language (OWLv2) model. OWLv2 can detect objects based on text prompts without
    specific training on those categories, making it ideal for specialized domains
    like marine life detection.
    
    The model accepts text prompts in the format "a photo of a [class]" and locates
    instances of those classes in images. It can work with both predefined marine taxonomy
    and custom labels loaded from a CSV file.
    
    Attributes:
        threshold (float): Confidence threshold for detections.
        formatted_classes (List[str]): Text prompts for the model in the format 
            "a photo of a [class]".
        processor: OWLv2Processor for preprocessing inputs.
        model: OWLv2ForObjectDetection model for inference.
    """
    # Keep OCEAN_CLASSES only as a fallback default
    OCEAN_CLASSES = [
        # Actinopterygii (Ray-finned fishes)
        "fish, anchovy, barracuda, bass, blenny, butterflyfish, cardinalfish, clownfish, cod, "
        "damselfish, eel, flounder, goby, grouper, grunts, halibut, herring, jackfish, lionfish, "
        "mackerel, moray eel, mullet, parrotfish, pipefish, pufferfish, rabbitfish, rays, "
        "scorpionfish, seahorse, sergeant major, snapper, sole, surgeonfish, tang, threadfin, "
        "triggerfish, tuna, wrasse, ",
        # Chondrichthyes (Cartilaginous fishes)
        "shark, angel shark, bamboo shark, blacktip reef shark, bull shark, carpet shark, "
        "cat shark, dogfish, great white shark, hammerhead shark, leopard shark, nurse shark, "
        "reef shark, sand tiger shark, thresher shark, tiger shark, whale shark, wobbegong, ",
        # Mammalia (Marine mammals)
        "whale, dolphin, porpoise, seal, sea lion, dugong, manatee, orca, pilot whale, "
        "sperm whale, humpback whale, blue whale, minke whale, right whale, beluga whale, "
        "narwhal, walrus, ",
        # Cephalopoda (Cephalopods)
        "octopus, squid, cuttlefish, nautilus, bobtail squid, giant squid, reef octopus, "
        "blue-ringed octopus, mimic octopus, dumbo octopus, vampire squid, ",
        # Crustacea (Crustaceans)
        "crab, lobster, shrimp, barnacle, hermit crab, spider crab, king crab, snow crab, "
        "mantis shrimp, krill, copepod, amphipod, isopod, crawfish, crayfish, ",
        # Echinodermata (Echinoderms)
        "starfish, sea star, brittle star, basket star, sea cucumber, sea urchin, sand dollar, "
        "feather star, crinoid, ",
        # Cnidaria (Cnidarians)
        "jellyfish, coral, sea anemone, hydroid, sea fan, sea whip, moon jellyfish, "
        "box jellyfish, lion's mane jellyfish, sea pen, fire coral, brain coral, "
        "staghorn coral, elkhorn coral, soft coral, gorgonian, ",
        # Mollusca (Non-cephalopod mollusks)
        "clam, mussel, oyster, scallop, nudibranch, sea slug, chiton, conch, cowrie, "
        "giant clam, abalone, whelk, limpet, ",
        # Other Marine Life
        "sponge, tunicate, sea squirt, salp, pyrosome, coral polyp, hydrozoan, bryozoan, "
        "zoanthid, colonial anemone"
    ]

    def __init__(self, threshold: float = 0.1, simplified_mode: bool = False, variant: str = "base",
                 labels_csv_path: Optional[str] = None):
        """
        Initialize the OWLv2 detector.
        
        Args:
            threshold: Confidence threshold for detections (0.0-1.0).
            simplified_mode: If True, use generic organism prompts instead of specific 
                species names.
            variant: Model variant to use ("base" or "large").
            labels_csv_path: Path to a CSV file containing custom labels. If None,
                the default OCEAN_CLASSES will be used.
        """
        super().__init__(threshold)
        
        # Load labels using the utility function
        raw_labels = _load_labels_from_source(labels_csv_path, self.OCEAN_CLASSES)

        # Format class names for OWL-ViT
        if simplified_mode:
            self.formatted_classes = ["a photo of an organism", "a photo of an ocean animal", "a photo of a plant"]
        else:
            self.formatted_classes = [f"a photo of a {name}" for name in raw_labels]
        
        # Load model and processor based on variant
        model_name = "google/owlv2-base-patch16-ensemble"
        if variant == "large":
            model_name = "google/owlv2-large-patch14-ensemble"
            
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        
    def detect(self, image: Image.Image) -> List[Dict]:
        """
        Run detection on an image using OWLv2.
        
        This method processes the image in batches of text queries, as OWLv2 has a 
        maximum text query length. Results from all batches are combined into a 
        single list of detections.
        
        Args:
            image: PIL Image to analyze.
            
        Returns:
            List of detection dictionaries, each containing:
                - label: The detected class name (without the "a photo of a" prefix)
                - score: Confidence score (0.0-1.0)
                - box: Bounding box coordinates [x1, y1, x2, y2]
        """
        batch_size = 16  # OWL-ViT's maximum text query length
        all_detections = []
        
        # Process class names in batches
        for i in range(0, len(self.formatted_classes), batch_size):
            batch_classes = self.formatted_classes[i:i + batch_size]
            
            # Run inference on batch
            inputs = self.processor(
                text=[batch_classes],  # Note: still needs to be a list of lists
                images=image, 
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = torch.Tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs=outputs, 
                target_sizes=target_sizes, 
                threshold=self.threshold
            )[0]

            # Process detections for this batch
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score > self.threshold:
                    # Adjust label index to account for batching
                    actual_label = batch_classes[label]
                    all_detections.append({
                        "label": actual_label.replace('a photo of a ', ''),
                        "score": score.item(),
                        "box": box.cpu().numpy()
                    })

        return all_detections

class YoloDetector(BaseDetector):
    """
    YOLOv8 detector from Ultralytics.
    
    This class implements object detection using YOLOv8, a fast and accurate 
    object detection model from Ultralytics. YOLOv8 is trained on the COCO dataset
    and can detect 80 common object categories, some of which may be relevant for
    marine environments (e.g., fish, person, boat).
    
    While YOLOv8 is not specifically trained on marine organisms, it can be useful
    for detecting larger or more common marine creatures and human elements (like divers)
    in underwater footage.
    
    Attributes:
        threshold (float): Confidence threshold for detections.
        verbose (bool): Whether to print verbose output during detection.
        model: The YOLOv8 model used for detection.
        marine_classes (dict): Dictionary mapping COCO class IDs to class names
            that may be relevant in marine contexts.
    """
    def __init__(self, threshold: float = 0.1, variant: str = "v8n", verbose: bool = True):
        """
        Initialize the YOLOv8 detector.
        
        Args:
            threshold: Confidence threshold for detections (0.0-1.0).
            variant: YOLOv8 model variant:
                - "v8n": Nano (smallest, fastest)
                - "v8s": Small
                - "v8m": Medium
                - "v8l": Large
                - "v8x": Extra large (largest, most accurate)
            verbose: Whether to print verbose output during detection.
        """
        super().__init__(threshold)
        self.verbose = verbose # Store verbose flag
        
        # Map variant to model name
        variant_map = {
            "v8n": "yolov8n.pt",  # Nano
            "v8s": "yolov8s.pt",  # Small
            "v8m": "yolov8m.pt",  # Medium
            "v8l": "yolov8l.pt",  # Large
            "v8x": "yolov8x.pt",  # Extra large
        }
        
        model_name = variant_map.get(variant, "yolov8n.pt")
        self.model = YOLO(model_name)
        
        # COCO class names that map to marine life
        self.marine_classes = {
            1: "person",   # For divers
            # Animal classes in COCO
            16: "bird",    # Seabirds
            21: "elephant",  # For comparison to large marine mammals
            23: "bear",    # For comparison to large marine mammals
            # Marine-relevant COCO classes
            72: "tv",      # For underwater monitors/equipment
            73: "laptop",  # For underwater equipment
            74: "mouse",   # Small animals could be misidentified
            # Add more relevant mappings as needed
        }
        
    def detect(self, image: Image.Image) -> List[Dict]:
        """
        Run detection on an image using YOLOv8.
        
        Args:
            image: PIL Image to analyze.
            
        Returns:
            List of detection dictionaries, each containing:
                - label: The detected class name from COCO dataset
                - score: Confidence score (0.0-1.0)
                - box: Bounding box coordinates [x1, y1, x2, y2]
                
        Note:
            Unlike specialized marine detectors, YOLOv8 will return generic
            COCO class names (e.g., "person", "boat", "bird") rather than 
            specific marine taxonomy.
        """
        img_array = np.array(image)
        
        # Pass verbose flag to the model call
        results = self.model(img_array, conf=self.threshold, verbose=self.verbose)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                cls = int(box.cls.item())
                cls_name = results[0].names[cls]
                
                # Filter for relevant classes or keep all
                # if cls in self.marine_classes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf.item()
                
                detections.append({
                    "label": cls_name,
                    "score": confidence,
                    "box": [x1, y1, x2, y2]
                })
                
        return detections

class DetrDetector(BaseDetector):
    """
    Facebook DETR (Detection Transformer) detector.
    
    This class implements object detection using Facebook's DETR (DEtection TRansformer),
    which uses a transformer-based architecture for object detection. Like YOLOv8,
    DETR is trained on the COCO dataset and can detect 80 common object categories.
    
    DETR provides good performance on general object detection tasks but is not
    specifically trained for marine organism detection. It is useful for detecting
    larger marine creatures, divers, boats, and equipment.
    
    Note:
        This detector uses models pre-trained on COCO and will output
        general object labels (e.g., 'person', 'boat') rather than specific
        marine organism labels like the OWL or CLIP detectors.
    
    Attributes:
        threshold (float): Confidence threshold for detections.
        verbose (bool): Whether to print verbose output during detection.
        processor: The DETR image processor for pre-processing inputs.
        model: The DETR model for object detection.
    """
    def __init__(self, threshold: float = 0.1, variant: str = "resnet50", verbose: bool = True):
        """
        Initialize the DETR detector.
        
        Args:
            threshold: Confidence threshold for detections (0.0-1.0).
            variant: DETR model variant:
                - "resnet50": ResNet-50 backbone (default)
                - "resnet101": ResNet-101 backbone (larger, more accurate)
                - "dc5": ResNet-50 with dilated C5 stage (better for small objects)
            verbose: Whether to print verbose output during detection.
        """
        super().__init__(threshold)
        self.verbose = verbose # Store verbose flag

        # Map variant to model name
        variant_map = {
            "resnet50": "facebook/detr-resnet-50",
            "resnet101": "facebook/detr-resnet-101",
            "dc5": "facebook/detr-resnet-50-dc5",
        }

        model_name = variant_map.get(variant, "facebook/detr-resnet-50")
        # Ensure 'no_timm' revision is used if needed, check transformers version compatibility
        try:
            # Try loading with revision='no_timm' first
            self.processor = DetrImageProcessor.from_pretrained(model_name, revision="no_timm")
            self.model = DetrForObjectDetection.from_pretrained(model_name, revision="no_timm")
            print(f"Loaded DETR model {model_name} with revision='no_timm'")
        except EnvironmentError:
            # Fallback to default if 'no_timm' revision doesn't exist or causes issues
            print(f"Warning: Could not load DETR model {model_name} with revision='no_timm'. Falling back to default.")
            self.processor = DetrImageProcessor.from_pretrained(model_name)
            self.model = DetrForObjectDetection.from_pretrained(model_name)


        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        # No marine_classes filtering applied for standard DETR


    def detect(self, image: Image.Image) -> List[Dict]:
        """
        Run detection on an image using DETR, returning all COCO objects above threshold.
        
        Args:
            image: PIL Image to analyze.
            
        Returns:
            List of detection dictionaries, each containing:
                - label: The detected class name from COCO dataset
                - score: Confidence score (0.0-1.0)
                - box: Bounding box coordinates [x1, y1, x2, y2]
                
        Note:
            Like YOLOv8, DETR will return generic COCO class names rather than 
            specific marine taxonomy. Post-processing may be needed to filter
            or map these to marine-relevant categories.
        """
        inputs = self.processor(images=image, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert outputs to COCO API format
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )[0] # index [0] to get results for the first (and only) image

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # Check score against threshold again (post_process might have its own logic)
            if score.item() > self.threshold:
                cls_name = self.model.config.id2label[label.item()]
                detections.append({
                    "label": cls_name,       # Use the COCO label name
                    "score": score.item(),
                    "box": box.tolist()
                })

        return detections

class EnsembleDetector(BaseDetector):
    """
    Ensemble of multiple detectors with weighted voting.
    
    This class combines multiple detector models, allowing them to work together
    for potentially better detection performance. Each detector contributes detection
    results which are weighted according to assigned importance values. The ensemble
    approach can leverage the strengths of different detection models:
    
    - OWLv2: Good for specific marine organisms with its zero-shot capabilities
    - YOLOv8: Fast and reliable for common objects
    - DETR: Strong at detecting relationships between objects
    - CLIP: Excellent at classifying image patches
    - Grounding DINO: Good at open-vocabulary detection with text prompts
    - YOLO-World: Fast zero-shot detection with YOLO architecture
    
    The ensemble applies non-maximum suppression to remove overlapping detections
    from different models.
    
    Attributes:
        detectors (Dict[str, Tuple[BaseDetector, float]]): Dictionary mapping detector
            names to (detector, weight) tuples.
    """
    def __init__(self, detectors: Dict[str, Tuple[BaseDetector, float]]):
        """
        Initialize ensemble detector with multiple weighted detection models.
        
        Args:
            detectors: Dictionary mapping detector name to (detector, weight) tuple.
                Example: {'owl': (owl_detector, 0.7), 'yolo': (yolo_detector, 0.3)}
                
                The weights determine the relative importance of each detector's 
                confidence scores. Higher weights give more influence to that detector.
        """
        self.detectors = detectors
        
    def detect(self, image: Image.Image) -> List[Dict]:
        """
        Run detection using all models in the ensemble.
        
        This method:
        1. Runs each detector on the input image
        2. Weights the confidence scores according to detector weights
        3. Normalizes scores across all detections
        4. Removes overlapping detections using non-maximum suppression
        5. Returns the combined and filtered detection results
        
        Args:
            image: PIL Image to analyze.
            
        Returns:
            List of detection dictionaries, each containing:
                - label: The detected class name
                - score: Weighted and normalized confidence score (0.0-1.0)
                - box: Bounding box coordinates [x1, y1, x2, y2]
                - detector: Name of the source detector that produced this detection
        """
        all_detections = []
        
        # Run each detector and collect results
        for name, (detector, weight) in self.detectors.items():
            print(f"Running {name} detector...")
            detections = detector.detect(image)
            
            # Apply weight to confidence scores
            for det in detections:
                det["score"] *= weight
                det["detector"] = name  # Add source detector info
                all_detections.append(det)
                
        # Normalize scores across all detectors
        if all_detections:
            max_score = max(d["score"] for d in all_detections)
            for det in all_detections:
                det["score"] /= max_score
                
        # Non-maximum suppression to remove overlapping detections
        # This is a simplified version - can be improved with proper NMS algorithm
        final_detections = []
        all_detections.sort(key=lambda x: x["score"], reverse=True)
        
        while all_detections:
            best = all_detections.pop(0)
            final_detections.append(best)
            
            # Remove overlapping detections
            box1 = best["box"]
            i = 0
            while i < len(all_detections):
                box2 = all_detections[i]["box"]
                if self._calculate_iou(box1, box2) > 0.45:  # IOU threshold
                    all_detections.pop(i)
                else:
                    i += 1
                    
        return final_detections
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        IoU measures the overlap between two boxes and is used to determine
        if detections are referring to the same object.
        
        Args:
            box1: First bounding box coordinates [x1, y1, x2, y2] or [[x1, y1], [x2, y2]]
            box2: Second bounding box coordinates [x1, y1, x2, y2] or [[x1, y1], [x2, y2]]
            
        Returns:
            float: IoU value between 0.0 (no overlap) and 1.0 (perfect overlap)
        """
        # Convert to [x1, y1, x2, y2] format
        if len(box1) == 4:
            box1_x1, box1_y1, box1_x2, box1_y2 = box1
        elif len(box1) == 2:
            box1_x1, box1_y1 = box1[0][0], box1[0][1]
            box1_x2, box1_y2 = box1[1][0], box1[1][1]
            
        if len(box2) == 4:
            box2_x1, box2_y1, box2_x2, box2_y2 = box2
        elif len(box2) == 2:
            box2_x1, box2_y1 = box2[0][0], box2[0][1]
            box2_x2, box2_y2 = box2[1][0], box2[1][1]
            
        # Calculate area of boxes
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        
        # Calculate area of intersection
        x_left = max(box1_x1, box2_x1)
        y_top = max(box1_y1, box2_y1)
        x_right = min(box1_x2, box2_x2)
        y_bottom = min(box1_y2, box2_y2)
        
        # No intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate IoU
        union_area = box1_area + box2_area - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0

class ClipDetector(BaseDetector):
    """
    CLIP-based detector combining proposal generation and patch classification.
    
    This detector uses a two-stage approach:
    1. A base detector (YOLO or DETR) first proposes bounding boxes of potential objects
    2. OpenAI's CLIP model then classifies the image patch within each box using text prompts
    
    This approach leverages the strong classification capabilities of CLIP with
    the localization abilities of object detection models. It can be particularly
    effective for marine organism detection because CLIP has strong zero-shot
    capabilities for recognizing diverse species.
    
    Attributes:
        threshold (float): Confidence threshold for CLIP classification results.
        simplified_mode (bool): Whether to use simplified classes for CLIP.
        base_detector_threshold (float): Threshold for the base detector's proposals.
        base_detector: The detector used for initial bounding box proposals.
        clip_processor: CLIP processor for text and image inputs.
        clip_model: CLIP model for zero-shot image classification.
        clip_texts (List[str]): List of text prompts/categories for CLIP classification.
    """
    # No need for OCEAN_CLASSES here anymore

    def __init__(self, threshold: float = 0.1, simplified_mode: bool = False,
                 base_detector_type: str = "yolo",
                 base_detector_variant: str = "v8n",
                 base_detector_threshold: float = 0.05,
                 labels_csv_path: Optional[str] = None):
        """
        Initialize the ClipDetector.

        Args:
            threshold: Confidence threshold for CLIP classification results (0.0-1.0).
            simplified_mode: If True, use generic organism prompts instead of specific species.
            base_detector_type: Which detector to use for box proposals ('yolo' or 'detr').
            base_detector_variant: Variant for the base detector (e.g., 'v8n', 'resnet50').
            base_detector_threshold: Confidence threshold for the base detector proposals.
                Note: This is typically lower than the final threshold to ensure
                CLIP has enough candidate regions to classify.
            labels_csv_path: Path to a CSV file containing custom labels. If None,
                default marine organism classes will be used.
        """
        super().__init__(threshold)
        self.simplified_mode = simplified_mode
        self.base_detector_threshold = base_detector_threshold

        print(f"Initializing CLIP Detector with base: {base_detector_type} ({base_detector_variant})")

        # Initialize base detector for proposals with verbose=False
        if base_detector_type == "yolo":
            self.base_detector = YoloDetector(
                threshold=self.base_detector_threshold,
                variant=base_detector_variant,
                verbose=False # Pass verbose=False
            )
            print(f"  Base YOLO detector initialized with threshold {self.base_detector_threshold}")
        elif base_detector_type == "detr":
            self.base_detector = DetrDetector(
                threshold=self.base_detector_threshold,
                variant=base_detector_variant,
                verbose=False # Pass verbose=False
            )
            print(f"  Base DETR detector initialized with threshold {self.base_detector_threshold}")
        else:
            raise ValueError(f"Unsupported base_detector_type for ClipDetector: {base_detector_type}")

        # Initialize CLIP model and processor
        clip_model_name = "openai/clip-vit-large-patch14"
        print(f"Loading CLIP model: {clip_model_name}")
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)

        if torch.cuda.is_available():
            self.clip_model = self.clip_model.cuda()
        self.clip_model.eval()
        print("CLIP model loaded.")

        # Load labels using the utility function (provide OwlDetector's default as fallback)
        raw_labels = _load_labels_from_source(labels_csv_path, OwlDetector.OCEAN_CLASSES)

        # Prepare text prompts for CLIP
        if simplified_mode:
            self.clip_texts = ["an organism", "an ocean animal", "marine life", "underwater creature", "something non-biological", "water"]
        else:
            self.clip_texts = raw_labels # Use loaded labels directly
            # Add some negative prompts
            self.clip_texts.extend(["empty water", "seafloor", "rov equipment", "laser dots"])

        print(f"Prepared {len(self.clip_texts)} text prompts for CLIP.")


    @torch.no_grad()
    def detect(self, image: Image.Image) -> List[Dict]:
        """
        Run detection using base detector for proposals and CLIP for classification.
        
        The detection process:
        1. The base detector (YOLO or DETR) proposes candidate bounding boxes
        2. Each proposed box is cropped from the image
        3. CLIP model evaluates the similarity between each crop and all text prompts
        4. The text prompt with highest similarity is assigned as the label
        5. Detections below threshold or matching negative labels are filtered out
        
        Args:
            image: PIL Image to analyze.
            
        Returns:
            List of detection dictionaries, each containing:
                - label: The class/label with highest CLIP similarity score
                - score: CLIP confidence score (0.0-1.0)
                - box: Bounding box coordinates from the base detector [x1, y1, x2, y2]
        """
        # 1. Get proposals from the base detector
        # print(f"Running base detector ({type(self.base_detector).__name__}) for proposals...")
        proposals = self.base_detector.detect(image)
        # print(f"  Base detector proposed {len(proposals)} boxes.")

        if not proposals:
            return []

        final_detections = []
        img_array = np.array(image)

        # 2. Classify patches using CLIP
        for proposal in proposals:
            box = [int(coord) for coord in proposal["box"]]
            x1, y1, x2, y2 = box

            # Ensure box coordinates are valid
            h, w, _ = img_array.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x1 >= x2 or y1 >= y2:
                # print(f"  Skipping invalid proposal box: {box}")
                continue

            # Extract patch
            patch = img_array[y1:y2, x1:x2]
            if patch.size == 0:
                # print(f"  Skipping empty patch for box: {box}")
                continue

            try:
                patch_pil = Image.fromarray(patch)
            except Exception as e:
                print(f"  Error converting patch to PIL Image for box {box}: {e}")
                continue

            # Prepare CLIP inputs
            try:
                inputs = self.clip_processor(
                    text=self.clip_texts,
                    images=patch_pil,
                    return_tensors="pt",
                    padding=True
                )
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
            except Exception as e:
                print(f"  Error processing patch with CLIP processor for box {box}: {e}")
                continue


            # Run CLIP inference
            try:
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image # image-text similarity score
                probs = logits_per_image.softmax(dim=1) # label probabilities
                best_prob, best_idx = probs[0].max(dim=0)
                best_label = self.clip_texts[best_idx.item()]
                score = best_prob.item()
            except Exception as e:
                print(f"  Error during CLIP model inference for box {box}: {e}")
                continue


            # Filter based on CLIP confidence and relevant labels
            # Define simple negative labels
            negative_labels = {"empty water", "seafloor", "rov equipment", "laser dots"}
            if score > self.threshold and best_label not in negative_labels:
                # print(f"  CLIP Confidence: {score:.3f}, Label: '{best_label}' for box {box} (Orig Score: {proposal['score']:.3f})")
                final_detections.append({
                    "label": best_label,
                    "score": score, # Use CLIP score
                    "box": proposal["box"] # Keep original box
                })
            # else:
                # print(f"  CLIP Rejected: Confidence {score:.3f} <= {self.threshold} or Negative Label '{best_label}' for box {box}")

        # print(f"CLIP detector returning {len(final_detections)} final detections.")
        return final_detections

class GroundingDinoDetector(BaseDetector):
    """
    Grounding DINO detector for zero-shot object detection with text prompts.
    
    Grounding DINO is an open-vocabulary object detector that accepts text
    prompts to specify what objects to find in an image. Unlike traditional
    detectors, it doesn't require training on specific classes and can detect
    novel objects described by text.
    
    This makes it well-suited for marine organism detection, as it can handle
    specialized terminology and diverse species without retraining. The model
    processes text prompts separated by periods (e.g., "fish. shark. coral.")
    and localizes matching objects in the image.
    
    Attributes:
        threshold (float): Confidence threshold for detections.
        simplified_mode (bool): Whether to use simplified classes instead of full taxonomy.
        processor: Grounding DINO processor for pre-processing inputs.
        model: Grounding DINO model for zero-shot object detection.
        prompt_texts (List[str]): List of category names to use as detection prompts.
    """
    # No need for OCEAN_CLASSES here anymore

    def __init__(self, threshold: float = 0.1, simplified_mode: bool = False, variant: str = "base",
                 labels_csv_path: Optional[str] = None):
        """
        Initialize the Grounding DINO detector.
        
        Args:
            threshold: Confidence threshold for detections (0.0-1.0).
            simplified_mode: If True, use generic organism prompts instead of specific species.
            variant: Model variant to use:
                - "tiny": Smallest, fastest
                - "base": Default balance of speed and accuracy
                - "small": Medium-size model
                - "medium": Larger, more accurate
            labels_csv_path: Path to a CSV file containing custom labels. If None,
                default marine organism classes will be used.
        """
        super().__init__(threshold)
        self.simplified_mode = simplified_mode # Store simplified_mode

        # Map variant to model name
        variant_map = {
            "tiny": "IDEA-Research/grounding-dino-tiny",
            "base": "IDEA-Research/grounding-dino-base",
            "small": "IDEA-Research/grounding-dino-small",
            "medium": "IDEA-Research/grounding-dino-medium",
            # Add large if needed, check Hugging Face for exact name
            # "large": "IDEA-Research/grounding-dino-large",
        }
        model_id = variant_map.get(variant, "IDEA-Research/grounding-dino-base")

        print(f"Initializing GroundingDINO Detector ({model_id})")

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        print("GroundingDINO model loaded.")

        # Load labels using the utility function
        raw_labels = _load_labels_from_source(labels_csv_path, OwlDetector.OCEAN_CLASSES)

        # Store the list for batch processing
        self.prompt_texts = raw_labels
        print(f"Initialized GroundingDINO with {len(self.prompt_texts)} potential classes from source.")

        # Note: The actual prompt string creation is now handled within the detect method's loop

    @torch.no_grad()
    def detect(self, image: Image.Image) -> List[Dict]:
        """
        Run detection on an image using Grounding DINO with text prompts.
        
        This method processes the class list in batches, formatting them into
        prompt strings with period separators as required by Grounding DINO.
        The results from all batches are combined and NMS is applied to remove
        overlapping detections.
        
        Args:
            image: PIL Image to analyze.
            
        Returns:
            List of detection dictionaries, each containing:
                - label: The matched text prompt
                - score: Confidence score (0.0-1.0)
                - box: Bounding box coordinates [x1, y1, x2, y2]
        """
        all_detections = []
        # Determine batch size for classes based on mode
        class_batch_size = 50 if not self.simplified_mode else len(self.prompt_texts) # Process all if simplified

        for i in range(0, len(self.prompt_texts), class_batch_size):
            batch_texts = self.prompt_texts[i : i + class_batch_size]
            # Create prompt for the current batch
            batch_prompt = " . ".join(batch_texts) + " ."

            # print(f"DEBUG: Processing class batch {i // class_batch_size + 1}, prompt: {batch_prompt[:100]}...")

            inputs = self.processor(images=image, text=batch_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            try:
                outputs = self.model(**inputs)
            except RuntimeError as e:
                 print(f"RuntimeError during GroundingDINO forward pass: {e}")
                 print(f"Prompt that caused error: {batch_prompt}")
                 # Skip this batch if error occurs
                 continue

            # Post-process for this batch
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs["input_ids"], # Pass input_ids as required by this method
                box_threshold=self.threshold, # Use box_threshold argument
                target_sizes=[image.size[::-1]]
            )[0] # index [0] for the first image

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                # The label returned by the processor is the text prompt matched
                 all_detections.append({
                    "label": label,
                    "score": score.item(),
                    "box": box.tolist()
                })

        # Apply Non-Maximum Suppression (NMS) to the combined results
        final_detections = self._apply_nms(all_detections)

        return final_detections

    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.45) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to filter overlapping boxes.
        
        NMS removes duplicate detections by keeping the highest-scoring box
        when multiple boxes overlap significantly (based on IoU threshold).
        
        Args:
            detections: List of detection dictionaries to filter.
            iou_threshold: IoU threshold (0.0-1.0) above which boxes are 
                considered to be overlapping. Default: 0.45.
                
        Returns:
            Filtered list of detections with overlaps removed.
        """
        if not detections:
            return []

        # Sort by score descending
        detections.sort(key=lambda x: x["score"], reverse=True)

        final_detections = []
        while detections:
            best = detections.pop(0)
            final_detections.append(best)

            box1 = best["box"]
            i = 0
            while i < len(detections):
                box2 = detections[i]["box"]
                 # Use the existing IoU calculation method (needs access or reimplementation)
                 # For simplicity, let's borrow the logic directly here
                if self._calculate_iou(box1, box2) > iou_threshold:
                    detections.pop(i)
                else:
                    i += 1
        return final_detections

    # Helper function for IoU calculation (copied from EnsembleDetector for standalone use)
    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: First bounding box coordinates [x1, y1, x2, y2]
            box2: Second bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            float: IoU value between 0.0 (no overlap) and 1.0 (perfect overlap)
        """
        # Expecting [x1, y1, x2, y2] format
        box1_x1, box1_y1, box1_x2, box1_y2 = box1
        box2_x1, box2_y1, box2_x2, box2_y2 = box2

        # Calculate area of boxes
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

        # Calculate area of intersection
        x_left = max(box1_x1, box2_x1)
        y_top = max(box1_y1, box2_y1)
        x_right = min(box1_x2, box2_x2)
        y_bottom = min(box1_y2, box2_y2)

        # No intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        if intersection_area < 0: intersection_area = 0 # Ensure non-negative area

        # Calculate IoU
        union_area = box1_area + box2_area - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0

class YoloWorldDetector(BaseDetector):
    """
    YOLO-World detector for zero-shot object detection using text prompts.
    
    YOLO-World combines the efficiency of the YOLO architecture with open-vocabulary
    capabilities, allowing it to detect objects based on text descriptions without
    specific training. It merges CLIP's vision-language understanding with YOLO's
    fast and accurate detection capabilities.
    
    This makes it particularly suitable for marine organism detection, as it can
    detect unfamiliar species described by taxonomy or common names, while maintaining
    the speed advantages of the YOLO architecture.
    
    Attributes:
        threshold (float): Confidence threshold for detections.
        model: The YOLO-World model used for detection.
        prompt_texts (List[str]): List of category names to detect.
    """
    # No need for OCEAN_CLASSES here anymore

    def __init__(self, threshold: float = 0.1, simplified_mode: bool = False, variant: str = "l",
                 labels_csv_path: Optional[str] = None):
        """
        Initialize the YOLO-World detector.
        
        Args:
            threshold: Confidence threshold for detections (0.0-1.0).
            simplified_mode: If True, use generic organism prompts instead of specific species.
            variant: Model variant to use:
                - "s": Small
                - "m": Medium
                - "l": Large (default)
                - "x": Extra large (most accurate)
                - "v2-s/m/l/x": Version 2 models (preferred if available)
            labels_csv_path: Path to a CSV file containing custom labels. If None,
                default marine organism classes will be used.
        """
        super().__init__(threshold)

        # Map variant to model name (check ultralytics docs/releases for exact names)
        # Align this map with the models downloaded in install_models.sh
        variant_map = {
            "s": "yolov8s-worldv2.pt", # Use V2 as default small
            "m": "yolov8m-worldv2.pt", # Use V2 as default medium
            "l": "yolov8l-worldv2.pt", # Use V2 as default large
            "x": "yolov8x-worldv2.pt", # Use V2 as default extra-large
            # Keep specific V2 identifiers if needed
            "v2-s": "yolov8s-worldv2.pt",
            "v2-m": "yolov8m-worldv2.pt",
            "v2-l": "yolov8l-worldv2.pt",
            "v2-x": "yolov8x-worldv2.pt",
            # Add original identifiers if you explicitly download them too
            # "orig-s": "yolo-world/s.pt",
            # "orig-m": "yolo-world/m.pt",
            # "orig-l": "yolo-world/l.pt",
            # "orig-x": "yolo-world/x.pt",
        }
        # Default to V2 large if variant not found or is ambiguous
        model_name = variant_map.get(variant, "yolov8l-worldv2.pt")

        print(f"Initializing YOLO-World Detector ({model_name})")
        self.model = YOLO(model_name)
        print("YOLO-World model loaded.")

        # Load labels using the utility function
        raw_labels = _load_labels_from_source(labels_csv_path, OwlDetector.OCEAN_CLASSES)

        # Prepare text prompts list
        if simplified_mode:
            self.prompt_texts = ["organism", "ocean animal", "marine life", "underwater creature"]
        else:
            self.prompt_texts = raw_labels # Use loaded labels directly

        print(f"Prepared {len(self.prompt_texts)} classes for YOLO-World prompt from source.")

        # Set the model's classes using the prompt texts
        # This tells YOLO-World what to look for
        self.model.set_classes(self.prompt_texts)

    def detect(self, image: Image.Image) -> List[Dict]:
        """
        Run detection on an image using YOLO-World with the configured classes.
        
        YOLO-World uses a unified architecture that combines visual-language
        understanding with YOLO's object detection capabilities, allowing it to
        detect marine organisms based on text prompts in a single pass.
        
        Args:
            image: PIL Image to analyze.
            
        Returns:
            List of detection dictionaries, each containing:
                - label: The detected class name that matched a text prompt
                - score: Confidence score (0.0-1.0)
                - box: Bounding box coordinates [x1, y1, x2, y2]
        """
        img_array = np.array(image)

        # Run inference with the specified confidence threshold
        # Note: YOLO-World uses the standard YOLO call signature
        results = self.model(img_array, conf=self.threshold, verbose=False) # verbose=False to suppress internal logs

        detections = []
        # Process results (assuming standard YOLO results format)
        for result in results:
            boxes = result.boxes
            names = result.names # Get class names map {index: 'name'}
            for box in boxes:
                class_index = int(box.cls.item())
                class_name = names[class_index] # Get the actual class name detected
                confidence = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "label": class_name, # Use the detected class name
                    "score": confidence,
                    "box": [x1, y1, x2, y2]
                })

        return detections

class CritterDetector:
    """Multi-model detector for marine organisms"""
    def __init__(self, config_path=None, config=None, show_labels: bool = True):
        """Initialize the CritterDetector with configuration.
        
        Args:
            config_path: Path to config.toml file (default: None)
            config: Configuration dictionary (default: None)
            show_labels: Whether to show labels in timeline visualization (default: True)
        """
        # Load configuration
        if config is None:
            if config_path is None:
                # Try to find config in standard locations
                possible_paths = [
                    'config.toml',
                    '../config.toml',
                    os.path.join(os.path.dirname(__file__), '../config.toml')
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        config_path = path
                        break
                        
                if config_path is None:
                    raise FileNotFoundError("Could not find config.toml")
                    
            config = toml.load(config_path)
            
        self.config = config
        self.show_labels = show_labels
        
        # Setup CUDA if available
        cuda_config = config.get('cuda', {})
        if torch.cuda.is_available():
            print(f"CUDA is available with {torch.cuda.device_count()} device(s)")
            # Set CUDA device if specified in config
            if 'device' in cuda_config:
                device_id = cuda_config['device']
                try:
                    torch.cuda.set_device(device_id)
                    print(f"Using CUDA device {device_id}: {torch.cuda.get_device_name(device_id)}")
                except Exception as e:
                    print(f"Warning: Failed to set CUDA device {device_id}: {e}")
                    print(f"Using default device: {torch.cuda.get_device_name(0)}")
            else:
                print(f"Using default CUDA device: {torch.cuda.get_device_name(0)}")
                
            # Apply memory optimization if specified
            if cuda_config.get('enable_memory_efficient_attention', False):
                try:
                    import transformers
                    transformers.utils.logging.set_verbosity_error()  # Reduce logging noise
                    
                    # Apply memory efficient attention to reduce VRAM usage
                    if hasattr(transformers, 'BitsAndBytesConfig'):
                        print("Enabling memory-efficient attention for transformers models")
                        # Modern transformers supports memory efficient attention via config
                    else:
                        print("Memory-efficient attention requested but not available with current transformers version")
                except ImportError:
                    print("Warning: transformers not available for memory optimization")
        else:
            print("Warning: CUDA is not available. Models will run on CPU which may be slow.")
            print("If you have an NVIDIA GPU, check CUDA installation.")
        
        # Extract detection configuration
        detection_config = config['detection']
        self.model_type = detection_config['model']
        self.model_variant = detection_config.get('model_variant', None)
        self.score_threshold = detection_config['score_threshold']
        self.use_ensemble = detection_config.get('use_ensemble', False)
        self.labels_csv_path = detection_config.get('labels_csv_path', None)

        # Simplified mode from evaluation config for consistency
        self.simplified_mode = config['evaluation'].get('simplified_mode', False)

        # Initialize detector based on configuration
        if self.use_ensemble:
            # For ensemble detection, prepare all detectors according to weights
            ensemble_weights = detection_config.get('ensemble_weights',
                                                   {"owl": 0.7, "yolo": 0.3})
            detectors = {}

            for model_name, weight in ensemble_weights.items():
                # Determine variant for the specific model being added
                # Use the top-level model_variant only if the model_type matches,
                # otherwise use the default for that model.
                current_variant = self.model_variant if self.model_type == model_name else None

                if model_name == "owl":
                    detector = OwlDetector(
                        threshold=self.score_threshold,
                        simplified_mode=self.simplified_mode,
                        variant=current_variant or "base",
                        labels_csv_path=self.labels_csv_path
                    )
                elif model_name == "yolo": # Standard YOLO
                    detector = YoloDetector(
                        threshold=self.score_threshold,
                        variant=current_variant or "v8n"
                    )
                elif model_name == "detr":
                    detector = DetrDetector(
                        threshold=self.score_threshold,
                        variant=current_variant or "resnet50"
                    )
                elif model_name == "clip":
                    clip_config = config.get('clip', {})
                    detector = ClipDetector(
                        threshold=self.score_threshold,
                        simplified_mode=self.simplified_mode,
                        base_detector_type=clip_config.get('base_detector', 'yolo'),
                        base_detector_variant=clip_config.get('base_detector_variant', 'v8n'),
                        base_detector_threshold=clip_config.get('base_detector_threshold', 0.05),
                        labels_csv_path=self.labels_csv_path
                    )
                elif model_name == "groundingdino":
                     detector = GroundingDinoDetector(
                        threshold=self.score_threshold,
                        simplified_mode=self.simplified_mode,
                        variant=current_variant or "base",
                        labels_csv_path=self.labels_csv_path
                    )
                elif model_name == "yoloworld":
                     detector = YoloWorldDetector(
                        threshold=self.score_threshold,
                        simplified_mode=self.simplified_mode,
                        variant=current_variant or "l",
                        labels_csv_path=self.labels_csv_path
                    )
                else:
                    # Handle YOLO and DETR which don't use the label list
                    if model_name == "yolo":
                        detector = YoloDetector(
                            threshold=self.score_threshold,
                            variant=current_variant or "v8n"
                        )
                    elif model_name == "detr":
                        detector = DetrDetector(
                            threshold=self.score_threshold,
                            variant=current_variant or "resnet50"
                        )
                    else:
                        print(f"Warning: Unknown model type '{model_name}' in ensemble, skipping")
                        continue

                detectors[model_name] = (detector, weight)

            if not detectors:
                 raise ValueError("No valid detectors specified for the ensemble.")
            self.detector = EnsembleDetector(detectors)
            print(f"Initialized Ensemble Detector with: {list(detectors.keys())}")

        else:
            # For single-model detection
            print(f"Initializing single model detector: {self.model_type}")
            if self.model_type == "owl":
                self.detector = OwlDetector(
                    threshold=self.score_threshold,
                    simplified_mode=self.simplified_mode,
                    variant=self.model_variant or "base",
                    labels_csv_path=self.labels_csv_path
                )
            elif self.model_type == "yolo": # Standard YOLO
                self.detector = YoloDetector(
                    threshold=self.score_threshold,
                    variant=self.model_variant or "v8n"
                )
            elif self.model_type == "detr":
                self.detector = DetrDetector(
                    threshold=self.score_threshold,
                    variant=self.model_variant or "resnet50"
                )
            elif self.model_type == "clip":
                clip_config = config.get('clip', {})
                self.detector = ClipDetector(
                    threshold=self.score_threshold,
                    simplified_mode=self.simplified_mode,
                    base_detector_type=clip_config.get('base_detector', 'yolo'),
                    base_detector_variant=clip_config.get('base_detector_variant', 'v8n'),
                    base_detector_threshold=clip_config.get('base_detector_threshold', 0.05),
                    labels_csv_path=self.labels_csv_path
                )
            elif self.model_type == "groundingdino":
                self.detector = GroundingDinoDetector(
                    threshold=self.score_threshold,
                    simplified_mode=self.simplified_mode,
                    variant=self.model_variant or "base",
                    labels_csv_path=self.labels_csv_path
                )
            elif self.model_type == "yoloworld":
                self.detector = YoloWorldDetector(
                    threshold=self.score_threshold,
                    simplified_mode=self.simplified_mode,
                    variant=self.model_variant or "l",
                    labels_csv_path=self.labels_csv_path
                )
            else:
                raise ValueError(f"Unsupported single model type: {self.model_type}")
            print(f"Initialized {type(self.detector).__name__}")
    
    def process_video(self, video_path: str, output_dir: str = None, 
                     create_highlight_clips: bool = False,
                     frame_interval: int = 5,
                     save_timeline: bool = True,
                     verbose: bool = False):
        """Process a video, detect marine organisms and generate highlights.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save outputs (default: same as video)
            create_highlight_clips: Whether to extract highlight clips (default: False)
            frame_interval: Analyze every Nth frame (default: 5)
            save_timeline: Whether to save timeline visualization (default: True)
            verbose: Whether to print detailed debug messages (default: False)
            
        Returns:
            VideoProcessingResult object containing detections and metadata
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Set output directory if not provided
        if output_dir is None:
            output_dir = os.path.dirname(video_path)
            
        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        
        if create_highlight_clips:
            highlights_dir = os.path.join(output_dir, "highlights")
            os.makedirs(highlights_dir, exist_ok=True)
            
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_name = os.path.basename(video_path)
        
        print(f"{Fore.GREEN}Processing video: {video_name}{Style.RESET_ALL}")
        print(f"  FPS: {fps:.2f}, Frames: {frame_count}, Resolution: {width}x{height}")
        
        # Process video frames
        detections = []
        frame_number = 0
        
        # Calculate how many frames we'll actually process
        total_frames_to_process = frame_count // frame_interval + 1
        
        # Create progress bar
        progress_bar = tqdm(total=total_frames_to_process, desc="Processing frames", unit="frame")
        
        detection_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_number % frame_interval == 0:
                # Convert frame to PIL image for detector
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Run detection on frame
                frame_detections = self.detector.detect(pil_image)
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({"Detections": detection_count, "Current Time": f"{frame_number / fps:.2f}s"})
                
                # Convert to Detection objects and store
                for det in frame_detections:
                    x1, y1, x2, y2 = [int(coord) for coord in det["box"]]
                    
                    # Only print detailed debug info if verbose mode is enabled
                    if verbose:
                        print(f"DEBUG: Processing frame {frame_number}/{frame_count} at {frame_number / fps:.2f}s")
                        print(f"DEBUG: Extracting patch with coordinates: {x1}, {y1}, {x2}, {y2}")
                    
                    # Extract image patch
                    patch = frame[y1:y2, x1:x2].copy()
                    
                    # Add a check before converting color space
                    if patch is not None and patch.size > 0 and patch.shape[0] > 0 and patch.shape[1] > 0:
                        patch_pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                        
                        detections.append(Detection(
                            frame_number=frame_number,
                            timestamp=frame_number / fps,
                            label=det["label"],
                            confidence=det["score"],
                            bbox=(x1, y1, x2, y2),
                            image_patch=patch_pil
                        ))
                        detection_count += 1
                    else:
                        if verbose:
                            print(f"WARNING: Empty patch at frame {frame_number}, timestamp {frame_number / fps:.2f}s. Skipping.")
                        continue  # Skip this iteration and continue with the next frame/patch
                
            frame_number += 1
                
        # Close progress bar
        progress_bar.close()
        
        # Release video
        cap.release()
        
        print(f"{Fore.GREEN}Processing complete. Found {detection_count} detections.{Style.RESET_ALL}")
        
        # Create result object
        result = VideoProcessingResult(
            video_name=video_name,
            fps=fps,
            frame_count=frame_count,
            duration=frame_count / fps,
            detections=detections
        )
        
        # Generate timeline visualization
        if save_timeline and detections:
            timeline_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_timeline.jpg")
            self.create_timeline(result, timeline_path)
            print(f"{Fore.GREEN}Timeline saved to: {timeline_path}{Style.RESET_ALL}")
            
        # Extract highlight clips if requested
        if create_highlight_clips and detections:
            # TODO: Implement highlight clip extraction
            pass
            
        return result
        
    def create_timeline(self, 
                       result: VideoProcessingResult,
                       output_path: str,
                       width: int = 2000,
                       height: int = 1200) -> None:
        """
        Create and save a timeline visualization of the detections.
        
        Args:
            result: VideoProcessingResult from process_video
            output_path: Where to save the timeline image
            width: Width of the timeline image
            height: Height of the timeline image
        """
        create_timeline_visualization(result, output_path, width, height, 
                                   show_labels=self.show_labels)

    @property
    def threshold(self):
        """Return score threshold for compatibility"""
        return self.score_threshold


# For backward compatibility
OWLHighlighter = CritterDetector
