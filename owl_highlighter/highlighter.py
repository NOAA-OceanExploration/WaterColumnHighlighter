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
    """Loads labels from CSV or uses default list."""
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
    """Base class for all detectors"""
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        
    def detect(self, image: Image.Image) -> List[Dict]:
        """Run detection on an image"""
        raise NotImplementedError("Subclasses must implement detect()")

class OwlDetector(BaseDetector):
    """OWLv2 detector from Google"""
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
        """Run detection on an image using OWLv2"""
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
    """YOLOv8 detector"""
    def __init__(self, threshold: float = 0.1, variant: str = "v8n", verbose: bool = True):
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
        """Run detection on an image using YOLOv8"""
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
    Note: This detector uses models pre-trained on COCO and will output
    general object labels (e.g., 'person', 'boat') rather than specific
    marine organism labels like the OWL or CLIP detectors.
    """
    def __init__(self, threshold: float = 0.1, variant: str = "resnet50", verbose: bool = True):
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
        """Run detection on an image using DETR, returning all COCO objects above threshold."""
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
    """Ensemble of multiple detectors with weighted voting"""
    def __init__(self, detectors: Dict[str, Tuple[BaseDetector, float]]):
        """
        Initialize ensemble detector
        
        Args:
            detectors: Dictionary mapping detector name to (detector, weight) tuple
        """
        self.detectors = detectors
        
    def detect(self, image: Image.Image) -> List[Dict]:
        """Run detection using all models in ensemble"""
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
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
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
    CLIP-based detector that uses a base detector for proposals
    and CLIP for classifying patches within those proposals.
    """
    # No need for OCEAN_CLASSES here anymore

    def __init__(self, threshold: float = 0.1, simplified_mode: bool = False,
                 base_detector_type: str = "yolo",
                 base_detector_variant: str = "v8n",
                 base_detector_threshold: float = 0.05,
                 labels_csv_path: Optional[str] = None):
        """
        Initializes the ClipDetector.

        Args:
            threshold (float): Confidence threshold for CLIP classification.
            simplified_mode (bool): Whether to use simplified classes for CLIP.
            base_detector_type (str): Which detector to use for box proposals ('yolo' or 'detr').
            base_detector_variant (str): Variant for the base detector.
            base_detector_threshold (float): Confidence threshold for the base detector proposals.
            labels_csv_path (Optional[str]): Path to a CSV file containing labels.
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
        Run detection: Base detector proposes boxes, CLIP classifies patches.
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
    Grounding DINO detector (e.g., IDEA-Research/grounding-dino-base).
    Uses text prompts (classes separated by '.') for zero-shot detection.
    """
    # No need for OCEAN_CLASSES here anymore

    def __init__(self, threshold: float = 0.1, simplified_mode: bool = False, variant: str = "base",
                 labels_csv_path: Optional[str] = None):
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
        """Run detection on an image using Grounding DINO, processing classes in batches."""
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
        """Apply Non-Maximum Suppression to filter overlapping boxes."""
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
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
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
    """
    # No need for OCEAN_CLASSES here anymore

    def __init__(self, threshold: float = 0.1, simplified_mode: bool = False, variant: str = "l",
                 labels_csv_path: Optional[str] = None):
        """
        Initializes the YoloWorldDetector.

        Args:
            threshold (float): Confidence threshold for detection.
            simplified_mode (bool): Whether to use simplified classes.
            variant (str): Model variant (e.g., "v2-s", "v2-m", "v2-l", "v2-x"). Default is "l".
            labels_csv_path (Optional[str]): Path to a CSV file containing labels.
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
        """Run detection on an image using YOLO-World."""
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
