import os
import cv2
from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from typing import List, Optional, Dict
from .models import Detection, VideoProcessingResult
from .visualization import create_timeline_visualization
from colorama import Fore, Style

class OWLHighlighter:
    # Define comprehensive ocean classes as a class attribute
    OCEAN_CLASSES = [
        # Actinopterygii (Ray-finned fishes)
        "fish, anchovy, barracuda, bass, blenny, butterflyfish, cardinalfish, clownfish, cod, "
        "damselfish, eel, flounder, goby, grouper, grunts, halibut, herring, jackfish, lionfish, "
        "mackerel, moray eel, mullet, parrotfish, pipefish, pufferfish, rabbitfish, rays, "
        "scorpionfish, seahorse, sergeant major, snapper, sole, surgeonfish, tang, threadfin, "
        "triggerfish, tuna, wrasse, "

        # Chondrichthyes (Cartilaginous fishes)
        "shark, angel shark, bamboo shark, blacktip reef shark, bull shark, carpet shark, "
        "cat shark, dogfish, great white shark, hammerhead shark, leopard shark, nurse shark, "
        "reef shark, sand tiger shark, thresher shark, tiger shark, whale shark, wobbegong, "

        # Mammalia (Marine mammals)
        "whale, dolphin, porpoise, seal, sea lion, dugong, manatee, orca, pilot whale, "
        "sperm whale, humpback whale, blue whale, minke whale, right whale, beluga whale, "
        "narwhal, walrus, "

        # Cephalopoda (Cephalopods)
        "octopus, squid, cuttlefish, nautilus, bobtail squid, giant squid, reef octopus, "
        "blue-ringed octopus, mimic octopus, dumbo octopus, vampire squid, "

        # Crustacea (Crustaceans)
        "crab, lobster, shrimp, barnacle, hermit crab, spider crab, king crab, snow crab, "
        "mantis shrimp, krill, copepod, amphipod, isopod, crawfish, crayfish, "

        # Echinodermata (Echinoderms)
        "starfish, sea star, brittle star, basket star, sea cucumber, sea urchin, sand dollar, "
        "feather star, crinoid, "

        # Cnidaria (Cnidarians)
        "jellyfish, coral, sea anemone, hydroid, sea fan, sea whip, moon jellyfish, "
        "box jellyfish, lion's mane jellyfish, sea pen, fire coral, brain coral, "
        "staghorn coral, elkhorn coral, soft coral, gorgonian, "

        # Mollusca (Non-cephalopod mollusks)
        "clam, mussel, oyster, scallop, nudibranch, sea slug, chiton, conch, cowrie, "
        "giant clam, abalone, whelk, limpet, "

        # Annelida (Worms and similar)
        "tube worm, bristle worm, feather duster worm, christmas tree worm, flatworm, "
        "ribbon worm, peanut worm, arrow worm, "

        # Reptilia (Marine reptiles)
        "sea snake, sea turtle, "

        # Other Marine Life
        "crocodile fish, frogfish, stonefish, sea moth, batfish, flying gurnard, remora, "
        "sea robin, pinecone fish, seamoth, "

        # Colonial/Compound Organisms
        "sponge, tunicate, sea squirt, salp, pyrosome, coral polyp, hydrozoan, bryozoan, "
        "zoanthid, colonial anemone"
    ]

    def __init__(self, model_name: str = "google/owlv2-base-patch16-ensemble", 
                 score_threshold: float = 0.1):
        """Initialize the OWL Highlighter.
        
        Args:
            model_name: Name of the OWL model to use
            score_threshold: Confidence threshold for detections (default: 0.1)
        """
        # Ensure the threshold is set from the parameter
        self.threshold = float(score_threshold)  # Convert to float to ensure proper type
        
        # Split OCEAN_CLASSES into a list and format them
        self.formatted_classes = [
            f"a photo of a {name.strip()}" 
            for class_string in self.OCEAN_CLASSES 
            for name in class_string.split(", ")
            if name.strip()
        ]
        
        # Load model and processor
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

    def process_video(self, 
                     video_path: str,
                     sample_rate: Optional[int] = None) -> VideoProcessingResult:
        """
        Process a video file and return detections.
        
        Args:
            video_path: Path to the video file
            sample_rate: Optional frames to skip (e.g., 30 for 1fps at 30fps video)
        """
        print(f"\n{Fore.CYAN}Processing video: {os.path.basename(video_path)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Using detection threshold: {self.threshold}{Style.RESET_ALL}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if sample_rate is None:
            sample_rate = int(fps)  # Default to 1 frame per second
        
        # Print video details
        print(f"{Fore.CYAN}Video details:{Style.RESET_ALL}")
        print(f"  • FPS: {fps}")
        print(f"  • Total frames: {frame_count}")
        print(f"  • Duration: {frame_count / fps:.2f} seconds")
        print(f"  • Sampling every {sample_rate} frames ({fps/sample_rate:.1f} fps)")
        
        video_name = os.path.basename(video_path)
        detections = []
        frames_processed = 0

        for frame_num in range(0, frame_count, sample_rate):
            # Update progress
            progress = (frame_num / frame_count) * 100
            print(f"\r{Fore.CYAN}Processing: [{frame_num}/{frame_count}] {progress:.1f}% complete{Style.RESET_ALL}", 
                  end="", flush=True)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                print(f"\n{Fore.RED}Warning: Failed to read frame {frame_num}{Style.RESET_ALL}")
                continue

            # Convert frame to PIL Image
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Run inference
            frame_detections = self._run_inference(pil_frame)
            frames_processed += 1
            
            # Process detections
            timestamp = frame_num / fps
            for det in frame_detections:
                bbox = det["box"]
                # Print detection details
                print(f"\n{Fore.MAGENTA}★ Detection at {timestamp:.1f}s: {det['label']} ({det['score']:.2f}){Style.RESET_ALL}")
                
                # Crop the detected object
                image_patch = pil_frame.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                
                detection = Detection(
                    frame_number=frame_num,
                    timestamp=timestamp,
                    label=det["label"].replace('a photo of a ', ''),
                    confidence=det["score"],
                    bbox=tuple(bbox),
                    image_patch=image_patch
                )
                detections.append(detection)

        cap.release()
        
        # Print summary
        print(f"\n{Fore.GREEN}✓ Processing complete:{Style.RESET_ALL}")
        print(f"  • Processed {frames_processed} frames")
        print(f"  • Found {len(detections)} detections")
        
        return VideoProcessingResult(
            video_name=video_name,
            fps=fps,
            frame_count=frame_count,
            duration=frame_count / fps,
            detections=detections
        )

    def _run_inference(self, image):
        """Run inference on a single image."""
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
                        "label": actual_label,
                        "score": score.item(),
                        "box": box.cpu().numpy()
                    })

        return all_detections

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
        create_timeline_visualization(result, output_path, width, height)