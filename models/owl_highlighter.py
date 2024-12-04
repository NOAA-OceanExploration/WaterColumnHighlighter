import os
import cv2
from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import toml
from colorama import init, Fore, Style

init()  # Initialize colorama


class YOLOHighlighter:
    def __init__(self, video_dir, output_dir, class_names, config):
        print(f"{Fore.CYAN}Initializing Highlighter...{Style.RESET_ALL}")
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.class_names = [f"a photo of a {name.strip()}" for name in class_names.split(", ")]
        self.threshold = config.get('score_thr', 0.1)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Load model and processor
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        print(f"{Fore.GREEN}✓ Highlighter initialized successfully{Style.RESET_ALL}")

    def process_videos(self):
        """Process all videos in the specified directory."""
        video_files = [f for f in os.listdir(self.video_dir) if f.endswith('.mov')]
        print(f"{Fore.CYAN}Found {len(video_files)} videos to process{Style.RESET_ALL}")
        
        for video_name in video_files:
            video_path = os.path.join(self.video_dir, video_name)
            print(f"\n{Fore.YELLOW}Processing video: {video_name}{Style.RESET_ALL}")
            self._process_video(video_path, video_name)

    def _process_video(self, video_path, video_name):
        """Process a single video."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"{Fore.CYAN}Video details:{Style.RESET_ALL}")
        print(f"  • FPS: {fps}")
        print(f"  • Total frames: {frame_count}")
        print(f"  • Duration: {frame_count / fps:.2f} seconds")

        for frame_num in range(0, frame_count, int(fps)):
            print(f"\r{Fore.CYAN}Processing frame {frame_num}/{frame_count} ({frame_num / frame_count * 100:.1f}%){Style.RESET_ALL}", end="")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert frame to PIL Image
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Run inference
            detections = self._run_inference(pil_frame)
            
            # Log detections if any found above threshold
            if detections:
                timestamp = frame_num / fps
                print(f"\n{Fore.MAGENTA}★ Highlight detected in {video_name} at frame {frame_num} ({timestamp:.1f}s){Style.RESET_ALL}")

        print(f"\n{Fore.GREEN}✓ Completed processing {video_name}{Style.RESET_ALL}")
        cap.release()

    def _run_inference(self, image):
        """Run inference on a single image."""
        inputs = self.processor(text=[self.class_names], images=image, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert outputs to Pascal VOC Format
        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=self.threshold
        )[0]

        return results["scores"].any().item()  # Return True if any detection above threshold


if __name__ == "__main__":
    print(f"{Fore.CYAN}Starting Highlighter application...{Style.RESET_ALL}")
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.toml')
    config = toml.load(config_path)

    # Define ocean organism classes
    ocean_classes = (
        "fish, shark, whale, dolphin, seal, sea lion, turtle, jellyfish, octopus, squid, "
        "crab, lobster, shrimp, starfish, sea cucumber, sea urchin, coral, anemone, "
        "manta ray, stingray, eel, barracuda, clownfish, angelfish, seahorse"
    )

    # Initialize Highlighter
    highlighter = YOLOHighlighter(
        video_dir=config['paths']['video_dir'],
        output_dir=".",
        class_names=ocean_classes,
        config={'score_thr': config['yolo'].get('score_thr', 0.1)}
    )
    highlighter.process_videos()
