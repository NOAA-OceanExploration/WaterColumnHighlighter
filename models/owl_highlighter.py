import os
import cv2
from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import toml
from colorama import init, Fore, Style

init()  # Initialize colorama


class OWLHighlighter:
    def __init__(self, video_dir, output_dir, class_names, config):
        print(f"{Fore.CYAN}Initializing Highlighter...{Style.RESET_ALL}")
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.class_names = [f"a photo of a {name.strip()}" for name in class_names.split(", ")]
        self.threshold = config.get('score_thr', 0.05)
        
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

        # Create video-specific output directory
        video_output_dir = os.path.join(self.output_dir, os.path.splitext(video_name)[0])
        os.makedirs(video_output_dir, exist_ok=True)

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
            
            # Save frame if detections found
            if detections:
                timestamp = frame_num / fps
                frame_filename = f"frame_{frame_num:06d}_{timestamp:.1f}s.jpg"
                frame_path = os.path.join(video_output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                print(f"\n{Fore.MAGENTA}★ Highlight detected and saved in {video_name} at frame {frame_num} ({timestamp:.1f}s){Style.RESET_ALL}")

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

    # Define comprehensive ocean organism classes
    ocean_classes = (
        # Fish
        "fish, anchovy, barracuda, bass, blenny, butterflyfish, cardinalfish, clownfish, cod, "
        "damselfish, eel, flounder, goby, grouper, grunts, halibut, herring, jackfish, lionfish, "
        "mackerel, moray eel, mullet, parrotfish, pipefish, pufferfish, rabbitfish, rays, "
        "scorpionfish, seahorse, sergeant major, snapper, sole, surgeonfish, tang, threadfin, "
        "triggerfish, tuna, wrasse, "
        
        # Sharks
        "shark, angel shark, bamboo shark, blacktip reef shark, bull shark, carpet shark, "
        "cat shark, dogfish, great white shark, hammerhead shark, leopard shark, nurse shark, "
        "reef shark, sand tiger shark, thresher shark, tiger shark, whale shark, wobbegong, "
        
        # Marine Mammals
        "whale, dolphin, porpoise, seal, sea lion, dugong, manatee, orca, pilot whale, "
        "sperm whale, humpback whale, blue whale, minke whale, right whale, beluga whale, "
        "narwhal, walrus, "
        
        # Cephalopods
        "octopus, squid, cuttlefish, nautilus, bobtail squid, giant squid, reef octopus, "
        "blue-ringed octopus, mimic octopus, dumbo octopus, vampire squid, "
        
        # Crustaceans
        "crab, lobster, shrimp, barnacle, hermit crab, spider crab, king crab, snow crab, "
        "mantis shrimp, krill, copepod, amphipod, isopod, crawfish, crayfish, "
        
        # Echinoderms
        "starfish, sea star, brittle star, basket star, sea cucumber, sea urchin, sand dollar, "
        "feather star, crinoid, "
        
        # Cnidarians
        "jellyfish, coral, sea anemone, hydroid, sea fan, sea whip, moon jellyfish, "
        "box jellyfish, lion's mane jellyfish, sea pen, fire coral, brain coral, "
        "staghorn coral, elkhorn coral, soft coral, gorgonian, "
        
        # Mollusks (non-cephalopod)
        "clam, mussel, oyster, scallop, nudibranch, sea slug, chiton, conch, cowrie, "
        "giant clam, abalone, whelk, limpet, "
        
        # Worms and Similar
        "tube worm, bristle worm, feather duster worm, christmas tree worm, flatworm, "
        "ribbon worm, peanut worm, arrow worm, "
        
        # Other Marine Life
        "sea snake, sea turtle, crocodile fish, frogfish, stonefish, sea moth, batfish, "
        "flying gurnard, remora, sea robin, pinecone fish, seamoth, "
        
        # Colonial/Compound Organisms
        "sponge, tunicate, sea squirt, salp, pyrosome, coral polyp, hydrozoan, bryozoan, "
        "zoanthid, colonial anemone"
    )

    # Initialize Highlighter
    highlighter = OWLHighlighter(
        video_dir=config['paths']['video_dir'],
        output_dir=config['paths']['highlight_output_dir'],
        class_names=ocean_classes,
        config={'score_thr': config['owl'].get('score_thr', 0.05)}
    )
    highlighter.process_videos()
