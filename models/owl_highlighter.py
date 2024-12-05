import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import toml
from colorama import init, Fore, Style

init()  # Initialize colorama


class OWLHighlighter:
    def __init__(self, video_dir, output_dir, timeline_dir, class_names, config):
        print(f"{Fore.CYAN}Initializing Highlighter...{Style.RESET_ALL}")
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.timeline_dir = timeline_dir
        self.class_names = [f"a photo of a {name.strip()}" for name in class_names.split(", ")]
        self.threshold = config.get('score_thr', 0.05)
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.timeline_dir, exist_ok=True)

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

        # Store detections for timeline visualization
        frame_detections = []

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
            
            # Save frame and store detection info if detections found
            if detections:
                timestamp = frame_num / fps
                frame_filename = f"frame_{frame_num:06d}_{timestamp:.1f}s.jpg"
                frame_path = os.path.join(video_output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                
                # Store detection information for timeline
                frame_detections.append((frame_num, timestamp, detections))
                
                print(f"\n{Fore.MAGENTA}★ Highlight detected and saved in {video_name} at frame {frame_num} ({timestamp:.1f}s){Style.RESET_ALL}")

        # Generate timeline visualization after processing all frames
        if frame_detections:
            print(f"\n{Fore.CYAN}Generating timeline visualization...{Style.RESET_ALL}")
            self._create_timeline_visualization(video_name, frame_detections, frame_count, fps)

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

        # Return detected labels instead of just boolean
        detected_labels = []
        for score, label in zip(results["scores"], results["labels"]):
            if score > self.threshold:
                detected_labels.append(self.class_names[label])
        return detected_labels

    def _create_timeline_visualization(self, video_name, frame_detections, total_frames, fps):
        """Create a visual timeline of detections in the video."""
        # Set dimensions for the timeline visualization
        width = 2000  # Expanded width for more space
        height = 500
        padding = 40
        timeline_y = height - 100  # Adjusted to leave space for the legend
        
        # Create a blank image with a light background
        background_color = (245, 245, 250)  # Light blue-gray
        timeline_img = Image.new('RGB', (width, height), background_color)
        draw = ImageDraw.Draw(timeline_img)
        
        # Expanded taxonomic classification and color scheme
        taxonomic_groups = {
            # Chordata (vertebrates)
            'actinopterygii': {  # Ray-finned fishes
                'color': (65, 105, 225),  # Royal Blue
                'patterns': ['fish', 'anchovy', 'barracuda', 'bass', 'blenny', 'butterflyfish', 
                             'cardinalfish', 'clownfish', 'cod', 'damselfish', 'eel', 'flounder', 
                             'goby', 'grouper', 'grunts', 'halibut', 'herring', 'jackfish', 
                             'lionfish', 'mackerel', 'moray eel', 'mullet', 'parrotfish', 
                             'pipefish', 'pufferfish', 'rabbitfish', 'rays', 'scorpionfish', 
                             'seahorse', 'sergeant major', 'snapper', 'sole', 'surgeonfish', 
                             'tang', 'threadfin', 'triggerfish', 'tuna', 'wrasse']
            },
            'chondrichthyes': {  # Cartilaginous fishes
                'color': (220, 20, 60),  # Crimson
                'patterns': ['shark', 'angel shark', 'bamboo shark', 'blacktip reef shark', 
                             'bull shark', 'carpet shark', 'cat shark', 'dogfish', 
                             'great white shark', 'hammerhead shark', 'leopard shark', 
                             'nurse shark', 'reef shark', 'sand tiger shark', 'thresher shark', 
                             'tiger shark', 'whale shark', 'wobbegong']
            },
            'mammalia': {  # Marine mammals
                'color': (75, 0, 130),  # Indigo
                'patterns': ['whale', 'dolphin', 'porpoise', 'seal', 'sea lion', 'dugong', 
                             'manatee', 'orca', 'pilot whale', 'sperm whale', 'humpback whale', 
                             'blue whale', 'minke whale', 'right whale', 'beluga whale', 
                             'narwhal', 'walrus']
            },
            # Mollusca
            'cephalopoda': {  # Cephalopods
                'color': (255, 69, 0),  # Red-Orange
                'patterns': ['octopus', 'squid', 'cuttlefish', 'nautilus', 'bobtail squid', 
                             'giant squid', 'reef octopus', 'blue-ringed octopus', 
                             'mimic octopus', 'dumbo octopus', 'vampire squid']
            },
            'bivalvia': {  # Bivalves
                'color': (255, 165, 0),  # Orange
                'patterns': ['clam', 'mussel', 'oyster', 'scallop', 'giant clam']
            },
            # Cnidaria
            'anthozoa': {  # Corals and anemones
                'color': (255, 127, 80),  # Coral
                'patterns': ['coral', 'anemone', 'sea fan', 'sea whip', 'brain coral', 
                             'staghorn coral', 'elkhorn coral', 'soft coral', 'gorgonian']
            },
            'scyphozoa': {  # True jellyfish
                'color': (147, 112, 219),  # Medium Purple
                'patterns': ['jellyfish', 'moon jellyfish', 'box jellyfish', 
                             'lion\'s mane jellyfish']
            },
            # Echinodermata
            'echinodermata': {  # Echinoderms
                'color': (34, 139, 34),  # Forest Green
                'patterns': ['starfish', 'sea star', 'brittle star', 'basket star', 
                             'sea cucumber', 'sea urchin', 'sand dollar', 'feather star', 
                             'crinoid']
            },
            # Crustacea
            'crustacea': {  # Crustaceans
                'color': (210, 105, 30),  # Chocolate
                'patterns': ['crab', 'lobster', 'shrimp', 'barnacle', 'hermit crab', 
                             'spider crab', 'king crab', 'snow crab', 'mantis shrimp', 
                             'krill', 'copepod', 'amphipod', 'isopod', 'crawfish', 'crayfish']
            },
            # Other groups
            'other': {
                'color': (128, 128, 128),  # Gray
                'patterns': ['sponge', 'tunicate', 'sea squirt', 'salp', 'pyrosome', 
                             'coral polyp', 'hydrozoan', 'bryozoan', 'zoanthid', 
                             'colonial anemone']
            }
        }

        # Font setup
        try:
            title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
            label_font = ImageFont.truetype("DejaVuSans.ttf", 12)
            scientific_font = ImageFont.truetype("DejaVuSans-Oblique.ttf", 10)
        except:
            title_font = label_font = scientific_font = ImageFont.load_default()

        # Draw title
        title = f"Timeline: {os.path.splitext(video_name)[0]}"
        draw.text((padding, padding), title, fill=(50, 50, 50), font=title_font)
        
        # Draw main timeline
        timeline_color = (100, 100, 100)
        draw.line([(padding, timeline_y), (width - padding, timeline_y)], 
                 fill=timeline_color, width=3)
        
        # Calculate scaling factor
        usable_width = width - (2 * padding)
        scale = usable_width / total_frames
        
        # Draw each detection individually unless they are close in time
        previous_frame = None
        for frame_num, timestamp, detections in sorted(frame_detections):
            x = int(padding + (frame_num * scale))
            
            # Check if this detection is close to the previous one
            if previous_frame is not None and frame_num - previous_frame <= int(fps * 2):
                # Continue the current group
                continue
            
            # Draw detection markers and labels
            y_offset = timeline_y - 100
            for detection in detections:
                # Determine organism type and color
                detected_group = 'other'
                for group, info in taxonomic_groups.items():
                    if any(pattern in detection.lower() for pattern in info['patterns']):
                        detected_group = group
                        break
                color = taxonomic_groups[detected_group]['color']
                
                # Draw colored dot
                dot_radius = 3
                draw.ellipse([x - dot_radius, y_offset - dot_radius, 
                             x + dot_radius, y_offset + dot_radius], 
                            fill=color)
                
                # Draw label
                label = detection.replace('a photo of a ', '').capitalize()
                draw.text((x + 5, y_offset - 7), label, 
                         fill=color, font=label_font)
                
                y_offset -= 20
            
            previous_frame = frame_num

        # Add timestamp markers
        duration = total_frames / fps
        marker_interval = 10  # seconds
        for t in range(0, int(duration) + 1, marker_interval):
            x = int(padding + ((t * fps) * scale))
            draw.line([(x, timeline_y), (x, timeline_y + 10)], 
                     fill=timeline_color, width=2)
            draw.text((x - 15, timeline_y + 15), f"{t}s", 
                     fill=timeline_color, font=label_font)
        
        # Add legend with taxonomic information
        legend_y = height - 60
        legend_x = padding
        for taxon, info in taxonomic_groups.items():
            if taxon != 'other':
                # Draw color sample
                draw.rectangle([legend_x, legend_y, legend_x + 10, legend_y + 10], 
                             fill=info['color'])
                # Add taxonomic name
                draw.text((legend_x + 15, legend_y), taxon.capitalize(), 
                         fill=info['color'], font=label_font)
                # Add example organisms in italics
                examples = ', '.join(info['patterns'][:2])
                if info['patterns']:
                    draw.text((legend_x + 15, legend_y + 12), 
                             f"e.g., {examples}", 
                             fill=info['color'], font=scientific_font)
                legend_x += 180
                if legend_x > width - 200:  # Start new row
                    legend_x = padding
                    legend_y += 30

        # Save visualization
        output_path = os.path.join(self.timeline_dir, 
                                  f"{os.path.splitext(video_name)[0]}_timeline.png")
        timeline_img.save(output_path)
        print(f"\n{Fore.GREEN}✓ Timeline visualization saved to {output_path}{Style.RESET_ALL}")


if __name__ == "__main__":
    print(f"{Fore.CYAN}Starting Highlighter application...{Style.RESET_ALL}")
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.toml')
    config = toml.load(config_path)

    # Define comprehensive ocean organism classes
    ocean_classes = (
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
    )

    # Initialize Highlighter with timeline directory
    highlighter = OWLHighlighter(
        video_dir=config['paths']['video_dir'],
        output_dir=config['paths']['highlight_output_dir'],
        timeline_dir=config['paths']['timeline_output_dir'],
        class_names=ocean_classes,
        config={'score_thr': config['owl'].get('score_thr', 0.05)}
    )
    highlighter.process_videos()
