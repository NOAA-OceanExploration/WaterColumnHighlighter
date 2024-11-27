import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.ops import nms
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
import supervision as sv
import toml
from colorama import init, Fore, Style
init()  # Initialize colorama

class YOLOHighlighter:
    def __init__(self, video_dir, output_dir, class_names, config):
        print(f"{Fore.CYAN}Initializing YOLOHighlighter...{Style.RESET_ALL}")
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.class_names = class_names
        self.max_num_boxes = config.get('max_num_boxes', 100)
        self.score_thr = config.get('score_thr', 0.05)
        self.nms_thr = config.get('nms_thr', 0.5)
        os.makedirs(self.output_dir, exist_ok=True)

        # Construct the relative path using os.path.join
        config_file_path = os.path.join(os.getcwd(), "YOLO-World", "configs", "pretrain", "yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py")
        self.cfg = Config.fromfile(config_file_path)
        self.cfg.work_dir = "."
        self.cfg.load_from = "pretrained_weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth"
        self.runner = Runner.from_cfg(self.cfg)
        self.runner.call_hook("before_run")
        self.runner.load_or_resume()
        self.pipeline = Compose(self.cfg.test_dataloader.dataset.pipeline)
        self.runner.pipeline = self.pipeline
        self.runner.model.eval()
        print(f"{Fore.GREEN}✓ YOLOHighlighter initialized successfully{Style.RESET_ALL}")

    def process_videos(self):
        video_files = [f for f in os.listdir(self.video_dir) if f.endswith('.mp4')]
        print(f"{Fore.CYAN}Found {len(video_files)} videos to process{Style.RESET_ALL}")
        
        for video_name in video_files:
            if video_name.endswith('.mp4'):
                video_path = os.path.join(self.video_dir, video_name)
                print(f"\n{Fore.YELLOW}Processing video: {video_name}{Style.RESET_ALL}")
                self.process_video(video_path, video_name)

    def process_video(self, video_path, video_name):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"{Fore.CYAN}Video details:{Style.RESET_ALL}")
        print(f"  • FPS: {fps}")
        print(f"  • Total frames: {frame_count}")
        print(f"  • Duration: {frame_count/fps:.2f} seconds")

        for frame_num in range(0, frame_count, int(fps)):
            print(f"\r{Fore.CYAN}Processing frame {frame_num}/{frame_count} ({frame_num/frame_count*100:.1f}%){Style.RESET_ALL}", end="")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert frame to PIL Image for YOLO
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Run YOLO inference
            detections = self.run_yolo_inference(pil_frame)

            # Update detection printing
            if detections:
                print(f"\n{Fore.GREEN}✓ Frame {frame_num}: Detected {len(detections.class_id)} objects{Style.RESET_ALL}")

            # Update highlight detection printing
            if any(detections.class_id == self.class_names):
                print(f"\n{Fore.MAGENTA}★ Highlight detected in {video_name} at frame {frame_num} ({frame_num/fps:.1f}s){Style.RESET_ALL}")

        print(f"\n{Fore.GREEN}✓ Completed processing {video_name}{Style.RESET_ALL}")
        cap.release()

    def run_yolo_inference(self, image):
        print(f"\r{Fore.CYAN}Running YOLO inference...{Style.RESET_ALL}", end="")
        data_info = self.runner.pipeline(dict(img_id=0, img_path=image, texts=[[t.strip()] for t in self.class_names.split(",")] + [[" "]]))
        data_batch = dict(
            inputs=data_info["inputs"].unsqueeze(0),
            data_samples=[data_info["data_samples"]],
        )

        with autocast(enabled=False), torch.no_grad():
            output = self.runner.model.test_step(data_batch)[0]
            self.runner.model.class_names = self.class_names
            pred_instances = output.pred_instances

        # Apply Non-Maximum Suppression (NMS)
        keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=self.nms_thr)
        pred_instances = pred_instances[keep_idxs]
        pred_instances = pred_instances[pred_instances.scores.float() > self.score_thr]

        if len(pred_instances.scores) > self.max_num_boxes:
            indices = pred_instances.scores.float().topk(self.max_num_boxes)[1]
            pred_instances = pred_instances[indices]

        # Convert predictions to numpy
        pred_instances = pred_instances.cpu().numpy()

        # Create detections object
        detections = sv.Detections(
            xyxy=pred_instances['bboxes'],
            class_id=pred_instances['labels'],
            confidence=pred_instances['scores']
        )

        return detections

if __name__ == "__main__":
    print(f"{Fore.CYAN}Starting YOLOHighlighter application...{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Loading configuration...{Style.RESET_ALL}")
    # Load configuration from config.toml located one level above
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.toml')
    print(f"Looking for config file at: {config_path}")
    config = toml.load(config_path)

    # Define the class names for ocean organisms
    ocean_organism_classes = (
        "fish, shark, whale, dolphin, seal, sea lion, turtle, jellyfish, octopus, squid, "
        "crab, lobster, shrimp, starfish, sea cucumber, sea urchin, coral, anemone, "
        "manta ray, stingray, eel, barracuda, clownfish, angelfish, seahorse, "
        "moray eel, hammerhead shark, great white shark, blue whale, humpback whale, "
        "orca, beluga whale, narwhal, walrus, manatee, dugong, "
        "sea otter, penguin, albatross, pelican, puffin, "
        "krill, plankton, barnacle, mussel, clam, oyster, "
        "seaweed, kelp, sponge, nudibranch"
    )

    # Initialize YOLOHighlighter
    highlighter = YOLOHighlighter(
        video_dir=config['paths']['video_dir'],
        output_dir=".",
        class_names=ocean_organism_classes,
        config={
            'max_num_boxes': config['yolo'].get('max_num_boxes', 100),
            'score_thr': config['yolo'].get('score_thr', 0.05),
            'nms_thr': config['yolo'].get('nms_thr', 0.5)
        }
    )

    print(f"{Fore.YELLOW}Initializing YOLOHighlighter...{Style.RESET_ALL}")
    # Process videos to generate highlights
    print(f"{Fore.CYAN}Beginning video processing...{Style.RESET_ALL}")
    highlighter.process_videos()
    print(f"\n{Fore.GREEN}✓ All videos processed successfully!{Style.RESET_ALL}")
