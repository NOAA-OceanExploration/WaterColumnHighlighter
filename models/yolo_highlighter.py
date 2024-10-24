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

class YOLOHighlighter:
    def __init__(self, video_dir, output_dir, class_names, config):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.class_names = class_names
        self.max_num_boxes = config.get('max_num_boxes', 100)
        self.score_thr = config.get('score_thr', 0.05)
        self.nms_thr = config.get('nms_thr', 0.5)
        os.makedirs(self.output_dir, exist_ok=True)

        # Load YOLO model configuration
        self.cfg = Config.fromfile(
            "configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py"
        )
        self.cfg.work_dir = "."
        self.cfg.load_from = "pretrained_weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth"
        self.runner = Runner.from_cfg(self.cfg)
        self.runner.call_hook("before_run")
        self.runner.load_or_resume()
        self.pipeline = Compose(self.cfg.test_dataloader.dataset.pipeline)
        self.runner.pipeline = self.pipeline
        self.runner.model.eval()

    def process_videos(self):
        for video_name in os.listdir(self.video_dir):
            if video_name.endswith('.mp4'):
                video_path = os.path.join(self.video_dir, video_name)
                self.process_video(video_path, video_name)

    def process_video(self, video_path, video_name):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_num in range(0, frame_count, int(fps)):  # Process one frame per second
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert frame to PIL Image for YOLO
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Run YOLO inference
            detections = self.run_yolo_inference(pil_frame)

            # Check if any ocean organisms are detected
            if any(detections.class_id == self.class_names):
                print(f"Highlight detected in {video_name} at frame {frame_num}")

        cap.release()

    def run_yolo_inference(self, image):
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
    # Load configuration from config.toml
    config = toml.load('config.toml')

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
        output_dir="/path/to/output",
        class_names=ocean_organism_classes,
        config={
            'max_num_boxes': config['yolo'].get('max_num_boxes', 100),
            'score_thr': config['yolo'].get('score_thr', 0.05),
            'nms_thr': config['yolo'].get('nms_thr', 0.5)
        }
    )

    # Process videos to generate highlights
    highlighter.process_videos()
