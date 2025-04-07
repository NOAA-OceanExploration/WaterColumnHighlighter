#!/bin/bash

# Install PyTorch and core dependencies
pip install torch torchvision
pip install opencv-python==4.9.0.80
pip install pillow numpy colorama toml
pip install scipy

# Install Hugging Face transformers with various detection models
pip install transformers>=4.39.0
pip install safetensors

# Install Ultralytics for YOLOv8 and YOLO-World (ensure recent version)
pip install --upgrade ultralytics

# Install timm for EfficientDet
pip install timm

# Install additional dependencies
pip install "numpy<2.0"

# Download and cache the OWL model
echo "Downloading OWLv2 model..."
python -c "
from transformers import Owlv2Processor, Owlv2ForObjectDetection
processor = Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble')
model = Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble')
"

# Download and cache the DETR model
echo "Downloading DETR model..."
python -c "
from transformers import DetrImageProcessor, DetrForObjectDetection
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
"

# Download YOLOv8 model
echo "Downloading YOLOv8 model..."
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
"

# Download and cache the CLIP model
echo "Downloading CLIP model..."
python -c "
from transformers import CLIPProcessor, CLIPModel
processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
"

# Download and cache the Grounding DINO model
echo "Downloading Grounding DINO model..."
python -c "
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
processor = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
model = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')
print('Grounding DINO model loaded successfully!')
"

# Download YOLO-World model
echo "Downloading YOLO-World model..."
python -c "
from ultralytics import YOLO
model = YOLO('yolov8l-worldv2.pt') # Try the V2 large variant identifier
"

echo "Setup is complete."
