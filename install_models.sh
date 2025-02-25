#!/bin/bash

# Install PyTorch and core dependencies
pip install torch torchvision
pip install opencv-python==4.9.0.80
pip install pillow numpy colorama toml
pip install scipy

# Install Hugging Face transformers with various detection models
pip install transformers==4.36.2
pip install safetensors

# Install Ultralytics for YOLOv8
pip install ultralytics

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

echo "Setup is complete."
