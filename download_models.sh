#!/bin/bash

# ======================================================================
# CritterDetector Model Pre-download Script
# ======================================================================
#
# This script pre-downloads and caches various detection models used by
# the CritterDetector system. Running this script is optional but helpful for:
#
# 1. Setting up environments without internet access
# 2. Ensuring specific model versions are cached before runtime
# 3. Avoiding delays during first use of each model
# 4. Testing that all model dependencies are correctly installed
#
# IMPORTANT: You must install the CritterDetector package and its dependencies
# first (using `pip install .`) before running this script.
#
# Models downloaded:
# - OWLv2: Google's Open-vocabulary detector (base variant)
# - DETR: Facebook's Detection Transformer with ResNet-50 backbone 
# - YOLOv8: Ultralytics' object detection model (nano variant)
# - CLIP: OpenAI's vision-language model for patch classification
# - Grounding DINO: Open-vocabulary detector with text prompts
# - YOLO-World: Fast zero-shot object detection
#
# ======================================================================

echo "=========================================================================================="
echo "CritterDetector - Model Pre-download Script"
echo "=========================================================================================="
echo "This script pre-downloads and caches detection models used by CritterDetector."
echo "Running this is optional, as models are automatically downloaded on first use."
echo "=========================================================================================="

# Download and cache the OWLv2 model (Base)
echo -e "\n[1/6] Downloading OWLv2 model (Base)..."
echo "    This is Google's Open-vocabulary detector with zero-shot capabilities for marine organisms."
python -c "
from transformers import Owlv2Processor, Owlv2ForObjectDetection
processor = Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble')
model = Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble')
print('OWLv2 Base model cached successfully.')
" || echo "Failed to cache OWLv2 Base model. Check your internet connection and transformers package version."

# Download and cache the OWLv2 model (Large - Optional, uncomment if needed)
# echo -e "\n[Optional] Downloading OWLv2 model (Large)..."
# echo "    This is the larger variant of Google's Open-vocabulary detector (more accurate but slower)."
# python -c "
# from transformers import Owlv2Processor, Owlv2ForObjectDetection
# processor = Owlv2Processor.from_pretrained('google/owlv2-large-patch14-ensemble')
# model = Owlv2ForObjectDetection.from_pretrained('google/owlv2-large-patch14-ensemble')
# print('OWLv2 Large model cached successfully.')
# " || echo "Failed to cache OWLv2 Large model. Check your internet connection and transformers package version."

# Download and cache the DETR model (ResNet-50)
echo -e "\n[2/6] Downloading DETR model (ResNet-50)..."
echo "    This is Facebook's Detection Transformer for general object detection."
python -c "
from transformers import DetrImageProcessor, DetrForObjectDetection
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50', revision='no_timm')
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50', revision='no_timm')
print('DETR ResNet-50 (no_timm) model cached successfully.')
" || python -c "
# Fallback if 'no_timm' revision fails
from transformers import DetrImageProcessor, DetrForObjectDetection
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
print('DETR ResNet-50 (default) model cached successfully.')
" || echo "Failed to cache DETR ResNet-50 model. Check your internet connection and transformers package version."

# Pre-download YOLOv8 model (Ultralytics handles download, this just ensures it's cached)
echo -e "\n[3/6] Pre-downloading YOLOv8 model (Nano)..."
echo "    This is Ultralytics' fast object detection model for general objects."
python -c "
from ultralytics import YOLO
try:
    model = YOLO('yolov8n.pt')
    print('YOLOv8 Nano model cached successfully.')
except Exception as e:
    print(f'Failed to cache YOLOv8 Nano model: {e}')
"

# Download and cache the CLIP model
echo -e "\n[4/6] Downloading CLIP model (Large Patch14)..."
echo "    This is OpenAI's vision-language model for patch classification."
python -c "
from transformers import CLIPProcessor, CLIPModel
processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
print('CLIP ViT-L/14 model cached successfully.')
" || echo "Failed to cache CLIP model. Check your internet connection and transformers package version."

# Download and cache the Grounding DINO model (Base)
echo -e "\n[5/6] Downloading Grounding DINO model (Base)..."
echo "    This is an open-vocabulary detector that accepts text prompts for zero-shot detection."
python -c "
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
processor = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
model = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')
print('Grounding DINO Base model cached successfully.')
" || echo "Failed to cache Grounding DINO Base model. Check your internet connection and transformers package version."

# Pre-download YOLO-World model (Ultralytics handles download, this just ensures it's cached)
echo -e "\n[6/6] Pre-downloading YOLO-World model (Large v2)..."
echo "    This is Ultralytics' YOLO-World model that combines YOLO architecture with zero-shot capabilities."
python -c "
from ultralytics import YOLO
try:
    model = YOLO('yolov8l-worldv2.pt')
    print('YOLO-World Large v2 model cached successfully.')
except Exception as e:
    print(f'Failed to cache YOLO-World Large v2 model: {e}')
"

echo -e "\n=========================================================================================="
echo "Model pre-download process completed!"
echo "If any model failed to download, you can either try again later or"
echo "let the system download it automatically when needed."
echo "=========================================================================================="
