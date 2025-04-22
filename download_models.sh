#!/bin/bash

# This script pre-downloads and caches models used by CritterDetector.
# Running this is optional, as models will typically be downloaded on first use,
# but it can be useful for setting up environments without internet access
# or ensuring specific model versions are cached.

# Ensure necessary Python packages are installed (e.g., via `pip install .`)
# before running this script.

echo "Pre-downloading and caching models..."

# Download and cache the OWLv2 model (Base)
echo "Downloading OWLv2 model (Base)..."
python -c "
from transformers import Owlv2Processor, Owlv2ForObjectDetection
processor = Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble')
model = Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble')
print('OWLv2 Base model cached.')
" || echo "Failed to cache OWLv2 Base model."

# Download and cache the OWLv2 model (Large - Optional, uncomment if needed)
# echo "Downloading OWLv2 model (Large)..."
# python -c "
# from transformers import Owlv2Processor, Owlv2ForObjectDetection
# processor = Owlv2Processor.from_pretrained('google/owlv2-large-patch14-ensemble')
# model = Owlv2ForObjectDetection.from_pretrained('google/owlv2-large-patch14-ensemble')
# print('OWLv2 Large model cached.')
# " || echo "Failed to cache OWLv2 Large model."

# Download and cache the DETR model (ResNet-50)
echo "Downloading DETR model (ResNet-50)..."
python -c "
from transformers import DetrImageProcessor, DetrForObjectDetection
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50', revision='no_timm')
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50', revision='no_timm')
print('DETR ResNet-50 (no_timm) model cached.')
" || python -c "
# Fallback if 'no_timm' revision fails
from transformers import DetrImageProcessor, DetrForObjectDetection
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
print('DETR ResNet-50 (default) model cached.')
" || echo "Failed to cache DETR ResNet-50 model."

# Pre-download YOLOv8 model (Ultralytics handles download, this just ensures it's cached)
echo "Pre-downloading YOLOv8 model (Nano)..."
python -c "
from ultralytics import YOLO
try:
    model = YOLO('yolov8n.pt')
    print('YOLOv8 Nano model cached.')
except Exception as e:
    print(f'Failed to cache YOLOv8 Nano model: {e}')
"

# Download and cache the CLIP model
echo "Downloading CLIP model (Large Patch14)..."
python -c "
from transformers import CLIPProcessor, CLIPModel
processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
print('CLIP ViT-L/14 model cached.')
" || echo "Failed to cache CLIP model."

# Download and cache the Grounding DINO model (Base)
echo "Downloading Grounding DINO model (Base)..."
python -c "
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
processor = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
model = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')
print('Grounding DINO Base model cached.')
" || echo "Failed to cache Grounding DINO Base model."

# Pre-download YOLO-World model (Ultralytics handles download, this just ensures it's cached)
echo "Pre-downloading YOLO-World model (Large v2)..."
python -c "
from ultralytics import YOLO
try:
    model = YOLO('yolov8l-worldv2.pt')
    print('YOLO-World Large v2 model cached.')
except Exception as e:
    print(f'Failed to cache YOLO-World Large v2 model: {e}')
"

echo "Model pre-download process finished."
