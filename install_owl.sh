#!/bin/bash

# Install PyTorch and core dependencies
pip install torch torchvision
pip install opencv-python==4.9.0.80
pip install pillow numpy colorama toml

# Install Hugging Face transformers with OWLv2 model
pip install transformers==4.36.2
pip install safetensors

pip install scipy

# Download and cache the model
echo "Downloading OWLv2 model..."
python -c "
from transformers import Owlv2Processor, Owlv2ForObjectDetection
processor = Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble')
model = Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble')
"

conda install "numpy<2.0"

echo "Setup is complete."
