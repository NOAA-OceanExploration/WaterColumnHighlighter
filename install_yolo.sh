#!/bin/bash

# Install PyTorch first
conda install pytorch torchvision

# Install basic dependencies
pip install opencv-python pillow numpy colorama toml supervision mmengine

# Step 4: Clone YOLO-World repository
echo "Cloning YOLO-World repository..."
cd models
if [ ! -d "YOLO-World" ]; then
  git clone --recursive https://github.com/AILab-CVC/YOLO-World
else
  echo "YOLO-World repository already exists. Skipping cloning."
fi

# Step 5: Install YOLO-World package
echo "Installing YOLO-World as a package..."
cd YOLO-World
pip install torch wheel -q
pip install -e .
cd ../..

# Step 6: Download pretrained weights
echo "Downloading pretrained weights..."
mkdir -p models/pretrained_weights
wget -P models/pretrained_weights/ https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth

# Step 7: Inform the user
echo "Setup is complete."
