#!/bin/bash

# Install core Python packages
pip install torch torchvision
pip install opencv-python
pip install pandas
pip install scikit-learn
pip install toml
pip install wandb
pip install tqdm
pip install matplotlib
pip install boto3  # For AWS S3 integration

# Install additional packages for YOLO-World and video processing
pip install mmengine
pip install supervision
pip install pillow  # For image processing
pip install mmcv  # For YOLO-World model support
pip install mmyolo  # For YOLO-World model support

# Install any other necessary packages
pip install fastapi
pip install jinja2
pip install uvicorn
pip install starlette

pip install "numpy<2" 

cd models
git clone --recursive https://github.com/AILab-CVC/YOLO-World
mkdir -p pretrained_weights
cd pretrained_weights
wget -P pretrained_weights/ https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth
cd ..