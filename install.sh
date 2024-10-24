#!/bin/bash

# Install required packages without specific versions
pip install --prefer-binary torch torchvision
pip install --prefer-binary opencv-python
pip install --prefer-binary pandas
pip install --prefer-binary numpy
pip install --prefer-binary boto3  # Added for AWS S3 integration
pip install --prefer-binary scikit-learn
pip install --prefer-binary toml
pip install --prefer-binary wandb
pip install --prefer-binary Flask
pip install --prefer-binary moviepy
pip install --prefer-binary tqdm
pip install --prefer-binary psutil
pip install --prefer-binary requests

# Install additional packages without specific versions
pip install --prefer-binary fastapi
pip install --prefer-binary jinja2
pip install --prefer-binary transformers
pip install --prefer-binary matplotlib
pip install --prefer-binary uvicorn
pip install --prefer-binary starlette
