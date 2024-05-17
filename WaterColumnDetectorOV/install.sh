#!/bin/bash

# Set the name of the conda environment
ENV_NAME="ocean-detection"

# Set the Python version
PYTHON_VERSION="3.9"

# Create the conda environment
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Initialize conda for bash
conda init bash

# Activate the conda environment
conda activate $ENV_NAME

# Install the required libraries
pip install fastapi uvicorn pydantic torch transformers Pillow opencv-python numpy

# Deactivate the conda environment
conda deactivate

echo "Conda environment '$ENV_NAME' created and libraries installed successfully!"