# Upgrade pip
pip install --upgrade pip

# Install core ML dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8 support
pip install transformers
pip install wandb
pip install toml
pip install boto3
pip install tqdm
pip install scikit-learn

# Install image processing dependencies
pip install opencv-python-headless
pip install pillow

# Install data manipulation dependencies
pip install numpy
pip install pandas