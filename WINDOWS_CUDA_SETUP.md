# Windows CUDA Setup Guide for CritterDetector

This guide provides detailed instructions for setting up CritterDetector on a Windows machine with CUDA support.

## System Requirements

- Windows 10 or 11 (64-bit)
- NVIDIA GPU with CUDA support
- Python 3.8 - 3.11 (3.9 or 3.10 recommended)
- [Git](https://git-scm.com/download/win) (for cloning the repository)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended for environment management)

## Step 1: Check Your NVIDIA GPU and Install CUDA Toolkit

1. **Verify your GPU supports CUDA:**
   - Press `Windows + X` and select "Device Manager"
   - Expand "Display adapters" and confirm you have an NVIDIA GPU listed

2. **Install NVIDIA GPU Drivers:**
   - Download and install the latest drivers from [NVIDIA's website](https://www.nvidia.com/Download/index.aspx)
   - Restart your computer after installation

3. **Install CUDA Toolkit:**
   - Go to [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Select your version of Windows and download the installer
   - Run the installer and follow the prompts (Express installation is recommended)
   - The installer might need to update your graphics driver

4. **Verify CUDA Installation:**
   - Open Command Prompt and type: `nvcc --version`
   - If installed correctly, it will display the CUDA version

## Step 2: Set Up Python Environment

### Using Miniconda (Recommended)

1. **Install Miniconda:**
   - Download from [Miniconda website](https://docs.conda.io/en/latest/miniconda.html)
   - During installation, check "Add Miniconda3 to the system PATH"

2. **Create a CUDA-enabled environment:**
   - Open Command Prompt or Anaconda Prompt
   - Create a new environment:
     ```
     conda create -n critterdetector python=3.9
     conda activate critterdetector
     ```

3. **Install PyTorch with CUDA:**
   - Go to [PyTorch website](https://pytorch.org/get-started/locally/)
   - Select your CUDA version (match it to your installed CUDA Toolkit)
   - Copy the conda command and run it, for example:
     ```
     conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
     ```
     (Replace 11.8 with your CUDA version)

4. **Verify PyTorch CUDA Support:**
   ```
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device available')"
   ```
   This should output `True` followed by your GPU name.

### Using pip (Alternative)

If you prefer pip instead of conda:

1. **Create and activate a virtual environment:**
   ```
   python -m venv critterdetector-env
   critterdetector-env\Scripts\activate
   ```

2. **Install PyTorch with CUDA:**
   - Go to [PyTorch website](https://pytorch.org/get-started/locally/)
   - Select pip and your CUDA version
   - Run the provided command, for example:
     ```
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
     (Replace cu118 with your CUDA version)

## Step 3: Install CritterDetector

1. **Clone the repository:**
   ```
   git clone https://github.com/yourusername/CritterDetector.git
   cd CritterDetector
   ```

2. **Install the package:**
   ```
   pip install -e .
   ```

3. **Pre-download model weights:**
   ```
   download_models.bat
   ```

## Step 4: Directory Setup

1. **Create required directories:**
   ```
   mkdir models videos checkpoints highlights timelines evaluation_results annotations verification_frames
   ```

2. **Move your videos:**
   - Place your video files in the `videos` directory

## Step 5: Run CritterDetector

1. **Process a video:**
   ```
   python -m owl_highlighter.run_highlighter --video_path videos/your_video.mp4 --output_dir highlights --verbose
   ```

2. **Verify CUDA is being used:**
   - The `--verbose` flag will display CUDA information
   - Look for output showing "CUDA available: True" and your GPU name

## Troubleshooting

### CUDA Not Available

If PyTorch doesn't detect CUDA:

1. **Verify CUDA installation:**
   ```
   nvcc --version
   ```

2. **Verify PyTorch installation has CUDA support:**
   ```
   python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
   ```

3. **Make sure CUDA versions match:**
   - The CUDA version in PyTorch should match or be compatible with your installed CUDA Toolkit
   - If not, reinstall PyTorch with the correct CUDA version

### Memory Errors

If you encounter CUDA out-of-memory errors:

1. **Reduce batch size:** Edit `config.toml` and reduce `batch_size` under `[training]`

2. **Use a smaller model variant:** Edit `config.toml` and change `model_variant` under `[detection]` to a smaller version (e.g., "v8n" instead of "v8l")

3. **Process fewer frames:** Use the `--frame_interval` option with a higher value

## Advanced Configuration

- **Custom CUDA device selection:**
  If you have multiple GPUs, you can select specific ones by setting environment variables:
  ```
  set CUDA_VISIBLE_DEVICES=0  # Use only the first GPU
  ```

- **Memory optimization:**
  Add the following to your config.toml:
  ```toml
  [cuda]
  enable_memory_efficient_attention = true
  ``` 