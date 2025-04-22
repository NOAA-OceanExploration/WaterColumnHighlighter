![WaterColumnHighlighter Logo](WHC.png)

# CritterDetector

CritterDetector is an advanced deep learning framework designed to automate the detection and extraction of highlight segments—specifically, moments featuring marine organisms—from underwater video footage. By leveraging state-of-the-art object detection models like DETR (Detection Transformer) and bidirectional long short-term memory (BiLSTM) networks, the tool effectively analyzes both spatial and temporal features to classify video segments. The project is particularly beneficial for researchers, marine biologists, and underwater videographers seeking to streamline the identification of noteworthy events in extensive underwater recordings.

## Features

Automatic Highlight Detection: Employs deep learning algorithms to automatically identify and extract highlight segments featuring marine organisms from underwater videos.
Multiple Detection Models: Supports several state-of-the-art detection models:
OWLv2: Google's Open-vocabulary detector with specialized marine organism prompts
YOLOv8: Ultralytics' fast and accurate object detection with multiple size options
DETR: Facebook's Detection Transformer for object detection (outputs general COCO labels)
CLIP: OpenAI's model used here as a patch classifier, leveraging another detector (YOLO or DETR) for initial bounding box proposals.
Grounding DINO: Open-vocabulary detector similar to OWL, uses text prompts for zero-shot detection.
YOLO-World: Fast zero-shot detector from Ultralytics, uses text prompts with YOLO architecture.
Ensemble Detection: Combines multiple models with customizable weights to leverage the strengths of each detector for improved accuracy
Model Variant Selection: Choose between different model variants:
OWLv2: base or large
YOLOv8: nano, small, medium, large, or extra large
DETR: resnet50, resnet101, or dc5
CLIP: Uses a configurable base detector (YOLO or DETR) with its own variants.
Grounding DINO: tiny, small, base, medium (check Hugging Face for latest)
YOLO-World: s, m, l, x, v2-s, v2-m, v2-l, v2-x (check Ultralytics for latest)
Customizable Labels: Define target marine organism labels for zero-shot detectors (OWL, CLIP, GroundingDINO, YOLO-World) in an external CSV file (`marine_labels.csv` by default).
Customizable Configuration: All detector options are configurable via TOML config file
Temporal Dynamics Modeling: Utilizes BiLSTM networks to capture temporal dependencies across video frames, enhancing the accuracy of highlight detection.
Model Saving and Checkpointing: Implements model checkpointing and early stopping to prevent overfitting and enable training resumption.
Evaluation Tools: Includes comprehensive evaluation scripts to assess temporal detection accuracy against ground truth annotations.

## Installation

Follow the appropriate instructions for your operating system.

### Standard Installation (Linux/macOS)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/CritterDetector.git # Replace with your actual repo URL
    cd CritterDetector
    ```

2.  **Install the package and dependencies:**
    This command uses the `setup.py` file to install the `owl_highlighter` package and all required libraries listed therein. It's recommended to use a virtual environment (like venv or conda).
    ```bash
    # Optional: Create and activate a virtual environment
    # python -m venv venv
    # source venv/bin/activate 

    pip install .
    ```
    For development purposes (allowing changes in the code to be reflected without reinstalling):
    ```bash
    pip install -e .
    ```

3.  **(Optional) Pre-download models:**
    The required models will typically be downloaded automatically on first use. However, you can pre-download and cache them using the provided script. This is useful for offline environments or ensuring specific model versions are cached.
    ```bash
    bash download_models.sh
    ```

4.  **Login to Weights & Biases (if using for logging):**
    ```bash
    wandb login
    ```

### Windows Installation (using Conda)

These instructions guide you through installing CritterDetector on Windows using the Conda package manager, which simplifies the installation of PyTorch with CUDA support.

1.  **Install Conda:**
    If you don't have Conda, download and install Miniconda (recommended) or Anaconda from [https://docs.conda.io/projects/miniconda/en/latest/](https://docs.conda.io/projects/miniconda/en/latest/) or [https://www.anaconda.com/download](https://www.anaconda.com/download). Follow the installer instructions. It's recommended to allow the installer to add Conda to your PATH environment variable or use the Anaconda Prompt/PowerShell.

2.  **Create a Conda Environment:**
    Open Anaconda Prompt (or your terminal configured for Conda) and create a new environment (e.g., named `critterdetector`) with a specific Python version (e.g., 3.9 or 3.10 recommended):
    ```bash
    conda create -n critterdetector python=3.9
    ```
    Activate the environment:
    ```bash
    conda activate critterdetector
    ```
    You should see `(critterdetector)` at the beginning of your prompt. **All subsequent commands should be run within this activated environment.**

3.  **Install PyTorch with CUDA:**
    The easiest way to install PyTorch with the correct CUDA version is using Conda. Go to the [PyTorch official website](https://pytorch.org/get-started/locally/) and select the options appropriate for your system (e.g., Stable, Windows, Conda, your CUDA version or CPU). Copy the generated `conda install` command and run it in your activated environment. It will look something like this ( **verify the command on the PyTorch website!**):
    ```bash
    # Example command - GET THE CORRECT ONE FROM PYTORCH.ORG
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 
    ```

4.  **Install Git:**
    If you don't have Git, download and install it from [https://git-scm.com/download/win](https://git-scm.com/download/win). During installation, ensure "Git Bash Here" is added to the context menu, and it's recommended to use the default settings for line endings (Checkout Windows-style, commit Unix-style).

5.  **Clone the CritterDetector Repository:**
    Navigate to the directory where you want to store the project using the `cd` command in your Anaconda Prompt (or Git Bash). Then, clone the repository:
    ```bash
    git clone https://github.com/yourusername/CritterDetector.git # Replace with your actual repo URL
    cd CritterDetector
    ```

6.  **Install CritterDetector and Dependencies:**
    Now, install the CritterDetector package and its Python dependencies using pip within your active Conda environment:
    ```bash
    pip install .
    ```
    Or for development:
    ```bash
    pip install -e .
    ```

7.  **(Optional) Pre-download models:**
    The `download_models.sh` script is a Bash script. To run it on Windows, you need a Bash environment like **Git Bash** (installed with Git) or **Windows Subsystem for Linux (WSL)**.
    *   **Using Git Bash:** Right-click inside the `CritterDetector` project folder in File Explorer and select "Git Bash Here". Then run:
        ```bash
        bash download_models.sh
        ```
    *   **Using WSL:** Open your WSL terminal, navigate to the project directory (e.g., `/mnt/c/path/to/CritterDetector`), ensure your Conda environment is activated within WSL (you might need to install Miniconda inside WSL too), and run `bash download_models.sh`.
    *   **Alternatively:** You can skip this step. The models will be downloaded automatically when the code needs them for the first time, provided you have an internet connection.

8.  **Login to Weights & Biases (if using for logging):**
    In your activated Conda environment (Anaconda Prompt or Git Bash), run:
    ```bash
    wandb login
    ```
    Follow the prompts to log in to your W&B account.

You should now have CritterDetector installed and ready to use within your `critterdetector` Conda environment on Windows. Remember to always run `conda activate critterdetector` before using the tool in a new terminal session.

## Configuration

CritterDetector uses a TOML configuration file (`config.toml`) to manage settings for paths, training, data processing, models, and evaluation.

```toml
[paths]
model_save_path = "WaterColumnHighlighter/models"
dataset_cache_dir = "WaterColumnHighlighter/models"
checkpoint_dir = "/media/patc/puck_of_destiny/critter_detector/checkpoints"
video_dir = "/home/patc/data/Example_Dive/Compressed"
csv_dir = "/media/patc/puck_of_destiny/patrick_work/Data/Annotations"
# mode = "train" # Mode is typically set via command line argument now
highlight_output_dir = "/media/patc/puck_of_destiny/critter_detector/highlights"
timeline_output_dir = "/media/patc/puck_of_destiny/critter_detector/timelines"
evaluation_output_dir = "/home/patc/data/Example_Dive/evaluation_results"
annotation_csv = "/home/patc/data/Example_Dive/Annotations/SeaTubeAnnotations_20220809T080000.000Z_20220809T200000.000Z.csv" # Ensure this path is correct
verification_output_dir = "/home/patc/data/Example_Dive/annotation_verification_frames" # New path for verify_annotations.py output

[training] # Settings specific to the LSTM trainer (models/wch_trainer.py)
window_size = 20        # Number of frames in each input sequence for LSTM model
stride = 1
batch_size = 4
num_epochs = 1
learning_rate = 0.001
k_folds = 5
early_stopping_patience = 10
gradient_clip = 1.0
checkpoint_steps = 1000
scheduler_factor = 0.5
scheduler_patience = 5
optimizer = "Adam"     # Options: "Adam", "SGD", etc.
loss_function = "FocalLoss"  # Options: "FocalLoss", "CrossEntropy", etc.
mixed_precision = true

[data] # Settings specific to the LSTM trainer (models/wch_trainer.py)
frame_rate = 29        # Frames per second to sample from videos

[augmentation] # Settings specific to the LSTM trainer (models/wch_trainer.py)
random_crop_size = 200
color_jitter_brightness = 0.1
color_jitter_contrast = 0.1
color_jitter_saturation = 0.1
color_jitter_hue = 0.1

[logging]
wandb_project = "critter_detector"
wandb_entity = "patrickallencooper"
log_interval = 10

[model] # Settings specific to the LSTM trainer (models/wch_trainer.py)
model_type = "lstm"    # Options: "lstm" or "detr"
feature_extractor = "resnet"  # Options for LSTM model: "resnet" or "detr"
fine_tune = false      # Whether to fine-tune the feature extractor or DETR model
hidden_dim = 32        # Hidden dimension size for LSTM
num_layers = 5         # Number of layers for LSTM

[detection] # Settings for the owl_highlighter inference
model = "yoloworld"  # Options: "owl", "yolo", "detr", "clip", "groundingdino", "yoloworld"
model_variant = "l"  # Options depend on model (see features list)
score_threshold = 0.25 # Detection confidence threshold
use_ensemble = false    # Whether to use ensemble detection with multiple models
ensemble_weights = {"owl": 0.7, "yoloworld": 0.3}  # Relative weights for ensemble models
labels_csv_path = "marine_labels.csv" # Path to CSV file with organism labels for zero-shot models

[clip] # Settings specific to the ClipDetector
base_detector = "yolo"             # Which detector to use for proposals ("yolo" or "detr")
base_detector_variant = "v8n"      # Variant for the base detector (e.g., "v8n", "resnet50")
base_detector_threshold = 0.05   # Confidence threshold for base detector proposals

# [owl] - These settings are potentially deprecated or handled internally by transformers
# max_num_boxes = 10    # Maximum number of boxes
# nms_thr = 0.5         # Non-Maximum Suppression threshold
# score_thr = 0.1       # Detection confidence threshold (use [detection].score_threshold)

[aws] # Settings specific to the LSTM trainer (models/wch_trainer.py)
use_aws = false       # Set to true to enable AWS training
s3_bucket_name = "your-s3-bucket-name"
s3_data_prefix = "data/"
aws_region = "us-west-2"

[evaluation]
temporal_tolerance = 300.0  # Time window in seconds for matching detections
simplified_mode = false      # Use single "organism" label instead of detailed categories
skip_organism_filter = true  # Skip positive filtering, only remove operational terms (affects evaluate_detections.py)

```

## Usage

### Training (LSTM Model)

To start training the BiLSTM model (`models/wch_trainer.py`) with your dataset:

```bash
python models/wch_trainer.py --mode train --subsample_ratio 0.1 # Add other arguments as needed
```
*   `--mode train`: Specifies training mode.
*   `--model_type lstm`: (Optional, set in config) Selects the LSTM model.
*   `--feature_extractor resnet`: (Optional, set in config) Selects the feature extractor for LSTM.
*   `--subsample_ratio 0.1`: Use 10% of the dataset for training (adjust as needed).
*   Other arguments like `--fine_tune` can be added.

Training uses settings from the `[training]`, `[data]`, `[augmentation]`, `[model]`, and `[aws]` sections of `config.toml`.

### Inference (Highlight Detection)

To run detection on a video using the configured detector (`owl_highlighter`):

```bash
python -m owl_highlighter.run_highlighter --video_path /path/to/your/video.mp4 --output_dir /path/to/output
```
*   `--video_path`: Path to the input video file.
*   `--output_dir`: (Optional) Directory to save results (timeline image). Defaults to video directory.
*   `--config_path`: (Optional) Path to `config.toml` if not in standard locations.
*   `--frame_interval`: (Optional) Analyze every Nth frame (default: 5).
*   `--no-timeline`: (Optional) Disable saving the timeline visualization.
*   `--show-labels`: (Optional, default: True) Show labels on the timeline visualization. Set `--no-show-labels` to hide them.

Inference uses settings from the `[detection]`, `[clip]` sections of `config.toml`.

### Model Checkpointing (LSTM Trainer)

During training, the `wch_trainer.py` script saves checkpoints based on the `checkpoint_steps` setting in `config.toml`. These checkpoints contain the model state, optimizer state, epoch, and global step. They are saved to subdirectories within the path specified by `checkpoint_dir` in `config.toml` (e.g., `checkpoint_dir/fold_1/`). Training can be resumed automatically from the latest checkpoint found for the current fold.

## Evaluation

The evaluation module (`owl_highlighter.evaluate_detections.py`) assesses the performance of the configured detection model against ground truth annotations.

Run evaluation with:
```bash
python -m owl_highlighter.evaluate_detections
```

This script uses settings from the `[evaluation]`, `[detection]`, and relevant path sections in `config.toml`.

It will:
1.  Process all videos found in the `video_dir` specified in the config.
2.  Compare detections generated by the configured model (`[detection]` section) with ground truth annotations from `annotation_csv`.
3.  Generate evaluation metrics based on the `temporal_tolerance` setting.
4.  Save results (plots and text report) to the `evaluation_output_dir`.

### Annotation Format for Evaluation

The evaluation script expects annotations in CSV format with specific columns (case-insensitive matching attempted):
- **Dive ID**: Identifier for the video/dive (used to match annotations to video folders/files).
- **Start Date**: Timestamp of the annotation (must be parsable into datetime).
- **Comment**: Text description (used for filtering operational annotations).
- **Taxonomy/Taxon/Taxon Path**: (Optional) Used for positive organism filtering if `skip_organism_filter` is `false`.

Example:
```csv
Dive ID,Start Date,Comment,Taxon
EX2304,2023-07-15T15:53:48.000Z,"Saw a large fish",Actinopterygii
EX2304,2023-07-15T16:00:26.264Z,"Jellyfish pulsing",Scyphozoa
2673,2022-08-09T10:15:00.000Z,"ROV arm deployed",Operational_Note
```

### Annotation Verification

A separate script (`verify_annotations.py`) helps visualize annotation accuracy by extracting frame sequences around each annotation timestamp.

Run verification with:
```bash
python verify_annotations.py
```

This script uses paths from `config.toml` (`video_dir`, `annotation_csv`, `verification_output_dir`). It reads annotations, finds the corresponding video and frame, and saves a sequence of frames (e.g., +/- 5 frames) with annotation details overlaid into the `verification_output_dir`. Each annotation gets its own subfolder.

## Contributing

Contributions to CritterDetector are welcome! If you have suggestions for improvements or encounter any issues, please open an issue or submit a pull request.

## License

MIT
