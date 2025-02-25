![WaterColumnHighlighter Logo](WHC.png)

# CritterDetector

CritterDetector is an advanced deep learning framework designed to automate the detection and extraction of highlight segments—specifically, moments featuring marine organisms—from underwater video footage. By leveraging state-of-the-art object detection models like DETR (Detection Transformer) and bidirectional long short-term memory (BiLSTM) networks, the tool effectively analyzes both spatial and temporal features to classify video segments. The project is particularly beneficial for researchers, marine biologists, and underwater videographers seeking to streamline the identification of noteworthy events in extensive underwater recordings.

## Features

Automatic Highlight Detection: Employs deep learning algorithms to automatically identify and extract highlight segments featuring marine organisms from underwater videos.
Multiple Detection Models: Now supports several state-of-the-art detection models:
OWL-ViT: Google's Open-vocabulary detector with specialized marine organism prompts
YOLOv8: Ultralytics' fast and accurate object detection with multiple size options
DETR: Facebook's Detection Transformer for object detection
Ensemble Detection: Combines multiple models with customizable weights to leverage the strengths of each detector for improved accuracy
Model Variant Selection: Choose between different model variants:
OWL-ViT: base or large
YOLOv8: nano, small, medium, large, or extra large
DETR: resnet50, resnet101, or dc5
Customizable Configuration: All detector options are configurable via TOML config file
Temporal Dynamics Modeling: Utilizes BiLSTM networks to capture temporal dependencies across video frames, enhancing the accuracy of highlight detection.
Model Saving and Checkpointing: Implements model checkpointing and early stopping to prevent overfitting and enable training resumption.
Evaluation Tools: Includes comprehensive evaluation scripts to assess temporal detection accuracy against ground truth annotations.

Installation
To install all required dependencies including the new model options:
```bash
./install_owl.sh
```
This script will install dependencies for all supported models and cache them locally.

```bash
pip install torch torchvision opencv-python numpy pandas scikit-learn transformers wandb toml tqdm matplotlib boto3
```

Make sure to log in to your Weights & Biases account:

```bash
wandb login
```

## Configuration

CritterDetector now uses an enhanced TOML configuration file with support for multiple detection models:

```toml
[paths]
video_dir = "/path/to/videos"
csv_dir = "/path/to/csvs"
model_save_path = "/path/to/save/model"
checkpoint_dir = "/path/to/checkpoints"
dataset_cache_dir = "/path/to/dataset_cache"
evaluation_output_dir = "evaluation_results"
annotation_csv = "/path/to/annotations.csv"

[data]
frame_rate = 1  # Frames per second to sample from videos

[training]
batch_size = 4
num_epochs = 10
learning_rate = 0.001
window_size = 10  # Number of frames in each input sequence
stride = 1  # Step size for the sliding window
hidden_dim = 256
num_layers = 2
k_folds = 5
early_stopping_patience = 3
gradient_clip = 1.0
checkpoint_steps = 1000

[augmentation]
random_crop_size = 224
color_jitter_brightness = 0.2
color_jitter_contrast = 0.2
color_jitter_saturation = 0.2
color_jitter_hue = 0.1

[logging]
log_interval = 10

[aws]
use_aws = false  # Set to true to enable AWS training
s3_bucket_name = "your-s3-bucket-name"
s3_data_prefix = "data/"
aws_region = "us-west-2"

[evaluation]
temporal_tolerance = 2.0  # Time window in seconds for matching detections
min_confidence = 0.1     # Minimum confidence threshold for detections

[detection]
model = "owl"  # Options: "owl", "yolo", "detr"
model_variant = "base"  # Options depend on model: 
                       # - for owl: "base", "large"
                       # - for yolo: "v8n", "v8s", "v8m", "v8l", "v8x"
                       # - for detr: "resnet50", "resnet101", "dc5"
score_threshold = 0.1  # Detection confidence threshold
use_ensemble = true    # Whether to use ensemble detection with multiple models
ensemble_weights = {"owl": 0.7, "yolo": 0.3}  # Relative weights for ensemble models

[owl]
max_num_boxes = 10    # Maximum number of boxes
nms_thr = 0.5         # Non-Maximum Suppression threshold
score_thr = 0.1       # Detection confidence threshold
```

## Usage

To start training the model with your dataset:

```bash
python critter_detector.py --video_dir /path/to/videos --csv_dir /path/to/csvs --mode train
```

For AWS training, ensure `use_aws` is set to `true` in the `config.toml` and provide the necessary S3 bucket details.

## Model Checkpointing

During training, the script saves checkpoints after each epoch. These checkpoints contain the model state, optimizer state, and the current epoch number. They are saved to the path specified in `config.toml` and can be used for resuming training or model evaluation.

## Evaluation

The evaluation module has been updated to work with all detection models. Run evaluation with:
```bash
python -m owl_highlighter.evaluate_detections
```

You can customize the evaluation behavior through the configuration file:

```toml
[evaluation]
temporal_tolerance = 30.0  # Time window in seconds for matching detections
simplified_mode = false      # Use single "organism" label instead of detailed categories
```

The script will:
1. Process all videos in the configured video directory
2. Compare detections with ground truth annotations
3. Generate evaluation metrics including:
   - Precision: Proportion of detections that match ground truth events
   - Recall: Proportion of ground truth events that were detected
   - F1 Score: Harmonic mean of precision and recall
4. Create visualizations in the evaluation_output_dir:
   - Metrics distribution plots
   - Per-video results
   - Summary statistics

### Annotation Format

The evaluation script expects annotations in CSV format with the following columns:
- Dive ID: Identifier for the video/dive
- Start Date: Timestamp of the annotation (ISO format)
- Taxon: Type of organism (optional, not used for temporal evaluation)

Example annotation format:
```csv
Dive ID,Start Date,Taxon
2853,2023-07-15T15:53:48.000Z,fish
2853,2023-07-15T16:00:26.264Z,jellyfish
```

### Output

The evaluation generates:
- A metrics distribution plot showing precision, recall, and F1 scores across all videos
- A detailed text report with per-video metrics and overall statistics
- Summary statistics including mean and standard deviation of key metrics

## Contributing

Contributions to CritterDetector are welcome! If you have suggestions for improvements or encounter any issues, please open an issue or submit a pull request.

## License

MIT
