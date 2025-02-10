![WaterColumnHighlighter Logo](WHC.png)

# CritterDetector

CritterDetector is an advanced deep learning framework designed to automate the detection and extraction of highlight segments—specifically, moments featuring marine organisms—from underwater video footage. By leveraging state-of-the-art object detection models like DETR (Detection Transformer) and bidirectional long short-term memory (BiLSTM) networks, the tool effectively analyzes both spatial and temporal features to classify video segments. The project is particularly beneficial for researchers, marine biologists, and underwater videographers seeking to streamline the identification of noteworthy events in extensive underwater recordings.

## Features

- **Automatic Highlight Detection:** Employs deep learning algorithms to automatically identify and extract highlight segments featuring marine organisms from underwater videos.
- **DETR Integration by Default:** Utilizes the pretrained DETR model for object detection and feature extraction, capturing complex relationships within images without additional training.
- **Temporal Dynamics Modeling:** Utilizes BiLSTM networks to capture temporal dependencies across video frames, enhancing the accuracy of highlight detection.
- **Customizable Training:** Offers extensive configuration options for data augmentation, model parameters, and training settings to adapt to various underwater datasets.
- **AWS Training Support:** Seamlessly switch between local and AWS EC2 training environments, with data loading from S3 when on AWS.
- **Model Saving and Checkpointing:** Implements model checkpointing and early stopping to prevent overfitting and enable training resumption.
- **Evaluation Tools:** Includes comprehensive evaluation scripts to assess temporal detection accuracy against ground truth annotations.

## Installation

Before running CritterDetector, ensure you have the following prerequisites installed:

- Python 3.6 or higher
- PyTorch
- torchvision
- OpenCV (cv2)
- NumPy
- toml
- wandb (Weights & Biases)
- boto3 (for AWS S3 integration)

You can install most of the required packages using pip:

```bash
pip install torch torchvision opencv-python numpy pandas scikit-learn transformers wandb toml tqdm matplotlib boto3
```

Make sure to log in to your Weights & Biases account:

```bash
wandb login
```

## Configuration

Adjust the `config.toml` file to specify paths and parameters:

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

CritterDetector includes an evaluation script to assess the temporal accuracy of detections against ground truth annotations. The evaluation focuses on temporal overlap between detected events and annotated events, rather than spatial accuracy or classification precision.

### Configuration

Configure evaluation parameters in `config.toml`:

```toml
[paths]
evaluation_output_dir = "evaluation_results"  # Where to save evaluation results
annotation_csv = "/path/to/annotations.csv"   # Path to ground truth annotations

[evaluation]
temporal_tolerance = 2.0  # Time window in seconds for matching detections
min_confidence = 0.1     # Minimum confidence threshold for detections
```

### Running Evaluation

To evaluate the detector's performance:

```bash
python -m owl_highlighter.evaluate_detections
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
