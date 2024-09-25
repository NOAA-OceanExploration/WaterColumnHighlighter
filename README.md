![WaterColumnHighlighter Logo](WHC.png)

# CritterDetector

CritterDetector is an advanced deep learning framework designed to automate the detection and extraction of highlight segments—specifically, moments featuring marine organisms—from underwater video footage. By leveraging state-of-the-art object detection models like DETR (Detection Transformer) and bidirectional long short-term memory (BiLSTM) networks, the tool effectively analyzes both spatial and temporal features to classify video segments. The project is particularly beneficial for researchers, marine biologists, and underwater videographers seeking to streamline the identification of noteworthy events in extensive underwater recordings.

## Features

- **Automatic Highlight Detection:** Employs deep learning algorithms to automatically identify and extract highlight segments featuring marine organisms from underwater videos.
- **DETR Integration by Default:** Utilizes the pretrained DETR model for object detection and feature extraction, capturing complex relationships within images without additional training.
- **Temporal Dynamics Modeling:** Utilizes BiLSTM networks to capture temporal dependencies across video frames, enhancing the accuracy of highlight detection.
- **Customizable Training:** Offers extensive configuration options for data augmentation, model parameters, and training settings to adapt to various underwater datasets.
- **Model Saving and Checkpointing:** Implements model checkpointing and early stopping to prevent overfitting and enable training resumption.

## Installation

Before running WaterColumnHighlighter, ensure you have the following prerequisites installed:

- Python 3.6 or higher
- PyTorch
- torchvision
- OpenCV (cv2)
- NumPy
- toml
- wandb (Weights & Biases)

You can install most of the required packages using pip:

```bash
pip install torch torchvision opencv-python numpy pandas scikit-learn transformers wandb toml tqdm matplotlib
```

Make sure to log in to your Weights & Biases account:

```bash
wandb login
```

## Configuration

Adjust the `config.toml` file to specify paths and training parameters:

```toml
[paths]
video_dir = "/path/to/videos"
csv_dir = "/path/to/csvs"
model_save_path = "/path/to/save/model"
checkpoint_dir = "/path/to/checkpoints"
dataset_cache_dir = "/path/to/dataset_cache"

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

```

Paths: Update the paths to your video directory, CSV annotations, model save location, and cache directories.
Data: Set the frame rate for sampling frames from the videos.
Training: Configure hyperparameters such as batch size, number of epochs, learning rate, model architecture, and training strategies.
Augmentation: Adjust data augmentation parameters to improve model generalization.
Logging: Set intervals for logging training progress.

## Usage

To start training the model with your dataset, simply run the script:

```bash
python critter_detector.py --video_dir /path/to/videos --csv_dir /path/to/csvs --mode train
```

Note: The model uses the pretrained DETR model by default for feature extraction, without additional training.

## Model Checkpointing

During training, the script saves checkpoints after each epoch. These checkpoints contain the model state, optimizer state, and the current epoch number. They are saved to the path specified in `config.toml` and can be used for resuming training or model evaluation.

## Final Model Saving

After the training is complete, the final model is saved to the specified directory. This model can be used for detecting highlights in new underwater videos.

## Contributing

Contributions to WaterColumnHighlighter are welcome! If you have suggestions for improvements or encounter any issues, please open an issue or submit a pull request.

## License
MIT
