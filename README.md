# WaterColumnHighlighter

WaterColumnHighlighter is a deep learning project aimed at identifying and extracting highlight segments from underwater videos. This tool leverages the power of convolutional neural networks (CNNs) and long short-term memory (LSTM) networks to analyze video data and classify segments as highlights or non-highlights. It's particularly useful for researchers, marine biologists, and enthusiasts looking to automatically detect interesting or significant moments in lengthy underwater footage.

## Features

- **Automatic Highlight Detection:** Automatically identifies highlights in underwater video footage using deep learning.
- **Pretrained Model Integration:** Utilizes pretrained CNN models for efficient feature extraction from video frames.
- **Temporal Dynamics Analysis:** Employs LSTM networks to understand the temporal relationships between video frames for accurate highlight detection.
- **Customizable Training:** Offers configurable training options to adapt the model to specific types of underwater footage.
- **Model Saving and Checkpointing:** Enables saving of model checkpoints during training and the final trained model for later use or further training.

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
pip install torch torchvision opencv-python numpy toml wandb
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

[training]
batch_size = 4
num_epochs = 10
learning_rate = 0.001
clip_length = 10
hidden_dim = 256
num_layers = 2
```

Ensure the paths to your video and CSV directories are correct, and modify the training parameters as needed.

## Usage

To start training the model with your dataset, simply run the script:

```bash
python water_column_highlighter.py
```

The script will process the videos, train the model based on the provided dataset, and save the model upon completion. Training progress and metrics are logged to Weights & Biases for easy monitoring and analysis.

## Model Checkpointing

During training, the script saves checkpoints after each epoch. These checkpoints contain the model state, optimizer state, and the current epoch number. They are saved to the path specified in `config.toml` and can be used for resuming training or model evaluation.

## Final Model Saving

After the training is complete, the final model is saved to the specified directory. This model can be used for detecting highlights in new underwater videos.

## Contributing

Contributions to WaterColumnHighlighter are welcome! If you have suggestions for improvements or encounter any issues, please open an issue or submit a pull request.

## License
MIT
