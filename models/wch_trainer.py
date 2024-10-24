import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import csv
import numpy as np
from PIL import Image
import pickle
from datetime import datetime
import tarfile
import tempfile
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse
from torchvision.transforms import (
    Compose,
    Resize,
    Normalize,
    ToTensor,
    RandomCrop,
    RandomHorizontalFlip,
    ColorJitter,
    ToPILImage,
)
import toml
import wandb
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torch.optim import Adam

import math
from torch.optim.lr_scheduler import LambdaLR

# Import DETR modules
from transformers import DetrImageProcessor, DetrForObjectDetection

import boto3
from botocore.exceptions import NoCredentialsError

# Load configuration
config = toml.load('config.toml')


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between
    the initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps.
        num_cycles: The number of cycles (half cycles) in the cosine decay.
        last_epoch: The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


def save_checkpoint(model, optimizer, epoch, loss, global_step, checkpoint_dir):
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir, f'checkpoint_step_{global_step}.pth'
    )

    try:
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'global_step': global_step,
            },
            checkpoint_path,
        )
        print(f'Checkpoint saved at step {global_step}: {checkpoint_path}')
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


# Helpers
def process_chunk(args):
    chunk, dataset = args
    labels = np.array([dataset[i][1] for i in chunk])
    return np.sum(labels == 1), np.sum(labels == 0)


def calculate_dataset_metrics(dataset):
    total_frames = int(len(dataset) * 0.0001)

    # Randomly select the frames
    selected_indices = random.sample(range(len(dataset)), total_frames)

    # Determine the number of processes to use
    num_processes = max(1, cpu_count() - 8)

    # Split the selected indices into chunks for parallel processing
    chunk_size = max(1, total_frames // num_processes)
    chunks = [selected_indices[i: i + chunk_size] for i in range(0, total_frames, chunk_size)]

    # Create a pool of worker processes
    with Pool(num_processes) as pool:
        # Use tqdm to show progress
        results = list(
            tqdm(
                pool.imap(process_chunk, [(chunk, dataset) for chunk in chunks]),
                total=len(chunks),
                desc="Calculating dataset metrics",
            )
        )

    # Sum up the results
    positive_frames, negative_frames = np.sum(results, axis=0)

    positive_percentage = (positive_frames / total_frames) * 100
    negative_percentage = (negative_frames / total_frames) * 100

    metrics = {
        "Total Frames": total_frames,
        "Positive Frames (with organism)": int(positive_frames),
        "Negative Frames (without organism)": int(negative_frames),
        "Positive Percentage": positive_percentage,
        "Negative Percentage": negative_percentage,
    }

    return metrics


def print_dataset_metrics(metrics):
    print("\nDataset Metrics:")
    print(
        f"Total Frames: {metrics['Total Frames']}"
    )
    print(
        f"Positive Frames (with organism): {metrics['Positive Frames (with organism)']} ({metrics['Positive Percentage']:.2f}%)"
    )
    print(
        f"Negative Frames (without organism): {metrics['Negative Frames (without organism)']} ({metrics['Negative Percentage']:.2f}%)"
    )

    # Write metrics to a file
    with open("dataset_metrics.txt", "w") as f:
        f.write("Dataset Metrics:\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.2f}\n")
            else:
                f.write(f"{key}: {value}\n")



class SlidingWindowVideoDataset(Dataset):
    def __init__(
        self,
        video_dir,
        csv_dir,
        window_size,
        stride,
        transform=None,
        cache_dir='./dataset_cache',
    ):
        self.video_dir = video_dir
        self.csv_dir = csv_dir
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.cache_dir = cache_dir
        self.frame_info = []
        self.total_positive_frames = 0
        self.total_negative_frames = 0
        self._load_clips_and_labels()

    def _find_csv_file(self, dive_dir):
        expedition_dir = dive_dir.split('_')[0]
        csv_expedition_dir = os.path.join(self.csv_dir, expedition_dir)
        if os.path.exists(csv_expedition_dir):
            csv_files = [f for f in os.listdir(csv_expedition_dir) if f.endswith('.csv')]
            if csv_files:
                return os.path.join(csv_expedition_dir, csv_files[0])
        return None

    def _load_clips_and_labels(self):
        os.makedirs(self.cache_dir, exist_ok=True)

        for dive_dir in tqdm(os.listdir(self.video_dir), desc="Processing dives"):
            if dive_dir.startswith("EX"):
                dive_cache_path = os.path.join(
                    self.cache_dir, f"{dive_dir}_cache.pkl"
                )

                if os.path.exists(dive_cache_path):
                    print(f"Loading cached data for dive: {dive_dir}")
                    with open(dive_cache_path, 'rb') as f:
                        dive_frame_info = pickle.load(f)
                else:
                    print(f"Processing dive: {dive_dir}")
                    dive_frame_info = self._process_dive(dive_dir)

                    with open(dive_cache_path, 'wb') as f:
                        pickle.dump(dive_frame_info, f)

                self.frame_info.extend(dive_frame_info)

                positive_frames = sum(1 for _, _, label in dive_frame_info if label == 1)
                negative_frames = sum(1 for _, _, label in dive_frame_info if label == 0)
                self.total_positive_frames += positive_frames
                self.total_negative_frames += negative_frames

                print(f"Dive {dive_dir} processed:")
                print(f"  Positive frames: {positive_frames}")
                print(f"  Negative frames: {negative_frames}")
                print(f"  Total frames: {len(dive_frame_info)}")
                print(f"Cumulative totals:")
                print(f"  Total positive frames: {self.total_positive_frames}")
                print(f"  Total negative frames: {self.total_negative_frames}")
                print(f"  Total frames processed: {len(self.frame_info)}")
                print("--------------------")

        print("\nDataset construction completed.")
        print(f"Final totals:")
        print(f"  Total positive frames: {self.total_positive_frames}")
        print(f"  Total negative frames: {self.total_negative_frames}")
        print(f"  Total frames in dataset: {len(self.frame_info)}")

    def _process_dive(self, dive_dir):
        video_dive_dir = os.path.join(self.video_dir, dive_dir)
        compressed_dir = os.path.join(video_dive_dir, "Compressed")
        tar_file = os.path.join(video_dive_dir, "Compressed.tar")
        csv_file = self._find_csv_file(dive_dir)

        if csv_file is None:
            print(f"CSV file not found for dive: {dive_dir}")
            return []

        print(f"Processing dive: {dive_dir}")
        print(f"CSV file: {csv_file}")

        # Read CSV file manually to handle inconsistent number of fields
        rows = []
        with open(csv_file, 'r') as f:
            csv_reader = csv.reader(f)
            headers = next(csv_reader)  # Read the header row
            for row_index, row in enumerate(csv_reader, start=2):  # Start from 2 as 1 is header
                if len(row) != len(headers):
                    print(
                        f"Warning: Inconsistent number of fields in row {row_index}. Expected {len(headers)}, got {len(row)}."
                    )
                    # Pad or truncate the row to match header length
                    row = (row + [''] * len(headers))[:len(headers)]
                rows.append(row)

        # Create DataFrame from the processed rows
        df = pd.DataFrame(rows, columns=headers)

        # Data validation
        if 'Start Date' not in df.columns or 'End Date' not in df.columns:
            print(f"CSV columns: {df.columns}")
            raise ValueError("CSV is missing 'Start Date' or 'End Date' column")

        # Custom date parsing function
        def parse_date(date_string):
            if pd.isna(date_string) or date_string == 'Start Date':
                return pd.NaT
            try:
                return datetime.strptime(date_string, '%Y%m%dT%H%M%S.%fZ')
            except ValueError:
                print(f"Warning: Unable to parse date: {date_string}")
                return pd.NaT

        # Apply custom date parsing
        df['Start Date'] = df['Start Date'].apply(parse_date)
        df['End Date'] = df['End Date'].apply(parse_date)

        # Remove rows with NaT values
        df = df.dropna(subset=['Start Date', 'End Date'])

        print(f"Total annotations in CSV: {len(df)}")
        print(f"Sample of parsed dates:")
        print(df[['Start Date', 'End Date']].head())

        dive_frame_info = []

        if os.path.exists(compressed_dir):
            for video_name in tqdm(os.listdir(compressed_dir), desc=f"Processing videos in {dive_dir}"):
                if video_name.endswith('.mp4'):
                    video_path = os.path.join(compressed_dir, video_name)
                    dive_frame_info.extend(
                        self._process_video(video_path, df, video_name)
                    )
        elif os.path.exists(tar_file):
            with tarfile.open(tar_file, 'r') as tar:
                for member in tqdm(
                    tar.getmembers(), desc=f"Processing videos in {dive_dir} tar"
                ):
                    if member.name.endswith('.mp4'):
                        video_file = tar.extractfile(member)
                        with tempfile.NamedTemporaryFile(
                            suffix='.mp4', delete=False
                        ) as temp_video:
                            temp_video.write(video_file.read())
                            temp_video_path = temp_video.name
                        dive_frame_info.extend(
                            self._process_video(
                                temp_video_path, df, os.path.basename(member.name)
                            )
                        )
                        os.unlink(temp_video_path)
        else:
            print(
                f"Neither Compressed directory nor Compressed.tar file found for dive: {dive_dir}"
            )

        return dive_frame_info

    def _process_video(self, video_file, df, original_video_name):
        video_frame_info = []
        video_timestamp = (
            original_video_name.split('_')[2]
            if len(original_video_name.split('_')) > 2
            else None
        )
        if not video_timestamp:
            print(
                f"Skipping video file with unexpected naming format: {original_video_name}"
            )
            return video_frame_info

        try:
            video_start_time = pd.to_datetime(video_timestamp, format='%Y%m%dT%H%M%S%Z')
        except ValueError:
            print(
                f"Skipping video file with incorrect timestamp format: {original_video_name}"
            )
            return video_frame_info

        print(f"Processing video: {video_file}")
        print(f"Video start time: {video_start_time}")

        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = pd.Timedelta(seconds=frame_count / fps)
        video_end_time = video_start_time + video_duration

        # Filter annotations relevant to this video
        video_annotations = df[
            (df['Start Date'] >= video_start_time) & (df['Start Date'] <= video_end_time)
        ]
        print(f"Annotations for this video: {len(video_annotations)}")

        # Create a set of frame numbers with annotations
        annotated_frames = set()
        for _, row in video_annotations.iterrows():
            annotation_time = row['Start Date']
            frame_number = int(
                (annotation_time - video_start_time).total_seconds() * fps
            )
            annotated_frames.add(frame_number)

        # Process frames
        target_fps = config['data']['frame_rate']
        frame_step = int(fps / target_fps)
        window_frames = int(5 * fps)  # 5-second window

        for frame_num in range(0, frame_count, frame_step):
            highlight = 0
            for check_frame in range(
                max(0, frame_num - window_frames),
                min(frame_count, frame_num + window_frames + 1),
            ):
                if check_frame in annotated_frames:
                    highlight = 1
                    break
            video_frame_info.append((original_video_name, frame_num, highlight))

            if frame_num % 1000 == 0:  # Log every 1000th frame for debugging
                frame_time = video_start_time + pd.Timedelta(seconds=frame_num / fps)
                print(f"Frame {frame_num}, Time: {frame_time}, Highlight: {highlight}")

        cap.release()

        positive_frames = sum(1 for _, _, label in video_frame_info if label == 1)
        print(f"Video {original_video_name} processed:")
        print(f"  Total frames: {len(video_frame_info)}")
        print(f"  Positive frames: {positive_frames}")
        print(f"  Negative frames: {len(video_frame_info) - positive_frames}")

        return video_frame_info

    def __len__(self):
        return len(self.frame_info)

    def __getitem__(self, idx):
        center_frame_info = self.frame_info[idx]
        video_file, center_frame_num, label = center_frame_info

        half_window = self.window_size // 2
        start_idx = max(0, idx - half_window)
        end_idx = min(len(self.frame_info), idx + half_window + 1)

        frame_sequence = []
        for i in range(start_idx, end_idx):
            frame_info = self.frame_info[i]
            video_file, frame_num, _ = frame_info
            cap = cv2.VideoCapture(video_file)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = Image.fromarray(frame)

            if self.transform:
                frame = self.transform(frame)

            frame_sequence.append(frame)

        # Zero-pad if necessary
        if len(frame_sequence) < self.window_size:
            padding = [
                torch.zeros_like(frame_sequence[0])
                for _ in range(self.window_size - len(frame_sequence))
            ]
            if idx < half_window:
                frame_sequence = padding + frame_sequence
            else:
                frame_sequence = frame_sequence + padding

        frame_sequence = torch.stack(frame_sequence)
        return frame_sequence, torch.tensor(label, dtype=torch.float32)

class BidirectionalLSTMModel(nn.Module):
    def __init__(self, hidden_dim, num_layers, feature_extractor='resnet', fine_tune=False):
        super(BidirectionalLSTMModel, self).__init__()
        self.feature_extractor_type = feature_extractor
        self.fine_tune = fine_tune
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.feature_extractor_type == 'detr':
            self.processor = DetrImageProcessor.from_pretrained(
                "facebook/detr-resnet-50", revision="no_timm"
            )
            self.detr_model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                output_hidden_states=True,
                revision="no_timm",
            )
            self.detr_model = self.detr_model.to(self.device)

            if not self.fine_tune:
                for param in self.detr_model.parameters():
                    param.requires_grad = False

            with torch.no_grad():
                dummy_input = torch.rand(3, 224, 224).to(self.device)
                inputs = self.processor(images=dummy_input, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.detr_model(**inputs)
                last_hidden_state = outputs.last_hidden_state  # Shape: (1, num_queries, hidden_size)
                hidden_size = last_hidden_state.size(-1)
                feature_size = hidden_size
        elif self.feature_extractor_type == 'resnet':
            self.resnet = resnet50(pretrained=True)
            self.resnet.fc = nn.Identity()
            self.resnet = self.resnet.to(self.device)
            if not self.fine_tune:
                for param in self.resnet.parameters():
                    param.requires_grad = False
            with torch.no_grad():
                dummy_input = torch.rand(1, 3, 224, 224).to(self.device)
                resnet_output = self.resnet(dummy_input)
                feature_size = resnet_output.size(1)
        else:
            raise ValueError(f"Unsupported feature extractor: {self.feature_extractor_type}")

        self.lstm = nn.LSTM(
            feature_size, hidden_dim, num_layers, batch_first=True, bidirectional=True
        )
        self.lstm = self.lstm.to(self.device)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 because of bidirectional
        self.fc = self.fc.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)

        if self.feature_extractor_type == 'detr':
            # Convert c_in to list of PIL Images
            imgs_pil = [ToPILImage()(img.cpu()) for img in c_in]
            inputs = self.processor(images=imgs_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.detr_model(**inputs)
            last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size * timesteps, num_queries, hidden_size)
            # Take mean over num_queries
            features = last_hidden_state.mean(dim=1)  # Shape: (batch_size * timesteps, hidden_size)
        elif self.feature_extractor_type == 'resnet':
            features = self.resnet(c_in)
        else:
            raise ValueError(f"Unsupported feature extractor: {self.feature_extractor_type}")

        features = features.view(batch_size, timesteps, -1)
        lstm_out, _ = self.lstm(features)
        center_frame_output = lstm_out[:, lstm_out.size(1) // 2, :]
        output = self.fc(center_frame_output)
        return torch.sigmoid(output)


class DETRModel(nn.Module):
    def __init__(self, num_classes=1, fine_tune=False):
        super(DETRModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50", revision="no_timm"
        )
        self.detr_model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_classes,
            revision="no_timm",
        )
        self.detr_model = self.detr_model.to(self.device)
        self.fine_tune = fine_tune

        if not self.fine_tune:
            for param in self.detr_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = x.to(self.device)
        # x shape: (batch_size, C, H, W)
        imgs_pil = [ToPILImage()(img.cpu()) for img in x]
        inputs = self.processor(images=imgs_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.detr_model(**inputs)
        # For classification, we can take the logits from class labels
        logits = outputs.logits  # Shape: (batch_size, num_queries, num_classes + 1)
        # We can average over the queries
        logits = logits.mean(dim=1)  # Shape: (batch_size, num_classes + 1)
        # Since we're dealing with presence of objects, we can consider the class probabilities
        probs = torch.softmax(logits, dim=-1)
        # Return probability of class 1 (assuming binary classification)
        return probs[:, 1].unsqueeze(1)


def download_data_from_s3(bucket_name, prefix, local_dir):
    s3 = boto3.client('s3')
    try:
        for obj in s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)['Contents']:
            s3.download_file(bucket_name, obj['Key'], os.path.join(local_dir, os.path.basename(obj['Key'])))
    except NoCredentialsError:
        print("Credentials not available for AWS S3.")
        return False
    return True

def train(video_dir, csv_dir, model_type='lstm', feature_extractor='resnet', fine_tune=False):
    # Load configuration
    use_aws = config['aws'].get('use_aws', False)
    s3_bucket_name = config['aws'].get('s3_bucket_name', '')
    s3_data_prefix = config['aws'].get('s3_data_prefix', '')
    aws_region = config['aws'].get('aws_region', 'us-west-2')

    if use_aws:
        # Set up AWS S3
        boto3.setup_default_session(region_name=aws_region)
        local_data_dir = '/tmp/data'  # Temporary directory for data on EC2
        os.makedirs(local_data_dir, exist_ok=True)

        # Download data from S3
        if not download_data_from_s3(s3_bucket_name, s3_data_prefix, local_data_dir):
            print("Failed to download data from S3.")
            return

        # Update paths to local data directory
        video_dir = os.path.join(local_data_dir, 'videos')
        csv_dir = os.path.join(local_data_dir, 'csvs')

    print("Starting training")
    if model_type == 'lstm':
        transform = Compose(
            [
                Resize((224, 224)),
                RandomCrop((config['augmentation']['random_crop_size'], config['augmentation']['random_crop_size'])),
                RandomHorizontalFlip(),
                ColorJitter(
                    brightness=config['augmentation']['color_jitter_brightness'],
                    contrast=config['augmentation']['color_jitter_contrast'],
                    saturation=config['augmentation']['color_jitter_saturation'],
                    hue=config['augmentation']['color_jitter_hue'],
                ),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif model_type == 'detr':
        transform = Compose(
            [
                Resize((224, 224)),
                ToTensor(),
            ]
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    dataset = SlidingWindowVideoDataset(
        video_dir=video_dir,
        csv_dir=csv_dir,
        window_size=config['training']['window_size'] if model_type == 'lstm' else 1,
        stride=config['training']['stride'],
        transform=transform,
        cache_dir=config['paths']['dataset_cache_dir'],
    )
    print(f"Dataset created with {len(dataset)} frames")

    k = config['training']['k_folds']
    kfold = KFold(n_splits=k, shuffle=True)
    models = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_indices, val_indices) in enumerate(kfold.split(range(len(dataset)))):
        print(f'Fold {fold + 1}')
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=2,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=2,
        )

        print(f"Training dataloader created with {len(train_dataloader)} batches")
        print(f"Validation dataloader created with {len(val_dataloader)} batches")

        if model_type == 'lstm':
            model = BidirectionalLSTMModel(
                hidden_dim=config['training']['hidden_dim'],
                num_layers=config['training']['num_layers'],
                feature_extractor=feature_extractor,
                fine_tune=fine_tune,
            )
        elif model_type == 'detr':
            model = DETRModel(num_classes=1, fine_tune=fine_tune)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.to(device)

        criterion = FocalLoss(alpha=0.25, gamma=2)
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['learning_rate'])

        # Use scheduler parameters from config
        scheduler_factor = config['training'].get('scheduler_factor', 0.5)
        scheduler_patience = config['training'].get('scheduler_patience', 5)

        # Initialize the new scheduler
        total_steps = config['training']['num_epochs'] * len(train_dataloader)
        warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=scheduler_factor  # Use the scheduler factor from config
        )

        wandb.watch(model, log='all')

        best_val_accuracy = 0.0
        early_stopping_patience = config['training']['early_stopping_patience']
        early_stopping_counter = 0

        checkpoint_dir = os.path.join(config['paths']['checkpoint_dir'], f'fold_{fold + 1}')
        os.makedirs(checkpoint_dir, exist_ok=True)

        global_step = 0
        for epoch in range(config['training']['num_epochs']):
            print(f'Epoch {epoch + 1}/{config["training"]["num_epochs"]}')
            model.train()
            train_loss = 0.0
            for batch_idx, (frame_sequences, labels) in enumerate(train_dataloader):
                frame_sequences, labels = frame_sequences.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(frame_sequences)
                labels = labels.view(-1, 1).float()
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), config['training']['gradient_clip']
                )
                optimizer.step()
                scheduler.step()  # Step the scheduler every batch

                train_loss += loss.item()
                global_step += 1

                if global_step % config['training']['checkpoint_steps'] == 0:
                    save_checkpoint(
                        model, optimizer, epoch, loss.item(), global_step, checkpoint_dir
                    )

                if batch_idx % config['logging']['log_interval'] == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(
                        f'Batch {batch_idx + 1}/{len(train_dataloader)}, Loss: {loss.item()}, LR: {current_lr}'
                    )
                    wandb.log(
                        {
                            "fold": fold + 1,
                            "epoch": epoch + 1,
                            "batch": batch_idx + 1,
                            "train_loss": loss.item(),
                            "learning_rate": current_lr,
                            "global_step": global_step,
                        }
                    )

            train_loss /= len(train_dataloader)

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_idx, (frame_sequences, labels) in enumerate(val_dataloader):
                    frame_sequences, labels = frame_sequences.to(device), labels.to(device)
                    outputs = model(frame_sequences)
                    labels = labels.view(-1, 1).float()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    predicted = (outputs >= 0.5).float()
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
                    if batch_idx % config['logging']['log_interval'] == 0:
                        print(
                            f'Validation Batch {batch_idx + 1}/{len(val_dataloader)}, Loss: {loss.item()}'
                        )

                    wandb.log(
                        {
                            "fold": fold + 1,
                            "epoch": epoch + 1,
                            "batch": batch_idx + 1,
                            "val_loss": loss.item(),
                            "global_step": global_step,
                        }
                    )

            val_loss /= len(val_dataloader)
            val_accuracy = val_correct / val_total

            print(
                f'Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}'
            )

            wandb.log(
                {
                    "fold": fold + 1,
                    "epoch": epoch + 1,
                    "train_loss_avg": train_loss,
                    "val_loss_avg": val_loss,
                    "val_accuracy": val_accuracy,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "global_step": global_step,
                }
            )

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                early_stopping_counter = 0
                best_model_path = (
                    f"{config['paths']['model_save_path']}_fold_{fold + 1}_best_val_acc_{best_val_accuracy:.4f}.pth"
                )
                torch.save(model.state_dict(), best_model_path)
                print(f'Best model saved to {best_model_path}')
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

        best_model_path = (
            f"{config['paths']['model_save_path']}_fold_{fold + 1}_best_val_acc_{best_val_accuracy:.4f}.pth"
        )
        model.load_state_dict(torch.load(best_model_path))
        models.append(model)

    return models, dataset


def test(models, dataset, model_type='lstm', mode='test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = DataLoader(
        dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=2
    )
    correct = 0
    total = 0
    false_positives = 0
    false_negatives = 0
    results = []

    with torch.no_grad():
        for frame_sequences, labels in data_loader:
            frame_sequences, labels = frame_sequences.to(device), labels.to(device)
            ensemble_outputs = torch.zeros(frame_sequences.size(0), 1).to(device)
            for model in models:
                model.eval()
                outputs = model(frame_sequences)
                ensemble_outputs += outputs
            ensemble_outputs /= len(models)
            predicted = (ensemble_outputs >= 0.5).float()
            correct += (predicted == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)

            false_positives += ((predicted == 1) & (labels.unsqueeze(1) == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels.unsqueeze(1) == 1)).sum().item()

            for i in range(len(labels)):
                results.append(f"Label: {labels[i].item()}, Predicted: {predicted[i].item()}")

    accuracy = correct / total

    with open(f"{mode}_results.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"False Positives: {false_positives}\n")
        f.write(f"False Negatives: {false_negatives}\n")
        f.write("Results:\n")
        for result in results:
            f.write(f"{result}\n")

    print(f"{mode.capitalize()} Accuracy: {accuracy}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")

    wandb.log(
        {
            f"{mode}_accuracy": accuracy,
            f"{mode}_false_positives": false_positives,
            f"{mode}_false_negatives": false_negatives,
        }
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Highlight Detection Training')
    parser.add_argument('--video_dir', type=str, required=True, help='Path to the video directory')
    parser.add_argument('--csv_dir', type=str, required=True, help='Path to the CSV directory')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help='Mode: train or test')
    parser.add_argument('--model_paths', type=str, nargs='*', help='Paths to the trained models for testing')
    parser.add_argument('--force_dataset_rebuild', action='store_true', help='Force rebuilding of the dataset cache')
    parser.add_argument('--model_type', type=str, choices=['lstm', 'detr'], default='lstm', help='Type of model to use: lstm or detr')
    parser.add_argument('--feature_extractor', type=str, choices=['resnet', 'detr'], default='resnet', help='Feature extractor to use with LSTM model')
    parser.add_argument('--fine_tune', action='store_true', help='Fine-tune the feature extractor or DETR model')
    args = parser.parse_args()

    video_dir = args.video_dir
    csv_dir = args.csv_dir

    if args.force_dataset_rebuild:
        dataset_cache_path = os.path.join(config['paths']['dataset_cache_dir'], 'dataset_cache.pkl')
        if os.path.exists(dataset_cache_path):
            os.remove(dataset_cache_path)
            print("Removed existing dataset cache. Rebuilding...")

    # Initialize wandb
    wandb.init(project='critter_detector')

    if args.mode == 'train':
        models, dataset = train(
            video_dir,
            csv_dir,
            model_type=args.model_type,
            feature_extractor=args.feature_extractor,
            fine_tune=args.fine_tune
        )
        test(models, dataset, model_type=args.model_type, mode='train')
    elif args.mode == 'test':
        if args.model_type == 'lstm':
            transform = Compose(
                [
                    Resize((224, 224)),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        elif args.model_type == 'detr':
            transform = Compose(
                [
                    Resize((224, 224)),
                    ToTensor(),
                ]
            )
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")

        dataset = SlidingWindowVideoDataset(
            video_dir=video_dir,
            csv_dir=csv_dir,
            window_size=config['training']['window_size'] if args.model_type == 'lstm' else 1,
            stride=config['training']['stride'],
            transform=transform,
            cache_dir=config['paths']['dataset_cache_dir'],
        )
        # Calculate and print dataset metrics for test mode as well
        metrics = calculate_dataset_metrics(dataset)
        print_dataset_metrics(metrics)
        wandb.log(metrics)

        model_paths = args.model_paths
        models = []
        for path in model_paths:
            if args.model_type == 'lstm':
                model = BidirectionalLSTMModel(
                    hidden_dim=config['training']['hidden_dim'],
                    num_layers=config['training']['num_layers'],
                    feature_extractor=args.feature_extractor,
                    fine_tune=args.fine_tune
                )
            elif args.model_type == 'detr':
                model = DETRModel(num_classes=1, fine_tune=args.fine_tune)
            else:
                raise ValueError(f"Unsupported model type: {args.model_type}")

            model.load_state_dict(torch.load(path))
            model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            models.append(model)
        test(models, dataset, model_type=args.model_type, mode='test')

    wandb.finish()




