import os
import pandas as pd
import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, ColorJitter
from torchvision.models import resnet50
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import argparse
import datetime
from datetime import timedelta
import toml
import shutil
import wandb
from sklearn.model_selection import KFold
import tarfile
import tempfile
import resource
import signal

# Path to the temporary directory on the external drive
temp_dir = "/Volumes/My Passport for Mac/Data/Temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

tempfile.tempdir = temp_dir

# Increase the limit on the number of open files
soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard_limit), hard_limit))

# Handle SIGTERM to ensure proper cleanup
def sigterm_handler(signum, frame):
    print("Received SIGTERM, exiting gracefully...")
    cleanup_semaphores()
    exit(0)

signal.signal(signal.SIGTERM, sigterm_handler)

# Function to clean up semaphores
def cleanup_semaphores():
    import ctypes
    libc = ctypes.CDLL('libc.so.6')
    for i in range(4096):
        try:
            libc.semctl(i, 0, 0, 0)
        except Exception:
            pass

# Call cleanup_semaphores at exit
import atexit
atexit.register(cleanup_semaphores)

def convert_video(input_path, output_path):
    from moviepy.editor import VideoFileClip
    clip = None
    try:
        clip = VideoFileClip(input_path)
        clip.write_videofile(output_path, codec='libx264')
        print(f"Successfully converted {input_path} and saved to {output_path}")
    except Exception as e:
        print(f"Failed to convert video {input_path} to {output_path}: {e}")
    finally:
        if clip:
            clip.reader.close()
            if clip.audio:
                clip.audio.reader.close_proc()
        if os.path.exists(output_path) and "need_cleanup" in output_path:
            os.remove(output_path)
            print(f"Temporary file deleted: {output_path}")
        else:
            print(f"No cleanup needed for: {output_path}")

# Load configuration FIRST
config = toml.load('../config.toml')

# THEN Initialize wandb with the loaded configuration
wandb.init(project="video_highlight_detection", entity="patrickallencooper", config=config['training'])
config_wandb = wandb.config

class VideoDataset(Dataset):
    def __init__(self, video_dir, csv_dir, clip_length=10, transform=None):
        print("Initializing VideoDataset")
        self.video_dir = video_dir
        self.csv_dir = csv_dir
        self.clip_length = clip_length
        self.transform = transform
        self.clips, self.labels = self._load_clips_and_labels()

    def _find_csv_file(self, dive_dir):
        expedition_dir = dive_dir.split('_')[0]
        csv_expedition_dir = os.path.join(self.csv_dir, expedition_dir)
        if os.path.exists(csv_expedition_dir):
            csv_files = [f for f in os.listdir(csv_expedition_dir) if f.endswith('.csv')]
            if csv_files:
                return os.path.join(csv_expedition_dir, csv_files[0])
        return None

    def _load_clips_and_labels(self):
        clips = []
        labels = []

        print("Loading clips and labels")

        for dive_dir in os.listdir(self.video_dir):
            if dive_dir.startswith("EX"):
                video_dive_dir = os.path.join(self.video_dir, dive_dir)
                compressed_dir = os.path.join(video_dive_dir, "Compressed")
                tar_file = os.path.join(video_dive_dir, "Compressed.tar")

                csv_file = self._find_csv_file(dive_dir)

                if csv_file is None:
                    print(f"CSV file not found for dive: {dive_dir}")
                    continue

                df = pd.read_csv(csv_file)
                print(f"Processing dive: {dive_dir}")

                if os.path.exists(compressed_dir):
                    for video_name in os.listdir(compressed_dir):
                        if not video_name.endswith('.mp4'):
                            continue

                        video_path = os.path.join(compressed_dir, video_name)
                        clip_labels = self._process_video(video_path, df)
                        clips.extend(clip_labels[0])
                        labels.extend(clip_labels[1])

                elif os.path.exists(tar_file):
                    with tarfile.open(tar_file, 'r') as tar:
                        for member in tar.getmembers():
                            if member.name.endswith('.mp4'):
                                video_file = tar.extractfile(member)
                                clip_labels = self._process_video(video_file, df)
                                clips.extend(clip_labels[0])
                                labels.extend(clip_labels[1])

                else:
                    print(f"Neither Compressed directory nor Compressed.tar file found for dive: {dive_dir}")

        print(f"Total clips: {len(clips)}")
        print(f"Total labels: {len(labels)}")
        return clips, labels

    def _process_video(self, video_file, df):
        clips = []
        labels = []
        target_fps = 29
        clip_frames = self.clip_length * target_fps

        if isinstance(video_file, str):
            video_name = os.path.basename(video_file)
            video_timestamp = video_name.split('_')[2] if len(video_name.split('_')) > 2 else None
            cap = cv2.VideoCapture(video_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            video_name = video_file.name.lstrip('/')
            video_timestamp = video_name.split('_')[2] if len(video_name.split('_')) > 2 else None
            temp_dir = tempfile.mkdtemp()
            temp_video_path = os.path.join(temp_dir, 'temp_video.mp4')
            with open(temp_video_path, 'wb') as f:
                f.write(video_file.read())
            convert_video(temp_video_path, temp_video_path)
            cap = cv2.VideoCapture(temp_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            shutil.rmtree(temp_dir)

        if not video_timestamp:
            print(f"Skipping video file with unexpected naming format: {video_name}")
            return clips, labels

        try:
            video_start_time = datetime.datetime.strptime(video_timestamp, '%Y%m%dT%H%M%SZ')
        except ValueError:
            print(f"Skipping video file with incorrect timestamp format: {video_name}")
            return clips, labels

        datetime_format = '%Y-%m-%d %H:%M:%S'
        df['Start Date'] = pd.to_datetime(df['Start Date'], format=datetime_format, errors='coerce')
        df['End Date'] = pd.to_datetime(df['End Date'], format=datetime_format, errors='coerce')
        df = df.dropna(subset=['Start Date', 'End Date'])
        frame_step = int(fps / target_fps)

        for i in range(0, frame_count, clip_frames * frame_step):
            start_frame = i
            end_frame = min(start_frame + clip_frames * frame_step, frame_count)
            clip_end_time = video_start_time + timedelta(seconds=end_frame / fps)
            filtered_df = df[(df['Start Date'] >= video_start_time) & (df['End Date'] <= clip_end_time)]
            filtered_df = filtered_df[filtered_df['Taxon Path'].str.contains('Biology / Organism')]

            if not filtered_df.empty:
                annotation_start_time = filtered_df['Start Date'].min()
                annotation_end_time = filtered_df['End Date'].max()
                if annotation_start_time < video_start_time or annotation_end_time > clip_end_time:
                    print(f"Warning: Annotation timestamps outside clip range for {video_name}, clip {i}")

            highlight = len(filtered_df) > 0
            if isinstance(video_file, str):
                clips.append((video_file, start_frame, end_frame))
            else:
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    frames = []
                    for _ in range(clip_frames):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if cap.get(cv2.CAP_PROP_POS_FRAMES) % frame_step == 0:
                            frames.append(frame)
                    if len(frames) < clip_frames:
                        padding_frames = [np.zeros_like(frames[0]) for _ in range(clip_frames - len(frames))]
                        frames.extend(padding_frames)
                    clips.append((frames, start_frame, end_frame))
                except Exception as e:
                    print(f"Error: Failed to extract clip from video for {video_name}, clip {i}: {e}")
            labels.append(int(highlight))

        if isinstance(video_file, str):
            cap.release()
        return clips, labels

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        video_path, start_frame, end_frame = self.clips[idx]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        frames = torch.stack(frames)
        label = self.labels[idx]
        return frames, label

class HighlightDetectionModel(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(HighlightDetectionModel, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
        with torch.no_grad():
            dummy_input = torch.rand(1, 3, 224, 224)
            resnet_output = self.resnet(dummy_input)
            resnet_output_size = resnet_output.size(1)
        self.lstm = nn.LSTM(resnet_output_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.resnet(x)
        x = x.view(batch_size, timesteps, -1)
        x, (hn, cn) = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return torch.sigmoid(x)

def train(video_dir, csv_dir):
    print("Starting training")
    transform = Compose([
        Resize((224, 224)),
        RandomCrop((200, 200)),
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_dir = temp_dir
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_path = os.path.join(dataset_dir, "extracted_dataset.pt")

    if os.path.exists(dataset_path):
        dataset = torch.load(dataset_path)
        print(f"Loaded extracted dataset from {dataset_path}")
    else:
        dataset = VideoDataset(
            video_dir=video_dir,
            csv_dir=csv_dir,
            clip_length=config['training']['clip_length'],
            transform=transform
        )
        print(f"Dataset created with {len(dataset)} clips")
        torch.save(dataset, dataset_path)
        print(f"Saved extracted dataset to {dataset_path}")

    k = config['training']['k_folds']
    kfold = KFold(n_splits=k, shuffle=True)
    models = []

    for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold+1}')
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

        print(f"Training dataloader created with {len(train_dataloader)} batches")
        print(f"Validation dataloader created with {len(val_dataloader)} batches")

        model = HighlightDetectionModel(
            hidden_dim=config['training']['hidden_dim'],
            num_layers=config['training']['num_layers']
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion = nn.BCELoss()
        optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

        wandb.watch(model, log='all')

        best_val_accuracy = 0.0
        early_stopping_patience = config['training']['early_stopping_patience']
        early_stopping_counter = 0

        for epoch in range(1):
            print(f'Epoch {epoch+1}/1')
            model.train()
            train_loss = 0.0
            for batch_idx, (clips, labels) in enumerate(train_dataloader):
                clips, labels = clips.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(clips)
                labels = labels.view(-1, 1).float()
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
                optimizer.step()
                train_loss += loss.item()
                print(f'Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item()}')

                # Log metrics by batch
                wandb.log({
                    "fold": fold + 1,
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                    "train_loss": loss.item()
                })

            train_loss /= len(train_dataloader)

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_idx, (clips, labels) in enumerate(val_dataloader):
                    clips, labels = clips.to(device), labels.to(device)
                    outputs = model(clips)
                    labels = labels.view(-1, 1).float()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    predicted = (outputs >= 0.5).float()
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
                    print(f'Validation Batch {batch_idx+1}/{len(val_dataloader)}, Loss: {loss.item()}')

                    # Log validation metrics by batch
                    wandb.log({
                        "fold": fold + 1,
                        "epoch": epoch + 1,
                        "batch": batch_idx + 1,
                        "val_loss": loss.item()
                    })

            val_loss /= len(val_dataloader)
            val_accuracy = val_correct / val_total

            print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')

            wandb.log({
                "fold": fold + 1,
                "epoch": epoch + 1,
                "train_loss_avg": train_loss,
                "val_loss_avg": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

            scheduler.step(val_accuracy)

            checkpoint_path = f"{config['paths']['model_save_path']}_fold_{fold+1}_epoch_{epoch+1}_batch_{64}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                early_stopping_counter = 0
                best_model_path = f"{config['paths']['model_save_path']}_fold_{fold+1}_best_val_acc_{best_val_accuracy:.4f}.pth"
                torch.save(model.state_dict(), best_model_path)
                print(f'Best model saved to {best_model_path}')
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

        best_model_path = f"{config['paths']['model_save_path']}_fold_{fold+1}_best_val_acc_{best_val_accuracy:.4f}.pth"
        model.load_state_dict(torch.load(best_model_path))
        models.append(model)

    return models, dataset

def test(models, dataset, mode='test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    correct = 0
    total = 0
    false_positives = 0
    false_negatives = 0
    results = []

    with torch.no_grad():
        for clips, labels in data_loader:
            clips, labels = clips.to(device), labels.to(device)
            ensemble_outputs = torch.zeros(clips.size(0), 1).to(device)
            for model in models:
                model.eval()
                outputs = model(clips)
                ensemble_outputs += outputs
            ensemble_outputs /= len(models)
            predicted = (ensemble_outputs >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Highlight Detection Training')
    parser.add_argument('--video_dir', type=str, required=True, help='Path to the video directory')
    parser.add_argument('--csv_dir', type=str, required=True, help='Path to the CSV directory')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help='Mode: train or test')
    parser.add_argument('--model_paths', type=str, nargs='*', help='Paths to the trained models for testing')
    args = parser.parse_args()

    video_dir = args.video_dir
    csv_dir = args.csv_dir

    if args.mode == 'train':
        models, dataset = train(video_dir, csv_dir)
        test(models, dataset, mode='train')
    elif args.mode == 'test':
        dataset_dir = temp_dir
        dataset_path = os.path.join(dataset_dir, "extracted_dataset.pt")
        dataset = torch.load(dataset_path)
        model_paths = args.model_paths
        models = [HighlightDetectionModel(
            hidden_dim=config['training']['hidden_dim'],
            num_layers=config['training']['num_layers']
        ) for _ in model_paths]
        for model, path in zip(models, model_paths):
            model.load_state_dict(torch.load(path))
        test(models, dataset, mode='test')
