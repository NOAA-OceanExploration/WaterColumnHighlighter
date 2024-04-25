import cv2
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, ColorJitter
from torchvision.models import resnet50
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from moviepy.editor import VideoFileClip
import numpy as np
import os
import argparse
import datetime
from datetime import timedelta
import toml
import tempfile
import shutil
import wandb
import xtarfile as tarfile
from sklearn.model_selection import KFold

import shutil
import tempfile
from moviepy.editor import VideoFileClip
import os

# Path to the temporary directory on the external drive
temp_dir = "/Volumes/My Passport for Mac/Data/Temp"

# Check if the directory exists, and create it if it does not
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Now set it as the default directory for temporary files
import tempfile
tempfile.tempdir = temp_dir

def convert_video(input_path, output_path):
    try:
        with VideoFileClip(input_path) as clip:
            clip.write_videofile(output_path, codec='libx264')
            print(f"Successfully converted {input_path} and saved to {output_path}")
    except Exception as e:
        print(f"Failed to convert video {input_path} to {output_path}: {e}")
    finally:
        # Be careful with cleanup: Ensure you're not deleting needed files
        if os.path.exists(output_path) and "need_cleanup" in output_path:
            os.remove(output_path)
            print(f"Temporary file deleted: {output_path}")
        else:
            print(f"No cleanup needed for: {output_path}")


# Increase open file limit
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))

# Load configuration FIRST
config = toml.load('config.toml')

# THEN Initialize wandb with the loaded configuration
wandb.init(project="video_highlight_detection", config=config['training'])
config_wandb = wandb.config

class VideoDataset(Dataset):
    def __init__(self, video_dir, csv_dir, clip_length=10, transform=None):
        self.video_dir = video_dir
        self.csv_dir = csv_dir
        self.clip_length = clip_length
        self.transform = transform
        self.clips, self.labels = self._load_clips_and_labels()

    def _load_clips_and_labels(self):
        clips = []
        labels = []
        for dive_dir in os.listdir(self.video_dir):
            if dive_dir.startswith("EX"):
                video_dive_dir = os.path.join(self.video_dir, dive_dir)
                compressed_dir = os.path.join(video_dive_dir, "Compressed")
                tar_file = os.path.join(video_dive_dir, "Compressed.tar")
                
                # Find the corresponding CSV file for the dive based on timestamp range
                csv_file = None
                for expedition_dir in os.listdir(self.csv_dir):
                    if expedition_dir.startswith("EX") and dive_dir.startswith(expedition_dir):
                        csv_expedition_dir = os.path.join(self.csv_dir, expedition_dir)
                        csv_files = [f for f in os.listdir(csv_expedition_dir) if f.endswith('.csv')]
                        for csv_f in csv_files:
                            start_timestamp, end_timestamp = csv_f.split('_')[1:]
                            start_timestamp = start_timestamp.split('.')[0]
                            end_timestamp = end_timestamp.split('.')[0]
                            
                            if os.path.exists(compressed_dir):
                                video_files = [f for f in os.listdir(compressed_dir) if f.endswith('_ROVHD_Low.mp4')]
                            elif os.path.exists(tar_file):
                                with tarfile.open(tar_file, 'r') as tar:
                                    video_files = [f for f in tar.getnames() if f.endswith('_ROVHD_Low.mp4')]
                            else:
                                video_files = []
                            
                            for video_file in video_files:
                                video_timestamp = video_file.split('_')[2]
                                if start_timestamp <= video_timestamp <= end_timestamp:
                                    csv_file = os.path.join(csv_expedition_dir, csv_f)
                                    break
                            
                            if csv_file is not None:
                                break
                        
                        if csv_file is not None:
                            break
                
                if csv_file is None:
                    print(f"CSV file not found for dive: {dive_dir}")
                    continue
                
                df = pd.read_csv(csv_file)
                print(f"Processing dive: {dive_dir}")
                
                if os.path.exists(compressed_dir):
                    # Process extracted video files
                    for video_name in os.listdir(compressed_dir):
                        if not video_name.endswith('_ROVHD_Low.mp4'):
                            continue
                        
                        video_path = os.path.join(compressed_dir, video_name)
                        clips, labels = self._process_video(video_path, df)
                elif os.path.exists(tar_file):
                    # Extract video files directly from the tar archive and process them
                    with tarfile.open(tar_file, 'r') as tar:
                        for member in tar.getmembers():
                            if member.name.endswith('_ROVHD_Low.mp4'):
                                video_file = tar.extractfile(member)
                                clips, labels = self._process_video(video_file, df)
                else:
                    print(f"Neither Compressed directory nor Compressed.tar file found for dive: {dive_dir}")
        
        print(f"Total clips: {len(clips)}")
        print(f"Total labels: {len(labels)}")
        return clips, labels
    
    def _process_video(self, video_file, df):
        clips = []
        labels = []

        if isinstance(video_file, str):
            video_name = os.path.basename(video_file)
            video_timestamp = video_name.split('_')[2]
            cap = cv2.VideoCapture(video_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            video_name = video_file.name.lstrip('/')
            video_timestamp = video_name.split('_')[2]
            
            # Convert the video file to a supported format
            temp_dir = tempfile.mkdtemp()
            temp_video_path = os.path.join(temp_dir, 'temp_video.mp4')
            with open(temp_video_path, 'wb') as f:
                f.write(video_file.read())
            convert_video(temp_video_path, temp_video_path)
            
            # Read the converted video file
            cap = cv2.VideoCapture(temp_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Remove the temporary directory
            shutil.rmtree(temp_dir)

        video_start_time = datetime.datetime.strptime(video_timestamp, '%Y%m%dT%H%M%SZ')

        duration = frame_count / fps
        clip_frames = self.clip_length * fps
        num_clips = int(frame_count // clip_frames)

        # Specify the format of the datetime strings in the 'Start Date' and 'End Date' columns
        datetime_format = '%Y-%m-%d %H:%M:%S'
        
        # Convert 'Start Date' and 'End Date' columns to datetime
        df['Start Date'] = pd.to_datetime(df['Start Date'], format=datetime_format, errors='coerce')
        df['End Date'] = pd.to_datetime(df['End Date'], format=datetime_format, errors='coerce')
        
        # Drop rows with invalid datetime values
        df = df.dropna(subset=['Start Date', 'End Date'])

        for i in range(num_clips):
            start_frame = i * clip_frames
            end_frame = start_frame + clip_frames
            clip_end_time = video_start_time + timedelta(seconds=end_frame / fps)

            # Filter annotations based on Taxon Path and timestamp range
            filtered_df = df[(df['Start Date'] >= video_start_time) & (df['End Date'] <= clip_end_time)]
            filtered_df = filtered_df[filtered_df['Taxon Path'].str.contains('Biology / Organism')]

            # Validate timestamp range
            if not filtered_df.empty:
                annotation_start_time = filtered_df['Start Date'].min()
                annotation_end_time = filtered_df['End Date'].max()

                if annotation_start_time < video_start_time or annotation_end_time > clip_end_time:
                    print(f"Warning: Annotation timestamps outside clip range for {video_name}, clip {i}")
                    # Handle the misalignment based on your requirements

            highlight = len(filtered_df) > 0
            if isinstance(video_file, str):
                clips.append((video_file, start_frame, end_frame))
            else:
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    ret, frame = cap.read()
                    if ret:
                        clips.append((frame, start_frame, end_frame))
                    else:
                        print(f"Error: Failed to read frame from video for {video_name}, clip {i}")
                except:
                    print(f"Error: Failed to extract clip from video for {video_name}, clip {i}")
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
        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer
        
        # Get the output size of the ResNet model
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Highlight Detection Training')
    parser.add_argument('--video_dir', type=str, required=True, help='Path to the video directory')
    parser.add_argument('--csv_dir', type=str, required=True, help='Path to the CSV directory')
    args = parser.parse_args()

    video_dir = args.video_dir
    csv_dir = args.csv_dir

    print(f"Video directory: {video_dir}")
    print(f"CSV directory: {csv_dir}")

    # Initialize dataset with data augmentation
    transform = Compose([
        Resize((224, 224)),
        RandomCrop((200, 200)),
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Check if the extracted dataset exists
    if os.path.exists('extracted_dataset.pt'):
        # Load the saved dataset from disk
        dataset = torch.load('extracted_dataset.pt')
        print("Loaded extracted dataset from disk.")
    else:
        # Create a new instance of VideoDataset
        dataset = VideoDataset(
            video_dir=video_dir,
            csv_dir=csv_dir,
            clip_length=config['training']['clip_length'],
            transform=transform
        )
        print(f"Dataset created with {len(dataset)} clips")
        
        # Save the extracted dataset to disk
        torch.save(dataset, 'extracted_dataset.pt')
        print("Saved extracted dataset to disk.")

    # Perform k-fold cross-validation
    k = config['training']['k_folds']
    kfold = KFold(n_splits=k, shuffle=True)

    models = []

    for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold+1}')
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")

        train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

        print(f"Training dataloader created with {len(train_dataloader)} batches")
        print(f"Validation dataloader created with {len(val_dataloader)} batches")

        # Initialize model, loss, optimizer, and scheduler
        model = HighlightDetectionModel(
            hidden_dim=config['training']['hidden_dim'],
            num_layers=config['training']['num_layers']
        )
        criterion = nn.BCELoss()
        optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

        wandb.watch(model, log='all')

        best_val_accuracy = 0.0
        early_stopping_patience = config['training']['early_stopping_patience']
        early_stopping_counter = 0

        # Training loop with wandb logging
        for epoch in range(config['training']['num_epochs']):
            print(f'Epoch {epoch+1}/{config["training"]["num_epochs"]}')
            model.train()
            train_loss = 0.0
            for clips, labels in train_dataloader:
                optimizer.zero_grad()
                outputs = model(clips)
                labels = labels.view(-1, 1).float()
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_dataloader)

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for clips, labels in val_dataloader:
                    outputs = model(clips)
                    labels = labels.view(-1, 1).float()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    predicted = (outputs >= 0.5).float()
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= len(val_dataloader)
            val_accuracy = val_correct / val_total

            print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')

            # Log metrics to wandb
            wandb.log({
                "fold": fold + 1,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

            # Update learning rate based on validation accuracy
            scheduler.step(val_accuracy)

            # Save model checkpoint
            checkpoint_path = f"{config['paths']['model_save_path']}_fold_{fold+1}_checkpoint_{epoch+1}.pth"
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

            # Save the best model based on validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                early_stopping_counter = 0
                best_model_path = f"{config['paths']['model_save_path']}_fold_{fold+1}_best.pth"
                torch.save(model.state_dict(), best_model_path)
                print(f'Best model saved to {best_model_path}')
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

            # Save the best model for the current fold
            best_model_path = f"{config['paths']['model_save_path']}_fold_{fold+1}_best.pth"
            model.load_state_dict(torch.load(best_model_path))
            models.append(model)

        # Evaluate the ensemble model
        ensemble_correct = 0
        ensemble_total = 0
        with torch.no_grad():
            for clips, labels in val_dataloader:
                ensemble_outputs = torch.zeros(clips.size(0), 1)
                for model in models:
                    model.eval()
                    outputs = model(clips)
                    ensemble_outputs += outputs
                ensemble_outputs /= len(models)
                predicted = (ensemble_outputs >= 0.5).float()
                labels = labels.view(-1, 1).float()
                ensemble_correct += (predicted == labels).sum().item()
                ensemble_total += labels.size(0)

        ensemble_accuracy = ensemble_correct / ensemble_total
        print(f'Ensemble Accuracy: {ensemble_accuracy}')

        # Log the ensemble accuracy to wandb
        wandb.log({"ensemble_accuracy": ensemble_accuracy})

        # Save the ensemble model
        ensemble_model_path = config['paths']['model_save_path'] + '_ensemble.pth'
        torch.save(models, ensemble_model_path)
        print(f'Ensemble model saved to {ensemble_model_path}')

        # Finish the wandb run at the end of training
        wandb.finish()