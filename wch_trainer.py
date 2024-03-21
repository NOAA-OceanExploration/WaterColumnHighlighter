import cv2
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, ColorJitter
from torchvision.models import resnet50
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import toml
import wandb
import boto3
from sklearn.model_selection import KFold

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
        for video_name in os.listdir(self.video_dir):
            if not video_name.endswith('.mp4'):
                continue
            video_path = os.path.join(self.video_dir, video_name)
            csv_path = os.path.join(self.csv_dir, video_name.replace('.mp4', '.csv'))
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            clip_frames = self.clip_length * fps
            num_clips = int(frame_count // clip_frames)
            for i in range(num_clips):
                start_frame = i * clip_frames
                end_frame = start_frame + clip_frames
                highlight = any((df['start'] <= start_frame/fps) & (df['end'] >= end_frame/fps))
                clips.append((video_path, start_frame, end_frame))
                labels.append(int(highlight))
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
        self.lstm = nn.LSTM(self.resnet.fc.in_features, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.resnet(x)
        x = x.view(batch_size, timesteps, -1)
        x, (hn, cn) = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return torch.sigmoid(x)

# Initialize dataset with data augmentation
transform = Compose([
    Resize((224, 224)),
    RandomCrop((200, 200)),
    RandomHorizontalFlip(),
    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = VideoDataset(
    video_dir=config['paths']['video_dir'],
    csv_dir=config['paths']['csv_dir'],
    clip_length=config['training']['clip_length'],
    transform=transform
)

# Perform k-fold cross-validation
k = config['training']['k_folds']
kfold = KFold(n_splits=k, shuffle=True)

models = []

for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
    print(f'Fold {fold+1}')
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

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

        # Save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            early_stopping_counter = 0
            best_model_path = f"{config['paths']['model_save_path']}_fold_{fold+1}_best.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved to {best_model_path}')

            # Save the best model to S3 bucket
            s3 = boto3.client('s3')
            s3.upload_file(best_model_path, config['paths']['s3_bucket'], os.path.basename(best_model_path))
            print(f'Best model uploaded to S3 bucket: {config["paths"]["s3_bucket"]}')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # Save the best model for the current fold
    best_model_path = f"{config['paths']['model_save_path']}_fold_{fold+1}_best.pth"
    model.load_state_dict(torch.load(best_model_path))
    models.append(model)

    # Upload the best model to S3 bucket
    s3.upload_file(best_model_path, config['paths']['s3_bucket'], os.path.basename(best_model_path))
    print(f'Best model for fold {fold+1} uploaded to S3 bucket: {config["paths"]["s3_bucket"]}')

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

# Upload the ensemble model to S3 bucket
s3.upload_file(ensemble_model_path, config['paths']['s3_bucket'], os.path.basename(ensemble_model_path))
print(f'Ensemble model uploaded to S3 bucket: {config["paths"]["s3_bucket"]}')

# Finish the wandb run at the end of training
wandb.finish()
