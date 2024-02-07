import cv2
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torchvision.models import resnet50
from torch.optim import Adam
import numpy as np
import os
import toml
import wandb

wandb.init(project="video_highlight_detection", config=config['training'])
config_wandb = wandb.config

# Load configuration
config = toml.load('config.toml')

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

# Initialize dataset and dataloader with parameters from config
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = VideoDataset(
    video_dir=config['paths']['video_dir'],
    csv_dir=config['paths']['csv_dir'],
    clip_length=config['training']['clip_length'],
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

# Initialize model, loss, and optimizer with parameters from config
model = HighlightDetectionModel(
    hidden_dim=config['training']['hidden_dim'],
    num_layers=config['training']['num_layers']
)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])

wandb.watch(model, log='all')

# Training loop with model saving, checkpointing, and wandb logging
for epoch in range(config['training']['num_epochs']):
    for clips, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(clips)
        labels = labels.view(-1, 1).float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    # Log metrics to wandb
    wandb.log({"epoch": epoch + 1, "loss": loss.item()})

    # Save model checkpoint as before
    checkpoint_path = f"{config['paths']['model_save_path']}_checkpoint_{epoch+1}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

# Finish the wandb run at the end of training
wandb.finish()