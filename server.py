import os
import cv2
import torch
import pandas as pd
from flask import Flask, request, jsonify, render_template
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from wch_trainer import HighlightDetectionModel

app = Flask(__name__)

# Load the trained model
model_path = 'path/to/your/trained/model.pth'
model = HighlightDetectionModel(hidden_dim=512, num_layers=2)  # Adjust based on your model architecture
model.load_state_dict(torch.load(model_path))
model.eval()

# Define the video transformation pipeline
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'})

    video_file = request.files['video']
    video_path = 'uploads/' + video_file.filename
    video_file.save(video_path)

    # Process the video and generate clips
    clips = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_length = 10  # Adjust based on your requirements
    clip_frames = clip_length * fps
    num_clips = int(frame_count // clip_frames)

    for i in range(num_clips):
        start_frame = i * clip_frames
        end_frame = start_frame + clip_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(frame)
            frames.append(frame)
        clips.append(torch.stack(frames))

    cap.release()

    # Perform inference on the clips
    highlights = []
    with torch.no_grad():
        for clip in clips:
            output = model(clip.unsqueeze(0))
            highlight = int(output.item() >= 0.5)
            highlights.append(highlight)

    # Generate the CSV file
    csv_data = {'Clip': list(range(1, len(highlights) + 1)), 'Highlight': highlights}
    df = pd.DataFrame(csv_data)
    csv_path = 'results/' + os.path.splitext(video_file.filename)[0] + '_results.csv'
    df.to_csv(csv_path, index=False)

    return jsonify({'csv_url': csv_path})

if __name__ == '__main__':
    app.run()