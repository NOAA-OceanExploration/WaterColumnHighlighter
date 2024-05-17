import os
import tempfile
import secrets
from typing import List
from fastapi import FastAPI, UploadFile, File, Request, Depends, HTTPException, WebSocket
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import cv2
import numpy as np
from starlette.status import HTTP_401_UNAUTHORIZED
from starlette.websockets import WebSocketDisconnect
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
import asyncio

app = FastAPI()

# Create the "static" directory if it doesn't exist
static_dir = "static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory="templates")

security = HTTPBasic()

# Replace with your desired username
USERNAME = "admin"

# Retrieve the password from the environment variable
PASSWORD = "test"
print(f"Retrieved password: {PASSWORD}")

model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

ocean_labels = [
    "fish, shark, whale, dolphin, octopus, squid, jellyfish, stingray, eel, crab, lobster, shrimp, clam, oyster, mussel, snail, starfish, sea urchin, coral, anemone, sponge, seaweed, kelp, plankton, sea turtle, sea lion, seal, penguin, seagull, pelican, albatross, sea snake, sea slug, nudibranch, sea spider, sea cucumber, sea squirt, sea lily, sea fan, sea pen, sea whip, sea mouse, sea butterfly, sea angel, sea walnut, sea gooseberry, ctenophore, salp, pteropod, heteropod, copepod, krill, amphipod, isopod, barnacle, bryozoan, hydrozoan"
]

class DetectionArea(BaseModel):
    start_time: float
    end_time: float

class DetectionResult(BaseModel):
    detections: List[DetectionArea]
    detections_url: str

sessions = {}

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, username: str = Depends(get_current_username)):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    sessions[session_id] = websocket
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received WebSocket message: {data}")
            if data == "start_detection":
                await websocket.send_text("Detection started")
            elif data.startswith("progress:"):
                progress = int(data.split(":")[1])
                print(f"Sending progress update: {progress}%")
                await websocket.send_text(f"Progress: {progress}%")
    except WebSocketDisconnect:
        print("WebSocket disconnected")
        del sessions[session_id]

@app.post("/detect/{session_id}")
async def detect_objects(session_id: str, video: UploadFile = File(...), username: str = Depends(get_current_username)):
    print("Received request to detect objects in video.")

    # Save the uploaded video file to a temporary location
    print("Saving video file to temporary location...")
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        temp_video.write(video.file.read())
        temp_video_path = temp_video.name
    print(f"Video file saved to: {temp_video_path}")

    # Load the video file to process it
    print("Loading video file...")
    cap = cv2.VideoCapture(temp_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print(f"Video loaded. FPS: {fps}, Frame Count: {frame_count}, Duration: {duration}s")

    detections = []
    print("Processing video frames...")
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_idx}")
            break

        if frame_idx % int(fps * 5) == 0:  # Process every 5 seconds
            print(f"Processing frame {frame_idx}...")
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(images=image, text=ocean_labels, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]]
            )

            print(f"Detection results for frame {frame_idx}: {results}")

            for result in results:
                print(f"Scores for frame {frame_idx}: {result['scores']}")
                if any(score > 0.3 for score in result["scores"]):
                    start_time = max(0, frame_idx / fps - 2.5)
                    end_time = min(frame_idx / fps + 2.5, duration)
                    detections.append(DetectionArea(start_time=start_time, end_time=end_time))
                    print(f"Detection added: Start: {start_time:.2f}s, End: {end_time:.2f}s")

            # Save the frame as an image file for inspection
            cv2.imwrite(f"frame_{frame_idx}.jpg", frame)

        # Send progress update to the frontend
        progress = int((frame_idx + 1) / frame_count * 100)
        if session_id in sessions:
            print(f"Sending progress update: {progress}%")
            await sessions[session_id].send_text(f"progress:{progress}")
            await asyncio.sleep(0)  # Yield control to the event loop to ensure the message is sent immediately

    # Release resources and clean up
    print("Releasing video file...")
    cap.release()
    os.remove(temp_video_path)

    print("Merging overlapping detections...")
    # Merge overlapping detections
    merged_detections = []
    for detection in detections:
        merged = False
        for merged_detection in merged_detections:
            if detection.start_time <= merged_detection.end_time:
                merged_detection.start_time = min(merged_detection.start_time, detection.start_time)
                merged_detection.end_time = max(merged_detection.end_time, detection.end_time)
                merged = True
                break
        if not merged:
            merged_detections.append(detection)

    print(f"Total detections: {len(detections)}")
    print(f"Merged detections: {len(merged_detections)}")

    print("Generating timeline visualization...")
    # Create a figure and axis for the timeline visualization
    fig, ax = plt.subplots(figsize=(24, 8), dpi=300)

    # Set the x-axis limits based on the video duration
    ax.set_xlim(0, duration)

    # Set the y-axis limits
    ax.set_ylim(0, 1)

    # Remove the y-axis ticks and labels
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Set the x-axis label and tick marks
    ax.set_xlabel("Time (seconds)")
    ax.xaxis.set_major_locator(MultipleLocator(10))  # Set major tick marks every 10 seconds
    ax.xaxis.set_minor_locator(MultipleLocator(1))   # Set minor tick marks every 1 second
    ax.grid(True, which='major', linestyle='-', alpha=0.7)
    ax.grid(True, which='minor', linestyle='--', alpha=0.3)

    # Add rectangles for each detection
    for detection in merged_detections:
        start_time = detection.start_time
        end_time = detection.end_time
        rect = patches.Rectangle((start_time, 0), end_time - start_time, 1, linewidth=1, edgecolor='g', facecolor='g', alpha=0.7)
        ax.add_patch(rect)

    # Set the plot title with relevant information
    ax.set_title(f"Timeline Visualization - Clip: {video.filename}, Sensitivity: box_threshold={0.4}, text_threshold={0.3}")

    # Save the timeline visualization image
    timeline_filename = os.path.join(static_dir, f"timeline_{secrets.token_hex(6)}.png")
    plt.tight_layout()
    plt.savefig(timeline_filename)
    print(f"Timeline visualization saved to: {timeline_filename}")

    print("Generating detections text file...")
    # Create and save the detection results text file
    detections_text = "Timecode areas requiring attention:\n\n"
    for idx, detection in enumerate(detections, start=1):
        detections_text += f"Detection {idx}:\n"
        detections_text += f"  Start: {detection.start_time:.2f}s\n"
        detections_text += f"  End: {detection.end_time:.2f}s\n\n"
    detections_text_filename = os.path.join(static_dir, f"detections_{secrets.token_hex(6)}.txt")
    with open(detections_text_filename, 'w') as file:
        file.write(detections_text)
    detections_url = f"/static/{os.path.basename(detections_text_filename)}"
    print(f"Detections text file saved to: {detections_text_filename}")
    print(f"Detections URL: {detections_url}")

    print("Returning detection results...")
    return FileResponse(timeline_filename, media_type="image/png", filename="timeline.png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
