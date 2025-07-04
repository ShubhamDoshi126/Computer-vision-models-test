import cv2
from ultralytics import YOLO

# Load YOLOv8n pretrained model
model = YOLO("best.pt")  # This will auto-download if not present locally

# Load input video file
input_path = "fire_test2.mp4"    # 🔁 Replace with your actual file
cap = cv2.VideoCapture(input_path)

# Setup output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("annotated_output.mp4", fourcc, fps, (width, height))

# Frame-by-frame detection
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    print(f"Processing frame {frame_count}...")

    # Run object detection (will detect general objects, not specifically fire/smoke)
    results = model.predict(frame, conf=0.3, iou=0.5)  # Adjust confidence threshold if needed
    annotated_frame = results[0].plot()  # Get annotated frame with boxes/labels

    # Write frame to output video
    out.write(annotated_frame)

# Cleanup
cap.release()
out.release()
print("✅ Detection complete. Output saved to: annotated_output.mp4")