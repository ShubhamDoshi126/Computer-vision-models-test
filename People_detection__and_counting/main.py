# Import required libraries
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP conflict
import cv2
from ultralytics import YOLO
import cvzone

# Configuration
INPUT_VIDEO = "people_id_And_counting_test.mp4"  # Your video file
OUTPUT_VIDEO = "output_annotated.mp4"  # Output video with annotations
COUNTING_LINE_X = 443  # Vertical line X position for counting
CONFIDENCE_THRESHOLD = 0.5  # Detection confidence threshold
RESIZE_WIDTH = 1020
RESIZE_HEIGHT = 600

def main():
    # Load YOLOv8 model
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')  # Using yolov8n.pt (nano version)
    names = model.names
    
    # Track previous center positions
    hist = {}
    
    # IN/OUT counters
    in_count = 0
    out_count = 0
    
    # Open input video
    print(f"Opening video: {INPUT_VIDEO}")
    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {INPUT_VIDEO}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {fps} FPS, {total_frames} frames")
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (RESIZE_WIDTH, RESIZE_HEIGHT))
    
    frame_count = 0
    processed_frames = 0
    
    print("Processing video... (Display disabled for faster processing)")
    print("Processing will run in background - check console for progress")
    
    while True:
        # Read video frame
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip every other frame for faster processing
        if frame_count % 2 != 0:
            continue
        
        processed_frames += 1
        
        # Resize frame
        frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        
        # Detect and track persons (class 0)
        results = model.track(frame, persist=True, classes=[0], conf=CONFIDENCE_THRESHOLD)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, class_id, conf in zip(boxes, ids, class_ids, confidences):
                x1, y1, x2, y2 = box
                c = names[class_id]
                
                # Calculate center point
                cx = int((x1 + x2) // 2)
                cy = int((y1 + y2) // 2)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                
                # Add labels
                cvzone.putTextRect(frame, f'{c.upper()}', (x1, y1 - 10), 
                                 scale=1, thickness=1, colorT=(255, 255, 255), 
                                 colorR=(0, 255, 0), offset=5, border=2)
                cvzone.putTextRect(frame, f'ID: {track_id}', (x1, y2 + 10), 
                                 scale=1, thickness=2, colorT=(255, 255, 255), 
                                 colorR=(0, 0, 255), offset=5, border=2)
                cvzone.putTextRect(frame, f'Conf: {conf:.2f}', (x1, y2 + 40), 
                                 scale=0.8, thickness=1, colorT=(255, 255, 255), 
                                 colorR=(255, 0, 0), offset=5, border=2)
                
                # Check line crossing for counting
                if track_id in hist:
                    prev_cx, prev_cy = hist[track_id]
                    
                    # Draw tracking line
                    cv2.line(frame, (prev_cx, prev_cy), (cx, cy), (255, 0, 255), 2)
                    
                    # Check crossing direction
                    if prev_cx < COUNTING_LINE_X <= cx:
                        in_count += 1
                        print(f"Person {track_id} crossed IN. Total IN: {in_count}")
                    elif prev_cx > COUNTING_LINE_X >= cx:
                        out_count += 1
                        print(f"Person {track_id} crossed OUT. Total OUT: {out_count}")
                
                # Update history
                hist[track_id] = (cx, cy)
        
        # Draw counting line
        cv2.line(frame, (COUNTING_LINE_X, 0), (COUNTING_LINE_X, frame.shape[0]), (0, 0, 255), 3)
        
        # Add line label
        cvzone.putTextRect(frame, 'COUNTING LINE', (COUNTING_LINE_X - 80, 30), 
                         scale=1, thickness=2, colorT=(255, 255, 255), 
                         colorR=(0, 0, 255), offset=5, border=2)
        
        # Display counts
        cvzone.putTextRect(frame, f'IN: {in_count}', (40, 60), 
                         scale=2, thickness=2, colorT=(255, 255, 255), 
                         colorR=(0, 128, 0))
        cvzone.putTextRect(frame, f'OUT: {out_count}', (40, 100), 
                         scale=2, thickness=2, colorT=(255, 255, 255), 
                         colorR=(0, 0, 255))
        cvzone.putTextRect(frame, f'TOTAL: {in_count + out_count}', (40, 140), 
                         scale=2, thickness=2, colorT=(255, 255, 255), 
                         colorR=(255, 0, 0))
        
        # Add frame info
        cvzone.putTextRect(frame, f'Frame: {processed_frames}/{total_frames//2}', 
                         (RESIZE_WIDTH - 200, 60), scale=1, thickness=1, 
                         colorT=(255, 255, 255), colorR=(128, 128, 128))
        
        # Write frame to output video
        out.write(frame)
        
        # Optional: Display frame (disable for faster processing)
        # cv2.imshow("People Counter", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
        # Progress indicator
        if processed_frames % 30 == 0:
            progress = (processed_frames / (total_frames // 2)) * 100
            print(f"Progress: {progress:.1f}% - Frame {processed_frames}/{total_frames//2}")
    
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print("\n" + "="*50)
    print("PROCESSING COMPLETE!")
    print("="*50)
    print(f"Input video: {INPUT_VIDEO}")
    print(f"Output video: {OUTPUT_VIDEO}")
    print(f"Total frames processed: {processed_frames}")
    print(f"People counted IN: {in_count}")
    print(f"People counted OUT: {out_count}")
    print(f"Total people detected: {in_count + out_count}")
    print(f"Net count (IN - OUT): {in_count - out_count}")
    
    if os.path.exists(OUTPUT_VIDEO):
        file_size = os.path.getsize(OUTPUT_VIDEO) / (1024 * 1024)  # MB
        print(f"Output file size: {file_size:.2f} MB")
        print(f"Output saved successfully: {OUTPUT_VIDEO}")
    else:
        print("Warning: Output file was not created successfully")

if __name__ == "__main__":
    main()