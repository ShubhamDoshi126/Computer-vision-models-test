import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import defaultdict
import os
import argparse
from datetime import datetime
import json

class CCTVActivityDetector:
    def __init__(self, detection_model_path="yolov8n.pt", confidence_threshold=0.5):
        """
        Initialize the CCTV Activity Detector
        
        Args:
            detection_model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detections
        """
        # Load YOLO model for person detection
        self.detector = YOLO(detection_model_path)
        self.confidence_threshold = confidence_threshold
        
        # Track persons across frames
        self.person_tracks = defaultdict(list)
        self.track_id_counter = 0
        
        # Activity classification based on movement patterns
        self.activity_classes = {
            'standing': 'Standing',
            'walking': 'Walking',
            'running': 'Running',
            'sitting': 'Sitting',
            'lying': 'Lying Down',
            'unknown': 'Unknown'
        }
        
        # Colors for different activities
        self.activity_colors = {
            'Standing': (0, 255, 0),    # Green
            'Walking': (255, 165, 0),   # Orange
            'Running': (255, 0, 0),     # Red
            'Sitting': (0, 0, 255),     # Blue
            'Lying Down': (128, 0, 128), # Purple
            'Unknown': (128, 128, 128)   # Gray
        }
    
    def calculate_movement_activity(self, current_box, previous_boxes, frame_count):
        """
        Determine activity based on bounding box movement patterns
        
        Args:
            current_box: Current bounding box [x1, y1, x2, y2]
            previous_boxes: List of previous bounding boxes
            frame_count: Current frame number
            
        Returns:
            Predicted activity string
        """
        if len(previous_boxes) < 3:
            return 'Unknown'
        
        # Calculate movement metrics
        movements = []
        aspect_ratios = []
        
        for i in range(min(5, len(previous_boxes))):
            prev_box = previous_boxes[-(i+1)]
            
            # Calculate center movement
            curr_center = ((current_box[0] + current_box[2]) / 2, 
                          (current_box[1] + current_box[3]) / 2)
            prev_center = ((prev_box[0] + prev_box[2]) / 2, 
                          (prev_box[1] + prev_box[3]) / 2)
            
            movement = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                             (curr_center[1] - prev_center[1])**2)
            movements.append(movement)
            
            # Calculate aspect ratio (height/width)
            height = current_box[3] - current_box[1]
            width = current_box[2] - current_box[0]
            aspect_ratio = height / width if width > 0 else 0
            aspect_ratios.append(aspect_ratio)
        
        avg_movement = np.mean(movements)
        avg_aspect_ratio = np.mean(aspect_ratios)
        
        # Activity classification logic
        if avg_movement < 5:
            if avg_aspect_ratio > 2.0:
                return 'Standing'
            elif avg_aspect_ratio < 1.2:
                return 'Lying Down'
            else:
                return 'Sitting'
        elif avg_movement < 15:
            return 'Walking'
        else:
            return 'Running'
    
    def draw_annotations(self, frame, detections, frame_count):
        """
        Draw bounding boxes and activity labels on frame
        
        Args:
            frame: Input frame
            detections: Detection results
            frame_count: Current frame number
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, detection['bbox'])
            confidence = detection['confidence']
            activity = detection['activity']
            person_id = detection['person_id']
            
            # Get color for activity
            color = self.activity_colors.get(activity, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label text
            label = f"Person {person_id}: {activity} ({confidence:.2f})"
            
            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - text_height - 10),
                         (x1 + text_width, y1),
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 2)
        
        # Add frame info
        info_text = f"Frame: {frame_count} | People: {len(detections)}"
        cv2.putText(annotated_frame, info_text,
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   (255, 255, 255), 2)
        
        return annotated_frame
    
    def process_video(self, video_path, output_path=None, save_detections=False):
        """
        Process CCTV video and generate annotated output
        
        Args:
            video_path: Path to input video
            output_path: Path for output video (optional)
            save_detections: Whether to save detection data to JSON
            
        Returns:
            Dictionary with processing results
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
        # Setup output video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Storage for detection results
        all_detections = []
        frame_count = 0
        
        print("Starting video processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run YOLO detection
            results = self.detector(frame, conf=self.confidence_threshold)
            
            # Process detections
            frame_detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Filter for person class (class 0 in COCO dataset)
                        if int(box.cls[0]) == 0:  # Person class
                            bbox = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0])
                            
                            # Simple tracking (in production, use more sophisticated tracking)
                            person_id = self.assign_person_id(bbox)
                            
                            # Store current detection for activity analysis
                            if person_id not in self.person_tracks:
                                self.person_tracks[person_id] = []
                            
                            self.person_tracks[person_id].append(bbox)
                            
                            # Keep only recent detections for activity analysis
                            if len(self.person_tracks[person_id]) > 10:
                                self.person_tracks[person_id] = self.person_tracks[person_id][-10:]
                            
                            # Determine activity
                            activity = self.calculate_movement_activity(
                                bbox, self.person_tracks[person_id][:-1], frame_count
                            )
                            
                            detection = {
                                'frame': frame_count,
                                'person_id': person_id,
                                'bbox': bbox.tolist(),
                                'confidence': confidence,
                                'activity': activity,
                                'timestamp': frame_count / fps
                            }
                            
                            frame_detections.append(detection)
                            all_detections.append(detection)
            
            # Draw annotations
            annotated_frame = self.draw_annotations(frame, frame_detections, frame_count)
            
            # Write output frame
            if output_path:
                out.write(annotated_frame)
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        
        print(f"Video processing completed! Processed {frame_count} frames")
        
        # Save detection results
        if save_detections:
            json_path = video_path.replace('.mp4', '_detections.json')
            with open(json_path, 'w') as f:
                json.dump(all_detections, f, indent=2)
            print(f"Detection data saved to: {json_path}")
        
        # Generate summary statistics
        summary = self.generate_summary(all_detections)
        
        return {
            'total_frames': frame_count,
            'total_detections': len(all_detections),
            'unique_persons': len(set(d['person_id'] for d in all_detections)),
            'summary': summary,
            'output_path': output_path
        }
    
    def assign_person_id(self, bbox):
        """
        Simple person ID assignment based on bounding box similarity
        In production, use more sophisticated tracking algorithms
        """
        # Simple centroid-based tracking
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        # Find closest existing track
        min_distance = float('inf')
        best_id = None
        
        for person_id, track in self.person_tracks.items():
            if track:
                last_bbox = track[-1]
                last_center = ((last_bbox[0] + last_bbox[2]) / 2, 
                              (last_bbox[1] + last_bbox[3]) / 2)
                
                distance = np.sqrt((center[0] - last_center[0])**2 + 
                                 (center[1] - last_center[1])**2)
                
                if distance < min_distance and distance < 100:  # Threshold for same person
                    min_distance = distance
                    best_id = person_id
        
        if best_id is None:
            # Create new ID
            self.track_id_counter += 1
            best_id = self.track_id_counter
        
        return best_id
    
    def generate_summary(self, detections):
        """Generate summary statistics"""
        if not detections:
            return {}
        
        # Activity distribution
        activity_counts = defaultdict(int)
        person_activities = defaultdict(set)
        
        for detection in detections:
            activity_counts[detection['activity']] += 1
            person_activities[detection['person_id']].add(detection['activity'])
        
        return {
            'activity_distribution': dict(activity_counts),
            'persons_detected': len(person_activities),
            'total_detection_frames': len(detections),
            'most_common_activity': max(activity_counts, key=activity_counts.get) if activity_counts else None
        }

def main():
    parser = argparse.ArgumentParser(description='CCTV Human Activity Detection')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--model', '-m', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--save-json', action='store_true', help='Save detection data to JSON')
    
    args = parser.parse_args()
    
    # Create detector
    detector = CCTVActivityDetector(
        detection_model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    # Generate output path if not provided
    if not args.output:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_annotated.mp4"
    
    try:
        # Process video
        results = detector.process_video(
            video_path=args.input,
            output_path=args.output,
            save_detections=args.save_json
        )
        
        # Print results
        print("\n" + "="*50)
        print("PROCESSING RESULTS")
        print("="*50)
        print(f"Total frames processed: {results['total_frames']}")
        print(f"Total detections: {results['total_detections']}")
        print(f"Unique persons detected: {results['unique_persons']}")
        print(f"Output video saved: {results['output_path']}")
        
        if results['summary']:
            print(f"\nActivity Distribution:")
            for activity, count in results['summary']['activity_distribution'].items():
                print(f"  {activity}: {count} detections")
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())