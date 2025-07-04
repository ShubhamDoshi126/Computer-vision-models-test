# Fire and Smoke Detection using YOLOv8

A computer vision project that uses YOLOv8 to detect fire and smoke in video footage. This implementation processes video files frame-by-frame and outputs annotated videos with detected objects highlighted.

## Features

- **Real-time Detection**: Frame-by-frame analysis of video files
- **Custom Model**: Uses a trained YOLOv8 model (`best.pt`) specifically for fire and smoke detection
- **Video Processing**: Processes entire video files and saves annotated output
- **Configurable Parameters**: Adjustable confidence threshold and IoU settings
- **Progress Tracking**: Real-time frame processing feedback

## Requirements

```bash
pip install ultralytics opencv-python
```

## Project Structure

```
Fire and smoke detection/
├── main.py           # Main detection script
├── best.pt           # Custom trained YOLOv8 model for fire/smoke detection
├── yolov8n.pt        # YOLOv8 nano pretrained model
└── README.md         # This file
```

## Usage

1. **Prepare your video file**: Place your input video in the project directory
2. **Update the input path**: Modify the `input_path` variable in `main.py` to point to your video file
3. **Run the detection**:
   ```bash
   python main.py
   ```
4. **Check the output**: The annotated video will be saved as `annotated_output.mp4`

## Configuration

You can modify the following parameters in `main.py`:

- **Input file**: Change `input_path` to your video file path
- **Confidence threshold**: Adjust `conf` parameter (default: 0.3)
- **IoU threshold**: Adjust `iou` parameter (default: 0.5)
- **Output file**: Change the output filename in `cv2.VideoWriter()`

```python
# Example configuration
input_path = "your_video.mp4"           # Input video file
results = model.predict(frame, conf=0.3, iou=0.5)  # Detection parameters
```

## Model Information

- **Model Type**: YOLOv8 (You Only Look Once version 8)
- **Custom Model**: `best.pt` - Specifically trained for fire and smoke detection
- **Backup Model**: `yolov8n.pt` - YOLOv8 nano pretrained model
- **Detection Classes**: Fire and smoke objects

## Output

The script generates:
- **Annotated Video**: `annotated_output.mp4` with bounding boxes and labels
- **Console Output**: Frame-by-frame processing progress
- **Detection Results**: Confidence scores and object classifications

## Performance Notes

- Processing time depends on video length and resolution
- Higher confidence thresholds reduce false positives but may miss detections
- GPU acceleration is automatically used if available (CUDA-compatible GPU)

## Troubleshooting

### Common Issues

1. **Model file not found**:
   - Ensure `best.pt` is in the project directory
   - Check file permissions

2. **Video file not loading**:
   - Verify the input video path is correct
   - Ensure the video format is supported by OpenCV

3. **Poor detection accuracy**:
   - Adjust confidence threshold (`conf` parameter)
   - Try different IoU threshold values
   - Ensure lighting conditions are adequate

4. **Slow processing**:
   - Consider resizing input video for faster processing
   - Use GPU acceleration if available

## Technical Details

### Dependencies
- **OpenCV (cv2)**: Video processing and frame manipulation
- **Ultralytics**: YOLOv8 implementation and model handling

### Video Processing Pipeline
1. Load video file using OpenCV VideoCapture
2. Initialize video writer for output
3. Process each frame through YOLOv8 model
4. Annotate frames with detection results
5. Write annotated frames to output video
6. Release resources and save final video

## Future Enhancements

- [ ] Real-time webcam detection
- [ ] Multiple video format support
- [ ] Batch processing for multiple videos
- [ ] Integration with alert systems
- [ ] Performance optimization
- [ ] GUI interface

## License

This project is for educational and research purposes. Please ensure compliance with applicable laws and regulations when using for commercial purposes.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this fire and smoke detection system.

---

**Note**: This implementation is designed for demonstration purposes. For production use in safety-critical applications, additional validation and testing are recommended.
