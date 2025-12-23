# Detection and Tracking Modules

Simple, clean modules for object detection and tracking in video sequences.

## Structure

```
src/
├── detector/          # Object detection module
│   ├── __init__.py
│   └── detector.py    # ObjectDetector class
├── tracker/           # Object tracking module
│   ├── __init__.py
│   └── tracker.py     # ObjectTracker and Track classes
└── demo.ipynb         # Complete demonstration notebook
```

## Quick Start

### Detection

```python
from detector import ObjectDetector

# Initialize with parameters
detector = ObjectDetector(
    history=500,
    dist_threshold=800,
    min_area=30,
    max_area=100000
)

# Detect objects in a frame
detections = detector.detect(frame)
# Returns: [[cx, cy, w, h], [cx, cy, w, h], ...]
```

### Tracking

```python
from tracker import ObjectTracker

# Initialize tracker
tracker = ObjectTracker(
    max_distance=30.0,
    min_hits=3,
    max_age=30
)

# Update with detections
tracks = tracker.update(detections)
confirmed = tracker.get_confirmed_tracks()

# Access track info
for track in confirmed:
    print(f"ID: {track.id}, Position: {track.prediction}")
    print(f"History: {track.history}")
```

## ObjectDetector Parameters

- `history`: Number of frames for background learning (default: 500)
- `dist_threshold`: Distance threshold for foreground detection (default: 800)
- `min_area`: Minimum object area (default: 30)
- `max_area`: Maximum object area (default: 100000)
- `erode_size`: Erosion kernel size (default: (3, 3))
- `dilate_size`: Dilation kernel size (default: (8, 8))
- `min_aspect_ratio`: Minimum width/height ratio (default: 0.4)
- `max_aspect_ratio`: Maximum width/height ratio (default: 2.5)
- `detect_shadows`: Whether to detect shadows (default: False)

## ObjectTracker Parameters

- `max_distance`: Maximum distance for matching detection to track (default: 30.0)
- `min_hits`: Minimum hits before track is confirmed (default: 3)
- `max_age`: Maximum frames without detection before deletion (default: 30)
- `max_history`: Maximum trajectory history length (default: 20)

## Complete Example

See `demo.ipynb` for a complete working example with visualization.

```python
import cv2
import glob
from detector import ObjectDetector
from tracker import ObjectTracker

# Initialize
detector = ObjectDetector()
tracker = ObjectTracker()

# Load frames
files = sorted(glob.glob("../data/images/*/blurred_frames/*.png"))

# Process
for i, file_path in enumerate(files):
    frame = cv2.imread(file_path)
    
    # Detect and track
    detections = detector.detect(frame)
    tracks = tracker.update(detections)
    confirmed = tracker.get_confirmed_tracks()
    
    print(f"Frame {i}: {len(confirmed)} tracks")
```

## Requirements

- opencv-python (cv2)
- numpy
- scipy (for tracker's Hungarian algorithm)

Install with:
```bash
pip install opencv-python numpy scipy
```

## Notes

- Both modules are simple and focused on core functionality
- Easy to integrate into larger pipelines
- Parameters can be tuned for different use cases
- Use `reset()` method to process multiple videos

