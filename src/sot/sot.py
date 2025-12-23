"""Small Object Tracker (SOT) - Main pipeline class."""
import cv2
import numpy as np
import glob
from typing import List, Dict, Any, Optional, Callable
from .pipeline import Pipeline, PipelineStep
from .steps import DetectionStep, TrackingStep, VisualizationStep


class SOT:
    """Small Object Tracker with configurable pipeline.
    
    Args:
        detector_params: Parameters for detector
        tracker_params: Parameters for tracker
        auto_setup: Automatically setup basic detection+tracking pipeline
    
    Example:
        # Basic usage
        sot = SOT()
        results = sot.process_video("frames/*.jpg", "output.mp4")
        
        # Custom pipeline
        sot = SOT(auto_setup=False)
        sot.pipeline.add_step(MyCustomStep())
        sot.pipeline.add_step(DetectionStep())
        sot.pipeline.add_step(TrackingStep())w
    """
    
    def __init__(
        self,
        detector_params: Dict[str, Any] = None,
        tracker_params: Dict[str, Any] = None,
        auto_setup: bool = True
    ):
        self.pipeline = Pipeline()
        
        self.detector_params = detector_params or {}
        self.tracker_params = tracker_params or {}
        
        if auto_setup:
            self._setup_default_pipeline()
    
    def _setup_default_pipeline(self):
        self.add_detection()
        self.add_tracking()
        self.add_visualization()
    
    def add_detection(self, **params):
        detector_params = {**self.detector_params, **params}
        self.pipeline.add_step(DetectionStep(**detector_params))
        return self
    
    def add_tracking(self, **params):
        tracker_params = {**self.tracker_params, **params}
        self.pipeline.add_step(TrackingStep(**tracker_params))
        return self
    
    def add_visualization(self, **params):
        self.pipeline.add_step(VisualizationStep(**params))
        return self
    
    def add_step(self, step: PipelineStep):
        self.pipeline.add_step(step)
        return self
    
    def remove_step(self, name: str):
        self.pipeline.remove_step(name)
        return self
    
    def process_frame(self, frame: np.ndarray, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a single frame through the pipeline.
        
        Args:
            frame: Input frame
            context: Optional context from previous frame
        
        Returns:
            Context dictionary with results
        """
        return self.pipeline.process(frame, context)
    
    def process_video(
        self,
        input_path: str,
        output_path: str = None,
        num_frames: int = None,
        fps: int = 30,
        callback: Callable[[int, Dict[str, Any]], None] = None
    ) -> Dict[str, Any]:
        """Process video frames through pipeline.
        
        Args:
            input_path: Path to frames (glob pattern) or video file
            output_path: Path to save output video (optional)
            num_frames: Maximum number of frames to process
            fps: Output video FPS
            callback: Function called after each frame: callback(frame_idx, context)
        
        Returns:
            Dictionary with processing statistics
        """
        if '*' in input_path:
            files = sorted(glob.glob(input_path))
        else:
            cap = cv2.VideoCapture(input_path)
            files = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                files.append(frame)
            cap.release()
        
        if not files:
            raise ValueError(f"No frames found at {input_path}")
        
        if num_frames:
            files = files[:num_frames]
        
        print(f"Processing {len(files)} frames...")
        
        writer = None
        if output_path:
            if isinstance(files[0], str):
                sample = cv2.imread(files[0])
            else:
                sample = files[0]
            h, w = sample.shape[:2]
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        self.pipeline.reset()
        
        stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'max_tracks': 0
        }
        
        context = {}
        
        for i, file_or_frame in enumerate(files):
            if isinstance(file_or_frame, str):
                frame = cv2.imread(file_or_frame)
            else:
                frame = file_or_frame
            
            if frame is None:
                continue
            
            # Add frame number to context
            context['frame_number'] = i
            
            # Process through pipeline
            context = self.process_frame(frame, context)
            
            # Update stats
            stats['frames_processed'] += 1
            stats['total_detections'] += context.get('num_detections', 0)
            num_confirmed = context.get('num_confirmed', 0)
            stats['total_tracks'] = context.get('num_tracks', 0)
            stats['max_tracks'] = max(stats['max_tracks'], num_confirmed)
            
            if writer and 'visualization' in context:
                writer.write(context['visualization'])
            
            if callback:
                callback(i, context)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(files)} frames...")
        
        if writer:
            writer.release()
            print(f"\nVideo saved to {output_path}")
        
        tracker_step = self.pipeline.get_step("tracking")
        if tracker_step:
            stats['unique_objects'] = tracker_step.tracker.next_id
        
        print(f"\nProcessing complete!")
        print(f"  Frames: {stats['frames_processed']}")
        print(f"  Total detections: {stats['total_detections']}")
        print(f"  Unique objects: {stats.get('unique_objects', 'N/A')}")
        print(f"  Max simultaneous tracks: {stats['max_tracks']}")
        
        return stats
    
    def reset(self):
        """Reset pipeline state."""
        self.pipeline.reset()
    
    def get_detector(self):
        """Get detector from pipeline."""
        detection_step = self.pipeline.get_step("detection")
        return detection_step.detector if detection_step else None
    
    def get_tracker(self):
        """Get tracker from pipeline."""
        tracking_step = self.pipeline.get_step("tracking")
        return tracking_step.tracker if tracking_step else None
    
    def list_steps(self) -> List[str]:
        """List all pipeline steps."""
        return self.pipeline.list_steps()

