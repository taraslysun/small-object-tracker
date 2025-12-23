"""
Motion Filtering for Separating Moving Objects from Static Background

This module provides multiple strategies to distinguish objects that are 
translating through space (bees flying) from static or locally-moving 
background elements (swaying leaves, branches).
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class TrackDisplacementFilter:
    """
    Filter tracks based on net displacement over time.
    
    Static objects (leaves swaying) have high local motion but low net displacement.
    Moving objects (bees flying) have significant net displacement.
    """
    
    def __init__(self, min_displacement: float = 50.0, 
                 displacement_window: int = 10,
                 min_track_age: int = 5):
        """
        Args:
            min_displacement: Minimum total displacement (pixels) to consider moving
            displacement_window: Number of frames to measure displacement over
            min_track_age: Minimum track age before filtering (avoid premature rejection)
        """
        self.min_displacement = min_displacement
        self.displacement_window = displacement_window
        self.min_track_age = min_track_age
        self.track_start_positions = {}
    
    def filter_tracks(self, tracks: List) -> List:
        """
        Filter tracks keeping only those with significant displacement.
        
        Args:
            tracks: List of Track objects with history
            
        Returns:
            Filtered list of tracks
        """
        filtered = []
        
        for track in tracks:
            # Always keep young tracks (not enough data yet)
            if len(track.history) < self.min_track_age:
                filtered.append(track)
                continue
            
            # Calculate net displacement
            if track.id not in self.track_start_positions:
                self.track_start_positions[track.id] = track.history[0]
            
            start_pos = self.track_start_positions[track.id]
            current_pos = track.history[-1]
            
            # Use recent displacement window
            if len(track.history) >= self.displacement_window:
                window_start = track.history[-self.displacement_window]
                displacement = np.linalg.norm(
                    np.array(current_pos) - np.array(window_start)
                )
            else:
                displacement = np.linalg.norm(
                    np.array(current_pos) - np.array(start_pos)
                )
            
            # Keep track if it has moved significantly
            if displacement >= self.min_displacement:
                filtered.append(track)
        
        return filtered
    
    def reset(self):
        """Reset tracked start positions."""
        self.track_start_positions.clear()


class MotionConsistencyFilter:
    """
    Filter objects based on motion consistency.
    
    Moving objects have consistent motion direction.
    Static/swaying objects have inconsistent/oscillating motion.
    """
    
    def __init__(self, min_consistency: float = 0.6, 
                 history_length: int = 10):
        """
        Args:
            min_consistency: Minimum motion consistency score (0-1)
            history_length: Number of frames to analyze
        """
        self.min_consistency = min_consistency
        self.history_length = history_length
    
    def filter_tracks(self, tracks: List) -> List:
        """
        Filter tracks based on motion direction consistency.
        
        Args:
            tracks: List of Track objects
            
        Returns:
            Filtered list of tracks
        """
        filtered = []
        
        for track in tracks:
            if len(track.history) < 3:
                filtered.append(track)
                continue
            
            # Calculate motion vectors
            history = track.history[-self.history_length:]
            if len(history) < 3:
                filtered.append(track)
                continue
            
            motion_vectors = []
            for i in range(1, len(history)):
                vec = np.array(history[i]) - np.array(history[i-1])
                if np.linalg.norm(vec) > 0.1:  # Ignore very small movements
                    motion_vectors.append(vec)
            
            if len(motion_vectors) < 2:
                filtered.append(track)
                continue
            
            # Calculate consistency (how aligned are motion vectors)
            consistency = self._calculate_consistency(motion_vectors)
            
            if consistency >= self.min_consistency:
                filtered.append(track)
        
        return filtered
    
    def _calculate_consistency(self, vectors: List[np.ndarray]) -> float:
        """
        Calculate motion consistency score.
        
        Returns:
            Consistency score between 0 (random) and 1 (perfectly aligned)
        """
        if len(vectors) < 2:
            return 0.0
        
        # Normalize vectors
        normalized = []
        for v in vectors:
            norm = np.linalg.norm(v)
            if norm > 0:
                normalized.append(v / norm)
        
        if len(normalized) < 2:
            return 0.0
        
        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                similarity = np.dot(normalized[i], normalized[j])
                similarities.append(similarity)
        
        # Return average similarity (ranges from -1 to 1, map to 0 to 1)
        avg_similarity = np.mean(similarities)
        return (avg_similarity + 1.0) / 2.0


class OpticalFlowMotionFilter:
    """
    Use optical flow to distinguish global motion from local motion.
    
    Swaying leaves have high local optical flow but are part of static scene.
    Flying bees create optical flow that moves across the frame.
    """
    
    def __init__(self, min_flow_magnitude: float = 2.0,
                 max_local_variance: float = 10.0):
        """
        Args:
            min_flow_magnitude: Minimum flow magnitude to consider moving
            max_local_variance: Maximum variance in local flow (static regions have low variance)
        """
        self.min_flow_magnitude = min_flow_magnitude
        self.max_local_variance = max_local_variance
        self.prev_gray = None
    
    def filter_detections(self, detections: List, frame: np.ndarray) -> List:
        """
        Filter detections based on optical flow analysis.
        
        Args:
            detections: List of [cx, cy, w, h] detections
            frame: Current frame (BGR)
            
        Returns:
            Filtered detections
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return detections
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        filtered = []
        for det in detections:
            cx, cy, w, h = det[:4]
            x1, y1 = int(max(0, cx - w/2)), int(max(0, cy - h/2))
            x2, y2 = int(min(magnitude.shape[1], cx + w/2)), int(min(magnitude.shape[0], cy + h/2))
            
            # Extract flow in detection region
            roi_magnitude = magnitude[y1:y2, x1:x2]
            roi_flow_x = flow[y1:y2, x1:x2, 0]
            roi_flow_y = flow[y1:y2, x1:x2, 1]
            
            if roi_magnitude.size == 0:
                continue
            
            # Calculate flow statistics
            avg_magnitude = np.mean(roi_magnitude)
            flow_variance = np.var(roi_flow_x) + np.var(roi_flow_y)
            
            # Keep if:
            # 1. Has sufficient motion magnitude
            # 2. Has consistent flow direction (low variance = not oscillating)
            if avg_magnitude >= self.min_flow_magnitude:
                # Additional check: consistent flow direction
                if flow_variance < self.max_local_variance or avg_magnitude > self.min_flow_magnitude * 2:
                    filtered.append(det)
        
        self.prev_gray = gray
        return filtered
    
    def reset(self):
        """Reset previous frame."""
        self.prev_gray = None


class BackgroundRegionFilter:
    """
    Learn static regions of the frame and filter detections in those areas.
    
    If a region consistently has detections that don't move, it's likely
    background (swaying vegetation).
    """
    
    def __init__(self, grid_size: int = 32, 
                 min_frames: int = 30,
                 max_static_ratio: float = 0.8):
        """
        Args:
            grid_size: Size of grid cells for spatial analysis
            min_frames: Minimum frames before marking region as static
            max_static_ratio: Maximum ratio of frames with static detections
        """
        self.grid_size = grid_size
        self.min_frames = min_frames
        self.max_static_ratio = max_static_ratio
        self.frame_count = 0
        self.grid_detection_counts = defaultdict(int)  # Cell -> count of frames with detections
        self.grid_static_counts = defaultdict(int)     # Cell -> count of static detections
        self.frame_shape = None
    
    def update(self, detections: List, frame_shape: Tuple[int, int]):
        """
        Update static region map with new detections.
        
        Args:
            detections: List of [cx, cy, w, h] detections
            frame_shape: (height, width) of frame
        """
        self.frame_count += 1
        self.frame_shape = frame_shape
        
        # Track which grid cells have detections
        occupied_cells = set()
        for det in detections:
            cx, cy = det[0], det[1]
            cell_x = int(cx / self.grid_size)
            cell_y = int(cy / self.grid_size)
            cell = (cell_x, cell_y)
            occupied_cells.add(cell)
        
        # Update counts
        for cell in occupied_cells:
            self.grid_detection_counts[cell] += 1
            # Mark as static if in same cell for many frames
            if self.grid_detection_counts[cell] > self.min_frames:
                self.grid_static_counts[cell] += 1
    
    def filter_detections(self, detections: List) -> List:
        """
        Filter out detections in static background regions.
        
        Args:
            detections: List of [cx, cy, w, h] detections
            
        Returns:
            Filtered detections
        """
        if self.frame_count < self.min_frames:
            return detections  # Not enough data yet
        
        filtered = []
        for det in detections:
            cx, cy = det[0], det[1]
            cell_x = int(cx / self.grid_size)
            cell_y = int(cy / self.grid_size)
            cell = (cell_x, cell_y)
            
            # Check if this cell is considered static
            if cell in self.grid_detection_counts:
                detection_count = self.grid_detection_counts[cell]
                static_ratio = self.grid_static_counts[cell] / max(detection_count, 1)
                
                # Keep if region is not predominantly static
                if static_ratio < self.max_static_ratio:
                    filtered.append(det)
            else:
                # New region, keep it
                filtered.append(det)
        
        return filtered
    
    def reset(self):
        """Reset learned static regions."""
        self.frame_count = 0
        self.grid_detection_counts.clear()
        self.grid_static_counts.clear()


class CompositeMotionFilter:
    """
    Combines multiple filtering strategies for robust separation.
    
    Uses a voting system where detections/tracks must pass multiple filters.
    """
    
    def __init__(self, 
                 use_displacement: bool = True,
                 use_consistency: bool = True,
                 use_optical_flow: bool = True,
                 use_background_regions: bool = True,
                 min_votes: int = 2):
        """
        Args:
            use_*: Enable/disable individual filters
            min_votes: Minimum number of filters that must approve
        """
        self.filters = []
        
        if use_displacement:
            self.displacement_filter = TrackDisplacementFilter(
                min_displacement=50.0, displacement_window=10
            )
            self.filters.append('displacement')
        
        if use_consistency:
            self.consistency_filter = MotionConsistencyFilter(
                min_consistency=0.5, history_length=10
            )
            self.filters.append('consistency')
        
        if use_optical_flow:
            self.flow_filter = OpticalFlowMotionFilter(
                min_flow_magnitude=2.0, max_local_variance=10.0
            )
            self.filters.append('optical_flow')
        
        if use_background_regions:
            self.background_filter = BackgroundRegionFilter(
                grid_size=32, min_frames=30, max_static_ratio=0.8
            )
            self.filters.append('background_regions')
        
        self.min_votes = min_votes
        self.use_displacement = use_displacement
        self.use_consistency = use_consistency
        self.use_optical_flow = use_optical_flow
        self.use_background_regions = use_background_regions
    
    def filter_tracks_and_detections(self, tracks: List, 
                                     detections: List, 
                                     frame: np.ndarray) -> Tuple[List, List]:
        """
        Apply all filters and return filtered tracks and detections.
        
        Args:
            tracks: List of Track objects
            detections: List of [cx, cy, w, h] detections
            frame: Current frame
            
        Returns:
            (filtered_tracks, filtered_detections)
        """
        h, w = frame.shape[:2]
        
        # Track-level filtering
        filtered_tracks = tracks
        if self.use_displacement and hasattr(self, 'displacement_filter'):
            filtered_tracks = self.displacement_filter.filter_tracks(filtered_tracks)
        
        if self.use_consistency and hasattr(self, 'consistency_filter'):
            filtered_tracks = self.consistency_filter.filter_tracks(filtered_tracks)
        
        # Detection-level filtering
        filtered_detections = detections
        
        if self.use_optical_flow and hasattr(self, 'flow_filter'):
            filtered_detections = self.flow_filter.filter_detections(filtered_detections, frame)
        
        if self.use_background_regions and hasattr(self, 'background_filter'):
            self.background_filter.update(detections, (h, w))
            filtered_detections = self.background_filter.filter_detections(filtered_detections)
        
        return filtered_tracks, filtered_detections
    
    def reset(self):
        """Reset all filters."""
        if hasattr(self, 'displacement_filter'):
            self.displacement_filter.reset()
        if hasattr(self, 'flow_filter'):
            self.flow_filter.reset()
        if hasattr(self, 'background_filter'):
            self.background_filter.reset()

