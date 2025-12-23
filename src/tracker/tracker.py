"""Simple object tracker using Kalman filtering and Hungarian algorithm."""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Optional


class Track:
    """Represents a single tracked object."""
    
    def __init__(self, track_id: int, x: float, y: float):
        self.id = track_id
        self.kf = cv2.KalmanFilter(4, 2)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kf.statePost = np.array([x, y, 0, 0], np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        
        self.prediction = np.array([x, y])
        self.age = 0
        self.hits = 1
        self.history = [(int(x), int(y))]
    
    def predict(self) -> np.ndarray:
        """Predict next position."""
        pred = self.kf.predict()
        self.prediction = np.array([float(pred[0]), float(pred[1])])
        self.age += 1
        return self.prediction
    
    def update(self, x: float, y: float):
        """Update track with new measurement."""
        measurement = np.array([[x], [y]], np.float32)
        self.kf.correct(measurement)
        self.prediction = np.array([x, y])
        self.age = 0
        self.hits += 1
        self.history.append((int(x), int(y)))


class ObjectTracker:
    """Simple multi-object tracker using Kalman filter and Hungarian algorithm.
    
    Args:
        max_distance: Maximum distance for matching detection to track
        min_hits: Minimum hits before track is confirmed
        max_age: Maximum age (frames without detection) before track is deleted
        max_history: Maximum length of track history to keep
    """
    
    def __init__(
        self,
        max_distance: float = 30.0,
        min_hits: int = 3,
        max_age: int = 30,
        max_history: int = 20
    ):
        self.max_distance = max_distance
        self.min_hits = min_hits
        self.max_age = max_age
        self.max_history = max_history
        
        self.tracks: List[Track] = []
        self.next_id = 0
    
    def update(self, detections: List[List[float]]) -> List[Track]:
        """Update tracker with new detections.
        
        Args:
            detections: List of detections, each as [x, y, ...] (only x, y used)
        
        Returns:
            List of active tracks
        """
        for track in self.tracks:
            track.predict()
        
        assigned_tracks = []
        assigned_dets = []
        
        if len(self.tracks) > 0 and len(detections) > 0:
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            
            for t_idx, track in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    dist = np.linalg.norm(track.prediction - np.array(det[:2]))
                    cost_matrix[t_idx, d_idx] = dist
            
            row_inds, col_inds = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_inds, col_inds):
                if cost_matrix[r, c] < self.max_distance:
                    self.tracks[r].update(detections[c][0], detections[c][1])
                    assigned_tracks.append(r)
                    assigned_dets.append(c)
        
        for d_idx, det in enumerate(detections):
            if d_idx not in assigned_dets:
                new_track = Track(self.next_id, det[0], det[1])
                self.tracks.append(new_track)
                self.next_id += 1
        
        clean_tracks = []
        for track in self.tracks:
            if track.age < self.max_age:
                if len(track.history) > self.max_history:
                    track.history = track.history[-self.max_history:]
                clean_tracks.append(track)
        
        self.tracks = clean_tracks
        return self.tracks
    
    def get_confirmed_tracks(self) -> List[Track]:
        """Get only confirmed tracks (hits >= min_hits)."""
        return [t for t in self.tracks if t.hits >= self.min_hits]
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_id = 0

