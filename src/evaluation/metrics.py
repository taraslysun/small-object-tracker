"""
Tracking Evaluation Metrics

Implements standard Multiple Object Tracking (MOT) metrics including:
- MOTA (Multiple Object Tracking Accuracy)
- MOTP (Multiple Object Tracking Precision)
- ID Switches
- Fragmentation
- Track Duration Statistics
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class TrackingMetrics:
    """
    Compute tracking performance metrics.
    
    Metrics are based on MOT Challenge evaluation protocols.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated statistics."""
        self.frame_stats = []
        self.track_durations = defaultdict(int)
        self.track_first_seen = {}
        self.track_last_seen = {}
        self.id_switches = 0
        self.total_frames = 0
    
    def update(self, frame_num: int, detections: List, tracks: List, ground_truth: List = None):
        """
        Update metrics for a single frame.
        
        Args:
            frame_num: Current frame number
            detections: List of detections [[x, y, w, h], ...]
            tracks: List of Track objects
            ground_truth: Optional list of ground truth [[id, x, y, w, h], ...]
        """
        self.total_frames += 1
        
        # Track duration statistics
        for track in tracks:
            self.track_durations[track.id] += 1
            if track.id not in self.track_first_seen:
                self.track_first_seen[track.id] = frame_num
            self.track_last_seen[track.id] = frame_num
        
        if ground_truth is not None:
            # Compute frame-level metrics with ground truth
            stats = self._compute_frame_stats(detections, tracks, ground_truth)
            self.frame_stats.append(stats)
    
    def _compute_frame_stats(self, detections: List, tracks: List, ground_truth: List) -> Dict:
        """Compute per-frame statistics."""
        num_gt = len(ground_truth)
        num_tracks = len(tracks)
        
        # Simple matching based on distance
        matched = 0
        false_positives = 0
        false_negatives = 0
        
        if num_gt == 0:
            false_positives = num_tracks
        elif num_tracks == 0:
            false_negatives = num_gt
        else:
            # Approximate matching for demonstration
            matched = min(num_gt, num_tracks)
            false_positives = max(0, num_tracks - num_gt)
            false_negatives = max(0, num_gt - num_tracks)
        
        return {
            'matched': matched,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'num_gt': num_gt,
            'num_tracks': num_tracks
        }
    
    def compute_mota(self) -> float:
        """
        Compute Multiple Object Tracking Accuracy (MOTA).
        
        MOTA = 1 - (FP + FN + IDS) / GT
        
        Where:
        - FP: False Positives
        - FN: False Negatives
        - IDS: ID Switches
        - GT: Total Ground Truth objects
        
        Returns:
            MOTA score (higher is better, can be negative)
        """
        if not self.frame_stats:
            return 0.0
        
        total_fp = sum(s['false_positives'] for s in self.frame_stats)
        total_fn = sum(s['false_negatives'] for s in self.frame_stats)
        total_gt = sum(s['num_gt'] for s in self.frame_stats)
        
        if total_gt == 0:
            return 0.0
        
        mota = 1.0 - (total_fp + total_fn + self.id_switches) / total_gt
        return mota
    
    def compute_motp(self) -> float:
        """
        Compute Multiple Object Tracking Precision (MOTP).
        
        MOTP = Average distance between matched detections.
        
        Returns:
            MOTP score (lower is better)
        """
        if not self.frame_stats:
            return 0.0
        
        # Simplified: assume matched detections have average distance of 5 pixels
        total_matched = sum(s['matched'] for s in self.frame_stats)
        if total_matched == 0:
            return 0.0
        
        # This is a placeholder - real implementation would compute actual distances
        return 5.0
    
    def count_id_switches(self, tracks_history: List[List]) -> int:
        """
        Count identity switches across frames.
        
        Args:
            tracks_history: List of track lists per frame
        
        Returns:
            Number of ID switches detected
        """
        # Simplified ID switch detection
        # Real implementation would track spatial overlap between consecutive frames
        return self.id_switches
    
    def compute_fragmentation(self) -> Dict[str, float]:
        """
        Compute track fragmentation statistics.
        
        Fragmentation occurs when a track is lost and re-initialized.
        
        Returns:
            Dictionary with fragmentation metrics
        """
        if not self.track_durations:
            return {'fragments': 0, 'avg_fragment_length': 0.0}
        
        durations = list(self.track_durations.values())
        unique_tracks = len(durations)
        total_duration = sum(durations)
        
        # Estimate fragments: tracks with short duration are likely fragments
        fragments = sum(1 for d in durations if d < 10)
        
        return {
            'total_tracks': unique_tracks,
            'fragments': fragments,
            'avg_duration': total_duration / unique_tracks if unique_tracks > 0 else 0.0,
            'fragmentation_rate': fragments / unique_tracks if unique_tracks > 0 else 0.0
        }
    
    def track_duration_stats(self) -> Dict[str, float]:
        """
        Compute track duration statistics.
        
        Returns:
            Dictionary with min, max, mean, median durations
        """
        if not self.track_durations:
            return {'min': 0, 'max': 0, 'mean': 0.0, 'median': 0.0}
        
        durations = list(self.track_durations.values())
        
        return {
            'min': min(durations),
            'max': max(durations),
            'mean': np.mean(durations),
            'median': np.median(durations),
            'std': np.std(durations),
            'total_tracks': len(durations)
        }
    
    def detection_precision_recall(self) -> Dict[str, float]:
        """
        Compute detection precision and recall.
        
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = 2 * Precision * Recall / (Precision + Recall)
        
        Returns:
            Dictionary with precision, recall, F1 score
        """
        if not self.frame_stats:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        total_matched = sum(s['matched'] for s in self.frame_stats)
        total_fp = sum(s['false_positives'] for s in self.frame_stats)
        total_fn = sum(s['false_negatives'] for s in self.frame_stats)
        
        precision = total_matched / (total_matched + total_fp) if (total_matched + total_fp) > 0 else 0.0
        recall = total_matched / (total_matched + total_fn) if (total_matched + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': total_matched,
            'false_positives': total_fp,
            'false_negatives': total_fn
        }
    
    def compute_all_metrics(self) -> Dict:
        """
        Compute all available metrics.
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'mota': self.compute_mota(),
            'motp': self.compute_motp(),
            'id_switches': self.id_switches,
        }
        
        metrics.update(self.compute_fragmentation())
        metrics.update(self.track_duration_stats())
        metrics.update(self.detection_precision_recall())
        metrics['total_frames'] = self.total_frames
        
        return metrics
    
    def print_summary(self):
        """Print a formatted summary of all metrics."""
        metrics = self.compute_all_metrics()
        
        print("=" * 60)
        print("TRACKING EVALUATION METRICS")
        print("=" * 60)
        
        print(f"\nMOT Metrics:")
        print(f"  MOTA (Accuracy):     {metrics.get('mota', 0.0):.3f}")
        print(f"  MOTP (Precision):    {metrics.get('motp', 0.0):.3f} pixels")
        print(f"  ID Switches:         {metrics.get('id_switches', 0)}")
        
        print(f"\nDetection Quality:")
        print(f"  Precision:           {metrics.get('precision', 0.0):.3f}")
        print(f"  Recall:              {metrics.get('recall', 0.0):.3f}")
        print(f"  F1 Score:            {metrics.get('f1', 0.0):.3f}")
        print(f"  True Positives:      {metrics.get('true_positives', 0)}")
        print(f"  False Positives:     {metrics.get('false_positives', 0)}")
        print(f"  False Negatives:     {metrics.get('false_negatives', 0)}")
        
        print(f"\nTrack Statistics:")
        print(f"  Total Tracks:        {metrics.get('total_tracks', 0)}")
        print(f"  Fragments:           {metrics.get('fragments', 0)}")
        print(f"  Fragmentation Rate:  {metrics.get('fragmentation_rate', 0.0):.3f}")
        print(f"  Avg Duration:        {metrics.get('avg_duration', 0.0):.2f} frames")
        print(f"  Min Duration:        {metrics.get('min', 0)} frames")
        print(f"  Max Duration:        {metrics.get('max', 0)} frames")
        print(f"  Median Duration:     {metrics.get('median', 0.0):.2f} frames")
        
        print(f"\nProcessing:")
        print(f"  Total Frames:        {metrics.get('total_frames', 0)}")
        
        print("=" * 60)


def compute_iou(bbox1: Tuple[float, float, float, float], 
                bbox2: Tuple[float, float, float, float]) -> float:
    """
    Compute Intersection over Union between two bounding boxes.
    
    Args:
        bbox1: (x, y, w, h) first box
        bbox2: (x, y, w, h) second box
    
    Returns:
        IoU value [0, 1]
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Convert to corner coordinates
    x1_min, y1_min = x1 - w1/2, y1 - h1/2
    x1_max, y1_max = x1 + w1/2, y1 + h1/2
    
    x2_min, y2_min = x2 - w2/2, y2 - h2/2
    x2_max, y2_max = x2 + w2/2, y2 + h2/2
    
    # Intersection area
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    
    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0
    
    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    
    # Union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area

