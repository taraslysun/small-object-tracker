"""Simple object detector using background subtraction."""
import cv2
import numpy as np
from typing import List, Optional, Tuple


class ObjectDetector:
    """Object detector using KNN background subtraction.
    
    Args:
        history: Number of frames for background learning
        dist_threshold: Distance threshold for foreground detection
        min_area: Minimum object area
        max_area: Maximum object area
        erode_size: Erosion kernel size (width, height)
        dilate_size: Dilation kernel size (width, height)
        min_aspect_ratio: Minimum width/height ratio
        max_aspect_ratio: Maximum width/height ratio
        detect_shadows: Whether to detect shadows
    """
    
    def __init__(
        self,
        history: int = 500,
        dist_threshold: float = 800.0,
        min_area: float = 30.0,
        max_area: float = 100000.0,
        erode_size: Tuple[int, int] = (3, 3),
        dilate_size: Tuple[int, int] = (8, 8),
        min_aspect_ratio: float = 0.4,
        max_aspect_ratio: float = 2.5,
        detect_shadows: bool = False
    ):
        self.history = history
        self.dist_threshold = dist_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        
        self.back_sub = cv2.createBackgroundSubtractorKNN(
            history=history,
            dist2Threshold=dist_threshold,
            detectShadows=detect_shadows
        )
        
        self.kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, erode_size)
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilate_size)
    
    def detect(self, frame: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> List[List[float]]:
        """Detect objects in a frame.
        
        Args:
            frame: Input frame (BGR image)
            roi_mask: Optional region of interest mask
        
        Returns:
            List of detections, each as [centroid_x, centroid_y, width, height]
        """
        if frame is None or frame.size == 0:
            return []
        
        # Apply background subtraction
        fg_mask = self.back_sub.apply(frame)
        
        if roi_mask is not None:
            fg_mask = cv2.bitwise_and(fg_mask, roi_mask)
        
        clean_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_dilate)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, self.kernel_erode)
        
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if not (self.min_area < area < self.max_area):
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            if not (self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio):
                continue
            
            # centroid
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                detections.append([cX, cY, w, h])
        
        return detections
    
    def reset(self):
        """Reset the background model."""
        self.back_sub = cv2.createBackgroundSubtractorKNN(
            history=self.history,
            dist2Threshold=self.dist_threshold,
            detectShadows=False
        )
