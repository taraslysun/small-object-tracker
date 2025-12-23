"""Built-in pipeline steps for object detection and tracking."""
import cv2
import numpy as np
from typing import Any, Dict, Optional
from .pipeline import PipelineStep


class DetectionStep(PipelineStep):
    """Object detection step using background subtraction."""
    
    def __init__(self, detector=None, **detector_params):
        super().__init__("detection")
        
        if detector is not None:
            self.detector = detector
        else:
            from detector import ObjectDetector
            self.detector = ObjectDetector(**detector_params)
    
    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        roi_mask = context.get('roi_mask', None)
        detections = self.detector.detect(frame, roi_mask=roi_mask)
        context['detections'] = detections
        context['num_detections'] = len(detections)
        return context
    
    def reset(self):
        self.detector.reset()


class TrackingStep(PipelineStep):
    """Object tracking step using Kalman filter and Hungarian algorithm."""
    
    def __init__(self, tracker=None, **tracker_params):
        super().__init__("tracking")
        
        if tracker is not None:
            self.tracker = tracker
        else:
            from tracker import ObjectTracker
            self.tracker = ObjectTracker(**tracker_params)
    
    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        detections = context.get('detections', [])
        tracks = self.tracker.update(detections)
        confirmed = self.tracker.get_confirmed_tracks()
        
        context['tracks'] = tracks
        context['confirmed_tracks'] = confirmed
        context['num_tracks'] = len(tracks)
        context['num_confirmed'] = len(confirmed)
        return context
    
    def reset(self):
        self.tracker.reset()


class VisualizationStep(PipelineStep):
    """Visualization step for drawing tracks and detections."""
    
    def __init__(
        self,
        draw_detections: bool = False,
        draw_tracks: bool = True,
        draw_trajectories: bool = True,
        draw_ids: bool = True,
        draw_stats: bool = True
    ):
        super().__init__("visualization")
        self.draw_detections = draw_detections
        self.draw_tracks = draw_tracks
        self.draw_trajectories = draw_trajectories
        self.draw_ids = draw_ids
        self.draw_stats = draw_stats
    
    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        vis_frame = frame.copy()
        
        if self.draw_detections and 'detections' in context:
            for det in context['detections']:
                cx, cy, w, h = det[:4]
                x, y = int(cx - w/2), int(cy - h/2)
                cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        
        if 'confirmed_tracks' in context:
            for track in context['confirmed_tracks']:
                if self.draw_trajectories and len(track.history) > 1:
                    pts = np.array(track.history, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(vis_frame, [pts], False, (0, 255, 255), 2)
                
                if self.draw_tracks:
                    color = ((track.id * 50) % 255, (track.id * 100) % 255, (track.id * 200) % 255)
                    cx, cy = int(track.prediction[0]), int(track.prediction[1])
                    cv2.circle(vis_frame, (cx, cy), 8, color, 2)
                    
                    if self.draw_ids:
                        cv2.putText(vis_frame, str(track.id), (cx-10, cy-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if self.draw_stats:
            frame_num = context.get('frame_number', 0)
            num_tracks = context.get('num_confirmed', 0)
            
            cv2.putText(vis_frame, f"Frame: {frame_num}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Tracks: {num_tracks}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        context['visualization'] = vis_frame
        return context


class ROIFilterStep(PipelineStep):
    """Region of Interest filter step."""
    
    def __init__(self, roi_points: list = None):
        super().__init__("roi_filter")
        self.roi_points = roi_points
        self.roi_mask = None
    
    def set_roi(self, roi_points: list, frame_shape: tuple):
        """Set ROI from points."""
        self.roi_points = roi_points
        h, w = frame_shape[:2]
        self.roi_mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(roi_points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(self.roi_mask, [pts], 255)
    
    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.roi_mask is None and self.roi_points is not None:
            self.set_roi(self.roi_points, frame.shape)
        
        if self.roi_mask is not None:
            context['roi_mask'] = self.roi_mask
        
        return context


class OpticalFlowStep(PipelineStep):
    """Optical flow computation step."""
    
    def __init__(self):
        super().__init__("optical_flow")
        self.prev_gray = None
    
    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            context['optical_flow'] = flow
            context['flow_magnitude'] = magnitude
            context['flow_angle'] = angle
        
        self.prev_gray = gray
        return context
    
    def reset(self):
        self.prev_gray = None


class BackgroundSeparationStep(PipelineStep):
    """Separate static background from moving foreground using optical flow."""
    
    def __init__(self, flow_threshold: float = 2.0):
        super().__init__("background_separation")
        self.flow_threshold = flow_threshold
    
    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        if 'flow_magnitude' in context:
            magnitude = context['flow_magnitude']
            
            # Static regions have low flow
            static_mask = (magnitude < self.flow_threshold).astype(np.uint8) * 255
            moving_mask = (magnitude >= self.flow_threshold).astype(np.uint8) * 255
            
            context['static_mask'] = static_mask
            context['moving_mask'] = moving_mask
            
            # Apply to detections if available
            if 'detections' in context:
                filtered_detections = []
                for det in context['detections']:
                    cx, cy = int(det[0]), int(det[1])
                    if cy < magnitude.shape[0] and cx < magnitude.shape[1]:
                        if magnitude[cy, cx] >= self.flow_threshold:
                            filtered_detections.append(det)
                context['filtered_detections'] = filtered_detections
        
        return context


class FramePreprocessingStep(PipelineStep):
    """Frame preprocessing (blur, resize, etc)."""
    
    def __init__(self, blur_ksize: int = 5, resize_factor: float = 1.0):
        super().__init__("preprocessing")
        self.blur_ksize = blur_ksize
        self.resize_factor = resize_factor
    
    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        processed = frame.copy()
        
        if self.resize_factor != 1.0:
            h, w = frame.shape[:2]
            new_h, new_w = int(h * self.resize_factor), int(w * self.resize_factor)
            processed = cv2.resize(processed, (new_w, new_h))
        
        if self.blur_ksize > 0:
            processed = cv2.GaussianBlur(processed, (self.blur_ksize, self.blur_ksize), 0)
        
        context['preprocessed_frame'] = processed
        context['frame'] = processed
        return context


class SIFTFeatureStep(PipelineStep):
    """SIFT feature detection and description step."""
    
    def __init__(self, nfeatures: int = 0, nOctaveLayers: int = 3, 
                 contrastThreshold: float = 0.04, edgeThreshold: float = 10,
                 sigma: float = 1.6):
        super().__init__("sift_features")
        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            nOctaveLayers=nOctaveLayers,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold,
            sigma=sigma
        )
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_gray = None
    
    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect SIFT keypoints and descriptors
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        context['sift_keypoints'] = keypoints
        context['sift_descriptors'] = descriptors
        context['num_sift_features'] = len(keypoints)
        
        # Store for matching in next frame
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_gray = gray
        
        return context
    
    def reset(self):
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_gray = None


class SIFTMatchingStep(PipelineStep):
    """SIFT feature matching between consecutive frames."""
    
    def __init__(self, ratio_threshold: float = 0.75, min_match_count: int = 4):
        super().__init__("sift_matching")
        self.ratio_threshold = ratio_threshold
        self.min_match_count = min_match_count
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.prev_keypoints = None
        self.prev_descriptors = None
    
    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        keypoints = context.get('sift_keypoints', [])
        descriptors = context.get('sift_descriptors', None)
        
        if descriptors is None or len(keypoints) == 0:
            return context
        
        # Match with previous frame
        if self.prev_descriptors is not None and len(self.prev_descriptors) > 0:
            try:
                # Use KNN matching with k=2 for ratio test
                matches = self.bf_matcher.knnMatch(self.prev_descriptors, descriptors, k=2)
                
                # Apply ratio test (Lowe's ratio test)
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < self.ratio_threshold * n.distance:
                            good_matches.append(m)
                
                context['sift_matches'] = good_matches
                context['num_sift_matches'] = len(good_matches)
                context['prev_sift_keypoints'] = self.prev_keypoints
                
            except cv2.error:
                context['sift_matches'] = []
                context['num_sift_matches'] = 0
        else:
            context['sift_matches'] = []
            context['num_sift_matches'] = 0
        
        # Store for next frame
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
        return context
    
    def reset(self):
        self.prev_keypoints = None
        self.prev_descriptors = None


class SIFTEnhancedDetectionStep(PipelineStep):
    """
    Enhanced detection using SIFT features to validate and refine detections.
    
    Two modes:
    - uniform_objects=True: Keep detections with FEW features (for bees, balls, etc.)
    - uniform_objects=False: Keep detections with MANY features (for textured objects)
    """
    
    def __init__(self, detector=None, min_features_in_detection: int = 2, 
                 feature_density_threshold: float = 0.001, 
                 uniform_objects: bool = True, max_features_in_detection: int = 5,
                 **detector_params):
        super().__init__("sift_enhanced_detection")
        
        if detector is not None:
            self.detector = detector
        else:
            from detector import ObjectDetector
            self.detector = ObjectDetector(**detector_params)
        
        self.min_features_in_detection = min_features_in_detection
        self.feature_density_threshold = feature_density_threshold
        self.uniform_objects = uniform_objects
        self.max_features_in_detection = max_features_in_detection
    
    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        roi_mask = context.get('roi_mask', None)
        
        # Get standard detections
        detections = self.detector.detect(frame, roi_mask=roi_mask)
        
        # Get SIFT keypoints if available
        keypoints = context.get('sift_keypoints', [])
        
        if len(keypoints) > 0 and len(detections) > 0:
            # Validate detections using SIFT features
            validated_detections = []
            detection_features = []
            
            for det in detections:
                cx, cy, w, h = det[:4]
                x1, y1 = int(cx - w/2), int(cy - h/2)
                x2, y2 = int(cx + w/2), int(cy + h/2)
                
                # Count features inside detection bbox
                features_in_bbox = 0
                for kp in keypoints:
                    kp_x, kp_y = kp.pt
                    if x1 <= kp_x <= x2 and y1 <= kp_y <= y2:
                        features_in_bbox += 1
                
                # Calculate feature density (features per pixel)
                bbox_area = w * h
                feature_density = features_in_bbox / bbox_area if bbox_area > 0 else 0
                
                # Decide whether to keep detection based on mode
                if self.uniform_objects:
                    # For uniform objects (bees): keep LOW feature count
                    # (objects are uniform, high features = textured background)
                    keep_detection = (features_in_bbox < self.max_features_in_detection and 
                                     feature_density < self.feature_density_threshold * 5)
                else:
                    # For textured objects (vehicles): keep HIGH feature count  
                    # (objects have features, low features = plain background)
                    keep_detection = (features_in_bbox >= self.min_features_in_detection or 
                                     feature_density >= self.feature_density_threshold)
                
                if keep_detection:
                    validated_detections.append(det)
                    detection_features.append(features_in_bbox)
            
            context['detections'] = validated_detections
            context['detection_feature_counts'] = detection_features
            context['num_detections'] = len(validated_detections)
        else:
            context['detections'] = detections
            context['num_detections'] = len(detections)
        
        return context
    
    def reset(self):
        self.detector.reset()


class SIFTVisualizationStep(PipelineStep):
    """Visualization step for SIFT features and matches."""
    
    def __init__(self, draw_keypoints: bool = True, draw_matches: bool = True,
                 draw_detections: bool = False, max_keypoints: int = 100):
        super().__init__("sift_visualization")
        self.draw_keypoints = draw_keypoints
        self.draw_matches = draw_matches
        self.draw_detections = draw_detections
        self.max_keypoints = max_keypoints
    
    def process(self, frame: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        vis_frame = frame.copy()
        
        # Draw SIFT keypoints
        if self.draw_keypoints and 'sift_keypoints' in context:
            keypoints = context['sift_keypoints']
            # Limit number of keypoints for visualization
            if len(keypoints) > self.max_keypoints:
                # Sort by response (strength) and take top N
                keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:self.max_keypoints]
            
            vis_frame = cv2.drawKeypoints(vis_frame, keypoints, None, 
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        if self.draw_matches and 'sift_matches' in context and len(context.get('sift_matches', [])) > 0:
            matches = context['sift_matches']
            prev_kps = context.get('prev_sift_keypoints', [])
            curr_kps = context.get('sift_keypoints', [])
            
            if len(prev_kps) > 0 and len(curr_kps) > 0:
                for match in matches[:90]:
                    if match.queryIdx < len(prev_kps) and match.trainIdx < len(curr_kps):
                        pt1 = tuple(map(int, prev_kps[match.queryIdx].pt))
                        pt2 = tuple(map(int, curr_kps[match.trainIdx].pt))
                        cv2.line(vis_frame, pt1, pt2, (0, 255, 0), 1)
        
        if self.draw_detections and 'detections' in context:
            detections = context['detections']
            feature_counts = context.get('detection_feature_counts', [0] * len(detections))
            
            for i, det in enumerate(detections):
                cx, cy, w, h = det[:4]
                x, y = int(cx - w/2), int(cy - h/2)
                cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Show feature count
                if i < len(feature_counts):
                    cv2.putText(vis_frame, f"F:{feature_counts[i]}", (x, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Add SIFT stats
        num_features = context.get('num_sift_features', 0)
        num_matches = context.get('num_sift_matches', 0)
        
        cv2.putText(vis_frame, f"SIFT Features: {num_features}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(vis_frame, f"SIFT Matches: {num_matches}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        context['sift_visualization'] = vis_frame
        return context

