import cv2
import numpy as np
import glob
import os
from scipy.optimize import linear_sum_assignment

INPUT_FOLDER = "data/images/bees_medium/blurred_frames"
DETECTIONS_FILE = "detections.npy"
OUTPUT_VIDEO = "bee_tracking_result.mp4"

MAX_DIST_THRESHOLD = 30.0
MIN_HITS = 3
MAX_AGE = 30

TAIL_LENGTH = 20
FONT = cv2.FONT_HERSHEY_SIMPLEX

class KalmanBee:
    def __init__(self, initial_x, initial_y):
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

        self.kf.statePost = np.array([initial_x, initial_y, 0, 0], np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

    def predict(self):
        prediction = self.kf.predict()
        return float(prediction[0]), float(prediction[1])

    def correct(self, x, y):
        measurement = np.array([[x], [y]], np.float32)
        self.kf.correct(measurement)
        return float(self.kf.statePost[0]), float(self.kf.statePost[1])

class Track:
    def __init__(self, id, x, y):
        self.id = id
        self.kf = KalmanBee(x, y)
        self.prediction = np.array([x, y])
        self.age = 0
        self.total_visible_count = 1
        self.history = []

    def update(self, x, y):
        self.kf.correct(x, y)
        self.prediction = np.array([x, y])
        self.age = 0
        self.total_visible_count += 1
        self.history.append((int(x), int(y)))
        if len(self.history) > TAIL_LENGTH:
            self.history.pop(0)

    def predict_next(self):
        pred_x, pred_y = self.kf.predict()
        self.prediction = np.array([pred_x, pred_y])
        self.age += 1
        return self.prediction

def main():
    if not os.path.exists(DETECTIONS_FILE):
        print(f"Error: {DETECTIONS_FILE} not found.")
        return

    raw_detections = np.load(DETECTIONS_FILE)
    detections_map = {}
    for row in raw_detections:
        frame_idx, x, y = int(row[0]), int(row[1]), int(row[2])
        if frame_idx not in detections_map:
            detections_map[frame_idx] = []
        detections_map[frame_idx].append([x, y])

    files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.png")))
    if not files:
        files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")))
    
    if not files:
        print("Error: No images found.")
        return

    sample = cv2.imread(files[0])
    h, w, _ = sample.shape
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

    tracks = []
    track_id_counter = 0
    
    total_bees_tracked = 0
    longest_track_len = 0

    print(f"Starting tracking on {len(files)} frames...")

    for i, file_path in enumerate(files):
        frame = cv2.imread(file_path)
        
        current_detections = detections_map.get(i, [])

        for t in tracks:
            t.predict_next()

        assigned_tracks = []
        assigned_dets = []

        if len(tracks) > 0 and len(current_detections) > 0:
            cost_matrix = np.zeros((len(tracks), len(current_detections)))
            
            for t_idx, t in enumerate(tracks):
                for d_idx, d in enumerate(current_detections):
                    dist = np.linalg.norm(t.prediction.flatten() - np.array(d))
                    cost_matrix[t_idx, d_idx] = dist

            row_inds, col_inds = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_inds, col_inds):
                if cost_matrix[r, c] < MAX_DIST_THRESHOLD:
                    tracks[r].update(current_detections[c][0], current_detections[c][1])
                    assigned_tracks.append(r)
                    assigned_dets.append(c)

        for d_idx, d in enumerate(current_detections):
            if d_idx not in assigned_dets:
                new_track = Track(track_id_counter, d[0], d[1])
                tracks.append(new_track)
                track_id_counter += 1
                total_bees_tracked += 1

        clean_tracks = []
        for t_idx, t in enumerate(tracks):
            if t.age < MAX_AGE:
                clean_tracks.append(t)
                if t.total_visible_count > longest_track_len:
                    longest_track_len = t.total_visible_count
            else:
                pass
        
        tracks = clean_tracks

        for t in tracks:
            if t.total_visible_count >= MIN_HITS:
                
                if len(t.history) > 1:
                    pts = np.array(t.history, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], False, (0, 255, 255), 2)

                color = ((t.id * 50) % 255, (t.id * 100) % 255, (t.id * 200) % 255)
                
                cx, cy = int(float(t.prediction[0])), int(float(t.prediction[1]))
                
                cv2.circle(frame, (cx, cy), 10, color, 2)
                cv2.putText(frame, str(t.id), (cx - 10, cy - 10), FONT, 0.5, color, 2)

                if t.age > 0:
                    cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)

        cv2.putText(frame, f"Frame: {i}", (10, 30), FONT, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Active Bees: {len(tracks)}", (10, 60), FONT, 0.7, (0, 255, 0), 2)

        video_writer.write(frame)

        if i % 100 == 0:
            print(f"Tracking frame {i}/{len(files)}... Active Tracks: {len(tracks)}")

    video_writer.release()
    print("\n" + "="*30)
    print(f"DONE! Video saved to {OUTPUT_VIDEO}")
    print(f"Total Unique Bees ID'd: {total_bees_tracked}")
    print(f"Longest Track Duration: {longest_track_len} frames")
    print("="*30)

if __name__ == "__main__":
    main()